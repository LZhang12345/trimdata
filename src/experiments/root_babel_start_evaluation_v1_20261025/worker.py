"""One frozen native START bank, real reload qualification, accepted-QA continuation."""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from . import banks, common as c, media

STOP = False
PROCESS = None
MODULE = "experiments.root_babel_start_evaluation_v1_20261025.worker"


def stopping(signum, frame):
    global STOP
    STOP = True
    if PROCESS is not None and PROCESS.poll() is None:
        os.killpg(PROCESS.pid, signal.SIGUSR1)


def budget(cfg):
    size = sum(p.stat().st_size for p in Path(cfg["output"]).rglob("*") if p.is_file())
    c.require(size <= cfg["output_limit_bytes"], "Accepted output envelope exhausted; preserve progress")
    return size


def runtime(cfg):
    import torch
    c.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8" and
              all(os.environ.get(k) == "8" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
              "Pre-CUDA deterministic GEMM and CPU thread environment differs")
    expected = dict(torch="2.1.2+cu121", transformers="4.37.2", numpy="1.26.4",
                    tokenizers="0.15.1", PIL="10.2.0")
    root = Path(cfg["python"]).parent.parent.resolve()
    modules = {}
    for name, version in expected.items():
        m = importlib.import_module(name)
        dist = metadata.distribution("Pillow" if name == "PIL" else name)
        origin = Path(m.__file__).resolve()
        c.require(str(m.__version__) == dist.version == version and origin.is_relative_to(root)
                  and Path(dist.locate_file("")).resolve().is_relative_to(root),
                  "Actual module/distribution/version/origin differs: " + name)
        modules[name] = dict(version=str(m.__version__), distribution=dist.version, source=c.desc(origin))
    c.require(torch.cuda.is_available() and torch.cuda.device_count() == 1
              and torch.version.cuda == "12.1" and "L40S" in torch.cuda.get_device_name(0),
              "One actual CUDA12.1 L40S required")
    torch.manual_seed(1234)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return dict(modules=modules, torch_C=c.desc(torch._C.__file__), CUDA=torch.version.cuda,
                GPU=torch.cuda.get_device_name(0), compute_capability=list(torch.cuda.get_device_capability(0)),
                cudnn=torch.backends.cudnn.version(), deterministic_algorithms=True, tf32=False,
                CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                CPU_thread_environment={k:os.environ[k] for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")})


@contextmanager
def offline():
    previous = socket.socket.connect
    def denied(*args, **kwargs):
        raise RuntimeError("NetworkForbiddenInCachedStartingModelEvaluation")
    socket.socket.connect = denied
    try:
        yield
    finally:
        socket.socket.connect = previous


def entries(journal, rows):
    result = []
    for row in rows:
        if row["semantic_key"] in journal.present:
            raw, audit = journal.db.execute("SELECT raw,audit FROM answers WHERE k=?",
                                           (row["semantic_key"],)).fetchone()
            result.append(dict(semantic_key=row["semantic_key"], raw=json.loads(raw), audit=json.loads(audit)))
    return result


def same_repeat(first, second):
    c.require(len(first) == len(second) == 2, "Exactly two real accepted QAs and repeats required")
    for a, b in zip(first, second):
        c.require(a["semantic_key"] == b["semantic_key"] and
                  a["audit"]["native_generation"] == b["audit"]["native_generation"] and
                  {k: v for k, v in a["raw"].items() if k != "answer_id"} ==
                  {k: v for k, v in b["raw"].items() if k != "answer_id"},
                  "Actual reloaded prompt/pixels/IDs/raw answer changed")
        x, y = a["audit"]["actual"], b["audit"]["actual"]
        c.require(x["job_id"] == y["job_id"] and x["nonce"] == y["nonce"] and
                  x["process_nonce"] != y["process_nonce"], "Genuine fresh process in same physical job required")


def expected_inputs(row, loaded, store):
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from experiments.session6_native_start_execution_v6_20261024.audit import tensor_digest
    tokenizer, model, processor, _ = loaded
    c.require(not model.config.mm_use_im_start_end and not model.config.mm_use_im_patch_token,
              "Native image-token policy differs")
    conversation = conv_templates["vicuna_v1"].copy()
    conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + row["native_input"]["text"])
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    tokens = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    rgb = store.rgb(row["native_input"]["image"])
    try:
        pixels = process_images([rgb], processor, model.config)[0].unsqueeze(0).to(torch.float16)
        return dict(prompt_utf8_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    input_ids_sha256=tensor_digest(tokens), input_token_count=tokens.shape[-1],
                    pixel_values_sha256=tensor_digest(pixels), image_size=list(rgb.size))
    finally:
        rgb.close()


def native(cfg, rows, journal, loaded, store, context, deadline):
    from experiments.session6_native_start_execution_v6_20261024 import audit, kernel
    missing = journal.missing()
    recorder = audit.Recorder(missing, loaded, dict(images=store.images), str(store.virtual), journal.function, context)
    args = SimpleNamespace(**cfg["decoding"], num_chunks=1, chunk_idx=0, model_base=None,
        model_path=str(Path(cfg["output"]) / "START-Stage1-native-c121"), image_folder=str(store.virtual),
        question_file=str(Path(cfg["output"]) / "native-source-view.jsonl"),
        answers_file=str(journal.directory / "native-answers.jsonl"))
    module = importlib.import_module(kernel.MODULES["loader"])
    # CPU feed adaptation only: the exact native dataset/loader body is retained;
    # no CUDA child forks and no worker inherits a live ZIP handle or model.
    defaults = module.create_data_loader.__defaults__
    c.require(defaults == (1, 4), "Pinned native loader defaults changed")
    previous = audit.expected_inputs
    module.create_data_loader.__defaults__ = (1, 0)
    audit.expected_inputs = lambda row, value, inventory, root: expected_inputs(row, value, store)
    try:
        with store.installed(), recorder.installed():
            return kernel.evaluate("loader", args, rows, journal, loaded, deadline,
                {kernel.MODULES["loader"]: c.desc(c.LLAVA / "llava/eval/model_vqa_loader.py")},
                recorder.take, stop_requested=lambda: STOP)
    finally:
        audit.expected_inputs = previous
        module.create_data_loader.__defaults__ = defaults


def bound_inputs(cfg):
    roster, inventory, targets = banks.load(cfg)
    snapshot = None
    if "archive_mapping" in cfg:
        support = c.checked(roster["support"])
        ref = support["media_source"]
        c.require(ref == cfg["archive_mapping"]["archive"] and c.desc(ref["path"]) == ref,
                  "Actual qualified original image archive changed")
        snapshot = media.fingerprint(ref["path"])
    # Bind every historical full RGB source exactly once, never once per image.
    proof_refs = {c.key(v["source_proof"]): v["source_proof"] for v in inventory["images"].values()}
    proofs = {k: c.checked(v) for k, v in proof_refs.items()}
    for value in inventory["images"].values():
        proof = proofs[c.key(value["source_proof"])]["images"][value["source_proof_key"]]
        c.require(all(proof.get(k) == value[k] for k in
                      ("encoded_bytes", "encoded_sha256", "rgb_sha256", "width", "height")),
                  "Real complete RGB proof/media identity differs")
    return roster, inventory, targets, snapshot


def result_score(cfg, rows, journal, targets):
    from experiments.session6_official_eval_v2_20261024 import scoring
    accepted = entries(journal, rows)
    c.require(len(accepted) == len(rows), "Partial bank cannot score")
    if cfg["bank"] == "POPE":
        scores = {}
        for category in ("adversarial", "popular", "random"):
            records = [dict(prediction=a["raw"]["text"], label=targets[r["original_row_index"]]["label"])
                       for r, a in zip(rows, accepted) if targets[r["original_row_index"]]["category"] == category]
            try:
                scores[category] = scoring.pope(records)
            except ZeroDivisionError:
                # Keep the original undefined native denominator as counterevidence,
                # never invent a finite precision/recall/F1 or rerun completed NN.
                scores[category] = dict(status="ORIGINAL_NATIVE_SCORER_UNDEFINED_DENOMINATOR",
                                        questions=len(records), official_values=None)
        if any(s.get("official_values") is None for s in scores.values()):
            return dict(status="COMPLETE_RAW_POPE8910_OFFICIAL_SCORE_UNDEFINED",
                        questions=len(rows), categories=scores, official_score=None,
                        native_scorer_not_relaxed=True, scientific_round_complete=False)
        correct = sum(sum(s["per_question_credit"]) for s in scores.values())
        return dict(status="COMPLETE_OFFICIAL_POPE8910_START", accuracy=100 * correct / len(rows),
                    questions=len(rows), categories=scores, macro_not_primary_native7=True)
    return dict(status="COMPLETE_NATIVE_TEST_PREDICTIONS_NOT_OFFICIAL_SERVER_SCORE",
                questions=len(rows), official_score=None, required_external_server=True)


def child(cfg, binding, current, attempt, phase):
    from experiments.session6_start_model_qualification_v4_eager_20261024 import model, worker as qualifier
    from experiments.session6_native_start_execution_v6_20261024 import audit, identity as identity_io
    roster, inventory, targets = banks.load(cfg)
    frozen = c.read(Path(cfg["output"]) / "INPUTS.json")
    c.require(frozen["roster_sha256"] == c.key(roster) and frozen["media"] == cfg["media"]
              and frozen["configuration"] == binding["configuration"], "Prepared actual full input identity differs")
    store = media.Store(cfg, roster["records"], inventory, frozen["archive_snapshot"])
    current = dict(current, process_nonce=uuid.uuid4().hex, pid=os.getpid(), phase=phase)
    out = attempt / phase
    out.mkdir(exist_ok=False)
    observed = runtime(cfg)
    c.seal(out / "RUNTIME.json", observed)
    assets = model.verified_assets(cfg["native_assets"], cfg["cache_root"])
    capsule = Path(cfg["output"]) / "capsule"
    with offline():
        loaded3 = model.compose(assets, projector_path=capsule / "mm_projector.bin" if phase != "first" else None,
                                config_path=capsule if phase != "first" else None)
        live, _ = qualifier.identity(loaded3, cfg["native_assets"], observed,
                                     diagnostic_path=out / "BEFORE_MODEL_GUARD.json")
        actual_identity = identity_io.json_domain(live)
        c.seal(out / "ACTUAL_MODEL_IDENTITY.json", actual_identity)
        reference = c.checked(cfg["reference_components"])
        compared = ("parameter_sha256", "buffers", "generation_config", "tokenizer", "processor",
                    "source_commit", "language_attention", "vision_attention", "training_updates", "adapter_active")
        component_match = {k: actual_identity[k] == reference[k] for k in compared}
        c.seal(out / "REFERENCE_COMPONENT_COMPARISON.json", dict(matches=component_match,
               reference=cfg["reference_components"], actual_identity=c.desc(out / "ACTUAL_MODEL_IDENTITY.json"),
               cross_site_runtime_equal_claim=False, effective_config_path_relabelled=False))
        c.require(all(component_match.values()), "Initial START parameters/buffers/tokenizer/processor/reference differ")
        first_path = Path(cfg["output"]) / "MODEL_IDENTITY.json"
        if first_path.exists():
            expected = c.read(first_path)
            actual_identity = identity_io.persist_and_compare(live, expected, out, c.key(expected))
        else:
            c.require(phase == "first", "No initial identity for reload")
            c.seal(first_path, actual_identity)
        if not capsule.exists():
            c.require(phase == "first", "Missing exact no-training reload capsule")
            model.save_no_training_capsule(loaded3[1], cfg["native_assets"], capsule)
        else:
            capsule_ref = c.read(capsule / "ROUNDTRIP.json")
            c.require(capsule_ref["assets"] == cfg["native_assets"] and
                      c.desc(capsule_ref["projector"]["path"]) == capsule_ref["projector"] and
                      c.desc(capsule_ref["config"]["path"]) == capsule_ref["config"], "Reload capsule changed")
        loaded = (*loaded3, getattr(loaded3[1].config, "max_sequence_length", 2048))
        function = dict(schema="root-babel-native-START-bank-function-v1", bank=cfg["bank"],
            model_identity_sha256=c.key(actual_identity), model_generation_config=actual_identity["generation_config"],
            configuration=binding["configuration"], sources_sha256=c.key(cfg["sources"]),
            QA_roster_sha256=c.key(roster), media=cfg["media"], decoding=cfg["decoding"],
            native_loop=c.desc(c.LLAVA / "llava/eval/model_vqa_loader.py"),
            job_id=current["job_id"], nonce=binding["nonce"], training_updates=0)
        c.seal(Path(cfg["output"]) / "FUNCTION.json", function)
        rows = roster["records"]
        prefix = rows[:2]
        deadline = time.monotonic() + max(0, current["end_epoch"] - time.time() - cfg["reserve_seconds"])
        class Bounded(audit.AuditedJournal):
            def append(self, *args):
                budget(cfg)
                super().append(*args)
        which = "first" if phase == "first" else "repeat"
        if not (Path(cfg["output"]) / "QUALIFIED_GENERATION.json").exists():
            j = Bounded(Path(cfg["output"]) / which, function, prefix)
            try:
                native(cfg, prefix, j, loaded, store, current, deadline)
                accepted = entries(j, prefix)
                if j.missing():
                    return dict(status="PARTIAL_QUALIFICATION_NO_PRODUCTION", remaining=len(j.missing()))
            finally:
                j.close()
            c.seal(Path(cfg["output"]) / (which.upper() + "_PREFIX.json"),
                   dict(function=function, entries=accepted, actual=current))
            if phase == "first":
                return dict(status="TWO_REAL_QAS_ACCEPTED_FRESH_RELOAD_REQUIRED", remaining=len(rows) - 2)
            first = c.read(Path(cfg["output"]) / "FIRST_PREFIX.json")
            c.require(first["function"] == function, "Reload function differs")
            same_repeat(first["entries"], accepted)
            c.seal(Path(cfg["output"]) / "QUALIFIED_GENERATION.json", dict(
                status="PASS_SCOPED_ACTUAL_BABEL_TWO_QA_FRESH_RELOAD_FOUR_OUTPUTS",
                function=function, first=c.desc(Path(cfg["output"]) / "FIRST_PREFIX.json"),
                repeat=c.desc(Path(cfg["output"]) / "REPEAT_PREFIX.json"), forwards=4,
                bank_complete=False, student_or_optimizer_runtime_qualified=False, scientific_round_complete=False))
        q = c.read(Path(cfg["output"]) / "QUALIFIED_GENERATION.json")
        c.require(q["function"] == function, "Foreign actual generation qualification")
        journal = Bounded(Path(cfg["output"]) / "production", function, rows)
        try:
            for entry in c.checked(q["first"])["entries"]:
                if entry["semantic_key"] not in journal.present:
                    journal.append(entry["semantic_key"], entry["raw"], entry["audit"])
            before = len(journal.present)
            native(cfg, rows, journal, loaded, store, current, deadline)
            remaining = len(journal.missing())
            result = dict(status="PARTIAL_REAL_BANK_MISSING_ONLY" if remaining else "COMPLETE_REAL_BANK",
                          accepted=len(journal.present), remaining=remaining, new_answers=len(journal.present) - before)
            if not remaining:
                result["native_answers"] = journal.export(Path(cfg["output"]) / "RAW_NATIVE_ANSWERS.jsonl")
                result["score"] = result_score(cfg, rows, journal, targets)
            return result
        finally:
            journal.close()
            store.close()


def subprocess_phase(binding_path, cfg, current, attempt, phase):
    global PROCESS
    destination = attempt / (phase + ".json")
    argv = [cfg["python"], "-B", "-m", MODULE, "--binding", str(binding_path),
            "--child", phase, "--attempt", str(attempt), "--result", str(destination)]
    c.seal(attempt / (phase + "_INTENT.json"), dict(argv=argv, actual=current, UTC=c.now()))
    PROCESS = subprocess.Popen(argv, start_new_session=True)
    try:
        timeout = max(1, current["end_epoch"] - time.time() - 90)
        code = PROCESS.wait(timeout=timeout)
        c.require(code == 0 and destination.exists(), "Child failed; preserve actual failure, no blind requeue")
        return c.read(destination)
    except subprocess.TimeoutExpired:
        os.killpg(PROCESS.pid, signal.SIGTERM)
        try:
            PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(PROCESS.pid, signal.SIGKILL)
            PROCESS.wait(timeout=10)
        raise RuntimeError("OwnedChildDeadlineExceededProgressPreserved")
    finally:
        PROCESS = None


def run(binding_path, child_phase=None, attempt_path=None, result_path=None):
    began = time.monotonic()
    binding = c.read(binding_path)
    cfg = c.checked(binding["configuration"])
    c.validate(cfg)
    current = c.actual(binding, cfg)
    output = Path(cfg["output"])
    if child_phase:
        expected = output / "attempts" / (current["job_id"] + "-r" + str(current["restart"]))
        c.require(Path(attempt_path) == expected and Path(result_path) == expected / (child_phase + ".json"),
                  "Exact current child output route required")
        try:
            value = child(cfg, binding, current, expected, child_phase)
            return c.seal(result_path, dict(value, actual=current, elapsed_seconds=time.monotonic() - began))
        except BaseException as error:
            c.seal(expected / (child_phase + "_FAILURE.json"), dict(error_class=type(error).__name__,
                   message_sha256=hashlib.sha256(str(error).encode()).hexdigest(), actual=current,
                   qualification=False, elapsed_seconds=time.monotonic() - began))
            raise
    output.mkdir(parents=True, exist_ok=True)
    with (output / "EXECUTION.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c.seal(output / "OWNERSHIP.json", dict(owner=c.OWNER, job_id=current["job_id"], nonce=binding["nonce"],
                                             configuration=binding["configuration"], review=binding["review"]))
        attempt = output / "attempts" / (current["job_id"] + "-r" + str(current["restart"]))
        attempt.mkdir(parents=True, exist_ok=False)
        try:
            c.seal(attempt / "START.json", dict(actual=current, binding=c.desc(binding_path), UTC=c.now()))
            budget(cfg)
            vfs = os.statvfs(output)
            c.require(vfs.f_bavail * vfs.f_frsize >= cfg["output_limit_bytes"] and vfs.f_favail >= 100,
                      "Insufficient actual output filesystem bytes/inodes")
            roster, inventory, targets, snapshot = bound_inputs(cfg)
            target_hash = c.key(targets) if cfg["bank"] == "POPE" else None
            c.seal(output / "INPUTS.json", dict(configuration=binding["configuration"], media=cfg["media"],
                roster_sha256=c.key(roster), questions=len(roster["records"]), images=len(inventory["images"]),
                official_target_join_sha256=target_hash, archive_snapshot=snapshot))
            c.seal(output / "QA_FUNCTION_CLAIM.json", dict(actual_job_id=current["job_id"], nonce=binding["nonce"],
                configuration=binding["configuration"], bank=cfg["bank"], input_identity=c.desc(output / "INPUTS.json"),
                scope="Untouched START only; no student, no training, one missing-only owner"))
            if (output / "COMPLETE.json").exists():
                return c.desc(output / "COMPLETE.json")
            result = None
            if not (output / "FIRST_PREFIX.json").exists():
                result = subprocess_phase(binding_path, cfg, current, attempt, "first")
            if not STOP and (output / "FIRST_PREFIX.json").exists() and time.time() < current["end_epoch"] - 600:
                result = subprocess_phase(binding_path, cfg, current, attempt, "repeat")
            c.require(result is not None, "No child work performed; insufficient safe lifetime")
            receipt = c.seal(attempt / "RESULT.json", dict(result, configuration=binding["configuration"],
                    actual=current, total_wall_seconds=time.monotonic() - began, UTC=c.now(),
                    full15_complete=False, scientific_round_complete=False, all_allocated_costs_count=True))
            if result["status"] == "COMPLETE_REAL_BANK":
                return c.seal(output / "COMPLETE.json", dict(result=receipt, configuration=binding["configuration"],
                    bank=cfg["bank"], questions=len(roster["records"]), actual=current,
                    status="COMPLETE_ONE_START_BANK_NOT_FULL15_OR_STUDENT_QUALIFICATION"))
            if current["restart"] + 1 < cfg["max_attempts"]:
                c.seal(attempt / "REQUEUE_INTENT.json", dict(actual=current, result=receipt,
                    argv=["scontrol", "requeue", current["job_id"]], same_physical_job_only=True))
                try:
                    rpc = subprocess.run(["scontrol", "requeue", current["job_id"]], capture_output=True,
                                         text=True, timeout=30)
                except subprocess.TimeoutExpired:
                    c.seal(attempt / "REQUEUE_AMBIGUOUS.json", dict(no_retry=True, actual=current))
                    raise
                c.seal(attempt / "REQUEUE_ACK.json", dict(returncode=rpc.returncode,
                    stdout=rpc.stdout, stderr=rpc.stderr, actual=current))
                c.require(rpc.returncode == 0, "Same-job continuation rejected; no successor assumed")
            return receipt
        except BaseException as error:
            c.seal(attempt / "FAILURE.json", dict(actual=current, error_class=type(error).__name__,
                message_sha256=hashlib.sha256(str(error).encode()).hexdigest(),
                full_bank_pass=False, elapsed_seconds=time.monotonic() - began))
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binding", required=True)
    p.add_argument("--child", choices=("first", "repeat"))
    p.add_argument("--attempt")
    p.add_argument("--result")
    args = p.parse_args()
    c.require(bool(args.child) == bool(args.attempt) == bool(args.result), "Child route arguments incomplete")
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stopping)
    print(json.dumps(run(args.binding, args.child, args.attempt, args.result)))


if __name__ == "__main__":
    main()
