"""Root comparator final checkpoints, unchanged official inference and QA journals."""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from experiments.root_native_trained_evaluation_v3_20261026 import worker as original
from experiments.root_native_trained_evaluation_v3_20261026 import banks, generation, media
from experiments.root_native_trained_eval_qual_v3_20261026 import worker as qualified
from experiments.root_native_trained_loader_v3_20261026 import model
from . import contract as c, historical

u = c.u
ROOT, HERE = c.ROOT, c.HERE
PYTHON = qualified.PYTHON
budget = original.budget
publish_same = original.publish_same
accepted_prefix_count = original.accepted_prefix_count
accepted_progress = original.accepted_progress


def sources(base, phase3):
    result = original.sources(base, phase3)
    for path, digest in historical.route.sources().items():
        c.require(path not in result or result[path] == digest, "Conflicting bound dependency")
        result[path] = digest
    for path in HERE.iterdir():
        if path.suffix in (".py", ".sbatch") and not path.name.startswith("test_"):
            result[str(path.relative_to(ROOT))] = u.sha(path.read_bytes())
    return dict(sorted(result.items()))


def documents(binding, load=c.checked):
    cfg = load(binding["configuration"])
    training = c.validate_configuration(cfg, load)
    c.require(cfg["python"] == PYTHON, "Exact compatible existing native environment")
    base, phase3 = load(cfg["official_POPE_configuration"]), load(cfg["phase3_configuration"])
    c.require(cfg["sources"] == sources(base, phase3) and cfg["frozen_scope"] == base["frozen_scope"]
              and cfg["decoding"] == base["decoding"]
              and cfg["scienceqa_decoding"] == dict(conv_mode="vicuna_v1", temperature=0,
                  single_pred_prompt=True, max_new_tokens=1024, native_num_beams_keyword=False,
                  native_top_p_keyword=False)
              and all(cfg[key] is True for key in ("no_model_downloads", "no_base_copies",
                                                  "no_START_journal_reuse")),
              "Unchanged frozen official banks and native decoding")
    review = load(binding["independent_source_config_review"])
    c.require(review["status"] == c.REVIEW_STATUS and review["independent"] is True
              and review["reviewer"] not in (None, "", "Session5")
              and review["configurations"][cfg["student_id"]] == binding["configuration"]
              and review["sources"] == cfg["sources"], "Root personal exact-source/config review required")
    for path, digest in cfg["sources"].items():
        c.require(u.sha((ROOT / path).read_bytes()) == digest, "Frozen evaluation source changed: " + path)
    observation = load(cfg["actual_media_observation"])
    c.require(observation["native_pool_image_root"] == "/data/hf_cache/datasets/dsmm_llava_665k_v1/media"
              and cfg["bank_inputs"]["TextVQA"]["image_root"] == observation["native_pool_image_root"] + "/textvqa/train_images"
              and observation["TextVQA_native_cached_sample"][0]["sha256"]
              == "5532753f1d943f12f72a8ec3dd9692b2414dfad892a431de95cb649d83356042",
              "Exact existing TextVQA locator observation")
    qref = u.desc(u.literal(cfg["required_actual_qualification_result_path"]))
    q = load(qref)
    qr = load(u.desc(u.literal(cfg["required_independent_qualification_review_path"])))
    c.require(q["status"] == "PASS_ACTUAL_NATIVE_TRAINED_ADAPTER_TWO_QA_FRESH_RELOAD"
              and q["configuration"] == cfg["qualified_loader_configuration"]
              and q["adapter_active"] is True and q["adapter_merged"] is False
              and q["distinct_QAs"] == 2 and q["neural_outputs"] == 4 and q["full_bank_scored"] is False
              and q["actual"]["job_id"] == c.QUALIFIER
              and qr["status"] == "PASS_INDEPENDENT_ACTUAL_NATIVE_TRAINED_EVAL_QUALIFICATION_V3"
              and qr["independent"] is True and qr["readout"] == qref
              and qr["configuration"] == q["configuration"]
              and qr["actual_independent_reviewer"]["job_id"] == c.REVIEWER
              and qr["subject_actual_accounting"]["job_id"] == c.QUALIFIER
              and qr["subject_neural_generations_verified"] == 4 and qr["distinct_QAs"] == 2
              and qr["all_saved_checkpoint_payload_SHA_verified"] is True
              and qr["actual_trained_tensor_and_persistent_buffer_guards_verified"] is True,
              "Actual loader10942687 result and real10942858 review, not afterok or source approval")
    anchor = load(load(cfg["qualified_loader_configuration"])["training_configuration"])
    c.require(all(training["providers"][key] == anchor["providers"][key] for key in ("assets", "environment"))
              and training["recipe"] == anchor["recipe"], "Exact qualified model/processor/native recipe")
    claim, grant = load(binding["ownership"]), load(binding["storage_reservation"])
    c.require(binding["status"] == c.BINDING_STATUS and binding["owner"] == "Root"
              and binding["nonce"] == cfg["nonce"] == claim["nonce"]
              and claim["job_id"] == binding["job_id"] and claim["owner"] == "Root"
              and claim["scope"] == c.SCOPE and claim["configuration"] == binding["configuration"]
              and claim["training_configuration"] == cfg["training_configuration"]
              and claim["no_live_duplicate_checkpoint_evaluation"] is True,
              "Exact one-owner Root function; never reuse S1/START ownership")
    c.require(grant["configuration"] == binding["configuration"] and grant["output"] == cfg["output"]
              and grant["reserved_bytes"] == cfg["output_limit_bytes"]
              and grant["final_checkpoint_retained_for_evaluation"] is True
              and grant["checkpoint_retention"] == cfg["checkpoint_retention"],
              "Actual result reservation and final-checkpoint retention")
    return cfg, base


def current_storage(cfg, grant, used, statvfs=os.statvfs):
    other, inodes = grant["other_outstanding_reserved_bytes"], grant["reserved_inodes"]
    other_inodes = grant["other_outstanding_reserved_inodes"]
    c.require(type(other) is int and other >= 0 and type(inodes) is int and inodes > 0
              and type(other_inodes) is int and other_inodes >= 0
              and type(used) is int and 0 <= used <= cfg["output_limit_bytes"],
              "Explicit current result/collective storage accounting")
    fs = statvfs(cfg["output"])
    required = cfg["output_limit_bytes"] - used + other
    c.require(fs.f_bavail * fs.f_frsize >= required and fs.f_favail >= inodes + other_inodes,
              "Current evaluation bytes/inodes insufficient; preserve all paid QA records")
    return dict(available_bytes=fs.f_bavail * fs.f_frsize, available_inodes=fs.f_favail,
                required_remaining_bytes=required, reservation_other_bytes=other,
                this_is_not_a_global_uncommitted_capacity_grant=True)


def prior_attempt(binding, actual, out, load=u.checked, describe=u.desc):
    if not actual["restart"]:
        return None
    prior = out / ("r" + str(actual["restart"] - 1))
    previous = load(describe(prior / "RESULT.json"))
    intent = load(describe(prior / "REQUEUE_INTENT.json"))
    c.require(previous["status"] == "ACCEPTED_QA_PROGRESS_REQUIRES_SAME_JOB_CONTINUATION"
              and previous["new_answers"] > 0 and previous["configuration"] == binding["configuration"]
              and previous["actual"]["job_id"] == actual["job_id"]
              and previous["actual"]["nonce"] == actual["nonce"]
              and previous["actual"]["restart"] == actual["restart"] - 1
              and intent["next_restart"] == actual["restart"]
              and intent["job_id"] == actual["job_id"] and intent["nonce"] == actual["nonce"]
              and intent["configuration"] == binding["configuration"]
              and intent["previous_result"] == describe(prior / "RESULT.json")
              and not (prior / "FAILURE.json").exists(), "Failed/foreign predecessor cannot auto-resume")
    return previous


def actor(binding, cfg):
    job, restart = binding["job_id"], os.environ.get("SLURM_RESTART_COUNT", "0")
    c.require(os.getuid() == 2708899 and job.isdecimal() and os.environ.get("SLURM_JOB_ID") == job
              and restart.isdecimal() and int(restart) < cfg["max_physical_attempts"]
              and not os.environ.get("SLURM_ARRAY_JOB_ID") and ROOT == c.SOURCE_ROOT
              and Path(sys.executable).absolute() == Path(PYTHON),
              "Own current finite scalar evaluator actor")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True, timeout=30)
    fields = u.fields(raw)
    expected = dict(JobId=job, JobState="RUNNING", JobName=c.NAME, UserId="luyangz(2708899)",
        Partition="general", Account="cx", QOS="normal", NumCPUs="8", NumTasks="1", NumNodes="1",
        Restarts=restart, TimeLimit="06:00:00", TresPerNode="gres/gpu:L40S:1",
        Command=str(HERE / "run.sbatch"), WorkDir=str(ROOT), Comment=c.COMMENT + cfg["nonce"],
        NodeList=socket.gethostname().split(".")[0])
    c.require(all(fields.get(key) == value for key, value in expected.items()), "Current evaluation actor differs")
    tres = dict(value.split("=", 1) for value in fields["AllocTRES"].split(",") if "=" in value)
    c.require(tres.get("cpu") == "8" and tres.get("mem") in ("64G", "65536M")
              and tres.get("gres/gpu") == "1", "Current one L40S/CPU8/64GiB required")
    c.require(subprocess.check_output(["scontrol", "write", "batch_script", job, "-"], timeout=30)
              == (HERE / "run.sbatch").read_bytes(), "Own exact frozen spool")
    c.require(os.environ.get("TZ") == "America/New_York", "Exact scheduler timezone")
    time.tzset()
    return dict(job_id=job, restart=int(restart), node=fields["NodeList"], nonce=cfg["nonce"], scheduler=raw,
        end_epoch=int(time.mktime(time.strptime(fields["EndTime"], "%Y-%m-%dT%H:%M:%S"))),
        process_nonce=uuid.uuid4().hex, pid=os.getpid())


def function(cfg, identity, roster, complete, bank):
    value = original.function(cfg, identity, roster, complete, bank)
    return dict(value, schema="root264-comparator-final-four-bank-function-v1", role=c.SCOPE,
                training_logical_protocol=historical.student.PACKAGE,
                physical_extension_protocol=historical.route.PACKAGE)


def strict_loaded_support(proof):
    values, injection = proof["portable_value_verification"], proof["native_adapter_construction"]
    c.require(values["exact_parameter_inventory"] is True and values["frozen_parameters"] == 682
              and values["trained_parameters"] == 452 and values["LoRA_tensors"] == 448
              and values["projector_tensors"] == 4 and values["portable_trainable_values_exact"] is True
              and len(injection["language_adapter_targets"]) == 224
              and len(injection["loaded_vision_generic_targets_excluded"]) == 72
              and proof["persistent_buffer_verification"]["exact_persistent_buffer_support"] is True
              and proof["persistent_buffer_verification"]["persistent_buffer_values_exact"] is True,
              "Every final checkpoint must reconstruct full1134/452/224/72 support and buffers")


def run(binding_path):
    binding = u.checked(u.desc(binding_path))
    cfg, base = documents(binding)
    actual, began = actor(binding, cfg), time.monotonic()
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out / "EXECUTION.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    attempt = out / ("r" + str(actual["restart"]))
    try:
        attempt.mkdir(exist_ok=False)
    except OSError:
        os.close(fd)
        raise
    try:
        previous = prior_attempt(binding, actual, out)
        grant = u.checked(binding["storage_reservation"])
        u.publish(attempt / "CURRENT_STORAGE.json", current_storage(cfg, grant, budget(cfg)))
        _, complete_ref, complete, physical_proofs = historical.complete_student(cfg)
        u.publish(attempt / "START.json", dict(actual=actual, configuration=binding["configuration"],
                  student_completion=complete_ref, historical_physical_proofs=physical_proofs))
        bankdata = {bank: banks.load(bank, cfg) for bank in cfg["banks"]}
        ready = {}
        for bank, (roster, inventory, _, physical, snapshot) in bankdata.items():
            store = media.Store(physical, roster["records"], inventory, snapshot)
            try:
                names = sorted({row["native_input"]["image"] for row in roster["records"]})
                for name in names:
                    image = store.rgb(name)
                    image.close()
                ready[bank] = dict(questions=len(roster["records"]), all_encoded_and_RGB_verified=True,
                                   images=len(names), forwards=0)
            finally:
                store.close()
        u.publish(attempt / "FULL_MEDIA_READINESS.json", ready)
        publish_same(out / "STUDENT_INPUT.json", dict(training_configuration=cfg["training_configuration"],
                     student_completion=complete_ref, accepted=complete["accepted"], physical_proofs=physical_proofs))
        runtime = qualified.official.runtime(base, attempt)
        u.publish(attempt / "RUNTIME.json", runtime)
        with qualified.official.offline():
            loaded3, proof = model.compose(complete["accepted"], cfg["training_configuration"])
            strict_loaded_support(proof)
            identity = model.identity(loaded3, proof, runtime)
            identity_ref = publish_same(out / "MODEL_IDENTITY.json", identity)
            u.publish(attempt / "MODEL_RECONSTRUCTION.json", proof)
            loaded = (*loaded3, getattr(loaded3[1].config, "max_sequence_length", 2048))
            reports, new = {}, 0
            for bank, (roster, inventory, targets, physical, snapshot) in bankdata.items():
                rows, path = roster["records"], out / bank
                path.mkdir(exist_ok=True)
                fn = function(cfg, identity, roster, complete_ref, bank)
                publish_same(path / "FUNCTION.json", fn)
                journal = generation.AuditedJournal(path / "journal", fn, rows)
                store = media.Store(physical, rows, inventory, snapshot)
                try:
                    for entry in qualified.official.entries(journal, rows):
                        generation.validate_entry(bank, entry, journal.roster[entry["semantic_key"]], fn)
                    accepted_prefix_count(previous, bank, len(journal.present))
                    before = len(journal.present)
                    if time.time() < actual["end_epoch"] - 300:
                        generation.native(bank, cfg, rows, journal, loaded, store, actual,
                                          time.monotonic() + max(0, actual["end_epoch"] - time.time() - 240))
                    new += len(journal.present) - before
                    reports[bank] = dict(questions=len(rows), accepted=len(journal.present),
                                        missing=len(journal.missing()), new=len(journal.present) - before)
                    if not journal.missing():
                        raw = journal.export(path / "PREDICTIONS.jsonl")
                        score = banks.score(bank, roster, journal, targets)
                        publish_same(path / "SCORE.json", dict(
                            status="COMPLETE_ROOT_COMPARATOR_NATIVE_BANK_AND_OFFICIAL_SCORING",
                            bank=bank, function=fn, predictions=raw, score=score,
                            student_completion=complete_ref, complete_questions=len(rows),
                            scientific_improvement_demonstrated=False))
                finally:
                    journal.close()
                    store.close()
                budget(cfg)
        done = accepted_progress(previous, reports, new)
        value = dict(status="COMPLETE_ROOT_COMPARATOR_FOUR_OFFICIAL_BANKS" if done else
                     "ACCEPTED_QA_PROGRESS_REQUIRES_SAME_JOB_CONTINUATION", actual=actual,
                     configuration=binding["configuration"], student_completion=complete_ref,
                     accepted_checkpoint=complete["accepted"], model_identity=identity_ref, banks=reports,
                     new_answers=new, elapsed_seconds=time.monotonic() - began,
                     all_prior_and_current_costs_count=True, scientific_improvement_demonstrated=False)
        result_ref = u.publish(attempt / "RESULT.json", value)
        if done:
            publish_same(out / "COMPLETE.json", value)
        else:
            c.require(new > 0 and actual["restart"] + 1 < cfg["max_physical_attempts"],
                      "No progress or QA-derived attempt allowance exhausted")
            intent = u.publish(attempt / "REQUEUE_INTENT.json", dict(job_id=actual["job_id"],
                nonce=actual["nonce"], next_restart=actual["restart"] + 1, previous_result=result_ref,
                configuration=binding["configuration"], same_job_only=True, accepted_per_QA_journal=True))
            result = subprocess.run(["scontrol", "requeue", actual["job_id"]], text=True,
                                    capture_output=True, timeout=30)
            u.publish(attempt / "REQUEUE_ACK.json", dict(intent=intent, returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr, ambiguous_ack_requires_Root_reconciliation=True))
            c.require(result.returncode == 0, "Ambiguous/failed continuation; never duplicate")
    except BaseException as error:
        u.publish(attempt / "FAILURE.json", dict(status="ROOT_COMPARATOR_FAILED_PRESERVE_QAS_NO_AUTO_RETRY",
                  actual=actual, error=repr(error), elapsed_seconds=time.monotonic() - began))
        raise
    finally:
        os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    run(parser.parse_args().binding)
