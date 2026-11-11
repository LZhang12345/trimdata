"""Four-bank evaluations of Tool3 COINCIDE budget-curve producers with the Tool1 b6 evaluator (t3coin v2).

v2 (2026-11-06) is t3coin v1 with two repairs from the independent review of v1 (v1 was never submitted; its
coincide-b05-s17 plan is preserved under t3-coincide-budget-eval-v1): model.py binds the Tool1 trainable-support
reference to `support` instead of the name `reference` that the unchanged inference loops rebind (v1 would raise
KeyError before inference), and the held submission requires the Tool3 RESULT job accounting COMPLETED 0:0 (as
eval v5 does for completed producers). New control root t3-coincide-budget-eval-v2. Text below is the v1
description.


Derived from helper_tool1_baseline_eval_v4_20261105 / _v5_20261106 (the controllers of Tool1's dsmm-t1e-b6-* budget
evaluations). Everything that defines the measurement is unchanged: the native Python, the v2 inference body
(model.py differs from helper_tool1_baseline_eval_v2_20261102/model.py only in producer custody), the four full
official banks (POPE 8910, GQA 12578, TextVQA 5000, ScienceQA-image 2017 = 28,505 QA), the frozen bank/scoring
reference, decoding, generation bridge, official scorers including GQA, the output root, the General route
(account cx, partition general, QoS normal, gres gpu:L40S:1, exclude babel-p9-20, 8 CPUs, 64G, 12 h, no requeue),
held submission under the shared ADMISSION.lock with INTENT/ACK/HELD_READBACK, and release only after an
independent exact source/config PASS bound to the plan.

Only the producer custody differs (Tool3 read-only custody adapter):
  * producers are the six registered Tool3 chain cells coincide-b05/b20-s17/29/43 (baseline-train-v3); no Tool1
    roster exists for them. The plan binds Tool3's own receipts in its control folder (frozen CELLS.json entry,
    the unique LINK_EXIT with status RESULT_SUCCESS and a FINAL_CHUNK at end_step == max_steps == natural steps,
    the link MODEL_CONFIG, the job SUBMIT_ACK, the CHAIN_AMENDMENT_V3_2 route receipt) and a read-only GPU-node
    ATTESTATION (`attest`, run in an srun --overlap step) that re-hashed the Tool3 RESULT.json and every output
    file it lists. The worker re-hashes RESULT.json and all listed files on its own node before any inference,
    and model.py re-hashes the listed files again.
  * the producer has completed when the plan is made, so no dependency is bound (as in eval v5).
  * the 452 native trainable names come from a sealed Tool1 b6 MODEL_CONFIG; the Tool3 frozen base assets and the
    saved model config.json must equal that reference.
  * disclosed differences: Tool3 K = int(N*budget) = 33,038 (b05) / 132,152 (b20) vs Tool1 b6 33,032 / 132,147;
    COINCIDE native operator v3 selection has no selection seed; Tool3 chain trainer source (resumable chunks).
  * --nice=200 (author-delegated budget-curve priority) is part of the submitted argv.
Tool3 files, Tool3 jobs and Tool1 b6 jobs are never modified by this controller.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import time
from types import ModuleType


HOME = Path("/home/luyangz/data_selection_multimodal")
SOURCE = HOME / "experiments/helper_tool1_baseline_eval_t3coin_v2_20261106"
CONTROL_ROOT = HOME / "collaboration/helper-tool1-baseline-independent-20261101/t3-coincide-budget-eval-v2"
TOOL3_CONTROL = HOME / "collaboration/helper-tool3-baselines-20261104"
TOOL3_TRAINING = TOOL3_CONTROL / "training"
TOOL3_REGISTRY = TOOL3_CONTROL / "CELLS.json"
CHECKPOINT_ROOT = Path("/data/hf_cache/project_checkpoints/DS-MultiModal/tool3-baselines-20261104")
OUTPUT_ROOT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-baseline-eval-v2-20261102")
PYTHON = "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python"
OWNER = "Tool1-owned-published-baseline-full-official-evaluation"
PRODUCER_OWNER = "Tool3-owned-baseline-native-training"
REVIEW_STATUS = "OWNER_REVIEWED_TOOL1_T3COIN_FULL_EVAL_V2_NOT_INDEPENDENT_ACCEPTANCE"
INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_T3COIN_EVAL_V2"
DIAGNOSIS_SCHEMA = "tool1-t3coin-eval-failed-attempt-diagnosis-v2"
DIAGNOSIS_STATUS = "OWNER_DIAGNOSED_FAILED_EVALUATION_ATTEMPT"
DIAGNOSIS_REVIEW_STATUS = "PASS_INDEPENDENT_FAILED_EVALUATION_DIAGNOSIS_TOOL1_T3COIN_EVAL_V2"
ATTESTATION_SCHEMA = "tool1-t3coin-producer-custody-attestation-v2"
UID = 2708899
ROUTE = dict(account="cx", partition="general", qos="normal")
NICE = "200"
MAX_ATTEMPTS = 6
TERMINAL = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
INTERRUPTED = ("TIMEOUT", "PREEMPTED", "NODE_FAIL")
HELPER = {
    "path": str(HOME / "experiments/helper_method_eval_bank_qual_v1_20261102/prepare.py"),
    "bytes": 51200, "sha256": "f0dc4f6a548eedc45d5ba3d92e0251328d33bb82b274d815f14b0aca25fa7357",
}
TOOL1_INFERENCE_SOURCE = {"path": str(HOME / "experiments/helper_tool1_baseline_eval_v2_20261102/model.py"),
                          "bytes": 18681,
                          "sha256": "df783c9ab4d02b94782af82570ce839c80aaa813b8eb795ac1e37caa3ffb4e2a"}
MODEL_SOURCE = {"path": str(SOURCE / "model.py"), "bytes": 20488, "sha256": "f5f7598885af3503ed713b282899801c2fa5a3dac3c10351cdb14835f7da47b1"}
BANK_QUALIFICATION = {
    "path": str(HOME / "collaboration/helper-tool2-method-20261101/eval-bank-qual-v1/"
                "ACTUAL_INFERENCE_RESULT-11022388.json"),
    "bytes": 318676, "sha256": "bde9f9a4fd0efb2191670d153975daf663ff1990fb80c9609feef378cf01bc2a",
}
TRAINABLE_REFERENCE = {
    "path": str(HOME / "collaboration/helper-tool1-baseline-independent-20261101/baseline-eval-v4/"
                "b6-uniform-mk1235-b05-k33032-train43/attempt-0/MODEL_CONFIG-11068026.json"),
    "bytes": 734587, "sha256": "6ce0144f4b8bd44fcd87ae1cf907e1f53ffbaa5782d948d11bb1d2705b6b6054",
}
ROUTE_AMENDMENT = {
    "path": str(TOOL3_CONTROL / "CHAIN_AMENDMENT_V3_2.json"),
    "bytes": 1224, "sha256": "be38d2a21cf3b5d0b7d80b4ee5f78c6a9cb2d55113addd32f878f6f0121319b7",
}
PRODUCER_MODEL_SHA = "b0c099075001ee60a2547e7c3a72e917fddf0f657d42fbb9b2dd7731ff14814a"
PRODUCER_ROLE = "TOOL3_BASELINE_RESUMABLE_FULL_EPOCH_NOT_EVALUATED"
PRODUCER_STATUS = "ACTUAL_FULL_NATIVE_BASELINE_EPOCH_NOT_EVALUATED"
AUDIT_CHECKS = ("chunks_tile_all_steps", "deterministic_attention_executed", "exposure_exact_once", "final_step",
                "microbatches")
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors", "config.json", "non_lora_trainables.bin")
TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
ROLE = "TOOL1_BASELINE_FULL_EPOCH_FOUR_OFFICIAL_BANK_EVALUATION"
RESULT_STATUS = "COMPLETE_TOOL1_BASELINE_FOUR_OFFICIAL_BANKS"
COMMENT_PREFIX = "dsmm-t1-t3coin-eval-v2:"
TIME = "12:00:00"
BANKS = {"POPE": 8910, "GQA": 12578, "TextVQA": 5000, "ScienceQA_image": 2017}


def _cell(budget, seed):
    k, comparator = {"b05": (33038, 33032), "b20": (132152, 132147)}[budget]
    return dict(budget=budget, K=k, training_seed=seed, steps=(k + 3) // 4 // 32, tool1_b6_K=comparator,
                short="coin-%s-s%d" % (budget, seed))


CELLS = {"coincide-%s-s%d" % (budget, seed): _cell(budget, seed)
         for budget in ("b05", "b20") for seed in (17, 29, 43)}


class NotReady(Exception):
    """The Tool3 producer has no RESULT_SUCCESS link yet; nothing may be planned or submitted."""


def require(value, message):
    if not value:
        raise ValueError(message)


def cell_spec(cell):
    require(isinstance(cell, str) and cell in CELLS, "One of the six registered Tool3 COINCIDE budget cells required")
    return CELLS[cell]


def load_helpers():
    path = Path(HELPER["path"])
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = next_fd
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(HELPER["bytes"] + 1)
        after = os.fstat(stream.fileno())
    require(stat.S_ISREG(before.st_mode) and before.st_uid == 2708899
            and len(raw) == before.st_size == HELPER["bytes"]
            and hashlib.sha256(raw).hexdigest() == HELPER["sha256"]
            and all(getattr(before, field) == getattr(after, field)
                    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
            "Exact immutable executed metadata helper required")
    module = ModuleType("tool1_frozen_evaluation_metadata_helpers_t3coin_v2")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    return module


# ---------------------------------------------------------------- Tool3 read-only custody adapter

def data_file(p, path, limit, keep=False):
    """Hash one regular project-user file on the shared HF NAS through a stable literal FD (read-only)."""
    fd = p.open_literal(Path(path), os.O_RDONLY)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == UID and 0 < before.st_size <= limit,
                "Bounded regular project-user producer file required: " + str(path))
        mount = p.mount_for_fd(fd, before.st_dev, "group")
        digest, size, kept = hashlib.sha256(), 0, []
        while True:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            digest.update(block)
            size += len(block)
            if keep:
                kept.append(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    require(size == before.st_size and all(getattr(before, key) == getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
            "Producer file changed while hashed: " + str(path))
    ref = dict(path=str(path), bytes=size, sha256=digest.hexdigest(), inode=before.st_ino, mount=mount)
    return (b"".join(kept) if keep else None), ref


def rehash_outputs(p, cell, listed):
    require(isinstance(listed, dict) and set(REQUIRED_FILES) <= set(listed)
            and all(re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in listed),
            "Tool3 RESULT must list the native adapter/projector/config output files")
    training = CHECKPOINT_ROOT / cell / "training"
    found = {}
    for name in sorted(listed):
        _, ref = data_file(p, training / name, 8 << 30)
        found[name] = dict(bytes=ref["bytes"], sha256=ref["sha256"])
    return found


def attest(p, cell):
    """Read-only GPU-node custody attestation of one Tool3 producer (run in an srun --overlap step)."""
    cell_spec(cell)
    node = socket.gethostname().split(".")[0]
    require(os.getuid() == UID and p.valid_singleton_node(node)
            and os.environ.get("SLURM_JOB_ID", "").isdecimal()
            and os.environ.get("SLURM_STEP_ID", "").isdecimal(), "Run inside an srun step on a Babel compute node")
    output = CHECKPOINT_ROOT / cell
    raw, result_ref = data_file(p, output / "RESULT.json", 64 << 20, keep=True)
    result = json.loads(raw)
    listed = result["output_files"]
    rehashed = rehash_outputs(p, cell, listed)
    adapter_raw, _ = data_file(p, output / "training" / "adapter_config.json", 1 << 20, keep=True)
    return dict(
        schema=ATTESTATION_SCHEMA, cell=cell, node=node, slurm_job_id=os.environ["SLURM_JOB_ID"],
        slurm_step_id=os.environ["SLURM_STEP_ID"], uid=os.getuid(),
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), controller=p.descriptor(SOURCE / "prepare.py"),
        result=result_ref, status=result["status"], job_id=result["job_id"], link_id=result["config"]["link_id"],
        full_epoch_training_complete=result["full_epoch_training_complete"], audit_checks=result["audit_checks"],
        overflow_steps=result["overflow_steps"],
        config_sha256=hashlib.sha256(p.canonical(result["config"])).hexdigest(),
        output_files_listed=listed, output_files_rehashed=rehashed,
        training_entries=sorted(os.listdir(output / "training")), adapter_config=json.loads(adapter_raw),
        read_only=True)


def link_exits(p, cell):
    """Tool3's own LINK_EXIT receipts of the cell with status RESULT_SUCCESS (normally exactly one)."""
    folder = TOOL3_TRAINING / cell
    found = []
    for path in sorted(folder.glob("LINK_EXIT-*.json")):
        match = re.fullmatch(r"LINK_EXIT-([1-9][0-9]*)-r([0-9]+)\.json", path.name)
        if match is None:
            continue
        ref = p.descriptor(path)
        value = json.loads(p.checked(ref))
        if value.get("status") == "RESULT_SUCCESS":
            found.append((match[1], match[1] + "-r" + match[2], ref, value))
    return found


def tool3_receipts(p, cell, spec):
    found = link_exits(p, cell)
    if not found:
        raise NotReady("No Tool3 RESULT_SUCCESS link for " + cell)
    require(len(found) == 1, "Ambiguous: several Tool3 RESULT_SUCCESS links; ask Tool3 (QUESTIONS_FOR_TOOL3.json)")
    job, link, exit_ref, value = found[0]
    chunk = value.get("chunk_record") or {}
    require(value.get("returncode") == 0 and value.get("result_exists") is True and value.get("progress") is True
            and chunk.get("status") == "FINAL_CHUNK" and chunk.get("job_id") == job and chunk.get("link_id") == link
            and chunk.get("end_step") == chunk.get("max_steps") == spec["steps"]
            and spec["steps"] in (chunk.get("saved_steps") or []),
            "Tool3 RESULT_SUCCESS link must end the full natural horizon (end_step == max_steps == steps)")
    folder = TOOL3_TRAINING / cell
    config_ref = p.descriptor(folder / ("MODEL_CONFIG-%s.json" % link))
    ack_ref = p.descriptor(folder / ("SUBMIT_ACK-%s.json" % job))
    ack = json.loads(p.checked(ack_ref))
    require(ack.get("job_id") == job and ack.get("returncode") == 0
            and ack.get("stdout", "").split(";")[0].strip() == job, "Tool3 SUBMIT_ACK of the RESULT job required")
    return dict(job=job, link=link, link_exit=exit_ref, model_config=config_ref, submit_ack=ack_ref,
                end_step=chunk["end_step"], max_steps=chunk["max_steps"])


def frozen_registry(p, cell):
    ref = p.descriptor(TOOL3_REGISTRY)
    registry = json.loads(p.checked(ref))
    require(registry.get("schema") == "tool3-baseline-cells-v1" and cell in registry.get("cells", {}),
            "Tool3 registry entry required")
    return dict(registry=ref, registry_entry=registry["cells"][cell])


def validate_attestation(p, cell, attestation_ref, receipts, model_config, reference):
    value = json.loads(p.checked(attestation_ref))
    listed = value.get("output_files_listed")
    adapter = value.get("adapter_config") or {}
    require(Path(attestation_ref["path"]).parent == CONTROL_ROOT / cell
            and re.fullmatch(r"ATTESTATION-[0-9TZ]+\.json", Path(attestation_ref["path"]).name) is not None
            and value.get("schema") == ATTESTATION_SCHEMA and value.get("cell") == cell
            and value.get("read_only") is True and value.get("uid") == UID
            and value.get("controller") == p.descriptor(SOURCE / "prepare.py")
            and p.valid_singleton_node(value.get("node"))
            and value["result"]["path"] == str(CHECKPOINT_ROOT / cell / "RESULT.json")
            and value["result"]["mount"]["target"] == "/data/hf_cache"
            and value.get("status") == PRODUCER_STATUS and value.get("job_id") == receipts["job"]
            and value.get("link_id") == receipts["link"] and value.get("full_epoch_training_complete") is True
            and value.get("audit_checks") == {name: True for name in AUDIT_CHECKS}
            and value.get("config_sha256") == hashlib.sha256(p.canonical(model_config)).hexdigest()
            and isinstance(listed, dict) and set(REQUIRED_FILES) <= set(listed)
            and value.get("output_files_rehashed") == listed
            and value.get("training_entries") == sorted(listed),
            "Exact read-only GPU-node attestation of the Tool3 RESULT and every listed output file required")
    require(adapter.get("r") == 128 and adapter.get("lora_alpha") == 256 and adapter.get("lora_dropout") == .05
            and adapter.get("bias") == "none" and adapter.get("peft_type") == "LORA"
            and adapter.get("task_type") == "CAUSAL_LM" and set(adapter.get("target_modules") or []) == TARGET_MODULES
            and len(adapter["target_modules"]) == len(TARGET_MODULES)
            and adapter.get("base_model_name_or_path") == model_config["assets"]["vicuna"]["snapshot"],
            "Native LoRA configuration of the Tool1 recipe required (target-module order is not semantic)")
    require(listed["config.json"] == reference["output_files"]["config.json"]
            and all(model_config["assets"][role] == reference["config"]["assets"][role]
                    for role in ("vicuna", "clip336", "stage1_projector")),
            "Tool3 frozen base assets and saved model config.json must equal the Tool1 b6 reference")
    return value


def make_plan(p, cell, attestation_path, frozen=None):
    spec = cell_spec(cell)
    receipts = tool3_receipts(p, cell, spec)
    frozen = frozen_registry(p, cell) if frozen is None else frozen
    entry = frozen["registry_entry"]
    model_config = json.loads(p.checked(receipts["model_config"]))
    p.checked(ROUTE_AMENDMENT)
    require(entry.get("K") == spec["K"] and entry.get("training_seed") == spec["training_seed"]
            and entry.get("short") == spec["short"] and entry.get("route") in ("general", "general_budget")
            and dict(model_config.get("registry_entry") or {}, route=None) == dict(entry, route=None)
            and model_config.get("cell") == cell and model_config.get("job_id") == receipts["job"]
            and model_config.get("link_id") == receipts["link"] and model_config.get("K") == spec["K"]
            and model_config.get("training_seed") == spec["training_seed"]
            and model_config.get("role") == PRODUCER_ROLE and model_config.get("input") == entry["records"]
            and model_config.get("model_source", {}).get("sha256") == PRODUCER_MODEL_SHA
            and model_config.get("output") == str(CHECKPOINT_ROOT / cell),
            "Tool3 link config must be the registered full-epoch cell (route is the only registry change)")
    reference = json.loads(p.checked(TRAINABLE_REFERENCE))["qualification"]
    attestation_ref = p.descriptor(Path(attestation_path))
    attestation = validate_attestation(p, cell, attestation_ref, receipts, model_config, reference)
    producer = dict(cell=cell, job_id=receipts["job"], K=spec["K"], training_seed=spec["training_seed"],
                    steps=spec["steps"], model_source_sha256=PRODUCER_MODEL_SHA)
    custody = dict(
        owner=PRODUCER_OWNER, read_only=True, registry=frozen["registry"], registry_entry=entry,
        route_amendment=ROUTE_AMENDMENT, link_exit=receipts["link_exit"], link_id=receipts["link"],
        end_step=receipts["end_step"], max_steps=receipts["max_steps"], model_config=receipts["model_config"],
        submit_ack=receipts["submit_ack"], attestation=attestation_ref,
        result={key: attestation["result"][key] for key in ("path", "bytes", "sha256")},
        output_files=attestation["output_files_listed"])
    function = dict(
        owner=OWNER, cell=cell, method=entry["method"], selection_source=entry.get("selection_source"),
        selection_seed=None, training_seed=spec["training_seed"], K=spec["K"], budget=spec["budget"],
        producer=producer, producer_custody=custody,
        disclosed_differences=dict(
            K="Tool3 K=int(N*budget)=%d vs Tool1 b6 K=%d (same 660,763-record pool)" % (spec["K"], spec["tool1_b6_K"]),
            selection_seed="COINCIDE native operator v3 selection is seedless; Tool1 b6 cells use selection seeds",
            trainer="Tool3 baseline-train-v3 resumable chunked trainer (same native recipe/assets); not Tool1's"),
        tool1_b6_inference_source=TOOL1_INFERENCE_SOURCE, trainable_reference=TRAINABLE_REFERENCE,
        prior_executed_bank_qualification=BANK_QUALIFICATION, metadata_helper=HELPER,
        model_source=MODEL_SOURCE, native_python=PYTHON, full_QA=28505, banks=BANKS,
        GQA_main="UNCHANGED_NATIVE_NORMALIZED_SCORE", no_model_downloads=True, no_base_copies=True,
        no_training_updates=True, no_tool3_writes=True, no_Root_acceptance_or_grant_transfer=True)
    return dict(
        schema="tool1-t3coin-budget-evaluations-v2", owner=OWNER, cell=cell, function=function,
        function_identity=hashlib.sha256(p.canonical(function)).hexdigest(),
        sources={"prepare.py": p.descriptor(SOURCE / "prepare.py"), "model.py": MODEL_SOURCE,
                 "run.sbatch": p.descriptor(SOURCE / "run.sbatch")},
        job_name="dsmm-t1e-t3" + spec["short"], dependency=None,
        resource=dict(account=ROUTE["account"], partition=ROUTE["partition"], qos=ROUTE["qos"], nice=int(NICE),
                      gpu="L40S", GPUs=1, CPUs=8, memory="64G", time=TIME, exclude="babel-p9-20"),
        automatic_requeue=False, independently_reviewed=False)


# ---------------------------------------------------------------- reviews and contract

def independent_path(cell):
    cell_spec(cell)
    return CONTROL_ROOT / ("INDEPENDENT_REVIEW-%s.json" % cell)


def validate_independent(review, plan_ref, plan, cell):
    require(review.get("status") == INDEPENDENT_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER, PRODUCER_OWNER)
            and review.get("blocking_findings") == [] and review.get("sources") == plan["sources"]
            and isinstance(review.get("plans"), dict) and review["plans"].get(cell) == plan_ref,
            "Independent exact source/config PASS bound to this evaluation plan is required")


def custody_frozen(plan):
    custody = plan["function"]["producer_custody"]
    return dict(registry=custody["registry"], registry_entry=custody["registry_entry"])


def contract(p, cell, independent=False):
    folder = CONTROL_ROOT / cell
    info = os.stat(folder, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == 2708899 and not info.st_mode & 0o022,
            "Private owned evaluation control directory required")
    plan_ref = p.descriptor(folder / "PLAN.json")
    plan = json.loads(p.checked(plan_ref))
    custody = plan["function"]["producer_custody"]
    require(plan == make_plan(p, cell, custody["attestation"]["path"], custody_frozen(plan)),
            "Exact source/config/producer-bound evaluation plan required")
    review_ref = p.descriptor(folder / "OWNER_REVIEW.json")
    review = json.loads(p.checked(review_ref))
    require(review["status"] == REVIEW_STATUS and review["independent"] is False
            and review["plan"] == plan_ref and review["sources"] == plan["sources"]
            and review["blocking_findings"] == [] and review["cell"] == cell,
            "Truthful exact private-owner review required; no independent PASS claimed")
    qualification = json.loads(p.checked(BANK_QUALIFICATION))
    require(qualification["job_id"] == "11022388" and qualification["actual_scoped_native_QA"] == 8
            and qualification["native_generation_bridge"]["calls"] == 8
            and all(value["accepted_scoped_QA"] == 2 for value in qualification["banks"].values())
            and qualification["score_computed"] is False,
            "Actual eight-QA native bank integration prerequisite required")
    p.checked(MODEL_SOURCE)
    p.checked(TOOL1_INFERENCE_SOURCE)
    independent_ref = None
    if independent:
        independent_ref = p.descriptor(independent_path(cell))
        validate_independent(json.loads(p.checked(independent_ref)), plan_ref, plan, cell)
    return folder, plan_ref, plan, review_ref, independent_ref


def private_directory(path):
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.stat(path, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == 2708899 and not info.st_mode & 0o022,
            "Owned private control directory required: " + str(path))


def make_plan_file(p, cell, attestation_path):
    cell_spec(cell)
    private_directory(CONTROL_ROOT)
    private_directory(CONTROL_ROOT / cell)
    path = CONTROL_ROOT / cell / "PLAN.json"
    require(not path.exists(), "Existing evaluation plan is immutable; use a new version")
    p.checked(MODEL_SOURCE)
    return p.seal(path, make_plan(p, cell, attestation_path))


def owner_review(p, cell, notes):
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete self-review notes required")
    plan_ref = p.descriptor(CONTROL_ROOT / cell / "PLAN.json")
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, cell, plan["function"]["producer_custody"]["attestation"]["path"],
                              custody_frozen(plan)), "Self-review must bind the current exact plan")
    return p.seal(CONTROL_ROOT / cell / "OWNER_REVIEW.json", dict(
        status=REVIEW_STATUS, independent=False, ai_review_disclosed=True, reviewer=OWNER, cell=cell,
        plan=plan_ref, sources=plan["sources"], function_identity=plan["function_identity"],
        blocking_findings=[], notes=notes))


# ---------------------------------------------------------------- scheduler

def accounting(p, job):
    value = p.query(["/usr/bin/sacct", "-X", "-nP", "-j", job,
                     "--format=JobIDRaw,State,ExitCode,Start,Elapsed"]).strip().split("|")
    require(len(value) == 5 and value[0] == job, "Exact scheduler accounting record required")
    return dict(job_id=job, state=value[1].split(" ")[0], exit_code=value[2], start=value[3], elapsed=value[4])


def check_attempt(attempt):
    require(type(attempt) is int and 0 <= attempt < MAX_ATTEMPTS, "Finite evaluation attempt index required")
    return attempt


def comment(plan, attempt):
    return COMMENT_PREFIX + plan["function_identity"] + ":a%d" % check_attempt(attempt)


def argv(cell, plan, attempt):
    require(plan["dependency"] is None, "Completed Tool3 producers bind no dependency")
    return ["/usr/bin/sbatch", "--parsable", "--hold", "--no-requeue", "--export=NIL",
            "--account=" + ROUTE["account"], "--partition=" + ROUTE["partition"], "--qos=" + ROUTE["qos"],
            "--nice=" + NICE, "--gres=gpu:L40S:1",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G", "--time=" + TIME,
            "--exclude=babel-p9-20", "--job-name=" + plan["job_name"], "--comment=" + comment(plan, attempt),
            "--chdir=" + str(HOME), "--input=/dev/null", "--output=/dev/null", "--error=/dev/null",
            str(SOURCE / "run.sbatch"), cell, str(check_attempt(attempt))]


def scheduler(p, cell, plan, job, attempt, held=False, running=False):
    raw = p.query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = p.fields(raw)
    expected = dict(
        JobId=job, UserId="luyangz(2708899)", Account=ROUTE["account"], Partition=ROUTE["partition"],
        QOS=ROUTE["qos"], JobName=plan["job_name"], Comment=comment(plan, attempt),
        NumCPUs="8", NumTasks="1", TresPerNode="gres/gpu:L40S:1", TimeLimit=TIME,
        ExcNodeList="babel-p9-20", Requeue="0", Command=str(SOURCE / "run.sbatch"), WorkDir=str(HOME),
        StdIn="/dev/null", StdOut="/dev/null", StdErr="/dev/null", Dependency="(null)")
    if not running:
        expected["Nice"] = NICE  # later author priority renices are not bound once released
    nodes = row.get("NumNodes") == "1" if running else row.get("NumNodes") in ("1", "1-1")
    require(all(row.get(key) == value for key, value in expected.items()) and nodes
            and shlex.split(row.get("SubmitLine", "")) == argv(cell, plan, attempt),
            "Actual physical evaluation request/source/resources differ")
    if held:
        require(row.get("JobState") == "PENDING" and row.get("Reason") == "JobHeldUser"
                and row.get("RunTime") == "00:00:00", "Verified never-started held request required")
    if running:
        require(row.get("JobState") == "RUNNING" and row.get("Restarts") == "0"
                and row.get("NodeList") == socket.gethostname().split(".")[0]
                and p.valid_singleton_node(row["NodeList"]), "Actual owned ready running allocation required")
    tres = dict(value.split("=", 1) for value in row.get("AllocTRES" if running else "ReqTRES", "").split(",")
                if "=" in value)
    require(tres.get("cpu") == "8" and tres.get("mem") in ("64G", "65536M")
            and tres.get("gres/gpu") == "1", "Actual total allocation/request resources differ")
    spool = subprocess.check_output(["/usr/bin/scontrol", "write", "batch_script", job, "-"],
                                    timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
    require(spool == p.checked(plan["sources"]["run.sbatch"]), "Exact immutable bound spool required")
    return raw, spool


def producer_terminal(p, plan):
    record = accounting(p, plan["function"]["producer"]["job_id"])
    require(record["state"] == "COMPLETED" and record["exit_code"] == "0:0",
            "Tool3 RESULT job must account COMPLETED 0:0 (otherwise ask Tool3; QUESTIONS_FOR_TOOL3.json)")
    return record


def general_census(p, plan):
    queue = p.query(["/usr/bin/squeue", "-u", "luyangz", "-r", "-h", "-o", "%i|%q|%k|%j|%T|%b"])
    rows = [line.split("|") for line in queue.splitlines() if line.strip()]
    require(all(len(row) == 6 for row in rows)
            and not any(plan["function_identity"] in row[2] or row[3] == plan["job_name"] for row in rows),
            "Expanded census invalid or exact live function already exists")
    general = p.query(["/usr/bin/sacctmgr", "-nP", "show", "qos", "normal",
                       "format=Name,MaxSubmitPU,MaxTRESPU"]).strip().split("|")
    gpu_limit = dict(x.split("=", 1) for x in general[2].split(",") if "=" in x).get("gres/gpu", "") \
        if len(general) == 3 else ""
    require(len(general) == 3 and general[0] == "normal" and general[1].isdigit() and gpu_limit.isdigit(),
            "Actual General QoS limits required")
    normal = [row for row in rows if row[1] == "normal"]
    census = dict(requests=len(normal), running_gpu=sum(row[4] == "RUNNING" and "gpu" in row[5] for row in normal),
                  pending=sum(row[4] == "PENDING" for row in normal), limits=general, expanded_requests=len(rows))
    require(census["requests"] < int(general[1]), "General route needs a free normal-QoS submission slot")
    return census


# ---------------------------------------------------------------- attempts

def diagnosed_failure(p, folder, ack_ref, ack, record, log_path):
    """Owner DIAGNOSIS + independent DIAGNOSIS_REVIEW of a FAILED t3coin evaluation attempt."""
    job = ack["job_id"]
    diagnosis_ref = p.descriptor(folder / ("DIAGNOSIS-%s.json" % job))
    diagnosis = json.loads(p.checked(diagnosis_ref))
    evidence, scheduled = diagnosis.get("evidence") or {}, diagnosis.get("scheduler") or {}
    require(diagnosis.get("schema") == DIAGNOSIS_SCHEMA and diagnosis.get("status") == DIAGNOSIS_STATUS
            and diagnosis.get("owner") == OWNER and diagnosis.get("cell") == ack["cell"]
            and diagnosis.get("job_id") == job and diagnosis.get("ack") == ack_ref
            and scheduled.get("state") == "FAILED" and scheduled.get("exit_code") == record["exit_code"]
            and diagnosis.get("controller") == p.descriptor(SOURCE / "prepare.py")
            and isinstance(diagnosis.get("cause"), str) and diagnosis["cause"].strip() != ""
            and isinstance(diagnosis.get("repair"), str) and diagnosis["repair"].strip() != ""
            and diagnosis.get("full_evaluation_result_absent") is True
            and (evidence.get("worker_log") or {}).get("path") == str(log_path),
            "Exact owner DIAGNOSIS bound to the FAILED evaluation request and this controller required")
    excerpt = evidence.get("log_excerpt")
    require(isinstance(excerpt, str) and excerpt.strip() != "" and excerpt.encode() in p.checked(evidence["worker_log"])
            and evidence.get("log_excerpt_sha256") == hashlib.sha256(excerpt.encode()).hexdigest(),
            "The diagnosis excerpt must be contained in the exact worker log and hashed")
    review_ref = p.descriptor(folder / ("DIAGNOSIS_REVIEW-%s.json" % job))
    review = json.loads(p.checked(review_ref))
    require(review.get("status") == DIAGNOSIS_REVIEW_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER, PRODUCER_OWNER) and review.get("blocking_findings") == []
            and review.get("diagnosis") == diagnosis_ref and review.get("job_id") == job
            and review.get("controller") == p.descriptor(SOURCE / "prepare.py"),
            "Independent PASS review bound to this exact evaluation DIAGNOSIS and this controller required")
    return dict(diagnosis=diagnosis_ref, review=review_ref)


def predecessor_allows(p, cell, folder, attempt):
    """Attempt 0 is the first request of this version; attempt N follows attempt N-1. Never repeat a completed
    evaluation, never run beside a live one."""
    require(not any((folder / ("attempt-%d" % i)).exists() for i in range(attempt + 1, MAX_ATTEMPTS)),
            "Later attempt folders already exist; attempts are strictly sequential")
    if attempt == 0:
        return None
    previous = folder / ("attempt-%d" % (attempt - 1))
    ack_ref = p.descriptor(previous / "ACK.json")
    ack = json.loads(p.checked(ack_ref))
    require(ack["attempt"] == attempt - 1 and ack["cell"] == cell
            and re.fullmatch(r"[1-9][0-9]*", ack["job_id"]) is not None, "Exact predecessor ACK required")
    log_path = folder / ("worker-%s.log" % ack["job_id"])
    record = accounting(p, ack["job_id"])
    require(record["state"] in TERMINAL, "Predecessor evaluation still active; never run two")
    require(not (previous / ("FULL_EVALUATION_RESULT-%s.json" % ack["job_id"])).exists()
            and record["state"] != "COMPLETED", "A completed evaluation is never repeated")
    diagnosis = None
    if record["state"] == "FAILED":
        diagnosis = diagnosed_failure(p, folder, ack_ref, ack, record, log_path)
    elif record["state"] == "CANCELLED":
        require(record["start"] in ("None", "Unknown") and record["elapsed"] == "00:00:00" and not log_path.exists(),
                "Only a never-started cancelled request may be followed")
    else:
        require(record["state"] in INTERRUPTED, "A predecessor without a result needs diagnosis")
    return dict(ack=ack_ref, accounting=record, diagnosis=diagnosis)


def admit(p, cell, attempt, release=False):
    check_attempt(attempt)
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, cell, independent=release)
    afolder = folder / ("attempt-%d" % attempt)
    with p.admission_lock():
        require(contract(p, cell, independent=release) == (folder, plan_ref, plan, review_ref, independent_ref),
                "Contract changed while acquiring shared admission lock")
        if not release:
            predecessor = predecessor_allows(p, cell, folder, attempt)
            private_directory(afolder)
            require(not any((afolder / name).exists() for name in
                            ("INTENT.json", "ACK.json", "AMBIGUOUS.json", "FUNCTION_CLAIM.json")),
                    "Existing submission or ambiguous ACK: reconcile, never duplicate")
            producer = producer_terminal(p, plan)
            census = general_census(p, plan)
            claim = p.seal(afolder / "FUNCTION_CLAIM.json", dict(
                owner=OWNER, cell=cell, attempt=attempt, function=plan["function"],
                function_identity=plan["function_identity"], plan=plan_ref, predecessor=predecessor))
            intent = p.seal(afolder / "INTENT.json", dict(
                owner=OWNER, cell=cell, attempt=attempt, plan=plan_ref, review=review_ref, claim=claim,
                argv=argv(cell, plan, attempt), route=dict(ROUTE, nice=int(NICE)), general_census=census,
                producer_accounting=producer, exact_dependency=None))
            try:
                response = subprocess.run(argv(cell, plan, attempt), text=True, capture_output=True,
                                          timeout=45, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as error:
                p.seal(afolder / "AMBIGUOUS.json", dict(intent=intent, error=repr(error)))
                raise
            good = response.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;[\w.-]+)?\n?", response.stdout)
            if good:
                job = response.stdout.strip().split(";")[0]
                ack_ref = p.seal(afolder / "ACK.json", dict(owner=OWNER, cell=cell, attempt=attempt, job_id=job,
                                                            intent=intent, plan=plan_ref))
            p.seal(afolder / "SUBMISSION_RESULT.json", dict(
                intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
            require(good, "No unambiguous actual ACK; reconcile exact intent before another submission")
            raw, spool = scheduler(p, cell, plan, job, attempt, held=True)
            p.seal(afolder / "HELD_READBACK.json", dict(
                ACK=ack_ref, job_id=job, scheduler=raw, spool_base64=base64.b64encode(spool).decode()))
            return dict(cell=cell, attempt=attempt, job_id=job, state="HELD_VERIFIED", dependency=None)
        ack_ref = p.descriptor(afolder / "ACK.json")
        ack = json.loads(p.checked(ack_ref))
        require(ack["owner"] == OWNER and ack["plan"] == plan_ref and ack["cell"] == cell
                and ack["attempt"] == attempt, "Exact own ACK required")
        require(not (afolder / "RELEASE_INTENT.json").exists(), "Existing release requires reconciliation")
        job = ack["job_id"]
        scheduler(p, cell, plan, job, attempt, held=True)
        intent = p.seal(afolder / "RELEASE_INTENT.json", dict(ACK=ack_ref, job_id=job,
                                                              independent_review=independent_ref))
        response = subprocess.run(["/usr/bin/scontrol", "release", job], text=True, capture_output=True,
                                  timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
        p.seal(afolder / "RELEASE_RESULT.json", dict(
            intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        require(response.returncode == 0, "Release failed or ambiguous; preserve exact identity")
        raw, _ = scheduler(p, cell, plan, job, attempt)
        row = p.fields(raw)
        require(row.get("JobState") in ("PENDING", "RUNNING", "CONFIGURING")
                and row.get("Reason") not in ("JobHeldUser", "JobHeldAdmin", "DependencyNeverSatisfied"),
                "Released evaluation must be eligible")
        p.seal(afolder / "RELEASE_READBACK.json", dict(job_id=job, scheduler=raw, expected_dependency=None))
        return dict(cell=cell, attempt=attempt, job_id=job, state=row.get("JobState") + ":" + row.get("Reason", ""))


# ---------------------------------------------------------------- worker

def read_producer(p, cell, plan):
    """Re-hash the bound Tool3 RESULT and every listed output file on the evaluation node before any DATA."""
    custody = plan["function"]["producer_custody"]
    raw, ref = data_file(p, CHECKPOINT_ROOT / cell / "RESULT.json", 64 << 20, keep=True)
    require({key: ref[key] for key in ("path", "bytes", "sha256")} == custody["result"],
            "Tool3 RESULT.json differs from the attested plan custody")
    qualification = json.loads(raw)
    require(qualification["output_files"] == custody["output_files"]
            and rehash_outputs(p, cell, qualification["output_files"]) == custody["output_files"],
            "Tool3 output files differ from the attested plan custody")
    producer_ref = dict(path=ref["path"], bytes=ref["bytes"], sha256=ref["sha256"],
                        actual_open_FD_mount=ref["mount"], inode=ref["inode"])
    return qualification, producer_ref


def work(p, cell, attempt):
    check_attempt(attempt)
    cell_folder, plan_ref, plan, review_ref, independent_ref = contract(p, cell, independent=True)
    folder = cell_folder / ("attempt-%d" % attempt)
    job = os.environ.get("SLURM_JOB_ID", "")
    ack_ref = p.descriptor(folder / "ACK.json")
    ack = json.loads(p.checked(ack_ref))
    require(os.getuid() == 2708899 and job.isdecimal() and ack["job_id"] == job and ack["cell"] == cell
            and ack["attempt"] == attempt
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0"
            and not os.environ.get("SLURM_ARRAY_JOB_ID"), "Actual unrestarted owned evaluation required")
    release = json.loads(p.checked(p.descriptor(folder / "RELEASE_INTENT.json")))
    require(release == dict(ACK=ack_ref, job_id=job, independent_review=independent_ref),
            "Actual independently reviewed release intent required before DATA")
    scheduler(p, cell, plan, job, attempt, running=True)
    node = socket.gethostname().split(".")[0]
    p.validate_process(job, os.getpid(), node, os.environ.get("SLURM_STEP_ID"),
                       p.query(["/usr/bin/scontrol", "listpids", job]),
                       p.fields(p.query(["/usr/bin/scontrol", "show", "step", job + ".batch", "-o"])),
                       (Path("/proc") / str(os.getpid()) / "status").read_text())
    producer = plan["function"]["producer"]
    qualification, producer_ref = read_producer(p, cell, plan)
    require(qualification["job_id"] == producer["job_id"] and qualification["status"] == PRODUCER_STATUS
            and qualification["config"]["cell"] == cell, "Actual completed plan-bound Tool3 producer required")
    actor = dict(job_id=job, pid=os.getpid(), node=node, producer_job=producer["job_id"], plan=plan_ref, cell=cell,
                 attempt=attempt)
    p.seal(folder / ("ACTOR_BEFORE-" + job + ".json"), actor)
    evaluator = json.loads(p.checked(p.EVAL_CONFIG))
    parent = p.open_literal(OUTPUT_ROOT.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            os.mkdir(OUTPUT_ROOT.name, 0o700, dir_fd=parent)
        except FileExistsError:
            info = os.stat(OUTPUT_ROOT.name, dir_fd=parent, follow_symlinks=False)
            require(stat.S_ISDIR(info.st_mode) and info.st_uid == 2708899
                    and not info.st_mode & 0o022, "Private own output root required")
    finally:
        os.close(parent)
    output = OUTPUT_ROOT / ("job-" + job)
    directory = p.open_literal(OUTPUT_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.mkdir(output.name, 0o700, dir_fd=directory)
    finally:
        os.close(directory)
    p.seal(output / "OWNER.json", dict(owner=OWNER, cell=cell, job_id=job, producer_job=producer["job_id"],
                                       function_identity=plan["function_identity"]))
    config = dict(
        role=ROLE, owner=OWNER, job_id=job, cell=cell, method=plan["function"]["method"],
        selection_seed=plan["function"]["selection_seed"], training_seed=plan["function"]["training_seed"],
        producer=producer, qualification=qualification, model_result_reference=producer_ref, actual_actor=actor,
        model_source=MODEL_SOURCE, eval_sources=evaluator["sources"], reference_evaluator=p.EVAL_CONFIG,
        official_runtime_reference=evaluator["official_POPE_configuration"], output=str(output),
        trainable_reference=TRAINABLE_REFERENCE)
    config_ref = p.seal(folder / ("MODEL_CONFIG-" + job + ".json"), config)
    command = [PYTHON, "-u", "-B", "-X", "pycache_prefix=/dev/null/no-cache",
               MODEL_SOURCE["path"], "--config", config_ref["path"]]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HOME="/data/hf_cache", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               TORCH_EXTENSIONS_DIR="/data/hf_cache/project_envs/DS-MultiModal/torch_extensions",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="8")
    p.seal(folder / ("MODEL_INTENT-" + job + ".json"), dict(config=config_ref, argv=command, actor=actor))
    started = time.monotonic()
    result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, timeout=42000)
    p.seal(folder / ("MODEL_EXIT-" + job + ".json"), dict(
        returncode=result.returncode, elapsed_seconds=time.monotonic() - started, output=str(output)))
    require(result.returncode == 0, "Full evaluation failed; preserve exact log, accepted QAs and producer")
    report = json.loads((output / "RESULT.json").read_bytes())
    require(report["status"] == RESULT_STATUS and report["job_id"] == job and report["cell"] == cell
            and report["actual_accepted_native_QA"] == 28505,
            "Full28505-QA completion required; no partial-bank success")
    p.seal(folder / ("FULL_EVALUATION_RESULT-" + job + ".json"), report)
    scheduler(p, cell, plan, job, attempt, running=True)
    p.seal(folder / ("ACTOR_AFTER-" + job + ".json"), actor)
    print(json.dumps(dict(job_id=job, producer_job=producer["job_id"], cell=cell, accepted_QA=28505,
                          status=report["status"], output=str(output))), flush=True)


def readiness(p):
    """Read-only HOME census of the six Tool3 cells (RESULT_SUCCESS links and this controller's state)."""
    report = {}
    for cell, spec in CELLS.items():
        folder = CONTROL_ROOT / cell
        try:
            receipts = tool3_receipts(p, cell, spec)
            state = dict(ready=True, job=receipts["job"], link=receipts["link"], end_step=receipts["end_step"])
        except NotReady:
            state = dict(ready=False)
        except ValueError as error:
            state = dict(ready=False, blocked=str(error))
        state.update(plan=(folder / "PLAN.json").exists(), owner_review=(folder / "OWNER_REVIEW.json").exists(),
                     independent_review=independent_path(cell).exists(),
                     attempts=sorted(path.name for path in folder.glob("attempt-*")) if folder.is_dir() else [])
        report[cell] = state
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("attest", "plan", "self-review", "submit", "release", "worker",
                                            "readiness"))
    parser.add_argument("--cell", default=None)
    parser.add_argument("--attestation", default=None)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    p = load_helpers()
    if args.command == "readiness":
        print(json.dumps(readiness(p), sort_keys=True))
        return
    cell_spec(args.cell)
    if args.command == "worker":
        work(p, args.cell, args.attempt)
    elif args.command == "attest":
        print(json.dumps(attest(p, args.cell), sort_keys=True))
    elif args.command == "plan":
        require(args.attestation is not None, "plan needs --attestation <CONTROL/cell/ATTESTATION-*.json>")
        print(json.dumps(make_plan_file(p, args.cell, args.attestation), sort_keys=True))
    elif args.command == "self-review":
        print(json.dumps(owner_review(p, args.cell, args.note), sort_keys=True))
    else:
        print(json.dumps(admit(p, args.cell, args.attempt, release=args.command == "release"), sort_keys=True))


if __name__ == "__main__":
    main()
