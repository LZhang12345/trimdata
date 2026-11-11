"""Roster-bound four-bank evaluations of Tool1 baseline producers on the General route (v4).

v4 (2026-11-05) is v3 with exactly the additions needed by the author's instruction of 2026-11-05 to move Tool1's
high-priority pending evaluations from the congested Preempt queue to General:
  * route: requests are planned, submitted, read back and run on account cx, partition general, QoS normal
    (gres gpu:L40S:1, exclude babel-p9-20, the same CPUs/memory/time, no requeue); the admission gate is the
    actual normal-QoS MaxSubmitPU of this user. Control lives in a new root (baseline-eval-v4); outputs keep the
    per-job directories of the bound v2 output root; model.py is the same pinned v2 file.
  * producers: the newest acknowledged trainer successor may be a v7 (PLAN-v7.json) or v8 General
    (PLAN-v8.json, helper_tool1_baseline_train_v8_20261105) request of the same finite function; the ACK's own
    plan is bound and its schema must match its file name.
  * predecessor: attempt 0 follows the cell's newest v3 evaluation request (instead of the v2 request). A
    never-started v3 request is withdrawn by `withdraw --cell C` under the shared ADMISSION.lock (readback,
    hold, JobHeldUser readback, scancel, accounting CANCELLED with Start=None/Elapsed 00:00:00, no worker log),
    sealing WITHDRAW_INTENT/WITHDRAW_HOLD/WITHDRAW_CANCEL/WITHDRAWN receipts in CONTROL/<cell>/; attempt 0 accepts
    a CANCELLED v3 predecessor only with that receipt. A started or completed v3 request is never withdrawn.
Text below this paragraph is the v3 description.

Roster-bound four-bank evaluations for Preempt-trained Tool1 baseline producers (v3: successors).

v3 (2026-11-05) is v2 (helper_tool1_baseline_eval_v2_20261102/prepare.py, unchanged; model.py pinned from the v2
folder) with two additions required after the 2026-11-05 external deletion of the shared HF snapshot links:
  * producer successors: when the trainer cell has v7 successor requests (baseline-train-v5/<cell>/successor-<n>/),
    the evaluation binds the newest successor's ACK/FUNCTION_CLAIM/HELD_READBACK, the v7 PLAN-v7.json and the
    original PLAN.json (same finite function), waits on afterok of that successor job (no dependency once the bound
    producer job has already sealed its TRAINING_RESULT: Slurm rejects afterok on a purged job), and the worker
    requires the producer RESULT.json of that job; otherwise it binds the original producer exactly as v2. The
    producer descriptor given to the pinned v2 model.py keeps exactly its six v2 keys (producer_successor is
    recorded beside it in the function).
  * explicit attempts: CONTROL/<cell>/attempt-<N>/ holds each held request; attempt 0 follows the cell's v2 request
    only if that v2 request is terminal and FAILED (owner DIAGNOSIS + independent DIAGNOSIS_REVIEW), interrupted or
    CANCELLED without ever starting (DependencyNeverSatisfied after a failed producer); attempt N>0 follows a terminal
    FAILED (diagnosed) or interrupted attempt N-1. A completed evaluation is never repeated.
Text below this paragraph is the v2 description.

Roster-bound four-bank evaluations for Preempt-trained Tool1 baseline producers (v2).

Each evaluation binds its producer's exact trainer roster entry, plan, ACK, claim and held readback,
waits on afterok for that producer, and runs the same reviewed four-bank inference/scoring body.
Preempt route only while the actual General queue is full; no automatic requeue.
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
SOURCE = HOME / "experiments/helper_tool1_baseline_eval_v4_20261105"
V2_SOURCE = HOME / "experiments/helper_tool1_baseline_eval_v2_20261102"
CONTROL_ROOT = HOME / "collaboration/helper-tool1-baseline-independent-20261101/baseline-eval-v4"
V3_SOURCE = HOME / "experiments/helper_tool1_baseline_eval_v3_20261105"
V3_CONTROL = HOME / "collaboration/helper-tool1-baseline-independent-20261101/baseline-eval-v3"
V2_CONTROL = HOME / "collaboration/helper-tool1-baseline-independent-20261101/baseline-eval-v2"
PRODUCER_CONTROL = HOME / "collaboration/helper-tool1-baseline-independent-20261101/baseline-train-v5"
CHECKPOINT_ROOT = Path("/data/hf_cache/project_checkpoints/DS-MultiModal/helper-tool1-baseline-train-v4-20261102")
OUTPUT_ROOT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-baseline-eval-v2-20261102")
PYTHON = "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python"
OWNER = "Tool1-owned-published-baseline-full-official-evaluation"
PRODUCER_OWNER = "Tool1-owned-published-baseline-native-training"
REVIEW_STATUS = "OWNER_REVIEWED_TOOL1_BASELINE_FULL_EVAL_V4_NOT_INDEPENDENT_ACCEPTANCE"
INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_BASELINE_EVAL_V4"
DIAGNOSIS_SCHEMA = "tool1-baseline-eval-failed-attempt-diagnosis-v4"
DIAGNOSIS_STATUS = "OWNER_DIAGNOSED_FAILED_EVALUATION_ATTEMPT"
DIAGNOSIS_REVIEW_STATUS = "PASS_INDEPENDENT_FAILED_EVALUATION_DIAGNOSIS_TOOL1_BASELINE_EVAL_V4"
WITHDRAWN_SCHEMA = "tool1-baseline-eval-v3-never-started-withdrawal-v4"
ROUTE = dict(account="cx", partition="general", qos="normal")
MAX_ATTEMPTS = 6
MAX_SUCCESSORS = 8
TERMINAL = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
INTERRUPTED = ("TIMEOUT", "PREEMPTED", "NODE_FAIL")
SUCCESSOR_PLAN_SCHEMAS = {"PLAN-v7.json": "tool1-baseline-native-preempt-successor-v7",
                          "PLAN-v8.json": "tool1-baseline-native-general-successor-v8"}
HELPER = {
    "path": str(HOME / "experiments/helper_method_eval_bank_qual_v1_20261102/prepare.py"),
    "bytes": 51200, "sha256": "f0dc4f6a548eedc45d5ba3d92e0251328d33bb82b274d815f14b0aca25fa7357",
}
MODEL_SOURCE = {"path": str(V2_SOURCE / "model.py"), "bytes": 18681,
                "sha256": "df783c9ab4d02b94782af82570ce839c80aaa813b8eb795ac1e37caa3ffb4e2a"}
BANK_QUALIFICATION = {
    "path": str(HOME / "collaboration/helper-tool2-method-20261101/eval-bank-qual-v1/"
                "ACTUAL_INFERENCE_RESULT-11022388.json"),
    "bytes": 318676, "sha256": "bde9f9a4fd0efb2191670d153975daf663ff1990fb80c9609feef378cf01bc2a",
}
PRODUCER_MODEL_SHA = "b53089293b49a5a2d9fb8d433999105d06c70629f3be788bbf35d3089a728b11"
PRODUCER_STATUS = "ACTUAL_FULL_NATIVE_BASELINE_EPOCH_NOT_EVALUATED"
ROLE = "TOOL1_BASELINE_FULL_EPOCH_FOUR_OFFICIAL_BANK_EVALUATION"
RESULT_STATUS = "COMPLETE_TOOL1_BASELINE_FOUR_OFFICIAL_BANKS"
COMMENT_PREFIX = "dsmm-t1-baseline-eval-v4:"
V3_COMMENT_PREFIX = "dsmm-t1-baseline-eval-v3:"
TIME = "12:00:00"


def require(value, message):
    if not value:
        raise ValueError(message)


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
    module = ModuleType("tool1_frozen_evaluation_metadata_helpers_v4")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    return module


def roster_cell(p, roster_ref, cell):
    require(isinstance(roster_ref, dict) and roster_ref.get("path", "").startswith(str(PRODUCER_CONTROL) + "/"),
            "Producer roster must be a pinned trainer control file")
    roster = json.loads(p.checked(roster_ref))
    require(roster.get("schema") == "tool1-baseline-train-roster-v1" and isinstance(roster.get("cells"), dict)
            and isinstance(cell, str) and re.fullmatch(r"[a-z0-9][a-z0-9-]{2,60}", cell) is not None
            and cell in roster["cells"] and roster["cells"][cell].get("kind") == "production"
            and roster["cells"][cell].get("job_name", "").startswith("dsmm-t1p-"),
            "Exact production producer cell from the pinned trainer roster required")
    return roster["cells"][cell]


def producer_refs(p, cell):
    """The newest acknowledged producer request: a v7/v8 successor if any exists, else the original request."""
    folder = PRODUCER_CONTROL / cell
    newest = None
    for index in range(1, MAX_SUCCESSORS + 1):
        if not (folder / ("successor-%d" % index) / "ACK.json").exists():
            break
        newest = index
    if newest is None:
        return {name: p.descriptor(folder / (name + ".json"))
                for name in ("PLAN", "ACK", "FUNCTION_CLAIM", "HELD_READBACK")}
    successor = folder / ("successor-%d" % newest)
    refs = {name: p.descriptor(successor / (name + ".json")) for name in ("ACK", "FUNCTION_CLAIM", "HELD_READBACK")}
    plan_path = Path(json.loads(p.checked(refs["ACK"]))["plan"]["path"])
    require(plan_path.parent == folder and plan_path.name in SUCCESSOR_PLAN_SCHEMAS,
            "A successor ACK must bind its own v7 or v8 plan in the producer cell folder")
    refs["PLAN"] = p.descriptor(plan_path)
    refs["ORIGINAL_PLAN"] = p.descriptor(folder / "PLAN.json")
    return refs


def validate_producer_documents(p, cell, refs, roster_ref, spec):
    values = {name: json.loads(p.checked(ref)) for name, ref in refs.items()}
    plan, ack, claim = values["PLAN"], values["ACK"], values["FUNCTION_CLAIM"]
    successor = None
    if "ORIGINAL_PLAN" in refs:
        original = values["ORIGINAL_PLAN"]
        successor = ack.get("successor")
        require(set(refs) == {"PLAN", "ACK", "FUNCTION_CLAIM", "HELD_READBACK", "ORIGINAL_PLAN"}
                and plan.get("schema") == SUCCESSOR_PLAN_SCHEMAS.get(Path(refs["PLAN"]["path"]).name)
                and plan.get("original_plan") == refs["ORIGINAL_PLAN"]
                and original.get("function") == plan["function"]
                and original.get("function_identity") == plan["function_identity"]
                and type(successor) is int and 1 <= successor <= MAX_SUCCESSORS
                and claim.get("successor") == successor and claim.get("plan") == refs["PLAN"],
                "Exact v7/v8 successor producer binding to the original finite function required")
    require(ack["cell"] == cell and ack["plan"] == refs["PLAN"] and ack["owner"] == PRODUCER_OWNER
            and re.fullmatch(r"[1-9][0-9]*", ack["job_id"]) is not None
            and values["HELD_READBACK"]["job_id"] == ack["job_id"]
            and claim["function_identity"] == plan["function_identity"] and claim["cell"] == cell
            and claim["owner"] == plan["owner"] == PRODUCER_OWNER and plan["cell"] == cell
            and plan["function"]["kind"] == "production" and plan["function"]["roster"] == roster_ref
            and plan["function"]["spec"] == spec
            and plan["sources"]["model.py"]["sha256"] == PRODUCER_MODEL_SHA,
            "Exact actual roster-bound producer binding required; no borrowed custody")
    k = spec["K"]
    require(type(k) is int and 1024 <= k <= 660763 and (k + 3) // 4 // 32 >= 2
            and spec.get("training_seed") in (17, 29, 43), "Bounded producer K, horizon and training seed required")
    return {"cell": cell, "job_id": ack["job_id"], "K": k, "training_seed": spec["training_seed"],
            "steps": (k + 3) // 4 // 32, "model_source_sha256": PRODUCER_MODEL_SHA, "successor": successor}


FRESH = object()


def producer_completion(p, cell, producer, completion=FRESH):
    """The bound producer job's own sealed TRAINING_RESULT, frozen into the plan when it already existed at plan
    time (Slurm rejects afterok on a completed job that slurmctld has purged: 'Job dependency problem')."""
    if completion is FRESH:
        # A Slurm requeue after a sealed success replays the existing-RESULT path and seals another TRAINING_RESULT
        # of the same job; every one must be complete for this job, and the earliest is bound (stable forever).
        names = sorted((PRODUCER_CONTROL / cell).glob("TRAINING_RESULT-%s-r*.json" % producer["job_id"]),
                       key=lambda path: int(path.name.rsplit("-r", 1)[1][:-5]))
        for name in names:
            value = json.loads(p.checked(p.descriptor(name)))
            require(value.get("status") == PRODUCER_STATUS and value.get("job_id") == producer["job_id"]
                    and value.get("cell") == cell, "Every sealed TRAINING_RESULT of the producer job must be complete")
        completion = p.descriptor(names[0]) if names else None
    if completion is not None:
        value = json.loads(p.checked(completion))
        require(Path(completion["path"]).parent == PRODUCER_CONTROL / cell
                and re.fullmatch(r"TRAINING_RESULT-%s-r[0-9]+\.json" % producer["job_id"],
                                 Path(completion["path"]).name) is not None
                and value.get("status") == PRODUCER_STATUS and value.get("job_id") == producer["job_id"]
                and value.get("cell") == cell, "Exact completed producer TRAINING_RESULT required")
    return completion



def newest_v3_ack(cell):
    """The newest contiguous v3 attempt ACK of this cell (None when the cell never had a v3 request)."""
    newest = None
    for attempt in range(MAX_ATTEMPTS):
        path = V3_CONTROL / cell / ("attempt-%d" % attempt) / "ACK.json"
        if not path.exists():
            break
        newest = path
    return newest


def make_plan(p, cell, roster_ref, completion=FRESH):
    spec = roster_cell(p, roster_ref, cell)
    refs = producer_refs(p, cell)
    producer = validate_producer_documents(p, cell, refs, roster_ref, spec)
    # The pinned v2 model.py validates the producer descriptor with exactly its six v2 keys.
    successor = producer.pop("successor")
    completion = producer_completion(p, cell, producer, completion)
    function = dict(
        owner=OWNER, cell=cell, method=spec["method"], selection_seed=spec["selection_seed"],
        training_seed=spec["training_seed"], K=spec["K"], producer=producer, producer_successor=successor,
        producer_receipts=refs,
        producer_roster=roster_ref, prior_executed_bank_qualification=BANK_QUALIFICATION, metadata_helper=HELPER,
        model_source=MODEL_SOURCE, native_python=PYTHON, full_QA=28505,
        banks={"POPE": 8910, "GQA": 12578, "TextVQA": 5000, "ScienceQA_image": 2017},
        GQA_main="UNCHANGED_NATIVE_NORMALIZED_SCORE", no_model_downloads=True,
        no_base_copies=True, no_training_updates=True, no_Root_acceptance_or_grant_transfer=True)
    v3_ack = newest_v3_ack(cell)
    return dict(
        schema="tool1-baseline-roster-evaluations-v4", owner=OWNER, cell=cell, function=function,
        v3_request=p.descriptor(v3_ack) if v3_ack is not None else None,
        function_identity=hashlib.sha256(p.canonical(function)).hexdigest(),
        sources={"prepare.py": p.descriptor(SOURCE / "prepare.py"),
                 "model.py": MODEL_SOURCE, "run.sbatch": p.descriptor(SOURCE / "run.sbatch")},
        job_name="dsmm-t1e-" + spec["job_name"][len("dsmm-t1p-"):],
        producer_completion=completion,
        dependency=None if completion is not None else "afterok:" + producer["job_id"], resource=dict(
            account=ROUTE["account"], partition=ROUTE["partition"], qos=ROUTE["qos"], gpu="L40S", GPUs=1,
            CPUs=8, memory="64G", time=TIME, exclude="babel-p9-20"),
        automatic_requeue=False, independently_reviewed=False)


def independent_path(roster_ref, cell):
    """One create-once independent review file per (pinned producer roster, cell), so evaluations can be
    reviewed in batches as their producer ACKs appear; a later review never replaces a bound one."""
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{2,60}", cell) is not None, "Literal evaluation cell required")
    return CONTROL_ROOT / ("INDEPENDENT_REVIEW-%s-%s.json" % (roster_ref["sha256"][:16], cell))


def validate_independent(review, plan_ref, plan, cell):
    require(review.get("status") == INDEPENDENT_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER, PRODUCER_OWNER)
            and review.get("blocking_findings") == [] and review.get("sources") == plan["sources"]
            and isinstance(review.get("plans"), dict) and review["plans"].get(cell) == plan_ref,
            "Independent exact source/config PASS bound to this evaluation plan is required")


def contract(p, cell, roster_ref, independent=False):
    folder = CONTROL_ROOT / cell
    info = os.stat(folder, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == 2708899 and not info.st_mode & 0o022,
            "Private owned evaluation control directory required")
    plan_ref = p.descriptor(folder / "PLAN.json")
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, cell, roster_ref, plan.get("producer_completion")),
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
    independent_ref = None
    if independent:
        independent_ref = p.descriptor(independent_path(roster_ref, cell))
        validate_independent(json.loads(p.checked(independent_ref)), plan_ref, plan, cell)
    return folder, plan_ref, plan, review_ref, independent_ref


def parent_status(p, job):
    value = p.query(["/usr/bin/sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State,ExitCode"]).strip()
    rows = [row.split("|") for row in value.splitlines() if row.strip()]
    require(len(rows) == 1 and len(rows[0]) == 3 and rows[0][0] == job, "Exact producer scheduler record required")
    state = rows[0]
    require(state[1] in ("RUNNING", "PENDING", "REQUEUED") or state[1:] == ["COMPLETED", "0:0"],
            "Producer failed/cancelled or unresolved; never remove its dependency")
    return state


def check_attempt(attempt):
    require(type(attempt) is int and 0 <= attempt < MAX_ATTEMPTS, "Finite evaluation attempt index required")
    return attempt


def comment(plan, attempt):
    return COMMENT_PREFIX + plan["function_identity"] + ":a%d" % check_attempt(attempt)


def argv(cell, plan, attempt):
    return ["/usr/bin/sbatch", "--parsable", "--hold", "--no-requeue", "--export=NIL",
            "--account=" + ROUTE["account"], "--partition=" + ROUTE["partition"], "--qos=" + ROUTE["qos"],
            "--gres=gpu:L40S:1",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G", "--time=" + TIME,
            "--exclude=babel-p9-20"] + (
                [] if plan["dependency"] is None else ["--dependency=" + plan["dependency"]]) + [
            "--job-name=" + plan["job_name"], "--comment=" + comment(plan, attempt),
            "--chdir=" + str(HOME), "--input=/dev/null", "--output=/dev/null", "--error=/dev/null",
            str(SOURCE / "run.sbatch"), cell, plan["function"]["producer_roster"]["path"],
            plan["function"]["producer_roster"]["sha256"], str(check_attempt(attempt))]


def validate_dependency(value, parent, completed):
    valid = re.fullmatch(r"afterok:" + re.escape(parent) + r"(?:\((?:unfulfilled|fulfilled)\))?", value or "")
    require(valid is not None or (completed and value == "(null)"), "Exact successful-parent dependency required")


def scheduler(p, cell, plan, job, attempt, held=False, running=False):
    raw = p.query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = p.fields(raw)
    expected = dict(
        JobId=job, UserId="luyangz(2708899)", Account=ROUTE["account"], Partition=ROUTE["partition"],
        QOS=ROUTE["qos"],
        JobName=plan["job_name"], Comment=comment(plan, attempt),
        NumCPUs="8", NumTasks="1", TresPerNode="gres/gpu:L40S:1", TimeLimit=TIME,
        ExcNodeList="babel-p9-20", Requeue="0", Command=str(SOURCE / "run.sbatch"), WorkDir=str(HOME),
        StdIn="/dev/null", StdOut="/dev/null", StdErr="/dev/null")
    nodes = row.get("NumNodes") == "1" if running else row.get("NumNodes") in ("1", "1-1")
    require(all(row.get(key) == value for key, value in expected.items()) and nodes
            and shlex.split(row.get("SubmitLine", "")) == argv(cell, plan, attempt),
            "Actual physical evaluation request/source/resources differ")
    producer_job = plan["function"]["producer"]["job_id"]
    producer = parent_status(p, producer_job)
    completed = producer[1:] == ["COMPLETED", "0:0"]
    if plan["dependency"] is None:
        require(completed and row.get("Dependency") == "(null)", "A completed producer binds no dependency")
    else:
        validate_dependency(row.get("Dependency"), producer_job, completed)
    if held:
        require(row.get("JobState") == "PENDING" and row.get("Reason") == "JobHeldUser"
                and row.get("RunTime") == "00:00:00", "Verified never-started held request required")
    if running:
        require(row.get("JobState") == "RUNNING" and row.get("Restarts") == "0" and completed
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


def general_full(rows, max_submit, max_gpus):
    normal = [row for row in rows if row[1] == "normal"]
    running = sum(row[4] == "RUNNING" and "gpu" in row[5] for row in normal)
    pending = sum(row[4] == "PENDING" for row in normal)
    return (len(normal) >= max_submit or (running >= max_gpus and pending > 0)), dict(
        requests=len(normal), running_gpu=running, pending=pending)


def private_directory(path):
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.stat(path, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == 2708899 and not info.st_mode & 0o022,
            "Owned private control directory required: " + str(path))


def make_plan_file(p, cell, roster_ref):
    roster_cell(p, roster_ref, cell)
    private_directory(CONTROL_ROOT)
    private_directory(CONTROL_ROOT / cell)
    path = CONTROL_ROOT / cell / "PLAN.json"
    require(not path.exists(), "Existing evaluation plan is immutable; use a new version")
    p.checked(MODEL_SOURCE)
    return p.seal(path, make_plan(p, cell, roster_ref))


def owner_review(p, cell, roster_ref, notes):
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete self-review notes required")
    plan_ref = p.descriptor(CONTROL_ROOT / cell / "PLAN.json")
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, cell, roster_ref, plan.get("producer_completion")),
            "Self-review must bind the current exact plan")
    return p.seal(CONTROL_ROOT / cell / "OWNER_REVIEW.json", dict(
        status=REVIEW_STATUS, independent=False, ai_review_disclosed=True, reviewer=OWNER, cell=cell,
        plan=plan_ref, sources=plan["sources"], function_identity=plan["function_identity"],
        blocking_findings=[], notes=notes))


def accounting(p, job):
    value = p.query(["/usr/bin/sacct", "-X", "-nP", "-j", job,
                     "--format=JobIDRaw,State,ExitCode,Start,Elapsed"]).strip().split("|")
    require(len(value) == 5 and value[0] == job, "Exact scheduler accounting record required")
    return dict(job_id=job, state=value[1].split(" ")[0], exit_code=value[2], start=value[3], elapsed=value[4])


def diagnosed_failure(p, folder, ack_ref, ack, record, log_path):
    """Owner DIAGNOSIS + independent DIAGNOSIS_REVIEW of a FAILED evaluation request (v2 or v3 attempt)."""
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
            and (evidence.get("worker_log") or {}).get("path") == str(log_path)
            and isinstance(evidence.get("records", []), list),
            "Exact owner DIAGNOSIS bound to the FAILED evaluation request and this controller required")
    excerpt = evidence.get("log_excerpt")
    require(isinstance(excerpt, str) and excerpt.strip() != "" and excerpt.encode() in p.checked(evidence["worker_log"])
            and evidence.get("log_excerpt_sha256") == hashlib.sha256(excerpt.encode()).hexdigest(),
            "The diagnosis excerpt must be contained in the exact worker log and hashed")
    for ref in evidence.get("records", []):
        p.checked(ref)
    review_ref = p.descriptor(folder / ("DIAGNOSIS_REVIEW-%s.json" % job))
    review = json.loads(p.checked(review_ref))
    require(review.get("status") == DIAGNOSIS_REVIEW_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER, PRODUCER_OWNER) and review.get("blocking_findings") == []
            and review.get("diagnosis") == diagnosis_ref and review.get("job_id") == job
            and review.get("controller") == p.descriptor(SOURCE / "prepare.py"),
            "Independent PASS review bound to this exact evaluation DIAGNOSIS and this controller required")
    return dict(diagnosis=diagnosis_ref, review=review_ref)


def predecessor_allows(p, cell, folder, attempt, plan):
    """Attempt 0 follows the cell's newest v3 request (if any); attempt N follows v4 attempt N-1. Never repeat a
    completed evaluation, never run beside a live one."""
    require(not any((folder / ("attempt-%d" % i)).exists() for i in range(attempt + 1, MAX_ATTEMPTS)),
            "Later attempt folders already exist; attempts are strictly sequential")
    if attempt == 0:
        if plan["v3_request"] is None:
            return None
        ack_ref = plan["v3_request"]
        ack = json.loads(p.checked(ack_ref))
        require(Path(ack_ref["path"]) == newest_v3_ack(cell), "Attempt 0 follows the newest v3 request")
        log_path = V3_CONTROL / cell / ("worker-%s.log" % ack["job_id"])
        completed = Path(ack_ref["path"]).parent / ("FULL_EVALUATION_RESULT-%s.json" % ack["job_id"])
        source = "v3"
    else:
        previous = folder / ("attempt-%d" % (attempt - 1))
        ack_ref = p.descriptor(previous / "ACK.json")
        ack = json.loads(p.checked(ack_ref))
        require(ack["attempt"] == attempt - 1, "Exact predecessor ACK required")
        log_path = folder / ("worker-%s.log" % ack["job_id"])
        completed = previous / ("FULL_EVALUATION_RESULT-%s.json" % ack["job_id"])
        source = "v4"
    require(ack["cell"] == cell and re.fullmatch(r"[1-9][0-9]*", ack["job_id"]) is not None,
            "Exact predecessor evaluation ACK required")
    record = accounting(p, ack["job_id"])
    require(record["state"] in TERMINAL, "Predecessor evaluation still active; never run two")
    require(not completed.exists() and record["state"] != "COMPLETED", "A completed evaluation is never repeated")
    diagnosis = withdrawal = None
    if record["state"] == "FAILED":
        diagnosis = diagnosed_failure(p, folder, ack_ref, ack, record, log_path)
    elif record["state"] == "CANCELLED":
        require(record["start"] in ("None", "Unknown") and record["elapsed"] == "00:00:00" and not log_path.exists(),
                "Only a never-started cancelled request (e.g. DependencyNeverSatisfied) may be followed")
        if source == "v3":
            withdrawal = withdrawn_receipt(p, cell, ack_ref, ack)
    else:
        require(record["state"] in INTERRUPTED, "A predecessor without a result needs diagnosis")
    return dict(source=source, ack=ack_ref, accounting=record, diagnosis=diagnosis, withdrawal=withdrawal)



def v3_request(p, cell):
    path = newest_v3_ack(cell)
    require(path is not None, "No v3 evaluation request exists for this cell")
    ack_ref = p.descriptor(path)
    ack = json.loads(p.checked(ack_ref))
    v3_plan = json.loads(p.checked(ack["plan"]))
    require(ack.get("owner") == OWNER and ack.get("cell") == cell and type(ack.get("attempt")) is int
            and path.parent.name == "attempt-%d" % ack["attempt"]
            and re.fullmatch(r"[1-9][0-9]*", str(ack.get("job_id"))) is not None
            and ack["plan"]["path"] == str(V3_CONTROL / cell / "PLAN.json") and v3_plan.get("cell") == cell
            and v3_plan.get("owner") == OWNER
            and v3_plan.get("sources", {}).get("run.sbatch", {}).get("path") == str(V3_SOURCE / "run.sbatch"),
            "Exact own v3 evaluation request of this cell required")
    return path, ack_ref, ack, v3_plan


def v3_traces(cell, attempt_folder, job):
    """Any worker log or attempt record of the v3 request means it started; it is then never withdrawn."""
    names = [V3_CONTROL / cell / ("worker-%s.log" % job)] + [
        attempt_folder / (name % job) for name in ("ACTOR_BEFORE-%s.json", "MODEL_CONFIG-%s.json",
                                                   "MODEL_INTENT-%s.json", "FULL_EVALUATION_RESULT-%s.json")]
    return sorted(str(path) for path in names if path.exists())


def v3_readback(p, job, v3_plan, attempt, held):
    raw = p.query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = p.fields(raw)
    expected = dict(JobId=job, UserId="luyangz(2708899)", JobName=v3_plan["job_name"],
                    Comment=V3_COMMENT_PREFIX + v3_plan["function_identity"] + ":a%d" % attempt,
                    Command=str(V3_SOURCE / "run.sbatch"), JobState="PENDING", RunTime="00:00:00", Restarts="0")
    if held:
        expected["Reason"] = "JobHeldUser"
    require(all(row.get(key) == value for key, value in expected.items())
            and row.get("BatchHost") in (None, "", "(null)"),
            "Withdrawal is only for this cell's own never-started PENDING v3 request")
    return raw


def withdrawn_receipt(p, cell, ack_ref, ack):
    path = CONTROL_ROOT / cell / ("WITHDRAWN-%s.json" % ack["job_id"])
    require(path.is_file(), "A CANCELLED v3 predecessor needs this controller's WITHDRAWN receipt")
    ref = p.descriptor(path)
    value = json.loads(p.checked(ref))
    require(value.get("schema") == WITHDRAWN_SCHEMA and value.get("owner") == OWNER and value.get("cell") == cell
            and value.get("job_id") == ack["job_id"] and value.get("ack") == ack_ref
            and value.get("never_started") is True and value.get("controller") == p.descriptor(SOURCE / "prepare.py"),
            "Exact WITHDRAWN receipt of this never-started v3 request by this controller required")
    return ref


def withdraw(p, cell):
    """Withdraw the cell's newest v3 evaluation request while it is a never-started PENDING request."""
    with p.admission_lock():
        path, ack_ref, ack, v3_plan = v3_request(p, cell)
        job, attempt = ack["job_id"], ack["attempt"]
        private_directory(CONTROL_ROOT)
        private_directory(CONTROL_ROOT / cell)
        folder = CONTROL_ROOT / cell
        require(not (folder / ("WITHDRAW_INTENT-%s.json" % job)).exists(),
                "Existing withdrawal attempt requires reconciliation")
        require(not v3_traces(cell, path.parent, job), "The v3 request started; it is never withdrawn")
        before = v3_readback(p, job, v3_plan, attempt, held=False)
        intent = p.seal(folder / ("WITHDRAW_INTENT-%s.json" % job), dict(
            owner=OWNER, cell=cell, job_id=job, attempt=attempt, ack=ack_ref, plan=ack["plan"],
            reason="author 2026-11-05: move Tool1 high-priority pending work from congested Preempt to General",
            scheduler_before=before, controller=p.descriptor(SOURCE / "prepare.py")))
        response = subprocess.run(["/usr/bin/scontrol", "hold", job], text=True, capture_output=True, timeout=60,
                                  env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
        p.seal(folder / ("WITHDRAW_HOLD-%s.json" % job), dict(
            intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        require(response.returncode == 0, "Hold failed (the request may have started); no withdrawal")
        held = v3_readback(p, job, v3_plan, attempt, held=True)
        require(not v3_traces(cell, path.parent, job), "The v3 request started; it is never withdrawn")
        response = subprocess.run(["/usr/bin/scancel", job], text=True, capture_output=True, timeout=60,
                                  env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
        cancel = p.seal(folder / ("WITHDRAW_CANCEL-%s.json" % job), dict(
            intent=intent, held_scheduler=held, returncode=response.returncode, stdout=response.stdout,
            stderr=response.stderr))
        require(response.returncode == 0, "Cancel failed or ambiguous; reconcile the held request")
        for _ in range(24):
            record = accounting(p, job)
            if record["state"] == "CANCELLED":
                break
            time.sleep(5)
        require(record["state"] == "CANCELLED" and record["start"] in ("None", "Unknown")
                and record["elapsed"] == "00:00:00" and not v3_traces(cell, path.parent, job),
                "Withdrawn v3 request must account as never-started CANCELLED")
        receipt = p.seal(folder / ("WITHDRAWN-%s.json" % job), dict(
            schema=WITHDRAWN_SCHEMA, owner=OWNER, cell=cell, job_id=job, attempt=attempt, ack=ack_ref,
            plan=ack["plan"], intent=intent, cancel=cancel, accounting=record, never_started=True,
            controller=p.descriptor(SOURCE / "prepare.py")))
        return dict(cell=cell, v3_attempt=attempt, job_id=job, state="WITHDRAWN_NEVER_STARTED", receipt=receipt)


def admit(p, cell, roster_ref, attempt, release=False):
    check_attempt(attempt)
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, cell, roster_ref, independent=release)
    afolder = folder / ("attempt-%d" % attempt)
    with p.admission_lock():
        require(contract(p, cell, roster_ref, independent=release)
                == (folder, plan_ref, plan, review_ref, independent_ref),
                "Contract changed while acquiring shared admission lock")
        if not release:
            predecessor = predecessor_allows(p, cell, folder, attempt, plan)
            private_directory(afolder)
            require(not any((afolder / name).exists() for name in
                            ("INTENT.json", "ACK.json", "AMBIGUOUS.json", "FUNCTION_CLAIM.json")),
                    "Existing submission or ambiguous ACK: reconcile, never duplicate")
            parent_status(p, plan["function"]["producer"]["job_id"])
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
            full, census = general_full(rows, int(general[1]), int(gpu_limit))
            require(census["requests"] < int(general[1]),
                    "General route needs a free normal-QoS submission slot for this user")
            claim = p.seal(afolder / "FUNCTION_CLAIM.json", dict(
                owner=OWNER, cell=cell, attempt=attempt, function=plan["function"],
                function_identity=plan["function_identity"], plan=plan_ref, predecessor=predecessor))
            intent = p.seal(afolder / "INTENT.json", dict(
                owner=OWNER, cell=cell, attempt=attempt, plan=plan_ref, review=review_ref, claim=claim,
                argv=argv(cell, plan, attempt), route=dict(ROUTE),
                general_census=dict(census, limits=general, general_full_for_running=full),
                expanded_requests=len(rows), exact_dependency=plan["dependency"]))
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
            return dict(cell=cell, attempt=attempt, job_id=job, state="HELD_VERIFIED", dependency=plan["dependency"])
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
                "Released evaluation must remain dependency-pending or eligible")
        p.seal(afolder / "RELEASE_READBACK.json", dict(job_id=job, scheduler=raw,
                                                      expected_dependency=plan["dependency"]))
        return dict(cell=cell, attempt=attempt, job_id=job, state=row.get("JobState") + ":" + row.get("Reason", ""))


def read_producer(p, cell):
    path = CHECKPOINT_ROOT / cell / "RESULT.json"
    fd = p.open_literal(path, os.O_RDONLY)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_uid == 2708899
                and 0 < before.st_size <= 64 << 20, "Bounded regular own producer result required")
        mount = p.mount_for_fd(stream.fileno(), before.st_dev, "group")
        raw = stream.read((64 << 20) + 1)
        after = os.fstat(stream.fileno())
    require(len(raw) == before.st_size and all(getattr(before, key) == getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
            "Actual opened producer result changed")
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                actual_open_FD_mount=mount, inode=before.st_ino)


def work(p, cell, roster_ref, attempt):
    check_attempt(attempt)
    cell_folder, plan_ref, plan, review_ref, independent_ref = contract(p, cell, roster_ref, independent=True)
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
    qualification, producer_ref = read_producer(p, cell)
    require(qualification["job_id"] == producer["job_id"] and qualification["status"] == PRODUCER_STATUS
            and qualification["cell"] == cell, "Actual completed roster-bound producer required")
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
        official_runtime_reference=evaluator["official_POPE_configuration"], output=str(output))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "self-review", "submit", "release", "withdraw", "worker"))
    parser.add_argument("--cell", required=True)
    parser.add_argument("--roster", required=True)
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    p = load_helpers()
    roster_ref = p.descriptor(Path(args.roster))
    require(roster_ref["sha256"] == args.roster_sha256, "Producer roster bytes differ from the requested pin")
    if args.command == "worker":
        work(p, args.cell, roster_ref, args.attempt)
    elif args.command == "withdraw":
        roster_cell(p, roster_ref, args.cell)
        print(json.dumps(withdraw(p, args.cell), sort_keys=True))
    elif args.command == "plan":
        print(json.dumps(make_plan_file(p, args.cell, roster_ref), sort_keys=True))
    elif args.command == "self-review":
        print(json.dumps(owner_review(p, args.cell, roster_ref, args.note), sort_keys=True))
    else:
        print(json.dumps(admit(p, args.cell, roster_ref, args.attempt, release=args.command == "release"),
                         sort_keys=True))


if __name__ == "__main__":
    main()
