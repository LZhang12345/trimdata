"""Tool1 baseline students on Preempt with accepted checkpoints and qualified-only requeue (v4 controller).

v4 repairs the v3 independent-review blockers (B1/B2: a classified comparator that never unpickles
training_args.bin and separates run identity from numeric/control parity) and folds the production
findings: tolerant acceptance-receipt parsing (N2), final-step resume (N3, model.py), crash-safe
post-success retirement (N4), quarantine storage accounting (N5) and roster-specific independent review
files (N9).  v3 probes 11023177/11023178 were cancelled before start.

The finite roster is a separately pinned JSON file (bound into every plan).  Each Slurm attempt
re-proves its inputs, quarantines unaccepted/partial state, resumes only from this trajectory's
own accepted checkpoint, and seals per-attempt receipts.  Production cells cannot be released
until the actual restart-parity probe has an independent PASS review.
"""

import argparse
import base64
from contextlib import ExitStack, contextmanager
import fcntl
import gc
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


HOME = Path("/home/luyangz/data_selection_multimodal")
SOURCE = HOME / "experiments/helper_tool1_baseline_train_v4_20261102"
MODEL_DIR = SOURCE
CONTROL_ROOT = HOME / "collaboration/helper-tool1-baseline-independent-20261101/baseline-train-v4"
LOCK = HOME / "collaboration/babel-two-track-20261019/ADMISSION.lock"
INDEPENDENT = None
RESTART_REVIEW = CONTROL_ROOT / "INDEPENDENT_RESTART_REVIEW.json"
OWNER = "Tool1-owned-published-baseline-native-training"
REVIEW_STATUS = "SELF_REVIEWED_TOOL1_BASELINE_TRAIN_V4_NOT_INDEPENDENT_ACCEPTANCE"
INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_BASELINE_TRAIN_V4"
RESTART_STATUS = "PASS_INDEPENDENT_ACTUAL_RESTART_PARITY_TOOL1_BASELINE_TRAIN_V4"
ROSTER_SCHEMA = "tool1-baseline-train-roster-v1"
UID = 2708899
N = 660763
POOL = {
    "path": "/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/POOL.json",
    "bytes": 18754787,
    "sha256": "c76040a17ca0dc1e99fea14514618edade3171a9b05f6ccc3e4751a5939e4964",
}
POOL_RECORDS = {
    "path": "/data/user_data/luyangz/native_campaign_20261023/root-wholepool-authority-v1/authority/records.json",
    "bytes": 760064389,
    "sha256": "1c53e08b940af27483201efc4c2fa27e3f5c4f7862484e078736c718673c1d36",
}
MODEL_SOURCE = {"path": str(MODEL_DIR / "model.py"), "bytes": 20966,
                "sha256": "b53089293b49a5a2d9fb8d433999105d06c70629f3be788bbf35d3089a728b11"}
COMPARE_SOURCE = {"path": str(MODEL_DIR / "compare.py"), "bytes": 15217,
                  "sha256": "ea57b77c0fb324eb4e5c38f1f6925fa208fa4727fd49c50ac038a8426783e719"}
REFERENCE_CONFIG = {
    "path": str(HOME / "collaboration/root-general-all-plan-20261028/submission-check-20261029/"
                "baseline-ready-v1/cells/coincide-native10-s17/CONFIG_FINAL.json"),
    "bytes": 67760, "sha256": "450160a8c561fbd33f4e98acca9805cc04ed727955dfb253c58e461cdd0fe9ec",
}
AUTHORITY = {
    "author_instructions": [
        "Push toward the real submission of the baseline jobs recorded in collaboration/EXPERIMENTAL_ANCHOR_20261101.md.",
        "If the queue for general partition is full, submit to preempt.",
    ],
    "standing_rule": "Preempt training requires actual compatible accepted checkpoint/optimizer/RNG/cursor "
                     "resume qualification; no unqualified auto-requeue.",
    "original_session4_session6_root_claims_preserved": True,
    "no_checkpoint_RNG_grant_or_claim_transfer": True,
}
PYTHON = "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python"
OUTPUT_ROOT = Path("/data/hf_cache/project_checkpoints/DS-MultiModal/helper-tool1-baseline-train-v4-20261102")
EXCLUDED = "babel-p9-20"
SLURM_ENV = {"HOME": str(HOME), "PATH": "/usr/bin:/bin", "USER": "luyangz",
             "LOGNAME": "luyangz", "LC_ALL": "C"}
KINDS = ("production", "probe-reference", "probe-requeue")
CELL = None
SPEC = None
ROSTER_REF = None
CONTROL = PLAN = REVIEW = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def parent_fd(path):
    text = os.fspath(path)
    require(isinstance(text, str) and text.startswith("/") and not text.startswith("//")
            and not text.endswith("/") and "\0" not in text, "Literal absolute file path required")
    parts = text[1:].split("/")
    require(parts and all(part not in ("", ".", "..") for part in parts), "Literal path components required")
    with ExitStack() as stack:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        stack.callback(os.close, fd)
        for component in parts[:-1]:
            fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=fd)
            stack.callback(os.close, fd)
        return os.dup(fd), parts[-1]


def open_literal(path, flags, mode=0o600):
    directory, name = parent_fd(path)
    try:
        return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, mode, dir_fd=directory)
    finally:
        os.close(directory)


def raw_home(path):
    fd = open_literal(path, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == UID and before.st_size <= 4 << 20,
                "Small regular owned HOME file required")
        chunks, remaining = [], before.st_size + 1
        while remaining:
            block = os.read(fd, min(1 << 20, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw, after = b"".join(chunks), os.fstat(fd)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(len(raw) == before.st_size and all(getattr(before, k) == getattr(after, k) for k in keys),
                "HOME file changed while open")
        return raw
    finally:
        os.close(fd)


def descriptor(path):
    path = Path(path)
    raw = raw_home(path)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def checked(ref):
    path = Path(ref["path"])
    raw = raw_home(path)
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"],
            "Pinned HOME descriptor changed: " + str(path))
    return raw


def seal(path, value):
    path = Path(path)
    raw = canonical(value) + b"\n"
    directory, name = parent_fd(path)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=directory)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def control_exists(name):
    require(isinstance(name, str) and name and "/" not in name and name not in (".", ".."),
            "Literal control filename required")
    directory = open_literal(CONTROL, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(directory)


def private_directory(parent, name):
    directory = open_literal(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=directory)
        except FileExistsError:
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
                    "Owned private directory required: " + str(Path(parent) / name))
    finally:
        os.close(directory)


def ref_ok(value):
    return (isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}
            and isinstance(value["path"], str) and value["path"].startswith("/")
            and type(value["bytes"]) is int and value["bytes"] > 0
            and isinstance(value["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None)


def natural_steps(k):
    return (k + 3) // 4 // 32


def validate_roster(roster):
    require(isinstance(roster, dict) and roster.get("schema") == ROSTER_SCHEMA
            and isinstance(roster.get("cells"), dict) and roster["cells"], "Exact roster schema required")
    names = set()
    for cell, spec in roster["cells"].items():
        require(re.fullmatch(r"[a-z0-9][a-z0-9-]{2,60}", cell) is not None, "Literal roster cell name required")
        require(isinstance(spec, dict) and spec.get("kind") in KINDS
                and spec.get("method") in ("uniform", "datatailor")
                and type(spec.get("selection_seed")) is int and spec.get("training_seed") in (17, 29, 43)
                and type(spec.get("K")) is int and 1024 <= spec["K"] <= N
                and ref_ok(spec.get("selection")) and spec.get("selection_kind") in ("home", "group")
                and ref_ok(spec.get("records")) and spec["records"]["path"].startswith("/data/hf_cache/")
                and spec.get("selection_status") in (None, "PASS_SELECTION")
                and type(spec.get("save_steps")) is int and spec.get("retain") in (1, 2)
                and type(spec.get("minutes")) is int and 30 <= spec["minutes"] <= 4320
                and isinstance(spec.get("job_name"), str)
                and re.fullmatch(r"dsmm-t1[a-z0-9-]{3,50}", spec["job_name"]) is not None,
                "Roster cell fields are incomplete or out of bounds: " + cell)
        require(spec["job_name"] not in names, "Duplicate roster job name")
        names.add(spec["job_name"])
        if spec["selection_kind"] == "home":
            require(spec["selection"]["path"].startswith(str(HOME) + "/"), "HOME selection must live in HOME")
        else:
            require(spec["selection"]["path"].startswith("/data/hf_cache/"), "Group selection must live on HF NAS")
        steps = natural_steps(spec["K"])
        require(steps >= 2, "At least two native optimizer boundaries required")
        if spec["kind"] == "production":
            require(spec.get("probe_max_steps") is None and spec.get("stop_after") is None
                    and spec.get("depends_on") is None and spec.get("compare_with") is None
                    and spec["retain"] == 1 and 1 <= spec["save_steps"] <= steps and spec.get("requeue") is True,
                    "Production cells: natural horizon, single retained state and qualified requeue only")
        else:
            limit = spec.get("probe_max_steps")
            require(type(limit) is int and 2 <= limit < steps and spec["retain"] == 2
                    and 1 <= spec["save_steps"] < limit, "Bounded probe horizon required")
            if spec["kind"] == "probe-reference":
                require(spec.get("stop_after") is None and spec.get("requeue") is False
                        and spec.get("depends_on") is None and spec.get("compare_with") is None,
                        "Reference probe never stops or requeues")
            else:
                require(type(spec.get("stop_after")) is int and spec["stop_after"] % spec["save_steps"] == 0
                        and 0 < spec["stop_after"] < limit and spec.get("requeue") is True
                        and spec.get("depends_on") == spec.get("compare_with")
                        and roster["cells"].get(spec["depends_on"], {}).get("kind") == "probe-reference",
                        "Requeue probe stops at an accepted boundary and compares to its reference")
                ref = roster["cells"][spec["depends_on"]]
                for key in ("method", "selection_seed", "training_seed", "K", "selection", "records",
                            "probe_max_steps", "save_steps", "retain"):
                    require(ref.get(key) == spec.get(key), "Requeue probe must match its reference: " + key)
    inventory = roster.get("input_inventory")
    require(inventory is None or (isinstance(inventory, list) and all(
        isinstance(x, str) and x.startswith("/data/hf_cache/project_features/DS-MultiModal/") and ".." not in x
        for x in inventory)), "Read-only HF-cache inventory paths required")
    return roster


def load_roster(ref):
    require(ref_ok(ref) and ref["path"].startswith(str(CONTROL_ROOT) + "/"), "Roster must be a pinned control file")
    return validate_roster(json.loads(checked(ref)))


def independent_path(roster_ref):
    """One independent review file per pinned roster: a later roster never replaces a bound review."""
    return CONTROL_ROOT / ("INDEPENDENT_REVIEW-%s.json" % roster_ref["sha256"][:16])


def configure(cell, roster_ref):
    global CELL, SPEC, ROSTER_REF, CONTROL, PLAN, REVIEW, INDEPENDENT
    roster = load_roster(roster_ref)
    require(isinstance(cell, str) and cell in roster["cells"], "Exact finite roster cell required")
    CELL, SPEC, ROSTER_REF = cell, roster["cells"][cell], roster_ref
    INDEPENDENT = independent_path(roster_ref)
    CONTROL = CONTROL_ROOT / cell
    PLAN, REVIEW = CONTROL / "PLAN.json", CONTROL / "OWNER_REVIEW.json"
    return roster


def horizon():
    k = SPEC["K"]
    return {"loader_batches": (k + 3) // 4, "natural_optimizer_boundaries": natural_steps(k),
            "probe_max_steps": SPEC.get("probe_max_steps"),
            "optimized_records": natural_steps(k) * 128, "unoptimized_tail_records": k - natural_steps(k) * 128,
            "top_up": False}


def dependency_job():
    if SPEC.get("depends_on") is None:
        return None
    ack = json.loads(checked(descriptor(CONTROL_ROOT / SPEC["depends_on"] / "ACK.json")))
    require(ack["cell"] == SPEC["depends_on"] and re.fullmatch(r"[1-9][0-9]*", ack["job_id"]) is not None,
            "Dependency ACK must be the exact reference probe")
    return ack["job_id"]


def function_contract():
    return {
        "owner": OWNER, "purpose": "tool1-baseline-native-one-epoch-preempt-accepted-checkpoints",
        "cell": CELL, "kind": SPEC["kind"], "roster": ROSTER_REF, "spec": SPEC,
        "pool": POOL, "pool_records": POOL_RECORDS, "N": N, "horizon": horizon(),
        "model_source": MODEL_SOURCE, "compare_source": COMPARE_SOURCE,
        "native_reference": REFERENCE_CONFIG, "native_python": PYTHON, "checkpoint_root": str(OUTPUT_ROOT),
        "authority": AUTHORITY, "benchmark_QA_reads": 0, "selected_payload_copies": 0, "selection_rebuilt": False,
        "independent_training_acceptance": False, "benchmark_performance_measured": False,
    }


def minutes_text(minutes):
    """Slurm's own TimeLimit display: HH:MM:SS below one day, D-HH:MM:SS otherwise."""
    days, rest = divmod(minutes, 1440)
    if days:
        return "%d-%02d:%02d:00" % (days, rest // 60, rest % 60)
    return "%02d:%02d:00" % (rest // 60, rest % 60)


def make_plan(source_ref, batch_ref):
    function = function_contract()
    function_id = hashlib.sha256(canonical(function)).hexdigest()
    return {
        "schema": "tool1-baseline-native-preempt-v4", "owner": OWNER, "cell": CELL,
        "function": function, "function_identity": function_id,
        "sources": {"prepare.py": source_ref, "run.sbatch": batch_ref, "model.py": MODEL_SOURCE,
                    "compare.py": COMPARE_SOURCE},
        "resources": {"account": "cx", "partition": "preempt", "qos": "preempt_qos", "gres": "gpu:L40S:1",
                      "cpus": 8, "memory": "64G", "time": minutes_text(SPEC["minutes"]), "exclude": EXCLUDED,
                      "requeue": SPEC["requeue"]},
        "dependency": None if dependency_job() is None else "afterok:" + dependency_job(),
        "job_name": SPEC["job_name"], "comment": "dsmm-t1-baseline-train-v4:" + function_id,
        "output_directory": str(CONTROL), "checkpoint_output": str(OUTPUT_ROOT / CELL),
        "new_owned_experiment_claim": True, "no_grant_or_existing_claim_transfer": True,
    }


def validate_owner_review(peer, plan_ref, plan):
    require(peer.get("status") == REVIEW_STATUS and peer.get("independent") is False
            and peer.get("ai_review_disclosed") is True and peer.get("blocking_findings") == []
            and peer.get("plan") == plan_ref and peer.get("sources") == plan["sources"]
            and peer.get("function_identity") == plan["function_identity"]
            and peer.get("cell") == CELL and peer.get("reviewer") == OWNER,
            "Truthful current owner self-review bound to this exact plan is required")


def validate_independent(review, plan_ref, plan):
    require(review.get("status") == INDEPENDENT_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER) and review.get("blocking_findings") == []
            and review.get("sources") == plan["sources"] and review.get("roster") == ROSTER_REF
            and isinstance(review.get("plans"), dict) and review["plans"].get(CELL) == plan_ref,
            "Independent exact source/config/roster PASS bound to this plan is required")


def validate_restart(review):
    require(review.get("status") == RESTART_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER) and review.get("blocking_findings") == []
            and review.get("model_source") == MODEL_SOURCE and review.get("verdict") in (
                "PASS_BITWISE_RESTART_PARITY", "PASS_REVIEWED_RESTART_PARITY_WITHIN_MEASURED_NONDETERMINISM")
            and isinstance(review.get("parity_report"), dict) and ref_ok(review["parity_report"]),
            "Independent actual restart-parity PASS for this exact model source is required")
    parity = json.loads(checked(review["parity_report"]))
    require(parity.get("verdict") == review["verdict"] or (
        review["verdict"] == "PASS_REVIEWED_RESTART_PARITY_WITHIN_MEASURED_NONDETERMINISM"
        and parity.get("verdict") == "INCONCLUSIVE_NONDETERMINISTIC_BASELINE"),
        "Restart review verdict must match the actual parity report")
    return parity


def current_contract(independent=False):
    require(Path(__file__).absolute() == SOURCE / "prepare.py" and os.getuid() == UID,
            "Exact owned staged source and UID required")
    fd = open_literal(CONTROL, os.O_RDONLY | os.O_DIRECTORY)
    try:
        info = os.fstat(fd)
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not (info.st_mode & 0o022),
                "Literal private owned control directory required")
    finally:
        os.close(fd)
    source_ref, batch_ref = descriptor(SOURCE / "prepare.py"), descriptor(SOURCE / "run.sbatch")
    plan_ref = descriptor(PLAN)
    plan = json.loads(checked(plan_ref))
    require(plan == make_plan(source_ref, batch_ref), "Exact reviewed finite plan required")
    review_ref = descriptor(REVIEW)
    validate_owner_review(json.loads(checked(review_ref)), plan_ref, plan)
    checked(MODEL_SOURCE)
    checked(COMPARE_SOURCE)
    checked(REFERENCE_CONFIG)
    if SPEC["selection_kind"] == "home":
        checked(SPEC["selection"])
    independent_ref = None
    if independent:
        independent_ref = descriptor(INDEPENDENT)
        validate_independent(json.loads(checked(independent_ref)), plan_ref, plan)
    return plan_ref, plan, review_ref, independent_ref


def fields(raw):
    rows = [line for line in raw.splitlines() if line.strip()]
    require(len(rows) == 1, "One exact Slurm object required")
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=", rows[0]))
    require(matches and matches[0].start() == 0, "Malformed Slurm metadata")
    result = {}
    for i, match in enumerate(matches):
        require(match[1] not in result, "Duplicate Slurm key")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(rows[0])
        result[match[1]] = rows[0][match.end():end].strip()
    return result


def query(args):
    return subprocess.check_output(args, text=True, timeout=30, env=SLURM_ENV, stdin=subprocess.DEVNULL)


@contextmanager
def admission_lock():
    fd = open_literal(LOCK, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def qos_limits(name):
    raw = query(["/usr/bin/sacctmgr", "-nP", "show", "qos", name, "format=Name,MaxSubmitPU,MaxJobsPU,MaxTRESPU"])
    values = [line.split("|") for line in raw.splitlines() if line.strip()]
    require(len(values) == 1 and len(values[0]) == 4 and values[0][0] == name
            and values[0][1].isdecimal() and values[0][2].isdecimal(), "Actual finite QoS limits required: " + name)
    tres = dict(x.split("=", 1) for x in values[0][3].split(",") if "=" in x)
    require(tres.get("gres/gpu", "").isdecimal(), "Actual QoS GPU limit required: " + name)
    return {"name": name, "max_submit": int(values[0][1]), "max_jobs": int(values[0][2]),
            "max_gpus": int(tres["gres/gpu"]), "raw": raw}


def general_full(rows, general):
    normal = [row for row in rows if row[4] == "normal"]
    running_gpus = sum(row[2] == "RUNNING" and "gpu" in row[5] for row in normal)
    pending = sum(row[2] == "PENDING" for row in normal)
    return (len(normal) >= general["max_submit"]
            or (running_gpus >= general["max_gpus"] and pending > 0)), {
        "normal_requests": len(normal), "normal_running_gpu_requests": running_gpus,
        "normal_pending": pending, "limits": general}


def queue_and_limits(function_id):
    queue = query(["/usr/bin/squeue", "-u", "luyangz", "-r", "-h", "-o", "%i|%j|%T|%P|%q|%b|%k"])
    rows = [line.split("|") for line in queue.splitlines() if line.strip()]
    require(all(len(row) == 7 for row in rows), "Expanded queue census is incomplete")
    require(not any(row[1] == SPEC["job_name"] or function_id in row[6] for row in rows),
            "An exact live baseline controller already exists")
    preempt = qos_limits("preempt_qos")
    require(sum(row[4] == "preempt_qos" for row in rows) < preempt["max_submit"],
            "Actual preempt_qos submission limit is full")
    full, general = general_full(rows, qos_limits("normal"))
    require(full, "Preempt route is only used while the actual General queue is full for this user")
    return {"expanded_jobs": len(rows), "preempt_qos_requests": sum(row[4] == "preempt_qos" for row in rows),
            "preempt_limits": preempt, "general": general}


def argv(plan):
    args = ["/usr/bin/sbatch", "--parsable", "--hold",
            "--requeue" if SPEC["requeue"] else "--no-requeue", "--export=NIL",
            "--partition=preempt", "--account=cx", "--qos=preempt_qos", "--gres=gpu:L40S:1",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
            "--time=" + plan["resources"]["time"], "--exclude=" + EXCLUDED,
            "--job-name=" + SPEC["job_name"], "--comment=" + plan["comment"]]
    if plan["dependency"] is not None:
        args.append("--dependency=" + plan["dependency"])
    return args + ["--chdir=" + str(HOME), "--input=/dev/null", "--output=/dev/null", "--error=/dev/null",
                   str(SOURCE / "run.sbatch"), CELL, ROSTER_REF["path"], ROSTER_REF["sha256"]]


def expected_identity(plan, job):
    return {
        "JobId": job, "UserId": "luyangz(2708899)", "Account": "cx", "Partition": "preempt",
        "QOS": "preempt_qos", "JobName": SPEC["job_name"], "Comment": plan["comment"],
        "NumCPUs": "8", "NumTasks": "1", "TresPerNode": "gres/gpu:L40S:1",
        "TimeLimit": plan["resources"]["time"], "ExcNodeList": EXCLUDED,
        "Requeue": "1" if SPEC["requeue"] else "0", "Command": str(SOURCE / "run.sbatch"),
        "WorkDir": str(HOME), "StdIn": "/dev/null", "StdOut": "/dev/null", "StdErr": "/dev/null",
    }


def validate_dependency(row, plan):
    value = row.get("Dependency") or ""
    if plan["dependency"] is None:
        require(value == "(null)", "No dependency expected")
    else:
        parent = plan["dependency"].split(":", 1)[1]
        require(re.fullmatch(r"afterok:" + re.escape(parent) + r"(?:\((?:unfulfilled|fulfilled)\))?", value)
                is not None or value == "(null)", "Exact successful reference dependency required")


def request_tres(row, key):
    tres = dict(x.split("=", 1) for x in row.get(key, "").split(",") if "=" in x)
    require(tres.get("cpu") == "8" and tres.get("mem") in ("64G", "65536M") and tres.get("gres/gpu") == "1",
            "Requested/allocated CPU, memory or GPU differs")


def held_readback(job, plan):
    raw = query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = fields(raw)
    expected = dict(expected_identity(plan, job), JobState="PENDING", Reason="JobHeldUser", Restarts="0")
    require(all(row.get(k) == v for k, v in expected.items()) and row.get("NumNodes") in ("1", "1-1")
            and row.get("RunTime") == "00:00:00", "Held never-started request identity/resources differ")
    require(shlex.split(row.get("SubmitLine", "")) == argv(plan), "Held submission argv differs")
    validate_dependency(row, plan)
    request_tres(row, "ReqTRES")
    spool = subprocess.check_output(["/usr/bin/scontrol", "write", "batch_script", job, "-"], timeout=30,
                                    env=SLURM_ENV, stdin=subprocess.DEVNULL)
    require(spool == checked(plan["sources"]["run.sbatch"]), "Held spool does not match reviewed source")
    return raw, spool


def make_plan_file():
    require(Path(__file__).absolute() == SOURCE / "prepare.py" and os.getuid() == UID,
            "Exact owned staged source and UID required")
    private_directory(CONTROL_ROOT.parent, CONTROL_ROOT.name)
    private_directory(CONTROL_ROOT, CELL)
    require(not control_exists("PLAN.json"), "Existing plan is immutable; use a new version")
    checked(MODEL_SOURCE)
    checked(COMPARE_SOURCE)
    return seal(PLAN, make_plan(descriptor(SOURCE / "prepare.py"), descriptor(SOURCE / "run.sbatch")))


def owner_review(notes):
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete self-review notes required")
    plan_ref = descriptor(PLAN)
    plan = json.loads(checked(plan_ref))
    require(plan == make_plan(descriptor(SOURCE / "prepare.py"), descriptor(SOURCE / "run.sbatch")),
            "Self-review must bind the current exact plan")
    return seal(REVIEW, {
        "status": REVIEW_STATUS, "independent": False, "ai_review_disclosed": True, "reviewer": OWNER,
        "cell": CELL, "plan": plan_ref, "sources": plan["sources"], "function_identity": plan["function_identity"],
        "blocking_findings": [], "notes": notes,
    })


def submit():
    plan_ref, plan, review_ref, _ = current_contract()
    with admission_lock():
        require(current_contract()[:3] == (plan_ref, plan, review_ref),
                "Plan, review or sources changed while acquiring the admission lock")
        require(not any(control_exists(name) for name in
                        ("INTENT.json", "ACK.json", "AMBIGUOUS.json", "FUNCTION_CLAIM.json")),
                "Existing or ambiguous attempt: reconcile, never retry")
        census = queue_and_limits(plan["function_identity"])
        claim = seal(CONTROL / "FUNCTION_CLAIM.json", {
            "owner": OWNER, "cell": CELL, "function": plan["function"],
            "function_identity": plan["function_identity"], "plan": plan_ref,
            "not_a_training_grant_or_existing_claim_transfer": True, "scientific_output_not_accepted": True,
        })
        intent = seal(CONTROL / "INTENT.json", {
            "owner": OWNER, "cell": CELL, "plan": plan_ref, "review": review_ref, "claim": claim,
            "argv": argv(plan), "census": census,
        })
        try:
            response = subprocess.run(argv(plan), capture_output=True, text=True, timeout=45,
                                      env=SLURM_ENV, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as error:
            seal(CONTROL / "AMBIGUOUS.json", {"intent": intent, "error": repr(error),
                                              "reconcile_exact_comment_before_retry": True})
            raise
        good = response.returncode == 0 and re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?\n?", response.stdout)
        if good:
            job = response.stdout.strip().split(";")[0]
            ack = seal(CONTROL / "ACK.json", {"owner": OWNER, "cell": CELL, "job_id": job, "intent": intent,
                                              "plan": plan_ref})
        seal(CONTROL / "SUBMISSION_RESULT.json", {"intent": intent, "returncode": response.returncode,
                                                   "stdout": response.stdout, "stderr": response.stderr})
        require(good, "No unambiguous scalar ACK; reconcile intent and comment, never blindly retry")
        raw, spool = held_readback(job, plan)
        seal(CONTROL / "HELD_READBACK.json", {"ACK": ack, "job_id": job, "scheduler": raw,
                                               "scheduler_spool_base64": base64.b64encode(spool).decode("ascii")})
        return {"cell": CELL, "job_id": job, "state": "HELD_VERIFIED", "dependency": plan["dependency"]}


def release():
    plan_ref, plan, review_ref, independent_ref = current_contract(independent=True)
    restart_ref = None
    if SPEC["kind"] == "production":
        restart_ref = descriptor(RESTART_REVIEW)
        validate_restart(json.loads(checked(restart_ref)))
    intent_ref = descriptor(CONTROL / "INTENT.json")
    intent = json.loads(checked(intent_ref))
    ack_ref = descriptor(CONTROL / "ACK.json")
    ack = json.loads(checked(ack_ref))
    require(intent["plan"] == plan_ref and intent["review"] == review_ref and ack["intent"] == intent_ref
            and ack["plan"] == plan_ref and ack["cell"] == CELL and ack["owner"] == OWNER
            and re.fullmatch(r"[1-9][0-9]*", ack["job_id"]) is not None, "Own actual intent/ACK/plan binding required")
    with admission_lock():
        require(current_contract(independent=True) == (plan_ref, plan, review_ref, independent_ref),
                "Plan, reviews or sources changed while acquiring the admission lock")
        require(not control_exists("RELEASE_INTENT.json"), "Existing release requires reconciliation")
        held_readback(ack["job_id"], plan)
        release_intent = seal(CONTROL / "RELEASE_INTENT.json", {
            "ACK": ack_ref, "job_id": ack["job_id"], "independent_review": independent_ref,
            "restart_review": restart_ref})
        response = subprocess.run(["/usr/bin/scontrol", "release", ack["job_id"]], capture_output=True, text=True,
                                  timeout=30, env=SLURM_ENV, stdin=subprocess.DEVNULL)
        seal(CONTROL / "RELEASE_RESULT.json", {"intent": release_intent, "returncode": response.returncode,
                                               "stdout": response.stdout, "stderr": response.stderr})
        require(response.returncode == 0, "Release failed or is ambiguous; no retry")
        raw = query(["/usr/bin/scontrol", "show", "job", "-o", ack["job_id"]])
        row = fields(raw)
        require(all(row.get(k) == v for k, v in expected_identity(plan, ack["job_id"]).items())
                and row.get("JobState") in ("PENDING", "CONFIGURING", "RUNNING")
                and row.get("Reason") not in ("JobHeldUser", "JobHeldAdmin", "DependencyNeverSatisfied"),
                "Post-release request identity or eligibility differs")
        validate_dependency(row, plan)
        seal(CONTROL / "RELEASE_READBACK.json", {"ACK": ack_ref, "scheduler": raw})
        return {"cell": CELL, "job_id": ack["job_id"], "state": row.get("JobState") + ":" + row.get("Reason", "")}


def parse_mount(fdinfo, mountinfo, device, kind):
    require(kind in ("pool", "group"), "Known input filesystem kind required")
    ids = re.findall(r"^mnt_id:\s*([0-9]+)\s*$", fdinfo, re.MULTILINE)
    require(len(ids) == 1, "Exact open FD mount ID required")
    found = []
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(" - ")
        a, b = left.split(), right.split()
        if a and a[0] == ids[0]:
            require(separator and len(a) >= 6 and len(b) >= 3, "Malformed FD mount metadata")
            found.append((a, b))
    require(len(found) == 1, "One mount entry must match actual FD")
    a, b = found[0]
    require(a[2] == "%d:%d" % (os.major(device), os.minor(device)) and b[0] in ("nfs", "nfs4"),
            "Actual FD must be on matching NFS device")
    if kind == "group":
        require(a[4] == "/data/hf_cache" and b[0] == "nfs4"
                and b[1] == "babel-nas-2:/srv/nfs/data/hf_cache", "Exact shared HF NAS required")
    else:
        require(a[4] in ("/data", "/data/user_data", "/data/user_data/luyangz")
                and bool(b[1]) and ":" in b[1], "Pool FD must belong to a real user-data NFS mount")
    return {"mount_id": int(ids[0]), "device": a[2], "target": a[4], "filesystem": b[0], "source": b[1]}


def mount_for_fd(fd, device, kind):
    proc = Path("/proc") / str(os.getpid())
    return parse_mount((proc / "fdinfo" / str(fd)).read_text(), (proc / "mountinfo").read_text(), device, kind)


def allowed_data():
    allowed = [(POOL, "pool"), (POOL_RECORDS, "pool"), (SPEC["records"], "group")]
    if SPEC["selection_kind"] == "group":
        allowed.append((SPEC["selection"], "group"))
    return allowed


def read_data(ref, kind):
    require((ref, kind) in allowed_data(), "Exact pinned cell input descriptor and filesystem kind required")
    fd = open_literal(ref["path"], os.O_RDONLY)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == UID and before.st_size == ref["bytes"],
                "Opened DATA file must be regular, owned by the project user and exact size")
        mount = mount_for_fd(fd, before.st_dev, kind)
        chunks, remaining = [], ref["bytes"] + 1
        while remaining:
            block = os.read(fd, min(1 << 20, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw, after = b"".join(chunks), os.fstat(fd)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"]
                and all(getattr(before, field) == getattr(after, field) for field in stable),
                "Opened DATA bytes or FD identity changed")
        return raw, {"reference": ref, "mount": mount, "inode": after.st_ino, "uid": after.st_uid}
    finally:
        os.close(fd)


def validate_selection(pool, rows, selection, selected_rows):
    """Selected file must be exactly the pool rows named by the frozen selection, in its order."""
    ids = selection.get("record_ids")
    require(isinstance(ids, list) and all(isinstance(i, str) for i in ids)
            and selection.get("count") == len(ids) == len(set(ids)) == SPEC["K"]
            and selection.get("records_json") == SPEC["records"] and selection.get("pool") == POOL
            and selection.get("method") == SPEC["method"]
            and selection.get("selection_seed") == SPEC["selection_seed"], "Frozen selection identity differs")
    if SPEC["selection_status"] is not None:
        require(selection.get("status") == SPEC["selection_status"], "Qualified selection status required")
    require(isinstance(pool.get("record_ids"), list) and isinstance(rows, list)
            and pool.get("count") == len(rows) == len(pool["record_ids"]) == N
            and pool.get("records_json", {}).get("sha256") == POOL_RECORDS["sha256"], "Complete frozen pool required")
    lookup = dict(zip(pool["record_ids"], rows))
    require(len(lookup) == N, "Unique pool record identities required")
    require(isinstance(selected_rows, list) and len(selected_rows) == SPEC["K"]
            and all(i in lookup for i in ids) and all(type(row) is dict for row in selected_rows)
            and selected_rows == [lookup[i] for i in ids],
            "Selected file must equal the pool's whole original conversations in frozen order")
    payload = canonical(selected_rows) + b"\n"
    return {"K": SPEC["K"], "ordered_ID_sha256": hashlib.sha256(canonical(ids)).hexdigest(),
            "canonical_selected_rows_sha256": hashlib.sha256(payload).hexdigest(),
            "image_rows": sum("image" in row for row in selected_rows),
            "text_only_rows": sum("image" not in row for row in selected_rows)}


def valid_singleton_node(node):
    if not isinstance(node, str) or len(node) > 64:
        return False
    match = re.fullmatch(r"babel-([a-z])([0-9]+)-([0-9]+)", node)
    return match is not None and not (match[1] == "p" and match[2].lstrip("0") == "9"
                                      and match[3].lstrip("0") == "20")


def validate_process(job, pid, node, raw_step, pids, step, status):
    own = [line.split() for line in pids.splitlines() if line.split()[:2] == [str(pid), job]]
    require(raw_step in ("-5", "batch") and len(own) == 1 and len(own[0]) >= 3 and own[0][2] == "batch"
            and re.findall(r"^NSpid:\s+(.+)$", status, re.MULTILINE) == [str(pid)]
            and step.get("StepId") == job + ".batch" and step.get("UserId") == str(UID)
            and step.get("Name") == "batch" and step.get("State") == "RUNNING"
            and step.get("NodeList") == node, "Fresh PID, namespace and symbolic batch identity required")


def restart_count():
    value = os.environ.get("SLURM_RESTART_COUNT", "0")
    require(re.fullmatch(r"[0-9]+", value) is not None and int(value) <= 20, "Bounded restart count required")
    value = int(value)
    require(value == 0 or SPEC["requeue"], "A no-requeue cell cannot be a restarted attempt")
    return value


def actual_actor(plan_ref, plan, review_ref, independent_ref, restart):
    job = os.environ.get("SLURM_JOB_ID", "")
    require(os.getuid() == UID and re.fullmatch(r"[1-9][0-9]*", job) is not None
            and not os.environ.get("SLURM_ARRAY_JOB_ID"), "Actual owned scalar allocation required")
    intent_ref = descriptor(CONTROL / "INTENT.json")
    intent = json.loads(checked(intent_ref))
    ack_ref = descriptor(CONTROL / "ACK.json")
    ack = json.loads(checked(ack_ref))
    claim = json.loads(checked(intent["claim"]))
    require(intent["plan"] == plan_ref and intent["review"] == review_ref
            and ack["job_id"] == job and ack["intent"] == intent_ref and ack["plan"] == plan_ref
            and ack["owner"] == OWNER and ack["cell"] == CELL
            and claim["function_identity"] == plan["function_identity"] and claim["owner"] == OWNER
            and claim["function"] == plan["function"] and claim["plan"] == plan_ref,
            "Actual own claim, plan, review, intent and ACK required")
    release_intent = json.loads(checked(descriptor(CONTROL / "RELEASE_INTENT.json")))
    require(release_intent["ACK"] == ack_ref and release_intent["job_id"] == job
            and release_intent["independent_review"] == independent_ref,
            "Own independently reviewed release intent required before DATA")
    if SPEC["kind"] == "production":
        require(release_intent["restart_review"] == descriptor(RESTART_REVIEW), "Restart review changed after release")
        validate_restart(json.loads(checked(release_intent["restart_review"])))
    raw = query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = fields(raw)
    node = socket.gethostname().split(".")[0]
    expected = dict(expected_identity(plan, job), JobState="RUNNING", Restarts=str(restart), NodeList=node)
    require(valid_singleton_node(node) and all(row.get(k) == v for k, v in expected.items())
            and row.get("NumNodes") in ("1", "1-1"), "Actual running attempt identity/resources differ")
    require(shlex.split(row.get("SubmitLine", "")) == argv(plan), "Running submission argv differs")
    request_tres(row, "AllocTRES")
    spool = subprocess.check_output(["/usr/bin/scontrol", "write", "batch_script", job, "-"], timeout=30,
                                    env=SLURM_ENV, stdin=subprocess.DEVNULL)
    require(spool == checked(plan["sources"]["run.sbatch"]), "Actual running spool differs")
    pids = query(["/usr/bin/scontrol", "listpids", job])
    step_raw = query(["/usr/bin/scontrol", "show", "step", job + ".batch", "-o"])
    status = (Path("/proc") / str(os.getpid()) / "status").read_text()
    validate_process(job, os.getpid(), node, os.environ.get("SLURM_STEP_ID"), pids, fields(step_raw), status)
    forbidden = ("BASH_ENV", "ENV", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")
    require(all(key not in os.environ for key in forbidden) and os.environ.get("PATH") == "/usr/bin:/bin",
            "Actual restricted bootstrap environment differs")
    log_path = CONTROL / ("worker-%s-r%d.log" % (job, restart))
    log_fd = open_literal(log_path, os.O_RDONLY)
    try:
        log_stat, out_stat = os.fstat(log_fd), os.fstat(1)
        require(stat.S_ISREG(log_stat.st_mode) and log_stat.st_uid == UID
                and (log_stat.st_dev, log_stat.st_ino) == (out_stat.st_dev, out_stat.st_ino)
                == (os.fstat(2).st_dev, os.fstat(2).st_ino), "Actual per-attempt HOME log binding required")
    finally:
        os.close(log_fd)
    return {"job_id": job, "restart_count": restart, "pid": os.getpid(), "node": node,
            "parent_scheduler": raw, "step_scheduler": step_raw, "process_table": pids,
            "scheduler_spool_base64": base64.b64encode(spool).decode("ascii"), "secure_log_path": str(log_path)}


def sha_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def checkpoint_files(path):
    files = {}
    for item in sorted(Path(path).rglob("*")):
        require(not item.is_symlink(), "Checkpoint symlink is not accepted")
        if item.is_file() and item.name != "ACCEPTED.json":
            files[str(item.relative_to(path))] = {"bytes": item.stat().st_size, "sha256": sha_file(item)}
    return files


def plan_resume(output, identity, restart):
    """Pick this trajectory's newest verified accepted state; move everything else to quarantine."""
    training = output / "training"
    if not training.exists():
        return None, []
    require(training.is_dir() and not training.is_symlink(), "Training directory must be a real directory")
    accepted, moved = [], []
    for item in sorted(training.iterdir()):
        receipt_path = item / "ACCEPTED.json"
        if (item.is_dir() and not item.is_symlink() and re.fullmatch(r"checkpoint-[1-9][0-9]*", item.name)
                and receipt_path.is_file()):
            try:
                receipt = json.loads(receipt_path.read_bytes())
            except ValueError:
                receipt = None
            if (isinstance(receipt, dict) and receipt.get("status") == "ACCEPTED_FULL_NATIVE_STATE"
                    and receipt.get("identity") == identity
                    and receipt.get("global_step") == int(item.name.split("-")[1])
                    and checkpoint_files(item) == receipt.get("files")):
                accepted.append((receipt["global_step"], item, receipt))
                continue
    keep = max(accepted, key=lambda x: x[0]) if accepted else None
    quarantine = output / "quarantine" / ("r%d" % restart)
    for item in sorted(training.iterdir()):
        if keep is not None and item == keep[1]:
            continue
        quarantine.mkdir(parents=True, exist_ok=True)
        target = quarantine / item.name
        require(not target.exists(), "Quarantine target exists")
        os.rename(item, target)
        moved.append({"from": str(item), "to": str(target)})
    if keep is None:
        return None, moved
    return {"path": str(keep[1]), "global_step": keep[0], "accepted": keep[2]}, moved


def retire_completed(training, identity, tag):
    """Delete a completed trajectory's accepted checkpoints; ACCEPTED.json goes last, so a preempted
    retirement is resumable from the remaining receipt or from its sealed intent."""
    retired = []
    for item in sorted(training.glob("checkpoint-*")):
        receipt_path = item / "ACCEPTED.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_bytes())
        else:
            prior = sorted(CONTROL.glob("POST_SUCCESS_RETIRE_INTENT-*-%s.json" % item.name))
            require(prior, "Unreceipted checkpoint in a completed trajectory")
            receipt = json.loads(prior[-1].read_bytes())
        require(receipt["identity"] == identity and item.name == "checkpoint-%d" % receipt["global_step"],
                "Foreign checkpoint in completed trajectory")
        seal(CONTROL / ("POST_SUCCESS_RETIRE_INTENT-%s-%s.json" % (tag, item.name)), receipt)
        paths = sorted(item.rglob("*"), key=lambda p: (len(p.parts), str(p)), reverse=True)
        for path in [x for x in paths if x != receipt_path] + [x for x in paths if x == receipt_path]:
            path.unlink() if path.is_file() or path.is_symlink() else path.rmdir()
        item.rmdir()
        retired.append(seal(CONTROL / ("POST_SUCCESS_RETIRED-%s-%s.json" % (tag, item.name)), {
            "global_step": receipt["global_step"],
            "logical_bytes": sum(v["bytes"] for v in receipt["files"].values())}))
    return retired


def quarantine_account(output):
    """Storage accounting for quarantined attempt state; retained for a reviewed cleanup manifest."""
    root = output / "quarantine"
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()] if root.is_dir() else []
    return {"root": str(root), "attempt_directories": sorted(p.name for p in root.iterdir()) if root.is_dir() else [],
            "files": len(files), "logical_bytes": sum(p.stat().st_size for p in files),
            "deleted": False, "retained_for_reviewed_cleanup_manifest": True}


def inventory_inputs(paths):
    """Read-only DataTailor budget input inventory inside this owned GPU allocation."""
    report = []
    for root in paths:
        folder = Path(root)
        require(folder.is_dir() and not folder.is_symlink(), "Inventory folder must exist: " + root)
        for item in sorted(folder.iterdir()):
            if item.is_dir() and not item.is_symlink():
                entry = {"directory": str(item), "files": {}}
                for name in ("SELECTION_CANDIDATE.json", "records.json"):
                    path = item / name
                    if path.is_file() and not path.is_symlink():
                        entry["files"][name] = {"bytes": path.stat().st_size, "sha256": sha_file(path),
                                                "uid": path.stat().st_uid}
                candidate = item / "SELECTION_CANDIDATE.json"
                if candidate.is_file() and candidate.stat().st_size <= 64 << 20:
                    value = json.loads(candidate.read_bytes())
                    ids = value.get("record_ids") if isinstance(value, dict) else None
                    entry["candidate"] = {k: value.get(k) for k in (
                        "count", "method", "selection_seed", "fraction", "records_json", "pool", "status", "budget")
                        if isinstance(value, dict)}
                    entry["candidate"]["record_ids_count"] = len(ids) if isinstance(ids, list) else None
                    entry["candidate"]["record_ids_unique"] = len(set(ids)) if isinstance(ids, list) else None
                report.append(entry)
    return report


def work(roster):
    restart = restart_count()
    plan_ref, plan, review_ref, independent_ref = current_contract(independent=True)
    actor = actual_actor(plan_ref, plan, review_ref, independent_ref, restart)
    job = actor["job_id"]
    tag = "%s-r%d" % (job, restart)
    seal(CONTROL / ("ACTOR_BEFORE-" + tag + ".json"), actor)
    if SPEC["kind"] == "probe-reference" and roster.get("input_inventory"):
        seal(CONTROL / ("INPUT_INVENTORY-" + tag + ".json"), {
            "paths": roster["input_inventory"], "entries": inventory_inputs(roster["input_inventory"])})
    pool_raw, pool_fd = read_data(POOL, "pool")
    rows_raw, rows_fd = read_data(POOL_RECORDS, "pool")
    selected_raw, selected_fd = read_data(SPEC["records"], "group")
    if SPEC["selection_kind"] == "group":
        selection_raw, selection_fd = read_data(SPEC["selection"], "group")
    else:
        selection_raw, selection_fd = checked(SPEC["selection"]), {"reference": SPEC["selection"], "home": True}
    pool = json.loads(pool_raw)
    image_root = pool["image_root"]
    selected = validate_selection(pool, json.loads(rows_raw), json.loads(selection_raw), json.loads(selected_raw))
    del pool, pool_raw, rows_raw, selected_raw, selection_raw
    gc.collect()
    require(isinstance(image_root, str) and image_root.startswith("/") and Path(image_root).is_dir(),
            "Pool image root must be an existing absolute directory")
    input_ref = dict(SPEC["records"], canonical_sha256=selected["canonical_selected_rows_sha256"])
    seal(CONTROL / ("ACTUAL_INPUT-" + tag + ".json"), dict(
        selected, pool_fd=pool_fd, pool_records_fd=rows_fd, selected_fd=selected_fd, selection_fd=selection_fd,
        input=input_ref))
    require(Path(PYTHON).is_file() and os.access(PYTHON, os.X_OK), "Pinned native Python is unavailable")
    reference = json.loads(checked(REFERENCE_CONFIG))
    assets = json.loads(checked(reference["providers"]["assets"]))
    private_directory(OUTPUT_ROOT.parent, OUTPUT_ROOT.name)
    output = OUTPUT_ROOT / CELL
    owner_value = {"owner": OWNER, "cell": CELL, "job_id": job, "function_identity": plan["function_identity"],
                   "kind": SPEC["kind"], "author_checkpoint_root": str(OUTPUT_ROOT),
                   "discretionary_storage_threshold": None, "not_a_Root_grant_or_production_permit": True}
    if restart == 0:
        directory = open_literal(OUTPUT_ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.mkdir(output.name, 0o700, dir_fd=directory)
        finally:
            os.close(directory)
        seal(output / "OWNER.json", owner_value)
    else:
        require(json.loads((output / "OWNER.json").read_bytes()) == owner_value,
                "Restarted attempt must own the same output trajectory")
    result_path = output / "RESULT.json"
    if result_path.exists():
        finished = json.loads(result_path.read_bytes())
        require(finished["job_id"] == job and finished["cell"] == CELL and restart > 0,
                "Existing result must be this job's completed earlier attempt")
        resume, moved = None, []
    else:
        resume, moved = plan_resume(output, plan["function_identity"], restart)
        attempt_dir = CONTROL / ("attempt-" + tag)
        os.mkdir(attempt_dir, 0o700)
        seal(CONTROL / ("RESUME_DECISION-" + tag + ".json"), {"resume": resume, "quarantined": moved})
        require(restart == 0 or resume is not None or SPEC["kind"] == "production",
                "A restarted probe must resume from its accepted state")
        config = {
            "role": "TOOL1_RESTART_PARITY_PROBE" if SPEC["kind"] != "production"
                    else "TOOL1_BASELINE_FULL_EPOCH_NOT_EVALUATED",
            "owner": OWNER, "job_id": job, "cell": CELL, "method": SPEC["method"],
            "selection_seed": SPEC["selection_seed"], "training_seed": SPEC["training_seed"], "K": SPEC["K"],
            "horizon": horizon(), "checkpointing": {"save_steps": SPEC["save_steps"], "retain_accepted": SPEC["retain"]},
            "probe_stop_after_step": SPEC.get("stop_after") if restart == 0 else None,
            "function_identity": plan["function_identity"],
            "attempt": {"restart_count": restart, "directory": str(attempt_dir)},
            "resume": resume, "reference_assets": reference["providers"]["assets"], "recipe": reference["recipe"],
            "input": input_ref, "image_root": image_root, "assets": assets["assets"], "sources": reference["sources"],
            "output": str(output), "model_source": MODEL_SOURCE,
        }
        config_ref = seal(CONTROL / ("MODEL_CONFIG-" + tag + ".json"), config)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   HF_HOME="/data/hf_cache", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                   TORCH_EXTENSIONS_DIR="/data/hf_cache/project_envs/DS-MultiModal/torch_extensions",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        command = [PYTHON, "-B", "-X", "pycache_prefix=/dev/null/no-cache", "-m", "torch.distributed.run",
                   "--standalone", "--nnodes=1", "--nproc-per-node=1", "--max-restarts=0", "--no-python",
                   PYTHON, "-B", "-X", "pycache_prefix=/dev/null/no-cache", MODEL_SOURCE["path"],
                   "--config", config_ref["path"]]
        seal(CONTROL / ("MODEL_INTENT-" + tag + ".json"), {"config": config_ref, "argv": command, "actor": actor})
        started = time.monotonic()
        result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, timeout=SPEC["minutes"] * 60 - 900)
        seal(CONTROL / ("MODEL_EXIT-" + tag + ".json"), {
            "returncode": result.returncode, "elapsed_seconds": time.monotonic() - started, "config": config_ref})
        stop = attempt_dir / "PROBE_STOP.json"
        if result.returncode != 0 and SPEC["kind"] == "probe-requeue" and restart == 0 and stop.is_file():
            value = json.loads(stop.read_bytes())
            require(value == {"global_step": SPEC["stop_after"], "job_id": job}, "Unexpected probe stop marker")
            requeue_intent = seal(CONTROL / ("PROBE_REQUEUE_INTENT-" + tag + ".json"), {"stop": value})
            response = subprocess.run(["/usr/bin/scontrol", "requeue", job], capture_output=True, text=True,
                                      timeout=30, env=SLURM_ENV, stdin=subprocess.DEVNULL)
            seal(CONTROL / ("PROBE_REQUEUE_RESULT-" + tag + ".json"), {
                "intent": requeue_intent, "returncode": response.returncode,
                "stdout": response.stdout, "stderr": response.stderr})
            require(response.returncode == 0, "Self-requeue failed; preserve accepted state")
            time.sleep(600)
            raise RuntimeError("Self-requeue did not terminate this attempt")
        require(result.returncode == 0, "Actual native model failed; inspect preserved worker log")
        require(result_path.is_file(), "Actual model returned without a result")
        finished = json.loads(result_path.read_bytes())
    expected_status = ("ACTUAL_FULL_NATIVE_BASELINE_EPOCH_NOT_EVALUATED" if SPEC["kind"] == "production"
                       else "ACTUAL_RESTART_PARITY_PROBE_NOT_SCIENTIFIC")
    require(finished["status"] == expected_status and finished["job_id"] == job and finished["cell"] == CELL,
            "Actual model result differs")
    if SPEC["kind"] == "probe-requeue":
        reference_output = OUTPUT_ROOT / SPEC["compare_with"]
        command = [PYTHON, "-B", "-X", "pycache_prefix=/dev/null/no-cache", COMPARE_SOURCE["path"],
                   "--reference", str(reference_output), "--probe", str(output),
                   "--stop-step", str(SPEC["stop_after"]), "--steps", str(SPEC["probe_max_steps"]),
                   "--save-steps", str(SPEC["save_steps"]), "--out", str(output / ("PARITY-" + tag + ".json"))]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
        compared = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, timeout=3600)
        seal(CONTROL / ("COMPARE_EXIT-" + tag + ".json"), {"returncode": compared.returncode, "argv": command})
        require(compared.returncode == 0, "Parity comparison failed to run")
        seal(CONTROL / ("PARITY-" + tag + ".json"), json.loads((output / ("PARITY-" + tag + ".json")).read_bytes()))
    if SPEC["kind"] == "production":
        retire_completed(output / "training", plan["function_identity"], tag)
    actor_after = actual_actor(plan_ref, plan, review_ref, independent_ref, restart)
    require((actor_after["job_id"], actor_after["pid"], actor_after["node"]) == (job, actor["pid"], actor["node"]),
            "Baseline actor changed")
    seal(CONTROL / ("ACTOR_AFTER-" + tag + ".json"), actor_after)
    report = {"schema": "tool1-baseline-native-training-result-v4", "status": expected_status, "cell": CELL,
              "kind": SPEC["kind"], "function_identity": plan["function_identity"], "plan": plan_ref,
              "independent_review": independent_ref, "job_id": job, "restart_count": restart,
              "model_result": str(result_path), "input": input_ref, "selected": selected,
              "quarantine": quarantine_account(output)}
    seal(CONTROL / ("TRAINING_RESULT-" + tag + ".json"), report)
    print(json.dumps(report, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "self-review", "submit", "release", "worker"))
    parser.add_argument("--cell", required=True)
    parser.add_argument("--roster", required=True)
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    roster_ref = descriptor(Path(args.roster))
    require(roster_ref["sha256"] == args.roster_sha256, "Roster bytes differ from the requested pin")
    roster = configure(args.cell, roster_ref)
    if args.command == "worker":
        work(roster)
    elif args.command == "plan":
        print(json.dumps(make_plan_file(), sort_keys=True))
    elif args.command == "self-review":
        print(json.dumps(owner_review(args.note), sort_keys=True))
    else:
        print(json.dumps({"submit": submit, "release": release}[args.command](), sort_keys=True))


if __name__ == "__main__":
    main()
