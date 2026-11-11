"""Plan, hold-submit, release and run Tool1's CELLS_V7 custody and parameter-identity inventory (one GPU allocation).

CELLS_V7 custody-only fork of helper_tool1_cells_v4_custody_20261107/admit4.py (2026-11-10): identical inventory logic,
bound to CELLS_V7.json/custody7.py (four producer kinds: Tool1 trainer v6 original, v8 successor, v10 original and the
Tool3 chain-v3 RESULT under the t3coin v2 accept rule) with its own control/output roots, owner, statuses, job prefix
and comment. Plan-time custody goes through custody7.plan_custody (pinned records of the cell's kind, this
controller's exact readers). The native child is the byte-identical identity3.py (it requires the v3 ROLE string and
the "custody3.py" source key, which here names custody7.py). The CELLS_V4 docstring follows.

CELLS_V4 custody-only fork of helper_tool1_cells_v3_20261104/admit3.py (2026-11-07): identical inventory logic, bound
to CELLS_V4.json/custody4.py (General successor-trainer custody) with its own control/output roots, owner, statuses,
job prefix and comment. The original v3 docstring follows.

For the Tool1-trained baseline students registered in CELLS_V3.json whose trainer sealed a complete production
epoch, one run verifies on-node, read-only:
  * the producer's own HOME records (pinned PLAN/ACK/FUNCTION_CLAIM/HELD_READBACK, the single sealed
    TRAINING_RESULT-<job>-r<N>.json) through custody3.trainer_custody (bound into the plan);
  * the producer RESULT.json under /data/hf_cache/project_checkpoints (complete epoch, horizon (K + 3) // 4 // 32,
    finite losses) and the stream sha256 of every final training/ inference file it names;
  * the parameter digest of the model as the v3 evaluators load it: identity3.py (native Python child, same GPU)
    composes START + the cell's active unmerged LoRA + trained projector with the exact compose_student of the
    VizWiz v3 worker and seals MODEL_IDENTITY.json create-once under the inventory output root;
  * when Tool1's four-bank evaluation v2 of the same producer already COMPLETED, its parameter_sha256 must equal
    the inventory digest (FAIL otherwise; never relabelled).
The RESULT (COMPLETE_CELLS_V3_CUSTODY_INVENTORY_V1, per-cell PASS/FAIL) is sealed into the owned HOME control
run folder; v3 evaluation plans bind the run by its ACK and late-bind this RESULT at release and in the worker.

Babel rule: CPU checks run inside this CPU+GPU allocation. No downloads, no dataset/label reads, no training.
Routes: one reviewed route per run plan (--route general|preempt) gated by routes.py on the actual census under
ADMISSION.lock; --no-requeue. An interrupted run is followed by a new run index (fresh plan/review).

  admit7.py plan        --run N --route R --cell C [--cell C ...]
  admit7.py self-review --run N --route R --note "..."
  admit7.py submit|release --run N --route R
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
SOURCE = HOME / "experiments/helper_tool1_cells_v7_extsuite_20261110"
CELLS3 = HOME / "experiments/helper_tool1_cells_v3_20261104"
VZ3 = HOME / "experiments/helper_tool1_vizwiz_eval_v3_20261104"
V1 = HOME / "experiments/helper_tool1_vizwiz_eval_v1_20261102"
CONTROL = HOME / "collaboration/helper-tool1-baseline-independent-20261101/cells-v7-inventory-v1"
FOUR_BANK_CONTROL = HOME / "collaboration/helper-tool1-baseline-independent-20261101/baseline-eval-v2"
OUTPUT_ROOT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-cells-v7-inventory-v1-20261110")
PYTHON = "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python"
OWNER = "Tool1-owned-CELLS-V7-custody-and-parameter-identity-inventory-v1"
REVIEW_STATUS = "OWNER_REVIEWED_TOOL1_CELLS_V7_INVENTORY_NOT_INDEPENDENT_ACCEPTANCE"
INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_CELLS_V7_INVENTORY_V1"
# identity3.py (byte-identical child) requires this exact v3 role string.
ROLE = "TOOL1_CELLS_V3_CUSTODY_AND_PARAMETER_IDENTITY_INVENTORY"
FOUR_BANK_STATUS = "COMPLETE_TOOL1_BASELINE_FOUR_OFFICIAL_BANKS"
FOUR_BANK_OWNER = "Tool1-owned-published-baseline-full-official-evaluation"
JOB_PREFIX = "dsmm-t1v7-inv-r"
COMMENT_PREFIX = "dsmm-t1-cells-v7-inventory-v1:"
UID = 2708899
EXCLUDED = "babel-p9-20"
TIME = "02:00:00"
MAX_RUNS = 32
MAX_CELLS = 9
ROUTES = ("general", "preempt")
MAX_RESULT_BYTES = 2 << 20
MAX_PRODUCER_RESULT = 64 << 20
MAX_FOUR_BANK_RESULT = 64 << 20
TERMINAL = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
HELPER = {
    "path": str(HOME / "experiments/helper_method_eval_bank_qual_v1_20261102/prepare.py"),
    "bytes": 51200, "sha256": "f0dc4f6a548eedc45d5ba3d92e0251328d33bb82b274d815f14b0aca25fa7357",
}
SOURCES = {
    "identity3.py": {"path": str(CELLS3 / "identity3.py"), "bytes": 8449, "sha256": "32caf6198cc8377ab2bfcccc2ccfc557ad96a57dda346f606fa318c35376d500"},
    # Source key kept as "custody3.py" because identity3.py requires that exact key; it names custody7.py.
    "custody3.py": {"path": str(SOURCE / "custody7.py"), "bytes": 43616, "sha256": "e835ae87213f19df8f5f61e5bd36c9d47d7ea6e852c5f1f280f5868db52c41c8"},
    "CELLS_V7.json": {"path": str(SOURCE / "CELLS_V7.json"), "bytes": 7796, "sha256": "6276480cafcad44030510ede2de94ebf3420c83139e7054c3785bbb7c657bf8f"},
    "model.py": {"path": str(VZ3 / "model.py"), "bytes": 26547, "sha256": "b48ef7a4e54eab94f1bcacf05181f02d3f9b63e0f435328044628699ebfefe30"},
    "banks.py": {"path": str(V1 / "banks.py"), "bytes": 17979,
                 "sha256": "1121c76f1bd7e25c70c8fce41c0461037c12e243e87944a81e9c8878cb035816"},
    "routes.py": {"path": str(V1 / "routes.py"), "bytes": 4797,
                  "sha256": "5067bf831884699edcd0f5d86b5e9c1a3af65def6edf68aab4cb157c3d388709"},
}
CHILD_SOURCES = ("identity3.py", "custody3.py", "model.py", "banks.py")


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
    require(stat.S_ISREG(before.st_mode) and before.st_uid == UID
            and len(raw) == before.st_size == HELPER["bytes"]
            and hashlib.sha256(raw).hexdigest() == HELPER["sha256"]
            and all(getattr(before, field) == getattr(after, field)
                    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
            "Exact immutable executed metadata helper required")
    module = ModuleType("tool1_cells_v7_inventory_metadata_helpers")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    return module


def bound_module(p, name, module_name):
    module = ModuleType(module_name)
    module.__file__ = SOURCES[name]["path"]
    exec(compile(p.checked(SOURCES[name]), SOURCES[name]["path"], "exec", dont_inherit=True), vars(module))
    return module


def bound_custody(p):
    custody3 = bound_module(p, "custody3.py", "tool1_cells_v7_custody_inventory")
    require(custody3.INVENTORY_OWNER == OWNER and str(CONTROL) == custody3.INVENTORY_CONTROL
            and str(OUTPUT_ROOT) == custody3.INVENTORY_OUTPUT, "Inventory identity differs from custody7.py")
    return custody3


def bound_cells(p, custody3):
    return custody3.check_cells(json.loads(p.checked(SOURCES["CELLS_V7.json"])))


def check_run(run):
    require(type(run) is int and 0 <= run < MAX_RUNS, "Finite inventory run index required")
    return run


def check_route(route):
    require(isinstance(route, str) and route in ROUTES, "Reviewed route general or preempt required")
    return route


def run_folder(run):
    return CONTROL / ("run-%d" % check_run(run))


def job_name(run):
    return JOB_PREFIX + str(check_run(run))


def trainer_custody(p, custody3, cells, cell):
    """Same plan-time custody the CELLS_V7 evaluators bind (custody7.plan_custody over the pinned HOME records)."""
    return custody3.plan_custody(cells, cell, p.checked, p.descriptor, os.listdir)


def four_bank_binding(p, cell, producer):
    """Optional cross-check source: Tool1's four-bank evaluation v2 ACK of the same producer (result late-read)."""
    path = FOUR_BANK_CONTROL / cell / "ACK.json"
    if not path.exists():
        return None
    ack_ref = p.descriptor(path)
    ack = json.loads(p.checked(ack_ref))
    require(ack.get("cell") == cell and ack.get("owner") == FOUR_BANK_OWNER
            and re.fullmatch(r"[1-9][0-9]*", str(ack.get("job_id"))) is not None,
            "Four-bank evaluation ACK must be Tool1's own for this cell")
    return dict(ack=ack_ref, job_id=ack["job_id"], producer_job=producer["job_id"],
                result_path=str(FOUR_BANK_CONTROL / cell / ("FULL_EVALUATION_RESULT-%s.json" % ack["job_id"])))


def make_plan(p, run, route, cell_list):
    routes = bound_module(p, "routes.py", "tool1_cells_v7_inventory_routes")
    custody3 = bound_custody(p)
    cells = bound_cells(p, custody3)
    require(isinstance(cell_list, list) and 1 <= len(cell_list) <= MAX_CELLS and len(set(cell_list)) == len(cell_list)
            and cell_list == sorted(cell_list), "Sorted unique finite cell list required")
    targets = {}
    for cell in cell_list:
        custody = trainer_custody(p, custody3, cells, cell)
        targets[cell] = dict(custody, four_bank=four_bank_binding(p, cell, custody["producer"]),
                             model_identity=custody3.identity_path(run, cell))
    function = dict(
        owner=OWNER, purpose="on-node-CELLS-V7-four-kind-producer-custody-and-loaded-parameter-digest", run=check_run(run),
        cells=targets, bound_sources=SOURCES, metadata_helper=HELPER, evaluator_source_configuration=p.EVAL_CONFIG,
        native_python=PYTHON,
        output_root=str(OUTPUT_ROOT), inference_protocol_for_composition="vizwiz-test START configuration",
        downloads=False, dataset_or_label_reads=False, training_updates=0, generation_forwards=0,
        writes_under_data="create-once MODEL_IDENTITY.json and child records under the inventory output run folder",
        held_out_claim=False, result_location=str(run_folder(run)))
    require(TIME == routes.time_text(120) and EXCLUDED == routes.EXCLUDED, "Route table and inventory resources differ")
    return dict(
        schema="tool1-cells-v7-inventory-v1-plan-v1", owner=OWNER, run=run, route=check_route(route),
        cells=cell_list, function=function, function_identity=hashlib.sha256(p.canonical(function)).hexdigest(),
        sources=dict({"admit7.py": p.descriptor(SOURCE / "admit7.py"),
                      "inv_run7.sbatch": p.descriptor(SOURCE / "inv_run7.sbatch")}, **SOURCES),
        resource=dict(routes.ROUTES[routes.check_route(route)], gres=routes.GRES, GPUs=1, CPUs=8, memory="64G",
                      time=TIME, exclude=routes.EXCLUDED),
        expected_readback=routes.expected_route_fields(route), automatic_requeue=False,
        independently_reviewed=False)


def validate_independent(review, plan_ref, plan, run, route):
    require(review.get("status") == INDEPENDENT_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER) and review.get("blocking_findings") == []
            and review.get("sources") == plan["sources"] and review.get("plan") == plan_ref
            and review.get("route") == route == plan["route"] and review.get("run") == run
            and review.get("cells") == plan["cells"],
            "Independent exact source/config PASS bound to this inventory run plan is required")


def contract(p, run, route, independent=False):
    folder = run_folder(run)
    info = os.stat(folder, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Private owned inventory run directory required")
    plan_ref = p.descriptor(folder / ("PLAN-%s.json" % check_route(route)))
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, run, route, plan["cells"]), "Exact source/custody-bound inventory plan required")
    review_ref = p.descriptor(folder / ("OWNER_REVIEW-%s.json" % route))
    review = json.loads(p.checked(review_ref))
    require(review["status"] == REVIEW_STATUS and review["independent"] is False and review["plan"] == plan_ref
            and review["sources"] == plan["sources"] and review["blocking_findings"] == []
            and review["route"] == route and review["run"] == run,
            "Truthful exact private-owner review required; no independent PASS claimed")
    independent_ref = None
    if independent:
        independent_ref = p.descriptor(folder / ("INDEPENDENT_REVIEW-%s.json" % route))
        validate_independent(json.loads(p.checked(independent_ref)), plan_ref, plan, run, route)
    return plan_ref, plan, review_ref, independent_ref


def comment(plan, run):
    return COMMENT_PREFIX + plan["function_identity"] + ":r%d" % run


def argv(plan, run):
    resource = plan["resource"]
    return ["/usr/bin/sbatch", "--parsable", "--hold", "--no-requeue", "--export=NIL",
            "--account=" + resource["account"], "--partition=" + resource["partition"], "--qos=" + resource["qos"],
            "--gres=gpu:L40S:1", "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G", "--time=" + TIME,
            "--exclude=" + EXCLUDED, "--job-name=" + job_name(run), "--comment=" + comment(plan, run),
            "--chdir=" + str(HOME), "--input=/dev/null", "--output=/dev/null", "--error=/dev/null",
            str(SOURCE / "inv_run7.sbatch"), str(run), check_route(plan["route"])]


def scheduler(p, plan, run, job, held=False, running=False):
    raw = p.query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = p.fields(raw)
    expected = dict(
        plan["expected_readback"], JobId=job, UserId="luyangz(2708899)", JobName=job_name(run),
        Comment=comment(plan, run), NumCPUs="8", NumTasks="1", TimeLimit=TIME, Command=str(SOURCE / "inv_run7.sbatch"),
        WorkDir=str(HOME), Dependency="(null)", StdIn="/dev/null", StdOut="/dev/null", StdErr="/dev/null")
    # Slurm shows a pending single-node request as NumNodes=1-1 and a running one as 1.
    nodes = row.get("NumNodes") == "1" if running else row.get("NumNodes") in ("1", "1-1")
    require(plan["expected_readback"]["Requeue"] == "0"
            and all(row.get(key) == value for key, value in expected.items())
            and nodes and shlex.split(row.get("SubmitLine", "")) == argv(plan, run),
            "Actual physical inventory request/source/resources differ")
    if held:
        require(row.get("JobState") == "PENDING" and row.get("Reason") == "JobHeldUser"
                and row.get("RunTime") == "00:00:00", "Verified never-started held request required")
    if running:
        require(row.get("JobState") == "RUNNING" and row.get("Restarts") == "0"
                and row.get("NodeList") == socket.gethostname().split(".")[0]
                and p.valid_singleton_node(row["NodeList"]), "Actual owned running allocation required")
    tres = dict(value.split("=", 1) for value in row.get("AllocTRES" if running else "ReqTRES", "").split(",")
                if "=" in value)
    require(tres.get("cpu") == "8" and tres.get("mem") in ("64G", "65536M") and tres.get("gres/gpu") == "1",
            "Actual total allocation/request resources differ")
    spool = subprocess.check_output(["/usr/bin/scontrol", "write", "batch_script", job, "-"],
                                    timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
    require(spool == p.checked(plan["sources"]["inv_run7.sbatch"]), "Exact immutable bound spool required")
    return raw, spool


def private_directory(path):
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.stat(path, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Owned private control directory required: " + str(path))


def make_plan_file(p, run, route, cell_list):
    private_directory(CONTROL)
    private_directory(run_folder(run))
    path = run_folder(run) / ("PLAN-%s.json" % check_route(route))
    require(not path.exists(), "Existing inventory plan is immutable; use a new run")
    for ref in SOURCES.values():
        p.checked(ref)
    return p.seal(path, make_plan(p, run, route, sorted(cell_list)))


def owner_review(p, run, route, notes):
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete self-review notes required")
    folder = run_folder(run)
    plan_ref = p.descriptor(folder / ("PLAN-%s.json" % check_route(route)))
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, run, route, plan["cells"]), "Self-review must bind the exact plan")
    return p.seal(folder / ("OWNER_REVIEW-%s.json" % route), dict(
        status=REVIEW_STATUS, independent=False, ai_review_disclosed=True, reviewer=OWNER, run=run, route=route,
        plan=plan_ref, cells=plan["cells"], sources=plan["sources"], function_identity=plan["function_identity"],
        blocking_findings=[], notes=notes))


def previous_terminal(p, run):
    """Runs are strictly sequential: every earlier run that was submitted must be terminal."""
    previous = []
    for index in range(run):
        path = run_folder(index) / "ACK.json"
        if not path.exists():
            continue
        ack = json.loads(p.checked(p.descriptor(path)))
        value = p.query(["/usr/bin/sacct", "-X", "-nP", "-j", ack["job_id"],
                         "--format=JobIDRaw,State,ExitCode"]).strip().split("|")
        require(len(value) == 3 and value[0] == ack["job_id"] and value[1].split(" ")[0] in TERMINAL,
                "An earlier inventory run must be terminal before another observation")
        previous.append(dict(run=index, job_id=ack["job_id"], state=value[1], exit_code=value[2]))
    return previous


def admit(p, run, route, release=False):
    check_run(run)
    routes = bound_module(p, "routes.py", "tool1_cells_v7_inventory_routes")
    plan_ref, plan, review_ref, independent_ref = contract(p, run, route, independent=release)
    folder = run_folder(run)
    with p.admission_lock():
        require(contract(p, run, route, independent=release) == (plan_ref, plan, review_ref, independent_ref),
                "Contract changed while acquiring shared admission lock")
        if not release:
            predecessors = previous_terminal(p, run)
            rows, gate = routes.actual_gate(p.query, route)
            require(not any(plan["function_identity"] in row[2] or row[3].startswith(JOB_PREFIX) for row in rows),
                    "A live cells-v7 inventory already exists on some route")
            require(not any((folder / x).exists() for x in
                            ("INTENT.json", "ACK.json", "AMBIGUOUS.json", "FUNCTION_CLAIM.json")),
                    "Existing submission or ambiguous ACK: reconcile, never duplicate")
            claim = p.seal(folder / "FUNCTION_CLAIM.json", dict(
                owner=OWNER, run=run, route=route, cells=plan["cells"], function=plan["function"],
                function_identity=plan["function_identity"], plan=plan_ref, predecessors=predecessors))
            intent = p.seal(folder / "INTENT.json", dict(
                owner=OWNER, run=run, route=route, cells=plan["cells"], plan=plan_ref, review=review_ref, claim=claim,
                argv=argv(plan, run), route_gate=gate, expanded_requests=len(rows)))
            try:
                response = subprocess.run(argv(plan, run), text=True, capture_output=True, timeout=45,
                                          env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as error:
                p.seal(folder / "AMBIGUOUS.json", dict(intent=intent, error=repr(error)))
                raise
            good = response.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;[\w.-]+)?\n?", response.stdout)
            if good:
                job = response.stdout.strip().split(";")[0]
                ack_ref = p.seal(folder / "ACK.json", dict(owner=OWNER, run=run, route=route, job_id=job,
                                                          cells=plan["cells"], intent=intent, plan=plan_ref))
            p.seal(folder / "SUBMISSION_RESULT.json", dict(
                intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
            require(good, "No unambiguous actual ACK; reconcile exact intent before another submission")
            raw, spool = scheduler(p, plan, run, job, held=True)
            p.seal(folder / "HELD_READBACK.json", dict(
                ACK=ack_ref, job_id=job, scheduler=raw, spool_base64=base64.b64encode(spool).decode()))
            return dict(run=run, route=route, job_id=job, cells=plan["cells"], state="HELD_VERIFIED")
        ack_ref = p.descriptor(folder / "ACK.json")
        ack = json.loads(p.checked(ack_ref))
        require(ack["owner"] == OWNER and ack["plan"] == plan_ref and ack["run"] == run and ack["route"] == route
                and ack["cells"] == plan["cells"], "Exact own ACK required")
        require(not (folder / "RELEASE_INTENT.json").exists(), "Existing release requires reconciliation")
        job = ack["job_id"]
        scheduler(p, plan, run, job, held=True)
        intent = p.seal(folder / "RELEASE_INTENT.json", dict(ACK=ack_ref, job_id=job,
                                                            independent_review=independent_ref))
        response = subprocess.run(["/usr/bin/scontrol", "release", job], text=True, capture_output=True,
                                  timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
        p.seal(folder / "RELEASE_RESULT.json", dict(
            intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        require(response.returncode == 0, "Release failed or ambiguous; preserve exact identity")
        raw, _ = scheduler(p, plan, run, job)
        row = p.fields(raw)
        require(row.get("JobState") in ("PENDING", "RUNNING", "CONFIGURING")
                and row.get("Reason") not in ("JobHeldUser", "JobHeldAdmin", "DependencyNeverSatisfied"),
                "Released inventory must be eligible or running")
        p.seal(folder / "RELEASE_READBACK.json", dict(job_id=job, route=route, scheduler=raw))
        return dict(run=run, route=route, job_id=job, state=row.get("JobState") + ":" + row.get("Reason", ""))


def read_owned_data(p, path, limit, nas=True):
    """Bounded, no-follow, FD-stable read of one owned regular file (on the shared HF NAS unless nas=False)."""
    fd = p.open_literal(path, os.O_RDONLY)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_uid == UID and 0 < before.st_size <= limit,
                "Bounded regular owned file required: " + str(path))
        mount = p.mount_for_fd(stream.fileno(), before.st_dev, "group") if nas else None
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    require(len(raw) == before.st_size and all(getattr(before, key) == getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "Opened file changed")
    return raw, dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()), mount


def stream_owned(p, ref):
    """Stream sha256 of one owned regular inference file; size and digest must equal the producer's record."""
    fd = p.open_literal(ref["path"], os.O_RDONLY)
    with os.fdopen(fd, "rb", buffering=0) as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_uid == UID and before.st_size == ref["bytes"],
                "Owned regular inference file of the recorded size required: " + ref["path"])
        mount = p.mount_for_fd(stream.fileno(), before.st_dev, "group")
        result, seen = hashlib.sha256(), 0
        while True:
            block = stream.read(8 << 20)
            if not block:
                break
            result.update(block)
            seen += len(block)
        after = os.fstat(stream.fileno())
    require(seen == ref["bytes"] and result.hexdigest() == ref["sha256"]
            and all(getattr(before, key) == getattr(after, key)
                    for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
            "Inference file differs from the producer RESULT: " + ref["path"])
    return dict(ref, inode=after.st_ino, mount=mount)


def four_bank_digest(p, binding):
    """Parameter digest of Tool1's COMPLETE four-bank evaluation of the same producer, or None if not complete."""
    if binding is None or not Path(binding["result_path"]).exists():
        return None, None
    raw, ref, _ = read_owned_data(p, Path(binding["result_path"]), MAX_FOUR_BANK_RESULT, nas=False)
    report = json.loads(raw)
    producer = report.get("config", {}).get("producer", {})
    require(report.get("status") == FOUR_BANK_STATUS and report.get("job_id") == binding["job_id"]
            and producer.get("job_id") == binding["producer_job"]
            and re.fullmatch(r"[0-9a-f]{64}", str(report.get("parameter_sha256"))) is not None,
            "Four-bank evaluation result is not this producer's complete evaluation")
    return report["parameter_sha256"], ref


def mkdir_private(path):
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.stat(path, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Private own output directory required: " + str(path))


def work(p, run, route):
    check_run(run)
    plan_ref, plan, review_ref, independent_ref = contract(p, run, route, independent=True)
    folder = run_folder(run)
    job = os.environ.get("SLURM_JOB_ID", "")
    ack_ref = p.descriptor(folder / "ACK.json")
    ack = json.loads(p.checked(ack_ref))
    require(os.getuid() == UID and job.isdecimal() and ack["job_id"] == job and ack["run"] == run
            and ack["route"] == route
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0" and not os.environ.get("SLURM_ARRAY_JOB_ID"),
            "Actual unrestarted owned inventory allocation required")
    release = json.loads(p.checked(p.descriptor(folder / "RELEASE_INTENT.json")))
    require(release == dict(ACK=ack_ref, job_id=job, independent_review=independent_ref),
            "Actual independently reviewed release intent required before DATA")
    raw, _ = scheduler(p, plan, run, job, running=True)
    node = socket.gethostname().split(".")[0]
    p.validate_process(job, os.getpid(), node, os.environ.get("SLURM_STEP_ID"),
                       p.query(["/usr/bin/scontrol", "listpids", job]),
                       p.fields(p.query(["/usr/bin/scontrol", "show", "step", job + ".batch", "-o"])),
                       (Path("/proc") / str(os.getpid()) / "status").read_text())
    actor = dict(job_id=job, pid=os.getpid(), node=node, plan=plan_ref, run=run, route=route)
    p.seal(folder / ("ACTOR_BEFORE-" + job + ".json"), actor)
    custody3 = bound_custody(p)
    cells = {}
    child_cells = {}
    for cell, target in plan["function"]["cells"].items():
        try:
            producer = target["producer"]
            raw_result, result_ref, mount = read_owned_data(p, Path(producer["model_result"]), MAX_PRODUCER_RESULT)
            files = custody3.producer_files(json.loads(raw_result), producer)
            observed = {name: stream_owned(p, ref) for name, ref in files.items()}
            four_bank_sha, four_bank_ref = four_bank_digest(p, target["four_bank"])
            cells[cell] = dict(status="PENDING_PARAMETER_DIGEST", producer=producer, trainer_records=target[
                "trainer_records"], training_result=target["training_result"], producer_result=result_ref,
                producer_result_mount=mount, inference_files=files, observed_files=observed,
                four_bank=dict(binding=target["four_bank"], result=four_bank_ref, parameter_sha256=four_bank_sha))
            child_cells[cell] = dict(producer=producer, inference_files=files, model_identity=target["model_identity"])
        except (ValueError, OSError, KeyError, TypeError) as error:
            cells[cell] = dict(status="FAIL", reason=repr(error)[:2000], producer=target.get("producer"))
    output = OUTPUT_ROOT / ("run-%d" % run)
    mkdir_private(OUTPUT_ROOT)
    mkdir_private(output)
    attempt = output / ("job-" + job)
    os.mkdir(attempt, 0o700)
    child_result = None
    if child_cells:
        evaluator = json.loads(p.checked(p.EVAL_CONFIG))
        config = dict(role=ROLE, owner=OWNER, job_id=job, run=run, cells=child_cells,
                      sources={name: SOURCES[name] for name in CHILD_SOURCES}, eval_sources=evaluator["sources"],
                      output=str(output),
                      attempt=str(attempt), actual_actor=actor)
        config_ref = p.seal(folder / ("CHILD_CONFIG-" + job + ".json"), config)
        command = [PYTHON, "-u", "-B", "-X", "pycache_prefix=/dev/null/no-cache",
                   SOURCES["identity3.py"]["path"], "--config", config_ref["path"]]
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   HF_HOME="/data/hf_cache", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                   TORCH_EXTENSIONS_DIR="/data/hf_cache/project_envs/DS-MultiModal/torch_extensions",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                   OPENBLAS_NUM_THREADS="8")
        p.seal(folder / ("CHILD_INTENT-" + job + ".json"), dict(config=config_ref, argv=command, actor=actor))
        started = time.monotonic()
        result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, timeout=6000)
        p.seal(folder / ("CHILD_EXIT-" + job + ".json"), dict(
            returncode=result.returncode, elapsed_seconds=time.monotonic() - started, attempt=str(attempt)))
        require(result.returncode == 0, "Identity child failed; preserve its log and every receipt")
        raw_child, child_ref, _ = read_owned_data(p, attempt / "CHILD_RESULT.json", MAX_RESULT_BYTES)
        child_result = json.loads(raw_child)
        require(child_result.get("job_id") == job and sorted(child_result.get("cells", {})) == sorted(child_cells),
                "Exact own identity child result required")
        for cell in child_cells:
            found = child_result["cells"][cell]
            raw_identity, identity_ref, _ = read_owned_data(p, Path(child_cells[cell]["model_identity"]),
                                                            MAX_RESULT_BYTES)
            identity = json.loads(raw_identity)
            digest = identity.get("parameter_sha256")
            expected = cells[cell]["four_bank"]["parameter_sha256"]
            consistent = (identity_ref == found["model_identity"] and digest == found["parameter_sha256"]
                          and identity.get("cell") == cell and identity.get("job_id") == job
                          and identity.get("producer") == child_cells[cell]["producer"]
                          and identity.get("inference_files") == child_cells[cell]["inference_files"]
                          and re.fullmatch(r"[0-9a-f]{64}", str(digest)) is not None)
            parity = None if expected is None else expected == digest
            cells[cell].update(model_identity=identity_ref, parameter_sha256=digest, four_bank_parity=parity,
                               status="PASS" if consistent and parity is not False else "FAIL",
                               reason=None if consistent and parity is not False else (
                                   "four-bank parameter digest differs" if consistent else "identity record differs"))
        child_result = child_ref
    report = dict(
        schema=custody3.INVENTORY_SCHEMA, status=custody3.INVENTORY_STATUS, owner=OWNER, run=run, plan=plan_ref,
        cells=cells, child_result=child_result, worker_returncode=0, actual=dict(actor, scheduler=raw),
        summary={cell: value["status"] for cell, value in cells.items()}, downloads=False,
        dataset_or_label_reads=False, training_updates=0, scientific_performance_measured=False)
    require(len(p.canonical(report)) < MAX_RESULT_BYTES, "Inventory result must stay a small HOME control record")
    reference = p.seal(folder / ("RESULT-" + job + ".json"), report)
    scheduler(p, plan, run, job, running=True)
    p.seal(folder / ("ACTOR_AFTER-" + job + ".json"), actor)
    print(json.dumps(dict(job_id=job, run=run, result=reference, summary=report["summary"]), sort_keys=True),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "self-review", "submit", "release", "worker"))
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    p = load_helpers()
    if args.command == "worker":
        work(p, args.run, args.route)
    elif args.command == "plan":
        print(json.dumps(make_plan_file(p, args.run, args.route, args.cell), sort_keys=True))
    elif args.command == "self-review":
        print(json.dumps(owner_review(p, args.run, args.route, args.note), sort_keys=True))
    else:
        print(json.dumps(admit(p, args.run, args.route, release=args.command == "release"), sort_keys=True))


if __name__ == "__main__":
    main()
