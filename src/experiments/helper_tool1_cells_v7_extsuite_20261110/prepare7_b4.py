"""Plan, hold-submit, release and run Tool1 BENCH4 evaluations of CELLS_V7 (four-kind custody) students.

CELLS_V7 custody-only fork of helper_tool1_bench4_eval_v3_20261104/prepare4.py (2026-11-10), in the reviewed CELLS_V4 VizWiz pattern
(helper_tool1_cells_v4_custody_20261107/prepare4.py). Generation, journals, export and scoring code are the
byte-identical v3 files (model4.py, banks, routes; the v3 OWNER, ROLE, CONTROL_ROOT and OUTPUT_ROOT are unchanged, so the
v3 registered scoring steps and the BENCH4 judge see the result exactly like the other v3 cells). Differences, all
custody/admission:
  * custody: the "CELLS_V3.json"/"custody3.py" source keys (read by the byte-identical model4.py) name CELLS_V7.json and
    custody7.py; plan-time custody goes through custody7.plan_custody (pinned records of the cell's producer kind:
    Tool1 trainer v6 original, v8 successor, v10 original, or the Tool3 chain-v3 RESULT under the t3coin v2 accept
    rule) and binds the cells-v7 inventory run by its ACK;
  * admission: own job names (dsmm-t1v7-b4-*), comment prefix, plan schema and review statuses; every request
    carries --dependency=afterok:<cells-v7 inventory job> (the job the plan's custody binds). Release requires that
    inventory's sealed PASS RESULT or a still pending/running inventory job (RELEASE_INVENTORY_GATE.json); the worker
    late-binds the PASS RESULT before any data exactly as in v3 (custody_ready in work()).
The v3 docstring follows.

Plan, hold-submit, release and run Tool1 BENCH4 evaluations (one job per model, four banks in order).

Mirrors the reviewed Tool1 BENCH6 controller (helper_tool1_bench6_eval_v1_20261103/prepare6.py): exact pinned
sources, PLAN and truthful owner review, an independent exact source/config review per plan before release, held
submission under the shared ADMISSION.lock with durable FUNCTION_CLAIM/INTENT/ACK/SUBMISSION_RESULT/HELD_READBACK,
and a worker that re-verifies its own running allocation before any data. One job per cell loads the model once and
runs mmmu-val, llava-wild, mm-vet and mmbench-en-dev in that order; each (bank, cell) has its own resume-stable
SQLite journal, so attempt N>0 is an explicit, result-dependent successor that generates only missing QAs.

Inputs: every plan binds the BENCH4 input inventory by its exact run ACK (and through it its PLAN); the worker and
the release gate require that run's sealed RESULT (COMPLETE_BENCH4_INPUT_INVENTORY_V1, all four banks PASS) and
pass its label-free QUESTIONS descriptors to model4.py. Custody is late-bound from the completed on-node Tool1
custody inventory that the BENCH6 CELLS roster (bound by hash) names for the cell.

Routes: every plan is made for one reviewed route (--route general|preempt); routes.py (reviewed v1 file) gates
submission on the actual census under ADMISSION.lock. Both routes are --no-requeue. Continue a preempted,
timed-out or node-failed attempt with
  prepare4.py submit  --cell C --attempt N+1 --route R
  prepare4.py release --cell C --attempt N+1 --route R
A FAILED predecessor without a result needs attempt-N/DIAGNOSIS-<job>.json and an independent
DIAGNOSIS_REVIEW-<job>.json bound to that job and this controller.

v3 (new SOURCE/CONTROL/OUTPUT roots, owner, role, comment, job names and independent statuses; the bound v1
controller is unchanged): CELLS_V3.json (helper_tool1_cells_v3_20261104) Tool1-trained baseline students.
Custody comes from custody3.py: a plan binds the producer's own PLAN/ACK/TRAINING_RESULT (complete epoch
only) and the cells-v3 inventory run by its exact ACK; the release gate and the worker late-bind that run's
sealed RESULT (PASS, producer RESULT/file descriptors, loaded parameter digest). The BENCH4 input inventory
binding, banks4.py (unchanged v1 file), generation, journals and export are unchanged; the horizon follows K.

  prepare7_b4.py plan        --cell C --route R --inventory <cells-v7-inventory run-N/ACK.json> --inputs-ack PATH
  prepare7_b4.py self-review --cell C --route R --note "..."
  prepare7_b4.py submit|release --cell C --attempt N --route R
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
V3SRC = HOME / "experiments/helper_tool1_bench4_eval_v3_20261104"
BENCH4V1 = HOME / "experiments/helper_tool1_bench4_eval_v1_20261104"
CELLS3 = HOME / "experiments/helper_tool1_cells_v3_20261104"
V1 = HOME / "experiments/helper_tool1_vizwiz_eval_v1_20261102"
CONTROL_ROOT = HOME / "collaboration/helper-tool1-baseline-independent-20261101/bench4-eval-v3"
INPUT_CONTROL = HOME / "collaboration/helper-tool1-baseline-independent-20261101/bench4-inventory-v1"
OUTPUT_ROOT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-bench4-eval-v3-20261104")
PYTHON = "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python"
OWNER = "Tool1-owned-BENCH4-CELLS-V3-trained-baseline-evaluation-v3"
INPUT_OWNER = "Tool1-owned-BENCH4-input-inventory-acquisition-and-exposure-audit"
INPUT_STATUS = "COMPLETE_BENCH4_INPUT_INVENTORY_V1"
REVIEW_STATUS = "OWNER_REVIEWED_TOOL1_BENCH4_EVAL_V3_CUSTODY_V7_NOT_INDEPENDENT_ACCEPTANCE"
INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_BENCH4_EVAL_V3_CUSTODY_V7"
DIAGNOSIS_SCHEMA = "tool1-bench4-v3-failed-attempt-diagnosis-v1"
DIAGNOSIS_STATUS = "OWNER_DIAGNOSED_FAILED_ATTEMPT"
DIAGNOSIS_REVIEW_STATUS = "PASS_INDEPENDENT_FAILED_ATTEMPT_DIAGNOSIS_TOOL1_BENCH4_EVAL_V3"
DIAGNOSIS_BASES = ("NO_QA_COMMITTED_BY_FAILED_ATTEMPT", "JOURNAL_INTEGRITY_VERIFIED")
ROLE = "TOOL1_BENCH4_CELLS_V3_TRAINED_BASELINE_EVALUATION"
COMPLETE_STATUS = "COMPLETE_ALL_PLANNED_BANKS"
PROGRESS_STATUS = "ACCEPTED_QA_PROGRESS_REQUIRES_REVIEWED_MISSING_ONLY_SUCCESSOR"
COMMENT_PREFIX = "dsmm-t1-bench4-v3c7:"
UID = 2708899
EXCLUDED = "babel-p9-20"
RESERVE_SECONDS = 900
MAX_ATTEMPTS = 8
ROUTES = ("general", "preempt")
MINUTES = 240
BANKS = ("mmmu-val", "llava-wild", "mm-vet", "mmbench-en-dev")
TERMINAL = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
INTERRUPTED = ("TIMEOUT", "PREEMPTED", "NODE_FAIL")
HELPER = {
    "path": str(HOME / "experiments/helper_method_eval_bank_qual_v1_20261102/prepare.py"),
    "bytes": 51200, "sha256": "f0dc4f6a548eedc45d5ba3d92e0251328d33bb82b274d815f14b0aca25fa7357",
}
SOURCES = {
    "model4.py": {"path": str(V3SRC / "model4.py"), "bytes": 33729, "sha256": "16f25d036cb798d9447819717b2848aa0f12947a8e93d5d1024e31137fa3f44d"},
    "banks4.py": {"path": str(BENCH4V1 / "banks4.py"), "bytes": 30685, "sha256": "8a3b14c9ce0676db1d7bed24f3d56069222d4816933546eb3e9487ae9e4aadbe"},
    # Keys kept because the byte-identical model4.py reads sources["CELLS_V3.json"] and sources["custody3.py"].
    "CELLS_V3.json": {"path": str(SOURCE / "CELLS_V7.json"), "bytes": 7796, "sha256": "6276480cafcad44030510ede2de94ebf3420c83139e7054c3785bbb7c657bf8f"},
    "custody3.py": {"path": str(SOURCE / "custody7.py"), "bytes": 43616, "sha256": "e835ae87213f19df8f5f61e5bd36c9d47d7ea6e852c5f1f280f5868db52c41c8"},
    "routes.py": {"path": str(V1 / "routes.py"), "bytes": 4797,
                  "sha256": "5067bf831884699edcd0f5d86b5e9c1a3af65def6edf68aab4cb157c3d388709"},
}
GENERATION_SOURCES = ("model4.py", "banks4.py", "CELLS_V3.json", "custody3.py")


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
    module = ModuleType("tool1_frozen_evaluation_metadata_helpers")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    return module


def bound_module(p, name, module_name):
    raw = p.checked(SOURCES[name])
    module = ModuleType(module_name)
    module.__file__ = SOURCES[name]["path"]
    exec(compile(raw, SOURCES[name]["path"], "exec", dont_inherit=True), vars(module))
    return module


def bound_cells(p):
    return bound_custody(p).check_cells(json.loads(p.checked(SOURCES["CELLS_V3.json"])))


def bound_routes(p):
    return bound_module(p, "routes.py", "tool1_bench4_routes")


def bound_banks(p):
    return bound_module(p, "banks4.py", "tool1_bench4_banks_plan")


def check_route(route):
    require(isinstance(route, str) and route in ROUTES, "Reviewed route general or preempt required")
    return route


def plan_name(route):
    return "PLAN-%s.json" % check_route(route)


def review_name(route):
    return "OWNER_REVIEW-%s.json" % check_route(route)


def independent_name(route):
    return "INDEPENDENT_REVIEW-%s.json" % check_route(route)


def check_cell(cells, cell):
    require(isinstance(cell, str) and cell in cells["cells"],
            "Registered CELLS_V7 model required; planning further needs its complete TRAINING_RESULT")
    return cell


def check_attempt(attempt):
    require(type(attempt) is int and 0 <= attempt < MAX_ATTEMPTS, "Finite attempt index required")
    return attempt


def bound_custody(p):
    return bound_module(p, "custody3.py", "tool1_cells_v7_custody")


def plan_custody(p, cells, cell, inventory_ref):
    """Plan-time custody through custody7.plan_custody (the pinned records of the cell's producer kind, read with
    this controller's exact readers) plus the cells-v7 inventory ACK."""
    custody3 = bound_custody(p)
    custody = custody3.plan_custody(cells, cell, p.checked, p.descriptor, os.listdir)
    binding = custody3.inventory_binding(inventory_ref, json.loads(p.checked(inventory_ref)))
    require(cell in binding["cells"], "The bound cells-v7 inventory run does not cover this cell")
    return dict(custody, inventory=binding)


def custody_ready(p, plan):
    """Release/worker gate: the bound inventory run sealed a COMPLETE RESULT with PASS for this exact custody."""
    custody = plan["function"]["custody"]
    result_ref = p.descriptor(Path(custody["inventory"]["result_path"]))
    return bound_custody(p).inventory_custody(result_ref, json.loads(p.checked(result_ref)), custody["inventory"],
                                              custody)


def inventory_gate(p, plan):
    """Release gate: the bound inventory's PASS RESULT (v3 custody_ready), or its job still pending/running so that the
    afterok request can only start after it COMPLETED; the worker re-checks custody_ready before any data."""
    custody = plan["function"]["custody"]
    job = custody["inventory"]["job_id"]
    require(plan["dependency"] == "afterok:" + job, "Dependency must name exactly the bound inventory job")
    if Path(custody["inventory"]["result_path"]).exists():
        return dict(mode="inventory_result_pass", custody=custody_ready(p, plan))
    state, code = sacct_state(p, job)
    require(state in ("PENDING", "RUNNING"), "Bound inventory job is neither live nor sealed a RESULT: " + state)
    return dict(mode="inventory_dependency_live", inventory_job=job, state=state, exit_code=code)


def job_name(cells, cell, attempt):
    return "dsmm-t1v7-b4-%s-a%d" % (cells["cells"][cell]["short"], attempt)


def comment(plan, attempt):
    return COMMENT_PREFIX + plan["function_identity"] + ":a%d" % attempt


def time_text(minutes):
    require(type(minutes) is int and 0 < minutes < 24 * 60, "Sub-day walltime required")
    return "%02d:%02d:00" % divmod(minutes, 60)


def inputs_binding(p, ack_ref):
    """The BENCH4 input inventory run named by its exact ACK; its RESULT is late-bound by job identity."""
    ack = json.loads(p.checked(ack_ref))
    require(Path(ack_ref["path"]).parent.parent == INPUT_CONTROL and ack.get("owner") == INPUT_OWNER
            and re.fullmatch(r"[1-9]\d*", str(ack.get("job_id"))) is not None and type(ack.get("run")) is int
            and Path(ack_ref["path"]).parent.name == "run-%d" % ack["run"], "Exact BENCH4 input inventory ACK required")
    return dict(ack=ack_ref, plan=ack["plan"], run=ack["run"], job_id=ack["job_id"],
                result_path=str(INPUT_CONTROL / ("run-%d" % ack["run"]) / ("RESULT-%s.json" % ack["job_id"])))


def inputs_ready(p, plan):
    """Release/worker gate: the bound inventory run sealed a complete RESULT with all four banks PASS."""
    binding = plan["function"]["inputs_inventory"]
    result_ref = p.descriptor(Path(binding["result_path"]))
    result = json.loads(p.checked(result_ref))
    require(result.get("status") == INPUT_STATUS and result["actual"]["job_id"] == binding["job_id"]
            and result.get("worker_returncode") == 0 and sorted(result["banks"]) == sorted(BANKS)
            and all(result["banks"][name]["status"] == "PASS" for name in BANKS),
            "Completed BENCH4 input inventory with every bank PASS required before release")
    banks = {name: dict(questions=result["banks"][name]["questions"], items=result["banks"][name]["items"])
             for name in BANKS}
    return dict(inventory_result=result_ref, banks=banks)


def make_plan(p, cell, inventory_ref, route, inputs_ack_ref):
    routes = bound_routes(p)
    banks = bound_banks(p)
    cells = bound_cells(p)
    spec = cells["cells"][check_cell(cells, cell)]
    names = banks.check_bank_list(list(BANKS))
    require(tuple(names) == tuple(banks.ORDER) == BANKS, "Exact reviewed BENCH4 bank order required")
    custody = plan_custody(p, cells, cell, inventory_ref)
    protocols = {name: banks.protocol(name) for name in names}
    function = dict(
        owner=OWNER, cell=cell, banks=names, method=spec["method"], training_seed=spec["training_seed"],
        K=spec["K"], custody=custody, inventory=inventory_ref, inputs_inventory=inputs_binding(p, inputs_ack_ref),
        bank_inputs={name: {k: protocols[name].get(k) for k in ("bank", "kind", "split", "edition", "items",
                                                                 "sources", "root_support", "decoding", "suffix",
                                                                 "image_mode", "prompt_policy", "scoring",
                                                                 "judge_required")}
                     for name in names},
        start_template=banks.START_TEMPLATE, frozen_scope=banks.FROZEN_SCOPE,
        bound_sources=SOURCES, metadata_helper=HELPER, evaluator_source_configuration=p.EVAL_CONFIG,
        native_python=PYTHON, output_root=str(OUTPUT_ROOT), labels_bound_or_opened_by_generation=False,
        scoring="REGISTERED_BENCH4_SCORING_STEP_AFTER_COMPLETE_EXPORTS_JUDGE_BANKS_DEFERRED", no_model_downloads=True,
        no_dataset_downloads=True, no_training_updates=True, read_only_foreign_checkpoints=True,
        labels_never_used_for_selection_or_tuning=True, not_held_out_until_exposure_audit=True)
    # The route is a plan/resource choice, never part of the journal function: duplicate and live-attempt
    # checks span both routes, and either reviewed route may continue the same journals.
    return dict(
        schema="tool1-bench4-eval-v3-custody-v7-plan-v1", owner=OWNER, cell=cell, route=check_route(route), function=function,
        function_identity=hashlib.sha256(p.canonical(function)).hexdigest(),
        sources=dict({"prepare7_b4.py": p.descriptor(SOURCE / "prepare7_b4.py"),
                      "b4_run7.sbatch": p.descriptor(SOURCE / "b4_run7.sbatch")}, **SOURCES),
        dependency="afterok:" + custody["inventory"]["job_id"],
        resource=dict(routes.ROUTES[routes.check_route(route)], gres=routes.GRES, GPUs=1, CPUs=8, memory="64G",
                      minutes=MINUTES, time=routes.time_text(MINUTES), exclude=routes.EXCLUDED),
        expected_readback=routes.expected_route_fields(route), automatic_requeue=False,
        successor="explicit missing-only attempt N+1 on a reviewed route after a terminal predecessor",
        independently_reviewed=False)


def validate_independent(review, plan_ref, plan, cell, route):
    """One immutable review file per plan."""
    require(review.get("status") == INDEPENDENT_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER) and review.get("blocking_findings") == []
            and review.get("sources") == plan["sources"] and review.get("route") == route == plan["route"]
            and review.get("cell") == cell and review.get("plan") == plan_ref,
            "Independent exact source/config PASS bound to this evaluation plan is required")


def private_directory(path):
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.stat(path, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Owned private control directory required: " + str(path))


def contract(p, cell, route, independent=False):
    folder = CONTROL_ROOT / cell
    info = os.stat(folder, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Private owned evaluation control directory required")
    plan_ref = p.descriptor(folder / plan_name(route))
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, cell, plan["function"]["inventory"], route, plan["function"]["inputs_inventory"]["ack"]),
            "Exact source/config/custody/route-bound evaluation plan required")
    review_ref = p.descriptor(folder / review_name(route))
    review = json.loads(p.checked(review_ref))
    require(review["status"] == REVIEW_STATUS and review["independent"] is False and review["plan"] == plan_ref
            and review["sources"] == plan["sources"] and review["blocking_findings"] == []
            and review["cell"] == cell and review["route"] == route,
            "Truthful exact private-owner review required; no independent PASS claimed")
    independent_ref = None
    if independent:
        independent_ref = p.descriptor(folder / independent_name(route))
        validate_independent(json.loads(p.checked(independent_ref)), plan_ref, plan, cell, route)
    return folder, plan_ref, plan, review_ref, independent_ref


def argv(cells, cell, attempt, plan):
    resource = plan["resource"]
    return ["/usr/bin/sbatch", "--parsable", "--hold", "--no-requeue", "--export=NIL",
            "--account=" + resource["account"], "--partition=" + resource["partition"], "--qos=" + resource["qos"],
            "--gres=gpu:L40S:1", "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
            "--time=" + time_text(MINUTES), "--exclude=" + EXCLUDED,
            "--job-name=" + job_name(cells, cell, attempt), "--comment=" + comment(plan, attempt),
            "--chdir=" + str(HOME), "--input=/dev/null", "--output=/dev/null", "--error=/dev/null",
            "--dependency=" + plan["dependency"],
            str(SOURCE / "b4_run7.sbatch"), cell, str(attempt), check_route(plan["route"])]


def scheduler(p, cells, cell, attempt, plan, job, held=False, running=False):
    raw = p.query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = p.fields(raw)
    expected = dict(
        plan["expected_readback"], JobId=job, UserId="luyangz(2708899)", JobName=job_name(cells, cell, attempt),
        Comment=comment(plan, attempt), NumCPUs="8", NumTasks="1", TimeLimit=time_text(MINUTES),
        Command=str(SOURCE / "b4_run7.sbatch"), WorkDir=str(HOME),
        StdIn="/dev/null", StdOut="/dev/null", StdErr="/dev/null")
    # Slurm shows afterok:<job>(unfulfilled) while unsatisfied and afterok:<job>[(fulfilled)] or (null) afterwards.
    dependencies = (("(null)", plan["dependency"], plan["dependency"] + "(fulfilled)")
                    + (() if running else (plan["dependency"] + "(unfulfilled)",)))
    # Slurm shows a pending single-node request as NumNodes=1-1 and a running one as 1.
    nodes = row.get("NumNodes") == "1" if running else row.get("NumNodes") in ("1", "1-1")
    require(set(plan["expected_readback"]) == {"Account", "Partition", "QOS", "Requeue", "TresPerNode", "ExcNodeList"}
            and plan["expected_readback"]["Requeue"] == "0"
            and all(row.get(key) == value for key, value in expected.items()) and nodes
            and row.get("Dependency") in dependencies
            and shlex.split(row.get("SubmitLine", "")) == argv(cells, cell, attempt, plan),
            "Actual physical evaluation request/source/resources differ")
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
    require(spool == p.checked(plan["sources"]["b4_run7.sbatch"]), "Exact immutable bound spool required")
    return raw, spool


def make_plan_file(p, cell, inventory_path, route, inputs_ack_path):
    cells = bound_cells(p)
    check_cell(cells, cell)
    check_route(route)
    private_directory(CONTROL_ROOT)
    private_directory(CONTROL_ROOT / cell)
    path = CONTROL_ROOT / cell / plan_name(route)
    require(not path.exists(), "Existing evaluation plan is immutable; use a new version")
    for name in SOURCES:
        p.checked(SOURCES[name])
    plan = make_plan(p, cell, p.descriptor(inventory_path), route, p.descriptor(inputs_ack_path))
    for other in ROUTES:
        sibling = CONTROL_ROOT / cell / plan_name(other)
        if other != route and sibling.exists():
            require(json.loads(p.checked(p.descriptor(sibling)))["function"] == plan["function"],
                    "Both reviewed routes of one cell must bind the identical evaluation function")
    return p.seal(path, plan)


def owner_review(p, cell, route, notes):
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete self-review notes required")
    folder = CONTROL_ROOT / cell
    plan_ref = p.descriptor(folder / plan_name(route))
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, cell, plan["function"]["inventory"], route, plan["function"]["inputs_inventory"]["ack"]),
            "Self-review must bind the exact plan")
    return p.seal(folder / review_name(route), dict(
        status=REVIEW_STATUS, independent=False, ai_review_disclosed=True, reviewer=OWNER, cell=cell, route=route,
        plan=plan_ref, sources=plan["sources"], function_identity=plan["function_identity"],
        blocking_findings=[], notes=notes))


def sacct_state(p, job):
    value = p.query(["/usr/bin/sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State,ExitCode"]).strip().split("|")
    require(len(value) == 3 and value[0] == job, "Exact scheduler accounting record required")
    return value[1].split(" ")[0], value[2]


def diagnosis_names(job):
    return "DIAGNOSIS-%s.json" % job, "DIAGNOSIS_REVIEW-%s.json" % job


def owner_diagnosis(p, previous, ack, code, plan):
    """Owner DIAGNOSIS of a FAILED predecessor bound to that job, its ACK, scheduler state and exit code, a
    contained hashed worker-log excerpt, a no-QA-commit or journal-integrity proof, the repair reference,
    this plan function and this controller."""
    job = ack["job_id"]
    diagnosis_ref = p.descriptor(previous / diagnosis_names(job)[0])
    diagnosis = json.loads(p.checked(diagnosis_ref))
    scheduled, evidence = diagnosis.get("scheduler") or {}, diagnosis.get("evidence") or {}
    journal, repair = diagnosis.get("journal") or {}, diagnosis.get("repair") or {}
    require(diagnosis.get("schema") == DIAGNOSIS_SCHEMA and diagnosis.get("status") == DIAGNOSIS_STATUS
            and diagnosis.get("owner") == OWNER and diagnosis.get("cell") == plan["cell"]
            and diagnosis.get("attempt") == ack["attempt"] and diagnosis.get("job_id") == job
            and diagnosis.get("ack") == p.descriptor(previous / "ACK.json")
            and scheduled.get("state") == "FAILED" and scheduled.get("exit_code") == code
            and diagnosis.get("function_identity") == plan["function_identity"]
            and diagnosis.get("controller") == p.descriptor(SOURCE / "prepare7_b4.py")
            and isinstance(diagnosis.get("cause"), str) and diagnosis["cause"].strip() != ""
            and (evidence.get("worker_log") or {}).get("path") == str(previous / ("worker-" + job + ".log"))
            and journal.get("basis") in DIAGNOSIS_BASES and journal.get("attempt_result_absent") is True
            and isinstance(journal.get("proof"), str) and journal["proof"].strip() != ""
            and isinstance(repair.get("reference"), str) and repair["reference"].strip() != ""
            and isinstance(repair.get("records"), list) and repair["records"]
            and isinstance(evidence.get("records", []), list),
            "Exact owner DIAGNOSIS bound to the FAILED predecessor job, this plan and this controller required")
    excerpt = evidence.get("log_excerpt")
    require(isinstance(excerpt, str) and excerpt.strip() != "" and excerpt.encode() in p.checked(evidence["worker_log"])
            and evidence.get("log_excerpt_sha256") == hashlib.sha256(excerpt.encode()).hexdigest(),
            "The diagnosis excerpt must be contained in the exact worker log and hashed")
    for ref in evidence.get("records", []) + repair["records"]:
        p.checked(ref)
    return diagnosis_ref


def diagnosed_failure(p, previous, ack, code, plan):
    diagnosis_ref = owner_diagnosis(p, previous, ack, code, plan)
    review_ref = p.descriptor(previous / diagnosis_names(ack["job_id"])[1])
    review = json.loads(p.checked(review_ref))
    require(review.get("status") == DIAGNOSIS_REVIEW_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER) and review.get("blocking_findings") == []
            and review.get("diagnosis") == diagnosis_ref and review.get("job_id") == ack["job_id"]
            and review.get("controller") == p.descriptor(SOURCE / "prepare7_b4.py"),
            "Independent PASS review bound to this exact DIAGNOSIS and this controller required")
    return dict(diagnosis=diagnosis_ref, review=review_ref)


def predecessor_allows(p, folder, attempt, plan):
    """Successor only after a terminal predecessor that reported partial progress, was interrupted, or FAILED
    with an independently reviewed diagnosis; CANCELLED, OUT_OF_MEMORY or COMPLETED without a result never."""
    require(not any((folder / ("attempt-%d" % index)).exists() for index in range(attempt + 1, MAX_ATTEMPTS)),
            "Later attempt folders already exist; successors are strictly sequential")
    if attempt == 0:
        return None
    previous = folder / ("attempt-%d" % (attempt - 1))
    ack = json.loads(p.checked(p.descriptor(previous / "ACK.json")))
    require(re.fullmatch(r"[1-9]\d*", ack["job_id"]) is not None and ack["attempt"] == attempt - 1,
            "Exact predecessor ACK required")
    state, code = sacct_state(p, ack["job_id"])
    require(state in TERMINAL, "Predecessor still active; never run two writers on one journal")
    receipt = previous / ("ATTEMPT_RESULT-" + ack["job_id"] + ".json")
    diagnosis = None
    if receipt.exists():
        result = json.loads(p.checked(p.descriptor(receipt)))
        require(result["status"] == PROGRESS_STATUS and result["new_answers"] > 0,
                "Only a partial predecessor with real new accepted QAs permits a successor")
    elif state == "FAILED":
        diagnosis = diagnosed_failure(p, previous, ack, code, plan)
    else:
        require(state in INTERRUPTED, "A predecessor without a result needs diagnosis, not a blind successor")
    return dict(job_id=ack["job_id"], route=ack.get("route"), state=state, exit_code=code,
                result_receipt_present=receipt.exists(), diagnosis=diagnosis)


def admit(p, cell, attempt, route, release=False):
    cells = bound_cells(p)
    check_attempt(attempt)
    routes = bound_routes(p)
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, cell, route, independent=release)
    afolder = folder / ("attempt-%d" % attempt)
    with p.admission_lock():
        require(contract(p, cell, route, independent=release) == (folder, plan_ref, plan, review_ref, independent_ref),
                "Contract changed while acquiring shared admission lock")
        if not release:
            predecessor = predecessor_allows(p, folder, attempt, plan)
            rows, gate = routes.actual_gate(p.query, route)
            prefix = job_name(cells, cell, 0)[:-1]
            require(not any(plan["function_identity"] in row[2] or row[3].startswith(prefix) for row in rows),
                    "A live attempt for this cell already exists on some route")
            private_directory(afolder)
            require(not any((afolder / x).exists() for x in
                            ("INTENT.json", "ACK.json", "AMBIGUOUS.json", "FUNCTION_CLAIM.json")),
                    "Existing submission or ambiguous ACK: reconcile, never duplicate")
            claim = p.seal(afolder / "FUNCTION_CLAIM.json", dict(
                owner=OWNER, cell=cell, attempt=attempt, route=route, function=plan["function"],
                function_identity=plan["function_identity"], plan=plan_ref, predecessor=predecessor))
            intent = p.seal(afolder / "INTENT.json", dict(
                owner=OWNER, cell=cell, attempt=attempt, route=route, plan=plan_ref, review=review_ref, claim=claim,
                argv=argv(cells, cell, attempt, plan), route_gate=gate, expanded_requests=len(rows)))
            try:
                response = subprocess.run(argv(cells, cell, attempt, plan), text=True, capture_output=True,
                                          timeout=45, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as error:
                p.seal(afolder / "AMBIGUOUS.json", dict(intent=intent, error=repr(error)))
                raise
            good = response.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;[\w.-]+)?\n?", response.stdout)
            if good:
                job = response.stdout.strip().split(";")[0]
                ack_ref = p.seal(afolder / "ACK.json", dict(
                    owner=OWNER, cell=cell, attempt=attempt, route=route, job_id=job, intent=intent, plan=plan_ref))
            p.seal(afolder / "SUBMISSION_RESULT.json", dict(
                intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
            require(good, "No unambiguous actual ACK; reconcile exact intent before another submission")
            raw, spool = scheduler(p, cells, cell, attempt, plan, job, held=True)
            p.seal(afolder / "HELD_READBACK.json", dict(
                ACK=ack_ref, job_id=job, scheduler=raw, spool_base64=base64.b64encode(spool).decode()))
            return dict(cell=cell, attempt=attempt, route=route, job_id=job, state="HELD_VERIFIED")
        ack_ref = p.descriptor(afolder / "ACK.json")
        ack = json.loads(p.checked(ack_ref))
        require(ack["owner"] == OWNER and ack["plan"] == plan_ref and ack["cell"] == cell and ack["route"] == route
                and ack["attempt"] == attempt, "Exact own ACK required")
        require(not (afolder / "RELEASE_INTENT.json").exists(), "Existing release requires reconciliation")
        job = ack["job_id"]
        scheduler(p, cells, cell, attempt, plan, job, held=True)
        inputs = inputs_ready(p, plan)
        gate = inventory_gate(p, plan)
        p.seal(afolder / "RELEASE_INVENTORY_GATE.json", dict(ACK=ack_ref, job_id=job, gate=gate))
        intent = p.seal(afolder / "RELEASE_INTENT.json", dict(ACK=ack_ref, job_id=job,
                                                             independent_review=independent_ref,
                                                             inputs_result=inputs["inventory_result"]))
        response = subprocess.run(["/usr/bin/scontrol", "release", job], text=True, capture_output=True,
                                  timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
        p.seal(afolder / "RELEASE_RESULT.json", dict(
            intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        require(response.returncode == 0, "Release failed or ambiguous; preserve exact identity")
        raw, _ = scheduler(p, cells, cell, attempt, plan, job)
        row = p.fields(raw)
        require(row.get("JobState") in ("PENDING", "RUNNING", "CONFIGURING")
                and row.get("Reason") not in ("JobHeldUser", "JobHeldAdmin", "DependencyNeverSatisfied"),
                "Released evaluation must be eligible or running")
        p.seal(afolder / "RELEASE_READBACK.json", dict(job_id=job, route=route, scheduler=raw))
        return dict(cell=cell, attempt=attempt, route=route, job_id=job,
                    state=row.get("JobState") + ":" + row.get("Reason", ""))


def time_left_seconds(text):
    text = text.strip()
    match = re.fullmatch(r"(?:(\d+)-(\d+):|(\d+):)?(\d+):(\d+)", text)
    require(match is not None, "Actual remaining Slurm time required")
    days, day_hours, hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    require(minutes < 60 and seconds < 60 and day_hours < 24, "Malformed remaining Slurm time")
    return (((days * 24 + day_hours + hours) * 60) + minutes) * 60 + seconds


def sources_sha256(p, eval_sources):
    # Only generation sources enter the journal identity: admission routes, the controller and the scorer
    # can be versioned without orphaning accepted QAs of the same model, inputs and decoding.
    closure = dict(bound={k: SOURCES[k]["sha256"] for k in GENERATION_SOURCES},
                   evaluator_sources=hashlib.sha256(p.canonical(eval_sources)).hexdigest(), helper=HELPER["sha256"])
    return hashlib.sha256(p.canonical(closure)).hexdigest()


def output_attempt(cell, job):
    for path in (OUTPUT_ROOT, OUTPUT_ROOT / "cells", OUTPUT_ROOT / "cells" / cell,
                 OUTPUT_ROOT / "cells" / cell / "attempts"):
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        info = os.stat(path, follow_symlinks=False)
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
                "Private own output directory required: " + str(path))
    attempt = OUTPUT_ROOT / "cells" / cell / "attempts" / ("job-" + job)
    os.mkdir(attempt, 0o700)
    return attempt


def work(p, cell, attempt, route):
    cells = bound_cells(p)
    check_attempt(attempt)
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, cell, route, independent=True)
    afolder = folder / ("attempt-%d" % attempt)
    job = os.environ.get("SLURM_JOB_ID", "")
    ack_ref = p.descriptor(afolder / "ACK.json")
    ack = json.loads(p.checked(ack_ref))
    require(os.getuid() == UID and job.isdecimal() and ack["job_id"] == job and ack["cell"] == cell
            and ack["attempt"] == attempt and ack["route"] == route
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0"
            and not os.environ.get("SLURM_ARRAY_JOB_ID"), "Actual unrestarted owned evaluation required")
    release = json.loads(p.checked(p.descriptor(afolder / "RELEASE_INTENT.json")))
    inputs = inputs_ready(p, plan)
    require(release == dict(ACK=ack_ref, job_id=job, independent_review=independent_ref,
                            inputs_result=inputs["inventory_result"]),
            "Actual independently reviewed release intent bound to the completed inputs required before DATA")
    scheduler(p, cells, cell, attempt, plan, job, running=True)
    node = socket.gethostname().split(".")[0]
    p.validate_process(job, os.getpid(), node, os.environ.get("SLURM_STEP_ID"),
                       p.query(["/usr/bin/scontrol", "listpids", job]),
                       p.fields(p.query(["/usr/bin/scontrol", "show", "step", job + ".batch", "-o"])),
                       (Path("/proc") / str(os.getpid()) / "status").read_text())
    left = time_left_seconds(p.query(["/usr/bin/squeue", "-h", "-j", job, "-o", "%L"]))
    require(left > RESERVE_SECONDS + 600, "Too little allocation left for a durable attempt")
    actor = dict(job_id=job, pid=os.getpid(), node=node, plan=plan_ref, cell=cell, attempt=attempt, route=route)
    p.seal(afolder / ("ACTOR_BEFORE-" + job + ".json"), actor)
    output = output_attempt(cell, job)
    p.seal(output / "OWNER.json", dict(owner=OWNER, cell=cell, attempt=attempt, route=route, job_id=job,
                                       banks=plan["function"]["banks"], function_identity=plan["function_identity"]))
    evaluator = json.loads(p.checked(p.EVAL_CONFIG))
    config = dict(
        role=ROLE, owner=OWNER, job_id=job, cell=cell, attempt=attempt, route=route, banks=plan["function"]["banks"],
        custody=dict(plan["function"]["custody"], **custody_ready(p, plan)), inputs=inputs, sources={k: SOURCES[k] for k in GENERATION_SOURCES},
        sources_sha256=sources_sha256(p, evaluator["sources"]), eval_sources=evaluator["sources"],
        actual_actor=actor, deadline_epoch=time.time() + left - RESERVE_SECONDS, output=str(output))
    config_ref = p.seal(afolder / ("MODEL_CONFIG-" + job + ".json"), config)
    command = [PYTHON, "-u", "-B", "-X", "pycache_prefix=/dev/null/no-cache",
               SOURCES["model4.py"]["path"], "--config", config_ref["path"]]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HOME="/data/hf_cache", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               TORCH_EXTENSIONS_DIR="/data/hf_cache/project_envs/DS-MultiModal/torch_extensions",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="8")
    p.seal(afolder / ("MODEL_INTENT-" + job + ".json"), dict(config=config_ref, argv=command, actor=actor))
    started = time.monotonic()
    result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, timeout=max(60, left - 300))
    p.seal(afolder / ("MODEL_EXIT-" + job + ".json"), dict(
        returncode=result.returncode, elapsed_seconds=time.monotonic() - started, output=str(output)))
    require(result.returncode == 0, "Evaluation failed; preserve the log, accepted QAs and every receipt")
    raw = (output / "RESULT.json").read_bytes()
    require(len(raw) <= 2 << 20, "Attempt result must stay a small control record")
    report = json.loads(raw)
    require(report["job_id"] == job and report["cell"] == cell and report["planned_banks"] == plan["function"]["banks"]
            and report["status"] in (COMPLETE_STATUS, PROGRESS_STATUS), "Exact own attempt result required")
    p.seal(afolder / ("ATTEMPT_RESULT-" + job + ".json"), report)
    scheduler(p, cells, cell, attempt, plan, job, running=True)
    p.seal(afolder / ("ACTOR_AFTER-" + job + ".json"), actor)
    print(json.dumps(dict(job_id=job, cell=cell, attempt=attempt, route=route, status=report["status"],
                          banks={b["protocol"]: b["status"] for b in report["banks"]})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "self-review", "submit", "release", "worker"))
    parser.add_argument("--cell", required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--inventory")
    parser.add_argument("--inputs-ack")
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    p = load_helpers()
    if args.command == "worker":
        work(p, args.cell, args.attempt, args.route)
    elif args.command == "plan":
        require(args.inventory and args.inputs_ack,
                "The cells-v7 inventory run ACK and the BENCH4 input inventory ACK paths are required")
        print(json.dumps(make_plan_file(p, args.cell, Path(args.inventory), args.route, Path(args.inputs_ack)),
                         sort_keys=True))
    elif args.command == "self-review":
        print(json.dumps(owner_review(p, args.cell, args.route, args.note), sort_keys=True))
    else:
        print(json.dumps(admit(p, args.cell, args.attempt, args.route, release=args.command == "release"),
                         sort_keys=True))


if __name__ == "__main__":
    main()
