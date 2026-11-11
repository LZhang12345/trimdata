"""Plan, hold-submit, release and run Tool1 VizWiz-test evaluations of CELLS_V7 (four-kind custody) students.

CELLS_V7 custody-only fork of helper_tool1_cells_v4_custody_20261107/prepare4.py (2026-11-10): identical v3 owner,
role, control/output roots, generation/export/scoring files and afterok-inventory admission; the "CELLS_V3.json"/
"custody3.py" source keys (read by the byte-identical v3 model.py) name CELLS_V7.json/custody7.py; plan-time custody
goes through custody7.plan_custody (pinned records of the cell's producer kind); own job names dsmm-t1v7-vz-*,
comment prefix, plan schema and review statuses; every request carries --dependency=afterok:<cells-v7 inventory job>.
The CELLS_V4 docstring follows.

CELLS_V4 custody-only fork of helper_tool1_vizwiz_eval_v3_20261104/prepare.py (2026-11-07). Generation, export and
scoring code are the byte-identical v3 files (model.py, banks.py, score.py; routes.py) and the evaluation keeps the v3
owner, role, control and output roots, so the unchanged v3 registered scoring step (score_step.py) sees the result
exactly like the other v3 cells. Differences from v3, all custody/admission:
  * custody: SOURCES "CELLS_V3.json"/"custody3.py" keys (read by the byte-identical model.py) name CELLS_V4.json and
    custody4.py, which bind the General successor-trainer records (trainer v8 PLAN-v8 + successor-<n> ACK/CLAIM/
    HELD_READBACK, pinned TRAINING_RESULT and producer RESULT.json) and the cells-v4 inventory;
  * admission: own job names (dsmm-t1v4-vz-*), comment prefix and review statuses; every request carries
    --dependency=afterok:<cells-v4 inventory job> (the same job the plan's custody binds). Release then requires
    either that inventory's sealed PASS RESULT (v3 gate) or a still pending/running inventory job (sealed in
    RELEASE_INVENTORY_GATE.json); the worker late-binds the PASS RESULT before any data exactly as in v3
    (custody_ready in work()). A cell FAIL or failed inventory therefore never reaches model loading.
The original v3 docstring follows.


Mirrors helper_tool1_baseline_eval_v1_20261102/prepare.py: exact pinned sources, PLAN and truthful
owner review, independent exact-source/config review before release, held submission under the
shared ADMISSION.lock with durable INTENT/ACK/SUBMISSION_RESULT/HELD_READBACK, and a worker that
re-verifies its own running allocation before any data. One (protocol, cell) has one stable QA
journal; attempt N>0 is an explicit, result-dependent, missing-only successor.

Routes: every plan is made for one reviewed route (--route general|preempt) and records it in
PLAN-<route>.json, the sbatch argv and the expected scheduler readback (Account/Partition/QOS/
Requeue=0). routes.py gates submission on the actual census under ADMISSION.lock. Both routes are
--no-requeue; a preempted, timed-out or node-failed attempt leaves every committed QA in the
WAL/synchronous=FULL SQLite journal, and the only continuation is the explicit successor
  prepare7_vz.py submit  --protocol P --cell C --attempt N+1 --route R
  prepare7_vz.py release --protocol P --cell C --attempt N+1 --route R
which regenerates nothing already accepted.

v2 (new SOURCE/CONTROL/OUTPUT roots, owner, comment, job names and independent status; the bound v1
evaluator is unchanged): plans only CELLS_V2.json cells whose plan_status is ELIGIBLE (custody from
owners' records, verified by the on-node inventory v3). banks.py, score.py and routes.py are the
unchanged v1 files, bound by exact pins; model.py is the v1 worker generalized to CELLS_V2.

v3 (new SOURCE/CONTROL/OUTPUT roots, owner, role, comment, job names and independent status; the bound v2
evaluator is unchanged): VizWiz-test only (VQAv2 left the suite on 2026-11-04) for the Tool1-trained baseline
students registered in CELLS_V3.json (helper_tool1_cells_v3_20261104). Custody comes from custody3.py: a plan
binds the producer's own PLAN/ACK/TRAINING_RESULT (complete production epoch only) and the cells-v3 inventory
run by its exact ACK; the release gate and the worker late-bind that run's sealed RESULT (PASS, producer
RESULT/file descriptors, loaded parameter digest). VizWiz bank checks use the pinned vqa-inventory-v3
RESULT-11033169 as in v2. Generation, journal function, export and scoring code are unchanged; the horizon
follows each cell's K (no checkpoint-516 assumption).

  prepare7_vz.py plan --protocol vizwiz-test --cell C --route R --inventory <cells-v7-inventory run-N/ACK.json>
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
VZ3 = HOME / "experiments/helper_tool1_vizwiz_eval_v3_20261104"
V1 = HOME / "experiments/helper_tool1_vizwiz_eval_v1_20261102"
CONTROL_ROOT = HOME / "collaboration/helper-tool1-baseline-independent-20261101/vizwiz-eval-v3"
OUTPUT_ROOT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-vizwiz-eval-v3-20261104")
PYTHON = "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python"
OWNER = "Tool1-owned-VizWiz-CELLS-V3-trained-baseline-evaluation-v3"
REVIEW_STATUS = "OWNER_REVIEWED_TOOL1_VIZWIZ_EVAL_V3_CUSTODY_V7_NOT_INDEPENDENT_ACCEPTANCE"
INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_VIZWIZ_EVAL_V3_CUSTODY_V7"
BANK_INVENTORY = {"path": str(HOME / "collaboration/helper-tool1-baseline-independent-20261101/vqa-inventory-v3/run-1/RESULT-11033169.json"),
                  "bytes": 79937, "sha256": "3332dfd597514b41149d30e4244c8a0754bedebfb750ee9713f2116f5a6cd45d"}
INVENTORY_SCHEMA = "tool1-vqa-vizwiz-readonly-inventory-v1"
INVENTORY_STATUS = "COMPLETE_READ_ONLY_INVENTORY_V1"
ROLE = "TOOL1_VIZWIZ_CELLS_V3_TRAINED_BASELINE_EVALUATION"
COMPLETE_STATUS = "COMPLETE_FULL_BANK_PREDICTIONS"
PROGRESS_STATUS = "ACCEPTED_QA_PROGRESS_REQUIRES_REVIEWED_MISSING_ONLY_SUCCESSOR"
COMMENT_PREFIX = "dsmm-t1-vz-eval-v3c7:"
UID = 2708899
EXCLUDED = "babel-p9-20"
RESERVE_SECONDS = 900
MAX_ATTEMPTS = 8
ROUTES = ("general", "preempt")
MINUTES = {"vizwiz-test": 240}
SHORT_PROTOCOL = {"vizwiz-test": "vz"}
TERMINAL = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
INTERRUPTED = ("TIMEOUT", "PREEMPTED", "NODE_FAIL")
HELPER = {
    "path": str(HOME / "experiments/helper_method_eval_bank_qual_v1_20261102/prepare.py"),
    "bytes": 51200, "sha256": "f0dc4f6a548eedc45d5ba3d92e0251328d33bb82b274d815f14b0aca25fa7357",
}
SOURCES = {
    "model.py": {"path": str(VZ3 / "model.py"), "bytes": 26547, "sha256": "b48ef7a4e54eab94f1bcacf05181f02d3f9b63e0f435328044628699ebfefe30"},
    "banks.py": {"path": str(V1 / "banks.py"), "bytes": 17979, "sha256": "1121c76f1bd7e25c70c8fce41c0461037c12e243e87944a81e9c8878cb035816"},
    "score.py": {"path": str(V1 / "score.py"), "bytes": 15544, "sha256": "c1a59d6eb0b14789e905152a5448b22bfc8c4402d6c276992b0dfd168cbbf1e2"},
    "routes.py": {"path": str(V1 / "routes.py"), "bytes": 4797, "sha256": "5067bf831884699edcd0f5d86b5e9c1a3af65def6edf68aab4cb157c3d388709"},
    # Keys kept because the byte-identical model.py reads sources["CELLS_V3.json"] and sources["custody3.py"].
    "CELLS_V3.json": {"path": str(SOURCE / "CELLS_V7.json"), "bytes": 7796, "sha256": "6276480cafcad44030510ede2de94ebf3420c83139e7054c3785bbb7c657bf8f"},
    "custody3.py": {"path": str(SOURCE / "custody7.py"), "bytes": 43616, "sha256": "e835ae87213f19df8f5f61e5bd36c9d47d7ea6e852c5f1f280f5868db52c41c8"},
}
REVIEWED_COMPONENTS = {
    "experiments/helper_tool1_benchmark_scoring_v2_20261102/scoring.py":
        "e32738f1c2364f85850f42a6aa9f4ed5fbd5fd9c1b035195ea1be1b6e32038fc",
    "experiments/helper_tool1_vqa_submission_v2_20261102/adapter.py":
        "03d90b8aeab65607a10f0e8aab44f1f0fcd05a5821f6705729b1128fe6c51204",
    "experiments/helper_tool1_vqa_native_inputs_v2_20261102/inputs.py":
        "5b296dc058975a1b4b2a87a29dd9cc581b9e8942a34a4935d498b7d016170d98",
    "experiments/helper_tool1_vizwiz_score_v1_20261101/scorer.py":
        "0b7c9b89724ef294424b4f2c5d6cd8246a69c48062877dc67e38fd83ff2973d8",
    "experiments/baselines_session3/campaign300/vqa_scoring.py":
        "178806484683a8b52a9f8e3e0ec1739ddb1b2e13ae5f45363a8f71eb1d2bde8a",
    "experiments/baselines_session3/campaign300/official_vqa/vqaEval.py":
        "f08edfcad5be0112500993e245c706b6cb928eadebe203f89f838e5e0d04bec8",
    "experiments/babel_q2_sft/contract.py":
        "eae5fa8d146d73fb2e04a0ce59348ef92c329b2f244a932f02d10a60270a5cea",
    "collaboration/published-foundation-20261022/reference_sources/LLaVA/"
    "c121f0432da27facab705978f83c4ada465e46fd/llava/eval/m4c_evaluator.py":
        "8b49e1033e29d551c0bdd86d9149e2b54370e4cd17e52176ac9061b3521f0774",
}


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
    return bound_module(p, "routes.py", "tool1_vizwiz_routes")


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


def bank_inventory(p):
    """The pinned vqa-inventory-v3 RESULT that verified the VizWiz-test bank inputs for the v2 plans."""
    inventory = json.loads(p.checked(BANK_INVENTORY))
    require(inventory.get("schema") == INVENTORY_SCHEMA and inventory.get("status") == INVENTORY_STATUS
            and inventory.get("label_bytes_parsed") == 0, "Completed read-only bank inventory required")
    return inventory


def scoring_from_inventory(inventory, name):
    """Generation binds no labels: VizWiz is scored only by the registered scoring step (score_step.py)."""
    banks = inventory.get("banks", {})
    value = banks.get(name)
    require(isinstance(value, dict) and value.get("status") == "PASS", "Inventory did not verify these bank inputs")
    scoring = value.get("scoring", {})
    allowed = {"vizwiz-test": (), "vqav2-testdev": ("test2015_questions",),
               "vqav2-val2014": ("annotation_archive",)}[name]
    return {field: scoring.get(field) for field in allowed}


def job_name(cells, name, cell, attempt):
    return "dsmm-t1v7-%s-%s-a%d" % (SHORT_PROTOCOL[name], cells["cells"][cell]["short"], attempt)


def comment(plan, attempt):
    return COMMENT_PREFIX + plan["function_identity"] + ":a%d" % attempt


def time_text(minutes):
    require(type(minutes) is int and 0 < minutes < 24 * 60, "Sub-day walltime required")
    return "%02d:%02d:00" % divmod(minutes, 60)


def make_plan(p, name, cell, inventory_ref, route):
    routes = bound_routes(p)
    banks = bound_module(p, "banks.py", "tool1_vizwiz_banks_plan")
    cells = bound_cells(p)
    protocol = banks.check_runnable(name)
    spec = cells["cells"][check_cell(cells, cell)]
    custody = plan_custody(p, cells, cell, inventory_ref)
    scoring = scoring_from_inventory(bank_inventory(p), name)
    function = dict(
        owner=OWNER, protocol=name, bank=protocol["bank"], split=protocol["split"], cell=cell,
        method=spec["method"], training_seed=spec["training_seed"], K=spec["K"], custody=custody,
        scoring=scoring, inventory=inventory_ref, bank_inventory=BANK_INVENTORY,
        questions=protocol["questions"], decoding=protocol["decoding"],
        prompt_policy=protocol["prompt_policy"], start_configuration=protocol["start_configuration"],
        bound_sources=SOURCES, reviewed_components=REVIEWED_COMPONENTS, metadata_helper=HELPER,
        evaluator_source_configuration=p.EVAL_CONFIG, native_python=PYTHON, output_root=str(OUTPUT_ROOT),
        external_server_required=protocol["external_server_required"], no_model_downloads=True,
        no_dataset_downloads=True, no_training_updates=True, read_only_foreign_checkpoints=True,
        labels_never_used_for_selection_or_tuning=True, not_held_out_until_exposure_audit=True)
    # The route is a plan/resource choice, never part of the journal function: duplicate and live-attempt
    # checks therefore span both routes, and either reviewed route may continue the same journal.
    return dict(
        schema="tool1-vizwiz-eval-v3-custody-v7-plan-v1", owner=OWNER, protocol=name, cell=cell,
        route=check_route(route), function=function,
        function_identity=hashlib.sha256(p.canonical(function)).hexdigest(),
        sources=dict({"prepare7_vz.py": p.descriptor(SOURCE / "prepare7_vz.py"),
                      "vz_run7.sbatch": p.descriptor(SOURCE / "vz_run7.sbatch")}, **SOURCES),
        dependency="afterok:" + custody["inventory"]["job_id"],
        resource=dict(routes.ROUTES[routes.check_route(route)], gres=routes.GRES, GPUs=1, CPUs=8, memory="64G",
                      minutes=MINUTES[name], time=routes.time_text(MINUTES[name]), exclude=routes.EXCLUDED),
        expected_readback=routes.expected_route_fields(route), automatic_requeue=False,
        successor="explicit missing-only attempt N+1 on a reviewed route after a terminal predecessor",
        independently_reviewed=False)


def validate_independent(review, plan_ref, plan, name, cell, route):
    """One immutable review file per plan: adding cells later never rewrites a released attempt's review."""
    require(review.get("status") == INDEPENDENT_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER) and review.get("blocking_findings") == []
            and review.get("sources") == plan["sources"] and review.get("route") == route == plan["route"]
            and review.get("protocol") == name and review.get("cell") == cell and review.get("plan") == plan_ref,
            "Independent exact source/config PASS bound to this evaluation plan is required")


def private_directory(path):
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.stat(path, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Owned private control directory required: " + str(path))


def contract(p, name, cell, route, independent=False):
    folder = CONTROL_ROOT / name / cell
    info = os.stat(folder, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Private owned evaluation control directory required")
    plan_ref = p.descriptor(folder / plan_name(route))
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, name, cell, plan["function"]["inventory"], route),
            "Exact source/config/custody/route-bound evaluation plan required")
    review_ref = p.descriptor(folder / review_name(route))
    review = json.loads(p.checked(review_ref))
    require(review["status"] == REVIEW_STATUS and review["independent"] is False and review["plan"] == plan_ref
            and review["sources"] == plan["sources"] and review["blocking_findings"] == []
            and review["protocol"] == name and review["cell"] == cell and review["route"] == route,
            "Truthful exact private-owner review required; no independent PASS claimed")
    independent_ref = None
    if independent:
        independent_ref = p.descriptor(folder / independent_name(route))
        validate_independent(json.loads(p.checked(independent_ref)), plan_ref, plan, name, cell, route)
    return folder, plan_ref, plan, review_ref, independent_ref


def argv(cells, name, cell, attempt, plan):
    resource = plan["resource"]
    return ["/usr/bin/sbatch", "--parsable", "--hold", "--no-requeue", "--export=NIL",
            "--account=" + resource["account"], "--partition=" + resource["partition"], "--qos=" + resource["qos"],
            "--gres=gpu:L40S:1", "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
            "--time=" + time_text(MINUTES[name]), "--exclude=" + EXCLUDED,
            "--job-name=" + job_name(cells, name, cell, attempt), "--comment=" + comment(plan, attempt),
            "--chdir=" + str(HOME), "--input=/dev/null", "--output=/dev/null", "--error=/dev/null",
            "--dependency=" + plan["dependency"],
            str(SOURCE / "vz_run7.sbatch"), name, cell, str(attempt), check_route(plan["route"])]


def scheduler(p, cells, name, cell, attempt, plan, job, held=False, running=False):
    raw = p.query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = p.fields(raw)
    expected = dict(
        plan["expected_readback"], JobId=job, UserId="luyangz(2708899)", JobName=job_name(cells, name, cell, attempt),
        Comment=comment(plan, attempt), NumCPUs="8", NumTasks="1", TimeLimit=time_text(MINUTES[name]),
        Command=str(SOURCE / "vz_run7.sbatch"), WorkDir=str(HOME),
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
            and shlex.split(row.get("SubmitLine", "")) == argv(cells, name, cell, attempt, plan),
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
    require(spool == p.checked(plan["sources"]["vz_run7.sbatch"]), "Exact immutable bound spool required")
    return raw, spool


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


def reviewed_components_present(p):
    """Plan/submit-time check that the reviewed scoring closure is staged byte-exactly on HOME."""
    for relative, digest in REVIEWED_COMPONENTS.items():
        require(p.descriptor(HOME / relative)["sha256"] == digest, "Reviewed component not staged exactly: " + relative)
    return dict(REVIEWED_COMPONENTS)


def make_plan_file(p, name, cell, inventory_path, route):
    cells = bound_cells(p)
    check_cell(cells, cell)
    check_route(route)
    reviewed_components_present(p)
    private_directory(CONTROL_ROOT)
    private_directory(CONTROL_ROOT / name)
    private_directory(CONTROL_ROOT / name / cell)
    path = CONTROL_ROOT / name / cell / plan_name(route)
    require(not path.exists(), "Existing evaluation plan is immutable; use a new version")
    for name_ in SOURCES:
        p.checked(SOURCES[name_])
    plan = make_plan(p, name, cell, p.descriptor(inventory_path), route)
    for other in ROUTES:
        sibling = CONTROL_ROOT / name / cell / plan_name(other)
        if other != route and sibling.exists():
            require(json.loads(p.checked(p.descriptor(sibling)))["function"] == plan["function"],
                    "Both reviewed routes of one journal must bind the identical evaluation function")
    return p.seal(path, plan)


def owner_review(p, name, cell, route, notes):
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete self-review notes required")
    folder = CONTROL_ROOT / name / cell
    plan_ref = p.descriptor(folder / plan_name(route))
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, name, cell, plan["function"]["inventory"], route),
            "Self-review must bind the exact plan")
    return p.seal(folder / review_name(route), dict(
        status=REVIEW_STATUS, independent=False, ai_review_disclosed=True, reviewer=OWNER, protocol=name,
        cell=cell, route=route, plan=plan_ref, sources=plan["sources"], function_identity=plan["function_identity"],
        blocking_findings=[], notes=notes))


def sacct_state(p, job):
    value = p.query(["/usr/bin/sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State,ExitCode"]).strip().split("|")
    require(len(value) == 3 and value[0] == job, "Exact scheduler accounting record required")
    return value[1].split(" ")[0], value[2]


def predecessor_allows(p, folder, attempt):
    """Successor only after a terminal predecessor that reported partial progress or was interrupted.

    A no-requeue attempt killed by preemption, its time limit or a node failure writes no result, but
    every QA it committed is durable in the journal; FAILED, CANCELLED, OUT_OF_MEMORY or a COMPLETED
    job without a result needs diagnosis, never a blind successor.
    """
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
    if receipt.exists():
        result = json.loads(p.checked(p.descriptor(receipt)))
        require(result["status"] == PROGRESS_STATUS and result["new_answers"] > 0
                and result["completion"]["complete"] is False,
                "Only a partial predecessor with real new accepted QAs permits a successor")
    else:
        require(state in INTERRUPTED, "A predecessor without a result needs diagnosis, not a blind successor")
    return dict(job_id=ack["job_id"], route=ack.get("route"), state=state, exit_code=code,
                result_receipt_present=receipt.exists())


def admit(p, name, cell, attempt, route, release=False):
    cells = bound_cells(p)
    check_attempt(attempt)
    routes = bound_routes(p)
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, name, cell, route, independent=release)
    afolder = folder / ("attempt-%d" % attempt)
    with p.admission_lock():
        require(contract(p, name, cell, route, independent=release)
                == (folder, plan_ref, plan, review_ref, independent_ref),
                "Contract changed while acquiring shared admission lock")
        if not release:
            predecessor = predecessor_allows(p, folder, attempt)
            reviewed_components_present(p)
            rows, gate = routes.actual_gate(p.query, route)
            prefix = job_name(cells, name, cell, 0)[:-1]
            require(not any(plan["function_identity"] in row[2] or row[3].startswith(prefix) for row in rows),
                    "A live attempt for this journal already exists on some route")
            private_directory(afolder)
            require(not any((afolder / x).exists() for x in
                            ("INTENT.json", "ACK.json", "AMBIGUOUS.json", "FUNCTION_CLAIM.json")),
                    "Existing submission or ambiguous ACK: reconcile, never duplicate")
            claim = p.seal(afolder / "FUNCTION_CLAIM.json", dict(
                owner=OWNER, protocol=name, cell=cell, attempt=attempt, route=route, function=plan["function"],
                function_identity=plan["function_identity"], plan=plan_ref, predecessor=predecessor))
            intent = p.seal(afolder / "INTENT.json", dict(
                owner=OWNER, protocol=name, cell=cell, attempt=attempt, route=route, plan=plan_ref,
                review=review_ref, claim=claim, argv=argv(cells, name, cell, attempt, plan), route_gate=gate,
                expanded_requests=len(rows)))
            try:
                response = subprocess.run(argv(cells, name, cell, attempt, plan), text=True, capture_output=True,
                                          timeout=45, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as error:
                p.seal(afolder / "AMBIGUOUS.json", dict(intent=intent, error=repr(error)))
                raise
            good = response.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;[\w.-]+)?\n?", response.stdout)
            if good:
                job = response.stdout.strip().split(";")[0]
                ack_ref = p.seal(afolder / "ACK.json", dict(
                    owner=OWNER, protocol=name, cell=cell, attempt=attempt, route=route, job_id=job, intent=intent,
                    plan=plan_ref))
            p.seal(afolder / "SUBMISSION_RESULT.json", dict(
                intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
            require(good, "No unambiguous actual ACK; reconcile exact intent before another submission")
            raw, spool = scheduler(p, cells, name, cell, attempt, plan, job, held=True)
            p.seal(afolder / "HELD_READBACK.json", dict(
                ACK=ack_ref, job_id=job, scheduler=raw, spool_base64=base64.b64encode(spool).decode()))
            return dict(protocol=name, cell=cell, attempt=attempt, route=route, job_id=job, state="HELD_VERIFIED")
        ack_ref = p.descriptor(afolder / "ACK.json")
        ack = json.loads(p.checked(ack_ref))
        require(ack["owner"] == OWNER and ack["plan"] == plan_ref and ack["cell"] == cell and ack["route"] == route
                and ack["protocol"] == name and ack["attempt"] == attempt, "Exact own ACK required")
        require(not (afolder / "RELEASE_INTENT.json").exists(), "Existing release requires reconciliation")
        job = ack["job_id"]
        scheduler(p, cells, name, cell, attempt, plan, job, held=True)
        gate = inventory_gate(p, plan)
        p.seal(afolder / "RELEASE_INVENTORY_GATE.json", dict(ACK=ack_ref, job_id=job, gate=gate))
        intent = p.seal(afolder / "RELEASE_INTENT.json", dict(ACK=ack_ref, job_id=job,
                                                             independent_review=independent_ref))
        response = subprocess.run(["/usr/bin/scontrol", "release", job], text=True, capture_output=True,
                                  timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
        p.seal(afolder / "RELEASE_RESULT.json", dict(
            intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        require(response.returncode == 0, "Release failed or ambiguous; preserve exact identity")
        raw, _ = scheduler(p, cells, name, cell, attempt, plan, job)
        row = p.fields(raw)
        require(row.get("JobState") in ("PENDING", "RUNNING", "CONFIGURING")
                and row.get("Reason") not in ("JobHeldUser", "JobHeldAdmin", "DependencyNeverSatisfied"),
                "Released evaluation must be eligible or running")
        p.seal(afolder / "RELEASE_READBACK.json", dict(job_id=job, route=route, scheduler=raw))
        return dict(protocol=name, cell=cell, attempt=attempt, route=route, job_id=job,
                    state=row.get("JobState") + ":" + row.get("Reason", ""))


def time_left_seconds(text):
    text = text.strip()
    match = re.fullmatch(r"(?:(\d+)-(\d+):|(\d+):)?(\d+):(\d+)", text)
    require(match is not None, "Actual remaining Slurm time required")
    days, day_hours, hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    require(minutes < 60 and seconds < 60 and day_hours < 24, "Malformed remaining Slurm time")
    return (((days * 24 + day_hours + hours) * 60) + minutes) * 60 + seconds


def sources_sha256(p, eval_sources):
    # Admission route code is excluded so the resumable journal identity never depends on the queue used.
    closure = dict(bound={k: v["sha256"] for k, v in SOURCES.items() if k != "routes.py"}, reviewed=REVIEWED_COMPONENTS,
                   evaluator_sources=hashlib.sha256(p.canonical(eval_sources)).hexdigest(), helper=HELPER["sha256"])
    return hashlib.sha256(p.canonical(closure)).hexdigest()


def output_attempt(name, cell, job):
    for path in (OUTPUT_ROOT, OUTPUT_ROOT / name, OUTPUT_ROOT / name / cell, OUTPUT_ROOT / name / cell / "attempts"):
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        info = os.stat(path, follow_symlinks=False)
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
                "Private own output directory required: " + str(path))
    attempt = OUTPUT_ROOT / name / cell / "attempts" / ("job-" + job)
    os.mkdir(attempt, 0o700)
    return attempt


def work(p, name, cell, attempt, route):
    cells = bound_cells(p)
    check_attempt(attempt)
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, name, cell, route, independent=True)
    afolder = folder / ("attempt-%d" % attempt)
    job = os.environ.get("SLURM_JOB_ID", "")
    ack_ref = p.descriptor(afolder / "ACK.json")
    ack = json.loads(p.checked(ack_ref))
    require(os.getuid() == UID and job.isdecimal() and ack["job_id"] == job and ack["cell"] == cell
            and ack["protocol"] == name and ack["attempt"] == attempt and ack["route"] == route
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0"
            and not os.environ.get("SLURM_ARRAY_JOB_ID"), "Actual unrestarted owned evaluation required")
    release = json.loads(p.checked(p.descriptor(afolder / "RELEASE_INTENT.json")))
    require(release == dict(ACK=ack_ref, job_id=job, independent_review=independent_ref),
            "Actual independently reviewed release intent required before DATA")
    scheduler(p, cells, name, cell, attempt, plan, job, running=True)
    node = socket.gethostname().split(".")[0]
    p.validate_process(job, os.getpid(), node, os.environ.get("SLURM_STEP_ID"),
                       p.query(["/usr/bin/scontrol", "listpids", job]),
                       p.fields(p.query(["/usr/bin/scontrol", "show", "step", job + ".batch", "-o"])),
                       (Path("/proc") / str(os.getpid()) / "status").read_text())
    left = time_left_seconds(p.query(["/usr/bin/squeue", "-h", "-j", job, "-o", "%L"]))
    require(left > RESERVE_SECONDS + 600, "Too little allocation left for a durable attempt")
    actor = dict(job_id=job, pid=os.getpid(), node=node, plan=plan_ref, protocol=name, cell=cell, attempt=attempt,
                 route=route)
    p.seal(afolder / ("ACTOR_BEFORE-" + job + ".json"), actor)
    output = output_attempt(name, cell, job)
    p.seal(output / "OWNER.json", dict(owner=OWNER, protocol=name, cell=cell, attempt=attempt, route=route,
                                       job_id=job, function_identity=plan["function_identity"]))
    evaluator = json.loads(p.checked(p.EVAL_CONFIG))
    config = dict(
        role=ROLE, owner=OWNER, job_id=job, protocol=name, cell=cell, attempt=attempt, route=route,
        custody=dict(plan["function"]["custody"], **custody_ready(p, plan)), scoring=plan["function"]["scoring"], sources=SOURCES,
        sources_sha256=sources_sha256(p, evaluator["sources"]), eval_sources=evaluator["sources"],
        actual_actor=actor, deadline_epoch=time.time() + left - RESERVE_SECONDS, output=str(output))
    config_ref = p.seal(afolder / ("MODEL_CONFIG-" + job + ".json"), config)
    command = [PYTHON, "-u", "-B", "-X", "pycache_prefix=/dev/null/no-cache",
               SOURCES["model.py"]["path"], "--config", config_ref["path"]]
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
    require(report["job_id"] == job and report["protocol"] == name and report["cell"] == cell
            and report["status"] in (COMPLETE_STATUS, PROGRESS_STATUS), "Exact own attempt result required")
    p.seal(afolder / ("ATTEMPT_RESULT-" + job + ".json"), report)
    scheduler(p, cells, name, cell, attempt, plan, job, running=True)
    p.seal(afolder / ("ACTOR_AFTER-" + job + ".json"), actor)
    print(json.dumps(dict(job_id=job, protocol=name, cell=cell, attempt=attempt, route=route,
                          status=report["status"], completion=report["completion"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "self-review", "submit", "release", "worker"))
    parser.add_argument("--protocol", choices=sorted(MINUTES), required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--inventory")
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    p = load_helpers()
    if args.command == "worker":
        work(p, args.protocol, args.cell, args.attempt, args.route)
    elif args.command == "plan":
        require(args.inventory, "The cells-v7 inventory run ACK.json path is required")
        print(json.dumps(make_plan_file(p, args.protocol, args.cell, Path(args.inventory), args.route),
                         sort_keys=True))
    elif args.command == "self-review":
        print(json.dumps(owner_review(p, args.protocol, args.cell, args.route, args.note), sort_keys=True))
    else:
        print(json.dumps(admit(p, args.protocol, args.cell, args.attempt, args.route,
                               release=args.command == "release"), sort_keys=True))


if __name__ == "__main__":
    main()
