"""Afterok registered scoring of CELLS_V7 generations (BENCH6, BENCH4-local, VizWiz-test), reusing the v3 scoring steps.

CELLS_V7 adapter in the reviewed CELLS_V4 pattern (helper_tool1_cells_v4_custody_20261107/score4.py), generalized to
three bound v3 scoring steps and to multi-cell batches. Each step is executed from its exact pinned bytes as a module
(never as __main__); every scoring constant, scorer, label source and SCORE schema is the step's own:
  * bench6: helper_tool1_bench6_eval_v3_20261104/score_step6.py (label_sources, validate_generation, score_cell; MME
    and SEED official converters; SCORE files in this adapter's batch cells/ folder);
  * bench4: helper_tool1_bench4_score_v3_20261105/score_step4_v3.py kind local (MMMU-val official parse/eval port,
    MMBench-EN dev CircularEval; bank_inputs/score_local_cell; labels/questions/answer dict from the one BENCH4 input
    inventory RESULT the generation plans bind);
  * vizwiz: helper_tool1_vizwiz_eval_v3_20261104/score_step.py (registered acquisition reuse, score_batch -> pinned
    official VQAEval; SCORE files in the v3 step's cells/ folder, create-once-or-identical).
Changes from score4, all admission/custody:
  * plan: binds, per cell, the newest generation ACK made by the pinned CELLS_V7 controller of that bank
    (prepare7_b6.py / prepare7_b4.py / prepare7_vz.py; plan schema, cell, CELLS_V7/custody7 source pins checked) and its
    job ID; the request carries --dependency=afterok:<job>[:<job>...] over every bound generation job;
  * worker (after all bound jobs COMPLETED): late-binds the single complete ATTEMPT_RESULT-<job>.json of exactly each
    bound ACK through the step's own generation_result()/generation_binding()/validate_generation(), seals
    LATE_BINDING-<job>.json, then runs the step's unchanged scoring core. A partial (missing-only successor)
    generation is never scored: the worker refuses before any label read and a new batch follows the successor;
  * a cell bound by any v3 step batch or another batch of this adapter (same scorer) is refused.
If any bound generation job fails, the batch stays DependencyNeverSatisfied: cancel it with a receipt and plan a new
batch for the remaining cells (never blind-resubmit).

  score7.py plan        --scorer S --batch B --route R --cell C [--cell C ...]
  score7.py self-review --scorer S --batch B --route R --note "..."
  score7.py submit|release --scorer S --batch B --attempt N --route R
"""

import argparse
import base64
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
from types import ModuleType


HOME = Path("/home/luyangz/data_selection_multimodal")
SOURCE = HOME / "experiments/helper_tool1_cells_v7_extsuite_20261110"
C = HOME / "collaboration/helper-tool1-baseline-independent-20261101"
UID = 2708899
CELLS5 = {"path": str(SOURCE / "CELLS_V7.json"), "bytes": 7796, "sha256": "6276480cafcad44030510ede2de94ebf3420c83139e7054c3785bbb7c657bf8f"}
CUSTODY5 = {"path": str(SOURCE / "custody7.py"), "bytes": 43616, "sha256": "e835ae87213f19df8f5f61e5bd36c9d47d7ea6e852c5f1f280f5868db52c41c8"}
SCORERS = {
    "bench6": dict(
        step={"path": str(HOME / "experiments/helper_tool1_bench6_eval_v3_20261104/score_step6.py"), "bytes": 33657,
              "sha256": "3758b4a1c7171de6b5213e89088dc2665514717aaef5aeeedd473eb5198445fd"},
        prepare={"path": str(SOURCE / "prepare7_b6.py"), "bytes": 39727, "sha256": "12d2a8a5e1e4b31da69a3046c7a0b09bd9a17b5cba2447385a0ebdb7c17438c5"},
        prepare_name="prepare7_b6.py", generation_schema="tool1-bench6-eval-v3-custody-v7-plan-v1",
        control=C / "bench6-eval-v3" / "scoring-afterok-v7", short="b6",
        step_owner="Tool1-owned-BENCH6-registered-scoring-v3",
        generation_owner="Tool1-owned-BENCH6-CELLS-V3-trained-baseline-evaluation-v3"),
    "bench4": dict(
        step={"path": str(HOME / "experiments/helper_tool1_bench4_score_v3_20261105/score_step4_v3.py"),
              "bytes": 58039, "sha256": "445ae87c6f36b7742374ac3a75403773b894ba56fb8f87f7a5e4822d2da1848e"},
        prepare={"path": str(SOURCE / "prepare7_b4.py"), "bytes": 42210, "sha256": "069be846b61df3b8269bb9b5940f9b32873e8dc998411bccf273adfd334389f3"},
        prepare_name="prepare7_b4.py", generation_schema="tool1-bench4-eval-v3-custody-v7-plan-v1",
        control=C / "bench4-eval-v3" / "scoring-afterok-v7", short="b4",
        step_owner="Tool1-owned-BENCH4-CELLS-V3-registered-local-scoring-v3",
        generation_owner="Tool1-owned-BENCH4-CELLS-V3-trained-baseline-evaluation-v3"),
    "vizwiz": dict(
        step={"path": str(HOME / "experiments/helper_tool1_vizwiz_eval_v3_20261104/score_step.py"), "bytes": 38369,
              "sha256": "33ad9775ff6215319e32c5f4bee347e385cf869b85792d875c363287ff91b05b"},
        prepare={"path": str(SOURCE / "prepare7_vz.py"), "bytes": 39772, "sha256": "65f43663fd2fac3eaa37ff939283434cb375b67fbb7b607774a9b1b4deda06cc"},
        prepare_name="prepare7_vz.py", generation_schema="tool1-vizwiz-eval-v3-custody-v7-plan-v1",
        control=C / "vizwiz-eval-v3" / "vizwiz-test-scoring-afterok-v7", short="vz",
        step_owner="Tool1-owned-VizWiz-test-registered-scoring",
        generation_owner="Tool1-owned-VizWiz-CELLS-V3-trained-baseline-evaluation-v3"),
}
REVIEW_STATUS = "OWNER_REVIEWED_TOOL1_CELLS_V7_AFTEROK_SCORING_NOT_INDEPENDENT_ACCEPTANCE"
INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_CELLS_V7_AFTEROK_SCORING"
COMMENT_PREFIX = "dsmm-t1-score-v7a:"
PLAN_SCHEMA = "tool1-cells-v7-afterok-scoring-plan-v1"
MAX_BATCHES = 8
MAX_ATTEMPTS = 4
MAX_CELLS = 8
CPUS, MEMORY, MEMORY_MB, MINUTES = "4", "32G", "32768M", 120
GENERATION_LIVE = ("PENDING", "RUNNING", "COMPLETED")
DEPENDENCY_ITEM = re.compile(r"afterok:([1-9][0-9]*)(?:_\*)?(?:\((unfulfilled|fulfilled)\))?")


def require(value, message):
    if not value:
        raise ValueError(message)


def read_exact(ref):
    path = ref["path"]
    require(path.startswith("/") and "/../" not in path and "\0" not in path, "Literal absolute path required")
    with ExitStack() as stack:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        stack.callback(os.close, fd)
        parts = path[1:].split("/")
        for component in parts[:-1]:
            fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            stack.callback(os.close, fd)
        source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=fd)
        stack.callback(os.close, source)
        before = os.fstat(source)
        raw = bytearray()
        while len(raw) <= ref["bytes"]:
            block = os.read(source, ref["bytes"] + 1 - len(raw))
            if not block:
                break
            raw.extend(block)
        after = os.fstat(source)
    require(stat.S_ISREG(before.st_mode) and before.st_uid == UID and len(raw) == ref["bytes"]
            and hashlib.sha256(bytes(raw)).hexdigest() == ref["sha256"]
            and all(getattr(before, k) == getattr(after, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns")),
            "Exact immutable pinned bytes required: " + path)
    return bytes(raw)


def check_scorer(scorer):
    require(isinstance(scorer, str) and scorer in SCORERS, "Scorer bench6, bench4 or vizwiz required")
    return SCORERS[scorer]


def load_step(scorer):
    """The bound v3 scoring step as a module (its own pins, constants and functions; never run as __main__)."""
    config = check_scorer(scorer)
    module = ModuleType("tool1_score_step_%s_bound_for_cells_v7_afterok" % scorer)
    module.__file__ = config["step"]["path"]
    exec(compile(read_exact(config["step"]), config["step"]["path"], "exec", dont_inherit=True), vars(module))
    require(module.OWNER == config["step_owner"], "Pinned v3 scoring step identity differs: " + scorer)
    if scorer == "bench4":
        require(module.KINDS == ("local",), "BENCH4 v3 step must be local-only")
    else:
        require(module.GENERATION_OWNER == config["generation_owner"], "Pinned v3 step generation owner differs")
    return module


def check_batch(batch):
    require(type(batch) is int and 0 <= batch < MAX_BATCHES, "Finite afterok scoring batch index required")
    return batch


def check_attempt(attempt):
    require(type(attempt) is int and 0 <= attempt < MAX_ATTEMPTS, "Finite scoring attempt index required")
    return attempt


def batch_folder(scorer, batch):
    return check_scorer(scorer)["control"] / ("batch-%d" % check_batch(batch))


def job_name(scorer, batch, attempt):
    return "dsmm-t1v7-sc-%s-b%d-a%d" % (check_scorer(scorer)["short"], check_batch(batch), check_attempt(attempt))


def comment(plan, attempt):
    return COMMENT_PREFIX + plan["function_identity"] + ":a%d" % attempt


def time_text(minutes):
    return "%02d:%02d:00" % divmod(minutes, 60)


def cells5(p):
    cells = json.loads(p.checked(CELLS5))
    require(cells.get("schema") == "tool1-baseline-cells-v7-extsuite-custody-v1" and isinstance(cells.get("cells"), dict),
            "Exact CELLS_V7 roster required")
    return cells["cells"]


def bench4_family(s, p):
    """The bound BENCH4 v3 controller's own roots/owner (step v3_family), with the CELLS_V7 roster."""
    family = s.v3_family(p)
    require(family["owner"] == SCORERS["bench4"]["generation_owner"], "BENCH4 v3 generation owner differs")
    return dict(family, cells=sorted(cells5(p)))


def generation_folder(scorer, s, p, cell):
    if scorer == "vizwiz":
        return s.EVAL_CONTROL / "vizwiz-test" / cell
    if scorer == "bench6":
        return s.EVAL_CONTROL / cell
    return bench4_family(s, p)["control"] / cell


def generation_ack(scorer, s, p, cell):
    """The newest acknowledged attempt of this cell's v3-lineage generation journal, made by the pinned CELLS_V7
    controller of this bank (plan schema, cell, controller and CELLS_V7/custody7 pins)."""
    config = check_scorer(scorer)
    folder = generation_folder(scorer, s, p, cell)
    attempts = [index for index in range(8) if (folder / ("attempt-%d" % index) / "ACK.json").exists()]
    require(attempts and attempts == list(range(len(attempts))), "Sequential acknowledged generation attempts required")
    ack_ref = p.descriptor(folder / ("attempt-%d" % attempts[-1]) / "ACK.json")
    ack = json.loads(p.checked(ack_ref))
    require(ack.get("owner") == config["generation_owner"] and ack.get("cell") == cell
            and ack.get("attempt") == attempts[-1] and re.fullmatch(r"[1-9]\d*", str(ack.get("job_id"))) is not None
            and (scorer != "vizwiz" or ack.get("protocol") == "vizwiz-test"), "Exact generation ACK required")
    plan = json.loads(p.checked(ack["plan"]))
    require(plan.get("schema") == config["generation_schema"] and plan.get("cell") == cell
            and plan.get("sources", {}).get(config["prepare_name"]) == config["prepare"]
            and plan["sources"].get("CELLS_V3.json") == CELLS5 and plan["sources"].get("custody3.py") == CUSTODY5,
            "Generation must be the pinned CELLS_V7 controller's own attempt")
    return ack_ref, ack, plan


def bound_elsewhere(scorer, s, p, batch):
    """Cells bound by any v3 step batch (both routes) or by another batch of this adapter for the same scorer."""
    if scorer == "bench4":
        taken = set(s.bound_cells_elsewhere(p, "local", None))
    else:
        taken = set(s.bound_cells_elsewhere(p, None))
    for other in range(MAX_BATCHES):
        for route in s.ROUTES:
            path = batch_folder(scorer, other) / ("PLAN-%s.json" % route)
            if other != batch and path.exists():
                taken |= set(json.loads(p.checked(p.descriptor(path)))["function"]["generation_acks"])
    return taken


def bench4_inputs(s, p, generation_plans):
    """The one BENCH4 input inventory RESULT every bound generation plan binds (late-bound path, sealed run)."""
    paths = {plan["function"]["inputs_inventory"]["result_path"] for plan in generation_plans.values()}
    require(len(paths) == 1, "Every bound BENCH4 generation must bind the same input inventory run")
    inventory_ref = p.descriptor(Path(paths.pop()))
    return inventory_ref, s.validate_inventory(json.loads(p.checked(inventory_ref)))


def scorer_function(scorer, s, p, batch, generation_plans):
    """The bound step's own scoring function fields (everything except the late-bound generation results)."""
    if scorer == "vizwiz":
        banks = s.bound_module(p, "banks.py", "tool1_vizwiz_score_banks_cells_v7")
        protocol = banks.protocol("vizwiz-test")
        require(s.bank_constants_agree(banks), "Bound bank protocol and scoring-step constants disagree")
        return dict(owner=s.OWNER, protocol="vizwiz-test", native_source=protocol["native_source"],
                    official_scorer=s.OFFICIAL_SCORER, acquisition=s.acquisition_function(),
                    registry_claim=str(s.REGISTRY), reviewed_components=s.REVIEWED_COMPONENTS,
                    bound_sources=s.SOURCES, metadata_helper=s.HELPER,
                    score_files=str(s.CONTROL / "cells" / "<cell>" / s.SCORE_NAME), questions_per_cell=s.QUESTIONS,
                    per_image_accuracy_retained=False)
    if scorer == "bench6":
        banks = s.bound_module(p, "banks6.py", "tool1_bench6_score_banks_cells_v7")
        require(tuple(banks.WAVE1) == s.WAVE1, "Bound bank protocol and scoring step disagree on wave 1")
        return dict(owner=s.OWNER, banks=list(s.WAVE1), labels=s.label_sources(banks),
                    scorers={name: banks.protocol(name)["scoring"] for name in s.WAVE1},
                    official_references={"mme": banks.MME_CONVERTER, "seed": banks.SEED_CONVERTER},
                    bound_sources=s.SOURCES, metadata_helper=s.HELPER,
                    score_files=str(batch_folder(scorer, batch) / "cells" / "<cell>" / "SCORE_<protocol>.json"),
                    per_question_correctness_retained=False, mme_points_never_averaged_with_percentages=True)
    banks = s.bound_module(p, "banks4.py", "tool1_bench4_score_banks_cells_v7")
    require(tuple(banks.ORDER) == s.BANKS, "Bound bank protocol and scoring step disagree")
    inventory_ref, inventory = bench4_inputs(s, p, generation_plans)
    names = list(s.KIND_BANKS["local"])
    return dict(owner=s.OWNER, kind="local", banks=names, inventory_result=inventory_ref,
                labels={name: s.plain(inventory["banks"][name]["labels"]) for name in names},
                questions={name: s.plain(inventory["banks"][name]["questions"]) for name in names},
                scorers={name: banks.protocol(name)["scoring"] for name in names}, bound_sources=s.SOURCES,
                reference_origins={k: s.REFERENCE_ORIGINS[k] for k in s.REFERENCE_ORIGINS}, metadata_helper=s.HELPER,
                mmmu_answer_dict=s.answer_dict_ref(inventory),
                score_files=str(batch_folder(scorer, batch) / "cells" / "<cell>" / "SCORE_<protocol>.json"))


def make_plan(scorer, s, p, batch, route, acks):
    config = check_scorer(scorer)
    routes = s.bound_module(p, "routes.py", "tool1_score_routes_cells_v7_" + scorer)
    registered = cells5(p)
    require(isinstance(acks, dict) and 1 <= len(acks) <= MAX_CELLS and set(acks) <= set(registered)
            and all(s.exact_ref(ref) for ref in acks.values()), "Registered CELLS_V7 generation ACKs required")
    jobs, plans = {}, {}
    for cell, ack_ref in sorted(acks.items()):
        ack = json.loads(p.checked(ack_ref))
        plan = json.loads(p.checked(ack["plan"]))
        require(ack.get("owner") == config["generation_owner"] and ack.get("cell") == cell
                and plan.get("schema") == config["generation_schema"]
                and plan.get("sources", {}).get(config["prepare_name"]) == config["prepare"],
                "Bound ACK is not this cell's CELLS_V7 generation")
        jobs[cell], plans[cell] = ack["job_id"], plan
    function = dict(
        scorer_function(scorer, s, p, batch, plans), scorer=scorer, batch=batch, generation_acks=acks,
        generation_jobs=jobs, generation_controller=config["prepare"], cells_v7=CELLS5, custody7=CUSTODY5,
        scoring_step=config["step"],
        late_binding="worker reads the single complete ATTEMPT_RESULT-<job>.json of exactly each bound ACK after every "
                     "afterok generation job COMPLETED (the step's own generation binding and validation)",
        labels_decoded_only_inside_scorer=True, model_forwards=0, training_updates=0,
        used_for_selection_or_tuning=False)
    dependency = "afterok:" + ":".join(jobs[cell] for cell in sorted(jobs))
    return dict(
        schema=PLAN_SCHEMA, owner=s.OWNER, scorer=scorer, batch=batch, route=s.check_route(route),
        function=function, function_identity=hashlib.sha256(p.canonical(function)).hexdigest(),
        sources=dict({"score7.py": p.descriptor(SOURCE / "score7.py"),
                      "score7_run.sbatch": p.descriptor(SOURCE / "score7_run.sbatch"),
                      "score_step": config["step"], config["prepare_name"]: config["prepare"], "CELLS_V7.json": CELLS5,
                      "custody7.py": CUSTODY5}, **s.SOURCES),
        resource=dict(routes.ROUTES[routes.check_route(route)], gres=routes.GRES, GPUs=1, CPUs=int(CPUS),
                      memory=MEMORY, minutes=MINUTES, time=time_text(MINUTES), exclude=routes.EXCLUDED),
        expected_readback=routes.expected_route_fields(route), dependency=dependency,
        automatic_requeue=False, independently_reviewed=False)


def make_plan_file(scorer, s, p, batch, route, cells):
    check_batch(batch)
    s.check_route(route)
    require(isinstance(cells, list) and cells and len(set(cells)) == len(cells) and len(cells) <= MAX_CELLS,
            "Explicit unique CELLS_V7 cells required")
    require(not set(cells) & bound_elsewhere(scorer, s, p, batch), "A cell is already bound by another scoring batch")
    if scorer == "vizwiz":
        require(not any((s.CONTROL / "cells" / cell / s.SCORE_NAME).exists() for cell in cells),
                "A cell already has a registered VizWiz SCORE")
        s.reviewed_components_present(p)
    acks = {cell: generation_ack(scorer, s, p, cell)[0] for cell in sorted(cells)}
    plan = make_plan(scorer, s, p, batch, route, acks)
    folder = batch_folder(scorer, batch)
    s.private_directory(check_scorer(scorer)["control"])
    s.private_directory(folder)
    path = folder / ("PLAN-%s.json" % route)
    require(not path.exists(), "Existing scoring plan is immutable; use a new batch")
    for other in s.ROUTES:
        sibling = folder / ("PLAN-%s.json" % other)
        if other != route and sibling.exists():
            require(json.loads(p.checked(p.descriptor(sibling)))["function"] == plan["function"],
                    "Both routes of one batch must bind the identical scoring function")
    return p.seal(path, plan)


def owner_review(scorer, s, p, batch, route, notes):
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete self-review notes required")
    folder = batch_folder(scorer, batch)
    plan_ref = p.descriptor(folder / ("PLAN-%s.json" % s.check_route(route)))
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(scorer, s, p, batch, route, plan["function"]["generation_acks"]),
            "Self-review must bind the exact plan")
    return p.seal(folder / ("OWNER_REVIEW-%s.json" % route), dict(
        status=REVIEW_STATUS, independent=False, ai_review_disclosed=True, reviewer="Tool1-owner-cells-v7-scoring",
        scorer=scorer, batch=batch, route=route, plan=plan_ref, sources=plan["sources"],
        function_identity=plan["function_identity"], blocking_findings=[], notes=notes))


def validate_independent(scorer, review, plan_ref, plan, batch, route):
    require(review.get("status") == INDEPENDENT_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", "Tool1-owner-cells-v7-scoring")
            and review.get("blocking_findings") == [] and review.get("sources") == plan["sources"]
            and review.get("route") == route == plan["route"] and review.get("scorer") == scorer
            and review.get("batch") == batch and review.get("plan") == plan_ref,
            "Independent exact source/config PASS bound to this afterok scoring plan is required")


def contract(scorer, s, p, batch, route, independent=False):
    folder = batch_folder(scorer, batch)
    info = os.stat(folder, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Private owned scoring control directory required")
    plan_ref = p.descriptor(folder / ("PLAN-%s.json" % s.check_route(route)))
    plan = json.loads(p.checked(plan_ref))
    require(plan.get("scorer") == scorer
            and plan == make_plan(scorer, s, p, batch, route, plan["function"]["generation_acks"]),
            "Exact source/config/route-bound afterok scoring plan required")
    review_ref = p.descriptor(folder / ("OWNER_REVIEW-%s.json" % route))
    review = json.loads(p.checked(review_ref))
    require(review["status"] == REVIEW_STATUS and review["independent"] is False and review["plan"] == plan_ref
            and review["sources"] == plan["sources"] and review["blocking_findings"] == []
            and review["scorer"] == scorer and review["batch"] == batch and review["route"] == route,
            "Truthful exact private-owner review required; no independent PASS claimed")
    independent_ref = None
    if independent:
        independent_ref = p.descriptor(folder / ("INDEPENDENT_REVIEW-%s.json" % route))
        validate_independent(scorer, json.loads(p.checked(independent_ref)), plan_ref, plan, batch, route)
    return folder, plan_ref, plan, review_ref, independent_ref


def argv(scorer, batch, attempt, plan):
    resource = plan["resource"]
    return ["/usr/bin/sbatch", "--parsable", "--hold", "--no-requeue", "--export=NIL",
            "--account=" + resource["account"], "--partition=" + resource["partition"], "--qos=" + resource["qos"],
            "--gres=gpu:L40S:1", "--nodes=1", "--ntasks=1", "--cpus-per-task=" + CPUS, "--mem=" + MEMORY,
            "--time=" + resource["time"], "--exclude=" + resource["exclude"],
            "--job-name=" + job_name(scorer, batch, attempt), "--comment=" + comment(plan, attempt),
            "--chdir=" + str(HOME), "--input=/dev/null", "--output=/dev/null", "--error=/dev/null",
            "--dependency=" + plan["dependency"],
            str(SOURCE / "score7_run.sbatch"), scorer, str(batch), str(attempt), plan["route"]]


def dependency_ok(value, plan, running):
    """Slurm prints afterok:<job>(unfulfilled) per remaining job (comma separated); satisfied ones may show
    (fulfilled) or disappear; (null) once all are satisfied. Only bound jobs may appear."""
    if value in ("(null)", plan["dependency"]):
        return True
    bound = set(plan["function"]["generation_jobs"].values())
    items = [DEPENDENCY_ITEM.fullmatch(item) for item in str(value).split(",")]
    return (bool(items) and all(match is not None and match[1] in bound for match in items)
            and not (running and any(match[2] == "unfulfilled" for match in items)))


def scheduler(scorer, p, batch, attempt, plan, job, held=False, running=False):
    raw = p.query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = p.fields(raw)
    expected = dict(
        plan["expected_readback"], JobId=job, UserId="luyangz(2708899)", JobName=job_name(scorer, batch, attempt),
        Comment=comment(plan, attempt), NumCPUs=CPUS, NumTasks="1", TimeLimit=plan["resource"]["time"],
        Command=str(SOURCE / "score7_run.sbatch"), WorkDir=str(HOME),
        StdIn="/dev/null", StdOut="/dev/null", StdErr="/dev/null")
    nodes = row.get("NumNodes") == "1" if running else row.get("NumNodes") in ("1", "1-1")
    require(plan["expected_readback"]["Requeue"] == "0" and all(row.get(k) == v for k, v in expected.items())
            and nodes and dependency_ok(row.get("Dependency"), plan, running)
            and shlex.split(row.get("SubmitLine", "")) == argv(scorer, batch, attempt, plan),
            "Actual physical scoring request/source/resources differ")
    if held:
        require(row.get("JobState") == "PENDING" and row.get("Reason") == "JobHeldUser"
                and row.get("RunTime") == "00:00:00", "Verified never-started held request required")
    if running:
        require(row.get("JobState") == "RUNNING" and row.get("Restarts") == "0"
                and row.get("NodeList") == socket.gethostname().split(".")[0]
                and p.valid_singleton_node(row["NodeList"]), "Actual owned running allocation required")
    tres = dict(value.split("=", 1) for value in row.get("AllocTRES" if running else "ReqTRES", "").split(",")
                if "=" in value)
    require(tres.get("cpu") == CPUS and tres.get("mem") in (MEMORY, MEMORY_MB) and tres.get("gres/gpu") == "1",
            "Actual total allocation/request resources differ")
    spool = subprocess.check_output(["/usr/bin/scontrol", "write", "batch_script", job, "-"],
                                    timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
    require(spool == p.checked(plan["sources"]["score7_run.sbatch"]), "Exact immutable bound spool required")
    return raw, spool


def resolve_result(scorer, s, p, cell):
    """The step's own single-complete-result binding: (result ref, BENCH4 release inputs_result or None)."""
    if scorer == "vizwiz":
        return s.generation_result(p, cell), None
    if scorer == "bench6":
        banks = s.bound_module(p, "banks6.py", "tool1_bench6_score_banks_cells_v7_gate")
        return s.generation_result(p, cell, banks), None
    return s.generation_binding(p, bench4_family(s, p), cell)


def generation_gate(scorer, s, p, plan, final=False):
    """Bound generation jobs: live or COMPLETED (submit/release); all COMPLETED with exactly their own complete
    results (worker)."""
    function = plan["function"]
    require(plan["dependency"] == "afterok:" + ":".join(function["generation_jobs"][c]
                                                        for c in sorted(function["generation_jobs"])),
            "Bound dependency differs from the bound generation jobs")
    states, results = {}, {}
    for cell, ack_ref in sorted(function["generation_acks"].items()):
        ack = json.loads(p.checked(ack_ref))
        job = function["generation_jobs"][cell]
        require(ack["job_id"] == job, "Bound generation identity differs: " + cell)
        require(generation_ack(scorer, s, p, cell)[0] == ack_ref, "A newer generation attempt exists; plan a new batch")
        state, code = s.sacct_state(p, job)
        require(state in (("COMPLETED",) if final else GENERATION_LIVE),
                "Generation job state does not permit scoring: %s %s" % (cell, state))
        states[cell] = dict(generation_job=job, state=state, exit_code=code)
        if state == "COMPLETED":
            result, inputs = resolve_result(scorer, s, p, cell)
            require(result["path"] == str(Path(ack_ref["path"]).parent / ("ATTEMPT_RESULT-%s.json" % job))
                    and json.loads(p.checked(result)).get("job_id") == job,
                    "The single complete generation result must be the bound job's own: " + cell)
            if scorer == "bench4":
                require(inputs == function["inventory_result"],
                        "Generation released with another BENCH4 input inventory: " + cell)
            results[cell] = result
    return dict(states=states, generation_results=results)


def predecessor_allows(scorer, s, p, folder, attempt):
    require(not any((folder / ("attempt-%d" % index)).exists() for index in range(attempt + 1, MAX_ATTEMPTS)),
            "Later scoring attempt folders already exist")
    if attempt == 0:
        return None
    previous = folder / ("attempt-%d" % (attempt - 1))
    ack = json.loads(p.checked(p.descriptor(previous / "ACK.json")))
    state, code = s.sacct_state(p, ack["job_id"])
    require(state in s.TERMINAL, "Predecessor scoring attempt still active")
    require(not (previous / ("RESULT-" + ack["job_id"] + ".json")).exists(), "Batch already scored")
    require(state in s.INTERRUPTED, "A failed scoring attempt needs diagnosis, not a blind successor")
    return dict(job_id=ack["job_id"], route=ack.get("route"), state=state, exit_code=code)


def admit(scorer, s, p, batch, attempt, route, release=False):
    check_attempt(attempt)
    routes = s.bound_module(p, "routes.py", "tool1_score_routes_cells_v7_admit_" + scorer)
    folder, plan_ref, plan, review_ref, independent_ref = contract(scorer, s, p, batch, route, independent=release)
    afolder = folder / ("attempt-%d" % attempt)
    with p.admission_lock():
        require(contract(scorer, s, p, batch, route, independent=release)
                == (folder, plan_ref, plan, review_ref, independent_ref),
                "Contract changed while acquiring shared admission lock")
        if not release:
            predecessor = predecessor_allows(scorer, s, p, folder, attempt)
            extra = {}
            if scorer == "vizwiz":
                extra["batch0_acquisition"] = s.acquisition_record(p)
                s.reviewed_components_present(p)
                extra["registry_claim"] = s.registry_claim(p, plan_ref, create=False)[0]
            gate_now = generation_gate(scorer, s, p, plan)
            rows, gate = routes.actual_gate(p.query, route)
            prefix = job_name(scorer, batch, 0)[:-1]
            require(not any(plan["function_identity"] in row[2] or row[3].startswith(prefix) for row in rows),
                    "A live attempt of this scoring batch already exists on some route")
            s.private_directory(afolder)
            require(not any((afolder / x).exists() for x in
                            ("INTENT.json", "ACK.json", "AMBIGUOUS.json", "FUNCTION_CLAIM.json")),
                    "Existing submission or ambiguous ACK: reconcile, never duplicate")
            claim = p.seal(afolder / "FUNCTION_CLAIM.json", dict(
                owner=s.OWNER, scorer=scorer, batch=batch, attempt=attempt, route=route, function=plan["function"],
                function_identity=plan["function_identity"], plan=plan_ref, predecessor=predecessor,
                generation_gate=gate_now, **extra))
            intent = p.seal(afolder / "INTENT.json", dict(
                owner=s.OWNER, scorer=scorer, batch=batch, attempt=attempt, route=route, plan=plan_ref,
                review=review_ref, claim=claim, argv=argv(scorer, batch, attempt, plan), route_gate=gate,
                expanded_requests=len(rows)))
            try:
                response = subprocess.run(argv(scorer, batch, attempt, plan), text=True, capture_output=True,
                                          timeout=45, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as error:
                p.seal(afolder / "AMBIGUOUS.json", dict(intent=intent, error=repr(error)))
                raise
            good = response.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;[\w.-]+)?\n?", response.stdout)
            if good:
                job = response.stdout.strip().split(";")[0]
                ack_ref = p.seal(afolder / "ACK.json", dict(owner=s.OWNER, scorer=scorer, batch=batch,
                                                           attempt=attempt, route=route, job_id=job, intent=intent,
                                                           plan=plan_ref))
            p.seal(afolder / "SUBMISSION_RESULT.json", dict(
                intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
            require(good, "No unambiguous actual ACK; reconcile exact intent before another submission")
            raw, spool = scheduler(scorer, p, batch, attempt, plan, job, held=True)
            p.seal(afolder / "HELD_READBACK.json", dict(
                ACK=ack_ref, job_id=job, scheduler=raw, spool_base64=base64.b64encode(spool).decode()))
            return dict(scorer=scorer, batch=batch, attempt=attempt, route=route, job_id=job, state="HELD_VERIFIED")
        ack_ref = p.descriptor(afolder / "ACK.json")
        ack = json.loads(p.checked(ack_ref))
        require(ack["owner"] == s.OWNER and ack["plan"] == plan_ref and ack["scorer"] == scorer
                and ack["batch"] == batch and ack["attempt"] == attempt and ack["route"] == route,
                "Exact own ACK required")
        require(not (afolder / "RELEASE_INTENT.json").exists(), "Existing release requires reconciliation")
        job = ack["job_id"]
        scheduler(scorer, p, batch, attempt, plan, job, held=True)
        gate = generation_gate(scorer, s, p, plan)
        p.seal(afolder / "RELEASE_GENERATION_GATE.json", dict(ACK=ack_ref, job_id=job, gate=gate))
        intent = p.seal(afolder / "RELEASE_INTENT.json", dict(ACK=ack_ref, job_id=job,
                                                             independent_review=independent_ref))
        response = subprocess.run(["/usr/bin/scontrol", "release", job], text=True, capture_output=True,
                                  timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
        p.seal(afolder / "RELEASE_RESULT.json", dict(
            intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        require(response.returncode == 0, "Release failed or ambiguous; preserve exact identity")
        raw, _ = scheduler(scorer, p, batch, attempt, plan, job)
        row = p.fields(raw)
        require(row.get("JobState") in ("PENDING", "RUNNING", "CONFIGURING")
                and row.get("Reason") not in ("JobHeldUser", "JobHeldAdmin", "DependencyNeverSatisfied"),
                "Released scoring must be eligible, dependency-pending or running")
        p.seal(afolder / "RELEASE_READBACK.json", dict(job_id=job, route=route, scheduler=raw))
        return dict(scorer=scorer, batch=batch, attempt=attempt, route=route, job_id=job,
                    state=row.get("JobState") + ":" + row.get("Reason", ""))


def score_core(scorer, s, p, folder, afolder, job, plan_ref, late):
    """The bound step's unchanged scoring core over the late-bound complete generation results."""
    if scorer == "vizwiz":
        registry_ref, _ = s.registry_claim(p, plan_ref, create=False)
        modules = {name: s.bound_module(p, name, "tool1_vizwiz_score_cells_v7_" + name.split(".")[0])
                   for name in ("score.py", "banks.py", "acquire.py")}
        acquisition, preflight, cells = s.score_batch(
            p, late, job, registry_ref, modules,
            lambda value: p.seal(afolder / ("ACQUISITION-" + job + ".json"), value))
        return dict(status=s.RESULT_STATUS, registry_claim=registry_ref, acquisition_mode=acquisition["mode"],
                    annotation=acquisition["annotation"], acquisition_receipt=acquisition["receipt"],
                    reviewed_preflight=preflight, cells=cells, per_image_accuracy_retained=False)
    s.private_directory(folder / "cells")
    cells = {}
    if scorer == "bench6":
        banks = s.bound_module(p, "banks6.py", "tool1_bench6_score_banks_cells_v7_worker")
        score = s.bound_module(p, "score6.py", "tool1_bench6_score_cells_v7_worker")
        sources = late["labels"]
        labels = dict(mme=s.read_ref(sources["mme_archive"]), mme_ref=sources["mme_archive"],
                      seed=json.loads(s.read_ref(sources["seed_annotation"])), seed_ref=sources["seed_annotation"])
        for cell, result_ref in sorted(late["generation_results"].items()):
            generation = s.validate_generation(json.loads(p.checked(result_ref)), cell, banks)
            s.private_directory(folder / "cells" / cell)
            cells[cell] = s.score_cell(banks, score, cell, generation, labels, folder / "cells" / cell)
        return dict(status=s.RESULT_STATUS, cells=cells,
                    labels={"mme_archive": sources["mme_archive"], "seed_annotation": sources["seed_annotation"]})
    banks = s.bound_module(p, "banks4.py", "tool1_bench4_score_banks_cells_v7_worker")
    score = s.bound_module(p, "score4.py", "tool1_bench4_score_cells_v7_worker")
    answer_dict = json.loads(s.read_ref(late["mmmu_answer_dict"], 1 << 20))
    output_root = bench4_family(s, p)["output_root"]
    cache = {}
    for cell, result_ref in sorted(late["generation_results"].items()):
        generation = s.validate_generation(json.loads(p.checked(result_ref)), cell, output_root)
        s.private_directory(folder / "cells" / cell)
        cells[cell] = s.score_local_cell(banks, score, cell, generation, late, answer_dict, folder / "cells" / cell,
                                         cache)
    return dict(status=s.RESULT_STATUS["local"], cells=cells)


def work(scorer, s, p, batch, attempt, route):
    check_attempt(attempt)
    folder, plan_ref, plan, review_ref, independent_ref = contract(scorer, s, p, batch, route, independent=True)
    afolder = folder / ("attempt-%d" % attempt)
    job = os.environ.get("SLURM_JOB_ID", "")
    ack_ref = p.descriptor(afolder / "ACK.json")
    ack = json.loads(p.checked(ack_ref))
    require(os.getuid() == UID and job.isdecimal() and ack["job_id"] == job and ack["scorer"] == scorer
            and ack["batch"] == batch and ack["attempt"] == attempt and ack["route"] == route
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0" and not os.environ.get("SLURM_ARRAY_JOB_ID"),
            "Actual unrestarted owned scoring allocation required")
    release = json.loads(p.checked(p.descriptor(afolder / "RELEASE_INTENT.json")))
    require(release == dict(ACK=ack_ref, job_id=job, independent_review=independent_ref),
            "Actual independently reviewed release intent required before labels")
    raw, _ = scheduler(scorer, p, batch, attempt, plan, job, running=True)
    node = socket.gethostname().split(".")[0]
    p.validate_process(job, os.getpid(), node, os.environ.get("SLURM_STEP_ID"),
                       p.query(["/usr/bin/scontrol", "listpids", job]),
                       p.fields(p.query(["/usr/bin/scontrol", "show", "step", job + ".batch", "-o"])),
                       (Path("/proc") / str(os.getpid()) / "status").read_text())
    require(s.time_left_seconds(p.query(["/usr/bin/squeue", "-h", "-j", job, "-o", "%L"])) > 900,
            "Too little allocation left for a durable scoring attempt")
    actor = dict(job_id=job, pid=os.getpid(), node=node, plan=plan_ref, scorer=scorer, batch=batch, attempt=attempt,
                 route=route)
    p.seal(afolder / ("ACTOR_BEFORE-" + job + ".json"), actor)
    gate = generation_gate(scorer, s, p, plan, final=True)
    require(sorted(gate["generation_results"]) == sorted(plan["function"]["generation_acks"]),
            "Every bound cell needs its own complete generation result before labels")
    late = dict(plan["function"], generation_results=gate["generation_results"])
    p.seal(afolder / ("LATE_BINDING-" + job + ".json"), dict(plan=plan_ref, gate=gate,
                                                             generation_results=late["generation_results"]))
    body = score_core(scorer, s, p, folder, afolder, job, plan_ref, late)
    result = p.seal(afolder / ("RESULT-" + job + ".json"), dict(
        body, owner=s.OWNER, scorer=scorer, batch=batch, attempt=attempt, route=route, job_id=job, plan=plan_ref,
        scoring_step=check_scorer(scorer)["step"], generation_gate=gate, labels_decoded_only_inside_scorer=True,
        model_forwards=0, training_updates=0, independently_reviewed=False, scheduler=raw))
    scheduler(scorer, p, batch, attempt, plan, job, running=True)
    p.seal(afolder / ("ACTOR_AFTER-" + job + ".json"), actor)
    print(json.dumps(dict(job_id=job, scorer=scorer, batch=batch, result=result), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "self-review", "submit", "release", "worker"))
    parser.add_argument("--scorer", choices=sorted(SCORERS), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--route", choices=("general", "preempt"), required=True)
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    s = load_step(args.scorer)
    p = s.load_helpers()
    if args.command == "worker":
        work(args.scorer, s, p, args.batch, args.attempt, args.route)
    elif args.command == "plan":
        print(json.dumps(make_plan_file(args.scorer, s, p, args.batch, args.route, args.cell), sort_keys=True))
    elif args.command == "self-review":
        print(json.dumps(owner_review(args.scorer, s, p, args.batch, args.route, args.note), sort_keys=True))
    else:
        print(json.dumps(admit(args.scorer, s, p, args.batch, args.attempt, args.route,
                               release=args.command == "release"), sort_keys=True))


if __name__ == "__main__":
    main()
