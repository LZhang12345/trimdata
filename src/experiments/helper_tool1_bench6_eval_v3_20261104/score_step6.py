"""Registered BENCH6 wave-1 scoring step for Tool1: MME perception/cognition and SEED-Bench image.

Runs in this stdlib process inside one owned CPU+GPU allocation (Babel rule; no model is loaded) on a reviewed
route chosen at plan time (--route general|preempt, gated by routes.py on the actual census under
ADMISSION.lock). A batch plan binds, per cell, the exact HOME ATTEMPT_RESULT of the cell's generation attempt
whose status is COMPLETE_ALL_PLANNED_BANKS, so it can only be made after generation finished (no afterok).
Labels are read only here and only from Root's existing shared-cache copies (no acquisition): the MME release
zip (per-image question<TAB>answer files) and SEED-Bench.json, both pinned in banks6.py. Each (cell, bank)
gets a create-once HOME SCORE file with the official totals and sub-scores; MME stays on its point scale.

  score_step6.py plan        --batch B --route R --cell C [--cell C ...]
  score_step6.py self-review --batch B --route R --note "..."
  score_step6.py submit|release --batch B --attempt N --route R
An interrupted attempt (TIMEOUT/PREEMPTED/NODE_FAIL, no RESULT) continues with --attempt N+1; any other
terminal state without a RESULT needs diagnosis. SCORE files are create-once-or-identical.

v3: the v1 scoring step for the CELLS_V3 lineage (bench6-eval-v3 control and output roots, v3 generation
owner, own job names, comment, plan schema and independent status). Any registered CELLS_V3 cell may be
scored once its single complete generation result exists (that result, with prior parameter parity, is the
eligibility evidence). score6.py and banks6.py are the unchanged v1 files; scoring code is unchanged.
"""

import argparse
import base64
import hashlib
import io
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
SOURCE = HOME / "experiments/helper_tool1_bench6_eval_v3_20261104"
BENCH6V1 = HOME / "experiments/helper_tool1_bench6_eval_v1_20261103"
CELLS3 = HOME / "experiments/helper_tool1_cells_v3_20261104"
V1 = HOME / "experiments/helper_tool1_vizwiz_eval_v1_20261102"
EVAL_CONTROL = HOME / "collaboration/helper-tool1-baseline-independent-20261101/bench6-eval-v3"
CONTROL = EVAL_CONTROL / "scoring"
OUTPUT_ROOT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-bench6-eval-v3-20261104")
OWNER = "Tool1-owned-BENCH6-registered-scoring-v3"
GENERATION_OWNER = "Tool1-owned-BENCH6-CELLS-V3-trained-baseline-evaluation-v3"
REVIEW_STATUS = "OWNER_REVIEWED_TOOL1_BENCH6_SCORING_V3_NOT_INDEPENDENT_ACCEPTANCE"
INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_BENCH6_SCORING_V3"
GENERATION_COMPLETE = "COMPLETE_ALL_PLANNED_BANKS"
BANK_COMPLETE = "COMPLETE_FULL_BANK_PREDICTIONS"
RESULT_STATUS = "COMPLETE_BENCH6_WAVE1_OFFICIAL_SCORES"
COMMENT_PREFIX = "dsmm-t1-b6-score-v3:"
UID = 2708899
EXCLUDED = "babel-p9-20"
ROUTES = ("general", "preempt")
MINUTES = 60
MAX_BATCHES = 8
MAX_ATTEMPTS = 4
GENERATION_ATTEMPTS = 8
MAX_EXPORT = 256 << 20
WAVE1 = ("mme-perception", "mme-cognition", "seed-image")
TERMINAL = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
INTERRUPTED = ("TIMEOUT", "PREEMPTED", "NODE_FAIL")
STABLE = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
HELPER = {
    "path": str(HOME / "experiments/helper_method_eval_bank_qual_v1_20261102/prepare.py"),
    "bytes": 51200, "sha256": "f0dc4f6a548eedc45d5ba3d92e0251328d33bb82b274d815f14b0aca25fa7357",
}
SOURCES = {
    "score6.py": {"path": str(BENCH6V1 / "score6.py"), "bytes": 11992, "sha256": "4f42c00a3220aa35d93c7b16e84d79294fa8aca23c3ebed819a13e07b62dbb9b"},
    "banks6.py": {"path": str(BENCH6V1 / "banks6.py"), "bytes": 29018, "sha256": "98d7df85feae6b0097e48ab3c17e8f80414c6dcd70fa03d6fd61e6c2dbbebdf6"},
    "CELLS_V3.json": {"path": str(CELLS3 / "CELLS_V3.json"), "bytes": 52268, "sha256": "e1bcb9c713f3250234d0b10c790006d3b101807d728b59718f6a75963baf5500"},
    "routes.py": {"path": str(V1 / "routes.py"), "bytes": 4797,
                  "sha256": "5067bf831884699edcd0f5d86b5e9c1a3af65def6edf68aab4cb157c3d388709"},
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
            and all(getattr(before, field) == getattr(after, field) for field in STABLE),
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


def check_batch(batch):
    require(type(batch) is int and 0 <= batch < MAX_BATCHES, "Finite scoring batch index required")
    return batch


def check_attempt(attempt):
    require(type(attempt) is int and 0 <= attempt < MAX_ATTEMPTS, "Finite scoring attempt index required")
    return attempt


def check_route(route):
    require(isinstance(route, str) and route in ROUTES, "Reviewed route general or preempt required")
    return route


def batch_folder(batch):
    return CONTROL / ("batch-%d" % check_batch(batch))


def job_name(batch, attempt):
    return "dsmm-t1-b6sc3-b%d-a%d" % (check_batch(batch), check_attempt(attempt))


def comment(plan, attempt):
    return COMMENT_PREFIX + plan["function_identity"] + ":a%d" % attempt


def private_directory(path):
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.stat(path, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Owned private control directory required: " + str(path))


def exact_ref(value):
    return isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"} and type(value["bytes"]) is int \
        and isinstance(value["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None


def read_ref(ref, limit=MAX_EXPORT):
    """Bounded, nonblocking, no-follow read of one pinned file with stable FD identity; bytes hashed once."""
    require(exact_ref(ref) and 0 < ref["bytes"] <= limit, "Bounded exact descriptor required")
    fd = os.open(ref["path"], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size == ref["bytes"], "Pinned file type/size differs")
        chunks, remaining = [], ref["bytes"] + 1
        while remaining:
            block = os.read(fd, min(remaining, 8 << 20))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw, after = b"".join(chunks), os.fstat(fd)
    finally:
        os.close(fd)
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"]
            and all(getattr(before, k) == getattr(after, k) for k in STABLE), "Pinned file differs: " + ref["path"])
    return raw


def validate_generation(result, cell, banks):
    """A complete, label-free, parity-checked BENCH6 wave-1 generation result of exactly this cell."""
    require(isinstance(result, dict) and result.get("status") == GENERATION_COMPLETE and result.get("cell") == cell
            and result.get("planned_banks") == list(WAVE1) and result.get("prior_parameter_parity") is True
            and result.get("labels_opened") is False
            and re.fullmatch(r"[0-9a-f]{64}", str(result.get("parameter_sha256"))) is not None
            and [b.get("protocol") for b in result.get("banks", [])] == list(WAVE1),
            "Complete wave-1 generation result with parameter parity required: " + cell)
    for report in result["banks"]:
        name = report["protocol"]
        value = banks.protocol(name)
        bank = str(OUTPUT_ROOT / name / cell)
        require(report.get("status") == BANK_COMPLETE and report.get("labels_opened") is False
                and report.get("completion") == dict(protocol=name, questions=value["questions"],
                                                     accepted=value["questions"], missing=0, complete=True)
                and report["roster"]["questions"] == value["questions"], "Complete bank report required: " + name)
        for field, file in (("predictions", "PREDICTIONS.jsonl"), ("questions_export", "QUESTIONS.jsonl")):
            require(exact_ref(report.get(field)) and report[field]["path"] == bank + "/" + file,
                    "Exact bank-level export descriptor required: " + name + " " + field)
    return result


def generation_result(p, cell, banks):
    """The single complete generation ATTEMPT_RESULT of this cell, found through its own ACK records."""
    folder = EVAL_CONTROL / cell
    found = []
    for attempt in range(GENERATION_ATTEMPTS):
        ack_path = folder / ("attempt-%d" % attempt) / "ACK.json"
        if not ack_path.exists():
            continue
        ack = json.loads(p.checked(p.descriptor(ack_path)))
        require(ack.get("owner") == GENERATION_OWNER and ack.get("cell") == cell and ack.get("attempt") == attempt
                and re.fullmatch(r"[1-9]\d*", str(ack.get("job_id"))) is not None, "Exact generation ACK required")
        path = ack_path.parent / ("ATTEMPT_RESULT-" + ack["job_id"] + ".json")
        if path.exists():
            ref = p.descriptor(path)
            if json.loads(p.checked(ref)).get("status") == GENERATION_COMPLETE:
                found.append(ref)
    require(len(found) == 1, "Exactly one complete BENCH6 generation result required for " + cell)
    validate_generation(json.loads(p.checked(found[0])), cell, banks)
    return found[0]


def registered_cells(p):
    """Registered CELLS_V3 cells; each still needs its own single complete parity-checked generation result."""
    cells = json.loads(p.checked(SOURCES["CELLS_V3.json"]))
    require(cells.get("schema") == "tool1-baseline-cells-v3-custody-v1" and isinstance(cells.get("cells"), dict),
            "Exact CELLS_V3 roster required")
    return dict(cells["cells"])


def label_sources(banks):
    mme = banks.protocol("mme-perception")["label_source"]
    require(mme == banks.protocol("mme-cognition")["label_source"], "One MME release zip for both splits")
    return dict(mme_archive=mme["archive"], mme_root=mme["root"],
                seed_annotation=banks.protocol("seed-image")["label_source"]["annotation"])


def make_plan(p, batch, route, results):
    routes = bound_module(p, "routes.py", "tool1_bench6_score_routes")
    banks = bound_module(p, "banks6.py", "tool1_bench6_score_banks")
    cells = registered_cells(p)
    require(tuple(banks.WAVE1) == WAVE1, "Bound bank protocol and scoring step disagree on wave 1")
    require(isinstance(results, dict) and results and set(results) <= set(cells)
            and all(exact_ref(ref) for ref in results.values()), "Nonempty exact roster of generation results required")
    function = dict(
        owner=OWNER, batch=batch, banks=list(WAVE1), generation_results=results, labels=label_sources(banks),
        scorers={name: banks.protocol(name)["scoring"] for name in WAVE1},
        official_references={"mme": banks.MME_CONVERTER, "seed": banks.SEED_CONVERTER},
        bound_sources=SOURCES, metadata_helper=HELPER,
        score_files=str(batch_folder(batch) / "cells" / "<cell>" / "SCORE_<protocol>.json"),
        labels_decoded_only_inside_scorer=True, per_question_correctness_retained=False, model_forwards=0,
        training_updates=0, used_for_selection_or_tuning=False, mme_points_never_averaged_with_percentages=True)
    return dict(
        schema="tool1-bench6-registered-scoring-plan-v3", owner=OWNER, batch=batch, route=check_route(route),
        function=function, function_identity=hashlib.sha256(p.canonical(function)).hexdigest(),
        sources=dict({"score_step6.py": p.descriptor(SOURCE / "score_step6.py"),
                      "score_run.sbatch": p.descriptor(SOURCE / "score_run.sbatch")}, **SOURCES),
        resource=dict(routes.ROUTES[routes.check_route(route)], gres=routes.GRES, GPUs=1, CPUs=4, memory="32G",
                      minutes=MINUTES, time=routes.time_text(MINUTES), exclude=routes.EXCLUDED),
        expected_readback=routes.expected_route_fields(route), automatic_requeue=False,
        independently_reviewed=False)


def bound_cells_elsewhere(p, batch):
    taken = set()
    for other in range(MAX_BATCHES):
        for route in ROUTES:
            path = batch_folder(other) / ("PLAN-%s.json" % route)
            if other != batch and path.exists():
                taken |= set(json.loads(p.checked(p.descriptor(path)))["function"]["generation_results"])
    return taken


def make_plan_file(p, batch, route, cells):
    check_batch(batch)
    check_route(route)
    banks = bound_module(p, "banks6.py", "tool1_bench6_score_banks_file")
    roster = registered_cells(p)
    require(isinstance(cells, list) and cells and len(set(cells)) == len(cells) and set(cells) <= set(roster),
            "Explicit unique registered CELLS_V3 cells required")
    require(not set(cells) & bound_cells_elsewhere(p, batch), "A cell is already bound by another scoring batch")
    results = {cell: generation_result(p, cell, banks) for cell in sorted(cells)}
    plan = make_plan(p, batch, route, results)
    private_directory(CONTROL)
    private_directory(batch_folder(batch))
    path = batch_folder(batch) / ("PLAN-%s.json" % route)
    require(not path.exists(), "Existing scoring plan is immutable; use a new batch")
    for other in ROUTES:
        sibling = batch_folder(batch) / ("PLAN-%s.json" % other)
        if other != route and sibling.exists():
            require(json.loads(p.checked(p.descriptor(sibling)))["function"] == plan["function"],
                    "Both routes of one batch must bind the identical scoring function")
    return p.seal(path, plan)


def owner_review(p, batch, route, notes):
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete self-review notes required")
    folder = batch_folder(batch)
    plan_ref = p.descriptor(folder / ("PLAN-%s.json" % check_route(route)))
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, batch, route, plan["function"]["generation_results"]),
            "Self-review must bind the exact plan")
    return p.seal(folder / ("OWNER_REVIEW-%s.json" % route), dict(
        status=REVIEW_STATUS, independent=False, ai_review_disclosed=True, reviewer=OWNER, batch=batch, route=route,
        plan=plan_ref, sources=plan["sources"], function_identity=plan["function_identity"], blocking_findings=[],
        notes=notes))


def validate_independent(review, plan_ref, plan, batch, route):
    require(review.get("status") == INDEPENDENT_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER) and review.get("blocking_findings") == []
            and review.get("sources") == plan["sources"] and review.get("route") == route == plan["route"]
            and review.get("batch") == batch and review.get("plan") == plan_ref,
            "Independent exact source/config PASS bound to this scoring plan is required")


def contract(p, batch, route, independent=False):
    folder = batch_folder(batch)
    info = os.stat(folder, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Private owned scoring control directory required")
    plan_ref = p.descriptor(folder / ("PLAN-%s.json" % check_route(route)))
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, batch, route, plan["function"]["generation_results"]),
            "Exact source/config/route-bound scoring plan required")
    banks = bound_module(p, "banks6.py", "tool1_bench6_score_banks_contract")
    for cell, ref in plan["function"]["generation_results"].items():
        validate_generation(json.loads(p.checked(ref)), cell, banks)
    review_ref = p.descriptor(folder / ("OWNER_REVIEW-%s.json" % route))
    review = json.loads(p.checked(review_ref))
    require(review["status"] == REVIEW_STATUS and review["independent"] is False and review["plan"] == plan_ref
            and review["sources"] == plan["sources"] and review["blocking_findings"] == []
            and review["batch"] == batch and review["route"] == route,
            "Truthful exact private-owner review required; no independent PASS claimed")
    independent_ref = None
    if independent:
        independent_ref = p.descriptor(folder / ("INDEPENDENT_REVIEW-%s.json" % route))
        validate_independent(json.loads(p.checked(independent_ref)), plan_ref, plan, batch, route)
    return folder, plan_ref, plan, review_ref, independent_ref


def argv(batch, attempt, plan):
    resource = plan["resource"]
    return ["/usr/bin/sbatch", "--parsable", "--hold", "--no-requeue", "--export=NIL",
            "--account=" + resource["account"], "--partition=" + resource["partition"], "--qos=" + resource["qos"],
            "--gres=gpu:L40S:1", "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=32G",
            "--time=" + resource["time"], "--exclude=" + EXCLUDED, "--job-name=" + job_name(batch, attempt),
            "--comment=" + comment(plan, attempt), "--chdir=" + str(HOME), "--input=/dev/null",
            "--output=/dev/null", "--error=/dev/null", str(SOURCE / "score_run.sbatch"), str(batch), str(attempt),
            check_route(plan["route"])]


def scheduler(p, batch, attempt, plan, job, held=False, running=False):
    raw = p.query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = p.fields(raw)
    expected = dict(
        plan["expected_readback"], JobId=job, UserId="luyangz(2708899)", JobName=job_name(batch, attempt),
        Comment=comment(plan, attempt), NumCPUs="4", NumTasks="1", TimeLimit=plan["resource"]["time"],
        Command=str(SOURCE / "score_run.sbatch"), WorkDir=str(HOME), Dependency="(null)",
        StdIn="/dev/null", StdOut="/dev/null", StdErr="/dev/null")
    nodes = row.get("NumNodes") == "1" if running else row.get("NumNodes") in ("1", "1-1")
    require(plan["expected_readback"]["Requeue"] == "0" and all(row.get(k) == v for k, v in expected.items())
            and nodes and shlex.split(row.get("SubmitLine", "")) == argv(batch, attempt, plan),
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
    require(tres.get("cpu") == "4" and tres.get("mem") in ("32G", "32768M") and tres.get("gres/gpu") == "1",
            "Actual total allocation/request resources differ")
    spool = subprocess.check_output(["/usr/bin/scontrol", "write", "batch_script", job, "-"],
                                    timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
    require(spool == p.checked(plan["sources"]["score_run.sbatch"]), "Exact immutable bound spool required")
    return raw, spool


def sacct_state(p, job):
    value = p.query(["/usr/bin/sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State,ExitCode"]).strip().split("|")
    require(len(value) == 3 and value[0] == job, "Exact scheduler accounting record required")
    return value[1].split(" ")[0], value[2]


def predecessor_allows(p, folder, attempt):
    require(not any((folder / ("attempt-%d" % index)).exists() for index in range(attempt + 1, MAX_ATTEMPTS)),
            "Later scoring attempt folders already exist")
    if attempt == 0:
        return None
    previous = folder / ("attempt-%d" % (attempt - 1))
    ack = json.loads(p.checked(p.descriptor(previous / "ACK.json")))
    state, code = sacct_state(p, ack["job_id"])
    require(state in TERMINAL, "Predecessor scoring attempt still active")
    require(not (previous / ("RESULT-" + ack["job_id"] + ".json")).exists(), "Batch already scored")
    require(state in INTERRUPTED, "A failed scoring attempt needs diagnosis, not a blind successor")
    return dict(job_id=ack["job_id"], route=ack.get("route"), state=state, exit_code=code)


def admit(p, batch, attempt, route, release=False):
    check_attempt(attempt)
    routes = bound_module(p, "routes.py", "tool1_bench6_score_routes_admit")
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, batch, route, independent=release)
    afolder = folder / ("attempt-%d" % attempt)
    with p.admission_lock():
        require(contract(p, batch, route, independent=release) == (folder, plan_ref, plan, review_ref, independent_ref),
                "Contract changed while acquiring shared admission lock")
        if not release:
            predecessor = predecessor_allows(p, folder, attempt)
            rows, gate = routes.actual_gate(p.query, route)
            prefix = job_name(batch, 0)[:-1]
            require(not any(plan["function_identity"] in row[2] or row[3].startswith(prefix) for row in rows),
                    "A live attempt of this scoring batch already exists on some route")
            private_directory(afolder)
            require(not any((afolder / x).exists() for x in
                            ("INTENT.json", "ACK.json", "AMBIGUOUS.json", "FUNCTION_CLAIM.json")),
                    "Existing submission or ambiguous ACK: reconcile, never duplicate")
            claim = p.seal(afolder / "FUNCTION_CLAIM.json", dict(
                owner=OWNER, batch=batch, attempt=attempt, route=route, function=plan["function"],
                function_identity=plan["function_identity"], plan=plan_ref, predecessor=predecessor))
            intent = p.seal(afolder / "INTENT.json", dict(
                owner=OWNER, batch=batch, attempt=attempt, route=route, plan=plan_ref, review=review_ref, claim=claim,
                argv=argv(batch, attempt, plan), route_gate=gate, expanded_requests=len(rows)))
            try:
                response = subprocess.run(argv(batch, attempt, plan), text=True, capture_output=True, timeout=45,
                                          env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as error:
                p.seal(afolder / "AMBIGUOUS.json", dict(intent=intent, error=repr(error)))
                raise
            good = response.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;[\w.-]+)?\n?", response.stdout)
            if good:
                job = response.stdout.strip().split(";")[0]
                ack_ref = p.seal(afolder / "ACK.json", dict(owner=OWNER, batch=batch, attempt=attempt, route=route,
                                                           job_id=job, intent=intent, plan=plan_ref))
            p.seal(afolder / "SUBMISSION_RESULT.json", dict(
                intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
            require(good, "No unambiguous actual ACK; reconcile exact intent before another submission")
            raw, spool = scheduler(p, batch, attempt, plan, job, held=True)
            p.seal(afolder / "HELD_READBACK.json", dict(
                ACK=ack_ref, job_id=job, scheduler=raw, spool_base64=base64.b64encode(spool).decode()))
            return dict(batch=batch, attempt=attempt, route=route, job_id=job, state="HELD_VERIFIED")
        ack_ref = p.descriptor(afolder / "ACK.json")
        ack = json.loads(p.checked(ack_ref))
        require(ack["owner"] == OWNER and ack["plan"] == plan_ref and ack["batch"] == batch
                and ack["attempt"] == attempt and ack["route"] == route, "Exact own ACK required")
        require(not (afolder / "RELEASE_INTENT.json").exists(), "Existing release requires reconciliation")
        job = ack["job_id"]
        scheduler(p, batch, attempt, plan, job, held=True)
        intent = p.seal(afolder / "RELEASE_INTENT.json", dict(ACK=ack_ref, job_id=job,
                                                             independent_review=independent_ref))
        response = subprocess.run(["/usr/bin/scontrol", "release", job], text=True, capture_output=True,
                                  timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
        p.seal(afolder / "RELEASE_RESULT.json", dict(
            intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        require(response.returncode == 0, "Release failed or ambiguous; preserve exact identity")
        raw, _ = scheduler(p, batch, attempt, plan, job)
        row = p.fields(raw)
        require(row.get("JobState") in ("PENDING", "RUNNING", "CONFIGURING")
                and row.get("Reason") not in ("JobHeldUser", "JobHeldAdmin", "DependencyNeverSatisfied"),
                "Released scoring must be eligible or running")
        p.seal(afolder / "RELEASE_READBACK.json", dict(job_id=job, route=route, scheduler=raw))
        return dict(batch=batch, attempt=attempt, route=route, job_id=job,
                    state=row.get("JobState") + ":" + row.get("Reason", ""))


def publish_once(path, value):
    """Create-once SCORE file (O_EXCL + fsync); an existing file must hold the identical bytes."""
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    except FileExistsError:
        with os.fdopen(os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC), "rb") as stream:
            require(stream.read((4 << 20) + 1) == raw, "Existing SCORE file differs")
    else:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def score_cell(banks, score, cell, generation, labels, folder):
    """Every wave-1 bank of one bound complete generation; returns aggregate records only."""
    records = {}
    for report in generation["banks"]:
        name = report["protocol"]
        value = banks.protocol(name)
        questions = score.parse_jsonl(read_ref(report["questions_export"]), MAX_EXPORT)
        require(banks.key(questions) == report["roster"]["ordered_native_inputs_sha256"]
                and len(questions) == value["questions"], "Exported questions differ from the generated roster")
        predictions = banks.parse_predictions(read_ref(report["predictions"]), [{"native_input": q} for q in questions])
        if value["family"] == "mme":
            summary = score.mme_score(name, value["split"], questions, predictions, io.BytesIO(labels["mme"]))
        else:
            summary = score.seed_score(name, questions, predictions, labels["seed"])
        document = dict(summary, owner=OWNER, cell=cell, parameter_sha256=generation["parameter_sha256"],
                        predictions=report["predictions"], questions_export=report["questions_export"],
                        labels={"mme": labels["mme_ref"], "seed": labels["seed_ref"]}[value["family"]],
                        scientific_claims=False, independently_reviewed=False, held_out_claim=False)
        reference = publish_once(folder / ("SCORE_%s.json" % name), document)
        headline = {k: summary[k] for k in ("total", "max_total", "accuracy", "status") if k in summary}
        records[name] = dict(score=reference, **headline)
    return records


def time_left_seconds(text):
    match = re.fullmatch(r"(?:(\d+)-(\d+):|(\d+):)?(\d+):(\d+)", text.strip())
    require(match is not None, "Actual remaining Slurm time required")
    days, day_hours, hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    require(minutes < 60 and seconds < 60 and day_hours < 24, "Malformed remaining Slurm time")
    return (((days * 24 + day_hours + hours) * 60) + minutes) * 60 + seconds


def work(p, batch, attempt, route):
    check_attempt(attempt)
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, batch, route, independent=True)
    afolder = folder / ("attempt-%d" % attempt)
    job = os.environ.get("SLURM_JOB_ID", "")
    ack_ref = p.descriptor(afolder / "ACK.json")
    ack = json.loads(p.checked(ack_ref))
    require(os.getuid() == UID and job.isdecimal() and ack["job_id"] == job and ack["batch"] == batch
            and ack["attempt"] == attempt and ack["route"] == route
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0" and not os.environ.get("SLURM_ARRAY_JOB_ID"),
            "Actual unrestarted owned scoring allocation required")
    release = json.loads(p.checked(p.descriptor(afolder / "RELEASE_INTENT.json")))
    require(release == dict(ACK=ack_ref, job_id=job, independent_review=independent_ref),
            "Actual independently reviewed release intent required before labels")
    raw, _ = scheduler(p, batch, attempt, plan, job, running=True)
    node = socket.gethostname().split(".")[0]
    p.validate_process(job, os.getpid(), node, os.environ.get("SLURM_STEP_ID"),
                       p.query(["/usr/bin/scontrol", "listpids", job]),
                       p.fields(p.query(["/usr/bin/scontrol", "show", "step", job + ".batch", "-o"])),
                       (Path("/proc") / str(os.getpid()) / "status").read_text())
    require(time_left_seconds(p.query(["/usr/bin/squeue", "-h", "-j", job, "-o", "%L"])) > 600,
            "Too little allocation left for scoring")
    actor = dict(job_id=job, pid=os.getpid(), node=node, plan=plan_ref, batch=batch, attempt=attempt, route=route)
    p.seal(afolder / ("ACTOR_BEFORE-" + job + ".json"), actor)
    banks = bound_module(p, "banks6.py", "tool1_bench6_score_banks_worker")
    score = bound_module(p, "score6.py", "tool1_bench6_score_worker")
    sources = plan["function"]["labels"]
    labels = dict(mme=read_ref(sources["mme_archive"]), mme_ref=sources["mme_archive"],
                  seed=json.loads(read_ref(sources["seed_annotation"])), seed_ref=sources["seed_annotation"])
    private_directory(folder / "cells")
    cells = {}
    for cell, result_ref in sorted(plan["function"]["generation_results"].items()):
        generation = validate_generation(json.loads(p.checked(result_ref)), cell, banks)
        private_directory(folder / "cells" / cell)
        cells[cell] = score_cell(banks, score, cell, generation, labels, folder / "cells" / cell)
    result = dict(status=RESULT_STATUS, owner=OWNER, batch=batch, attempt=attempt, job_id=job, cells=cells,
                  labels={"mme_archive": sources["mme_archive"], "seed_annotation": sources["seed_annotation"]},
                  model_forwards=0, training_updates=0, scheduler=raw)
    reference = p.seal(afolder / ("RESULT-" + job + ".json"), result)
    scheduler(p, batch, attempt, plan, job, running=True)
    p.seal(afolder / ("ACTOR_AFTER-" + job + ".json"), actor)
    print(json.dumps(dict(job_id=job, batch=batch, result=reference), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "self-review", "submit", "release", "worker"))
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    p = load_helpers()
    if args.command == "worker":
        work(p, args.batch, args.attempt, args.route)
    elif args.command == "plan":
        print(json.dumps(make_plan_file(p, args.batch, args.route, args.cell), sort_keys=True))
    elif args.command == "self-review":
        print(json.dumps(owner_review(p, args.batch, args.route, args.note), sort_keys=True))
    else:
        print(json.dumps(admit(p, args.batch, args.attempt, args.route, release=args.command == "release"),
                         sort_keys=True))


if __name__ == "__main__":
    main()
