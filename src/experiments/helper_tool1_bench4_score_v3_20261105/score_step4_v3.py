"""Registered BENCH4 LOCAL scoring step v3 for Tool1's CELLS_V3 students (MMMU-val + MMBench-EN dev).

v3 (2026-11-05) is helper_tool1_bench4_eval_v1_20261104/score_step4.py (unchanged, still used for CELLS_BENCH6 cells)
forked to the BENCH4 v3 generation lineage. Author, 2026-11-05: "do whatever you think could be most efficient for
us, focus on the coverage first." Changes only:
  * generation results are bound through the exact BENCH4 v3 controller (helper_tool1_bench4_eval_v3_20261104/
    prepare4.py, pinned by sha256): its CONTROL_ROOT, OWNER, OUTPUT_ROOT, MAX_ATTEMPTS and CELLS_V3 roster
    (custody3.check_cells), with the v1 validation semantics (single COMPLETE_ALL_PLANNED_BANKS ATTEMPT_RESULT,
    parameter parity, labels never opened, complete bank reports, exact bank-level export descriptors);
  * kind `local` only (the MM-Vet/LLaVA-W judge is the separate judge v1.4 step; the inherited judge code below is
    unreachable: check_kind and the sbatch bootstrap refuse `judge`);
  * own versioned SOURCE, CONTROL (bench4-eval-v3/scoring), owner, statuses, comment and job prefix dsmm-t1-b4sc3-.
Scorers (score4.py, banks4.py), references, labels, answer dict, resources and routes are the exact v1 pins.
Text below this paragraph is the v1 description.

Registered BENCH4 scoring step for Tool1: local MMMU/MMBench scoring and the GPT-judge MM-Vet/LLaVA-W step.

Runs in this stdlib process inside one owned CPU+GPU allocation (Babel rule; no model is loaded) on a reviewed route
chosen at plan time (--route general|preempt, gated by routes.py on the actual census under ADMISSION.lock). A batch
plan binds, per cell, the exact HOME ATTEMPT_RESULT of the cell's BENCH4 generation whose status is
COMPLETE_ALL_PLANNED_BANKS, and the one BENCH4 input-inventory RESULT that every bound generation released with, so
it can only be made after generation finished (no afterok). Two kinds:

  local  mmmu-val (official MMMU parse/eval port, score4.py, answer_dict_val.json acquired by the inventory) and
         mmbench-en-dev (local dev CircularEval over the TSV groups + explicit rotations, per ability). Labels are read
         only here, from the inventory's LABELS files and acquisition.
  judge  mm-vet (official MM-Vet grader) and llava-wild (LLaVA c121 eval_gpt_review_bench relative score) through
         judge4.py: PRIMARY gpt-4o-2024-08-06 for every bound cell, optional CALIBRATION gpt-4-0613 on MM-Vet for
         start/uniform-s17 only. The OpenAI key is read in-process only by the judge worker after every check and the
         budget projection passed; one append-only ledger (exclusive flock, one live judge job) caches every paid
         response and stops before the 70 USD line (hard author budget 81.29 USD).

Each (cell, bank[, judge model]) gets a create-once HOME SCORE file (identical bytes on replay).

  score_step4_v3.py plan        --kind local --batch B --route R --cell C [--cell C ...]
  score_step4_v3.py self-review --kind local --batch B --route R --note "..."
  score_step4_v3.py submit|release --kind local --batch B --attempt N --route R
  score_step4_v3.py withdraw    --kind local --batch B --route R --note "..."
An interrupted attempt (TIMEOUT/PREEMPTED/NODE_FAIL) or a judge attempt that stopped at its deadline/budget with a
PARTIAL RESULT continues with --attempt N+1. A FAILED attempt without a RESULT continues only after an owner
attempt-N/DIAGNOSIS-<job>.json and an independent DIAGNOSIS_REVIEW-<job>.json bound to that job and this controller.
`withdraw` retires a batch that has no live job and no complete RESULT (WITHDRAWN.json), so its cells may be bound
by a new batch; published SCORE files and the judge ledger/cache are kept (a rebound judge cell never re-pays).
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
SOURCE = HOME / "experiments/helper_tool1_bench4_score_v3_20261105"
EVAL = HOME / "experiments/helper_tool1_bench4_eval_v1_20261104"
EVAL_V3 = HOME / "experiments/helper_tool1_bench4_eval_v3_20261104"
BENCH6 = HOME / "experiments/helper_tool1_bench6_eval_v1_20261103"
V1 = HOME / "experiments/helper_tool1_vizwiz_eval_v1_20261102"
V3_CONTROL = HOME / "collaboration/helper-tool1-baseline-independent-20261101/bench4-eval-v3"
INPUT_CONTROL = HOME / "collaboration/helper-tool1-baseline-independent-20261101/bench4-inventory-v1"
CONTROL = V3_CONTROL / "scoring"
LEDGER = HOME / "collaboration/helper-tool1-baseline-independent-20261101/bench4-eval-v1/scoring/judge-ledger/JUDGE_LEDGER.jsonl"
INPUT_ROOT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-bench4-inputs-v1-20261104")
OWNER = "Tool1-owned-BENCH4-CELLS-V3-registered-local-scoring-v3"
REVIEW_STATUS = "OWNER_REVIEWED_TOOL1_BENCH4_V3_LOCAL_SCORING_NOT_INDEPENDENT_ACCEPTANCE"
INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_BENCH4_SCORING_V3"
GENERATION_COMPLETE = "COMPLETE_ALL_PLANNED_BANKS"
BANK_COMPLETE = "COMPLETE_FULL_BANK_PREDICTIONS"
INPUT_STATUS = "COMPLETE_BENCH4_INPUT_INVENTORY_V1"
RESULT_STATUS = {"local": "COMPLETE_BENCH4_V3_LOCAL_OFFICIAL_SCORES", "judge": "COMPLETE_BENCH4_JUDGE_SCORES"}
PARTIAL_STATUS = "PARTIAL_BENCH4_JUDGE_SCORES_STOPPED_BEFORE_DEADLINE_OR_BUDGET"
COMMENT_PREFIX = "dsmm-t1-b4-score-v3:"
UID = 2708899
EXCLUDED = "babel-p9-20"
ROUTES = ("general", "preempt")
KINDS = ("local",)
MINUTES = {"local": 60, "judge": 240}
CPUS = {"local": "4", "judge": "2"}
MEMORY = {"local": "32G", "judge": "16G"}
MEMORY_MB = {"local": "32768M", "judge": "16384M"}
JOB_PREFIX = {"local": "dsmm-t1-b4sc3-", "judge": "dsmm-t1-b4j-"}
MAX_BATCHES = 12
MAX_ATTEMPTS = 6
MAX_EXPORT = 256 << 20
BANKS = ("mmmu-val", "llava-wild", "mm-vet", "mmbench-en-dev")
KIND_BANKS = {"local": ("mmmu-val", "mmbench-en-dev"), "judge": ("mm-vet", "llava-wild")}
TERMINAL = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
INTERRUPTED = ("TIMEOUT", "PREEMPTED", "NODE_FAIL")
STABLE = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
RESERVE_SECONDS = 600
DIAGNOSIS_SCHEMA = "tool1-bench4-v3-failed-local-scoring-attempt-diagnosis-v3"
DIAGNOSIS_STATUS = "OWNER_DIAGNOSED_FAILED_V3_LOCAL_SCORING_ATTEMPT"
DIAGNOSIS_REVIEW_STATUS = "PASS_INDEPENDENT_FAILED_SCORING_DIAGNOSIS_TOOL1_BENCH4_V3"
WITHDRAWN_STATUS = "OWNER_WITHDRAWN_BENCH4_V3_LOCAL_SCORING_BATCH_NO_LIVE_JOB_NO_COMPLETE_RESULT"
HELPER = {
    "path": str(HOME / "experiments/helper_method_eval_bank_qual_v1_20261102/prepare.py"),
    "bytes": 51200, "sha256": "f0dc4f6a548eedc45d5ba3d92e0251328d33bb82b274d815f14b0aca25fa7357",
}
REF = EVAL / "reference"
SOURCES = {
    "score4.py": {"path": str(EVAL / "score4.py"), "bytes": 18135, "sha256": "f3b0d38c24cb291a195b6ec0190debad7de7b78b6cfd2d71a505601966eff29f"},
    "judge4.py": {"path": str(EVAL / "judge4.py"), "bytes": 26998, "sha256": "f1c003476ec6a446934a76477eebe5af190367fd281613e4cd934f8c5e200be8"},
    "banks4.py": {"path": str(EVAL / "banks4.py"), "bytes": 30685, "sha256": "8a3b14c9ce0676db1d7bed24f3d56069222d4816933546eb3e9487ae9e4aadbe"},
    "CELLS_BENCH6.json": {"path": str(BENCH6 / "CELLS_BENCH6.json"), "bytes": 84088,
                          "sha256": "1b38da4831510de4db37ef1498f69343768d0ed917ae85892f339a5674c82f85"},
    "bench4_v3_prepare4.py": {"path": str(EVAL_V3 / "prepare4.py"), "bytes": 39588,
                              "sha256": "f03c756601f32c005d9c86e6799fc0e47857e8fd232e390696e5c5184dc2ee32"},
    "routes.py": {"path": str(V1 / "routes.py"), "bytes": 4797,
                  "sha256": "5067bf831884699edcd0f5d86b5e9c1a3af65def6edf68aab4cb157c3d388709"},
    "mm-vet_evaluator.py": {"path": str(REF / "mmvet-acde843e/mm-vet_evaluator.py"), "bytes": 15232,
                            "sha256": "68cfd25dc1c83d9dbf695c86854749083fc3899398ef88ca2d70cea3ddd82cbf"},
    "rule.json": {"path": str(REF / "llava-c121/rule.json"), "bytes": 9098,
                  "sha256": "dd7345a44ce9f7a24e51a30ee030cdab32346f9d08e3359c80545ff1312b1e07"},
    "eval_gpt_review_bench.py": {"path": str(REF / "llava-c121/eval_gpt_review_bench.py"), "bytes": 4172,
                                 "sha256": "fd6a2c3ed99b0b2881e056160c03aab17f7ea5879b9e03fd993381b4d47e63f2"},
    "summarize_gpt_review.py": {"path": str(REF / "llava-c121/summarize_gpt_review.py"), "bytes": 2438,
                                "sha256": "e5d8856867207f3729877d233f74cb12240d5c998650f22b1aec0ba67c21740b"},
    "convert_mmvet_for_eval.py": {"path": str(REF / "llava-c121/convert_mmvet_for_eval.py"), "bytes": 397,
                                  "sha256": "ec52e1f4d5703d460a622b7a9391812af727bb7a32137549674d23f9e979316d"},
    "mmmu_eval_utils.py": {"path": str(REF / "mmmu-268471d0/eval_utils.py"), "bytes": 8992,
                           "sha256": "cc1a89b4a27f697702d340bc6a7578b5237135754ddaa9ef105239a3d9847367"},
    "mmmu_main_eval_only.py": {"path": str(REF / "mmmu-268471d0/main_eval_only.py"), "bytes": 4122,
                               "sha256": "d527738e3beeccb0620d41bdaa94324f89e376a9935b28a48f4975b1815234d6"},
    "vlmevalkit_matching_util.py": {"path": str(REF / "vlmevalkit-a8adc93e/matching_util.py"), "bytes": 5275,
                                    "sha256": "e88638369165759cc40c3f6dad025b58b2d2cf1b6d2f28f80118755f1578a889"},
}
REFERENCE_ORIGINS = {
    "mm-vet_evaluator.py": "https://raw.githubusercontent.com/yuweihao/MM-Vet/"
                           "acde843e02dae30047bb456ee4a8ab28dee7e0ae/mm-vet_evaluator.py",
    "rule.json": "https://raw.githubusercontent.com/haotian-liu/LLaVA/c121f0432da27facab705978f83c4ada465e46fd/"
                 "llava/eval/table/rule.json",
    "eval_gpt_review_bench.py": "LLaVA c121f043 llava/eval/eval_gpt_review_bench.py",
    "summarize_gpt_review.py": "LLaVA c121f043 llava/eval/summarize_gpt_review.py",
    "convert_mmvet_for_eval.py": "LLaVA c121f043 scripts/convert_mmvet_for_eval.py",
    "mmmu_eval_utils.py": "https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/"
                          "268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py",
    "mmmu_main_eval_only.py": "https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/"
                              "268471d0d488258990025331c7528359c324aa25/mmmu/main_eval_only.py",
    "vlmevalkit_matching_util.py": "open-compass/VLMEvalKit@a8adc93e vlmeval/utils/matching_util.py",
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


def check_kind(kind):
    require(isinstance(kind, str) and kind in KINDS, "Scoring kind local or judge required")
    return kind


def check_batch(batch):
    require(type(batch) is int and 0 <= batch < MAX_BATCHES, "Finite scoring batch index required")
    return batch


def check_attempt(attempt):
    require(type(attempt) is int and 0 <= attempt < MAX_ATTEMPTS, "Finite scoring attempt index required")
    return attempt


def check_route(route):
    require(isinstance(route, str) and route in ROUTES, "Reviewed route general or preempt required")
    return route


def batch_folder(kind, batch):
    return CONTROL / ("%s-batch-%d" % (check_kind(kind), check_batch(batch)))


def job_name(kind, batch, attempt):
    return JOB_PREFIX[check_kind(kind)] + "b%d-a%d" % (check_batch(batch), check_attempt(attempt))


def comment(plan, attempt):
    return COMMENT_PREFIX + plan["function_identity"] + ":a%d" % attempt


def time_text(minutes):
    require(type(minutes) is int and 0 < minutes < 24 * 60, "Sub-day walltime required")
    return "%02d:%02d:00" % divmod(minutes, 60)


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


def plain(ref):
    require(isinstance(ref, dict), "Descriptor required")
    return {"path": ref["path"], "bytes": ref["bytes"], "sha256": ref["sha256"]}


def jsonl(raw):
    require(raw.endswith(b"\n"), "Complete newline-terminated JSONL required")
    return [json.loads(line) for line in raw[:-1].split(b"\n")]


# ------------------------------------------------------------------------------------------ bound generation
def v3_family(p):
    """CELLS_V3 students, read only through the exact bound BENCH4 v3 controller's own roots, owner and roster."""
    v3 = bound_module(p, "bench4_v3_prepare4.py", "tool1_bench4_score_v3_controller")
    require(v3.COMPLETE_STATUS == GENERATION_COMPLETE and tuple(v3.BANKS) == BANKS
            and v3.SOURCES["banks4.py"] == SOURCES["banks4.py"] and Path(v3.CONTROL_ROOT) == V3_CONTROL,
            "BENCH4 v3 generation semantics differ")
    return dict(control=Path(v3.CONTROL_ROOT), owner=v3.OWNER, output_root=Path(v3.OUTPUT_ROOT),
                attempts=v3.MAX_ATTEMPTS, cells=sorted(v3.bound_cells(p)["cells"]))


def validate_generation(result, cell, output_root):
    """A complete, label-free, parity-checked BENCH4 generation result of exactly this cell."""
    require(isinstance(result, dict) and result.get("status") == GENERATION_COMPLETE and result.get("cell") == cell
            and result.get("planned_banks") == list(BANKS) and result.get("prior_parameter_parity") is True
            and result.get("labels_opened") is False
            and re.fullmatch(r"[0-9a-f]{64}", str(result.get("parameter_sha256"))) is not None
            and [b.get("protocol") for b in result.get("banks", [])] == list(BANKS),
            "Complete BENCH4 generation result with parameter parity required: " + cell)
    for report in result["banks"]:
        name, completion = report["protocol"], report.get("completion") or {}
        bank = str(Path(output_root) / name / cell)
        require(report.get("status") == BANK_COMPLETE and report.get("labels_opened") is False
                and completion.get("complete") is True and completion.get("missing") == 0
                and completion.get("accepted") == completion.get("items") == report["roster"]["items"],
                "Complete bank report required: " + name)
        for field, file in (("predictions", "PREDICTIONS.jsonl"), ("function", "FUNCTION.json")):
            require(exact_ref(report.get(field)) and report[field]["path"] == bank + "/" + file,
                    "Exact bank-level export descriptor required: " + name + " " + field)
    return result


def generation_binding(p, family, cell):
    """The single complete generation ATTEMPT_RESULT of this cell and the inventory RESULT its release bound."""
    require(cell in family["cells"], "Registered CELLS_V3 cell required: %r" % (cell,))
    folder = family["control"] / cell
    found = []
    for attempt in range(family["attempts"]):
        ack_path = folder / ("attempt-%d" % attempt) / "ACK.json"
        if not ack_path.exists():
            continue
        ack = json.loads(p.checked(p.descriptor(ack_path)))
        require(ack.get("owner") == family["owner"] and ack.get("cell") == cell and ack.get("attempt") == attempt
                and re.fullmatch(r"[1-9]\d*", str(ack.get("job_id"))) is not None, "Exact generation ACK required")
        path = ack_path.parent / ("ATTEMPT_RESULT-" + ack["job_id"] + ".json")
        if path.exists():
            ref = p.descriptor(path)
            if json.loads(p.checked(ref)).get("status") == GENERATION_COMPLETE:
                release = json.loads(p.checked(p.descriptor(ack_path.parent / "RELEASE_INTENT.json")))
                found.append((ref, release["inputs_result"]))
    require(len(found) == 1, "Exactly one complete BENCH4 generation result required for " + cell)
    validate_generation(json.loads(p.checked(found[0][0])), cell, family["output_root"])
    return found[0]


def validate_inventory(result):
    require(result.get("status") == INPUT_STATUS and result.get("worker_returncode") == 0
            and sorted(result.get("banks", {})) == sorted(BANKS)
            and all(result["banks"][name]["status"] == "PASS" for name in BANKS)
            and all(exact_ref(result["banks"][name]["labels"]) for name in BANKS)
            and all(str(result["banks"][name]["labels"]["path"]).startswith(str(INPUT_ROOT) + "/") for name in BANKS)
            and result["acquisition"]["mmmu_answer_dict"]["file"]["status"] == "PASS",
            "Completed BENCH4 input inventory with scoring-only LABELS required")
    return result


def answer_dict_ref(inventory):
    record = inventory["acquisition"]["mmmu_answer_dict"]["file"]
    return {"path": record["path"], "bytes": record["bytes"], "sha256": record["sha256"]}


def make_plan(p, kind, batch, route, results, inventory_ref, calibration):
    routes = bound_module(p, "routes.py", "tool1_bench4_score_routes")
    banks = bound_module(p, "banks4.py", "tool1_bench4_score_banks")
    judge = bound_module(p, "judge4.py", "tool1_bench4_score_judge")
    family = v3_family(p)
    require(tuple(banks.ORDER) == BANKS and tuple(judge.JUDGE_BANKS) == KIND_BANKS["judge"],
            "Bound bank protocol, judge and scoring step disagree")
    require(isinstance(results, dict) and results and set(results) <= set(family["cells"])
            and all(exact_ref(ref) for ref in results.values()), "Nonempty exact roster of generation results required")
    inventory = validate_inventory(json.loads(p.checked(inventory_ref)))
    require(type(calibration) is bool and (kind == "judge" or not calibration), "Calibration is a judge-only option")
    names = list(KIND_BANKS[check_kind(kind)])
    labels = {name: plain(inventory["banks"][name]["labels"]) for name in names}
    questions = {name: plain(inventory["banks"][name]["questions"]) for name in names}
    function = dict(
        owner=OWNER, kind=kind, batch=batch, banks=names, generation_results=results, inventory_result=inventory_ref,
        labels=labels, questions=questions, scorers={name: banks.protocol(name)["scoring"] for name in names},
        bound_sources=SOURCES, reference_origins={k: REFERENCE_ORIGINS[k] for k in REFERENCE_ORIGINS},
        metadata_helper=HELPER, model_forwards=0, training_updates=0, used_for_selection_or_tuning=False,
        score_files=str(batch_folder(kind, batch) / "cells" / "<cell>" / "SCORE_<protocol>[__<judge>].json"),
        labels_decoded_only_inside_scorer=True)
    if kind == "local":
        function["mmmu_answer_dict"] = answer_dict_ref(inventory)
    else:
        prompt = judge.mmvet_prompt(p.checked(SOURCES["mm-vet_evaluator.py"]))
        rules = judge.wild_rules(p.checked(SOURCES["rule.json"]), p.checked(SOURCES["eval_gpt_review_bench.py"]))
        runs = [dict(model=judge.PRIMARY, bank=name, cell=cell, role="PRIMARY")
                for cell in sorted(results) for name in names]
        if calibration:
            chosen = [cell for cell in judge.CALIBRATION_CELLS if cell in results]
            require(chosen, "Calibration needs start and/or uniform-s17 in the batch")
            runs += [dict(model=judge.CALIBRATION, bank="mm-vet", cell=cell, role="CALIBRATION") for cell in chosen]
        function["judge"] = dict(
            runs=runs, primary=judge.PRIMARY, calibration=judge.CALIBRATION, temperature=judge.TEMPERATURE,
            calibration_cells=list(judge.CALIBRATION_CELLS), prices_usd_per_mtok=judge.PRICES_USD_PER_MTOK,
            stop_usd=judge.STOP_USD, hard_budget_usd=judge.HARD_BUDGET_USD, endpoint=judge.ENDPOINT,
            key_file_path_only=judge.KEY_PATH, ledger=str(LEDGER), repeat_policy=judge.REPEAT_POLICY,
            prompt_hashes=judge.prompt_hashes(prompt, rules), mmvet_max_tokens=judge.MMVET_MAX_TOKENS,
            wild_max_tokens=judge.WILD_MAX_TOKENS, official_judges={"mm-vet": "gpt-4-0613 temperature 0",
                                                                    "llava-wild": "gpt-4-0314 temperature 0.2"},
            deviation="Primary judge gpt-4o-2024-08-06 at temperature 0 for every cell (author-approved cost "
                      "decision); gpt-4-0613 calibration on MM-Vet for start/uniform-s17 only.")
    return dict(
        schema="tool1-bench4-registered-scoring-plan-v1", owner=OWNER, kind=kind, batch=batch,
        route=check_route(route), function=function,
        function_identity=hashlib.sha256(p.canonical(function)).hexdigest(),
        sources=dict({"score_step4_v3.py": p.descriptor(SOURCE / "score_step4_v3.py"),
                      "score_run_v3.sbatch": p.descriptor(SOURCE / "score_run_v3.sbatch")}, **SOURCES),
        resource=dict(routes.ROUTES[routes.check_route(route)], gres=routes.GRES, GPUs=1, CPUs=int(CPUS[kind]),
                      memory=MEMORY[kind], minutes=MINUTES[kind], time=routes.time_text(MINUTES[kind]),
                      exclude=routes.EXCLUDED),
        expected_readback=routes.expected_route_fields(route), automatic_requeue=False,
        independently_reviewed=False)


def bound_cells_elsewhere(p, kind, batch):
    taken = set()
    for other in range(MAX_BATCHES):
        for route in ROUTES:
            path = batch_folder(kind, other) / ("PLAN-%s.json" % route)
            if other != batch and path.exists() and not (batch_folder(kind, other) / "WITHDRAWN.json").exists():
                taken |= set(json.loads(p.checked(p.descriptor(path)))["function"]["generation_results"])
    return taken


def make_plan_file(p, kind, batch, route, cells, calibration=False):
    check_kind(kind)
    check_batch(batch)
    check_route(route)
    family = v3_family(p)
    require(isinstance(cells, list) and cells and len(set(cells)) == len(cells) and set(cells) <= set(family["cells"]),
            "Explicit unique registered CELLS_V3 cells required")
    require(not set(cells) & bound_cells_elsewhere(p, kind, batch), "A cell is already bound by another batch")
    bindings = {cell: generation_binding(p, family, cell) for cell in sorted(cells)}
    inventories = {json.dumps(binding[1], sort_keys=True) for binding in bindings.values()}
    require(len(inventories) == 1, "Every bound generation must have released with the same input inventory")
    inventory_ref = next(iter(bindings.values()))[1]
    plan = make_plan(p, kind, batch, route, {cell: b[0] for cell, b in bindings.items()}, inventory_ref, calibration)
    private_directory(CONTROL)
    private_directory(batch_folder(kind, batch))
    path = batch_folder(kind, batch) / ("PLAN-%s.json" % route)
    require(not path.exists(), "Existing scoring plan is immutable; use a new batch")
    for other in ROUTES:
        sibling = batch_folder(kind, batch) / ("PLAN-%s.json" % other)
        if other != route and sibling.exists():
            require(json.loads(p.checked(p.descriptor(sibling)))["function"] == plan["function"],
                    "Both routes of one batch must bind the identical scoring function")
    return p.seal(path, plan)


def replan(p, plan, route):
    function = plan["function"]
    calibration = any(run["role"] == "CALIBRATION" for run in function.get("judge", {}).get("runs", []))
    return make_plan(p, plan["kind"], plan["batch"], route, function["generation_results"],
                     function["inventory_result"], calibration)


def owner_review(p, kind, batch, route, notes):
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete self-review notes required")
    folder = batch_folder(kind, batch)
    plan_ref = p.descriptor(folder / ("PLAN-%s.json" % check_route(route)))
    plan = json.loads(p.checked(plan_ref))
    require(plan == replan(p, plan, route), "Self-review must bind the exact plan")
    return p.seal(folder / ("OWNER_REVIEW-%s.json" % route), dict(
        status=REVIEW_STATUS, independent=False, ai_review_disclosed=True, reviewer=OWNER, kind=kind, batch=batch,
        route=route, plan=plan_ref, sources=plan["sources"], function_identity=plan["function_identity"],
        blocking_findings=[], notes=notes))


def validate_independent(review, plan_ref, plan, kind, batch, route):
    require(review.get("status") == INDEPENDENT_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER) and review.get("blocking_findings") == []
            and review.get("sources") == plan["sources"] and review.get("route") == route == plan["route"]
            and review.get("kind") == kind and review.get("batch") == batch and review.get("plan") == plan_ref,
            "Independent exact source/config PASS bound to this scoring plan is required")


def contract(p, kind, batch, route, independent=False):
    folder = batch_folder(kind, batch)
    info = os.stat(folder, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Private owned scoring control directory required")
    plan_ref = p.descriptor(folder / ("PLAN-%s.json" % check_route(route)))
    plan = json.loads(p.checked(plan_ref))
    require(plan.get("kind") == kind and plan == replan(p, plan, route),
            "Exact source/config/route-bound scoring plan required")
    output_root = v3_family(p)["output_root"]
    for cell, ref in plan["function"]["generation_results"].items():
        validate_generation(json.loads(p.checked(ref)), cell, output_root)
    review_ref = p.descriptor(folder / ("OWNER_REVIEW-%s.json" % route))
    review = json.loads(p.checked(review_ref))
    require(review["status"] == REVIEW_STATUS and review["independent"] is False and review["plan"] == plan_ref
            and review["sources"] == plan["sources"] and review["blocking_findings"] == []
            and review["kind"] == kind and review["batch"] == batch and review["route"] == route,
            "Truthful exact private-owner review required; no independent PASS claimed")
    independent_ref = None
    if independent:
        independent_ref = p.descriptor(folder / ("INDEPENDENT_REVIEW-%s.json" % route))
        validate_independent(json.loads(p.checked(independent_ref)), plan_ref, plan, kind, batch, route)
    return folder, plan_ref, plan, review_ref, independent_ref


def argv(kind, batch, attempt, plan):
    resource = plan["resource"]
    return ["/usr/bin/sbatch", "--parsable", "--hold", "--no-requeue", "--export=NIL",
            "--account=" + resource["account"], "--partition=" + resource["partition"], "--qos=" + resource["qos"],
            "--gres=gpu:L40S:1", "--nodes=1", "--ntasks=1", "--cpus-per-task=" + CPUS[kind],
            "--mem=" + MEMORY[kind], "--time=" + time_text(MINUTES[kind]), "--exclude=" + EXCLUDED,
            "--job-name=" + job_name(kind, batch, attempt), "--comment=" + comment(plan, attempt),
            "--chdir=" + str(HOME), "--input=/dev/null", "--output=/dev/null", "--error=/dev/null",
            str(SOURCE / "score_run_v3.sbatch"), kind, str(batch), str(attempt), check_route(plan["route"])]


def scheduler(p, kind, batch, attempt, plan, job, held=False, running=False):
    raw = p.query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = p.fields(raw)
    expected = dict(
        plan["expected_readback"], JobId=job, UserId="luyangz(2708899)", JobName=job_name(kind, batch, attempt),
        Comment=comment(plan, attempt), NumCPUs=CPUS[kind], NumTasks="1", TimeLimit=time_text(MINUTES[kind]),
        Command=str(SOURCE / "score_run_v3.sbatch"), WorkDir=str(HOME), Dependency="(null)",
        StdIn="/dev/null", StdOut="/dev/null", StdErr="/dev/null")
    nodes = row.get("NumNodes") == "1" if running else row.get("NumNodes") in ("1", "1-1")
    require(plan["expected_readback"]["Requeue"] == "0" and all(row.get(k) == v for k, v in expected.items())
            and nodes and shlex.split(row.get("SubmitLine", "")) == argv(kind, batch, attempt, plan),
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
    require(tres.get("cpu") == CPUS[kind] and tres.get("mem") in (MEMORY[kind], MEMORY_MB[kind])
            and tres.get("gres/gpu") == "1", "Actual total allocation/request resources differ")
    spool = subprocess.check_output(["/usr/bin/scontrol", "write", "batch_script", job, "-"],
                                    timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
    require(spool == p.checked(plan["sources"]["score_run_v3.sbatch"]), "Exact immutable bound spool required")
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
    receipt = previous / ("RESULT-" + ack["job_id"] + ".json")
    diagnosis = None
    if receipt.exists():
        require(json.loads(p.checked(p.descriptor(receipt))).get("status") == PARTIAL_STATUS,
                "Batch already scored")
    elif state == "FAILED":
        diagnosis = diagnosed_failure(p, previous, ack, code)
    else:
        require(state in INTERRUPTED, "A failed scoring attempt needs diagnosis, not a blind successor")
    return dict(job_id=ack["job_id"], route=ack.get("route"), state=state, exit_code=code,
                partial_result=receipt.exists(), diagnosis=diagnosis)


def diagnosed_failure(p, previous, ack, code):
    """Owner DIAGNOSIS + independent DIAGNOSIS_REVIEW of a FAILED predecessor, bound to its job, ACK, scheduler
    state/exit code, a contained hashed worker-log excerpt and this controller."""
    job = ack["job_id"]
    diagnosis_ref = p.descriptor(previous / ("DIAGNOSIS-%s.json" % job))
    diagnosis = json.loads(p.checked(diagnosis_ref))
    evidence, scheduled = diagnosis.get("evidence") or {}, diagnosis.get("scheduler") or {}
    require(diagnosis.get("schema") == DIAGNOSIS_SCHEMA and diagnosis.get("status") == DIAGNOSIS_STATUS
            and diagnosis.get("owner") == OWNER and diagnosis.get("kind") == ack["kind"]
            and diagnosis.get("batch") == ack["batch"] and diagnosis.get("attempt") == ack["attempt"]
            and diagnosis.get("job_id") == job and diagnosis.get("ack") == p.descriptor(previous / "ACK.json")
            and scheduled.get("state") == "FAILED" and scheduled.get("exit_code") == code
            and diagnosis.get("controller") == p.descriptor(SOURCE / "score_step4_v3.py")
            and isinstance(diagnosis.get("cause"), str) and diagnosis["cause"].strip() != ""
            and isinstance(diagnosis.get("repair"), str) and diagnosis["repair"].strip() != ""
            and (evidence.get("worker_log") or {}).get("path") == str(previous / ("worker-" + job + ".log")),
            "Exact owner DIAGNOSIS bound to the FAILED scoring predecessor and this controller required")
    excerpt = evidence.get("log_excerpt")
    require(isinstance(excerpt, str) and excerpt.strip() != "" and excerpt.encode() in p.checked(evidence["worker_log"])
            and evidence.get("log_excerpt_sha256") == hashlib.sha256(excerpt.encode()).hexdigest(),
            "The diagnosis excerpt must be contained in the exact worker log and hashed")
    review_ref = p.descriptor(previous / ("DIAGNOSIS_REVIEW-%s.json" % job))
    review = json.loads(p.checked(review_ref))
    require(review.get("status") == DIAGNOSIS_REVIEW_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER) and review.get("blocking_findings") == []
            and review.get("diagnosis") == diagnosis_ref and review.get("job_id") == job
            and review.get("controller") == p.descriptor(SOURCE / "score_step4_v3.py"),
            "Independent PASS review bound to this exact scoring DIAGNOSIS and this controller required")
    return dict(diagnosis=diagnosis_ref, review=review_ref)


def withdraw(p, kind, batch, route, notes):
    """Retire a batch with no live job and no complete RESULT so that its cells may be planned again."""
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete withdrawal notes required")
    folder = batch_folder(kind, batch)
    plan_ref = p.descriptor(folder / ("PLAN-%s.json" % check_route(route)))
    plan = json.loads(p.checked(plan_ref))
    with p.admission_lock():
        rows = [line.split("|") for line in p.query(["/usr/bin/squeue", "-u", "luyangz", "-r", "-h", "-o",
                                                      "%i|%q|%k|%j|%T|%b"]).splitlines() if line.strip()]
        prefix = job_name(kind, batch, 0)[:-1]
        require(all(len(row) == 6 for row in rows)
                and not any(plan["function_identity"] in row[2] or row[3].startswith(prefix) for row in rows),
                "A live attempt of this batch exists; never withdraw a batch with a live job")
        results = []
        for attempt in range(MAX_ATTEMPTS):
            for path in sorted((folder / ("attempt-%d" % attempt)).glob("RESULT-*.json")):
                status = json.loads(p.checked(p.descriptor(path))).get("status")
                require(status != RESULT_STATUS[kind], "A complete batch is never withdrawn")
                results.append(dict(path=str(path), status=status))
        return p.seal(folder / "WITHDRAWN.json", dict(
            status=WITHDRAWN_STATUS, owner=OWNER, kind=kind, batch=batch, plan=plan_ref,
            function_identity=plan["function_identity"], cells=sorted(plan["function"]["generation_results"]),
            results=results, notes=notes, controller=p.descriptor(SOURCE / "score_step4_v3.py")))


def admit(p, kind, batch, attempt, route, release=False):
    check_attempt(attempt)
    routes = bound_module(p, "routes.py", "tool1_bench4_score_routes_admit")
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, kind, batch, route, independent=release)
    afolder = folder / ("attempt-%d" % attempt)
    require(not (folder / "WITHDRAWN.json").exists(), "A withdrawn batch is never submitted or released")
    with p.admission_lock():
        require(contract(p, kind, batch, route, independent=release)
                == (folder, plan_ref, plan, review_ref, independent_ref),
                "Contract changed while acquiring shared admission lock")
        if not release:
            predecessor = predecessor_allows(p, folder, attempt)
            rows, gate = routes.actual_gate(p.query, route)
            prefix = job_name(kind, batch, 0)[:-1]
            require(not any(plan["function_identity"] in row[2] or row[3].startswith(prefix) for row in rows),
                    "A live attempt of this scoring batch already exists on some route")
            require(kind != "judge" or not any(row[3].startswith(JOB_PREFIX["judge"]) for row in rows),
                    "Only one live judge job at a time (shared ledger and budget)")
            private_directory(afolder)
            require(not any((afolder / x).exists() for x in
                            ("INTENT.json", "ACK.json", "AMBIGUOUS.json", "FUNCTION_CLAIM.json")),
                    "Existing submission or ambiguous ACK: reconcile, never duplicate")
            claim = p.seal(afolder / "FUNCTION_CLAIM.json", dict(
                owner=OWNER, kind=kind, batch=batch, attempt=attempt, route=route, function=plan["function"],
                function_identity=plan["function_identity"], plan=plan_ref, predecessor=predecessor))
            intent = p.seal(afolder / "INTENT.json", dict(
                owner=OWNER, kind=kind, batch=batch, attempt=attempt, route=route, plan=plan_ref, review=review_ref,
                claim=claim, argv=argv(kind, batch, attempt, plan), route_gate=gate, expanded_requests=len(rows)))
            try:
                response = subprocess.run(argv(kind, batch, attempt, plan), text=True, capture_output=True,
                                          timeout=45, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as error:
                p.seal(afolder / "AMBIGUOUS.json", dict(intent=intent, error=repr(error)))
                raise
            good = response.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;[\w.-]+)?\n?", response.stdout)
            if good:
                job = response.stdout.strip().split(";")[0]
                ack_ref = p.seal(afolder / "ACK.json", dict(owner=OWNER, kind=kind, batch=batch, attempt=attempt,
                                                           route=route, job_id=job, intent=intent, plan=plan_ref))
            p.seal(afolder / "SUBMISSION_RESULT.json", dict(
                intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
            require(good, "No unambiguous actual ACK; reconcile exact intent before another submission")
            raw, spool = scheduler(p, kind, batch, attempt, plan, job, held=True)
            p.seal(afolder / "HELD_READBACK.json", dict(
                ACK=ack_ref, job_id=job, scheduler=raw, spool_base64=base64.b64encode(spool).decode()))
            return dict(kind=kind, batch=batch, attempt=attempt, route=route, job_id=job, state="HELD_VERIFIED")
        ack_ref = p.descriptor(afolder / "ACK.json")
        ack = json.loads(p.checked(ack_ref))
        require(ack["owner"] == OWNER and ack["plan"] == plan_ref and ack["kind"] == kind and ack["batch"] == batch
                and ack["attempt"] == attempt and ack["route"] == route, "Exact own ACK required")
        require(not (afolder / "RELEASE_INTENT.json").exists(), "Existing release requires reconciliation")
        job = ack["job_id"]
        scheduler(p, kind, batch, attempt, plan, job, held=True)
        intent = p.seal(afolder / "RELEASE_INTENT.json", dict(ACK=ack_ref, job_id=job,
                                                             independent_review=independent_ref))
        response = subprocess.run(["/usr/bin/scontrol", "release", job], text=True, capture_output=True,
                                  timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
        p.seal(afolder / "RELEASE_RESULT.json", dict(
            intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        require(response.returncode == 0, "Release failed or ambiguous; preserve exact identity")
        raw, _ = scheduler(p, kind, batch, attempt, plan, job)
        row = p.fields(raw)
        require(row.get("JobState") in ("PENDING", "RUNNING", "CONFIGURING")
                and row.get("Reason") not in ("JobHeldUser", "JobHeldAdmin", "DependencyNeverSatisfied"),
                "Released scoring must be eligible or running")
        p.seal(afolder / "RELEASE_READBACK.json", dict(job_id=job, route=route, scheduler=raw))
        return dict(kind=kind, batch=batch, attempt=attempt, route=route, job_id=job,
                    state=row.get("JobState") + ":" + row.get("Reason", ""))


# ------------------------------------------------------------------------------------------------- scoring
def publish_once(path, value):
    """Create-once SCORE file (O_EXCL + fsync); an existing file must hold the identical bytes."""
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    except FileExistsError:
        with os.fdopen(os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC), "rb") as stream:
            require(stream.read((32 << 20) + 1) == raw, "Existing SCORE file differs")
    else:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def bank_inputs(banks, generation, function, name, cache):
    """Exported predictions of one bank aligned with the inventory roster the generation journal was bound to."""
    report = [r for r in generation["banks"] if r["protocol"] == name][0]
    if name not in cache:
        cache[name] = (jsonl(read_ref(function["questions"][name])), jsonl(read_ref(function["labels"][name])))
    rows, labels = cache[name]
    bank_function = json.loads(read_ref(report["function"], 1 << 20))
    require(bank_function.get("questions_sha256") == function["questions"][name]["sha256"]
            and bank_function.get("protocol") == name and bank_function.get("cell") == generation["cell"]
            and bank_function.get("full_roster_sha256") == report["roster"]["ordered_semantic_keys_sha256"],
            "Generation journal was bound to another roster: " + name)
    require(banks.validate_questions(name, rows, expected_items=len(rows)) == report["roster"],
            "Inventory roster differs from the generated roster: " + name)
    predictions = banks.parse_predictions(name, read_ref(report["predictions"]), rows)
    return report, rows, labels, predictions


def score_local_cell(banks, score, cell, generation, function, answer_dict, folder, cache):
    records = {}
    for name in KIND_BANKS["local"]:
        report, rows, labels, predictions = bank_inputs(banks, generation, function, name, cache)
        if name == "mmmu-val":
            summary = score.score_mmmu(rows, predictions, answer_dict)
            summary.pop("parsed_predictions")
            source = function["mmmu_answer_dict"]
        else:
            summary = score.score_mmbench(rows, predictions, labels)
            source = function["labels"][name]
        document = dict(summary, owner=OWNER, cell=cell, protocol=name, parameter_sha256=generation["parameter_sha256"],
                        predictions=report["predictions"], questions=function["questions"][name], labels=source,
                        scorer=banks.protocol(name)["scoring"], scientific_claims=False, independently_reviewed=False,
                        held_out_claim=False)
        reference = publish_once(folder / ("SCORE_%s.json" % name), document)
        records[name] = dict(score=reference, overall=summary["overall"])
    return records


def run_cost(ledger, model, bank, cell):
    total, calls = 0.0, 0
    for call_id, intent in ledger.intents.items():
        if (intent["model"], intent["bank"], intent["cell"]) != (model, bank, cell):
            continue
        outcome = ledger.outcomes.get(call_id)
        total += intent["max_cost_usd"] if outcome is None else outcome.get("cost_usd", outcome.get("charged_usd"))
        calls += 1
    return dict(usd=total, ledger_calls=calls)


def judge_units(judge, prompt, rules, run, inputs):
    labels, predictions = inputs
    if run["bank"] == "mm-vet":
        return [(run["cell"], l["key"], [{"role": "user", "content": judge.mmvet_question(prompt, l, p["text"])}])
                for l, p in zip(labels, predictions)]
    return [(run["cell"], l["question_id"], judge.wild_messages(rules, l, p["text"])) for l, p in zip(labels, predictions)]


def score_judge_run(judge, client, ledger, run, inputs, prompt, rules, hashes, generation, function, folder):
    labels, predictions = inputs
    model, bank, cell = run["model"], run["bank"], run["cell"]
    if bank == "mm-vet":
        summary, scores, evidence = judge.score_mmvet(client, model, cell, prompt, labels, predictions)
    else:
        summary, evidence = judge.score_wild(client, model, cell, rules, labels, predictions)
        scores = None
    for item in evidence:
        item.pop("cached", None)
    document = dict(
        summary, owner=OWNER, cell=cell, protocol=bank, role=run["role"], judge_model_requested=model,
        judge_models_returned=sorted({model}), temperature=judge.TEMPERATURE, repeat_policy=judge.REPEAT_POLICY,
        prompt_hashes=hashes, prices_usd_per_mtok=judge.PRICES_USD_PER_MTOK, run_cost=run_cost(ledger, model, bank, cell),
        parameter_sha256=generation["parameter_sha256"], predictions=[r for r in generation["banks"]
                                                                      if r["protocol"] == bank][0]["predictions"],
        labels=function["labels"][bank], scorer=function["scorers"][bank], official_judge=function["judge"][
            "official_judges"][bank], deviation=function["judge"]["deviation"], scientific_claims=False,
        independently_reviewed=False, held_out_claim=False)
    reference = publish_once(folder / ("SCORE_%s__%s.json" % (bank, model)), document)
    evidence_ref = publish_once(folder / ("JUDGE_EVIDENCE_%s__%s.json" % (bank, model)),
                                dict(cell=cell, protocol=bank, model=model, items=evidence))
    return dict(score=reference, evidence=evidence_ref, overall=summary.get("total", summary.get("overall")),
                cost=document["run_cost"]), scores


def time_left_seconds(text):
    match = re.fullmatch(r"(?:(\d+)-(\d+):|(\d+):)?(\d+):(\d+)", text.strip())
    require(match is not None, "Actual remaining Slurm time required")
    days, day_hours, hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    require(minutes < 60 and seconds < 60 and day_hours < 24, "Malformed remaining Slurm time")
    return (((days * 24 + day_hours + hours) * 60) + minutes) * 60 + seconds


def judge_work(p, plan, folder, afolder, job, deadline, transport_factory=None, key_reader=None):
    """All runs of one judge batch; returns the RESULT body. Key read only after checks and budget projection."""
    banks = bound_module(p, "banks4.py", "tool1_bench4_judge_banks_worker")
    judge = bound_module(p, "judge4.py", "tool1_bench4_judge_worker")
    function = plan["function"]
    prompt = judge.mmvet_prompt(p.checked(SOURCES["mm-vet_evaluator.py"]))
    rules = judge.wild_rules(p.checked(SOURCES["rule.json"]), p.checked(SOURCES["eval_gpt_review_bench.py"]))
    hashes = judge.prompt_hashes(prompt, rules)
    require(hashes == function["judge"]["prompt_hashes"], "Pinned judge prompts differ from the plan")
    cache, inputs, generations = {}, {}, {}
    for cell, ref in sorted(function["generation_results"].items()):
        generations[cell] = validate_generation(json.loads(p.checked(ref)), cell, v3_family(p)["output_root"])
        for name in KIND_BANKS["judge"]:
            _, rows, labels, predictions = bank_inputs(banks, generations[cell], function, name, cache)
            require([l["question_id"] for l in labels] == [r["native_id"] for r in rows], "Judge labels misaligned")
            inputs[(cell, name)] = (labels, predictions)
    private_directory(LEDGER.parent)
    ledger = judge.Ledger(LEDGER, job)
    try:
        spend_before = ledger.spend()
        projection = sum(judge.project(run["model"], run["bank"],
                                       judge_units(judge, prompt, rules, run, inputs[(run["cell"], run["bank"])]),
                                       ledger) for run in function["judge"]["runs"])
        if spend_before + projection > judge.STOP_USD:
            return dict(status="REFUSED_BUDGET_PROJECTION", spend_before_usd=spend_before, projected_usd=projection,
                        stop_usd=judge.STOP_USD, api_key_read=False)
        secret = (key_reader or judge.read_key)()
        transport = (transport_factory or (lambda s: (lambda body: judge.https_transport(s, body))))(secret)
        client = judge.Client(ledger, transport, deadline=deadline)
        private_directory(folder / "cells")
        runs, primary_scores, calibration_scores, stopped = [], {}, {}, None
        for run in function["judge"]["runs"]:
            cell_folder = folder / "cells" / run["cell"]
            private_directory(cell_folder)
            try:
                record, scores = score_judge_run(judge, client, ledger, run, inputs[(run["cell"], run["bank"])],
                                                 prompt, rules, hashes, generations[run["cell"]], function, cell_folder)
            except judge.BudgetStop as error:
                stopped = str(error)
                break
            runs.append(dict(run, **record))
            if run["bank"] == "mm-vet":
                (primary_scores if run["role"] == "PRIMARY" else calibration_scores)[run["cell"]] = scores
        calibrations = {}
        for cell, scores in calibration_scores.items():
            if cell in primary_scores:
                calibrations[cell] = publish_once(folder / "cells" / cell / "CALIBRATION_mm-vet.json", dict(
                    cell=cell, primary=judge.PRIMARY, calibration=judge.CALIBRATION,
                    agreement=judge.calibration_agreement(primary_scores[cell], scores)))
        complete = stopped is None and len(runs) == len(function["judge"]["runs"])
        return dict(status=RESULT_STATUS["judge"] if complete else PARTIAL_STATUS, runs=runs, stop_reason=stopped,
                    calibration=calibrations, spend_before_usd=spend_before, spend_after_usd=ledger.spend(),
                    spend_by_model_usd=ledger.by_model(), projected_usd=projection, paid_calls=client.paid_calls,
                    cached_calls=client.cached_calls, torn_ledger_lines=ledger.torn, api_key_read=True,
                    stop_usd=judge.STOP_USD, hard_budget_usd=judge.HARD_BUDGET_USD)
    finally:
        ledger.close()


def work(p, kind, batch, attempt, route):
    check_attempt(attempt)
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, kind, batch, route, independent=True)
    afolder = folder / ("attempt-%d" % attempt)
    job = os.environ.get("SLURM_JOB_ID", "")
    ack_ref = p.descriptor(afolder / "ACK.json")
    ack = json.loads(p.checked(ack_ref))
    require(os.getuid() == UID and job.isdecimal() and ack["job_id"] == job and ack["kind"] == kind
            and ack["batch"] == batch and ack["attempt"] == attempt and ack["route"] == route
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0" and not os.environ.get("SLURM_ARRAY_JOB_ID"),
            "Actual unrestarted owned scoring allocation required")
    release = json.loads(p.checked(p.descriptor(afolder / "RELEASE_INTENT.json")))
    require(release == dict(ACK=ack_ref, job_id=job, independent_review=independent_ref),
            "Actual independently reviewed release intent required before labels")
    raw, _ = scheduler(p, kind, batch, attempt, plan, job, running=True)
    node = socket.gethostname().split(".")[0]
    p.validate_process(job, os.getpid(), node, os.environ.get("SLURM_STEP_ID"),
                       p.query(["/usr/bin/scontrol", "listpids", job]),
                       p.fields(p.query(["/usr/bin/scontrol", "show", "step", job + ".batch", "-o"])),
                       (Path("/proc") / str(os.getpid()) / "status").read_text())
    left = time_left_seconds(p.query(["/usr/bin/squeue", "-h", "-j", job, "-o", "%L"]))
    require(left > RESERVE_SECONDS + 300, "Too little allocation left for scoring")
    actor = dict(job_id=job, pid=os.getpid(), node=node, plan=plan_ref, kind=kind, batch=batch, attempt=attempt,
                 route=route)
    p.seal(afolder / ("ACTOR_BEFORE-" + job + ".json"), actor)
    function = plan["function"]
    if kind == "local":
        banks = bound_module(p, "banks4.py", "tool1_bench4_score_banks_worker")
        score = bound_module(p, "score4.py", "tool1_bench4_score_worker")
        answer_dict = json.loads(read_ref(function["mmmu_answer_dict"], 1 << 20))
        private_directory(folder / "cells")
        cells, cache = {}, {}
        for cell, result_ref in sorted(function["generation_results"].items()):
            generation = validate_generation(json.loads(p.checked(result_ref)), cell, v3_family(p)["output_root"])
            private_directory(folder / "cells" / cell)
            cells[cell] = score_local_cell(banks, score, cell, generation, function, answer_dict,
                                           folder / "cells" / cell, cache)
        body = dict(status=RESULT_STATUS["local"], cells=cells)
    else:
        body = judge_work(p, plan, folder, afolder, job, deadline=time.time() + left - RESERVE_SECONDS)
    result = dict(body, owner=OWNER, kind=kind, batch=batch, attempt=attempt, job_id=job, model_forwards=0,
                  training_updates=0, scheduler=raw)
    reference = p.seal(afolder / ("RESULT-" + job + ".json"), result)
    scheduler(p, kind, batch, attempt, plan, job, running=True)
    p.seal(afolder / ("ACTOR_AFTER-" + job + ".json"), actor)
    print(json.dumps(dict(job_id=job, kind=kind, batch=batch, status=result["status"], result=reference),
                     sort_keys=True), flush=True)
    require(result["status"] in (RESULT_STATUS[kind], PARTIAL_STATUS), "Scoring did not run: " + result["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "self-review", "submit", "release", "worker", "withdraw"))
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    p = load_helpers()
    if args.command == "worker":
        work(p, args.kind, args.batch, args.attempt, args.route)
    elif args.command == "plan":
        print(json.dumps(make_plan_file(p, args.kind, args.batch, args.route, args.cell, args.calibration),
                         sort_keys=True))
    elif args.command == "self-review":
        print(json.dumps(owner_review(p, args.kind, args.batch, args.route, args.note), sort_keys=True))
    elif args.command == "withdraw":
        print(json.dumps(withdraw(p, args.kind, args.batch, args.route, args.note), sort_keys=True))
    else:
        print(json.dumps(admit(p, args.kind, args.batch, args.attempt, args.route,
                               release=args.command == "release"), sort_keys=True))


if __name__ == "__main__":
    main()
