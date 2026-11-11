import argparse
from contextlib import ExitStack, redirect_stdout
import fcntl
import importlib
import io
import json
import os
from pathlib import Path
import random
import signal
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid
from types import SimpleNamespace

from . import audit, common as c, identity as identity_io, inputs, kernel
from experiments.session6_start_model_qualification_v4_eager_20261024 import model as start, worker as qualifier
from experiments.session6_official_eval_v2_20261024 import common as evaluation, scoring

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def actual(binding, config):
    phase = binding["phase"]
    c.require(phase in c.PHASES and binding["owner"] == c.OWNER, "Owned START phase required")
    job = os.environ.get("SLURM_JOB_ID", "")
    c.require(job.isdecimal(), "Actual allocated job required")
    task = None
    if phase == "predict":
        raw_task = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        c.require(os.environ.get("SLURM_ARRAY_JOB_ID") == binding["job_id"] and raw_task.isdecimal(),
                  "Exact bound finite array required")
        task = int(raw_task)
        c.require(str(task) == raw_task and 0 <= task < c.SHARDS, "Foreign/noncanonical array task")
    else:
        c.require(not os.environ.get("SLURM_ARRAY_JOB_ID") and job == binding["job_id"],
                  "Exact scalar preparation/scoring job required")
    raw = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True, timeout=30)
    value = qualifier.fields(raw)
    route = c.route(phase)
    expected = dict(JobId=job, JobName=c.job_name(phase), JobState="RUNNING",
        Account="158416130568", Partition=route["partition"], QOS="normal", NumCPUs=str(route["cpus"]), NumTasks="1",
        TimeLimit="00:20:00", Requeue="1", WorkDir=str(c.ROOT), Command=str(c.HERE / "run.sbatch"),
        NodeList=socket.gethostname().split(".")[0])
    if route["GPU_GRES"]:
        expected["TresPerNode"] = "gres/" + route["GPU_GRES"]
    c.require(all(value.get(k) == v for k, v in expected.items())
              and value.get("NumNodes") in ("1", "1-1")
              and value.get("UserId", "").split("(")[0] == "u.lz341613",
              "Actual owned phase route/resources differ")
    if task is not None:
        c.require(value.get("ArrayJobId") == binding["job_id"] and value.get("ArrayTaskId") == str(task),
                  "Actual scheduler array identity differs")
    tres = dict(item.split("=", 1) for item in value["ReqTRES"].split(",") if "=" in item)
    c.require(tres.get("mem") in (str(route["memory_GiB"]) + "G",
                                str(route["memory_GiB"] * 1024) + "M"), "Actual memory differs")
    allocated = dict(item.split("=", 1) for item in value["AllocTRES"].split(",") if "=" in item)
    if route["GPU_GRES"]:
        c.require(tres.get("gres/gpu") == allocated.get("gres/gpu") == "1",
                  "Actual H100 GPU allocation differs")
    else:
        c.require(not any(key.startswith("gres/gpu") for key in {*tres, *allocated})
                  and value.get("TresPerNode") in (None, "(null)", "N/A"),
                  "Pure CPU scoring must not request or allocate a GPU")
    intent = c.checked(binding["intent"])
    c.require(intent["configuration"] == binding["configuration"] and intent["owner"] == c.OWNER
              and intent["phase"] == phase and value.get("Comment") == c.PREFIX + intent["nonce"],
              "Actual pre-RPC phase intent differs")
    restart = value.get("Restarts", "")
    c.require(restart.isdecimal() and os.environ.get("SLURM_RESTART_COUNT", restart) == restart,
              "Actual immutable restart identity differs")
    c.require(Path(sys.executable).absolute() == Path(config["python"]).absolute(),
              "Actual interpreter differs")
    spool = subprocess.check_output(["scontrol", "write", "batch_script", job, "-"], timeout=30)
    c.require(spool == (c.HERE / "run.sbatch").read_bytes(), "Actual wrapper spool differs")
    return dict(job_id=job, parent_job_id=binding["job_id"], task_id=task,
                restart=int(restart), phase=phase, node=value["NodeList"], scheduler=raw,
                process_nonce=uuid.uuid4().hex, pid=os.getpid())


def model_result(config):
    result = c.checked(config["model_result"])
    c.require(result["status"] == c.MODEL_STATUS
              and result["model_identity_sha256"] == c.key(result["model_identity"])
              and result["training_updates"] == 0 and result["neural_forwards"] == 0
              and result["official_benchmark_predictions"] == 0,
              "Actual qualified untouched model/no-training result required")
    original_config = c.checked(result["configuration"])
    c.model_common.validate_config(original_config)
    context = result["actual"]
    c.require(isinstance(context["job_id"], str) and context["job_id"].isdecimal()
              and type(context["restart"]) is int and context["restart"] >= 0,
              "Actual immutable model producer job/restart required")
    producer_binding = c.checked(c.desc(c.model_common.BASE / "registration/BINDING.json"))
    c.require(producer_binding["owner"] == c.model_common.OWNER
              and producer_binding["job_id"] == context["job_id"]
              and producer_binding["configuration"] == result["configuration"]
              and result["configuration"]["path"] == str(c.model_common.BASE / "control/CONFIG.json")
              and config["model_result"]["path"] == str(c.model_common.BASE / "output/attempts" /
                  (context["job_id"] + "-r" + str(context["restart"])) / "RESULT.json"),
              "Model result does not belong to the exact reviewed registered producer")
    c.model_common.validate_review(c.checked(producer_binding["review"]),
                                   result["configuration"], original_config)
    intent = c.checked(producer_binding["intent"])
    scheduler = qualifier.fields(context["scheduler"])
    c.require(intent["owner"] == c.model_common.OWNER
              and intent["configuration"] == result["configuration"]
              and scheduler.get("JobId") == context["job_id"]
              and scheduler.get("JobName") == original_config["job_name"]
              and scheduler.get("Comment") == c.model_common.PREFIX + intent["nonce"]
              and scheduler.get("Restarts") == str(context["restart"])
              and scheduler.get("NodeList") == context["node"],
              "Original model producer scheduler/intent identity differs")
    c.require(all(config["sources"].get(path) == digest
                  for path, digest in original_config["sources"].items()),
              "Qualified model source bytes changed")
    return result, original_config


def load_model(config, observed_runtime, *, diagnostic_path):
    result, original_config = model_result(config)
    assets = start.verified_assets(original_config["native_assets"], config["cache_root"])
    loaded = start.compose(assets)
    identity, _ = qualifier.identity(loaded, original_config["native_assets"], observed_runtime,
                                     diagnostic_path=diagnostic_path)
    identity = identity_io.persist_and_compare(identity, result["model_identity"],
        diagnostic_path.parent, result["model_identity_sha256"])
    return (*loaded, getattr(loaded[1].config, "max_sequence_length", 2048)), identity, assets


def function(config, metadata, identity):
    source = c.desc(c.LLAVA / "llava/eval/model_vqa_loader.py")
    return dict(schema="session6-native-full-TextVQA-START-prediction-function-v1",
        role="ONE_SHARED_UNTOUCHED_START_NOT_TRAINING_SEED_REPLICATION",
        bank="TextVQA", QA_roster=metadata["full_QA_roster"], media=config["media"],
        model_identity_sha256=c.key(identity),
        model_generation_config=identity["generation_config"],
        evaluation_protocol=config["evaluation_protocol"],
        pipeline_sources_sha256=c.key(config["sources"]), native_loop=source,
        conv_mode="vicuna_v1", temperature=0, do_sample=False, num_beams=1,
        max_new_tokens=128, top_p=None, image_aspect_ratio="pad",
        vocabulary_growth=False, training_updates=0)


def native(config, rows, journal, loaded, media, context, deadline):
    missing = journal.missing()
    recorder = audit.Recorder(missing, loaded, media, config["image_root"], journal.function, context)
    args = SimpleNamespace(temperature=0, top_p=None, num_beams=1, max_new_tokens=128,
        num_chunks=1, chunk_idx=0, conv_mode="vicuna_v1", model_base=None,
        model_path=str(c.BASE / "native-model-label/START-Stage1-native-c121"),
        image_folder=config["image_root"], question_file=str(c.BASE / "native-QA.jsonl"),
        answers_file=str(journal.directory / "native-answers.jsonl"))
    sources = {kernel.MODULES["loader"]: c.desc(c.LLAVA / "llava/eval/model_vqa_loader.py")}
    with recorder.installed():
        return kernel.evaluate("loader", args, rows, journal, loaded, deadline, sources, recorder.take,
                               stop_requested=lambda: STOP_REQUESTED)


def entries(journal, *, present_only=False):
    result = []
    for key in journal.order:
        value = journal.db.execute("SELECT raw,audit FROM answers WHERE k=?", (key,)).fetchone()
        if value is None and present_only:
            continue
        c.require(value is not None, "Missing accepted semantic answer")
        result.append(dict(semantic_key=key, raw=json.loads(value[0]), audit=json.loads(value[1])))
    return result


class PrepareBoundaryJournal(audit.AuditedJournal):
    stop_after_first = False

    def append(self, semantic_id, raw, record):
        super().append(semantic_id, raw, record)
        if self.stop_after_first and len(self.present) == 1:
            raise kernel.AcceptedQABoundary("Durable first real QA; force genuine fresh-process continuation")


def fresh_process_proof(accepted, context):
    c.require(len(accepted) == 2, "Exactly two accepted real preparation QAs required")
    first, second = (entry["audit"]["actual"] for entry in accepted)
    c.require(first["job_id"] == second["job_id"] == context["job_id"]
              and first["phase"] == second["phase"] == "prepare"
              and type(first["restart"]) is int and type(second["restart"]) is int
              and first["restart"] < second["restart"] <= context["restart"]
              and first["process_nonce"] != second["process_nonce"],
              "Actual first accepted QA and missing-only second QA must span fresh same-job processes")
    return dict(status="PASS_ACTUAL_FRESH_PROCESS_MISSING_ONLY_NATIVE_QA",
                first_actual=first, second_actual=second,
                first_semantic_key=accepted[0]["semantic_key"],
                second_semantic_key=accepted[1]["semantic_key"],
                first_accepted_entry_sha256=c.key(accepted[0]), accepted_rows_not_replayed=True)


def prepare_bank(config, binding, context, attempt):
    ready_path = Path(config["output"]) / "BANK_READY.json"
    if ready_path.is_file():
        c.ready(config, binding["configuration"])
        return dict(status="ALREADY_READY_NO_NEURAL_REPLAY", ready=c.desc(ready_path), remaining=0)
    metadata, roster, media = inputs.load(config)
    provider = c.checked(config["media_provider"])
    c.require(provider["status"] == "PASS_FULL_OFFICIAL_TEXTVQA_ENCODED_RGB_PIXELS"
              and provider["media"] == config["media"] and provider["image_root"] == config["image_root"]
              and provider["images"] == c.IMAGES and provider["encoded_bytes"] == c.ENCODED_BYTES
              and provider["physical_verification"]["encoded_and_RGB_verified"] is True,
              "Reuse the genuine completed full-pixel provider; no metadata-as-pixel upgrade")
    c.checked(config["evaluation_protocol"])
    proof = provider["physical_verification"]
    observed = qualifier.runtime()
    c.seal(attempt / "ACTUAL_IMPORTED_RUNTIME.json", observed)
    loaded, identity, _ = load_model(config, observed,
        diagnostic_path=attempt / "BEFORE_NATIVE_PREPARE_MODEL_IDENTITY_GUARD.json")
    bound_function = function(config, metadata, identity)
    prefix = roster["records"][:2]
    directory = Path(config["output"]) / "prepare-qualification"
    first = PrepareBoundaryJournal(directory / "accepted", bound_function, prefix)
    try:
        before = entries(first, present_only=True)
        before_first = c.key(before[0]) if before else None
        first.stop_after_first = not before
        generated = native(config, prefix, first, loaded, media, context, float("inf"))
        accepted_entries = entries(first, present_only=True)
        if before_first:
            c.require(c.key(accepted_entries[0]) == before_first,
                      "Previously accepted real QA changed across fresh-process resume")
        if first.missing():
            c.require(len(accepted_entries) == 1, "Unexpected native preparation boundary")
            progress = c.seal(attempt / "ACCEPTED_QA_BEFORE_REQUEUE.json",
                dict(actual=context, configuration=binding["configuration"], function=bound_function,
                     accepted_entry=accepted_entries[0], missing_semantic_keys=[r["semantic_key"] for r in first.missing()]))
            return dict(status="REAL_QA_ACCEPTED_FRESH_PROCESS_REQUEUE_REQUIRED", remaining=1,
                        progress=progress, new_neural_answers=generated["new_answers"])
    finally:
        first.close()
    fresh = fresh_process_proof(accepted_entries, context)
    c.seal(attempt / "FRESH_PROCESS_MISSING_ONLY_PROOF.json", fresh)
    repeat = audit.AuditedJournal(directory / "repeat", bound_function, prefix)
    try:
        repeated_run = native(config, prefix, repeat, loaded, media, context, float("inf"))
        if repeat.missing():
            return dict(status="REPEAT_QA_BOUNDARY_REQUEUE_REQUIRED", remaining=len(repeat.missing()),
                        new_neural_answers=generated["new_answers"] + repeated_run["new_answers"])
        repeated_entries = entries(repeat)
    finally:
        repeat.close()
    for first_entry, repeated in zip(accepted_entries, repeated_entries):
        c.require(first_entry["semantic_key"] == repeated["semantic_key"]
                  and first_entry["audit"]["native_generation"] == repeated["audit"]["native_generation"]
                  and {k: v for k, v in first_entry["raw"].items() if k != "answer_id"}
                  == {k: v for k, v in repeated["raw"].items() if k != "answer_id"},
                  "Actual native prompt/pixel/token/output repeat changed")
    accepted = c.seal(attempt / "QUALIFIED_PREFIX.json",
                      dict(function=bound_function, entries=accepted_entries, scope="2QA_NOT_FULL_BANK"))
    ready = dict(status=c.READY_STATUS, actual=context,
        configuration=binding["configuration"], producer_binding=c.desc(c.registration("prepare") / "BINDING.json"),
        sources=config["sources"], questions=c.QUESTIONS, images=c.IMAGES,
        physical_media=proof, media_provider=config["media_provider"],
        model_result=config["model_result"], function=bound_function, qualified_prefix=accepted,
        native_qualification_forwards=4, distinct_accepted_prefix_QA=2,
        actual_fresh_process_missing_only=fresh,
        actual_full_bank_predictions=2, full_bank_complete=False, official_scores=None,
        training_updates=0, scientific_round_complete=False)
    return dict(status="ACTUAL_BANK_NATIVE_GENERATION_READY",
                ready=c.seal(ready_path, ready), remaining=0,
                new_neural_answers=generated["new_answers"] + repeated_run["new_answers"])


def seed_prefix(journal, ready):
    prefix = c.checked(ready["qualified_prefix"])
    c.require(prefix["function"] == journal.function and len(prefix["entries"]) == 2,
              "Foreign qualified prefix function/count")
    reused = 0
    for entry in prefix["entries"]:
        key = entry["semantic_key"]
        if key in journal.roster and key not in journal.present:
            journal.append(key, entry["raw"], entry["audit"])
            reused += 1
    return reused


def predict(config, binding, context, attempt, began):
    ready = c.ready(config, binding["configuration"])
    metadata, roster, media = inputs.load(config)
    rows = c.partition(roster["records"], context["task_id"])
    directory = Path(config["output"]) / "shards" / ("task-" + str(context["task_id"]).zfill(2))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "WORKER.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = audit.AuditedJournal(directory, ready["function"], rows)
        try:
            reused = seed_prefix(journal, ready)
            if not journal.missing():
                return dict(status="COMPLETE_NO_NEURAL_REPLAY", reused_prefix_QA=reused,
                            new_neural_answers=0, remaining=0, answers=journal.export(directory / "answers.jsonl"))
            observed = qualifier.runtime()
            c.seal(attempt / "ACTUAL_IMPORTED_RUNTIME.json", observed)
            loaded, identity, _ = load_model(config, observed,
                diagnostic_path=attempt / "BEFORE_NATIVE_PREDICT_MODEL_IDENTITY_GUARD.json")
            c.require(function(config, metadata, identity) == journal.function, "Changed actual prediction function")
            before = len(journal.present)
            native(config, rows, journal, loaded, media, context, began + config["attempt_seconds"])
            remaining = len(journal.missing())
            result = dict(status="COMPLETE_SHARD" if not remaining else "ACCEPTED_QA_BOUNDARY_REQUEUE_REQUIRED",
                reused_prefix_QA=reused, new_neural_answers=len(journal.present) - before,
                total_accepted=len(journal.present), expected_shard_questions=len(rows), remaining=remaining)
            if not remaining:
                result["answers"] = journal.export(directory / "answers.jsonl")
            return result
        finally:
            journal.close()


def image_cluster_interval(records, credits, repeats=1000, seed=67124):
    c.require(len(records) == len(credits) > 0 and repeats >= 2, "Complete credits/cluster units required")
    groups = {}
    for row, credit in zip(records, credits):
        group = groups.setdefault(row["image_reference"], [0.0, 0])
        group[0] += credit
        group[1] += 1
    units = [groups[key] for key in sorted(groups)]
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        sampled = [units[rng.randrange(len(units))] for _ in units]
        values.append(100 * sum(row[0] for row in sampled) / sum(row[1] for row in sampled))
    values.sort()
    return dict(method="single_START_image_cluster_percentile_not_paired_student_or_training_seeds",
                repeats=repeats, seed=seed, clusters=len(units),
                lower=values[int(.025 * (repeats - 1))], upper=values[int(.975 * (repeats - 1))])


def answer_producer(entry, task, prepare_binding, predict_binding, intents, prefix):
    context = entry["audit"]["actual"]
    phase = context["phase"]
    c.require(phase in ("prepare", "predict") and type(context["restart"]) is int
              and context["restart"] >= 0 and isinstance(context["job_id"], str)
              and context["job_id"].isdecimal(), "Actual answer producer identity required")
    binding = prepare_binding if phase == "prepare" else predict_binding
    c.require(context["parent_job_id"] == binding["job_id"], "Foreign answer-producing parent job")
    if phase == "prepare":
        c.require(context["task_id"] is None and context["job_id"] == binding["job_id"]
                  and prefix.get(entry["semantic_key"]) == entry,
                  "Only the exact two qualified preparation answers may be reused")
    else:
        c.require(type(context["task_id"]) is int and context["task_id"] == task,
                  "Answer belongs to a different finite shard")
    scheduler = qualifier.fields(context["scheduler"])
    c.require(scheduler.get("JobId") == context["job_id"]
              and scheduler.get("JobName") == c.job_name(phase)
              and scheduler.get("Comment") == c.PREFIX + intents[phase]["nonce"]
              and scheduler.get("Restarts") == str(context["restart"])
              and scheduler.get("NodeList") == context["node"]
              and scheduler.get("Requeue") == "1",
              "Answer producer scheduler/intent/restart differs")
    if phase == "predict":
        c.require(scheduler.get("ArrayJobId") == binding["job_id"]
                  and scheduler.get("ArrayTaskId") == str(task), "Foreign actual array answer")


def ground_truth_row(row, entry, join, annotations, prompt_processor):
    c.require(join["semantic_key"] == row["semantic_key"]
              and entry["raw"]["prompt"] == row["native_input"]["text"]
              and type(join["annotation_row"]) is int
              and 0 <= join["annotation_row"] < len(annotations), "Original QA/prompt/GT row differs")
    annotation = annotations[join["annotation_row"]]
    c.require(type(annotation["image_id"]) is type(row["native_id"])
              and annotation["image_id"] == row["native_id"]
              and annotation["question"].lower() == prompt_processor(entry["raw"]["prompt"]),
              "Individual official GT image/question join differs")
    return annotation


def normalized_credits(rows, ordered, joins, credit_inputs, credits, evaluator):
    c.require(len(rows) == len(ordered) == len(joins) == len(credit_inputs) == len(credits),
              "Raw/normalized per-QA report must retain the complete scored denominator")
    records = []
    for row, entry, join, credit_input, credit in zip(rows, ordered, joins, credit_inputs, credits):
        normalized = evaluator.answer_processor(credit_input["prediction"])
        direct_credit = evaluator._compute_answer_scores(credit_input["answers"]).get(normalized, 0.0)
        c.require(direct_credit == credit, "Per-QA official normalized answer/credit differs")
        records.append(dict(semantic_key=row["semantic_key"], native_id=row["native_id"],
            original_row_index=row["original_row_index"], annotation_row=join["annotation_row"],
            image_reference=row["image_reference"], raw_prediction=entry["raw"]["text"],
            official_normalized_prediction=normalized, official_consensus_credit=credit))
    return records


def score(config, binding, context, attempt):
    ready = c.ready(config, binding["configuration"])
    metadata, roster, _ = inputs.load(config)
    c.require(all((Path(config["output"]) / "shards" / ("task-" + str(index).zfill(2)) /
                   "raw_answers.sqlite3").is_file() for index in range(c.SHARDS)),
              "Missing shard; no incomplete full-bank score")
    prepare_binding = c.checked(c.desc(c.registration("prepare") / "BINDING.json"))
    predict_binding = c.checked(c.desc(c.registration("predict") / "BINDING.json"))
    c.require(binding["dependency"] == "afterany:" + predict_binding["job_id"],
              "CPU scoring must be the exact registered continuation of its prediction array")
    intents = {}
    for phase, producer in (("prepare", prepare_binding), ("predict", predict_binding)):
        c.require(producer["owner"] == c.OWNER and producer["phase"] == phase
                  and producer["configuration"] == binding["configuration"], "Foreign answer producer binding")
        c.validate_review(c.checked(producer["review"]), binding["configuration"], config)
        intents[phase] = c.checked(producer["intent"])
        c.require(intents[phase]["owner"] == c.OWNER and intents[phase]["phase"] == phase
                  and intents[phase]["configuration"] == binding["configuration"],
                  "Answer producer durable intent differs")
    qualified = c.checked(ready["qualified_prefix"])
    c.require(qualified["function"] == ready["function"] and len(qualified["entries"]) == 2,
              "Exact qualified prefix required")
    prefix = {entry["semantic_key"]: entry for entry in qualified["entries"]}
    all_entries = {}
    with ExitStack() as stack:
        for index in range(c.SHARDS):
            directory = Path(config["output"]) / "shards" / ("task-" + str(index).zfill(2))
            c.require((directory / "raw_answers.sqlite3").is_file(), "Missing shard; no incomplete full-bank score")
            stream = stack.enter_context((directory / "WORKER.lock").open("a+"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            rows = c.partition(roster["records"], index)
            journal = audit.AuditedJournal(directory, ready["function"], rows)
            try:
                c.require(not journal.missing(), "Incomplete shard; retain missing QA rather than score proxy")
                for row, entry in zip(rows, entries(journal)):
                    audit.validate_entry(entry, row, ready["function"])
                    answer_producer(entry, index, prepare_binding, predict_binding, intents, prefix)
                    key = entry["semantic_key"]
                    c.require(key not in all_entries, "Duplicate semantic answer across shards")
                    all_entries[key] = entry
            finally:
                journal.close()
        c.require(set(all_entries) == {row["semantic_key"] for row in roster["records"]}
                  and len(all_entries) == c.QUESTIONS, "Incomplete full5000QA answer union")
        ordered = [all_entries[row["semantic_key"]] for row in roster["records"]]
    answer_path = attempt / "native_answers.jsonl"
    with answer_path.open("xb") as stream:
        for entry in ordered:
            stream.write(c.canonical(entry["raw"]))
        stream.flush()
        os.fsync(stream.fileno())
    alignment = c.checked(metadata["GT_alignment"])
    annotations = c.checked(config["annotations"])["data"]
    c.require(alignment["questions"] == c.QUESTIONS and len(alignment["records"]) == c.QUESTIONS
              and len(annotations) == c.QUESTIONS, "Complete original official GT projection required")
    sys.path.insert(0, str(c.LLAVA))
    native_scorer = importlib.import_module("llava.eval.eval_textvqa")
    c.require(c.desc(native_scorer.__file__) == c.desc(c.LLAVA / "llava/eval/eval_textvqa.py"),
              "Foreign native TextVQA scorer")
    credits_input = []
    for row, entry, join in zip(roster["records"], ordered, alignment["records"]):
        annotation = ground_truth_row(row, entry, join, annotations, native_scorer.prompt_processor)
        credits_input.append(dict(prediction=entry["raw"]["text"],
                                  answers=annotation["answers"]))
    result = scoring.textvqa(credits_input)
    per_question = normalized_credits(roster["records"], ordered, alignment["records"],
        credits_input, result["per_question_credit"], native_scorer.TextVQAAccuracyEvaluator())
    per_question_reference = c.seal(attempt / "PER_QA_OFFICIAL_SCORES.json", dict(
        status="COMPLETE5000_ORIGINAL_QA_RAW_NORMALIZED_OFFICIAL_CREDIT",
        function=ready["function"], scorer=result["scorer"], questions=len(per_question),
        records=per_question))
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        native_scorer.eval_single(config["annotations"]["path"], str(answer_path))
    c.require("Samples: 5000\n" in stdout.getvalue()
              and ("Accuracy: {:.2f}%".format(result["score"])) in stdout.getvalue(),
              "Direct pinned full-bank scorer and per-QA scoring disagree")
    result.update(status="COMPLETE_OFFICIAL_FULL_TEXTVQA_START_ONLY",
        actual=context, function=ready["function"], raw_answers=c.desc(answer_path),
        raw_and_audit=c.seal(attempt / "semantic_raw_audits.json", ordered),
        per_question_official_scores=per_question_reference,
        native_scorer_stdout=stdout.getvalue(), model_reference="ONE_SHARED_START",
        coverage=dict(expected=c.QUESTIONS, accepted=len(ordered), unique_images=c.IMAGES),
        length_stops=sum(e["audit"]["native_generation"]["stop_reason"] == "LENGTH" for e in ordered),
        uncertainty=image_cluster_interval(roster["records"], result["per_question_credit"]),
        all15={name: ({"score": result["score"], "status": "COMPLETE"} if name == "TextVQA"
                     else {"score": None, "status": "NOT_EVALUATED_IN_THIS_BANK"})
               for name in evaluation.METRICS},
        DERIVED_NATIVE7_LOCAL_TASK_MACRO_V1=None, NATIVE5_MICRO=None,
        primary_complete=False, student_comparison=False, scientific_round_complete=False)
    return result


def run(binding_path, expected_phase):
    began = time.monotonic()
    c.admission_context()
    binding = c.read(binding_path)
    c.require(binding["phase"] == expected_phase, "Wrapper phase differs from registered binding")
    config = c.checked(binding["configuration"])
    c.validate_config(config)
    c.validate_review(c.checked(binding["review"]), binding["configuration"], config)
    context = actual(binding, config)
    if context["phase"] in ("prepare", "predict"):
        signal.signal(signal.SIGUSR1, request_stop)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / "attempts" / (context["job_id"] + "-r" + str(context["restart"]))
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        c.seal(attempt / "START.json", dict(actual=context, configuration=binding["configuration"],
            sources=config["sources"], UTC=c.now(),
            CPU_preparation_inside_GPU_allocation=context["phase"] in ("prepare", "predict")))
        c.require(shutil.disk_usage(output).free >= config["reserved_bytes"], "Insufficient accepted output storage")
        if context["phase"] == "prepare":
            result = prepare_bank(config, binding, context, attempt)
        elif context["phase"] == "predict":
            result = predict(config, binding, context, attempt, began)
        elif context["phase"] == "dispatch":
            from . import admit
            c.ready(config, binding["configuration"])
            result = admit.dispatch_predictions(binding, context)
        else:
            result = score(config, binding, context, attempt)
        result.update(actual=context, elapsed_seconds=time.monotonic() - began, UTC=c.now())
        reference = c.seal(attempt / "RESULT.json", result)
        if result.get("remaining", 0):
            c.require(context["restart"] + 1 < config["max_attempts"],
                      "Finite accepted attempts exhausted with missing QA; preserve and diagnose")
            c.seal(attempt / "REQUEUE_INTENT.json", dict(job_id=context["job_id"],
                restart=context["restart"], result=reference, before_RPC=True, missing_only=True))
            response = subprocess.run(["scontrol", "requeue", context["job_id"]],
                                      capture_output=True, text=True, timeout=45)
            c.seal(attempt / "REQUEUE_RPC.json", dict(rc=response.returncode, stdout=response.stdout,
                                                    stderr=response.stderr))
            c.require(response.returncode == 0, "Same-job requeue failed; no capacity-unsafe new successor")
        return reference
    except BaseException as error:
        c.seal(attempt / "FAILURE.json", dict(actual=context, configuration=binding["configuration"],
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            UTC=c.now(), elapsed_seconds=time.monotonic() - began, scientific_PASS=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--expected-phase", required=True, choices=c.PHASES)
    args = parser.parse_args()
    print(c.canonical(run(args.binding, args.expected_phase)).decode())
