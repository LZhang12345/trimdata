"""LESS-approximation gradient features (Tool2) v2, Preempt with requeue: one shard of the candidate pool per job.
v2: grad.py gradient checkpointing (v1 OOM on long first turns) and a less-qual unit (first 32 + 16 longest of shard 9).

Derived from record-signal v5: identical seeded candidate pool for image sources (3x Uniform per source, probe holdout
and centroid-probe training sample excluded) plus text_only candidates (3x Uniform), sharded; a separate batch
computes the target (GQA-val, ScienceQA-val dev references) gradients. Child: grad.py (warm-up adapter = Tool2
proportional 10%% cell, LoRA gradients count-sketched to 8192 dims). Selection is a later CPU step.
"""

import argparse
import base64
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
import sys
import time
from types import ModuleType


HOME = Path("/home/luyangz/data_selection_multimodal")
SOURCE = HOME / "experiments/helper_tool2_less_approx_v2_20261105"
V1_SOURCE = HOME / "experiments/helper_adapter_eval_v1_20261102"
CONTROL_ROOT = HOME / "collaboration/helper-tool2-method-20261101/less-approx-v2"
PROBE_ROOT = Path("/data/hf_cache/project_features/DS-MultiModal/helper-tool2-less-approx-v2-20261105")
PROBE_OUTPUT_ROOT = Path("/data/hf_cache/project_checkpoints/DS-MultiModal/helper-tool2-signal-probes-v1-20261102")
PROBE_CONTROLS = {"v1": HOME / "collaboration/helper-tool2-method-20261101/signal-probe-v1",
                  "v2": HOME / "collaboration/helper-tool2-method-20261101/signal-probe-v2",
                  "v3": HOME / "collaboration/helper-tool2-method-20261101/signal-probe-v3",
                  "v4": HOME / "collaboration/helper-tool2-method-20261101/signal-probe-v4"}
SEARCH = ("v4", "v2", "v3", "v1")
PYTHON = "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python"
OWNER = "Tool2-owned-LESS-approx-baseline"
REVIEW_STATUS = "OWNER_REVIEWED_LESS_APPROX_NOT_INDEPENDENT_ACCEPTANCE"
JOB_PREFIX = "dsmm-t2-less2-"
UID = 2708899
HELPER = {
    "path": str(HOME / "experiments/helper_method_policy_selection_qual_v1_20261102/preview.py"),
    "bytes": 52317, "sha256": "d90354f056aae1abc66b9b1619e133540d4e30be86d094608f1d970d703bbe0d",
}
MODEL_SOURCE = {"path": str(SOURCE / "grad.py"), "bytes": 16845,
                "sha256": "c7e4f3382b77b9479483078e0d76213ac7d8ca3fa87964a147480059bbf9ef38"}
WARMUP = {"name": "warmup-prop-sel1234-t29",
          "training_result": str(HOME / "collaboration/helper-tool2-method-20261101/allocation-train-v2/prop-sel1234-train29/TRAINING_RESULT-11022798.json"),
          "training_dir": "/data/hf_cache/project_checkpoints/DS-MultiModal/helper-tool2-final-only-allocation-20261102/allocation-prop-sel1234-train29/training",
          "job": "11022798", "trainer_sha256": "4451b6c3f4a12c6947cbf3bad5fa0030629466a66bde142fca46877ba1159e2f", "policy": "proportional"}
DEV_REFERENCES = (
    {"name": "dev-gqa-val", "path": "/data/hf_cache/datasets/dsmm_dev_reference_v1/references/gqa_val_512.json",
     "bytes": 122162, "sha256": "f764c91e8d62fbbe4f258c36e7de4953343d6002cadde9d9f87f050d42f824b1", "count": 512},
    {"name": "dev-scienceqa-val", "path": "/data/hf_cache/datasets/dsmm_dev_reference_v1/references/scienceqa_val_512.json",
     "bytes": 282213, "sha256": "8f43e96e35a520453719f4938a331fb58c21d6a8afbf80bd2aecafd0374a40f0", "count": 512},
)
REFERENCE_CONFIG = {
    "path": str(HOME / "collaboration/root-general-all-plan-20261028/submission-check-20261029/"
                "baseline-ready-v1/cells/coincide-native10-s17/CONFIG_FINAL.json"),
    "bytes": 67760, "sha256": "450160a8c561fbd33f4e98acca9805cc04ed727955dfb253c58e461cdd0fe9ec",
}
EVAL_CONFIG = {
    "path": str(HOME / "collaboration/claude-baseline-recovery-20261030/coincide-v3-eval/"
                "coincide-native10-s17/CONFIG_12cfec3e9a7b4b028e866356bf8c92c1.json"),
    "bytes": 144004, "sha256": "df65d7d5d82d91c64d68753a6c96f083c0479081d7ee16f38f26052beb3e510e",
}
PROBE_HELPER = {"path": str(HOME / "experiments/helper_signal_probe_v2_20261102/prepare.py"), "bytes": 25455,
                "sha256": "dc70049b508efd11eaa5cdd00783ce2981b219bcf19200554f8188b6cb95d5fd"}
DESIGN = {"path": str(HOME / "experiments/helper_signal_probe_v1_20261102/design.py"), "bytes": 6098,
          "sha256": "0f33893bf6c7ce122efca5d68300af84f479cb7eed763ba01ed8dfc3cc304028"}
SHARDS = 24
IMAGE_GROUPS = ("coco", "gqa", "ocr_vqa", "textvqa", "vg")
UNIFORM_COUNTS = {"coco": 36001, "gqa": 7215, "ocr_vqa": 8040, "textvqa": 2249, "vg": 8530}
TEXT_ONLY_UNIFORM = 4041
CANDIDATE_FACTOR = 3
BATCHES = {"less-s%02d" % k: {"group": "candidates", "shard": k, "probe": "centroid", "models": ["warmup"]}
           for k in range(SHARDS)}
BATCHES["less-targets"] = {"group": "targets", "shard": 0, "probe": "centroid", "models": ["warmup"]}
BATCHES["less-qual"] = {"group": "candidates", "shard": 9, "probe": "centroid", "models": ["warmup"], "qual": True}
QUAL_HEAD, QUAL_LONGEST = 32, 16
EVALUATOR_QUALIFICATION_JOB = "11024165"  # V3 pin 11023448 was PREEMPTED (not requeued); replacement b0
INFERENCE_QUALIFICATION_JOB = "11022388"
GPU_QUALIFICATION_JOB = "11021226"
SELECTION_QUALIFICATION_JOB = "11022768"
MINUTES = 180
PROBE_ID = re.compile(r"[a-z0-9][a-z0-9_+-]{0,63}")


def require(value, message):
    if not value:
        raise ValueError(message)


def load_exact(ref, name):
    path = Path(ref["path"])
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(ref["bytes"] + 1)
        after = os.fstat(stream.fileno())
    require(stat.S_ISREG(before.st_mode) and before.st_uid == UID and len(raw) == ref["bytes"]
            and hashlib.sha256(raw).hexdigest() == ref["sha256"]
            and all(getattr(before, key) == getattr(after, key)
                    for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
            "Exact immutable bound source required: " + str(path))
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    return module


def modules():
    p = load_exact(HELPER, "tool2_less_custody_helpers")
    d = load_exact(DESIGN, "tool2_adeval_design")
    probe = load_exact(PROBE_HELPER, "tool2_adeval_probe_v2")
    return p, (d, probe)


def probe_spec(d, probe_id):
    require(probe_id in BATCHES, "Exact record-signal shard required")
    return dict(BATCHES[probe_id], probe_id=probe_id, shards=SHARDS)


def control(probe_id, attempt):
    require(type(attempt) is int and 1 <= attempt <= 9, "Finite explicit attempt required")
    return CONTROL_ROOT / probe_id / ("attempt-" + str(attempt))


def job_name(probe_id, attempt):
    return JOB_PREFIX + probe_id + "-a" + str(attempt)


def probe_job_ids(p, models, live_only=True):
    result = []
    state = p.query(["/usr/bin/sacct", "-X", "-nP", "-j", EVALUATOR_QUALIFICATION_JOB, "--format=State"]).strip().split("|")[0]
    if not live_only or state in ("PENDING", "RUNNING", "REQUEUED", "SUSPENDED"):
        result.append(EVALUATOR_QUALIFICATION_JOB)
    for model in models:
        if model in ("start", "warmup"):
            continue
        version, probe_id = model.split(":")
        versions = SEARCH if version == "p" else (version,)
        acks = [path for v in versions for path in sorted((PROBE_CONTROLS[v] / probe_id).glob("attempt-*/ACK.json"))]
        require(acks, "Probe must have an actual admission: " + model)
        for path in acks:
            job = json.loads(p.checked(p.descriptor(path)))["job_id"]
            state = p.query(["/usr/bin/sacct", "-X", "-nP", "-j", job, "--format=State"]).strip().split("|")[0]
            if not live_only or state in ("PENDING", "RUNNING", "REQUEUED", "SUSPENDED"):
                result.append(job)
    return result


def make_plan(p, d, probe_id, attempt, dependencies=None):
    spec = probe_spec(d, probe_id)
    if dependencies is None:
        dependencies = probe_job_ids(p, spec["models"])
    require(set(dependencies) <= set(probe_job_ids(p, spec["models"], live_only=False)),
            "Dependencies must be this batch's own probe admissions")
    function = dict(
        owner=OWNER, batch=spec, attempt=attempt, model_source=MODEL_SOURCE, custody_helper=HELPER,
        probe_helper=PROBE_HELPER, design=DESIGN, eval_configuration=EVAL_CONFIG,
        pool=p.POOL, groups=p.GROUPS, records=p.RECORDS, reference_configuration=REFERENCE_CONFIG,
        dependency_probe_jobs=dependencies, dependency_kind="afterany; missing/failed probes are recorded, not imputed",
        datasets="one shard of the record-signal v5 candidate pool (image sources) plus 3x Uniform text_only candidates, or the dev-reference target batch; first-turn LoRA gradients of the warm-up adapter, count-sketched",
        warmup=WARMUP, dev_references=list(DEV_REFERENCES), less_deviations=["reused 10% random warm-up adapter (Tool2 proportional, 1 epoch) instead of 5% x 4 epochs", "one checkpoint", "SGD not Adam-preconditioned gradients", "first QA turn only", "count-sketch instead of TRAK projection", "targets: GQA-val 512 and ScienceQA-val 512 only (TextVQA/POPE dev overlap reported banks)", "candidate pool 3x Uniform per source (stratified subsample)"],
        evaluator_qualification_job=EVALUATOR_QUALIFICATION_JOB,
        prior_inference_qualification=INFERENCE_QUALIFICATION_JOB,
        reported_benchmark_QA_reads=0, training_updates=0, preempt_rerun_not_resume=True)
    return dict(
        schema="tool2-less-approx-plan-v1", owner=OWNER, function=function,
        function_identity=hashlib.sha256(p.canonical(function)).hexdigest(),
        sources={"prepare.py": p.descriptor(SOURCE / "prepare.py"), "run.sbatch": p.descriptor(SOURCE / "run.sbatch"),
                 "grad.py": MODEL_SOURCE},
        job_name=job_name(probe_id, attempt), dependency=dependencies,
        resource=dict(account="cx", partition="preempt", qos="preempt_qos", gpu="L40S", GPUs=1, CPUs=8,
                      memory="64G", minutes=MINUTES, exclude="babel-p9-20"),
        automatic_requeue=False, independently_reviewed=False)


def contract(p, d, probe_id, attempt):
    folder = control(probe_id, attempt)
    plan_ref = p.descriptor(folder / "PLAN.json")
    plan = json.loads(p.checked(plan_ref))
    require(plan == make_plan(p, d, probe_id, attempt, plan.get("dependency")), "Exact bound evaluation plan required")
    review_ref = p.descriptor(folder / "OWNER_REVIEW.json")
    review = json.loads(p.checked(review_ref))
    require(review["status"] == REVIEW_STATUS and review["independent"] is False
            and review["plan"] == plan_ref and review["sources"] == plan["sources"]
            and review["blocking_findings"] == [], "Truthful exact owner review required")
    for ref in (MODEL_SOURCE, REFERENCE_CONFIG):
        p.checked(ref)
    for job in (GPU_QUALIFICATION_JOB, SELECTION_QUALIFICATION_JOB):
        require(p.query(["/usr/bin/sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State,ExitCode"]).strip()
                == job + "|COMPLETED|0:0", "Exact completed prerequisite required: " + job)
    return folder, plan_ref, plan, review_ref


def argv(probe_id, attempt, plan):
    dependency = ["--dependency=afterany:" + ":".join(plan["dependency"])] if plan["dependency"] else []
    return ["/usr/bin/sbatch", "--parsable", "--hold", "--requeue", "--export=NIL",
            "--account=cx", "--partition=preempt", "--qos=preempt_qos", "--gres=gpu:L40S:1",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G", "--time=03:00:00",
            "--exclude=babel-p9-20", *dependency, "--job-name=" + plan["job_name"],
            "--comment=dsmm-t2-less-approx-v2:" + plan["function_identity"],
            "--chdir=" + str(HOME), "--input=/dev/null", "--output=/dev/null", "--error=/dev/null",
            str(SOURCE / "run.sbatch"), probe_id, str(attempt)]


def dependency_ok(value, expected, terminal):
    if value in ("(null)", ""):
        return all(terminal(job) for job in expected)
    found = re.findall(r"afterany:(\d+)(?:\((?:unfulfilled|fulfilled)\))?", value)
    return sorted(found) == sorted(expected) and re.fullmatch(
        r"(afterany:\d+(\((?:unfulfilled|fulfilled)\))?)([,:]afterany:\d+(\((?:unfulfilled|fulfilled)\))?)*", value)


def scheduler(p, probe_id, attempt, plan, job, held=False, running=False):
    raw = p.query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = p.fields(raw)
    expected = dict(
        JobId=job, UserId="luyangz(2708899)", Account="cx", Partition="preempt", QOS="preempt_qos",
        JobName=plan["job_name"], Comment="dsmm-t2-less-approx-v2:" + plan["function_identity"],
        NumCPUs="8", NumTasks="1", TresPerNode="gres/gpu:L40S:1", TimeLimit="03:00:00",
        ExcNodeList="babel-p9-20", Requeue="1", Command=str(SOURCE / "run.sbatch"), WorkDir=str(HOME),
        StdIn="/dev/null", StdOut="/dev/null", StdErr="/dev/null")
    require(all(row.get(key) == value for key, value in expected.items())
            and row.get("NumNodes") in ("1", "1-1")
            and shlex.split(row.get("SubmitLine", "")) == argv(probe_id, attempt, plan),
            "Actual physical probe request/source/resources differ")
    terminal = lambda job: p.query(["/usr/bin/sacct", "-X", "-nP", "-j", job, "--format=State"]).strip().split("|")[0] \
        not in ("PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "")
    require(dependency_ok(row.get("Dependency", ""), plan["dependency"], terminal),
            "Exact probe afterany dependency required")
    if held:
        require(row.get("JobState") == "PENDING" and row.get("Reason") == "JobHeldUser"
                and row.get("RunTime") == "00:00:00", "Verified never-started held request required")
    if running:
        require(row.get("JobState") == "RUNNING" and row.get("Restarts") == os.environ.get("SLURM_RESTART_COUNT", "0")
                and row.get("NodeList") == socket.gethostname().split(".")[0]
                and p.valid_singleton_node(row["NodeList"]), "Actual owned running probe allocation required")
    tres = dict(value.split("=", 1) for value in row.get("AllocTRES" if running else "ReqTRES", "").split(",")
                if "=" in value)
    require(tres.get("cpu") == "8" and tres.get("mem") in ("64G", "65536M") and tres.get("gres/gpu") == "1",
            "Actual total probe resources differ")
    spool = subprocess.check_output(["/usr/bin/scontrol", "write", "batch_script", job, "-"],
                                    timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
    require(spool == p.checked(plan["sources"]["run.sbatch"]), "Exact immutable bound spool required")
    return raw, spool


def admit(p, d, probe_id, attempt, release=False):
    folder, plan_ref, plan, review_ref = contract(p, d, probe_id, attempt)
    with p.admission_lock():
        require(contract(p, d, probe_id, attempt) == (folder, plan_ref, plan, review_ref),
                "Contract changed while acquiring shared admission lock")
        if not release:
            require(not any((folder / name).exists() for name in
                            ("INTENT.json", "ACK.json", "AMBIGUOUS.json", "FUNCTION_CLAIM.json")),
                    "Existing submission or ambiguous ACK: reconcile, never duplicate")
            queue = p.query(["/usr/bin/squeue", "-u", "luyangz", "-r", "-h", "-o", "%i|%j|%q|%k"])
            rows = [line.split("|") for line in queue.splitlines() if line.strip()]
            require(all(len(row) == 4 for row in rows)
                    and not any(row[1].startswith(JOB_PREFIX + probe_id + "-a")
                                or plan["function_identity"] in row[3] for row in rows),
                    "Expanded census invalid or a live attempt of this probe already exists")
            limits = p.query(["/usr/bin/sacctmgr", "-nP", "show", "qos", "preempt_qos",
                              "format=Name,MaxSubmitPU,MaxJobsPU,MaxTRESPU"]).strip().split("|")
            require(len(limits) == 4 and limits[0] == "preempt_qos" and limits[1].isdigit()
                    and sum(row[2] == "preempt_qos" for row in rows) < int(limits[1]),
                    "Actual Preempt submission limit reached or unavailable")
            claim = p.seal(folder / "FUNCTION_CLAIM.json", dict(
                owner=OWNER, function=plan["function"], function_identity=plan["function_identity"], plan=plan_ref))
            intent = p.seal(folder / "INTENT.json", dict(
                owner=OWNER, plan=plan_ref, review=review_ref, claim=claim, argv=argv(probe_id, attempt, plan),
                actual_QoS_limits=limits, expanded_requests=len(rows)))
            try:
                response = subprocess.run(argv(probe_id, attempt, plan), text=True, capture_output=True,
                                          timeout=45, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as error:
                p.seal(folder / "AMBIGUOUS.json", dict(intent=intent, error=repr(error)))
                raise
            good = response.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;[\w.-]+)?\n?", response.stdout)
            if good:
                job = response.stdout.strip().split(";")[0]
                ack_ref = p.seal(folder / "ACK.json", dict(owner=OWNER, job_id=job, intent=intent, plan=plan_ref))
            p.seal(folder / "SUBMISSION_RESULT.json", dict(
                intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
            require(good, "No unambiguous actual ACK; reconcile exact intent before another submission")
            raw, spool = scheduler(p, probe_id, attempt, plan, job, held=True)
            p.seal(folder / "HELD_READBACK.json", dict(
                ACK=ack_ref, scheduler=raw, spool_base64=base64.b64encode(spool).decode()))
            return job
        ack_ref = p.descriptor(folder / "ACK.json")
        ack = json.loads(p.checked(ack_ref))
        require(ack["owner"] == OWNER and ack["plan"] == plan_ref, "Exact own ACK required")
        job = ack["job_id"]
        scheduler(p, probe_id, attempt, plan, job, held=True)
        intent = p.seal(folder / "RELEASE_INTENT.json", dict(ACK=ack_ref, job_id=job))
        response = subprocess.run(["/usr/bin/scontrol", "release", job], text=True, capture_output=True,
                                  timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
        p.seal(folder / "RELEASE_RESULT.json", dict(
            intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        require(response.returncode == 0, "Release failed or ambiguous; preserve exact identity")
        raw, _ = scheduler(p, probe_id, attempt, plan, job)
        p.seal(folder / "RELEASE_READBACK.json", dict(job_id=job, scheduler=raw))
        return job


def payload(rows):
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode() + b"\n"


def sealed(raw, name, stack):
    fd = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    stack.append(fd)
    view = memoryview(raw)
    while view:
        written = os.write(fd, view)
        require(written > 0, "Non-progressing sealed input write")
        view = view[written:]
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
    digest, offset = hashlib.sha256(), 0
    while offset < len(raw):
        block = os.pread(fd, min(1 << 20, len(raw) - offset), offset)
        require(bool(block), "Short sealed input readback")
        digest.update(block)
        offset += len(block)
    require(digest.hexdigest() == hashlib.sha256(raw).hexdigest() and fcntl.fcntl(fd, fcntl.F_GET_SEALS) == seals,
            "Sealed input bytes differ")
    return {"path": "/proc/" + str(os.getpid()) + "/fd/" + str(fd), "bytes": len(raw),
            "sha256": digest.hexdigest()}


def read_result(p, path, limit=8 << 20):
    fd = p.open_literal(path, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == UID and 0 < before.st_size <= limit,
                "Bounded owned probe result required")
        p.mount_for_fd(fd, before.st_dev, "group")
        raw = b""
        while len(raw) <= before.st_size:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            raw += block
        require(len(raw) == before.st_size, "Probe result changed while reading")
        return raw
    finally:
        os.close(fd)


def resolve_models(p, models):
    specs, missing = [], []
    for model in models:
        if model == "start":
            specs.append({"name": "start", "adapter": None})
            continue
        if model == "warmup":
            result = json.loads(p.checked(p.descriptor(Path(WARMUP["training_result"]))))
            require(result["status"] == "ACTUAL_FULL_NATIVE_ALLOCATION_EPOCH_NOT_EVALUATED"
                    and str(result["job_id"]) == WARMUP["job"] and result["full_epoch_training_complete"] is True
                    and result["config"]["policy"] == WARMUP["policy"]
                    and result["config"]["model_source"]["sha256"] == WARMUP["trainer_sha256"]
                    and result["config"]["output"] + "/training" == WARMUP["training_dir"],
                    "Exact completed warm-up adapter required")
            specs.append({"name": WARMUP["name"], "adapter": {
                "training_dir": WARMUP["training_dir"], "output_files": result["output_files"],
                "optimizer_groups": result["observation"]["optimizer_inventory"]["original_groups"],
                "producer_job": WARMUP["job"]}})
            continue
        version, probe_id = model.split(":")
        versions = SEARCH if version == "p" else (version,)
        receipts = next((found for v in versions
                         for found in [sorted((PROBE_CONTROLS[v] / probe_id).glob("attempt-*/PROBE_RESULT-*.json"))]
                         if found), [])
        if not receipts:
            missing.append(model)
            continue
        receipt = json.loads(p.checked(p.descriptor(receipts[-1])))
        job = receipts[-1].stem.split("-")[-1]
        parent = PROBE_OUTPUT_ROOT / (probe_id + "-" + job)
        raw = read_result(p, parent / "RESULT.json")
        require(hashlib.sha256(raw).hexdigest() == receipt["result_sha256"], "Probe result differs from receipt")
        result = json.loads(raw)
        require(result["status"] == "ACTUAL_SIGNAL_PROBE_TRAINED_AND_HELDOUT_EVALUATED" and result["job_id"] == job,
                "Actual completed probe required")
        specs.append({"name": version + "-" + probe_id, "adapter": {
            "training_dir": str(parent / "training"), "output_files": result["output_files"],
            "optimizer_groups": result["observation"]["optimizer_inventory"]["original_groups"],
            "probe_job": job, "result_sha256": receipt["result_sha256"]}})
    return specs, missing


def work(p, d, probe_id, attempt):
    design, probe = d
    folder, plan_ref, plan, review_ref = contract(p, d, probe_id, attempt)
    job = os.environ.get("SLURM_JOB_ID", "")
    ack = json.loads(p.checked(p.descriptor(folder / "ACK.json")))
    require(os.getuid() == UID and job.isdecimal() and ack["job_id"] == job
            and os.environ.get("SLURM_RESTART_COUNT", "0").isdecimal() and not os.environ.get("SLURM_ARRAY_JOB_ID"),
            "Actual owned (possibly requeued) allocation required")
    tag = job + "-r" + os.environ.get("SLURM_RESTART_COUNT", "0")
    release = json.loads(p.checked(p.descriptor(folder / "RELEASE_INTENT.json")))
    require(release["job_id"] == job, "Actual release intent required before DATA")
    scheduler(p, probe_id, attempt, plan, job, running=True)
    node = socket.gethostname().split(".")[0]
    p.validate_process(job, os.getpid(), node, os.environ.get("SLURM_STEP_ID"),
                       p.query(["/usr/bin/scontrol", "listpids", job]),
                       p.fields(p.query(["/usr/bin/scontrol", "show", "step", job + ".batch", "-o"])),
                       (Path("/proc") / str(os.getpid()) / "status").read_text())
    actor = dict(job_id=job, pid=os.getpid(), node=node, plan=plan_ref)
    p.seal(folder / ("ACTOR_BEFORE-" + tag + ".json"), actor)
    require(p.query(["/usr/bin/sacct", "-X", "-nP", "-j", EVALUATOR_QUALIFICATION_JOB, "--format=State,ExitCode"]).strip()
            == "COMPLETED|0:0", "Evaluator qualification batch must have completed successfully")
    batch = plan["function"]["batch"]
    specs, missing = resolve_models(p, batch["models"])
    require(not missing, "Every model of a record-signal shard must exist: " + ",".join(missing))
    p.seal(folder / ("RESOLVED_MODELS-" + tag + ".json"), dict(models=specs, missing=missing))
    require(specs, "No evaluable model in this batch")
    pool_raw, pool_fd = p.read_data(p.POOL, "pool")
    groups_raw, groups_fd = p.read_data(p.GROUPS, "group")
    records_raw, records_fd = p.read_data(p.RECORDS, "records")
    pool_value = json.loads(pool_raw)
    group_map = json.loads(groups_raw)["groups"]
    records = json.loads(records_raw)
    del records_raw, groups_raw
    ids = pool_value["record_ids"]
    group_of = [group_map[ident] for ident in ids]
    chosen, images = design.holdout(group_of, records)
    eligible = design.eligible(group_of, records, chosen, images)
    probe_spec_row = next(row for row in design.probes() if row["probe_id"] == batch["probe"])
    trained, _ = design.sample(probe_spec_row, eligible)
    excluded = set(trained)
    candidates = []
    for group in IMAGE_GROUPS:
        members = [index for index in eligible[group] if index not in excluded]
        rng = __import__("random").Random("tool2-recsig5-candidates:" + group)
        candidates.extend(rng.sample(members, min(CANDIDATE_FACTOR * UNIFORM_COUNTS[group], len(members))))
    text_members = [index for index in eligible["text_only"] if index not in excluded]
    candidates.extend(__import__("random").Random("tool2-less-approx-candidates:text_only").sample(
        text_members, min(CANDIDATE_FACTOR * TEXT_ONLY_UNIFORM, len(text_members))))
    candidates.sort()
    members = candidates
    shard = members[batch["shard"]::batch["shards"]] if batch["group"] == "candidates" else []
    if batch.get("qual"):
        def turn_length(i):
            turns = records[i]["conversations"]
            return len(turns[0]["value"]) + len(turns[1]["value"])
        head = shard[:QUAL_HEAD]
        longest = sorted((i for i in shard[QUAL_HEAD:]), key=lambda i: (-turn_length(i), i))[:QUAL_LONGEST]
        shard = head + longest
    descriptors, datasets = [], []
    try:
        if batch["group"] == "candidates":
            rows = [dict(records[i], id=ids[i]) for i in shard]
            raw = probe.payload(rows)
            datasets.append(dict(probe.sealed(raw, "tool2-less", descriptors), name=("records-candidates-qual" if batch.get("qual") else "records-candidates-s"
                                 + str(batch["shard"])), count=len(rows), generate=False, scorer=None,
                                 blank_variant=False))
        else:
            rows = []
            for ref in DEV_REFERENCES:
                raw = Path(ref["path"]).read_bytes()
                require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"],
                        "Exact dev reference required: " + ref["name"])
                values = json.loads(raw)
                require(len(values) == ref["count"], "Dev reference count differs")
                datasets.append(dict(probe.sealed(raw, "tool2-less-dev", descriptors), name=ref["name"],
                                     count=ref["count"], generate=False, scorer=None, blank_variant=False))
        p.seal(folder / ("SHARD_INPUT-" + tag + ".json"), dict(group=batch["group"], shard=batch["shard"],
               shards=batch["shards"], candidate_pool=len(members), excluded_probe_training=len(excluded),
               shard_sources={g: sum(1 for i in shard if group_of[i] == g) for g in IMAGE_GROUPS + ("text_only",)}, shard_records=len(rows),
               record_ids_sha256=hashlib.sha256(p.canonical([ids[i] for i in shard])).hexdigest()))
        image_root = pool_value["image_root"]
        del records, group_map, pool_value, pool_raw, group_of
        gc.collect()
        reference_config = json.loads(p.checked(REFERENCE_CONFIG))
        assets = json.loads(p.checked(reference_config["providers"]["assets"]))["assets"]
        evaluator = json.loads(p.checked(EVAL_CONFIG))
        project = PROBE_ROOT.parent
        parent = p.open_literal(project, os.O_RDONLY | os.O_DIRECTORY)
        try:
            try:
                os.mkdir(PROBE_ROOT.name, 0o700, dir_fd=parent)
            except FileExistsError:
                info = os.stat(PROBE_ROOT.name, dir_fd=parent, follow_symlinks=False)
                require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
                        "Owned private evaluation root required")
        finally:
            os.close(parent)
        output = PROBE_ROOT / ("job-" + job)
        directory = p.open_literal(PROBE_ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            try:
                os.mkdir(output.name, 0o700, dir_fd=directory)
            except FileExistsError:
                info = os.stat(output.name, dir_fd=directory, follow_symlinks=False)
                require(os.environ.get("SLURM_RESTART_COUNT", "0") != "0" and stat.S_ISDIR(info.st_mode)
                        and info.st_uid == UID, "Existing output only for this job's own requeued restart")
        finally:
            os.close(directory)
        config = dict(role="OWNED_LESS_APPROX_GRADIENT_FEATURES_NO_UPDATES", owner=OWNER, job_id=job,
                      models=specs, datasets=datasets, image_root=image_root, assets=assets,
                      eval_sources=evaluator["sources"], official_runtime_reference=evaluator["official_POPE_configuration"],
                      output=str(output), model_source=MODEL_SOURCE)
        config_ref = p.seal(folder / ("MODEL_CONFIG-" + tag + ".json"), config)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   HF_HOME="/data/hf_cache", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                   OPENBLAS_NUM_THREADS="8")
        command = [PYTHON, "-u", "-B", "-X", "pycache_prefix=/dev/null/no-cache", MODEL_SOURCE["path"],
                   "--config", config_ref["path"]]
        p.seal(folder / ("MODEL_INTENT-" + tag + ".json"), dict(config=config_ref, argv=command, actor=actor))
        started = time.monotonic()
        result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, timeout=10200)
        p.seal(folder / ("MODEL_EXIT-" + tag + ".json"), dict(
            returncode=result.returncode, elapsed_seconds=time.monotonic() - started, output=str(output)))
        require(result.returncode == 0, "Actual evaluation failed; per-model results are preserved")
    finally:
        for fd in descriptors:
            os.close(fd)
    report = json.loads((output / "RESULT.json").read_bytes())
    require(report["status"] == "ACTUAL_LESS_APPROX_GRADIENT_FEATURES" and report["job_id"] == job,
            "Actual gradient-feature result differs")
    p.seal(folder / ("GRAD_RESULT-" + tag + ".json"), dict(report=report, missing=missing, pool_fd=pool_fd,
                                                          groups_fd=groups_fd, records_fd=records_fd))
    scheduler(p, probe_id, attempt, plan, job, running=True)
    print(json.dumps(dict(job_id=job, datasets=report["datasets"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit", "release", "worker"))
    parser.add_argument("--probe", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    p, d = modules()
    if args.command == "worker":
        work(p, d, args.probe, args.attempt)
    else:
        print(json.dumps(dict(job_id=admit(p, d, args.probe, args.attempt, release=args.command == "release"))))


if __name__ == "__main__":
    main()
