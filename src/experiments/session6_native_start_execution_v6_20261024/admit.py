import argparse
import fcntl
from pathlib import Path
import re
import subprocess
import uuid

from . import common as c
from experiments.session6_start_model_qualification_v4_eager_20261024.worker import fields


def call(argv):
    return subprocess.check_output(argv, text=True, timeout=45)


def held(raw, config, phase, parent, nonce, task=None, dependency=None):
    value = fields(raw)
    route = c.route(phase)
    expected = dict(JobName=c.job_name(phase), JobState="PENDING", Reason="JobHeldUser",
        Partition=route["partition"], QOS="normal", Account="158416130568", NumTasks="1",
        NumCPUs=str(route["cpus"]), TimeLimit="00:20:00", Requeue="1", Restarts="0",
        RunTime="00:00:00", WorkDir=str(c.ROOT), Command=str(c.HERE / "run.sbatch"),
        Comment=c.PREFIX + nonce)
    if route["GPU_GRES"]:
        expected["TresPerNode"] = "gres/" + route["GPU_GRES"]
    if task is None:
        expected["JobId"] = parent
    else:
        expected.update(ArrayJobId=parent, ArrayTaskId=str(task))
    tres = dict(item.split("=", 1) for item in value.get("ReqTRES", "").split(",") if "=" in item)
    c.require(all(value.get(k) == v for k, v in expected.items())
              and value.get("JobId", "").isdecimal() and value.get("NumNodes") in ("1", "1-1")
              and value.get("UserId", "").split("(")[0] == "u.lz341613"
              and value.get("AllocTRES") in (None, "(null)")
              and tres.get("mem") in (str(route["memory_GiB"]) + "G",
                                      str(route["memory_GiB"] * 1024) + "M"),
              "Exact never-started held phase/task/owner/resources differ")
    if route["GPU_GRES"]:
        c.require(tres.get("gres/gpu") == "1", "H100 route requires one actual GPU")
    else:
        c.require(not any(key.startswith("gres/gpu") for key in tres)
                  and value.get("TresPerNode") in (None, "(null)", "N/A"),
                  "Pure CPU continuation/scoring must not request GPU GRES")
    if dependency:
        c.require(re.fullmatch(re.escape(dependency) + r"(?:\((?:unfulfilled|fulfilled)\))?",
                               value.get("Dependency", "")) is not None,
                  "Exact owned scientific dependency differs")


def validated(configuration_path, review_path):
    actor = c.admission_context()
    reference = c.desc(configuration_path)
    config = c.checked(reference)
    c.validate_config(config)
    review_ref = c.desc(review_path)
    c.validate_review(c.checked(review_ref), reference, config)
    return actor, reference, config, review_ref


def census(config, required, phases):
    queue = call(["squeue", "-r", "-h", "-u", "u.lz341613", "-q", "normal", "-o", "%i|%j|%T|%b"])
    limits = call(["sacctmgr", "-nP", "show", "qos", "normal", "format=Name,MaxSubmitPU,MaxTRESPU,MaxJobsPU"])
    rows = [row.split("|") for row in limits.splitlines() if row.startswith("normal|")]
    c.require(len(rows) == 1 and rows[0][1].isdecimal()
              and len(queue.splitlines()) + required <= int(rows[0][1]),
              "Actual expanded normal submission slots cannot hold the entire finite bundle")
    names = [row.split("|")[1] for row in queue.splitlines()]
    c.require(not any(c.job_name(phase) in names for phase in phases), "Exact owned bundle already live")
    return dict(expanded_normal_queue=queue, actual_site_limits=limits,
                new_expanded_requests=required)


def submit_held(actor, reference, config, review_ref, phase, capacity, dependency=None):
    c.require(actor == c.admission_context(), "Actual actor/root changed before held-request creation")
    directory = c.registration(phase)
    c.require(not (directory / "INTENT.json").exists(),
              "Reconcile existing durable intent/ACK before any further submission")
    directory.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    binding_path = directory / "BINDING.json"
    route = c.route(phase)
    count = c.SHARDS if phase == "predict" else 1
    argv = ["sbatch", "--parsable", "--hold", "--requeue", "--nodes=1", "--ntasks=1",
        "--account=158416130568", "--partition=" + route["partition"], "--qos=normal",
        "--cpus-per-task=" + str(route["cpus"]), "--mem=" + str(route["memory_GiB"]) + "G",
        "--time=20", "--open-mode=append", "--job-name=" + c.job_name(phase),
        "--comment=" + c.PREFIX + nonce, "--chdir=" + str(c.ROOT),
        "--output=" + str(directory / "job-%A_%a.out")]
    if route["GPU_GRES"]:
        argv += ["--gres=" + route["GPU_GRES"], "--signal=B:USR1@180"]
    if phase == "predict":
        argv.append("--array=0-" + str(c.SHARDS - 1))
    if dependency:
        c.require(re.fullmatch(r"after(?:ok|any):\d+", dependency) is not None,
                  "Only an exact real owned parent dependency may be registered")
        argv += ["--dependency=" + dependency, "--kill-on-invalid-dep=yes"]
    argv += [str(c.HERE / "run.sbatch"), str(binding_path), config["python"], phase]
    intent = c.seal(directory / "INTENT.json", dict(owner=c.OWNER, phase=phase,
        configuration=reference, review=review_ref, nonce=nonce, argv=argv, expanded_tasks=count,
        actual_actor_before_any_write_or_RPC=actor, capacity=capacity, dependency=dependency,
        execution_route=route, UTC=c.now(), before_submission=True,
        no_private_GPU_concurrency_cap=True, finite_same_job_requeue=True))
    try:
        response = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as error:
        c.seal(directory / "AMBIGUOUS_RPC.json", dict(intent=intent, error=str(error), reconcile_required=True))
        raise
    c.seal(directory / "RPC_RESPONSE.json", dict(rc=response.returncode,
        stdout=response.stdout, stderr=response.stderr))
    c.require(response.returncode == 0 and re.fullmatch(r"\d+(?:;[\w.-]+)?\s*", response.stdout),
              "Failed/ambiguous ACK: preserve bundle held and reconcile exact nonce, never repeat RPC")
    job = response.stdout.strip().split(";")[0]
    c.seal(directory / "ACK.json", dict(job_id=job, intent=intent, UTC=c.now()))
    binding = c.seal(binding_path, dict(owner=c.OWNER, phase=phase, job_id=job,
        configuration=reference, review=review_ref, intent=intent, dependency=dependency))
    requests = [(job + "_" + str(index), index) for index in range(c.SHARDS)] if phase == "predict" else [(job, None)]
    readbacks = []
    for target, index in requests:
        raw = call(["scontrol", "show", "job", target, "-o"])
        held(raw, config, phase, job, nonce, index, dependency)
        readbacks.append(raw)
    spool = subprocess.check_output(["scontrol", "write", "batch_script", job, "-"], timeout=30)
    c.require(spool == (c.HERE / "run.sbatch").read_bytes(), "Actual owned held spool differs")
    c.seal(directory / "HELD_READBACK.json", dict(tasks=readbacks,
        wrapper_sha256=c.sha(c.HERE / "run.sbatch"), expanded_tasks=count))
    return dict(job_id=job, expanded_tasks=count, phase=phase, binding=binding)


def release(request):
    c.admission_context()
    phase, job = request["phase"], request["job_id"]
    directory = c.registration(phase)
    binding = c.checked(request["binding"])
    c.require(request["binding"]["path"] == str(directory / "BINDING.json")
              and binding["owner"] == c.OWNER and binding["phase"] == phase
              and binding["job_id"] == job, "Release only the exact registered owned request")
    c.seal(directory / "RELEASE_INTENT.json", dict(job_id=job, binding=request["binding"], UTC=c.now()))
    response = call(["scontrol", "release", job])
    c.seal(directory / "RELEASE_ACK.json", dict(job_id=job, stdout=response, UTC=c.now()))
    c.seal(directory / "RELEASE_READBACK.json", dict(job_id=job,
        actual=call(["squeue", "-r", "-h", "-j", job, "-o", "%i|%j|%T|%b|%R"]), UTC=c.now()))


def submit(configuration_path, review_path, phase):
    actor, reference, config, review_ref = validated(configuration_path, review_path)
    c.require(phase == "prepare", "Only the initial reviewed prepare+CPU-continuation bundle is manually admitted")
    lock = Path(config["admission_lock"])
    c.require(lock.is_file(), "Canonical admission lock required")
    with lock.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        for member in ("prepare", "dispatch"):
            c.require(not (c.registration(member) / "INTENT.json").exists(),
                      "Existing prepare/controller intent must be reconciled, not duplicated")
        capacity = census(config, 2, ("prepare", "dispatch"))
        prepare = submit_held(actor, reference, config, review_ref, "prepare", capacity)
        dispatch = submit_held(actor, reference, config, review_ref, "dispatch", capacity,
                               dependency="afterok:" + prepare["job_id"])
        bundle = c.seal(c.BASE / "INITIAL_BUNDLE_HELD.json", dict(
            owner=c.OWNER, configuration=reference, prepare=prepare, dispatch=dispatch, UTC=c.now()))
        release(dispatch)
        release(prepare)
        return dict(status="SUBMITTED_RELEASED_PREPARE_AND_FINITE_CPU_CONTINUATION",
                    prepare=prepare, CPU_continuation=dispatch, bundle=bundle)


def dispatch_predictions(binding, context):
    actor, reference, config, review_ref = validated(binding["configuration"]["path"], binding["review"]["path"])
    c.require(binding["owner"] == c.OWNER and binding["phase"] == "dispatch"
              and binding["job_id"] == context["job_id"] and reference == binding["configuration"],
              "Only the exact actual owned CPU continuation may dispatch the finite roster")
    prepare_binding = c.checked(c.desc(c.registration("prepare") / "BINDING.json"))
    c.require(binding["dependency"] == "afterok:" + prepare_binding["job_id"],
              "CPU continuation must depend on its exact owned real prepare")
    c.ready(config, reference)
    lock = Path(config["admission_lock"])
    with lock.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        committed = c.BASE / "PREDICTION_BUNDLE_COMMITTED.json"
        if committed.exists():
            previous = c.checked(c.desc(committed))
            c.require(previous["owner"] == c.OWNER and previous["configuration"] == reference,
                      "Foreign committed scientific roster")
            for name in ("prediction", "CPU_score"):
                saved = c.checked(previous[name]["binding"])
                c.require(saved["configuration"] == reference
                          and saved["job_id"] == previous[name]["job_id"],
                          "Committed roster binding differs; reconcile rather than resubmit")
            return dict(status="ALREADY_COMMITTED_NO_DUPLICATE_SUBMISSION", committed=c.desc(committed))
        for member in ("predict", "score"):
            c.require(not (c.registration(member) / "INTENT.json").exists(),
                      "Partial finite roster intent/ACK requires reconciliation, no duplicate RPC")
        capacity = census(config, c.SHARDS + 1, ("predict", "score"))
        prediction = submit_held(actor, reference, config, review_ref, "predict", capacity)
        score = submit_held(actor, reference, config, review_ref, "score", capacity,
                           dependency="afterany:" + prediction["job_id"])
        c.seal(c.BASE / "PREDICTION_BUNDLE_HELD.json", dict(owner=c.OWNER,
            configuration=reference, prediction=prediction, CPU_score=score, expanded_requests=c.SHARDS + 1))
        release(score)
        release(prediction)
        receipt = c.seal(committed, dict(owner=c.OWNER, configuration=reference,
            prediction=prediction, CPU_score=score, actual_CPU_continuation=context,
            expanded_requests=c.SHARDS + 1, UTC=c.now()))
        return dict(status="FINITE32_PREDICTION_AND_DEPENDENT_CPU_SCORE_SUBMITTED",
                    prediction=prediction, CPU_score=score, committed=receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--phase", choices=["prepare"], required=True)
    args = parser.parse_args()
    print(c.canonical(submit(args.configuration, args.review, args.phase)).decode())
