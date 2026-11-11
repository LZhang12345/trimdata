"""Finite reviewed feature arrays; all-eight author policy, no automatic refill."""
import argparse
import fcntl
from pathlib import Path
import re
import subprocess
import uuid

from experiments.root_native_campaign_20261023 import common as c
from . import worker


def call(argv):
    return subprocess.check_output(argv, text=True, timeout=45).strip()


def verify(binding, spec):
    raw = call(["scontrol", "show", "job", "-o", binding["job_id"]])
    fields = dict(re.findall(r"(\w+)=(\S+)", raw))
    intent = c.checked(binding["intent"])
    expected = dict(JobId=binding["job_id"], JobName=spec["job_name"],
                    Account=spec["site"]["account"], Partition=spec["site"]["partition"],
                    QOS=spec["site"]["qos"], WorkDir=str(worker.ROOT),
                    Command=str(worker.ROOT / spec["wrapper"]),
                    TresPerNode="gres/gpu:" + spec["site"]["gpu_type"] + ":1",
                    Comment="s7-fullpool:" + intent["nonce"])
    c.require(all(fields.get(key) == value for key, value in expected.items()),
              "Accepted array identity/route differs")
    c.require(fields["UserId"].split("(")[0] == spec["site"]["user"], "Wrong actual array owner")
    c.require(fields.get("NumCPUs") == str(spec["allocation"]["cpus"])
              and fields.get("ArrayTaskId", "").split("%")[0] == "0-" + str(len(spec["global_shard_indices"]) - 1),
              "Accepted expanded array resources/routing differ")
    spool = subprocess.check_output(["scontrol", "write", "batch_script", binding["job_id"], "-"], timeout=30)
    c.require(spool == (worker.ROOT / spec["wrapper"]).read_bytes(), "Accepted array source differs")
    return raw


def submit(spec_path, review_path, registration):
    description = c.desc(spec_path)
    spec, review = c.checked(description), c.read(review_path)
    worker.checked_review(spec, description, review)
    site, allocation = spec["site"], spec["allocation"]
    c.require(site["name"] == "babel" and site["account"] == "cx" and site["gpu_type"] == "L40S"
              and (site["partition"], site["qos"]) in (("general", "normal"), ("preempt", "preempt_qos")),
              "This deployment uses actual Babel canonical inputs and exact L40S routes")
    wrapper = Path(spec["wrapper"])
    c.require(not wrapper.is_absolute() and ".." not in wrapper.parts and str(wrapper) in spec["sources"],
              "Frozen array wrapper required")
    directory = Path(registration)
    directory.mkdir(parents=True, exist_ok=True)
    with Path(spec["admission_lock"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "BINDING.json"
        if path.exists():
            binding = c.read(path)
            c.require(binding["configuration"] == description and binding["review"] == c.desc(review_path),
                      "This finite array is already bound differently")
            return dict(binding=binding, actual=verify(binding, spec), repeated_submission=False)
        c.require(not (directory / "INTENT.json").exists(),
                  "Ambiguous prior array intent: reconcile exact nonce before another submission")
        function_registration = Path(spec["function_registration"])
        c.require(not function_registration.exists(),
                  "Existing cross-version function intent requires exact reconciliation, not another controller")
        queue = call(["squeue", "-r", "-h", "-u", site["user"], "-q", site["qos"], "-o", "%i|%j|%T|%b|%r"])
        c.require(not any(row.split("|")[1].startswith(spec["live_function_prefix"]) for row in queue.splitlines()),
                  "Exact scientific array is already live")
        qos = call(["sacctmgr", "-n", "-P", "show", "qos", site["qos"], "format=Name,MaxTRESPU,MaxSubmitPU"])
        rows = [row.split("|") for row in qos.splitlines() if row.startswith(site["qos"] + "|")]
        count = len(spec["global_shard_indices"])
        c.require(len(rows) == 1 and rows[0][2].isdigit()
                  and len(queue.splitlines()) + count <= int(rows[0][2]),
                  "Expanded finite array exceeds actual QoS/user submission headroom")
        gpu_limit = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", rows[0][1])
        c.require(gpu_limit is not None and spec["array_concurrency"] <= int(gpu_limit.group(1)),
                  "Array concurrency exceeds actual site/account GPU QoS")
        commitments = []
        for row in queue.splitlines():
            parts = row.split("|")
            c.require(len(parts) == 5, "Malformed expanded scheduler census")
            gpu = re.search(r"gpu(?::[^:,]+)?:(\d+)", parts[3])
            c.require(parts[3] in ("N/A", "(null)") or gpu is not None, "Unknown GPU multiplicity")
            commitments.append(dict(job_id=parts[0], job_name=parts[1], state=parts[2],
                GPUs=int(gpu.group(1)) if gpu else 0, reason=parts[4],
                project_scope="multimodal" if parts[1].startswith(("dsmm", "s7-", "s5-", "s6-")) else "unrelated"))
        nonce = uuid.uuid4().hex
        argv = ["sbatch", "--parsable", "--hold", "--requeue",
                "--account=" + site["account"], "--partition=" + site["partition"], "--qos=" + site["qos"],
                "--gres=gpu:L40S:1", "--nodes=1", "--ntasks=1",
                "--array=0-" + str(count - 1) + "%" + str(spec["array_concurrency"]),
                "--cpus-per-task=" + str(allocation["cpus"]), "--mem=" + str(allocation["memory_mb"]) + "M",
                "--time=" + str(allocation["minutes"]), "--signal=B:USR1@" + str(allocation["reserve_seconds"]),
                "--open-mode=append", "--chdir=" + str(worker.ROOT), "--job-name=" + spec["job_name"],
                "--comment=s7-fullpool:" + nonce, "--output=" + str(directory / "job-%A_%a.out")]
        if spec["dependency"]:
            argv.append("--dependency=afterok:" + spec["dependency"])
        argv += [str(worker.ROOT / wrapper), str(path)]
        intent = c.seal(directory / "INTENT.json", dict(owner="Session7", nonce=nonce,
            configuration=description, review=c.desc(review_path), argv=argv,
            actual_QoS_queue=queue, actual_QoS_limits=qos,
            expanded_GPU_commitments=commitments,
            global_shard_indices=spec["global_shard_indices"], before_submission=True,
            no_student_or_old_refill=True, UTC=c.now()))
        c.seal(function_registration, dict(owner="Session7", intent=intent,
               configuration=description, registration=str(directory), nonce=nonce))
        try:
            response = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired as error:
            c.seal(directory / "AMBIGUOUS_RPC.json", dict(intent=intent, error=str(error)))
            raise
        c.seal(directory / "RPC_RESPONSE.json", dict(rc=response.returncode,
               stdout=response.stdout, stderr=response.stderr))
        c.require(response.returncode == 0 and re.fullmatch(r"\d+(?:;[\w.-]+)?", response.stdout.strip()),
                  "Array submission failed/ambiguous; preserve intent and reconcile, never duplicate")
        job = response.stdout.strip().split(";")[0]
        c.seal(directory / "ACK.json", dict(job_id=job, intent=intent, UTC=c.now()))
        binding = dict(schema="s7-finalpool-published-array-binding-v1", owner="Session7", job_id=job,
                       configuration=description, review=c.desc(review_path), intent=intent)
        c.seal(path, binding)
        before = verify(binding, spec)
        c.seal(directory / "HELD_READBACK.json", dict(actual=before, UTC=c.now()))
        c.seal(directory / "RELEASE_INTENT.json", dict(job_id=job, author_policy=spec["author_policy"],
               dependency=spec["dependency"], UTC=c.now(), no_other_project_control=True))
        release = subprocess.run(["scontrol", "release", job], capture_output=True, text=True, timeout=30)
        c.seal(directory / "RELEASE_RESPONSE.json", dict(rc=release.returncode,
               stdout=release.stdout, stderr=release.stderr))
        c.require(release.returncode == 0, "Array release unacknowledged; reconcile exact request")
        after = call(["scontrol", "show", "job", "-o", job])
        c.seal(directory / "AFTER_RELEASE.json", dict(actual=after, UTC=c.now()))
        return dict(binding=binding, actual=after, expanded_tasks=count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    print(submit(args.configuration, args.review, args.registration))
