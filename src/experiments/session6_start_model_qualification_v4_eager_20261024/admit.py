import argparse
import fcntl
from pathlib import Path
import re
import subprocess
import uuid

from . import common as c
from .worker import fields


def call(argv):
    return subprocess.check_output(argv, text=True, timeout=45)


def submit(configuration_path, review_path, directory):
    reference = c.desc(configuration_path)
    config = c.checked(reference)
    c.validate_config(config)
    review_ref = c.desc(review_path)
    c.validate_review(c.checked(review_ref), reference, config)
    directory = Path(directory)
    c.require(directory.resolve() == (c.BASE / "registration").resolve(),
              "Exact owned registration directory required")
    directory.mkdir(parents=True, exist_ok=True)
    lock = Path(config["admission_lock"])
    c.require(lock.is_file(), "Shared ACES admission lock missing")
    with lock.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        c.require(not (directory / "INTENT.json").exists(), "Reconcile prior durable intent; never duplicate")
        queue = call(["squeue", "-r", "-h", "-u", "u.lz341613", "-q", "normal", "-o", "%i|%j|%T|%b"])
        limits = call(["sacctmgr", "-nP", "show", "qos", "normal", "format=Name,MaxSubmitPU,MaxTRESPU,MaxJobsPU"])
        rows = [row.split("|") for row in limits.splitlines() if row.startswith("normal|")]
        c.require(len(rows) == 1 and rows[0][1].isdecimal()
                  and len(queue.splitlines()) + 1 <= int(rows[0][1]),
                  "Actual expanded normal submission capacity unavailable")
        c.require(not any(row.split("|")[1] == config["job_name"] for row in queue.splitlines()),
                  "Owned model qualification already live")
        nonce = uuid.uuid4().hex
        binding_path = directory / "BINDING.json"
        argv = ["sbatch", "--parsable", "--hold", "--requeue", "--nodes=1", "--ntasks=1",
            "--account=158416130568", "--partition=gpu", "--qos=normal", "--gres=gpu:h100:1",
            "--cpus-per-task=8", "--mem=64G", "--time=15", "--open-mode=append",
            "--job-name=" + config["job_name"], "--comment=" + c.PREFIX + nonce,
            "--chdir=" + str(c.ROOT), "--output=" + str(directory / "job-%j.out"),
            str(c.HERE / "run.sbatch"), str(binding_path), config["python"]]
        intent = c.seal(directory / "INTENT.json", dict(owner=c.OWNER, configuration=reference,
            review=review_ref, nonce=nonce, argv=argv, expanded_normal_queue=queue,
            actual_site_limits=limits, UTC=c.now(), before_submission=True,
            new_downloads=False, new_Babel_requests=False))
        try:
            response = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired as error:
            c.seal(directory / "AMBIGUOUS_RPC.json", dict(intent=intent, error=str(error), reconcile_required=True))
            raise
        c.seal(directory / "RPC_RESPONSE.json", dict(rc=response.returncode, stdout=response.stdout, stderr=response.stderr))
        c.require(response.returncode == 0 and re.fullmatch(r"\d+(?:;[\w.-]+)?\s*", response.stdout),
                  "Failed or ambiguous sbatch ACK; reconcile exact nonce before another RPC")
        job = response.stdout.strip().split(";")[0]
        c.seal(directory / "ACK.json", dict(job_id=job, intent=intent, UTC=c.now()))
        binding = c.seal(binding_path, dict(owner=c.OWNER, job_id=job,
            configuration=reference, review=review_ref, intent=intent))
        raw = call(["scontrol", "show", "job", job, "-o"])
        value = fields(raw)
        expected = dict(JobId=job, JobName=config["job_name"], JobState="PENDING", Reason="JobHeldUser",
            Partition="gpu", QOS="normal", Account="158416130568", NumTasks="1", NumCPUs="8",
            TimeLimit="00:15:00", Restarts="0", RunTime="00:00:00", WorkDir=str(c.ROOT),
            Command=str(c.HERE / "run.sbatch"), TresPerNode="gres/gpu:h100:1", Comment=c.PREFIX + nonce)
        tres = dict(item.split("=", 1) for item in value.get("ReqTRES", "").split(",") if "=" in item)
        c.require(all(value.get(k) == v for k, v in expected.items())
                  and value.get("NumNodes") in ("1", "1-1")
                  and value.get("UserId", "").split("(")[0] == "u.lz341613"
                  and value.get("AllocTRES") in (None, "(null)")
                  and tres.get("mem") in ("64G", "65536M") and tres.get("gres/gpu") == "1",
                  "Exact never-started held model qualification differs")
        spool = subprocess.check_output(["scontrol", "write", "batch_script", job, "-"], timeout=30)
        c.require(spool == (c.HERE / "run.sbatch").read_bytes(), "Actual held wrapper spool differs")
        c.seal(directory / "HELD_READBACK.json", dict(actual=raw, wrapper_sha256=c.sha(c.HERE / "run.sbatch")))
        c.seal(directory / "RELEASE_INTENT.json", dict(job_id=job, binding=binding, UTC=c.now()))
        response = call(["scontrol", "release", job])
        c.seal(directory / "RELEASE_ACK.json", dict(job_id=job, stdout=response, UTC=c.now()))
        c.seal(directory / "RELEASE_READBACK.json", dict(job_id=job,
            actual=call(["scontrol", "show", "job", job, "-o"]), UTC=c.now()))
        return dict(job_id=job, status="SUBMITTED_RELEASED_NOT_COMPLETED", binding=binding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    print(c.canonical(submit(args.configuration, args.review, args.registration)).decode())
