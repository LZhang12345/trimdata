"""Finite feature-qualification admission for the exact S6-reviewed S7 spec."""
import argparse
import fcntl
from pathlib import Path
import re
import subprocess
import uuid

from . import common as c


def call(argv):
    return subprocess.check_output(argv, text=True, timeout=45).strip()


def finish(binding, spec, directory):
    intent, ack = c.checked(binding["intent"]), c.read(directory / "ACK.json")
    c.require(ack["job_id"] == binding["job_id"] and ack["intent"] == binding["intent"]
              and intent["configuration"] == binding["configuration"], "Exact feature intent/ACK differs")
    raw = call(["scontrol", "show", "job", "-o", binding["job_id"]])
    fields = dict(re.findall(r"(\w+)=(\S+)", raw))
    site = spec["site"]
    expected = dict(
        JobId=binding["job_id"], JobName=spec["job_name"], Account="cx" if site == "babel" else "158416130568",
        Partition="general" if site == "babel" else "gpu", QOS="normal", WorkDir=str(c.ROOT),
        Command=str(c.ROOT / spec["wrapper"]), TresPerNode="gres/gpu:" + ("L40S" if site == "babel" else "h100") + ":1",
        NumTasks="1", NumCPUs="8", TimeLimit="00:55:00", Comment="dsmm-s7-native:" + intent["nonce"])
    c.require(all(fields.get(k) == v for k, v in expected.items()), "Actual feature resource/ownership differs")
    tres = dict(part.split("=", 1) for part in fields.get("ReqTRES", "").split(",") if "=" in part)
    c.require(tres.get("mem") in {"64G", "65536M"} and tres.get("gres/gpu") == "1",
              "Actual feature RAM/GPU differs")
    c.require(fields.get("NumNodes") in {"1", "1-1"}
              and fields.get("UserId", "").split("(")[0] == ("luyangz" if site == "babel" else "u.lz341613"),
              "Actual feature user/node count differs")
    spool = subprocess.check_output(["scontrol", "write", "batch_script", binding["job_id"], "-"], timeout=30)
    c.require(spool == (c.ROOT / spec["wrapper"]).read_bytes(), "Current feature payload differs")
    operation = directory / ("readback-" + uuid.uuid4().hex)
    c.seal(operation / "BEFORE.json", dict(actual=raw, binding=c.desc(directory / "BINDING.json"), UTC=c.now()))
    if fields.get("JobState") == "PENDING" and fields.get("Reason") == "JobHeldUser":
        c.require(fields.get("RunTime") == "00:00:00" and fields.get("Restarts") == "0"
                  and fields.get("AllocTRES") in (None, "(null)"), "Exact never-started held request required")
        c.seal(operation / "RELEASE_INTENT.json", dict(job_id=binding["job_id"], UTC=c.now()))
        call(["scontrol", "release", binding["job_id"]])
    after = call(["scontrol", "show", "job", "-o", binding["job_id"]])
    c.seal(operation / "AFTER.json", dict(actual=after, no_student_controls=True, UTC=c.now()))
    return dict(binding=binding, actual=after)


def submit(spec_path, review_path, registration):
    spec, review = c.read(spec_path), c.read(review_path)
    configuration = c.desc(spec_path)
    c.verify_sources(spec["sources"])
    c.verify_sources(spec["dependencies"])
    c.require(review["status"] == "PASS_EXACT_S7_PREREQUISITE_SOURCE_AND_CONFIG"
              and review["configuration"] == configuration and review["sources"] == spec["sources"]
              and review["dependencies"] == spec["dependencies"], "Genuine S6 exact feature spec review required")
    directory = Path(registration)
    directory.mkdir(parents=True, exist_ok=True)
    with Path(spec["admission_lock"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "BINDING.json"
        if path.exists():
            binding = c.read(path)
            c.require(binding["configuration"] == configuration and binding["review"] == c.desc(review_path),
                      "Existing feature function/source review differs")
            return finish(binding, spec, directory)
        c.require(not (directory / "INTENT.json").exists(), "Unresolved feature RPC intent; reconcile, do not repeat")
        site = spec["site"]
        user = "luyangz" if site == "babel" else "u.lz341613"
        queue = call(["squeue", "-r", "-h", "-u", user, "-o", "%i|%j|%T|%b"])
        normal_queue = call(["squeue", "-r", "-h", "-u", user, "-q", "normal", "-o", "%i|%j|%T|%b"])
        c.require(not any(line.split("|")[1] == spec["job_name"] for line in queue.splitlines()),
                  "Exact feature qualifier is already live")
        limits = call(["sacctmgr", "-n", "-P", "show", "qos", "normal", "format=Name,MaxTRESPU,MaxSubmitPU"])
        rows = [line.split("|") for line in limits.splitlines() if line.startswith("normal|")]
        c.require(len(rows) == 1 and (not rows[0][2].isdecimal() or len(normal_queue.splitlines()) < int(rows[0][2])),
                  "Actual site/user submit ceiling reached or unknown")
        nonce = uuid.uuid4().hex
        argv = ["sbatch", "--parsable", "--hold", "--requeue",
                "--account=" + ("cx" if site == "babel" else "158416130568"),
                "--partition=" + ("general" if site == "babel" else "gpu"), "--qos=normal",
                "--gres=gpu:" + ("L40S" if site == "babel" else "h100") + ":1",
                "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G", "--time=55",
                "--signal=B:USR1@300", "--open-mode=append", "--chdir=" + str(c.ROOT),
                "--job-name=" + spec["job_name"], "--comment=dsmm-s7-native:" + nonce,
                "--output=" + str(directory / "job-%j.out"), str(c.ROOT / spec["wrapper"]), str(path)]
        intent = c.seal(directory / "INTENT.json", dict(owner=c.OWNER, nonce=nonce, function=spec["function"],
            configuration=configuration, review=c.desc(review_path), argv=argv, actual_queue=queue, actual_normal_queue=normal_queue,
            actual_qos=limits, UTC=c.now(), before_submission=True, no_student_submission=True))
        try:
            response = subprocess.run(argv, text=True, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired as error:
            c.seal(directory / "AMBIGUOUS_RPC.json", dict(intent=intent, error=str(error), reconcile_required=True))
            raise
        c.seal(directory / "RPC_RESPONSE.json", dict(rc=response.returncode, stdout=response.stdout, stderr=response.stderr))
        c.require(response.returncode == 0 and re.fullmatch(r"\d+(?:;[\w.-]+)?", response.stdout.strip()),
                  "Feature submission failed/ambiguous; preserve exact intent")
        job = response.stdout.strip().split(";")[0]
        c.seal(directory / "ACK.json", dict(job_id=job, intent=intent, UTC=c.now()))
        binding = dict(schema="s7-native-feature-qualification-binding-v1", owner=c.OWNER, job_id=job,
                       configuration=configuration, review=c.desc(review_path), intent=intent)
        c.seal(path, binding)
        return finish(binding, spec, directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    print(submit(args.configuration, args.review, args.registration))
