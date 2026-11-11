"""Scalar, intent-first prerequisite admission; Babel release remains withheld."""
import argparse
import fcntl
from pathlib import Path
import re
import socket
import subprocess
import uuid

from . import common as c


def call(argv):
    return subprocess.check_output(argv, text=True, timeout=45).strip()


def route(spec):
    c.require(spec["site"] in ("babel", "aces"), "Unknown prerequisite site")
    return dict(account="cx" if spec["site"] == "babel" else "158416130568",
                partition="general" if spec["site"] == "babel" else "gpu", qos="normal",
                gpu_type="L40S" if spec["site"] == "babel" else "h100",
                user="luyangz" if spec["site"] == "babel" else "u.lz341613")


def clock(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def verify_spec(spec, configuration, review):
    c.require(spec["owner"] == c.OWNER and spec["phase"] in ("datatailor32", "tiny8_collection"),
              "Only these two finite owned prerequisites are admitted")
    c.require(spec["schema"] == ("s7-native-feature-qualification-spec-v1" if spec["phase"] == "datatailor32"
                                else "s7-tiny8-raw-collection-spec-v2"), "Wrong prerequisite schema")
    wrapper = Path(spec["wrapper"])
    c.require(not wrapper.is_absolute() and ".." not in wrapper.parts and str(wrapper) in spec["sources"],
              "The exact wrapper must belong to the frozen source packet")
    c.require(spec["code_root"] == str(c.ROOT), "Wrong immutable prerequisite source root")
    expected = dict(gpus=1, cpus=8, memory_mb=65536, minutes=55) if spec["phase"] == "datatailor32" else (
        dict(gpus=1, cpus=2, memory_mb=16384, minutes=10))
    c.require(spec["allocation"] == expected, "Unreviewed prerequisite resource envelope")
    c.require(spec["site"] == ("babel" if spec["phase"] == "datatailor32" else "aces"),
              "This version binds exact existing site/runtime assets")
    c.verify_sources(spec["sources"])
    c.verify_sources(spec["dependencies"])
    c.require(review["status"] == "PASS_EXACT_S7_PREREQUISITE_SOURCE_AND_CONFIG"
              and review["configuration"] == configuration and review["sources"] == spec["sources"]
              and review["dependencies"] == spec["dependencies"], "Exact independent source/config acceptance required")


def readback(binding, spec):
    fields = dict(re.findall(r"(\w+)=(\S+)", call(["scontrol", "show", "job", "-o", binding["job_id"]])))
    profile, allocation = route(spec), spec["allocation"]
    intent = c.checked(binding["intent"])
    expected = dict(JobId=binding["job_id"], JobName=spec["job_name"], Account=profile["account"],
                    Partition=profile["partition"], QOS=profile["qos"], WorkDir=str(c.ROOT),
                    Command=str(c.ROOT / spec["wrapper"]), NumTasks="1",
                    NumCPUs=str(allocation["cpus"]), TimeLimit=clock(allocation["minutes"]),
                    TresPerNode="gres/gpu:" + profile["gpu_type"] + ":1",
                    Comment="dsmm-s7-native:" + intent["nonce"])
    c.require(all(fields.get(key) == value for key, value in expected.items()), "Actual prerequisite binding/resources differ")
    c.require(fields.get("NumNodes") in ("1", "1-1") and fields["UserId"].split("(")[0] == profile["user"],
              "Actual prerequisite user/nodes differ")
    tres = dict(item.split("=", 1) for item in fields["ReqTRES"].split(",") if "=" in item)
    memory = re.fullmatch(r"(\d+)([MG])", tres.get("mem", ""))
    c.require(memory is not None and int(memory[1]) * (1024 if memory[2] == "G" else 1) == allocation["memory_mb"]
              and tres.get("gres/gpu") == "1", "Actual prerequisite RAM/GPU differs")
    spool = subprocess.check_output(["scontrol", "write", "batch_script", binding["job_id"], "-"], timeout=30)
    c.require(spool == (c.ROOT / spec["wrapper"]).read_bytes(), "Accepted prerequisite wrapper differs")
    return fields


def actual(binding, spec):
    c.require(binding["owner"] == c.OWNER and binding["job_id"] == c.os.environ.get("SLURM_JOB_ID")
              and not c.os.environ.get("SLURM_ARRAY_JOB_ID"), "Exact owned scalar allocation required")
    fields = readback(binding, spec)
    c.require(fields["JobState"] == "RUNNING" and fields["NodeList"] == socket.gethostname().split(".")[0],
              "Actual running compute node differs")
    restart = fields["Restarts"]
    c.require(restart.isdecimal() and c.os.environ.get("SLURM_RESTART_COUNT", restart) == restart,
              "Stale restart identity")
    c.os.environ["SLURM_RESTART_COUNT"] = restart
    return dict(job_id=binding["job_id"], restart=int(restart), node=fields["NodeList"],
                start=fields["StartTime"], allocated_gpus=1, allocated_cpus=spec["allocation"]["cpus"])


def _finish(binding, spec, directory):
    fields = readback(binding, spec)
    evidence = directory / ("readback-" + uuid.uuid4().hex)
    c.seal(evidence / "BEFORE.json", fields)
    if spec["site"] == "babel":
        c.seal(evidence / "WITHHELD.json", dict(
            job_id=binding["job_id"], reason="ROOT_TWO_GENERAL_RESERVATION_AND_LIVE_HEADROOM_REQUIRE_SEPARATE_RELEASE",
            automatic_release=False, no_other_job_control=True))
    elif fields["JobState"] == "PENDING" and fields["Reason"] == "JobHeldUser":
        c.require(fields["RunTime"] == "00:00:00" and fields["Restarts"] == "0"
                  and fields.get("AllocTRES") in (None, "(null)"), "Release only a verified never-started owned request")
        c.seal(evidence / "RELEASE_INTENT.json", dict(job_id=binding["job_id"], UTC=c.now()))
        response = subprocess.run(["scontrol", "release", binding["job_id"]], capture_output=True, text=True, timeout=30)
        c.seal(evidence / "RELEASE_RESPONSE.json", dict(rc=response.returncode, stdout=response.stdout, stderr=response.stderr))
        c.require(response.returncode == 0, "Owned prerequisite release unacknowledged")
    after = call(["scontrol", "show", "job", "-o", binding["job_id"]])
    c.seal(evidence / "AFTER.json", dict(raw=after, UTC=c.now()))
    return dict(binding=binding, actual=after, Babel_release_withheld=spec["site"] == "babel")


def submit(spec_path, review_path, registration):
    configuration = c.desc(spec_path)
    spec, review = c.checked(configuration), c.read(review_path)
    verify_spec(spec, configuration, review)
    directory = Path(registration)
    directory.mkdir(parents=True, exist_ok=True)
    profile, allocation = route(spec), spec["allocation"]
    with Path(spec["admission_lock"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        binding_path = directory / "BINDING.json"
        if binding_path.exists():
            binding = c.read(binding_path)
            c.require(binding["configuration"] == configuration and binding["review"] == c.desc(review_path),
                      "A different prerequisite is already bound")
            return _finish(binding, spec, directory)
        queue = call(["squeue", "-r", "-h", "-u", profile["user"], "-q", "normal", "-o", "%i|%j|%T|%b|%k"])
        intent_path = directory / "INTENT.json"
        if intent_path.exists():
            intent_description = c.desc(intent_path)
            intent = c.checked(intent_description)
            c.require(intent["configuration"] == configuration and intent["review"] == c.desc(review_path),
                      "Unresolved intent belongs to a different prerequisite")
            ack_path = directory / "ACK.json"
            if ack_path.exists():
                ack = c.read(ack_path)
                c.require(ack["intent"] == intent_description, "Existing ACK belongs to another intent")
                job = ack["job_id"]
            else:
                matches = [row.split("|")[0] for row in queue.splitlines()
                           if row.split("|")[1] == spec["job_name"]
                           and row.split("|")[4] == "dsmm-s7-native:" + intent["nonce"]]
                c.require(len(matches) == 1, "Unresolved RPC intent: reconcile exact scheduler acknowledgement, never resubmit")
                job = matches[0]
                c.seal(ack_path, dict(job_id=job, intent=intent_description, reconciled_from_actual_queue=True, UTC=c.now()))
        else:
            c.require(not any(row.split("|")[1] == spec["job_name"] for row in queue.splitlines()),
                      "Exact prerequisite already has a live request")
            qos = call(["sacctmgr", "-n", "-P", "show", "qos", "normal", "format=Name,MaxTRESPU,MaxSubmitPU"])
            rows = [row.split("|") for row in qos.splitlines() if row.startswith("normal|")]
            c.require(len(rows) == 1 and rows[0][2].isdecimal()
                      and len(queue.splitlines()) < int(rows[0][2]), "Actual normal-QoS submission limit reached or unknown")
            nonce = uuid.uuid4().hex
            argv = ["sbatch", "--parsable", "--hold", "--requeue", "--nodes=1", "--ntasks=1",
                    "--account=" + profile["account"], "--partition=" + profile["partition"], "--qos=normal",
                    "--gres=gpu:" + profile["gpu_type"] + ":1", "--cpus-per-task=" + str(allocation["cpus"]),
                    "--mem=" + str(allocation["memory_mb"]) + "M", "--time=" + str(allocation["minutes"]),
                    "--open-mode=append", "--chdir=" + str(c.ROOT), "--job-name=" + spec["job_name"],
                    "--comment=dsmm-s7-native:" + nonce, "--output=" + str(directory / "job-%j.out"),
                    str(c.ROOT / spec["wrapper"]), str(binding_path)]
            if spec["phase"] == "datatailor32":
                argv.insert(3, "--signal=B:USR1@300")
            intent_description = c.seal(intent_path, dict(
                owner=c.OWNER, nonce=nonce, function=spec["function"], configuration=configuration,
                review=c.desc(review_path), argv=argv, actual_normal_queue=queue, actual_qos=qos,
                UTC=c.now(), before_submission=True, no_student_submission=True))
            try:
                response = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired as error:
                c.seal(directory / "AMBIGUOUS_RPC.json", dict(intent=intent_description, error=str(error)))
                raise
            c.seal(directory / "RPC_RESPONSE.json", dict(rc=response.returncode, stdout=response.stdout, stderr=response.stderr))
            c.require(response.returncode == 0 and re.fullmatch(r"\d+(?:;[\w.-]+)?", response.stdout.strip()),
                      "Submission failed or ambiguous; preserve intent and reconcile")
            job = response.stdout.strip().split(";")[0]
            c.seal(directory / "ACK.json", dict(job_id=job, intent=intent_description, UTC=c.now()))
        binding = dict(schema="s7-native-feature-qualification-binding-v1", owner=c.OWNER, job_id=job,
                       configuration=configuration, review=c.desc(review_path), intent=intent_description)
        c.seal(binding_path, binding)
        return _finish(binding, spec, directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    print(submit(args.configuration, args.review, args.registration))
