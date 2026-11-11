"""One separately reviewed H100 job; rematerialize aliases, then genuine same-job restart."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import uuid

from . import protocol as p


def actual(binding, study, config, runner=subprocess.check_output):
    profile = p.checked(config["site_profile"])
    job = os.environ.get("SLURM_JOB_ID", "")
    p.require(job.isdecimal() and binding["schema"] == "h100-native-engineering-binding-v1"
              and binding["owner"] == "Root" and binding["job_id"] == job
              and not os.environ.get("SLURM_ARRAY_JOB_ID"), "Exact scalar H100 binding required")
    intent = p.checked(binding["intent"])
    p.require(intent["status"] == "INTENT_BEFORE_SUBMISSION" and intent["owner"] == "Root"
              and intent["design"] == study["design"] and intent["driver_sources"] == study["driver_sources"]
              and intent["site_profile"] == config["site_profile"] and intent["allocation"] == config["allocation"],
              "Durable exact H100 submission intent required")
    raw = runner(["scontrol", "show", "job", "-o", job], text=True)
    fields = dict(re.findall(r"(?:^|\s)([^\s=]+)=(\S+)", raw))
    expected = {
        "JobId": job, "JobName": "dsmm-h100-eng-" + study["id"], "JobState": "RUNNING",
        "Account": profile["account"], "Partition": profile["partition"], "QOS": profile["qos"],
        "NumNodes": "1", "NumTasks": "1", "WorkDir": profile["code_root"],
        "Command": str(p.HERE / "run.sbatch"), "Comment": "dsmm-h100-eng:" + intent["nonce"],
        "TresPerNode": "gres/gpu:h100:1",
    }
    for name, value in expected.items():
        p.require(fields.get(name) == value, "Actual H100 scheduler field differs: " + name)
    p.require(fields.get("UserId", "").split("(")[0] == profile["slurm_user"]
              and fields.get("NodeList") == socket.gethostname().split(".")[0]
              and fields.get("StartTime") not in (None, "", "Unknown", "N/A")
              and fields.get("Restarts", "").isdecimal(), "Unknown physical H100 attempt")
    restart = fields["Restarts"]
    supplied = os.environ.get("SLURM_RESTART_COUNT")
    p.require(supplied is None or supplied == restart, "Stale H100 restart environment")
    os.environ["SLURM_RESTART_COUNT"] = restart
    resources = config["allocation"]
    p.require(fields.get("NumCPUs") == fields.get("CPUs/Task") == str(resources["cpus"])
              and fields.get("TimeLimit") == "%02d:%02d:00" % divmod(resources["minutes"], 60),
              "Actual H100 CPU/walltime request differs")
    tres = dict(item.split("=", 1) for item in fields.get("ReqTRES", "").split(",") if "=" in item)
    p.require(tres.get("gres/gpu") == "1" and tres.get("mem") in
              (str(resources["memory_mb"]) + "M", str(resources["memory_mb"] // 1024) + "G"),
              "Actual H100 GPU/memory request differs")
    p.require(p.within(study["report"], profile["registry_root"]), "H100 report outside owned registry")
    return profile, raw, fields


def invoke(study, study_ref, arm, phase, logical, profile, attempt, stop_at, current_stop, fields):
    from experiments.root_native_media_transport_20261023 import transport
    attempt.mkdir(parents=True, exist_ok=False)
    current_stop[0] = attempt / "STOP_REQUESTED"
    selected = p.checked(logical["providers"]["selection"])
    started = time.monotonic()
    alias_ref = transport.prepare(logical["providers"]["pool"], selected["records_json"],
                                  logical["providers"]["media_transport"], attempt / "transport")
    modules, _ = p.base.target_modules(study["native_target"])
    physical = p.physical_config(logical, alias_ref, validate_native=modules["config"].validate,
                                 run_identity=modules["common"].run_identity)
    physical_ref = p.seal(attempt / "PHYSICAL_CONFIG.json", physical)
    p.require(stop_at - time.time() > study["minimum_launch_seconds"],
              "Transport consumed launch reserve; preserve aliases, do not relabel a partial reference")
    env = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HOME=profile["cache_root"],
        HF_HUB_CACHE=str(Path(profile["cache_root"]) / "hub"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS=str(logical["allocation"]["cpus"]),
        NATIVE_STOP_PATH=str(current_stop[0]), NATIVE_ATTEMPT=str(attempt),
        NATIVE_RUN_IDENTITY=p.identity(physical, modules),
        H100_ENGINEERING_NODE=fields["NodeList"], H100_ENGINEERING_START=fields["StartTime"],
    )
    command = [
        profile["python"], "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", "-m",
        "experiments.root_h100_native_qualification_v10_20261024.child",
        "--study", study_ref["path"], "--arm", arm, "--phase", phase,
        "--physical-config", physical_ref["path"], "--alias", alias_ref["path"],
        "--attempt", str(attempt), "--stop-at-epoch", str(stop_at),
    ]
    p.seal(attempt / "INVOCATION.json", {
        "command": command, "study": study_ref, "logical_configuration": study["native_configurations"][arm],
        "physical_configuration": physical_ref, "alias": alias_ref, "arm": arm, "phase": phase,
        "actual_job": os.environ["SLURM_JOB_ID"], "actual_restart": int(os.environ["SLURM_RESTART_COUNT"]),
        "actual_node": fields["NodeList"], "actual_start": fields["StartTime"], "UTC": p.now(),
    })
    with (attempt / "torchrun.log").open("xb", buffering=0) as log:
        code = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT).wait()
    current_stop[0] = None
    result = attempt / "CHILD_RESULT.json"
    p.seal(attempt / "SUPERVISOR_RESULT.json", {
        "rc": code, "seconds_including_transport": time.monotonic() - started,
        "alias": alias_ref, "child_result": p.desc(result) if result.exists() else None,
    })
    p.require(code == 0 and result.exists(), "Actual H100 failure; inspect raw evidence before versioned repair")
    return p.desc(result)


def cached_phase(descriptor, phase, study, job):
    value = p.checked(descriptor)
    arm = "reference" if phase == "reference" else "resume"
    expected = "ACCEPTED_NATIVE_CONTINUATION" if phase == "interrupted" else "NATIVE_TRAJECTORY_COMPLETE"
    p.require(value["error"] is None and value["phase"] == phase and value["status"] == expected
              and value["hardware"] == "H100" and value["study_id"] == study["id"]
              and value["driver_sources"] == study["driver_sources"] and value["native_target"] == study["native_target"]
              and value["native_configuration"] == study["native_configurations"][arm]
              and value["actual_job_id"] == job, "Foreign or incomplete cached H100 phase")
    return value


def _run(binding, context):
    study = p.checked(binding["study"])
    preliminary = p.checked(study['native_configurations']['reference'])
    profile, raw, fields = actual(binding, study, preliminary)
    from experiments.root_native_qualification_v10_20261024.preflight import runtime_preflight
    runtime = runtime_preflight(study, site='aces')
    p.seal(Path(study['report'])/'runtime-preflight'/
           (binding['job_id']+'-r'+os.environ['SLURM_RESTART_COUNT']+'.json'), runtime)
    configs, modules = p.configs(study)
    report = Path(study["report"])
    report.mkdir(parents=True, exist_ok=True)
    with (report / "EXECUTION.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = report / "attempts" / (
            binding["job_id"] + "-r" + os.environ["SLURM_RESTART_COUNT"] + "-" + uuid.uuid4().hex[:10]
        )
        attempt.mkdir(parents=True, exist_ok=False)
        context.update(attempt=attempt, actual_slurm=raw, binding=binding)
        p.seal(attempt / "START.json", {"binding": binding, "actual_slurm": raw, "UTC": p.now()})
        stop = [None]
        def stop_requested(*_):
            if stop[0] is not None:
                stop[0].touch()
        signal.signal(signal.SIGUSR1, stop_requested)
        signal.signal(signal.SIGTERM, stop_requested)
        stop_at = int(os.environ["SLURM_JOB_END_TIME"]) - configs["reference"]["allocation"]["reserve_seconds"]
        p.require(stop_at > time.time(), "Actual H100 allocation reserve exhausted")
        if (report / "RAW_READOUT.json").exists():
            prior = p.checked(p.desc(report / "RAW_READOUT.json"))
            p.require(prior["study_identity"] == p.key({k: v for k, v in study.items() if k != "review"})
                      and prior["status"] == "PASS_H100_TRANSPORT_ENGINEERING_READOUT_UNREVIEWED",
                      "Existing readout is foreign or failed; preserve evidence and diagnose")
            return
        reference = report / "PHASE_REFERENCE.json"
        interrupted = report / "PHASE_INTERRUPTED.json"
        resumed = report / "PHASE_RESUMED.json"
        p.require(reference.exists() or not (interrupted.exists() or resumed.exists()), "Out-of-order H100 phase state")
        p.require(interrupted.exists() or not resumed.exists(), "Resumed H100 phase lacks accepted cut")
        descriptors = {}
        for arm, phase, receipt_name in (
            ("reference", "reference", "PHASE_REFERENCE.json"),
            ("resume", "interrupted", "PHASE_INTERRUPTED.json"),
            ("resume", "resumed", "PHASE_RESUMED.json"),
        ):
            receipt = report / receipt_name
            if receipt.exists():
                descriptors[phase] = p.read(receipt)
                cached_phase(descriptors[phase], phase, study, binding["job_id"])
                continue
            if phase == "resumed":
                before = cached_phase(descriptors["interrupted"], "interrupted", study, binding["job_id"])
                p.require(before["actual_job_id"] == binding["job_id"]
                          and before["actual_restart_count"] < int(os.environ["SLURM_RESTART_COUNT"])
                          and before["actual_start"] != fields["StartTime"],
                          "Real same-job physical restart required before resumed arm")
            result = invoke(study, binding["study"], arm, phase, configs[arm], profile,
                            attempt / phase, stop_at, stop, fields)
            value = p.checked(result)
            expected = "ACCEPTED_NATIVE_CONTINUATION" if phase == "interrupted" else "NATIVE_TRAJECTORY_COMPLETE"
            p.require(value["status"] == expected, "Wrong actual H100 phase endpoint; no success-shaped continuation")
            p.seal(receipt, result)
            descriptors[phase] = result
            if phase == "interrupted":
                p.require(not (report / "REQUEUE_INTENT.json").exists(),
                          "Ambiguous H100 requeue requires reconciliation, not another RPC")
                intent = p.seal(report / "REQUEUE_INTENT.json", {
                    "job_id": binding["job_id"], "actual_start": fields["StartTime"],
                    "restart": int(os.environ["SLURM_RESTART_COUNT"]), "interrupted": result, "same_job_only": True,
                })
                response = subprocess.run(["scontrol", "requeue", binding["job_id"]], capture_output=True, text=True)
                p.seal(report / "REQUEUE_RESPONSE.json", {
                    "intent": intent, "rc": response.returncode, "stdout": response.stdout, "stderr": response.stderr,
                })
                p.require(response.returncode == 0, "H100 requeue acknowledgement unresolved; no new successor")
                return
        from .readout import compare
        phase_ref = p.seal(report / "PHASE_RESULTS.json", descriptors)
        try:
            result = compare(study, descriptors)
        except BaseException as error:
            p.seal(attempt / "READOUT_FAILURE.json", {"error": repr(error), "phases": phase_ref, "UTC": p.now()})
            raise
        p.seal(report / "RAW_READOUT.json", result)
        p.require(result["status"] == "PASS_H100_TRANSPORT_ENGINEERING_READOUT_UNREVIEWED",
                  "Actual H100 parity failed; preserve and diagnose before any repair")


def run(binding):
    started, context = time.monotonic(), {}
    try:
        result = _run(binding, context)
    except BaseException as error:
        if "attempt" in context:
            p.seal(context["attempt"] / "FAILURE.json", {
                "status": "H100_EXECUTION_OR_QUALIFICATION_FAILURE_NOT_METHOD_NEGATIVE",
                "error": repr(error), "actual_slurm": context["actual_slurm"], "binding": binding,
                "supervisor_seconds_including_validation_transport_and_readout": time.monotonic() - started,
                "scheduler_startup_prior_attempts_and_setup_costs_must_be_added": True,
            })
        raise
    if "attempt" in context:
        p.seal(context["attempt"] / "RESULT.json", {
            "status": "SCOPED_H100_ATTEMPT_RETURNED_NOT_PROMOTED",
            "supervisor_seconds_including_validation_transport_and_readout": time.monotonic() - started,
            "scheduler_startup_prior_attempts_and_setup_costs_must_be_added": True,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    run(p.read(args.binding))


if __name__ == "__main__":
    main()
