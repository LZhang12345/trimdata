"""Submit one reviewed three-job prerequisite batch under the canonical lock."""

import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import uuid

from . import worker as w


def census(query=subprocess.check_output):
    raw = query(["squeue", "-r", "-h", "-u", "luyangz", "-o", "%i|%u|%q|%T|%b|%j"],
                text=True, timeout=30)
    rows = [line.split("|") for line in raw.splitlines() if line.strip()]
    w.require(all(len(row) == 6 and re.fullmatch(r"\d+(?:_\d+)?", row[0])
                  and row[1] == "luyangz" for row in rows)
              and len({row[0] for row in rows}) == len(rows), "Expanded all-user census required")
    limits = query(["sacctmgr", "-nP", "show", "qos", "normal",
                    "format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU"], text=True, timeout=30)
    values = [line.split("|") for line in limits.splitlines() if line.startswith("normal|")]
    w.require(len(values) == 1 and len(values[0]) >= 4, "Exact actual normal QoS required")
    _, tres, max_jobs, max_submit, *_ = values[0]
    resources = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
    w.require(resources.get("gres/gpu") == "8" and max_jobs == "10" and max_submit == "50",
              "Actual scheduler limits changed; re-review instead of inventing capacity")
    normal = [row for row in rows if row[2] == "normal"]
    w.require(len(normal) + len(w.UNITS) <= int(max_submit),
              "Insufficient real submission slots for the finite batch")
    w.require(not any(row[5] in {"dsmm-s1-prereq-" + name for name in w.UNITS} for row in rows),
              "Existing prerequisite job requires reconciliation, not duplicate submission")
    return dict(raw_expanded=raw, raw_qos=limits, normal_requests=len(normal),
                batch_requests=len(w.UNITS), private_GPU_cap=False)


def submit(plan_ref, review_ref, run=subprocess.run, query=subprocess.check_output):
    w.require(os.getuid() == 2708899, "Actual Babel user required before writes/RPC")
    plan = w.load_plan(plan_ref, review_ref)
    w.require(Path(plan["remote_source_root"]) == w.ROOT and Path.cwd() == w.ROOT,
              "Private staged source root required")
    lock = w.literal(w.LOCK)
    w.require(lock.is_file(), "Existing canonical admission lock required")
    with lock.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        w.load_plan(plan_ref, review_ref)
        w.require(not w.CONTROL.exists(), "Existing batch intent/ACK requires reconciliation")
        current = census(query)
        w.literal(w.CONTROL)
        w.CONTROL.mkdir(parents=True, exist_ok=False)
        intent = w.publish(w.CONTROL / "BATCH_INTENT.json",
                           dict(owner="Session1", plan=plan_ref, review=review_ref,
                                units=list(w.UNITS), census=current, models=0, forwards=0,
                                output_limit_bytes=3 * plan["output_limit_bytes_per_unit"]))
        bindings = {}
        for unit in w.UNITS:
            directory = w.CONTROL / unit
            directory.mkdir(exist_ok=False)
            nonce = uuid.uuid4().hex
            name = "dsmm-s1-prereq-" + unit
            function = w.digest(w.canonical(dict(scope=unit, inputs=plan["function_inputs"][unit])))
            claim = w.publish(directory / "FUNCTION_CLAIM.json",
                              dict(owner="Session1", scope=unit, function_identity=function,
                                   plan=plan_ref, models=0, forwards=0,
                                   no_provider_payload_mutation=True))
            argv = w.submission_argv(unit, nonce)
            unit_intent = w.publish(directory / "INTENT.json",
                                    dict(batch_intent=intent, claim=claim, nonce=nonce, argv=argv))
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith(("SLURM_", "SBATCH_", "SRUN_"))}
            try:
                response = run(argv, capture_output=True, text=True, timeout=60, env=environment)
            except (OSError, subprocess.TimeoutExpired) as error:
                w.publish(directory / "AMBIGUOUS_SUBMISSION.json",
                          dict(intent=unit_intent, error_class=type(error).__name__, automatic_retry=False))
                raise
            reply = w.publish(directory / "SUBMISSION_RESPONSE.json",
                              dict(intent=unit_intent, returncode=response.returncode,
                                   stdout=response.stdout, stderr=response.stderr))
            match = re.fullmatch(r"(\d+)(?:;[A-Za-z0-9_-]+)?\s*", response.stdout)
            w.require(response.returncode == 0 and match is not None,
                      "Unknown submission ACK; reconcile durable intent, never auto-resubmit")
            job = match[1]
            w.publish(directory / "ACK.json", dict(job_id=job, response=reply, nonce=nonce))
            binding = dict(owner="Session1", unit=unit, job_id=job, nonce=nonce, job_name=name,
                           plan=plan_ref, review=review_ref, intent=unit_intent, claim=claim)
            raw = w.scheduler(binding, False, query)
            w.publish(directory / "HELD_READBACK.json", dict(job_id=job, scheduler=raw))
            bindings[unit] = w.publish(directory / "BINDING.json", binding)
        w.load_plan(plan_ref, review_ref)
        for ref in bindings.values():
            w.scheduler(w.checked(ref), False, query)
        w.publish(w.CONTROL / "RELEASE_INTENT.json", dict(bindings=bindings))
        for unit, ref in bindings.items():
            binding = w.checked(ref)
            response = run(["scontrol", "release", binding["job_id"]],
                           capture_output=True, text=True, timeout=30)
            w.publish(w.CONTROL / unit / "RELEASE_RESPONSE.json",
                      dict(job_id=binding["job_id"], returncode=response.returncode,
                           stdout=response.stdout, stderr=response.stderr))
            w.require(response.returncode == 0, "Release failed; reconcile exact job, no new submission")
        return w.publish(w.CONTROL / "BATCH_ACK.json",
                         dict(owner="Session1", bindings=bindings, jobs_submitted=3,
                              GPUs_per_job=1, initial_requested_GPU_hours=0.75,
                              models=0, forwards=0, scientific_round_complete=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--review", required=True)
    arguments = parser.parse_args()
    plan_ref, _ = w.snapshot(arguments.plan, 2 << 20)
    review_ref, _ = w.snapshot(arguments.review, 2 << 20)
    print(submit(plan_ref, review_ref))
