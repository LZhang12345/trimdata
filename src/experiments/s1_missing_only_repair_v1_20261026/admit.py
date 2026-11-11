"""A finite S1 repair DAG; canonical admission, source custody and no auto retry."""

import argparse
import fcntl
import os
import re
import subprocess
import uuid

from . import worker as w
from experiments.s1_parallel_prerequisites_v2_20261026 import admit as prior


def submit(plan_ref, review_ref):
    w.base.require(os.getuid() == 2708899 and str(w.ROOT) == w.base.checked(plan_ref)["remote_source_root"],
                   "Actual private Babel repair actor required")
    w.load_plan(plan_ref, review_ref)
    lock = w.base.literal(w.LOCK)
    with lock.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        w.load_plan(plan_ref, review_ref)
        w.base.require(not w.CONTROL.exists(), "Existing repair intent/ACK must be reconciled")
        census = prior.census()
        w.base.require(not any("dsmm-s1-repair-" in row for row in census["raw_expanded"].splitlines()),
                       "Existing live S1 repair requires reconciliation")
        w.base.require(not any("dsmm-dt-union-qual" in row
                               or ("VQAv2" in row and "|RUNNING|" in row)
                               for row in census["raw_expanded"].splitlines()),
                       "Competing live DT union or VQAv2 generation owner")
        w.base.literal(w.CONTROL).mkdir(parents=True, exist_ok=False)
        batch = w.base.publish(w.CONTROL / "BATCH_INTENT.json",
                               dict(owner="Session1", plan=plan_ref, review=review_ref,
                                    units=list(w.UNITS), actual_census=census))
        bindings = {}
        for unit in w.UNITS:
            directory = w.CONTROL / unit
            directory.mkdir(exist_ok=False)
            parents = {}
            parent_refs = {}
            if unit == "dt16-union":
                parent_refs["dt3-header"] = bindings["dt3-header"]
                parents["dt3-header"] = w.base.checked(bindings["dt3-header"])["job_id"]
            nonce = uuid.uuid4().hex
            claim = w.base.publish(directory / "FUNCTION_CLAIM.json",
                                   dict(owner="Session1", unit=unit, plan=plan_ref,
                                        no_duplicate_live_provider=True, paid_sources_read_only=True))
            argv = w.submission_argv(unit, nonce, parents)
            intent = w.base.publish(directory / "INTENT.json",
                                    dict(batch=batch, claim=claim, argv=argv, nonce=nonce, parents=parents))
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith(("SLURM_", "SBATCH_", "SRUN_"))}
            try:
                reply = subprocess.run(argv, capture_output=True, text=True, timeout=60, env=environment)
            except (OSError, subprocess.TimeoutExpired) as error:
                w.base.publish(directory / "AMBIGUOUS_SUBMISSION.json",
                                dict(intent=intent, error_class=type(error).__name__, automatic_retry=False))
                raise
            response = w.base.publish(directory / "SUBMISSION_RESPONSE.json",
                                      dict(intent=intent, returncode=reply.returncode,
                                           stdout=reply.stdout, stderr=reply.stderr))
            match = re.fullmatch(r"(\d+)(?:;[A-Za-z0-9_-]+)?\s*", reply.stdout)
            w.base.require(reply.returncode == 0 and match is not None,
                           "Ambiguous repair ACK; retain intent and never auto-resubmit")
            job = match[1]
            w.base.publish(directory / "ACK.json", dict(job_id=job, response=response))
            binding = dict(owner="Session1", unit=unit, job_id=job, nonce=nonce, plan=plan_ref,
                           review=review_ref, intent=intent, claim=claim, parents=parents,
                           parent_bindings=parent_refs)
            raw = w.scheduler(binding, False)
            w.base.publish(directory / "HELD_READBACK.json", dict(job_id=job, scheduler=raw))
            bindings[unit] = w.base.publish(directory / "BINDING.json", binding)
        for ref in bindings.values():
            w.scheduler(w.base.checked(ref), False)
        w.base.publish(w.CONTROL / "RELEASE_INTENT.json", dict(bindings=bindings))
        for unit, ref in bindings.items():
            binding = w.base.checked(ref)
            reply = subprocess.run(["scontrol", "release", binding["job_id"]],
                                   capture_output=True, text=True, timeout=30)
            w.base.publish(w.CONTROL / unit / "RELEASE_RESPONSE.json",
                            dict(job_id=binding["job_id"], returncode=reply.returncode,
                                 stdout=reply.stdout, stderr=reply.stderr))
            w.base.require(reply.returncode == 0, "Exact repair release failed; reconcile, no new job")
        return w.base.publish(w.CONTROL / "BATCH_ACK.json",
                               dict(owner="Session1", bindings=bindings, jobs_submitted=3,
                                    initial_requested_GPU_hours=2.0, models=0, forwards=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--review", required=True)
    args = parser.parse_args()
    plan, _ = w.base.snapshot(args.plan, 2 << 20)
    review, _ = w.base.snapshot(args.review, 2 << 20)
    print(submit(plan, review))
