"""One versioned schema repair; reuse the completed paid parent without a new dependency."""
import argparse
import fcntl
import os
import re
import subprocess
import uuid

from . import worker as w


def submit(plan_ref, review_ref):
    w.base.require(os.getuid()==2708899, "Actual S1 Babel actor required")
    plan=w.load_plan(plan_ref,review_ref)
    w.base.require(str(w.ROOT)==plan["remote_source_root"],"Exact private repaired root")
    with w.base.literal(w.LOCK).open("r+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        w.load_plan(plan_ref,review_ref)
        w.base.require(not w.CONTROL.exists(),"Existing repaired-union intent/ACK needs reconciliation")
        raw=subprocess.check_output(["squeue","-r","-h","-u","luyangz","-o","%i|%u|%q|%T|%b|%j"],text=True,timeout=30)
        rows=[line.split("|") for line in raw.splitlines() if line.strip()]
        w.base.require(all(len(row)==6 and row[1]=="luyangz" and re.fullmatch(r"\d+(?:_\d+)?",row[0])
                           for row in rows) and len({row[0] for row in rows})==len(rows),"Expanded actual census")
        limits=subprocess.check_output(["sacctmgr","-nP","show","qos","normal",
            "format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU"],text=True,timeout=30)
        matches=[line.split("|") for line in limits.splitlines() if line.startswith("normal|")]
        w.base.require(len(matches)==1 and matches[0][1:4]==["gres/gpu=8","10","50"]
                       and sum(row[2]=="normal" for row in rows)+1<=50,"Actual one-request capacity")
        w.base.require(not any("dsmm" in row[5].lower() and "union" in row[5].lower() for row in rows),
                       "Competing live union owner")
        parent=w.base.checked(plan["completed_parent_binding"])
        result=w.base.checked(plan["completed_parent_result"])
        w.base.require(parent["job_id"]=="10936649" and parent["plan"]==plan["completed_parent_plan"]
                       and result["job_id"]==parent["job_id"]
                       and result["status"]=="DT3_HEADER_AND_TYPED16_INPUTS_REPAIRED",
                       "Reuse the actual completed one-header repair only")
        directory=w.CONTROL/"dt16-union"
        w.base.literal(directory).mkdir(parents=True,exist_ok=False)
        nonce=uuid.uuid4().hex
        parents={"dt3-header":parent["job_id"]}
        argv=w.submission_argv("dt16-union",nonce,parents)
        claim=w.base.publish(directory/"FUNCTION_CLAIM.json",dict(owner="Session1",unit="dt16-union",
            plan=plan_ref,legacy_header_schema_repair=True,completed_parent=plan["completed_parent_binding"],
            no_duplicate_live_provider=True,paid_sources_read_only=True))
        intent=w.base.publish(directory/"INTENT.json",dict(argv=argv,claim=claim,nonce=nonce,
            expanded_census=raw,actual_qos=limits))
        env={key:value for key,value in os.environ.items() if not key.startswith(("SLURM_","SBATCH_","SRUN_"))}
        try:
            reply=subprocess.run(argv,capture_output=True,text=True,timeout=60,env=env)
        except (OSError,subprocess.TimeoutExpired) as error:
            w.base.publish(directory/"AMBIGUOUS_SUBMISSION.json",dict(intent=intent,
                error_class=type(error).__name__,automatic_retry=False))
            raise
        response=w.base.publish(directory/"SUBMISSION_RESPONSE.json",dict(returncode=reply.returncode,
            stdout=reply.stdout,stderr=reply.stderr))
        match=re.fullmatch(r"(\d+)(?:;[A-Za-z0-9_-]+)?\s*",reply.stdout)
        w.base.require(reply.returncode==0 and match is not None,"Ambiguous repaired union ACK; no retry")
        job=match[1]
        w.base.publish(directory/"ACK.json",dict(job_id=job,response=response))
        binding=dict(owner="Session1",unit="dt16-union",job_id=job,nonce=nonce,plan=plan_ref,review=review_ref,
            intent=intent,claim=claim,parents=parents,parent_bindings={"dt3-header":plan["completed_parent_binding"]})
        observed=w.scheduler(binding,False)
        w.base.publish(directory/"HELD_READBACK.json",dict(raw=observed))
        ref=w.base.publish(directory/"BINDING.json",binding)
        w.base.publish(w.CONTROL/"RELEASE_INTENT.json",dict(binding=ref))
        reply=subprocess.run(["scontrol","release",job],capture_output=True,text=True,timeout=30)
        w.base.publish(w.CONTROL/"RELEASE_RESPONSE.json",dict(returncode=reply.returncode,
            stdout=reply.stdout,stderr=reply.stderr))
        w.base.require(reply.returncode==0,"Exact repaired-union release failed; reconcile")
        return ref


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",required=True);parser.add_argument("--review",required=True)
    args=parser.parse_args()
    print(submit(w.base.snapshot(args.plan,2<<20)[0],w.base.snapshot(args.review,2<<20)[0]))
