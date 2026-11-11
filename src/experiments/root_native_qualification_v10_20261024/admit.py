"""One reviewed General qualifier, exact held ACK/spool and same-job requeue."""
import argparse
import fcntl
from pathlib import Path
import re
import subprocess
import uuid

from . import common as c


def call(argv):
    return subprocess.check_output(argv, text=True, timeout=45)


def submit(study_ref, registration):
    study = c.checked(study_ref)
    c.require(study["driver_sources"] == c.sources(), "Exact reviewed V7 source changed")
    review = c.checked(study["review"])
    c.require(review["status"] == "PASS_INDEPENDENT_ENGINEERING_DRIVER_AND_STUDY"
              and review["driver_sources"] == study["driver_sources"]
              and review["study_identity"] == c.key({k:v for k,v in study.items() if k!="review"}),
              "Genuine exact independent study review required")
    config = c.checked(study["native_configurations"]["reference"])
    profile = c.checked(config["site_profile"])
    c.require(profile["site"] == "babel" and profile["partition"] == "general"
              and profile["gpu_type"] == "L40S", "Only the current General V7 route is admitted")
    from .preflight import verify_shared
    verify_shared(study, config, raw=False)
    directory = Path(registration)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path=Path(study["admission_lock"])
    c.require(lock_path.is_absolute() and lock_path.is_file()
              and lock_path.name=="ADMISSION.lock","Existing canonical shared admission lock required")
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        c.require(not (directory/"INTENT.json").exists(),
                  "Existing intent/ACK must be reconciled, never silently returned or resubmitted")
        queue = call(["squeue","-r","-h","-u",profile["slurm_user"],"-o","%i|%j|%T|%q|%b"])
        c.require(not any(line.split("|")[1] == "dsmm-eng-"+study["id"] for line in queue.splitlines()),
                  "An exact named qualifier is already live")
        limits = call(["sacctmgr","-nP","show","qos",profile["qos"],
                       "format=Name,MaxSubmitPU,MaxTRESPU,MaxJobsPU"])
        rows = [line.split("|") for line in limits.splitlines() if line.startswith(profile["qos"]+"|")]
        c.require(len(rows)==1, "Actual site/account limits unavailable")
        ceiling = rows[0][1]
        active = sum(line.split("|")[3] == profile["qos"] for line in queue.splitlines())
        c.require(not ceiling.isdecimal() or active < int(ceiling), "Actual submission limit reached")
        allocation=config["allocation"]
        nonce=uuid.uuid4().hex
        binding_path=directory/"BINDING.json"
        argv=["sbatch","--parsable","--hold","--requeue","--nodes=1","--ntasks=1",
              "--account="+profile["account"],"--partition="+profile["partition"],"--qos="+profile["qos"],
              "--gres=gpu:L40S:1","--cpus-per-task="+str(allocation["cpus"]),
              "--mem="+str(allocation["memory_mb"])+"M","--time="+str(allocation["minutes"]),
              "--signal=B:USR1@"+str(allocation["reserve_seconds"]),"--open-mode=append",
              "--job-name=dsmm-eng-"+study["id"],"--comment=dsmm-eng:"+nonce,
              "--chdir="+str(c.ROOT),"--output="+str(directory/"job-%j.out"),
              str(c.HERE/"run.sbatch"),str(binding_path),profile["python"]]
        intent=c.seal(directory/"INTENT.json",dict(status="INTENT_BEFORE_SUBMISSION",owner="Root",
            operator="Session6",study=study_ref,driver_sources=study["driver_sources"],
            site_profile=config["site_profile"],allocation=allocation,nonce=nonce,argv=argv,
            actual_expanded_user_queue=queue,actual_site_limits=limits,UTC=c.now()))
        try:
            response=subprocess.run(argv,text=True,capture_output=True,timeout=60)
        except subprocess.TimeoutExpired as error:
            c.seal(directory/"AMBIGUOUS_RPC.json",dict(intent=intent,error=str(error),reconcile_required=True))
            raise
        c.seal(directory/"RPC_RESPONSE.json",dict(intent=intent,rc=response.returncode,
            stdout=response.stdout,stderr=response.stderr))
        c.require(response.returncode==0 and re.fullmatch(r"\d+(?:;[\w.-]+)?\s*",response.stdout),
                  "Failed/ambiguous ACK; reconcile exact nonce, no blind retry")
        job=response.stdout.strip().split(";")[0]
        c.seal(directory/"ACK.json",dict(job_id=job,intent=intent,UTC=c.now()))
        binding=c.seal(binding_path,dict(schema="native-engineering-binding-v10",owner="Root",
            operator="Session6",job_id=job,study=study_ref,intent=intent))
        raw=call(["scontrol","show","job",job,"-o"])
        from .supervisor import parse_job
        fields=parse_job(raw)
        expected=dict(JobId=job,JobName="dsmm-eng-"+study["id"],JobState="PENDING",Reason="JobHeldUser",
            Restarts="0",RunTime="00:00:00",Account=profile["account"],Partition="general",QOS=profile["qos"],
            NumCPUs=str(allocation["cpus"]),NumTasks="1",WorkDir=str(c.ROOT),Command=str(c.HERE/"run.sbatch"),
            Comment="dsmm-eng:"+nonce,TresPerNode="gres/gpu:L40S:1",
            TimeLimit=f"{allocation['minutes']//60:02d}:{allocation['minutes']%60:02d}:00")
        c.require(all(fields.get(k)==v for k,v in expected.items())
                  and fields.get("AllocTRES") in (None,"(null)")
                  and fields.get("NumNodes") in ("1","1-1")
                  and fields.get("UserId","").split("(",1)[0]==profile["slurm_user"],
                  "Actual held request user/time/nodes/resources differs")
        memory=dict(item.split("=",1) for item in fields["ReqTRES"].split(",") if "=" in item).get("mem")
        c.require(memory in (str(allocation["memory_mb"])+"M",str(allocation["memory_mb"]//1024)+"G"),
                  "Actual held memory differs")
        spool=subprocess.check_output(["scontrol","write","batch_script",job,"-"],timeout=30)
        c.require(spool==(c.HERE/"run.sbatch").read_bytes(), "Held spool differs from reviewed V7 wrapper")
        c.seal(directory/"HELD_READBACK.json",dict(job_id=job,raw=raw,spool_sha256=c.sha(c.HERE/"run.sbatch")))
        c.seal(directory/"RELEASE_INTENT.json",dict(job_id=job,binding=binding,UTC=c.now()))
        release=call(["scontrol","release",job])
        c.seal(directory/"RELEASE_ACK.json",dict(job_id=job,stdout=release,UTC=c.now()))
        c.seal(directory/"RELEASE_READBACK.json",dict(job_id=job,raw=call(["scontrol","show","job",job,"-o"]),UTC=c.now()))
        return dict(job_id=job,binding=binding,status="SUBMITTED_RELEASED_NOT_COMPLETED")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--study",required=True)
    parser.add_argument("--registration",required=True)
    args=parser.parse_args()
    print(c.canonical(submit(c.desc(args.study),args.registration)).decode())
