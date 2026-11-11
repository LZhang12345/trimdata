"""Finite A job execution; accepted checkpoints, real physical identity and held continuations."""
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import subprocess
import time
from zoneinfo import ZoneInfo

from . import protocol as p,collect
from experiments.root_production_epoch_qualification_v1_20261024 import runtime

CAPSULE_INSPECTION_POLICY=dict(maximum_seconds=1800,before_remaining_reserve_seconds=750,
    after_remaining_reserve_seconds=180,minimum_attempt_seconds=30,
    deadline_source="FRESH_ACTUAL_SCHEDULER_ENDTIME",
    insufficient_or_timeout="PRESERVE_ACCEPTED_PREFIX_EXIT75_EXTERNAL_CONTINUATION_NO_REQUEUE",
    guard_body_unchanged=True)


class CapsuleInspectionDeferred(ValueError):
    def __init__(self,receipt):
        super().__init__("Capsule inspection not verified; external continuation required, no automatic retry")
        self.receipt=receipt


def inspection_budget(end_epoch,now,phase):
    p.require(phase in ("before","after"),"Explicit capsule inspection phase")
    reserve=CAPSULE_INSPECTION_POLICY[phase+"_remaining_reserve_seconds"]
    seconds=max(0,min(CAPSULE_INSPECTION_POLICY["maximum_seconds"],int(end_epoch-now-reserve)))
    return dict(phase=phase,actual_end_epoch=end_epoch,measured_epoch=now,actual_remaining_seconds=end_epoch-now,
        reserve_seconds=reserve,timeout_seconds=seconds,policy=CAPSULE_INSPECTION_POLICY)


def _text(value):
    return value.decode("utf-8",errors="replace") if isinstance(value,bytes) else (value or "")


def _deferred(output,configuration,actual,budget,reason,rpc=None):
    receipt=p.seal(str(output)+".DEFERRED.json",dict(
        status="CAPSULE_INSPECTION_NOT_VERIFIED_REQUIRES_EXTERNAL_CONTINUATION",
        configuration=configuration,actual=actual,budget=budget,reason=reason,rpc=rpc,
        verified_capsule_or_training_PASS=False,preserve_all_accepted_states_and_raw_evidence=True,
        scheduler_action_taken=False,automatic_retry=False,exit_status_required=75))
    raise CapsuleInspectionDeferred(receipt)


def external_continuation(attempt,configuration,actual,cursor,accepted,error,phase):
    p.seal(attempt/"EXTERNAL_CAPSULE_INSPECTION_REQUIRED.json",dict(
        status="ACCEPTED_PREFIX_PRESERVED_EXTERNAL_CAPSULE_INSPECTION_REQUIRED_NOT_TRAINING_COMPLETE",
        configuration=configuration,actual=actual,recorded_accepted_cursor=cursor,
        recorded_accepted_reference=accepted,inspection_deferral=error.receipt,phase=phase,
        accepted_capsule_reverified_by_this_guard=False,full_epoch_complete=False,
        scheduler_action_taken=False,automatic_retry=False,exit_status=75,preserve_all_states_and_receipts=True))
    raise SystemExit(75)


def fields(raw):
    rows=[line for line in raw.splitlines() if line.strip()]
    p.require(len(rows)==1,"One exact current scalar scheduler record")
    matches=list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=",rows[0]))
    p.require(matches and matches[0].start()==0,"Malformed scheduler record")
    result={}
    for i,match in enumerate(matches):
        p.require(match[1] not in result,"Duplicate scheduler key")
        result[match[1]]=rows[0][match.end():matches[i+1].start() if i+1<len(matches) else len(rows[0])].strip()
    return result


def wall(minutes):return "%02d:%02d:00"%divmod(minutes,60)


def activation_target(module):
    p.require(module==p.PACKAGE+".driver","Exact declared native driver module, independent of __main__ entry")
    return module


def current(binding_ref,activation_ref,runner=None,minimum_seconds=600):
    runner=subprocess.check_output if runner is None else runner
    b=p.checked(binding_ref);a=p.checked(activation_ref);request=p.checked(b["request"])
    cfg=p.checked(a["configuration"]);p.supported(cfg)
    p.require(os.getuid()==2708899 and pwd.getpwuid(os.getuid()).pw_name=="luyangz"
        and p.ROOT==p.SOURCE_ROOT and b["schema"]=="coincide-fixed10-native-v2-binding-v1"
        and b["owner"]=="Root" and b["sources"]==p.sources()
        and a["binding"]==binding_ref and activation_target(a["target_module"])==p.PACKAGE+".driver"
        and a["target_sources"]==p.sources()
        and b["configuration"]==a["configuration"]==request["configuration"],
        "Exact new owned A binding/activation/source, not an old scope")
    intent=p.checked(b["intent"]);ack=p.checked(b["ack"])
    p.require(intent["request"]==b["request"] and intent["nonce"]==b["nonce"]
        and ack["intent"]==b["intent"] and ack["job_id"]==b["job_id"]
        and ack["response"]["returncode"]==0 and ack["response"]["stdout"].strip().split(";")[0]==b["job_id"],
        "Actual original intent/rawACK, never a future ID")
    job=os.environ.get("SLURM_JOB_ID","");restart=os.environ.get("SLURM_RESTART_COUNT","0")
    p.require(job==b["job_id"] and job.isdecimal() and restart.isdecimal()
        and not os.environ.get("SLURM_ARRAY_JOB_ID"),"Actual current scalar A job/incarnation")
    raw=runner(["scontrol","show","job","-o",job],text=True,timeout=30);f=fields(raw)
    expected=dict(JobId=job,JobName=b["job_name"],JobState="RUNNING",UserId="luyangz(2708899)",
        Account="cx",Partition="general",QOS="normal",NumNodes="1",NumTasks="1",NumCPUs="8",
        TimeLimit=wall(request["outer_allocation"]["minutes"]),Command=b["command"]["path"],WorkDir=str(p.ROOT),
        Comment="dsmm-coincide-native:"+b["nonce"],TresPerNode="gres/gpu:L40S:1",Restarts=restart,
        NodeList=socket.gethostname().split(".")[0])
    p.require(all(f.get(key)==value for key,value in expected.items()),"Actual A site/resources/identity changed")
    tres=dict(item.split("=",1) for item in f["AllocTRES"].split(",") if "=" in item)
    p.require(tres.get("cpu")=="8" and tres.get("mem") in ("64G","65536M")
        and tres.get("gres/gpu")=="1","Actual allocated CPU8/64GiB/L40S1")
    p.require(b["command"]==p.desc(p.HERE/"run.sbatch") and b["spool"]==b["command"]
        and runner(["scontrol","write","batch_script",job,"-"],timeout=30)==(p.HERE/"run.sbatch").read_bytes(),
        "Exact immutable executed A spool")
    end=int(datetime.datetime.fromisoformat(f["EndTime"]).replace(tzinfo=ZoneInfo("America/New_York")).timestamp())
    p.require(end>time.time()+minimum_seconds
        and (not os.environ.get("SLURM_JOB_END_TIME") or int(os.environ["SLURM_JOB_END_TIME"])==end),
        "Actual scheduler deadline, not requested-wall guess")
    return b,request,dict(job_id=job,restart=int(restart),node=f["NodeList"],end_epoch=end,scheduler=raw),a


def accepted_cursor_guard(guard,cursor,continuation,configuration):
    p.require(guard["latest_step"]==cursor,"Actual accepted cursor differs from history")
    if cursor==0:
        p.require(continuation is None and guard["capsules"]==[],"Fresh A cannot inherit another state")
        return
    value=p.checked(continuation);archive=p.checked(value["accepted_archive"])
    matches=[row for row in guard["capsules"] if row["checkpoint"]==str(Path(value["accepted"]["path"]).parent)]
    p.require(len(matches)==1 and value["configuration"]==configuration and value["global_step"]==cursor
        and value["config_identity"]==p.n.run_identity(p.checked(configuration))
        and value["adapter_sources"]==p.sources() and matches[0]["accepted"]==value["accepted"]
        and value["accepted"]["bytes"]==value["accepted_archive"]["bytes"]
        and value["accepted"]["sha256"]==value["accepted_archive"]["sha256"]
        and p.checked(matches[0]["accepted"])==archive and archive["global_step"]==cursor
        and archive["identity"]==value["config_identity"],"Current17-state/header/history, never replay a prefix")


def execution(binding_ref,activation_ref,actual):
    b=p.checked(binding_ref);a=p.checked(activation_ref);cfg=p.checked(a["configuration"])
    root=Path(cfg["report"])/"ROOT_LONG"
    ref=p.desc(root/"requests"/("restart-"+str(actual["restart"])+".json"));e=p.checked(ref)
    p.require(e["schema"]=="coincide-fixed10-native-v2-execution-v1" and e["owner"]=="Root"
        and e["binding"]==binding_ref and e["activation"]==activation_ref
        and e["configuration"]==a["configuration"] and e["config_identity"]==p.n.run_identity(cfg)
        and e["actual_restart"]==actual["restart"] and e["sources"]==p.sources()
        and e["mode"] in ("shared_canary","production"),"Exact current A execution request")
    review=p.checked(e["review"])
    p.require(review["status"]=="PASS_INDEPENDENT_EXACT_COINCIDE_NATIVE_EXECUTION_POLICY_V2"
        and review["policy_identity"]==e["policy_identity"] and review["sources"]==p.sources()
        ,"Independent exact COINCIDE execution policy")
    p.independent(review)
    history=e["accepted_history"];p.require(len(history)==actual["restart"],"One progressing accepted segment per restart")
    cursor=0
    for h in history:
        value=p.checked(h);header=p.checked(value["accepted_archive"])
        p.require(value["configuration"]==a["configuration"] and value["actual_job_id"]==b["job_id"]
            and value["config_identity"]==p.n.run_identity(cfg) and value["adapter_sources"]==p.sources()
            and cursor<value["global_step"]<=cfg["expected_horizon"]["natural_updates"]
            and header["identity"]==value["config_identity"] and header["global_step"]==value["global_step"],
            "Foreign or nonprogressing accepted history")
        cursor=value["global_step"]
    if actual["restart"]:
        route=p.checked(p.desc(root/"routes"/("restart-"+str(actual["restart"])+".json")))
        intent=p.checked(route["intent"]);peer=p.checked(route["independent_actual_transition_review"])
        previous=p.checked(intent["previous_execution"]);cont=p.checked(intent["continuation"])
        p.require(route["execution"]==ref and intent["binding"]==binding_ref
            and intent["action"]=="requeuehold" and intent["from_restart"]==actual["restart"]-1
            and intent["to_restart"]==actual["restart"] and intent["next_request"]==ref
            and previous["actual_restart"]==actual["restart"]-1
            and previous["configuration"]==a["configuration"] and intent["continuation"]==history[-1]
            and cont["actual_job_id"]==b["job_id"]
            and peer["status"]=="PASS_INDEPENDENT_ACTUAL_COINCIDE_SUCCESSOR_RECONCILIATION_V2"
            and peer["intent"]==route["intent"] and peer["next_request"]==ref
            and peer["actual_job_id"]==b["job_id"] and peer["actual_restart"]==actual["restart"]
            and peer["original_ACK_absence_preserved"] is True
            ,
            "Every actual transition needs independent accounting; absent ACK is never manufactured")
        p.independent(peer)
    permit=p.checked(e["authorization"])
    p.require(permit["configuration"]==a["configuration"] and permit["physical_binding"]==binding_ref
        and permit["phase"]==e["mode"] and permit["policy_identity"]==e["policy_identity"],"Exact current Root authorization")
    return ref,e,cfg,cursor,root


def physical_check(binding_ref,activation_ref,cfg,output,*,phase="before"):
    b,_,actual,a=current(binding_ref,activation_ref,minimum_seconds=1)
    budget=inspection_budget(actual["end_epoch"],time.time(),phase)
    p.seal(str(output)+".BUDGET.json",dict(budget,configuration=a["configuration"],actual=actual))
    if budget["timeout_seconds"]<CAPSULE_INSPECTION_POLICY["minimum_attempt_seconds"]:
        _deferred(output,a["configuration"],actual,budget,"Insufficient actual parent lifetime for verified inspection")
    profile=p.checked(cfg["site_profile"])
    argv=[profile["python"],"-B","-m",p.PACKAGE+".capsule","--binding",binding_ref["path"],
        "--activation",activation_ref["path"],"--output",str(output)]
    began=time.time();budget=inspection_budget(actual["end_epoch"],began,phase)
    if budget["timeout_seconds"]<CAPSULE_INSPECTION_POLICY["minimum_attempt_seconds"]:
        _deferred(output,a["configuration"],actual,budget,"Actual lifetime expired during launch preparation")
    try:
        result=subprocess.run(argv,cwd=p.ROOT,env=runtime.launch_environment(os.environ),
            text=True,capture_output=True,timeout=budget["timeout_seconds"])
    except subprocess.TimeoutExpired as error:
        rpc=p.seal(str(output)+".RPC.json",dict(argv=argv,returncode=None,timed_out=True,
            timeout_seconds=budget["timeout_seconds"],stdout=_text(error.stdout),stderr=_text(error.stderr),
            inspection_budget=budget,actual_started_epoch=began,actual_return_epoch=time.time()))
        _deferred(output,a["configuration"],actual,budget,"Actual bounded inspection timeout; no verified result",rpc)
    p.seal(str(output)+".RPC.json",dict(argv=argv,returncode=result.returncode,stdout=result.stdout,
        stderr=result.stderr,inspection_budget=budget,actual_started_epoch=began,actual_return_epoch=time.time()))
    p.require(result.returncode==0,"Actual owned-GPU semantic capsule guard failed; retain raw failure")
    value=p.checked(p.desc(output))
    p.require(value["config_identity"]==p.n.run_identity(cfg)
        and value["physical_job_id"]==p.checked(binding_ref)["job_id"],"Exact actual current capsule proof")
    return value


def child_exit(code,path,configuration):
    p.require(code==0,"Unaccepted child/torchrun failure is not an automatic retry")
    ref=p.desc(Path(path)/"CHILD_EXIT.json");value=p.checked(ref)
    p.require(value["schema"]=="root-longwall-child-exit-v1" and value["configuration"]==configuration
        and value["config_identity"]==p.n.run_identity(p.checked(configuration)) and value["adapter_sources"]==p.sources()
        and value["torchrun_process_exit_code"]==0 and value["logical_exit_code"] in (0,75),
        "Only authenticated child success/accepted continuation")
    receipt=p.checked(value["receipt"])
    p.require(receipt["configuration"]==configuration and receipt["config_identity"]==value["config_identity"]
        and receipt["adapter_sources"]==p.sources()
        and receipt["status"]==("COMPLETE_STUDENT_TRAINING_NOT_EVALUATION" if value["logical_exit_code"]==0
            else "ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION"),"Exact logical receipt")
    return ref,value["logical_exit_code"]


def run_conditional_student(binding_ref,activation_ref):
    started=time.time();b,request,actual,a=current(binding_ref,activation_ref)
    eref,e,cfg,cursor,root=execution(binding_ref,activation_ref,actual)
    name=b["job_id"]+"-r"+str(actual["restart"])+"-"+b["nonce"][:8]
    attempt=root/"attempts"/name;attempt.mkdir(parents=True,exist_ok=False)
    try:guard=physical_check(binding_ref,activation_ref,cfg,attempt/"BEFORE_TRAINING_CAPSULES.json",phase="before")
    except CapsuleInspectionDeferred as error:
        external_continuation(attempt,a["configuration"],actual,cursor,
            e["accepted_history"][-1] if e["accepted_history"] else None,error,"before")
    accepted_cursor_guard(guard,cursor,e["accepted_history"][-1] if e["accepted_history"] else None,a["configuration"])
    childpath=Path(cfg["report"])/"attempts"/name/"child"
    launch=p.seal(attempt/"CHILD_BINDING.json",dict(schema="root-longwall-student-child-binding-v1",owner="Root",
        configuration=a["configuration"],authorization=e["authorization"],adapter_sources=p.sources(),
        physical_binding=binding_ref,activation=activation_ref,physical_job_id=b["job_id"],
        actual_restart_count=actual["restart"],nonce=b["nonce"],command=b["command"],child_attempt=str(childpath),
        current_storage_guard=p.desc(attempt/"BEFORE_TRAINING_CAPSULES.json"),accepted_prior_attempts=e["accepted_history"]))
    stop=actual["end_epoch"]-600;profile=p.checked(cfg["site_profile"])
    argv=[profile["python"],"-B","-m","torch.distributed.run","--standalone","--nproc_per_node=1",
        "-m",p.PACKAGE+".child","--configuration",a["configuration"]["path"],"--authorization",e["authorization"]["path"],
        "--launch-binding",launch["path"],"--attempt",str(childpath),"--stop-at-epoch",str(stop)]
    env=runtime.launch_environment(os.environ)
    env.update(NATIVE_RUN_IDENTITY=p.n.run_identity(cfg),NATIVE_ATTEMPT=str(childpath),
        NATIVE_STOP_PATH=str(childpath/"STOP"),HF_HOME="/data/hf_cache",HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",SLURM_JOB_END_TIME=str(actual["end_epoch"]),OMP_NUM_THREADS="8")
    p.seal(attempt/"ATTEMPT_START.json",dict(actual=actual,execution=eref,argv=argv,accepted_cursor=cursor,
        sources=p.sources(),started_epoch=started,expected_horizon=cfg["expected_horizon"],scientific_max_steps=-1))
    requested=[False]
    def request_stop(*_):requested[0]=True
    handlers={sig:signal.signal(sig,request_stop) for sig in (signal.SIGUSR1,signal.SIGTERM)}
    try:
        with (attempt/"torchrun.log").open("xb") as log:
            process=subprocess.Popen(argv,cwd=p.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if (requested[0] or time.time()>=stop) and childpath.is_dir():(childpath/"STOP").touch(exist_ok=True)
                if time.time()>=actual["end_epoch"]-60:
                    process.terminate()
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=15)
                    raise ValueError("Unaccepted actual deadline failure; retain all evidence")
                time.sleep(1)
            code=process.returncode
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    p.seal(attempt/"PROCESS_RETURN.json",dict(returncode=code,actual=actual,
        elapsed_seconds=time.time()-started,all_in_cost_requires_sacct_D=True))
    outcome,logical=child_exit(code,childpath,a["configuration"])
    p.seal(attempt/"CHILD_RETURN.json",dict(returncode=code,logical_exit_code=logical,
        child_exit=outcome,actual=actual,elapsed_seconds=time.time()-started,all_in_cost_requires_sacct_D=True))
    if logical==0:
        return collect.collect(a["configuration"],e["accepted_history"]+[p.desc(childpath/"STUDENT_COMPLETE.json")],root/"COMPLETE.json")
    continuation=p.desc(childpath/"ACCEPTED_CONTINUATION_V1.json");value=p.checked(continuation)
    p.require(cursor<value["global_step"]<cfg["expected_horizon"]["natural_updates"],"Strict positive accepted progress only")
    try:after=physical_check(binding_ref,activation_ref,cfg,attempt/"AFTER_TRAINING_CAPSULES.json",phase="after")
    except CapsuleInspectionDeferred as error:
        external_continuation(attempt,a["configuration"],actual,value["global_step"],continuation,error,"after")
    accepted_cursor_guard(after,value["global_step"],continuation,a["configuration"])
    dest=root/"transitions"/("r"+str(actual["restart"]));dest.mkdir(parents=True,exist_ok=False)
    next_value=dict(e,actual_restart=actual["restart"]+1,accepted_history=e["accepted_history"]+[continuation])
    next_ref=p.seal(root/"requests"/("restart-"+str(actual["restart"]+1)+".json"),next_value)
    intent=p.seal(dest/"INTENT.json",dict(owner="Root",action="requeuehold",binding=binding_ref,
        previous_execution=eref,execution=eref,continuation=continuation,from_restart=actual["restart"],
        to_restart=actual["restart"]+1,next_request=next_ref,physical_guard=p.desc(attempt/"AFTER_TRAINING_CAPSULES.json")))
    p.seal(root/"routes"/("restart-"+str(actual["restart"]+1)+".json"),dict(intent=intent,execution=next_ref,
        independent_actual_transition_review=None,Root_must_reconcile_before_release=True))
    p.seal(attempt/"ROOT_CONTINUATION_REQUIRED.json",dict(intent=intent,next_request=next_ref,
        accepted=continuation,automatic_release=False,Root_owns_actual_transition=True))
    lock=Path("/home/luyangz/data_selection_multimodal/collaboration/babel-two-track-20261019/ADMISSION.lock")
    with lock.open("r+") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);current(binding_ref,activation_ref,minimum_seconds=120)
        response=subprocess.run(["scontrol","requeuehold",b["job_id"]],text=True,capture_output=True,timeout=30)
        ack=p.seal(dest/"ACK.json",dict(intent=intent,response=dict(returncode=response.returncode,
            stdout=response.stdout,stderr=response.stderr),may_be_absent_after_parent_termination=True))
        p.require(response.returncode==0,"Actual requeuehold failed; retain intent/raw response, no API retry")
    return p.seal(attempt/"HELD_CONTINUATION_REQUESTED.json",dict(intent=intent,ack=ack,
        accepted=continuation,full_epoch_completed=False,Root_must_independently_reconcile_and_release=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("binding","activation"):parser.add_argument("--"+name)
    parser.add_argument("--entry-contract",action="store_true")
    args=parser.parse_args()
    if args.entry_contract:
        target=activation_target(p.PACKAGE+".driver")
        print(p.canonical(dict(runtime_module=__name__,activation_target=target,
            child_module=p.PACKAGE+".child",capsule_module=p.PACKAGE+".capsule",
            actual_scheduler_or_model_called=False)).decode())
        return
    p.require(args.binding is not None and args.activation is not None,"Actual binding and activation paths")
    print(p.canonical(run_conditional_student(p.desc(args.binding),p.desc(args.activation))).decode())


if __name__=="__main__":main()
