"""Root materializes a finite request/postACK DAG; this module never submits/releases."""
import argparse
import os
from pathlib import Path
import re
from . import protocol as p


def request(configuration,nonce,load=p.checked):
    cfg=load(configuration);p.supported(cfg)
    p.require(re.fullmatch("[0-9a-f]{32}",nonce) is not None
        and cfg["owner"]=="Root" and cfg["allocation"]["minutes"]==720,
        "Finite reviewed12h10percent trajectory, not runtime padding")
    command=p.desc(p.HERE/"run.sbatch")
    control=Path(cfg["report"])/"PHYSICAL"/nonce
    binding=control/"BINDING.json";activation=control/"ACTIVATION.json"
    name="dsmm-coincide-native-s"+str(cfg["training_seed"])
    argv=["sbatch","--parsable","--hold","--open-mode=append","--job-name="+name,"--account=cx","--partition=general",
        "--qos=normal","--gres=gpu:L40S:1","--cpus-per-task=8","--mem=65536M","--nodes=1",
        "--ntasks=1","--time=12:00:00","--requeue","--comment=dsmm-coincide-native:"+nonce,
        "--chdir="+str(p.ROOT),"--output="+str(control/"slurm-%j.out"),
        "--error="+str(control/"slurm-%j.err"),command["path"],str(binding),str(activation)]
    return dict(schema="coincide-fixed10-native-v2-request-v1",owner="Root",nonce=nonce,
        configuration=configuration,sources=p.sources(),native_sources=cfg["sources"],
        scientific_id=cfg["id"],job_name=name,control_root=str(control),command=command,
        outer_allocation=dict(gpus=1,cpus=8,memory_mb=65536,minutes=720),
        expected_horizon=p.horizon(66076),sbatch_argv=argv,
        test_only_argv=argv[:1]+["--test-only"]+argv[1:],actual_submission_count=0,
        duplicate_ready_before_intent_required=True,Root_owns_ADMISSION_lock_and_actual_submission=True)


def post_ack(request_ref,intent_ref,ack_ref,spool_ref,policy_ref,load=p.checked,publish=p.seal):
    value=load(request_ref);intent=load(intent_ref);ack=load(ack_ref);policy=load(policy_ref)
    cfg=load(value["configuration"]);p.supported(cfg)
    response=ack["response"];raw=response["stdout"]
    p.require(value["schema"]=="coincide-fixed10-native-v2-request-v1" and value["sources"]==p.sources()
        and intent["owner"]=="Root" and intent["request"]==request_ref and intent["nonce"]==value["nonce"]
        and intent["command"]==value["command"] and intent["submitted_with_hold"] is True
        and ack["intent"]==intent_ref and response["returncode"]==0
        and re.fullmatch(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?\n?",raw) is not None,
        "Actual held pre-submit intent/rawACK; ambiguous acknowledgement needs Root reconciliation, no retry")
    job=raw.strip().split(";")[0]
    p.require(ack["job_id"]==job and spool_ref==value["command"]==p.desc(p.HERE/"run.sbatch"),
        "Actual exact spooled bytes after scheduler acceptance")
    p.require(policy["status"]=="PASS_INDEPENDENT_EXACT_COINCIDE_NATIVE_EXECUTION_POLICY_V2"
        and policy["sources"]==p.sources() and policy["configuration"]==value["configuration"]
        and policy["full_epoch_no_scientific_step_cap"] is True,
        "Genuine actual exact-source/config/finite execution policy")
    p.independent(policy)
    control=Path(value["control_root"]);config_ref=value["configuration"]
    binding=publish(control/"BINDING.json",dict(schema="coincide-fixed10-native-v2-binding-v1",
        owner="Root",sources=p.sources(),configuration=config_ref,request=request_ref,
        intent=intent_ref,ack=ack_ref,job_id=job,nonce=value["nonce"],job_name=value["job_name"],
        command=value["command"],spool=spool_ref))
    activation=publish(control/"ACTIVATION.json",dict(owner="Root",binding=binding,
        configuration=config_ref,target_module=p.PACKAGE+".driver",target_sources=p.sources()))
    permit=publish(control/"AUTHORIZATION.json",dict(schema="coincide-fixed10-native-v2-authorization-v1",
        owner="Root",configuration=config_ref,config_identity=p.n.run_identity(cfg),
        physical_binding=binding,adapter_sources=p.sources(),runtime=p.RUNTIME,phase="production",
        expected_horizon=p.horizon(66076),policy_identity=policy["policy_identity"],review=policy_ref))
    execution=publish(Path(cfg["report"])/"ROOT_LONG/requests/restart-0.json",
        dict(schema="coincide-fixed10-native-v2-execution-v1",owner="Root",binding=binding,
            activation=activation,configuration=config_ref,config_identity=p.n.run_identity(cfg),
            actual_restart=0,sources=p.sources(),mode="production",accepted_history=[],
            authorization=permit,policy_identity=policy["policy_identity"],review=policy_ref))
    return dict(binding=binding,activation=activation,authorization=permit,initial_execution=execution,
        job_id=job,actual_release_performed=False,Root_must_verify_pending_spool_census_QoS_before_release=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration",type=Path);parser.add_argument("--nonce")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--request",type=Path)
    for name in ("intent","ack","spool","policy"):parser.add_argument("--"+name,type=Path)
    args=parser.parse_args();p.require(os.getuid()==2708899,"Actual Root binding user")
    if args.request:
        p.require(all(getattr(args,name) is not None for name in ("intent","ack","spool","policy")),"Existing actual postACK graph")
        result=post_ack(p.desc(args.request),p.desc(args.intent),p.desc(args.ack),p.desc(args.spool),p.desc(args.policy))
    else:
        p.require(args.configuration is not None and args.nonce is not None,"Exact config/nonce")
        result=request(p.desc(args.configuration),args.nonce)
    print(p.canonical(p.seal(args.output,result)).decode())


if __name__=="__main__":main()
