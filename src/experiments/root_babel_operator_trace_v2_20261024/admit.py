"""Root-only finite one-job held admission, then individually reviewed release."""
import argparse
import fcntl
import os
from pathlib import Path
import pwd
import re
import subprocess
import uuid
from . import protocol as p

def fields(raw):return dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw))

def actor(profile):
    p.require(profile['site']=='babel' and profile['slurm_user']=='luyangz'
        and pwd.getpwuid(os.getuid()).pw_name==profile['slurm_user']
        and str(p.ROOT)==profile['code_root']==p.BASE+'/src',
        'Actual Babel project owner/source root required before mutation or scheduler RPC')

def held_fields(raw,job,cell,intent,profile):
    f=fields(raw)
    expected=dict(JobId=job,JobName='dsmm-babel-optrace-'+cell,JobState='PENDING',Reason='JobHeldUser',
        Account=profile['account'],Partition='general',QOS='normal',NumTasks='1',NumCPUs='8',
        WorkDir=str(p.ROOT),Command=str(p.HERE/'run.sbatch'),Comment='dsmm-babel-optrace:'+intent['nonce'],
        TimeLimit='00:55:00',TresPerNode='gres/gpu:L40S:1',Restarts='0',StartTime='Unknown',RunTime='00:00:00')
    for name,value in expected.items():p.require(f.get(name)==value,'Held replay differs: '+name)
    p.require(f.get('NumNodes') in ('1','1-1') and f.get('UserId')==profile['slurm_user']+'('+str(os.getuid())+')'
        and f.get('AllocTRES') in (None,'(null)') and f.get('ArrayJobId') in (None,'(null)'),'Held request may have run or belongs to another UID')
    tres=dict(x.split('=',1) for x in f.get('ReqTRES','').split(',') if '=' in x)
    p.require(tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1','Held memory/GPU request differs')
    return f

def source_review(ref):
    value=p.checked(ref)
    p.require(value['status']=='PASS_INDEPENDENT_BABEL_NATIVE7_OPERATOR_TRACE_SOURCE'
        and value['driver_sources']==p.sources() and value['reviewer'] not in ('/root/native_campaign_build','native_campaign_build'),
        'Independent exact execution/controller source review required')

def lock_path():
    path=Path(p.LOCK)
    p.require(path.is_file() and all(not x.is_symlink() for x in (path,*path.parents)),
        'Use existing authoritative project lock, never a competing private lock')
    return path

def capacity(profile):
    raw=subprocess.check_output(['sacctmgr','-nP','show','qos','normal','format=Name,MaxSubmitPU'],text=True,timeout=30)
    rows=[r.split('|') for r in raw.splitlines() if r.startswith('normal|')]
    p.require(len(rows)==1 and rows[0][1].isdigit(),'Unknown actual normal submission ceiling')
    queue=subprocess.check_output(['squeue','-r','-h','-u',profile['slurm_user'],'-o','%i|%q'],text=True,timeout=30)
    expanded=[row.split('|') for row in queue.splitlines() if row.strip()]
    p.require(all(len(row)==2 and re.fullmatch(r'[0-9]+(?:_[0-9]+)?',row[0]) and row[1] for row in expanded),
        'Malformed actual expanded all-user QoS census')
    count=sum(row[1]==profile['qos'] for row in expanded)
    p.require(count<int(rows[0][1]),'Actual all-user site submission ceiling reached; no new RPC')
    return dict(qos=raw,all_user_queue=queue,expanded_pending_running_tasks=count,actual_max_submit=int(rows[0][1]))

def spool(job,path):
    subprocess.run(['scontrol','write','batch_script',job,str(path)],text=True,capture_output=True,timeout=30,check=True)
    p.require(p.n.sha(path)==p.n.sha(p.HERE/'run.sbatch'),'Actual held spool differs from reviewed wrapper')
    return p.desc(path)

def admit(design_ref,review_ref):
    source_review(review_ref);design=p.validate_design(p.checked(design_ref))
    from experiments.root_native_campaign_v7_20261024.config import validate_profile
    profile=validate_profile(p.checked(design['configurations']['T1']['site_profile']))
    actor(profile)
    p.require(profile['site']=='babel' and profile['account']=='cx' and profile['qos']=='normal'
        and profile['partition']=='general' and profile['gpu_type']=='L40S' and profile['code_root']==str(p.ROOT),
        'Exact Babel General L40S route required')
    root=p.literal(Path(design['report']).parent/'admission');root.mkdir(parents=True,exist_ok=True)
    with lock_path().open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (root/'ADMISSION_RESULT.json').exists():
            prior=p.read(root/'ADMISSION_RESULT.json');p.require(prior['design']==design_ref,'Admission identity collision')
            return p.desc(root/'ADMISSION_RESULT.json')
        # Partial/ambiguous batch is never automatically repeated.
        p.require(not (root/'BATCH_INTENT.json').exists(),'Prior partial admission requires Root reconciliation')
        p.seal(root/'BATCH_INTENT.json',dict(owner='Root',design=design_ref,cells=p.CELLS,
            source_review=review_ref,driver_sources=p.sources(),UTC=p.now()))
        results={}
        for cell in p.CELLS:
            dest=root/cell;dest.mkdir(exist_ok=True);nonce=uuid.uuid4().hex
            census=capacity(profile);future=str(dest/'RUN_BINDING.json')
            argv=['sbatch','--parsable','--hold','--no-requeue','--account='+profile['account'],'--partition=general','--qos=normal',
                '--gres=gpu:L40S:1','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=64G','--time=00:55:00',
                '--signal=B:USR1@300','--open-mode=append','--job-name=dsmm-babel-optrace-'+cell,'--chdir='+str(p.ROOT),
                '--comment=dsmm-babel-optrace:'+nonce,'--output='+str(dest/'slurm-%j.out'),
                str(p.HERE/'run.sbatch'),future,profile['python']]
            value=dict(status='INTENT_BEFORE_SUBMISSION',owner='Root',cell=cell,design=design_ref,
                driver_sources=p.sources(),nonce=nonce,argv=argv,actual_census=census,UTC=p.now())
            intent=p.seal(dest/'INTENT.json',value)
            try:reply=subprocess.run(argv,text=True,capture_output=True,timeout=60)
            except subprocess.TimeoutExpired:
                p.seal(dest/'AMBIGUOUS_RPC.json',dict(intent=intent,status='TIMEOUT_MAY_HAVE_SUBMITTED_NO_RETRY'))
                raise
            raw=p.seal(dest/'SBATCH_RESPONSE.json',dict(intent=intent,returncode=reply.returncode,
                stdout=reply.stdout,stderr=reply.stderr))
            match=re.fullmatch(r'(\d+)(?:;[^\s]+)?\s*',reply.stdout)
            if reply.returncode!=0 or not match:
                p.seal(dest/'AMBIGUOUS_RPC.json',dict(intent=intent,response=raw,status='NO_EXACT_SCALAR_ACK_NO_RETRY'))
                raise ValueError('Unresolved scheduler acknowledgement; preserve and reconcile')
            job=match[1]
            ack=p.seal(dest/'ACK.json',dict(job_id=job,intent=intent,response=raw,UTC=p.now()))
            current=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30)
            held_fields(current,job,cell,value,profile)
            actual_spool=spool(job,dest/'ACTUAL_SPOOL.sh')
            results[cell]=p.seal(dest/'HELD_BINDING.json',dict(schema='babel-native7-operator-trace-held-binding-v2',
                owner='Root',cell=cell,job_id=job,intent=intent,ack=ack,design=design_ref,
                driver_sources=p.sources(),actual_held_scheduler=current,spool=actual_spool,spool_reconciled=True))
        return p.seal(root/'ADMISSION_RESULT.json',dict(owner='Root',design=design_ref,held_bindings=results,
            status='ONE_FINITE_TRACE_REQUEST_HELD_NOT_RELEASED',no_native_qualification_claimed=True))

def release(launch_ref,review_ref):
    # Explicit metadata-only conditional contract; old full-study release is
    # retired rather than dereferencing DATA from an ordinary login process.
    from .launch import release as conditional_release
    return conditional_release(launch_ref,review_ref)

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('admit','release'))
    a.add_argument('--artifact',required=True);a.add_argument('--source-review',required=True);x=a.parse_args()
    result=admit(p.desc(x.artifact),p.desc(x.source_review)) if x.action=='admit' else release(p.desc(x.artifact),p.desc(x.source_review))
    print(p.n.canonical(result).decode())

if __name__=='__main__':main()
