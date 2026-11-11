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
    p.require(profile['site']=='aces' and profile['slurm_user']=='u.lz341613'
        and pwd.getpwuid(os.getuid()).pw_name==profile['slurm_user']
        and str(p.ROOT)==profile['code_root']==p.BASE+'/v3/src',
        'Actual ACES project owner/source root required before mutation or scheduler RPC')

def held_fields(raw,job,cell,intent,profile):
    f=fields(raw)
    expected=dict(JobId=job,JobName='dsmm-optrace-'+cell,JobState='PENDING',Reason='JobHeldUser',
        Account=profile['account'],Partition='gpu',QOS='normal',NumTasks='1',NumCPUs='8',
        WorkDir=str(p.ROOT),Command=str(p.HERE/'run.sbatch'),Comment='dsmm-optrace:'+intent['nonce'],
        TimeLimit='00:20:00',TresPerNode='gres/gpu:h100:1',Restarts='0',StartTime='Unknown',RunTime='00:00:00')
    for name,value in expected.items():p.require(f.get(name)==value,'Held replay differs: '+name)
    p.require(f.get('NumNodes') in ('1','1-1') and f.get('UserId')==profile['slurm_user']+'('+str(os.getuid())+')'
        and f.get('AllocTRES') in (None,'(null)'),'Held request may have run or belongs to another UID')
    tres=dict(x.split('=',1) for x in f.get('ReqTRES','').split(',') if '=' in x)
    p.require(tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1','Held memory/GPU request differs')
    return f

def source_review(ref):
    value=p.checked(ref)
    p.require(value['status']=='PASS_INDEPENDENT_NATIVE7_OPERATOR_TRACE_SOURCE'
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
    queue=subprocess.check_output(['squeue','-r','-h','-u',profile['slurm_user'],'-o','%i'],text=True,timeout=30)
    count=len([x for x in queue.splitlines() if x.strip()])
    p.require(count<int(rows[0][1]),'Actual all-user site submission ceiling reached; no new RPC')
    return dict(qos=raw,all_user_queue=queue,expanded_pending_running_tasks=count,actual_max_submit=int(rows[0][1]))

def spool(job,path):
    subprocess.run(['scontrol','write','batch_script',job,str(path)],text=True,capture_output=True,timeout=30,check=True)
    p.require(p.n.sha(path)==p.n.sha(p.HERE/'run.sbatch'),'Actual held spool differs from reviewed wrapper')
    return p.desc(path)

def verify_retirement(reference,profile,runner=subprocess.check_output):
    p.require(reference is not None,'Exact predecessor retirement required before successor admission')
    value=p.checked(reference)
    p.require(value['status']=='EXACT_NEVER_STARTED_T1_RETIRED' and value['owner']=='Root'
        and value['job_id']=='2250058' and value['predecessor_study']==p.PREDECESSOR,
        'Only the exact original never-started T1 may be replaced')
    prior=p.checked(p.PREDECESSOR)
    p.require(prior['jobs']=={'T1':'2250058'} and value['held_binding']==prior['held_bindings']['T1'],
        'Retirement lost exact old job/study binding')
    held=p.checked(value['held_binding']);intent=p.checked(held['intent']);before=fields(value['pre_cancel_scheduler'])
    p.require(held['owner']=='Root' and held['cell']=='T1' and held['job_id']=='2250058' and held['spool_reconciled'] is True,'Foreign original held binding')
    old_wrapper=p.BASE+'/v3/src/experiments/root_native_operator_trace_v1_20261024/run.sbatch'
    expected=dict(JobId='2250058',JobName='dsmm-optrace-T1',JobState='PENDING',Reason='JobHeldUser',
        UserId=profile['slurm_user']+'('+str(os.getuid())+')',Account=profile['account'],Partition='gpu',QOS='normal',
        NumCPUs='8',NumTasks='1',TimeLimit='00:55:00',Restarts='0',RunTime='00:00:00',StartTime='Unknown',
        WorkDir=str(p.ROOT),Command=old_wrapper,Comment='dsmm-optrace:'+intent['nonce'],TresPerNode='gres/gpu:h100:1')
    p.require(all(before.get(k)==v for k,v in expected.items()) and before.get('NumNodes') in ('1','1-1')
        and before.get('AllocTRES') in (None,'(null)'),'Pre-cancel old request was not exact held/never-started')
    tres=dict(x.split('=',1) for x in before.get('ReqTRES','').split(',') if '=' in x)
    p.require(tres.get('cpu')=='8' and tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1','Original held resources changed')
    spool=value['pre_cancel_spool'];p.literal(spool['path'])
    p.require(p.desc(spool['path'])==spool and spool['sha256']==prior['driver_sources']['experiments/root_native_operator_trace_v1_20261024/run.sbatch'],
        'Preserved old scheduler spool/source differs')
    def terminal(raw):
        rows=[r.split('|') for r in raw.splitlines() if r.split('|')[0]=='2250058']
        p.require(len(rows)==1 and len(rows[0])>=6 and rows[0][1].split()[0]=='CANCELLED'
            and rows[0][2:4]==['0:0','0'] and rows[0][4] in ('Unknown','None')
            and rows[0][5]=='0','Old T1 is not cancelled with absent actual start and zero runtime/restarts')
    terminal(value['terminal_accounting'])
    actual=runner(['sacct','-X','-nP','-j','2250058','-o','JobID,State,ExitCode,ElapsedRaw,Start,Restarts'],text=True,timeout=30)
    terminal(actual)
    # A purged scalar -j query is an error, not an empty successful result.
    # Query all expanded requests belonging to the exact bound owner instead;
    # failed scheduler calls propagate and malformed IDs never prove absence.
    queue=runner(['squeue','-r','-h','-u',profile['slurm_user'],'-o','%i'],text=True,timeout=30)
    live=[row.strip() for row in queue.splitlines() if row.strip()]
    p.require(all(re.fullmatch(r'[0-9]+(?:_[0-9]+)?',job) for job in live),
        'Actual expanded all-user queue contains malformed job identity')
    p.require(not any(job.split('_',1)[0]=='2250058' for job in live),
        'Original T1 remains scheduler-live; do not duplicate')
    return reference

def admit(design_ref,review_ref,retirement_ref=None):
    source_review(review_ref);design=p.validate_design(p.checked(design_ref))
    from experiments.root_native_campaign_v7_20261024.config import validate_profile
    profile=validate_profile(p.checked(design['configurations']['T1']['site_profile']))
    actor(profile)
    p.require(profile['site']=='aces' and profile['account']=='158416130568' and profile['qos']=='normal'
        and profile['partition']=='gpu' and profile['gpu_type']=='h100' and profile['code_root']==str(p.ROOT),
        'Exact ACES H100 route required')
    root=p.literal(Path(design['report']).parent/'admission');root.mkdir(parents=True,exist_ok=True)
    with lock_path().open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        verify_retirement(retirement_ref,profile)
        if (root/'ADMISSION_RESULT.json').exists():
            prior=p.read(root/'ADMISSION_RESULT.json');p.require(prior['design']==design_ref,'Admission identity collision')
            return p.desc(root/'ADMISSION_RESULT.json')
        # Partial/ambiguous batch is never automatically repeated.
        p.require(not (root/'BATCH_INTENT.json').exists(),'Prior partial admission requires Root reconciliation')
        p.seal(root/'BATCH_INTENT.json',dict(owner='Root',design=design_ref,cells=p.CELLS,
            source_review=review_ref,driver_sources=p.sources(),predecessor_retirement=retirement_ref,UTC=p.now()))
        results={}
        for cell in p.CELLS:
            dest=root/cell;dest.mkdir(exist_ok=True);nonce=uuid.uuid4().hex
            census=capacity(profile);future=str(dest/'RUN_BINDING.json')
            argv=['sbatch','--parsable','--hold','--account='+profile['account'],'--partition=gpu','--qos=normal',
                '--gres=gpu:h100:1','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=64G','--time=00:20:00',
                '--signal=B:USR1@300','--open-mode=append','--job-name=dsmm-optrace-'+cell,'--chdir='+str(p.ROOT),
                '--comment=dsmm-optrace:'+nonce,'--output='+str(dest/'slurm-%j.out'),
                str(p.HERE/'run.sbatch'),future,profile['python']]
            value=dict(status='INTENT_BEFORE_SUBMISSION',owner='Root',cell=cell,design=design_ref,
                driver_sources=p.sources(),nonce=nonce,argv=argv,actual_census=census,predecessor_retirement=retirement_ref,UTC=p.now())
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
            results[cell]=p.seal(dest/'HELD_BINDING.json',dict(schema='native7-operator-trace-held-binding-v4',
                owner='Root',cell=cell,job_id=job,intent=intent,ack=ack,design=design_ref,
                driver_sources=p.sources(),actual_held_scheduler=current,spool=actual_spool,spool_reconciled=True,predecessor_retirement=retirement_ref))
        return p.seal(root/'ADMISSION_RESULT.json',dict(owner='Root',design=design_ref,held_bindings=results,
            status='ONE_FINITE_TRACE_REQUEST_HELD_NOT_RELEASED',no_native_qualification_claimed=True))

def release(study_ref,review_ref):
    source_review(review_ref);study=p.checked(study_ref);pair,_=p.validate(study)
    for cfg in pair.values():actor(p.checked(cfg['site_profile']))
    with lock_path().open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        results={}
        for cell,cfg in pair.items():
            held_ref=study['held_bindings'][cell];held=p.checked(held_ref);intent=p.checked(held['intent'])
            dest=Path(held_ref['path']).parent;job=held['job_id'];profile=p.checked(cfg['site_profile'])
            verify_retirement(held['predecessor_retirement'],profile)
            p.require(job==study['jobs'][cell] and held['design']==study['design'],'Wrong final job/design')
            p.require(not (dest/'RELEASE_INTENT.json').exists(),'Release already attempted; reconcile before another RPC')
            raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30)
            held_fields(raw,job,cell,intent,profile);spool(job,dest/'PRE_RELEASE_SPOOL.sh')
            binding=p.seal(dest/'RUN_BINDING.json',dict(schema='native7-operator-trace-run-binding-v4',owner='Root',
                cell=cell,job_id=job,held_binding=held_ref,study=study_ref))
            before=p.seal(dest/'RELEASE_INTENT.json',dict(owner='Root',job_id=job,binding=binding,
                actual_held_scheduler=raw,UTC=p.now()))
            try:r=subprocess.run(['scontrol','release',job],capture_output=True,text=True,timeout=30)
            except subprocess.TimeoutExpired:
                p.seal(dest/'RELEASE_AMBIGUOUS.json',dict(intent=before,status='NO_AUTOMATIC_RELEASE_RETRY'));raise
            results[cell]=p.seal(dest/'RELEASE_RESPONSE.json',dict(intent=before,returncode=r.returncode,
                stdout=r.stdout,stderr=r.stderr))
            p.require(r.returncode==0,'Release failed; no automatic retry')
        return results

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('admit','release'))
    a.add_argument('--artifact',required=True);a.add_argument('--source-review',required=True);a.add_argument('--predecessor-retirement');x=a.parse_args()
    result=admit(p.desc(x.artifact),p.desc(x.source_review),p.desc(x.predecessor_retirement) if x.predecessor_retirement else None) if x.action=='admit' else release(p.desc(x.artifact),p.desc(x.source_review))
    print(p.n.canonical(result).decode())

if __name__=='__main__':main()
