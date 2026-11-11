"""Root-only finite held admission/release hooks. This module is never a daemon."""
import argparse,fcntl,os,pwd,re,subprocess,uuid
from pathlib import Path
from . import protocol as p
from .worker import fields

def actor(profile):
    p.require(profile['site']=='babel' and profile['slurm_user']=='luyangz' and pwd.getpwuid(os.getuid()).pw_name=='luyangz' and profile['code_root']==str(p.ROOT)==p.BASE+'/src','Actual Root Babel source/user required before mutation/RPC')
def lock_path():
    value=Path(p.LOCK);p.require(value.is_file() and all(not x.is_symlink() for x in (value,*value.parents)),'Existing canonical project admission lock required');return value

def source_review(ref):
    value=p.checked(ref);p.require(value['status']=='PASS_INDEPENDENT_FINITE_DETERMINISTIC_REPLICATION_SOURCE' and value['driver_sources']==p.sources() and p.independent(value['reviewer']),'Independent exact source review required')
def held_fields(raw,held,profile):
    f=fields(raw);intent=p.checked(held['intent']);cell=held['cell'];job=held['job_id']
    expected=dict(JobId=job,JobName='dsmm-d8rep29-'+cell,JobState='PENDING',Reason='JobHeldUser',UserId='luyangz('+str(os.getuid())+')',Account='cx',Partition='general',QOS='normal',NumTasks='1',NumCPUs='8',WorkDir=str(p.ROOT),Command=str(p.HERE/'run.sbatch'),Comment='dsmm-d8rep29:'+intent['nonce'],TimeLimit='00:55:00',TresPerNode='gres/gpu:L40S:1',Restarts='0',StartTime='Unknown',RunTime='00:00:00')
    p.require(all(f.get(k)==v for k,v in expected.items()) and f.get('NumNodes') in ('1','1-1') and f.get('AllocTRES') in (None,'(null)') and f.get('ArrayJobId') in (None,'(null)'),'Exact never-started held request differs')
    tres=dict(x.split('=',1) for x in f.get('ReqTRES','').split(',') if '=' in x);p.require(tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1','HeldGPU/memory differs');return f

def census(profile,*,need_submission_slot=True):
    limits=subprocess.check_output(['sacctmgr','-nP','show','qos','normal','format=Name,MaxSubmitPU,MaxTRESPU,MaxJobsPU'],text=True,timeout=30)
    rows=[r.split('|') for r in limits.splitlines() if r.startswith('normal|')];p.require(len(rows)==1 and rows[0][1].isdigit(),'Actual site submission ceiling missing')
    raw=subprocess.check_output(['squeue','-r','-h','-u','luyangz','-o','%i|%j|%T|%q|%P|%r'],text=True,timeout=30)
    parsed=[row.split('|') for row in raw.splitlines() if row.strip()];p.require(all(len(row)==6 and re.fullmatch(r'\d+(?:_\d+)?',row[0]) for row in parsed),'Malformed expanded all-user census')
    count=sum(row[3]=='normal' for row in parsed);ceiling=int(rows[0][1])
    p.require(count<ceiling if need_submission_slot else count<=ceiling,'Actual all-user MaxSubmitPU ceiling reached')
    return dict(raw=raw,limits=limits,rows=parsed,normal_submission_count=count,normal_MaxSubmitPU=ceiling,requires_new_submission_slot=need_submission_slot)

def submit(design_ref,source_ref,authorization_ref):
    source_review(source_ref);design=p.validate_design(p.checked(design_ref));profile=p.checked(design['configurations']['reference']['site_profile']);actor(profile)
    root=p.literal(Path(design['report']).parent/'admission');root.mkdir(parents=True,exist_ok=True)
    with lock_path().open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);p.require(not (root/'INTENT.json').exists(),'Prior intent/ACK requires reconciliation; never resubmit blindly')
        current=census(profile);auth=p.checked(authorization_ref);eligible=eligible_root_general(auth,current);p.require(eligible<=2,'Author two-General eligible commitment window exceeded');cell=design['cell'];p.require(not any(row[1]=='dsmm-d8rep29-'+cell for row in current['rows']),'Exact finite controller already live')
        nonce=uuid.uuid4().hex;binding=root/'RUN_BINDING.json'
        argv=['sbatch','--parsable','--hold','--requeue','--nodes=1','--ntasks=1','--account=cx','--partition=general','--qos=normal','--gres=gpu:L40S:1','--cpus-per-task=8','--mem=64G','--time=00:55:00','--signal=B:USR1@300','--open-mode=append','--job-name=dsmm-d8rep29-'+cell,'--comment=dsmm-d8rep29:'+nonce,'--chdir='+str(p.ROOT),'--output='+str(root/'slurm-%j.out'),str(p.HERE/'run.sbatch'),str(binding),profile['python']]
        intent=p.seal(root/'INTENT.json',dict(status='INTENT_BEFORE_SUBMISSION',owner='Root',cell=cell,design=design_ref,driver_sources=p.sources(),source_review=source_ref,Root_window_authorization=authorization_ref,eligible_Root_GPUs=eligible,prepared_by=p.AUTHOR_IDENTITY,nonce=nonce,argv=argv,actual_census=current,UTC=p.now()))
        try:r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
        except subprocess.TimeoutExpired:
            p.seal(root/'AMBIGUOUS_RPC.json',dict(intent=intent,no_automatic_retry=True));raise
        response=p.seal(root/'SBATCH_RESPONSE.json',dict(intent=intent,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));match=re.fullmatch(r'(\d+)(?:;[^\s]+)?\s*',r.stdout)
        if r.returncode!=0 or not match:
            p.seal(root/'AMBIGUOUS_RPC.json',dict(intent=intent,response=response,no_automatic_retry=True));raise ValueError('Missing exact scalar ACK; reconcile')
        job=match[1];ack=p.seal(root/'ACK.json',dict(job_id=job,intent=intent,response=response,UTC=p.now()))
        held=dict(schema='native-deterministic-replication-held-binding-v1',owner='Root',cell=cell,job_id=job,intent=intent,ack=ack,design=design_ref,driver_sources=p.sources())
        raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30);held_fields(raw,held,profile)
        spool=subprocess.check_output(['scontrol','write','batch_script',job,'-'],timeout=30);p.require(spool==(p.HERE/'run.sbatch').read_bytes(),'Actual held spool differs')
        held.update(actual_held_scheduler=raw,spool_sha256=p.sha(p.HERE/'run.sbatch'),spool_reconciled=True)
        return p.seal(root/'HELD_BINDING.json',held)

def eligible_root_general(authorization,census_value,runner=subprocess.check_output):
    p.require(authorization['status']=='ROOT_REGISTERED_TWO_GENERAL_FOUNDATION_WINDOW' and authorization['owner']=='Root' and authorization['maximum_eligible_root_general_gpus']==2,'Explicit current author two-General window required')
    roster=authorization['current_root_general_commitments'];p.require(isinstance(roster,list) and len({r['job_id'] for r in roster})==len(roster),'Exact Root commitment roster required')
    p.require(all(type(r['gpus']) is int and r['gpus']>0 for r in roster),'Registered GPU multiplicity must be a positive integer')
    indexed={row[0]:row for row in census_value['rows']};registered={r['job_id']:r for r in roster};eligible=0
    for record in roster:
        job=record['job_id'];p.require(job.isdecimal(),'Scalar owned Root registration required')
        if job not in indexed:continue
        row=indexed[job];p.require(row[1]==record['job_name'] and row[3]=='normal' and row[4]=='general','Current Root roster queue identity differs')
        if row[2]=='PENDING' and row[5]=='Dependency':
            # A genuine future request remains in the queued census. Only a
            # fresh owned afterany edge to another registered current Root
            # commitment proves the two cannot run simultaneously. Unknown,
            # fulfilled, external or complex dependency graphs fail closed.
            raw=runner(['scontrol','show','job','-o',job],text=True,timeout=30);f=fields(raw)
            dep=re.fullmatch(r'afterany:(\d+)\(unfulfilled\)',f.get('Dependency',''))
            req=dict(x.split('=',1) for x in f.get('ReqTRES','').split(',') if '=' in x)
            p.require(dep is not None and dep[1]!=job and dep[1] in registered and dep[1] in indexed,'Only actual unfulfilled afterany of a registered current Root job is serialized')
            p.require(f.get('JobId')==job and f.get('UserId')=='luyangz('+str(os.getuid())+')' and f.get('JobState')=='PENDING' and f.get('Reason')=='Dependency' and f.get('Account')=='cx' and f.get('Partition')=='general' and f.get('QOS')=='normal' and f.get('JobName')==record['job_name'] and req.get('gres/gpu')==str(record['gpus']),'Actual dependency request ownership/resources differ')
            pred=indexed[dep[1]];p.require(pred[2] in ('RUNNING','PENDING') and pred[3:5]==['normal','general'] and registered[dep[1]]['gpus']>=record['gpus'],'Dependency serialization does not preserve the occupied Root GPU slot')
            continue
        if row[2]=='RUNNING' or (row[2]=='PENDING' and row[5]!='JobHeldUser'):eligible+=record['gpus']
    p.require(authorization['all_root_commitments_reconciled'] is True,'Root commitment reconciliation missing')
    return eligible

def root_two_general_window(authorization,census_value,runner=subprocess.check_output):
    eligible=eligible_root_general(authorization,census_value,runner)
    p.require(eligible<2,'Author two-General eligible commitment window full')
    return eligible

def release(study_ref,source_ref,authorization_ref):
    source_review(source_ref);study=p.checked(study_ref);configs,_=p.validate_study(study,gpu=False);profile=p.checked(configs['reference']['site_profile']);actor(profile)
    held=p.checked(study['held_binding']);root=Path(study['held_binding']['path']).parent;job=held['job_id']
    with lock_path().open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);p.require(not (root/'RELEASE_INTENT.json').exists(),'Existing release intent requires reconciliation')
        current=census(profile,need_submission_slot=False);auth=p.checked(authorization_ref)
        p.require(any(r==dict(job_id=job,job_name='dsmm-d8rep29-'+study['cell'],gpus=1) for r in auth['current_root_general_commitments']),'Current held candidate must be registered in complete Root census')
        root_two_general_window(auth,current)
        raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30);held_fields(raw,held,profile)
        spool=subprocess.check_output(['scontrol','write','batch_script',job,'-'],timeout=30);p.require(spool==(p.HERE/'run.sbatch').read_bytes(),'Pre-release held spool changed')
        binding=p.seal(root/'RUN_BINDING.json',dict(schema='native-deterministic-replication-run-binding-v1',owner='Root',prepared_by=p.AUTHOR_IDENTITY,job_id=job,study=study_ref,held_binding=study['held_binding'],Root_window_authorization=authorization_ref))
        intent=p.seal(root/'RELEASE_INTENT.json',dict(job_id=job,binding=binding,actual_held=raw,actual_census=current,Root_window_authorization=authorization_ref,UTC=p.now()))
        try:r=subprocess.run(['scontrol','release',job],capture_output=True,text=True,timeout=30)
        except subprocess.TimeoutExpired:
            p.seal(root/'RELEASE_AMBIGUOUS.json',dict(intent=intent,no_automatic_retry=True));raise
        receipt=p.seal(root/'RELEASE_ACK.json',dict(intent=intent,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));p.require(r.returncode==0,'Release not acknowledged; reconcile');return receipt

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('submit-held','release'));a.add_argument('--artifact',required=True);a.add_argument('--source-review',required=True);a.add_argument('--Root-window-authorization');v=a.parse_args()
    if not v.Root_window_authorization:a.error('--Root-window-authorization required')
    result=submit(p.desc(v.artifact),p.desc(v.source_review),p.desc(v.Root_window_authorization)) if v.action=='submit-held' else release(p.desc(v.artifact),p.desc(v.source_review),p.desc(v.Root_window_authorization));print(p.canonical(result).decode())
if __name__=='__main__':main()
