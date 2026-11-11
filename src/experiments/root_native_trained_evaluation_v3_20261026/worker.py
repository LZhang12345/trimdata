"""Late-bound final native student, four official banks, durable missing-only QAs."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid
from . import banks,generation,media
from experiments.root_gqa_bank_qualification_v1_20261026 import worker as u
from experiments.root_native_trained_loader_v3_20261026 import model
from experiments.root_native_trained_eval_qual_v3_20261026 import worker as qualified
from experiments.root_s1_shared_student_v1_20261025 import protocol as student

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SCOPE='FINAL_NATIVE_STUDENT_FOUR_COMPLETE_OFFICIAL_BANKS_NOT_START'
PYTHON=qualified.PYTHON

def sources(base,phase3):
    result=qualified.sources(base,phase3)
    extra=('root_aces_scienceqa_start_v1_20261025/__init__.py','root_aces_scienceqa_start_v1_20261025/common.py',
        'root_aces_scienceqa_start_v1_20261025/audit.py','root_scienceqa_native_bank_v1_20261025/__init__.py',
        'root_scienceqa_native_bank_v1_20261025/bank.py','session6_scienceqa_bank_v1_20261024/__init__.py',
        'session6_scienceqa_bank_v1_20261024/scoring.py','root_native_selection_20261023/__init__.py')
    for rel in extra:
        p=ROOT/'experiments'/rel;result[str(p.relative_to(ROOT))]=u.sha(p.read_bytes())
    for p in HERE.iterdir():
        if p.suffix in ('.py','.sbatch') and not p.name.startswith('test_'):result[str(p.relative_to(ROOT))]=u.sha(p.read_bytes())
    return dict(sorted(result.items()))

def complete_student(cfg,load=u.checked):
    """Actual complete metadata is bound on first execution, never a future SHA."""
    training=load(cfg['training_configuration']);path=u.literal(cfg['student_completion_path'])
    u.require(training['id']==cfg['student_id'],'Exact prepared student identity required')
    u.require(path==Path(training['report'])/'ROOT_LONG/COMPLETE.json','Exact original full-trajectory collector path required')
    complete_ref=u.desc(path);complete=load(complete_ref)
    u.require(complete['status']=='COMPLETE_ACTUAL_NATURAL_STUDENT_TRAJECTORY_NOT_EVALUATION'
        and complete['configuration']==cfg['training_configuration'] and complete['config_identity']==student.n.run_identity(training)
        and complete['adapter_sources']==student.sources() and complete['runtime']==student.RUNTIME
        and complete['native_objective_scaling_unchanged'] is True and complete['actual_no_skipped_optimizer_updates'] is True
        and complete['downstream_evaluation_complete'] is False,'Actual unchanged natural student completion required, never partial/START')
    selected=load(training['providers']['selection']);horizon=student.horizon(selected['count'])
    u.require(complete['expected']==horizon and complete['actual_optimizer_updates']==horizon['natural_updates']
        and complete['actual_optimized_records']==horizon['optimized_records']
        and len(complete['actual_unoptimized_tail_record_ids'])==horizon['unoptimized_tail_count'],
        'Natural epoch/exposure horizon incomplete or changed')
    accepted=load(complete['accepted'])
    u.require(accepted['status']=='ACCEPTED_FULL_NATIVE_STATE' and accepted['identity']==complete['config_identity']
        and accepted['global_step']==horizon['natural_updates'],'Final accepted native state does not match completed epoch')
    return training,complete_ref,complete

def function(cfg,identity,roster,complete,bank):
    return dict(schema='formal-native-trained-active-peft-official-bank-function-v1',bank=bank,role=SCOPE,
        training_configuration=cfg['training_configuration'],student_completion=complete,accepted_checkpoint=identity['accepted'],
        model_identity_sha256=qualified.official.c.key(identity),model_generation_config=identity['generation_config'],
        training_optimizer_updates=identity['training_optimizer_updates'],adapter_active=True,adapter_merged=False,
        full_roster_sha256=qualified.official.c.key(roster),sources_sha256=qualified.official.c.key(cfg['sources']),
        decoding=cfg['scienceqa_decoding'] if bank=='ScienceQA_image' else cfg['decoding'])

def documents(binding,load=u.checked):
    cfg=load(binding['configuration']);review=load(binding['independent_source_config_review'])
    u.require(cfg['status']=='ROOT_CONFIGURED_FINAL_NATIVE_STUDENT_FOUR_BANK_EVALUATION_V3' and cfg['owner']=='Root'
        and cfg['scope']==SCOPE and cfg['source_root']==str(ROOT) and cfg['python']==PYTHON
        and cfg['banks']==['POPE','GQA','TextVQA','ScienceQA_image'] and cfg['counts']==banks.COUNTS
        and cfg['resources']==dict(cpus=8,memory_mb=65536,gpus=1,minutes=360)
        and cfg['max_physical_attempts']==sum(banks.COUNTS.values())+1 and cfg['output_limit_bytes']==512<<20
        and cfg['no_model_downloads'] is True and cfg['no_base_copies'] is True and cfg['no_START_journal_reuse'] is True,
        'Finite late-bound final student/four full banks required')
    base=load(cfg['official_POPE_configuration']);phase3=load(cfg['phase3_configuration'])
    u.require(cfg['sources']==sources(base,phase3) and cfg['frozen_scope']==base['frozen_scope']
        and cfg['decoding']==base['decoding'] and cfg['scienceqa_decoding']==dict(conv_mode='vicuna_v1',temperature=0,
        single_pred_prompt=True,max_new_tokens=1024,native_num_beams_keyword=False,native_top_p_keyword=False),
        'Original official bank-kind decoding policies required')
    observation=load(cfg['actual_media_observation'])
    u.require(observation['native_pool_image_root']=='/data/hf_cache/datasets/dsmm_llava_665k_v1/media'
        and cfg['bank_inputs']['TextVQA']['image_root']==observation['native_pool_image_root']+'/textvqa/train_images'
        and observation['TextVQA_native_cached_sample'][0]['sha256']=='5532753f1d943f12f72a8ec3dd9692b2414dfad892a431de95cb649d83356042',
        'Actual owned TextVQA file-locator observation required; all3166 checked separately before production')
    u.require(review['status']=='PASS_INDEPENDENT_FINAL_NATIVE_STUDENT_FOUR_BANK_SOURCE_CONFIG_V3'
        and review['independent'] is True and review['reviewer'] not in ('Root','/root/telemetry_finish','')
        and review['configurations'][cfg['student_id']]==binding['configuration'] and review['sources']==cfg['sources'],
        'Independent exact source/config review required')
    for k,v in cfg['sources'].items():u.require(u.sha((ROOT/k).read_bytes())==v,'Bound production source changed: '+k)
    qref=u.desc(u.literal(cfg['required_actual_qualification_result_path']));q=load(qref)
    qr=load(u.desc(u.literal(cfg['required_independent_qualification_review_path'])))
    u.require(q['status']=='PASS_ACTUAL_NATIVE_TRAINED_ADAPTER_TWO_QA_FRESH_RELOAD'
        and q['configuration']==cfg['qualified_loader_configuration'] and q['adapter_active'] is True
        and q['adapter_merged'] is False and q['distinct_QAs']==2 and q['neural_outputs']==4
        and q['full_bank_scored'] is False and qr['status']=='PASS_INDEPENDENT_ACTUAL_NATIVE_TRAINED_EVAL_QUALIFICATION_V3'
        and qr['independent'] is True and qr['readout']==qref and qr['configuration']==q['configuration'],
        'Actual two-process trained-loader GPU qualification and independent actual review required')
    claim=load(binding['ownership']);grant=load(binding['storage_reservation'])
    u.require(binding['status']=='ROOT_BOUND_FINAL_NATIVE_STUDENT_FOUR_BANK_EVALUATION_V3' and binding['owner']=='Root'
        and binding['nonce']==cfg['nonce'] and claim['job_id']==binding['job_id'] and claim['nonce']==cfg['nonce']
        and claim['configuration']==binding['configuration'] and claim['owner']=='Root' and claim['scope']==SCOPE
        and claim['training_configuration']==cfg['training_configuration'] and claim['no_live_duplicate_checkpoint_evaluation'] is True,
        'One genuine owner for final checkpoint/all four functions required')
    u.require(grant['configuration']==binding['configuration'] and grant['output']==cfg['output']
        and grant['reserved_bytes']==cfg['output_limit_bytes'] and grant['final_checkpoint_retained_for_evaluation'] is True,
        'Actual small result grant/checkpoint retention required')
    u.require(u.literal(cfg['output']).is_relative_to(Path('/data/hf_cache/project_features/DS-MultiModal/native_trained_evaluation_20261026')),
        'Separate shared derived results only')
    return cfg,base

def actor(binding,cfg):
    job=binding['job_id'];restart=os.environ.get('SLURM_RESTART_COUNT','0')
    u.require(os.getuid()==2708899 and job.isdecimal() and os.environ.get('SLURM_JOB_ID')==job and restart.isdecimal()
        and int(restart)<cfg['max_physical_attempts'] and not os.environ.get('SLURM_ARRAY_JOB_ID'),'Finite true scalar evaluator job/restart required')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30);f=u.fields(raw)
    wanted=dict(JobId=job,JobState='RUNNING',JobName='dsmm-native-trained-eval-v3',UserId='luyangz(2708899)',Partition='general',Account='cx',QOS='normal',
        NumCPUs='8',NumTasks='1',NumNodes='1',Restarts=restart,TimeLimit='06:00:00',TresPerNode='gres/gpu:L40S:1',
        Command=str(HERE/'run.sbatch'),WorkDir=str(ROOT),Comment='dsmm-trained-eval:'+cfg['nonce'],NodeList=socket.gethostname().split('.')[0])
    u.require(all(f.get(k)==v for k,v in wanted.items()),'Actual General evaluator resources/identity differ')
    tres=dict(x.split('=',1) for x in f['AllocTRES'].split(',') if '=' in x)
    u.require(tres.get('cpu')=='8' and tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1','Actual evaluator allocation differs')
    u.require(subprocess.check_output(['scontrol','write','batch_script',job,'-'],timeout=30)==(HERE/'run.sbatch').read_bytes(),'Actual evaluator spool differs')
    u.require(os.environ.get('TZ')=='America/New_York','Pinned scheduler timezone required');time.tzset()
    return dict(job_id=job,restart=int(restart),node=f['NodeList'],nonce=cfg['nonce'],scheduler=raw,
        end_epoch=int(time.mktime(time.strptime(f['EndTime'],'%Y-%m-%dT%H:%M:%S'))),process_nonce=uuid.uuid4().hex,pid=os.getpid())

def budget(cfg):
    size=sum(p.stat().st_size for p in Path(cfg['output']).rglob('*') if p.is_file())
    u.require(size<=cfg['output_limit_bytes'],'Actual result envelope exhausted; retain paid QAs')
    return size

def publish_same(path,value):
    if path.exists():u.require(u.checked(u.desc(path))==value,'Previously bound evaluation identity changed')
    else:u.publish(path,value)
    return u.desc(path)

def accepted_prefix_count(previous,bank,present):
    expected=previous['banks'][bank]['accepted'] if previous is not None else 0
    u.require(present==expected,'Accepted bank journal count changed outside recorded prior attempt; no replay/duplicate')
    return expected

def accepted_progress(previous,reports,new):
    prior_count=sum(v['accepted'] for v in previous['banks'].values()) if previous is not None else 0
    u.require(type(new) is int and new>=0 and all(0<=v['accepted']<=v['questions'] and v['missing']==v['questions']-v['accepted'] for v in reports.values())
        and sum(v['accepted'] for v in reports.values())==prior_count+new,'Accepted QA progress accounting differs')
    done=all(v['missing']==0 for v in reports.values())
    u.require(done or new>0,'No progress is not a valid automatic continuation')
    return done

def run(binding_path):
    binding=u.checked(u.desc(binding_path));cfg,base=documents(binding);actual=actor(binding,cfg);began=time.monotonic()
    # Required actual final completion and every full-bank readiness check happen
    # immediately. This is not an empty GPU waiter for future student/media jobs.
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    fd=os.open(out/'EXECUTION.lock',os.O_RDWR|os.O_CREAT,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:os.close(fd);raise
    attempt=out/('r'+str(actual['restart']));attempt.mkdir(exist_ok=False)
    try:
        previous=None
        if actual['restart']:
            previous=u.checked(u.desc(out/('r'+str(actual['restart']-1))/'RESULT.json'))
            intent=u.checked(u.desc(out/('r'+str(actual['restart']-1))/'REQUEUE_INTENT.json'))
            u.require(previous['status']=='ACCEPTED_QA_PROGRESS_REQUIRES_SAME_JOB_CONTINUATION'
                and previous['new_answers']>0
                and previous['configuration']==binding['configuration'] and previous['actual']['job_id']==actual['job_id']
                and previous['actual']['nonce']==actual['nonce'] and previous['actual']['restart']==actual['restart']-1
                and intent['next_restart']==actual['restart'] and intent['previous_result']==u.desc(out/('r'+str(actual['restart']-1))/'RESULT.json'),
                'Failed/unaccepted predecessor is not automatic continuation')
            u.require(not (out/('r'+str(actual['restart']-1))/'FAILURE.json').exists(),'Failed predecessor needs explicit Root recovery, not automatic continuation')
        training,complete_ref,complete=complete_student(cfg)
        u.publish(attempt/'START.json',dict(actual=actual,configuration=binding['configuration'],student_completion=complete_ref))
        bankdata={b:banks.load(b,cfg) for b in cfg['banks']}
        ready={}
        for bank,(roster,inventory,targets,physical,snapshot) in bankdata.items():
            store=media.Store(physical,roster['records'],inventory,snapshot)
            try:
                names=sorted({r['native_input']['image'] for r in roster['records']})
                for name in names:
                    image=store.rgb(name);image.close()
                ready[bank]=dict(questions=len(roster['records']),all_encoded_and_RGB_verified=True,images=len(names),forwards=0)
            finally:store.close()
        u.publish(attempt/'FULL_MEDIA_READINESS.json',ready)
        publish_same(out/'STUDENT_INPUT.json',dict(training_configuration=cfg['training_configuration'],student_completion=complete_ref,accepted=complete['accepted']))
        runtime=qualified.official.runtime(base,attempt);u.publish(attempt/'RUNTIME.json',runtime)
        with qualified.official.offline():
            loaded3,proof=model.compose(complete['accepted'],cfg['training_configuration'])
            identity=model.identity(loaded3,proof,runtime);identity_ref=publish_same(out/'MODEL_IDENTITY.json',identity)
            u.publish(attempt/'MODEL_RECONSTRUCTION.json',proof)
            loaded=(*loaded3,getattr(loaded3[1].config,'max_sequence_length',2048));reports={};new=0
            for bank,(roster,inventory,targets,physical,snapshot) in bankdata.items():
                rows=roster['records'];path=out/bank;path.mkdir(exist_ok=True)
                fn=function(cfg,identity,roster,complete_ref,bank);publish_same(path/'FUNCTION.json',fn)
                j=generation.AuditedJournal(path/'journal',fn,rows);store=media.Store(physical,rows,inventory,snapshot)
                try:
                    for entry in qualified.official.entries(j,rows):generation.validate_entry(bank,entry,j.roster[entry['semantic_key']],fn)
                    accepted_prefix_count(previous,bank,len(j.present))
                    before=len(j.present)
                    if time.time()<actual['end_epoch']-300:
                        generation.native(bank,cfg,rows,j,loaded,store,actual,
                            time.monotonic()+max(0,actual['end_epoch']-time.time()-240))
                    new+=len(j.present)-before;reports[bank]=dict(questions=len(rows),accepted=len(j.present),missing=len(j.missing()),new=len(j.present)-before)
                    if not j.missing():
                        raw=j.export(path/'PREDICTIONS.jsonl');score=banks.score(bank,roster,j,targets)
                        publish_same(path/'SCORE.json',dict(status='COMPLETE_TRAINED_NATIVE_BANK_PREDICTIONS_AND_OFFICIAL_SCORING',
                            bank=bank,function=fn,predictions=raw,score=score,student_completion=complete_ref,complete_questions=len(rows),
                            scientific_improvement_demonstrated=False))
                finally:j.close();store.close()
                budget(cfg)
        done=accepted_progress(previous,reports,new)
        status='COMPLETE_FINAL_NATIVE_STUDENT_FOUR_OFFICIAL_BANKS' if done else 'ACCEPTED_QA_PROGRESS_REQUIRES_SAME_JOB_CONTINUATION'
        value=dict(status=status,actual=actual,configuration=binding['configuration'],student_completion=complete_ref,accepted_checkpoint=complete['accepted'],
            model_identity=identity_ref,banks=reports,new_answers=new,elapsed_seconds=time.monotonic()-began,all_prior_and_current_costs_count=True,
            scientific_improvement_demonstrated=False)
        result_ref=u.publish(attempt/'RESULT.json',value)
        if done:publish_same(out/'COMPLETE.json',value)
        else:
            u.require(new>0 and actual['restart']+1<cfg['max_physical_attempts'],'No progress/finite continuation exhausted; Root diagnosis required')
            intent=u.publish(attempt/'REQUEUE_INTENT.json',dict(job_id=actual['job_id'],nonce=actual['nonce'],next_restart=actual['restart']+1,
                previous_result=result_ref,configuration=binding['configuration'],same_job_only=True,accepted_per_QA_journal=True))
            result=subprocess.run(['scontrol','requeue',actual['job_id']],text=True,capture_output=True,timeout=30)
            u.publish(attempt/'REQUEUE_ACK.json',dict(intent=intent,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                ambiguous_ack_requires_Root_reconciliation=True))
            u.require(result.returncode==0,'Requeue failed/ambiguous; do not duplicate successor')
    except BaseException as e:
        u.publish(attempt/'FAILURE.json',dict(status='TRAINED_EVALUATION_FAILED_PRESERVE_ALL_QAS_NO_AUTO_RETRY',actual=actual,error=repr(e),elapsed_seconds=time.monotonic()-began));raise
    finally:os.close(fd)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);a=p.parse_args();run(a.binding)
