"""Two real POPE QAs on paid checkpoint28, fresh reload, no production score."""
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid
from types import SimpleNamespace

from experiments.root_gqa_bank_qualification_v1_20261026 import worker as u
from experiments.root_native_trained_loader_v4_20261026 import model
from experiments.root_babel_start_evaluation_v2_20261025 import worker as official
from experiments.session6_native_start_execution_v6_20261024 import audit
from experiments.session6_official_eval_v2_20261024.journal import Journal

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
MODULE='experiments.root_native_trained_eval_qual_v4_20261026.worker'
SCOPE='PAID_CANARY28_TRAINED_ADAPTER_TWO_POPE_QA_FRESH_RELOAD_NOT_FORMAL_SCORE'
PYTHON='/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python'

def sources(base,phase3):
    result=dict(base['sources'])
    for k,v in phase3['producer_sources'].items():
        u.require(k not in result or result[k]==v,'Conflicting frozen official/native source')
        result[k]=v
    result.update(u.sources())
    for directory in (HERE,Path(model.__file__).parent):
        for p in directory.iterdir():
            if p.suffix in ('.py','.sbatch') and not p.name.startswith('test_'):result[str(p.relative_to(ROOT))]=u.sha(p.read_bytes())
    result.update(model.sources())
    return dict(sorted(result.items()))

def documents(binding,load=u.checked):
    cfg=load(binding['configuration']);review=load(binding['independent_source_config_review'])
    base=load(cfg['official_POPE_configuration']);phase3=load(cfg['phase3_configuration'])
    u.require(cfg['status']=='ROOT_CONFIGURED_NATIVE_TRAINED_EVAL_QUALIFICATION_V4' and cfg['owner']=='Root'
        and cfg['scope']==SCOPE and cfg['source_root']==str(ROOT) and cfg['sources']==sources(base,phase3)
        and cfg['resources']==dict(cpus=8,memory_mb=65536,gpus=1,minutes=60)
        and cfg['banks']==['POPE'] and cfg['questions_per_bank']==2 and cfg['fresh_model_processes']==2
        and cfg['decoding']==base['decoding'] and base['bank']=='POPE'
        and cfg['production_scoring_authorized'] is False and cfg['frozen_base_copy_authorized'] is False,
        'Exact bounded trained-model qualification scope required')
    provenance=load(cfg['pinned_PEFT_source_export']);peft=provenance['peft']
    u.require(cfg['generation_API_policy']==model.generation.POLICY and provenance['neural_forwards']==0
        and peft['version']=='0.7.1' and peft['files']['peft/peft_model.py']['sha256']==model.generation.PEFT_MODEL_SHA
        and peft['files']['peft/__init__.py']['sha256']==model.generation.PEFT_INIT_SHA,
        'Actual pinned PEFT source/API acquisition evidence required, not runtime PASS')
    canary=load(cfg['actual_shared_canary']);canary_review=load(cfg['actual_shared_canary_review'])
    u.require(cfg['actual_shared_canary']==phase3['evidence']['actual_v4_canary']
        and cfg['actual_shared_canary']['sha256']=='17f488f922340197169418df322f17df4bb87db2ec62d0cfe1d4dc70885c85b7'
        and cfg['actual_shared_canary_review']==phase3['evidence']['actual_v4_canary_review']
        and canary['accepted_prefix_end']==28 and canary['actual_same_job_restore'] is True
        and canary['ordinary_accepted_checkpoint_control'] is True and canary_review['independent'] is True,
        'Actual paid shared canary and independent review required')
    continuation=load(canary['accepted_segments'][-1])
    u.require(continuation['accepted']==cfg['accepted_checkpoint'] and continuation['global_step']==28
        and continuation['configuration']==cfg['training_configuration'] and continuation['config_identity']==canary['config_identity'],
        'Exact paid accepted28 checkpoint/training configuration required')
    u.require(review['status']=='PASS_INDEPENDENT_NATIVE_TRAINED_EVAL_QUAL_SOURCE_CONFIG_V4'
        and review['independent'] is True and review['reviewer'] not in ('Root','/root/telemetry_finish','')
        and review['configuration']==binding['configuration'] and review['sources']==cfg['sources'],
        'Independent exact trained qualification source/config review required')
    for k,v in cfg['sources'].items():u.require(u.sha((ROOT/k).read_bytes())==v,'Frozen native/official source changed: '+k)
    claim=load(binding['ownership']);grant=load(binding['storage_reservation'])
    u.require(binding['status']=='ROOT_BOUND_NATIVE_TRAINED_EVAL_QUALIFICATION_V4' and binding['owner']=='Root'
        and claim['owner']=='Root' and claim['scope']==SCOPE and claim['configuration']==binding['configuration']
        and claim['job_id']==binding['job_id'] and claim['nonce']==binding['nonce']==cfg['nonce']
        and claim['accepted_checkpoint']==cfg['accepted_checkpoint'] and claim['no_live_duplicate_qualification'] is True,
        'Exclusive actual checkpoint qualification owner required')
    u.require(grant['configuration']==binding['configuration'] and grant['output']==cfg['output']
        and grant['reserved_bytes']==64<<20 and grant['accepted_checkpoint_retained_for_consumer'] is True,
        'Small shared output grant and active checkpoint retention required')
    u.require(u.literal(cfg['output']).is_relative_to(Path('/data/hf_cache/project_features/DS-MultiModal/native_trained_eval_qualification_20261026')),
        'Separate trained evaluation qualification output required')
    return cfg,base

def actor(binding,cfg):
    job=binding['job_id'];u.require(job.isdecimal() and os.getuid()==2708899 and os.environ.get('SLURM_JOB_ID')==job
        and not os.environ.get('SLURM_ARRAY_JOB_ID') and os.environ.get('SLURM_RESTART_COUNT','0')=='0','Fresh real scalar qualification actor required')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30);f=u.fields(raw)
    wanted=dict(JobId=job,JobState='RUNNING',JobName='dsmm-native-trained-eval-qual-v4',UserId='luyangz(2708899)',Partition='general',Account='cx',QOS='normal',
        NumCPUs='8',NumTasks='1',NumNodes='1',Restarts='0',TimeLimit='01:00:00',TresPerNode='gres/gpu:L40S:1',
        Command=str(HERE/'run.sbatch'),WorkDir=str(ROOT),Comment='dsmm-trained-eval-qual:'+cfg['nonce'],NodeList=socket.gethostname().split('.')[0])
    u.require(all(f.get(k)==v for k,v in wanted.items()),'Actual trained qualification GeneralCPU/GPU route differs')
    tres=dict(x.split('=',1) for x in f['AllocTRES'].split(',') if '=' in x)
    u.require(tres.get('cpu')=='8' and tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1','Actual resources differ')
    u.require(subprocess.check_output(['scontrol','write','batch_script',job,'-'],timeout=30)==(HERE/'run.sbatch').read_bytes(), 'Actual qualification spool differs')
    u.require(os.environ.get('TZ')=='America/New_York','Pinned scheduler timezone required');time.tzset()
    return dict(job_id=job,restart=0,node=f['NodeList'],nonce=cfg['nonce'],scheduler=raw,
        end_epoch=int(time.mktime(time.strptime(f['EndTime'],'%Y-%m-%dT%H:%M:%S'))))

def trained_function(cfg,identity,roster,bank='POPE'):
    return dict(schema='native-trained-active-adapter-official-bank-function-v1',bank=bank,role='ENGINEERING_CHECKPOINT_RELOAD_QUALIFICATION_ONLY',
        accepted_checkpoint=cfg['accepted_checkpoint'],training_configuration=cfg['training_configuration'],
        model_identity_sha256=official.c.key(identity),model_generation_config=identity['generation_config'],
        training_optimizer_updates=identity['training_optimizer_updates'],adapter_active=True,adapter_merged=False,
        full_roster_sha256=official.c.key(roster),decoding=cfg['decoding'],sources_sha256=official.c.key(cfg['sources']))

class TrainedJournal(Journal):
    def append(self,key,raw,record):
        audit.validate_entry(dict(semantic_key=key,raw=raw,audit=record),self.roster[key],self.function)
        return super().append(key,raw,record)

def native(cfg,rows,journal,loaded,store,current,deadline):
    """Original native loader body; adapter label and physical image feed only."""
    from experiments.session6_native_start_execution_v6_20261024 import kernel
    missing=journal.missing()
    recorder=audit.Recorder(missing,loaded,dict(images=store.images),str(store.virtual),journal.function,current)
    args=SimpleNamespace(**cfg['decoding'],num_chunks=1,chunk_idx=0,model_base=None,
        model_path=str(Path(cfg['output'])/'TRAINED-Stage1-PEFT-native-c121'),image_folder=str(store.virtual),
        question_file=str(Path(cfg['output'])/'native-source-view.jsonl'),answers_file=str(journal.directory/'native-answers.jsonl'))
    module=importlib.import_module(kernel.MODULES['loader']);defaults=module.create_data_loader.__defaults__
    u.require(defaults==(1,4),'Pinned native loader defaults changed')
    previous=audit.expected_inputs;module.create_data_loader.__defaults__=(1,0)
    audit.expected_inputs=lambda row,value,inventory,root:official.expected_inputs(row,value,store)
    try:
        with store.installed(),recorder.installed():
            return kernel.evaluate('loader',args,rows,journal,loaded,deadline,
                {kernel.MODULES['loader']:official.c.desc(official.c.LLAVA/'llava/eval/model_vqa_loader.py')},recorder.take,
                stop_requested=lambda:False)
    finally:
        audit.expected_inputs=previous;module.create_data_loader.__defaults__=defaults

def child(binding_path,phase):
    import torch
    binding=u.checked(u.desc(binding_path));cfg,base=documents(binding);current=actor(binding,cfg)
    current.update(process_nonce=uuid.uuid4().hex,pid=os.getpid(),phase=phase)
    out=Path(cfg['output'])/phase;out.mkdir(exist_ok=False)
    runtime=official.runtime(base,out);u.publish(out/'RUNTIME.json',runtime)
    roster,inventory,targets,snapshot=official.bound_inputs(base);rows=roster['records'][:2]
    physical=dict(base,output=str(out));store=official.media.Store(physical,roster['records'],inventory,snapshot)
    with official.offline():
        loaded3,proof=model.compose(cfg['accepted_checkpoint'],cfg['training_configuration'],diagnostic_path=out/'LOADED_PARAMETER_INVENTORY.json')
        identity=model.identity(loaded3,proof,runtime);u.publish(out/'MODEL_IDENTITY.json',identity);u.publish(out/'MODEL_RECONSTRUCTION.json',proof)
        function=trained_function(cfg,identity,roster)
        loaded=(*loaded3,getattr(loaded3[1].config,'max_sequence_length',2048))
        journal=TrainedJournal(out/'journal',function,rows)
        try:
            result=native(cfg,rows,journal,loaded,store,current,time.monotonic()+max(0,current['end_epoch']-time.time()-180))
            entries=official.entries(journal,rows)
            u.require(not journal.missing() and len(entries)==2,'Exactly two complete real native QAs required')
            after=model.identity(loaded3,proof,runtime)
            u.require(after==identity,'Native generation changed parameter/buffer/tokenizer/processor identity')
            u.publish(out/'POST_GENERATION_MODEL_IDENTITY.json',after)
            bridge=model.generation.readout(loaded3[1])
            u.require(bridge['calls']==2 and all(r['success'] for r in bridge['records']),'Exactly two successful original native bridge calls')
            u.publish(out/'GENERATION_BRIDGE_READOUT.json',bridge)
        finally:journal.close();store.close()
        return u.publish(out/'RESULT.json',dict(status='COMPLETE_TWO_REAL_TRAINED_QAS',actual=current,function=function,entries=entries,
            model_identity=u.desc(out/'MODEL_IDENTITY.json'),reconstruction=u.desc(out/'MODEL_RECONSTRUCTION.json'),new_answers=result['new_answers'],
            post_generation_model_identity=u.desc(out/'POST_GENERATION_MODEL_IDENTITY.json'),generation_bridge_readout=u.desc(out/'GENERATION_BRIDGE_READOUT.json')))

def compare(first,second,load=u.checked):
    u.require(first['status']==second['status']=='COMPLETE_TWO_REAL_TRAINED_QAS' and first['function']==second['function']
        and first['function']['adapter_active'] is True and first['function']['training_optimizer_updates']>0,
        'Two same trained/checkpoint function outputs required, never START')
    u.require(load(first['model_identity'])==load(second['model_identity']),'Fresh reconstructed parameter/buffer/tokenizer/processor identity differs')
    for child in(first,second):
        u.require(load(child['post_generation_model_identity'])==load(child['model_identity']),'Actual post-generation full model identity differs')
        b=load(child['generation_bridge_readout'])
        u.require(b['status']=='COMPLETE_PINNED_PEFT071_NATIVE_POSITIONAL_BRIDGE_CALLS'and b['runtime_API_proof']==load(child['model_identity'])['generation_API_bridge']
            and b['policy']==model.generation.POLICY and b['calls']==len(b['records'])==2 and b['adapter_active']is True and b['adapter_merged']is False
            and all(all(r[k]is True for k in('success','input_object_unchanged','all_keyword_objects_unchanged','prepare_hook_restored','original_PEFT_generate_invoked','output_object_unchanged'))and r['input_keyword']=='inputs'for r in b['records']),
            'Four actual original PEFT/native API bridge calls required')
    official.same_repeat(first['entries'],second['entries'])
    u.require(first['actual']['job_id']==second['actual']['job_id'] and first['actual']['nonce']==second['actual']['nonce']
        and first['actual']['process_nonce']!=second['actual']['process_nonce'],'Two genuine fresh model processes required')
    return dict(status='PASS_ACTUAL_NATIVE_TRAINED_ADAPTER_TWO_QA_FRESH_RELOAD',function=first['function'],
        neural_outputs=4,distinct_QAs=2,adapter_active=True,adapter_merged=False,
        full_bank_scored=False,formal_student_evaluation_complete=False,scientific_improvement_demonstrated=False)

def run(binding_path):
    binding=u.checked(u.desc(binding_path));cfg,_=documents(binding);actual=actor(binding,cfg)
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    u.publish(out/'START.json',dict(actual=actual,configuration=binding['configuration']))
    try:
        refs=[]
        for phase in ('first','repeat'):
            argv=[PYTHON,'-B','-s','-u','-m',MODULE,'--binding',str(binding_path),'--child',phase]
            u.publish(out/(phase.upper()+'_INTENT.json'),dict(argv=argv,actual=actual))
            result=subprocess.run(argv,timeout=max(1,actual['end_epoch']-time.time()-120),check=False)
            u.require(result.returncode==0,'Fresh trained child failed; preserve evidence, no blindretry')
            refs.append(u.desc(out/phase/'RESULT.json'))
        value=compare(*[u.checked(r) for r in refs]);value.update(actual=actual,configuration=binding['configuration'],sources=cfg['sources'],
            accepted_checkpoint=cfg['accepted_checkpoint'],engineering_canary_only=True,child_results=refs,
            elapsed_seconds=time.monotonic()-began,all_prior_training_and_current_costs_count=True)
        u.publish(out/'RESULT.json',value)
    except BaseException as e:
        u.publish(out/'FAILURE.json',dict(status='TRAINED_EVALUATION_QUALIFICATION_FAILED_NO_AUTO_RETRY',actual=actual,error=repr(e),elapsed_seconds=time.monotonic()-began));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--child',choices=['first','repeat']);a=p.parse_args()
    child(a.binding,a.child) if a.child else run(a.binding)
