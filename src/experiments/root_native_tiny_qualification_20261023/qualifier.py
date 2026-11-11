"""One real H100, one original author batch8, engineering evidence only."""
import argparse
import fcntl
import importlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_campaign_20261023.config import validate_profile
from experiments.root_native_campaign_20261023.selection import validate_pool
from experiments.root_native_baselines_20261023 import features,pins
from experiments.root_native_baselines_20261023.store import Store
from .prepare import sources,one_batch


def validate(config):
    c.require(config['schema']=='tiny-native-batch8-qualification-config-v1' and config['owner']=='Root'
        and config['method']=='coincide' and config['role']=='qualify_features'
        and config['engineering_only'] is True and config['full_pool_qualified'] is False,'Only bounded Tiny engineering work authorized')
    c.require(config['qualifier_sources']==sources(),'Qualifier source changed')
    pins.verify(config['sources']);c.verify_sources(config['native_sources'])
    c.require(config['measurement_contract']==features.CONTRACTS['coincide'],'Actual author feature contract changed')
    pool,rows,ids=validate_pool(config['pool'],allow_engineering=True)
    c.require(config['exact_batch']==one_batch(ids,rows),'Original batch8 or padding peers changed')
    c.require(config['shard_index']==config['exact_batch']['shard_index'] and config['shard_count']==config['exact_batch']['shard_count'],'Native batch stripe changed')
    profile=validate_profile(c.checked(config['site_profile']))
    c.require(profile['site']=='aces' and profile['gpu_type']=='h100' and Path(profile['code_root']).resolve()==c.ROOT.resolve(),'Exact staged ACES H100 profile required')
    env=c.checked(config['environment'])
    c.require(env['status']=='PASS_ENVIRONMENT' and env['python_executable']==profile['python']
        and '/envs/tiny-' in profile['python'] and 'native-py310-cu121-v1' not in profile['python'],'Separate actual Tiny environment required')
    expected=dict(torch='2.2.0+cu121',torchvision='0.17.0+cu121',transformers='4.37.2',tiktoken='0.5.2',numpy='1.26.4')
    c.require(all(env['versions'].get(k)==v for k,v in expected.items()),'Explicit proxy runtime pins differ')
    assets=c.checked(config['feature_assets']);c.require(assets['status']=='PASS_PINNED_TINY_PROXY_ASSETS','Actual full Tiny snapshot receipt absent')
    for role in ('tiny_proxy','tiny_siglip'):
        pins.verify_tiny_snapshot(assets['assets'][role],role)
        c.require(c.within(assets['assets'][role]['snapshot'],profile['cache_root']),'Tiny assets outside approved cache')
    eligibility=c.checked(config['eligibility'])
    c.require(eligibility['status']=='PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY' and eligibility['pool']==config['pool']
        and eligibility['missing_media_count']==eligibility['protected_QA_image_RGB_overlap_count']==0,'Actual engineering input protection/media absent')
    c.require(config['allocation']==dict(gpus=1,cpus=8,memory_mb=65536,minutes=55,reserve_seconds=300),'Finite1H100 explicit CPU+GPU resource contract differs')
    c.require(c.within(config['output'],profile['output_root']),'Unapproved qualifier output')
    identity=features.measurement_identity(config,pool)
    c.require(config['measurement_identity']==identity,'Qualifier measurement identity differs')
    claim=c.checked(config['ownership'])
    c.require(claim['status']=='ACTIVE_EXCLUSIVE_TINY_QUALIFIER' and claim['owner']=='Root'
        and claim['measurement_identity']==identity and claim['record_ids']==config['exact_batch']['record_ids']
        and claim['site_profile']==config['site_profile'],'Exact eight-record ownership absent')
    reserve=c.checked(config['storage_reservation'])
    c.require(reserve['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and reserve['output']==config['output']
        and reserve['measurement_identity']==identity,'Measured qualifier storage absent')
    review=c.checked(config['review'])
    c.require(review['status']=='PASS_INDEPENDENT_EXACT_TINY_QUALIFIER_CONFIG' and review['qualifier_sources']==config['qualifier_sources']
        and review['published_sources']==config['sources'] and review['native_sources']==config['native_sources']
        and review['config_identity']==c.key({k:v for k,v in config.items() if k!='review'}),'Exact independent qualifier/config review absent')
    return profile,env,pool,rows,ids,identity


def actual(binding,config,profile):
    job=os.environ.get('SLURM_JOB_ID','')
    c.require(job.isdecimal() and job==binding['job_id'] and not os.environ.get('SLURM_ARRAY_JOB_ID'),'Exact scalar qualifier job required')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=25)
    fields=dict(re.findall(r'(\w+)=(\S+)',raw))
    expected=dict(JobState='RUNNING',JobName='dsmm-tiny-qual8-v1',Account=profile['account'],
        Partition=profile['partition'],QOS=profile['qos'],NumNodes='1',NumTasks='1',NumCPUs='8',
        WorkDir=profile['code_root'],Command=str(Path(__file__).parent/'run.sbatch'),TresPerNode='gres/gpu:h100:1',TimeLimit='00:55:00')
    c.require(all(fields.get(k)==v for k,v in expected.items()),'Actual Tiny qualifier route/resources/command differ')
    c.require(fields['UserId'].split('(')[0]==profile['slurm_user'] and fields['NodeList']==os.uname().nodename.split('.')[0],'Actual qualifier user/node differs')
    c.require(fields.get('Comment')=='dsmm-tiny-qual8:'+c.checked(binding['intent'])['nonce'],'Qualifier ownership nonce differs')
    tres=dict(item.split('=',1) for item in fields['ReqTRES'].split(',') if '=' in item)
    c.require(tres.get('mem') in ('64G','65536M'),'Actual qualifier host-memory request differs')
    return raw


def main():
    started=time.monotonic();parser=argparse.ArgumentParser();parser.add_argument('--binding',required=True);a=parser.parse_args()
    binding=c.read(a.binding)
    c.require(binding['schema']=='tiny-native-batch8-qualification-binding-v1' and binding['owner']=='Root','Wrong qualifier binding')
    config=c.checked(binding['configuration']);profile,env,pool,rows,ids,identity=validate(config)
    c.require(Path(sys.executable).absolute()==Path(profile['python']).absolute(),'Run the isolated reviewed Torch2.2 interpreter')
    raw=actual(binding,config,profile)
    os.environ.update(HF_HOME=profile['cache_root'],HF_HUB_CACHE=str(Path(profile['cache_root'])/'hub'),HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    runtime={}
    for name in ('torch','torchvision','transformers','tiktoken','numpy'):
        module=importlib.import_module(name);version=getattr(module,'__version__',None)
        if version is None:
            from importlib.metadata import version as distribution_version
            version=distribution_version(name)
        c.require(version==env['versions'][name],'Actual isolated package version differs: '+name);runtime[name]=version
    import torch
    c.require(torch.cuda.is_available() and torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0),'One actual visible H100 required')
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    lock=open(output/'EXECUTION.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=output/f"attempt-{binding['job_id']}-r{os.environ.get('SLURM_RESTART_COUNT','0')}"
    c.seal(attempt/'START.json',dict(binding=c.desc(a.binding),actual_slurm=raw,runtime=runtime,UTC=c.now(),actual_GPU=torch.cuda.get_device_name(0)))
    store=Store(output/'features.sqlite3',identity,config['exact_batch']['record_ids'],'coincide')
    before=len(store.existing)
    requested=[False]
    def stop(*_):requested[0]=True
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    deadline=int(os.environ['SLURM_JOB_END_TIME'])-300
    should_stop=lambda:requested[0] or time.time()>=deadline
    sys.path.insert(0,str(pins.COIN_CODE));module=importlib.import_module('tinyllava.eval.score.coincide.extract_embed')
    original_loader=module.load_pretrained_model;original_collator=module.DataCollatorForSupervisedDataset
    def audited_loader(*args,**kwargs):
        tokenizer,model,processor,context=original_loader(*args,**kwargs)
        attention=[type(layer.self_attn).__module__+'.'+type(layer.self_attn).__name__ for layer in model.model.layers]
        c.require(model.config._attn_implementation=='eager' and all(n.endswith('.Attention') for n in attention),'Proxy attention differs from author eager path')
        c.seal(attempt/'MODEL_AND_PROCESSOR_AUDIT.json',dict(model=type(model).__module__+'.'+type(model).__name__,
            tokenizer=type(tokenizer).__module__+'.'+type(tokenizer).__name__,processor=type(processor).__module__+'.'+type(processor).__name__,
            attention=attention,attn_implementation=model.config._attn_implementation,context=context,
            runtime=runtime,author_tiktoken_unpinned_our_runtime_pin='0.5.2'))
        return tokenizer,model,processor,context
    class AuditedCollator(original_collator):
        def __call__(self,instances):
            batch=super().__call__(instances)
            lengths=[min(len(row['input_ids']),self.tokenizer.model_max_length) for row in instances]
            c.require(len(instances)==8 and max(lengths)>min(lengths) and batch['input_ids'].shape[1]==max(lengths),
                'Actual qualifier batch has no unequal-length padding; diagnose engineering coverage')
            for j,row in enumerate(instances):
                length=min(len(row['input_ids']),self.tokenizer.model_max_length)
                c.require(torch.equal(batch['input_ids'][j,:length],row['input_ids'][:length]),'Collation changed author tokens')
                c.require(torch.equal(batch['labels'][j,:length],row['labels'][:length]),'Collation changed author masks')
            c.seal(attempt/'ACTUAL_BATCH_PADDING.json',dict(records=8,valid_tokens=batch['attention_mask'].sum(dim=1).tolist(),
                original_token_lengths=lengths,padded_positions=[max(lengths)-n for n in lengths],
                attention_mask_false_tokens=(~batch['attention_mask']).sum(dim=1).tolist(),supervised_tokens=(batch['labels']!=-100).sum(dim=1).tolist(),
                native_tokens_and_labels_preserved=True,full_pool_qualified=False))
            return batch
    module.load_pretrained_model=audited_loader;module.DataCollatorForSupervisedDataset=AuditedCollator
    pool=dict(pool,_verified_image_assets=c.checked(pool['image_manifest'])['assets'])
    try:
        features.coincide(config,pool,rows,ids,store,should_stop,engineering=True)
        c.require(len(store.existing)==8,'Engineering batch did not complete; preserve raw partial state, no full PASS')
    except Exception as error:
        c.seal(attempt/'FAILURE.json',dict(error_type=type(error).__name__,error=str(error),GPU_hours=(time.monotonic()-started)/3600,
            actual_records=len(store.existing),qualification_promoted=False));raise
    finally:
        module.load_pretrained_model=original_loader;module.DataCollatorForSupervisedDataset=original_collator;store.close()
    c.seal(attempt/'RESULT.json',dict(status='COMPLETE_ACTUAL_TINY_BATCH8_ENGINEERING_EVIDENCE' if before==0 else 'ALREADY_COMPLETE_EXISTING_ENGINEERING_EVIDENCE_ONLY',records=8,
        record_ids=config['exact_batch']['record_ids'],database=c.desc(output/'features.sqlite3'),GPU_hours=(time.monotonic()-started)/3600,
        newly_acquired_records=8-before,cached_records=before,qualification_promoted=False,full_pool_features_qualified=False,benchmark_or_method_effect_measured=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
