"""Finite owned CPU+GPU provider, fail-closed before paid acquisition.

No scheduler admission or automatic successor submission occurs here. A partial
feature shard can requeue only its exact physical job, with immutable intent/ACK.
Qualification output is engineering evidence, never a full-pool PASS receipt.
"""
import argparse
import importlib
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_campaign_20261023.config import validate_profile,ASSET_PINS
from experiments.root_native_campaign_20261023.selection import validate_pool
from . import pins,features
from .store import Store


def actual_environment(descriptor,postprocess=False):
    env=c.checked(descriptor)
    required=('torch','transformers','numpy')+ (('scipy','faiss') if postprocess else ())
    observed={}
    for name in required:
        module=importlib.import_module(name);version=module.__version__
        c.require(env['versions'].get(name)==version,'Actual provider package version changed: '+name)
        observed[name]=version
    return observed


def validate(config):
    c.require(config['schema']=='published-native-feature-job-v1' and config['owner']=='Root','Wrong provider owner/schema')
    c.require(config['method'] in features.CONTRACTS and config['role'] in ('qualify_features','features'),'Wrong finite published acquisition role')
    c.require(config['measurement_contract']==features.CONTRACTS[config['method']],'Published feature contract altered')
    c.verify_sources(config['native_sources']);pins.verify(config['sources'])
    profile=validate_profile(c.checked(config['site_profile']))
    c.require(Path(profile['code_root']).resolve()==c.ROOT.resolve(),'Wrong staged source tree')
    pool,rows,ids=validate_pool(config['pool'],allow_engineering=config['role']=='qualify_features')
    eligibility=c.checked(config['eligibility'])
    c.require(eligibility['status']=='PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY' and eligibility['pool']==config['pool']
        and eligibility['protected_QA_image_RGB_overlap_count']==0 and eligibility['missing_media_count']==0,'Broad pool protected/media qualification absent')
    env=c.checked(config['environment'])
    c.require(env['status']=='PASS_ENVIRONMENT' and env['python_executable']==profile['python'],'Actual exact provider environment required')
    c.require(env['versions']['transformers']=='4.37.2' and env['versions']['torch'].startswith('2.1.2'),'Pinned native API runtime required')
    assets=c.checked(config['feature_assets'])
    if config['method']=='datatailor':
        c.require(assets['status']=='PASS_ASSETS' and assets['official_commit']==c.PIN,'DataTailor requires the actual frozen Stage1 assets')
        for name,pin in ASSET_PINS.items():
            value=assets['assets'][name]
            c.require((value['repository'],value['revision'])==pin,'DataTailor Stage1 identity differs')
        c.require(env['attention']=='xformers','DataTailor source adaptation requires the separately qualified native attention path')
    else:
        c.require(assets['status']=='PASS_PINNED_TINY_PROXY_ASSETS','COINCIDE requires its own proxy assets, not Stage1/Qwen features')
        tiny=assets['assets']['tiny_proxy'];vision=assets['assets']['tiny_siglip']
        pins.verify_tiny_snapshot(tiny,'tiny_proxy');pins.verify_tiny_snapshot(vision,'tiny_siglip')
        c.require((tiny['repository'],tiny['revision'])==(pins.TINY_REPOSITORY,pins.TINY_REVISION),'Tiny proxy pin differs')
        c.require((vision['repository'],vision['revision'])==('bczhou/TinyLLaVA-2.0B-SigLIP','5e462ef0248c232e7a68486da1fb6c2371f994d2'),'Tiny vision pin differs')
        for asset,expected_sha,expected_size in ((tiny,pins.TINY_WEIGHTS_SHA,pins.TINY_WEIGHTS_BYTES),
            (vision,'f601859f6bfb0d0e55282547fdab4f5fb0a1ae75f03f4edb6335cb791963326d',795547232)):
            weights=[d for d in asset['files'] if Path(d['path']).name=='model.safetensors']
            c.require(len(weights)==1 and weights[0]['sha256']==expected_sha and weights[0]['bytes']==expected_size,'Proxy model weights not exact')
    for value in assets['assets'].values():
        c.require(c.within(value['snapshot'],profile['cache_root']),'Published base assets must use approved shared cache')
        for descriptor in value['files']:c.require(c.desc(descriptor['path'])=={k:descriptor[k] for k in ('path','bytes','sha256')},'Feature asset bytes changed')
    identity=features.measurement_identity(config,pool)
    claim=c.checked(config['ownership'])
    c.require(claim['status']=='ACTIVE_EXCLUSIVE_PUBLISHED_FEATURE_SHARD' and claim['owner']=='Root'
        and claim['measurement_identity']==identity and claim['shard_index']==config['shard_index']
        and claim['shard_count']==config['shard_count'] and claim['site_profile']==config['site_profile'],'Feature ownership/measurement/shard differs')
    allocation=config['allocation']
    c.require(allocation['gpus']==1 and allocation['cpus']>=2 and allocation['memory_mb']>=32768,'Explicit CPU+GPU allocation required')
    c.require(0<allocation['reserve_seconds']<allocation['minutes']*60<=profile['max_minutes']*60,'Deadline/reserve/site duration invalid')
    c.require(allocation['cpus']<=profile['max_cpus_per_request'] and allocation['memory_mb']<=profile['max_memory_mb'],'Provider resources exceed actual inventoried site')
    c.require(c.within(config['output'],profile['output_root']),'Feature output outside approved owned root')
    storage=c.checked(config['storage_reservation'])
    c.require(storage['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and storage['output']==config['output']
        and storage['measurement_identity']==identity,'Actual collective feature storage reservation absent')
    review=c.checked(config['review'])
    c.require(review['status']=='PASS_INDEPENDENT_EXACT_PUBLISHED_FEATURE_CONFIG' and review['sources']==config['sources']
        and review['native_sources']==config['native_sources'] and review['config_identity']==c.key({k:v for k,v in config.items() if k!='review'}),'Exact independent source/configuration review required')
    if config['role']=='features':
        q=c.checked(config['qualification'])
        c.require(q['status']=='PASS_ACTUAL_PUBLISHED_FEATURE_RUNTIME' and q['method']==config['method']
            and q['sources']==config['sources'] and q['native_sources']==config['native_sources']
            and q['feature_assets']==config['feature_assets'] and q['environment']==config['environment']
            and q['measurement_contract']==config['measurement_contract'],'Actual matching model/kernel/masking/runtime qualification absent')
        for name in ('native_feature_parity','image_RGB_identity','full_conversation_masking','missing_only_restart'):
            evidence=c.checked(q['evidence'][name])
            c.require(evidence['status']=='PASS_ACTUAL_'+name.upper(),'Actual raw qualification evidence absent: '+name)
    return profile,pool,rows,ids,identity


def actual(binding,config,profile):
    job=os.environ.get('SLURM_JOB_ID','')
    c.require(job.isdecimal() and job==binding['job_id'] and not os.environ.get('SLURM_ARRAY_JOB_ID'),'Actual scalar provider job differs')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=25)
    fields=dict(re.findall(r'(\w+)=(\S+)',raw))
    for name,value in dict(JobName='dsmm-published-'+config['id'],JobState='RUNNING',Account=profile['account'],
        Partition=profile['partition'],QOS=profile['qos'],NumNodes='1',NumTasks='1',WorkDir=profile['code_root'],
        Command=str(Path(__file__).parent/'run.sbatch'),TresPerNode='gres/gpu:'+profile['gpu_type']+':1').items():
        c.require(fields.get(name)==value,'Actual provider Slurm '+name+' differs')
    c.require(fields['UserId'].split('(')[0]==profile['slurm_user'],'Actual provider user differs')
    c.require(fields.get('Comment')=='dsmm-published:'+c.checked(binding['intent'])['nonce'],'Actual provider ownership nonce differs')
    c.require(int(fields['NumCPUs'])>=config['allocation']['cpus'],'Actual CPUs insufficient')
    tres=dict(item.split('=',1) for item in fields['ReqTRES'].split(',') if '=' in item)
    m=re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)',tres.get('mem',''))
    c.require(m is not None and float(m[1])*{'K':1/1024,'':1,'M':1,'G':1024,'T':1024**2}[m[2]]==config['allocation']['memory_mb'],'Actual provider RAM request differs')
    return raw


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);a=p.parse_args()
    binding=c.read(a.binding)
    c.require(binding['schema']=='published-native-feature-binding-v1' and binding['owner']=='Root','Wrong provider binding')
    config=c.checked(binding['configuration']);profile,pool,rows,ids,identity=validate(config)
    if Path(sys.executable).absolute()!=Path(profile['python']).absolute():
        os.execv(profile['python'],[profile['python'],'-B','-u','-m','experiments.root_native_baselines_20261023.provider','--binding',a.binding])
    raw=actual(binding,config,profile)
    os.environ['OMP_NUM_THREADS']=str(config['allocation']['cpus'])
    os.environ['MKL_NUM_THREADS']=str(config['allocation']['cpus'])
    runtime=actual_environment(config['environment'])
    os.environ.update(HF_HOME=profile['cache_root'],HF_HUB_CACHE=str(Path(profile['cache_root'])/'hub'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    c.require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One real visible GPU required')
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    lock=open(output/'EXECUTION.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=output/f"attempt-{binding['job_id']}-r{os.environ.get('SLURM_RESTART_COUNT','0')}"
    c.seal(attempt/'START.json',dict(binding=c.desc(a.binding),measurement_identity=identity,actual_slurm=raw,runtime=runtime,UTC=c.now(),engineering_only=config['role']=='qualify_features'))
    if (output/'COMPLETE.json').exists():
        receipt=c.read(output/'COMPLETE.json')
        c.require(receipt['measurement_identity']==identity and c.desc(receipt['database']['path'])==receipt['database'],'Completed feature shard changed')
        return 0
    roster_batches=features.batches(ids,config['method'],config['shard_index'],config['shard_count'])
    engineering=config['role']=='qualify_features'
    if engineering:roster_batches=roster_batches[:(4 if config['method']=='coincide' else 32)]
    roster=[ids[i] for batch in roster_batches for i in batch]
    c.require(roster,'Empty feature shard')
    store=Store(output/'features.sqlite3',identity,roster,config['method'])
    stop=[False]
    def request(*_):stop[0]=True
    signal.signal(signal.SIGUSR1,request);signal.signal(signal.SIGTERM,request)
    deadline=int(os.environ['SLURM_JOB_END_TIME'])-config['allocation']['reserve_seconds']
    should_stop=lambda:stop[0] or time.time()>=deadline
    c.require(not should_stop(),'No allocation time after setup/reserve')
    before=len(store.existing)
    pool=dict(pool,_verified_image_assets=c.checked(pool['image_manifest'])['assets'])
    try:
        getattr(features,config['method'])(config,pool,rows,ids,store,should_stop,engineering)
        complete=len(store.existing)==len(roster)
        counts=dict(before=before,after=len(store.existing),required=len(roster))
    except Exception as error:
        c.seal(attempt/'FAILURE.json',dict(error_type=type(error).__name__,error=str(error),measurement_identity=identity,
            completed=len(store.existing),GPU_elapsed_seconds=time.monotonic()-started,failed_compute_counts=True))
        raise
    finally:store.close()
    receipt=dict(status=('COMPLETE_ENGINEERING_FEATURE_SHARD' if engineering else 'COMPLETE_PUBLISHED_FEATURE_SHARD') if complete else 'DURABLE_MISSING_ONLY_CONTINUATION',
        method=config['method'],measurement_identity=identity,pool=config['pool'],logical_pool=features.logical_pool(pool),
        feature_assets=config['feature_assets'],environment=config['environment'],measurement_contract=config['measurement_contract'],
        sources=config['sources'],native_sources=config['native_sources'],shard_index=config['shard_index'],shard_count=config['shard_count'],
        roster_sha256=c.key(roster),counts=counts,database=c.desc(output/'features.sqlite3'),engineering_only=engineering,
        setup_and_failed_costs_count=True,actual_GPU_hours=(time.monotonic()-started)/3600)
    c.seal(attempt/'RESULT.json',receipt)
    if complete:c.seal(output/'COMPLETE.json',receipt);return 0
    c.seal(attempt/'REQUEUE_INTENT.json',dict(job_id=binding['job_id'],accepted_database=receipt['database'],UTC=c.now()))
    response=subprocess.run(['scontrol','requeue',binding['job_id']],capture_output=True,text=True,timeout=25)
    c.seal(attempt/'REQUEUE_RESPONSE.json',dict(rc=response.returncode,stdout=response.stdout,stderr=response.stderr))
    c.require(response.returncode==0,'Same-job requeue unacknowledged; no replacement submission')
    return 0


if __name__=='__main__':raise SystemExit(main())
