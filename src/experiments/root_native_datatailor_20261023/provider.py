"""Explicit new native DataTailor path; no legacy constructor fallback."""
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
from experiments.root_native_campaign_v3_20261023 import common as c
from experiments.root_native_campaign_v3_20261023.config import validate_profile,ASSET_PINS
from experiments.root_native_campaign_v3_20261023.selection import validate_pool
from experiments.root_native_baselines_20261023 import pins
from experiments.root_native_baselines_20261023.store import Store
from . import features

HERE=Path(__file__).resolve().parent


def sources():
    out=pins.sources()
    out.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in HERE.iterdir()
        if p.suffix in ('.py','.sbatch') and not p.name.startswith('test_')})
    for name in ('experiments/root_native_qualification_20261023/masking.py',
                 'experiments/root_native_qualification_20261023/common.py',
                 'experiments/root_native_campaign_20261023/common.py'):
        out[name]=c.sha(c.ROOT/name)
    return dict(sorted(out.items()))


def validate(config):
    c.require(config['schema']=='native-datatailor-feature-job-v1' and config['owner']=='Root'
        and config['method']=='datatailor' and config['role'] in ('qualify_features','features'),
        'Only explicit native DataTailor feature roles authorized')
    c.require(config['sources']==sources() and config['native_sources']==c.sources(),'Bound adapter/native sources changed')
    c.require(config['measurement_contract']==features.CONTRACT,'BF16 native DataTailor contract altered')
    c.require('media_transport' not in config and 'runtime_images' not in config,
        'This adapter requires canonical persistent images; transport needs independent new qualification')
    profile=validate_profile(c.checked(config['site_profile']))
    c.require(Path(profile['code_root']).resolve()==c.ROOT.resolve(),'Wrong actual staged source root')
    pool,rows,ids=validate_pool(config['pool'],allow_engineering=config['role']=='qualify_features')
    selected=features.roster(config,rows,ids);c.require(selected,'Empty native feature shard')
    if config['role']=='qualify_features':
        c.require(config['shard_index']==0 and config['shard_count']==1
            and config['engineering_ids']==features.engineering_ids(rows,ids)
            and config['engineering_pause_after']==16,'Exact32record engineering roster with16record accepted interruption required')
    eligibility=c.checked(config['eligibility'])
    c.require(eligibility['status']=='PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY' and eligibility['pool']==config['pool']
        and eligibility['protected_QA_image_RGB_overlap_count']==0 and eligibility['missing_media_count']==0,
        'Actual protected/missing-media boundaries required')
    env=c.checked(config['environment'])
    c.require(env['status']=='PASS_ENVIRONMENT' and env['python_executable']==profile['python']
        and env['versions']['torch'].startswith('2.1.2') and env['versions']['transformers']=='4.37.2'
        and env['attention']=='xformers','Exact actual native runtime absent')
    assets=c.checked(config['feature_assets'])
    c.require(assets['status']=='PASS_ASSETS' and assets['official_commit']==c.PIN,'Actual frozen Stage1 assets required')
    for name,pin in ASSET_PINS.items():
        asset=assets['assets'][name]
        c.require((asset['repository'],asset['revision'])==pin and c.within(asset['snapshot'],profile['cache_root']),
            'Wrong Stage1 asset/cache')
        for d in asset['files']:c.require(c.desc(d['path'])=={k:d[k] for k in ('path','bytes','sha256')},'Asset bytes changed')
    identity=features.identity(config,pool);claim=c.checked(config['ownership'])
    c.require(claim['status']=='ACTIVE_EXCLUSIVE_NATIVE_DATATAILOR_FEATURE_SHARD' and claim['owner']=='Root'
        and claim['measurement_identity']==identity and claim['site_profile']==config['site_profile']
        and claim['shard_index']==config['shard_index'] and claim['shard_count']==config['shard_count']
        and claim['roster_sha256']==c.key(selected),'Exact acquisition ownership absent')
    allocation=config['allocation']
    c.require(allocation['gpus']==1 and allocation['cpus']>=2 and allocation['memory_mb']>=32768
        and 0<allocation['reserve_seconds']<allocation['minutes']*60<=profile['max_minutes']*60,
        'Explicit finite CPU+GPU allocation required')
    c.require(allocation['cpus']<=profile['max_cpus_per_request'] and allocation['memory_mb']<=profile['max_memory_mb'],
        'Site resource bounds exceeded')
    c.require(c.within(config['output'],profile['output_root']),'Unapproved feature output')
    storage=c.checked(config['storage_reservation'])
    c.require(storage['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and storage['output']==config['output']
        and storage['measurement_identity']==identity,'Actual measured feature storage reservation required')
    review=c.checked(config['review'])
    c.require(review['status']=='PASS_INDEPENDENT_EXACT_NATIVE_DATATAILOR_CONFIG' and review['sources']==config['sources']
        and review['native_sources']==config['native_sources']
        and review['config_identity']==c.key({k:v for k,v in config.items() if k!='review'}),
        'Independent exact adapter/config review absent')
    if config['role']=='features':
        q=c.checked(config['qualification'])
        c.require(q['status']=='PASS_ACTUAL_NATIVE_DATATAILOR_RUNTIME' and q['sources']==config['sources']
            and q['native_sources']==config['native_sources'] and q['feature_assets']==config['feature_assets']
            and q['environment']==config['environment'] and q['site_profile']==config['site_profile']
            and q['measurement_contract']==features.CONTRACT,'Actual matching native BF16 feature qualification required')
        for name in ('native_BF16_xformers_forward','author_SVD_parity','image_RGB_identity','full_conversation_masking','missing_only_restart'):
            raw=c.checked(q['evidence'][name])
            c.require(raw['status']=='PASS_ACTUAL_'+name.upper(),'Actual independent raw evidence absent: '+name)
    return profile,pool,rows,ids,selected,identity


def actual(binding,config,profile):
    job=os.environ.get('SLURM_JOB_ID','')
    c.require(binding['schema']=='native-datatailor-feature-binding-v1' and binding['owner']=='Root'
        and job.isdecimal() and job==binding['job_id'] and not os.environ.get('SLURM_ARRAY_JOB_ID'),'Actual scalar binding differs')
    intent=c.checked(binding['intent'])
    c.require(intent['status']=='INTENT_BEFORE_SUBMISSION' and intent['owner']=='Root'
        and intent['configuration']==binding['configuration'] and intent['sources']==config['sources'],
        'Durable exact pre-submission intent absent')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=25)
    fields=dict(re.findall(r'(\w+)=(\S+)',raw))
    exact=dict(JobId=job,JobName='dsmm-datatailor-'+config['id'],JobState='RUNNING',Account=profile['account'],
        Partition=profile['partition'],QOS=profile['qos'],NumNodes='1',NumTasks='1',WorkDir=profile['code_root'],
        Command=str(HERE/'run.sbatch'),TresPerNode='gres/gpu:'+profile['gpu_type']+':1',
        Comment='dsmm-datatailor:'+intent['nonce'],Restarts=os.environ.get('SLURM_RESTART_COUNT','0'))
    c.require(all(fields.get(k)==v for k,v in exact.items()),'Actual job route/type/command/nonce/restart differs')
    c.require(fields['UserId'].split('(')[0]==profile['slurm_user'] and fields['NodeList']==os.uname().nodename.split('.')[0],
        'Actual user/node differs')
    c.require(int(fields['NumCPUs'])>=config['allocation']['cpus'],'Insufficient actual CPUs')
    tres=dict(x.split('=',1) for x in fields['ReqTRES'].split(',') if '=' in x)
    m=re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)',tres.get('mem',''))
    c.require(m is not None and float(m[1])*{'K':1/1024,'':1,'M':1,'G':1024,'T':1024**2}[m[2]]==config['allocation']['memory_mb'],
        'Actual host-memory request differs')
    return raw


def main():
    started=time.monotonic();parser=argparse.ArgumentParser();parser.add_argument('--binding',required=True);args=parser.parse_args()
    binding=c.read(args.binding);config=c.checked(binding['configuration'])
    profile,pool,rows,ids,roster,identity=validate(config)
    c.require(Path(sys.executable).absolute()==Path(profile['python']).absolute(),'Use the independently reviewed native interpreter')
    slurm=actual(binding,config,profile)
    environment=c.checked(config['environment']);versions={}
    for name in ('torch','transformers','numpy'):
        module=importlib.import_module(name);versions[name]=module.__version__
        c.require(versions[name]==environment['versions'][name],'Actual runtime differs: '+name)
    import torch
    c.require(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported(),'One actual BF16-capable GPU required')
    os.environ.update(HF_HOME=profile['cache_root'],HF_HUB_CACHE=str(Path(profile['cache_root'])/'hub'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS=str(config['allocation']['cpus']),MKL_NUM_THREADS=str(config['allocation']['cpus']))
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    with open(output/'EXECUTION.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=output/f"attempt-{binding['job_id']}-r{os.environ.get('SLURM_RESTART_COUNT','0')}"
        c.seal(attempt/'START.json',dict(binding=c.desc(args.binding),actual_slurm=slurm,runtime=versions,UTC=c.now()))
        if (output/'COMPLETE.json').exists():
            complete=c.read(output/'COMPLETE.json')
            c.require(complete['measurement_identity']==identity and c.desc(complete['database']['path'])==complete['database'],
                'Completed feature capsule changed')
            return 0
        store=Store(output/'features.sqlite3',identity,roster,'datatailor');before=len(store.existing)
        prefix=sorted(store.existing)
        def prefix_digest():
            values=[]
            for ident in prefix:
                row=store.db.execute('SELECT record_id,hex(vector),info,tau,length,nan_count,identity FROM features WHERE record_id=?',(ident,)).fetchone()
                values.append(row)
            return c.key(values)
        preserved_prefix=prefix_digest() if config['role']=='qualify_features' else None
        requested=[False]
        def stop(*_):requested[0]=True
        signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
        deadline=int(os.environ['SLURM_JOB_END_TIME'])-config['allocation']['reserve_seconds']
        should_stop=lambda:requested[0] or time.time()>=deadline or (config['role']=='qualify_features'
            and len(store.existing)-before>=config['engineering_pause_after'])
        c.require(not should_stop(),'No allocation time after accepted-boundary reserve')
        try:
            features.compute(config,pool,rows,ids,store,should_stop,config['role']=='qualify_features',attempt)
            complete=len(store.existing)==len(roster);count=len(store.existing)
            if config['role']=='qualify_features':
                c.require(prefix_digest()==preserved_prefix,'Missing-only continuation changed earlier feature records')
                c.seal(attempt/'MISSING_ONLY_PREFIX.json',dict(previous_records=len(prefix),
                    prefix_semantic_sha256_before=preserved_prefix,prefix_semantic_sha256_after=preserved_prefix,
                    actual_restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0')),new_records=count-before,
                    old_feature_records_preserved=True,independent_qualification_required=True))
        except BaseException as error:
            c.seal(attempt/'FAILURE.json',dict(error=repr(error),measurement_identity=identity,
                completed=len(store.existing),elapsed_seconds=time.monotonic()-started,failed_cost_counts=True))
            raise
        finally:store.close()
        receipt=dict(status=('COMPLETE_ENGINEERING_FEATURE_SHARD' if config['role']=='qualify_features' else 'COMPLETE_PUBLISHED_FEATURE_SHARD')
            if complete else 'DURABLE_MISSING_ONLY_CONTINUATION',method='datatailor',adapter='explicit-eager-native-bf16-v1',
            measurement_identity=identity,pool=config['pool'],logical_pool=features.logical_pool(pool),feature_assets=config['feature_assets'],
            environment=config['environment'],measurement_contract=features.CONTRACT,sources=config['sources'],native_sources=config['native_sources'],
            shard_index=config['shard_index'],shard_count=config['shard_count'],roster_sha256=c.key(roster),
            counts=dict(before=before,after=count,required=len(roster)),database=c.desc(output/'features.sqlite3'),
            engineering_only=config['role']=='qualify_features',actual_GPU_hours=(time.monotonic()-started)/3600,
            qualification_promoted=False,setup_and_failed_costs_count=True)
        c.seal(attempt/'RESULT.json',receipt)
        if complete:c.seal(output/'COMPLETE.json',receipt);return 0
        c.seal(attempt/'REQUEUE_INTENT.json',dict(job_id=binding['job_id'],database=receipt['database'],UTC=c.now()))
        response=subprocess.run(['scontrol','requeue',binding['job_id']],capture_output=True,text=True,timeout=25)
        c.seal(attempt/'REQUEUE_RESPONSE.json',dict(rc=response.returncode,stdout=response.stdout,stderr=response.stderr))
        c.require(response.returncode==0,'Same-job requeue unacknowledged; no successor submission')
    return 0


if __name__=='__main__':raise SystemExit(main())
