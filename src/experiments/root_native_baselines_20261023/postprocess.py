"""Owned CPU+GPU full-pool geometry/selection, with no automatic student release."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import resource
import subprocess
import sys
import time
from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_campaign_20261023.config import validate_profile
from experiments.root_native_campaign_20261023.selection import validate_pool
from . import pins,select
from .provider import actual,actual_environment


def validate(config):
    c.require(config['schema']=='published-native-postprocess-job-v1' and config['owner']=='Root'
        and config['method'] in ('coincide','datatailor') and config['role'] in ('qualify_operators','postprocess'),'Wrong postprocess schema/owner/method/role')
    pins.verify(config['sources']);c.verify_sources(config['native_sources'])
    profile=validate_profile(c.checked(config['site_profile']))
    c.require(Path(profile['code_root']).resolve()==c.ROOT.resolve(),'Wrong staged published source tree')
    pool,_,_=validate_pool(config['pool'])
    env=c.checked(config['environment'])
    c.require(env['status']=='PASS_ENVIRONMENT' and env['python_executable']==profile['python'],'Exact published operator environment required')
    identity=c.key(dict(method=config['method'],pool=config['pool'],feature_shards=config['feature_shards'],
        sources=config['sources'],fractions=config['fractions'],environment=config['environment'],
        site_profile=config['site_profile'],cpus=config['allocation']['cpus']))
    allocation=config['allocation']
    c.require(allocation['gpus']==1 and allocation['cpus']>=2 and allocation['memory_mb']>=192*1024,
        'Exact full-pool Ward/FAISS host geometry requires explicit GPU+192GiB admission, or a separately reviewed measured-memory repair')
    c.require(0<allocation['reserve_seconds']<allocation['minutes']*60<=profile['max_minutes']*60
        and allocation['memory_mb']<=profile['max_memory_mb'] and allocation['cpus']<=profile['max_cpus_per_request'],'Actual resource/time limits exceeded')
    c.require(config['fractions']==['0.05','0.10','0.20'] and config['selection_seed']==1234,'Finite published comparison budget/seed differs')
    c.require(c.within(config['output'],profile['output_root']),'Unapproved postprocess output')
    claim=c.checked(config['ownership'])
    c.require(claim['status']=='ACTIVE_EXCLUSIVE_PUBLISHED_POSTPROCESS' and claim['owner']=='Root'
        and claim['function_identity']==identity and claim['site_profile']==config['site_profile'],'Unowned/duplicate full-pool geometry')
    reservation=c.checked(config['storage_reservation'])
    c.require(reservation['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION'
        and reservation['output']==config['output'] and reservation['function_identity']==identity,'Full operator storage reservation absent')
    review=c.checked(config['review'])
    c.require(review['status']=='PASS_INDEPENDENT_EXACT_PUBLISHED_POSTPROCESS_CONFIG'
        and review['config_identity']==c.key({k:v for k,v in config.items() if k!='review'})
        and review['sources']==config['sources'] and review['native_sources']==config['native_sources'],'Exact source/config independent review absent')
    if config['role']=='postprocess':
        q=c.checked(config['qualification'])
        c.require(q['status']=='PASS_ACTUAL_PUBLISHED_OPERATOR_RUNTIME' and q['method']==config['method']
            and q['sources']==config['sources'] and q['site_profile']==config['site_profile']
            and q['environment']==config['environment'],'Actual exact full-operator GPU/CPU qualification required')
        for name in ('author_operator_parity','native_dtype_and_quota','largest_source_memory_profile'):
            evidence=c.checked(q['evidence'][name]);c.require(evidence['status']=='PASS_ACTUAL_'+name.upper(),'Missing actual operator evidence: '+name)
    else:
        c.require(config.get('engineering_only') is True and config.get('qualification') is None,
            'Explicit separately reviewed qualification role; never invent its own PASS')
    return profile,identity


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--binding',required=True);a=parser.parse_args()
    binding=c.read(a.binding)
    c.require(binding['schema']=='published-native-postprocess-binding-v1' and binding['owner']=='Root','Wrong postprocess binding')
    config=c.checked(binding['configuration']);profile,identity=validate(config)
    if Path(sys.executable).absolute()!=Path(profile['python']).absolute():
        os.execv(profile['python'],[profile['python'],'-B','-u','-m','experiments.root_native_baselines_20261023.postprocess','--binding',a.binding])
    raw=actual(binding,config,profile)
    os.environ['OMP_NUM_THREADS']=str(config['allocation']['cpus'])
    os.environ['MKL_NUM_THREADS']=str(config['allocation']['cpus'])
    runtime=actual_environment(config['environment'],postprocess=True)
    import torch
    import faiss
    torch.set_num_threads(config['allocation']['cpus']);faiss.omp_set_num_threads(config['allocation']['cpus'])
    c.require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Actual owned GPU required even when Ward is CPU-side')
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    lock=open(output/'EXECUTION.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=output/f"attempt-{binding['job_id']}-r{os.environ.get('SLURM_RESTART_COUNT','0')}"
    c.seal(attempt/'START.json',dict(binding=c.desc(a.binding),identity=identity,actual_slurm=raw,runtime=runtime,UTC=c.now()))
    final_name='ENGINEERING_OPERATOR_EVIDENCE.json' if config['role']=='qualify_operators' else 'POSTPROCESS_COMPLETE.json'
    if (output/final_name).exists():
        receipt=c.read(output/final_name);c.require(receipt['identity']==identity,'Completed postprocess function changed')
        for descriptor in receipt['candidate_selections']:c.checked(descriptor)
        return 0
    stop=[False]
    def request(*_):stop[0]=True
    signal.signal(signal.SIGUSR1,request);signal.signal(signal.SIGTERM,request)
    deadline=int(os.environ['SLURM_JOB_END_TIME'])-config['allocation']['reserve_seconds']
    should_stop=lambda:stop[0] or time.time()>=deadline
    c.require(not should_stop(),'No allocation time after setup/reserve')
    feature_desc=select.collect(config['pool'],config['method'],config['feature_shards'],output/'collection',config.get('transfer'))
    context=dict(environment=config['environment'],runtime=runtime,site=profile['site'],
        cpus=config['allocation']['cpus'],GPU_name=torch.cuda.get_device_name(0),
        GPU_capability=list(torch.cuda.get_device_capability(0)))
    result=select.materialize_values(feature_desc,config['pool'],output/'geometry',should_stop,context)
    if result.get('status')=='DURABLE_OPERATOR_CONTINUATION':
        c.seal(attempt/'CONTINUATION.json',dict(identity=identity,result=result,GPU_hours=(time.monotonic()-started)/3600))
        c.seal(attempt/'REQUEUE_INTENT.json',dict(job_id=binding['job_id'],identity=identity,UTC=c.now()))
        response=subprocess.run(['scontrol','requeue',binding['job_id']],capture_output=True,text=True,timeout=25)
        c.seal(attempt/'REQUEUE_RESPONSE.json',dict(rc=response.returncode,stdout=response.stdout,stderr=response.stderr))
        c.require(response.returncode==0,'Exact-job postprocess requeue unacknowledged; no replacement')
        return 0
    candidates=[]
    for fraction in config['fractions']:
        # Native MMD is not interruptible at arbitrary operations. Qualification
        # must measure the largest cluster before admitting its finite walltime.
        c.require(not should_stop(),'Deadline reached before starting another native budget operator; preserve existing outputs and diagnose')
        candidates.append(select.candidate(config['pool'],result,fraction,1234,output/('p'+fraction)))
    receipt=dict(status='COMPLETE_ENGINEERING_OPERATOR_EVIDENCE' if config['role']=='qualify_operators' else 'COMPLETE_PUBLISHED_CANDIDATES_PENDING_INDEPENDENT_OUTPUT_REVIEW',identity=identity,
        candidate_selections=candidates,GPU_hours=(time.monotonic()-started)/3600,
        setup_failed_and_acquisition_costs_also_count=True,student_release=False,qualification_promoted=False,
        platform=sys.platform,peak_RSS_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_RSS_units='KiB_on_Linux_bytes_on_Darwin',torch_max_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_allocator_does_not_measure_FAISS_GPU_memory=True)
    c.seal(output/final_name,receipt);c.seal(attempt/'RESULT.json',receipt)
    return 0


if __name__=='__main__':raise SystemExit(main())
