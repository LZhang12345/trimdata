"""Read paid DT16 once; measure faithful native geometry on bounded real prefixes.

No model/checkpoint load, new feature acquisition, full Ward matrix, fabricated
selection or automatic successor. Original/composite custody reader is unchanged.
"""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import resource
import socket
import struct
import subprocess
import sys
import time

from experiments.root_native_campaign_v7_20261024 import common as c
from experiments.root_native_campaign_v7_20261024 import selection as native

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
AUTHOR='/root/telemetry_finish'
SCOPE='DT16_UNION_AND_BOUNDED_NATIVE_WARD_PROFILE_V1'
N=660763
BUDGETS={'0.05':33038,'0.10':66076,'0.15':99114,'0.20':132152}


def require_ref(ref):
    c.require(set(ref)=={'path','bytes','sha256'} and Path(ref['path']).is_absolute()
        and '..' not in Path(ref['path']).parts and type(ref['bytes']) is int and ref['bytes']>0
        and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,'Exact immutable input descriptor required')


def fields(raw):
    rows=[r for r in raw.splitlines() if r.strip()]
    c.require(len(rows)==1,'Exactly one scheduler row required')
    tokens=list(re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=',rows[0]));result={}
    c.require(tokens and tokens[0].start()==0,'Malformed scheduler metadata')
    for i,t in enumerate(tokens):
        c.require(t[1] not in result,'Duplicate scheduler key')
        result[t[1]]=rows[0][t.end():tokens[i+1].start() if i+1<len(tokens) else len(rows[0])].strip()
    return result


def source_map(original):
    result=dict(original['sources'])
    for p in HERE.iterdir():
        if p.suffix in ('.py','.sbatch') and not p.name.startswith('test_'):
            result[str(p.relative_to(ROOT))]=c.sha(p)
    for path,sha in result.items():
        p=ROOT/path
        c.require(not Path(path).is_absolute() and '..' not in Path(path).parts
            and p.is_file() and not p.is_symlink() and c.sha(p)==sha,'Frozen source mismatch: '+path)
    return dict(sorted(result.items()))


def validate_config(cfg):
    c.require(cfg['status']=='ROOT_CONFIGURED_DT16_UNION_QUALIFICATION_V1' and cfg['owner']=='Root'
        and cfg['scope']==SCOPE and cfg['source_root']==str(ROOT),'Exact separate engineering scope/root required')
    c.require(re.fullmatch('[0-9a-f]{32}',cfg['nonce']) is not None,'Exact nonce required')
    for key in ('pool','eligibility','original_configuration','source_reference_configuration','actual_parity','actual_parity_review'):
        require_ref(cfg[key])
    entries=cfg['feature_shards']
    c.require(len(entries)==16 and {e['shard_index'] for e in entries}==set(range(16))
        and all(type(e['shard_index']) is int for e in entries),'Exactly16distinct existing global shards required')
    for e in entries:
        c.require(e['kind'] in ('original','composite_v3','oracle_overlay_v1'),'Unknown actual custody kind')
        require_ref(e['database'] if e['kind']=='original' else e['manifest'])
        if e['kind']=='original':require_ref(e['completion'])
    c.require(cfg['probe_sizes']==[512,1024,2048] and cfg['relationship_probe_size']==512
        and cfg['budgets']==BUDGETS,'Fixed bounded native kernel probes and four-budget targets required')
    c.require(cfg['allocation']==dict(gpus=1,cpus=8,memory_mb=65536,minutes=90)
        and cfg['required_output_bytes']==(128<<20),'Finite explicit1L40S/CPU8/64G90min request required')
    out=Path(cfg['output'])
    c.require(out.is_absolute() and '..' not in out.parts
        and c.within(out,'/data/hf_cache/project_features/DS-MultiModal/dt_union_qualification_20261025')
        and not any(p.is_symlink() for p in (out,*out.parents)),'Separate project-derived output required')
    c.require(cfg['allow_new_forwards'] is False and cfg['full_ward_authorized'] is False
        and cfg['selection_or_student_authorized'] is False,'This job cannot acquire or promote full operators/students')
    return cfg


def function_identity(cfg):
    return c.key({k:cfg[k] for k in ('scope','pool','eligibility','original_configuration',
        'feature_shards','actual_parity','actual_parity_review','probe_sizes','relationship_probe_size','budgets')})


def documents(binding_ref,load=c.checked):
    b=load(binding_ref);cfg=validate_config(load(b['configuration']))
    c.require(b['status']=='ROOT_BOUND_DT16_UNION_QUALIFICATION_V1' and b['owner']=='Root'
        and b['nonce']==cfg['nonce'],'Exact Root binding required')
    original=load(cfg['original_configuration']);source_reference=load(cfg['source_reference_configuration']);packet=load(b['source_packet'])
    c.require(packet['sources']==source_map(source_reference),'Exact original327+additive source packet required')
    review=load(b['independent_source_config_review'])
    c.require(review['status']=='PASS_INDEPENDENT_DT16_UNION_QUALIFICATION_SOURCE_CONFIG_V1'
        and review['independent'] is True and review['reviewer'] not in ('',None,AUTHOR,'Root')
        and review['configuration']==b['configuration'] and review['source_packet']==b['source_packet'],
        'Independent exact actual source/config review required')
    claim=load(b['ownership'])
    c.require(claim['status']=='ROOT_EXCLUSIVE_DT16_UNION_QUALIFICATION_V1' and claim['owner']=='Root'
        and claim['scope']==SCOPE and claim['job_id']==b['job_id'] and claim['nonce']==b['nonce']
        and claim['configuration']==b['configuration'] and claim['function_identity']==function_identity(cfg)
        and claim['no_competing_union_or_geometry_provider'] is True,'Root exact nonduplicate engineering ownership required')
    grant=load(b['storage_reservation'])
    c.require(grant['status']=='ROOT_MEASURED_DT16_QUALIFICATION_OUTPUT_RESERVED_V1'
        and grant['output']==cfg['output'] and grant['reserved_bytes']==cfg['required_output_bytes']
        and grant['configuration']==b['configuration'],'Actual small collective storage reservation required')
    return b,cfg,original


def actor(binding_ref):
    b,cfg,original=documents(binding_ref);job=os.environ.get('SLURM_JOB_ID','')
    c.require(job.isdecimal() and job==b['job_id'] and os.getuid()==2708899
        and os.environ.get('SLURM_RESTART_COUNT','0')=='0' and not os.environ.get('SLURM_ARRAY_JOB_ID'),
        'Exact fresh scalar physical job required')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30);f=fields(raw)
    wanted=dict(JobId=job,JobState='RUNNING',UserId='luyangz(2708899)',Account='cx',Partition='general',QOS='normal',
        JobName='dsmm-dt-union-qual-v1',NumNodes='1',NumTasks='1',NumCPUs='8',Restarts='0',TimeLimit='01:30:00',
        TresPerNode='gres/gpu:L40S:1',Comment='dsmm-dt-union-qual:'+b['nonce'],Command=b['command']['path'],
        WorkDir=str(ROOT),NodeList=socket.gethostname().split('.')[0])
    c.require(all(f.get(k)==v for k,v in wanted.items()) and f.get('ArrayJobId') in (None,'(null)'),
        'Actual General scalar route/resources/nonce/command differ')
    tres=dict(v.split('=',1) for v in f['AllocTRES'].split(',') if '=' in v)
    c.require(tres.get('cpu')=='8' and tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1',
        'One actualGPU/CPU8/64G required')
    c.require(b['command']==c.desc(HERE/'run.sbatch'),'Exact wrapper source required')
    spool=subprocess.check_output(['scontrol','write','batch_script',job,'-'],timeout=30)
    c.require(spool==(HERE/'run.sbatch').read_bytes(),'Actual wrapper spool differs')
    return b,cfg,original,dict(job_id=job,node=f['NodeList'],raw_scheduler=raw,actual_GPU_allocation_count=1,
        actual_CPU_allocation_count=8,actual_GPU_computation_claimed=False)


def memory_counts(n):
    """Necessary lower bounds, never an upper bound or a full-Ward RAM PASS."""
    c.require(type(n) is int and n>=0,'Integer block count required')
    return dict(records=n,hidden_float32_bytes=n*4096*4,weight_float64_bytes=n*4096*8,
        dense_float64_bytes=n*n*8,condensed_float64_bytes=n*(n-1)//2*8,
        dense_plus_condensed_lower_bound_bytes=(n*n+n*(n-1)//2)*8,
        possible_nn_chain_copy_additional_bytes=n*(n-1)//2*8,
        original_extra_unused_condensed_virtual_bytes=n*(n-1)//2*8,
        is_peak_upper_bound=False,full_operation_memory_qualified=False)


def eligible_blocks(indices):
    from experiments.root_native_baselines_20261023 import operators
    blocks=operators.source_blocks(indices)
    c.require(len(blocks)==12 and sorted(p for block in blocks for p in block)==list(range(len(indices))),
        'Native twelve author source blocks must partition all eligible positions')
    return blocks


def union_digest(values,ids):
    """Digest exact paid values in immutable pool order; do not export vector copies."""
    import numpy as np
    c.require(set(values)==set(ids) and len(ids)==len(set(ids)),'Exact full semantic union required')
    digest=hashlib.sha256();lengths=[];info=[];tau=[]
    for ident in ids:
        v=values[ident];vector=np.asarray(v['vector'])
        c.require(vector.dtype==np.dtype('<f4') and vector.shape==(4096,) and np.isfinite(vector).all()
            and np.isfinite(v['info']) and np.isfinite(v['tau']) and v['tau']>0
            and type(v['length']) is int and v['length']>0 and v['nan_count']==0,'Invalid paid vector/scalar/length')
        key=ident.encode();digest.update(struct.pack('<I',len(key)));digest.update(key)
        digest.update(vector.tobytes(order='C'));digest.update(struct.pack('<ddq',v['info'],v['tau'],v['length']))
        lengths.append(v['length']);info.append(v['info']);tau.append(v['tau'])
    return dict(count=len(ids),record_ids_sha256=c.key(ids),ordered_paid_value_sha256=digest.hexdigest(),
        vector_dimension=4096,vector_dtype='<f4',all_nan_counts_zero=True,
        length_min=min(lengths),length_max=max(lengths),info_min=float(min(info)),info_max=float(max(info)),
        tau_min=float(min(tau)),tau_max=float(max(tau)),new_features_acquired=0)


def measure_probe(hidden,info,indices):
    """Unchanged Ward+relationship kernels on a real prefix, never full-pool labels."""
    import torch
    from experiments.root_native_baselines_20261023 import operators
    before=resource.getrusage(resource.RUSAGE_SELF);start=time.monotonic()
    labels=operators.datatailor_ward_block(hidden)
    elapsed=time.monotonic()-start;after=resource.getrusage(resource.RUSAGE_SELF)
    result=dict(n=len(hidden),source_indices_sha256=c.key(indices),hidden_sha256=hashlib.sha256(hidden.numpy().tobytes()).hexdigest(),
        labels_sha256=hashlib.sha256(labels.numpy().tobytes()).hexdigest(),clusters=int(labels.max()),
        elapsed_seconds=elapsed,process_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        process_max_RSS_KiB=after.ru_maxrss,memory_lower_bounds=memory_counts(len(hidden)),
        engineering_prefix_only=True,full_source_labels=False)
    if len(hidden)==512:
        start=time.monotonic()
        try:
            unique,rep=operators.datatailor_relationships(hidden,info,labels,indices)
            result['relationship']=dict(status='BOUNDED_NATIVE_RELATIONSHIP_COMPLETED',elapsed_seconds=time.monotonic()-start,
                uniqueness_sha256=hashlib.sha256(unique.numpy().tobytes()).hexdigest(),
                representativeness_sha256=hashlib.sha256(rep.numpy().tobytes()).hexdigest(),finite=bool(torch.isfinite(unique).all() and torch.isfinite(rep).all()))
        except ValueError as error:
            result['relationship']=dict(status='BOUNDED_NATIVE_RELATIONSHIP_REJECTED',error_repr=repr(error),
                elapsed_seconds=time.monotonic()-start,no_epsilon_or_fake_zero=True)
    return result


def runtime(original):
    import torch
    c.require(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA L40S',
        'A genuinely allocated visible L40S is required; no model forwards are requested')
    expected=c.checked(original['environment'])
    c.require(Path(sys.executable).absolute()==Path(expected['python_executable']).absolute(),
        'Exact existing feature interpreter required')
    result={}
    for name in ('numpy','torch','scipy'):
        module=importlib.import_module(name)
        result[name]=dict(version=module.__version__,source=c.desc(module.__file__))
    from scipy.cluster import _hierarchy
    c.require(all(result[name]['version']==expected['versions'][name] for name in ('numpy','torch')),
        'Observed NumPy/Torch versions differ from original native features')
    result['expected_feature_environment']=original['environment']
    result['scipy_hierarchy_binary']=c.desc(_hierarchy.__file__)
    result['GPU']=torch.cuda.get_device_name(0);torch.set_num_threads(8)
    return result


def validate_original_complete(receipt,entry,pool_ref,identity,roster,original):
    fields={'actual','counts','database','elapsed_seconds','environment','feature_assets',
        'measurement_identity','method','native_sources','no_student_or_scientific_PASS','pool',
        'roster_sha256','shard_count','shard_index','sources','status'}
    c.require(set(receipt)==fields,'Exact legacy production completion schema required')
    c.require(receipt['status']=='COMPLETE_PUBLISHED_FEATURE_SHARD' and receipt['method']=='datatailor'
        and receipt['no_student_or_scientific_PASS'] is True and receipt['pool']==pool_ref
        and receipt['measurement_identity']==identity and receipt['database']==entry['database']
        and entry['identity']==identity and receipt['shard_count']==16
        and receipt['shard_index']==entry['shard_index'] and receipt['roster_sha256']==c.key(roster)
        and receipt['counts']['after']==receipt['counts']['required']==len(roster)
        and receipt['sources']==original['measurement_sources']
        and receipt['native_sources']==original['native_sources']
        and receipt['environment']==original['environment']
        and receipt['feature_assets']==original['feature_assets'],
        'Original legacy header must authenticate the complete real shard and frozen production function')


def execute(cfg,original,attempt):
    from experiments.root_dt_oracle_overlay_recovery_v1_20261025 import consumer
    import numpy as np
    import torch
    started=time.monotonic();pool,records,ids=native.validate_pool(cfg['pool'],eligibility_desc=cfg['eligibility'])
    c.require(len(ids)==len(records)==pool['count']==N and ids==pool['record_ids'],'Complete actual660763 pool required')
    indices=pool['source_indices'];blocks=eligible_blocks(indices)
    c.seal(attempt/'POOL_BLOCK_COUNTS.json',dict(pool=cfg['pool'],count=N,
        blocks=[dict(author_source_block=i,**memory_counts(len(block))) for i,block in enumerate(blocks)],
        full_scope_feature_float32_payload_bytes=N*4096*4,full_ward_attempted=False))
    del records;gc.collect()
    identity=consumer.native.identity(original,pool)
    declared={entry['identity'] for entry in cfg['feature_shards'] if entry['kind']=='original'}
    c.require(declared=={identity},'Actual production measurement projection differs: computed='+identity+' declared='+repr(sorted(declared)))
    for entry in cfg['feature_shards']:
        if entry['kind']=='original':
            receipt=c.checked(entry['completion'])
            roster=consumer.native.roster(dict(role='features',shard_count=16,shard_index=entry['shard_index']),[],ids)
            validate_original_complete(receipt,entry,cfg['pool'],identity,roster,original)
    values=consumer.load_full_pool(cfg['feature_shards'],ids,cfg['pool'],identity,cfg['actual_parity'],cfg['actual_parity_review'])
    union=union_digest(values,ids)
    union.update(original_measurement=identity,feature_shards=cfg['feature_shards'],pool=cfg['pool'],
        reader_sources_preserved=True,all16_original_custody_and_locks_validated=True,elapsed_seconds=time.monotonic()-started)
    union_ref=c.seal(attempt/'FULL_PAID_UNION.json',dict(status='ACTUAL_FULL_DT16_UNION_VALIDATED_ENGINEERING_ONLY_V1',**union))
    largest=max(range(12),key=lambda i:(len(blocks[i]),-i));positions=blocks[largest]
    c.require(len(positions)>=max(cfg['probe_sizes']),'Largest real source cannot support declared prefix probes')
    needed=positions[:max(cfg['probe_sizes'])]
    hidden=torch.from_numpy(np.stack([values[ids[p]]['vector'] for p in needed]))
    info=torch.tensor([values[ids[p]]['info'] for p in needed],dtype=torch.float32)
    del values;gc.collect()  # Full paid union not copied/exported; probes fit independently of full geometry.
    probes=[]
    for size in cfg['probe_sizes']:
        result=measure_probe(hidden[:size],info[:size],[indices[p] for p in needed[:size]])
        result['source_block']=largest;result['record_ids_sha256']=c.key([ids[p] for p in needed[:size]])
        c.seal(attempt/('PROBE_%04d.json'%size),result);probes.append(result)
    return dict(full_paid_union=union_ref,largest_actual_source_block=largest,
        actual_source_blocks=[dict(author_source_block=i,**memory_counts(len(block))) for i,block in enumerate(blocks)],
        probes=probes,four_budget_targets=BUDGETS,full_ward_memory_or_walltime_qualified=False,
        full_relationships_or_selection_qualified=False,student_training_authorized=False,
        interpretation='Use measured prefix costs plus exact source sizes to design full native qualification. Quadratic/cubic extrapolations are estimates only; largest full-block peak and time remain unmeasured.')


def run(binding_ref):
    b,cfg,original,actual=actor(binding_ref)
    out=Path(cfg['output'])/(b['job_id']+'-'+b['nonce'])
    c.require(not out.exists() and not any(p.is_symlink() for p in (out,*out.parents)),'Fresh literal output required')
    out.mkdir(parents=True,exist_ok=False)
    stat=os.statvfs(out);c.require(stat.f_bavail*stat.f_frsize>=cfg['required_output_bytes'],'Actual output free space insufficient')
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF)
    c.seal(out/'START.json',dict(binding=binding_ref,actual=actual,UTC=c.now(),source_packet=b['source_packet']))
    try:
        observed=runtime(original);c.seal(out/'RUNTIME.json',observed)
        result=execute(cfg,original,out);after=resource.getrusage(resource.RUSAGE_SELF)
        return c.seal(out/'RESULT.json',dict(status='DT16_UNION_AND_BOUNDED_NATIVE_PROFILE_COMPLETED_NOT_FULL_OPERATOR_PASS_V1',
            binding=binding_ref,configuration=b['configuration'],source_packet=b['source_packet'],actual=actual,runtime=observed,
            result=result,elapsed_seconds=time.monotonic()-started,
            process_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,process_max_RSS_KiB=after.ru_maxrss,
            scheduler_allocation_cost_requires_sacct_D=True,automatic_resubmit=False,new_neural_forwards=0,
            new_selection_produced=False,checkpoint_or_model_loaded=False,all_in_cost_includes_paid_providers=True))
    except BaseException as error:
        c.seal(out/'FAILURE.json',dict(status='DT16_UNION_OR_BOUNDED_PROFILE_FAILED',binding=binding_ref,
            actual=actual,error_class=type(error).__name__,error_repr=repr(error),elapsed_seconds=time.monotonic()-started,
            automatic_resubmit=False,scheduler_allocation_cost_requires_sacct_D=True));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binding',required=True)
    print(json.dumps(run(c.desc(p.parse_args().binding))))
