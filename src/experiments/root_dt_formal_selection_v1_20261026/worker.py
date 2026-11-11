"""Join paid full source geometry, then call the unchanged four-budget selector.

Outputs are candidates awaiting independent actual operator/output review, never
invented Native7 method PASS or student readiness. No Ward/model is rerun.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time

from experiments.root_dt_full_geometry_v2_20261026 import worker as geometry
from experiments.root_dt_union_qualification_v1_20261025.worker import fields
from experiments.root_native_baselines_20261023 import operators
from experiments.root_native_campaign_v7_20261024 import selection as native

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
HOME=Path('/home/luyangz/data_selection_multimodal')
CONTROL=HOME/'collaboration/root-dt-formal-selection-v1-20261026'
LOCK=HOME/'collaboration/babel-two-track-20261019/ADMISSION.lock'
OUTPUT=Path('/data/hf_cache/project_features/DS-MultiModal/dt_formal_selection_20261026/v1')
BUDGETS=['0.05','0.10','0.15','0.20']
SCOPE='FOUR_NATIVE_DATATAILOR_BUDGET_CANDIDATES_FROM_PAID_FULL12_GEOMETRY'
REVIEW='PASS_INDEPENDENT_DT_FORMAL_SELECTION_SOURCE_CONFIG_V1'
require=geometry.require
checked=geometry.checked
ref=geometry.ref
seal=geometry.seal

def literal(path):
    path=Path(path)
    require(path.is_absolute() and '..' not in path.parts
        and not any(p.is_symlink() for p in (path,*path.parents)), 'Literal no-symlink path required')
    return path

def function_identity(cfg):
    return geometry.hashlib.sha256(json.dumps({k:cfg[k] for k in
        ('scope','geometry_configuration','producer_binding','geometry_binding','budgets','sources')},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()

def documents(binding):
    cfg=checked(binding['configuration']);peer=checked(binding['independent_review'])
    require(cfg['status']=='ROOT_CONFIGURED_FORMAL_DT_SELECTION_V1' and cfg['owner']=='Root'
        and cfg['scope']==SCOPE and cfg['source_root']==str(ROOT) and cfg['output']==str(OUTPUT)
        and cfg['budgets']==BUDGETS and cfg['resources']==dict(cpus=8,memory_mb=65536,gpus=1,minutes=60)
        and cfg['reserved_bytes']==4<<30 and cfg['reserved_inodes']==256
        and cfg['new_forwards']==0 and cfg['ward_recomputation_authorized'] is False
        and cfg['student_READY'] is False, 'Exact finite full-pool selection scope required')
    require(peer['status']==REVIEW and peer['independent'] is True
        and peer['reviewer'] not in ('','Root','/root/repair_independent_review')
        and peer['configuration']==binding['configuration'] and peer['sources']==cfg['sources'],
        'Non-author independent exact source/config review required')
    for name,digest in cfg['sources'].items():
        p=ROOT/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
            and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==digest,
            'Frozen source changed: '+name)
    gcfg=checked(cfg['geometry_configuration']);gb=checked(cfg['geometry_binding']);pb=checked(cfg['producer_binding'])
    require(gb['job_id']=='10937866' and gb['mode']=='geometry' and pb['job_id']=='10937865'
        and pb['mode']=='prepare' and gb['producer_binding']==cfg['producer_binding']
        and gb['configuration']==pb['configuration']==cfg['geometry_configuration']
        and gcfg['dt_config']['pool']==cfg['pool'] and gcfg['dt_config']['eligibility']==cfg['eligibility'],
        'Exact actual full-geometry provider graph required')
    prior_review=checked(gb['independent_review'])
    require(prior_review['status']=='PASS_INDEPENDENT_DT_FULL_GEOMETRY_SOURCE_CONFIG_V1'
        and prior_review['independent'] is True and prior_review['configuration']==cfg['geometry_configuration']
        and prior_review['sources']==gcfg['sources'] and pb['independent_review']==gb['independent_review'],
        'Actual full geometry exact source/config review required')
    for name,digest in gcfg['sources'].items():
        require(cfg['sources'].get(name)==digest,'Original395 source bytes cannot be replaced')
    require(binding['status']=='ROOT_BOUND_FORMAL_DT_SELECTION_V1' and binding['owner']=='Root'
        and binding['nonce']==cfg['nonce'], 'Exact fresh physical binding required')
    ownership=checked(binding['ownership'])
    require(ownership['owner']=='Root' and ownership['configuration']==binding['configuration']
        and ownership['job_id']==binding['job_id'] and ownership['scope']==SCOPE
        and ownership['no_live_duplicate_selector'] is True, 'Exclusive physical intent required')
    return cfg,gcfg,gb,pb

def actor(binding,cfg):
    j=os.environ.get('SLURM_JOB_ID','')
    require(j.isdecimal() and j==binding['job_id'] and os.getuid()==2708899
        and not os.environ.get('SLURM_ARRAY_JOB_ID') and os.environ.get('SLURM_RESTART_COUNT','0')=='0',
        'Fresh actual scalar collector required')
    raw=subprocess.check_output(['scontrol','show','job','-o',j],text=True,timeout=30);f=fields(raw)
    expected=dict(JobId=j,JobName='dsmm-root-dt-select-v1',JobState='RUNNING',UserId='luyangz(2708899)',
        Account='cx',Partition='general',QOS='normal',NumCPUs='8',NumTasks='1',NumNodes='1',Restarts='0',
        TimeLimit='01:00:00',TresPerNode='gres/gpu:L40S:1',WorkDir=str(ROOT),Command=str(HERE/'run.sbatch'),
        Comment='dsmm-root-dt-select:'+cfg['nonce'],NodeList=socket.gethostname().split('.')[0])
    require(all(f.get(k)==v for k,v in expected.items()),'Actual General CPU8/64G/L40S route differs')
    tres=dict(x.split('=',1) for x in f['AllocTRES'].split(',') if '=' in x)
    require(tres.get('cpu')=='8' and tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1',
        'Actual allocation differs')
    require(subprocess.check_output(['scontrol','write','batch_script',j,'-'],timeout=30)==(HERE/'run.sbatch').read_bytes(),
        'Actual spool differs')
    return dict(job_id=j,node=f['NodeList'],scheduler=raw,new_forwards=0,ward_recomputations=0)

def accounting_ok(raw,physical,selector):
    rows=[r.split('|') for r in raw.splitlines() if r.strip()]
    exact=[r for r in rows if len(r)>=4 and r[0]==physical and r[1]==selector]
    require(len(exact)==1 and exact[0][2:4]==['COMPLETED','0:0'],
        'Exact full geometry task has not completed successfully')
    return exact[0]

def load_geometry(cfg,gcfg,gb):
    manifest,mref=geometry.actual_manifest(gb,gcfg)
    require(manifest['paid_union']==gcfg['paid_union'] and manifest['new_features']==0
        and manifest['measurement_projection']=='EXACT_QUALIFIED_PLAN_V4_measurement_sources',
        'Exact paid prepare manifest required')
    results=[];accounting=[]
    for index in range(12):
        out=literal(Path(gcfg['output'])/('geometry-%02d'%index))
        rref=ref(out/'RESULT.json');result=checked(rref);start=checked(ref(out/'START.json'))
        require(result['status']=='ACTUAL_FULL_SOURCE_RELATIONSHIPS_COMPLETE' and result['block']==index
            and result['producer_manifest']==mref and result['configuration']==cfg['geometry_configuration']
            and result['actual']==start['actual'] and start['binding']==cfg['geometry_binding']
            and start['configuration']==cfg['geometry_configuration']
            and result['actual']['array_job']==gb['job_id'] and result['actual']['task']==str(index)
            and result['new_neural_forwards']==0, 'Exact completed full source result/custody differs')
        selector=gb['job_id']+'_'+str(index);physical=result['actual']['job_id']
        raw=subprocess.check_output(['sacct','-n','-P','-X','-j',selector,
            '--format=JobIDRaw,JobID,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES'],text=True,timeout=30)
        record=accounting_ok(raw,physical,selector)
        require(len(record)>=7 and record[4].isdecimal() and record[5].isdecimal(), 'Actual geometry allocation costs required')
        accounting.append(dict(selector=selector,physical_job_id=physical,raw=raw,elapsed_allocated_seconds=int(record[4]),allocated_CPUs=int(record[5]),allocated_TRES=record[6]))
        results.append(dict(result=result,reference=rref,start=ref(out/'START.json')))
    return manifest,mref,results,accounting

def join_relationships(pool,ids,manifest,rows,info):
    import numpy as np
    import torch
    n=len(ids);blocks=operators.source_blocks(pool['source_indices'])
    require(len(blocks)==len(manifest['blocks'])==len(rows)==12
        and sorted(p for b in blocks for p in b)==list(range(n)), 'Twelve complete source blocks required')
    unique=torch.empty(n,dtype=torch.float32);represent=torch.empty(n,dtype=torch.float32);seen=set();proof=[]
    for index,(positions,block,item) in enumerate(zip(blocks,manifest['blocks'],rows)):
        result=item['result']
        require(block['block']==result['block']==index and block['count']==len(positions)
            and block['source_indices']==[pool['source_indices'][p] for p in positions], 'Original source block scope differs')
        arrays={}
        for name,description in [('positions',block['positions']),('info',block['info']),
                ('labels',result['labels']),('uniqueness',result['uniqueness']),('representativeness',result['representativeness'])]:
            require(ref(literal(description['path']))==description,'Geometry array bytes changed')
            arrays[name]=np.load(description['path'],allow_pickle=False)
        p=arrays['positions'];require(p.dtype==np.dtype('<i8') and p.tolist()==positions
            and not seen.intersection(positions), 'Overlapping/missing/foreign global positions')
        seen.update(positions)
        require(arrays['info'].dtype==np.dtype('<f4') and arrays['info'].shape==(len(positions),)
            and np.array_equal(arrays['info'],info[positions].numpy()), 'Prepared informativeness does not match exact paid16 values')
        labels=arrays['labels'];require(labels.shape==(len(positions),) and np.issubdtype(labels.dtype,np.integer)
            and labels.min()==1 and set(labels.tolist())==set(range(1,int(labels.max())+1)), 'Native Ward labels invalid')
        for key,target in [('uniqueness',unique),('representativeness',represent)]:
            value=arrays[key];require(value.dtype==np.dtype('<f4') and value.shape==(len(positions),)
                and np.isfinite(value).all(),'Exact finite FP32 relationships required')
            target[positions]=torch.from_numpy(value)
        proof.append(dict(block=index,count=len(positions),result=item['reference'],
            positions=block['positions'],labels=result['labels'],uniqueness=result['uniqueness'],representativeness=result['representativeness']))
    require(seen==set(range(n)), 'Full eligible global position coverage required')
    return unique,represent,proof

def paid_values(gcfg):
    import numpy as np
    import torch
    from experiments.root_dt_oracle_overlay_recovery_v1_20261025 import consumer
    from experiments.root_dt_union_qualification_v1_20261025.worker import (
        runtime, union_digest, validate_original_complete)
    old=gcfg['dt_config'];original=checked(old['original_configuration']);observed=runtime(original)
    measurement=geometry.measurement_projection(original)
    pool,records,ids=native.validate_pool(old['pool'],eligibility_desc=old['eligibility'])
    require(len(ids)==len(records)==pool['count']==660763 and ids==pool['record_ids'],'Exact full660763pool required')
    identity=consumer.native.identity(measurement,pool)
    for entry in old['feature_shards']:
        if entry['kind']=='original':
            roster=consumer.native.roster(dict(role='features',shard_count=16,shard_index=entry['shard_index']),[],ids)
            validate_original_complete(checked(entry['completion']),entry,old['pool'],identity,roster,measurement)
    values=consumer.load_full_pool(old['feature_shards'],ids,old['pool'],identity,old['actual_parity'],old['actual_parity_review'])
    digest=union_digest(values,ids);prior=checked(gcfg['paid_union'])
    require(digest['ordered_paid_value_sha256']==prior['ordered_paid_value_sha256'], 'Paid full union checksum changed')
    info=torch.from_numpy(np.asarray([values[i]['info'] for i in ids],dtype=np.float32))
    tau=torch.from_numpy(np.asarray([values[i]['tau'] for i in ids],dtype=np.float32))
    lengths=[values[i]['length'] for i in ids];del values;gc.collect()
    return pool,records,ids,info,tau,lengths,digest,observed

def select_budgets(records,indices,info,tau,unique,represent):
    results={}
    for fraction in BUDGETS:
        result=operators.datatailor_select(records,indices,info,tau,unique,represent,float(fraction))
        require(result['top_up'] is False and result['whole_conversations'] is True
            and result['positions']==sorted(set(result['positions']))
            and result['source_indices']==[indices[p] for p in result['positions']]
            and result['count']==len(result['positions'])==sum(result['source_quotas']),
            'Native no-top-up selection result differs')
        results[fraction]=result
    return results

def reserve(binding,cfg,actual):
    grant=checked(binding['storage_reservation'])
    require(grant['owner']=='Root' and grant['configuration']==binding['configuration']
        and grant['output']==cfg['output'] and grant['reserved_bytes']==cfg['reserved_bytes']
        and grant['reserved_inodes']==cfg['reserved_inodes'] and grant['other_outstanding_reserved_bytes']>=1<<40
        and grant['other_outstanding_reserved_inodes']>=1100000,'Measured separate4GiB grant and conservative other commitments required')
    with literal(LOCK).open('r+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=literal(cfg['output']);require(not out.exists(),'Existing selector output requires explicit reconciliation')
        out.parent.mkdir(parents=True,exist_ok=True);v=os.statvfs(out.parent)
        require(v.f_bavail*v.f_frsize>=cfg['reserved_bytes']+grant['other_outstanding_reserved_bytes']+(2<<30)
            and v.f_favail>=cfg['reserved_inodes']+grant['other_outstanding_reserved_inodes']+1000,'Actual shared filesystem insufficient')
        path=CONTROL/'FUNCTION.json';require(not path.exists(),'Static selector function already owned; do not duplicate')
        claim=seal(path,dict(owner='Root',function_identity=function_identity(cfg),configuration=binding['configuration'],
            actual=actual,scope=SCOPE,output=cfg['output'],no_global_FUNCTION_roster_mutation=True))
        out.mkdir();return out,claim,dict(available_bytes=v.f_bavail*v.f_frsize,available_inodes=v.f_favail,
            actual=actual,prior_grant=binding['storage_reservation'])

def run(binding_path):
    started=time.monotonic();binding=checked(ref(literal(binding_path)));cfg,gcfg,gb,pb=documents(binding);actual=actor(binding,cfg)
    manifest,mref,rows,accounting=load_geometry(cfg,gcfg,gb)
    out,claim,space=reserve(binding,cfg,actual)
    seal(out/'START.json',dict(configuration=binding['configuration'],actual=actual,claim=claim,storage=space))
    try:
        pool,records,ids,info,tau,lengths,digest,observed=paid_values(gcfg)
        require(manifest['ordered_paid_value_sha256']==digest['ordered_paid_value_sha256'],'Geometry and original paid union differ')
        unique,represent,proof=join_relationships(pool,ids,manifest,rows,info)
        choices=select_budgets(records,pool['source_indices'],info,tau,unique,represent);outputs=[];written=0
        for fraction,result in choices.items():
            directory=out/('budget-'+fraction);directory.mkdir();positions=result['positions'];chosen=[ids[p] for p in positions]
            content=(json.dumps([records[p] for p in positions],ensure_ascii=False,separators=(',',':'))+'\n').encode()
            written+=len(content);require(written<=((7<<29)), 'Combined selected records exceed3.5GiB reserved envelope')
            with (directory/'records.json').open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
            records_ref=ref(directory/'records.json');del content
            operator_result=seal(directory/'OPERATOR_RESULT.json',dict(**result,paid_union=gcfg['paid_union'],
                ordered_paid_value_sha256=digest['ordered_paid_value_sha256'],full_geometry_manifest=mref,full_geometry=proof,
                measurement_projection='EXACT_QUALIFIED_PLAN_V4_measurement_sources',original_whole_conversations=True,
                native_original_conversation_turn_counts=True,paid_feature_lengths_preserved_as_provenance=True,no_top_up=True))
            candidate=dict(schema='native-record-selection-v1',status='PENDING_INDEPENDENT_OPERATOR_OUTPUT_REVIEW',
                pool=cfg['pool'],method='datatailor',fraction=fraction,selection_seed=1234,count=len(chosen),record_ids=chosen,
                records_json=records_ref,groups=None,features=None,assets=None,algorithm_sources=cfg['sources'],method_review=None,
                budget=dict(unit='whole_conversation_count',value=len(chosen),fraction=fraction,eligible_pool_count=len(ids)),
                requested_K=native.budget_size(len(ids),fraction),operator_result=operator_result,
                source_quotas=result['source_quotas'],provisional_not_student_READY=True)
            outputs.append(dict(fraction=fraction,actual_K=len(chosen),requested_K=candidate['requested_K'],
                candidate=seal(directory/'SELECTION_CANDIDATE.json',candidate),records=records_ref,operator_result=operator_result))
        return seal(out/'RESULT.json',dict(status='COMPLETE_FOUR_NATIVE_DT_SELECTION_CANDIDATES_PENDING_INDEPENDENT_OUTPUT_REVIEW',
            actual=actual,configuration=binding['configuration'],claim=claim,runtime=observed,paid_union=gcfg['paid_union'],
            paid_value_digest=digest,geometry_configuration=cfg['geometry_configuration'],geometry_manifest=mref,geometry_results=proof,
            actual_geometry_accounting=accounting,outputs=outputs,selected_record_bytes=written,source_block_count=12,
            full_pool_count=len(ids),paid_feature_length_min=min(lengths),paid_feature_length_max=max(lengths),
            elapsed_seconds=time.monotonic()-started,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            new_forwards=0,ward_recomputations=0,student_READY=False,all_prior_feature_geometry_and_current_costs_count=True))
    except BaseException as error:
        seal(out/'FAILURE.json',dict(actual=actual,error=repr(error),elapsed_seconds=time.monotonic()-started,
            partial_outputs_preserved=True,no_automatic_retry=True));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);a=p.parse_args();print(json.dumps(run(a.binding)))
