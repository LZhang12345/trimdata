"""Preserve executed V1 and its paid 5/10/15% choices; finish only missing output.

The published selector and shared Native7 budget registry remain byte-identical.
15% is explicitly declared here as a wrapper budget, using Decimal floor. Cached
feature stores are authenticated/read once for the missing 20% choice; there are
zero new neural forwards, Ward fits or relationship computations.
"""
import argparse
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time

from experiments.root_dt_formal_selection_v1_20261026 import worker as prior

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
CONTROL=prior.HOME/'collaboration/root-dt-formal-selection-v2-20261026'
OUTPUT=Path('/data/hf_cache/project_features/DS-MultiModal/dt_formal_selection_20261026/v2')
SCOPE='MISSING_ONLY_DT_15_CANDIDATE_AND_20_SELECTION_PRESERVING_PAID_5_10_15_V1'
REVIEW='PASS_INDEPENDENT_DT_MISSING_ONLY_FORMAL_SELECTION_SOURCE_CONFIG_V2'
BUDGETS=prior.BUDGETS
require=prior.require
checked=prior.checked
ref=prior.ref
seal=prior.seal
literal=prior.literal

def budget_size(n,fraction):
    """Wrapper-only declared whole-conversation count; no alteration of Native7."""
    require(type(n) is int and n>0,'Positive whole eligible count required')
    try: value=Decimal(str(fraction))
    except (InvalidOperation,ValueError): raise ValueError('Undeclared DT wrapper budget')
    require(value.is_finite() and value in tuple(map(Decimal,BUDGETS)),
        'Only declared 5/10/15/20 percent DT wrapper budgets allowed')
    return int((Decimal(n)*value).to_integral_value(rounding=ROUND_FLOOR))

def function_identity(cfg):
    return hashlib.sha256(json.dumps({k:cfg[k] for k in ('scope','prior_configuration',
        'prior_binding','prior_artifacts','geometry_configuration','producer_binding',
        'geometry_binding','budgets','sources')},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def documents(binding):
    cfg=checked(binding['configuration']);peer=checked(binding['independent_review'])
    require(cfg['status']=='ROOT_CONFIGURED_MISSING_ONLY_FORMAL_DT_SELECTION_V2'
        and cfg['owner']=='Root' and cfg['scope']==SCOPE and cfg['source_root']==str(ROOT)
        and cfg['output']==str(OUTPUT) and cfg['budgets']==BUDGETS and cfg['selection_seed']==1234
        and cfg['resources']==dict(cpus=8,memory_mb=65536,gpus=1,minutes=60)
        and cfg['reserved_bytes']==4<<30 and cfg['reserved_inodes']==256
        and cfg['new_forwards']==0 and cfg['ward_recomputation_authorized'] is False
        and cfg['student_READY'] is False,'Exact finite missing-only scope required')
    require(peer['status']==REVIEW and peer['independent'] is True
        and peer['reviewer'] not in ('','/root/repair_independent_review')
        and peer['configuration']==binding['configuration'] and peer['sources']==cfg['sources'],
        'Non-author exact review required')
    for name,digest in cfg['sources'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
            'Relative frozen source path required')
        require(ref(literal(ROOT/name))['sha256']==digest,'Source changed: '+name)
    old=checked(cfg['prior_configuration']);ob=checked(cfg['prior_binding'])
    require(ob['job_id']=='10938402' and ob['status']=='ROOT_BOUND_FORMAL_DT_SELECTION_V1'
        and ob['owner']=='Root' and ob['configuration']==cfg['prior_configuration']
        and ob['nonce']==old['nonce'] and old['budgets']==BUDGETS
        and old['output']==str(prior.OUTPUT),'Executed V1 binding/config required')
    old_peer=checked(ob['independent_review']);ownership=checked(ob['ownership'])
    require(old_peer['status']==prior.REVIEW and old_peer['independent'] is True
        and old_peer['configuration']==cfg['prior_configuration'] and old_peer['sources']==old['sources']
        and ownership['configuration']==cfg['prior_configuration']
        and ownership['job_id']=='10938402' and ownership['no_live_duplicate_selector'] is True,
        'Original reviewed physical ownership required')
    for name,digest in old['sources'].items():
        require(cfg['sources'].get(name)==digest,'All executed398 sources must be retained unchanged')
    for key in ('pool','eligibility','geometry_configuration','producer_binding','geometry_binding'):
        require(cfg[key]==old[key],'Paid scientific identity differs: '+key)
    gcfg=checked(cfg['geometry_configuration']);gb=checked(cfg['geometry_binding']);pb=checked(cfg['producer_binding'])
    require(gb['job_id']=='10937866' and pb['job_id']=='10937865'
        and gb['mode']=='geometry' and pb['mode']=='prepare'
        and gb['producer_binding']==cfg['producer_binding']
        and gb['configuration']==pb['configuration']==cfg['geometry_configuration']
        and gcfg['dt_config']['pool']==cfg['pool'] and gcfg['dt_config']['eligibility']==cfg['eligibility'],
        'Exact paid12 provider graph required')
    gpeer=checked(gb['independent_review'])
    require(gpeer['status']=='PASS_INDEPENDENT_DT_FULL_GEOMETRY_SOURCE_CONFIG_V1'
        and gpeer['independent'] is True and gpeer['configuration']==cfg['geometry_configuration']
        and gpeer['sources']==gcfg['sources'] and pb['independent_review']==gb['independent_review'],
        'Geometry review differs')
    require(binding['owner']=='Root' and binding['status']=='ROOT_BOUND_MISSING_ONLY_FORMAL_DT_SELECTION_V2'
        and binding['nonce']==cfg['nonce'],'Fresh physical binding required')
    claim=checked(binding['ownership'])
    require(claim['owner']=='Root' and claim['configuration']==binding['configuration']
        and claim['job_id']==binding['job_id'] and claim['scope']==SCOPE
        and claim['no_live_duplicate_selector'] is True,'Fresh exclusive ownership required')
    return cfg,old,gcfg,gb

def terminal_predecessor():
    raw=subprocess.check_output(['sacct','-D','-n','-P','-X','-j','10938402',
        '--format=JobIDRaw,JobID,State,ExitCode,User,Account,Partition,QOS,ElapsedRaw,AllocCPUS,AllocTRES'],text=True,timeout=30)
    rows=[x.split('|') for x in raw.splitlines() if x.strip()]
    require(len(rows)==1 and len(rows[0])>=11,'Exact single predecessor accounting row required')
    row=rows[0]
    require(row[:8]==['10938402','10938402','FAILED','1:0','luyangz','cx','general','normal']
        and row[8].isdecimal() and int(row[8])>0 and row[9]=='8','Actual paid V1 failure required')
    tres=dict(x.split('=',1) for x in row[10].split(',') if '=' in x)
    require(tres.get('gres/gpu')=='1' and tres.get('mem') in ('64G','65536M'),
        'Original failed allocation cost differs')
    queue=subprocess.check_output(['squeue','-r','-h','-u','luyangz','-o','%i'],text=True,timeout=30)
    require('10938402' not in queue.split(),'Predecessor still in live queue')
    return dict(raw_accounting=raw,raw_queue=queue,elapsed_allocated_seconds=int(row[8]),
        allocated_CPUs=8,allocated_GPUs=1,old_failure_preserved=True)

def actor(binding,cfg):
    j=os.environ.get('SLURM_JOB_ID','')
    require(j.isdecimal() and j==binding['job_id'] and os.getuid()==2708899
        and not os.environ.get('SLURM_ARRAY_JOB_ID') and os.environ.get('SLURM_RESTART_COUNT','0')=='0',
        'Fresh actual scalar missing-only actor required')
    raw=subprocess.check_output(['scontrol','show','job','-o',j],text=True,timeout=30);f=prior.fields(raw)
    expected=dict(JobId=j,JobName='dsmm-root-dt-select-v2',JobState='RUNNING',UserId='luyangz(2708899)',
        Account='cx',Partition='general',QOS='normal',NumCPUs='8',NumTasks='1',NumNodes='1',Restarts='0',
        TimeLimit='01:00:00',TresPerNode='gres/gpu:L40S:1',WorkDir=str(ROOT),Command=str(HERE/'run.sbatch'),
        Comment='dsmm-root-dt-select-v2:'+cfg['nonce'],NodeList=socket.gethostname().split('.')[0])
    require(all(f.get(k)==v for k,v in expected.items()),'Actual General CPU8/64G/L40S route differs')
    tres=dict(x.split('=',1) for x in f['AllocTRES'].split(',') if '=' in x)
    require(tres.get('cpu')=='8' and tres.get('mem') in ('64G','65536M') and tres.get('gres/gpu')=='1',
        'Actual allocation differs')
    require(subprocess.check_output(['scontrol','write','batch_script',j,'-'],timeout=30)==(HERE/'run.sbatch').read_bytes(),
        'Actual spool differs')
    return dict(job_id=j,node=f['NodeList'],scheduler=raw,new_forwards=0,ward_recomputations=0)

def validate_choice(result,fraction,pool,ids):
    positions=result['positions'];indices=pool['source_indices'];blocks=prior.operators.source_blocks(indices)
    require(type(result['count']) is int and type(positions) is list
        and all(type(p) is int and 0<=p<len(ids) for p in positions)
        and positions==sorted(set(positions)) and result['ratio']==float(fraction)
        and result['count']==len(positions)==sum(result['source_quotas'])
        and result['source_indices']==[indices[p] for p in positions]
        and result['source_counts']==[len(b) for b in blocks]
        and len(result['source_quotas'])==12 and result['top_up'] is False
        and result['whole_conversations'] is True and result['author_original_order_preserved'] is True,
        'Original no-topup whole-conversation operator scope differs')
    chosen=set(positions)
    require(all(type(q) is int and 0<=q<=len(b) and sum(p in chosen for p in b)==q
        for b,q in zip(blocks,result['source_quotas'])),'Actual per-source independent floors differ')
    require(result['count']<=budget_size(len(ids),fraction),'Actual count exceeds declared nominal budget')
    return positions

def candidate(cfg,fraction,result,records_ref,op_ref,ids,sources):
    chosen=[ids[p] for p in result['positions']]
    return dict(schema='native-record-selection-v1',status='PENDING_INDEPENDENT_OPERATOR_OUTPUT_REVIEW',
        pool=cfg['pool'],method='datatailor',fraction=fraction,selection_seed=1234,count=len(chosen),record_ids=chosen,
        records_json=records_ref,groups=None,features=None,assets=None,algorithm_sources=sources,method_review=None,
        budget=dict(unit='whole_conversation_count',value=len(chosen),fraction=fraction,eligible_pool_count=len(ids)),
        requested_K=budget_size(len(ids),fraction),operator_result=op_ref,
        source_quotas=result['source_quotas'],provisional_not_student_READY=True)

def authenticate_prior(cfg,old,pool,records,ids,mref,proof,digest):
    artifacts=cfg['prior_artifacts'];start=checked(artifacts['START.json']);failure=checked(artifacts['FAILURE.json'])
    require(start['configuration']==cfg['prior_configuration'] and start['actual']['job_id']=='10938402'
        and failure['actual']==start['actual'] and failure['error']=="ValueError('Unregistered budget')"
        and failure['partial_outputs_preserved'] is True,'Authentic original partial failure required')
    require(set(p.name for p in literal(prior.OUTPUT).iterdir())==
        {'START.json','FAILURE.json','budget-0.05','budget-0.10','budget-0.15'},
        'Old output changed; explicit reconciliation required')
    output=[]
    for fraction in BUDGETS[:3]:
        entry=artifacts['budget-'+fraction];directory=literal(prior.OUTPUT/('budget-'+fraction))
        required={'OPERATOR_RESULT.json','records.json'}|({'SELECTION_CANDIDATE.json'} if fraction!='0.15' else set())
        require(set(p.name for p in directory.iterdir())==required,'Unexpected old partial output')
        for name in required:
            require(entry[name]['path']==str(directory/name),'Foreign prior artifact path')
        result=checked(entry['OPERATOR_RESULT.json']);positions=validate_choice(result,fraction,pool,ids)
        require(result['paid_union']==checked(cfg['geometry_configuration'])['paid_union']
            and result['ordered_paid_value_sha256']==digest['ordered_paid_value_sha256']
            and result['full_geometry_manifest']==mref and result['full_geometry']==proof
            and result['measurement_projection']=='EXACT_QUALIFIED_PLAN_V4_measurement_sources'
            and result['original_whole_conversations'] is True
            and result['native_original_conversation_turn_counts'] is True
            and result['paid_feature_lengths_preserved_as_provenance'] is True and result['no_top_up'] is True,
            'Paid prior operator provenance differs')
        require(checked(entry['records.json'])==[records[p] for p in positions],
            'Original selected records do not match original positions/conversations')
        cref=entry.get('SELECTION_CANDIDATE.json')
        if fraction!='0.15':
            require(checked(cref)==candidate(cfg,fraction,result,entry['records.json'],entry['OPERATOR_RESULT.json'],ids,old['sources']),
                'Original completed candidate changed')
        output.append(dict(fraction=fraction,actual_K=result['count'],requested_K=budget_size(len(ids),fraction),
            candidate=cref,records=entry['records.json'],operator_result=entry['OPERATOR_RESULT.json'],
            operator_result_reused=True,selected_records_reused=True))
    return output

def reserve(binding,cfg,actual):
    grant=checked(binding['storage_reservation'])
    require(grant['owner']=='Root' and grant['configuration']==binding['configuration']
        and grant['output']==cfg['output'] and grant['reserved_bytes']==cfg['reserved_bytes']
        and grant['reserved_inodes']==cfg['reserved_inodes'] and grant['other_outstanding_reserved_bytes']>=1<<40
        and grant['other_outstanding_reserved_inodes']>=1100000,'Measured separate4GiB reservation required')
    with literal(prior.LOCK).open('r+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        out=literal(cfg['output']);require(not out.exists(),'Existing continuation requires reconciliation')
        out.parent.mkdir(parents=True,exist_ok=True);v=os.statvfs(out.parent)
        require(v.f_bavail*v.f_frsize>=cfg['reserved_bytes']+grant['other_outstanding_reserved_bytes']+(2<<30)
            and v.f_favail>=cfg['reserved_inodes']+grant['other_outstanding_reserved_inodes']+1000,
            'Actual shared filesystem insufficient')
        path=CONTROL/'FUNCTION.json';require(not path.exists(),'Missing-only selector already owned')
        claim=seal(path,dict(owner='Root',function_identity=function_identity(cfg),configuration=binding['configuration'],
            actual=actual,scope=SCOPE,output=cfg['output'],no_global_FUNCTION_roster_mutation=True))
        out.mkdir();return out,claim,dict(available_bytes=v.f_bavail*v.f_frsize,available_inodes=v.f_favail,
            actual=actual,prior_grant=binding['storage_reservation'])

def finish_missing(cfg,old,out,pool,records,ids,info,tau,unique,represent,mref,proof,digest):
    outputs=authenticate_prior(cfg,old,pool,records,ids,mref,proof,digest)
    # Only the missing15% wrapper candidate is new. Its paid operator/records stay at V1.
    e=outputs[2];r15=checked(e['operator_result']);d15=out/'budget-0.15';d15.mkdir()
    e['candidate']=seal(d15/'SELECTION_CANDIDATE.json',candidate(cfg,'0.15',r15,e['records'],e['operator_result'],ids,cfg['sources']))
    # The only newly computed scientific selection is20%, using the unchanged published operator.
    result=prior.operators.datatailor_select(records,pool['source_indices'],info,tau,unique,represent,0.20)
    positions=validate_choice(result,'0.20',pool,ids);directory=out/'budget-0.20';directory.mkdir()
    content=(json.dumps([records[p] for p in positions],ensure_ascii=False,separators=(',',':'))+'\n').encode()
    require(len(content)<=7<<29,'New selected records exceed3.5GiB reservation')
    with (directory/'records.json').open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
    records_ref=ref(directory/'records.json')
    op=seal(directory/'OPERATOR_RESULT.json',dict(**result,paid_union=checked(cfg['geometry_configuration'])['paid_union'],
        ordered_paid_value_sha256=digest['ordered_paid_value_sha256'],full_geometry_manifest=mref,full_geometry=proof,
        measurement_projection='EXACT_QUALIFIED_PLAN_V4_measurement_sources',original_whole_conversations=True,
        native_original_conversation_turn_counts=True,paid_feature_lengths_preserved_as_provenance=True,no_top_up=True))
    cref=seal(directory/'SELECTION_CANDIDATE.json',candidate(cfg,'0.20',result,records_ref,op,ids,cfg['sources']))
    outputs.append(dict(fraction='0.20',actual_K=result['count'],requested_K=budget_size(len(ids),'0.20'),
        candidate=cref,records=records_ref,operator_result=op,operator_result_reused=False,selected_records_reused=False))
    return outputs,len(content)

def run(binding_path):
    if not __debug__: raise RuntimeError('Optimized execution forbidden')
    started=time.monotonic();binding=checked(ref(literal(binding_path)));cfg,old,gcfg,gb=documents(binding)
    actual=actor(binding,cfg);terminal=terminal_predecessor()
    manifest,mref,rows,accounting=prior.load_geometry(cfg,gcfg,gb)
    out,claim,space=reserve(binding,cfg,actual)
    seal(out/'START.json',dict(configuration=binding['configuration'],actual=actual,claim=claim,storage=space,
        predecessor_accounting=terminal,prior_configuration=cfg['prior_configuration'],prior_binding=cfg['prior_binding']))
    try:
        pool,records,ids,info,tau,lengths,digest,observed=prior.paid_values(gcfg)
        require(manifest['ordered_paid_value_sha256']==digest['ordered_paid_value_sha256'],'Paid union differs')
        unique,represent,proof=prior.join_relationships(pool,ids,manifest,rows,info)
        outputs,written=finish_missing(cfg,old,out,pool,records,ids,info,tau,unique,represent,mref,proof,digest)
        return seal(out/'RESULT.json',dict(status='COMPLETE_FOUR_NATIVE_DT_SELECTION_CANDIDATES_PENDING_INDEPENDENT_OUTPUT_REVIEW',
            actual=actual,configuration=binding['configuration'],claim=claim,runtime=observed,paid_union=gcfg['paid_union'],
            paid_value_digest=digest,geometry_configuration=cfg['geometry_configuration'],geometry_manifest=mref,
            geometry_results=proof,actual_geometry_accounting=accounting,prior_configuration=cfg['prior_configuration'],
            prior_binding=cfg['prior_binding'],predecessor_accounting=terminal,outputs=outputs,new_selected_record_bytes=written,
            source_block_count=12,full_pool_count=len(ids),paid_feature_length_min=min(lengths),paid_feature_length_max=max(lengths),
            recomputed_selection_budgets=['0.20'],reused_operator_budgets=BUDGETS[:3],
            reused_completed_candidates=['0.05','0.10'],new_candidate_only_budgets=['0.15'],
            cached_paid16_values_reread_not_reacquired=True,elapsed_seconds=time.monotonic()-started,
            max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_forwards=0,ward_recomputations=0,
            relationship_recomputations=0,student_READY=False,all_prior_feature_geometry_and_current_costs_count=True))
    except BaseException as error:
        seal(out/'FAILURE.json',dict(actual=actual,error=repr(error),elapsed_seconds=time.monotonic()-started,
            partial_outputs_preserved=True,no_automatic_retry=True));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);a=p.parse_args();print(json.dumps(run(a.binding)))
