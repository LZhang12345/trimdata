"""Finite result-gated controls, exact config reviews and immutable Native7 math."""
import copy
from pathlib import Path
from experiments.root_babel_operator_trace_v2_20261024 import protocol as trace
from experiments.root_native_qualification_v10_20261024 import common as b
from experiments.root_native_campaign_v7_20261024 import common as n
ROOT=n.ROOT;HERE=Path(__file__).resolve().parent
PACKAGE='experiments.root_native_deterministic_replication_v1_20261024'
BASE=trace.BASE;OUTPUT_BASE=trace.OUTPUT_BASE;LOCK=trace.LOCK;GIB=1<<30
ANCHOR=trace.ANCHOR;PAID_READOUT=trace.PAID_READOUT;PRIOR_STORAGE=trace.PRIOR_STORAGE
NATIVE_SHA='a704d9716719a9ad2a0cd0890c24fec35c15bd085b7c2c98fe38ea0c98961830'
TOLERANCE={'absolute':1e-6,'relative':1e-5}
CELLS={'D8':dict(runtime='explicit_cutlass_deterministic',arms=['reference','resume'],capsules=3,reserved_bytes=24*GIB)}
ALLOCATION=dict(gpus=1,cpus=8,memory_mb=65536,minutes=55,reserve_seconds=300,microbatch=4,loader_workers=0)
require=n.require;checked=n.checked;desc=n.desc;seal=n.seal;read=n.read;key=n.key;now=n.now;sha=n.sha;canonical=n.canonical;within=n.within
AUTHOR_IDENTITY='Root/background-research:21f634ffbb1d4ceda284aef8d9a4c63e'
BUILDER_IDENTITY='/root/replication_builder'
def independent(reviewer):
    return isinstance(reviewer,str) and bool(reviewer) and reviewer not in ('Root',AUTHOR_IDENTITY,BUILDER_IDENTITY,'replication_builder','/root/native_campaign_build','native_campaign_build')
RESULT_BASIS={
 'seed17_D8':dict(path=BASE+'/registry/engineering/babel-native7-followon-v1-D8/actual/RAW_READOUT.json',bytes=28846,sha256='56c0cb2ef8caf5d27acc88907d92cf277251f6aba71047742fe0d1602c95405b'),
 'native_F8':dict(path=BASE+'/registry/engineering/babel-native7-f8-phase-repair-v1-F8/actual/RAW_READOUT.json',bytes=18354,sha256='2f949296a73fe417afd530a82993011abf7f533bea3f8d5a493d7f65fa6670d6')}
def result_basis():
    d=checked(RESULT_BASIS['seed17_D8']);f=checked(RESULT_BASIS['native_F8'])
    require(d['status']=='PASS_ACTUAL_ENGINEERING_READOUT_UNREVIEWED' and d['study_id']=='babel-native7-followon-v1-D8' and d['qualification_promoted'] is False,'Actual finite seed17 D8 basis changed')
    require(f['status']=='FAIL_FINITE_FRESH8_READOUT' and f['study_id']=='babel-native7-f8-phase-repair-v1-F8' and f['qualification_promoted'] is False,'Actual negative native full8 basis changed')
    return RESULT_BASIS
def bare(v):return {k:x for k,x in v.items() if k!='review'}
def sources():
    own=[x for x in HERE.iterdir() if x.suffix in ('.py','.sbatch') and not x.name.startswith('test_')]
    policy=ROOT/'experiments/root_native_deterministic_attention_v1_20261024'
    return dict(sorted({**trace.sources(),**{str(x.relative_to(ROOT)):sha(x) for x in own+[policy/'__init__.py',policy/'policy.py',ROOT/'experiments/root_native_operator_trace_v1_20261024/__init__.py',ROOT/'experiments/root_native_operator_trace_v1_20261024/runtime_trace.py']}}.items()))
def literal(path):return trace.literal(path)
def lexical(path,base):
    value=Path(path);require(value.is_absolute() and '..' not in value.parts and value.is_relative_to(Path(base)),'Outside approved literal route');return value

def completed_trace(ref,cell):
    """Read only immutable HOME metadata. No live-parent validation or DATA stat."""
    require(cell in CELLS,'Only finite F8/D8 controls')
    raw=checked(ref);require(raw['status']=='ACTUAL_CONCRETE_OPERATOR_OBSERVATION_NOT_REPAIR_OR_QUALIFICATION' and raw['qualification_promoted'] is False,'Authentic completed diagnostic required; queue/running is NOT_READY')
    study=checked(raw['study']);launch=checked(study['conditional_launch'])
    from experiments.root_babel_operator_trace_v2_20261024.launch import validate_launch
    _,config,held,_=validate_launch(launch)
    require(study['native_configurations']=={'T1':launch['native_configuration']} and study['jobs']=={'T1':held['job_id']} and study['driver_sources']==trace.sources() and raw['driver_sources']==trace.sources(),'Completed trace actual source/job/config graph differs')
    require(raw['native_target']==study['native_target'] and study['native_target']['source_packet']['sha256']==NATIVE_SHA,'Completed trace Native7 target differs')
    child=checked(raw['child']);ops=checked(raw['operator_trace']);obs=checked(child['observation']);step=checked(child['files']['STEP-1.json'])
    require(child['study']==raw['study'] and child['native_configuration']==launch['native_configuration'] and child['actual_job_id']==held['job_id'] and child['actual_restart_count']==0 and child['driver_sources']==trace.sources() and child['native_target']==study['native_target'] and child['error'] is None,'Completed trace child identity differs')
    trace.expected_endpoint('T1',child['status'],step['global_step'],obs['actual_record_exposure'])
    require(obs['mask_failures']==0 and step['DS_skipped_steps']==0 and step['all_weights_finite'],'Diagnostic has an unrelated mask/nonfinite/update failure')
    require(child['executed_operator_trace']==raw['operator_trace'] and ops['observational_only'] is True and ops['dispatch_arguments_unmodified'] is True and ops['forward_calls']>0 and ops['backward_calls']>0 and raw['executed']==ops['executed'] and raw['pre_CUDA_policy']==ops['pre_CUDA_policy'],'No authenticated actual concrete FW/BW trace')
    pre=ops['pre_CUDA_policy'];require(pre['policy']=='observe_native_dispatch_v1' and pre['math_changes'] is False and pre['CUDA_initialized'] is False and pre['torch_version']=='2.1.2+cu121' and pre['xformers_version']=='0.0.23.post1','Unreviewed observed runtime')
    executed=[x for x in ops['executed'] if x['kind'] in ('forward','backward') and x['calls']>0]
    require(all(x['operator'] in pre['operators'] for x in executed),'Observed concrete operator lacks advertised identity')
    if cell=='D8':
        require(any(x['kind']=='backward' and pre['operators'][x['operator']]['IS_DETERMINISTIC'] is False for x in executed),'D8 is NOT_READY: trace did not identify targeted advertised nondeterministic backward')
        require(all(pre['operators'].get('xformers.ops.fmha.cutlass.'+name,{}).get('IS_DETERMINISTIC') is True for name in ('FwOp','BwOp')),'D8 is NOT_READY: CUTLASS pair not advertised deterministic')
    return raw,pre

def anchor(value):return trace.anchor(value)
def invariant(cfg,original,cell,arm):
    require(cfg['id']=='babel-native7-deterministic-replication-v1-seed29-'+cell+'-'+arm and cfg['report']==BASE+'/registry/engineering/babel-native7-deterministic-replication-v1-seed29-'+cell+'/'+arm and cfg['output']==OUTPUT_BASE+'/babel-native7-deterministic-replication-v1-seed29-'+cell+'-'+arm,'Exact new arm namespace required')
    require(cell in CELLS and arm in CELLS[cell]['arms'],'Unknown finite arm')
    for name in ('schema','owner','role','method','fraction','selection_seed','recipe','sources','official_commit','qualification_max_steps','site_profile'):
        require(cfg[name]==original[name],'Changed original scientific field: '+name)
    require(cfg['training_seed']==29,'Frozen paired initialization seed29 only; seed43 requires actual29 readout and separate design')
    require(cfg['sources']==n.sources() and cfg['recipe']==__import__('experiments.root_native_campaign_v7_20261024.config',fromlist=['RECIPE']).RECIPE,'Native7/91 math changed')
    require(cfg['allocation']==ALLOCATION and cfg['qualification_max_steps']==8 and cfg['checkpoint_seconds']>55*60,'Frozen8/world1/micro4/GA32/no periodic save required')
    for name in ('assets','environment','pool','selection','eligibility'):require(cfg['providers'][name]==original['providers'][name],'Changed actual1024 provider: '+name)
    require('media_transport' not in cfg['providers'] and 'runtime_images' not in cfg['providers'],'Only qualified persistent Babel fixture')
    require('readiness' not in cfg and cfg['replication_block']==dict(replicate_site='babel',registered_purpose='finite-'+cell+'-'+arm),'Exact-config branch and distinct intentional replication required')
    require(cfg['diagnostic_runtime']==dict(policy=CELLS[cell]['runtime'],math_changes=cell=='D8',objective_recipe_unchanged=True,runtime_requires_separate_qualification=True),'Runtime intervention must be disclosed')
    return cfg

def projection(study,configs):
    values={}
    for arm,cfg in configs.items():
        x=copy.deepcopy(cfg);x.pop('review',None);x.pop('storage_reservation',None);x['providers'].pop('ownership',None);values[arm]=x
    return dict(schema='native-deterministic-replication-design-v1',owner='Root',prepared_by=AUTHOR_IDENTITY,paired_initialization_seed=29,exposed_development_fixture=True,multi_flag_runtime_intervention=True,id=study['id'],cell=study['cell'],driver_sources=sources(),native_target=study['native_target'],anchor_study=ANCHOR,paid_anchor_readout=PAID_READOUT,result_basis=RESULT_BASIS,completed_trace=study['completed_trace'],configurations=values,report=study['report'],total_steps=8,interrupt_step=4,numeric_tolerance=TOLERANCE,runtime=CELLS[study['cell']]['runtime'],allocation=ALLOCATION,storage=CELLS[study['cell']],no_automatic_promotion=True)

def validate_design(value):
    require(value['schema']=='native-deterministic-replication-design-v1' and value['owner']=='Root' and value['driver_sources']==sources() and value['cell'] in CELLS,'Exact source/finite design required')
    require(value['id']=='babel-native7-deterministic-replication-v1-seed29-'+value['cell'],'Exact engineering identity')
    result_basis()
    completed_trace(value['completed_trace'],value['cell'])
    old,original,_=anchor(value)
    configs=value['configurations'];require(set(configs)==set(CELLS[value['cell']]['arms']),'Finite arm roster differs')
    for arm,cfg in configs.items():invariant(cfg,original,value['cell'],arm)
    require(projection(value,configs)==bare(value),'Full conditional design changed')
    review=checked(value['review']);require(review['status']=='PASS_INDEPENDENT_FINITE_DETERMINISTIC_REPLICATION_DESIGN' and review['design_identity']==key(bare(value)) and review['driver_sources']==sources() and independent(review['reviewer']),'Independent finite design review required')
    return value

def validate_study(study,*,gpu=False,reviewed=True):
    require(study['schema']=='native-deterministic-replication-study-v1' and study['owner']=='Root' and study['driver_sources']==sources() and study['cell'] in CELLS,'Exact finite study/source required')
    require(study['total_steps']==8 and study['interrupt_step']==4 and study['numeric_tolerance']==TOLERANCE and study['minimum_launch_seconds']==1200 and study['minimum_phase_seconds']==dict(reference=700,interrupted=420,resumed=700),'Frozen horizon/tolerance/real launch reserves required')
    design=validate_design(checked(study['design']));require(study['cell']==design['cell'] and study['id']==design['id'],'Design identity differs')
    held=checked(study['held_binding']);intent=checked(held['intent']);ack=checked(held['ack'])
    require(held['job_id']==study['job_id'] and study['job_id'].isdecimal() and held['owner']=='Root' and held['cell']==study['cell'] and held['design']==study['design'] and held['driver_sources']==sources() and held['spool_reconciled'] is True,'Exact held job/design graph required')
    require(intent['status']=='INTENT_BEFORE_SUBMISSION' and intent['owner']=='Root' and intent['cell']==study['cell'] and intent['design']==study['design'] and intent['driver_sources']==sources() and ack['job_id']==study['job_id'] and ack['intent']==held['intent'],'Durable exact intent/ACK before execution required')
    require(study['Root_collective_authorization']==dict(owner='Root',study_id=study['id'],total_reserved_bytes=CELLS[study['cell']]['reserved_bytes'],maximum_new_capsules=CELLS[study['cell']]['capsules'],collective_consumers_reconciled=True),'Exact Root collective storage authorization required')
    configs={arm:checked(ref) for arm,ref in study['native_configurations'].items()}
    require(projection(study,configs)==bare(design),'Job-bound config changed reviewed design')
    require(study['runtime']==CELLS[study['cell']]['runtime'],'Explicit runtime factor differs')
    modules,packet=b.target_modules(study['native_target'])
    for arm,cfg in configs.items():
        review=checked(cfg['review']);require(review['status']=='PASS_INDEPENDENT_EXACT_SOURCE_AND_CONFIG' and review['config_identity']==n.run_identity(cfg) and review['sources']==packet['sources'] and independent(review.get('reviewer')),'Genuine exact config review required; no plan/READY fallback')
        owner=checked(cfg['providers']['ownership']);storage=checked(cfg['storage_reservation'])
        require(owner['actual_held_job_id']==study['job_id'] and owner['qualification_study']==study['id'] and owner['qualification_arm']==arm and owner['held_binding']==study['held_binding'],'Exact current-job arm ownership required')
        require(storage['Root_collective_authorization']==study['Root_collective_authorization'] and storage['retained_accepted_states']==(2 if arm=='resume' else 1) and storage['allow_new_intermediate_retirement'] is False and storage['checkpoint_transaction_bytes']==8*GIB and storage['total_pair_reserved_bytes']==CELLS[study['cell']]['reserved_bytes'] and storage['fresh_recheck_required_before_neural'] is True,'Exact collective accepted-state reservation required')
        require(storage['prior_actual_measurement']==PRIOR_STORAGE and storage['measurement_timing']=='PRIOR_10913252_SNAPSHOT_NOT_CURRENT_ALLOCATION','Historical measurement cannot become current-job evidence')
        if gpu:
            modules['config'].validate(cfg)
            pool=checked(cfg['providers']['pool']);require(pool['schema']=='native-engineering-pool-v1' and pool['count']==1024,'Whole engineering1024 only')
            from experiments.root_native_qualification_v10_20261024.preflight import verify_shared
            verify_shared(checked(ANCHOR),cfg,raw=True)
    if reviewed:
        review=checked(study['review']);require(review['status']=='PASS_INDEPENDENT_FINITE_DETERMINISTIC_REPLICATION_STUDY' and review['study_identity']==key(bare(study)) and review['driver_sources']==sources() and independent(review['reviewer']),'Genuine final exact study review required')
    return configs,modules

def native_identity(config,modules):return modules['common'].run_identity(config)
