"""Independent Babel observation of unchanged Native7 math; no PASS promotion."""
import copy
from pathlib import Path
from experiments.root_native_qualification_v10_20261024 import common as b
from experiments.root_native_campaign_v7_20261024 import common as n
ROOT=n.ROOT
HERE=Path(__file__).resolve().parent
ID='babel-native7-operator-trace-v2'
PACKAGE='experiments.root_babel_operator_trace_v2_20261024'
BASE='/home/luyangz/data_selection_multimodal/native_campaign_v3_20261023'
OUTPUT_BASE='/data/user_data/luyangz/native_campaign_20261023/v3'
LOCK='/home/luyangz/data_selection_multimodal/collaboration/babel-two-track-20261019/ADMISSION.lock'
CELLS={'T1':1};GIB=1<<30
ALLOCATION=dict(gpus=1,cpus=8,memory_mb=65536,minutes=55,reserve_seconds=300,microbatch=4,loader_workers=0)
RUNTIME_BUDGET=dict(minutes=55,reserve_seconds=300,minimum_neural_seconds=600,minimum_initial_remaining_seconds=3180,final_readout_minimum_seconds=60)
TOLERANCE=dict(absolute=1e-6,relative=1e-5)
ANCHOR=dict(path=BASE+'/registry/engineering/babel-v7-engineering-parity-v10/STUDY.json',bytes=4984,sha256='45c64649873415a10544c34e67a11ccaaab4b1c006006fffb75aa8e179cd7c21')
PRIOR_STORAGE=dict(path=BASE+'/registry/engineering/babel-native7-operator-trace-v1/T1/attempt/ACTUAL_STORAGE.json',bytes=3254,sha256='9168b65638dc79cc810dd1bb1b201d7f0544aea4d80cf56235c338a6a71a095f')
PAID_READOUT=dict(path=BASE+'/registry/engineering/babel-v7-engineering-parity-v10/actual/RAW_READOUT.json',bytes=34218,sha256='05fa5b6b39cdbfedd775c29e24f9d8751935d5b37d2492c6d16ce47ef4dd6f60')
require=n.require;checked=n.checked;seal=n.seal;desc=n.desc;key=n.key;read=n.read;now=n.now

def sources():
    own=[x for x in HERE.iterdir() if x.suffix in ('.py','.sbatch') and not x.name.startswith('test_')]
    return dict(sorted({**b.sources(),**{str(x.relative_to(ROOT)):n.sha(x) for x in own}}.items()))

def bare(value):return {k:v for k,v in value.items() if k!='review'}

def literal(path):
    path=Path(path)
    require(path.is_absolute() and '..' not in path.parts and any(path.is_relative_to(Path(root)) for root in (BASE,OUTPUT_BASE)), 'Outside exact Babel diagnostic roots')
    require(all(not p.is_symlink() for p in (path,*path.parents)), 'Symlink in owned artifact route')
    return path

def anchor(study):
    require(study['anchor_study']==ANCHOR and study['paid_anchor_readout']==PAID_READOUT,'Only authentic frozen Babel V10 paid anchor')
    old=checked(ANCHOR);approval=checked(old['review'])
    require(old['schema']=='native-engineering-parity-study-v10' and old['id']=='babel-v7-engineering-parity-v10'
        and old['driver_sources']==b.sources() and old['owner']=='Root' and old['total_steps']==8 and old['interrupt_step']==4,'Babel V10 identity changed')
    require(approval['status']=='PASS_INDEPENDENT_ENGINEERING_DRIVER_AND_STUDY' and approval['study_identity']==key(bare(old)) and approval['driver_sources']==b.sources(),'Authentic independently reviewed anchor required')
    require(old['native_target']==study['native_target'] and old['native_target']['source_packet']['sha256']=='a704d9716719a9ad2a0cd0890c24fec35c15bd085b7c2c98fe38ea0c98961830','Native7 target changed')
    modules,packet=b.target_modules(old['native_target']);original=checked(old['native_configurations']['reference'])
    require(len(packet['sources'])==91 and original['sources']==packet['sources'] and original['recipe']==modules['config'].RECIPE,'Unchanged Native7/91 recipe required')
    profile=checked(original['site_profile']);require(profile['site']=='babel' and profile['partition']=='general' and profile['gpu_type']=='L40S','Only actual Babel General L40S anchor')
    require('media_transport' not in original['providers'] and 'runtime_images' not in original['providers'],'Persistent Babel images only')
    paid=checked(PAID_READOUT)
    require(paid['status']=='FAIL_ACTUAL_ENGINEERING_READOUT' and paid['qualification_promoted'] is False
        and paid['study_id']==old['id'] and paid['study_identity']==key(bare(old)) and paid['driver_sources']==b.sources()
        and paid['native_sources']==original['sources'] and paid['native_target']==old['native_target'],'Preserve actual failed Babel evidence, never relabel as PASS')
    return old,original,modules

def invariant_config(cfg,original,cell):
    require(cell=='T1','Exactly one intentional Babel hardware trace')
    for field in ('schema','owner','role','method','fraction','training_seed','selection_seed','recipe','sources','official_commit','checkpoint_seconds','qualification_max_steps','site_profile'):
        require(cfg[field]==original[field],'Changed frozen scientific field: '+field)
    require(cfg['role']=='qualify' and cfg['method']=='full' and cfg['fraction']=='1' and cfg['qualification_max_steps']==8 and cfg['training_seed']==17 and cfg['selection_seed']==1234,'Full1024, seed17, native horizon8 retained')
    require(cfg['allocation']==ALLOCATION and cfg['diagnostic_request_budget']==RUNTIME_BUDGET,'Unreviewed finite runtime budget')
    require({k:v for k,v in cfg['allocation'].items() if k not in ('minutes','reserve_seconds')}=={k:v for k,v in original['allocation'].items() if k not in ('minutes','reserve_seconds')},'Changed world/microbatch/objective allocation')
    require(cfg['checkpoint_seconds']>55*60,'No periodic checkpoint during one-step control')
    for name in ('assets','environment','pool','eligibility','selection'):require(cfg['providers'][name]==original['providers'][name],'Changed frozen provider: '+name)
    require('media_transport' not in cfg['providers'] and 'runtime_images' not in cfg['providers'],'No ACES transport/alias in persistent Babel trace')
    require(cfg['diagnostic_runtime']==dict(policy='observe_native_dispatch_v1',math_changes=False),'Only unchanged observational trace')
    require(cfg['id']==ID+'-T1' and cfg['replication_block']==dict(replicate_site='babel',registered_purpose='intentional-L40S-operator-dispatch-trace-T1'),'Distinct hardware observation ownership required')
    return True

def design_contract(study,pair):
    arms={}
    for cell,cfg in pair.items():
        value=copy.deepcopy(cfg);value.pop('review',None);value.pop('storage_reservation',None);value['providers'].pop('ownership',None);arms[cell]=value
    return dict(schema='babel-native7-operator-trace-held-design-v2',id=ID,owner='Root',purpose='BABEL_OPERATOR_DISPATCH_TRACE_ONLY',driver_sources=sources(),anchor_study=study['anchor_study'],paid_anchor_readout=study['paid_anchor_readout'],native_target=study['native_target'],runtime_budget=RUNTIME_BUDGET,cells=CELLS,total_scheduler_steps=8,numeric_tolerance=TOLERANCE,configurations=arms,report=study['report'],maximum_new_final_capsules=1,reserved_bytes=8*GIB,no_restore_or_scientific_promotion=True)

def validate_design(value):
    require(value['schema']=='babel-native7-operator-trace-held-design-v2' and value['id']==ID and value['owner']=='Root' and value['purpose']=='BABEL_OPERATOR_DISPATCH_TRACE_ONLY' and value['driver_sources']==sources() and value['cells']==CELLS and value['total_scheduler_steps']==8 and value['runtime_budget']==RUNTIME_BUDGET and value['numeric_tolerance']==TOLERANCE and value['anchor_study']==ANCHOR and value['paid_anchor_readout']==PAID_READOUT,'Exact finite Babel design/source required')
    approval=checked(value['review'])
    require(approval['status']=='PASS_INDEPENDENT_BABEL_NATIVE7_OPERATOR_TRACE_HELD_DESIGN' and approval['design_identity']==key(bare(value)) and approval['driver_sources']==sources() and approval['reviewer'] not in ('/root/native_campaign_build','native_campaign_build'),'Independent exact held design required')
    return value

def validate(study,*,reviewed=True):
    require(study.get('schema')=='babel-native7-operator-trace-runtime-study-v3',
        'Superseded full-study route is disabled; use explicit conditional runtime inside owned GPU')
    from .launch import validate_runtime
    return validate_runtime(study)

def expected_endpoint(cell,status,step,observed):
    require(cell=='T1' and status=='ACCEPTED_NATIVE_CONTINUATION' and step==1 and observed==128,'Stop exactly accepted update1/128records while schedule remains8')
