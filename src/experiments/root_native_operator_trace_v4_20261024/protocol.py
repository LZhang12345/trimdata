"""Exact Native7 math and observed replay contrasts; no numerical relaxation."""
import copy
from pathlib import Path
from experiments.root_h100_native_qualification_v10_20261024 import protocol as h
from experiments.root_native_qualification_v10_20261024 import common as b
from experiments.root_native_campaign_v7_20261024 import common as n
from experiments.root_native_replay_dispersion_v1_20261024 import protocol as prior

ROOT=n.ROOT
HERE=Path(__file__).resolve().parent
ID='aces-native7-operator-trace-v4'
CELLS={'T1':1}
GIB=1<<30
ALLOCATION=dict(gpus=1,cpus=8,memory_mb=65536,minutes=20,reserve_seconds=300,microbatch=4,loader_workers=0)
TOLERANCE={'absolute':1e-6,'relative':1e-5}
RUNTIME_BUDGET=dict(minutes=20,reserve_seconds=300,collector_max_seconds=300,minimum_neural_seconds=420,minimum_initial_remaining_seconds=1080,final_readout_minimum_seconds=60)
PREDECESSOR=dict(path='/scratch/user/u.lz341613/dsmm_native_campaign_20261023/v3/registry/engineering/aces-native7-operator-trace-v1/STUDY.json',bytes=7971,sha256='6ae0e9427b05e8d415da852173829241f8d61c4992be2980c0f43cc93e9cf5fa')
PACKAGE='experiments.root_native_operator_trace_v4_20261024'
BASE='/scratch/user/u.lz341613/dsmm_native_campaign_20261023'
LOCK='/scratch/user/u.lz341613/dsmm_root_initial24_aces_20261023/ADMISSION.lock'
require=n.require;checked=n.checked;seal=n.seal;desc=n.desc;key=n.key;read=n.read;now=n.now

def sources():
    paths=[p for p in HERE.iterdir() if p.suffix in ('.py','.sbatch') and not p.name.startswith('test_')]
    paths += [ROOT/'experiments/root_h100_engineering_launch_v10_20261024'/name for name in ('__init__.py','prepare.py')]
    return dict(sorted({**prior.sources(),**{str(p.relative_to(ROOT)):n.sha(p) for p in paths}}.items()))

def bare(value):return {k:v for k,v in value.items() if k!='review'}

def literal(path):
    p=Path(path)
    require(p.is_absolute() and '..' not in p.parts and p.is_relative_to(Path(BASE)),'Outside approved scratch namespace')
    require(all(not x.is_symlink() for x in (p,*p.parents)),'Symlink ancestor in owned diagnostic path')
    return p

def anchor(study):
    original=checked(study['anchor_study'])
    require(original['id']=='aces-v7-engineering-parity-v10' and original['schema']=='h100-transport-engineering-study-v10'
            and original['purpose']=='H100_TRANSPORT_ENGINEERING_ONLY','Only actual existing Native7 H100 engineering anchor')
    review=checked(original['review'])
    require(review['status']=='PASS_INDEPENDENT_H100_TRANSPORT_DRIVER_AND_STUDY'
            and review['study_identity']==key(bare(original)) and original['driver_sources']==h.sources(),
            'Anchor source/study metadata changed')
    require(original['native_target']==study['native_target'],'Target differs from actual Native7 anchor')
    modules,packet=b.target_modules(study['native_target'])
    require(study['native_target']['package']=='experiments.root_native_campaign_v7_20261024'
            and study['native_target']['source_packet']['sha256']==h.TARGET_PACKET
            and len(packet['sources'])==91,'Frozen Native7/91 required')
    original_config=checked(original['native_configurations']['reference'])
    require(original_config['sources']==packet['sources'] and original_config['recipe']==modules['config'].RECIPE,
            'Anchor native recipe/source mismatch')
    # This is an exact historical input bridge, not a fresh native PASS.
    raw=checked(original['raw_prefix_proof_copy']);proof=checked(original['actual_prefix_proof'])
    from experiments.root_native_qualification_v10_20261024.preflight import source_equivalence_bridge
    source_equivalence_bridge(proof,packet['sources'],raw=raw,
        assets=checked(original_config['providers']['assets']),environment=checked(original_config['providers']['environment']))
    return original,original_config,modules

def invariant_config(cfg,original,cell):
    require(cell in CELLS,'Unknown finite trace cell')
    fields=('schema','owner','role','method','fraction','training_seed','selection_seed','recipe','sources',
            'official_commit','checkpoint_seconds','qualification_max_steps')
    for name in fields:require(cfg[name]==original[name],'Replay changes native scientific field: '+name)
    require(cfg['role']=='qualify' and cfg['method']=='full' and cfg['fraction']=='1'
            and cfg['qualification_max_steps']==8 and cfg['training_seed']==17 and cfg['selection_seed']==1234,
            'The trace retains full1024, seed17, original8-update horizon')
    require(cfg['allocation']==ALLOCATION and cfg['checkpoint_seconds']>55*60,'Changed world/microbatch or periodic saving')
    require({k:v for k,v in cfg['allocation'].items() if k not in ('minutes','reserve_seconds')}=={k:v for k,v in original['allocation'].items() if k not in ('minutes','reserve_seconds')},'Runtime request changed scientific allocation')
    require(cfg['diagnostic_request_budget']==RUNTIME_BUDGET,'Unreviewed finite runtime budget')
    for name in ('assets','environment','pool','eligibility','selection'):
        require(cfg['providers'][name]==original['providers'][name],'Replay changes frozen provider: '+name)
    require('runtime_images' not in cfg['providers'],'Logical configuration cannot contain a stale physical alias')
    old=checked(original['providers']['media_transport']);new=checked(cfg['providers']['media_transport'])
    require({k:v for k,v in new.items() if k!='ownership'}=={k:v for k,v in old.items() if k!='ownership'},
            'Transport logical content changed')
    require(cfg['diagnostic_runtime']==dict(policy='observe_native_dispatch_v1',math_changes=False), 'Unreviewed trace runtime policy')
    require(cfg['id']==ID+'-'+cell and cfg['replication_block']==dict(replicate_site='aces',
            registered_purpose='operator-dispatch-trace-'+cell),'Root must register intentional distinct replay identities')
    return True

def design_contract(study,pair):
    arms={}
    for cell,cfg in pair.items():
        value=copy.deepcopy(cfg)
        value.pop('review',None);value.pop('storage_reservation',None)
        value['providers'].pop('ownership',None)
        transport=checked(value['providers'].pop('media_transport'))
        value['transport_contract']={k:v for k,v in transport.items() if k!='ownership'}
        arms[cell]=value
    return dict(schema='native7-operator-trace-held-design-v4',id=ID,owner='Root',purpose='OPERATOR_DISPATCH_TRACE_ONLY',
        driver_sources=sources(),anchor_study=study['anchor_study'],native_target=study['native_target'],runtime_budget=RUNTIME_BUDGET,replaces_predecessor=PREDECESSOR,
        cells=CELLS,total_scheduler_steps=8,numeric_tolerance=TOLERANCE,configurations=arms,
        report=study['report'],maximum_new_final_capsules=1,reserved_bytes=8*GIB,
        no_restore_or_scientific_promotion=True)

def validate_design(value):
    require(value['schema']=='native7-operator-trace-held-design-v4' and value['id']==ID and value['owner']=='Root'
            and value['purpose']=='OPERATOR_DISPATCH_TRACE_ONLY' and value['cells']==CELLS
            and value['total_scheduler_steps']==8 and value['numeric_tolerance']==TOLERANCE
            and value['driver_sources']==sources() and value['runtime_budget']==RUNTIME_BUDGET and value['replaces_predecessor']==PREDECESSOR,'Wrong finite replay design/source')
    review=checked(value['review'])
    require(review['status']=='PASS_INDEPENDENT_NATIVE7_OPERATOR_TRACE_HELD_DESIGN'
            and review['design_identity']==key(bare(value)) and review['driver_sources']==sources()
            and review['reviewer'] not in ('/root/native_campaign_build','native_campaign_build'),
            'Independent exact held design review required')
    return value

def validate(study,*,reviewed=True):
    require(study['schema']=='native7-operator-trace-study-v4' and study['id']==ID and study['owner']=='Root'
            and study['purpose']=='OPERATOR_DISPATCH_TRACE_ONLY' and study['cells']==CELLS
            and study['total_scheduler_steps']==8 and study['numeric_tolerance']==TOLERANCE
            and study['driver_sources']==sources() and study['runtime_budget']==RUNTIME_BUDGET and study['replaces_predecessor']==PREDECESSOR,'Wrong frozen replay study')
    old,original,modules=anchor(study)
    pair={cell:checked(ref) for cell,ref in study['native_configurations'].items()}
    require(set(pair)==set(CELLS),'Exact one-cell roster required')
    outputs=[];jobs=[]
    from experiments.root_native_media_transport_20261023 import transport
    for cell,cfg in pair.items():
        invariant_config(cfg,original,cell)
        profile=modules['config'].validate_profile(checked(cfg['site_profile']))
        require(profile==checked(original['site_profile']),'Replay cannot silently change the qualified site route')
        require(profile['site']=='aces' and profile['code_root']==str(ROOT),'Actual current ACES source route required')
        review=checked(cfg['review'])
        require(review['status']=='PASS_INDEPENDENT_EXACT_SOURCE_AND_CONFIG'
                and review['sources']==cfg['sources'] and review['config_identity']==n.run_identity(cfg),
                'Independent exact replay native configuration required')
        owner=checked(cfg['providers']['ownership']);store=checked(cfg['storage_reservation'])
        job=study['jobs'][cell];jobs.append(job)
        require(job.isdecimal() and owner['status']=='ACTIVE_EXCLUSIVE_NATIVE_PLAN_CELL_CLAIM'
                and owner['owner']=='Root' and owner['cell_id']==cfg['id'] and owner['plan']==cfg['plan']
                and owner['pool']==cfg['providers']['pool'] and owner['site_profile']==cfg['site_profile']
                and owner['replicate_site']=='aces' and owner['replay_cell']==cell and owner['actual_held_job_id']==job,
                'Distinct Root replay acquisition ownership required')
        require(store['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and store['output']==cfg['output']
                and store['cell_id']==cfg['id'] and store['retained_accepted_states']==1
                and store['allow_new_intermediate_retirement'] is False
                and store['checkpoint_transaction_bytes']==8*GIB,'One final capsule per control, no retirement')
        pool=checked(cfg['providers']['pool']);selected=checked(cfg['providers']['selection'])
        tr=checked(cfg['providers']['media_transport'])
        transport._inputs(cfg['providers']['pool'],selected['records_json'],cfg['providers']['media_transport'],{'job_id':job})
        require(pool['count']==selected['count']==1024 and pool['engineering_only'] is True
                and tr['scope']=='ENGINEERING_ONLY','Unchanged full engineering construction required')
        outputs.append(cfg['output']);literal(cfg['output']);literal(cfg['report'])
    require(len(set(jobs))==len(set(outputs))==1,'Trace requires one distinct owned job/output')
    require(bare(validate_design(checked(study['design'])))==design_contract(study,pair),'Final controls drifted from held design')
    if reviewed:
        review=checked(study['review'])
        require(review['status']=='PASS_INDEPENDENT_NATIVE7_OPERATOR_TRACE_DRIVER_AND_STUDY'
                and review['study_identity']==key(bare(study)) and review['driver_sources']==sources()
                and review['reviewer'] not in ('/root/native_campaign_build','native_campaign_build'),
                'Exact independent final replay study review required')
    return pair,modules

def expected_endpoint(cell,status,step,observed):
    require(cell in CELLS and step==CELLS[cell] and observed==128*CELLS[cell],
            'Incomplete or overlong fresh-process control')
    require(status==('ACCEPTED_NATIVE_CONTINUATION' if cell=='T1' else 'NATIVE_TRAJECTORY_COMPLETE'),
            'Trace must stop at accepted update1 while retaining native horizon8')
