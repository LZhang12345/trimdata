"""Finite feature-free execution consumer, not a selection/provider controller."""
from pathlib import Path
from experiments.root_native_campaign_v7_20261024 import common as n
from experiments.root_native_campaign_v7_20261024 import config as native_config
from experiments.root_production_epoch_qualification_v1_20261024 import protocol as qualification
ROOT=n.ROOT;HERE=Path(__file__).resolve().parent
PACKAGE='experiments.root_fulltrajectory_student_v4_20261025'
AUTHOR='/root/aces_fixture_metadata'
require=n.require;checked=n.checked;seal=n.seal;key=n.key;sha=n.sha;desc=n.desc;read=n.read;canonical=n.canonical;now=n.now
SEEDS=(17,29,43)
KEYS=('5:uniform','10:uniform','20:uniform','5:source_sqrt','10:source_sqrt','20:source_sqrt')
CHECKPOINT_ROOT='/data/hf_cache/project_checkpoints/DS-MultiModal/native_v3'
CANARY_ID='production-student-v1-5-uniform-s29'
EXCLUDED_IDS={'production-student-v1-5-uniform-s17','production-student-v1-5-source_sqrt-s17'}
GATE='REQUIRES_EXPLICIT_BABEL_SHARED_RUNTIME_TARGET_AUTHORIZATION_V4'
RUNTIME='explicit_xformers_cutlass_fw_bw_deterministic_v1'
NATURAL_ANCHOR_SEED=17
NATURAL_ANCHOR_READOUT=dict(path='/home/luyangz/data_selection_multimodal/native_campaign_v3_20261023/registry/engineering/production-epoch-qual-v1-babel-s17/actual/RAW_READOUT.json',bytes=40998,sha256='be05899087c3f0dfaec9592dad5a960b8cd9d55e93bd9b1a2043b49164dd814c')
SEED_POLICY=dict(scope='GENERIC_NATIVE91_OPERATOR_RESTORE_ANCHOR_NOT_TARGET_SEED_REPLICATION',
    actual_natural_anchor_seed=17,actual_natural_anchor_readout=NATURAL_ANCHOR_READOUT,
    allowed_target_initialization_seeds=[17,29,43],required_new_shared_filesystem_canary_seed=29,
    own_target_initialization_config_and_accepted_RNG_required=True,evidence_relabeling_forbidden=True)

CAPSULE_BYTES=5*(1<<30);TRANSIENT_BYTES=6*(1<<30)

def sources():
 own={str(f.relative_to(ROOT)):sha(f) for f in HERE.iterdir() if f.suffix in ('.py','.sbatch') and not f.name.startswith('test_')}
 from experiments.root_production_student_launcher_v4_20261025 import protocol as original_launcher
 from experiments.root_fulltrajectory_held_cohort_v2_20261025 import protocol as held
 from experiments.root_fulltrajectory_student_v2_20261025 import protocol as anchor
 return dict(sorted({**anchor.sources(),**own}.items()))

def bare(value):return {k:v for k,v in value.items() if k!='review'}

def horizon(count):
 require(type(count) is int and count>=128,'Nonempty supported selected set with at least global128 required')
 batches=(count+3)//4;steps=batches//32
 require(steps>0,'Missing natural one-epoch horizon')
 used=min(count,steps*128)
 return dict(selected_count=count,loader_batches=batches,natural_updates=steps,optimized_records=used,
     unoptimized_tail_count=count-used,microbatch=4,global_batch=128,gradient_accumulation=32,
     num_train_epochs=1,max_steps=-1,drop_last=False,
     objective='UNCHANGED_NATIVE_MEAN_VALID_SHIFTED_TARGETS_PER_MICROBATCH_THEN_NATIVE_GA32',
     tail_policy='NATIVE_LOADER_FLOOR_NO_TOPUP_NO_RESCALE')

def cell_id(selection_key,seed):
 require(selection_key in KEYS and seed in SEEDS,'Outside finite18 compatible selection/seed grid')
 return 'production-student-v1-'+selection_key.replace(':','-')+'-s'+str(seed)

def supported(config):
 require(config['id']==cell_id(str(int(round(float(config['fraction'])*100)))+':'+config['method'],config['training_seed'])
     and config['id'] not in EXCLUDED_IDS and config['fraction'] in ('0.05','0.10','0.20'),
     'Only remaining Root16; current V2/canonical pilot, S1/full/15percent excluded')
 return True

def contract(config,natural_qualification):
 """A gate declaration, deliberately NOT a qualification PASS."""
 require(config['role']=='student','Production student role only')
 return dict(schema='production-student-runtime-gate-contract-v1',status=GATE,
     native_sources=config['sources'],recipe=config['recipe'],site_profile=config['site_profile'],
     environment=config['providers']['environment'],assets=config['providers']['assets'],
     site='babel',runtime=RUNTIME,qualification_evidence=natural_qualification,
     student_adapter=dict(package=PACKAGE,sources=sources()),ordinary_native_launch_forbidden=True,
     requires_separate_outer_authorization=True,natural_runtime_seed_policy=SEED_POLICY,
     target_initialization_seed=config['training_seed'])
