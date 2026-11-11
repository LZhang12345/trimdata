"""Exact-six S1 recipient class; numerical runtime comes from genuine Root V4."""
from pathlib import Path
from experiments.root_fulltrajectory_student_v4_20261025 import protocol as base
from experiments.s1_fulltrajectory_held_cohort_v2_20261025 import protocol as held
n=base.n;native_config=base.native_config;qualification=base.qualification
ROOT=n.ROOT;HERE=Path(__file__).resolve().parent;PACKAGE='experiments.root_s1_shared_student_v1_20261025';AUTHOR='/root/aces_fixture_metadata'
require=n.require;checked=n.checked;seal=n.seal;key=n.key;sha=n.sha;desc=n.desc;read=n.read;canonical=n.canonical;now=n.now
SEEDS=base.SEEDS;RUNTIME=base.RUNTIME;CHECKPOINT_ROOT=base.CHECKPOINT_ROOT;CANARY_ID=base.CANARY_ID
CAPSULE_BYTES=base.CAPSULE_BYTES;TRANSIENT_BYTES=base.TRANSIENT_BYTES;SEED_POLICY=base.SEED_POLICY
NATURAL_ANCHOR_SEED=base.NATURAL_ANCHOR_SEED;NATURAL_ANCHOR_READOUT=base.NATURAL_ANCHOR_READOUT
GATE='REQUIRES_EXPLICIT_BABEL_S1_SHARED_RUNTIME_AUTHORIZATION_V1';KEYS=('10:0R','10:H0')
SELECTIONS={
 '0R':dict(method='s1_r1_select_v4_0R',selection=dict(path='/data/user_data/luyangz/native_campaign_20261023/s1-round1-2x2-v1/full-10922767-s0/0.10-0R/SELECTION.json',bytes=1431003,sha256='5fea3dae95e7c02e4cfc170422d750a9e6eb2f5fec88ab5e298876254b607f4e'),method_review=dict(path='/data/user_data/luyangz/native_campaign_20261023/s1-round1-2x2-v1/full-10922767-s0/0.10-0R/OUTPUT_REVIEW_V1-10924262-s0.json',bytes=2440,sha256='9fc04ecaced8b291bcc00e9b1d1a6bb7fd6b71d9c7b56c260758047c1bb6993f')),
 'H0':dict(method='s1_r1_select_v4_H0',selection=dict(path='/data/user_data/luyangz/native_campaign_20261023/s1-round1-2x2-v1/full-10922767-s0/0.10-H0/SELECTION.json',bytes=1430727,sha256='aecddec0092d1839ecd801f95e78279c819ff0538468e4d07be3b2c7e81cf544'),method_review=dict(path='/data/user_data/luyangz/native_campaign_20261023/s1-round1-2x2-v1/full-10922767-s0/0.10-H0/OUTPUT_REVIEW_V1-10924262-s0.json',bytes=2441,sha256='2cfe5ee317374e80e757a5e4cc5dd44cf532c471973dda34f70031688901b5aa'))}
FINAL_SELECTION_RECEIPT_SHA='1f3f45fad0c13ca2e55047c74a27677a370c0f04e96e16160867ca81bbb34cc3'

def sources():
 own={str(f.relative_to(ROOT)):sha(f) for f in HERE.iterdir() if f.suffix in ('.py','.sbatch') and not f.name.startswith('test_')}
 return dict(sorted({**base.sources(),**held.sources(),**own}.items()))
def bare(value):return {k:v for k,v in value.items() if k!='review'}
horizon=base.horizon

def cell_id(key,seed):
 require(key in KEYS and seed in SEEDS,'Exact six S1 recipient grid only')
 return 's1-student-v1-10-'+key.split(':')[1]+'-s'+str(seed)

def supported(cfg):
 require(cfg['id'] in held.OLD_JOBS and cfg['fraction']=='0.10' and cfg['training_seed'] in SEEDS,'Only six exact S1 science IDs/seeds/budget')
 arm=cfg['id'].split('-')[-2]
 require(arm in SELECTIONS and cfg['id']==cell_id('10:'+arm,cfg['training_seed'])
     and cfg['method']==SELECTIONS[arm]['method'] and cfg['providers']['selection']==SELECTIONS[arm]['selection'],
     'Paid S1 method/selection identity cannot be relabeled or regenerated')
 return arm

def contract(cfg,natural):
 return dict(schema='production-student-runtime-gate-contract-v1',status=GATE,native_sources=cfg['sources'],recipe=cfg['recipe'],
     site_profile=cfg['site_profile'],environment=cfg['providers']['environment'],assets=cfg['providers']['assets'],site='babel',runtime=RUNTIME,
     qualification_evidence=natural,student_adapter=dict(package=PACKAGE,sources=sources()),ordinary_native_launch_forbidden=True,
     requires_separate_outer_authorization=True,natural_runtime_seed_policy=SEED_POLICY,target_initialization_seed=cfg['training_seed'])
