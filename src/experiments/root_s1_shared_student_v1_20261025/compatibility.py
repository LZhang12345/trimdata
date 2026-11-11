"""Exact metadata-only compatibility; never migrate an old checkpoint identity."""
from . import protocol as p
from experiments.root_production_student_adapter_v2_20261025 import protocol as prior
from experiments.root_fulltrajectory_held_cohort_v2_20261025.protocol import validate_historical_profile
STATUS='PASS_INDEPENDENT_METADATA_ONLY_STUDENT_RUNTIME_COMPATIBILITY_S1_V1'
FIELDS=('site','partition','qos','account','gpu_type','slurm_user','python','cache_root','code_root','registry_root')

def verify(ref,config,qualified_profile,load=p.checked):
 value=load(ref)
 p.require(value['status']==STATUS and isinstance(value.get('reviewer'),str) and value['reviewer'].strip()
     and value['reviewer']!=p.AUTHOR and value['old_adapter_sources']==prior.sources()
     and value['new_adapter_sources']==p.sources() and value['native_sources']==p.n.sources()==config['sources']
     and value['recipe']==config['recipe'] and value['qualified_site_profile']==qualified_profile
     and value['actual_site_profile']==config['site_profile'] and value['allow_old_checkpoint_identity_migration'] is False
     and value['new_checkpoint_root']==p.CHECKPOINT_ROOT,
     'Independent exact metadata/source/profile compatibility required; no identity relabel')
 old=validate_historical_profile(load(qualified_profile),load)
 new=p.native_config.validate_profile(load(config['site_profile']))
 p.require(all(old[k]==new[k] for k in FIELDS) and new['site']=='babel' and new['output_root']==p.CHECKPOINT_ROOT
     and config['allocation']['minutes'] in (360,720,1440) and new['max_minutes']>=config['allocation']['minutes'],
     'Qualified runtime paths/actor/device differ or longwall not actually authorized')
 # Actual training/objective audit body is byte-identical, including the
 # read-only Indexed fix and all positive-target guards.
 p.require((p.HERE/'observer.py').read_bytes()==(prior.HERE/'observer.py').read_bytes(),
     'Longwall metadata compatibility cannot authorize altered objective observer')
 return value
