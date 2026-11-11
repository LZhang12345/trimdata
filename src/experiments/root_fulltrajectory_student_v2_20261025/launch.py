"""Longwall current allocation; no credential inferred from ambient step names."""
import os,re
from pathlib import Path
from . import protocol as p

def require_current(ref,config_ref,permit_ref,config,profile,runner=None):
 value=p.checked(ref)
 p.require(value['schema']=='root-longwall-student-child-binding-v1' and value['owner']=='Root'
     and value['configuration']==config_ref and value['authorization']==permit_ref
     and value['adapter_sources']==p.sources() and value['actual_restart_count']==int(os.environ.get('SLURM_RESTART_COUNT','0')),
     'Exact new longwall child binding required')
 from .driver import current
 b,request,actual,activation=current(value['physical_binding'],value['activation'],runner=runner,minimum_seconds=300)
 p.require(activation['configuration']==config_ref and value['physical_job_id']==b['job_id']
     and value['nonce']==b['nonce'] and value['command']==b['command'],'Current physical/config/nonce graph differs')
 return dict(binding=ref,actual_slurm=actual['scheduler'],actual_job_id=b['job_id'],actual_EndTime=actual['end_epoch'],
     scheduler_mutations=False,Root_owns_atomic_requeuehold=True)

def require_attempt(attempt,config,profile,current,binding):
 attempt=Path(attempt);report=Path(config['report']);job=current['actual_job_id']
 p.require(attempt.is_absolute() and '..' not in attempt.parts and p.n.within(report,profile['registry_root'])
     and not any(v.is_symlink() for v in (attempt,*attempt.parents)),'Literal actual owned longwall report/attempt required')
 name=job+'-r'+os.environ.get('SLURM_RESTART_COUNT','0')+'-'+binding['nonce'][:8]
 expected=report/'attempts'/name/'child'
 p.require(attempt==expected and binding['child_attempt']==str(expected) and not attempt.exists(),'Fresh immutable child attempt required')
 return expected

# Metadata preparation retains the established genuinely owned GPU registration.
from experiments.root_production_student_adapter_v2_20261025.launch import require_metadata_registration
