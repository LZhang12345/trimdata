"""Registered Root metadata draft, never a retirement, review or admission PASS."""
from pathlib import Path
from . import protocol as p
from experiments.root_production_student_adapter_v2_20261025.launch import require_metadata_registration


def draft(original_request,old_binding,predecessor_retirement,site_profile,output,registration):
 require_metadata_registration(registration)
 old,science,_=p.prior.reviewed(original_request)
 p.c.require(old['cell_id'] in p.OLD_JOBS,'Exact six original S1 scientific cells')
 profile=p.ncfg.validate_profile(p.c.checked(site_profile))
 p.c.require(profile['output_root']=='/data/hf_cache/project_checkpoints/DS-MultiModal/native_v3','New current shared checkpoint profile')
 control=Path(profile['registry_root'])/p.REGISTRY/old['cell_id']
 value=dict(schema=p.SCHEMA,owner='Root',status=p.STATUS,site='babel',cell_id=old['cell_id'],
     sources=p.sources(),native_sources=p.c.sources(),recipe=p.ncfg.RECIPE,original_request=original_request,
     old_binding=old_binding,predecessor_retirement=predecessor_retirement,scientific_design=science,
     selection_receipt=old['selection_receipt'],paired_root_requests=old['paired_root_requests'],
     future_release_gates=p.GATES,release_authorized=False,reserved_storage_bytes=0,
     site_profile=site_profile,outer_allocation=dict(gpus=1,cpus=8,memory_mb=65536,minutes=720),control_root=str(control),review=None)
 p.design(value,admission=True)
 p.c.require(p.c.within(output,profile['registry_root']),'Registered owned metadata output only')
 return p.c.seal(output,value)
