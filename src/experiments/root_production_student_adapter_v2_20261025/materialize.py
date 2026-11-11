"""Actual provider late binding under Root's owned GPU preparation allocation."""
import os
from pathlib import Path
from . import protocol as p,authorization

def draft(plan_ref,cell_id,profile_ref,provider_refs,storage_ref,output,report,natural_evidence,registration_ref):
 from .launch import require_metadata_registration
 require_metadata_registration(registration_ref)
 plan=p.checked(plan_ref)
 p.require(plan['status']=='PREPARED_LATE_BOUND_STUDENT_CANDIDATES_NOT_RELEASE_AUTHORITY'
     and plan['adapter_sources']==p.sources(),'Exact prepared execution-consumer plan required')
 cells=[v for v in plan['cells'] if v['id']==cell_id];p.require(len(cells)==1,'Unknown/duplicate finite cell')
 cell=cells[0];profile=p.native_config.validate_profile(p.checked(profile_ref))
 p.require(profile['site']=='babel' and profile['partition']=='general' and profile['gpu_type']=='L40S',
     'Only declared Babel General production scope')
 p.require(p.n.within(output,profile['output_root']) and p.n.within(report,profile['registry_root']),
     'Current approved owned output/registry only')
 required={'pool','selection','eligibility','environment','assets','ownership'}
 p.require(set(provider_refs)==required and provider_refs['pool']==cell['providers']['pool']
     and provider_refs['eligibility']==cell['providers']['eligibility'] and provider_refs['selection']==cell['selection'],
     'Provider mismatch or competing selection generation')
 storage=p.checked(storage_ref)
 p.require(storage['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and storage['cell_id']==cell_id
     and storage['output']==str(output) and storage['checkpoint_transaction_bytes']==p.CAPSULE_BYTES
     and storage['retained_accepted_states']==2 and storage['allow_new_intermediate_retirement'] is True,
     'Actual Root collective reservation/owned resume retention required')
 cfg=dict(schema='native-selection-run-v1',id=cell_id,owner='Root',role='student',official_commit=p.n.PIN,
     method=cell['method'],fraction=cell['fraction'],training_seed=cell['training_seed'],selection_seed=1234,
     sources=p.n.sources(),recipe=p.native_config.RECIPE,site_profile=profile_ref,
     allocation=dict(gpus=1,cpus=8,memory_mb=65536,minutes=55,reserve_seconds=300,microbatch=4,loader_workers=0),
     checkpoint_seconds=1200,output=str(output),report=str(report),providers=dict(provider_refs),
     storage_reservation=storage_ref,review=None,replication_block=dict(replicate_site='babel'),
     execution_adapter=dict(package=p.PACKAGE,sources=p.sources(),runtime=p.RUNTIME,natural_epoch=True,
         ordinary_native_launch_forbidden=True),student_candidate_plan=plan_ref,
     expected_horizon=cell['expected_horizon'])
 authorization.natural_evidence(natural_evidence,cfg)
 gate=p.seal(Path(report)/'RUNTIME_GATE_CONTRACT.json',p.contract(cfg,natural_evidence))
 cfg['providers']['qualification']=gate
 return p.seal(Path(report)/'CONFIG_DRAFT.json',cfg)
