"""Derive a fresh own S1 configuration from exact paid science; no acquisition."""
from pathlib import Path
from . import protocol as p,authorization
from experiments.root_fulltrajectory_student_v4_20261025 import shared_runtime as qualified
from experiments.s1_fulltrajectory_held_cohort_v2_20261025 import protocol as held
from experiments.root_production_student_adapter_v2_20261025.launch import require_metadata_registration


def draft(binding_ref,profile_ref,ownership_ref,storage_ref,compatibility_ref,output,report,registration_ref):
 require_metadata_registration(registration_ref)
 b=p.checked(binding_ref);request,science,_=held.reviewed(b['request']);anchor,_,_=qualified.anchor()
 profile=p.native_config.validate_profile(p.checked(profile_ref))
 p.require(profile['output_root']==p.CHECKPOINT_ROOT and p.n.within(output,p.CHECKPOINT_ROOT)
     and Path(output)!=Path(p.CHECKPOINT_ROOT) and p.n.within(report,profile['registry_root']), 'Measured approved new output/report only')
 p.require(science['logical_pool']==anchor['providers']['pool'] and science['logical_eligibility']==anchor['providers']['eligibility'],
     'Exact same qualified broad logical pool/eligibility; no nominal pool substitution')
 providers={k:anchor['providers'][k] for k in ('pool','eligibility','assets','environment')}
 providers.update(selection=science['selection'],ownership=ownership_ref)
 storage=p.checked(storage_ref)
 p.require(storage['status']=='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION' and storage['cell_id']==request['cell_id']
     and storage['output']==str(output) and storage['checkpoint_transaction_bytes']==p.CAPSULE_BYTES
     and storage['retained_accepted_states']==2 and storage['allow_new_intermediate_retirement'] is True,
     'Actual measured independent new S1 storage receipt')
 cfg=dict(schema='native-selection-run-v1',id=request['cell_id'],owner='Root',role='student',official_commit=p.n.PIN,
     method=science['method'],fraction='0.10',training_seed=science['training_seed'],selection_seed=1234,
     sources=p.n.sources(),recipe=p.native_config.RECIPE,site_profile=profile_ref,
     allocation=dict(gpus=1,cpus=8,memory_mb=65536,minutes=720,reserve_seconds=300,microbatch=4,loader_workers=0),
     checkpoint_seconds=1200,output=str(output),report=str(report),providers=providers,storage_reservation=storage_ref,
     review=None,replication_block=dict(replicate_site='babel'),
     execution_adapter=dict(package=p.PACKAGE,sources=p.sources(),runtime=p.RUNTIME,natural_epoch=True,ordinary_native_launch_forbidden=True),
     student_candidate_plan=request['original_request'],expected_horizon=p.horizon(66076),runtime_compatibility=compatibility_ref)
 p.supported(cfg)
 natural=p.checked(anchor['providers']['qualification'])['qualification_evidence']
 authorization.natural_evidence(natural,cfg,compatibility_ref=compatibility_ref)
 cfg['providers']['qualification']=p.seal(Path(report)/'RUNTIME_GATE_CONTRACT.json',p.contract(cfg,natural))
 return p.seal(Path(report)/'CONFIG_DRAFT.json',cfg)
