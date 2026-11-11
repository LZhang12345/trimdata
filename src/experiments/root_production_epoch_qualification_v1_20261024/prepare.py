"""Metadata-only pair draft. Exact reviews remain genuine external inputs."""
import argparse,copy,os,pwd
from pathlib import Path
from . import protocol as p
from .metadata import MetadataBindings

def draft(request,*,load=p.checked,save=p.seal):
 site=request['site'];seed=request['seed'];identifier=p.identifier(site,seed);spec=p.specification(site)
 p.require(request['owner']=='Root' and request['status']=='REGISTERED_BEFORE_EXECUTION' and request['driver_sources']==p.sources(),'Exact registered metadata request')
 held=load(request['held_binding']);intent=load(held['intent']);ack=load(held['ack']);job=held['job_id']
 p.require(held['owner']=='Root' and held['spool_reconciled'] is True and held['driver_sources']==p.sources() and intent['status']=='INTENT_BEFORE_SUBMISSION' and intent['driver_sources']==p.sources() and ack['job_id']==job and ack['intent']==held['intent'],'Actual held intent/ACK/spool source binding')
 source_review=load(request['source_review']);p.require(source_review['status']=='PASS_INDEPENDENT_DECLARED_EPOCH_RUNTIME_SOURCE' and source_review['driver_sources']==p.sources() and source_review['reviewer']!=p.AUTHOR_IDENTITY,'Genuine reviewed adapter source')
 anchor=load(request['anchor_study']);base=load(anchor['native_configurations']['reference']);p.require(anchor['native_target']['source_packet']['sha256']==p.TARGET and base['sources']==p.n.sources(),'Actual unchanged Native7 source anchor')
 profile=load(request['site_profile']);p.require(pwd.getpwuid(os.getuid()).pw_name==profile['slurm_user'],'Actual site metadata owner before writes');p.require(profile['site']==site and profile['code_root']==str(p.ROOT),'Approved installed site namespace');root=Path(profile['registry_root'])/'engineering'/identifier
 fixture=load(request['fixture']);pool=load(fixture['pool']);selection=load(fixture['selection']);eligibility=load(request['eligibility'])
 p.require(pool['count']==spec['count'] and pool['engineering_only'] is True and selection['pool']==fixture['pool'] and selection['count']==spec['count'] and selection['method']=='full' and selection['fraction']=='1' and selection['record_ids']==pool['record_ids'] and all(selection['records_json'][k]==pool['records_json'][k] for k in ('bytes','sha256')),'Exact complete engineering fixture/selection; path-only byte-identical selected copy allowed')
 p.require(eligibility['status']=='PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY' and eligibility['pool']==fixture['pool'] and eligibility['protected_QA_image_RGB_overlap_count']==eligibility['missing_media_count']==0,'Genuine separately reviewed actual eligibility required, no DRAFT promotion here')
 authorization=load(request['storage_authorization']);p.require(authorization['owner']=='Root' and authorization['study_id']==identifier and authorization['reserved_bytes']>=p.MIN_COLLECTIVE and authorization['capsules']==3 and authorization['capsule_cap_bytes']==p.CAPSULE_CAP and authorization['transient_reserve_bytes']==p.TRANSIENT_RESERVE,'Actual Root reservation includes three hardcapped5GiB capsules plus one6GiB transient; no pruning')
 plan=copy.deepcopy(load(base['plan']));plan.update(id=identifier,candidate_cells=[],bridge_cells=[],reference_cells=[],qualification_cells=[dict(id=identifier+'-'+arm,site=site,role='qualify',method='full',fraction='1',training_seed=seed,selection_seed=1234,gpus=1,microbatch=4,global_batch=128,qualification_max_steps=spec['total_steps'],engineering_only=True,track='engineering') for arm in ('reference','resume')]);plan_ref=save(root/'PLAN.json',plan)
 transport_ref=None
 if site=='aces':
  transport=copy.deepcopy(load(base['providers']['media_transport']));claim=copy.deepcopy(load(transport['ownership']));claim.update(job_id=job,logical_pool=fixture['pool'],records_json=selection['records_json']);transport['ownership']=save(root/'MEDIA_TRANSPORT_CLAIM.json',claim);transport_ref=save(root/'MEDIA_TRANSPORT.json',transport)
 configs={}
 for arm in ('reference','resume'):
  cfg=copy.deepcopy(base);output=str(Path(profile['output_root'])/(identifier+'-'+arm));owner=save(root/arm/'OWNERSHIP.json',dict(status='ACTIVE_EXCLUSIVE_NATIVE_PLAN_CELL_CLAIM',owner='Root',plan=plan_ref,cell_id=identifier+'-'+arm,pool=fixture['pool'],replicate_site=site,site_profile=request['site_profile'],actual_held_job_id=job,qualification_study=identifier,qualification_arm=arm,submission_intent=held['intent']))
  storage=save(root/arm/'STORAGE.json',dict(status='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION',owner='Root',cell_id=identifier+'-'+arm,output=output,retained_accepted_states=1 if arm=='reference' else 2,allow_new_intermediate_retirement=False,checkpoint_transaction_bytes=p.CAPSULE_CAP,transient_reserve_bytes=p.TRANSIENT_RESERVE,collective_reservation=request['storage_authorization'],actual_storage=authorization['actual_storage'],total_collective_reserved_bytes=authorization['reserved_bytes']))
  cfg.update(id=identifier+'-'+arm,output=output,report=str(root/arm),training_seed=seed,selection_seed=1234,site_profile=request['site_profile'],plan=plan_ref,storage_reservation=storage,qualification_max_steps=spec['total_steps'],allocation={k:spec[k] for k in ('gpus','cpus','memory_mb','minutes','reserve_seconds','microbatch','loader_workers')},execution_adapter=dict(package=p.PACKAGE,sources=p.sources(),specification=spec),replication_block=dict(replicate_site=site,registered_purpose='declared-production-epoch-runtime-qualification'),review=None)
  cfg.pop('readiness',None);cfg['providers'].pop('runtime_images',None);cfg['providers'].update(pool=fixture['pool'],selection=fixture['selection'],eligibility=request['eligibility'],ownership=owner)
  if transport_ref:cfg['providers']['media_transport']=transport_ref
  else:cfg['providers'].pop('media_transport',None)
  configs[arm]=save(root/arm/'CONFIG_DRAFT.json',cfg)
 study=dict(schema='declared-production-epoch-qualification-v1',id=identifier,owner='Root',site=site,seed=seed,cell='D8',job_id=job,held_binding=request['held_binding'],driver_sources=p.sources(),native_target=anchor['native_target'],native_configurations=configs,anchor_study=request['anchor_study'],result_basis=request['result_basis'],specification=spec,total_steps=spec['total_steps'],interrupt_step=4,numeric_tolerance=p.TOLERANCE,collective_reserved_bytes=authorization['reserved_bytes'],report=str(root/'actual'),minimum_launch_seconds=1000 if site=='babel' else 600,minimum_phase_seconds=dict(reference=900 if site=='babel' else 420,interrupted=450 if site=='babel' else 240,resumed=600 if site=='babel' else 300),review=None)
 return save(root/'STUDY_DRAFT.json',study)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--request',required=True);a.add_argument('--metadata-bindings',required=True);v=a.parse_args();print(p.canonical(draft(p.read(v.request),load=MetadataBindings(p.read(v.metadata_bindings)))).decode())
