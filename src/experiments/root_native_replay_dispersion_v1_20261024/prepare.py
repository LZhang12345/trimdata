"""Finite design/job-bound metadata only. No scheduler mutation or asset download."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import subprocess
from . import protocol as p

def shell(request,configs):
    original=p.checked(request['anchor_study'])
    profile=p.checked(p.checked(original['native_configurations']['reference'])['site_profile'])
    return dict(schema='native7-replay-study-v1',id=p.ID,owner='Root',purpose='FRESH_PROCESS_DISPERSION_ONLY',
        cells=p.CELLS,total_scheduler_steps=8,numeric_tolerance=p.TOLERANCE,driver_sources=p.sources(),
        native_target=original['native_target'],anchor_study=request['anchor_study'],
        native_configurations=configs,report=profile['registry_root']+'/engineering/'+p.ID+'/actual')

def design(request_ref):
    request=p.checked(request_ref)
    p.require(request['status']=='REGISTERED_BEFORE_EXECUTION' and request['owner']=='Root'
        and request['schema']=='native7-replay-design-request-v1' and request['driver_sources']==p.sources(),
        'Exact registered Root request required')
    s=shell(request,{})
    old,original,modules=p.anchor(s)
    # Reuse the actual old qualified INPUTS, not a claimed native parity PASS.
    p.h.configs(old)
    profile=modules['config'].validate_profile(p.checked(original['site_profile']))
    root=p.literal(profile['registry_root']+'/engineering/'+p.ID);root.mkdir(parents=True,exist_ok=True)
    with (root/'PREPARE.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cells=[dict(id=p.ID+'-'+cell,method='full',fraction='1',training_seed=17,selection_seed=1234,
            site='aces',role='qualify',gpus=1,microbatch=4,global_batch=128,qualification_max_steps=8,
            track='engineering',engineering_only=True,minimum_actual_records=1024) for cell in p.CELLS]
        plan=p.seal(root/'PLAN.json',dict(schema='native-campaign-plan-v1',owner='Root',id=p.ID,
            recipe=original['recipe'],official_commit=original['official_commit'],sources=original['sources'],
            qualification_cells=cells,candidate_cells=[],bridge_cells=[],reference_cells=[],no_automatic_refill=True))
        transport=copy.deepcopy(p.checked(original['providers']['media_transport']));transport['ownership']=None
        transport_ref=p.seal(root/'design/TRANSPORT_TEMPLATE.json',transport)
        pair={};refs={}
        for cell in p.CELLS:
            cfg=copy.deepcopy(original)
            for name in ('review','storage_reservation'):cfg[name]=None
            cfg.pop('readiness',None);cfg['plan']=plan;cfg['id']=p.ID+'-'+cell
            cfg['output']=profile['output_root']+'/'+p.ID+'-'+cell
            cfg['report']=str(root/cell)
            cfg['replication_block']=dict(replicate_site='aces',registered_purpose='fresh-process-replay-'+cell)
            cfg['providers']['ownership']=None;cfg['providers']['media_transport']=transport_ref
            cfg['providers'].pop('runtime_images',None)
            pair[cell]=cfg;refs[cell]=p.seal(root/'design'/cell/'NON_RUNNABLE_CONFIG.json',cfg)
        s=shell(request,refs)
        draft=p.seal(root/'HELD_DESIGN_DRAFT.json',p.design_contract(s,pair))
        return p.seal(root/'DESIGN_REVIEW_REQUEST.json',dict(status='PENDING_EXACT_INDEPENDENT_HELD_REPLAY_DESIGN',
            owner='Root',registered_request=request_ref,design=draft,plan=plan,non_runnable_configurations=refs,
            maximum_new_final_capsules=3,reservation_bytes=24*p.GIB,new_jobs_submitted=False))

def finalize(request_ref):
    request=p.checked(request_ref)
    p.require(request['schema']=='native7-replay-finalization-request-v1' and request['owner']=='Root'
        and request['status']=='REGISTERED_BEFORE_EXECUTION' and request['driver_sources']==p.sources(),
        'Exact registered finalization request required')
    first=p.checked(request['design_request']);initial=p.checked(first['registered_request'])
    reviewed=p.validate_design(p.checked(request['reviewed_design']))
    p.require(p.bare(reviewed)==p.checked(first['design']),'Held design changed')
    base=shell(initial,{});old,original,modules=p.anchor(base)
    profile=modules['config'].validate_profile(p.checked(original['site_profile']))
    root=p.literal(profile['registry_root']+'/engineering/'+p.ID)
    p.require(request['conditional_storage_authorization']==dict(owner='Root',study_id=p.ID,
        total_reserved_bytes=24*p.GIB,maximum_new_final_capsules=3,collective_consumers_reconciled=True),
        'Root must reconcile existing consumers and separately authorize24GiB')
    from .admit import held_fields
    from experiments.root_h100_engineering_launch_v10_20261024.prepare import quota_headroom,transport_claim_value
    from experiments.root_native_media_transport_20261023 import transport as media
    with (root/'PREPARE.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        final=root/'CONFIG_REVIEW_REQUEST.json'
        if final.exists():
            p.require(p.read(final)['registered_request']==request_ref,'Partial finalization identity collision')
            return p.desc(final)
        quota=subprocess.run(['showquota'],capture_output=True,text=True,timeout=45,check=True)
        # Reuse conservative existing parser; its32GiB headroom requirement is
        # stricter than this finite24GiB reservation, not a private GPU cap.
        space=quota_headroom(quota.stdout,os.statvfs(p.literal(p.BASE)))
        measurement=p.seal(root/'ACTUAL_STORAGE.json',dict(status='ACTUAL_SCRATCH_QUOTA_AND_HELD_JOB_CENSUS',
            UTC=p.now(),registered_request=request_ref,showquota=quota.stdout,**space))
        collective=p.seal(root/'ROOT_STORAGE_AUTHORIZATION.json',dict(owner='Root',study_id=p.ID,
            status='AUTHORIZE_EXCLUSIVE_REPLAY_24GIB_THREE_CAPSULE_RESERVATION',
            reserved_bytes=24*p.GIB,maximum_new_final_capsules=3,checkpoint_transaction_bytes=8*p.GIB,
            measurement=measurement,registered_request=request_ref,collective_consumers_reconciled=True,
            preserves_all_original_engineering_capsules=True))
        drafts={};jobs={};bindings={}
        for cell in p.CELLS:
            binding=p.checked(request['held_bindings'][cell]);intent=p.checked(binding['intent'])
            job=binding['job_id'];jobs[cell]=job;bindings[cell]=request['held_bindings'][cell]
            p.require(binding['cell']==cell and binding['design']==request['reviewed_design']
                and binding['driver_sources']==p.sources() and binding['spool_reconciled'] is True,
                'Exact held replay binding required')
            raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30)
            held_fields(raw,job,cell,intent,profile)
            cfg=p.checked(first['non_runnable_configurations'][cell])
            p.require(not p.literal(cfg['output']).exists(),'New replay output already exists; do not resume a fresh control')
            tr=copy.deepcopy(p.checked(cfg['providers']['media_transport']))
            claim=p.seal(root/cell/'MEDIA_CLAIM.json',transport_claim_value(job,cfg['providers']['pool'],
                cfg['providers']['selection'],tr['locator_index']))
            tr['ownership']=claim;cfg['providers']['media_transport']=p.seal(root/cell/'MEDIA_TRANSPORT.json',tr)
            cfg['providers']['ownership']=p.seal(root/cell/'OWNERSHIP.json',dict(
                status='ACTIVE_EXCLUSIVE_NATIVE_PLAN_CELL_CLAIM',owner='Root',cell_id=cfg['id'],plan=cfg['plan'],
                pool=cfg['providers']['pool'],replicate_site='aces',site_profile=cfg['site_profile'],
                replay_study=p.ID,replay_cell=cell,actual_held_job_id=job,submission_intent=binding['intent'],
                purpose='INTENTIONAL_FRESH_PROCESS_REPLICATE_NOT_A_DUPLICATE_CONTROLLER'))
            cfg['storage_reservation']=p.seal(root/cell/'STORAGE.json',dict(
                status='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION',owner='Root',output=cfg['output'],cell_id=cfg['id'],
                checkpoint_transaction_bytes=8*p.GIB,retained_accepted_states=1,
                allow_new_intermediate_retirement=False,collective_reservation=collective,actual_storage=measurement))
            selected=p.checked(cfg['providers']['selection'])
            media._inputs(cfg['providers']['pool'],selected['records_json'],cfg['providers']['media_transport'],{'job_id':job})
            p.invariant_config(cfg,original,cell)
            drafts[cell]=p.seal(root/cell/'CONFIG_DRAFT.json',cfg)
        s=shell(initial,drafts);s.update(design=request['reviewed_design'],jobs=jobs,held_bindings=bindings)
        p.require(p.design_contract(s,{k:p.checked(v) for k,v in drafts.items()})==p.bare(reviewed),
            'Final job-bound inputs changed the reviewed design')
        return p.seal(final,dict(status='PENDING_EXACT_CONFIG_AND_FINAL_STUDY_REVIEW',owner='Root',
            registered_request=request_ref,configurations=drafts,study=p.seal(root/'STUDY_DRAFT.json',s),
            measurement=measurement,collective_reservation=collective,no_native_qualification_claimed=True))

def bind(request_ref):
    request=p.checked(request_ref);draft=p.checked(request['config_review_request'])
    p.require(request['owner']=='Root' and request['status']=='REGISTERED_BEFORE_EXECUTION','Root bind intent required')
    study=p.checked(draft['study']);configs={}
    for cell,ref in draft['configurations'].items():
        cfg=p.checked(ref);review=p.checked(request['configuration_reviews'][cell])
        p.require(review['status']=='PASS_INDEPENDENT_EXACT_SOURCE_AND_CONFIG'
            and review['sources']==cfg['sources'] and review['config_identity']==p.n.run_identity(cfg),
            'Exact independent config review required')
        cfg['review']=request['configuration_reviews'][cell]
        configs[cell]=p.seal(Path(ref['path']).with_name('CONFIG.json'),cfg)
    study['native_configurations']=configs
    p.validate(study,reviewed=False)
    return p.seal(Path(draft['study']['path']).with_name('STUDY_CONFIGS_REVIEWED_DRAFT.json'),study)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('design','finalize','bind'))
    parser.add_argument('--request',required=True);a=parser.parse_args()
    print(p.n.canonical(globals()[a.action](p.desc(a.request))).decode())

if __name__=='__main__':main()
