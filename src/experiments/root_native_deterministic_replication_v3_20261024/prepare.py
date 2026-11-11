"""Metadata-only Root design/held-config binding; never invents native READY."""
import argparse,copy,subprocess
from pathlib import Path
from . import protocol as p

def registration(ref):
    request=p.checked(ref);p.require(request['status']=='REGISTERED_BEFORE_EXECUTION' and request['owner']=='Root' and request['driver_sources']==p.sources(),'Exact Root registration/source required');p.result_basis(request.get('result_basis'));return request

def design(request_ref):
    request=registration(request_ref);basis=p.result_basis(request.get('result_basis'));cell=request['cell'];p.completed_trace(request['completed_trace'],cell)
    shell=dict(result_basis=basis,id='babel-native7-deterministic-replication-v3-seed43-'+cell,cell=cell,anchor_study=p.ANCHOR,paid_anchor_readout=p.PAID_READOUT,completed_trace=request['completed_trace'])
    old,original,modules=p.anchor(dict(shell,native_target=p.checked(p.ANCHOR)['native_target']))
    profile=modules['config'].validate_profile(p.checked(original['site_profile']));root=p.literal(Path(profile['registry_root'])/'engineering'/shell['id']);root.mkdir(parents=True,exist_ok=True)
    plan=p.seal(root/'PLAN.json',dict(schema='native-campaign-plan-v1',owner='Root',id=shell['id'],recipe=original['recipe'],official_commit=original['official_commit'],sources=original['sources'],qualification_cells=[dict(id=shell['id']+'-'+arm,method='full',fraction='1',training_seed=43,selection_seed=1234,site='babel',role='qualify',gpus=1,microbatch=4,global_batch=128,qualification_max_steps=8) for arm in p.CELLS[cell]['arms']],candidate_cells=[],bridge_cells=[],reference_cells=[],no_automatic_refill=True))
    configs={}
    for arm in p.CELLS[cell]['arms']:
        cfg=copy.deepcopy(original);cfg.update(training_seed=43,id=shell['id']+'-'+arm,plan=plan,allocation=copy.deepcopy(p.ALLOCATION),report=str(root/arm),output=profile['output_root']+'/'+shell['id']+'-'+arm,review=None,storage_reservation=None,replication_block=dict(replicate_site='babel',registered_purpose='finite-'+cell+'-'+arm),diagnostic_runtime=dict(policy=p.CELLS[cell]['runtime'],math_changes=cell=='D8',objective_recipe_unchanged=True,runtime_requires_separate_qualification=True));cfg.pop('readiness',None);cfg['providers']['ownership']=None;configs[arm]=cfg
    shell.update(native_target=old['native_target'],report=str(root/'actual'))
    value=p.projection(shell,configs);draft=p.seal(root/'HELD_DESIGN_DRAFT.json',value)
    return p.seal(root/'DESIGN_REVIEW_REQUEST.json',dict(status='PENDING_GENUINE_INDEPENDENT_DESIGN_REVIEW',registered_request=request_ref,design=draft,no_jobs_submitted=True))

def freeze(request_ref):
    request=registration(request_ref);design=p.validate_design(p.checked(request['design']));held=p.checked(request['held_binding']);job=held['job_id'];cell=design['cell'];root=p.literal(Path(design['report']).parent)
    p.require(held['design']==request['design'] and held['driver_sources']==p.sources() and held['spool_reconciled'] is True,'Actual held binding required')
    auth=request['Root_collective_authorization'];p.require(auth==dict(owner='Root',study_id=design['id'],total_reserved_bytes=p.CELLS[cell]['reserved_bytes'],maximum_new_capsules=p.CELLS[cell]['capsules'],collective_consumers_reconciled=True),'Root must reconcile existing states and authorize collective measured storage')
    prior=p.checked(p.PRIOR_STORAGE);p.require(prior['actual_job_id']=='10913252' and prior['available_bytes']>=p.CELLS[cell]['reserved_bytes'],'Authentic prior snapshot insufficient; do not relabel as current')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30)
    from .admit import held_fields
    profile=p.checked(design['configurations']['reference']['site_profile']);held_fields(raw,held,profile)
    configs={}
    for arm,cfg in design['configurations'].items():
        cfg=copy.deepcopy(cfg)
        cfg['providers']['ownership']=p.seal(root/arm/'OWNERSHIP.json',dict(status='ACTIVE_EXCLUSIVE_NATIVE_PLAN_CELL_CLAIM',owner='Root',cell_id=cfg['id'],plan=cfg['plan'],pool=cfg['providers']['pool'],site_profile=cfg['site_profile'],replicate_site='babel',qualification_study=design['id'],qualification_arm=arm,actual_held_job_id=job,held_binding=request['held_binding'],purpose='INTENTIONAL_RESULT_GATED_FOUNDATION_CONTROL'))
        cfg['storage_reservation']=p.seal(root/arm/'STORAGE.json',dict(status='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION',owner='Root',cell_id=cfg['id'],output=cfg['output'],retained_accepted_states=2 if arm=='resume' else 1,allow_new_intermediate_retirement=False,checkpoint_transaction_bytes=8*p.GIB,total_pair_reserved_bytes=p.CELLS[cell]['reserved_bytes'],Root_collective_authorization=auth,prior_actual_measurement=p.PRIOR_STORAGE,measurement_timing='PRIOR_10913252_SNAPSHOT_NOT_CURRENT_ALLOCATION',fresh_recheck_required_before_neural=True))
        cfg['review']=None;configs[arm]=p.seal(root/arm/'CONFIG_DRAFT.json',cfg)
    study=dict(schema='native-deterministic-replication-study-v3',owner='Root',id=design['id'],cell=cell,driver_sources=p.sources(),design=request['design'],held_binding=request['held_binding'],job_id=job,native_target=design['native_target'],result_basis=design['result_basis'],anchor_study=p.ANCHOR,paid_anchor_readout=p.PAID_READOUT,completed_trace=design['completed_trace'],native_configurations=configs,Root_collective_authorization=auth,total_steps=8,interrupt_step=4,numeric_tolerance=p.TOLERANCE,runtime=design['runtime'],report=design['report'],minimum_launch_seconds=1200,minimum_phase_seconds=dict(reference=700,interrupted=420,resumed=700))
    return p.seal(root/'CONFIG_REVIEW_REQUEST.json',dict(status='PENDING_GENUINE_EXACT_CONFIG_AND_STUDY_REVIEW',configurations=configs,config_identities={arm:p.n.run_identity(p.checked(ref)) for arm,ref in configs.items()},study_draft=p.seal(root/'STUDY_DRAFT.json',study),actual_held_scheduler=raw,registered_request=request_ref))

def bind(request_ref):
    request=registration(request_ref);study=p.checked(request['study_draft']);refs={}
    for arm,ref in study['native_configurations'].items():
        cfg=p.checked(ref);review_ref=request['configuration_reviews'][arm];review=p.checked(review_ref)
        p.require(review['status']=='PASS_INDEPENDENT_EXACT_SOURCE_AND_CONFIG' and review['sources']==cfg['sources'] and review['config_identity']==p.n.run_identity(cfg) and p.independent(review['reviewer']),'Genuine independent exact review required')
        cfg['review']=review_ref;refs[arm]=p.seal(Path(ref['path']).with_name('CONFIG.json'),cfg)
    study['native_configurations']=refs;p.validate_study(study,gpu=False,reviewed=False)
    return p.seal(Path(request['study_draft']['path']).with_name('STUDY_CONFIGS_REVIEWED_DRAFT.json'),study)

def bind_study(request_ref):
    request=registration(request_ref);study=p.checked(request['study_draft']);study['review']=request['study_review'];p.validate_study(study,gpu=False)
    return p.seal(Path(request['study_draft']['path']).with_name('STUDY.json'),study)

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('design','freeze','bind','bind-study'));a.add_argument('--request',required=True);v=a.parse_args();result=globals()[v.action.replace('-','_')](p.desc(v.request));print(p.canonical(result).decode())
if __name__=='__main__':main()
