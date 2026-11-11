"""Interleave real drafts with a separately pinned independent review module.

No author PASS, NN, scheduler release or checkpoint mutation is performed.
"""
import argparse,hashlib,importlib,os,resource,time
from pathlib import Path
from experiments.root_s1_shared_student_v1_20261025 import protocol as p,shared_runtime as sr,authorization
from experiments.s1_fulltrajectory_held_cohort_v2_20261025 import protocol as held
from experiments.root_s1_telemetry_physical_v1_20261026 import route as physical
from experiments.root_production_student_adapter_v2_20261025.launch import require_metadata_registration

ROOT=p.ROOT;HERE=Path(__file__).resolve().parent
HOME=Path('/home/luyangz/data_selection_multimodal');CONTROL=HOME/'collaboration/root-babel-s1-closure-phase3-20261026'
REVIEWER='experiments.root_s1_phase3_independent_review_v1_20261026.review'
AUTHOR='/root/cache_alias_finish';REVIEW='PASS_INDEPENDENT_EXACT_SIX_S1_PHASE3_METADATA_V1'


def sources():
    own={str(f.relative_to(ROOT)):p.sha(f) for f in HERE.iterdir() if f.suffix in ('.py','.sbatch') and not f.name.startswith('test_')}
    return dict(sorted({**physical.sources(),**own}.items()))


def literal(path):
    path=Path(path);p.require(path.is_absolute() and '..' not in path.parts and not any(v.is_symlink() for v in (path,*path.parents)),'Literal owned closure metadata route')
    return path


def reviewer(cfg):
    p.require(cfg['reviewer_module']==REVIEWER,'Only the separately authored independently qualified review module')
    module=importlib.import_module(REVIEWER)
    path=Path(module.__file__).resolve()
    p.require(path==ROOT/Path(REVIEWER.replace('.','/')).with_suffix('.py')
        and p.desc(path)==cfg['reviewer_module_source'] and module.sources()==cfg['reviewer_sources'], 'Exact separately reviewed callback source closure')
    for rel,sha in cfg['reviewer_sources'].items():p.require(p.sha(ROOT/rel)==sha,'Independent reviewer source changed')
    external=p.checked(cfg['reviewer_source_review']);packet=p.checked(cfg['reviewer_source_packet'])
    p.require(external['status']=='PASS_NONAUTHOR_ROOT_INDEPENDENT_PHASE3_VALIDATOR_SOURCE_REVIEW_V1' and external['independent'] is True
        and external['reviewer']=='/root' and external['author']=='/root/repair_independent_review'
        and external['reviewer_sources']==cfg['reviewer_sources'] and external['source_packet']==cfg['reviewer_source_packet']
        and packet['module_source']==cfg['reviewer_module_source'] and packet['reviewer_sources']==cfg['reviewer_sources'],
        'Callback author cannot self-qualify its own decision module')
    return module


def decision(cfg,module,stage,draft,context):
    reviewer(cfg)
    returned=module.review(stage,draft,context)
    value=p.checked(returned)
    p.require(value.get('reviewer') not in (None,'','Root','/root',AUTHOR),'Author/Root cannot issue independent PASS through callback')
    reviewer(cfg)
    return returned,value


def consume_review(value,status,identity_field,identity):
    p.require(value['status']==status and value[identity_field]==identity,'Actual independent decision differs from exact draft identity')


def compose_gates(evidence,row,cfg,folder):
    common=dict(configuration=cfg,selected_count=66076,selection=p.checked(cfg)['providers']['selection'],
        selected_metadata=evidence['selected_metadata'],final_selection_receipt=p.checked(p.checked(row['binding'])['request'])['selection_receipt'],
        selected_image_integrity=evidence['selected_image_integrity'],selected_image_review=evidence['selected_image_review'],
        phase2_result=evidence['phase2_result'],actual_natural_evidence=evidence['natural_evidence'])
    inputs=p.seal(folder/'ACTUAL_FULL_SOURCE_INPUT_COMPOSITION.json',dict(status='ROOT_COMPOSED_ACTUAL_FROZEN_SOURCE_AND_IMAGE_INPUT_EVIDENCE_V1',
        **common,new_NN_execution_claimed=False))
    pilot=p.seal(folder/'SHARED_CANARY_SELECTED_DATA_COMPOSITION.json',dict(status='ROOT_COMPOSED_SHARED_CANARY_AND_SELECTED_DATA_NOT_OWN_NN_PILOT',
        **common,actual_v4_canary=evidence['actual_v4_canary'],actual_v4_canary_review=evidence['actual_v4_canary_review'],
        logical_numerical_selection_bridge=evidence['source_bridge'],new_selected_full_NN_pilot_executed=False,full_NN_training_PASS=False,
        scope='Qualified shared runtime plus exact66076 own selected records and encoded/RGB media; never relabel actual V4 canary as S1'))
    return dict(actual_matching_natural_runtime=evidence['natural_evidence'],actual_full_source_student_inputs=inputs,
        actual_full_selected_canonical_pilot=pilot,reviewed_longwall_runtime_entrypoint=evidence['physical_extension_review'],
        actual_collective_storage_reservation=row['storage'],reviewed_custom_method_selection_route_in_target=evidence['source_bridge'])


def close_cell(cfg,module,row,evidence,actual,registration):
    cell=row['cell_id'];binding=row['binding'];b=p.checked(binding);request,science,_=held.reviewed(b['request'])
    draftref=row['configuration_draft'];draft=p.checked(draftref)
    p.require(cell==request['cell_id']==draft['id'] and draft['review'] is None and draft['sources']==p.n.sources()
        and draft['expected_horizon']==p.horizon(66076),'Only genuine actual phase2 frozen config drafts')
    report=literal(draft['report']);folder=report/'closure-phase3-v1';p.require(not folder.exists(),'Existing partial closure requires reconciliation, not overwrite/retry')
    folder.mkdir(parents=True,exist_ok=False)
    context=dict(owner='Root',cell_id=cell,binding=binding,phase2_cell=row,phase2_result=evidence['phase2_result'],configuration=draftref,
        registration=registration,actual_metadata=actual,output_directory=str(CONTROL/'reviews'/cell),producer_sources=sources(),
        reviewer_sources=cfg['reviewer_sources'],physical_sources=physical.sources(),**{k:v for k,v in evidence.items() if k!='phase2_result'})
    p.seal(folder/'CONFIG_REVIEW_REQUEST.json',dict(configuration=draftref,config_identity=p.n.run_identity(draft),sources=draft['sources'],
        physical_binding=binding,actual_phase2_result=evidence['phase2_result'],no_new_NN_claim=True))
    cref,cv=decision(cfg,module,'native_config',draftref,context)
    consume_review(cv,'PASS_INDEPENDENT_EXACT_SOURCE_AND_CONFIG','config_identity',p.n.run_identity(draft))
    p.require(cv['sources']==draft['sources'],'Exact independently reviewed native91 sources')
    final=dict(draft,review=cref);configuration=p.seal(report/'CONFIG_FINAL.json',final)
    p.require(p.n.run_identity(final)==p.n.run_identity(draft),'No scientific identity change while attaching review')
    context.update(configuration=configuration,configuration_review=cref)
    certificate=sr.draft(configuration,binding,evidence['source_bridge'],actual_v4_canary=evidence['actual_v4_canary'],actual_v4_canary_review=evidence['actual_v4_canary_review'])
    certdraft=p.seal(folder/'RUNTIME_CERTIFICATE_DRAFT.json',certificate)
    certref,cert=decision(cfg,module,'runtime_certificate',certdraft,context)
    sr.verify(certref,final,configuration,binding,'production');context['runtime_certificate']=certref
    permit=dict(schema='babel-s1-shared-runtime-outer-authorization-v1',owner='Root',configuration=configuration,config_identity=p.n.run_identity(final),
        adapter_sources=p.sources(),runtime=p.RUNTIME,phase='production',stop_after_global_step=None,expected_horizon=p.horizon(66076),
        runtime_certificate=certref,physical_binding=binding,review=None)
    permitdraft=p.seal(folder/'PRODUCTION_AUTHORIZATION_DRAFT.json',permit)
    pref,pv=decision(cfg,module,'production_authorization',permitdraft,context)
    consume_review(pv,'PASS_INDEPENDENT_EXACT_BABEL_S1_SHARED_EXECUTION_AUTHORIZATION_V1','authorization_identity',p.key(p.bare(permit)))
    permitref=p.seal(report/'PRODUCTION_AUTHORIZATION.json',dict(permit,review=pref))
    authorization.verify(final,{k:p.checked(v) for k,v in final['providers'].items()},p.checked(final['site_profile']),permitref)
    context.update(authorization=permitref,authorization_review=pref)
    gates=compose_gates(evidence,row,configuration,folder)
    gates['exact_config_and_outer_permit_reviews']=p.seal(folder/'ACTUAL_CONFIG_PERMIT_REVIEW_COMPOSITION.json',
        dict(status='ROOT_BOUND_ACTUAL_CONFIG_AND_PRODUCTION_AUTHORIZATION_REVIEWS',configuration=configuration,configuration_review=cref,authorization=permitref,authorization_review=pref))
    p.require(set(gates)==set(held.GATES),'Exactly seven actual evidence gates; no invented future gate')
    context['gates']=gates
    activation=dict(schema='s1-fulltrajectory-actual-activation-v2',owner='Root',binding=binding,configuration=configuration,scientific_design=science,
        held_sources=held.sources(),target_module=physical.PACKAGE+'.driver',target_callable='run_conditional_student',target_sources=physical.sources(),
        gates=gates,logical_input_registration=dict(original_pool=science['logical_pool'],original_selection=science['selection'],actual_pool=final['providers']['pool'],actual_selection=final['providers']['selection']),
        physical_extension_policy=evidence['physical_extension_policy'],physical_extension_review=evidence['physical_extension_review'],review=None)
    adraft=p.seal(folder/'ACTIVATION_DRAFT.json',activation)
    areview,av=decision(cfg,module,'activation',adraft,context)
    consume_review(av,held.ACTIVATION_REVIEW,'activation_identity',p.key(held.bare(activation)))
    aref=p.seal(Path(request['control_root'])/'ACTIVATION.json',dict(activation,review=areview))
    held.activation(binding,aref);context.update(activation=aref,activation_review=areview)
    execution=dict(schema='babel-s1-shared-runtime-outer-execution-v1',owner='Root',binding=binding,activation=aref,configuration=configuration,
        config_identity=p.n.run_identity(final),actual_restart=0,sources=p.sources(),mode='production',max_physical_attempts=516,accepted_history=[],
        predecessor_transition=None,runtime_certificate=certref,authorization=permitref,physical_extension_policy=evidence['physical_extension_policy'],
        physical_extension_review=evidence['physical_extension_review'],physical_sources=physical.sources(),physical_source_packet=evidence['physical_source_packet'],review=None)
    edraft=p.seal(folder/'OUTER_EXECUTION_DRAFT.json',execution)
    ereview,ev=decision(cfg,module,'outer_execution',edraft,context)
    policy_identity=p.key({k:v for k,v in execution.items() if k not in ('review','actual_restart','accepted_history','predecessor_transition')})
    consume_review(ev,'PASS_INDEPENDENT_EXACT_BABEL_S1_SHARED_OUTER_EXECUTION_V1','execution_policy_identity',policy_identity)
    eref=p.seal(report/'ROOT_LONG/requests/restart-0.json',dict(execution,review=ereview))
    physical.execution(binding,aref,dict(restart=0))
    result=dict(cell_id=cell,job_id=b['job_id'],binding=binding,configuration=configuration,configuration_review=cref,runtime_certificate=certref,
        authorization=permitref,authorization_review=pref,activation=aref,activation_review=areview,execution=eref,execution_review=ereview,
        scheduler_release_performed=False,new_NN_execution=False,actual_training_started=False)
    return p.seal(folder/'CLOSURE_METADATA_RESULT.json',result)


def execute(config_ref,review_ref,registration_ref):
    os.environ['TZ']='America/New_York';time.tzset();cfg=p.checked(config_ref);review=p.checked(review_ref)
    p.require(cfg['schema']=='root-babel-six-S1-phase3-closure-config-v1' and cfg['owner']=='Root' and cfg['producer_sources']==sources()
        and cfg['physical_sources']==physical.sources() and cfg['logical_sources']==p.sources(),'Exact source-qualified six closure producer')
    p.require(review['status']==REVIEW and review['configuration']==config_ref and review['producer_sources']==sources()
        and review['reviewer_sources']==cfg['reviewer_sources'] and review['independent'] is True
        and review['reviewer'] not in (None,'','Root','/root',AUTHOR),'Exact independent producer/reviewer/config source qualification')
    actual=require_metadata_registration(registration_ref);p.require(os.getuid()==2708899 and ROOT==HOME/'native_campaign_v3_20261023/src','Actual canonical Root Babel metadata actor')
    module=reviewer(cfg);evidence=cfg['evidence'];previous=p.checked(evidence['phase2_result'])
    p.require(previous['status']=='SIX_MEASURED_CONFIG_DRAFTS_NOT_TRAINING_ACTIVATION'
        and [r['cell_id'] for r in previous['cells']]==list(held.OLD_JOBS),'Authentic six phase2 outputs only')
    for ref in evidence.values():p.checked(ref)
    out=literal(cfg['result']);p.require(CONTROL in out.parents and not out.exists(),'New immutable phase3 result')
    start=time.monotonic();u=resource.getrusage(resource.RUSAGE_SELF);done=[]
    p.seal(out.parent/'START.json',dict(configuration=config_ref,actual=actual,registration=registration_ref,producer_sources=sources(),reviewer_sources=cfg['reviewer_sources']))
    try:
        for row in previous['cells']:done.append(close_cell(cfg,module,row,evidence,actual,registration_ref))
        p.require(cfg['producer_sources']==sources() and cfg['reviewer_sources']==reviewer(cfg).sources(),'Executed producer/reviewer source drift')
        after=resource.getrusage(resource.RUSAGE_SELF)
        return p.seal(out,dict(status='SIX_ACTUAL_REVIEWED_METADATA_CLOSURES_NOT_SCHEDULER_RELEASE_OR_TRAINING',configuration=config_ref,
            producer_sources=sources(),reviewer_sources=cfg['reviewer_sources'],actual=actual,registration=registration_ref,cells=done,NN=0,scheduler_mutations=False,
            cost=dict(wall_seconds=time.monotonic()-start,CPU_user_seconds=after.ru_utime-u.ru_utime,CPU_system_seconds=after.ru_stime-u.ru_stime,
                      actual_allocation_cost_requires_sacct=True)))
    except Exception as error:
        p.seal(out.parent/'FAILED.json',dict(configuration=config_ref,actual=actual,completed_cells=done,error=type(error).__name__,reason=str(error),
            preserve_all_partial_drafts_reviews_and_bindings=True,automatic_retry=False));raise


if __name__=='__main__':
    a=argparse.ArgumentParser()
    for name in ('configuration','review','registration'):a.add_argument('--'+name,type=Path,required=True)
    v=a.parse_args();print(p.canonical(execute(p.desc(v.configuration),p.desc(v.review),p.desc(v.registration))).decode())
