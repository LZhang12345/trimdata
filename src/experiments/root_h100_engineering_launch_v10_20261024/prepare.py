"""Two-phase H100 engineering preparation around the separately reviewed S6 driver.

The held design has deliberately non-runnable configs. Only after Root obtains
and reconciles an actual held job does finalization register job-bound transport,
measured storage and native claims. No scheduler mutation, downloads or deletion.
"""
import argparse
import copy
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import os
from pathlib import Path
import re
import socket
import subprocess

from experiments.root_native_campaign_v7_20261024 import common as c
from experiments.root_native_qualification_v10_20261024.preflight import source_equivalence_bridge
from experiments.root_native_campaign_v7_20261024.config import RECIPE, validate_profile
from experiments.root_native_campaign_v7_20261024 import selection
from experiments.root_h100_native_qualification_v10_20261024 import protocol as p

HERE=Path(__file__).resolve().parent
BASE='/scratch/user/u.lz341613/dsmm_native_campaign_20261023'
STUDY='aces-v7-engineering-parity-v10'
NATIVE_SHA='a704d9716719a9ad2a0cd0890c24fec35c15bd085b7c2c98fe38ea0c98961830'
ARMS=('reference','resume')
GIB=1<<30
ALLOCATION=dict(gpus=1,cpus=8,memory_mb=65536,minutes=55,reserve_seconds=600,microbatch=4,loader_workers=0)
REQUIRED=('native_source_packet','driver_source_packet','site_profile','pool','eligibility','environment','assets',
          'transport_template','input_rebind_provenance','source_pool_metadata','actual_prefix_proof','raw_prefix_proof_copy')


def sources():
    return {str(f.relative_to(c.ROOT)):c.sha(f) for f in sorted(HERE.glob('*.py')) if not f.name.startswith('test_')}


def literal(path,root=BASE):
    path=Path(path)
    c.require(path.is_absolute() and '..' not in path.parts and path.is_relative_to(Path(root)), 'Outside approved H100 namespace')
    for a in (path,*path.parents):c.require(not a.is_symlink(),'Symlink ancestor needs explicit review: '+str(a))
    return path


def diagnose(request):
    issues=[]
    for name in REQUIRED:
        d=request.get(name)
        if not d:
            issues.append(dict(input=name,problem='EXACT_DESCRIPTOR_NOT_SUPPLIED'));continue
        try:c.checked(d)
        except (ValueError,OSError,KeyError) as e:issues.append(dict(input=name,path=d.get('path'),problem=str(e)))
    return dict(status='NOT_READY_EXACT_INPUTS_MISSING_OR_CHANGED' if issues else 'EXACT_METADATA_PRESENT_NOT_GPU_QUALIFIED',
        issues=issues,scientific_full_pool_qualified=False,H100_native_restart_qualified=False)


def inputs(request):
    c.require(request['schema']=='root-h100-engineering-metadata-request-v3' and request['owner']=='Root'
        and request['status']=='REGISTERED_BEFORE_EXECUTION' and request['study_id']==STUDY
        and request['preparation_sources']==sources(),'Exact registered Root metadata request required')
    state=diagnose(request);c.require(not state['issues'],c.canonical(state).decode())
    data={name:c.checked(request[name]) for name in REQUIRED}
    c.require(request['native_source_packet']['sha256']==NATIVE_SHA and len(data['native_source_packet']['sources'])==91,
              'Exact frozen V7/91 source packet required')
    c.verify_sources(data['native_source_packet']['sources'])
    c.require(data['driver_source_packet']['sources']==p.sources(),'Final sealed S6 H100 source closure changed')
    profile=validate_profile(data['site_profile'])
    c.require(profile['site']=='aces' and profile['account']=='158416130568' and profile['partition']=='gpu'
        and profile['qos']=='normal' and profile['gpu_type']=='h100' and profile['slurm_user']=='u.lz341613'
        and profile['python']==BASE+'/envs/native-py310-cu121-v1/bin/python'
        and profile['code_root']==str(c.ROOT),'Exact observed H100 route/interpreter/source root required')
    for field in ('code_root','registry_root','output_root','cache_root'):literal(profile[field])
    for field,value in (('max_gpus_per_node',1),('max_cpus_per_request',8),('max_memory_mb',65536),('max_minutes',55)):
        c.require(profile[field]>=value,'Inventoried H100 route cannot admit requested '+field)
    pool=data['pool'];transport=data['transport_template']
    c.require(pool['schema']=='native-engineering-pool-v1' and pool['count']==1024
        and pool['engineering_only'] is True and pool['runtime_transport_required'] is True
        and pool['status']=='PASS_FROZEN_LOGICAL_ENGINEERING_INPUTS','Only actual logical1024 fixture allowed')
    c.require(data['input_rebind_provenance']['target_pool']==request['pool'], 'Engineering rebind names another target pool')
    with selection.authority_eligibility(request['eligibility']):selection.validate_pool(request['pool'],allow_engineering=True)
    c.require(transport['scope']=='ENGINEERING_ONLY' and transport['canonical_image_manifest']==pool['image_manifest'],
              'Wrong transport scope/canonical identity')
    c.checked(transport['locator_index']);c.checked(transport['index_review'])
    eligibility=data['eligibility']
    c.require(eligibility['status']=='PASS_FROZEN_POOL_EVALUATION_ELIGIBILITY' and eligibility['pool']==request['pool']
        and eligibility['missing_media_count']==0 and eligibility['protected_QA_image_RGB_overlap_count']==0,
        'Actual frozen logical engineering eligibility receipt missing; no full-pool inference')
    c.checked(eligibility['evaluation_protocol'])
    env=data['environment'];assets=data['assets']
    c.require(env['status']=='PASS_ENVIRONMENT' and env['python_executable']==profile['python']
        and env['precision']=='fp16' and env['attention']=='xformers','Actual bootstrap FP16 environment absent')
    c.require(assets['status']=='PASS_ASSETS' and assets['official_commit']==c.PIN,'Actual pinned asset receipt absent')
    proof=data['actual_prefix_proof'];raw=data['raw_prefix_proof_copy']
    source_equivalence_bridge(proof,data['native_source_packet']['sources'],raw=raw,assets=assets,environment=env)
    c.require(proof['status']=='PASS_ACTUAL_TOKENIZER_PARITY_ONLY' and proof['records_checked']==1024
        and not proof['violations']
        and all(proof[x] is True for x in ('all_input_ids_identical','all_unaffected_labels_identical','all_independent_prefix_oracle_identical')),
        'Actual repaired1024 tokenizer proof absent')
    c.require(all(request['raw_prefix_proof_copy'][k]==proof['raw_remote_only'][k] for k in ('bytes','sha256'))
        and raw['status']==proof['status'] and raw['repair_sources']==proof['repair_sources'],'Local raw proof differs')
    return data,profile


def study_shell(request,profile,configs):
    return dict(schema='h100-transport-engineering-study-v10',owner='Root',id=STUDY,
        purpose='H100_TRANSPORT_ENGINEERING_ONLY',total_steps=8,interrupt_step=4,minimum_launch_seconds=600,
        numeric_tolerance=p.TOLERANCE,driver_sources=p.sources(),
        native_target=dict(package='experiments.root_native_campaign_v7_20261024',source_packet=request['native_source_packet']),
        native_configurations=configs,input_rebind_provenance=request['input_rebind_provenance'],
        source_pool_metadata=request['source_pool_metadata'],actual_prefix_proof=request['actual_prefix_proof'],
        raw_prefix_proof_copy=request['raw_prefix_proof_copy'],report=profile['registry_root']+'/engineering/'+STUDY+'/actual')


def design(request_desc):
    request=c.checked(request_desc);data,profile=inputs(request)
    root=literal(profile['registry_root']+'/engineering/'+STUDY);root.mkdir(parents=True,exist_ok=True)
    with open(literal(root/'PREPARE.lock'),'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        existing=root/'HELD_DESIGN_REVIEW_REQUEST.json'
        if existing.exists():
            c.require(c.read(existing)['registered_request']==request_desc,'Design request collision');return c.desc(existing)
        # Deliberately no actual-job claim and no claimed physical transport PASS.
        template=copy.deepcopy(data['transport_template']);template['ownership']=None
        transport=c.seal(root/'design'/'TRANSPORT_TEMPLATE.json',template)
        selected=selection.build(request['pool'],'full','1',1234,root/'full1024',allow_engineering=True)
        p.validate_rebind(data['input_rebind_provenance'],data['pool'],c.checked(selected),request['source_pool_metadata'])
        cells=[dict(id=STUDY+'-'+arm,method='full',fraction='1',training_seed=17,selection_seed=1234,
            site='aces',role='qualify',gpus=1,microbatch=4,global_batch=128,qualification_max_steps=8,
            track='engineering',engineering_only=True,minimum_actual_records=1024) for arm in ARMS]
        plan=c.seal(root/'PLAN.json',dict(schema='native-campaign-plan-v1',owner='Root',id=STUDY,
            recipe=RECIPE,official_commit=c.PIN,sources=data['native_source_packet']['sources'],
            candidate_cells=[],bridge_cells=[],reference_cells=[],qualification_cells=cells,
            no_automatic_refill=True,scientific_pool_qualified=False))
        pair={};drafts={}
        for arm in ARMS:
            cfg=dict(schema='native-selection-run-v1',owner='Root',id=STUDY+'-'+arm,plan=plan,
                role='qualify',method='full',fraction='1',training_seed=17,selection_seed=1234,recipe=RECIPE,
                official_commit=c.PIN,allocation=ALLOCATION,checkpoint_seconds=86400,qualification_max_steps=8,
                site_profile=request['site_profile'],sources=data['native_source_packet']['sources'],
                output=profile['output_root']+'/'+STUDY+'-'+arm,report=str(root/arm),review=None,
                replication_block=dict(replicate_site='aces',registered_purpose='engineering-parity-'+arm),
                providers={name:request[name] for name in ('assets','environment','pool','eligibility')},
                storage_reservation=None)
            cfg['providers'].update(selection=selected,media_transport=transport,ownership=None)
            pair[arm]=cfg;drafts[arm]=c.seal(root/'design'/arm/'NON_RUNNABLE_CONFIG.json',cfg)
        skeleton=study_shell(request,profile,drafts)
        held=c.seal(root/'HELD_DESIGN_DRAFT.json',p.design_contract(skeleton,pair))
        return c.seal(existing,dict(status='PENDING_INDEPENDENT_HELD_DESIGN_AND_NATIVE_PLAN_REVIEW',owner='Root',
            registered_request=request_desc,design=held,native_plan=plan,non_runnable_configurations=drafts,
            planned_storage=dict(total_pair_bytes=32*GIB,maximum_simultaneous_capsules=3,transaction_cap_bytes=10*GIB,
                reference_states=1,resume_states=2,actual_native_capsule_bytes_measured=False),
            actual_job_transport_ownership_bound=False,job_submitted=False,native_H100_qualified=False))


def readonly(argv):
    c.require(argv[0]=='showquota' or argv[:3]==['scontrol','show','job'],'Only scoped read-only census allowed')
    run=subprocess.run(argv,text=True,capture_output=True,timeout=45,check=False)
    c.require(run.returncode==0,'Actual census failed: '+argv[0])
    return dict(argv=argv,stdout=run.stdout,stderr=run.stderr,returncode=run.returncode)


def quantity(value):
    match=re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)',value)
    c.require(match is not None,'Unrecognized quota unit: '+value)
    return Decimal(match[1])*(Decimal(1024)**(' KMGT'.index(match[2]) if match[2] else 0))


def quota_headroom(raw,fs):
    rows=[row.split() for row in raw.splitlines() if row.strip().startswith('/scratch/user/u.lz341613 ')]
    c.require(len(rows)==1 and len(rows[0])==5,'Actual user-scratch quota row absent or ambiguous')
    _,used,limit,files,filelimit=rows[0]
    # showquota rounds at one decimal; subtract a full displayed unit to avoid
    # treating rounded disk use as exact free quota.
    units=re.search(r'[KMGT]$',used)
    uncertainty=quantity('1'+units[0]) if units else Decimal(1)
    available=max(0,int(quantity(limit)-quantity(used)-uncertainty))
    inode_headroom=int(filelimit)-int(files)
    free=min(fs.f_bavail*fs.f_frsize,available);inodes=min(fs.f_favail,inode_headroom)
    c.require(free>=32*GIB and inodes>=10000,'Measured scratch/quota cannot reserve32GiB/10000inodes')
    return dict(available_bytes_conservative=free,available_inodes=inodes,
        statvfs_available_bytes=fs.f_bavail*fs.f_frsize,statvfs_available_inodes=fs.f_favail,
        quota_row=rows[0],rounded_quota_uncertainty_bytes=int(uncertainty))


def held_fields(raw,job,profile,intent,script):
    f=dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)',raw))
    expected=dict(JobId=job,JobName='dsmm-h100-eng-'+STUDY,JobState='PENDING',Reason='JobHeldUser',
        Account=profile['account'],Partition='gpu',QOS='normal',NumTasks='1',
        WorkDir=profile['code_root'],Command=script,Comment='dsmm-h100-eng:'+intent['nonce'],
        TresPerNode='gres/gpu:h100:1',NumCPUs='8',TimeLimit='00:55:00')
    for k,v in expected.items():c.require(f.get(k)==v,'Held job differs in '+k)
    c.require(f.get('NumNodes') in ('1','1-1'), 'Held request must be exactly one node')
    c.require(f.get('UserId','').startswith('u.lz341613(') and f.get('Restarts')=='0'
        and f.get('StartTime')=='Unknown' and f.get('RunTime')=='00:00:00','Held request may have executed')
    mem=dict(x.split('=',1) for x in f.get('ReqTRES','').split(',') if '=' in x)
    c.require(mem.get('mem') in ('64G','65536M') and mem.get('gres/gpu')=='1','Wrong held GPU/RAM request')
    return f


def transport_claim_value(job,pool_desc,selection_desc,index_desc):
    """Bind the exact consumer descriptor, even when source bytes are identical."""
    pool=c.checked(pool_desc);selected=c.checked(selection_desc)
    records=selected['records_json']
    c.require(pool['count']==selected['count']==1024,
              'Only the full engineering selection may own this transport')
    c.require(all(records[k]==pool['records_json'][k] for k in ('bytes','sha256'))
        and c.checked(records)==c.checked(pool['records_json']),
        'Selected full engineering conversations changed')
    return dict(status='ACTIVE_EXCLUSIVE_MEDIA_TRANSPORT_CLAIM',owner='Root',
        job_id=job,logical_pool=pool_desc,records_json=records,transport_index=index_desc)


def finalize(late_desc):
    late=c.checked(late_desc)
    c.require(late['schema']=='root-h100-engineering-held-finalization-v3' and late['owner']=='Root'
        and late['status']=='REGISTERED_BEFORE_EXECUTION' and late['preparation_sources']==sources(),
        'Registered exact held-finalization intent required')
    phase1=c.checked(late['design_request']);request_desc=phase1['registered_request'];request=c.checked(request_desc)
    data,profile=inputs(request);root=literal(profile['registry_root']+'/engineering/'+STUDY)
    design_value=p.validate_design(c.checked(late['reviewed_design']))
    c.require({k:v for k,v in design_value.items() if k!='review'}==c.checked(phase1['design']), 'Reviewed held design differs')
    plan=c.checked(phase1['native_plan']);review=c.checked(late['native_plan_review'])
    c.require(review['status']=='PASS_INDEPENDENT_EXACT_PLAN_AND_SOURCE'
        and review['plan_identity_sha256']==c.key(plan) and review['sources']==plan['sources'], 'Exact independent two-cell native plan review required')
    intent=c.checked(late['submission_intent']);ack=c.checked(late['ack']);job=ack['job_id']
    c.require(str(job).isdecimal() and intent['status']=='INTENT_BEFORE_SUBMISSION' and intent['owner']=='Root'
        and intent['design']==late['reviewed_design'] and intent['driver_sources']==p.sources()
        and intent['site_profile']==request['site_profile'] and intent['allocation']==ALLOCATION
        and ack['intent']==late['submission_intent'] and ack['spool_reconciled'] is True,
        'Actual Root ACK/spool reconciliation must bind the exact durable intent')
    c.require(late['conditional_storage_authorization']==dict(owner='Root',study_id=STUDY,total_reserved_bytes=32*GIB,
        maximum_simultaneous_capsules=3,collective_consumers_reconciled=True), 'Root collective32GiB authorization absent')
    with open(literal(root/'PREPARE.lock'),'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        final=root/'FINAL_STUDY_REVIEW_REQUEST.json'
        if final.exists():
            c.require(c.read(final)['registered_finalization']==late_desc,'Finalization collision');return c.desc(final)
        scheduler=readonly(['scontrol','show','job','-o',str(job)])
        held_fields(scheduler['stdout'],str(job),profile,intent,str(p.HERE/'run.sbatch'))
        for arm in ARMS:c.require(not literal(profile['output_root']+'/'+STUDY+'-'+arm).exists(),'Fresh engineering output exists; reconcile')
        quota=readonly(['showquota']);fs=os.statvfs(literal(BASE));measured=quota_headroom(quota['stdout'],fs)
        measurement_path=root/'ACTUAL_HELD_STORAGE_READBACK.json'
        measured_value=dict(status='ACTUAL_HELD_JOB_AND_USER_SCRATCH_QUOTA_CENSUS',
            UTC=c.now(),node=socket.gethostname(),job_id=str(job),scheduler=scheduler,showquota=quota,
            registered_intent=late_desc,path=BASE,**measured)
        if measurement_path.exists():
            prior=c.read(measurement_path)
            c.require(prior['registered_intent']==late_desc and prior['job_id']==str(job), 'Partial-finalization readback collision')
            measurement=c.desc(measurement_path)
        else:measurement=c.seal(measurement_path,measured_value)
        reservation=c.seal(root/'ROOT_STORAGE_AUTHORIZATION.json',dict(status='AUTHORIZE_EXCLUSIVE_ENGINEERING_32GIB_THREE_CAPSULE_RESERVATION',
            owner='Root',study_id=STUDY,job_id=str(job),output_root=profile['output_root'],total_reserved_bytes=32*GIB,
            maximum_simultaneous_capsules=3,actual_measurement=measurement,registered_conditional_authorization=late_desc,
            collective_consumers_reconciled=True,native_capsule_bytes_measured=False))
        transport=copy.deepcopy(data['transport_template'])
        draft_pair={arm:c.checked(phase1['non_runnable_configurations'][arm]) for arm in ARMS}
        selection_desc=draft_pair['reference']['providers']['selection']
        c.require(selection_desc==draft_pair['resume']['providers']['selection'],
                  'Transport consumer selection must be identical across both arms')
        claim=c.seal(root/'MEDIA_TRANSPORT_CLAIM.json',transport_claim_value(
            str(job),request['pool'],selection_desc,transport['locator_index']))
        transport['ownership']=claim;transport_desc=c.seal(root/'MEDIA_TRANSPORT.json',transport)
        readiness=c.seal(root/'LOGICAL_INPUTS_READY_RUNTIME_ALIAS_REQUIRED.json',dict(status='PASS_READY_NATIVE_CAMPAIGN_INPUTS',
            scope='ENGINEERING_LOGICAL_ONLY',providers={name:request[name] for name in ('pool','assets','environment','eligibility')},
            actual_runtime_alias_pending=True,actual_H100_forward_or_restart_qualified=False,scientific_full_pool_qualified=False))
        configs={}
        for arm in ARMS:
            cfg=c.checked(phase1['non_runnable_configurations'][arm]);cfg['providers']['media_transport']=transport_desc
            cfg['providers']['ownership']=c.seal(root/arm/'OWNERSHIP.json',dict(status='ACTIVE_EXCLUSIVE_NATIVE_PLAN_CELL_CLAIM',owner='Root',
                plan=phase1['native_plan'],cell_id=cfg['id'],pool=request['pool'],replicate_site='aces',site_profile=request['site_profile'],
                qualification_study=STUDY,qualification_arm=arm,actual_held_job_id=str(job),submission_intent=late['submission_intent']))
            cfg['storage_reservation']=c.seal(root/arm/'STORAGE.json',dict(status='PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION',
                owner='Root',output=cfg['output'],cell_id=cfg['id'],checkpoint_transaction_bytes=10*GIB,
                retained_accepted_states=1 if arm=='reference' else 2,allow_new_intermediate_retirement=False,
                collective_reservation=reservation,actual_storage=measurement,total_pair_reserved_bytes=32*GIB,
                pair_maximum_simultaneous_capsules=3,capsule_cap_is_enforced_estimate_not_measured_native_size=True))
            cfg['review']=late['native_plan_review'];cfg['readiness']=readiness
            configs[arm]=c.seal(root/arm/'CONFIG.json',cfg)
        study=study_shell(request,profile,configs);study['design']=late['reviewed_design']
        p.configs(study,reviewed=False)
        from experiments.root_native_media_transport_20261023 import transport as frozen_transport
        frozen_transport._inputs(request['pool'],c.checked(selection_desc)['records_json'],
                                 transport_desc,{'job_id':str(job)})
        study_desc=c.seal(root/'STUDY_DRAFT.json',study)
        return c.seal(final,dict(status='PENDING_INDEPENDENT_EXACT_H100_STUDY_REVIEW_BEFORE_RELEASE',
            owner='Root',job_id=str(job),registered_finalization=late_desc,study=study_desc,
            configurations=configs,measured_storage=measurement,actual_job_transport_claim=claim,
            final_binding_schema='h100-native-engineering-binding-v1',
            binding_fields=dict(owner='Root',job_id=str(job),intent=late['submission_intent'],study='REVIEWED_STUDY_DESCRIPTOR_REQUIRED'),
            submitted_or_released_by_helper=False,H100_native_restart_qualified=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('diagnose','design','finalize'))
    parser.add_argument('--request',required=True);args=parser.parse_args();d=c.desc(args.request)
    result=diagnose(c.checked(d)) if args.action=='diagnose' else design(d) if args.action=='design' else finalize(d)
    print(c.canonical(result).decode())


if __name__=='__main__':main()
