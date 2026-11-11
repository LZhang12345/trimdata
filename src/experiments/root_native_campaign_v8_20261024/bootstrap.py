"""Materialize one independently reviewed finite cell from actual late inputs.

This is a scientific bootstrap, not the ACES environment/assets bootstrap. Missing
inputs yield durable NOT_READY, no neural call and no fabricated PASS. Root leaves
student requests held until their real prerequisites/qualification exist.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sys
from . import common as c
from .config import RECIPE,validate
from .plan import get_cell
from . import selection
from . import controls

def materialize(binding,attempt):
    plan=c.checked(binding['plan']);cell=get_cell(plan,binding['cell_id'])
    c.require(cell['id']==binding['cell_id'] and plan['recipe']==RECIPE,'Bound finite cell/recipe differs')
    c.verify_sources(plan['sources'])
    review=c.checked(binding['review'])
    c.require(review['status']=='PASS_INDEPENDENT_EXACT_PLAN_AND_SOURCE' and review['plan_identity_sha256']==c.key(plan)
        and review['sources']==plan['sources'],'Exact independent plan/source review required')
    ready_path=Path(binding['ready_path'])
    if not ready_path.is_file():return dict(status='NOT_READY',missing=['actual_ready_receipt'])
    readiness=c.read(ready_path)
    if readiness.get('status')!='PASS_READY_NATIVE_CAMPAIGN_INPUTS':return dict(status='NOT_READY',missing=['inputs_not_qualified'],actual_status=readiness.get('status'))
    provider_descs=readiness.get('providers',{})
    required=['environment','assets','pool','eligibility']
    archive_mode=('pool' in provider_descs and Path(provider_descs['pool']['path']).is_file()
        and c.checked(provider_descs['pool']).get('schema')=='native-archive-site-pool-view-v1')
    if archive_mode and cell['method'] not in ('uniform','source_sqrt','source_proportional','full'):
        return dict(status='NOT_READY',missing=['archive_feature_or_published_consumer_not_implemented'])
    implemented=cell['method'] in selection.METHODS+controls.METHODS
    if implemented and not archive_mode and cell['method'] not in ('full','uniform','uniform_actualK'):required.append('groups')
    if cell['method'] in ('source_loss_coverage','source_visual_coverage')+controls.DIRECTIONAL:required.append('features')
    if cell['role']=='student':required.append('qualification')
    missing=[name for name in required if name not in provider_descs or not Path(provider_descs[name]['path']).is_file()]
    owned=readiness.get('cells',{}).get(cell['id'],{})
    missing += [name for name in ('ownership','storage') if name not in owned or not Path(owned[name]['path']).is_file()]
    if archive_mode:
        missing += [name for name in ('site_consumer','selection','media_transport') if name not in owned or not Path(owned[name]['path']).is_file()]
    if not implemented:
        missing += ['selection'] if 'selection' not in owned or not Path(owned['selection']['path']).is_file() else []
    if cell['method']=='uniform_actualK':
        missing += ['published_selection'] if 'published_selection' not in owned or not Path(owned['published_selection']['path']).is_file() else []
    if missing:return dict(status='NOT_READY',missing=missing)
    profile_desc=readiness['site_profile'];profile=c.checked(profile_desc)
    c.require(profile['site']==cell['site'],'Wrong site profile for this finite cell')
    with selection.authority_eligibility(provider_descs['eligibility'],owned.get('site_consumer')):
        if archive_mode:
            selected=owned['selection']
            selection.validate_selection(selected,provider_descs['pool'],cell['method'],cell['fraction'])
        elif not implemented:
            selected=owned['selection']
            selection.validate_selection(selected,provider_descs['pool'],cell['method'],cell['fraction'])
        else:
            selection_root=Path(profile['registry_root'])/'selections'/c.key(dict(pool=provider_descs['pool'],
                groups=provider_descs.get('groups'),features=provider_descs.get('features') if 'features' in required else None,
                method=cell['method'],fraction=cell['fraction'],seed=cell['selection_seed'],published=owned.get('published_selection')))
            selection_root.mkdir(parents=True,exist_ok=True)
            with open(selection_root/'SELECTION.lock','a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                args=(provider_descs['pool'],cell['method'],cell['fraction'],cell['selection_seed'],selection_root,
                    provider_descs.get('groups'),provider_descs.get('features') if 'features' in required else None,provider_descs['assets'])
                if cell['method'] in controls.METHODS:
                    selected=controls.build(*args,published_desc=owned.get('published_selection'),allow_engineering=cell['role']=='qualify')
                else:selected=selection.build(*args,allow_engineering=cell['role']=='qualify')
    providers={name:provider_descs[name] for name in ('environment','assets','pool','eligibility')}
    providers.update(selection=selected,ownership=owned['ownership'])
    if 'media_transport' in owned:
        from experiments.root_native_media_transport_20261023.transport import prepare
        from .media import staging_boundary,require_launch_time
        providers['media_transport']=owned['media_transport']
        with staging_boundary():
            if archive_mode:
                from .site import prepare as prepare_archive
                with selection.authority_eligibility(provider_descs['eligibility'],owned['site_consumer']):
                    providers['runtime_images']=prepare_archive(providers['pool'],selected,cell['method'],
                        cell['fraction'],cell['selection_seed'],owned['media_transport'],attempt/'media-transport')
            else:
                providers['runtime_images']=prepare(providers['pool'],c.checked(selected)['records_json'],
                    owned['media_transport'],attempt/'media-transport')
            require_launch_time(binding['allocation'])
    if cell['role']=='student':providers['qualification']=provider_descs['qualification']
    allocation=dict(binding['allocation'])
    c.require(allocation['gpus']==cell['gpus'] and allocation['microbatch']==cell['microbatch'],'Bound globalbatch/microbatch differs')
    config=dict(schema='native-selection-run-v1',id=cell['id'],owner='Root',role=cell['role'],
        official_commit=c.PIN,recipe=RECIPE,sources=plan['sources'],site_profile=profile_desc,
        providers=providers,method=cell['method'],fraction=cell['fraction'],selection_seed=cell['selection_seed'],
        training_seed=cell['training_seed'],allocation=allocation,
        qualification_max_steps=cell.get('qualification_max_steps'),storage_reservation=owned['storage'],
        output=str(Path(profile['output_root'])/cell['id']),report=str(Path(profile['registry_root'])/'runs'/cell['id']),
        checkpoint_seconds=binding['checkpoint_seconds'],review=binding['review'],plan=binding['plan'],readiness=c.desc(ready_path),
        site_consumer=owned.get('site_consumer'))
    config['replication_block']=dict(replicate_site=cell['site'],registered_purpose=cell['track'])
    validate(config)
    config_desc=c.seal(attempt/'CONFIG.json',config)
    materialized=dict(schema='native-job-binding-v1',job_id=binding['job_id'],owner='Root',
        configuration=config_desc,intent=binding['intent'],plan=binding['plan'],cell_id=cell['id'])
    return dict(status='MATERIALIZED_READY',binding=c.seal(attempt/'BINDING.json',materialized))

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);a=p.parse_args()
    binding=c.read(a.binding)
    if binding.get('schema')=='native-stage1-feature-binding-v1':
        from . import features
        sys.argv=['native-feature-worker','--binding',a.binding]
        return features.main()
    job=os.environ.get('SLURM_JOB_ID','')
    c.require(binding['schema']=='native-campaign-late-binding-v1' and job==binding['job_id'] and job.isdecimal(),'Actual scalar bootstrap ownership differs')
    report=Path(binding['admission_directory'])
    attempt=report/f"bootstrap-{job}-r{os.environ.get('SLURM_RESTART_COUNT','0')}"
    attempt.mkdir(parents=True,exist_ok=True)
    from .media import StagingContinuation,requeue
    try:result=materialize(binding,attempt)
    except StagingContinuation as error:
        # Authenticate the actual complete scheduler route before any mutation.
        from .worker import actual
        from .config import validate_profile
        ready=c.read(binding['ready_path']);profile=validate_profile(c.checked(ready['site_profile']))
        actual(binding,dict(id=binding['cell_id'],allocation=binding['allocation']),profile)
        requeue(binding,attempt,error)
        return 0
    c.seal(attempt/'RESULT.json',result)
    if result['status']!='MATERIALIZED_READY':return 78
    from . import worker
    sys.argv=['native-campaign-worker','--binding',result['binding']['path']]
    worker.main()
    return 0

if __name__=='__main__':raise SystemExit(main())
