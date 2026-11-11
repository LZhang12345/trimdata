"""Assemble an engineering study from real reviewed native configurations.

Creates no ownership, environment, storage or GPU PASS receipt. Root supplies
those actual prerequisites and the independent review of the resulting study.
"""
import argparse
from pathlib import Path
from . import common as c
from . import study as validation
from .masking import CONTRACT, RECORDS_SHA256, REPAIRED_ID

def draft(args):
    target=dict(package=args.native_package,source_packet=c.desc(args.source_packet))
    modules,packet=c.target_modules(target)
    configs={arm:c.desc(getattr(args,arm+'_config')) for arm in ('reference','resume')}
    bodies={arm:c.checked(d) for arm,d in configs.items()}
    for body in bodies.values():
        c.require(body['sources']==packet['sources'] and body['recipe']==modules['config'].RECIPE,
            'Exact V5 recipe/source required in metadata draft; full validation is inside GPU allocation')
    steps=bodies['reference']['qualification_max_steps']
    c.require(steps==bodies['resume']['qualification_max_steps'] and steps>=4,'Mismatched real native engineering limits')
    body=dict(schema='native-engineering-parity-study-v10',owner='Root',id=args.study_id,
        native_target=target,driver_sources=c.sources(),native_configurations=configs,
        total_steps=steps,interrupt_step=args.interrupt_step,minimum_launch_seconds=args.minimum_launch_seconds,
        numeric_tolerance=dict(absolute=1e-6,relative=1e-5),report=args.report,
        admission_lock=args.admission_lock,
        masking_contract=dict(name=CONTRACT,
            oracle_sha256=c.sha(c.HERE/'masking.py'),
            records_sha256=RECORDS_SHA256,
            repaired_record_ids=[REPAIRED_ID],
            unaffected_records=1023,original_diagnosis=c.desc(args.original_diagnosis),
            actual_prefix_proof=c.desc(args.actual_prefix_proof)),
        scientific_pool_qualified=False,downstream_performance_measured=False)
    c.require(0<args.interrupt_step<steps,'Interruption must precede terminal state')
    for config in bodies.values():
        c.require(config['qualification_max_steps']==8 and config['allocation']['microbatch']==4,
            'Exact8/world1micro4 paired metadata required')
    return c.seal(args.output,body)

def bind(draft_descriptor,review_descriptor,output):
    body=c.checked(draft_descriptor)
    c.require('review' not in body,'Study draft already bound')
    study=dict(body,review=review_descriptor)
    review=c.checked(review_descriptor)
    c.require(review['status']=='PASS_INDEPENDENT_ENGINEERING_DRIVER_AND_STUDY'
        and review['driver_sources']==body['driver_sources']
        and review['study_identity']==c.key(body),'Exact independent V7 metadata review required')
    return c.seal(output,study)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    d=sub.add_parser('draft');d.add_argument('--native-package',default='experiments.root_native_campaign_v7_20261024')
    for name in ('source-packet','reference-config','resume-config','original-diagnosis','actual-prefix-proof','admission-lock','study-id','report','output'):d.add_argument('--'+name,required=True)
    d.add_argument('--interrupt-step',type=int,default=4);d.add_argument('--minimum-launch-seconds',type=int,default=600)
    b=sub.add_parser('bind-review')
    for name in ('draft','review','output'):b.add_argument('--'+name,required=True)
    a=p.parse_args()
    result=draft(a) if a.action=='draft' else bind(c.desc(a.draft),c.desc(a.review),a.output)
    print(c.canonical(dict(status='DRAFT_ONLY_NEEDS_INDEPENDENT_REVIEW' if a.action=='draft' else 'REVIEWED_STUDY_NOT_GPU_QUALIFIED',artifact=result)).decode())

if __name__=='__main__':main()
