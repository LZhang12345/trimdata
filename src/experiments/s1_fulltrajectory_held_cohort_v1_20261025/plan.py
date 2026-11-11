"""Finite S1 prospective requests; authoring does not reserve storage or submit."""
import argparse
from pathlib import Path
from . import protocol as p


def prepare(receipt_ref, root_requests_dir, site, fraction, seeds, output):
    percent = p.percent_of(fraction)
    receipt = p.c.checked(receipt_ref)
    rows = []
    for seed in seeds:
        paired = {m: p.c.desc(Path(root_requests_dir) / f'production-student-v1-{percent}-{m}-s{seed}'
                              / 'REQUEST_REVIEWED_V3.json') for m in ('uniform', 'source_sqrt')}
        anchor = p.c.checked(paired['source_sqrt'])
        profile = p.c.checked(anchor['site_profile'])
        for arm in p.ARMS:
            cid = p.cell_id(fraction, arm, seed)
            asd = anchor['scientific_design']
            request = dict(
                schema=p.SCHEMA, status=p.STATUS, owner=p.OWNER, sources=p.sources(), native_sources=p.c.sources(),
                recipe=p.ncfg.RECIPE, site=site, cell_id=cid, arm=arm, training_seed=seed, fraction=fraction,
                selection_receipt=receipt_ref, paired_root_requests=paired,
                scientific_design=dict(method=p.METHOD_PREFIX + arm, fraction=fraction, training_seed=seed,
                                       selection_seed=1234, selected_count=p.BUDGETS[percent],
                                       selection=receipt['finalized'][arm]['selection'],
                                       logical_pool=asd['logical_pool'], logical_eligibility=asd['logical_eligibility'],
                                       epoch=1, max_steps=-1, global_batch=128, replicate_site=site),
                site_profile=anchor['site_profile'], scheduler_time_evidence=anchor['scheduler_time_evidence'],
                timing_basis=anchor['timing_basis'], outer_allocation=anchor['outer_allocation'],
                control_root=str(Path(profile['registry_root']) / p.REGISTRY / cid),
                reserved_storage_bytes=0, release_authorized=False, future_release_gates=p.GATES, review=None)
            p.design(request)
            ref = p.c.seal(Path(output) / cid / 'REQUEST_DRAFT.json', request)
            rows.append(dict(cell_id=cid, request_draft=ref, request_identity=p.c.key(p.bare(request)),
                             minutes=request['outer_allocation']['minutes'], physical_job_id=None,
                             paired_root_requests=paired,
                             status='UNREGISTERED_REQUIRES_GENUINE_HELD_DESIGN_REVIEW'))
    return p.c.seal(Path(output) / 'ROSTER_DRAFT.json', dict(
        schema='s1-fulltrajectory-held-roster-v1', owner=p.OWNER, site=site, fraction=fraction, rows=rows,
        request_count=len(rows), proposed_single_attempt_GPU_hours=sum(x['minutes'] for x in rows) / 60,
        submitted_GPU_hours=0, actual_jobs_submitted=0, release_authorized=False, storage_reserved_bytes=0,
        full_epoch_completion_not_claimed=True, downstream_utility_established=False))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--receipt', required=True)
    a.add_argument('--root-requests-dir', required=True)
    a.add_argument('--site', required=True, choices=sorted(p.rp.SITES))
    a.add_argument('--fraction', required=True)
    a.add_argument('--seeds', default='17,29,43')
    a.add_argument('--output', required=True)
    v = a.parse_args()
    seeds = tuple(int(x) for x in v.seeds.split(','))
    print(p.c.canonical(prepare(p.c.desc(v.receipt), v.root_requests_dir, v.site, v.fraction, seeds,
                                v.output)).decode())


if __name__ == '__main__':
    main()
