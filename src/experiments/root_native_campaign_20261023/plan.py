"""Finite R1 roster, not fabricated R2/R3 findings or automatic refill."""
from . import common as c
from .config import RECIPE

VARIANTS={
 'source_proportional':dict(kind='simple_source_allocation_control',allocation='source_record_proportions',rank='seeded_hash',features=[]),
 'source_sqrt':dict(kind='source_allocation_variant',allocation='sqrt_source_record_counts_capped',rank='seeded_hash',features=[]),
 'source_loss_coverage':dict(kind='native_signal_coverage_variant_not_proven_utility',allocation='source_record_proportions',rank='source_normalized_loss_decile_coverage',features=['native_assistant_loss']),
 'source_visual_coverage':dict(kind='visual_coverage_control_not_learned_interaction',allocation='source_record_proportions',rank='native_CLIP12bit_bucket_coverage',features=['native_CLIP_mean_patch_embedding'])}

def build():
    own=[];bridge=[]
    def cell(method,fraction,seed,site,track):
        return dict(id=f'{track}-{site}-{method}-p{int(float(fraction)*100):02d}-s{seed}',
            method=method,fraction=fraction,training_seed=seed,selection_seed=1234,site=site,
            role='student',gpus=1,microbatch=4,global_batch=128,track=track,
            providers_required=['environment','assets','pool','groups','eligibility','qualification','ownership','storage']+
                (['features'] if method in ('source_loss_coverage','source_visual_coverage') else []))
    for method in VARIANTS:
        for fraction in ('0.05','0.10','0.20'):
            for seed in (17,29):own.append(cell(method,fraction,seed,'aces','candidate'))
    for method,site in (('uniform','aces'),('uniform','babel'),('source_proportional','babel')):
        for fraction in ('0.05','0.10','0.20'):
            for seed in (17,29):bridge.append(cell(method,fraction,seed,site,'bridge'))
    qualifiers=[dict(id='qualify-'+site,method='full',fraction='1',training_seed=17,
        selection_seed=1234,site=site,role='qualify',gpus=1,microbatch=4,global_batch=128,
        qualification_max_steps=6,track='engineering',engineering_only=True,minimum_actual_records=1024) for site in ('aces','babel')]
    return dict(schema='native-campaign-plan-v1',owner='Root',recipe=RECIPE,official_commit=c.PIN,
        author_fork_equivalence_established=False,variants=VARIANTS,candidate_cells=own,
        bridge_cells=bridge,qualification_cells=qualifiers,candidate_count=len(own),bridge_count=len(bridge),
        sources=c.sources(),selection_budget_unit='whole_conversation_count',
        nominal_original_N=665298,full_pool_counts=[33264,66529,133059],
        actual_pool_N_requires_complete_media_and_protected_eligibility=True,
        R1='24candidate+18bridge_trajectories_with_actual_paired_seed_readout',
        R2='WAIT_ACTUAL_R1_RESULTS_THEN_TARGETED_DEV_REPAIR',
        R3='WAIT_R2_THEN_FROZEN_FRESH_CONSTRUCTION_OR_POOL_REPLICATION',
        no_old_setup_training=True,no_automatic_refill=True)

def get_cell(plan,identifier):
    found=[row for name in ('candidate_cells','bridge_cells','qualification_cells') for row in plan[name] if row['id']==identifier]
    c.require(len(found)==1,'Unknown/duplicate finite plan cell')
    return found[0]

if __name__=='__main__':
    print(c.canonical(build()).decode())
