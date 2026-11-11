"""Finite R1 roster, not fabricated R2/R3 findings or automatic refill."""
from . import common as c
from .config import RECIPE

VARIANTS={
 'source_proportional':dict(kind='simple_source_allocation_control',allocation='source_record_proportions',rank='seeded_hash',features=[]),
 'source_sqrt':dict(kind='source_allocation_variant',allocation='sqrt_source_record_counts_capped',rank='seeded_hash',features=[]),
 'source_loss_coverage':dict(kind='native_signal_coverage_variant_not_proven_utility',allocation='source_record_proportions',rank='source_normalized_loss_decile_coverage',features=['native_assistant_loss']),
 'source_visual_coverage':dict(kind='visual_coverage_control_not_learned_interaction',allocation='source_record_proportions',rank='native_CLIP12bit_bucket_coverage',features=['native_CLIP_mean_patch_embedding'])}

def build():
    own=[];bridge=[];references=[]
    def cell(method,fraction,seed,site,track):
        return dict(id=f'{track}-{site}-{method}-p{int(float(fraction)*100):02d}-s{seed}',
            method=method,fraction=fraction,training_seed=seed,selection_seed=1234,site=site,
            role='student',gpus=1,microbatch=4,global_batch=128,track=track,
            providers_required=['environment','assets','pool','eligibility','qualification','ownership','storage']+
                (['groups'] if method not in ('uniform','full') else [])+
                (['features'] if method in ('source_loss_coverage','source_visual_coverage','source_loss_low','source_loss_high') else []))
    for method in ('source_sqrt','source_loss_coverage','source_visual_coverage'):
        for fraction in ('0.05','0.10','0.20'):
            for seed in (17,29):own.append(cell(method,fraction,seed,'aces','candidate'))
    for method in ('source_loss_low','source_loss_high'):
        for seed in (17,29):own.append(cell(method,'0.10',seed,'aces','candidate'))
    for seed in (17,29):
        row=cell('uniform','0.10',seed,'aces','candidate')
        row['selection_seed']=5927;row['id']+='-q5927';row['purpose']='independent_selection_draw_control'
        own.append(row)
    for method,site in (('uniform','aces'),('uniform','babel')):
        for fraction in ('0.05','0.10','0.20'):
            for seed in (17,29):bridge.append(cell(method,fraction,seed,site,'bridge'))
    for site in ('aces','babel'):
        for seed in (17,29):references.append(cell('full','1',seed,site,'reference'))
    qualifiers=[dict(id='qualify-'+site,method='full',fraction='1',training_seed=17,
        selection_seed=1234,site=site,role='qualify',gpus=1,microbatch=4,global_batch=128,
        qualification_max_steps=8,track='engineering',engineering_only=True,minimum_actual_records=1024) for site in ('aces','babel')]
    return dict(schema='native-campaign-plan-v1',owner='Root',recipe=RECIPE,official_commit=c.PIN,
        author_fork_equivalence_established=False,variants=VARIANTS,candidate_cells=own,
        bridge_cells=bridge,reference_cells=references,qualification_cells=qualifiers,
        candidate_count=len(own),bridge_count=len(bridge),reference_count=len(references),
        sources=c.sources(),selection_budget_unit='whole_conversation_count',
        nominal_original_N=665298,full_pool_counts=[33264,66529,133059],
        actual_pool_N_requires_complete_media_and_protected_eligibility=True,
        R1='24screening+12uniform+4full_reference_trajectories_with_actual_paired_seed_readout',
        loss_coverage_claim='expected_null_stratified_random_control_not_directional_utility',
        source_label_policy='pinned_author_original_index_intervals_not_assumed_semantic_domains',
        record_order_policy='common_original_source_order_controlled_adaptation',
        R2='WAIT_ACTUAL_R1_RESULTS_THEN_TARGETED_DEV_REPAIR',
        R3='WAIT_R2_THEN_FROZEN_FRESH_CONSTRUCTION_OR_POOL_REPLICATION',
        no_old_setup_training=True,no_automatic_refill=True)

def get_cell(plan,identifier):
    found=[row for name in ('candidate_cells','bridge_cells','reference_cells','qualification_cells') for row in plan.get(name,[]) if row['id']==identifier]
    c.require(len(found)==1,'Unknown/duplicate finite plan cell')
    return found[0]

def build_published():
    """First two qualified published operators, never renamed placeholders."""
    cells=[]
    for method in ('coincide','datatailor'):
        for fraction in ('0.05','0.10','0.20'):
            for seed in (17,29):
                cells.append(dict(id=f'published-babel-{method}-p{int(float(fraction)*100):02d}-s{seed}',
                    method=method,fraction=fraction,training_seed=seed,selection_seed=1234,
                    site='babel',role='student',gpus=1,microbatch=4,global_batch=128,
                    track='published_baseline',selection_origin='external_reviewed_published_operator',
                    providers_required=['environment','assets','pool','eligibility','qualification',
                        'ownership','storage','external_selection'],acquisition_reused_across_training_seeds=True,
                    nominal_count_may_exceed_author_floor_count=True,
                    matched_actual_K_uniform_required_before_method_claim=True))
    return dict(schema='native-campaign-plan-v1',owner='Root',recipe=RECIPE,official_commit=c.PIN,
        sources=c.sources(),candidate_cells=[],bridge_cells=cells,reference_cells=[],qualification_cells=[],
        candidate_count=0,bridge_count=len(cells),reference_count=0,
        methods=['coincide','datatailor'],selection_budget_unit='whole_conversation_count',
        nominal_original_N=665298,actual_eligible_N_not_assumed=True,
        selected_training_rows='unaltered_whole_original_conversations',training_partition='general',
        acquisition_site_may_be_aces=True,actual_features_and_GPU_qualification_ready=False,
        no_old_Qwen_or_16record_features=True,no_automatic_refill=True,
        remaining_published_operators='LESS_ICONS_ScalSelect_require_new_native_fidelity_and_acquisition_before_admission',
        R1='Three record fractions and two paired seeds; same native trainer as candidate/control block',
        R2='WAIT_ACTUAL_R1_RESULTS_THEN_TARGETED_DEV_REPAIR',
        R3='WAIT_R2_THEN_FROZEN_FRESH_CONSTRUCTION_OR_POOL_REPLICATION')

def build_complete():
    """One reviewed finite identity for both sites and all replacement lineage."""
    plan=build(); published=build_published()
    plan['bridge_cells']+=published['bridge_cells']
    plan['uniform_bridge_count']=12
    plan['published_baseline_count']=12
    plan['bridge_count']=len(plan['bridge_cells'])
    plan['scientific_trajectory_count']=52
    plan['published_operator_protocol']=published
    return plan

if __name__=='__main__':
    print(c.canonical(build()).decode())
