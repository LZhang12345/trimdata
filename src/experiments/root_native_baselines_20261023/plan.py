"""Twelve finite General student requests, using externally reviewed selections."""
from experiments.root_native_campaign_20261023 import common as c
from experiments.root_native_campaign_20261023.config import RECIPE


def build():
    cells = []
    for method in ('coincide', 'datatailor'):
        for fraction in ('0.05', '0.10', '0.20'):
            for seed in (17, 29):
                cells.append(dict(
                    id=f'published-babel-{method}-p{int(float(fraction)*100):02d}-s{seed}',
                    method=method, fraction=fraction, training_seed=seed,
                    selection_seed=1234, site='babel', role='student', gpus=1,
                    microbatch=4, global_batch=128, track='published_baseline',
                    selection_origin='external_reviewed_published_operator',
                    providers_required=['environment', 'assets', 'pool', 'eligibility',
                        'qualification', 'ownership', 'storage', 'external_selection'],
                    acquisition_reused_across_training_seeds=True,
                    nominal_count_may_exceed_author_floor_count=True,
                    matched_actual_K_uniform_required_before_method_claim=True))
    return dict(schema='native-campaign-plan-v1', owner='Root', recipe=RECIPE,
        official_commit=c.PIN, sources=c.sources(), candidate_cells=[],
        bridge_cells=cells, qualification_cells=[], candidate_count=0,
        bridge_count=len(cells), selection_budget_unit='whole_conversation_count',
        nominal_original_N=665298, actual_eligible_N_not_assumed=True,
        methods=['coincide', 'datatailor'],
        original_published_selected_fraction_unit='record_count',
        selected_training_rows='unaltered_whole_original_conversations',
        acquisition_site_may_be_aces=True, training_partition='general',
        actual_features_and_GPU_qualification_ready=False,
        no_old_Qwen_or_16record_features=True,
        R1='Full eligible pool acquisition once; three fractions and two paired training seeds',
        R2='WAIT_ACTUAL_R1_RESULTS_THEN_DIAGNOSE_AND_TARGET_DEV_REPAIR',
        R3='WAIT_R2_THEN_FROZEN_FRESH_CONSTRUCTION_OR_POOL_REPLICATION',
        no_automatic_refill=True)


if __name__ == '__main__':
    print(c.canonical(build()).decode())
