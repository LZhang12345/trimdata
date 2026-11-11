"""Independent selection-seed replication of bound uniform (seed 5927 vs 1234).

Purpose: measure selection-draw variance at a fixed budget on the same pool,
site and training seeds. The replicate is produced and validated by the BOUND
V3 ``selection.build``/``validate_selection`` with method 'uniform', so the
existing worker consumes it unchanged; this module only

* fixes the predeclared seeds (5927 chosen before any native outcome),
* checks that the replicate is a statistically independent draw from the
  reference (overlap within 6 hypergeometric SD of chance) - an engineering
  gate on the hash, not an outcome-based switch,
* records paired-cell structure and covariates in a sealed REPLICATION receipt,
* verifies that Root's plan carries the predeclared cells (two uniform
  selection_seed=5927 + four source-matched loss cells, 10%, training 17/29)
  and pairs each 5927 cell with exactly one bound uniform 1234 cell. Config
  requires config.selection_seed == cell.selection_seed, so the seed must be in
  the plan; Root owns the plan and this module creates no competing cells.

Claim boundary: same exposed pool; a second draw is NOT fresh-pool confirmation.
"""
import argparse
import json
import math
from pathlib import Path

from experiments.root_native_campaign_v3_20261023 import common as c
from experiments.root_native_campaign_v3_20261023 import selection as sel
from experiments.root_native_selection_20261023.selection import budget_size
from experiments.s1_native_selection_controls_20261023.controls_v1 import covariates
from experiments.s1_native_selection_controls_20261023.source_labels_v2 import eligibility_scope

SCHEMA = 'native-selection-seed-replication-v1'
REFERENCE_SEED = 1234
REPLICATE_SEED = 5927
FRACTION = '0.10'
TRAINING_SEEDS = (17, 29)
OVERLAP_SD_GATE = 6.0
PRODUCER_PATH = 'experiments/s1_native_selection_controls_20261023/seed_replication_v1.py'
CLAIM = 'selection_draw_variance_same_exposed_pool_not_fresh_confirmation'


def overlap_gate(n, k, overlap):
    # Hypergeometric overlap of two independent k-subsets of n.
    mean = k * k / n
    sd = math.sqrt(k * (k / n) * (1 - k / n) * (n - k) / (n - 1)) if n > 1 else 0.0
    z = (overlap - mean) / sd if sd else 0.0
    return dict(eligible=n, k=k, overlap=overlap, overlap_fraction=overlap / k, chance_mean=mean, chance_sd=sd,
                z=z, gate_abs_z_max=OVERLAP_SD_GATE, passed=abs(overlap - mean) <= OVERLAP_SD_GATE * sd + 1)


def plan_cells(plan, site='aces'):
    """Verify Root's plan carries the predeclared design; return exact cell IDs.

    Root owns the plan. This does not create cells; it checks that the plan has
    exactly two uniform selection_seed=5927 cells and four source-matched loss
    cells at 10%, training seeds 17/29 on ``site``, and that each 5927 cell has a
    bound uniform selection_seed=1234 cell with identical fraction/site/training
    seed/recipe fields to pair against.
    """
    cells = [x for name in ('candidate_cells', 'bridge_cells', 'reference_cells', 'qualification_cells') for x in plan.get(name, [])]
    c.require(len({x['id'] for x in cells}) == len(cells), 'Duplicate plan cell IDs')
    replicate = [x for x in cells if x['selection_seed'] == REPLICATE_SEED]
    c.require(sorted((x['method'], x['fraction'], x['site'], x['training_seed']) for x in replicate)
              == [('uniform', FRACTION, site, s) for s in TRAINING_SEEDS], 'Plan must have exactly two uniform sel5927 10% cells')
    directional = [x for x in cells if x['method'] in ('source_loss_low', 'source_loss_high')]
    c.require(sorted((x['method'], x['fraction'], x['site'], x['training_seed'], x['selection_seed']) for x in directional)
              == sorted((m, FRACTION, site, s, REFERENCE_SEED) for m in ('source_loss_high', 'source_loss_low') for s in TRAINING_SEEDS),
              'Plan must have exactly four source-matched loss cells at 10%')
    pairs = {}
    shared = ('fraction', 'site', 'training_seed', 'role', 'gpus', 'microbatch', 'global_batch')
    for row in replicate:
        refs = [x for x in cells if x['method'] == 'uniform' and x['selection_seed'] == REFERENCE_SEED
                and all(x[f] == row[f] for f in shared)]
        c.require(len(refs) == 1, 'Exactly one paired bound uniform sel1234 cell required for ' + row['id'])
        pairs[row['id']] = refs[0]['id']
    return dict(replicate_cell_ids=sorted(pairs), paired_reference_cell_ids=pairs,
                directional_cell_ids=sorted(x['id'] for x in directional), plan_identity_sha256=c.key(plan))


def _report(ref, rep, ids, rows):
    original = dict(zip(ids, rows))
    a, b = ref['record_ids'], rep['record_ids']
    return dict(overlap=overlap_gate(len(ids), len(a), len(set(a) & set(b))),
                reference_covariates=covariates(a, original), replicate_covariates=covariates(b, original))


def _producer():
    return dict(path=PRODUCER_PATH, sha256=c.sha(Path(__file__)))


def build(pool_desc, output, plan, reference_desc=None, allow_engineering=False, site='aces', eligibility_desc=None):
    """Build the 5927 replicate via bound uniform; reuse an existing 1234 reference if given."""
    with eligibility_scope(eligibility_desc):
        return _build(pool_desc, output, plan, reference_desc, allow_engineering, site)


def _build(pool_desc, output, plan, reference_desc, allow_engineering, site):
    cells = plan_cells(plan, site)
    pool, rows, ids = sel.validate_pool(pool_desc, allow_engineering)
    output = Path(output)
    if reference_desc is None:
        reference_desc = sel.build(pool_desc, 'uniform', FRACTION, REFERENCE_SEED, output / 'reference_uniform_sel1234', allow_engineering=allow_engineering)
    replicate_desc = sel.build(pool_desc, 'uniform', FRACTION, REPLICATE_SEED, output / 'replicate_uniform_sel5927', allow_engineering=allow_engineering)
    ref = sel.validate_selection(reference_desc, pool_desc, 'uniform', FRACTION, allow_engineering)
    rep = sel.validate_selection(replicate_desc, pool_desc, 'uniform', FRACTION, allow_engineering)
    c.require(ref['selection_seed'] == REFERENCE_SEED and rep['selection_seed'] == REPLICATE_SEED, 'Predeclared seeds differ')
    report = _report(ref, rep, ids, rows)
    c.require(report['overlap']['passed'], 'Replicate is not an independent draw of the reference (hash overlap gate)')
    return c.seal(output / 'REPLICATION.json', dict(
        schema=SCHEMA, status='PASS_INDEPENDENT_SELECTION_SEED_REPLICATE', pool=pool_desc, method='uniform', fraction=FRACTION,
        reference=reference_desc, replicate=replicate_desc, reference_seed=REFERENCE_SEED, replicate_seed=REPLICATE_SEED,
        seeds_predeclared_before_native_outcomes=True, training_seeds=list(TRAINING_SEEDS), site=site, plan_cells=cells,
        pairing='same pool, site, budget, training seeds and recipe; only selection seed differs',
        readout='per-benchmark paired difference (replicate - reference) per training seed; between-draw spread is a variance estimate, not a method effect',
        **report, claim_boundary=CLAIM, producer=_producer()))


def validate(descriptor, pool_desc, plan, allow_engineering=False, eligibility_desc=None):
    with eligibility_scope(eligibility_desc):
        return _validate(descriptor, pool_desc, plan, allow_engineering)


def _validate(descriptor, pool_desc, plan, allow_engineering):
    receipt = c.checked(descriptor)
    pool, rows, ids = sel.validate_pool(pool_desc, allow_engineering)
    c.require(receipt['schema'] == SCHEMA and receipt['status'] == 'PASS_INDEPENDENT_SELECTION_SEED_REPLICATE'
              and receipt['pool'] == pool_desc, 'Wrong exact replication receipt')
    c.require(receipt['reference_seed'] == REFERENCE_SEED and receipt['replicate_seed'] == REPLICATE_SEED
              and receipt['fraction'] == FRACTION and receipt['training_seeds'] == list(TRAINING_SEEDS), 'Predeclared replication design differs')
    c.require(receipt['claim_boundary'] == CLAIM and receipt['producer'] == _producer(), 'Replication claim/producer differs')
    ref = sel.validate_selection(receipt['reference'], pool_desc, 'uniform', FRACTION, allow_engineering)
    rep = sel.validate_selection(receipt['replicate'], pool_desc, 'uniform', FRACTION, allow_engineering)
    c.require(ref['selection_seed'] == REFERENCE_SEED and rep['selection_seed'] == REPLICATE_SEED, 'Selection seeds differ from receipt')
    report = _report(ref, rep, ids, rows)
    c.require(report['overlap']['passed'], 'Replicate is not an independent draw')
    for name, value in report.items():
        c.require(receipt[name] == value, 'Reported %s differs from recomputation' % name)
    c.require(receipt['plan_cells'] == plan_cells(plan, receipt['site']), 'Plan cells/pairing differ from the reviewed plan')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', help='Existing bound uniform seed-1234 10% SELECTION descriptor JSON')
    parser.add_argument('--site', default='aces', choices=('aces', 'babel'))
    parser.add_argument('--allow-engineering', action='store_true')
    parser.add_argument('--eligibility', help='Independently frozen pool/evaluation eligibility descriptor JSON; required unless --allow-engineering')
    args = parser.parse_args()
    if not args.allow_engineering and not args.eligibility:
        parser.error('--eligibility is required for production replication')
    from experiments.root_native_campaign_v3_20261023 import plan as root_plan
    print(json.dumps(build(c.read(args.pool), args.output, root_plan.build(),
                           c.read(args.reference) if args.reference else None, args.allow_engineering, args.site,
                           eligibility_desc=c.read(args.eligibility) if args.eligibility else None)))


if __name__ == '__main__':
    main()
