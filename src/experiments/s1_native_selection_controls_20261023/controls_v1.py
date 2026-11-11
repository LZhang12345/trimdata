"""Source-matched directional loss controls, explicit-K uniform and full reference.

source_loss_low/high use EXACTLY the bound source_proportional Hamilton quotas,
then take the lowest/highest stored native per-target-token loss within each
source (seeded hash only breaks exact ties). They test whether the stored loss
ranks utility in either direction; they are controls, not learned utility.

uniform_actualK spends a published selector's emitted actualK with the bound
uniform ranking (a prefix of the same seeded order), separating count from
selection. full_reference is the bound 'full' method at fraction 1.
"""
import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path

from experiments.root_native_campaign_v3_20261023 import common as c
from experiments.root_native_campaign_v3_20261023 import selection as sel
from experiments.root_native_selection_20261023.selection import budget_size

DIRECTIONAL = ('source_loss_low', 'source_loss_high')
METHODS = DIRECTIONAL + ('uniform_actualK',)
SCHEMA = 'native-record-selection-control-v1'


def tie(seed, ident):
    return c.key(['s1-directional-loss-v1', seed, ident])


def quotas(ids, count, groups):
    sources = defaultdict(list)
    for ident in ids:
        sources[groups[ident]].append(ident)
    return sources, sel.allocation({g: len(v) for g, v in sources.items()}, count, False)


def choose_directional(ids, direction, count, seed, groups, features):
    c.require(direction in DIRECTIONAL and 0 < count <= len(ids) == len(set(ids)), 'Invalid directional control')
    c.require(set(groups) == set(ids) and set(features) == set(ids), 'Complete labels and real loss features required')
    sources, quota = quotas(ids, count, groups)
    sign = 1.0 if direction == 'source_loss_low' else -1.0
    selected = set()
    for source, roster in sorted(sources.items()):
        ordered = sorted(roster, key=lambda ident: (sign * features[ident]['loss'], tie(seed, ident), ident))
        selected.update(ordered[:quota[source]])
    c.require(len(selected) == count, 'Exact selected-data count lost')
    return [ident for ident in ids if ident in selected]


def choose_uniform_k(ids, count, seed):
    # Identical ranking to bound uniform; nested prefix of every larger budget.
    return sel.choose(ids, 'uniform', count, seed)


def covariates(chosen, original, groups=None, features=None):
    rows = [original[i] for i in chosen]
    out = dict(count=len(rows),
               image_records=sum(bool(r.get('image')) for r in rows),
               human_assistant_pairs=sum(len(r['conversations']) // 2 for r in rows),
               assistant_characters=sum(len(t['value']) for r in rows for t in r['conversations'][1::2]),
               human_characters=sum(len(t['value']) for r in rows for t in r['conversations'][0::2]))
    if groups is not None:
        counts = defaultdict(int)
        for i in chosen:
            counts[groups[i]] += 1
        out['per_source_counts'] = dict(sorted(counts.items()))
    if features is not None:
        out['native_target_tokens'] = sum(features[i]['targets'] for i in chosen)
        out['mean_native_loss'] = sum(features[i]['loss'] for i in chosen) / len(chosen)
    return out


def _published_actual_k(descriptor, pool_desc, fraction, allow_engineering=False):
    # Bound validator binds review pool/records/method/seed/fraction/requested_K/actual_K.
    c.require(descriptor is not None, 'Reviewed published selection required for uniform_actualK')
    published = sel.validate_selection(descriptor, pool_desc, expected_fraction=fraction, allow_engineering=allow_engineering)
    c.require(published['method'] not in sel.METHODS, 'uniform_actualK must reference a published selector, not an own/control method')
    return published


def build(pool_desc, method, fraction, seed, output, group_desc=None, feature_desc=None, assets_desc=None,
          published_desc=None, allow_engineering=False):
    c.require(method in METHODS, 'Unregistered control method')
    pool, rows, ids = sel.validate_pool(pool_desc, allow_engineering)
    requested = budget_size(len(ids), fraction)
    groups = features = None
    if method in DIRECTIONAL:
        groups = sel.groups_for(group_desc, pool_desc, ids)
        features = sel.load_features(feature_desc, pool_desc, assets_desc, ids)
        chosen = choose_directional(ids, method, requested, seed, groups, features)
        k = requested
    else:
        published = _published_actual_k(published_desc, pool_desc, fraction, allow_engineering)
        k = published['count']
        c.require(0 < k <= requested, 'actualK must not exceed nominal budget')
        chosen = choose_uniform_k(ids, k, seed)
    original = dict(zip(ids, rows))
    output = Path(output)
    records = c.seal(output / 'records.json', [original[i] for i in chosen])
    return c.seal(output / 'SELECTION.json', dict(
        schema=SCHEMA, status='PASS_SELECTION', pool=pool_desc, method=method, fraction=str(fraction),
        selection_seed=seed, count=len(chosen), record_ids=chosen, records_json=records,
        groups=group_desc, features=feature_desc, assets=assets_desc, published=published_desc,
        budget=dict(unit='whole_conversation_count', value=len(chosen), fraction=str(fraction),
                    eligible_pool_count=len(ids), nominal_requested_K=requested),
        covariates=covariates(chosen, original, groups, features),
        claim_boundary='control_not_learned_utility_or_interaction',
        producer=dict(path='experiments/s1_native_selection_controls_20261023/controls_v1.py', sha256=c.sha(Path(__file__)))))


def validate(descriptor, pool_desc, expected_method=None, expected_fraction=None, allow_engineering=False):
    selected = c.checked(descriptor)
    pool, rows, ids = sel.validate_pool(pool_desc, allow_engineering)
    c.require(selected['schema'] == SCHEMA and selected['status'] == 'PASS_SELECTION' and selected['pool'] == pool_desc, 'Wrong exact control selection')
    c.require(selected['method'] in METHODS, 'Unregistered control method')
    if expected_method is not None:
        c.require(selected['method'] == expected_method, 'Wrong control method')
    if expected_fraction is not None:
        c.require(Decimal(str(expected_fraction)) == Decimal(selected['fraction']), 'Wrong control fraction')
    chosen = selected['record_ids']; original = dict(zip(ids, rows))
    c.require(0 < len(chosen) == len(set(chosen)) == selected['count'] and set(chosen) <= set(ids), 'Wrong control count/duplicates')
    c.require(c.checked(selected['records_json']) == [original[i] for i in chosen], 'Altered original conversation')
    requested = budget_size(len(ids), selected['fraction'])
    if selected['method'] in DIRECTIONAL:
        c.require(len(chosen) == requested, 'Directional control must spend the exact declared budget')
        groups = sel.groups_for(selected['groups'], pool_desc, ids)
        features = sel.load_features(selected['features'], pool_desc, selected['assets'], ids)
        expected = choose_directional(ids, selected['method'], requested, selected['selection_seed'], groups, features)
    else:
        published = _published_actual_k(selected['published'], pool_desc, selected['fraction'], allow_engineering)
        c.require(len(chosen) == published['count'] <= requested, 'uniform_actualK count differs from reviewed published actualK')
        expected = choose_uniform_k(ids, len(chosen), selected['selection_seed'])
    c.require(chosen == expected, 'Control subset does not match declared algorithm/seed')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--fraction', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--groups'); parser.add_argument('--features'); parser.add_argument('--assets')
    parser.add_argument('--published')
    args = parser.parse_args()
    load = lambda p: c.read(p) if p else None
    print(json.dumps(build(c.read(args.pool), args.method, args.fraction, args.seed, args.output,
                           load(args.groups), load(args.features), load(args.assets), load(args.published))))


if __name__ == '__main__':
    main()
