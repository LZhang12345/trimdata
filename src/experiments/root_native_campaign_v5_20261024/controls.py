"""V2 source-matched directional loss controls and matched-seed uniform_actualK.

Repair of controls_v1 (preserved unchanged). Selection algorithms are imported
unchanged from V1; V2 hardens the receipt:

1. ``validate`` requires the EXACT budget object, including nominal_requested_K,
   the exact top-level key set, recomputed covariates (counts, source counts,
   tokens, loss) and the exact producer, so forged unit/value/source reporting
   is rejected.
2. Source labels must pass ``source_labels_v2.validate_labels``: production
   needs pinned-665K author labels; engineering labels need an engineering pool.
3. uniform_actualK uses the SAME selection seed as the reviewed published
   selection (protocol 'matched_published_selection_seed'), so it differs from
   the published selector only in which records fill the identical count.

Claims: controls, not learned utility or interactions.
"""
import argparse
from decimal import Decimal
import json
from pathlib import Path

from . import common as c
from . import selection as sel
from experiments.root_native_selection_20261023.selection import budget_size
from experiments.s1_native_selection_controls_20261023 import controls_v1 as v1
from . import source_labels as labels_v2
from experiments.s1_native_selection_controls_20261023.controls_v1 import (
    DIRECTIONAL, METHODS, choose_directional, choose_uniform_k, covariates)

SCHEMA = 'native-record-selection-control-v2'
BUDGET_UNIT = 'whole_conversation_count'
CLAIM = 'control_not_learned_utility_or_interaction'
SEED_PROTOCOL = 'matched_published_selection_seed'
PRODUCER_PATH = 'experiments/root_native_campaign_v5_20261024/controls.py'
KEYS = frozenset(('schema', 'status', 'pool', 'method', 'fraction', 'selection_seed', 'count', 'record_ids', 'records_json',
                  'groups', 'features', 'assets', 'published', 'comparator_seed_protocol', 'budget', 'covariates',
                  'claim_boundary', 'producer'))


def producer():
    return dict(path=PRODUCER_PATH, sha256=c.sha(Path(__file__)), reused_v1_algorithms_sha256=c.sha(Path(v1.__file__)))


def expected_budget(chosen_count, fraction, eligible, requested):
    return dict(unit=BUDGET_UNIT, value=chosen_count, fraction=fraction, eligible_pool_count=eligible,
                nominal_requested_K=requested)


def _published(descriptor, pool_desc, fraction, allow_engineering):
    c.require(descriptor is not None, 'Reviewed published selection required for uniform_actualK')
    published = sel.validate_selection(descriptor, pool_desc, expected_fraction=fraction, allow_engineering=allow_engineering)
    c.require(published['schema'] == 'native-record-selection-v1' and published['method'] not in sel.METHODS + METHODS,
              'uniform_actualK must reference a reviewed published selector, not an own/control method')
    return published


def _inputs(method, pool_desc, ids, group_desc, feature_desc, assets_desc, published_desc, fraction, allow_engineering):
    if method in DIRECTIONAL:
        c.require(published_desc is None, 'Directional controls take no published selection')
        groups = labels_v2.validate_labels(group_desc, pool_desc, allow_engineering)
        c.require(set(groups) == set(ids), 'Source labels must cover every exact eligible ID')
        features = sel.load_features(feature_desc, pool_desc, assets_desc, ids)
        return groups, features, None
    # Labels/features/assets are not inputs to uniform ranking; callers (e.g. Root
    # bootstrap) may pass shared providers, which build records as None.
    return None, None, _published(published_desc, pool_desc, fraction, allow_engineering)


def build(pool_desc, method, fraction, seed, output, group_desc=None, feature_desc=None, assets_desc=None,
          published_desc=None, allow_engineering=False, eligibility_desc=None):
    with labels_v2.eligibility_scope(eligibility_desc):
        return _build(pool_desc, method, fraction, seed, output, group_desc, feature_desc, assets_desc,
                      published_desc, allow_engineering)


def _build(pool_desc, method, fraction, seed, output, group_desc, feature_desc, assets_desc, published_desc, allow_engineering):
    c.require(method in METHODS and type(seed) is int, 'Unregistered control method or non-integer seed')
    fraction = str(fraction)
    pool, rows, ids = sel.validate_pool(pool_desc, allow_engineering)
    requested = budget_size(len(ids), fraction)
    groups, features, published = _inputs(method, pool_desc, ids, group_desc, feature_desc, assets_desc,
                                          published_desc, fraction, allow_engineering)
    if method in DIRECTIONAL:
        chosen = choose_directional(ids, method, requested, seed, groups, features)
        protocol = None
    else:
        c.require(seed == published['selection_seed'], 'uniform_actualK must use the published selection seed (matched protocol)')
        k = published['count']
        c.require(0 < k <= requested, 'actualK must not exceed nominal budget')
        chosen = choose_uniform_k(ids, k, seed)
        protocol = SEED_PROTOCOL
        group_desc = feature_desc = assets_desc = None
    original = dict(zip(ids, rows))
    output = Path(output)
    records = c.seal(output / 'records.json', [original[i] for i in chosen])
    return c.seal(output / 'SELECTION.json', dict(
        schema=SCHEMA, status='PASS_SELECTION', pool=pool_desc, method=method, fraction=fraction,
        selection_seed=seed, count=len(chosen), record_ids=chosen, records_json=records,
        groups=group_desc, features=feature_desc, assets=assets_desc, published=published_desc,
        comparator_seed_protocol=protocol,
        budget=expected_budget(len(chosen), fraction, len(ids), requested),
        covariates=covariates(chosen, original, groups, features),
        claim_boundary=CLAIM, producer=producer()))


def validate(descriptor, pool_desc, expected_method=None, expected_fraction=None, allow_engineering=False,
             eligibility_desc=None):
    with labels_v2.eligibility_scope(eligibility_desc):
        return _validate(descriptor, pool_desc, expected_method, expected_fraction, allow_engineering)


def _validate(descriptor, pool_desc, expected_method, expected_fraction, allow_engineering):
    selected = c.checked(descriptor)
    c.require(isinstance(selected, dict) and set(selected) == KEYS, 'Control selection fields differ from V2 schema')
    pool, rows, ids = sel.validate_pool(pool_desc, allow_engineering)
    c.require(selected['schema'] == SCHEMA and selected['status'] == 'PASS_SELECTION' and selected['pool'] == pool_desc,
              'Wrong exact control selection')
    method, fraction, seed = selected['method'], selected['fraction'], selected['selection_seed']
    c.require(method in METHODS and isinstance(fraction, str) and type(seed) is int, 'Unregistered control method/fraction/seed')
    if expected_method is not None:
        c.require(method == expected_method, 'Wrong control method')
    if expected_fraction is not None:
        c.require(Decimal(str(expected_fraction)) == Decimal(fraction), 'Wrong control fraction')
    c.require(selected['claim_boundary'] == CLAIM and selected['producer'] == producer(), 'Control claim/producer differs')
    chosen = selected['record_ids']; original = dict(zip(ids, rows))
    c.require(isinstance(chosen, list) and 0 < len(chosen) == len(set(chosen)) == selected['count'] and set(chosen) <= set(ids),
              'Wrong control count/duplicates')
    c.require(c.checked(selected['records_json']) == [original[i] for i in chosen], 'Altered original conversation')
    requested = budget_size(len(ids), fraction)
    c.require(selected['budget'] == expected_budget(len(chosen), fraction, len(ids), requested),
              'Budget must be exact whole_conversation_count with true value/fraction/eligible/nominal_requested_K')
    groups, features, published = _inputs(method, pool_desc, ids, selected['groups'], selected['features'], selected['assets'],
                                          selected['published'], fraction, allow_engineering)
    if method in DIRECTIONAL:
        c.require(selected['comparator_seed_protocol'] is None, 'Directional controls carry no comparator seed protocol')
        c.require(len(chosen) == requested, 'Directional control must spend the exact declared budget')
        expected = choose_directional(ids, method, requested, seed, groups, features)
    else:
        c.require(selected['groups'] is None and selected['features'] is None and selected['assets'] is None,
                  'uniform_actualK receipt must not claim label/feature inputs')
        c.require(selected['comparator_seed_protocol'] == SEED_PROTOCOL and seed == published['selection_seed'],
                  'uniform_actualK must use the published selection seed (matched protocol)')
        c.require(len(chosen) == published['count'] <= requested, 'uniform_actualK count differs from reviewed published actualK')
        expected = choose_uniform_k(ids, len(chosen), seed)
    c.require(chosen == expected, 'Control subset does not match declared algorithm/seed')
    c.require(selected['covariates'] == covariates(chosen, original, groups, features), 'Reported covariates differ from recomputation')
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
    parser.add_argument('--allow-engineering', action='store_true')
    parser.add_argument('--eligibility', help='Independently frozen pool/evaluation eligibility descriptor JSON; required unless --allow-engineering')
    args = parser.parse_args()
    if not args.allow_engineering and not args.eligibility:
        parser.error('--eligibility is required for production controls')
    load = lambda p: c.read(p) if p else None
    print(json.dumps(build(c.read(args.pool), args.method, args.fraction, args.seed, args.output,
                           load(args.groups), load(args.features), load(args.assets), load(args.published),
                           allow_engineering=args.allow_engineering, eligibility_desc=load(args.eligibility))))


if __name__ == '__main__':
    main()
