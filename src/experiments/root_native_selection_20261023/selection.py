"""Whole-conversation selection; no neural score or published selector is invented."""
import argparse
from collections import Counter
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from . import common as c

def budget_size(n, fraction):
    fraction = Decimal(str(fraction))
    c.require(fraction in map(Decimal, ('0.05', '0.10', '0.20', '1')), 'Unregistered budget')
    return int((Decimal(n)*fraction).to_integral_value(rounding=ROUND_FLOOR))

def validate_pool(descriptor):
    pool = c.checked(descriptor)
    c.require(pool['schema'] == 'published-llava-pool-v1' and pool['status'] == 'PASS_POOL', 'Qualified native pool required')
    rows = c.checked(pool['records_json'])
    ids = pool['record_ids']; indices = pool['source_indices']
    c.require(len(rows) == len(ids) == len(set(ids)) == len(indices) == len(set(indices)) == pool['count'], 'Pool count/identity duplicates')
    c.require(indices == sorted(indices), 'Original source order differs')
    c.require(ids == [f'{i:06d}:{r.get("id", "")}' for i, r in zip(indices, rows)], 'Original record identity differs')
    if pool['pool_kind'] == 'full665k':
        c.require(len(rows) == 665298, 'Full original mixture must contain exactly 665298 records')
    else:
        c.require(pool['pool_kind'] == 'all_source_protected_bridge', 'COCO-only pool cannot stand in for broad mixture')
    c.checked(pool['protected_exclusions'])
    for row in rows:
        turns = row.get('conversations', [])
        c.require(len(turns) >= 2 and turns[0]['from'] == 'human', 'Whole native conversation required')
        c.require(all(t['from'] == ('human' if i % 2 == 0 else 'gpt') and isinstance(t['value'], str) for i, t in enumerate(turns)), 'Alternating native human/gpt turns required')
        c.require(len(turns) % 2 == 0, 'Unpaired conversation turn')
    return pool, rows, ids

def choose(ids, method, count, seed, groups=None):
    c.require(0 < count <= len(ids) == len(set(ids)), 'Invalid distinct record budget')
    def ranked(roster):
        return sorted(roster, key=lambda ident: (c.key(['native-selection-v1', seed, ident]), ident))
    if method == 'uniform':
        chosen = set(ranked(ids)[:count])
    elif method == 'source_stratified':
        c.require(groups is not None and set(groups) == set(ids), 'Exact complete frozen source labels required')
        sizes = Counter(groups.values())
        c.require(all(isinstance(g, str) and g for g in sizes), 'Nonempty source labels required')
        # Hamilton apportionment: exact record-count proportions, deterministic ties.
        quotas = {g: count*n//len(ids) for g,n in sizes.items()}
        remainder = count-sum(quotas.values())
        for g in sorted(sizes, key=lambda g: (-(count*sizes[g] % len(ids)), g))[:remainder]:
            quotas[g] += 1
        chosen = set()
        for g in sorted(sizes):
            chosen.update(ranked([i for i in ids if groups[i] == g])[:quotas[g]])
    elif method == 'full':
        c.require(count == len(ids), 'Full-data count differs')
        chosen = set(ids)
    else:
        raise ValueError('Published/learned methods require supplied qualified selection; no placeholder selector')
    return [ident for ident in ids if ident in chosen]  # native source order, not score order

def validate_selection(descriptor, pool_desc, expected_method=None, expected_fraction=None):
    selected = c.checked(descriptor)
    pool, rows, ids = validate_pool(pool_desc)
    c.require(selected['schema'] == 'native-record-selection-v1' and selected['status'] == 'PASS_SELECTION', 'Qualified exact selection required')
    c.require(selected['pool'] == pool_desc, 'Selection belongs to a different pool')
    chosen = selected['record_ids']
    c.require(len(chosen) == len(set(chosen)) == selected['count'], 'Duplicate/missing selection')
    c.require(set(chosen) <= set(ids), 'Foreign records selected')
    selected_rows = c.checked(selected['records_json'])
    original = dict(zip(ids, rows))
    c.require(selected_rows == [original[i] for i in chosen], 'Conversation contents/turns were changed')
    c.require(selected['budget'] == {'unit':'whole_conversation_count', 'value':len(chosen), 'fraction':selected['fraction'], 'eligible_pool_count':len(ids)}, 'Budget metadata differs')
    c.require(len(chosen) == budget_size(len(ids), selected['fraction']), 'Incorrect percentage budget')
    if expected_method is not None: c.require(selected['method'] == expected_method, 'Wrong selection method')
    if expected_fraction is not None: c.require(Decimal(selected['fraction']) == Decimal(str(expected_fraction)), 'Wrong fraction')
    if selected['method'] in ('uniform', 'source_stratified', 'full'):
        groups=None
        if selected['method']=='source_stratified':
            group_receipt=c.checked(selected['groups'])
            c.require(group_receipt['pool']==pool_desc and group_receipt['status']=='PASS_FROZEN_TRAIN_SOURCE_LABELS', 'Source labels must be frozen and pool matched')
            groups=group_receipt['groups']
        c.require(chosen==choose(ids,selected['method'],len(chosen),selected['selection_seed'],groups), 'Selected roster differs from deterministic method/seed')
    else:
        review = c.checked(selected['method_review'])
        c.require(review['status'] == 'PASS_EXACT_SELECTOR_IMPLEMENTATION_AND_OUTPUT' and review['pool'] == pool_desc and review['records_json'] == selected['records_json']
            and review['method']==selected['method'] and review['selection_seed']==selected['selection_seed'] and review['fraction']==selected['fraction'], 'Published selector requires independent implementation/output review')
    return selected

def build(pool_desc, method, fraction, seed, output, group_desc=None):
    pool, rows, ids = validate_pool(pool_desc)
    groups = None
    if group_desc:
        group_receipt = c.checked(group_desc)
        c.require(group_receipt['pool'] == pool_desc and group_receipt['status'] == 'PASS_FROZEN_TRAIN_SOURCE_LABELS', 'Source labels must be frozen and pool matched')
        groups = group_receipt['groups']
    count = budget_size(len(ids), fraction)
    chosen = choose(ids, method, count, seed, groups)
    original = dict(zip(ids, rows)); output = Path(output)
    records = c.seal(output/'records.json', [original[i] for i in chosen])
    return c.seal(output/'SELECTION.json', dict(schema='native-record-selection-v1', status='PASS_SELECTION',
        pool=pool_desc, method=method, fraction=str(fraction), selection_seed=seed, count=count,
        record_ids=chosen, records_json=records, groups=group_desc,
        budget=dict(unit='whole_conversation_count', value=count, fraction=str(fraction), eligible_pool_count=len(ids)),
        selection_inputs_are_training_only=True, scoring_cost_kind='metadata_only_not_free'))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pool', required=True)
    parser.add_argument('--method', choices=['uniform','source_stratified','full'], required=True)
    parser.add_argument('--fraction', choices=['0.05','0.10','0.20','1'], required=True)
    parser.add_argument('--seed', type=int, default=1234); parser.add_argument('--output', required=True)
    parser.add_argument('--groups'); args=parser.parse_args()
    print(c.canonical(build(c.desc(args.pool),args.method,args.fraction,args.seed,args.output,c.desc(args.groups) if args.groups else None)).decode())

if __name__ == '__main__': main()
