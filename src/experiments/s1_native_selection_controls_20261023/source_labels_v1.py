"""Frozen TRAIN source labels from the author's twelve original-index intervals.

Label identity is the DataTailor author interval containing a record's ORIGINAL
LLaVA-665K index (pinned file order). Semantic names (e.g. "VQAv2") are recorded
only as hypotheses with a content audit; they are never the label. A coarse
image-prefix labelling is available for diagnostics only, is named coarse, and
carries a status that the bound ``groups_for`` gate rejects.
"""
import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path

from experiments.root_native_campaign_v3_20261023 import common as c
from experiments.root_native_campaign_v3_20261023 import selection as sel
from experiments.copilot7_published_foundation_20261022 import datatailor_v1 as author

NOMINAL_N = 665298
PINNED_665K_SHA256 = 'ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8'
STARTS = (0, 57669, 80909, 157712, 240495, 249493, 315653, 364100, 450517, 522657, 602657, 624610)
ENDS = (57668, 80908, 157711, 240494, 249492, 315652, 364099, 450516, 522656, 602656, 624609, 665297)
LABEL_STATUS = 'PASS_FROZEN_TRAIN_SOURCE_LABELS'
COARSE_STATUS = 'DIAGNOSTIC_ONLY_COARSE_IMAGE_PREFIX_LABELS_NOT_12_DOMAIN'

# Hypotheses only, from the published LLaVA-1.5 mixture composition. The audit
# measures them; a failed hypothesis does not change the author-interval label.
HYPOTHESES = (
    dict(name='llava_conversation_58k', image_prefix='coco', text_only=False, signature=None),
    dict(name='llava_detail_23k', image_prefix='coco', text_only=False, signature=None),
    dict(name='llava_complex_reasoning_77k', image_prefix='coco', text_only=False, signature=None),
    dict(name='vqav2', image_prefix='coco', text_only=False, signature='Answer the question using a single word or phrase.'),
    dict(name='okvqa', image_prefix='coco', text_only=False, signature='Answer the question using a single word or phrase.'),
    dict(name='a_okvqa', image_prefix='coco', text_only=False, signature="Answer with the option's letter from the given choices directly."),
    dict(name='refcoco', image_prefix='coco', text_only=False, signature='bounding box coordinate'),
    dict(name='visual_genome', image_prefix='vg', text_only=False, signature=None),
    dict(name='gqa', image_prefix='gqa', text_only=False, signature='Answer the question using a single word or phrase.'),
    dict(name='ocr_vqa', image_prefix='ocr_vqa', text_only=False, signature=None),
    dict(name='textcaps', image_prefix='textvqa', text_only=False, signature=None),
    dict(name='sharegpt_text_only', image_prefix=None, text_only=True, signature=None),
)


def check_author_boundaries():
    """The bound DataTailor kernel and this producer must use identical intervals."""
    c.require(tuple(author.STARTS) == STARTS and tuple(author.ENDS) == ENDS, 'Author interval constants differ from DataTailor kernel')
    c.require(STARTS[0] == 0 and ENDS[-1] == NOMINAL_N - 1, 'Intervals must span the pinned 665298 records')
    c.require(all(STARTS[k + 1] == ENDS[k] + 1 for k in range(11)), 'Intervals must be a contiguous partition')
    c.require(len(HYPOTHESES) == 12, 'One hypothesis per author interval')


def interval_of(index):
    c.require(type(index) is int and 0 <= index < NOMINAL_N, 'Original 665K index required')
    return bisect_left(ENDS, index)


def label_name(k):
    return f'author_interval_{k:02d}'


def image_prefix(row):
    image = row.get('image')
    return image.split('/', 1)[0] if isinstance(image, str) and image else None


def _human_turns(row):
    return [t.get('value', '') for t in row.get('conversations', [])[0::2]]


def audit_intervals(catalog):
    """Measure interval contents against the published-mixture hypotheses."""
    check_author_boundaries()
    c.require(len(catalog) == NOMINAL_N, 'Content audit requires the complete pinned catalog')
    report = []
    for k, (start, end) in enumerate(zip(STARTS, ENDS)):
        rows = catalog[start:end + 1]
        hyp = HYPOTHESES[k]
        n = len(rows)
        prefixes = Counter(image_prefix(r) for r in rows)
        sig = hyp['signature']
        sig_fraction = None if sig is None else sum(any(sig in v for v in _human_turns(r)) for r in rows) / n
        multi_turn = sum(len(r.get('conversations', [])) > 2 for r in rows) / n
        text_only = prefixes.get(None, 0) / n
        top, top_count = max(prefixes.items(), key=lambda kv: (kv[1], str(kv[0])))
        purity = top_count / n
        consistent_prefix = (text_only == 1.0) if hyp['text_only'] else (top == hyp['image_prefix'] and text_only < 0.01)
        consistent_sig = None if sig_fraction is None else sig_fraction >= 0.5
        edge = dict(before_first=str(image_prefix(catalog[start - 1])) if start > 0 else 'START',
                    first=str(image_prefix(catalog[start])), last=str(image_prefix(catalog[end])),
                    after_last=str(image_prefix(catalog[end + 1])) if end + 1 < NOMINAL_N else 'END')
        report.append(dict(label=label_name(k), start=start, end=end, count=n,
                           hypothesis=hyp['name'], hypothesis_status='UNVERIFIED_SEMANTIC_NAME',
                           image_prefix_counts={str(p): v for p, v in sorted(prefixes.items(), key=lambda kv: str(kv[0]))},
                           majority_prefix=str(top), majority_prefix_purity=purity, text_only_fraction=text_only,
                           multi_turn_fraction=multi_turn, signature=sig, signature_fraction=sig_fraction,
                           prefix_consistent_with_hypothesis=consistent_prefix,
                           signature_consistent_with_hypothesis=consistent_sig, boundary_neighbours=edge))
    ambiguities = [
        'Author intervals are positional: valid only for the pinned file order (sha256 %s...); a reordered export would mislabel.' % PINNED_665K_SHA256[:12],
        'Intervals 00-06 share the coco image prefix; their mutual boundaries are not verifiable by image prefix, only by prompt signatures and the author constants.',
        'Semantic interval names are hypotheses from the published mixture composition; labels are author_interval_NN, not verified domain names.',
    ]
    for row in report:
        if not row['prefix_consistent_with_hypothesis']:
            ambiguities.append(f"{row['label']}: image-prefix content contradicts hypothesis {row['hypothesis']} "
                               f"(majority {row['majority_prefix']}, purity {row['majority_prefix_purity']:.4f}, text-only {row['text_only_fraction']:.4f})")
        elif row['majority_prefix_purity'] < 0.99:
            ambiguities.append(f"{row['label']}: mixed image prefixes (purity {row['majority_prefix_purity']:.4f})")
        if row['signature_consistent_with_hypothesis'] is False:
            ambiguities.append(f"{row['label']}: prompt signature fraction {row['signature_fraction']:.4f} does not support hypothesis {row['hypothesis']}")
    return report, ambiguities


def labels_for(ids, indices):
    c.require(len(ids) == len(indices) == len(set(ids)) == len(set(indices)), 'Unique eligible IDs/indices required')
    c.require(list(indices) == sorted(indices), 'Original order required')
    return {ident: label_name(interval_of(index)) for ident, index in zip(ids, indices)}


def coarse_labels_for(ids, rows):
    return {ident: 'coarse:' + (image_prefix(row) or 'text_only') for ident, row in zip(ids, rows)}


def _load_catalog(path, expected_sha):
    catalog_desc = c.desc(path)
    c.require(catalog_desc['sha256'] == expected_sha, 'Catalog is not the pinned LLaVA-665K export')
    catalog = json.loads(Path(path).read_text())
    c.require(isinstance(catalog, list) and len(catalog) == NOMINAL_N, 'Pinned catalog must contain 665298 records')
    return catalog_desc, catalog


def produce(pool_desc, catalog_path, output, allow_engineering=False, coarse=False,
            expected_sha=PINNED_665K_SHA256):
    check_author_boundaries()
    pool, rows, ids = sel.validate_pool(pool_desc, allow_engineering)
    indices = pool['source_indices']
    catalog_desc, catalog = _load_catalog(catalog_path, expected_sha)
    for ident, row, index in zip(ids, rows, indices):
        c.require(catalog[index] == row, 'Pool record differs from pinned catalog at its original index: ' + ident)
    report, ambiguities = audit_intervals(catalog)
    original_counts = {label_name(k): ENDS[k] - STARTS[k] + 1 for k in range(12)}
    if coarse:
        groups = coarse_labels_for(ids, rows)
        status, kind = COARSE_STATUS, 'coarse_image_prefix_not_12_domain'
    else:
        groups = labels_for(ids, indices)
        status, kind = LABEL_STATUS, 'author_original_index_interval'
    eligible = Counter(groups.values())
    retained = None if coarse else {g: dict(original=original_counts[g], eligible=eligible.get(g, 0),
                                          retained_fraction=eligible.get(g, 0) / original_counts[g]) for g in original_counts}
    return c.seal(Path(output) / ('COARSE_LABELS_DIAGNOSTIC.json' if coarse else 'SOURCE_LABELS.json'), dict(
        schema='native-train-source-labels-v1', status=status, label_kind=kind, pool=pool_desc,
        catalog=dict(sha256=catalog_desc['sha256'], bytes=catalog_desc['bytes'], count=NOMINAL_N),
        author_boundaries=dict(starts=list(STARTS), ends=list(ENDS),
                               kernel='experiments/copilot7_published_foundation_20261022/datatailor_v1.py',
                               kernel_sha256=c.sha(Path(author.__file__))),
        groups=groups, eligible_counts=dict(sorted(eligible.items())), per_interval_retention=retained,
        empty_author_intervals=[] if coarse else sorted(g for g in original_counts if not eligible.get(g)),
        interval_content_audit=report, ambiguities=ambiguities,
        twelve_author_interval_identity=not coarse, semantic_names_verified=False,
        producer=dict(path='experiments/s1_native_selection_controls_20261023/source_labels_v1.py', sha256=c.sha(Path(__file__)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', required=True, help='Pool descriptor JSON ({path,bytes,sha256})')
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--coarse', action='store_true')
    parser.add_argument('--allow-engineering', action='store_true')
    args = parser.parse_args()
    print(json.dumps(produce(c.read(args.pool), args.catalog, args.output, args.allow_engineering, args.coarse)))


if __name__ == '__main__':
    main()
