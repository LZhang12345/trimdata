"""V2 frozen TRAIN source labels: production is ALWAYS the pinned LLaVA-665K export.

Repair of source_labels_v1 (preserved unchanged). V1 let any caller override the
expected catalog SHA while still emitting the production status. V2:

* production (no engineering catalog) always requires the exact pinned
  665K SHA and emits PASS_FROZEN_TRAIN_SOURCE_LABELS;
* an unpinned catalog SHA is accepted ONLY with allow_engineering=True AND an
  explicit engineering-only pool, and emits a DISTINCT engineering status that
  the bound V3 ``groups_for`` gate rejects;
* ``validate_labels`` re-derives the author-interval labels from the pool's
  original indices and checks catalog/provenance/producer identity.

Interval logic, hypotheses and the content audit are imported unchanged from V1.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path

from . import common as c
from . import selection as sel
from experiments.copilot7_published_foundation_20261022 import datatailor_v1 as author
from experiments.s1_native_selection_controls_20261023 import source_labels_v1 as v1
from experiments.s1_native_selection_controls_20261023.source_labels_v1 import (
    NOMINAL_N, PINNED_665K_SHA256, STARTS, ENDS, LABEL_STATUS, COARSE_STATUS,
    check_author_boundaries, interval_of, label_name, audit_intervals, labels_for, coarse_labels_for)

SCHEMA = 'native-train-source-labels-v2'
ENGINEERING_LABEL_STATUS = 'ENGINEERING_ONLY_SOURCE_LABELS_UNPINNED_CATALOG_NOT_PRODUCTION'
PRODUCER_PATH = 'experiments/root_native_campaign_v5_20261024/source_labels.py'
PRODUCTION_CATALOG = 'pinned_llava_665k_original_order'
ENGINEERING_CATALOG = 'engineering_unpinned_catalog_not_llava_665k'


def _is_engineering_pool(pool):
    return pool.get('schema') == 'native-engineering-pool-v1' and pool.get('engineering_only') is True


def eligibility_scope(eligibility_desc):
    """Scope pool validation under the caller's EXPLICIT, independently frozen
    eligibility (Root selection.authority_eligibility). None inherits the
    caller's scope (e.g. Root bootstrap/config); never inferred from the pool.
    A different explicit descriptor than an inherited one is rejected."""
    if eligibility_desc is None:
        return nullcontext()
    c.require(isinstance(eligibility_desc, dict) and set(eligibility_desc) >= {'path', 'sha256'},
              'Eligibility must be an explicit sealed descriptor')
    inherited = sel._ELIGIBILITY.get()
    c.require(inherited is None or inherited == eligibility_desc, 'Conflicting pool eligibility descriptor in scope')
    return sel.authority_eligibility(eligibility_desc)


def _load_catalog(path, expected_sha):
    catalog_desc = c.desc(path)
    c.require(catalog_desc['sha256'] == expected_sha, 'Catalog SHA differs from the required catalog identity')
    catalog = json.loads(Path(path).read_text())
    c.require(isinstance(catalog, list) and len(catalog) == NOMINAL_N, 'Catalog must contain 665298 records')
    return catalog_desc, catalog


def produce(pool_desc, catalog_path, output, allow_engineering=False, coarse=False, engineering_catalog_sha=None,
            eligibility_desc=None):
    with eligibility_scope(eligibility_desc):
        return _produce(pool_desc, catalog_path, output, allow_engineering, coarse, engineering_catalog_sha)


def _produce(pool_desc, catalog_path, output, allow_engineering, coarse, engineering_catalog_sha):
    check_author_boundaries()
    pool, rows, ids = sel.validate_pool(pool_desc, allow_engineering)
    engineering_catalog = engineering_catalog_sha is not None
    if engineering_catalog:
        c.require(allow_engineering and _is_engineering_pool(pool),
                  'Unpinned catalog allowed only for an explicit engineering-only pool with allow_engineering')
        c.require(engineering_catalog_sha != PINNED_665K_SHA256, 'Pinned catalog must use the production path')
    expected_sha = engineering_catalog_sha if engineering_catalog else PINNED_665K_SHA256
    indices = pool['source_indices']
    catalog_desc, catalog = _load_catalog(catalog_path, expected_sha)
    for ident, row, index in zip(ids, rows, indices):
        c.require(catalog[index] == row, 'Pool record differs from catalog at its original index: ' + ident)
    report, ambiguities = audit_intervals(catalog)
    if engineering_catalog:
        ambiguities = ['ENGINEERING CATALOG: not the pinned LLaVA-665K export; labels and audit are fixture-only.'] + ambiguities
    original_counts = {label_name(k): ENDS[k] - STARTS[k] + 1 for k in range(12)}
    if coarse:
        groups = coarse_labels_for(ids, rows)
        status, kind = COARSE_STATUS, 'coarse_image_prefix_not_12_domain'
    else:
        groups = labels_for(ids, indices)
        status = ENGINEERING_LABEL_STATUS if engineering_catalog else LABEL_STATUS
        kind = 'author_original_index_interval'
    eligible = Counter(groups.values())
    retained = None if coarse else {g: dict(original=original_counts[g], eligible=eligible.get(g, 0),
                                          retained_fraction=eligible.get(g, 0) / original_counts[g]) for g in original_counts}
    if coarse:
        name = 'COARSE_LABELS_DIAGNOSTIC.json'
    else:
        name = 'ENGINEERING_SOURCE_LABELS.json' if engineering_catalog else 'SOURCE_LABELS.json'
    return c.seal(Path(output) / name, dict(
        schema=SCHEMA, status=status, label_kind=kind, pool=pool_desc,
        label_provenance=dict(catalog_kind=ENGINEERING_CATALOG if engineering_catalog else PRODUCTION_CATALOG,
                              pinned_catalog_sha256=PINNED_665K_SHA256, pool_engineering_only=_is_engineering_pool(pool),
                              production_eligible=not engineering_catalog and not coarse and not _is_engineering_pool(pool)),
        catalog=dict(sha256=catalog_desc['sha256'], bytes=catalog_desc['bytes'], count=NOMINAL_N),
        author_boundaries=dict(starts=list(STARTS), ends=list(ENDS),
                               kernel='experiments/copilot7_published_foundation_20261022/datatailor_v1.py',
                               kernel_sha256=c.sha(Path(author.__file__))),
        groups=groups, eligible_counts=dict(sorted(eligible.items())), per_interval_retention=retained,
        empty_author_intervals=[] if coarse else sorted(g for g in original_counts if not eligible.get(g)),
        interval_content_audit=report, ambiguities=ambiguities,
        twelve_author_interval_identity=not coarse, semantic_names_verified=False,
        producer=dict(path=PRODUCER_PATH, sha256=c.sha(Path(__file__)),
                      reused_v1_functions_sha256=c.sha(Path(v1.__file__)))))


def validate_labels(descriptor, pool_desc, allow_engineering=False, eligibility_desc=None):
    """Return {record_id: label}; production requires pinned-catalog author labels."""
    with eligibility_scope(eligibility_desc):
        return _validate_labels(descriptor, pool_desc, allow_engineering)


def _validate_labels(descriptor, pool_desc, allow_engineering):
    labels = c.checked(descriptor)
    pool, rows, ids = sel.validate_pool(pool_desc, allow_engineering)
    c.require(labels.get('schema') == SCHEMA and labels['pool'] == pool_desc, 'Exact V2 source labels for this pool required')
    c.require(labels['label_kind'] == 'author_original_index_interval' and labels['twelve_author_interval_identity'] is True,
              'Coarse/diagnostic labels cannot drive selection')
    prov = labels['label_provenance']
    if labels['status'] == LABEL_STATUS:
        c.require(prov['catalog_kind'] == PRODUCTION_CATALOG and labels['catalog']['sha256'] == PINNED_665K_SHA256,
                  'Production labels must come from the pinned LLaVA-665K catalog')
    else:
        c.require(labels['status'] == ENGINEERING_LABEL_STATUS and prov['catalog_kind'] == ENGINEERING_CATALOG,
                  'Unknown source-label status')
        c.require(allow_engineering and _is_engineering_pool(pool), 'Engineering labels cannot satisfy production selection')
    c.require(prov['pool_engineering_only'] == _is_engineering_pool(pool), 'Pool engineering provenance differs')
    c.require(labels['producer'] == dict(path=PRODUCER_PATH, sha256=c.sha(Path(__file__)),
                                         reused_v1_functions_sha256=c.sha(Path(v1.__file__))), 'Label producer differs')
    c.require(labels['groups'] == labels_for(ids, pool['source_indices']), 'Labels differ from author intervals of original indices')
    return labels['groups']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', required=True, help='Pool descriptor JSON ({path,bytes,sha256})')
    parser.add_argument('--catalog', required=True, help='Pinned LLaVA-665K export; no SHA override exists on the CLI')
    parser.add_argument('--output', required=True)
    parser.add_argument('--coarse', action='store_true')
    parser.add_argument('--allow-engineering', action='store_true')
    parser.add_argument('--eligibility', help='Independently frozen pool/evaluation eligibility descriptor JSON; required unless --allow-engineering')
    args = parser.parse_args()
    if not args.allow_engineering and not args.eligibility:
        parser.error('--eligibility is required for production label production')
    print(json.dumps(produce(c.read(args.pool), args.catalog, args.output, args.allow_engineering, args.coarse,
                             eligibility_desc=c.read(args.eligibility) if args.eligibility else None)))


if __name__ == '__main__':
    main()
