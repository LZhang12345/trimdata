"""Authenticate logical pool metadata and materialize selected images explicitly.

This API never monkeypatches checked(), rewrites an upstream receipt, issues a
local full-pool physical PASS, or qualifies a trainer. Callers must explicitly
bind the original eligibility and independent archive review. A future native
caller needs its own independently reviewed integration and actual GPU test.
"""
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from experiments.session5_native_archive_authority_20261024 import archive_authority_v1 as archive
from experiments.root_native_campaign_v7_20261024 import selection, source_labels

c = archive.c
METHODS = ('uniform', 'source_sqrt', 'source_proportional', 'full')


def exact_alias(original, local):
    """Authenticate both descriptions and the actual same-byte local object."""
    archive._descriptor(original)
    archive._descriptor(local)
    c.require((original['bytes'], original['sha256']) == (local['bytes'], local['sha256']),
              'Local metadata alias changes upstream bytes')
    return c.checked(local)


@dataclass(frozen=True)
class LogicalView:
    site_descriptor: dict
    rebind_descriptor: dict
    upstream_eligibility: dict
    rebind_review: dict
    site_pool: dict
    upstream_pool: dict
    rows: list
    ids: list
    bindings: archive.MetadataBindings

    def original_metadata(self, descriptor):
        """Only descriptors in the frozen export are eligible for local rebinding."""
        fresh = validate_logical(self.site_descriptor, self.rebind_descriptor,
                                 self.upstream_eligibility, self.rebind_review)
        return fresh.bindings.checked(descriptor)


def validate_logical(site_pool_desc, rebind_desc, expected_upstream_eligibility, expected_rebind_review):
    site, rows, ids = archive.validate_archive_pool(
        rebind_desc, expected_upstream_eligibility, expected_rebind_review)
    rebind = c.checked(rebind_desc)
    c.require(rebind['site_pool'] == site_pool_desc and c.checked(site_pool_desc) == site,
              'Caller site pool does not equal the authenticated archive view')
    c.require(site['schema'] == 'native-archive-site-pool-view-v1' and
              site['status'] == 'LOGICAL_ONLY_RUNTIME_TRANSPORT_REQUIRED' and
              site['runtime_transport_required'] is True and
              site['current_physical_media_readiness'] is False,
              'Logical archive view must never claim local physical readiness')
    config = c.checked(rebind['configuration'])
    bindings = archive.MetadataBindings(c.checked(config['metadata_bindings']))
    upstream_pool = bindings.checked(site['upstream_pool'])
    c.require(upstream_pool['record_ids'] == ids and len(rows) == site['count'] and
              bindings.checked(upstream_pool['records_json']) == rows,
              'Authenticated original rows or count differ')
    return LogicalView(site_pool_desc, rebind_desc, expected_upstream_eligibility,
                       expected_rebind_review, site, upstream_pool, rows, ids, bindings)


def validate_groups(view, original_descriptor, local_descriptor):
    fresh = validate_logical(view.site_descriptor, view.rebind_descriptor,
                             view.upstream_eligibility, view.rebind_review)
    return _validate_groups_on_fresh_view(fresh, original_descriptor, local_descriptor)


def _validate_groups_on_fresh_view(view, original_descriptor, local_descriptor):
    labels = exact_alias(original_descriptor, local_descriptor)
    c.require(labels['schema'] == source_labels.SCHEMA and
              labels['status'] == source_labels.LABEL_STATUS and
              labels['pool'] == view.site_pool['upstream_pool'] and
              labels['label_kind'] == 'author_original_index_interval' and
              labels['twelve_author_interval_identity'] is True,
              'Only original production source labels are supported')
    provenance = labels['label_provenance']
    c.require(provenance['catalog_kind'] == source_labels.PRODUCTION_CATALOG and
              provenance['pinned_catalog_sha256'] == source_labels.PINNED_665K_SHA256 and
              provenance['production_eligible'] is True and provenance['pool_engineering_only'] is False and
              labels['catalog']['sha256'] == source_labels.PINNED_665K_SHA256 and
              labels['catalog']['count'] == source_labels.NOMINAL_N,
              'Source labels lost the pinned production catalog identity')
    c.require(labels['producer'] == dict(path=source_labels.PRODUCER_PATH,
              sha256=c.sha(Path(source_labels.__file__)),
              reused_v1_functions_sha256=c.sha(Path(source_labels.v1.__file__))),
              'Source-label producer differs from the explicit Native7 producer')
    # The catalog has already been authenticated as part of the complete archive
    # partition. Recheck that it is exactly the pinned production annotation.
    catalog = view.upstream_pool['source_json']
    c.require(catalog['sha256'] == source_labels.PINNED_665K_SHA256 and
              view.site_pool['source_indices'] == view.upstream_pool['source_indices'],
              'Source interval labels refer to another original catalog')
    expected = source_labels.labels_for(view.ids, view.site_pool['source_indices'])
    c.require(labels['groups'] == expected, 'Source labels differ from original indices')
    return expected


def validate_selection(view, original_selection_desc, local_selection_desc, selected_records_local,
                       expected_method, expected_fraction, expected_seed, groups_local=None):
    """Verify a frozen upstream selection and exact same-byte row alias.

    Initial support deliberately covers the sixteen feature-independent cells.
    Published or learned-feature selections need an additional reviewed seam.
    """
    fresh = validate_logical(view.site_descriptor, view.rebind_descriptor,
                             view.upstream_eligibility, view.rebind_review)
    return _validate_selection_on_fresh_view(fresh, original_selection_desc, local_selection_desc,
                                            selected_records_local, expected_method,
                                            expected_fraction, expected_seed, groups_local)


def _validate_selection_on_fresh_view(view, original_selection_desc, local_selection_desc,
                                    selected_records_local, expected_method, expected_fraction,
                                    expected_seed, groups_local=None):
    selected = exact_alias(original_selection_desc, local_selection_desc)
    c.require(expected_method in METHODS and selected['schema'] == 'native-record-selection-v1' and
              selected['status'] == 'PASS_SELECTION' and selected['pool'] == view.site_pool['upstream_pool'] and
              selected['method'] == expected_method and selected['selection_seed'] == expected_seed,
              'Upstream selection method, seed, pool or schema differs')
    fraction = Decimal(str(expected_fraction))
    c.require(fraction == Decimal(selected['fraction']) and fraction in
              (Decimal('0.05'), Decimal('0.10'), Decimal('0.20'), Decimal('1')),
              'Unsupported or changed selected-data fraction')
    c.require((expected_method == 'full') == (fraction == 1), 'Full branch requires fraction1 and method full')
    count = len(view.ids) if fraction == 1 else int(Decimal(len(view.ids)) * fraction)
    chosen = selected['record_ids']
    c.require(count > 0 and len(chosen) == len(set(chosen)) == selected['count'] == count and
              set(chosen) <= set(view.ids), 'Selection must spend exact floor-K with no replacement or topup')
    c.require(selected['budget'] == dict(unit='whole_conversation_count', value=count,
              fraction=selected['fraction'], eligible_pool_count=len(view.ids)),
              'Selected-data unit or eligible denominator changed')
    groups = None
    if expected_method in ('source_sqrt', 'source_proportional'):
        c.require(groups_local is not None, 'Source allocation requires explicit same-byte label alias')
        groups = _validate_groups_on_fresh_view(view, selected['groups'], groups_local)
    c.require(chosen == selection.choose(view.ids, expected_method, count, expected_seed, groups),
              'Upstream selected IDs/order do not match the declared operator')
    selected_rows = exact_alias(selected['records_json'], selected_records_local)
    original = dict(zip(view.ids, view.rows))
    c.require(selected_rows == [original[key] for key in chosen],
              'Local selected records changed original whole rows or their order')
    return selected


def prepare_selected(view, original_selection_desc, local_selection_desc, selected_records_local,
                     expected_method, expected_fraction, expected_seed, transport_desc, report,
                     groups_local=None):
    # Revalidate current archive metadata rather than trusting a previously
    # constructed Python object after inputs or accepted boundaries changed.
    fresh = validate_logical(view.site_descriptor, view.rebind_descriptor,
                             view.upstream_eligibility, view.rebind_review)
    _validate_selection_on_fresh_view(fresh, original_selection_desc, local_selection_desc,
                                    selected_records_local, expected_method, expected_fraction,
                                    expected_seed, groups_local)
    cfg = c.checked(transport_desc)
    rebind = c.checked(fresh.rebind_descriptor)
    c.require(cfg['scope'] == 'ALL_SOURCE' and cfg['locator_index'] == rebind['locator_index'] and
              cfg['canonical_image_manifest'] == fresh.site_pool['image_manifest'],
              'Selected transport must use the complete canonical rebound image index')
    alias = archive.transport.prepare(fresh.site_descriptor, selected_records_local, transport_desc, report)
    archive.transport.validate(alias, fresh.site_descriptor, selected_records_local)
    return alias


def source_identity():
    """Explicit execution closure; a receipt never implies installed trainer support."""
    names = set(selection.c.sources()) | set(archive.source_identity())
    names.update(('experiments/root_native_site_consumer_v1_20261024/__init__.py',
                  'experiments/root_native_site_consumer_v1_20261024/consumer.py'))
    return {name: c.sha(c.ROOT / name) for name in sorted(names)}
