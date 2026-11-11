"""Original label producer with explicit cross-package eligibility forwarding."""
from contextlib import contextmanager
from experiments.root_native_campaign_v7_20261024.source_labels import *
from experiments.root_native_campaign_v7_20261024 import source_labels as original
from . import selection as caller_selection


@contextmanager
def eligibility_scope(eligibility_desc):
    inherited = caller_selection._ELIGIBILITY.get()
    original.c.require(eligibility_desc is None or inherited is None or inherited == eligibility_desc,
                       'Conflicting Native8 caller eligibility descriptor')
    with original.eligibility_scope(eligibility_desc if eligibility_desc is not None else inherited):
        yield


def validate_labels(descriptor, pool_desc, allow_engineering=False, eligibility_desc=None):
    with eligibility_scope(eligibility_desc):
        return original.validate_labels(descriptor, pool_desc, allow_engineering)


def produce(pool_desc, catalog_path, output, allow_engineering=False, coarse=False,
            engineering_catalog_sha=None, eligibility_desc=None):
    with eligibility_scope(eligibility_desc):
        return original.produce(pool_desc, catalog_path, output, allow_engineering, coarse,
                                engineering_catalog_sha)
