"""Explicit caller-bound archive context, retaining original receipt identities."""
from contextlib import contextmanager
from contextvars import ContextVar
from . import common as c
from experiments.root_native_site_consumer_v1_20261024 import consumer

_CONTEXT = ContextVar('native8_explicit_site_consumer_context', default=None)


@contextmanager
def context(descriptor):
    value = c.checked(descriptor) if descriptor is not None else None
    if value is not None:
        c.require(value['schema'] == 'native-explicit-site-consumer-context-v1' and value['owner'] == 'Root',
                  'Caller-bound Root archive context required')
    token = _CONTEXT.set(value)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def view(pool_desc, eligibility_desc=None):
    value = _CONTEXT.get()
    c.require(value is not None and value['site_pool'] == pool_desc,
              'Archive schema requires an explicit caller-bound site context')
    result = consumer.validate_logical(pool_desc, value['rebind'],
        value['expected_upstream_eligibility'], value['expected_rebind_review'])
    if eligibility_desc is not None:
        c.require(eligibility_desc == result.bindings.local_descriptor(value['expected_upstream_eligibility']),
                  'Runtime eligibility must be the authenticated same-byte upstream alias')
    return result, value


def selected(descriptor, pool_desc, method, fraction):
    result, value = view(pool_desc)
    c.require(descriptor == value['local_selection'] and method is not None and fraction is not None,
              'Exact local manifest and caller method/fraction required')
    return consumer.validate_selection(result, value['original_selection'], value['local_selection'],
        value['local_records'], method, fraction, value['selection_seed'], value.get('local_groups'))


def prepare(pool_desc, selected_desc, method, fraction, seed, transport_desc, report):
    result, value = view(pool_desc)
    c.require(selected_desc == value['local_selection'] and seed == value['selection_seed'],
              'Runtime selection/seed differs from frozen archive context')
    return consumer.prepare_selected(result, value['original_selection'], value['local_selection'],
        value['local_records'], method, fraction, seed, transport_desc, report, value.get('local_groups'))
