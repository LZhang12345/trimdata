"""Explicit new runtime object for a reviewed observational child overlay.

Nothing mutates the original runtime module. Root must bind this new executed
source extension separately from the unchanged scientific config/accepted state.
"""
from experiments.root_production_epoch_qualification_v1_20261024 import runtime as prior
from . import policy

launch_environment = prior.launch_environment
require_current = prior.require_current
ACTIVE_EXTENSION = None
ACTIVE_POLICY = None
ACTIVE_TRACE = None


def install():
    global ACTIVE_POLICY, ACTIVE_TRACE
    policy.require(ACTIVE_EXTENSION is not None and ACTIVE_POLICY is None and ACTIVE_TRACE is None,
                   'Reviewed single child-overlay scope required')
    import torch
    policy.require(not torch.cuda.is_initialized() and torch.__version__ == '2.1.2+cu121',
                   'Exact Torch/preCUDA process required')
    selected, attestation = policy.install()
    ACTIVE_POLICY = selected
    try:
        from experiments.root_babel_operator_trace_v2_20261024.runtime_trace import Trace
        trace = Trace()
        ACTIVE_TRACE = trace
        trace.install()
        attestation['observational_execution_extension'] = ACTIVE_EXTENSION
        return selected, trace, attestation
    except BaseException:
        cleanup()
        raise


def finish(selected, trace):
    global ACTIVE_POLICY, ACTIVE_TRACE
    policy.require(selected is ACTIVE_POLICY and trace is ACTIVE_TRACE and ACTIVE_EXTENSION is not None,
                   'Exact active observational runtime instances required')
    try:
        result = prior.finish(selected, trace)
        result['observational_execution_extension'] = ACTIVE_EXTENSION
        return result
    finally:
        cleanup()


def cleanup():
    global ACTIVE_POLICY, ACTIVE_TRACE
    try:
        if ACTIVE_TRACE is not None and getattr(ACTIVE_TRACE, 'originals', None):
            ACTIVE_TRACE.finish()
    finally:
        if ACTIVE_POLICY is not None:
            ACTIVE_POLICY.restore()
        ACTIVE_POLICY = None
        ACTIVE_TRACE = None
