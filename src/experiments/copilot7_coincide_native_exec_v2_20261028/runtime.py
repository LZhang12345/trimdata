"""Use the exact qualified telemetry runtime under the COINCIDE authorization."""
from contextlib import contextmanager
from contextvars import ContextVar
from experiments.root_native_deterministic_attention_v2_20261025 import runtime as qualified
from . import protocol as p

launch_environment=qualified.launch_environment
require_current=qualified.require_current
_PROOF=ContextVar("new_COINCIDE_qualified_runtime_proof_v2",default=None)


@contextmanager
def extension(proof):
    p.require(_PROOF.get() is None and qualified.ACTIVE_EXTENSION is None,"No nested numerical runtime scope")
    token=_PROOF.set(proof)
    try:yield
    finally:
        qualified.cleanup();qualified.ACTIVE_EXTENSION=None;_PROOF.reset(token)


def install():
    proof=_PROOF.get()
    p.require(proof is not None and qualified.ACTIVE_EXTENSION is None,"Actual separately reviewed COINCIDE physical scope")
    qualified.ACTIVE_EXTENSION=proof
    return qualified.install()


def finish(policy,trace):
    return qualified.finish(policy,trace)
