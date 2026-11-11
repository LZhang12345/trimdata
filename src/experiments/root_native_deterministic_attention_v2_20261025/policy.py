"""Evict diagnostic records rather than aborting valid native attention calls.

The original V1 deterministic flags, operators, input metadata definition and
exact original attention call are retained. The bounded cache is observational:
every uncached layout, including an evicted layout seen again, is checked before
dispatch. Ordered digests and explicit counters do not claim a complete roster.
"""
import collections
import functools
import hashlib
from pathlib import Path

from experiments.root_native_deterministic_attention_v1_20261024 import policy as prior
from experiments.root_native_operator_trace_v1_20261024.runtime_trace import input_metadata, identity

POLICY = prior.POLICY
CUBLAS = prior.CUBLAS
CACHE_ENTRIES = 512
TELEMETRY = 'bounded_lru_layout_evidence_streaming_digest_v2'
require = prior.require
assert_flags = prior.assert_flags
set_before_cuda = prior.set_before_cuda


def append_hash(digest, value):
    raw = repr(value).encode('utf-8')
    digest.update(len(raw).to_bytes(8, 'big'))
    digest.update(raw)


def installed_policy_wrapper(function):
    code = getattr(function, '__code__', None)
    return bool(getattr(function, '_dsmm_bounded_policy_v2', False)
        or (code is not None and code.co_name == 'explicit'
            and Path(code.co_filename).resolve() in (Path(prior.__file__).resolve(), Path(__file__).resolve())))


class CutlassPolicy(prior.CutlassPolicy):
    def __init__(self, torch, ops, fmha, fw, bw):
        super().__init__(torch, ops, fmha, fw, bw)
        self.supported = collections.OrderedDict()
        self.entries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.maximum_cached = 0
        self.support_validation_attempts = 0
        self.support_validation_successes = 0
        self.successful_returns = 0
        self.failure_events = 0
        self.last_failure = None
        self.dispatch_hash = hashlib.sha256(b'DSMM-CUTLASS-VALIDATED-DISPATCH-LAYOUTS-v2\0')
        self.support_hash = hashlib.sha256(b'DSMM-CUTLASS-UNCACHED-SUPPORT-CHECKS-v2\0')
        self.failure_hash = hashlib.sha256(b'DSMM-CUTLASS-FAILURE-EVENTS-v2\0')
        self.installed = False
        self.wrapper = None

    def failure(self, stage, error, metadata=None, reasons=None):
        text = repr(error)
        self.last_failure = dict(stage=stage, error_class=type(error).__name__,
            error_repr=text[:2048], error_repr_truncated=len(text)>2048,
            input=metadata, support_reasons=reasons)
        self.failure_events += 1
        append_hash(self.failure_hash, self.last_failure)

    def install(self):
        require(not self.installed, 'Attention policy already installed')
        require(not installed_policy_wrapper(self.ops.memory_efficient_attention)
                and not installed_policy_wrapper(self.fmha.memory_efficient_attention),
                'Nested bounded attention telemetry is forbidden')
        require(self.ops.memory_efficient_attention is self.original
                and self.fmha.memory_efficient_attention is self.original_fmha,
                'Attention entrypoint changed before installation')
        require(all(bool(getattr(op, 'IS_DETERMINISTIC', False)) for op in self.pair),
                'Chosen forward/backward pair is not advertised deterministic')
        assert_flags(self.torch)
        parent = self

        @functools.wraps(self.original)
        def explicit(query, key, value, attn_bias=None, p=0.0, scale=None, *, op=None):
            parent.entries += 1
            stage, metadata, reasons = 'deterministic_flags', None, None
            try:
                assert_flags(parent.torch)
                stage = 'requested_operator'
                require(op is None or op == parent.pair,
                        'Caller requested another attention operator; no fallback')
                stage = 'input_metadata'
                inp = parent.fmha.Inputs(query=query, key=key, value=value,
                                        attn_bias=attn_bias, p=p, scale=scale)
                metadata = input_metadata(inp)
                signature = repr(metadata)  # Exactly the original V1 cache identity.
                if signature not in parent.supported:
                    parent.cache_misses += 1
                    parent.support_validation_attempts += 1
                    stage = 'uncached_support_validation'
                    reasons = {identity(chosen): list(chosen.not_supported_reasons(inp))
                               for chosen in parent.pair}
                    append_hash(parent.support_hash, dict(input=metadata,
                        operators=[identity(x) for x in parent.pair], support_reasons=reasons))
                    require(all(not values for values in reasons.values()),
                        'Explicit deterministic operator unsupported for actual layout: '+repr(reasons))
                    parent.support_validation_successes += 1
                    # The cap bounds retained CPU evidence; it never caps valid execution.
                    if len(parent.supported) == CACHE_ENTRIES:
                        parent.supported.popitem(last=False)
                        parent.evictions += 1
                    parent.supported[signature] = dict(input=metadata,
                        operators=[identity(x) for x in parent.pair],
                        advertised_deterministic=[bool(x.IS_DETERMINISTIC) for x in parent.pair],
                        support_reasons=reasons)
                    parent.maximum_cached = max(parent.maximum_cached, len(parent.supported))
                else:
                    parent.cache_hits += 1
                    parent.supported.move_to_end(signature)
                stage = 'original_explicit_attention_dispatch'
                parent.calls += 1
                append_hash(parent.dispatch_hash, dict(input=metadata,
                    operators=[identity(x) for x in parent.pair]))
                result = parent.original(query, key, value, attn_bias, p, scale, op=parent.pair)
                parent.successful_returns += 1
                return result
            except BaseException as error:
                parent.failure(stage, error, metadata, reasons)
                raise

        explicit._dsmm_bounded_policy_v2 = True
        self.wrapper = explicit
        self.ops.memory_efficient_attention = explicit
        self.fmha.memory_efficient_attention = explicit
        self.installed = True
        return self

    def snapshot(self, flags):
        return dict(policy=POLICY, calls=self.calls, supported_layouts=list(self.supported.values()),
            final_flags=flags, no_operator_fallback=True, numerical_qualification_passed=False,
            telemetry=dict(schema=TELEMETRY, cache_capacity=CACHE_ENTRIES,
                retained_layouts=len(self.supported), maximum_cached_layouts=self.maximum_cached,
                roster_scope='CURRENT_BOUNDED_LRU_CACHE_NOT_COMPLETE_HISTORICAL_ROSTER',
                complete_historical_layout_roster=False,
                distinct_historical_layout_count_exact=(len(self.supported) if self.evictions == 0 else None),
                distinct_historical_layout_count_lower_bound=self.maximum_cached+(1 if self.evictions else 0),
                distinct_count_scope='SUCCESSFULLY_SUPPORT_VALIDATED_LAYOUTS_ONLY; REJECTED_LAYOUTS_EXCLUDED',
                distinct_count_is_not_cache_miss_count=True, policy_entries=self.entries,
                validated_dispatch_attempts=self.calls, successful_original_returns=self.successful_returns,
                cache_hits=self.cache_hits, cache_misses=self.cache_misses, cache_evictions=self.evictions,
                uncached_support_validation_attempts=self.support_validation_attempts,
                uncached_support_validation_successes=self.support_validation_successes,
                digest_algorithm='SHA256_DOMAIN_PREFIX_THEN_UINT64BE_LENGTH_AND_UTF8_REPR_PER_EVENT',
                support_digest_scope='COMPLETED_PAIR_REASON_RESULTS; SUPPORT_METHOD_EXCEPTIONS_USE_FAILURE_DIGEST',
                validated_dispatch_order_digest=self.dispatch_hash.hexdigest(),
                uncached_support_check_order_digest=self.support_hash.hexdigest(),
                failure_event_order_digest=self.failure_hash.hexdigest(),
                failure_events=self.failure_events, last_failure=self.last_failure,
                digest_scope='THIS_PROCESS_SEGMENT_ONLY_NOT_PRIOR_OR_FUTURE_COVERAGE',
                evicted_layouts_revalidated_before_dispatch=True,
                no_tensor_payload_reads_or_RNG_calls=True, math_policy_unchanged_from_v1=True))

    def finish(self):
        require(self.installed, 'Attention policy is not installed or already finished')
        try:
            flags = assert_flags(self.torch)
            return self.snapshot(flags)
        except BaseException as error:
            self.failure('finish_deterministic_flags', error)
            raise
        finally:
            self.restore()

    def restore(self):
        # Exact original entrypoints are restored even on early child/trace failure.
        self.ops.memory_efficient_attention = self.original
        self.fmha.memory_efficient_attention = self.original_fmha
        self.installed = False


def install():
    """Fresh-child entry, with the original pre-CUDA math/runtime requirements."""
    import torch
    require(torch.__version__ == '2.1.2+cu121', 'Unreviewed actual Torch build')
    attestation = set_before_cuda(torch)
    import xformers
    import xformers.ops as ops
    import xformers.ops.fmha as fmha
    from xformers.ops.fmha import cutlass
    require(xformers.__version__ == '0.0.23.post1', 'Unreviewed actual xFormers build')
    require(not torch.cuda.is_initialized(), 'Attention import initialized CUDA before native model')
    policy = CutlassPolicy(torch, ops, fmha, cutlass.FwOp, cutlass.BwOp).install()
    attestation['observational_telemetry_extension'] = dict(schema=TELEMETRY,
        unchanged_math_policy=POLICY, bounded_evidence_capacity=CACHE_ENTRIES,
        no_historical_complete_roster_claim=True, numerical_qualification_passed=False)
    return policy, attestation
