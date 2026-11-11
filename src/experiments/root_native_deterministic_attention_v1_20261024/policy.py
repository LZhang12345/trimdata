"""Explicit CUTLASS fw/bw policy; unsupported or changed policies fail loudly.

Only language calls already routed through pinned xFormers are intercepted.
This does not declare all model/optimizer kernels deterministic. Actual fresh
and restarted trajectories must still pass the original numerical tests.
"""
import functools
import os

POLICY='explicit_xformers_cutlass_fw_bw_deterministic_v1'
CUBLAS=':4096:8'


def require(value,message):
    if not value:raise ValueError(message)


def observed_flags(torch):
    return dict(CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        deterministic_warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic))


def assert_flags(torch):
    wanted=dict(CUBLAS_WORKSPACE_CONFIG=CUBLAS,deterministic_algorithms=True,
        deterministic_warn_only=False,matmul_allow_tf32=False,cudnn_allow_tf32=False,
        cudnn_benchmark=False,cudnn_deterministic=True)
    actual=observed_flags(torch)
    require(actual==wanted,'Deterministic runtime policy reset or ignored: '+repr(actual))
    return actual


def set_before_cuda(torch):
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==CUBLAS,
        'Launcher must set exact CUBLAS policy before Torch/torchrun import')
    require(not torch.cuda.is_initialized(),'Deterministic policy installed after CUDA initialization')
    before=observed_flags(torch)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    after=assert_flags(torch)
    require(not torch.cuda.is_initialized(),'Policy setup initialized CUDA')
    return dict(policy=POLICY,CUDA_initialized=False,before=before,after=after,
        unsupported_operations_must_raise=True,no_warn_only=True)


class CutlassPolicy:
    def __init__(self,torch,ops,fmha,fw,bw):
        self.torch=torch;self.ops=ops;self.fmha=fmha;self.pair=(fw,bw)
        self.original=ops.memory_efficient_attention;self.original_fmha=fmha.memory_efficient_attention
        self.supported={};self.calls=0

    def install(self):
        from experiments.root_native_operator_trace_v1_20261024.runtime_trace import input_metadata,identity
        require(all(bool(getattr(op,'IS_DETERMINISTIC',False)) for op in self.pair),
            'Chosen forward/backward pair is not advertised deterministic')
        assert_flags(self.torch)
        parent=self
        @functools.wraps(self.original)
        def explicit(query,key,value,attn_bias=None,p=0.0,scale=None,*,op=None):
            assert_flags(parent.torch)
            require(op is None or op==parent.pair,'Caller requested another attention operator; no fallback')
            inp=parent.fmha.Inputs(query=query,key=key,value=value,attn_bias=attn_bias,p=p,scale=scale)
            metadata=input_metadata(inp);signature=repr(metadata)
            if signature not in parent.supported:
                reasons={identity(chosen):list(chosen.not_supported_reasons(inp)) for chosen in parent.pair}
                require(all(not values for values in reasons.values()),
                    'Explicit deterministic operator unsupported for actual layout: '+repr(reasons))
                require(len(parent.supported)<512,'Unexpectedly unbounded operator layout roster')
                parent.supported[signature]=dict(input=metadata,operators=[identity(x) for x in parent.pair],
                    advertised_deterministic=[bool(x.IS_DETERMINISTIC) for x in parent.pair],support_reasons=reasons)
            parent.calls+=1
            return parent.original(query,key,value,attn_bias,p,scale,op=parent.pair)
        self.ops.memory_efficient_attention=explicit
        self.fmha.memory_efficient_attention=explicit
        return self

    def finish(self):
        flags=assert_flags(self.torch)
        self.ops.memory_efficient_attention=self.original
        self.fmha.memory_efficient_attention=self.original_fmha
        return dict(policy=POLICY,calls=self.calls,supported_layouts=list(self.supported.values()),
            final_flags=flags,no_operator_fallback=True,numerical_qualification_passed=False)


def install():
    """Fresh-child entrypoint; wrapper env must precede importing this module."""
    import torch
    require(torch.__version__=='2.1.2+cu121','Unreviewed actual Torch build')
    attestation=set_before_cuda(torch)
    import xformers
    import xformers.ops as ops
    import xformers.ops.fmha as fmha
    from xformers.ops.fmha import cutlass
    require(xformers.__version__=='0.0.23.post1','Unreviewed actual xFormers build')
    require(not torch.cuda.is_initialized(),'Attention import initialized CUDA before native model')
    policy=CutlassPolicy(torch,ops,fmha,cutlass.FwOp,cutlass.BwOp).install()
    return policy,attestation
