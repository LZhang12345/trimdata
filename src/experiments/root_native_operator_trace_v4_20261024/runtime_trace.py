"""Observe concrete xFormers apply calls; no policy/dispatch/tensor mutation.

Counters and shape metadata stay in CPU memory until finish. No RNG calls,
tensor payload reads, device synchronization, or per-call filesystem writes.
"""
import collections
import hashlib
import importlib
import os
from pathlib import Path


def identity(cls):
    return cls.__module__+'.'+cls.__qualname__


def tensor_metadata(value):
    return dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),
                device=str(value.device),requires_grad=bool(value.requires_grad))


def input_metadata(inp):
    return dict(query=tensor_metadata(inp.query),key=tensor_metadata(inp.key),
                value=tensor_metadata(inp.value),bias=identity(type(inp.attn_bias)),
                dropout=float(inp.p),scale=inp.scale)


class Trace:
    def __init__(self):
        self.calls=collections.Counter();self.signatures={};self.originals=[]
        self.before_cuda=None;self.installed=False

    def _record(self,kind,op,inp):
        name=identity(op);self.calls[(kind,name)]+=1
        # Metadata key has no tensor values and is bounded by32 signatures/operator.
        meta=input_metadata(inp);key=repr(meta);bucket=self.signatures.setdefault((kind,name),{})
        if key in bucket:bucket[key]['calls']+=1
        elif len(bucket)<32:bucket[key]=dict(metadata=meta,calls=1)
        else:self.calls[('unrecorded_signature',name)]+=1

    def install_ops(self,fw_ops,bw_ops):
        """Separate pure-Python seam exercises actual classmethod dispatch."""
        seen=set()
        for kind,ops in (('forward',fw_ops),('backward',bw_ops)):
            for op in ops:
                if (kind,op) in seen:continue
                seen.add((kind,op));original=getattr(op,'apply')
                local=op.__dict__.get('apply');self.originals.append((op,local))
                inp_index=0 if kind=='forward' else 1
                def observed(cls,*args,_old=original,_op=op,_kind=kind,_index=inp_index,**kwargs):
                    inp=args[_index] if len(args)>_index else kwargs['inp']
                    self._record(_kind,_op,inp)
                    return _old(*args,**kwargs)
                setattr(op,'apply',classmethod(observed))

    def install(self):
        if self.installed:raise ValueError('Trace already installed')
        import torch
        if torch.cuda.is_initialized():raise ValueError('Trace policy attestation must precede CUDA initialization')
        self.before_cuda=dict(policy='observe_native_dispatch_v1',math_changes=False,
            CUDA_initialized=False,CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
            deterministic_warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
            torch_version=torch.__version__,CUDA_version=torch.version.cuda)
        fmha=importlib.import_module('xformers.ops.fmha')
        import xformers
        self.before_cuda['xformers_version']=xformers.__version__
        sources={}
        for name in ('xformers.ops.fmha','xformers.ops.fmha.common','xformers.ops.fmha.dispatch',
                     'xformers.ops.fmha.flash','xformers.ops.fmha.cutlass'):
            path=Path(importlib.import_module(name).__file__).resolve()
            sources[name]=dict(path=str(path),bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.before_cuda['installed_operator_sources']=sources
        self.before_cuda['operators']={identity(op):dict(IS_DETERMINISTIC=bool(op.IS_DETERMINISTIC))
            for op in set(fmha.ALL_FW_OPS+fmha.ALL_BW_OPS)}
        if torch.cuda.is_initialized():raise ValueError('Operator import initialized CUDA before policy attestation')
        self.install_ops(fmha.ALL_FW_OPS,fmha.ALL_BW_OPS);self.installed=True
        return self

    def finish(self):
        # Restore descriptors exactly; an inherited method must remain inherited.
        for op,original in reversed(self.originals):
            if original is None:delattr(op,'apply')
            else:setattr(op,'apply',original)
        self.originals=[]
        records=[]
        for (kind,name),count in sorted(self.calls.items()):
            records.append(dict(kind=kind,operator=name,calls=count,
                signatures=list(self.signatures.get((kind,name),{}).values())))
        return dict(schema='native7-concrete-operator-trace-v1',pre_CUDA_policy=self.before_cuda,
                    executed=records,observational_only=True,dispatch_arguments_unmodified=True,
                    no_tensor_payload_reads_or_RNG_calls=True,
                    forward_calls=sum(v for (k,_),v in self.calls.items() if k=='forward'),
                    backward_calls=sum(v for (k,_),v in self.calls.items() if k=='backward'))


def weight_details(first,second):
    """CPU tensor evidence distinguishes signed zero/dtype from numeric updates."""
    import torch
    rows=[]
    if set(first)!=set(second):raise ValueError('Weight component support changed')
    for group in sorted(first):
        if set(first[group])!=set(second[group]):raise ValueError('Weight parameter support changed')
        for name,a in sorted(first[group].items()):
            b=second[group][name]
            if a.device.type!='cpu' or b.device.type!='cpu':raise ValueError('Probe may inspect CPU copies only')
            if a.shape!=b.shape:raise ValueError('Weight shape changed')
            av=a.double();bv=b.double();finite=bool(torch.isfinite(av).all() and torch.isfinite(bv).all())
            diff=av-bv
            same_dtype=a.dtype==b.dtype
            same_bytes=same_dtype and bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
            rows.append(dict(group=group,name=name,shape=list(a.shape),first_dtype=str(a.dtype),
                second_dtype=str(b.dtype),bitwise_equal=same_bytes,
                numerical_exact_equal=bool(torch.equal(av,bv)),all_finite=finite,
                differing_elements=int(torch.count_nonzero(diff)),
                maximum_absolute_error=float(diff.abs().max()) if diff.numel() else 0.,
                first_zero_elements=int(torch.count_nonzero(av==0)),second_zero_elements=int(torch.count_nonzero(bv==0)),
                signed_zero_differences=int(torch.count_nonzero((av==0)&(bv==0)&(torch.signbit(av)!=torch.signbit(bv))))))
    return dict(compared_trainable_tensors=len(rows),rows=rows,
        all_numeric_exact_equal=all(v['numerical_exact_equal'] for v in rows),
        all_bitwise_equal=all(v['bitwise_equal'] for v in rows),
        no_tensor_payload_exported=True,no_acceptance_tolerance_change=True)


class InitialWeightProbe:
    """Reuse two already-required observer CPU snapshots; no extra model pass."""
    def __init__(self,state_module):
        self.state=state_module;self.original=state_module.trainable_state
        self.first=None;self.result=None;self.calls=0

    def install(self):
        def observed(trainer):
            value=self.original(trainer);self.calls+=1
            if self.calls==1:self.first=value
            elif self.calls==2:
                self.result=weight_details(self.first,value);self.first=None
            else:raise ValueError('One-step trace unexpectedly requested another weight snapshot')
            return value
        self.state.trainable_state=observed
        return self

    def finish(self):
        self.state.trainable_state=self.original
        return dict(snapshot_calls=self.calls,comparison=self.result,
            scope='ACTUAL_FRESH_T1_INITIAL_TO_UPDATE1_NOT_RETROACTIVE_G1_INITIAL_TENSORS',
            retains_initial_CPU_copy_only_until_second_snapshot=True)
