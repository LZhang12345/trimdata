"""CPU tokenizer-only scope after authenticated native import, not a CUDA-free claim."""
from contextlib import contextmanager
import socket
from experiments.root_native_campaign_20261023 import common as c

@contextmanager
def pre_model_tokenizer_guard(report):
    """Permit an existing incidental context, deny new neural/device/I/O work.

    The worker's actual allocation/custody guards must already have passed.
    Explicit-device memory queries avoid current_device calling _lazy_init while
    that entrypoint is denied. No model exists in this pre-construction phase.
    """
    import torch
    import transformers
    initialized=bool(torch.cuda.is_initialized())
    device=int(torch.cuda.current_device()) if initialized else None
    before=dict(initialized=initialized,device=device,
        allocated_bytes=int(torch.cuda.memory_allocated(device)) if initialized else 0,
        reserved_bytes=int(torch.cuda.memory_reserved(device)) if initialized else 0)
    report.update(schema='root-dt-pre-model-tokenizer-context-v1',before=before,
        existing_context_allowed=True,CPU_only_input_and_mask_tensors_required=True,
        scope='Tokenizer preprocessing only after outer owned-allocation authentication; no model/weights/forward',
        blocked_operations={},completed_guard=False)
    changes=[]
    def install(obj,name,label):
        old=getattr(obj,name)
        def fail(*args,**kwargs):
            report['blocked_operations'][label]=report['blocked_operations'].get(label,0)+1
            raise RuntimeError('ForbiddenPreModelTokenizerOperation:'+label)
        changes.append((obj,name,old));setattr(obj,name,fail)
    for obj,name,label in [(socket.socket,'connect','network'),(torch,'load','checkpoint_load'),(torch,'save','checkpoint_save'),
        (torch.cuda,'_lazy_init','CUDA_lazy_init'),(torch.cuda,'init','CUDA_init'),(torch.cuda,'set_device','CUDA_set_device'),
        (transformers.PreTrainedModel,'__init__','pretrained_model_constructor'),
        (torch.nn.Parameter,'__new__','parameter_constructor'),(torch.nn.Module,'_call_impl','neural_forward')]:
        install(obj,name,label)
    try:yield report
    finally:
        try:
            after_initialized=bool(torch.cuda.is_initialized())
            # If state changed from uninitialized, do not initialize a device to inspect it.
            after=dict(initialized=after_initialized,device=device,
                allocated_bytes=int(torch.cuda.memory_allocated(device)) if initialized and after_initialized else 0,
                reserved_bytes=int(torch.cuda.memory_reserved(device)) if initialized and after_initialized else 0)
            report['after']=after
        finally:
            for obj,name,old in reversed(changes):setattr(obj,name,old)
        after['device']=int(torch.cuda.current_device()) if after_initialized else None
        report['completed_guard']=True
        report['CUDA_context_and_memory_unchanged']=after==before
        c.require(not report['blocked_operations'],'Forbidden pre-model operation attempted; retain diagnostics')
        c.require(after==before,'Pre-model tokenizer CUDA initialization/device/memory changed')
