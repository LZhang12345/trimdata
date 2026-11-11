"""Observation-only guards after the native image expansion/truncation seam."""
from . import common as c

def assert_expanded_targets(prepared,supervised=False):
    c.require(isinstance(prepared,tuple) and len(prepared)==6,'Unexpected native multimodal return contract')
    labels=prepared[-1]
    c.require(not supervised or labels is not None,'Native expansion discarded supervised labels')
    if labels is not None:
        c.require(labels.ndim==2 and labels.shape[-1]>1,'Malformed actual expanded labels')
        counts=(labels[...,1:]!=-100).sum(dim=-1)
        c.require(bool((counts>0).all().item()),
            'Native image expansion/truncation removed all supervised targets from a record; preserve and diagnose')
    return prepared

def install_expanded_target_guard(model):
    # PEFT delegates forward to its base model. Guard the actual called method,
    # rather than assigning a shadow attribute on the outer PEFT wrapper.
    base=model.get_base_model()
    original=base.prepare_inputs_labels_for_multimodal
    c.require(not getattr(original,'_dsmm_expanded_target_guard',False),'Target guard already installed')
    def observed(*args,**kwargs):
        labels=kwargs.get('labels',args[4] if len(args)>4 else None)
        return assert_expanded_targets(original(*args,**kwargs),supervised=labels is not None)
    observed._dsmm_expanded_target_guard=True
    base.prepare_inputs_labels_for_multimodal=observed
    return original
