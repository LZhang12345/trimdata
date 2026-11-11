"""Semantic hashes and explicit numerical comparisons, not pickle byte equality."""
import hashlib
import math
import random
import struct
from . import common as c

def digest(value):
    import numpy as np
    try:import torch
    except ImportError:torch=None
    h=hashlib.sha256()
    def visit(v):
        if torch is not None and torch.is_tensor(v):
            t=v.detach().cpu().contiguous()
            h.update(c.canonical(['tensor',str(t.dtype),list(t.shape)]))
            h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()).cast('B'))
        elif torch is not None and isinstance(v,torch.dtype):h.update(c.canonical(['torch.dtype',str(v)]))
        elif torch is not None and isinstance(v,torch.device):h.update(c.canonical(['torch.device',v.type,v.index]))
        elif isinstance(v,np.generic):visit(v.item())
        elif isinstance(v,bytes):h.update(c.canonical(['bytes',len(v)]));h.update(v)
        elif isinstance(v,np.ndarray):
            a=np.ascontiguousarray(v);h.update(c.canonical(['numpy',str(a.dtype),list(a.shape)]));h.update(memoryview(a).cast('B'))
        elif isinstance(v,dict):
            h.update(c.canonical(['dict',len(v)]))
            for k in sorted(v,key=lambda item:(str(type(item)),str(item))):visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(c.canonical([str(type(v)),len(v)]))
            for item in v:visit(item)
        elif isinstance(v,float):h.update(b'float'+struct.pack('!d',v))
        elif v is None or isinstance(v,(str,int,bool)):
            h.update(c.canonical([str(type(v)),v]))
        elif hasattr(v,'__dict__'):
            h.update((type(v).__module__+'.'+type(v).__qualname__).encode());visit(vars(v))
        else:raise TypeError('Unsupported semantic state: '+str(type(v)))
    visit(value);return h.hexdigest()

def rng_state():
    import numpy as np,torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def normalize_saved_rng(value):
    import torch
    out=dict(value)
    if torch.is_tensor(out.get('cuda')):out['cuda']=[out['cuda']]
    return out

def trainable_state(trainer):
    import importlib
    native=importlib.import_module('llava.train.train')
    params=trainer.model.named_parameters()
    adapter=native.get_peft_state_maybe_zero_3(params,'none')
    extra=native.get_peft_state_non_lora_maybe_zero_3(trainer.model.named_parameters())
    return dict(adapter=adapter,projector=extra)

def optimizer_state(trainer):
    engine=trainer.model_wrapped
    c.require(hasattr(engine,'optimizer'),'Actual DeepSpeed optimizer required')
    return engine.optimizer.state_dict()

def snapshot(trainer):
    import torch
    before=digest(rng_state())
    weights=trainable_state(trainer);optimizer=optimizer_state(trainer)
    state=dict(weights=digest(weights),optimizer=digest(optimizer),
        scheduler=digest(trainer.lr_scheduler.state_dict()),rng=before,
        global_step=int(trainer.state.global_step),actual_DS_global_steps=int(trainer.model_wrapped.global_steps),
        DS_skipped_steps=int(getattr(trainer.model_wrapped,'skipped_steps',0)),
        trainable_parameter_count=sum(p.numel() for group in weights.values() for p in group.values()),
        all_weights_finite=all(bool(torch.isfinite(p).all()) for group in weights.values() for p in group.values()))
    c.require(digest(rng_state())==before,'Observation consumed RNG state')
    return state

def compare_tree(a,b,atol=1e-6,rtol=1e-5):
    """All tensor contents compared in bounded chunks; metadata/scalars exact."""
    import numpy as np
    try:import torch
    except ImportError:torch=None
    mismatches=[];max_abs=0.;compared=0;sum_diff2=0.;sum_ref2=0.
    def array(value):
        if torch is not None and torch.is_tensor(value):
            t=value.detach().cpu()
            return t.float().numpy() if str(t.dtype)=='torch.bfloat16' else t.numpy()
        return value
    def visit(x,y,path):
        nonlocal max_abs,compared,sum_diff2,sum_ref2
        tx=torch is not None and torch.is_tensor(x);ty=torch is not None and torch.is_tensor(y)
        if tx!=ty:mismatches.append(path+':tensor_type');return
        if tx:
            if x.shape!=y.shape or x.dtype!=y.dtype:mismatches.append(path+':shape_dtype');return
            x,y=array(x),array(y)
        if isinstance(x,np.ndarray):
            if not isinstance(y,np.ndarray) or x.shape!=y.shape or x.dtype!=y.dtype:mismatches.append(path+':array_layout');return
            xf=x.reshape(-1);yf=y.reshape(-1)
            exact=x.dtype.kind not in 'fc' or path.rsplit('.',1)[-1] in ('step','global_steps','skipped_steps')
            failed=False
            for start in range(0,len(xf),1<<20):
                u=xf[start:start+(1<<20)];v=yf[start:start+(1<<20)]
                if not np.isfinite(u).all() or not np.isfinite(v).all():failed=True;continue
                diff=u.astype(np.float64)-v.astype(np.float64)
                if len(diff):max_abs=max(max_abs,float(np.abs(diff).max()))
                compared+=len(diff);sum_diff2+=float(np.square(diff).sum());sum_ref2+=float(np.square(u.astype(np.float64)).sum())
                if exact:
                    if not np.array_equal(u,v):failed=True
                elif not np.allclose(u,v,atol=atol,rtol=rtol):failed=True
            if failed:mismatches.append(path+':numeric')
        elif isinstance(x,dict):
            if not isinstance(y,dict) or set(x)!=set(y):mismatches.append(path+':keys');return
            for k in sorted(x,key=str):visit(x[k],y[k],path+'.'+str(k))
        elif isinstance(x,(list,tuple)):
            if type(x)!=type(y) or len(x)!=len(y):mismatches.append(path+':sequence');return
            for i,(u,v) in enumerate(zip(x,y)):visit(u,v,path+'.'+str(i))
        elif hasattr(x,'__dict__'):
            if type(x)!=type(y):mismatches.append(path+':object_type');return
            visit(vars(x),vars(y),path+'.state')
        elif type(x)!=type(y) or x!=y:mismatches.append(path+':scalar')
    visit(a,b,'state')
    return dict(pass_numeric=not mismatches,mismatch_count=len(mismatches),mismatches=mismatches[:100],
        max_absolute_error=max_abs,relative_l2_error=math.sqrt(sum_diff2/max(sum_ref2,1e-300)),
        tensor_elements_compared=compared,atol=atol,rtol=rtol)

def portable_weights(checkpoint):
    from pathlib import Path
    import torch
    p=Path(checkpoint)
    if (p/'adapter_model.safetensors').exists():
        from safetensors.torch import load_file
        adapter=load_file(str(p/'adapter_model.safetensors'),device='cpu')
    else:adapter=torch.load(p/'adapter_model.bin',map_location='cpu')
    return dict(adapter=adapter,projector=torch.load(p/'non_lora_trainables.bin',map_location='cpu'))

def native_optimizer(checkpoint):
    from pathlib import Path
    import torch
    paths=list(Path(checkpoint).rglob('*optim_states.pt'))
    c.require(len(paths)==1,'Qualification requires one real ZeRO optimizer shard')
    return torch.load(paths[0],map_location='cpu')['optimizer_state_dict']
