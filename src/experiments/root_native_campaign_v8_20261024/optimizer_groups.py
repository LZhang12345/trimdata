"""Keep author nonempty groups intact before HF captures LambdaLR indices.

DS0.12.6 otherwise removes empty groups after LambdaLR construction. This repair
removes only empty groups while the native optimizer is fresh, without replacing
the optimizer, any retained dictionary/parameter, or its hyperparameters.
"""
import copy
import math
from . import common as c


def _metadata(group):
    result={}
    for key,value in group.items():
        if key=='params':continue
        c.require(value is None or isinstance(value,(str,bool,int,float,tuple,list)), 'Unexpected optimizer group metadata')
        if isinstance(value,(tuple,list)):
            c.require(all(v is None or isinstance(v,(str,bool,int,float)) for v in value), 'Nested optimizer metadata unsupported')
            value=list(value)
        result[key]=value
    return result


def prune_before_scheduler(optimizer,named_parameters,*,scheduler,main_lr,projector_lr,weight_decay):
    c.require(scheduler is None, 'Cannot prune optimizer groups after a scheduler exists')
    c.require(not optimizer.state, 'Cannot prune optimizer groups after optimizer state exists')
    named={id(p):(name,p) for name,p in named_parameters if p.requires_grad}
    c.require(named and len({name for name,_ in named.values()})==len(named), 'Missing/duplicate named trainables')
    expected=[[],[],[],[]]
    for name,param in named.values():
        c.require(name.endswith('.weight') or name.endswith('.bias'), 'Unqualified trainable parameter kind')
        projector='mm_projector' in name
        c.require(projector or '.lora_A.' in name or '.lora_B.' in name, 'Unexpected native trainable support')
        expected[2*int(projector)+int('bias' in name)].append(name)
    groups=optimizer.param_groups
    c.require(isinstance(groups,list) and len(groups)==4, 'Expected pinned author four-group optimizer')
    c.require(expected[0] and not expected[1] and expected[2] and expected[3], 'Expected author LoRA/empty/projector-weight/projector-bias grouping')
    before=[];seen=[];kept=[];keep_indices=[]
    preserved=[]
    for i,group in enumerate(groups):
        params=group['params'];c.require(isinstance(params,list), 'Expected materialized author parameter list')
        c.require(all(id(p) in named for p in params), 'Foreign/frozen optimizer parameter')
        names=[named[id(p)][0] for p in params]
        c.require(names==expected[i], 'Author group order/named parameter support differs')
        wanted=main_lr if i<2 else projector_lr
        c.require(math.isfinite(float(wanted)) and wanted>0 and group['lr']==wanted, 'Declared semantic group initial LR differs')
        c.require(group['weight_decay']==(weight_decay if i%2==0 else 0.), 'Author group weight decay differs')
        c.require('initial_lr' not in group, 'Scheduler may already have captured optimizer groups')
        before.append(dict(original_index=i,names=names,parameters=[dict(name=n,
            shape=list(getattr(p,'ds_shape',p.shape)),dtype=str(p.dtype)) for n,p in zip(names,params)],
            hyperparameters=_metadata(group)))
        seen.extend(names)
        if params:
            kept.append(group);keep_indices.append(i)
            preserved.append((id(group),[id(p) for p in params],copy.deepcopy(_metadata(group))))
    c.require(len(seen)==len(set(seen))==len(named) and set(seen)=={n for n,_ in named.values()}, 'Optimizer omits/duplicates trainables')
    c.require(keep_indices==[0,2,3], 'Unexpected nonempty original optimizer mapping')
    # In-place list update preserves the optimizer object and every kept group.
    groups[:]=kept
    c.require([(id(g),[id(p) for p in g['params']],_metadata(g)) for g in groups]==preserved,
              'Empty pruning changed nonempty optimizer content')
    return dict(schema='native-pre-scheduler-optimizer-group-inventory-v1',
        status='CAPTURED_AND_PRUNED_EMPTY_GROUPS_BEFORE_SCHEDULER',
        optimizer_class=type(optimizer).__module__+'.'+type(optimizer).__qualname__,
        original_groups=before,kept_original_indices=keep_indices,removed_original_indices=[1],
        actual_group_count=3,all_nonempty_groups_objects_parameters_order_hyperparameters_preserved=True,
        before_scheduler_and_optimizer_state=True,main_initial_lr=main_lr,
        projector_initial_lr=projector_lr,weight_decay=weight_decay)


def model_inventory(model):
    value=getattr(model,'_native_optimizer_group_inventory',None)
    c.require(isinstance(value,dict) and value.get('schema')=='native-pre-scheduler-optimizer-group-inventory-v1'
        and value.get('kept_original_indices')==[0,2,3] and value.get('actual_group_count')==3,
        'Missing actual pre-scheduler named-group inventory')
    return copy.deepcopy(value)
