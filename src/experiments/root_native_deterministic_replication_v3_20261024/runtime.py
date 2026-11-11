"""Runtime is an explicit factor, never an implicit extension of Native7 PASS."""
import os
from . import protocol as p
FLAG_NAMES=('CUBLAS_WORKSPACE_CONFIG','deterministic_algorithms','deterministic_warn_only','matmul_allow_tf32','cudnn_allow_tf32','cudnn_benchmark','cudnn_deterministic')
def observed_policy(torch):
    from experiments.root_native_deterministic_attention_v1_20261024.policy import observed_flags
    return observed_flags(torch)
def launch_environment(base,pre,cell):
    env=dict(base)
    value=':4096:8'  # Exact inherited paid V10 wrapper and explicit D8 policy.
    if value is None:env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    else:env['CUBLAS_WORKSPACE_CONFIG']=value
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    for name in ('PYTHONHOME','PYTHONPATH','LD_PRELOAD'):env.pop(name,None)
    return env

def install(cell,pre):
    """Must run in fresh child before native construction/any CUDA init."""
    import torch
    p.require(not torch.cuda.is_initialized(),'Runtime control installed after CUDA initialization')
    expected={name:pre[name] for name in FLAG_NAMES};expected['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    policy=None
    if cell=='D8':
        from experiments.root_native_deterministic_attention_v1_20261024.policy import install as force
        policy,attestation=force()
    else:
        p.require(cell=='F8' and observed_policy(torch)==expected,'Fresh replay defaults differ from actual authentic trace; no silent flag repair')
        p.require('export CUBLAS_WORKSPACE_CONFIG=:4096:8' in (p.b.HERE/'run.sbatch').read_text(),'Paid V10 launch-policy source differs')
        attestation=dict(policy='paid_v10_native_dispatch',actual=observed_policy(torch),CUDA_initialized=False,math_changes=False,paid_V10_wrapper=p.desc(p.b.HERE/'run.sbatch'),trace_CUBLAS_WORKSPACE_CONFIG=pre.get('CUBLAS_WORKSPACE_CONFIG'),trace_to_paid_CUBLAS_difference_explicit=True)
    from experiments.root_babel_operator_trace_v2_20261024.runtime_trace import Trace
    trace=Trace().install()
    return policy,trace,attestation

def require_current(policy,pre,cell):
    import torch
    if cell=='D8':
        from experiments.root_native_deterministic_attention_v1_20261024.policy import assert_flags
        assert_flags(torch)
    else:
        # Native recipe explicitly disables TF32 at model construction. All
        # other process policy must remain that of the measured native trace.
        expected={name:pre[name] for name in FLAG_NAMES};expected['CUBLAS_WORKSPACE_CONFIG']=':4096:8';expected.update(matmul_allow_tf32=False)
        p.require(observed_policy(torch)==expected,'Native replay runtime policy drifted before training')

def finish(policy,trace):
    # Reverse install order: concrete apply wrappers, then explicit API policy.
    actual=trace.finish();forced=policy.finish() if policy else None
    if forced:p.require(forced['calls']>0 and forced['supported_layouts'],'Explicit operator was not exercised')
    return dict(executed_operator_trace=actual,explicit_policy=forced,scope='ACTUAL_RUNTIME_OBSERVATION_NOT_AUTOMATIC_PRODUCTION_QUALIFICATION')
