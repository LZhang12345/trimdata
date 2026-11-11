"""Declared D8 policy, source-pinned and mandatory before CUDA initialization."""
import os
from . import protocol as p

def launch_environment(base):
 env=dict(base,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
 for name in ('PYTHONHOME','PYTHONPATH','LD_PRELOAD'):env.pop(name,None)
 return env

def install():
 import torch
 p.require(not torch.cuda.is_initialized() and torch.__version__=='2.1.2+cu121','Exact Torch/preCUDA process required')
 from experiments.root_native_deterministic_attention_v1_20261024.policy import install as explicit
 policy,attestation=explicit()
 from experiments.root_babel_operator_trace_v2_20261024.runtime_trace import Trace
 return policy,Trace().install(),attestation

def require_current():
 import torch
 from experiments.root_native_deterministic_attention_v1_20261024.policy import assert_flags
 return assert_flags(torch)

def finish(policy,trace):
 observed=trace.finish();forced=policy.finish();p.require(forced['calls']>0 and forced['supported_layouts'],'No actual explicit operator exercise')
 return dict(executed_operator_trace=observed,explicit_policy=forced,scope='ACTUAL_DECLARED_RUNTIME_NOT_AUTOMATIC_PROMOTION')
