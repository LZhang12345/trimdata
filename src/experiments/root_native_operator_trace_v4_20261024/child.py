"""Fresh observed Native7 process; only diagnostic stopping differs."""
import argparse
import importlib
import os
from pathlib import Path
import sys
import time
import traceback
from . import protocol as p
from experiments.root_native_qualification_v10_20261024.observer import Audit

def run(study,study_ref,cell,physical_ref,alias_ref,attempt,stop_at):
    from .runtime_trace import Trace
    trace=Trace();trace.install()
    from .runtime_trace import InitialWeightProbe
    from experiments.root_native_qualification_v10_20261024 import state as observed_state
    initial_probe=InitialWeightProbe(observed_state).install()
    p.seal(Path(attempt)/'PRE_CUDA_RUNTIME_POLICY.json',trace.before_cuda)
    configs,modules=p.validate(study)
    anchor=p.checked(study['anchor_study'])
    from experiments.root_native_qualification_v10_20261024.preflight import runtime_preflight
    p.seal(Path(attempt)/'CURRENT_TOKENIZER_RUNTIME.json',runtime_preflight(anchor,site='aces'))
    logical=configs[cell];config=p.checked(physical_ref)
    expected=p.h.physical_config(logical,alias_ref,validate_native=modules['config'].validate,run_identity=p.n.run_identity)
    p.require(config==expected and os.environ['NATIVE_RUN_IDENTITY']==p.n.run_identity(config),'Unreviewed physical replay config')
    source=p.ROOT/modules['common'].NATIVE;sys.path.insert(0,str(source))
    patch=importlib.import_module('llava.train.llama_xformers_attn_monkey_patch')
    patch.replace_llama_attn_with_xformers_attn()
    module=importlib.import_module('llava.train.train')
    p.require(Path(module.__file__).resolve()==source/'llava/train/train.py','Foreign native trainer')
    phase='interrupted' if cell=='T1' else 'reference'
    audit=Audit(config,phase,Path(attempt)/'observations',p.CELLS[cell])
    parent=module.LLaVATrainer;module.LLaVATrainer=audit.install(parent)
    old=sys.argv[:];started=time.monotonic();error=None;status='FAILED_NATIVE_ATTEMPT';observation=None
    try:
        import torch
        p.require(torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name()
            and torch.version.cuda=='12.1' and os.environ.get('CUDA_HOME')=='/sw/eb/sw/CUDA/12.1.1',
            'Actual completed H100/CUDA route required')
        p.require(not (Path(config['output'])/'training').exists(),'Fresh process cannot resume a prior control')
        sys.argv=['native7-operator-trace','--config',physical_ref['path'],'--stop-at-epoch',str(stop_at)]
        modules['native'].main()
        if (Path(config['output'])/'TRAIN_COMPLETE.json').exists():status='NATIVE_TRAJECTORY_COMPLETE'
        elif (Path(attempt)/'CONTINUATION.json').exists():status='ACCEPTED_NATIVE_CONTINUATION'
        else:raise ValueError('Native returned without accepted state')
        p.expected_endpoint(cell,status,int(audit.trainer.state.global_step),len(audit.inputs))
    except BaseException as exc:
        error=repr(exc);traceback.print_exc()
    finally:
        module.LLaVATrainer=parent;sys.argv=old
        p.seal(Path(attempt)/'EXECUTED_OPERATOR_TRACE.json',trace.finish())
        p.seal(Path(attempt)/'LR_ZERO_WEIGHT_DETAILS.json',initial_probe.finish())
        try:observation=audit.finish(status)
        except BaseException as exc:error=(error or '')+'; audit: '+repr(exc);traceback.print_exc()
    out=Path(attempt)/'observations'
    files={str(f.relative_to(out)):p.desc(f) for f in sorted(out.rglob('*')) if f.is_file()}
    result=dict(schema='native7-operator-trace-child-v4',status=status,error=error,cell=cell,study_id=p.ID,
        study=study_ref,driver_sources=p.sources(),native_target=study['native_target'],
        native_configuration=study['native_configurations'][cell],physical_configuration=physical_ref,
        runtime_alias=alias_ref,initial_weight_probe=p.desc(Path(attempt)/'LR_ZERO_WEIGHT_DETAILS.json'),executed_operator_trace=p.desc(Path(attempt)/'EXECUTED_OPERATOR_TRACE.json'),actual_job_id=os.environ['SLURM_JOB_ID'],
        actual_restart_count=int(os.environ['SLURM_RESTART_COUNT']),elapsed_seconds=time.monotonic()-started,
        attempt=str(attempt),observation=observation,files=files,
        native_horizon=8,diagnostic_endpoint=p.CELLS[cell],no_restore_qualification=True)
    p.seal(Path(attempt)/'CHILD_RESULT.json',result)
    p.require(error is None,'Native replay attempt failed; preserve evidence and diagnose')
    return result

def main():
    a=argparse.ArgumentParser()
    for name in ('study','cell','physical-config','alias','attempt','stop-at-epoch'):a.add_argument('--'+name,required=True)
    args=a.parse_args();run(p.read(args.study),p.desc(args.study),args.cell,p.desc(args.physical_config),
        p.desc(args.alias),args.attempt,int(args.stop_at_epoch))

if __name__=='__main__':main()
