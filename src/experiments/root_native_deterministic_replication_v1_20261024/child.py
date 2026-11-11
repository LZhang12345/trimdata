"""One fully observed native trajectory; explicit runtime before CUDA init."""
import argparse,importlib,os,sys,time,traceback
from pathlib import Path
from . import protocol as p,runtime
from experiments.root_native_qualification_v10_20261024.observer import Audit

def run(study,study_ref,arm,phase,attempt,stop_at):
    attempt=Path(attempt);raw,pre=p.completed_trace(study['completed_trace'],study['cell'])
    policy,trace,attestation=runtime.install(study['cell'],pre)
    p.seal(attempt/'PRE_CUDA_RUNTIME_POLICY.json',attestation)
    configs,modules=p.validate_study(study,gpu=False);cfg=configs[arm]
    p.require((arm,phase) in (('reference','reference'),('resume','interrupted'),('resume','resumed')),'Unknown arm/phase')
    p.require(os.environ['NATIVE_RUN_IDENTITY']==p.n.run_identity(cfg),'Exact config/run identity differs')
    anchor=p.checked(study['anchor_study'])
    from experiments.root_native_qualification_v10_20261024.preflight import runtime_preflight
    p.seal(attempt/'CURRENT_TOKENIZER_RUNTIME.json',runtime_preflight(anchor,site='babel'))
    source=p.ROOT/modules['common'].NATIVE;sys.path.insert(0,str(source))
    patch=importlib.import_module('llava.train.llama_xformers_attn_monkey_patch');patch.replace_llama_attn_with_xformers_attn()
    module=importlib.import_module('llava.train.train');p.require(Path(module.__file__).resolve()==source/'llava/train/train.py','Foreign native trainer')
    parent=module.LLaVATrainer;audit=Audit(cfg,phase,attempt/'observations',4);observed=audit.install(parent)
    class BudgetObserved(observed):
        def train(self,*args,**kwargs):
            # NativeTrainer intercepts its constructor first; this method is
            # reached only by actual training, AFTER cold asset validation.
            runtime.require_current(policy,pre,study['cell'])
            p.require(stop_at-time.time()>=study['minimum_phase_seconds'][phase],'Insufficient actual post-validation walltime for declared phase; no partial reference')
            return super().train(*args,**kwargs)
    module.LLaVATrainer=BudgetObserved;old=sys.argv[:];started=time.monotonic();error=None;status='FAILED_NATIVE_ATTEMPT';observation=None;operator=None
    try:
        import torch
        p.require(torch.cuda.device_count()==1 and 'L40S' in torch.cuda.get_device_name() and torch.version.cuda=='12.1','Exact L40S/CUDA12.1 required')
        training=Path(cfg['output'])/'training'
        if phase!='resumed':p.require(not training.exists(),'Fresh arm must not reuse prior training state')
        sys.argv=['finite-followon-native7','--config',study['native_configurations'][arm]['path'],'--stop-at-epoch',str(stop_at)]
        # This unchanged Native7 main invokes its ACTUAL config.validate, all
        # assets/pool/eligibility/selection/ownership checks before any forward.
        modules['native'].main()
        if (Path(cfg['output'])/'TRAIN_COMPLETE.json').exists():status='NATIVE_TRAJECTORY_COMPLETE'
        elif (attempt/'CONTINUATION.json').exists():status='ACCEPTED_NATIVE_CONTINUATION'
        else:raise ValueError('Native returned without accepted state')
    except BaseException as exc:error=repr(exc);traceback.print_exc()
    finally:
        module.LLaVATrainer=parent;sys.argv=old
        try:operator=p.seal(attempt/'ACTUAL_RUNTIME_OPERATOR.json',runtime.finish(policy,trace))
        except BaseException as exc:error=(error or '')+'; runtime finalization: '+repr(exc);traceback.print_exc()
        try:observation=audit.finish(status)
        except BaseException as exc:error=(error or '')+'; audit: '+repr(exc);traceback.print_exc()
    out=attempt/'observations';files={str(x.relative_to(out)):p.desc(x) for x in sorted(out.rglob('*')) if x.is_file()}
    result=dict(status=status,error=error,arm=arm,phase=phase,study_id=study['id'],study=study_ref,driver_sources=p.sources(),native_target=study['native_target'],native_configuration=study['native_configurations'][arm],actual_job_id=os.environ['SLURM_JOB_ID'],actual_restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0')),attempt=str(attempt),observation=observation,executed_runtime=operator,files=files,elapsed_seconds=time.monotonic()-started,qualification_promoted=False)
    p.seal(attempt/'CHILD_RESULT.json',result);p.require(error is None,'Failed observed attempt preserved; no blind retry');return result

def main():
    a=argparse.ArgumentParser()
    for name in ('study','arm','phase','attempt'):a.add_argument('--'+name,required=True)
    a.add_argument('--stop-at-epoch',required=True,type=int);v=a.parse_args();run(p.read(v.study),p.desc(v.study),v.arm,v.phase,v.attempt,v.stop_at_epoch)
if __name__=='__main__':main()
