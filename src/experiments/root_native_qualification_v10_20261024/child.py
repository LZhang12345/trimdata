"""One observed native attempt; no changed training objective or ad-hoc resume."""
import argparse
import importlib
import os
from pathlib import Path
import sys
import time
import traceback
from . import common as c
from . import study as validation
from .observer import Audit

def run(study,arm,phase,attempt,stop_at_epoch):
    from .preflight import runtime_preflight
    runtime=runtime_preflight(study,site='babel')
    c.seal(Path(attempt)/'CURRENT_TOKENIZER_RUNTIME.json',runtime)
    configs,modules=validation.validate_study(study);config=configs[arm]
    c.require((arm,phase) in (('reference','reference'),('resume','interrupted'),('resume','resumed')),'Unknown engineering arm/phase')
    output=Path(attempt)/'observations';output.mkdir(parents=True,exist_ok=False)
    c.require(os.environ['NATIVE_RUN_IDENTITY']==validation.native_identity(config,modules),'Native function identity differs')
    native_common=modules['common'];source=c.ROOT/native_common.NATIVE
    sys.path.insert(0,str(source))
    # Same required import order as actual native trainer; native.main reasserts it.
    patch=importlib.import_module('llava.train.llama_xformers_attn_monkey_patch')
    patch.replace_llama_attn_with_xformers_attn()
    module=importlib.import_module('llava.train.train')
    c.require(Path(module.__file__).resolve()==source/'llava/train/train.py','Foreign native trainer import')
    parent=module.LLaVATrainer;audit=Audit(config,phase,output,study['interrupt_step'])
    module.LLaVATrainer=audit.install(parent)
    old_argv=sys.argv[:];started=time.monotonic();error=None;status='FAILED_NATIVE_ATTEMPT'
    try:
        sys.argv=['engineering-observed-native','--config',study['native_configurations'][arm]['path'],
            '--stop-at-epoch',str(stop_at_epoch)]
        modules['native'].main()
        final=Path(config['output'])/'TRAIN_COMPLETE.json';continuation=Path(attempt)/'CONTINUATION.json'
        if final.exists():status='NATIVE_TRAJECTORY_COMPLETE'
        elif continuation.exists():status='ACCEPTED_NATIVE_CONTINUATION'
        else:raise ValueError('Native returned without complete or accepted continuation')
    except BaseException as exc:
        error=repr(exc);traceback.print_exc()
    finally:
        module.LLaVATrainer=parent;sys.argv=old_argv
        try:observation=audit.finish(status)
        except BaseException as exc:
            observation=None;error=(error or '')+'; observation finalization: '+repr(exc);traceback.print_exc()
    files={str(p.relative_to(output)):c.desc(p) for p in sorted(output.rglob('*')) if p.is_file()}
    result=dict(status=status,arm=arm,phase=phase,study_id=study['id'],driver_sources=study['driver_sources'],
        native_target=study['native_target'],native_configuration=study['native_configurations'][arm],
        actual_job_id=os.environ['SLURM_JOB_ID'],actual_restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0')),
        attempt=str(Path(attempt).absolute()),observation=observation,files=files,error=error,
        elapsed_seconds=time.monotonic()-started,actual_result_requires_independent_review=True)
    c.seal(Path(attempt)/'CHILD_RESULT.json',result)
    c.require(error is None,'Observed native attempt failed; preserve all evidence')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--arm',choices=['reference','resume'],required=True)
    p.add_argument('--phase',choices=['reference','interrupted','resumed'],required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--stop-at-epoch',type=int,required=True);a=p.parse_args()
    run(c.read(a.study),a.arm,a.phase,a.attempt,a.stop_at_epoch)

if __name__=='__main__':main()
