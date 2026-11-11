"""OwnedGPU checkpoint-tensor semantic validation; no base-model construction/NN."""
import argparse,time
from pathlib import Path
from . import protocol as p

def check(binding_ref,path,output):
 from .worker import allocation
 b=p.checked(binding_ref);request=p.checked(b['request']);cfg,_,_,_=p.metadata(request)
 p.validate_binding(b,b['request'],request);actual=allocation(b,cfg,minimum_seconds=30)
 path=Path(path);p.lexical(path,Path(cfg['output'])/'training')
 p.require(path.parent==Path(cfg['output'])/'training' and path.name.startswith('checkpoint-')
     and path.name[len('checkpoint-'):].isdecimal() and not any(x.is_symlink() for x in (path,*path.parents)),'Exact literal own checkpoint only')
 p.literal_home(output);p.lexical(output,Path(cfg['report'])/'ROOT_LAUNCH/physical-checks');start=time.time()
 from experiments.root_native_campaign_v7_20261024.native import checkpoint_receipt
 accepted=checkpoint_receipt(path,p.a.n.run_identity(cfg))
 return p.seal(output,dict(status='ACTUAL_OWNED_GPU_CAPSULE_REVALIDATION_NOT_TRAINING_QUALIFICATION',
     physical_job_id=b['job_id'],actual_restart=actual['restart'],actual_scheduler=actual['scheduler'],configuration=b['configuration'],
     config_identity=p.a.n.run_identity(cfg),checkpoint=str(path),accepted=accepted,seconds=time.time()-start,
     CUDA_guard_process_exits_before_training=True,checkpoint_tensor_payloads_loaded_for_semantic_validation=True,
     base_model_constructed=False,neural_forward=False))

def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--output',required=True);v=a.parse_args()
 print(p.a.canonical(check(p.desc(v.binding),v.checkpoint,v.output)).decode())
if __name__=='__main__':main()
