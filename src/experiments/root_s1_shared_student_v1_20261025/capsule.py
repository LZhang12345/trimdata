"""Finite ownedGPU semantic capsule checks and conservative own-byte credit."""
import argparse,os,stat,time
from pathlib import Path
from . import protocol as p

def check(binding_ref,activation_ref,output):
 from .driver import current
 b,request,actual,a=current(binding_ref,activation_ref,minimum_seconds=30);cfg=p.checked(a['configuration']);identity=p.n.run_identity(cfg)
 training=Path(cfg['output'])/'training';latest=0;rows=[];allocated=0;inodes=set();start=time.time()
 from experiments.root_native_campaign_v7_20261024.native import checkpoint_receipt
 for path in sorted(training.glob('checkpoint-*')):
  p.require(not path.is_symlink() and path.name[len('checkpoint-'):].isdecimal() and (path/'ACCEPTED.json').is_file(),
      'Unaccepted/foreign capsule requires separate Root reconciliation, never credit or replay')
  receipt=checkpoint_receipt(path,identity);step=receipt['global_step'];p.require(path.name=='checkpoint-'+str(step),'Capsule cursor/path mismatch')
  own=0
  for name,ref in receipt['files'].items():
   file=path/name;st=file.lstat();p.require(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_uid==os.getuid() and not file.is_symlink(),
       'Only actual sameconfig owned singlelink capsule bytes may reduce unwritten allowance')
   inode=(st.st_dev,st.st_ino);p.require(inode not in inodes,'Duplicate inode byte credit');inodes.add(inode)
   own+=min(st.st_size,st.st_blocks*512)
  allocated+=own;latest=max(latest,step);rows.append(dict(checkpoint=str(path),accepted=p.desc(path/'ACCEPTED.json'),logical_bytes=receipt['logical_bytes'],allocated_lower_bound=own))
 storage=p.checked(cfg['storage_reservation']);total=storage['checkpoint_transaction_bytes']*(storage['retained_accepted_states']+1)+p.TRANSIENT_BYTES
 p.require(storage['checkpoint_transaction_bytes']==p.CAPSULE_BYTES and storage['retained_accepted_states']==2
     and storage['allow_new_intermediate_retirement'] is True and len(rows)<=2 and total==21<<30,
     'Exact Native91 twoaccepted+transaction+6GiB retention envelope required')
 other=storage['other_outstanding_reserved_bytes'];p.require(type(other) is int and other>=0,'Explicit conservative other unpaid claims required')
 fs=os.statvfs(p.checked(cfg['site_profile'])['output_root']);required=max(0,total-allocated)+other
 p.require(fs.f_bavail*fs.f_frsize>=required and fs.f_favail>=4000,'Actual current collective remaining checkpoint allowance cannot fit')
 return p.seal(output,dict(status='ACTUAL_OWNED_GPU_CAPSULE_REVALIDATION_NOT_TRAINING_QUALIFICATION',configuration=a['configuration'],config_identity=identity,
     physical_job_id=b['job_id'],actual_restart=actual['restart'],actual_node=actual['node'],actual_scheduler=actual['scheduler'],latest_step=latest,capsules=rows,
     own_materialized_allocated_lower_bound=allocated,total_claim_bytes=total,other_outstanding_reserved_bytes=other,required_unwritten_bytes=required,
     available_bytes=fs.f_bavail*fs.f_frsize,available_inodes=fs.f_favail,seconds=time.time()-start,
     checkpoint_tensor_payloads_loaded_for_semantic_validation=True,base_model_constructed=False,neural_forward=False))

def main():
 a=argparse.ArgumentParser()
 for name in ('binding','activation','output'):a.add_argument('--'+name,required=True)
 v=a.parse_args();print(p.canonical(check(p.desc(v.binding),p.desc(v.activation),v.output)).decode())
if __name__=='__main__':main()
