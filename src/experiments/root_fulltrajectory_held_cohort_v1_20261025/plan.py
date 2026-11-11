"""Finite prospective requests; authoring does not reserve storage or submit."""
import argparse
from pathlib import Path
from . import protocol as p

def prepare(parent_ref,profile_ref,scheduler_ref,timing_ref,control_root,output):
 parent=p.c.checked(parent_ref);profile=p.c.checked(profile_ref);site=profile['site']
 p.c.require(parent_ref['sha256']==p.ORIGINAL_SHA and site in p.SITES,'Exact paid parent and genuine site profile')
 rows=[]
 for cell in parent['cells']:
  if cell['method'] not in ('uniform','source_sqrt'):continue
  if site=='babel' and cell['id']=='production-student-v1-5-uniform-s17':continue
  percent=int(cell['selection_key'].split(':')[0]);minutes=(p.BABEL_MINUTES if site=='babel' else p.ACES_MINUTES)[percent]
  design=dict(method=cell['method'],fraction=cell['fraction'],training_seed=cell['training_seed'],selection_seed=1234,
      selected_count=cell['actual_K'],selection=cell['selection'],logical_pool=cell['providers']['pool'],logical_eligibility=cell['providers']['eligibility'],epoch=1,max_steps=-1,global_batch=128,replicate_site=site)
  request=dict(schema='root-fulltrajectory-held-request-v1',status='CONDITIONAL_HELD_ONLY_NOT_EXECUTION_AUTHORITY',owner='Root',
      sources=p.sources(),native_sources=p.c.sources(),recipe=p.ncfg.RECIPE,site=site,parent_candidate_plan=parent_ref,
      original_cell_id=cell['id'],scientific_design=design,site_profile=profile_ref,scheduler_time_evidence=scheduler_ref,timing_basis=timing_ref,
      outer_allocation=dict(gpus=1,cpus=8,memory_mb=65536,minutes=minutes),
      control_root=str(Path(control_root)/cell['id']),reserved_storage_bytes=0,release_authorized=False,
      future_release_gates=['actual_matching_natural_runtime','actual_full_source_student_inputs','actual_full_selected_canonical_pilot','reviewed_longwall_runtime_entrypoint','actual_collective_storage_reservation','exact_config_and_outer_permit_reviews'],review=None)
  p.design(request)
  ref=p.c.seal(Path(output)/cell['id']/'REQUEST_DRAFT.json',request)
  rows.append(dict(cell_id=cell['id'],request_draft=ref,minutes=minutes,status='UNREGISTERED_REQUIRES_GENUINE_HELD_DESIGN_REVIEW',physical_job_id=None))
 p.c.require(len(rows)==(17 if site=='babel' else 18),'Finite existing-source cohort only')
 return p.c.seal(Path(output)/'ROSTER_DRAFT.json',dict(schema='root-fulltrajectory-held-roster-v1',site=site,rows=rows,
     request_count=len(rows),proposed_single_attempt_GPU_hours=sum(x['minutes'] for x in rows)/60,
     submitted_GPU_hours=0,actual_jobs_submitted=0,all_requests_unregistered=True,release_authorized=False,
     storage_reserved_bytes=0,full_epoch_completion_not_claimed=True,
     active_canonical_Babel_uniform5_seed17_excluded=(site=='babel')))

def main():
 a=argparse.ArgumentParser();a.add_argument('--parent',required=True);a.add_argument('--profile',required=True);a.add_argument('--scheduler-time',required=True);a.add_argument('--timing',required=True);a.add_argument('--control-root',required=True);a.add_argument('--output',required=True);v=a.parse_args()
 print(p.c.canonical(prepare(p.c.desc(v.parent),p.c.desc(v.profile),p.c.desc(v.scheduler_time),p.c.desc(v.timing),v.control_root,v.output)).decode())
if __name__=='__main__':main()
