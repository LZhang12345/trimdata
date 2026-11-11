"""Register 21 actual candidates and all39 baseline paths; never select/acquire."""
import argparse
from . import protocol as p

def register(candidate_ref,output,registration_ref):
 from experiments.root_production_student_adapter_v2_20261025.launch import require_metadata_registration
 require_metadata_registration(registration_ref)
 value=p.checked(candidate_ref);completion=p.checked(value['s7_completion']);s7=p.checked(value['s7_plan'])
 p.require(value['status']=='PREPARED_LATE_BOUND_STUDENT_CANDIDATES_NOT_RELEASE_AUTHORITY'
     and value['native_sources']==p.a.n.sources() and value['adapter_sources']==p.a.sources()
     and len(value['cells'])==21 and value['eligible_N']==s7['eligible_N'],'Exact latebound21 and originalS7 population required')
 p.require(completion['status']=='COMPLETE_LOGICAL_POOL_AND_SEVEN_FEATURE_FREE_SELECTIONS'
     and completion['trainer_or_feature_PASS'] is False and completion['plan']==value['s7_plan']
     and completion['pool']==s7['pool'] and completion['eligibility']==s7['eligibility']
     and set(completion['selections'])==set(p.a.KEYS),'Actual completion, raw frozen plan and seven outputs must agree')
 p.require(s7['schema']=='s7-same-site-baseline-plan-v1' and s7['site']=='babel'
     and s7['status']=='FINAL_POOL_VERIFIED_PREPARATION_NOT_TRAINING_AUTHORITY','Authentic raw frozen S7 plan required')
 prepared={}
 for key,entry in completion['selections'].items():
  selected=p.checked(entry['selection']);method='full' if key=='full' else key.split(':')[1]
  fraction='1' if key=='full' else '0.'+key.split(':')[0].zfill(2)
  expected=s7['eligible_N'] if key=='full' else s7['eligible_N']*int(key.split(':')[0])//100
  p.require(selected['status']=='PASS_SELECTION' and selected['pool']==completion['pool']
      and selected['method']==method and selected['fraction']==fraction and selected['selection_seed']==1234
      and selected['count']==entry['count']==expected,'Exact completion selection metadata changed')
  prepared[key]=entry
 byid={v['id']:v for v in value['cells']};p.require(len(byid)==21,'Duplicate candidate')
 cells=s7['cells']+s7['full_references'];p.require(len(cells)==39 and len({v['id'] for v in cells})==39,'Original39 distinct baseline paths required')
 rows=[];supported=set()
 for cell in cells:
  p.require(cell['site']=='babel' and cell['training_seed'] in p.a.SEEDS and cell['selection_seed']==1234,'Babel finite paired-seed roster only')
  method=cell['method'];key=cell['selection_key'];seed=cell['training_seed']
  p.require(method in ('uniform','source_sqrt','coincide','datatailor','full'),'Unknown baseline')
  expected_id=('s7-babel-full-s'+str(seed)) if method=='full' else 's7-babel-'+method+'-p'+str(cell['percent']).zfill(2)+'-s'+str(seed)+'-q1234'
  p.require(cell['id']==expected_id,'S7 path/factor mismatch')
  candidate=None;selection=cell.get('selection');actual_K=cell.get('actual_K')
  if method in ('uniform','source_sqrt','full'):
   ident=p.a.cell_id(key,seed);candidate=byid[ident];supported.add(ident)
   entry=prepared[key]
   p.require(candidate['selection']==entry['selection'] and candidate['actual_K']==entry['count']
       and candidate['requested_K']==cell['requested_K'] and candidate['fraction']==cell['fraction']
       and candidate['providers']['pool']==completion['pool'] and candidate['providers']['eligibility']==completion['eligibility']
       and candidate['expected_horizon']==p.a.horizon(entry['count']),'Prepared candidate and authenticated completion differ')
   # FINAL_PLAN is deliberately raw/unbound. Merge only into this NEW registry;
   # never mutate or relabel the original plan as a fully prepared39 roster.
   p.require(selection in (None,entry['selection']) and actual_K in (None,entry['count']),'Raw S7 cell conflicts with prepared output')
   selection=entry['selection'];actual_K=entry['count']
  else:p.require(selection is None and actual_K is None,'Unapproved published outputs cannot be promoted by source125')
  rows.append(dict(s7_id=cell['id'],method=method,selection_key=key,training_seed=seed,fraction=cell['fraction'],
      actual_K=actual_K,selection=selection,candidate_id=candidate['id'] if candidate else None,
      status='SUPPORTED_CANDIDATE_REQUIRES_GENUINE_RUNTIME_CONFIG_PERMIT' if candidate else 'BLOCKED_NO_APPROVED125_PUBLISHED_METHOD_MATERIALIZATION',
      launch_authorized=False))
 p.require(supported==set(byid) and len(supported)==21,'Missing or repeated supported paths')
 return p.seal(output,dict(schema='root-production-student-path-registry-v1',owner='Root',candidate_plan=candidate_ref,
     s7_plan=value['s7_plan'],s7_completion=value['s7_completion'],eligible_N=value['eligible_N'],rows=rows,supported_paths=21,blocked_published_paths=18,
     actual_jobs_submitted=0,source_only_registration_not_READINESS=True,launcher_sources=p.sources()))

def main():
 a=argparse.ArgumentParser();a.add_argument('--candidate-plan',required=True);a.add_argument('--registration',required=True);a.add_argument('--output',required=True);v=a.parse_args()
 print(p.a.canonical(register(p.desc(v.candidate_plan),v.output,p.desc(v.registration))).decode())
if __name__=='__main__':main()
