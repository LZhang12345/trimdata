"""Late-binding candidate consumer; never produces selections or release receipts."""
import argparse
from fractions import Fraction
from . import protocol as p

def prepare(s7_ref,output,registration_ref):
 from .launch import require_metadata_registration
 require_metadata_registration(registration_ref)
 s7=p.checked(s7_ref)
 p.require(s7['status']=='COMPLETE_LOGICAL_POOL_AND_SEVEN_FEATURE_FREE_SELECTIONS'
     and set(s7['selections'])==set(p.KEYS)|{'full'} and s7['trainer_or_feature_PASS'] is False,
     'Genuine completed Root/S7 seven-selector handoff required')
 from experiments.root_native_campaign_v7_20261024.selection import authority_eligibility,validate_pool,validate_selection
 with authority_eligibility(s7['eligibility']):
  pool,_,_=validate_pool(s7['pool'],allow_engineering=False)
  p.require(pool['count']==660763 and pool.get('pool_kind')=='all_source_protected_bridge',
      'Only genuine current complete all-source protected population')
  cells=[]
  for key in p.KEYS:
   method='full' if key=='full' else key.split(':')[1]
   fraction='1' if key=='full' else {5:'0.05',10:'0.10',20:'0.20'}[int(key.split(':')[0])]
   entry=s7['selections'][key];selection=validate_selection(entry['selection'],s7['pool'],method,fraction,allow_engineering=False)
   requested=pool['count'] if key=='full' else int(Fraction(fraction)*pool['count'])
   p.require(selection['count']==entry['count']==requested and selection['selection_seed']==1234,
       'Feature-free exact selected-budget/seed mismatch; no top-up/underfill disguise')
   for seed in p.SEEDS:
    if p.cell_id(key,seed) in p.EXCLUDED_IDS:continue
    cells.append(dict(id=p.cell_id(key,seed),site='babel',role='student',selection_key=key,method=method,
        fraction=fraction,training_seed=seed,selection_seed=1234,selection=entry['selection'],
        requested_K=requested,actual_K=selection['count'],expected_horizon=p.horizon(selection['count']),
        providers=dict(pool=s7['pool'],eligibility=s7['eligibility']),
        runtime=p.RUNTIME,qualification_contract=None,outer_authorization=None,
        qualification_pending=True,release_authorized=False,physical_job_id=None))
 p.require(len(cells)==16 and len({v['id'] for v in cells})==16,'Exactly remaining Root16; no full/15/S1 or active duplicates')
 plan=dict(schema='feature-free-production-student-candidate-v1',
     status='PREPARED_LATE_BOUND_STUDENT_CANDIDATES_NOT_RELEASE_AUTHORITY',
     owner='Root',selection_provider_owner='Root/S7',s7_completion=s7_ref,s7_plan=s7['plan'],
     recipe=p.native_config.RECIPE,native_sources=p.n.sources(),adapter_sources=p.sources(),
     eligible_N=pool['count'],cells=cells,no_selection_acquisition=True,actual_submission_count=0,
     candidate_only=True,three_research_rounds_completed=False)
 return p.seal(output,plan)

def main():
 a=argparse.ArgumentParser();a.add_argument('--s7-completion',required=True);a.add_argument('--output',required=True);a.add_argument('--registration',required=True)
 v=a.parse_args();print(prepare(p.desc(v.s7_completion),v.output,p.desc(v.registration)))
if __name__=='__main__':main()
