"""Project exact paid21 metadata and materialize one already-held trajectory."""
import copy
from pathlib import Path
from . import protocol as p,materialize
from experiments.root_fulltrajectory_held_cohort_v2_20261025 import protocol as h

def project(original_ref,output,source_review_ref):
 original=p.checked(original_ref);review=p.checked(source_review_ref)
 p.require(original_ref['sha256']==h.ORIGINAL_SHA and original['status']=='PREPARED_LATE_BOUND_STUDENT_CANDIDATES_NOT_RELEASE_AUTHORITY'
     and original['native_sources']==p.n.sources() and len(original['cells'])==21,'Exact original paid21 required')
 p.require(review['status']=='PASS_INDEPENDENT_LONGWALL_STUDENT_SOURCE_V2' and review['sources']==p.sources()
     and review.get('reviewer') not in (None,'',p.AUTHOR),'Genuine independent complete longwall source review required')
 value=copy.deepcopy(original);value['adapter_sources']=p.sources();value['longwall_metadata_ancestry']=dict(original=original_ref,source_review=source_review_ref,checkpoint_identity_migration=False)
 return p.seal(output,value)

def draft(binding_ref,plan_ref,profile_ref,providers,storage_ref,output,report,natural_evidence,compatibility_ref,registration_ref):
 b=p.checked(binding_ref);request,cell,_=h.reviewed(b['request'])
 p.require(request['site']=='babel' and providers['pool']==cell['providers']['pool'] and providers['selection']==cell['selection'],
     'Exact existing conditional cohort scientific population')
 return materialize.draft(plan_ref,cell['id'],profile_ref,providers,storage_ref,output,report,natural_evidence,registration_ref,
     allocation_minutes=request['outer_allocation']['minutes'],runtime_compatibility=compatibility_ref)
