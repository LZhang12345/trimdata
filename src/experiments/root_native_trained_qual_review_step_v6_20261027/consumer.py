"""Explicit completed-step reviewer contract; never a fictitious batch prerequisite.

Root may call this from a separately frozen current12 physical consumer adapter.
The qualifier and trained-student batch dependencies remain unchanged. This
materialized reviewer role adds no afterok dependency on its running parent.
"""
from . import worker as w

COMPLETION='ROOT_AUTHENTICATED_COMPLETED_INDEPENDENT_REVIEW_STEP_V6'


def accounting(raw,identifier,cpus,node):
    rows=[line.split('|')for line in raw.splitlines()if line.strip()]
    rows=[r for r in rows if len(r)>=2 and r[0]==identifier and r[1]==identifier]
    w.require(len(rows)==1,'Exactly one real numeric registered step accounting row required')
    row=rows[0]
    w.require(len(row)==12 and row[2:4]==['COMPLETED','0:0'] and row[4]in('','luyangz') and row[5]=='cx' and row[6]in('','general') and row[7]in('','normal')
        and row[8]==str(cpus) and row[9].isdecimal() and int(row[9])>0 and row[11]==node,
        'Actual registered step must complete zero on its own allocated CPU route')
    tres=dict(x.split('=',1)for x in row[10].split(',')if'='in x)
    w.require(tres.get('cpu')==str(cpus),'Actual registered step CPU allocation differs')
    return dict(registered_step=identifier,allocated_CPUs=cpus,elapsed_seconds=int(row[9]),node=node,raw=raw,
        memory_is_shared_parent=True,no_dedicated_step_memory_claim=True,
        inherited_identity_fields=[name for index,name in ((4,'User'),(6,'Partition'),(7,'QOS'))if row[index]==''])


def materialized_review(cfg,value,qref,contract,load=w.checked):
    w.require(contract['schema']=='materialized-independent-review-registered-step-v6'
        and contract['parent_afterok_dependency_required']is False,'Explicit materialized registered-step review contract required')
    binding=load(contract['binding']);reviewcfg=load(binding['configuration']);peer=load(binding['independent_source_review'])
    w.require(binding['status']=='ROOT_BOUND_INDEPENDENT_TRAINED_QUAL_REGISTERED_STEP_V6'
        and binding['owner']=='Root' and binding['parent_job_id']=='10942711'
        and binding['registered_step'].startswith('10942711.')
        and binding['registered_step'].split('.')[1].isdecimal()
        and int(binding['registered_step'].split('.')[1])>0
        and reviewcfg['status']=='ROOT_CONFIGURED_INDEPENDENT_TRAINED_QUAL_REGISTERED_STEP_V6'
        and reviewcfg['subject_job_id']==cfg['required_loader_qualifier_job_id']=='10946795'
        and reviewcfg['subject_configuration']==cfg['qualified_loader_configuration']
        and peer['status']==w.SOURCE_REVIEW and peer['independent']is True
        and peer['reviewer']not in('',w.AUTHOR,w.REVIEWER)
        and peer['configuration']==binding['configuration'] and peer['sources']==reviewcfg['sources'],
        'Exact independently reviewed actual step route required')
    ownership=load(binding['ownership']);intent=load(binding['launch_intent'])
    w.require(ownership['registered_step']==binding['registered_step'] and ownership['configuration']==binding['configuration']
        and ownership['owner']=='Root' and ownership['no_competing_actual_readout_reviewer']is True
        and intent['status']=='ROOT_REGISTERED_STEP_LAUNCH_INTENT_V6' and intent['owner']=='Root'
        and intent['configuration']==binding['configuration'] and intent['independent_source_review']==binding['independent_source_review']
        and intent['parent_binding']==reviewcfg['parent_binding'] and intent['nonce']==binding['nonce']==reviewcfg['nonce']
        and intent['step_cpus']==4 and intent['no_competing_actual_readout_reviewer']is True,'Actual step launch/ownership chain differs')
    completion=load(contract['completion']);reviewref=contract['review']
    w.require(load(reviewref)==value and completion['status']==COMPLETION and completion['owner']=='Root'
        and completion['binding']==contract['binding'] and completion['review']==reviewref
        and completion['configuration']==binding['configuration'] and completion['srun_returncode']==0,
        'Real materialized review and independently authenticated completed-step receipt required')
    actual=value['actual_independent_reviewer'];node=actual['node']
    parent=load(reviewcfg['parent_binding']);pf=w.fields(actual['scheduler'])
    w.require(parent['schema']=='root-fulltrajectory-held-binding-v2' and parent['owner']=='Root' and parent['job_id']=='10942711'
        and parent['nonce']==reviewcfg['parent_nonce'] and pf['JobId']=='10942711' and pf['UserId']=='luyangz(2708899)'
        and pf['Account']=='cx' and pf['Partition']=='general' and pf['QOS']=='normal' and pf['NodeList']==node
        and actual['parent_binding']==reviewcfg['parent_binding'], 'Inherited step identity must come from exact authenticated parent')
    w.inspect_step(actual['step_scheduler'],binding['registered_step'],4,node)
    observed=accounting(completion['raw_sacct'],binding['registered_step'],4,node)
    w.require(completion['accounting']==observed and actual['registered_step']==binding['registered_step']
        and actual['route']=='REGISTERED_CPU_STEP_WITHIN_EXISTING_OWNED_GPU_ALLOCATION_V6'
        and actual['job_id']=='10942711' and actual['step_CPUs']==4
        and actual['parent_allocated_CPUs']==8 and actual['parent_allocated_memory_mb']==65536
        and actual['parent_allocated_GPUs']==1 and actual['memory_is_shared_parent']is True
        and actual['models']==actual['neural_generations']==0,'Actual completed step actor differs')
    w.require(value['status']==w.PASS and value['independent']is True and value['reviewer']==w.REVIEWER
        and value['configuration']==cfg['qualified_loader_configuration'] and value['readout']==qref
        and value['subject_actual_accounting']['job_id']=='10946795'
        and value['subject_neural_generations_verified']==4 and value['distinct_QAs']==2
        and all(value[k]is True for k in('all_saved_checkpoint_payload_SHA_verified',
            'actual_trained_tensor_and_persistent_buffer_guards_verified','actual_pinned_PEFT071_bridge_four_calls_verified',
            'independent_prompt_token_RGB_processor_recomputation','no_full_bank_or_student_improvement_claim',
            'completion_requires_separate_actual_sacct_step_receipt'))
        and value['registered_step_route_version']=='V6' and value['registered_step_binding']==contract['binding']
        and value['independent_review_configuration']==binding['configuration'],'Complete original independent V4 verification plus actual V6 route required')
    return dict(binding=contract['binding'],completion=contract['completion'],review=reviewref,accounting=observed,
        parent_afterok_dependency_required=False,batch_ACK_not_applicable_to_registered_step=True)
