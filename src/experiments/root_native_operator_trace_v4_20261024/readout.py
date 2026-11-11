"""Compact actual operator evidence, never a numerical qualification."""
from pathlib import Path
from . import protocol as p


def collect(study_ref):
    study=p.checked(study_ref);pair,_=p.validate(study)
    cfg=pair['T1'];receipt=p.read(Path(cfg['report'])/'CELL_COMPLETE.json')
    p.require(receipt['cell']=='T1' and receipt['study']==study_ref,'Foreign trace result')
    child=p.checked(receipt['child']);trace=p.checked(child['executed_operator_trace'])
    p.require(child['actual_job_id']==study['jobs']['T1'] and child['actual_restart_count']==0
        and child['driver_sources']==p.sources() and child['native_configuration']==study['native_configurations']['T1'],
        'Actual trace identity differs')
    p.require(trace['observational_only'] and trace['dispatch_arguments_unmodified']
        and trace['forward_calls']>0 and trace['backward_calls']>0,'No real observed forward/backward')
    o=p.checked(child['observation']);end=p.checked(child['files']['STEP-1.json'])
    p.expected_endpoint('T1',child['status'],end['global_step'],o['actual_record_exposure'])
    p.require(o['mask_failures']==0 and end['DS_skipped_steps']==0 and end['all_weights_finite'],
        'Trace did not complete finite unmasked native update')
    return p.seal(Path(study['report'])/'RAW_OPERATOR_TRACE.json',dict(
        status='ACTUAL_CONCRETE_OPERATOR_OBSERVATION_NOT_REPAIR_OR_QUALIFICATION',study=study_ref,
        child=receipt['child'],operator_trace=child['executed_operator_trace'],executed=trace['executed'],
        initial_weight_probe=child['initial_weight_probe'],pre_CUDA_policy=trace['pre_CUDA_policy'],native_target=study['native_target'],driver_sources=p.sources(),
        qualification_promoted=False,no_causal_kernel_claim=True,no_stochastic_variance_bound=True,
        all_in_cost_requires_actual_scheduler_accounting=True))
