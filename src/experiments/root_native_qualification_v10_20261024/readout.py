"""Actual raw parity readout; promotion requires a distinct independent review."""
import argparse
import json
from pathlib import Path
from . import common as c
from . import study as validation
from . import state
from .observer import validate_lr_trajectory

def compare_learning_rate_trajectories(paths,total,cut):
    rows={name:[json.loads(line) for line in (path/'LEARNING_RATES.jsonl').read_text().splitlines()]
        for name,path in paths.items()}
    checks=dict(reference=validate_lr_trajectory(rows['reference'],0,total,total),
        interrupted=validate_lr_trajectory(rows['interrupted'],0,cut,total),
        resumed=validate_lr_trajectory(rows['resumed'],cut,total,total))
    boundary_equal=rows['interrupted'][-1]==rows['resumed'][0]
    combined=rows['interrupted']+rows['resumed'][1:]
    exact=rows['reference']==combined
    c.require(boundary_equal and exact,'Actual semantic LR trajectory differs across physical restart')
    return dict(passed=True,phase_checks=checks,boundary_exact=boundary_equal,
        reference_and_restarted_curve_exact=exact,
        declared_main_LR=2e-4,declared_projector_weight_and_bias_LR=2e-5)


def verified_phase(descriptor):
    value=c.checked(descriptor)
    c.require(value['error'] is None,'Failed child cannot qualify')
    for item in value['files'].values():c.require(c.desc(item['path'])==item,'Raw child evidence changed')
    return value


def check_behavior(observation):
    gradients=observation['gradient_support'];attention=observation['actual_attention_forward_calls']
    return dict(actual_forward_backward=observation['actual_microbatches']>0
            and len(gradients)==8 and all(g['finite_calls']>0 and g['nonzero_finite_calls']>0 for g in gradients.values()),
        actual_xformers_attention=len(attention)==32 and all(count>0 for count in attention.values())
            and observation['actual_xformers_kernel_calls']>0 and observation['noncausal_kernel_calls']==0,
        actual_masking=observation['masks_checked']==observation['actual_record_exposure']
            and observation['mask_failures']==0 and observation['expanded_assistant_targets']>0,
        actual_memory=observation['CUDA_peak_allocated']>0 and observation['CUDA_peak_reserved']>0,
        actual_throughput=observation['observed_records_per_second']>0,
        all_pass=True)


def accepted_state(config,modules,step):
    training=Path(config['output'])/'training';path=training/f'checkpoint-{step}'
    identity=validation.native_identity(config,modules)
    receipt=modules['native'].checkpoint_receipt(path,identity)
    c.require(receipt['global_step']==step and receipt['world_size']==1,'Accepted state step/world differs')
    portable=c.read(path/'config.json')
    c.require(portable['model_type']=='llava_llama' and portable['hidden_size']==4096,'Missing/incorrect portable native7B config')
    c.require((path/'adapter_config.json').exists(),'Portable LoRA configuration absent')
    return path,receipt


def compare(study,phase_descriptors):
    configs,modules=validation.validate_study(study)
    phases={key:verified_phase(phase_descriptors[key]) for key in ('reference','interrupted','resumed')}
    c.require(all(p['study_id']==study['id'] and p['driver_sources']==study['driver_sources'] and p['native_target']==study['native_target'] for p in phases.values()),'Different study/driver/target evidence')
    c.require(phases['reference']['native_configuration']==study['native_configurations']['reference']
        and all(phases[k]['native_configuration']==study['native_configurations']['resume'] for k in ('interrupted','resumed')),'Actual native configuration substitution')
    c.require(phases['reference']['phase']=='reference' and phases['interrupted']['phase']=='interrupted' and phases['resumed']['phase']=='resumed','Phase substitution')
    c.require(phases['reference']['actual_job_id']==phases['interrupted']['actual_job_id']==phases['resumed']['actual_job_id'],'Same physical GPU job must own actual requeue test')
    c.require(phases['resumed']['actual_restart_count']>phases['interrupted']['actual_restart_count'],'Actual scheduler requeue not demonstrated')
    observations={key:c.checked(p['observation']) for key,p in phases.items()}
    criteria={key:check_behavior(value) for key,value in observations.items()}
    for value in criteria.values():value['all_pass']=all(v for k,v in value.items() if k!='all_pass')
    total=study['total_steps'];cut=study['interrupt_step']
    paths={key:Path(value['attempt'])/'observations' for key,value in phases.items()}
    lr_trajectories=compare_learning_rate_trajectories(paths,total,cut)
    initial={key:c.read(value/'FIRST_MICRO_STATE.json') for key,value in paths.items()}
    before={key:c.read(value/f'STEP-{cut}.json') for key,value in paths.items() if key!='resumed'}
    ends={key:c.read(paths[key]/f'STEP-{total}.json') for key in ('reference','resumed')}
    c.require(phases['reference']['status']=='NATIVE_TRAJECTORY_COMPLETE' and phases['resumed']['status']=='NATIVE_TRAJECTORY_COMPLETE'
        and phases['interrupted']['status']=='ACCEPTED_NATIVE_CONTINUATION','Incomplete reference or interruption schedule')
    order_reference=observations['reference']['ordered_record_ids']
    order_resume=observations['interrupted']['ordered_record_ids']+observations['resumed']['ordered_record_ids']
    order=dict(exact=order_reference==order_resume,reference_records=len(order_reference),resumed_records=len(order_resume),
        expected_records=total*128,reference_order_sha256=c.key(order_reference),resumed_order_sha256=c.key(order_resume))
    order['complete_global128']=order['reference_records']==order['resumed_records']==order['expected_records']
    pool=c.checked(configs['reference']['providers']['pool']);rows=c.checked(pool['records_json'])
    rowmap=dict(zip(pool['record_ids'],rows))
    c.require(set(order_reference)<=set(rowmap),'Unknown actual training records')
    selected=[rowmap[i] for i in order_reference]
    order['unique_records']=len(set(order_reference))
    order['without_repetition']=len(set(order_reference))==len(order_reference)
    order['actual_type_coverage']=dict(text_only=sum(not r.get('image') for r in selected),image=sum(bool(r.get('image')) for r in selected),
        multi_turn=sum(len(r['conversations'])>=4 for r in selected),long=sum(sum(len(t['value']) for t in r['conversations'])>=2000 for r in selected))
    order['actual_required_type_coverage']=all(order['actual_type_coverage'][k]>=minimum for k,minimum in [('text_only',32),('image',512),('multi_turn',128),('long',32)])
    initial_keys=['weights','optimizer','scheduler','rng','global_step','actual_DS_global_steps','DS_skipped_steps']
    initial_pair={name:initial['reference'][name]==initial['interrupted'][name] for name in initial_keys}
    c.require(total*128<=c.checked(configs['reference']['providers']['pool'])['count'],'Engineering parity should avoid unobserved sampler epoch/repetition')
    restorable=['weights','optimizer','scheduler','rng','global_step','actual_DS_global_steps','DS_skipped_steps']
    restored={name:before['interrupted'][name]==initial['resumed'][name] for name in restorable}
    c.require(initial['reference']['global_step']==initial['interrupted']['global_step']==0 and initial['resumed']['global_step']==cut,'Unexpected initial/resume trainer steps')
    ref_path,ref_receipt=accepted_state(configs['reference'],modules,total)
    resumed_path,resumed_receipt=accepted_state(configs['resume'],modules,total)
    cut_path,cut_receipt=accepted_state(configs['resume'],modules,cut)
    tol=study['numeric_tolerance'];comparisons={}
    comparisons['portable_weights']=state.compare_tree(state.portable_weights(ref_path),state.portable_weights(resumed_path),tol['absolute'],tol['relative'])
    comparisons['Adam_FP32_master_scaler']=state.compare_tree(state.native_optimizer(ref_path),state.native_optimizer(resumed_path),tol['absolute'],tol['relative'])
    import torch
    comparisons['saved_RNG']=state.compare_tree(state.normalize_saved_rng(torch.load(ref_path/'rng_state.pth',map_location='cpu')),
        state.normalize_saved_rng(torch.load(resumed_path/'rng_state.pth',map_location='cpu')),0,0)
    def scheduler(path):
        if (path/'scheduler.pt').exists():return torch.load(path/'scheduler.pt',map_location='cpu')
        candidates=list(path.rglob('*model_states.pt'));c.require(len(candidates)==1,'Missing actual DS model-state scheduler shard')
        return torch.load(candidates[0],map_location='cpu')['lr_scheduler']
    comparisons['scheduler']=state.compare_tree(scheduler(ref_path),scheduler(resumed_path),0,0)
    import json,numpy as np
    def losses(key):
        return [json.loads(line)['returned_native_scaled_loss'] for line in (paths[key]/'MICROBATCHES.jsonl').read_text().splitlines()]
    comparisons['microbatch_loss_trajectory']=state.compare_tree(np.asarray(losses('reference'),dtype=np.float64),
        np.asarray(losses('interrupted')+losses('resumed'),dtype=np.float64),tol['absolute'],tol['relative'])
    steps=dict(reference=ends['reference']['global_step'],resumed=ends['resumed']['global_step'],expected=total,
        optimizer_update_demonstrated=initial['reference']['weights']!=ends['reference']['weights'] and initial['interrupted']['weights']!=ends['resumed']['weights'],
        all_final_weights_finite=ends['reference']['all_weights_finite'] and ends['resumed']['all_weights_finite'],
        DS_skipped_steps_equal=ends['reference']['DS_skipped_steps']==ends['resumed']['DS_skipped_steps'])
    pass_all=all(v['all_pass'] for v in criteria.values()) and order['exact'] and order['complete_global128'] and order['without_repetition'] and order['actual_required_type_coverage']
    pass_all=pass_all and all(initial_pair.values()) and all(restored.values())
    pass_all=pass_all and all(v['pass_numeric'] for v in comparisons.values()) and steps['reference']==steps['resumed']==total
    pass_all=pass_all and steps['optimizer_update_demonstrated'] and steps['all_final_weights_finite'] and steps['DS_skipped_steps_equal']
    groups={key:c.read(value/'PARAMETER_GROUPS.json') for key,value in paths.items()}
    groups_equal=groups['reference']==groups['interrupted']==groups['resumed'];pass_all=pass_all and groups_equal
    checkpoint_evidence={key:dict(accepted=c.desc(path/'ACCEPTED.json'),logical_bytes=receipt['logical_bytes'],portable_config=c.desc(path/'config.json'))
        for key,path,receipt in [('reference_final',ref_path,ref_receipt),('resume_boundary',cut_path,cut_receipt),('resume_final',resumed_path,resumed_receipt)]}
    return dict(schema='native-engineering-actual-readout-v1',status='PASS_ACTUAL_ENGINEERING_READOUT_UNREVIEWED' if pass_all else 'FAIL_ACTUAL_ENGINEERING_READOUT',
        owner='Root',study_id=study['id'],study_identity=c.key({k:v for k,v in study.items() if k!='review'}),
        native_target=study['native_target'],driver_sources=study['driver_sources'],native_sources=configs['reference']['sources'],
        phase_results=phase_descriptors,criteria=criteria,ordered_exposure=order,initial_pair_state_exact=initial_pair,restart_loaded_state_exact=restored,
        numerical_comparisons=comparisons,steps=steps,optimizer_groups_equal=groups_equal,
        semantic_learning_rate_trajectories=lr_trajectories,accepted_states=checkpoint_evidence,
        observed_costs={key:dict(seconds=value['elapsed_seconds'],CUDA_peak_allocated=value['CUDA_peak_allocated'],
            CUDA_peak_reserved=value['CUDA_peak_reserved'],observed_records_per_second=value['observed_records_per_second'],
            observer_overhead_seconds=value['observer_overhead_seconds']) for key,value in observations.items()},
        actual_same_job_requeue=True,scientific_pool_qualified=False,downstream_performance_measured=False,
        qualification_promoted=False,raw_descriptors_and_independent_readout_review_required=True)


def promote(study,raw_descriptor,review_descriptor,output):
    """Independent review is an actual-result gate, never manufactured here."""
    configs,_=validation.validate_study(study);raw=c.checked(raw_descriptor);review=c.checked(review_descriptor)
    c.require(raw['status']=='PASS_ACTUAL_ENGINEERING_READOUT_UNREVIEWED','Failed/missing raw experiment cannot qualify')
    c.require(raw['study_identity']==c.key({k:v for k,v in study.items() if k!='review'}),'Actual study changed')
    c.require(review['status']=='PASS_INDEPENDENT_ACTUAL_NATIVE_ENGINEERING_READOUT' and review['readout']==raw_descriptor
        and review['study_identity']==raw['study_identity'] and review['reviewer']!='native_campaign_build','Actual independent readout review absent')
    cfg=configs['reference'];evidence={}
    for name in ('forward_backward','masking','restart','optimizer_groups','throughput'):
        evidence[name]=c.seal(Path(output)/('GPU_'+name.upper()+'.json'),dict(status='PASS_ACTUAL_GPU_'+name.upper(),sources=cfg['sources'],recipe=cfg['recipe'],
            actual_raw_readout=raw_descriptor,independent_actual_readout_review=review_descriptor,scope='1024_OR_MORE_RECORD_ENGINEERING_ONLY',
            actual_checkpoint_and_raw_phase_records=raw['phase_results']))
    return c.seal(Path(output)/'QUALIFICATION.json',dict(status='PASS_ACTUAL_NATIVE_GPU_MASKING_NUMERICS_RESTART',sources=cfg['sources'],recipe=cfg['recipe'],
        allocation=cfg['allocation'],site_profile=cfg['site_profile'],providers={k:cfg['providers'][k] for k in ('environment','assets')},evidence=evidence,
        engineering_readout=raw_descriptor,independent_actual_readout_review=review_descriptor,scientific_pool_qualified=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--phase-results',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(c.canonical(c.seal(a.output,compare(c.read(a.study),c.read(a.phase_results)))).decode())

if __name__=='__main__':main()
