"""Actual raw parity readout; promotion requires a distinct independent review."""
import argparse
import json
from pathlib import Path
from . import protocol as c
from . import protocol as validation
from experiments.root_native_qualification_v10_20261024 import state
from experiments.root_native_qualification_v10_20261024.observer import validate_lr_trajectory

STATE_KEYS=('weights','optimizer','scheduler','rng','global_step','actual_DS_global_steps','DS_skipped_steps')

def update_cursor_checks(initial,before,ends,total,cut):
    """Equal skipped trajectories do not demonstrate eight optimizer updates."""
    checkpoints={**{'initial_'+k:(value,cut if k=='resumed' else 0) for k,value in initial.items()},
        **{'cut_'+k:(value,cut) for k,value in before.items()},
        **{'final_'+k:(value,total) for k,value in ends.items()}}
    checks={name:dict(zero_skips=value['DS_skipped_steps']==0,
        trainer_cursor_exact=value['global_step']==step,
        DS_cursor_exact=value['actual_DS_global_steps']==step)
        for name,(value,step) in checkpoints.items()}
    return dict(passed=all(all(row.values()) for row in checks.values()),checkpoints=checks,
        expected_optimizer_updates=total)

def pre_restart_checks(before,loss_rows,total,cut,tolerance):
    """STEP4 is full semantic state digests, not a second tensor checkpoint.

    Exact digest equality is stronger than approximate equality when it passes;
    a failure does not provide a numeric error size. Retain that distinction.
    Loss tensors below are actually compared with the unchanged strict bounds.
    """
    import numpy as np
    c.require(set(before)=={'reference','interrupted'},'Both actual pre-restart phases required')
    exact={name:before['reference'][name]==before['interrupted'][name] for name in STATE_KEYS}
    steps=before['reference']['global_step']==before['interrupted']['global_step']==cut
    finite=before['reference']['all_weights_finite'] and before['interrupted']['all_weights_finite']
    c.require(len(loss_rows['reference'])==total*32 and len(loss_rows['interrupted'])==cut*32,
              'Prefix loss rows must cover all actual global128/micro4 updates')
    prefix=state.compare_tree(np.asarray(loss_rows['reference'][:cut*32],dtype=np.float64),
        np.asarray(loss_rows['interrupted'],dtype=np.float64),tolerance['absolute'],tolerance['relative'])
    passed=all(exact.values()) and steps and finite and prefix['pass_numeric']
    return dict(passed=passed,semantic_state_exact=exact,boundary_step_exact=steps,
        both_weights_finite=finite,microbatch_prefix_loss_comparison=prefix,
        prefix_loss_records=cut*32,semantic_digest_scope='FULL_STATE_HASHES_NO_SEPARATE_PRE_RESTART_TENSOR_NUMERIC_RECOMPUTATION',
        numeric_error_from_digest_mismatch_available=False)

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
    c.require(study['cell']=='D8','Strict paired readout requires declared D8 runtime')
    phases={key:verified_phase(phase_descriptors[key]) for key in ('reference','interrupted','resumed')}
    for phase in phases.values():
        runtime=c.checked(phase['executed_runtime']);forced=runtime['explicit_policy'];actual=runtime['executed_operator_trace']
        c.require(forced and forced['policy']=='explicit_xformers_cutlass_fw_bw_deterministic_v1' and forced['calls']>0 and forced['no_operator_fallback'] is True and actual['forward_calls']>0 and actual['backward_calls']>0,'Actual declared deterministic runtime was not exercised')
        c.require(all(v['operator'] in ('xformers.ops.fmha.cutlass.FwOp','xformers.ops.fmha.cutlass.BwOp') for v in actual['executed'] if v['kind'] in ('forward','backward')),'Unexpected operator/fallback in deterministic D8')
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
    initial_keys=STATE_KEYS
    initial_pair={name:initial['reference'][name]==initial['interrupted'][name] for name in initial_keys}
    c.require(total*128<=c.checked(configs['reference']['providers']['pool'])['count'],'Engineering parity should avoid unobserved sampler epoch/repetition')
    restorable=STATE_KEYS
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
    loss_rows={key:losses(key) for key in ('reference','interrupted','resumed')}
    pre_restart=pre_restart_checks(before,loss_rows,total,cut,tol)
    comparisons['microbatch_loss_trajectory']=state.compare_tree(np.asarray(losses('reference'),dtype=np.float64),
        np.asarray(losses('interrupted')+losses('resumed'),dtype=np.float64),tol['absolute'],tol['relative'])
    steps=dict(reference=ends['reference']['global_step'],resumed=ends['resumed']['global_step'],expected=total,
        optimizer_update_demonstrated=initial['reference']['weights']!=ends['reference']['weights'] and initial['interrupted']['weights']!=ends['resumed']['weights'],
        all_final_weights_finite=ends['reference']['all_weights_finite'] and ends['resumed']['all_weights_finite'],
        DS_skipped_steps_equal=ends['reference']['DS_skipped_steps']==ends['resumed']['DS_skipped_steps'])
    cursor_checks=update_cursor_checks(initial,before,ends,total,cut)
    pass_all=all(v['all_pass'] for v in criteria.values()) and order['exact'] and order['complete_global128'] and order['without_repetition'] and order['actual_required_type_coverage']
    pass_all=pass_all and all(initial_pair.values()) and all(restored.values()) and pre_restart['passed'] and cursor_checks['passed']
    pass_all=pass_all and all(v['pass_numeric'] for v in comparisons.values()) and steps['reference']==steps['resumed']==total
    pass_all=pass_all and steps['optimizer_update_demonstrated'] and steps['all_final_weights_finite'] and steps['DS_skipped_steps_equal']
    groups={key:c.read(value/'PARAMETER_GROUPS.json') for key,value in paths.items()}
    groups_equal=groups['reference']==groups['interrupted']==groups['resumed'];pass_all=pass_all and groups_equal
    checkpoint_evidence={key:dict(accepted=c.desc(path/'ACCEPTED.json'),logical_bytes=receipt['logical_bytes'],portable_config=c.desc(path/'config.json'))
        for key,path,receipt in [('reference_final',ref_path,ref_receipt),('resume_boundary',cut_path,cut_receipt),('resume_final',resumed_path,resumed_receipt)]}
    epoch_checks=epoch_evidence(study,configs,phases,observations)
    pass_all=pass_all and epoch_checks['passed']
    return dict(schema='declared-epoch-runtime-actual-readout-v1',status='PASS_ACTUAL_DECLARED_EPOCH_RUNTIME_UNREVIEWED' if pass_all else 'FAIL_ACTUAL_DECLARED_EPOCH_RUNTIME',epoch_and_objective_units=epoch_checks,
        owner='Root',prepared_by=c.AUTHOR_IDENTITY,result_basis=study['result_basis'],production_runtime_adapter_qualified=False,paired_initialization_seed=study['seed'],exposed_development_fixture=True,multi_flag_runtime_intervention=True,study_id=study['id'],study_identity=c.key({k:v for k,v in study.items() if k!='review'}),
        native_target=study['native_target'],driver_sources=study['driver_sources'],native_sources=configs['reference']['sources'],
        phase_results=phase_descriptors,criteria=criteria,ordered_exposure=order,initial_pair_state_exact=initial_pair,restart_loaded_state_exact=restored,pre_restart_reference_cut4=pre_restart,actual_update_cursor_checks=cursor_checks,
        numerical_comparisons=comparisons,steps=steps,optimizer_groups_equal=groups_equal,
        semantic_learning_rate_trajectories=lr_trajectories,accepted_states=checkpoint_evidence,
        observed_costs={key:dict(seconds=value['elapsed_seconds'],CUDA_peak_allocated=value['CUDA_peak_allocated'],
            CUDA_peak_reserved=value['CUDA_peak_reserved'],observed_records_per_second=value['observed_records_per_second'],
            observer_overhead_seconds=value['observer_overhead_seconds']) for key,value in observations.items()},
        actual_same_job_requeue=True,scientific_pool_qualified=False,downstream_performance_measured=False,
        qualification_promoted=False,raw_descriptors_and_independent_readout_review_required=True)




def epoch_evidence(study,configs,phases,observations):
    spec=study['specification'];inputs={}
    bound_pool=c.checked(configs['reference']['providers']['pool'])
    for name,child in phases.items():
        for item in child['epoch_files'].values():c.require(c.desc(item['path'])==item,'Epoch evidence bytes changed')
        files=child['epoch_files'];horizon=c.checked(files['ACTUAL_HORIZON.json']);loader=c.checked(files['ACTUAL_DATALOADER.json'])
        c.require(horizon['actual_max_steps']==spec['total_steps'] and horizon['selected_count']==spec['count'] and horizon['natural_epoch']==spec['natural_epoch'],'Actual horizon belongs to another endpoint')
        if spec['natural_epoch']:c.require(horizon['args_max_steps']==-1 and loader['loader_batches']==289 and loader['drop_last'] is False,'Natural epoch was secretly capped or drop_last differs')
        else:c.require(horizon['args_max_steps']==8,'ACES fixed fixture cap drift')
        rows=[json.loads(line) for line in Path(files['ACTUAL_OBJECTIVE_UNITS.jsonl']['path']).read_text().splitlines()]
        expected=(4 if name=='interrupted' else spec['total_steps']-4 if name=='resumed' else spec['total_steps'])*32
        c.require(len(rows)==expected and all(row['microbatch_size']==4 and row['expanded_shifted_valid_targets']==sum(row['per_record_expanded_shifted_targets'])>0 and len(row['per_record_expanded_shifted_targets'])==4 and min(row['per_record_expanded_shifted_targets'])>0 and row['declared_accumulation']==32 for row in rows),'Missing/nonpositive actual objective denominators or partial optimized microbatch')
        c.require([value for row in rows for value in row['record_ids']]==observations[name]['ordered_record_ids'] and all(row['record_ids']==[bound_pool['record_ids'][i] for i in row['fixture_indices']] and row['original_source_indices']==[bound_pool['source_indices'][i] for i in row['fixture_indices']] for row in rows),'Fixture-local/global original row identities differ')
        if name!='interrupted':
            stop=c.checked(files['ACTUAL_STOP.json']);c.require(stop['global_step']==stop['state_max_steps']==spec['total_steps'] and stop['args_max_steps']==horizon['args_max_steps'] and stop['control_should_training_stop'] is True,'Missing genuine natural/native endpoint')
        inputs[name]=rows
    # Time/micro counters excluded; actual indices/denominators/losses/scaling remain.
    combined=inputs['interrupted']+inputs['resumed'];same=inputs['reference']==combined
    pool=c.checked(configs['reference']['providers']['pool']);used=observations['reference']['ordered_record_ids'];unused=[value for value in pool['record_ids'] if value not in set(used)]
    expected_tail=1 if spec['natural_epoch'] else 0;c.require(len(unused)==expected_tail and len(used)==spec['total_steps']*128 and len(set(used))==len(used),'Selected cardinality/tail/order was changed, topped-up or repeated')
    rowmap=dict(zip(pool['record_ids'],c.checked(pool['records_json'])))
    from experiments.root_native_input_authority_20261023.pool_authority_v1 import source
    from collections import Counter
    source_counts=dict(Counter(source(rowmap[value]) for value in used))
    if spec['natural_epoch']:c.require(set(source_counts)=={'coco','gqa','ocr_vqa','vg','textvqa','text_only'},'Actual forwarded records omit an original source')
    return dict(passed=same,uninterrupted_and_restarted_objective_units_exact=same,selected_count=spec['count'],actual_optimized_count=len(used),unoptimized_tail_record_ids=unused,tail_count=len(unused),no_tail_replacement_or_topup=True,actual_forwarded_sources=source_counts,natural_epoch=spec['natural_epoch'],all_source_qualified_in_this_scope=spec['natural_epoch'],H100_broad_source_claim=False,objective_scaling='UNCHANGED_NATIVE_MICROBATCH_VALID_TARGET_MEAN_AND_GA32',actual_denominator_records=sum(len(x) for x in inputs.values()),epochs=1)

# Promotion intentionally absent; production consumer needs exact independent evidence binding.
