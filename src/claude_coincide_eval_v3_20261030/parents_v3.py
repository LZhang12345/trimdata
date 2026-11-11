"""Full516 recovered parent: actual v3 jobs, exact bindings and recomputed exposure.

afterok alone is insufficient. Every bound v3 training job must have its own
sacct record (COMPLETED 0:0 for the final job, FAILED 75:0 for accepted
continuations), its sealed RECOVERY_BINDING/ATTEMPT_RESULT/CHILD_EXIT and child
receipt; the natural epoch is then recomputed by collect_v3 (v2 accepted prefix
+ v3 segments) or by the unchanged v2 collect (fresh s43).
"""
from pathlib import Path

from experiments.copilot7_coincide_native_exec_v2_20261028 import protocol as native
from experiments.copilot7_coincide_native_exec_v2_20261028 import collect as v2collect
from experiments.root_native_qualification_v10_20261024.observer import validate_lr_trajectory
from experiments.s1_eval_parent_bridge_v1_20261027.parents import accounting
from . import common_v3 as c, protocol_v3 as p, collect_v3

COMPLETE = 'COMPLETE_STUDENT_TRAINING_NOT_EVALUATION'
CONTINUED = 'ACCEPTED_CONTINUATION_REQUIRES_V3_SUCCESSOR'
CHILD_CONTINUED = 'ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION'


def prefix_archive(accepted, cell, step, describe=c.desc):
    """Original step-S ACCEPTED.json if still present, else native retention's same-bytes archive."""
    retired = c.V2_REPORT / cell / 'retired-new-intermediates' / ('checkpoint-' + str(step)) / 'ACCEPTED_ARCHIVE.json'
    for candidate in (Path(accepted['path']), retired):
        if candidate.is_file():
            ref = describe(candidate)
            if c.same_bytes(ref, accepted):
                return ref
    raise ValueError('Accepted step-%d header is neither present nor archived with identical bytes' % step)


def attempt_chain(cfg, training, load=c.checked, account=accounting):
    """Verify each bound v3 physical attempt; returns ordered child receipts and sacct history."""
    identity = native.n.run_identity(training)
    jobs = p.training_chain(cfg['training_seed'], cfg['training_configuration'], cfg['training_records'], load)
    c.require(jobs == cfg['required_training_job_ids'], 'Configured training job chain changed')
    receipts, history, previous, previous_ref = [], [], None, None
    for k, rec in enumerate(cfg['training_records']):
        job, last = rec['job_id'], k == len(cfg['training_records']) - 1
        request = load(rec['request'])
        attempt = Path(rec['request']['path']).parent / 'attempts' / (job + '-r0')
        binding_ref = c.desc(attempt / 'RECOVERY_BINDING.json')
        binding, result = load(binding_ref), load(c.desc(attempt / 'ATTEMPT_RESULT.json'))
        c.require(binding['schema'] == c.TRAINING_BINDING_SCHEMA and binding['request'] == rec['request']
                  and binding['actual']['job_id'] == job and binding['configuration'] == cfg['training_configuration']
                  and binding['config_identity'] == identity and binding['v2_sources'] == native.sources()
                  and binding['v3_sources'] == request['v3_sources'] and binding['review'] == request['review']
                  and binding['predecessor'] == request['predecessor']
                  and binding['scientific_recipe_changed'] is False and binding['selected_ids_changed'] is False
                  and binding['expected_natural_updates'] == 516 and binding['missing_updates_only'] is True
                  and Path(binding['child_attempt']) == attempt / 'child',
                  'Exact v3 recovery binding of this physical training job')
        c.training_independent(load(binding['review']), binding['v3_sources'])
        status, code = (COMPLETE, 0) if last else (CONTINUED, 75)
        c.require(result['binding'] == binding_ref and result['status'] == status
                  and result['logical_exit_code'] == code and result['automatic_requeue'] is False,
                  'Only a final complete attempt preceded by accepted continuations')
        history.append(account(job, 'COMPLETED', '0:0') if last else account(job, 'FAILED', '75:0'))
        exit_value = load(c.desc(attempt / 'child' / 'CHILD_EXIT.json'))
        receipt_ref = result['receipt']
        receipt = load(receipt_ref)
        c.require(exit_value['schema'] == 'root-longwall-child-exit-v1' and exit_value['receipt'] == receipt_ref
                  and exit_value['logical_exit_code'] == code and exit_value['torchrun_process_exit_code'] == 0
                  and exit_value['configuration'] == cfg['training_configuration']
                  and Path(receipt_ref['path']).parent == attempt / 'child'
                  and receipt['actual_job_id'] == job and receipt['actual_restart_count'] == 0
                  and receipt['configuration'] == cfg['training_configuration'] and receipt['config_identity'] == identity
                  and receipt['status'] == (COMPLETE if last else CHILD_CONTINUED),
                  'Child receipt belongs to this exact v3 job and outcome')
        start = c.PREFIX[cfg['training_seed']]['step'] if k == 0 else previous['global_step']
        c.require(binding['resume_point']['latest_step'] == start, 'Resume cursor differs from accepted chain')
        if k > 0:
            pred = binding['predecessor']
            c.require(pred['step'] == previous['global_step'] and pred['accepted'] == previous['accepted']
                      and pred['attempt'] == str(Path(previous_ref['path']).parent),
                      'v3 successor resumed from a different accepted boundary')
        receipts.append(receipt_ref); previous, previous_ref = receipt, receipt_ref
    return receipts, history


def recompute(cfg, receipts, path, load=c.checked):
    if cfg['prefix']['mode'] == 'fresh':
        return v2collect.collect(cfg['training_configuration'], receipts, path, load=load)
    pred = load(cfg['training_records'][0]['request'])['predecessor']
    step = cfg['prefix']['step']
    c.require(pred['step'] == step and pred['attempt'] == cfg['prefix']['v2_child'], 'Configured prefix differs')
    pre = dict(child=Path(pred['attempt']), step=step, accepted=pred['accepted'],
               archive=prefix_archive(pred['accepted'], cfg['student_id'], step))
    return collect_v3.collect(cfg['training_configuration'], pre, receipts, path, native, validate_lr_trajectory, load=load)


def complete(cfg, output, preflight, load=c.checked, account=accounting):
    training = load(cfg['training_configuration'])
    native.supported(training)
    receipts, history = attempt_chain(cfg, training, load, account)
    recheck = recompute(cfg, receipts, Path(preflight) / 'RECOMPUTED_COMPLETION.json', load)
    value = load(recheck)
    tail = value['actual_unoptimized_tail_record_ids']
    c.require(value['status'] == 'COMPLETE_ACTUAL_NATURAL_STUDENT_TRAJECTORY_NOT_EVALUATION'
              and value['configuration'] == cfg['training_configuration']
              and value['config_identity'] == native.n.run_identity(training)
              and value['adapter_sources'] == native.sources() and value['runtime'] == native.RUNTIME
              and value['expected'] == native.horizon(66076) and value['actual_optimizer_updates'] == 516
              and value['actual_optimized_records'] == 66048 and len(tail) == len(set(tail)) == 28
              and value['attempts'] == receipts and value['native_objective_scaling_unchanged'] is True
              and value['actual_no_skipped_optimizer_updates'] is True
              and value['downstream_evaluation_complete'] is False,
              'Actual full natural516/66048/28 recovered completion; no partial75/header-only or afterok inference')
    if cfg['prefix']['mode'] == 'resume':
        cont = value['restart_continuity']
        c.require(value['recovery'] == 'V2_ACCEPTED_PREFIX_PLUS_V3_RESUMED_SEGMENTS'
                  and value['prefix']['step'] == cfg['prefix']['step'] and cont['record_order_equal'] is True
                  and cont['lr_rows_equal'] is True and cont['compared_microbatches'] > 0 and cont['compared_lr_rows'] > 0,
                  'Restart continuity must actually be compared against the interrupted trajectory')
    accepted = load(value['accepted'])
    c.require(accepted['status'] == 'ACCEPTED_FULL_NATIVE_STATE' and accepted['global_step'] == 516
              and accepted['identity'] == value['config_identity']
              and Path(value['accepted']['path']) == Path(training['output']) / 'training/checkpoint-516/ACCEPTED.json',
              'Exact final native capsule; payloads are revalidated by the trained loader')
    # Stable cell-level completion: identical recomputation required on every successor.
    stable = c.seal(Path(output) / 'PARENT' / 'STUDENT_COMPLETION_V3.json', value)
    proof = c.seal(Path(preflight) / 'PARENT_ACCEPTANCE.json', dict(
        status='ACTUAL_COINCIDE_V3_RECOVERED_FULL516_PARENT_VERIFIED_NOT_MODEL_FORWARD',
        configuration=cfg['training_configuration'], training_records=cfg['training_records'],
        completion=stable, recomputed_completion=recheck, accepted=value['accepted'], accounting=history,
        independent_parent_review_required=False, parent_scope_covered_by_source_config_review=True,
        no_parent_state_or_checkpoint_changes=True, new_NN=0))
    return training, stable, value, [proof]
