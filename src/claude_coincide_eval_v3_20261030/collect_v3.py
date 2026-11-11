"""Recovered natural-epoch accounting: accepted v2 prefix [0,S] + v3 segment(s) (S,516].

Same per-attempt checks as the unchanged v2 collect.collect. The v2 attempt that
the site interruption killed contributes only its accepted prefix: its first
S*32 objective units and LR rows 0..S, bound to the accepted step-S header.
Overlap rows it wrote after S (discarded updates) must equal the resumed
segment's data order and LR schedule, which verifies data-position and
scheduler continuity across the restart. Fresh cells use v2 collect unchanged.
"""
import json
import math
from pathlib import Path


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _objective_ok(rows):
    return all(row['microbatch_size'] == 4 and len(row['per_record_shifted_targets']) == 4
        and min(row['per_record_shifted_targets']) > 0
        and sum(row['per_record_shifted_targets']) == row['expanded_shifted_targets']
        and row['declared_accumulation'] == 32 and math.isfinite(row['native_unscaled_loss'])
        and math.isfinite(row['returned_native_scaled_loss']) for row in rows)


def prefix(p, cfg, expected, child, step, accepted, archive, validate_lr):
    """Accepted prefix of an interrupted v2 attempt; returns ids, last LR row and overlap rows."""
    child = Path(child); obs = child / 'student_observations'
    horizon = p.read(obs / 'ACTUAL_HORIZON.json')
    p.require(horizon['actual_resume_step'] == 0 and horizon['args_max_steps'] == -1
        and horizon['native_max_steps'] == expected['natural_updates'] and 0 < step < expected['natural_updates'],
        'Prefix attempt must start at 0 with the natural horizon')
    record = p.read(child / ('CHECKPOINT-' + str(step) + '.json'))
    p.require(record['accepted'] == accepted and archive['sha256'] == accepted['sha256']
        and archive['bytes'] == accepted['bytes'], 'Prefix accepted header/archive differs from its own record')
    header = p.checked(archive)
    p.require(header['status'] == 'ACCEPTED_FULL_NATIVE_STATE' and header['identity'] == p.n.run_identity(cfg)
        and header['global_step'] == step, 'Prefix accepted identity/step')
    objective = _jsonl(obs / 'OBJECTIVE_UNITS.jsonl'); lr = _jsonl(obs / 'LEARNING_RATES.jsonl')
    head, tail = objective[:step * 32], objective[step * 32:]
    p.require(len(head) == step * 32 and [r['microbatch'] for r in head] == list(range(1, step * 32 + 1))
        and _objective_ok(head), 'Incomplete/invalid prefix objective units')
    p.require([r['trainer_step_before'] for r in head] == [i // 32 for i in range(step * 32)],
        'Prefix objective units not aligned to optimizer steps')
    ids = [v for row in head for v in row['record_ids']]
    p.require(len(ids) == step * 128 and len(set(ids)) == len(ids), 'Prefix exposure count/repetition')
    validate_lr(lr[:step + 1], 0, step, expected['natural_updates'])
    p.require(all(row['skipped_steps'] == 0 for row in lr[:step + 1]), 'Skipped update in prefix')
    return dict(ids=ids, last_lr=lr[step], overlap_objective=tail, overlap_lr=lr[step + 1:],
        header=archive, discarded_microbatches=len(tail), discarded_lr_rows=len(lr) - step - 1)


def collect(configuration, pre, attempt_receipts, output, p, validate_lr, load=None):
    load = p.checked if load is None else load
    cfg = load(configuration); selected = load(cfg['providers']['selection']); expected = p.horizon(selected['count'])
    allowed = set(selected['record_ids'])
    first = prefix(p, cfg, expected, pre['child'], pre['step'], pre['accepted'], pre['archive'], validate_lr)
    p.require(set(first['ids']) <= allowed, 'Prefix exposure outside selection')
    cursor, ids, last_lr, bound, continuity = pre['step'], list(first['ids']), first['last_lr'], [], {}
    p.require(attempt_receipts, 'No resumed v3 attempt')
    for k, ref in enumerate(attempt_receipts):
        value = load(ref)
        p.require(value['configuration'] == configuration and value['config_identity'] == p.n.run_identity(cfg)
            and value['adapter_sources'] == p.sources() and value['runtime'] == p.RUNTIME, 'Foreign v3 attempt')
        attempt = Path(value['exposure']['path']).parent
        horizon = load(p.desc(attempt / 'ACTUAL_HORIZON.json'))
        p.require(horizon['actual_resume_step'] == cursor and horizon['args_max_steps'] == -1
            and horizon['native_max_steps'] == expected['natural_updates'] and horizon['selected_count'] == selected['count'],
            'Resumed attempt starts at a different accepted cursor')
        if value['status'] == 'ACCEPTED_STUDENT_CONTINUATION_REQUIRES_ROOT_SAME_JOB_ACTION':
            end = value['global_step']; last_accepted = value['accepted']
        else:
            p.require(value['status'] == 'COMPLETE_STUDENT_TRAINING_NOT_EVALUATION', 'Unaccepted attempt')
            complete = load(value['native_complete']); end = complete['global_step']; last_accepted = complete['accepted']
        p.require(cursor < end <= expected['natural_updates'], 'Nonincreasing accepted cursor')
        archive = value['accepted_archive']
        p.require(archive['sha256'] == last_accepted['sha256'] and archive['bytes'] == last_accepted['bytes'], 'Archive differs')
        state = load(archive)
        p.require(state['status'] == 'ACCEPTED_FULL_NATIVE_STATE' and state['identity'] == p.n.run_identity(cfg)
            and state['global_step'] == end, 'Accepted identity/cursor mismatch')
        exposure = load(value['exposure']); seen = exposure['ordered_record_ids']
        p.require(exposure['record_count'] == len(seen) == (end - cursor) * 128 and exposure['microbatches'] == (end - cursor) * 32
            and set(seen) <= allowed and len(set(seen)) == len(seen), 'Resumed exposure count/source/repetition')
        lr = _jsonl(attempt / 'LEARNING_RATES.jsonl')
        validate_lr(lr, cursor, end, expected['natural_updates'])
        p.require(all(row['skipped_steps'] == 0 for row in lr) and lr[0] == last_lr,
            'Skipped update or scheduler discontinuity across the recovery restart')
        objective = _jsonl(attempt / 'OBJECTIVE_UNITS.jsonl')
        p.require(len(objective) == (end - cursor) * 32 and [v for row in objective for v in row['record_ids']] == seen
            and _objective_ok(objective), 'Incomplete resumed objective-unit accounting')
        if k == 0:
            # Discarded post-S rows of the interrupted attempt must match the resumed order/schedule.
            n = min(len(first['overlap_objective']), len(objective))
            p.require([r['record_ids'] for r in first['overlap_objective'][:n]] == [r['record_ids'] for r in objective[:n]],
                'Resumed data position differs from the interrupted trajectory')
            m = min(len(first['overlap_lr']), len(lr) - 1)
            p.require(first['overlap_lr'][:m] == lr[1:m + 1], 'Resumed LR schedule differs from the interrupted trajectory')
            loss = [abs(a['native_unscaled_loss'] - b['native_unscaled_loss'])
                for a, b in zip(first['overlap_objective'][:n], objective[:n])]
            continuity = dict(compared_microbatches=n, compared_lr_rows=m, record_order_equal=True, lr_rows_equal=True,
                max_abs_microbatch_loss_difference=max(loss) if loss else None)
        ids += seen; cursor = end; last_lr = lr[-1]; bound.append(ref)
    p.require(cursor == expected['natural_updates'] and len(ids) == expected['optimized_records'] and len(set(ids)) == len(ids),
        'Recovered natural epoch incomplete or repeats records')
    optimized = set(ids); tail = [v for v in selected['record_ids'] if v not in optimized]
    p.require(len(tail) == expected['unoptimized_tail_count'], 'Unoptimized tail differs; no topup')
    return p.seal(output, dict(status='COMPLETE_ACTUAL_NATURAL_STUDENT_TRAJECTORY_NOT_EVALUATION',
        recovery='V2_ACCEPTED_PREFIX_PLUS_V3_RESUMED_SEGMENTS', configuration=configuration,
        config_identity=p.n.run_identity(cfg), prefix=dict(child=str(pre['child']), step=pre['step'],
            accepted=pre['accepted'], archive=pre['archive'], discarded_microbatches=first['discarded_microbatches'],
            discarded_lr_rows=first['discarded_lr_rows']),
        attempts=bound, accepted=last_accepted, expected=expected, actual_optimized_records=len(ids),
        actual_unoptimized_tail_record_ids=tail, ordered_record_ids_sha256=p.key(ids), actual_optimizer_updates=cursor,
        restart_continuity=continuity, native_objective_scaling_unchanged=True, actual_no_skipped_optimizer_updates=True,
        adapter_sources=p.sources(), runtime=p.RUNTIME, downstream_evaluation_complete=False,
        scientific_improvement_demonstrated=False))
