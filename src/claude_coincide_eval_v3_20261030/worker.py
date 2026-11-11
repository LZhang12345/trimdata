"""Recovered final native COINCIDE checkpoint, unchanged official bank loops (v3).

Body copied from copilot7_coincide_native_eval_v2_20261028.worker.run. Changes:
v3 actor bound to this package's REQUEST/INTENT/ACK, one v3 review gate instead
of v2 review/ownership/storage grants, v3 recovered parents, the qualified v7
basis bank loader (official TextVQA join) and no self-requeue: an accepted
partial pass exits 75 for a separately submitted, reviewed v3 successor job.
"""
import argparse
import fcntl
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid

from experiments.root_current_four_bank_eval_v1_20261027 import worker as core
from experiments.copilot7_coincide_native_eval_v2_20261028 import model, scoring
from experiments.copilot7_coincide_native_exec_v2_20261028 import protocol as native
from experiments.s1_eval_parent_bridge_v1_20261027.parents import accounting
from . import common_v3 as c, protocol_v3 as p, parents_v3 as parents

u = core.u
banks, media, generation, qualified = core.banks, core.media, core.generation, core.qualified
budget, publish_same = core.budget, core.publish_same
CONTINUE = 'ACCEPTED_QA_PROGRESS_REQUIRES_V3_SUCCESSOR'
DONE = 'COMPLETE_COINCIDE_V3_FOUR_OFFICIAL_BANKS'


def documents(request, load=c.checked):
    cfg = load(request['configuration'])
    training = load(cfg['training_configuration'])
    basis = load(cfg['qualified_basis'])
    p.validate_science(cfg, basis, training, load)
    review = load(request['review'])
    c.independent(review, cfg['sources'], request['configuration'], cfg['student_id'])
    c.require(request['sources'] == cfg['sources']
              and review['unchanged_loader_numerical_bodies'] == model.qualified_body_proof(),
              'Independent exact v3 source/config and scoped trained-loader review')
    for relative, expected in cfg['sources'].items():
        c.require(c.sha(c.ROOT / relative) == expected, 'Frozen evaluation source changed: ' + relative)
    # Qualified basis gates, unchanged and evaluated on the basis itself (as v2 did).
    qref = c.desc(core.c.literal(basis['required_actual_qualification_result_path']))
    core.checks.qualification_result(basis, load(qref), load)
    qpeer = load(c.desc(core.c.literal(basis['required_independent_qualification_review_path'])))
    core.checks.independent_result(basis, qpeer, qref, load)
    _, qcfg = core.c.prerequisite_roster(basis, load)
    anchor = load(qcfg['training_configuration'])
    c.require(training['recipe'] == anchor['recipe']
              and all(training['providers'][name] == anchor['providers'][name] for name in ('environment', 'assets')),
              'Exact already qualified native model/processor/runtime class')
    core.c.validate_node_policy(basis, load)
    return cfg, load(cfg['official_POPE_configuration'])


def actor(request_ref, request, cfg, query=subprocess.check_output):
    job = os.environ.get('SLURM_JOB_ID', '')
    c.require(os.getuid() == c.UID and job.isdecimal() and os.environ.get('SLURM_RESTART_COUNT', '0') == '0'
              and not os.environ.get('SLURM_ARRAY_JOB_ID') and c.ROOT == c.SOURCE_ROOT
              and Path(sys.executable).absolute() == Path(c.PYTHON),
              'Actual scalar first-incarnation source-bound General evaluator (no requeue)')
    folder = Path(request_ref['path']).parent
    intent_ref = c.desc(folder / 'INTENT.json')
    intent, ack = c.checked(intent_ref), c.checked(c.desc(folder / 'ACK.json'))
    c.records_identity(request_ref, request, intent, ack, job)
    c.require(ack['intent'] == intent_ref and request['seed'] == cfg['training_seed']
              and Path(request['configuration']['path']).parent == c.EVAL_RECORDS / cfg['student_id'],
              'Exact durable intent/ACK chain and cell configuration')
    raw = query(['scontrol', 'show', 'job', '-o', job], text=True, timeout=30)
    fields = c.fields(raw)
    c.check_actor(fields, c.expected_actor(job, request['seed'], request['nonce'], socket.gethostname().split('.')[0]))
    c.require(query(['scontrol', 'write', 'batch_script', job, '-'], timeout=30)
              == (c.HERE / 'run.sbatch').read_bytes(), 'Immutable submitted batch script')
    c.require(os.environ.get('TZ') == 'America/New_York', 'Exact scheduler timezone')
    time.tzset()
    return dict(job_id=job, restart=0, node=fields['NodeList'], nonce=cfg['nonce'], scheduler=raw,
        end_epoch=int(time.mktime(time.strptime(fields['EndTime'], '%Y-%m-%dT%H:%M:%S'))),
        process_nonce=uuid.uuid4().hex, pid=os.getpid())


def existing_attempts(output):
    return sorted(path.name for path in Path(output).iterdir() if path.is_dir() and re.fullmatch(r'\d+-r0', path.name))


def prior(request, actual, output, existing, cfg, load=c.checked, account=accounting):
    """Previous accepted QA progress of the bound successor chain; None for the first job."""
    chain = request['previous_attempts']
    c.require(type(chain) is list and sorted(chain) == existing and len(chain) + 1 < cfg['max_physical_attempts'],
              'Evaluation attempt directories differ from the bound successor chain')
    if not chain:
        c.require(request['predecessor'] is None, 'First evaluation job has no predecessor')
        return None
    pred, last = request['predecessor'], chain[-1]
    ref = c.desc(Path(output) / last / 'RESULT.json')
    value = load(ref)
    c.require(pred['attempt'] == last == pred['job_id'] + '-r0'
              and value['status'] == CONTINUE and value['new_answers'] > 0
              and value['configuration'] == request['configuration']
              and value['actual']['job_id'] == pred['job_id'] and value['actual']['nonce'] == actual['nonce']
              and not (Path(output) / last / 'FAILURE.json').exists()
              and (Path(output) / last / 'V3_SUCCESSOR_REQUIRED.json').is_file(),
              'Only an accepted, non-failed predecessor pass may be continued')
    account(pred['job_id'], 'FAILED', '75:0')
    return value


def storage(cfg, used, statvfs=os.statvfs):
    fs = statvfs(cfg['output'])
    required = cfg['output_limit_bytes'] - used
    c.require(type(used) is int and 0 <= used <= cfg['output_limit_bytes']
              and fs.f_bavail * fs.f_frsize >= required and fs.f_favail >= 4096,  # SQLite per-bank journals
              'Current evaluation bytes/inodes insufficient; preserve all paid QA records')
    return dict(available_bytes=fs.f_bavail * fs.f_frsize, available_inodes=fs.f_favail, used_bytes=used,
                required_remaining_bytes=required, measured_not_a_reservation=True)


def function(cfg, identity, roster, complete_ref, bank):
    value = core.original.function(cfg, identity, roster, complete_ref, bank)
    return dict(value, schema='coincide-fixed10-fourbank-function-v3', role=p.SCOPE,
        adaptation_label=native.ADAPTATION, training_logical_protocol=native.PACKAGE,
        training_recovery_protocol='claude_coincide_recovery_v3_20261030',
        generation_API_policy=model.generation.POLICY, GQA_conversion=cfg['GQA_conversion'])


def run(request_path):
    request_ref = c.desc(request_path)
    request = c.checked(request_ref)
    preliminary = c.checked(request['configuration'])
    actual, began = actor(request_ref, request, preliminary), time.monotonic()
    cfg, base = documents(request)
    output = Path(cfg['output'])
    c.require(not any(path.is_symlink() for path in (output, *output.parents)), 'Literal new evaluator output')
    output.mkdir(parents=True, exist_ok=True)
    fd = os.open(output / 'EXECUTION.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    owned = False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = existing_attempts(output)
        attempt = output / (actual['job_id'] + '-r0')
        attempt.mkdir(exist_ok=False)
        owned = True
        previous = prior(request, actual, output, existing, cfg)
        u.publish(attempt / 'CURRENT_STORAGE.json', storage(cfg, budget(cfg)))
        u.publish(attempt / 'EARLY_BOUND_ASSET_READINESS.json', core.early_asset_readiness(cfg))
        _, complete_ref, complete, physical = parents.complete(cfg, output, attempt / 'PARENT_PREFLIGHT')
        u.publish(attempt / 'START.json', dict(actual=actual, request=request_ref, configuration=request['configuration'],
            student_completion=complete_ref, physical_proofs=physical, previous_attempts=existing,
            previous_result=c.desc(output / request['previous_attempts'][-1] / 'RESULT.json') if previous is not None else None))
        bankdata = core.joined_bank_data(cfg, attempt)
        ready = {}
        for bank, (roster, inventory, _, physical_bank, snapshot) in bankdata.items():
            store = media.Store(physical_bank, roster['records'], inventory, snapshot)
            try:
                names = sorted({row['native_input']['image'] for row in roster['records']})
                for name in names:
                    image = store.rgb(name)
                    image.close()
                ready[bank] = dict(questions=len(roster['records']), images=len(names),
                                   all_encoded_and_RGB_verified=True, forwards=0)
            finally:
                store.close()
        u.publish(attempt / 'FULL_MEDIA_READINESS.json', ready)
        publish_same(output / 'STUDENT_INPUT.json', dict(training_configuration=cfg['training_configuration'],
            student_completion=complete_ref, accepted=complete['accepted']))
        runtime = qualified.official.runtime(base, attempt)
        u.publish(attempt / 'RUNTIME.json', runtime)
        with qualified.official.offline():
            loaded3, proof = model.compose_v4(complete['accepted'], cfg['training_configuration'])
            core.strict_loaded_support(proof)
            identity = model.identity_v4(loaded3, proof, runtime)
            identity_ref = publish_same(output / 'MODEL_IDENTITY.json', identity)
            u.publish(attempt / 'MODEL_RECONSTRUCTION.json', proof)
            loaded = (*loaded3, getattr(loaded3[1].config, 'max_sequence_length', 2048))
            reports, new = {}, 0
            for bank, (roster, inventory, targets, physical_bank, snapshot) in bankdata.items():
                rows, path = roster['records'], output / bank
                path.mkdir(exist_ok=True)
                fn = function(cfg, identity, roster, complete_ref, bank)
                publish_same(path / 'FUNCTION.json', fn)
                journal = generation.AuditedJournal(path / 'journal', fn, rows)
                store = media.Store(physical_bank, rows, inventory, snapshot)
                try:
                    for entry in qualified.official.entries(journal, rows):
                        generation.validate_entry(bank, entry, journal.roster[entry['semantic_key']], fn)
                    core.accepted_prefix_count(previous, bank, len(journal.present))
                    before = len(journal.present)
                    if time.time() < actual['end_epoch'] - 300:
                        generation.native(bank, cfg, rows, journal, loaded, store, actual,
                                          time.monotonic() + max(0, actual['end_epoch'] - time.time() - 240))
                    new += len(journal.present) - before
                    reports[bank] = dict(questions=len(rows), accepted=len(journal.present),
                        missing=len(journal.missing()), new=len(journal.present) - before)
                    if not journal.missing():
                        raw = journal.export(path / 'PREDICTIONS.jsonl')
                        value = scoring.score(bank, roster, journal, targets)
                        extra = {}
                        if bank == 'GQA':
                            # Same predictions under the basis (uniform-s17) unconverted algorithm.
                            extra['basis_unconverted_score'] = banks.score(bank, roster, journal, targets)
                        publish_same(path / 'SCORE.json', dict(status='COMPLETE_COINCIDE_V3_NATIVE_OFFICIAL_BANK',
                            bank=bank, function=fn, predictions=raw, score=value, **extra,
                            student_completion=complete_ref, complete_questions=len(rows),
                            scientific_improvement_demonstrated=False))
                finally:
                    journal.close()
                    store.close()
                budget(cfg)
        c.require(model.identity_v4(loaded3, proof, runtime) == identity, 'Generation mutated trained state')
        bridge = model.generation.readout(loaded3[1])
        c.require(bridge['calls'] == new, 'Every new QA needs the exact qualified V4 bridge')
        u.publish(attempt / 'GENERATION_BRIDGE_READOUT.json', bridge)
        done = core.accepted_progress(previous, reports, new)
        value = dict(status=DONE if done else CONTINUE,
            actual=actual, request=request_ref, configuration=request['configuration'],
            student_completion=complete_ref, accepted_checkpoint=complete['accepted'], model_identity=identity_ref,
            banks=reports, new_answers=new, elapsed_seconds=time.monotonic() - began,
            all_prior_and_current_costs_count=True, scientific_improvement_demonstrated=False,
            scheduler_mutations=False)
        ref = u.publish(attempt / 'RESULT.json', value)
        if done:
            publish_same(output / 'COMPLETE.json', value)
        else:
            c.require(new > 0 and len(existing) + 2 < cfg['max_physical_attempts'],
                      'No accepted progress; preserve without auto retry')
            u.publish(attempt / 'V3_SUCCESSOR_REQUIRED.json', dict(previous_result=ref, job_id=actual['job_id'],
                attempt=attempt.name, previous_attempts=existing + [attempt.name], per_QA_durable=True,
                separately_reviewed_successor_submission_required=True, scheduler_action=False))
        return 0 if done else 75
    except BaseException as error:
        if owned:
            u.publish(attempt / 'FAILURE.json', dict(status='COINCIDE_V3_EVALUATION_FAILED_NO_AUTO_RETRY',
                actual=actual, error=repr(error), elapsed_seconds=time.monotonic() - began))
        raise
    finally:
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    raise SystemExit(run(parser.parse_args().request))
