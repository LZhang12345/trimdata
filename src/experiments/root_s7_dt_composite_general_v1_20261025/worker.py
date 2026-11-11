"""Two distinct Root scalar General continuations: global DT shards0 and3.

Numerical constructor, store, hook, full-pool validator and consumer are reused.
No admission, legacy writes, feature relabeling or new Preempt work.
"""
import argparse
import fcntl
import importlib
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import subprocess
import sys
import time
import types

from experiments.root_native_campaign_20261023 import common as c

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path('/home/luyangz/data_selection_multimodal/session7_fullpool_published_src_v2')
CONTROL = Path('/home/luyangz/data_selection_multimodal/collaboration/root-s7-dt-composite-general-v1-20261025')
CAMPAIGN = Path('/data/user_data/luyangz/native_campaign_20261023/session7-fullpool-published-v2')
AUTHOR = '/root/scientific_packet_review'
SCOPE = 'ROOT_MISSING_ONLY_S7_DATATAILOR_GLOBAL0_AND3_GENERAL_OVERLAYS'
PARITY_SHA = 'a05181d263a2386926ba14415846d46c34057f89538057ac4d067977c681bdbc'
PARITY_REVIEW_SHA = 'b7f7dd1da0ff29ca893630b78f59790711329c271f22f1dfa1d57e511d4237a6'


def within(path, root):
    value = Path(path)
    c.require(value.is_absolute() and '..' not in value.parts and root in value.parents
              and not any(p.is_symlink() for p in (value, *value.parents)), 'Exact nonaliased owned route required')
    return value


def verify_config(cfg, ref, review):
    c.require(cfg['schema'] == 'root-s7-dt-composite-general-v1' and cfg['owner'] == 'Root'
              and cfg['scope'] == SCOPE and cfg['source_root'] == str(SOURCE_ROOT)
              and cfg['global_shard_indices'] in ([0], [3]) and cfg['shard_count'] == 16,
              'Only the two distinct original global shards are authorized')
    shard = cfg['global_shard_indices'][0]
    c.require(cfg['site'] == dict(name='babel', partition='general', qos='normal', account='cx', user='luyangz', gpu_type='L40S')
              and cfg['allocation'] == dict(cpus=8, gpus=1, memory_mb=65536, minutes=120, reserve_seconds=300)
              and cfg['maximum_restarts'] == 3 and cfg['method'] == 'datatailor', 'Exact finite General CPU+GPU route required')
    c.require(cfg['prior_job'] == ('10919220_0' if shard == 0 else '10919402_2')
              and cfg['original_spec']['sha256'] == ('d8c377a40d4d08ad02ca21753f06c5cfc34bdccfd851f8681b9a7d4b9eaef1d3' if shard == 0
                  else 'cea019c8f2ac448a214ea45392ff42f9e966fcae89458e5a0c391cf21d8bc285')
              and cfg['actual_parity']['sha256'] == PARITY_SHA and cfg['actual_parity']['bytes'] == 14353
              and cfg['actual_parity_review']['sha256'] == PARITY_REVIEW_SHA
              and cfg['actual_parity_review']['bytes'] == 278966, 'Genuine actual native33 proof and exact predecessor required')
    c.require(review['status'] == 'PASS_INDEPENDENT_ROOT_S7_COMPOSITE_SOURCE_CONFIG_V1'
              and review['configuration'] == ref and review['sources'] == cfg['sources']
              and isinstance(review['reviewer'], str) and review['reviewer'] not in ('', AUTHOR), 'Independent exact adapter/config review required')
    for relative, digest in cfg['sources'].items():
        path = Path(relative)
        c.require(not path.is_absolute() and '..' not in path.parts and c.sha(ROOT / path) == digest,
                  'Staged frozen source closure differs: ' + relative)
    plan = cfg['composite_shards'][str(shard)]
    c.require(isinstance(plan['original_measurement'], str)
              and re.fullmatch('[0-9a-f]{64}', plan['original_measurement'])
              and isinstance(plan['legacy_databases'], list)
              and (shard != 0 or plan['legacy_databases']), 'Actual original measurement and retained paid database descriptors required')
    for ref_db in plan['legacy_databases']:
        c.require(set(ref_db) == {'path', 'bytes', 'sha256'} and Path(ref_db['path']).is_absolute()
                  and '..' not in Path(ref_db['path']).parts and type(ref_db['bytes']) is int and ref_db['bytes'] > 0
                  and re.fullmatch('[0-9a-f]{64}', ref_db['sha256']), 'Exact immutable legacy database descriptor required')
    overlay = within(plan['overlay_output'], CAMPAIGN / 'datatailor-structured-overlay-v3-root')
    c.require(overlay.name == 'shard-%04d' % shard and cfg['required_storage_bytes'] == 2 << 30,
              'One explicit nonnested overlay and bounded storage request required')
    return shard


def duration(value):
    day, clock = value.split('-', 1) if '-' in value else ('0', value)
    h, m, s = map(int, clock.split(':'))
    return int(day) * 86400 + h * 3600 + m * 60 + s


def actual(binding, cfg, runner=subprocess.check_output):
    c.require(ROOT == SOURCE_ROOT and os.getuid() == 2708899
              and pwd.getpwuid(os.getuid()).pw_name == 'luyangz', 'Exact Root Babel actor/source before writes or DATA')
    job, restart = os.environ.get('SLURM_JOB_ID', ''), os.environ.get('SLURM_RESTART_COUNT', '0')
    c.require(job.isdecimal() and job == binding['job_id'] and binding['owner'] == 'Root'
              and not os.environ.get('SLURM_ARRAY_JOB_ID') and not os.environ.get('SLURM_ARRAY_TASK_ID')
              and restart.isdecimal() and 0 <= int(restart) <= cfg['maximum_restarts']
              and re.fullmatch('[0-9a-f]{32}', binding['nonce']), 'Current scalar Root binding/restart required')
    raw = runner(['scontrol', 'show', 'job', '-o', job], text=True, timeout=30)
    lines = [line for line in raw.splitlines() if line.strip()]
    c.require(len(lines) == 1, 'Exactly one current scalar scheduler record required')
    fields = dict(re.findall(r'(?:^|\s)([^\s=]+)=(\S+)', lines[0]))
    expected = dict(JobId=job, JobState='RUNNING', UserId='luyangz(2708899)', Account='cx', Partition='general', QOS='normal',
        JobName='dsmm-root-s7-composite-g%d-v1' % cfg['global_shard_indices'][0], Comment='root-s7-dt-composite-v1:' + binding['nonce'],
        Command=str(HERE / 'run.sbatch'), WorkDir=str(ROOT), NumCPUs='8', NumNodes='1', NumTasks='1',
        Restarts=restart, Requeue='1', TimeLimit='02:00:00', NodeList=socket.gethostname().split('.')[0], TresPerNode='gres/gpu:L40S:1')
    control = within(CONTROL / ('g%d' % cfg['global_shard_indices'][0]) / ('attempt-' + job + '-r' + restart), CONTROL)
    control.mkdir(parents=True, exist_ok=True)
    c.seal(control / 'SCHEDULER_PREFLIGHT.json', dict(raw=raw, expected=expected,
        differences={k: dict(observed=fields.get(k), expected=v) for k, v in expected.items() if fields.get(k) != v}))
    c.require(all(fields.get(k) == v for k, v in expected.items()), 'Actual owned General allocation/route differs')
    for field in ('ReqTRES', 'AllocTRES'):
        tres = dict(x.split('=', 1) for x in fields[field].split(',') if '=' in x)
        c.require(tres.get('cpu') == '8' and tres.get('mem') in ('64G', '65536M') and tres.get('gres/gpu') == '1'
                  and tres.get('gres/gpu:l40s') == '1', 'Exact requested/allocated L40S CPU/memory resources required')
    c.require(runner(['scontrol', 'write', 'batch_script', job, '-'], timeout=30) == (HERE / 'run.sbatch').read_bytes(),
              'Current exact wrapper spool required')
    remaining = duration(fields['TimeLimit']) - duration(fields['RunTime'])
    c.require(remaining > 300, 'No accepted-boundary reserve remains')
    return dict(job_id=job, restart=int(restart), node=fields['NodeList'], raw_slurm=raw,
                deadline=time.time() + remaining - 300, control_attempt=str(control))


def verify_binding_claims(binding, cfg, current, runner=subprocess.check_output):
    claim = c.checked(binding['function_claim'])
    c.require(claim['owner'] == 'Root' and claim['scope'] == SCOPE and claim['configuration'] == binding['configuration']
              and claim['prior_job'] == cfg['prior_job'] and claim['S7_takeover_authorized'] is True
              and claim['no_competing_overlay_provider'] is True, 'Registered Root function takeover required')
    quiescence = c.checked(binding['prior_quiescence'])
    c.require(quiescence['owner'] == 'Root' and quiescence['prior_job'] == cfg['prior_job']
              and quiescence['original_configuration'] == cfg['original_spec']
              and quiescence['paid_prefix_preserved'] is True and quiescence['old_writer_quiescent'] is True,
              'Actual original paid-prefix custody and quiescence receipt required')
    raw = runner(['squeue', '-r', '-h', '-u', 'luyangz', '-o', '%i|%T|%r'], text=True, timeout=30)
    for line in raw.splitlines():
        parts = line.split('|')
        c.require(len(parts) == 3 and re.fullmatch(r'[0-9]+(?:_[0-9]+)?', parts[0]), 'Malformed expanded queue census')
        if parts[0] == cfg['prior_job']:
            c.require(parts[1] == 'PENDING' and parts[2] in ('JobHeldUser', 'UserHeld'), 'Original writer must remain held or be retired')
    c.seal(Path(current['control_attempt']) / 'PRIOR_QUIESCENCE_CURRENT_QUEUE.json', dict(prior_job=cfg['prior_job'], raw_queue=raw))
    storage = c.checked(binding['storage_reservation'])
    c.require(storage['owner'] == 'Root' and storage['configuration'] == binding['configuration']
              and storage['output'] == cfg['composite_shards'][str(cfg['global_shard_indices'][0])]['overlay_output']
              and storage['reserved_bytes'] >= cfg['required_storage_bytes']
              and storage['collective_claims_accounted'] is True, 'Actual collectively measured overlay storage claim required')
    return storage


def native_config(cfg, old):
    for field in ('pool', 'eligibility', 'environment', 'feature_assets', 'native_sources', 'measurement_sources', 'shard_count'):
        c.require(cfg[field] == old[field], 'Original native scientific measurement changed: ' + field)
    return dict(method='datatailor', role='features', pool=old['pool'], feature_assets=old['feature_assets'],
        environment=old['environment'], sources=old['measurement_sources'], native_sources=old['native_sources'],
        shard_index=cfg['global_shard_indices'][0], shard_count=16,
        output=str(Path(old['output']) / ('shard-%04d' % cfg['global_shard_indices'][0])), site_profile=old['feature_profile'])


def load_pool(cfg, observed):
    from experiments.root_native_input_authority_20261023 import pool_authority_v1 as authority
    # Same allocation-callback-only adaptation as the accepted S7 worker. No
    # logical, exclusion, media or identity predicate is removed or replaced.
    validator = types.FunctionType(authority.validate_pool.__code__, dict(vars(authority), require_allocation=lambda: observed),
                                   authority.validate_pool.__name__, authority.validate_pool.__defaults__)
    validator.__kwdefaults__ = authority.validate_pool.__kwdefaults__
    return validator(cfg['pool'], cfg['eligibility'])


def verify_assets(spec):
    """Retain the exact original S7 full publisher/source/asset-byte guard."""
    from experiments.root_native_baselines_20261023 import pins
    from experiments.root_native_campaign_v3_20261023.config import ASSET_PINS
    pins.verify(spec['published_sources'])
    assets = c.checked(spec['feature_assets'])
    for role in ('vicuna', 'clip336', 'stage1_projector'):
        asset = assets['assets'][role]
        c.require((asset['repository'], asset['revision']) == ASSET_PINS[role], 'Exact existing Stage1 component required')
        for description in asset['files']:
            c.require(c.desc(description['path']) == {k: description[k] for k in ('path', 'bytes', 'sha256')},
                      'Stage1 component bytes changed')
    return spec['feature_assets']


def execute(cfg, binding, observed, storage):
    from experiments.copilot7_fullpool_published_20261024 import worker as prior_worker
    from experiments.copilot7_datatailor_composite_consumer_20261025 import consumer_v3 as consumer
    old = c.checked(cfg['original_spec'])
    model_config = native_config(cfg, old)
    runtime = prior_worker.runtime(old)
    pool, records, ids = load_pool(cfg, observed)
    c.require(pool['count'] == len(ids) == len(records) == 660763, 'Complete original eligible pool required')
    verify_assets(old)
    prior_worker.verify_fidelity(old)
    proof = consumer.engine.reviewed_parity(cfg['actual_parity'], cfg['actual_parity_review'])
    qualified = c.checked(proof['configuration'])
    c.require(qualified['inputs']['environment'] == cfg['environment'] and qualified['inputs']['assets'] == cfg['feature_assets']
              and qualified['native_sources'] == cfg['native_sources']
              and all(qualified['sources'].get(name) == digest for name, digest in cfg['measurement_sources'].items()),
              'Actual native33 qualification does not cover this original scientific function')
    shard = cfg['global_shard_indices'][0]
    plan = cfg['composite_shards'][str(shard)]
    c.require(plan['original_measurement'] == consumer.engine.native.identity(model_config, pool), 'Actual original feature identity changed')
    # Import the same pinned module before the reversible reviewed hook is installed.
    sys.path.insert(0, str(ROOT / c.NATIVE))
    module = importlib.import_module('llava.train.train')
    c.require(Path(module.__file__).resolve() == (ROOT / c.NATIVE / 'llava/train/train.py').resolve(), 'Foreign native module')
    output = within(plan['overlay_output'], CAMPAIGN / 'datatailor-structured-overlay-v3-root')
    output.parent.mkdir(parents=True, exist_ok=True)
    fs = os.statvfs(output.parent)
    c.require(fs.f_bavail * fs.f_frsize >= storage['reserved_bytes'] + storage['other_outstanding_reserved_bytes']
              and fs.f_favail >= 1000, 'Fresh actual collective space/inodes below reservation')
    c.seal(Path(observed['control_attempt']) / 'ACTUAL_RUNTIME_AND_SPACE.json', dict(runtime=runtime,
        free_bytes=fs.f_bavail * fs.f_frsize, free_inodes=fs.f_favail, storage=binding['storage_reservation'],
        original_measurement=plan['original_measurement'], original_spec=cfg['original_spec']))
    stop = [False]
    def request_stop(*_):
        stop[0] = True
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    def should_stop():
        capacity = os.statvfs(output.parent)
        c.require(capacity.f_bavail * capacity.f_frsize >= 512 << 20 and capacity.f_favail >= 500,
                  'Storage safety floor breached; preserve accepted rows and fail, never retry blindly')
        return stop[0] or time.time() >= observed['deadline']
    attempt = output / ('attempt-' + observed['job_id'] + '-r' + str(observed['restart']))
    return consumer.acquire(model_config, pool, records, ids, plan['legacy_databases'], cfg['actual_parity'],
        cfg['actual_parity_review'], module, output, attempt, should_stop, bound_configuration=binding['configuration'])


def continue_or_complete(ref, observed, cfg, runner=subprocess.check_output):
    progress = c.checked(ref)
    control = Path(observed['control_attempt'])
    c.seal(control / 'RESULT.json', dict(progress=ref, job_id=observed['job_id'], restart=observed['restart'],
        accepted=progress['after'], required=progress['required'], new_forwards=progress['new_forwards'],
        fullpool_or_student_PASS=False, all_attempt_allocation_cost_requires_sacct_D=True))
    if progress['status'] == 'COMPLETE_COMPATIBLE_COMPOSITE_FEATURE_SHARD':
        return 0
    c.require(progress['status'] == 'MISSING_ONLY_COMPATIBLE_FEATURE_PROGRESS' and progress['after'] < progress['required']
              and observed['restart'] < cfg['maximum_restarts'], 'Finite incomplete attempt requires Root diagnosis; no extra successor')
    c.require(not (control / 'REQUEUE_INTENT.json').exists(), 'Existing requeue attempt requires reconciliation; no retry')
    c.seal(control / 'REQUEUE_INTENT.json', dict(job_id=observed['job_id'], progress=ref, restart=observed['restart']))
    try:
        out = runner(['scontrol', 'requeue', observed['job_id']], text=True, timeout=30)
    except Exception as error:
        c.seal(control / 'REQUEUE_AMBIGUOUS.json', dict(error_class=type(error).__name__, no_automatic_retry=True))
        raise
    c.seal(control / 'REQUEUE_ACK.json', dict(stdout=out, requested_only=True))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding', required=True)
    args = p.parse_args(argv)
    binding = c.read(within(args.binding, CONTROL))
    cfg = c.checked(binding['configuration'])
    verify_config(cfg, binding['configuration'], c.checked(binding['review']))
    observed = actual(binding, cfg)
    try:
        storage = verify_binding_claims(binding, cfg, observed)
        control = Path(observed['control_attempt'])
        with (control.parent / 'ROOT_SCALAR_EXECUTION.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ref = execute(cfg, binding, observed, storage)
            return continue_or_complete(ref, observed, cfg)
    except BaseException as error:
        c.seal(Path(observed['control_attempt']) / 'FAILURE.json', dict(error_class=type(error).__name__,
               error=str(error), job_id=observed['job_id'], restart=observed['restart'], all_allocated_cost_counts=True))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
