"""Login-node config sealing and admission: exact request, held sbatch, read-back ACK, release.

Mirrors the v3 training submit.py under the shared ADMISSION.lock with immutable
REQUEST/INTENT/SBATCH_RESPONSE/ACK/RELEASE records. Run from the staged source
root (python3 -B -m claude_coincide_eval_v3_20261030.submit ...). /data/hf_cache
is not mounted on the login node; parent bytes, checkpoints, media and the
previous pass's RESULT.json are verified inside the allocation by worker.py.

  config --seed S --training-request <v3 training REQUEST.json> [...]   (ordered chain)
  submit --seed S --configuration <CONFIG_<nonce>.json> [--successor-of <eval REQUEST.json>] [--dry-run]
"""
import argparse
from pathlib import Path
import subprocess
import uuid
from . import common_v3 as c, protocol_v3 as p


def call(argv):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=120)
    return dict(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def sacct_state(job, runner=subprocess.check_output):
    raw = runner(['sacct', '-X', '-n', '-P', '-j', job, '--format=JobIDRaw,State,ExitCode'], text=True, timeout=60)
    rows = [line.split('|') for line in raw.splitlines() if line.strip() and line.split('|')[0] == job]
    c.require(len(rows) == 1, 'One exact accounting row for job ' + job + ': ' + raw)
    return rows[0][1].split()[0], rows[0][2], raw


def queued(job, runner=None):
    """Live squeue state, or '' when squeue no longer knows the job (finished and purged)."""
    if runner is not None:
        return runner(['squeue', '-h', '-j', job, '-o', '%T'], text=True, timeout=60).strip()
    result = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'], text=True, capture_output=True, timeout=60)
    if result.returncode != 0:
        c.require('Invalid job id' in result.stderr, 'Unexpected squeue failure: ' + result.stderr)
        return ''
    return result.stdout.strip()


def build_config(seed, training_requests, load=c.checked):
    cell = c.cell_id(seed)
    records = []
    for path in training_requests:
        c.require(Path(path).name == 'REQUEST.json', 'Pass v3 training REQUEST.json paths')
        folder = Path(path).absolute().parent
        records.append(dict(request=c.desc(folder / 'REQUEST.json'), intent=c.desc(folder / 'INTENT.json'),
                            ack=c.desc(folder / 'ACK.json'), job_id=c.read(folder / 'ACK.json')['job_id']))
    nonce = uuid.uuid4().hex
    cfg = p.configuration(seed, c.desc(c.CELLS / cell / 'CONFIG_FINAL.json'), records, nonce, load=load)
    return c.seal(c.EVAL_RECORDS / cell / ('CONFIG_' + nonce + '.json'), cfg)


def dependency(cfg, runner=subprocess.check_output):
    """afterok on the live final training job; an already purged job must be COMPLETED 0:0."""
    for job in cfg['required_training_job_ids'][:-1]:
        state, code, _ = sacct_state(job, runner)
        c.require((state, code) == ('FAILED', '75:0'), 'Earlier v3 training job is not an accepted continuation')
    final = cfg['dependency_training_job_id']
    live = queued(final, None if runner is subprocess.check_output else runner)
    if live:
        c.require(live in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'), 'Training job not progressing: ' + live)
        return ['--dependency=afterok:' + final], dict(job=final, kind='afterok_live', squeue=live)
    state, code, raw = sacct_state(final, runner)
    c.require((state, code) == ('COMPLETED', '0:0'), 'Final training job did not complete 0:0: ' + raw)
    return [], dict(job=final, kind='already_completed_0_0', sacct=raw)


def build(seed, configuration, successor_of=None, runner=subprocess.check_output):
    c.require(c.ROOT == c.SOURCE_ROOT, 'Submit from the staged v3 evaluation source root')
    cell = c.cell_id(seed)
    cfg_ref = c.desc(configuration)
    cfg = c.checked(cfg_ref)
    c.require(Path(cfg_ref['path']).parent == c.EVAL_RECORDS / cell and cfg['training_seed'] == seed,
              'Configuration belongs to this cell namespace')
    p.validate_science(cfg, c.checked(cfg['qualified_basis']), c.checked(cfg['training_configuration']))
    review_ref = c.desc(c.REVIEW)
    c.independent(c.read(review_ref['path']), cfg['sources'], cfg_ref, cell)
    if successor_of is None:
        predecessor, previous_attempts = None, []
        flags, dep = dependency(cfg, runner)
    else:
        prev_ref = c.desc(successor_of)
        prev = c.checked(prev_ref)
        prev_job = c.read(Path(prev_ref['path']).parent / 'ACK.json')['job_id']
        c.require(prev['schema'] == c.REQUEST_SCHEMA and prev['configuration'] == cfg_ref and prev['seed'] == seed,
                  'Successor must continue the same reviewed configuration')
        state, code, raw = sacct_state(prev_job, runner)
        c.require((state, code) == ('FAILED', '75:0'), 'Predecessor pass did not end in accepted continuation 75')
        predecessor = dict(job_id=prev_job, attempt=prev_job + '-r0', request=prev_ref, sacct=raw)
        previous_attempts = prev['previous_attempts'] + [prev_job + '-r0']
        flags, dep = [], dict(job=prev_job, kind='terminal_predecessor_pass', sacct=raw)
    return dict(schema=c.REQUEST_SCHEMA, owner='Claude-baseline-recovery', seed=seed, cell=cell,
        nonce=uuid.uuid4().hex, configuration=cfg_ref, review=review_ref, sources=cfg['sources'],
        sbatch=c.desc(c.HERE / 'run.sbatch'), predecessor=predecessor, previous_attempts=previous_attempts,
        dependency=dep, dependency_flags=flags, exclude=list(cfg['operational_node_exclusion']['excluded_nodes']),
        automatic_requeue=False, UTC=c.now()), cfg


def readback(f, name, request, flags):
    expected = dict(JobName=name, Partition='general', QOS='normal', Account='cx', TresPerNode='gres/gpu:L40S:1',
        Comment=c.COMMENT_PREFIX + request['nonce'], WorkDir=str(c.ROOT), Command=str(c.HERE / 'run.sbatch'),
        TimeLimit=c.TIME_LIMIT, JobState='PENDING', NumCPUs='8', Requeue='0', ExcNodeList=','.join(request['exclude']) or '(null)')
    bad = {k: (f.get(k), v) for k, v in expected.items() if f.get(k) != v}
    c.require(not bad, 'Read-back differs: ' + repr(bad))
    for flag in flags:
        # e.g. Dependency=afterok:10986422(unfulfilled)
        c.require(f.get('Dependency', '').split('(')[0] == flag.split('=', 1)[1], 'Dependency differs in read-back')


def submit(seed, configuration, successor_of=None, dry_run=False):
    import fcntl
    request, cfg = build(seed, configuration, successor_of)
    if dry_run:
        return request
    records = c.EVAL_RECORDS / request['cell'] / request['nonce']
    name = c.job_name(seed)
    argv = ['sbatch', '--parsable', '--hold', '--job-name=' + name, '--comment=' + c.COMMENT_PREFIX + request['nonce'],
        '--time=' + c.TIME_LIMIT, '--chdir=' + str(c.ROOT), '--no-requeue',
        '--output=' + str(records / 'slurm-%j.out')] + request['dependency_flags'] \
        + (['--exclude=' + ','.join(request['exclude'])] if request['exclude'] else []) \
        + [str(c.HERE / 'run.sbatch'), str(records / 'REQUEST.json')]
    with open(c.ADMISSION, 'r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        live = c.live_eval_jobs(subprocess.check_output(['squeue', '-u', c.USER, '-h', '-o', '%i|%j|%T'], text=True), seed)
        c.require(not live, 'Live evaluator already exists for this cell: ' + repr(live))
        ref = c.seal(records / 'REQUEST.json', request)
        intent = c.seal(records / 'INTENT.json', dict(request=ref, argv=argv, UTC=c.now()))
        response = call(argv)
        c.seal(records / 'SBATCH_RESPONSE.json', response)
        c.require(response['returncode'] == 0, 'sbatch failed; intent and raw response retained')
        job = response['stdout'].strip().split(';')[0]
        c.require(job.isdecimal(), 'Ambiguous sbatch acknowledgement; reconcile by comment before any retry')
        raw = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True)
        f = c.fields(raw)
        readback(f, name, request, request['dependency_flags'])
        c.seal(records / 'ACK.json', dict(intent=intent, job_id=job, state=f['JobState'], reason=f['Reason'],
            raw=raw, UTC=c.now()))
        release = call(['scontrol', 'release', job])
        c.seal(records / 'RELEASE.json', dict(job_id=job, response=release, UTC=c.now()))
        c.require(release['returncode'] == 0, 'Release failed; job remains held with ACK recorded')
    return dict(job_id=job, records=str(records))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest='command', required=True)
    k = sub.add_parser('config'); k.add_argument('--seed', type=int, required=True)
    k.add_argument('--training-request', action='append', required=True)
    s = sub.add_parser('submit'); s.add_argument('--seed', type=int, required=True)
    s.add_argument('--configuration', required=True); s.add_argument('--successor-of')
    s.add_argument('--dry-run', action='store_true')
    v = a.parse_args()
    if v.command == 'config':
        value = build_config(v.seed, v.training_request)
    else:
        value = submit(v.seed, v.configuration, v.successor_of, v.dry_run)
    print(c.canonical(value).decode())


if __name__ == '__main__':
    main()
