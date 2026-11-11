"""Session1 CLI: one reviewed conditional HELD scalar; no release/requeue API."""
import argparse
import fcntl
import os
import re
import subprocess
import uuid
from pathlib import Path
from experiments.root_fulltrajectory_held_cohort_v2_20261025 import admit as ra
from . import protocol as p

fields = ra.fields
wall = ra.wall


def census(site,query):
    return ra.census(site,query) # actualsite8GPU/50submit, no privateS1 slot reserve


def job_name(request):
    return 'dsmm-root-s1-ft-v2-' + request['site'] + '-' + request['cell_id']


def held(raw, job, b, request):
    lines = [x for x in raw.splitlines() if x.strip()]
    p.c.require(len(lines) == 1, 'One scalar held scheduler row required')
    f = fields(lines[0])
    s = p.rp.SITES[request['site']]
    expected = dict(JobId=job, JobName=b['job_name'], JobState='PENDING', Reason='JobHeldUser',
                    UserId=s['user'] + '(' + str(s['uid']) + ')', Account=s['account'], Partition=s['partition'],
                    QOS=s['qos'], NumCPUs='8', NumTasks='1', TimeLimit=wall(request['outer_allocation']['minutes']),
                    Command=b['command']['path'], WorkDir=str(p.ROOT), Comment=p.COMMENT + b['nonce'],
                    TresPerNode='gres/gpu:' + s['gpu'] + ':1', Restarts='0', RunTime='00:00:00', StartTime='Unknown',
                    ExitCode='0:0')
    p.c.require(all(f.get(k) == v for k, v in expected.items()) and f.get('NumNodes') in ('1', '1-1')
                and all(f.get(k) in (None, '(null)', 'None')
                        for k in ('AllocTRES', 'ArrayJobId', 'ArrayTaskId', 'NodeList', 'Dependency')),
                'Exact never-started owned scalar/route/nonce differs')
    t = dict(x.split('=', 1) for x in f.get('ReqTRES', '').split(',') if '=' in x)
    p.c.require(t.get('cpu') == '8' and t.get('mem') in ('64G', '65536M') and t.get('gres/gpu') == '1',
                'Held requested resources differ')
    return f


def submit(request_ref, run=subprocess.run, query=subprocess.check_output):
    preliminary = p.c.checked(request_ref)
    p.rp.actor(preliminary['site'])
    request, expected, profile = p.reviewed(request_ref, admission=True)
    s = p.rp.SITES[request['site']]
    lock = Path(s['lock'])
    p.c.require(lock.is_file() and not any(x.is_symlink() for x in (lock, *lock.parents)),
                'Existing canonical shared admission lock required')
    root = Path(request['control_root'])
    command = p.c.desc(Path(__file__).with_name('run.sbatch'))
    with lock.open('r+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        p.rp.actor(request['site'])
        request, expected, profile = p.reviewed(request_ref, admission=True)
        p.c.require(not root.exists(), 'Existing intent/ACK/function claim requires reconciliation; no retry')
        current = census(request['site'], query)
        name = job_name(request)
        p.c.require(not any(x.split('|')[5] in (name,'dsmm-s1-ft-babel-'+request['cell_id']) for x in current['raw'].splitlines() if x),
                    'Same logical trajectory already queued')
        root.mkdir(parents=True, exist_ok=False)
        nonce = uuid.uuid4().hex
        claim = p.c.seal(root / 'CONDITIONAL_FUNCTION_CLAIM.json', dict(
            owner=p.OWNER, status='HELD_CONDITIONAL_NO_ACTIVE_TRAINING_RIGHT', scientific_design=expected,
            site=request['site'], request=request_ref, reserved_storage_bytes=0,
            future_eligible_storage_reservation_required=True, source_ready_not_runtime_READY=True))
        argv = ['sbatch', '--parsable', '--hold', '--requeue', '--open-mode=append', '--nodes=1', '--ntasks=1',
                '--account=' + s['account'], '--partition=' + s['partition'], '--qos=' + s['qos'],
                '--gres=gpu:' + s['gpu'] + ':1', '--cpus-per-task=8', '--mem=64G',
                '--time=' + wall(request['outer_allocation']['minutes']), '--job-name=' + name,
                '--comment=' + p.COMMENT + nonce, '--chdir=' + str(p.ROOT), '--output=' + str(root / 'slurm-%j.out'),
                command['path'], str(root / 'BINDING.json'), profile['python']]
        intent = p.c.seal(root / 'SUBMISSION_INTENT.json', dict(
            owner=p.OWNER, request=request_ref, nonce=nonce, argv=argv, sources=p.sources(), function_claim=claim,
            actual_census=current, held_only=True, release_authorized=False, UTC=p.c.now()))
        env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'SBATCH_', 'SRUN_'))}
        try:
            r = run(argv, text=True, capture_output=True, timeout=60, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            p.c.seal(root / 'AMBIGUOUS_SUBMISSION.json', dict(intent=intent, error=type(exc).__name__,
                                                              automatic_retry=False))
            raise
        response = p.c.seal(root / 'SUBMISSION_RESPONSE.json', dict(intent=intent, returncode=r.returncode,
                                                                     stdout=r.stdout, stderr=r.stderr))
        m = re.fullmatch(r'(\d+)(?:;[^\s]+)?\s*', r.stdout)
        if r.returncode != 0 or m is None:
            p.c.seal(root / 'AMBIGUOUS_SUBMISSION.json', dict(intent=intent, response=response, automatic_retry=False))
            raise ValueError('Unknown scheduler acknowledgement: no automatic retry')
        job = m[1]
        ack = p.c.seal(root / 'ACK.json', dict(intent=intent, response=response, job_id=job, nonce=nonce))
        b = dict(schema='s1-fulltrajectory-held-binding-v2', owner=p.OWNER, job_id=job, nonce=nonce, job_name=name,
                 request=request_ref, command=command, sources=p.sources(), intent=intent, ack=ack,
                 function_claim=claim, held_only=True)
        raw = query(['scontrol', 'show', 'job', '-o', job], text=True, timeout=30)
        held(raw, job, b, request)
        spool = query(['scontrol', 'write', 'batch_script', job, '-'], timeout=30)
        p.c.require(spool == Path(command['path']).read_bytes(), 'Held spool differs')
        target = root / 'batch_script.sh'
        with target.open('xb') as out:
            out.write(spool)
            out.flush()
            os.fsync(out.fileno())
        b['spool'] = p.c.desc(target)
        p.c.seal(root / 'HELD_READBACK.json', dict(job_id=job, raw=raw, command=command, spool=b['spool'],
                                                    no_release=True))
        return p.c.seal(root / 'BINDING.json', b)


def main():
    a = argparse.ArgumentParser(description='One conditional HELD S1 request only. No release command.')
    a.add_argument('--request', required=True)
    v = a.parse_args()
    print(p.c.canonical(submit(p.c.desc(v.request))).decode())


if __name__ == '__main__':
    main()
