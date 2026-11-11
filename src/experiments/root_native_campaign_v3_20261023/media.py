"""Transport-only accepted interruption, before any neural work begins."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import time
from . import common as c

class StagingContinuation(InterruptedError):pass

@contextmanager
def staging_boundary():
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGUSR1,signal.SIGTERM)}
    def requested(*_):raise StagingContinuation('Scheduler requested a pre-neural media boundary')
    for sig in previous:signal.signal(sig,requested)
    try:
        try:yield
        except InterruptedError as error:raise StagingContinuation(str(error)) from error
    finally:
        for sig,value in previous.items():signal.signal(sig,value)

def require_launch_time(allocation):
    end=int(os.environ['SLURM_JOB_END_TIME'])
    if end-time.time()<=allocation['reserve_seconds']+120:
        raise StagingContinuation('Insufficient remaining time for a new native launch after verified staging')

def requeue(binding,attempt,reason):
    job=os.environ.get('SLURM_JOB_ID','')
    c.require(job.isdecimal() and job==binding['job_id'],'Foreign staging continuation')
    attempt=Path(attempt)
    c.seal(attempt/'MEDIA_STAGING_REQUEUE_INTENT.json',dict(job_id=job,
        actual_restart_count=os.environ.get('SLURM_RESTART_COUNT','0'),UTC=c.now(),reason=str(reason),
        neural_training_started=False,same_physical_job_only=True))
    response=subprocess.run(['scontrol','requeue',job],capture_output=True,text=True)
    c.seal(attempt/'MEDIA_STAGING_REQUEUE_RESPONSE.json',dict(rc=response.returncode,
        stdout=response.stdout,stderr=response.stderr))
    c.require(response.returncode==0,'Staging requeue acknowledgement unresolved; no new successor permitted')
