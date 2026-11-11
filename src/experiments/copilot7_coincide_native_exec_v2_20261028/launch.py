"""Authenticated current A child; Root alone admits/releases physical jobs."""
import os
from pathlib import Path
from . import protocol as p


def require_current(ref,config_ref,permit_ref,config,profile,runner=None):
    value=p.checked(ref)
    p.require(value["schema"]=="root-longwall-student-child-binding-v1" and value["owner"]=="Root"
        and value["configuration"]==config_ref and value["authorization"]==permit_ref
        and value["adapter_sources"]==p.sources()
        and value["actual_restart_count"]==int(os.environ.get("SLURM_RESTART_COUNT","0")),
        "Exact new A current child binding, never old physical scope")
    from .driver import current
    binding,_,actual,activation=current(value["physical_binding"],value["activation"],runner=runner,minimum_seconds=300)
    p.require(activation["configuration"]==config_ref and value["physical_job_id"]==binding["job_id"]
        and value["nonce"]==binding["nonce"] and value["command"]==binding["command"],
        "Current A physical/config/nonce graph")
    return dict(binding=ref,actual_slurm=actual["scheduler"],actual_job_id=binding["job_id"],
        actual_EndTime=actual["end_epoch"],scheduler_mutations=False,Root_owns_atomic_requeuehold=True)


def require_attempt(attempt,config,profile,current,binding):
    attempt=Path(attempt);report=Path(config["report"])
    name=current["actual_job_id"]+"-r"+os.environ.get("SLURM_RESTART_COUNT","0")+"-"+binding["nonce"][:8]
    expected=report/"attempts"/name/"child"
    p.require(attempt==expected and binding["child_attempt"]==str(expected) and not attempt.exists()
        and p.n.within(report,profile["registry_root"])
        and not any(path.is_symlink() for path in (attempt,*attempt.parents)),
        "Fresh literal owned A child attempt")
    return expected


from experiments.root_production_student_adapter_v2_20261025.launch import require_metadata_registration
