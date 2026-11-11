"""Fresh path/partition profile from an existing Root-registered General allocation."""
import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import socket
import subprocess

from . import protocol as p, paid
from .driver import fields
from experiments.root_production_student_adapter_v1_20261024.launch import require_metadata_registration


def minutes(text):
    p.require(text not in ("UNLIMITED", "INFINITE"), "Finite actual partition wall ceiling")
    days, clock = text.split("-", 1) if "-" in text else ("0", text)
    values = clock.split(":")
    p.require(len(values) == 3 and all(value.isdecimal() for value in (days, *values)),
              "Actual Slurm wall syntax")
    hours, mins, seconds = map(int, values)
    return int(days) * 1440 + hours * 60 + mins + bool(seconds)


def qualify(registration_ref, output, load=p.checked, publish=p.seal, query=subprocess.check_output):
    require_metadata_registration(registration_ref, runner=query)
    p.require(os.getuid() == 2708899 and p.ROOT == p.SOURCE_ROOT, "Actual staged COINCIDE metadata process")
    old = load(paid.PROFILE)
    job, node = os.environ.get("SLURM_JOB_ID", ""), socket.gethostname().split(".")[0]
    raw_job = query(["scontrol", "show", "job", "-o", job], text=True, timeout=30)
    current = fields(raw_job)
    p.require(current["JobId"] == job and current["JobState"] == "RUNNING"
              and current["UserId"] == "luyangz(2708899)" and current["Account"] == "cx"
              and current["Partition"] == "general" and current["QOS"] == "normal"
              and current["NodeList"] == node, "Current owned General allocation required")
    raw_partition = query(["scontrol", "show", "partition", "general", "-o"], text=True, timeout=30)
    partition = fields(raw_partition)
    raw_node = query(["scontrol", "show", "node", node, "-o"], text=True, timeout=30)
    native_node = fields(raw_node)
    advertised = re.fullmatch(r"gpu:L40S:([1-9][0-9]*)(?:\([^)]*\))?", native_node["Gres"])
    p.require(partition["State"] == "UP" and "normal" in partition["AllowQos"].split(",")
              and minutes(partition["MaxTime"]) >= 720 and advertised is not None,
              "Fresh advertised General normal/case-sensitive L40S and12h support")
    for path in (Path(p.CHECKPOINT_ROOT), p.REPORT_ROOT):
        p.require(not any(parent.is_symlink() for parent in (path, *path.parents)), "Literal new owned output root")
        path.mkdir(parents=True, exist_ok=True)
    value = dict(old, site="babel", partition="general", qos="normal", account="cx", gpu_type="L40S",
        code_root=str(p.ROOT), output_root=p.CHECKPOINT_ROOT, registry_root=str(p.REPORT_ROOT),
        max_minutes=minutes(partition["MaxTime"]), max_gpus_per_node=int(advertised[1]),
        max_cpus_per_request=int(native_node["CPUTot"]), max_memory_mb=int(native_node["RealMemory"]))
    p.require(value["python"] == "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python",
              "Preserve exact current qualified native Python; no FAISS environment substitution")
    for name in ("python", "cache_root", "code_root", "output_root", "registry_root"):
        p.require(Path(value[name]).exists(), "Actual current path missing: " + name)
    now = datetime.now(timezone.utc)
    evidence = dict(value, status="PASS_CURRENT_SITE_PATHS_AND_SCHEDULER",
        created_at=now.isoformat(), valid_until=(now + timedelta(days=7)).isoformat(),
        actual_parent_registration=registration_ref, actual_job_id=job, actual_node=node,
        actual_scheduler=raw_job, actual_partition=raw_partition, actual_node_query=raw_node,
        historic_profile=paid.PROFILE, qualified_new_GPU_or_research_claim=False)
    evidence.pop("inventory", None)
    output = Path(output)
    value["inventory"] = publish(output.with_name(output.stem + "_INVENTORY.json"), evidence)
    return publish(output, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(p.canonical(qualify(p.desc(args.registration), args.output)).decode())
