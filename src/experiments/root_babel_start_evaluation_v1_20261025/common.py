"""Frozen Root binding and actual General allocation; imports no ML libraries."""
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import sys
import time

from experiments.session6_official_eval_v2_20261024.common import *

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CODE = Path("/home/luyangz/data_selection_multimodal/native_campaign_v3_20261023/src")
CONTROL = Path("/home/luyangz/data_selection_multimodal/native_campaign_v3_20261023/registry/root-start-evaluation-v1")
OUTPUT = Path("/data/user_data/luyangz/native_campaign_20261023/start-evaluation-v1")
PYTHON = "/data/user_data/luyangz/native_campaign_20261023/envs/native-fp16-xformers-v2/bin/python"
OWNER = "Root_babel_START_evaluation_v1_20261025"
PREFIX = "dsmm-root-start-eval-v1:"
REVIEW = "PASS_INDEPENDENT_BABEL_START_EVAL_SOURCE_CONFIG_V1"


def validate(cfg):
    from .banks import COUNTS
    require(cfg["schema"] == "root-babel-start-evaluation-v1" and cfg["owner"] == OWNER
            and cfg["bank"] in COUNTS and cfg["python"] == PYTHON and cfg["cache_root"] == "/data/hf_cache"
            and cfg["output"] == str(OUTPUT / cfg["bank"])
            and cfg["resources"] == dict(cpus=8, memory_mb=65536, minutes=120, gpu="L40S", gpus=1,
                                         partition="general", account="cx", qos="normal")
            and cfg["max_attempts"] == 3 and cfg["training_updates"] == 0 and cfg["model_downloads"] is False
            and cfg["reserve_seconds"] == 180 and cfg["no_student_runtime_authorization"] is True,
            "Exact finite Root starting-model bank/config required")
    require(cfg["decoding"] == dict(conv_mode="vicuna_v1", temperature=0, do_sample=False,
            num_beams=1, top_p=None, max_new_tokens=128, use_cache=True), "Native greedy loader policy changed")
    require(0 < cfg["output_limit_bytes"] <= 640 * 1024**2 and cfg["collective_output_limit_bytes"] == 1024**3,
            "Bounded collective output authorization required")
    profile = checked(cfg["site_profile"])
    inventory = checked(profile["inventory"])
    require(profile["site"] == "babel" and profile["gpu_type"] == "L40S" and profile["python"] == PYTHON
            and profile["code_root"] == str(CODE), "Genuine Babel path/scheduler profile required")
    from datetime import datetime
    require(datetime.fromisoformat(inventory["valid_until"]).timestamp() > time.time(),
            "Profile expired; do not silently extend historical observations")
    for relative, digest in cfg["sources"].items():
        p = Path(relative)
        require(not p.is_absolute() and ".." not in p.parts and sha(ROOT / p) == digest,
                "Actual reviewed source changed: " + relative)


def actual(binding, cfg):
    job = os.environ.get("SLURM_JOB_ID", "")
    require(pwd.getpwuid(os.getuid()).pw_name == "luyangz" and os.getuid() == 2708899
            and ROOT == CODE and Path.cwd() == CODE and job.isdecimal()
            and binding["owner"] == OWNER and binding["job_id"] == job
            and not os.environ.get("SLURM_ARRAY_JOB_ID") and len(binding["nonce"]) == 32,
            "Actual Root scalar actor/source/working directory required before DATA")
    require(Path(sys.executable).absolute() == Path(PYTHON), "Wrong actual native interpreter")
    intent = checked(binding["intent"])
    require(intent["owner"] == OWNER and intent["nonce"] == binding["nonce"]
            and intent["configuration"] == binding["configuration"] and intent["bank"] == cfg["bank"],
            "Exact registered pre-submit function intent required")
    review = checked(binding["review"])
    require(review["status"] == REVIEW and review["reviewer"] not in ("/root/scientific_packet_review", "scientific_packet_review")
            and review["configuration"] == binding["configuration"] and review["sources"] == cfg["sources"],
            "Genuine independent source/config review required")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True, timeout=30)
    require(len(raw.strip().splitlines()) == 1, "One actual scalar scheduler record required")
    v = dict(re.findall(r"(?:^|\s)([^\s=]+)=(\S+)", raw))
    expected = dict(JobId=job, JobState="RUNNING", JobName="dsmm-root-start-" + cfg["bank"],
        UserId="luyangz(2708899)", Account="cx", Partition="general", QOS="normal", NumCPUs="8",
        NumTasks="1", WorkDir=str(CODE), Command=str(HERE / "run.sbatch"), TimeLimit="02:00:00",
        TresPerNode="gres/gpu:L40S:1", Comment=PREFIX + binding["nonce"],
        NodeList=socket.gethostname().split(".")[0], Requeue="1")
    require(all(v.get(k) == x for k, x in expected.items()) and v.get("NumNodes") in ("1", "1-1"),
            "Actual allocation/nonce/spool route/resources differ")
    for field in ("AllocTRES", "ReqTRES"):
        tres = dict(x.split("=", 1) for x in v.get(field, "").split(",") if "=" in x)
        require(tres.get("cpu") == "8" and tres.get("mem") in ("64G", "65536M")
                and tres.get("gres/gpu") == "1" and tres.get("gres/gpu:l40s") == "1",
                "Exact allocated/requested CPU/RAM/L40S required")
    restart = v.get("Restarts", "")
    require(restart.isdecimal() and int(restart) < cfg["max_attempts"]
            and os.environ.get("SLURM_RESTART_COUNT", restart) == restart, "Wrong/exhausted physical restart")
    spool = subprocess.check_output(["scontrol", "write", "batch_script", job, "-"], timeout=30)
    require(spool == (HERE / "run.sbatch").read_bytes(), "Actual batch spool changed")
    end = time.mktime(time.strptime(v["EndTime"], "%Y-%m-%dT%H:%M:%S"))
    require(end > time.time() + cfg["reserve_seconds"], "Actual allocation reserve exhausted")
    return dict(job_id=job, restart=int(restart), node=v["NodeList"], scheduler=raw,
                end_epoch=end, uid=os.getuid(), nonce=binding["nonce"])
