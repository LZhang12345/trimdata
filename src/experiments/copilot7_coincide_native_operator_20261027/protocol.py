"""Standard-library admission/identity contract for S1 execution of S7's operator."""
import hashlib
import importlib.util
import os
from pathlib import Path
import pwd
import socket
import subprocess


ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/home/luyangz/data_selection_multimodal")
CONTROL = BASE / "collaboration/session7-coincide-native-operator-20261027/v3"
LOCK = BASE / "collaboration/babel-two-track-20261019/ADMISSION.lock"
CLAIM = BASE / "collaboration/babel-two-track-20261019/FUNCTION_S7_COINCIDE_NATIVE_OPERATOR_V1.json"
OUTPUT = Path("/data/hf_cache/project_features/DS-MultiModal/coincide_native_operator_20261027/v3")
MATRIX_ROOT = Path("/data/hf_cache/project_features/DS-MultiModal/coincide_operator_inputs_20261026/v1")
PYTHON = "/data/hf_cache/project_envs/DS-MultiModal/coincide-faiss180-torch220-cu121-v2/bin/python"
WRAPPER = "experiments/copilot7_coincide_native_operator_20261027/run.sbatch"
JOB_NAME = "dsmm-s7-coincide-native-operator-v3"
COMMENT = "s7-coincide-native-operator:"
REVIEW_STATUS = "PASS_EXACT_S7_COINCIDE_NATIVE_OPERATOR_QUAL_SOURCE_CONFIG_V3"
FUNCTION = "COINCIDE.native660763.FAISS10000-50-spherical-seed1234.MMDgamma1-temp0.1.p05-p10-p20.v1"
ALLOCATION = dict(gpus=1, cpus=8, memory_mb=196608, minutes=360)
BUDGETS = {"0.05": 33038, "0.10": 66076, "0.20": 132152}
MEASUREMENT = "92cb292ae3b6fb36ce61b314455800c27748e9fe7e6fa4972420dd0126643301"
HELPER = ROOT / "experiments/copilot7_coincide_budget4_20261025/export_v3.py"
HELPER_SHA = "bdaf8fbb6b69fa01a9f73bfc9dd249fab5d5f275fb41df5da4c6d967375acaae"


def metadata():
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError("Exact reviewed standard-library metadata helper required")
    spec = importlib.util.spec_from_file_location("_s7_coin_operator_metadata", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require(Path(module.__file__) == HELPER, "Metadata helper origin differs")
    return module


m = metadata()


def key(value):
    return hashlib.sha256(m.canonical(value)).hexdigest()


def actor():
    m.require(os.getuid() == 2708899 and pwd.getpwuid(os.getuid()).pw_name == "luyangz",
              "Actual Babel actor required before mutation or DATA")
    os.umask(0o077)


def function_identity(spec):
    return key({name: spec[name] for name in (
        "function", "pool", "eligibility", "matrix_receipt", "feature_measurement",
        "operator_python", "environment_configuration", "environment_probe", "native_operator",
        "budgets", "sources", "output")})


def validate(configuration, review_ref):
    m.require(configuration["path"] == str(CONTROL / "CONFIG.json")
        and review_ref["path"] == str(CONTROL / "INDEPENDENT_REVIEW.json"),
        "Exact frozen operator configuration/review paths required")
    spec, review = m.checked(configuration), m.checked(review_ref)
    m.require(spec["schema"] == "s7-native-COINCIDE-operator-qualification-v3"
        and spec["function_owner"] == "Session7" and spec["admission_owner"] == "Session1"
        and spec["source_root"] == str(ROOT) and spec["output"] == str(OUTPUT)
        and spec["allocation"] == ALLOCATION and spec["operator_python"] == PYTHON
        and spec["function"] == FUNCTION and spec["feature_measurement"] == MEASUREMENT
        and spec["budgets"] == BUDGETS and spec["native_operator"] == dict(
            clusters=10000, spherical_iterations=50, seed=1234, gamma=1, temperature=0.1,
            precision="author_FP32_FAISS_and_FP64_CUDA_MMD", top_up=False),
        "Exact full-pool native operator, three budgets and delegated ownership required")
    m.require(review["status"] == REVIEW_STATUS and review["independent"] is True
        and review["configuration"] == configuration and review["sources"] == spec["sources"],
        "Genuine independent exact source/config review required")
    for relative, digest in spec["sources"].items():
        path = Path(relative)
        m.require(not path.is_absolute() and ".." not in path.parts
            and m.reference(ROOT / path)["sha256"] == digest, "Reviewed execution source changed: " + relative)
    authority = Path(spec["delegation"]["path"])
    m.require(ROOT in authority.parents and m.reference(authority) == spec["delegation"],
              "Exact staged author delegation required")
    text = authority.read_text()
    m.require("Session1" in text and "Priority3: faithful published-baseline" in text,
              "S1 scoped published-provider execution authority missing")
    m.require(spec["matrix_receipt"] == dict(path=str(MATRIX_ROOT / "COMPLETE_INPUTS.json"), bytes=6230,
            sha256="1e0c1c6ee7b9431cdb2d1b107651c7297924cfa788384fccdb38836dba0e0431")
        and spec["new_features_models_or_downloads_allowed"] is False
        and spec["student_or_evaluation_release_authorized"] is False
        and spec["automatic_successor_authorized"] is False
        and spec["minimum_output_reservation_bytes"] == 8 << 30
        and spec["minimum_output_reserved_inodes"] == 1000,
        "Reuse exact paid inputs; no acquisition or fabricated output capacity")
    return spec


def storage(spec, reference):
    m.require(isinstance(reference, dict)
        and str(reference.get("path", "")).startswith(str(BASE / "collaboration") + "/"),
        "Actual delegated S1/Root HOME storage reservation descriptor required before admission")
    value = m.checked(reference)
    m.require(value["status"] == "PASS_EXCLUSIVE_MEASURED_STORAGE_RESERVATION"
        and value["owner"] in ("Root", "Session1")
        and value["output"] == str(OUTPUT)
        and value["function_identity"] == function_identity(spec)
        and value["cross_owner_commitments_accounted"] is True
        and value["reserved_bytes"] >= spec["minimum_output_reservation_bytes"]
        and value["reserved_inodes"] >= spec["minimum_output_reserved_inodes"]
        and value["output"] != str(MATRIX_ROOT),
        "Distinct measured native-operator output lease must not spend the paid matrix/feature lease")
    return value


def binding_identity(binding, spec):
    m.require(binding["owner"] == "Session1" and binding["function_owner"] == "Session7"
        and binding["job_id"].isdecimal()
        and binding["intent"]["path"] == str(CONTROL / "registration/INTENT.json")
        and binding["function_claim"]["path"] == str(CLAIM),
        "Exact delegated executor and retained S7 function paths required")
    intent, claim = m.checked(binding["intent"]), m.checked(binding["function_claim"])
    m.require(intent["configuration"] == binding["configuration"] and intent["review"] == binding["review"]
        and intent["storage_reservation"] == binding["storage_reservation"]
        and claim == dict(owner="Session7", admitted_by="Session1", function=FUNCTION,
            function_identity=function_identity(spec), configuration=binding["configuration"],
            intent=binding["intent"]),
        "Exact registered delegated intent/function identity differs")
    storage(spec, binding["storage_reservation"])
    return intent


def observe(binding, spec, *, held=False, running=False):
    intent = binding_identity(binding, spec)
    raw = m.call(["scontrol", "show", "job", "-o", binding["job_id"]])
    m.require(len([line for line in raw.splitlines() if line.strip()]) == 1, "Exact scalar readback required")
    value = m.fields(raw)
    try:
        expected = dict(JobId=binding["job_id"], JobName=JOB_NAME, UserId="luyangz(2708899)",
            Account="cx", Partition="general", QOS="normal", NumCPUs="8", NumTasks="1",
            TimeLimit="06:00:00", Requeue="0", Restarts="0", TresPerNode="gres/gpu:L40S:1",
            Command=str(ROOT / WRAPPER), WorkDir=str(ROOT), Comment=COMMENT + intent["nonce"])
        m.require(all(value.get(name) == field for name, field in expected.items())
            and value["NumNodes"] in ("1", "1-1")
            and not any(name.startswith(("Array", "HetJob")) for name in value),
            "Exact delegated operator job/source/route/resource contract differs")
        tres = dict(token.split("=", 1) for token in value["ReqTRES"].split(",") if "=" in token)
        m.require(tres.get("cpu") == "8" and tres.get("node") == "1"
            and tres.get("gres/gpu") == tres.get("gres/gpu:l40s") == "1"
            and tres.get("mem") in ("192G", "196608M"), "Actual operator request must be1L40S8CPU192GiB")
        if held:
            required = dict(JobState="PENDING", Reason="JobHeldUser", Priority="0",
                RunTime="00:00:00", StartTime="Unknown", AllocTRES="(null)")
            m.require(all(value.get(name) == field for name, field in required.items())
                and "NodeList" in value and value["NodeList"] in ("", "(null)"),
                "Fresh never-started held ACK required")
        if running:
            allocated = dict(token.split("=", 1) for token in value["AllocTRES"].split(",") if "=" in token)
            m.require(value["JobState"] == "RUNNING" and value["NumNodes"] == "1"
                and value["NodeList"] == socket.gethostname().split(".")[0]
                and allocated.get("gres/gpu") == "1" and allocated.get("mem") in ("192G", "196608M"),
                "Actual exact owned allocation required before DATA or native computation")
        spool = subprocess.check_output(["scontrol", "write", "batch_script", binding["job_id"], "-"], timeout=30)
        m.require(spool == (ROOT / WRAPPER).read_bytes(), "Actual accepted operator spool differs")
    except Exception as error:
        error.raw_slurm = raw
        raise
    return dict(raw=raw, fields=value, source_spool_verified=True)
