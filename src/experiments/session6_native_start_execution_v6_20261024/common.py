import os
from pathlib import Path
import pwd
import sys

from experiments.session6_start_model_qualification_v4_eager_20261024 import common as model_common
from experiments.session6_start_model_qualification_v4_eager_20261024.common import (
    canonical, checked, desc, key, now, read, require, seal, sha,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = Path("/scratch/user/u.lz341613/dsmm_s6_official_eval_20261024/native-start-execution-v6")
OWNER = "Session6_native_START_execution_v6_20261024"
SCHEMA = "session6-native-start-execution-v6"
REVIEW_STATUS = "PASS_INDEPENDENT_S6_NATIVE_START_EXECUTION_V6_SOURCE_CONFIG"
PREFIX = "dsmm-s6-start-exec-v6:"
SHARDS = 32
QUESTIONS = 5000
IMAGES = 3166
ENCODED_BYTES = 916425313
PHASES = ("prepare", "predict", "score", "dispatch")
LLAVA = model_common.LLAVA
PROTOCOL_SHA = "f1ff1af008482dfba1ee9945065ee57d0e38592fef30ee75d0353545e9746416"
MODEL_STATUS = "PASS_SCOPED_ACTUAL_START_MODEL_NO_TRAINING_ROUNDTRIP"
READY_STATUS = "PASS_FULL_TEXTVQA_PHYSICAL_MEDIA_MODEL_NATIVE_GENERATION_READY"


def sources():
    result = model_common.sources()
    files = [p for p in HERE.glob("*.py") if not p.name.startswith("test_")]
    files += [HERE / "run.sbatch"]
    files += [ROOT / "experiments/session6_start_evaluation_v1_20261024/kernel.py",
              ROOT / "experiments/session6_official_eval_v2_20261024/journal.py",
              ROOT / "experiments/session6_official_eval_v2_20261024/scoring.py"]
    result.update((str(p.relative_to(ROOT)), sha(p)) for p in files)
    return dict(sorted(result.items()))


def validate_config(config):
    require(config["schema"] == SCHEMA and config["owner"] == OWNER
            and config["bank"] == "TextVQA" and config["shards"] == SHARDS
            and config["questions"] == QUESTIONS and config["unique_images"] == IMAGES
            and config["encoded_bytes"] == ENCODED_BYTES
            and config["output"] == str(BASE / "output")
            and config["python"] == model_common.PYTHON
            and config["cache_root"] == model_common.CACHE
            and config["sources"] == sources(), "Exact frozen full-TextVQA array configuration required")
    require(config["evaluation_protocol"]["sha256"] == PROTOCOL_SHA,
            "Inference protocol is not the Root input-protection protocol")
    require(config["metadata_result"]["sha256"] ==
            "49cc24e490a19f4bc42259d4b69f120f192689e2d1901e16b1d2b3a6466cf369",
            "Reuse the actual completed2250425 metadata result")
    require(config["media"]["sha256"] ==
            "a7ab4f3947b4fa470b0969e14ff5b6f5fe8ac71860e6535a755143e859d9b9c2"
            and config["annotations"]["sha256"] ==
            "4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d",
            "Exact completed official TextVQA media/GT identities required")
    root = Path(config["image_root"])
    approved = Path("/scratch/user/u.lz341613/dsmm_native_campaign_20261023/assets")
    require(root.is_absolute() and root.resolve().is_relative_to(approved.resolve()),
            "S5-owned exact official media must be in approved existing cache")
    require(config["reserved_bytes"] == 2 * 1024 ** 3
            and config["attempt_seconds"] == 900 and config["max_attempts"] == 8,
            "Exact finite attempt/storage policy required")
    require(config["identity_comparison"] == "COMPLETE_TYPED_PREIMAGE_AND_STRICT_JSON_DOMAIN_ALL_FIELDS"
            and config["automatic_continuation"] == "FINITE_CPU_AFTER_PREPARE_THEN32_AND_DEPENDENT_CPU_SCORE",
            "Strict all-component identity and finite exact continuation required")


def validate_review(review, reference, config):
    actor = review.get("reviewer")
    require(review["status"] == REVIEW_STATUS and isinstance(actor, str) and actor.strip()
            and actor != OWNER and not actor.startswith("Session6")
            and review["configuration"] == reference and review["sources"] == config["sources"],
            "Genuine independent exact-source/config review required")


def admission_context():
    uid = os.getuid()
    require(uid == 56219 and pwd.getpwuid(uid).pw_name == "u.lz341613",
            "Exact actual ACES actor UID/name required before any write or RPC")
    expected_root = BASE / "src"
    require(ROOT.resolve() == expected_root.resolve()
            and HERE.resolve() == (expected_root / "experiments/session6_native_start_execution_v6_20261024").resolve()
            and Path.cwd().resolve() == expected_root.resolve()
            and Path(sys.executable).absolute() == Path(model_common.PYTHON).absolute(),
            "Exact approved deployed source root/cwd/interpreter required before any write or RPC")
    return dict(uid=uid, user="u.lz341613", source_root=str(ROOT), cwd=str(Path.cwd()))


def route(phase):
    require(phase in PHASES, "Unknown START phase")
    cpu_only = phase in ("score", "dispatch")
    return dict(partition="cpu" if cpu_only else "gpu",
                cpus=(1 if phase == "dispatch" else 4) if cpu_only else 8,
                memory_GiB=(2 if phase == "dispatch" else 16) if cpu_only else 64, minutes=20,
                GPU_GRES=None if cpu_only else "gpu:h100:1")


def partition(rows, index, count=SHARDS):
    require(type(index) is int and type(count) is int and 0 <= index < count <= len(rows),
            "Exact nonempty finite shard index/count required")
    return rows[len(rows) * index // count:len(rows) * (index + 1) // count]


def job_name(phase):
    require(phase in PHASES, "Unknown START phase")
    return "dsmm-s6-start-tvq-" + phase + "-v6"


def registration(phase):
    require(phase in PHASES, "Unknown START phase")
    return BASE / "registration" / phase


def ready(config, reference):
    value = checked(desc(Path(config["output"]) / "BANK_READY.json"))
    binding = checked(value["producer_binding"])
    require(value["status"] == READY_STATUS and value["configuration"] == reference
            and binding["owner"] == OWNER and binding["phase"] == "prepare"
            and binding["configuration"] == reference
            and value["actual"]["job_id"] == binding["job_id"]
            and value["producer_binding"]["path"] == str(registration("prepare") / "BINDING.json")
            and value["sources"] == config["sources"]
            and value["questions"] == QUESTIONS and value["images"] == IMAGES,
            "Actual owned full-bank readiness, not a fixture or input-only declaration, required")
    return value
