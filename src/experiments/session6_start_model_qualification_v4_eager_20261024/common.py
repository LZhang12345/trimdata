import ast
import json
from pathlib import Path

from experiments.copilot7_native_official_eval_20261024.common import (
    canonical, desc, key, now, require, seal, sha,
)
from experiments.root_official_protected_inputs_v1_20261024.producer import checked

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OWNER = "Session6_START_model_qualification_v4_eager_20261024"
SCHEMA = "session6-start-model-qualification-v4-eager"
REVIEW_STATUS = "PASS_INDEPENDENT_S6_START_MODEL_V4_EAGER_SOURCE_CONFIG"
PREFIX = "dsmm-s6-start-model-v4eager:"
LLAVA = ROOT / "collaboration/published-foundation-20261022/reference_sources/LLaVA/c121f0432da27facab705978f83c4ada465e46fd"
PYTHON = "/scratch/user/u.lz341613/dsmm_native_campaign_20261023/envs/native-py310-cu121-v1/bin/python"
CACHE = "/scratch/user/u.lz341613/dsmm_native_campaign_20261023/assets/hf_cache"
BASE = Path("/scratch/user/u.lz341613/dsmm_s6_official_eval_20261024/start-model-qualification-v4-eager")


def read(path):
    return json.loads(Path(path).read_bytes())


def sources():
    pending = [p for p in HERE.glob("*.py") if not p.name.startswith("test_")]
    pending += [HERE / "run.sbatch", ROOT / "experiments/session6_start_evaluation_v1_20261024/model.py"]
    pending += list((LLAVA / "llava").rglob("*.py"))
    seen = set()

    def add_module(name):
        if not name.startswith("experiments"):
            return
        parts = name.split(".")
        for length in range(1, len(parts) + 1):
            package = ROOT.joinpath(*parts[:length]) / "__init__.py"
            if package.is_file():
                pending.append(package)
        module = ROOT.joinpath(*parts).with_suffix(".py")
        if module.is_file():
            pending.append(module)

    while pending:
        path = pending.pop()
        if path in seen:
            continue
        require(path.is_file() and path.resolve().is_relative_to(ROOT), "Foreign or missing START source")
        seen.add(path)
        if path.suffix != ".py":
            continue
        package = list(path.relative_to(ROOT).parts[:-1])
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.Import):
                for name in node.names:
                    add_module(name.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = ".".join(prefix + ([node.module] if node.module else []))
                add_module(module)
                for name in node.names:
                    add_module(module + "." + name.name)
    return dict(sorted((str(path.relative_to(ROOT)), sha(path)) for path in seen))


def verify_sources(expected):
    require(expected == sources(), "Independently reviewed START source closure differs")


def validate_config(config):
    require(config["schema"] == SCHEMA and config["owner"] == OWNER
            and config["job_name"] == "dsmm-s6-start-model-v4eager"
            and config["python"] == PYTHON and config["cache_root"] == CACHE
            and config["output"] == str(BASE / "output")
            and config["training_updates"] == 0 and config["benchmark_predictions"] == 0
            and config["projector_materialization"] == "EXACT_PINNED_FP16_ASSIGN_TRUE_NO_META",
            "Only the exact no-training, no-benchmark START model qualification is authorized")
    require(config["reserved_bytes"] == 1024 ** 3, "Exact small-capsule storage reservation required")
    require(config["diagnostic_before_guard"] is True, "Durable actual observations must precede every guard")
    require(config["causal_attention_binding"] == "EXPLICIT_EAGER_INTERNAL_BEFORE_CONSTRUCTOR",
            "Causal eager implementation must be explicitly bound before native construction")
    verify_sources(config["sources"])


def validate_review(review, reference, config):
    require(review["status"] == REVIEW_STATUS and review["reviewer"] != "Session6"
            and review["reviewer_model"] in ("gpt-6.1-sol", "claude-opus-5.5")
            and review["configuration"] == reference and review["sources"] == config["sources"],
            "Genuine independent exact-source/config review required")
