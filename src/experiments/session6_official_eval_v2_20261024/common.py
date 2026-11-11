"""Versioned evaluation identities and durable metadata."""
from pathlib import Path

from experiments.copilot7_native_official_eval_20261024.common import (
    canonical, checked, desc, key, now, read, require, seal, sha,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OWNER = "Session6_official_eval_v2_20261024"
LLAVA_COMMIT = "c121f0432da27facab705978f83c4ada465e46fd"
LLAVA = ROOT / "collaboration/published-foundation-20261022/reference_sources/LLaVA" / LLAVA_COMMIT
SCORER_PINS = {
    "m4c_evaluator.py": "8b49e1033e29d551c0bdd86d9149e2b54370e4cd17e52176ac9061b3521f0774",
    "eval_pope.py": "a4fe52fad8387f104dd66192f50089fe2991d651cf62fb79f07670cefd932845",
    "eval_science_qa.py": "e57b92cafc7a4c0d0228bc0554a4315ef808e844d009dcaafa3b0f6ac693fbaf",
}
METRICS = (
    "VQAv2", "GQA", "VizWiz", "ScienceQA_image", "TextVQA", "POPE",
    "MME_perception", "MME_cognition", "MMBench_EN", "MMBench_CN",
    "SEED_image", "LLaVA_Bench_in_the_Wild", "MM_Vet",
    "MMMU_validation", "NoCaps_validation",
)


def scorer_source(name):
    require(name in SCORER_PINS, "Unknown official scorer source")
    path = LLAVA / "llava/eval" / name
    require(sha(path) == SCORER_PINS[name], "Pinned official scorer changed: " + name)
    return path


def sources():
    paths = [HERE / name for name in (
        "__init__.py", "common.py", "journal.py", "scoring.py",
        "setup_worker.py", "admit_setup.py", "run_setup.sbatch",
    )]
    paths.append(ROOT / "experiments/copilot7_native_official_eval_20261024/common.py")
    paths.extend(scorer_source(name) for name in SCORER_PINS)
    return dict(sorted((str(p.relative_to(ROOT)), sha(p)) for p in paths))


def verify_sources(expected):
    require(expected == sources(), "Evaluation source closure differs from exact review")
