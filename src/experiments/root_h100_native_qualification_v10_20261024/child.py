"""Observed H100 native arm using a validated, current-attempt physical alias."""
import argparse
import importlib
import os
from pathlib import Path
import sys
import time
import traceback

from . import protocol as p
from experiments.root_native_qualification_v10_20261024.observer import Audit


def run(study, study_ref, arm, phase, physical_ref, alias_ref, attempt, stop_at):
    from experiments.root_native_qualification_v10_20261024.preflight import runtime_preflight
    runtime = runtime_preflight(study, site='aces')
    p.seal(Path(attempt)/'CURRENT_TOKENIZER_RUNTIME.json',runtime)
    configs, modules = p.configs(study)
    logical, config = configs[arm], p.checked(physical_ref)
    expected = p.physical_config(logical, alias_ref, validate_native=modules["config"].validate,
                                 run_identity=modules["common"].run_identity)
    p.require(config == expected, "Unreviewed physical H100 configuration")
    p.require((arm, phase) in (("reference", "reference"), ("resume", "interrupted"), ("resume", "resumed")),
              "Unknown H100 phase")
    p.require(os.environ["NATIVE_RUN_IDENTITY"] == p.identity(config, modules), "Wrong native H100 identity")
    source = p.ROOT / modules["common"].NATIVE
    sys.path.insert(0, str(source))
    patch = importlib.import_module("llava.train.llama_xformers_attn_monkey_patch")
    patch.replace_llama_attn_with_xformers_attn()
    module = importlib.import_module("llava.train.train")
    p.require(Path(module.__file__).resolve() == source / "llava/train/train.py", "Foreign native trainer")
    parent = module.LLaVATrainer
    audit = Audit(config, phase, Path(attempt) / "observations", study["interrupt_step"])
    module.LLaVATrainer = audit.install(parent)
    old = sys.argv[:]
    started = time.monotonic()
    error, status, observation = None, "FAILED_NATIVE_ATTEMPT", None
    try:
        import torch
        p.require(torch.cuda.device_count() == 1 and "H100" in torch.cuda.get_device_name(),
                  "Actual H100 required; L40S result cannot qualify this driver")
        p.require(os.environ.get("CUDA_HOME") == "/sw/eb/sw/CUDA/12.1.1"
                  and torch.version.cuda == "12.1", "Completed ACES CUDA12.1.1 route changed")
        sys.argv = ["h100-observed-native", "--config", physical_ref["path"], "--stop-at-epoch", str(stop_at)]
        modules["native"].main()
        complete = Path(config["output"]) / "TRAIN_COMPLETE.json"
        if complete.exists():
            p.require(p.read(complete)["identity"] == p.identity(config, modules), "Foreign terminal native state")
            status = "NATIVE_TRAJECTORY_COMPLETE"
        elif (Path(attempt) / "CONTINUATION.json").exists():
            status = "ACCEPTED_NATIVE_CONTINUATION"
        else:
            raise ValueError("H100 native returned without an accepted result")
    except BaseException as exc:
        error = repr(exc)
        traceback.print_exc()
    finally:
        module.LLaVATrainer, sys.argv = parent, old
        try:
            observation = audit.finish(status)
        except BaseException as exc:
            error = (error or "") + "; observation finalization: " + repr(exc)
            traceback.print_exc()
    output = Path(attempt) / "observations"
    files = {str(path.relative_to(output)): p.desc(path) for path in sorted(output.rglob("*")) if path.is_file()}
    result = {
        "status": status, "arm": arm, "phase": phase, "study_id": study["id"],
        "study": study_ref, "driver_sources": study["driver_sources"], "native_target": study["native_target"],
        "native_configuration": study["native_configurations"][arm], "physical_configuration": physical_ref,
        "runtime_alias": alias_ref, "actual_job_id": os.environ["SLURM_JOB_ID"],
        "actual_restart_count": int(os.environ["SLURM_RESTART_COUNT"]),
        "actual_node": os.environ["H100_ENGINEERING_NODE"], "actual_start": os.environ["H100_ENGINEERING_START"],
        "attempt": str(Path(attempt).absolute()), "observation": observation, "files": files, "error": error,
        "elapsed_seconds": time.monotonic() - started, "hardware": "H100",
        "actual_result_requires_independent_review": True,
    }
    p.seal(Path(attempt) / "CHILD_RESULT.json", result)
    p.require(error is None, "Actual H100 native failed; preserved raw trace and observations require diagnosis")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--arm", choices=("reference", "resume"), required=True)
    parser.add_argument("--phase", choices=("reference", "interrupted", "resumed"), required=True)
    parser.add_argument("--physical-config", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--stop-at-epoch", type=int, required=True)
    args = parser.parse_args()
    run(p.read(args.study), p.desc(args.study), args.arm, args.phase, p.desc(args.physical_config),
        p.desc(args.alias), args.attempt, args.stop_at_epoch)


if __name__ == "__main__":
    main()
