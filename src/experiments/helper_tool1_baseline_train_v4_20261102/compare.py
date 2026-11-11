"""Restart-parity comparison (v4): reference probe versus stop + Slurm requeue + accepted-state resume probe.

Every checkpoint file is classified before comparison:
  identity  - training_args.bin (pickled LLaVA TrainingArguments with cell paths/time): sha256 recorded,
              never unpickled, never judged;
  static    - README.md, tokenizer files, zero_to_fp32.py, latest, FROZEN_RECONSTRUCTION.json: must match
              by bytes (JSON by value); any difference is structural, never "nondeterminism";
  control   - rng_state.pth, scheduler.pt, trainer_state.json: must be exactly equal by value;
  config    - adapter_config.json: equal after sorting PEFT target_modules (set-order varies by process);
  numeric   - *.safetensors, *.bin tensors and DeepSpeed *_states.pt: tensors give the numeric verdict,
              non-tensor leaves (step counters, loss scaler, flags) are control state;
  skipped   - ACCEPTED.json, OBSERVATION.json (attempt receipts; losses are compared separately).
Any other file fails closed.  The checkpoint at the stop step is produced by both runs before any
interruption (baseline determinism); the later checkpoint, final adapter/projector and post-stop losses
measure resume parity.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


IDENTITY = {"training_args.bin"}
SKIPPED = {"ACCEPTED.json", "OBSERVATION.json"}
STATIC = {"README.md", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "added_tokens.json",
          "tokenizer.json", "zero_to_fp32.py", "latest", "FROZEN_RECONSTRUCTION.json"}
CONTROL = {"rng_state.pth", "scheduler.pt", "trainer_state.json"}
CONFIG = {"adapter_config.json"}
VOLATILE_FINAL_LOG_KEYS = {"train_runtime", "train_samples_per_second", "train_steps_per_second",
                           "train_loss", "total_flos", "epoch", "step"}


def require(value, message):
    if not value:
        raise ValueError(message)


def kind(relative):
    name = Path(relative).name
    if name in SKIPPED:
        return "skipped"
    if name in IDENTITY:
        return "identity"
    if name in STATIC:
        return "static"
    if name in CONTROL:
        return "control"
    if name in CONFIG:
        return "config"
    if name.endswith(".safetensors") or name.endswith("_states.pt") or (name.endswith(".bin")
                                                                       and name not in IDENTITY):
        return "numeric"
    return "unknown"


def new_stats():
    return {"tensors": 0, "unequal_tensors": 0, "max_abs": 0.0, "max_rel": 0.0, "control_mismatch": [],
            "structural": [], "nonfinite_difference": [], "first_difference": None}


def note(stats, where):
    if stats["first_difference"] is None:
        stats["first_difference"] = where


def compare(a, b, where, stats):
    import numpy
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.shape != b.shape or a.dtype != b.dtype:
            stats["structural"].append(where)
            note(stats, where)
            return
        stats["tensors"] += 1
        if torch.equal(a, b):
            return
        stats["unequal_tensors"] += 1
        note(stats, where)
        if a.is_floating_point():
            difference = (a.double() - b.double()).abs()
            finite = torch.isfinite(difference)
            if not bool(finite.all()):
                stats["nonfinite_difference"].append(where)
            if bool(finite.any()):
                largest = float(difference[finite].max())
                scale = float(a.double().abs().max())
                stats["max_abs"] = max(stats["max_abs"], largest)
                if math.isfinite(scale) and scale > 0:
                    stats["max_rel"] = max(stats["max_rel"], largest / scale)
        else:
            stats["control_mismatch"].append(where)
        return
    if isinstance(a, numpy.ndarray) or isinstance(b, numpy.ndarray):
        if not (isinstance(a, numpy.ndarray) and isinstance(b, numpy.ndarray)) or a.shape != b.shape or a.dtype != b.dtype:
            stats["structural"].append(where)
            note(stats, where)
        elif not numpy.array_equal(a, b):
            stats["control_mismatch"].append(where)
            note(stats, where)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            stats["structural"].append(where + " keys")
            note(stats, where)
        for key in sorted(set(a) & set(b), key=str):
            compare(a[key], b[key], where + "/" + str(key), stats)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            stats["structural"].append(where)
            note(stats, where)
            return
        for index, (left, right) in enumerate(zip(a, b)):
            compare(left, right, "%s[%d]" % (where, index), stats)
        return
    if isinstance(a, float) and isinstance(b, float):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return
        stats["control_mismatch"].append(where)
        if not (math.isfinite(a) and math.isfinite(b)):
            stats["nonfinite_difference"].append(where)
        note(stats, where)
        return
    if type(a) is type(b) and hasattr(a, "__dict__") and not isinstance(a, (str, bytes, int, float, bool)):
        compare(vars(a), vars(b), where + "{" + type(a).__name__ + "}", stats)
        return
    if a != b:
        stats["control_mismatch"].append(where)
        note(stats, where)


def sha(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load(path, category):
    if path.name.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    if path.suffix in (".pt", ".pth", ".bin"):
        import torch
        # Own job-produced DeepSpeed/HF tensors and RNG/scheduler/loss-scaler states only (never training_args).
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.suffix == ".json":
        value = json.loads(path.read_bytes())
        if category == "config" and isinstance(value, dict) and isinstance(value.get("target_modules"), list):
            value = dict(value, target_modules=sorted(value["target_modules"]))
        return value
    return path.read_bytes()


def merge(total, local):
    for key in ("tensors", "unequal_tensors"):
        total[key] += local[key]
    for key in ("max_abs", "max_rel"):
        total[key] = max(total[key], local[key])
    for key in ("control_mismatch", "structural", "nonfinite_difference"):
        total[key].extend(local[key])
    if local["first_difference"]:
        note(total, local["first_difference"])


def compare_files(left, right, names):
    total, per_file, identity = new_stats(), {}, {}
    for name in sorted(names):
        category = kind(name)
        a, b = left / name, right / name
        if category == "skipped":
            continue
        if category == "unknown":
            total["structural"].append("unclassified file " + name)
            note(total, name)
            continue
        if not (a.is_file() and b.is_file()):
            total["structural"].append("missing file " + name)
            note(total, name)
            continue
        if category == "identity":
            identity[name] = {"reference_sha256": sha(a), "probe_sha256": sha(b), "differs": sha(a) != sha(b)}
            continue
        if a.stat().st_size == b.stat().st_size and sha(a) == sha(b):
            per_file[name] = "BYTE_IDENTICAL"
            continue
        if category == "static":
            try:
                same = name.endswith(".json") and json.loads(a.read_bytes()) == json.loads(b.read_bytes())
            except ValueError:
                same = False
            per_file[name] = "VALUE_IDENTICAL" if same else "STATIC_DIFFERS"
            if not same:
                total["structural"].append("static file differs: " + name)
                note(total, name)
            continue
        local = new_stats()
        compare(load(a, category), load(b, category), name, local)
        if category in ("control", "config"):
            if local["unequal_tensors"]:
                local["control_mismatch"].append(name + " tensors")
            local["unequal_tensors"], local["max_abs"], local["max_rel"] = 0, 0.0, 0.0
            if category == "config" and (local["control_mismatch"] or local["structural"]):
                local["structural"].append(name + " configuration differs beyond target_modules order")
        clean = not (local["structural"] or local["unequal_tensors"] or local["control_mismatch"])
        per_file[name] = "VALUE_IDENTICAL" if clean else local
        merge(total, local)
    total["files"] = per_file
    total["identity_files"] = identity
    total["numeric_exact"] = total["unequal_tensors"] == 0
    total["control_exact"] = not total["control_mismatch"]
    total["structurally_valid"] = not total["structural"]
    return total


def compare_tree(left, right):
    require(left.is_dir() and right.is_dir(), "Both comparison directories must exist")
    names_left = {str(p.relative_to(left)) for p in left.rglob("*") if p.is_file()}
    names_right = {str(p.relative_to(right)) for p in right.rglob("*") if p.is_file()}
    result = compare_files(left, right, names_left & names_right)
    extra = sorted(name for name in names_left ^ names_right if kind(name) != "skipped")
    if extra:
        result["structural"].append("file sets differ: %s" % extra)
        result["structurally_valid"] = False
    return result


def observation(path):
    value = json.loads((path / "RESULT.json").read_bytes())
    return value, value["observation"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--stop-step", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    reference, probe = Path(args.reference), Path(args.probe)
    stop, steps, save = args.stop_step, args.steps, args.save_steps
    require(0 < stop < steps and stop % save == 0, "Stop must be an accepted boundary before the horizon")
    later = max(s for s in range(save, steps + 1, save))
    require(later > stop, "A post-resume accepted checkpoint is required")
    ref_result, ref_obs = observation(reference)
    probe_result, probe_obs = observation(probe)
    require(ref_result["steps"] == probe_result["steps"] == steps
            and ref_result["restart_count"] == 0 and probe_result["restart_count"] >= 1,
            "Reference must be unrestarted and the probe must be a restarted attempt")
    require([a["resumed_from_step"] for a in probe_obs["attempts"]] == [None, stop],
            "Probe must be one fresh attempt followed by one resume from the stop boundary")
    micro = 32
    losses_ref, losses_probe = ref_obs["losses"], probe_obs["losses"]
    require(len(losses_ref) == len(losses_probe) == micro * steps, "Complete per-microbatch losses required")
    pre = [abs(a - b) for a, b in zip(losses_ref[:micro * stop], losses_probe[:micro * stop])]
    post = [abs(a - b) for a, b in zip(losses_ref[micro * stop:], losses_probe[micro * stop:])]
    overflow_equal = ([e["overflow"] for e in ref_obs["step_events"]]
                      == [e["overflow"] for e in probe_obs["step_events"]])
    stop_tree = compare_tree(reference / "training" / ("checkpoint-%d" % stop),
                             probe / "training" / ("checkpoint-%d" % stop))
    later_tree = compare_tree(reference / "training" / ("checkpoint-%d" % later),
                              probe / "training" / ("checkpoint-%d" % later))
    final = compare_files(reference / "training", probe / "training",
                          {"adapter_model.safetensors", "non_lora_trainables.bin", "adapter_config.json",
                           "README.md"})
    config_equal = (json.loads((reference / "training" / "config.json").read_bytes())
                    == json.loads((probe / "training" / "config.json").read_bytes()))
    stable = lambda s: {"global_step": s["global_step"], "max_steps": s["max_steps"],
                        "log_history": [{k: v for k, v in row.items() if k not in VOLATILE_FINAL_LOG_KEYS}
                                        for row in s["log_history"]]}
    final_state_equal = (stable(json.loads((reference / "training" / "trainer_state.json").read_bytes()))
                         == stable(json.loads((probe / "training" / "trainer_state.json").read_bytes())))
    structural = not all(t["structurally_valid"] for t in (stop_tree, later_tree, final)) or not config_equal
    baseline_exact = (stop_tree["numeric_exact"] and stop_tree["control_exact"] and max(pre or [0.0]) == 0.0)
    resumed_numeric = (later_tree["numeric_exact"] and final["numeric_exact"] and max(post or [0.0]) == 0.0)
    resumed_control = (later_tree["control_exact"] and final["control_exact"] and final_state_equal
                       and overflow_equal)
    if structural:
        verdict = "FAIL_STRUCTURAL_STATE_DIFFERENCE"
    elif not stop_tree["control_exact"]:
        verdict = "INCONCLUSIVE_NONDETERMINISTIC_BASELINE"
    elif not resumed_control:
        verdict = "FAIL_RESUMED_RNG_SCHEDULER_OR_STEP_STATE_DIFFERS"
    elif baseline_exact and resumed_numeric:
        verdict = "PASS_BITWISE_RESTART_PARITY"
    elif baseline_exact:
        verdict = "FAIL_RESUME_PARITY_WITH_DETERMINISTIC_BASELINE"
    else:
        verdict = "INCONCLUSIVE_NONDETERMINISTIC_BASELINE"
    report = {
        "schema": "tool1-restart-parity-v4", "verdict": verdict, "reference": str(reference), "probe": str(probe),
        "stop_step": stop, "post_resume_checkpoint": later, "steps": steps,
        "reference_job": ref_result["job_id"], "probe_job": probe_result["job_id"],
        "probe_attempts": probe_obs["attempts"],
        "losses": {"pre_interruption_max_abs": max(pre or [0.0]), "post_resume_max_abs": max(post or [0.0]),
                   "pre_exact": max(pre or [0.0]) == 0.0, "post_exact": max(post or [0.0]) == 0.0,
                   "count": len(losses_ref)},
        "overflow_flags_equal": overflow_equal, "final_config_json_equal": config_equal,
        "checkpoint_at_stop": stop_tree, "checkpoint_after_resume": later_tree, "final_outputs": final,
        "final_trainer_state_stable_fields_equal": final_state_equal,
        "scope": "Engineering restart-parity probe only; not a scientific comparison.",
    }
    Path(args.out).write_bytes(json.dumps(report, sort_keys=True, allow_nan=False, default=str).encode() + b"\n")
    print(json.dumps({"verdict": verdict, "out": args.out}), flush=True)


if __name__ == "__main__":
    main()
