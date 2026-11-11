"""H100-specific raw comparison; no promotion from a Babel or synthetic receipt."""
from pathlib import Path

from . import protocol as p
from experiments.root_native_qualification_v10_20261024 import state
from experiments.root_native_qualification_v10_20261024.readout import check_behavior, verified_phase, compare_learning_rate_trajectories


def _phase(descriptor, study, config):
    value = verified_phase(descriptor)
    p.require(value["hardware"] == "H100" and value["study_id"] == study["id"]
              and value["driver_sources"] == study["driver_sources"]
              and value["native_target"] == study["native_target"]
              and value["native_configuration"] == config,
              "Foreign H100 phase/study/operator evidence")
    physical, logical = p.checked(value["physical_configuration"]), p.checked(config)
    alias = p.checked(value["runtime_alias"])
    expected = {**logical, "providers": {**logical["providers"], "runtime_images": value["runtime_alias"]}}
    p.require(physical == expected, "Physical config changed more than its registered image alias")
    selected = p.checked(logical["providers"]["selection"])
    from experiments.root_native_media_transport_20261023 import transport
    p.require(alias["status"] == "PASS_BYTE_IDENTICAL_RUNTIME_IMAGES" and alias["scope"] == "ENGINEERING_ONLY"
              and alias["transport"] == logical["providers"]["media_transport"]
              and alias["logical_pool"] == logical["providers"]["pool"]
              and alias["records_json"] == selected["records_json"]
              and alias["job_id"] == value["actual_job_id"] and alias["node"] == value["actual_node"]
              and int(alias["actual"]["restart"]) == value["actual_restart_count"]
              and alias["materializer_source_sha256"] == p.sha(transport.__file__),
              "Stale/foreign historical H100 alias evidence")
    pool = p.checked(logical["providers"]["pool"])
    canonical = p.checked(pool["image_manifest"])["assets"]
    names = {row["image"] for row in p.checked(selected["records_json"]) if row.get("image")}
    p.require(set(alias["assets"]) == names and alias["cost"]["images"] == len(names),
              "Incomplete historical selected-image alias evidence")
    for name in names:
        recorded, expected_image = alias["assets"][name], canonical[name]
        p.require({key: val for key, val in recorded.items() if key != "path"} ==
                  {key: val for key, val in expected_image.items() if key != "path"}
                  and recorded["path"] == str(Path(alias["image_root"]) / name),
                  "Historical encoded/RGB/shape identity changed")
    # Previous node-local files need not survive a real requeue. This checks
    # immutable acquisition provenance, not a fictitious reread of that node.
    return value, alias


def _accepted(config, modules, step):
    path = Path(config["output"]) / "training" / ("checkpoint-" + str(step))
    receipt = modules["native"].checkpoint_receipt(path, p.identity(config, modules))
    p.require(receipt["world_size"] == 1 and receipt["global_step"] == step, "Wrong H100 accepted state")
    portable = p.read(path / "config.json")
    p.require(portable["model_type"] == "llava_llama" and portable["hidden_size"] == 4096
              and (path / "adapter_config.json").exists(), "Missing native7B portable configuration")
    return path, receipt


def compare(study, descriptors):
    import json
    import numpy as np
    import torch

    configs, modules = p.configs(study)
    phases, aliases = {}, {}
    for phase in ("reference", "interrupted", "resumed"):
        arm = "reference" if phase == "reference" else "resume"
        phases[phase], aliases[phase] = _phase(descriptors[phase], study, study["native_configurations"][arm])
        p.require(phases[phase]["phase"] == phase, "H100 phase substitution")
    p.require(len({v["actual_job_id"] for v in phases.values()}) == 1
              and phases["resumed"]["actual_restart_count"] > phases["interrupted"]["actual_restart_count"]
              and phases["resumed"]["actual_start"] != phases["interrupted"]["actual_start"],
              "No actual H100 same-job physical restart")
    p.require(phases["reference"]["status"] == phases["resumed"]["status"] == "NATIVE_TRAJECTORY_COMPLETE"
              and phases["interrupted"]["status"] == "ACCEPTED_NATIVE_CONTINUATION", "Incomplete H100 schedule")
    observations = {name: p.checked(value["observation"]) for name, value in phases.items()}
    criteria = {name: check_behavior(value) for name, value in observations.items()}
    for checks in criteria.values():
        checks["all_pass"] = all(value for key, value in checks.items() if key != "all_pass")
    paths = {name: Path(value["attempt"]) / "observations" for name, value in phases.items()}
    lr_trajectories = compare_learning_rate_trajectories(paths,8,4)
    initial = {name: p.read(path / "FIRST_MICRO_STATE.json") for name, path in paths.items()}
    cut = p.read(paths["interrupted"] / "STEP-4.json")
    ends = {name: p.read(paths[name] / "STEP-8.json") for name in ("reference", "resumed")}
    keys = ("weights", "optimizer", "scheduler", "rng", "global_step", "actual_DS_global_steps", "DS_skipped_steps")
    paired_initial = {name: initial["reference"][name] == initial["interrupted"][name] for name in keys}
    restored = {name: cut[name] == initial["resumed"][name] for name in keys}
    p.require(initial["reference"]["global_step"] == initial["interrupted"]["global_step"] == 0
              and initial["resumed"]["global_step"] == 4, "Unexpected actual H100 prefix cursor")
    reference_ids = observations["reference"]["ordered_record_ids"]
    resumed_ids = observations["interrupted"]["ordered_record_ids"] + observations["resumed"]["ordered_record_ids"]
    pool = p.checked(configs["reference"]["providers"]["pool"])
    rowmap = dict(zip(pool["record_ids"], p.checked(pool["records_json"]), strict=True))
    p.require(set(reference_ids) <= set(rowmap), "Unknown actual H100 record IDs")
    rows = [rowmap[key] for key in reference_ids]
    coverage = {
        "text_only": sum(not row.get("image") for row in rows), "image": sum(bool(row.get("image")) for row in rows),
        "multi_turn": sum(len(row["conversations"]) >= 4 for row in rows),
        "long": sum(sum(len(turn["value"]) for turn in row["conversations"]) >= 2000 for row in rows),
    }
    order = {
        "exact": reference_ids == resumed_ids,
        "complete": len(reference_ids) == len(resumed_ids) == len(set(reference_ids)) == 1024,
        "reference_sha256": p.key(reference_ids), "resumed_sha256": p.key(resumed_ids),
        "type_coverage": coverage,
        "required_type_coverage": all(coverage[name] >= minimum for name, minimum in
                                      (("text_only", 32), ("image", 512), ("multi_turn", 128), ("long", 32))),
    }
    reference_path, reference_receipt = _accepted(configs["reference"], modules, 8)
    resume_path, resume_receipt = _accepted(configs["resume"], modules, 8)
    cut_path, cut_receipt = _accepted(configs["resume"], modules, 4)
    tolerance = study["numeric_tolerance"]
    def scheduler(path):
        if (path / "scheduler.pt").exists():
            return torch.load(path / "scheduler.pt", map_location="cpu")
        candidates = list(path.rglob("*model_states.pt"))
        p.require(len(candidates) == 1, "Missing world1 H100 DS scheduler shard")
        return torch.load(candidates[0], map_location="cpu")["lr_scheduler"]
    def losses(name):
        return [json.loads(line)["returned_native_scaled_loss"]
                for line in (paths[name] / "MICROBATCHES.jsonl").read_text().splitlines()]
    comparisons = {
        "portable_weights": state.compare_tree(state.portable_weights(reference_path),
                                               state.portable_weights(resume_path),
                                               tolerance["absolute"], tolerance["relative"]),
        "Adam_FP32_master_scaler": state.compare_tree(state.native_optimizer(reference_path),
                                                     state.native_optimizer(resume_path),
                                                     tolerance["absolute"], tolerance["relative"]),
        "scheduler": state.compare_tree(scheduler(reference_path), scheduler(resume_path), 0, 0),
        "saved_RNG": state.compare_tree(
            state.normalize_saved_rng(torch.load(reference_path / "rng_state.pth", map_location="cpu")),
            state.normalize_saved_rng(torch.load(resume_path / "rng_state.pth", map_location="cpu")), 0, 0),
        "microbatch_loss": state.compare_tree(np.asarray(losses("reference"), dtype=np.float64),
                                              np.asarray(losses("interrupted") + losses("resumed"), dtype=np.float64),
                                              tolerance["absolute"], tolerance["relative"]),
    }
    groups = {name: p.read(path / "PARAMETER_GROUPS.json") for name, path in paths.items()}
    groups_equal = groups["reference"] == groups["interrupted"] == groups["resumed"]
    steps = {
        "final_steps": ends["reference"]["global_step"] == ends["resumed"]["global_step"] == 8,
        "finite_weights": ends["reference"]["all_weights_finite"] and ends["resumed"]["all_weights_finite"],
        "skipped_steps_equal": ends["reference"]["DS_skipped_steps"] == ends["resumed"]["DS_skipped_steps"],
        "actual_parameter_movement": initial["reference"]["weights"] != ends["reference"]["weights"]
                                    and initial["interrupted"]["weights"] != ends["resumed"]["weights"],
    }
    passed = (all(checks["all_pass"] for checks in criteria.values()) and all(paired_initial.values())
              and all(restored.values()) and all(value["pass_numeric"] for value in comparisons.values())
              and order["exact"] and order["complete"] and order["required_type_coverage"]
              and groups_equal and all(steps.values()))
    return {
        "schema": "h100-transport-engineering-actual-readout-v1",
        "status": "PASS_H100_TRANSPORT_ENGINEERING_READOUT_UNREVIEWED" if passed else "FAIL_H100_TRANSPORT_ENGINEERING",
        "study_id": study["id"], "study_identity": p.key({k: v for k, v in study.items() if k != "review"}),
        "driver_sources": study["driver_sources"], "native_target": study["native_target"],
        "phase_results": descriptors, "criteria": criteria, "order": order, "initial_pair_exact": paired_initial,
        "restored_boundary_exact": restored, "numerical_comparisons": comparisons, "steps": steps,
        "optimizer_groups_equal": groups_equal, "semantic_learning_rate_trajectories":lr_trajectories,
        "accepted_states": {name: {"receipt": p.desc(path / "ACCEPTED.json"), "logical_bytes": receipt["logical_bytes"]}
                            for name, path, receipt in (("reference8", reference_path, reference_receipt),
                                                       ("resume4", cut_path, cut_receipt),
                                                       ("resume8", resume_path, resume_receipt))},
        "actual_transport_and_cost": {name: {"alias": phases[name]["runtime_alias"], "cost": value["cost"]}
                                      for name, value in aliases.items()},
        "actual_observations": observations, "hardware": "H100", "Babel_PASS_transferred": False,
        "qualification_promoted": False, "scientific_pool_qualified": False,
        "all_allocated_scheduler_attempts_and_setup_costs_must_be_added": True,
        "phase_transport_costs_are_nested_not_extra_parent_allocations": True,
        "independent_actual_readout_review_required": True,
    }
