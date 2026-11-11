"""Read-only checks of DS0.12.6 / Torch AdamW / HF4.37.2 native saved state."""
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import numbers
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if torch.is_tensor(value):
        require(value.numel() == 1, name + " must be scalar")
        value = value.item()
    require(not isinstance(value, (bool, np.bool_)) and isinstance(value, numbers.Real)
            and math.isfinite(float(value)) and int(value) == value and value >= minimum,
            name + " must be a finite integer >= " + str(minimum))
    return int(value)


def finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    require(not isinstance(value, (bool, np.bool_)) and isinstance(value, numbers.Real)
            and math.isfinite(float(value)), name + " must be finite")
    result = float(value)
    require(result > 0 if positive else result >= 0, name + " has invalid sign")
    return result


def tensor(value: Any, name: str, *, nonempty: bool = True) -> torch.Tensor:
    require(torch.is_tensor(value), name + " must be a tensor")
    require(not nonempty or value.numel() > 0, name + " must not be empty")
    require(not value.is_sparse and not value.is_complex(), name + " has unsupported tensor layout/type")
    if value.is_floating_point():
        flat = value.detach().reshape(-1)
        for offset in range(0, flat.numel(), 1 << 20):
            require(bool(torch.isfinite(flat[offset:offset + (1 << 20)]).all().item()),
                    "Nonfinite " + name)
    return value


def descriptor(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path.absolute()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked_json(ref: Mapping[str, Any]) -> Any:
    path = Path(ref["path"])
    require(descriptor(path) == dict(ref), "Changed metadata descriptor: " + str(path))
    return json.loads(path.read_text())


def safe_member(root: Path, relative: str) -> Path:
    name = Path(relative)
    require(not name.is_absolute() and name.parts and ".." not in name.parts,
            "Unsafe checkpoint member: " + relative)
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "Checkpoint member escapes root")
    for item in (path, *path.parents):
        if item == root.parent:
            break
        require(not item.is_symlink(), "Symlink in checkpoint member: " + str(item))
    return path


def _accepted_metadata(path: Path, identity: str) -> dict[str, Any]:
    require(not path.is_symlink(), "Checkpoint directory is a symlink")
    accepted_path = path / "ACCEPTED.json"
    require(not accepted_path.is_symlink(), "Accepted receipt is a symlink")
    accepted = json.loads(accepted_path.read_text())
    require(accepted["status"] == "ACCEPTED_FULL_NATIVE_STATE" and accepted["identity"] == identity,
            "Foreign accepted checkpoint identity/status")
    step = integer(accepted["global_step"], "accepted.global_step")
    require(path.name == "checkpoint-" + str(step), "Accepted cursor/directory differs")
    require(accepted["world_size"] == 1, "This guard qualifies world-size one only")
    require(isinstance(accepted["files"], dict) and bool(accepted["files"]), "Empty accepted inventory")
    require(sum(integer(ref["bytes"], "accepted member bytes") for ref in accepted["files"].values())
            == accepted["logical_bytes"], "Accepted logical bytes differ")
    return accepted


def _verify_inventory(path: Path, files: Mapping[str, Any], *, accepted: bool) -> None:
    require(path.is_dir() and not path.is_symlink() and bool(files), "Missing/unsafe checkpoint directory/inventory")
    require("ACCEPTED.json" not in files, "Inventory cannot include its own accepted receipt")
    actual_names = set()
    for member in path.rglob("*"):
        require(not member.is_symlink(), "Symlink in saved checkpoint")
        if member.is_file():
            actual_names.add(str(member.relative_to(path)))
    expected_names = set(files) | ({"ACCEPTED.json"} if accepted else set())
    require(actual_names == expected_names, "Actual checkpoint file inventory differs")
    for name, ref in files.items():
        actual = descriptor(safe_member(path, name))
        require(actual["bytes"] == ref["bytes"] and actual["sha256"] == ref["sha256"],
                "Changed checkpoint member: " + name)


def _rng(state: Mapping[str, Any], cuda_state_sizes: Sequence[int]) -> dict[str, Any]:
    require({"python", "numpy", "cpu", "cuda"} <= state.keys(), "Missing Python/NumPy/CPU/CUDA RNG state")
    random.Random(0).setstate(state["python"])
    np.random.RandomState(0).set_state(state["numpy"])
    cpu = tensor(state["cpu"], "CPU RNG")
    require(cpu.dtype == torch.uint8 and cpu.ndim == 1, "Malformed CPU RNG bytes")
    torch.Generator(device="cpu").set_state(cpu.cpu())
    require(len(cuda_state_sizes) == 1, "World-one CUDA RNG schema must come from its actual runtime")
    cuda = state["cuda"]
    if torch.is_tensor(cuda):
        cuda = [cuda]
    require(isinstance(cuda, (tuple, list)) and len(cuda) == len(cuda_state_sizes),
            "Missing/extra CUDA RNG devices")
    for index, (saved, size) in enumerate(zip(cuda, cuda_state_sizes, strict=True)):
        value = tensor(saved, "CUDA RNG " + str(index))
        require(value.dtype == torch.uint8 and value.ndim == 1
                and value.numel() == integer(size, "observed CUDA RNG size", 1),
                "Malformed CUDA RNG bytes/actual-runtime length")
        if torch.cuda.is_available():
            torch.Generator(device="cuda:" + str(index)).set_state(value.cpu())
    return {"python_numpy_cpu_restore_checked": True, "cuda_byte_sizes": list(cuda_state_sizes),
            "private_CUDA_generator_restore_checked": torch.cuda.is_available(),
            "global_RNG_was_not_modified": True}


def _scaler(state: Mapping[str, Any], steps: int) -> dict[str, Any]:
    require(type(state["overflow"]) is bool and type(state["dynamic_loss_scale"]) is bool,
            "Malformed DS overflow/dynamic flags")
    scaler = state["loss_scaler"]
    require(not isinstance(scaler, Mapping) and hasattr(scaler, "cur_scale")
            and type(getattr(scaler, "dynamic", None)) is bool, "Missing actual DS loss-scaler state")
    scale = finite_number(scaler.cur_scale, "loss scale", positive=True)
    require(scaler.dynamic == state["dynamic_loss_scale"], "DS/scaler dynamic flag differs")
    if scaler.dynamic:
        require(integer(scaler.cur_iter, "scaler.cur_iter") == steps, "Scaler/global step differs")
        last = integer(scaler.last_overflow_iter, "scaler.last_overflow_iter", -1)
        require(last < steps, "Scaler last overflow is in the future")
        finite_number(scaler.min_scale, "minimum loss scale", positive=True)
        require(scale >= scaler.min_scale, "Loss scale is below minimum")
        require(finite_number(scaler.scale_factor, "loss-scale factor", positive=True) > 1,
                "Loss-scale factor must exceed one")
        integer(scaler.scale_window, "scale window", 1)
        shift = integer(scaler.delayed_shift, "delayed shift", 1)
        require(1 <= integer(scaler.cur_hysteresis, "current hysteresis", 1) <= shift,
                "Invalid current hysteresis")
        require(type(scaler.consecutive_hysteresis) is bool
                and type(scaler.raise_error_at_min_scale) is bool and scaler.dtype == torch.float16,
                "Malformed dynamic loss-scaler policy")
        if state["overflow"]:
            require(steps > 0 and last == steps - 1, "Overflow/scaler history differs")
        elif steps:
            require(last != steps - 1, "Non-overflow flag contradicts scaler history")
    return {"dynamic": scaler.dynamic, "current_scale": scale, "overflow": state["overflow"]}


def _portable_name(name: str) -> str:
    return name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")


def validate_payloads(
    *, model_state: Mapping[str, Any], zero_state: Mapping[str, Any],
    rng_state: Mapping[str, Any], lora_state: Mapping[str, Any], projector_state: Mapping[str, Any],
    trainer_state: Mapping[str, Any], reconstruction: Mapping[str, Any],
    cuda_state_sizes: Sequence[int], scheduler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate state, not learning benefit. Zero LR/moments and overflow skips are valid."""
    require(integer(zero_state["zero_stage"], "ZeRO stage") == 3
            and integer(zero_state["partition_count"], "partition count") == 1,
            "Only actual DS world-one ZeRO3 state is supported")
    steps = integer(model_state["global_steps"], "DS global_steps")
    skipped = integer(model_state["skipped_steps"], "DS skipped_steps")
    require(skipped <= steps, "Skipped steps exceed attempted steps")
    applied = steps - skipped
    require(integer(trainer_state["global_step"], "Trainer global_step") == steps,
            "Trainer/DS global step differs")
    require(integer(model_state["global_samples"], "DS global_samples") == steps * 128,
            "DS global-sample counter differs from target global128")
    require(model_state["dp_world_size"] == model_state["mp_world_size"] == 1,
            "Wrong saved distributed dimensions")
    ds_config = model_state["ds_config"]
    require(isinstance(ds_config, Mapping) and ds_config["train_batch_size"] == 128
            and ds_config["train_micro_batch_size_per_gpu"] == 4
            and ds_config["gradient_accumulation_steps"] == 32
            and ds_config["fp16"]["enabled"] is True, "Saved native FP16/batch configuration differs")
    loss_scale = finite_number(ds_config["fp16"]["loss_scale"], "configured loss scale")
    require(zero_state["dynamic_loss_scale"] == (loss_scale == 0),
            "Configured/saved loss-scaling mode differs")
    require(model_state.get("frozen_param_shapes") is None
            and model_state.get("frozen_param_fragments") is None, "Frozen base was duplicated in checkpoint")
    if trainer_state.get("epoch") is not None:
        finite_number(trainer_state["epoch"], "Trainer epoch")
    trainable = reconstruction["trainable"]
    require(isinstance(trainable, list) and bool(trainable), "Missing trainable reconstruction inventory")
    expected = {row["name"]: row for row in trainable}
    require(len(expected) == len(trainable), "Duplicate reconstruction parameter names")
    lora_names = {name for name in expected if ".lora_A." in name or ".lora_B." in name}
    projector_names = {name for name in expected if "mm_projector" in name}
    require(bool(lora_names) and bool(projector_names) and lora_names.isdisjoint(projector_names)
            and lora_names | projector_names == set(expected), "Unexpected native trainable support")
    portable = {_portable_name(name): name for name in lora_names}
    require(len(portable) == len(lora_names) and set(lora_state) == set(portable)
            and set(projector_state) == projector_names, "Portable LoRA/projector support differs")
    parameters = {portable[name]: tensor(value, "LoRA " + name) for name, value in lora_state.items()}
    parameters.update({name: tensor(value, "projector " + name) for name, value in projector_state.items()})
    for name, value in parameters.items():
        require(list(value.shape) == list(expected[name]["shape"])
                and str(value.dtype) == expected[name]["dtype"], "Portable parameter shape/dtype differs: " + name)
    module = model_state["module"]
    require(isinstance(module, Mapping), "Missing DS module state")
    buffers = model_state["buffer_names"]
    require(isinstance(buffers, list) and all(isinstance(n, str) for n in buffers)
            and len(set(buffers)) == len(buffers), "Malformed DS persistent-buffer inventory")
    require(set(module) == set(expected) | set(buffers), "DS module trainable/buffer keys differ")
    for name, value in module.items():
        tensor(value, "DS module " + name, nonempty=False)
    shapes = model_state["param_shapes"]
    masters = zero_state["fp32_flat_groups"]
    require(isinstance(shapes, list) and isinstance(masters, list) and shapes
            and len(shapes) == len(masters), "Missing/misaligned FP32 master groups")
    named = [name for group in shapes for name in group]
    require(len(named) == len(set(named)) and set(named) == set(expected), "DS param_shapes support differs")
    optimizer = zero_state["optimizer_state_dict"]
    groups = optimizer["param_groups"]
    states = optimizer["state"]
    require(isinstance(groups, list) and bool(groups) and isinstance(states, Mapping),
            "Missing actual Adam groups/state")
    ids = [pid for group in groups for pid in group["params"]]
    require(len(ids) == len(set(ids)) == len(masters) and set(ids) == set(states),
            "Adam state/master group identity differs")
    positive_moments = 0
    for group_index, (shape_group, master, pid) in enumerate(zip(shapes, masters, ids, strict=True)):
        master = tensor(master, "FP32 master group " + str(group_index))
        require(master.dtype == torch.float32 and master.ndim == 1, "FP32 master layout differs")
        offset = 0
        for name, shape in shape_group.items():
            require(list(shape) == list(expected[name]["shape"]), "DS parameter shape differs: " + name)
            count = math.prod(integer(size, "parameter dimension", 1) for size in shape)
            value = parameters[name]
            require(offset + count <= master.numel(), "FP32 master is shorter than parameter layout")
            require(torch.equal(master[offset:offset + count].to(dtype=value.dtype).cpu(),
                                value.detach().reshape(-1).cpu()), "FP32 master/portable parameter differs: " + name)
            offset += count
        require(offset == master.numel(), "FP32 master has extra world-one elements")
        state = states[pid]
        for key, value in state.items():
            if torch.is_tensor(value):
                tensor(value, "Adam " + key)
        # DS0.12.6 initializes Torch AdamW once with zero gradients before any
        # engine step. FP16 overflows increment global/skipped, not Adam.step.
        require(integer(state["step"], "Adam step") == applied + 1,
                "Adam step differs from DS initialization plus successful updates")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = tensor(state[key], "Adam " + key)
            require(moment.dtype == torch.float32 and moment.shape == master.shape, "Adam moment layout differs")
            if key == "exp_avg_sq":
                require(bool((moment >= 0).all().item()), "Negative Adam second moment")
                positive_moments += int(bool(torch.count_nonzero(moment).item()))
    for group in groups:
        finite_number(group["lr"], "optimizer learning rate")
        finite_number(group["weight_decay"], "optimizer weight decay")
        finite_number(group["eps"], "optimizer epsilon", positive=True)
        require(len(group["betas"]) == 2 and all(
            0 <= finite_number(value, "Adam beta") < 1 for value in group["betas"]),
            "Invalid Adam betas")
    scheduler = scheduler_state if scheduler_state is not None else model_state["lr_scheduler"]
    require(isinstance(scheduler, Mapping) and len(scheduler["base_lrs"]) == len(groups)
            and len(scheduler["_last_lr"]) == len(groups), "Missing actual scheduler group state")
    require(integer(scheduler["last_epoch"], "scheduler last_epoch") == applied
            and integer(scheduler["_step_count"], "scheduler step count", 1) == applied + 1,
            "Scheduler does not match successful DS updates")
    for group, base, last in zip(groups, scheduler["base_lrs"], scheduler["_last_lr"], strict=True):
        finite_number(base, "scheduler base LR", positive=True)
        require(finite_number(last, "scheduler last LR") == float(group["lr"]),
                "Scheduler/optimizer learning rate differs")
    require(isinstance(scheduler["lr_lambdas"], list) and len(scheduler["lr_lambdas"]) == len(groups)
            and all(value is None or isinstance(value, dict) for value in scheduler["lr_lambdas"]),
            "Missing LambdaLR callable-state inventory")
    scaler = _scaler(zero_state, steps)
    if not scaler["dynamic"]:
        require(scaler["current_scale"] == loss_scale, "Static loss scale differs from configuration")
    if scaler["dynamic"]:
        require((skipped == 0) == (zero_state["loss_scaler"].last_overflow_iter == -1),
                "DS skipped-step/scaler overflow history differs")
    rng = _rng(rng_state, cuda_state_sizes)
    return {"status": "PASS_SEMANTIC_NATIVE_WORLD1_ZERO3_STATE", "global_steps": steps,
            "skipped_steps": skipped, "successful_updates": applied,
            "Adam_initialization_steps": 1, "master_groups": len(masters),
            "LoRA_tensors": len(lora_names), "projector_tensors": len(projector_names),
            "positive_second_moment_groups": positive_moments, "scaler": scaler, "RNG": rng,
            "learning_or_nonzero_movement_qualified": False}


def _load_and_validate(
    path: Path, files: Mapping[str, Any], global_step: int, *,
    cuda_state_sizes: Sequence[int], accepted: bool,
) -> dict[str, Any]:
    _verify_inventory(path, files, accepted=accepted)
    names = set(files)
    required = {"trainer_state.json", "config.json", "FROZEN_RECONSTRUCTION.json",
                "non_lora_trainables.bin", "rng_state.pth"}
    require(required <= names, "Missing semantic checkpoint files")
    model_files = [name for name in names if name.endswith("model_states.pt")]
    optim_files = [name for name in names if name.endswith("optim_states.pt")]
    require(len(model_files) == len(optim_files) == 1, "Missing/extra world-one DS shards")
    portable = names & {"adapter_model.bin", "adapter_model.safetensors"}
    require(len(portable) == 1, "Missing/ambiguous portable LoRA archive")
    def load(name: str) -> Any:
        return torch.load(path / name, map_location="cpu", weights_only=False)
    model = load(model_files[0])
    optimizer = load(optim_files[0])["optimizer_state_dict"]
    adapter_name = next(iter(portable))
    if adapter_name.endswith(".safetensors"):
        from safetensors.torch import load_file
        lora = load_file(str(path / adapter_name), device="cpu")
    else:
        lora = load(adapter_name)
    reconstruction = json.loads((path / "FROZEN_RECONSTRUCTION.json").read_text())
    config = json.loads((path / "config.json").read_text())
    require(config.get("model_type") == reconstruction["native_config"].get("model_type") == "llava_llama",
            "Portable model config is not LLaVA")
    scheduler = None
    if model["lr_scheduler"] is None:
        require("scheduler.pt" in names, "Missing DS/HF scheduler state")
        scheduler = load("scheduler.pt")
    result = validate_payloads(
        model_state=model, zero_state=optimizer, rng_state=load("rng_state.pth"),
        lora_state=lora, projector_state=load("non_lora_trainables.bin"),
        trainer_state=json.loads((path / "trainer_state.json").read_text()),
        reconstruction=reconstruction, cuda_state_sizes=cuda_state_sizes, scheduler_state=scheduler,
    )
    require(result["global_steps"] == global_step, "Expected/DS step differs")
    _verify_inventory(path, files, accepted=accepted)
    return {**result, "files_verified_before_and_after": True}


def validate_pre_acceptance(
    path: Path, files: Mapping[str, Any], global_step: int, *, cuda_state_sizes: Sequence[int],
) -> dict[str, Any]:
    """Refuse invalid newly saved state BEFORE Root publishes ACCEPTED.json."""
    path = Path(path).absolute()
    step = integer(global_step, "expected global_step")
    require(path.name == "checkpoint-" + str(step), "Saved cursor/directory differs")
    require(not (path / "ACCEPTED.json").exists(), "Pre-acceptance guard cannot adopt an already accepted state")
    inventory = {name: {"bytes": ref["bytes"], "sha256": ref["sha256"]} for name, ref in files.items()}
    return _load_and_validate(path, inventory, step, cuda_state_sizes=cuda_state_sizes, accepted=False)


def validate_checkpoint_directory(
    path: Path, identity: str, *, cuda_state_sizes: Sequence[int],
) -> dict[str, Any]:
    """Read only trusted project-created pickle files, inside owned CPU+GPU work."""
    path = Path(path).absolute()
    accepted = _accepted_metadata(path, identity)
    accepted_ref = descriptor(path / "ACCEPTED.json")
    result = _load_and_validate(path, accepted["files"], accepted["global_step"],
                                cuda_state_sizes=cuda_state_sizes, accepted=True)
    require(descriptor(path / "ACCEPTED.json") == accepted_ref, "Accepted receipt changed during validation")
    return {**result, "accepted": accepted, "accepted_descriptor": accepted_ref}
