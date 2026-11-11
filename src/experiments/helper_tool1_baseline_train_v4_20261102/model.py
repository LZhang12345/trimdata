"""Tool1 baseline native trajectory with accepted checkpoints; resume only from verified accepted states.

Roster-agnostic: the controller supplies cell, K, horizon and inputs.  The native recipe is the same
as the executed v1 trainer.  Checkpoints are HF/DeepSpeed states accepted only after a full file
inventory and an observation snapshot are sealed.  Resume restores through Root's verified
frozen-exclusion adapter and continues the persisted per-microbatch observation.
v4: a production attempt preempted right after its final accepted checkpoint resumes at the horizon.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys


ROOT = Path("/home/luyangz/data_selection_multimodal/coincide_native_exec_src_v2_20261028")
NATIVE = ROOT / ("collaboration/published-foundation-20261022/reference_sources/LLaVA/"
                 "c121f0432da27facab705978f83c4ada465e46fd")
OUTPUT_ROOT = Path("/data/hf_cache/project_checkpoints/DS-MultiModal/helper-tool1-baseline-train-v4-20261102")
ROLES = {
    "TOOL1_BASELINE_FULL_EPOCH_NOT_EVALUATED": "ACTUAL_FULL_NATIVE_BASELINE_EPOCH_NOT_EVALUATED",
    "TOOL1_RESTART_PARITY_PROBE": "ACTUAL_RESTART_PARITY_PROBE_NOT_SCIENTIFIC",
}
ACCEPTED = "ACCEPTED_FULL_NATIVE_STATE"
PROBE_STOP_EXIT = 75


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    """Create one immutable fsynced JSON file; never overwrite."""
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def inventory(path):
    path = Path(path)
    files = {}
    for item in sorted(path.rglob("*")):
        require(not item.is_symlink(), "Checkpoint symlink is not accepted: " + str(item))
        if item.is_file() and item.name != "ACCEPTED.json":
            files[str(item.relative_to(path))] = {"bytes": item.stat().st_size, "sha256": digest(item)}
    return files


def verify_accepted(path, identity):
    path = Path(path)
    accepted = json.loads((path / "ACCEPTED.json").read_bytes())
    require(accepted.get("status") == ACCEPTED and accepted.get("identity") == identity
            and accepted.get("global_step") == int(path.name.split("-")[-1]),
            "Accepted checkpoint identity or step differs: " + str(path))
    require(inventory(path) == accepted["files"], "Accepted checkpoint files changed: " + str(path))
    return accepted


def horizon(config):
    k = config["K"]
    require(type(k) is int and 1024 <= k <= 660763, "Bounded actual selected-record count required")
    natural = (k + 3) // 4 // 32
    value = config["horizon"]
    require(value["loader_batches"] == (k + 3) // 4 and value["natural_optimizer_boundaries"] == natural,
            "Native loader/optimizer arithmetic differs")
    if config["role"] == "TOOL1_RESTART_PARITY_PROBE":
        steps = value["probe_max_steps"]
        require(type(steps) is int and 2 <= steps < natural, "Bounded probe horizon required")
    else:
        steps = natural
        require(value.get("probe_max_steps") is None, "Production horizon cannot be truncated")
    return steps


def arguments(config, steps):
    assets = config["assets"]
    values = dict(
        model_name_or_path=assets["vicuna"]["snapshot"], version="v1",
        data_path=config["input"]["path"], image_folder=config["image_root"],
        vision_tower=assets["clip336"]["snapshot"],
        pretrain_mm_mlp_adapter=str(Path(assets["stage1_projector"]["snapshot"]) / "mm_projector.bin"),
        mm_projector_type="mlp2x_gelu", mm_vision_select_layer=-2,
        mm_use_im_start_end=False, mm_use_im_patch_token=False, image_aspect_ratio="pad",
        group_by_modality_length=True, lora_enable=True, lora_r=128, lora_alpha=256,
        lora_dropout=.05, lora_bias="none", mm_projector_lr=2e-5,
        fp16=True, bf16=False, tf32=False, num_train_epochs=1,
        per_device_train_batch_size=4, per_device_eval_batch_size=1,
        gradient_accumulation_steps=32, evaluation_strategy="no",
        save_strategy="steps", save_steps=config["checkpointing"]["save_steps"],
        learning_rate=2e-4, weight_decay=0, warmup_ratio=.03, lr_scheduler_type="cosine",
        model_max_length=2048, gradient_checkpointing=True, dataloader_num_workers=0,
        lazy_preprocess=True, output_dir=str(Path(config["output"]) / "training"),
        seed=config["training_seed"], data_seed=config["training_seed"], optim="adamw_torch", report_to="none",
        disable_tqdm=True, logging_steps=10, save_safetensors=True,
        deepspeed=str(ROOT / "experiments/root_native_campaign_v7_20261024/zero3.json"),
    )
    if config["role"] == "TOOL1_RESTART_PARITY_PROBE":
        values["max_steps"] = steps
    return [part for key, value in values.items() for part in ("--" + key, str(value))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_bytes())
    require(Path(__file__).absolute() == Path(config["model_source"]["path"])
            and digest(__file__) == config["model_source"]["sha256"],
            "Current private model source differs from its submitted binding")
    job = os.environ.get("SLURM_JOB_ID", "")
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    require(os.getuid() == 2708899 and job.isdecimal() and config["job_id"] == job
            and config["attempt"]["restart_count"] == restart
            and os.environ.get("WORLD_SIZE") == "1" and os.environ.get("RANK") == "0",
            "Actual owned world-one GPU child for this attempt required")
    output = Path(config["output"])
    cell = config["cell"]
    require(isinstance(cell, str) and re.fullmatch(r"[a-z0-9][a-z0-9-]{2,80}", cell) is not None
            and output == OUTPUT_ROOT / cell and config["role"] in ROLES
            and type(config["training_seed"]) is int and config["training_seed"] in (17, 29, 43),
            "Exact owned cell, output, role and seed required")
    steps = horizon(config)
    save_steps = config["checkpointing"]["save_steps"]
    keep = config["checkpointing"]["retain_accepted"]
    require(type(save_steps) is int and 1 <= save_steps <= steps and keep in (1, 2),
            "Bounded checkpoint cadence and retention required")
    stop_after = config.get("probe_stop_after_step")
    require(stop_after is None or (config["role"] == "TOOL1_RESTART_PARITY_PROBE" and restart == 0
                                   and type(stop_after) is int and stop_after % save_steps == 0
                                   and 0 < stop_after < steps),
            "Only a first-attempt probe may stop, and only at an accepted boundary")
    identity = config["function_identity"]
    attempt_dir = Path(config["attempt"]["directory"])
    require(attempt_dir.is_dir() and not any(attempt_dir.iterdir()), "Fresh private attempt directory required")
    for relative, expected in config["sources"].items():
        path = ROOT / relative
        require(".." not in Path(relative).parts and digest(path) == expected,
                "Pinned native/runtime source differs: " + relative)
    for role in ("vicuna", "clip336", "stage1_projector"):
        for ref in config["assets"][role]["files"]:
            require(Path(ref["path"]).stat().st_size == ref["bytes"]
                    and digest(ref["path"]) == ref["sha256"], "Cached asset differs")
    require(digest(config["input"]["path"]) == config["input"]["sha256"],
            "Actual immutable selected input differs")
    training = output / "training"
    resume = config["resume"]
    observation = {"microbatches": 0, "losses": [], "optimizer_boundaries": 0, "step_events": [],
                   "attempts": []}
    if resume is None:
        require(not training.exists() or not any(training.iterdir()), "Fresh trajectory requires an empty training dir")
    else:
        path = Path(resume["path"])
        require(path.parent == training and path.name == "checkpoint-%d" % resume["global_step"]
                and 0 < resume["global_step"] <= steps, "Resume checkpoint must be this trajectory's own state")
        accepted = verify_accepted(path, identity)
        require(canonical(accepted) == canonical(resume["accepted"]), "Resume acceptance receipt changed")
        saved = json.loads((path / "OBSERVATION.json").read_bytes())
        require(saved["optimizer_boundaries"] == resume["global_step"] == len(saved["step_events"])
                and saved["microbatches"] == 32 * resume["global_step"] == len(saved["losses"]),
                "Persisted observation does not match the accepted boundary")
        observation.update(saved)
        observation["optimizer_boundaries"] = saved["optimizer_boundaries"]
        others = [p for p in training.iterdir() if p.name != path.name]
        require(not others, "Only the selected accepted checkpoint may remain in the training directory")
    observation["attempts"] = list(observation.get("attempts", [])) + [
        {"job_id": job, "restart_count": restart, "resumed_from_step": None if resume is None else resume["global_step"]}]
    require(not any(name == "llava" or name.startswith("llava.") for name in sys.modules),
            "Fresh native import required")
    sys.path[:0] = [str(ROOT), str(NATIVE)]
    from experiments.root_native_deterministic_attention_v2_20261025 import policy
    selected, attestation = policy.install()
    import torch
    require(torch.__version__ == "2.1.2+cu121" and torch.cuda.device_count() == 1
            and "L40S" in torch.cuda.get_device_name(), "Exact native CUDA runtime required")
    patch = importlib.import_module("llava.train.llama_xformers_attn_monkey_patch")
    patch.replace_llama_attn_with_xformers_attn()
    module = importlib.import_module("llava.train.train")
    from experiments.root_native_campaign_v7_20261024 import optimizer_groups, structured_preprocess
    from experiments.root_native_campaign_v7_20261024 import resume_adapter
    from experiments.root_native_campaign_v7_20261024.runtime_guards import install_expanded_target_guard
    from experiments.root_native_campaign_v7_20261024.resume_adapter import assert_attention
    structured_preprocess.install(module)
    parent = module.LLaVATrainer
    reconstruction_config = {"providers": {"assets": config["reference_assets"]}, "recipe": config["recipe"]}
    retired = []
    captured = []
    from transformers import TrainerCallback

    class StepAudit(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_step_end(self, args, state, control, **kwargs):
            optimizer = self.trainer.model_wrapped.optimizer
            require(hasattr(optimizer, "overflow"), "Actual FP16 overflow state unavailable")
            observation["step_events"].append({
                "step": int(state.global_step), "overflow": bool(optimizer.overflow),
                "microbatches_seen": observation["microbatches"],
            })
            observation["optimizer_boundaries"] = int(state.global_step)
            return control

    def checkpoint_receipt(path, expected_identity):
        require(expected_identity == identity, "Foreign restore identity")
        return verify_accepted(path, identity)

    class BaselineTrainer(parent):
        def create_optimizer(self):
            result = super().create_optimizer()
            if not hasattr(self, "_tool1_prune"):
                self._tool1_prune = optimizer_groups.prune_before_scheduler(
                    self.optimizer, self.model.named_parameters(), scheduler=self.lr_scheduler,
                    main_lr=2e-4, projector_lr=2e-5, weight_decay=0)
                self.model._native_optimizer_group_inventory = self._tool1_prune
                if "optimizer_inventory" in observation:
                    require(observation["optimizer_inventory"] == json.loads(canonical(self._tool1_prune)),
                            "Resumed optimizer group inventory differs from the accepted trajectory")
                observation["optimizer_inventory"] = json.loads(canonical(self._tool1_prune))
            return result

        def compute_loss(self, model, inputs, return_outputs=False):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs)
            loss = result[0] if return_outputs else result
            value = float(loss.detach().float().item())
            require(math.isfinite(value), "Nonfinite actual native loss")
            observation["microbatches"] += 1
            observation["losses"].append(value)
            return result

        def _rotate_checkpoints(self, *args, **kwargs):
            return

        def _save_checkpoint(self, model, trial, metrics=None):
            step = int(self.state.global_step)
            path = training / ("checkpoint-%d" % step)
            require(not path.exists() and not (training / ("tmp-checkpoint-%d" % step)).exists(),
                    "Checkpoint target already exists; quarantine is required before replay")
            require(observation["optimizer_boundaries"] == step == len(observation["step_events"])
                    and observation["microbatches"] == 32 * step,
                    "Observation is not at the exact accepted boundary")
            super()._save_checkpoint(model, trial, metrics=metrics)
            require(path.is_dir() and (path / "rng_state.pth").is_file() and (path / "scheduler.pt").is_file()
                    and (path / "trainer_state.json").is_file()
                    and len(list(path.glob("global_step*/*optim_states.pt"))) == 1
                    and len(list(path.glob("global_step*/*model_states.pt"))) == 1
                    and (path / "adapter_model.safetensors").is_file(), "Incomplete native checkpoint")
            record = resume_adapter.reconstruction_record(self.model_wrapped.module, reconstruction_config)
            (path / "FROZEN_RECONSTRUCTION.json").write_bytes(canonical(record) + b"\n")
            (path / "OBSERVATION.json").write_bytes(canonical(observation) + b"\n")
            write_new(path / "ACCEPTED.json", {"status": ACCEPTED, "identity": identity, "global_step": step,
                                               "files": inventory(path), "job_id": job, "restart_count": restart})
            accepted = sorted((int(p.name.split("-")[-1]), p) for p in training.glob("checkpoint-*")
                              if (p / "ACCEPTED.json").is_file())
            for older_step, older in accepted[:-keep]:
                receipt = json.loads((older / "ACCEPTED.json").read_bytes())
                require(receipt.get("status") == ACCEPTED and receipt.get("identity") == identity
                        and receipt.get("global_step") == older_step, "Retirement target is not this trajectory's state")
                shutil.rmtree(older)
                retired.append({"global_step": older_step, "files": receipt["files"], "job_id": job,
                                "restart_count": restart})
                write_new(attempt_dir / ("RETIRED-checkpoint-%d.json" % older_step), retired[-1])
            if stop_after == step:
                write_new(attempt_dir / "PROBE_STOP.json", {"global_step": step, "job_id": job})
                sys.stdout.flush()
                os._exit(PROBE_STOP_EXIT)

        def train(self, *args, **kwargs):
            require(not args and kwargs in ({}, {"resume_from_checkpoint": True}),
                    "Only native fresh or accepted-state resume calls are allowed")
            require(bool(kwargs) == (resume is not None), "Native resume decision differs from the accepted plan")
            data = canonical(self.train_dataset.list_data_dict) + b"\n"
            require(len(self.train_dataset) == config["K"]
                    and hashlib.sha256(data).hexdigest() == config["input"]["canonical_sha256"],
                    "Actual native dataset differs from the verified frozen selection")
            require(self.args.world_size == 1 and self.args.per_device_train_batch_size == 4
                    and self.args.gradient_accumulation_steps == 32
                    and self.args.dataloader_num_workers == 0
                    and self.args.max_steps == (steps if config["role"] == "TOOL1_RESTART_PARITY_PROBE" else -1)
                    and self.args.num_train_epochs == 1 and self.args.save_steps == save_steps
                    and self.args.seed == config["training_seed"]
                    and self.args.data_seed == config["training_seed"],
                    "Actual natural-epoch training arguments differ")
            observation["attention"] = assert_attention(self.model, patch)
            base = self.model.get_base_model()
            original = base.prepare_inputs_labels_for_multimodal
            install_expanded_target_guard(self.model)
            self.add_callback(StepAudit(self))
            captured.append(self)
            try:
                if resume is None:
                    result = super().train()
                else:
                    with resume_adapter.install_verified_restore(reconstruction_config, identity, attempt_dir,
                                                                 checkpoint_receipt):
                        result = super().train(resume_from_checkpoint=resume["path"])
                require(self.state.global_step == self.state.max_steps == steps
                        and observation["microbatches"] == 32 * steps == len(observation["losses"])
                        and [event["step"] for event in observation["step_events"]] == list(range(1, steps + 1)),
                        "Actual native update or microbatch count differs across attempts")
                observation["overflow_boundaries"] = sum(event["overflow"] for event in observation["step_events"])
                observation["non_overflow_boundaries"] = steps - observation["overflow_boundaries"]
                return result
            finally:
                base.prepare_inputs_labels_for_multimodal = original

    module.LLaVATrainer = BaselineTrainer
    sys.argv = ["tool1-native-baseline-v2"] + arguments(config, steps)
    try:
        module.transformers.set_seed(config["training_seed"])
        require(torch.initial_seed() == config["training_seed"], "Model initialization seed differs")
        observation["model_initialization_seed"] = int(torch.initial_seed())
        module.train(attn_implementation="eager")
        require(len(captured) == 1, "Exactly one actual native trainer required")
        required = ("adapter_config.json", "config.json", "non_lora_trainables.bin", "trainer_state.json")
        require(all((training / name).is_file() for name in required)
                and any((training / name).is_file()
                        for name in ("adapter_model.safetensors", "adapter_model.bin")),
                "Final native adapter/projector or training state missing")
        operators = selected.finish()
        resumed_at_horizon = resume is not None and resume["global_step"] == steps
        require(operators["calls"] > 0 or resumed_at_horizon, "No actual deterministic attention execution")
        report = dict(status=ROLES[config["role"]], cell=cell, K=config["K"], steps=steps,
                      job_id=job, restart_count=restart, config=config, observation=observation,
                      runtime_attestation=attestation, operators=operators, retired_checkpoints=retired,
                      output_files={p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                                    for p in training.iterdir() if p.is_file()},
                      full_epoch_training_complete=config["role"] != "TOOL1_RESTART_PARITY_PROBE",
                      scientific_performance_measured=False, independently_reviewed=False)
        write_new(output / "RESULT.json", report)
        print(json.dumps({"job_id": job, "status": report["status"], "output": str(output)}), flush=True)
    finally:
        module.LLaVATrainer = parent
        selected.restore()


if __name__ == "__main__":
    main()
