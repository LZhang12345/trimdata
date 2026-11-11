"""Resumable, chunked Tool2 Preempt-cell native training (byte-for-byte Tool3 resumable trainer except identities).

Each invocation trains from the newest complete checkpoint until the job deadline,
saves a complete checkpoint, keeps only that one, and exits. The final chunk saves
the native adapter, writes RESULT.json and removes all intermediate checkpoints.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


ROOT = NATIVE = None  # set from the bound config (Babel HOME or the ACES staged copy of the same packet)
NATIVE_RELATIVE = ("collaboration/published-foundation-20261022/reference_sources/LLaVA/"
                   "c121f0432da27facab705978f83c4ada465e46fd")
ARMS = ("uni_coin", "m1u_coin", "m1u_coin_vn", "less_approx")  # v3: + less_approx (Tool2 LESS-approx baseline)
ROLES = ("TOOL2_PREEMPT_RESUMABLE_FULL_EPOCH_NOT_EVALUATED", "TOOL2_RESUME_QUALIFICATION_NOT_SCIENCE")
COMPLETE = "TOOL2_CHECKPOINT_COMPLETE.json"
AUDIT = "TOOL2_CHUNK_AUDIT.json"
SUCCESS = ("ACTUAL_FULL_NATIVE_TOOL2_PREEMPT_EPOCH_NOT_EVALUATED", "QUALIFICATION_COMPLETE")
EXPECTED_STEPS, MICROBATCHES_PER_STEP = 516, 32


class ChunkStop(Exception):
    """Training stopped at a saved boundary before the final step."""


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


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp-" + str(os.getpid()))
    with open(temporary, "wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def checkpoint_complete(path):
    marker = Path(path) / COMPLETE
    if not marker.is_file():
        return None
    value = json.loads(marker.read_bytes())
    for name, size in value["files"].items():
        if not (Path(path) / name).is_file() or (Path(path) / name).stat().st_size != size:
            return None
    return value


def prepare_training_dir(training):
    """Delete incomplete/staging checkpoints; keep only the newest complete one."""
    training.mkdir(parents=True, exist_ok=True)
    removed, complete = [], []
    for path in sorted(training.iterdir()):
        if path.is_dir() and path.name.startswith("tmp-checkpoint-"):
            shutil.rmtree(path)
            removed.append(path.name)
        elif path.is_dir() and path.name.startswith("checkpoint-"):
            marker = checkpoint_complete(path)
            if marker is None:
                shutil.rmtree(path)
                removed.append(path.name)
            else:
                complete.append((marker["global_step"], path))
    complete.sort()
    for _, path in complete[:-1]:
        shutil.rmtree(path)
        removed.append(path.name)
    return (complete[-1] if complete else None), removed


def arguments(config, training):
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
        gradient_accumulation_steps=32, evaluation_strategy="no", save_strategy="no",
        learning_rate=2e-4, weight_decay=0, warmup_ratio=.03, lr_scheduler_type="cosine",
        model_max_length=2048, gradient_checkpointing=True, dataloader_num_workers=0,
        lazy_preprocess=True, output_dir=str(training),
        seed=config["training_seed"], data_seed=config["training_seed"], optim="adamw_torch", report_to="none",
        disable_tqdm=True, logging_steps=10, save_safetensors=True,
        deepspeed=str(ROOT / "experiments/root_native_campaign_v7_20261024/zero3.json"),
    )
    if config["role"] == ROLES[1]:
        values["max_steps"] = config["qualification_max_steps"]
    return [part for key, value in values.items() for part in ("--" + key, str(value))]


def verify_inputs(config, output):
    """Hash pinned sources and selected input every chunk; hash large assets once per cell."""
    for relative, expected in config["sources"].items():
        require(".." not in Path(relative).parts and digest(ROOT / relative) == expected,
                "Pinned native/runtime source differs: " + relative)
    require(digest(config["input"]["path"]) == config["input"]["sha256"], "Actual selected input differs")
    files = [ref for role in ("vicuna", "clip336", "stage1_projector") for ref in config["assets"][role]["files"]]
    receipt = output / "ASSETS_VERIFIED.json"
    expected = {ref["path"]: [ref["bytes"], ref["sha256"]] for ref in files}
    if receipt.is_file() and json.loads(receipt.read_bytes())["assets"] == expected:
        for ref in files:
            require(Path(ref["path"]).stat().st_size == ref["bytes"], "Cached asset size changed")
        return "SIZE_CHECK_AFTER_PRIOR_FULL_HASH"
    for ref in files:
        require(Path(ref["path"]).stat().st_size == ref["bytes"] and digest(ref["path"]) == ref["sha256"],
                "Cached asset differs")
    write_json(receipt, {"assets": expected, "job_id": config["job_id"]})
    return "FULL_HASH"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_bytes())
    global ROOT, NATIVE
    ROOT = Path(config["root"])
    NATIVE = ROOT / NATIVE_RELATIVE
    require(Path(__file__).absolute() == Path(config["model_source"]["path"])
            and digest(__file__) == config["model_source"]["sha256"],
            "Current model source differs from its submitted binding")
    job = os.environ.get("SLURM_JOB_ID", "")
    link = config.get("link_id", job)
    require(link == job or link.startswith(job + "-r"), "Link identity must extend the actual job id")
    require(job.isdecimal() and config["job_id"] == job
            and os.environ.get("WORLD_SIZE") == "1" and os.environ.get("RANK") == "0",
            "Actual owned world-one GPU child required")
    output = Path(config["output"])
    require(config.get("arm") in ARMS and config["role"] in ROLES
            and config["selection_seed"] in (1234, 1235, 1236) and config["training_seed"] in (17, 29, 43)
            and output.name.startswith("t2pr-" + config["arm"] + "-sel" + str(config["selection_seed"])
                                       + "-train" + str(config["training_seed"]))
            and type(config["deadline_epoch"]) is int and type(config["save_steps"]) is int
            and config["save_steps"] > 0, "Exact resumable cell, role, deadline and save cadence required")
    qualify = config["role"] == ROLES[1]
    expected_steps = config["qualification_max_steps"] if qualify else EXPECTED_STEPS
    training = output / "training"
    if (output / "RESULT.json").is_file():
        require(json.loads((output / "RESULT.json").read_bytes())["status"] in SUCCESS, "Foreign RESULT.json")
        for path in (training.iterdir() if training.is_dir() else ()):
            if path.is_dir() and (path.name.startswith("checkpoint-") or path.name.startswith("tmp-checkpoint-")):
                shutil.rmtree(path)
        print(json.dumps({"job_id": job, "status": "ALREADY_COMPLETE", "output": str(output)}), flush=True)
        return
    asset_check = verify_inputs(config, output)
    latest, removed = prepare_training_dir(training)
    resume_step = latest[0] if latest else 0
    chunk_dir = output / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    journal_repaired = None
    if latest is not None:
        embedded = json.loads((latest[1] / AUDIT).read_bytes())
        require(embedded["end_step"] == resume_step, "Embedded audit differs from its checkpoint step")
        target = chunk_dir / ("chunk-" + embedded["link_id"] + ".json")
        current = json.loads(target.read_bytes()) if target.is_file() else None
        if current is None or (current["end_step"] or 0) < resume_step:
            write_json(target, embedded)
            journal_repaired = embedded["link_id"]
    require(not any(name == "llava" or name.startswith("llava.") for name in sys.modules),
            "Fresh native import required")
    sys.path[:0] = [str(ROOT), str(NATIVE)]
    from experiments.root_native_deterministic_attention_v2_20261025 import policy
    selected, attestation = policy.install()
    import torch
    require(torch.__version__ == "2.1.2+cu121" and torch.cuda.device_count() == 1
            and any(name in torch.cuda.get_device_name() for name in ("L40S", "H100")),
            "Exact native CUDA runtime required")
    patch = importlib.import_module("llava.train.llama_xformers_attn_monkey_patch")
    patch.replace_llama_attn_with_xformers_attn()
    module = importlib.import_module("llava.train.train")
    from experiments.root_native_campaign_v7_20261024 import optimizer_groups, structured_preprocess
    from experiments.root_native_campaign_v7_20261024.runtime_guards import install_expanded_target_guard
    from experiments.root_native_campaign_v7_20261024.resume_adapter import (
        assert_attention, restore_exact_trainable_and_buffers)
    import transformers.trainer as trainer_module
    from transformers import TrainerCallback
    structured_preprocess.install(module)
    parent = module.LLaVATrainer
    started = time.time()
    observation = {"microbatches": 0, "losses": [], "step_events": [], "indices": []}
    captured, restores = [], []
    original_load = trainer_module.deepspeed_load_checkpoint

    def verified_load(engine, checkpoint_path):
        require(checkpoint_complete(checkpoint_path) is not None,
                "Resume target is not a complete Tool2 checkpoint")
        loader = engine.load_module_state_dict

        def scoped(checkpoint, strict=True, custom_load_fn=None, fetch_z3_params=False):
            require(custom_load_fn is None and fetch_z3_params is False, "Unexpected DeepSpeed restore mode")
            require(checkpoint.get("frozen_param_fragments") is None,
                    "Frozen fragments unexpectedly present in excluded-frozen checkpoint")

            def exact(src, dst):
                require(dst is engine.module, "Foreign restore destination")
                restores.append(restore_exact_trainable_and_buffers(src, dst))
            return loader(checkpoint, strict=True, custom_load_fn=exact, fetch_z3_params=False)
        engine.load_module_state_dict = scoped
        try:
            return original_load(engine, checkpoint_path)
        finally:
            engine.load_module_state_dict = loader

    class Boundaries(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_step_end(self, args, state, control, **kwargs):
            optimizer = self.trainer.model_wrapped.optimizer
            require(hasattr(optimizer, "overflow"), "Actual FP16 overflow state unavailable")
            step = int(state.global_step)
            scheduler = self.trainer.lr_scheduler
            observation["step_events"].append({
                "step": step, "overflow": bool(optimizer.overflow), "microbatches_seen": observation["microbatches"],
                "lr": [float(value) for value in scheduler.get_last_lr()] if scheduler is not None else None,
                "loss_scale": float(getattr(optimizer, "loss_scale", float("nan")))})
            if step % 10 == 0 or step == state.max_steps:
                recent = observation["losses"][-320:]
                print(json.dumps({"step": step, "max_steps": int(state.max_steps),
                                  "mean_recent_microbatch_loss": sum(recent) / max(len(recent), 1),
                                  "overflow": bool(optimizer.overflow)}), flush=True)
            deadline = time.time() >= config["deadline_epoch"] or (
                qualify and config.get("stop_after_steps") is not None
                and step - resume_step >= config["stop_after_steps"])
            if step % config["save_steps"] == 0 or deadline or step >= state.max_steps:
                control.should_save = True
            if deadline and step < state.max_steps:
                control.should_training_stop = True
            return control

    class ResumableTrainer(parent):
        def create_optimizer(self):
            result = super().create_optimizer()
            if not hasattr(self, "_tool3_prune"):
                self._tool3_prune = optimizer_groups.prune_before_scheduler(
                    self.optimizer, self.model.named_parameters(), scheduler=self.lr_scheduler,
                    main_lr=2e-4, projector_lr=2e-5, weight_decay=0)
            return result

        def compute_loss(self, model, inputs, return_outputs=False):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs)
            loss = result[0] if return_outputs else result
            value = float(loss.detach().float().item())
            require(math.isfinite(value), "Nonfinite actual native loss")
            observation["microbatches"] += 1
            observation["losses"].append(value)
            return result

        def save_model(self, output_dir=None, _internal_call=False):
            if _internal_call:
                # DeepSpeed state already holds every trainable tensor; skip the 7B 16-bit gather.
                os.makedirs(output_dir, exist_ok=True)
                return None
            return super().save_model(output_dir, _internal_call=_internal_call)

        def _save_checkpoint(self, model, trial, metrics=None):
            super()._save_checkpoint(model, trial, metrics)
            step = int(self.state.global_step)
            path = training / ("checkpoint-" + str(step))
            files = {str(p.relative_to(path)): p.stat().st_size for p in sorted(path.rglob("*")) if p.is_file()}
            require("trainer_state.json" in files and "rng_state.pth" in files
                    and any(name.endswith("optim_states.pt") for name in files)
                    and any(name.endswith("model_states.pt") for name in files),
                    "Incomplete DeepSpeed/HF checkpoint")
            for name in files:
                with open(path / name, "rb") as stream:
                    os.fsync(stream.fileno())
            observation.setdefault("saved_steps", []).append(step)
            observation["end_step"] = step
            record = chunk_record("SAVED", None)
            # The progress record travels inside the checkpoint, so marker and audit commit together.
            write_json(path / AUDIT, record)
            files[AUDIT] = (path / AUDIT).stat().st_size
            write_json(path / COMPLETE, {"global_step": step, "files": files, "job_id": job, "link_id": link,
                                         "logical_bytes": sum(files.values())})
            write_json(chunk_dir / ("chunk-" + link + ".json"), record)
            for other in training.iterdir():
                if other.is_dir() and other != path and (other.name.startswith("checkpoint-")
                                                        or other.name.startswith("tmp-checkpoint-")):
                    shutil.rmtree(other)

        def train(self, resume_from_checkpoint=None, **kwargs):
            require(not kwargs, "Unexpected trainer arguments")
            data = canonical(self.train_dataset.list_data_dict) + b"\n"
            require(len(self.train_dataset) == 66076
                    and hashlib.sha256(data).hexdigest() == config["input"]["sha256"],
                    "Actual native dataset differs from qualified selection")
            require(self.args.world_size == 1 and self.args.per_device_train_batch_size == 4
                    and self.args.gradient_accumulation_steps == 32
                    and self.args.dataloader_num_workers == 0 and self.args.num_train_epochs == 1
                    and self.args.max_steps == (config["qualification_max_steps"] if qualify else -1)
                    and self.args.seed == config["training_seed"]
                    and self.args.data_seed == config["training_seed"],
                    "Actual training arguments differ")
            require(bool(resume_from_checkpoint) == (latest is not None),
                    "Native resume decision differs from complete-checkpoint state")
            observation["attention"] = assert_attention(self.model, patch)
            dataset_type = type(self.train_dataset)
            original_getitem = dataset_type.__getitem__

            def audited_getitem(dataset, index):
                observation["indices"].append(int(index))
                return original_getitem(dataset, index)
            dataset_type.__getitem__ = audited_getitem
            base = self.model.get_base_model()
            original = base.prepare_inputs_labels_for_multimodal
            install_expanded_target_guard(self.model)
            self.add_callback(Boundaries(self))
            captured.append(self)
            try:
                result = super().train(resume_from_checkpoint=resume_from_checkpoint)
            finally:
                base.prepare_inputs_labels_for_multimodal = original
                dataset_type.__getitem__ = original_getitem
            observation["end_step"] = int(self.state.global_step)
            observation["max_steps"] = int(self.state.max_steps)
            observation["final_overflow"] = bool(getattr(self.model_wrapped.optimizer, "overflow", True))
            if self.state.global_step < self.state.max_steps:
                raise ChunkStop()
            return result

    def chunk_record(status, operators):
        losses = observation["losses"]
        # The accelerate loader prefetches one batch ahead; keep only indices of microbatches actually trained.
        consumed = observation["indices"][:observation["microbatches"] * 4]
        return dict(status=status, job_id=job, link_id=link, resume_step=resume_step,
                    end_step=observation.get("end_step"), consumed_indices=consumed,
                    max_steps=observation.get("max_steps"), microbatches=observation["microbatches"],
                    step_events=observation["step_events"], saved_steps=observation.get("saved_steps", []),
                    restores=len(restores), removed_at_start=removed, asset_check=asset_check,
                    journal_repaired=journal_repaired,
                    first_losses=losses[:64], last_losses=losses[-64:],
                    loss_mean=(sum(losses) / len(losses)) if losses else None,
                    seconds=round(time.time() - started, 1),
                    operators_calls=operators.get("calls") if operators else int(selected.calls),
                    runtime_attestation=attestation, gpu=torch.cuda.get_device_name())

    trainer_module.deepspeed_load_checkpoint = verified_load
    module.LLaVATrainer = ResumableTrainer
    sys.argv = ["tool2-native-preempt-resumable"] + arguments(config, training)
    try:
        module.transformers.set_seed(config["training_seed"])
        require(torch.initial_seed() == config["training_seed"], "Model initialization seed differs")
        try:
            module.train(attn_implementation="eager")
        except ChunkStop:
            operators = selected.finish()
            require(latest is None or len(restores) == 1, "Resumed chunk did not restore exactly once")
            write_json(chunk_dir / ("chunk-" + link + ".json"), chunk_record("CHUNK_SAVED_CONTINUE", operators))
            print(json.dumps({"job_id": job, "status": "CHUNK_SAVED_CONTINUE",
                              "resume_step": resume_step, "end_step": observation.get("end_step")}), flush=True)
            return
        operators = selected.finish()
        require(len(captured) == 1, "Exactly one actual native trainer required")
        require(latest is None or len(restores) == 1, "Resumed chunk did not restore exactly once")
        write_json(chunk_dir / ("chunk-" + link + ".json"), chunk_record("FINAL_CHUNK", operators))
        required = ("adapter_config.json", "config.json", "non_lora_trainables.bin", "trainer_state.json")
        require(all((training / name).is_file() for name in required)
                and any((training / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")),
                "Final native adapter/projector or training state missing")
        chunks = sorted((json.loads(p.read_bytes()) for p in chunk_dir.glob("chunk-*.json")),
                        key=lambda value: (value["resume_step"], value["end_step"] or 0))
        tiled = [value for value in chunks if value["status"] in ("SAVED", "CHUNK_SAVED_CONTINUE", "FINAL_CHUNK")]
        cursor, coverage = 0, True
        for value in tiled:
            coverage = coverage and value["resume_step"] == cursor
            cursor = value["end_step"]
        exposure = [index for value in tiled for index in value["consumed_indices"]]
        checks = {
            "final_step": observation.get("end_step") == observation.get("max_steps") == expected_steps,
            "chunks_tile_all_steps": coverage and cursor == expected_steps,
            "microbatches": sum(value["microbatches"] for value in tiled) == expected_steps * MICROBATCHES_PER_STEP,
            "exposure_exact_once": len(exposure) == expected_steps * MICROBATCHES_PER_STEP * 4 == len(set(exposure))
            and all(0 <= index < 66076 for index in exposure),
            "deterministic_attention_executed": (operators.get("calls") or 0) > 0 or (
                resume_step == expected_steps and any((v.get("operators_calls") or 0) > 0 for v in tiled)),
        }
        passed = all(checks.values())
        report = dict(status=("ACTUAL_FULL_NATIVE_TOOL2_PREEMPT_EPOCH_NOT_EVALUATED" if passed and not qualify else
                              "QUALIFICATION_COMPLETE" if passed else "ADAPTER_SAVED_WITH_TRAINING_AUDIT_MISMATCH"),
                      job_id=job, config=config, audit_checks=checks, chunks=[v["link_id"] for v in tiled],
                      exposure_sha256=hashlib.sha256(canonical(exposure)).hexdigest(),
                      final_step_overflow=observation.get("final_overflow"),
                      overflow_steps=sum(1 for v in tiled for e in v["step_events"] if e["overflow"]),
                      output_files={p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                                    for p in training.iterdir() if p.is_file()},
                      full_epoch_training_complete=passed and not qualify, scientific_performance_measured=False,
                      independent_selection_output_acceptance=False)
        if not passed:
            # Keep the final checkpoint for diagnosis; never publish a success-looking RESULT.json.
            write_json(output / ("AUDIT_MISMATCH-" + link + ".json"), report)
            raise ValueError("Training audit mismatch: " + json.dumps(checks))
        write_json(output / "RESULT.json", report)
        for path in training.iterdir():
            if path.is_dir() and (path.name.startswith("checkpoint-") or path.name.startswith("tmp-checkpoint-")):
                shutil.rmtree(path)
        print(json.dumps({"job_id": job, "status": report["status"], "output": str(output)}), flush=True)
    finally:
        module.LLaVATrainer = parent
        trainer_module.deepspeed_load_checkpoint = original_load
        selected.restore()


if __name__ == "__main__":
    main()
