#!/usr/bin/env python3
"""Portable reproduction of the executed baseline trainer (LLaVA-1.5-7B stage-2 LoRA, one natural epoch).

Our runs used the official LLaVA trainer (`llava/train/train.py` at commit c121f04) with exactly these runtime
changes, which this script re-applies without editing any LLaVA file:

  1. Training attention: LLaVA's own xformers patch (`llava/train/llama_xformers_attn_monkey_patch.py`) with the
     xformers CUTLASS forward/backward kernels forced, plus deterministic torch/cuBLAS settings
     (src/experiments/root_native_deterministic_attention_v{1,2}_*). Inference used plain eager attention instead.
  2. Structured v1 label masking (src/experiments/root_native_campaign_v7_20261024/structured_preprocess.py). The
     official `preprocess_v1` finds turn boundaries by splitting text, which mis-masks conversations whose content
     contains a role marker such as "USER:"; the repair takes boundaries from the turn structure.
  3. Removal of the always-empty "LoRA, no weight decay" optimizer group before the LR scheduler is built
     (src/experiments/root_native_campaign_v7_20261024/optimizer_groups.py). DeepSpeed 0.12.6 otherwise removes that
     group after the scheduler has captured four base learning rates, which misaligns the per-group learning rates
     (the projector weights can receive the LoRA rate 2e-4 instead of 2e-5).

It also installs the observation-only guard used in our runs (every row keeps at least one supervised token after
image-token expansion) and checks, during training, that the scheduler drives exactly three optimizer groups with
the projector at 0.1x the LoRA learning rate.

This file is a portable re-packaging of those modules; the modules themselves are imported unchanged from `src/`.
It was not itself executed on our cluster: the T0b calibration in AGENT_CONTRACT.md qualifies it on your hardware.

Single GPU (the setting of every provided run): one process, per-device batch 4 x accumulation 32 = global 128.
    CUDA_VISIBLE_DEVICES=0 python scripts/train_portable.py --data-path records.json --image-folder /data/llava \
        --model /models/vicuna-7b-v1.5 --vision-tower /models/clip-vit-large-patch14-336 \
        --projector /models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
        --output-dir runs/uniform-b10-sel1234-t17-a1 --seed 17 --expected-steps 516
Multi-GPU is only for tasks the contract allows (full-data reference): launch with torchrun and
--allow-multi-gpu; per-device batch x accumulation x world size must equal 128.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDORED_LLAVA = (REPO / "src/collaboration/published-foundation-20261022/reference_sources/LLaVA/"
                  "c121f0432da27facab705978f83c4ada465e46fd")
ZERO3 = REPO / "scripts/zero3.json"
CUBLAS = ":4096:8"
MAIN_LR = 2e-4
PROJECTOR_LR = 2e-5
GLOBAL_BATCH = 128


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", required=True, help="records.json rebuilt by scripts/reconstruct_records.py")
    p.add_argument("--image-folder", required=True, help="root containing coco/ gqa/ ocr_vqa/ textvqa/ vg/")
    p.add_argument("--model", required=True, help="local dir of lmsys/vicuna-7b-v1.5 @ 3321f76e")
    p.add_argument("--vision-tower", required=True, help="local dir of openai/clip-vit-large-patch14-336 @ ce19dc91")
    p.add_argument("--projector", required=True, help="stage-1 mm_projector.bin (llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5)")
    p.add_argument("--output-dir", required=True, help="fresh, non-existing or empty directory (one per attempt)")
    p.add_argument("--seed", type=int, required=True, choices=(17, 29, 43), help="training seed (also data_seed)")
    p.add_argument("--expected-steps", type=int, required=True,
                   help="natural_steps from the contract (258/516/774/1032/5162); the run fails if it differs")
    p.add_argument("--per-device-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--llava-root", default=str(VENDORED_LLAVA),
                   help="checkout of haotian-liu/LLaVA at c121f0432da27facab705978f83c4ada465e46fd")
    p.add_argument("--allow-multi-gpu", action="store_true", help="only for tasks where the contract allows it")
    p.add_argument("--master-port", type=int, default=None)
    return p.parse_args()


def fail(message):
    sys.stderr.write("train_portable: ERROR: " + message + "\n")
    sys.exit(2)


def prepare_environment(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS)
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != CUBLAS:
        fail("CUBLAS_WORKSPACE_CONFIG must be %s" % CUBLAS)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        port = args.master_port or (29500 + int(visible) if visible.isdigit() else 29500)
        for key, value in dict(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK="0", LOCAL_RANK="0",
                               WORLD_SIZE="1").items():
            os.environ.setdefault(key, value)
    elif not args.allow_multi_gpu:
        fail("WORLD_SIZE=%d but every provided run is single-GPU; use one process per GPU "
             "(pass --allow-multi-gpu only where the contract allows it)" % world)
    if args.per_device_batch * args.grad_accum * world != GLOBAL_BATCH:
        fail("global batch must be %d (per-device %d x accumulation %d x world %d)"
             % (GLOBAL_BATCH, args.per_device_batch, args.grad_accum, world))
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        fail("output dir %s is not empty; use a fresh directory per attempt (LLaVA would silently resume)" % out)
    for path in (args.data_path, args.image_folder, args.model, args.vision_tower, args.projector,
                 Path(args.llava_root) / "llava/train/train.py", ZERO3):
        if not Path(path).exists():
            fail("missing input: %s" % path)
    return world


def install_deterministic_cutlass_attention():
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import xformers
    import xformers.ops as xops
    from xformers.ops import fmha
    from xformers.ops.fmha import cutlass
    if xformers.__version__ != "0.0.23.post1":
        print("train_portable: WARNING: xformers %s (executed runs used 0.0.23.post1)" % xformers.__version__)
    original = fmha.memory_efficient_attention

    def cutlass_attention(query, key, value, attn_bias=None, p=0.0, scale=None, *, op=None):
        return original(query, key, value, attn_bias=attn_bias, p=p, scale=scale, op=(cutlass.FwOp, cutlass.BwOp))

    xops.memory_efficient_attention = cutlass_attention
    fmha.memory_efficient_attention = cutlass_attention
    from llava.train.llama_xformers_attn_monkey_patch import replace_llama_attn_with_xformers_attn
    replace_llama_attn_with_xformers_attn()


def build_trainer_class(module, optimizer_groups, install_expanded_target_guard):
    from transformers import TrainerCallback

    class LearningRateGroupCheck(TrainerCallback):
        def on_step_end(self, args, state, control, lr_scheduler=None, **kwargs):
            if state.global_step not in (1, 10, 100) or lr_scheduler is None:
                return control
            groups = lr_scheduler.optimizer.param_groups
            rates = [g["lr"] for g in groups]
            base = list(getattr(lr_scheduler, "base_lrs", []))
            ratio = PROJECTOR_LR / MAIN_LR
            # base_lrs are captured when the scheduler is built; a bug leaves 4 of them for 3 groups. Current rates may
            # all be 0 if an fp16 overflow skipped the first scheduler step during warmup.
            ok = (len(groups) == 3 and len(base) == 3 and math.isclose(base[0], MAIN_LR, rel_tol=1e-6)
                  and all(math.isclose(b, PROJECTOR_LR, rel_tol=1e-6) for b in base[1:])
                  and ((rates[0] > 0 and all(math.isclose(r / rates[0], ratio, rel_tol=1e-6) for r in rates[1:]))
                       or all(r == 0 for r in rates)))
            print("train_portable: step %d group learning rates %s (scheduler base %s)"
                  % (state.global_step, rates, base), flush=True)
            if not ok:
                raise RuntimeError("optimizer learning-rate groups are wrong: %s (expected 3 groups, projector = "
                                   "0.1 x LoRA rate); the optimizer-group fix did not take effect" % rates)
            return control

    class PortableBaselineTrainer(module.LLaVATrainer):
        def create_optimizer(self):
            result = super().create_optimizer()
            if not getattr(self, "_portable_pruned", False):
                inventory = optimizer_groups.prune_before_scheduler(
                    self.optimizer, self.model.named_parameters(), scheduler=self.lr_scheduler,
                    main_lr=MAIN_LR, projector_lr=PROJECTOR_LR, weight_decay=0)
                self._portable_pruned = True
                print("train_portable: optimizer groups pruned; kept original indices %s"
                      % inventory["kept_original_indices"], flush=True)
            return result

        def train(self, *args, **kwargs):
            base = self.model.get_base_model()
            original = base.prepare_inputs_labels_for_multimodal
            install_expanded_target_guard(self.model)
            self.add_callback(LearningRateGroupCheck())
            try:
                return super().train(*args, **kwargs)
            finally:
                base.prepare_inputs_labels_for_multimodal = original

    return PortableBaselineTrainer


def arguments(args):
    values = dict(
        model_name_or_path=args.model, version="v1", data_path=args.data_path, image_folder=args.image_folder,
        vision_tower=args.vision_tower, pretrain_mm_mlp_adapter=args.projector,
        mm_projector_type="mlp2x_gelu", mm_vision_select_layer=-2,
        mm_use_im_start_end=False, mm_use_im_patch_token=False, image_aspect_ratio="pad",
        group_by_modality_length=True, lora_enable=True, lora_r=128, lora_alpha=256,
        lora_dropout=.05, lora_bias="none", mm_projector_lr=PROJECTOR_LR,
        fp16=True, bf16=False, tf32=False, num_train_epochs=1,
        per_device_train_batch_size=args.per_device_batch, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, evaluation_strategy="no", save_strategy="no",
        learning_rate=MAIN_LR, weight_decay=0, warmup_ratio=.03, lr_scheduler_type="cosine",
        model_max_length=2048, gradient_checkpointing=True, dataloader_num_workers=0,
        lazy_preprocess=True, output_dir=args.output_dir,
        seed=args.seed, data_seed=args.seed, optim="adamw_torch", report_to="none",
        disable_tqdm=True, logging_steps=10, save_safetensors=True, deepspeed=str(ZERO3),
    )
    return [part for key, value in values.items() for part in ("--" + key, str(value))]


def main():
    args = parse()
    prepare_environment(args)
    sys.path[:0] = [str(Path(args.llava_root).resolve()), str(REPO / "src")]
    install_deterministic_cutlass_attention()
    import llava.train.train as module
    from experiments.root_native_campaign_v7_20261024 import optimizer_groups, structured_preprocess
    from experiments.root_native_campaign_v7_20261024.runtime_guards import install_expanded_target_guard

    def preprocess_v1(sources, tokenizer, has_image=False):
        return structured_preprocess.preprocess_structured(module, sources, tokenizer, has_image)

    module.preprocess_v1 = preprocess_v1
    module.LLaVATrainer = build_trainer_class(module, optimizer_groups, install_expanded_target_guard)
    records = len(json.loads(Path(args.data_path).read_bytes()))
    print("train_portable: K=%d records, expected optimizer steps %d" % (records, args.expected_steps), flush=True)
    sys.argv = ["train_portable"] + arguments(args)
    module.transformers.set_seed(args.seed)
    module.train(attn_implementation="eager")

    if int(os.environ.get("RANK", "0")) == 0:
        state = json.loads((Path(args.output_dir) / "trainer_state.json").read_bytes())
        steps = int(state["global_step"])
        required = ["adapter_model.safetensors", "adapter_config.json", "non_lora_trainables.bin", "config.json"]
        missing = [name for name in required if not (Path(args.output_dir) / name).is_file()]
        summary = dict(records=records, global_step=steps, expected_steps=args.expected_steps,
                       complete_one_epoch=steps == args.expected_steps, missing_files=missing, seed=args.seed,
                       per_device_batch=args.per_device_batch, grad_accum=args.grad_accum,
                       world_size=int(os.environ.get("WORLD_SIZE", "1")))
        (Path(args.output_dir) / "PORTABLE_RUN_SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
        print("train_portable: " + json.dumps(summary), flush=True)
        if steps != args.expected_steps or missing:
            sys.exit(3)


if __name__ == "__main__":
    main()
