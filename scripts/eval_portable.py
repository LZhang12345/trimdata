#!/usr/bin/env python3
"""Run an official LLaVA (c121) evaluation module with our inference settings.

Our evaluator loaded each checkpoint as: base Vicuna-7B-v1.5 in fp16 + trained projector (non_lora_trainables.bin) +
LoRA adapter kept **unmerged and active** (PEFT 0.7.1), Hugging Face **eager** attention, greedy decoding, batch 1.
The official `llava.model.builder.load_pretrained_model` instead merges the LoRA (`merge_and_unload`), uses the
transformers default attention, and only takes the LoRA branch when the model *name* contains "llava" and "lora"
(our folder names such as `sel1234-t17` do not). This wrapper patches only those three points and then runs the
requested official module unchanged, e.g.

    python scripts/eval_portable.py llava.eval.model_vqa_loader \
        --model-path ckpts/uniform/b10/sel1234-t17 --model-base /models/vicuna-7b-v1.5 \
        --question-file playground/data/eval/pope/llava_pope_test.jsonl \
        --image-folder playground/data/eval/pope/val2014 \
        --answers-file out/uniform-b10-sel1234-t17/pope.jsonl --temperature 0 --conv-mode vicuna_v1

Add `--merged` (before the module name) to evaluate with the official merged LoRA instead; report it separately.
The same patch applies to llava.eval.model_vqa, model_vqa_science and model_vqa_mmbench. Scoring uses the official
scripts unchanged (see AGENT_CONTRACT.md section 3). This wrapper was not itself executed on our cluster; the T0a
calibration qualifies it on your hardware.
"""
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDORED_LLAVA = (REPO / "src/collaboration/published-foundation-20261022/reference_sources/LLaVA/"
                  "c121f0432da27facab705978f83c4ada465e46fd")
LORA_NAME = "llava-v1.5-7b-lora"


def install_eager_config_loader(config_class):
    """Make eager explicit before Transformers 4.37.2 dispatches attention.

    The builder passes an existing config to from_pretrained. In 4.37.2 its
    default getter returns "eager" while the internal value is None; an eager
    keyword equal to that getter is ignored and automatic dispatch picks SDPA.
    Set the config's implementation explicitly when the builder loads it.
    """
    original = config_class.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        config = original(cls, *args, **kwargs)
        config._attn_implementation = "eager"
        return config

    config_class.from_pretrained = from_pretrained


def main():
    argv = sys.argv[1:]
    merged = False
    llava_root = VENDORED_LLAVA
    while argv and argv[0].startswith("--"):
        flag = argv.pop(0)
        if flag == "--merged":
            merged = True
        elif flag == "--llava-root":
            llava_root = Path(argv.pop(0))
        else:
            sys.exit("eval_portable: unknown wrapper flag %s (wrapper flags go before the module name)" % flag)
    if not argv or not argv[0].startswith("llava.eval."):
        sys.exit("usage: eval_portable.py [--merged] [--llava-root DIR] llava.eval.<module> <module args...>")
    module_name, module_args = argv[0], argv[1:]
    sys.path.insert(0, str(llava_root.resolve()))

    import peft
    import torch
    import llava.model.builder as builder
    from llava.model.language_model.llava_llama import LlavaConfig

    install_eager_config_loader(LlavaConfig)

    if torch.cuda.device_count() != 1:
        sys.exit("eval_portable: expose exactly one GPU (e.g. CUDA_VISIBLE_DEVICES=3); found %d"
                 % torch.cuda.device_count())

    if not merged:
        def keep_unmerged(self, *args, **kwargs):
            return self.get_base_model()
        peft.PeftModel.merge_and_unload = keep_unmerged

    original = builder.load_pretrained_model

    def load_pretrained_model(model_path, model_base, model_name, *args, **kwargs):
        if model_base is not None and not ("llava" in model_name.lower() and "lora" in model_name.lower()):
            model_name = LORA_NAME
        kwargs.setdefault("attn_implementation", "eager")
        tokenizer, model, image_processor, context_len = original(model_path, model_base, model_name, *args, **kwargs)
        model = model.to(device="cuda", dtype=torch.float16).eval()
        from transformers.models.llama.modeling_llama import LlamaAttention
        attention_layers = [layer.self_attn for layer in model.get_model().layers]
        if (model.config._attn_implementation != "eager"
                or not attention_layers or any(type(layer) is not LlamaAttention for layer in attention_layers)):
            raise RuntimeError(
                "expected instantiated eager LlamaAttention layers; config=%s, classes=%s"
                % (model.config._attn_implementation, sorted({type(layer).__name__ for layer in attention_layers})))
        if model_base is not None:
            from peft.tuners.lora import LoraLayer
            layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
            if not merged and (not layers or any(m.merged for m in layers)):
                raise RuntimeError("expected active, unmerged LoRA layers; found %d layers" % len(layers))
            print("eval_portable: %s LoRA (%d LoRA layers), attention=%s (%d LlamaAttention layers), dtype=fp16"
                  % ("merged" if merged else "unmerged", len(layers), model.config._attn_implementation,
                     len(attention_layers)), flush=True)
        return tokenizer, model, image_processor, context_len

    builder.load_pretrained_model = load_pretrained_model
    sys.argv = [module_name] + module_args
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
