"""Versioned low-memory projector materialization; executed V1 stays immutable."""
import importlib
from pathlib import Path
import sys

from experiments.session6_start_evaluation_v1_20261024.model import (
    PINS, generation_configuration, parameter_digest, save_no_training_capsule, verified_assets,
)
from experiments.session6_official_eval_v2_20261024 import common as c


def materialize_projector(model, weights):
    import torch
    state = model.state_dict()
    expected = {name for name in state if "mm_projector" in name}
    c.require(len(expected) == 4 and set(weights) == expected
              and all(isinstance(value, torch.Tensor) and not value.is_meta
                      and value.shape == state[name].shape and bool(torch.isfinite(value).all())
                      for name, value in weights.items()),
              "Exact four finite concrete Stage1 projector tensors required")
    concrete = {name: value.detach().to(device="cpu", dtype=torch.float16)
                for name, value in weights.items()}
    c.require(all(bool(torch.isfinite(value).all()) for value in concrete.values()),
              "Stage1 projector overflowed during the frozen FP16 cast")
    loaded = model.load_state_dict(concrete, strict=False, assign=True)
    c.require(not loaded.unexpected_keys and set(loaded.missing_keys) == set(state) - expected,
              "Projector assignment changed unexpected native state")
    actual = dict(model.named_parameters())
    c.require(not any(value.is_meta for value in actual.values())
              and all(actual[name].dtype == torch.float16 and actual[name].device.type == "cpu"
                      and torch.equal(actual[name].detach(), concrete[name]) for name in expected),
              "Projector assignment did not materialize exact FP16 checkpoint values")


def explicit_eager_config(config):
    config._attn_implementation = "eager"
    c.require(config._attn_implementation_internal == "eager",
              "Actual explicit eager configuration binding failed")
    return config


def compose(assets, device="cuda:0", projector_path=None, config_path=None):
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(c.LLAVA))
    module = importlib.import_module("llava.model.language_model.llava_llama")
    c.require(Path(module.__file__).resolve() ==
              (c.LLAVA / "llava/model/language_model/llava_llama.py").resolve(),
              "Foreign native LLaVA implementation")
    config = explicit_eager_config(module.LlavaConfig.from_pretrained(
        config_path or assets["stage1_projector"]["snapshot"], local_files_only=True))
    config.mm_vision_tower = assets["clip336"]["snapshot"]
    config.image_aspect_ratio = "pad"
    config.mm_use_im_start_end = False
    config.mm_use_im_patch_token = False
    tokenizer = AutoTokenizer.from_pretrained(
        assets["vicuna"]["snapshot"], use_fast=False, model_max_length=2048,
        padding_side="right", local_files_only=True)
    tokenizer.pad_token = tokenizer.unk_token
    model, loading = module.LlavaLlamaForCausalLM.from_pretrained(
        assets["vicuna"]["snapshot"], config=config, torch_dtype=torch.float16,
        attn_implementation="eager", low_cpu_mem_usage=True, local_files_only=True,
        output_loading_info=True)
    c.require(not loading["unexpected_keys"] and not loading.get("mismatched_keys")
              and all("mm_projector" in key for key in loading["missing_keys"]),
              "Starting model has missing/foreign language weights")
    c.require(len(tokenizer) == model.get_input_embeddings().weight.shape[0],
              "Starting vocabulary changed; no synthetic token embeddings allowed")
    projector_path = Path(projector_path or
                          (Path(assets["stage1_projector"]["snapshot"]) / "mm_projector.bin"))
    materialize_projector(model, torch.load(projector_path, map_location="cpu"))
    vision = model.get_vision_tower()
    if not vision.is_loaded:
        vision.load_model()
    vision.to(device=device, dtype=torch.float16)
    model.to(device=device, dtype=torch.float16).requires_grad_(False).eval()
    c.require(not any(parameter.requires_grad for parameter in model.parameters())
              and not model.training and not hasattr(model, "peft_config"),
              "START must be frozen/evaluation-only with no active adapter")
    return tokenizer, model, vision.image_processor
