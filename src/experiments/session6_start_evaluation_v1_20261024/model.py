"""Explicit Stage1 composition without vocabulary or frozen-base duplication."""
import copy
import hashlib
import importlib
from pathlib import Path
import sys

from experiments.session6_official_eval_v2_20261024 import common as c

PINS = {
    "vicuna": ("lmsys/vicuna-7b-v1.5", "3321f76e3f527bd14065daf69dad9344000a201d"),
    "clip336": ("openai/clip-vit-large-patch14-336", "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"),
    "stage1_projector": (
        "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
        "5414da88308e4287a29f2e9609256458afb0a981",
    ),
}


def verified_assets(reference, cache_root):
    receipt = c.checked(reference)
    c.require(receipt["status"] == "PASS_ASSETS", "Exact existing Stage1 asset receipt required")
    assets = receipt["assets"]
    for role, expected in PINS.items():
        asset = assets[role]
        c.require((asset["repository"], asset["revision"]) == expected and asset["files"],
                  "Stage1 asset revision or inventory differs: " + role)
        c.require(Path(asset["snapshot"]).resolve().is_relative_to(Path(cache_root).resolve()),
                  "Model snapshot outside approved shared cache")
        for ref in asset["files"]:
            c.require(c.desc(ref["path"]) == {key: ref[key] for key in ("path", "bytes", "sha256")},
                      "Stage1 model/processor bytes changed")
    return assets


def compose(assets, device="cuda:0", projector_path=None, config_path=None):
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(c.LLAVA))
    module = importlib.import_module("llava.model.language_model.llava_llama")
    c.require(Path(module.__file__).resolve() ==
              (c.LLAVA / "llava/model/language_model/llava_llama.py").resolve(),
              "Foreign native LLaVA implementation")
    config = module.LlavaConfig.from_pretrained(
        config_path or assets["stage1_projector"]["snapshot"], local_files_only=True)
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
    missing = loading["missing_keys"]
    c.require(not loading["unexpected_keys"] and not loading.get("mismatched_keys")
              and all("mm_projector" in key for key in missing),
              "Starting model has missing/foreign language weights")
    c.require(len(tokenizer) == model.get_input_embeddings().weight.shape[0],
              "Starting vocabulary changed; no synthetic token embeddings allowed")
    projector_path = Path(projector_path or
                          (Path(assets["stage1_projector"]["snapshot"]) / "mm_projector.bin"))
    weights = torch.load(projector_path, map_location="cpu")
    expected = {name for name in model.state_dict() if "mm_projector" in name}
    c.require(set(weights) == expected and all(bool(torch.isfinite(v).all()) for v in weights.values()),
              "Stage1 projector support or finite values differ")
    model.load_state_dict({key: value.to(torch.float16) for key, value in weights.items()}, strict=False)
    vision = model.get_vision_tower()
    if not vision.is_loaded:
        vision.load_model()
    vision.to(device=device, dtype=torch.float16)
    model.to(device=device, dtype=torch.float16).requires_grad_(False).eval()
    c.require(not any(parameter.requires_grad for parameter in model.parameters())
              and not model.training and not hasattr(model, "peft_config"),
              "START must be frozen/evaluation-only with no active adapter")
    return tokenizer, model, vision.image_processor


def parameter_digest(model):
    import torch
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(c.canonical([name, str(value.dtype), list(value.shape)]))
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def generation_configuration(model):
    value = model.generation_config.to_dict()
    c.require(isinstance(value, dict) and "num_beams" in value
              and type(value["num_beams"]) is int and value["num_beams"] > 0,
              "Actual model GenerationConfig with an integer beam count required")
    return copy.deepcopy(value)


def save_no_training_capsule(model, assets_reference, output):
    import torch
    generation = generation_configuration(model)
    output = Path(output)
    c.require(not output.exists(), "No-training capsule must own a fresh directory")
    output.mkdir(parents=True)
    model.config.save_pretrained(output)
    state = {name: parameter.detach().cpu().clone()
             for name, parameter in model.named_parameters() if "mm_projector" in name}
    c.require(state and all(bool(torch.isfinite(v).all()) for v in state.values()),
              "No finite current projector to export")
    torch.save(state, output / "mm_projector.bin")
    return c.seal(output / "ROUNDTRIP.json", dict(
        status="NO_TRAINING_PROJECTOR_CONFIG_CAPSULE_NOT_BASE_WEIGHT_COPY",
        assets=assets_reference, projector=c.desc(output / "mm_projector.bin"),
        config=c.desc(output / "config.json"), training_updates=0,
        model_generation_config=generation,
        model_generation_config_sha256=c.key(generation),
        all_base_and_vision_weights_reused_from_exact_shared_cache=True))
