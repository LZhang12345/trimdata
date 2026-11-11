"""LESS-approximation gradient features (Tool2): per-record LoRA gradients of a warm-up adapter, count-sketched.

Derived from helper_adapter_eval_v3 model.py (same frozen loader, exact adapter checks, vicuna_v1 first-turn encoding,
answer+EOS loss). New: LoRA tensors get gradients (projector and base stay frozen; no optimizer, no updates), one
backward per record, and a fixed count-sketch (seeded index+sign per LoRA coordinate) to SKETCH_DIM dimensions;
each row is L2-normalized (cosine geometry as in LESS). Chunked .npy partials make a requeued job resume.
v2 (2026-11-05): v1 shards ran out of memory in the forward pass on long first turns (42.8 GiB allocated); v2 uses
the native trainer's non-reentrant gradient checkpointing (backbone only; LoRA dropout stays off, so features are
deterministic) and requires the native 2048-token model limit (tokenizer_model_max_length). No other change.
Deviations from published LESS (disclosed in the design receipt): reused 10% random warm-up adapter, one checkpoint,
SGD (not Adam-preconditioned) gradients, first QA turn only, count-sketch instead of TRAK's projector.
"""

import argparse
import gc
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path("/home/luyangz/data_selection_multimodal/claude_coincide_eval_src_v3_20261030")
NATIVE = ROOT / ("collaboration/published-foundation-20261022/reference_sources/LLaVA/"
                 "c121f0432da27facab705978f83c4ada465e46fd")
OUTPUT_ROOT = Path("/data/hf_cache/project_features/DS-MultiModal/helper-tool2-less-approx-v2-20261105")
SKETCH_DIM = 8192
SKETCH_SEED = 20261105
LOSS_SCALE = 256.0
CHUNK = 256
BLANK_RGB = (122, 116, 104)
GQA_SUFFIX = "\nAnswer the question using a single word or phrase."


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def portable_key(name):
    return name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")


def first_turn(record):
    turns = record["conversations"]
    require(len(turns) >= 2 and turns[0]["from"] == "human" and turns[1]["from"] == "gpt",
            "First human/assistant turn required")
    question = turns[0]["value"]
    if record.get("image") is not None and "<image>" not in question:
        question = "<image>\n" + question
    return question, turns[1]["value"]


def gqa_correct(prediction, answer):
    return prediction.strip().rstrip(".").lower() == answer.strip().lower()


def letter_correct(prediction, answer):
    text = prediction.strip()
    return bool(text) and text[0] == answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_bytes())
    job = os.environ.get("SLURM_JOB_ID", "")
    require(os.getuid() == 2708899 and job.isdecimal() and config["job_id"] == job
            and config["role"] == "OWNED_LESS_APPROX_GRADIENT_FEATURES_NO_UPDATES"
            and Path(config["output"]) == OUTPUT_ROOT / ("job-" + job), "Actual owned evaluator required")
    require(Path(__file__).absolute() == Path(config["model_source"]["path"])
            and digest(__file__) == config["model_source"]["sha256"], "Evaluator source changed")
    for relative, expected in config["eval_sources"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                and digest(ROOT / relative) == expected, "Frozen evaluator source changed: " + relative)
    assets = config["assets"]
    for role in ("vicuna", "clip336", "stage1_projector"):
        for reference in assets[role]["files"]:
            require(Path(reference["path"]).stat().st_size == reference["bytes"]
                    and digest(reference["path"]) == reference["sha256"], "Cached frozen model changed")
    datasets = {}
    for item in config["datasets"]:
        raw = Path(item["path"]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == item["sha256"], "Sealed dataset changed: " + item["name"])
        rows = json.loads(raw)
        require(len(rows) == item["count"], "Dataset count differs: " + item["name"])
        datasets[item["name"]] = (item, rows)
    require(not any(name == "llava" or name.startswith("llava.") for name in sys.modules),
            "Fresh native inference import required")
    sys.path[:0] = [str(ROOT), str(NATIVE)]
    import torch
    import numpy as np
    import peft
    from PIL import Image
    from safetensors.torch import load_file
    require(torch.__version__ == "2.1.2+cu121" and torch.cuda.device_count() == 1
            and "L40S" in torch.cuda.get_device_name(), "Exact inference device/build required")
    require(metadata.version("peft") == "0.7.1" and metadata.version("transformers") == "4.37.2",
            "Native inference package versions differ")
    from experiments.root_babel_start_evaluation_v2_20261025 import worker as official
    runtime_reference = config["official_runtime_reference"]
    runtime_raw = Path(runtime_reference["path"]).read_bytes()
    require(len(runtime_raw) == runtime_reference["bytes"]
            and hashlib.sha256(runtime_raw).hexdigest() == runtime_reference["sha256"],
            "Frozen official runtime configuration changed")
    output = Path(config["output"])
    attestation = official.runtime(json.loads(runtime_raw), output)
    from experiments.session6_start_model_qualification_v4_eager_20261024 import model as stage1
    from experiments.copilot7_coincide_native_eval_v2_20261028 import model as matched
    from llava import conversation
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token
    blank_path = output / "blank_mean_rgb_336.png"
    if not blank_path.is_file():
        temporary = output / "blank_mean_rgb_336.png.partial"
        Image.new("RGB", (336, 336), BLANK_RGB).save(temporary, format="PNG")
        os.replace(temporary, blank_path)

    def load(spec):
        if spec["adapter"] is None:
            return stage1.compose(assets), {"adapter": None}
        training = Path(spec["adapter"]["training_dir"])
        for name, reference in spec["adapter"]["output_files"].items():
            require("/" not in name and (training / name).stat().st_size == reference["bytes"]
                    and digest(training / name) == reference["sha256"], "Probe adapter artifact changed")
        names = [n for group in spec["adapter"]["optimizer_groups"] for n in group["names"]]
        require(len(names) == len(set(names)) == 452, "Exact native LoRA/projector support required")
        tokenizer, base, processor = stage1.compose(assets, config_path=training)
        adapter = json.loads((training / "adapter_config.json").read_bytes())
        require(adapter["r"] == 128 and adapter["lora_alpha"] == 256 and adapter["lora_dropout"] == .05,
                "Native saved LoRA configuration differs")
        model, injection = matched.attach_native_adapter(
            base, adapter, {"trainable": [{"name": name} for name in names]},
            lambda value: peft.PeftModel.from_pretrained(
                value, str(training), adapter_name="default", is_trainable=False, local_files_only=True))
        non_lora = torch.load(training / "non_lora_trainables.bin", map_location="cpu")
        incompatible = model.load_state_dict(non_lora, strict=False)
        require(not incompatible.unexpected_keys, "Projector loading touched unexpected state")
        model.to(device="cuda:0", dtype=torch.float16).eval().requires_grad_(False)
        lora = load_file(str(training / "adapter_model.safetensors"), device="cpu")
        expected = {name: non_lora[name] if name in non_lora else lora[portable_key(name)] for name in names}
        parameters = dict(model.named_parameters())
        for name, tensor in expected.items():
            require(torch.equal(parameters[name].detach().cpu(), tensor.to(torch.float16)),
                    "Loaded trainable tensor is not exact: " + name)
        layers = [module for module in model.modules() if hasattr(module, "lora_A")]
        require(len(layers) == 224 and all(not layer.merged and not layer.disable_adapters for layer in layers),
                "Adapters must remain active and unmerged")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(".lora_" in name)
        return (tokenizer, model, processor), {"adapter": spec["name"], "injection": injection,
                                               "exact_trainables": len(expected)}

    def encode(tokenizer, question, answer):
        conv = conversation.conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        full = conv.copy()
        full.messages[-1][1] = answer
        prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        full_ids = tokenizer_image_token(full.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        prefix = int(prompt_ids.shape[0])
        require(full_ids.shape[0] > prefix and torch.equal(full_ids[:prefix], prompt_ids),
                "Prompt tokens must be an exact prefix of the answered sequence")
        labels = full_ids.clone()
        labels[:prefix] = -100
        return prompt_ids, full_ids, labels

    def pixels(processor, model, record, blank):
        image = record.get("image")
        if image is None:
            return None
        path = blank_path if blank else (Path(config["image_root"]) / image if not image.startswith("/") else Path(image))
        with Image.open(path) as handle:
            rgb = handle.convert("RGB")
        return process_images([rgb], processor, model.config).to("cuda:0", dtype=torch.float16)

    def checkpointed(model):
        require(getattr(model.config, "tokenizer_model_max_length", None) == 2048,
                "Native 2048-token model limit required (model truncates expanded multimodal inputs)")
        model.config.use_cache = False
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbones = [m for m in base.modules() if type(m).__name__ == "LlavaLlamaModel"]
        require(len(backbones) == 1 and backbones[0].gradient_checkpointing is True,
                "Exactly one checkpointed LLaVA backbone required")
        backbones[0].training = True  # activates checkpointing only; submodules (LoRA dropout) remain in eval mode
        require(sum(1 for m in model.modules() if m.training) == 1, "Only the backbone flag may be in training mode")
        return dict(gradient_checkpointing="non-reentrant, backbone only", tokenizer_model_max_length=2048)

    def save_partial(path, state):
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(state))
        os.replace(temporary, path)

    def sketch_plan(model):
        names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        require(len(names) == 448 and all(".lora_" in n for n in names), "Exactly the 448 native LoRA tensors required")
        params = dict(model.named_parameters())
        plan = []
        for position, name in enumerate(names):
            generator = torch.Generator(device="cuda:0").manual_seed(SKETCH_SEED * 1000 + position)
            numel = params[name].numel()
            index = torch.randint(0, SKETCH_DIM, (numel,), generator=generator, device="cuda:0")
            sign = (torch.randint(0, 2, (numel,), generator=generator, device="cuda:0") * 2 - 1).to(torch.float32)
            plan.append((params[name], index, sign))
        return plan, hashlib.sha256(json.dumps(names).encode()).hexdigest()

    def gradient(model, plan, full_ids, labels, image):
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=full_ids[None].to("cuda:0"), labels=labels[None].to("cuda:0"), images=image).loss
        value = float(loss.float().item())
        require(math.isfinite(value), "Nonfinite loss")
        (loss.float() * LOSS_SCALE).backward()
        vector = torch.zeros(SKETCH_DIM, dtype=torch.float32, device="cuda:0")
        for parameter, index, sign in plan:
            require(parameter.grad is not None, "Every LoRA tensor must receive a gradient")
            grad = parameter.grad.detach().reshape(-1).to(torch.float32)
            require(bool(torch.isfinite(grad).all()), "Nonfinite gradient")
            vector.index_add_(0, index, grad * sign)
        norm = float(vector.norm().item())
        require(math.isfinite(norm) and norm > 0, "Degenerate gradient sketch")
        return (vector / norm).to(torch.float16).cpu().numpy(), value, norm

    results = {}
    spec = config["models"][0]
    require(len(config["models"]) == 1 and spec["adapter"] is not None, "Exactly one warm-up adapter required")
    final = output / "RESULT.json"
    if final.is_file():
        print(json.dumps({"job_id": job, "status": "ALREADY_COMPLETE"}), flush=True)
        return
    started = time.monotonic()
    (tokenizer, model, processor), proof = load(spec)
    proof["memory"] = checkpointed(model)
    plan, names_sha = sketch_plan(model)
    for name, (item, rows) in datasets.items():
        done_path = output / ("FEATURES-" + name + ".npy")
        if done_path.is_file():
            results[name] = json.loads((output / ("FEATURES-" + name + ".json")).read_bytes())
            continue
        chunks = sorted(output.glob("CHUNK-" + name + "-*.npz"))
        start_row = len(chunks) * CHUNK
        for begin in range(start_row, len(rows), CHUNK):
            block, losses, norms = [], [], []
            for record in rows[begin:begin + CHUNK]:
                question, answer = first_turn(record)
                _, full_ids, labels = encode(tokenizer, question, answer)
                vector, value, norm = gradient(model, plan, full_ids, labels, pixels(processor, model, record, False))
                block.append(vector)
                losses.append(value)
                norms.append(norm)
            temporary = output / ("CHUNK-" + name + "-%07d.partial.npz" % begin)
            np.savez(temporary, features=np.stack(block), losses=np.array(losses, dtype=np.float32),
                     norms=np.array(norms, dtype=np.float32))
            os.replace(temporary, output / ("CHUNK-" + name + "-%07d.npz" % begin))
            print(json.dumps({"chunk": name, "begin": begin, "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        chunks = sorted(output.glob("CHUNK-" + name + "-*.npz"))
        loaded = [np.load(path) for path in chunks]
        features = np.concatenate([x["features"] for x in loaded])
        losses = np.concatenate([x["losses"] for x in loaded])
        norms = np.concatenate([x["norms"] for x in loaded])
        require(features.shape == (len(rows), SKETCH_DIM), "Feature rows must cover the dataset exactly")
        temporary = output / ("FEATURES-" + name + ".partial.npy")
        with open(temporary, "wb") as stream:
            np.save(stream, features)
        os.replace(temporary, done_path)
        meta = dict(name=name, count=len(rows), record_ids=[r.get("id") for r in rows], losses=losses.tolist(),
                    raw_gradient_norms=norms.tolist(), features_sha256=digest(done_path), sketch_dim=SKETCH_DIM)
        (output / ("FEATURES-" + name + ".json")).write_text(json.dumps(meta))
        for path in chunks:
            path.unlink()
        results[name] = meta
    report = dict(status="ACTUAL_LESS_APPROX_GRADIENT_FEATURES", job_id=job, model=proof, lora_names_sha256=names_sha,
                  sketch=dict(dim=SKETCH_DIM, seed=SKETCH_SEED, kind="count-sketch index+sign per LoRA coordinate",
                              loss_scale=LOSS_SCALE, normalized=True),
                  datasets={k: {kk: v[kk] for kk in ("count", "features_sha256")} for k, v in results.items()},
                  seconds=time.monotonic() - started, training_updates=0, runtime_attestation=attestation,
                  loss_definition="single-turn first QA under vicuna_v1; answer+EOS tokens only; eager FP16")
    final.write_text(json.dumps(report))
    print(json.dumps({k: report[k] for k in ("status", "job_id", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
