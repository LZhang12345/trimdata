"""GPU worker: Tool1 BENCH4 generation (MMMU validation, LLaVA-Bench-in-the-Wild, MM-Vet, MMBench-EN dev).

Reuses the reviewed Tool1 BENCH6 worker (helper_tool1_bench6_eval_v1_20261103/model6.py): its custody, model
composition, eager-attention, parameter-parity, storage and atomic-publication helpers are copied byte-identically
(same frozen claude_coincide_eval_src_v3 closure, LLaVA c121 sources, read-only accepted checkpoints or the untouched
START model, active unmerged PEFT adapter with the audited positional bridge). New here: per-bank loops that repeat
the LLaVA c121 eval-script iteration (model_vqa for MM-Vet/LLaVA-Wild, model_vqa_mmbench for MMBench, the official
MMMU prompt with the c121 conversation for MMMU; see banks4.py) over the label-free BENCH4 inventory rosters, with
per-QA encoded-byte and RGB pixel proofs, the frozen session6 per-semantic-QA SQLite Journal (one resume-stable
journal per (bank, cell)), and atomic bank-level PREDICTIONS/EXPORT_SUMMARY/COMPLETE publication. No label file is
opened; a bank the allocation deadline prevents from finishing is left for the explicit missing-only successor.

v3: the BENCH4 v1 worker for CELLS_V3 Tool1-trained baseline students (custody3.py; horizon per K, final
training/ adapter named by the producer RESULT.json) with its own output root and role; loading, parity,
generation, durability and export are unchanged.
"""

import argparse
import fcntl
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
import time
from types import ModuleType, SimpleNamespace
import uuid


ROOT = Path("/home/luyangz/data_selection_multimodal/claude_coincide_eval_src_v3_20261030")
NATIVE = ROOT / ("collaboration/published-foundation-20261022/reference_sources/LLaVA/"
                 "c121f0432da27facab705978f83c4ada465e46fd")
HOME = Path("/home/luyangz/data_selection_multimodal")
OUTPUT_ROOT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-bench4-eval-v3-20261104")
CHECKPOINT_ROOT = Path("/data/hf_cache/project_checkpoints/DS-MultiModal")
UID = 2708899
ROLE = "TOOL1_BENCH4_CELLS_V3_TRAINED_BASELINE_EVALUATION"
LORA_TENSORS, PROJECTOR_TENSORS, LORA_LAYERS = 448, 4, 224
GENERATION_SOURCES = ("model4.py", "banks4.py", "CELLS_V3.json", "custody3.py")
INPUT_STATUS = "COMPLETE_BENCH4_INPUT_INVENTORY_V1"
COMPLETE_STATUS = "COMPLETE_ALL_PLANNED_BANKS"
PROGRESS_STATUS = "ACCEPTED_QA_PROGRESS_REQUIRES_REVIEWED_MISSING_ONLY_SUCCESSOR"
BANK_COMPLETE = "COMPLETE_FULL_BANK_PREDICTIONS"
BANK_PROGRESS = "ACCEPTED_QA_PROGRESS"
BANK_NOT_STARTED = "NOT_STARTED_ALLOCATION_DEADLINE"
BANK_START_MARGIN_SECONDS = 300
PORTABLE = re.compile(r"base_model\.model\.model\.layers\.\d+\."
                      r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
                      r"\.lora_[AB]\.weight")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_exact(ref, limit=64 << 20):
    """Bounded, nonblocking, no-follow read of a small pinned file; bytes are hashed once."""
    require(isinstance(ref, dict) and type(ref.get("bytes")) is int and 0 <= ref["bytes"] <= limit,
            "Small exact descriptor required")
    fd = os.open(ref["path"], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size == ref["bytes"], "Pinned file type/size differs")
        chunks, remaining = [], ref["bytes"] + 1
        while remaining:
            block = os.read(fd, min(remaining, 8 << 20))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw, after = b"".join(chunks), os.fstat(fd)
    finally:
        os.close(fd)
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"]
            and all(getattr(before, k) == getattr(after, k) for k in keys), "Pinned file differs: " + ref["path"])
    return raw


def verify_large(ref):
    """Streamed SHA256 with stable FD identity for bulky inference files."""
    fd = os.open(ref["path"], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size == ref["bytes"], "Inference file type/size differs")
        result, seen = hashlib.sha256(), 0
        while True:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            result.update(block)
            seen += len(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(seen == ref["bytes"] and result.hexdigest() == ref["sha256"]
            and all(getattr(before, k) == getattr(after, k) for k in keys), "Inference file differs: " + ref["path"])
    return dict(ref)


def load_bound(ref, name):
    module = ModuleType(name)
    module.__file__ = ref["path"]
    exec(compile(read_exact(ref), ref["path"], "exec", dont_inherit=True), vars(module))
    return module


def validate_cell(cells, cell, custody, custody3):
    """Registered CELLS_V3 cell; custody = plan-bound producer records plus the cells-v3 inventory's late-bound
    producer RESULT/file descriptors and the parameter digest of the model that inventory actually loaded."""
    spec = custody3.worker_custody(cells, cell, custody)
    require(spec["kind"] == "student", "CELLS_V3 registers trained students only")
    return spec


def producer_inputs(custody, custody3):
    """Final training/ inference files named by the producer's own RESULT.json; they must equal the inventory's.

    The horizon follows the cell's K ((K + 3) // 4 // 32 updates); the trainer retired its accepted
    checkpoint-<step> directories after success, so none is read."""
    result = json.loads(read_exact(custody["producer_result"]))
    files = custody3.producer_files(result, custody["producer"])
    require(files == custody["inference_files"], "Producer RESULT inference files differ from the inventory custody")
    return Path(custody["producer"]["training_dir"]), files


def portable_key(name):
    return name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")


def named_key(portable):
    require(PORTABLE.fullmatch(portable) is not None, "Unexpected saved LoRA tensor: " + portable)
    return re.sub(r"\.lora_([AB])\.weight$", r".lora_\1.default.weight", portable)


def trainable_names(lora_keys, projector_keys):
    names = [named_key(key) for key in sorted(lora_keys)]
    projector = sorted(projector_keys)
    require(len(names) == len(set(names)) == LORA_TENSORS, "Exact native language LoRA support required")
    require(len(projector) == PROJECTOR_TENSORS and all("mm_projector" in name for name in projector),
            "Exact four trained projector tensors required")
    return names + projector


def eager_attention(model, attention_type, original_forward):
    layers = [module for module in model.modules() if isinstance(module, attention_type)]
    require(len(layers) == 32
            and all(type(module) is attention_type
                    and module.forward.__func__ is original_forward for module in layers)
            and original_forward.__module__ == "transformers.models.llama.modeling_llama"
            and model.config._attn_implementation == "eager"
            and model.get_vision_tower().vision_tower.config._attn_implementation == "eager",
            "Matched original HF eager attention required; training monkey patch is not an inference backend")
    return dict(language_layers=len(layers), language_class=attention_type.__name__,
                language_forward_module=original_forward.__module__,
                language_attention="eager", vision_attention="eager", training_xformers_patch_installed=False)


def private_child(parent, name):
    require(isinstance(name, str) and name and "/" not in name and name not in (".", ".."), "Literal name required")
    fd = os.open(str(parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=fd)
        except FileExistsError:
            pass
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
    finally:
        os.close(fd)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Owned private output directory required: " + str(Path(parent) / name))
    return Path(parent) / name


def bank_directory(protocol_name, cell, root=OUTPUT_ROOT):
    return private_child(private_child(root, protocol_name), cell)


def check_storage(directory, envelope, statvfs=os.statvfs):
    value = statvfs(str(directory))
    available = value.f_bavail * value.f_frsize
    require(available >= 2 * envelope and value.f_favail >= 10000,
            "Current shared filesystem cannot hold twice this bank's measured output envelope")
    return dict(available_bytes=available, available_inodes=value.f_favail, envelope_bytes=envelope,
                not_a_collective_capacity_grant=True)


def publish_same(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return publish_bytes(path, raw)


def publish_bytes(path, raw):
    """Atomic create-once publication that a kill (e.g. preemption) can never leave half-written.

    Bytes go to an O_EXCL private temp in the same directory and are fsynced; link(2) then publishes
    the final name without ever replacing an existing one (rename(2) would), the temp name is
    unlinked and the directory fsynced. An existing name must already hold the identical bytes.
    """
    path = Path(path)
    ref = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    directory = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        temp = ".%s.partial-%s" % (path.name, uuid.uuid4().hex)
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                     dir_fd=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp, path.name, src_dir_fd=directory, dst_dir_fd=directory)
            except FileExistsError:
                require(read_exact(ref, limit=1 << 30) == raw, "Existing immutable output differs")
        finally:
            os.unlink(temp, dir_fd=directory)
        os.fsync(directory)
    finally:
        os.close(directory)
    return ref


def compose_student(torch, peft, load_file, stage1, matched, assets, checkpoint_dir, files):
    tokenizer, base, processor = stage1.compose(assets, config_path=checkpoint_dir)
    lora = load_file(files["adapter_model.safetensors"]["path"], device="cpu")
    non_lora = torch.load(files["non_lora_trainables.bin"]["path"], map_location="cpu")
    names = trainable_names(lora, non_lora)
    adapter_raw = Path(checkpoint_dir, "adapter_config.json").read_bytes()
    if "adapter_config.json" in files:
        require(hashlib.sha256(adapter_raw).hexdigest() == files["adapter_config.json"]["sha256"],
                "Accepted adapter configuration changed")
    adapter = json.loads(adapter_raw)
    require(adapter["r"] == 128 and adapter["lora_alpha"] == 256 and adapter["lora_dropout"] == .05
            and adapter["bias"] == "none" and adapter.get("task_type") == "CAUSAL_LM",
            "Native saved LoRA configuration differs")
    model, injection = matched.attach_native_adapter(
        base, adapter, {"trainable": [{"name": name} for name in names]},
        lambda value: peft.PeftModel.from_pretrained(
            value, str(checkpoint_dir), adapter_name="default", is_trainable=False, local_files_only=True))
    current = model.state_dict()
    require(set(non_lora) <= set(current), "Trained projector names differ from the loaded model")
    incompatible = model.load_state_dict(non_lora, strict=False)
    require(not incompatible.unexpected_keys and set(incompatible.missing_keys) == set(current) - set(non_lora),
            "Projector loading touched unexpected state")
    model.to(device="cuda:0", dtype=torch.float16).eval().requires_grad_(False)
    parameters = dict(model.named_parameters())
    for name in names:
        tensor = non_lora[name] if name in non_lora else lora[portable_key(name)]
        require(tensor.dtype == parameters[name].dtype == torch.float16 and tensor.shape == parameters[name].shape
                and bool(torch.isfinite(tensor).all()) and torch.equal(parameters[name].detach().cpu(), tensor),
                "Loaded trainable tensor is not exact: " + name)
    layers = [module for module in model.modules() if hasattr(module, "lora_A")]
    require(len(layers) == LORA_LAYERS and all(not layer.merged and not layer.disable_adapters for layer in layers),
            "Inference adapters must remain active and unmerged")
    require(model.active_adapters == ["default"] and model.peft_config["default"].inference_mode is True,
            "Exact active inference adapter required; never merge or relabel START")
    require(not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()),
            "Unqualified mutable normalization buffers")
    persistent = sorted(set(model.state_dict()) - set(parameters))
    require(not persistent, "Portable-only output requires separate proof for persistent buffers")
    del lora, non_lora
    gc.collect()
    return (tokenizer, model, processor), dict(native_injection=injection, loaded_trainable_tensors=len(names),
                                               active_unmerged_layers=len(layers), persistent_buffers=persistent)


def compose_start(torch, stage1, assets):
    tokenizer, model, processor = stage1.compose(assets)
    require(not hasattr(model, "peft_config") and not model.training
            and all(not p.requires_grad and p.dtype == torch.float16 and p.device.type == "cuda"
                    for p in model.parameters()), "Untouched frozen FP16 START required")
    require(not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules())
            and not set(model.state_dict()) - set(dict(model.named_parameters())),
            "START must have no mutable normalization or persistent buffers")
    return (tokenizer, model, processor), dict(adapter_active=False, training_updates=0)


def attempt_directory(config, cell, job, root=OUTPUT_ROOT):
    attempt = root / "cells" / cell / "attempts" / ("job-" + job)
    require(Path(config["output"]) == attempt and attempt.is_dir() and not attempt.is_symlink()
            and not (attempt / "RESULT.json").exists(), "Fresh controller-created attempt directory required")
    return attempt


def load_questions(config, banks, name):
    """Label-free roster of one bank from the completed BENCH4 input inventory named by the plan."""
    inputs = config["inputs"]
    result = json.loads(read_exact(inputs["inventory_result"], limit=2 << 20))
    require(result.get("status") == INPUT_STATUS and result["banks"][name]["status"] == "PASS"
            and result["banks"][name]["questions"] == inputs["banks"][name]["questions"]
            and result["banks"][name]["items"] == inputs["banks"][name]["items"],
            "Completed BENCH4 input inventory must name this exact roster")
    raw = read_exact(inputs["banks"][name]["questions"], limit=512 << 20)
    require(raw.endswith(b"\n"), "Complete newline-terminated roster required")
    rows = [json.loads(line) for line in raw[:-1].split(b"\n")]
    summary = banks.validate_questions(name, rows, expected_items=inputs["banks"][name]["items"])
    return rows, summary, inputs["banks"][name]["questions"]


PAD_SAFE_MODES = ("RGB", "RGBA", "P", "CMYK")


def open_image(row, mode, Image):
    """Exact inventory bytes, PIL object as the native script builds it, and an RGB pixel proof.

    model_vqa (and the MMMU adapter) convert to RGB; model_vqa_mmbench feeds load_image_from_base64 unchanged.
    c121 expand2square cannot pad a non-square L/LA/1/I image with its RGB mean colour (PIL TypeError), so only
    that case is converted to RGB and disclosed in the per-QA audit.
    """
    import io
    image = row["image"]
    raw = read_exact(dict(path=image["path"], bytes=image["bytes"], sha256=image["sha256"]), limit=64 << 20)
    native = Image.open(io.BytesIO(raw))
    rgb = native.convert("RGB")
    rgb.load()
    require(hashlib.sha256(rgb.tobytes()).hexdigest() == image["rgb_sha256"]
            and [rgb.width, rgb.height] == [image["width"], image["height"]],
            "Decoded RGB pixels differ from the inventory proof")
    if mode == "convert_rgb":
        native.close()
        return rgb, None
    require(mode == "native_base64_no_convert", "Unknown image mode")
    if native.mode not in PAD_SAFE_MODES and native.width != native.height:
        adaptation = "NONSQUARE_%s_CONVERTED_TO_RGB_C121_PAD_WOULD_RAISE" % native.mode
        native.close()
        return rgb, adaptation
    rgb.close()
    native.load()
    return native, None


def tensor_sha(tensor):
    value = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(str(value.dtype).encode() + repr(tuple(value.shape)).encode()
                          + value.numpy().tobytes()).hexdigest()


def generate_one(ctx, protocol, row):
    """One LLaVA c121 eval-script iteration (model_vqa / model_vqa_mmbench / MMMU adapter), audited."""
    llava = ctx.llava
    tokenizer, model, processor = ctx.loaded
    started = time.monotonic()
    qs = ctx.banks.llava_query(row["cur_prompt"], bool(model.config.mm_use_im_start_end), protocol["suffix"])
    conv = llava.conv_templates[protocol["decoding"]["conv_mode"]].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = llava.tokenizer_image_token(prompt, tokenizer, llava.IMAGE_TOKEN_INDEX,
                                            return_tensors="pt").unsqueeze(0).cuda()
    image, adaptation = open_image(row, protocol["image_mode"], ctx.Image)
    try:
        size = list(image.size)
        image_tensor = llava.process_images([image], processor, model.config)[0]
        pixels = image_tensor.unsqueeze(0).half().cuda()
        decoding = protocol["decoding"]
        output_ids = model.generate(
            input_ids,
            images=pixels,
            image_sizes=[image.size],
            do_sample=True if decoding["temperature"] > 0 else False,
            temperature=decoding["temperature"],
            top_p=decoding["top_p"],
            num_beams=decoding["num_beams"],
            max_new_tokens=decoding["max_new_tokens"],
            use_cache=True)
    finally:
        image.close()
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    new_tokens = int(output_ids.shape[-1])
    audit = dict(
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), input_ids_sha256=tensor_sha(input_ids),
        input_tokens=int(input_ids.shape[-1]), pixel_values_sha256=tensor_sha(pixels), image_size=size,
        image_encoded_sha256=row["image"]["sha256"], image_rgb_sha256=row["image"]["rgb_sha256"],
        image_mode_adaptation=adaptation,
        output_tokens=new_tokens, reached_max_new_tokens=new_tokens >= decoding["max_new_tokens"],
        generate_kwargs=dict(do_sample=decoding["temperature"] > 0, temperature=decoding["temperature"],
                             top_p=decoding["top_p"], num_beams=decoding["num_beams"],
                             max_new_tokens=decoding["max_new_tokens"], use_cache=True, image_sizes=True),
        elapsed_seconds=round(time.monotonic() - started, 4))
    return outputs, audit


def run_bank(ctx, name):
    """One bank's stable journal: generate only missing QAs, publish a complete export atomically."""
    banks = ctx.banks
    protocol = banks.protocol(name)
    rows, summary, questions_ref = load_questions(ctx.config, banks, name)
    bank_dir = ctx.bank_dirs[name]
    function = banks.journal_function(name, ctx.cell, ctx.model_identity, summary, ctx.sources_sha256, questions_ref)
    function_ref = publish_same(bank_dir / "FUNCTION.json", function)
    function_sha = ctx.jc.key(function)
    current = dict(ctx.actor, pid=os.getpid(), node=socket.gethostname().split(".")[0],
                   process_nonce=uuid.uuid4().hex, phase="TOOL1_BENCH4_" + name.upper())
    require(current["job_id"] == ctx.job, "Own actual child and parent allocation must match")
    journal = ctx.Journal(bank_dir / "journal", function, rows)
    seconds, stopped = [], False
    try:
        before = len(journal.present)
        with ctx.torch.inference_mode():
            for row in journal.missing():
                if time.monotonic() >= ctx.deadline:
                    stopped = True
                    break
                text, audit = generate_one(ctx, protocol, row)
                audit.update(function_sha256=function_sha, semantic_key=row["semantic_key"], actual=current)
                journal.append(row["semantic_key"], banks.raw_answer(name, row, text), audit)
                seconds.append(audit["elapsed_seconds"])
        state = banks.completion(name, len(journal.present), len(journal.missing()), len(rows))
        predictions = journal.export(ctx.attempt / (name + ".PREDICTIONS.jsonl")) if state["complete"] else None
        new_answers = len(journal.present) - before
    finally:
        journal.close()
    report = dict(protocol=name, bank=protocol["bank"], status=BANK_PROGRESS, completion=state,
                  new_answers=new_answers, deadline_stop=stopped, function=function_ref, roster=summary,
                  seconds_per_new_answer=(sum(seconds) / len(seconds)) if seconds else None,
                  max_seconds_per_answer=max(seconds) if seconds else None, labels_opened=False)
    if state["complete"]:
        exported = read_exact(predictions, limit=1 << 30)
        banks.parse_predictions(name, exported, rows)
        report["predictions"] = publish_bytes(bank_dir / "PREDICTIONS.jsonl", exported)
        report["export_summary"] = publish_same(bank_dir / "EXPORT_SUMMARY.json", dict(
            protocol=name, bank=protocol["bank"], cell=ctx.cell, roster=summary, function=function_ref,
            questions=questions_ref, predictions=report["predictions"], scoring=protocol["scoring"],
            judge_required=protocol["judge_required"], labels_opened=False,
            score_status=("JUDGE_SCORING_DEFERRED_TO_REGISTERED_BENCH4_JUDGE_STEP" if protocol["judge_required"]
                          else "SCORING_DEFERRED_TO_REGISTERED_BENCH4_SCORING_STEP")))
        publish_same(bank_dir / "COMPLETE.json", dict(
            protocol=name, cell=ctx.cell, completion=state, function=function_ref, predictions=report["predictions"],
            questions=questions_ref, export_summary=report["export_summary"]))
        report["status"] = BANK_COMPLETE
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_bytes())
    job = os.environ.get("SLURM_JOB_ID", "")
    require(os.getuid() == UID and job.isdecimal() and config["job_id"] == job and config["role"] == ROLE,
            "Actual owned evaluation job required")
    sources = config["sources"]
    require(sorted(sources) == sorted(GENERATION_SOURCES), "Exact generation source closure required")
    require(Path(__file__).absolute() == Path(sources["model4.py"]["path"])
            and digest(__file__) == sources["model4.py"]["sha256"], "Current inference source changed")
    banks = load_bound(sources["banks4.py"], "tool1_bench4_banks")
    cells = json.loads(read_exact(sources["CELLS_V3.json"]))
    custody3 = load_bound(sources["custody3.py"], "tool1_cells_v3_custody")
    cell, custody = config["cell"], config["custody"]
    names = banks.check_bank_list(config["banks"])
    spec = validate_cell(cells, cell, custody, custody3)
    student = spec["kind"] == "student"
    for relative, expected in config["eval_sources"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                and digest(ROOT / relative) == expected, "Frozen evaluator source changed: " + relative)
    template = json.loads(read_exact(banks.START_TEMPLATE))
    require(template.get("schema") == "root-babel-start-evaluation-v2" and template.get("bank") == "VizWiz"
            and template.get("network_allowed") is False and template.get("cache_root") == "/data/hf_cache"
            and template.get("native_assets") == banks.NATIVE_ASSETS and template.get("frozen_scope") == banks.FROZEN_SCOPE
            and template.get("decoding") == banks.START_DECODING and template.get("training_updates") == 0
            and isinstance(template.get("runtime_module_bindings"), dict) and template.get("model_downloads") is False,
            "Exact Root START environment template required")
    attempt = attempt_directory(config, cell, job)
    bank_dirs = {name: bank_directory(name, cell) for name in names}
    locks = []
    for name in names:
        lock = os.open(str(bank_dirs[name] / "EXECUTION.lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                       0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    storage = check_storage(OUTPUT_ROOT, sum(banks.protocol(name)["output_envelope"] for name in names))
    prior = json.loads(read_exact(custody["prior_model_identity"]))
    require(prior.get("parameter_sha256") == custody["prior_parameter_sha256"],
            "Prior four-bank MODEL_IDENTITY.json no longer names the planned parameter identity")
    files = None
    if student:
        checkpoint_dir, files = producer_inputs(custody, custody3)
        files = {key: verify_large(value) for key, value in files.items()}
    require(not any(n == "llava" or n.startswith("llava.") for n in sys.modules), "Fresh native inference import required")
    sys.path[:0] = [str(ROOT), str(NATIVE)]
    import torch
    import peft
    from safetensors.torch import load_file
    from PIL import Image, __version__ as pillow
    require(torch.__version__ == "2.1.2+cu121" and torch.cuda.device_count() == 1
            and "L40S" in torch.cuda.get_device_name(), "Exact inference device/build required")
    require(metadata.version("peft") == "0.7.1" and metadata.version("transformers") == "4.37.2"
            and metadata.version("xformers") == "0.0.23.post1" and pillow == "10.2.0",
            "Native inference package versions differ")
    from experiments.root_babel_start_evaluation_v2_20261025 import worker as official
    attestation = official.runtime(template, attempt)
    from transformers.models.llama.modeling_llama import LlamaAttention
    original_forward = LlamaAttention.forward
    from experiments.session6_start_model_qualification_v4_eager_20261024 import model as stage1
    from experiments.copilot7_coincide_native_eval_v2_20261028 import model as matched
    from experiments.root_native_trained_loader_v4_20261026 import generation as native_generation
    from experiments.session6_start_evaluation_v1_20261024 import model as identity_tools
    from experiments.session6_official_eval_v2_20261024 import common as jc
    from experiments.session6_official_eval_v2_20261024.journal import Journal
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    llava = SimpleNamespace(IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX, conv_templates=conv_templates,
                            process_images=process_images, tokenizer_image_token=tokenizer_image_token)
    for module_name in ("llava.constants", "llava.conversation", "llava.mm_utils"):
        relative = str(Path(sys.modules[module_name].__file__).resolve().relative_to(ROOT))
        require(config["eval_sources"].get(relative) == digest(sys.modules[module_name].__file__),
                "Native LLaVA prompt/image source outside the pinned closure: " + module_name)
    deadline = time.monotonic() + max(0.0, config["deadline_epoch"] - time.time())
    reports = []
    with official.offline():
        assets = stage1.verified_assets(template["native_assets"], "/data/hf_cache")
        if student:
            loaded3, load_proof = compose_student(torch, peft, load_file, stage1, matched, assets,
                                                  checkpoint_dir, files)
        else:
            loaded3, load_proof = compose_start(torch, stage1, assets)
        tokenizer, model, processor = loaded3
        require(model.config.mm_use_im_start_end is False and getattr(model.config, "image_aspect_ratio", None) == "pad",
                "LLaVA-1.5 image-token and pad policy required")
        attention = eager_attention(model, LlamaAttention, original_forward)
        parameter_sha = identity_tools.parameter_digest(model)
        require(parameter_sha == custody["prior_parameter_sha256"],
                "Loaded parameters differ from the exact model behind the prior four-bank evaluation")
        generation_config = identity_tools.generation_configuration(model)
        auxiliary = banks.key(dict(tokenizer_vocabulary=tokenizer.get_vocab(), processor=processor.to_dict(),
                                   model_generation_config=generation_config))
        bridge_installation = native_generation.install(model) if student else None
        ctx = SimpleNamespace(
            banks=banks, torch=torch, llava=llava, Image=Image, jc=jc, Journal=Journal, config=config,
            bank_dirs=bank_dirs, cell=cell, job=job, attempt=attempt, student=student, deadline=deadline,
            loaded=(tokenizer, model, processor), sources_sha256=config["sources_sha256"],
            actor=config["actual_actor"], model_identity=dict(
                parameter_sha256=parameter_sha, auxiliary_identity_sha256=auxiliary,
                generation_config=generation_config, adapter_active=student))
        for name in names:
            if time.monotonic() >= deadline - BANK_START_MARGIN_SECONDS:
                reports.append(dict(protocol=name, status=BANK_NOT_STARTED, new_answers=0))
                continue
            reports.append(run_bank(ctx, name))
    require(eager_attention(model, LlamaAttention, original_forward) == attention
            and identity_tools.parameter_digest(model) == parameter_sha
            and banks.key(dict(tokenizer_vocabulary=tokenizer.get_vocab(), processor=processor.to_dict(),
                               model_generation_config=identity_tools.generation_configuration(model))) == auxiliary,
            "Generation changed model parameters, tokenizer, processor or generation configuration")
    new_answers = sum(report["new_answers"] for report in reports)
    bridge = None
    if student:
        readout = native_generation.readout(model)
        checks = ("success", "input_object_unchanged", "all_keyword_objects_unchanged", "prepare_hook_restored",
                  "original_PEFT_generate_invoked", "output_object_unchanged")
        require(readout["calls"] == new_answers == len(readout["records"])
                and all(all(record[k] is True for k in checks) for record in readout["records"]),
                "Every new answer needs exactly one successful audited PEFT bridge call")
        bridge = dict(status=readout["status"], policy=readout["policy"], calls=readout["calls"],
                      runtime_API_proof=readout["runtime_API_proof"], records_sha256=banks.key(readout["records"]),
                      all_record_checks_true=True, adapter_active=True, adapter_merged=False)
    complete = all(report["status"] == BANK_COMPLETE for report in reports)
    report = dict(status=COMPLETE_STATUS if complete else PROGRESS_STATUS, cell=cell, job_id=job, banks=reports,
                  planned_banks=names, new_answers=new_answers, parameter_sha256=parameter_sha,
                  prior_parameter_parity=True, model_load=load_proof, attention=attention,
                  runtime_attestation=attestation, storage=storage, prior_model_identity=custody["prior_model_identity"],
                  bridge_installation=bridge_installation, native_generation_bridge=bridge,
                  accepted_inference_files=files, labels_opened=False, scientific_performance_measured=False,
                  independently_reviewed=False, superiority_established=False)
    with (attempt / "RESULT.json").open("xb") as stream:
        stream.write(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"job_id": job, "status": report["status"], "cell": cell,
                      "banks": {r["protocol"]: r["status"] for r in reports}}), flush=True)
    for lock in locks:
        os.close(lock)
    del model
    gc.collect()


if __name__ == "__main__":
    main()
