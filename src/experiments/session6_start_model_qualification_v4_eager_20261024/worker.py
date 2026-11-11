import argparse
import fcntl
import gc
import hashlib
import importlib
from importlib import metadata
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback

from . import common as c
from . import model as start


def fields(raw):
    return dict(re.findall(r"(?:^|\s)([^\s=]+)=(\S+)", raw))


def actual(binding, config):
    job = os.environ.get("SLURM_JOB_ID", "")
    c.require(job.isdecimal() and binding["owner"] == c.OWNER and binding["job_id"] == job
              and not os.environ.get("SLURM_ARRAY_JOB_ID"), "Exact scalar owned START allocation required")
    raw = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True, timeout=30)
    value = fields(raw)
    expected = dict(JobId=job, JobName=config["job_name"], JobState="RUNNING",
        Account="158416130568", Partition="gpu", QOS="normal", NumCPUs="8", NumTasks="1",
        TimeLimit="00:15:00", WorkDir=str(c.ROOT), Command=str(c.HERE / "run.sbatch"),
        TresPerNode="gres/gpu:h100:1", NodeList=socket.gethostname().split(".")[0])
    c.require(all(value.get(k) == v for k, v in expected.items())
              and value.get("NumNodes") in ("1", "1-1")
              and value.get("UserId", "").split("(")[0] == "u.lz341613",
              "Actual START allocation owner/route/resources differ")
    tres = dict(item.split("=", 1) for item in value["ReqTRES"].split(",") if "=" in item)
    c.require(tres.get("mem") in ("64G", "65536M") and tres.get("gres/gpu") == "1",
              "Actual START GPU/memory differs")
    intent = c.checked(binding["intent"])
    c.require(intent["configuration"] == binding["configuration"] and intent["owner"] == c.OWNER
              and value.get("Comment") == c.PREFIX + intent["nonce"], "Actual pre-RPC intent differs")
    restart = value.get("Restarts", "")
    c.require(restart.isdecimal() and os.environ.get("SLURM_RESTART_COUNT", restart) == restart,
              "Immutable actual restart identity required")
    c.require(Path(sys.executable).absolute() == Path(config["python"]).absolute(),
              "Wrong native interpreter")
    spool = subprocess.check_output(["scontrol", "write", "batch_script", job, "-"], timeout=30)
    c.require(spool == (c.HERE / "run.sbatch").read_bytes(), "Actual START wrapper spool differs")
    return dict(job_id=job, restart=int(restart), node=value["NodeList"], scheduler=raw)


def runtime():
    import torch
    expected = {"torch": "2.1.2+cu121", "transformers": "4.37.2", "numpy": "1.26.4"}
    environment = Path(c.PYTHON).parent.parent.resolve()
    modules = {}
    for name, version in expected.items():
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        distribution = metadata.distribution(name)
        c.require(module.__version__ == distribution.version == version
                  and origin.is_relative_to(environment)
                  and Path(distribution.locate_file("")).resolve().is_relative_to(environment),
                  "Imported native version/distribution/origin differs: " + name)
        modules[name] = dict(version=module.__version__, distribution=distribution.version,
                             origin=c.desc(origin))
    numpy = importlib.import_module("numpy")
    binaries = sorted(p for p in Path(numpy.__file__).parent.rglob("*")
                      if p.is_file() and (p.name.endswith(".so") or ".so." in p.name))
    c.require(binaries, "Actual NumPy binary inventory absent")
    c.require(torch.cuda.is_available() and torch.cuda.device_count() == 1
              and torch.version.cuda == "12.1", "One actual CUDA12.1 H100 required")
    torch.manual_seed(1234)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.cuda.get_device_properties(0)
    c.require("H100" in device.name, "Actual accelerator is not H100")
    return dict(modules=modules, numpy_binary_inventory=[c.desc(p) for p in binaries],
                torch_C=c.desc(torch._C.__file__), CUDA=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), GPU=device.name,
                compute_capability=list(torch.cuda.get_device_capability(0)),
                tf32=False, deterministic_algorithms=True)


def buffer_manifest(model):
    import torch
    result = {}
    for name, value in model.named_buffers():
        value = value.detach().cpu().contiguous()
        result[name] = dict(dtype=str(value.dtype), shape=list(value.shape),
            sha256=hashlib.sha256(memoryview(
                value.reshape(-1).view(torch.uint8).numpy()).cast("B")).hexdigest())
    return result


def diagnostic(loaded):
    from collections import Counter
    import torch
    _, model, _ = loaded
    configurations = {}
    vision = model.get_vision_tower()
    for label, config in (("causal", model.config), ("vision", vision.config)):
        configurations[label] = {
            name: getattr(config, name, None) for name in
            ("_attn_implementation", "_attn_implementation_internal", "_attn_implementation_autoset")
        }
        configurations[label]["config_class"] = type(config).__module__ + "." + type(config).__name__
    attention = []
    training = []
    for name, module in model.named_modules():
        if module.training:
            training.append(name)
        if name.endswith("self_attn"):
            module_file = getattr(sys.modules.get(type(module).__module__), "__file__", None)
            attention.append(dict(name=name, role="vision" if "vision_tower" in name else "causal",
                class_name=type(module).__name__, module=type(module).__module__,
                source=c.desc(module_file) if module_file and Path(module_file).is_file() else None,
                source_available=bool(module_file and Path(module_file).is_file())))
    parameters = []
    for name, value in model.named_parameters():
        parameters.append(dict(name=name, dtype=str(value.dtype), device=str(value.device),
            requires_grad=value.requires_grad, is_meta=value.is_meta, shape=list(value.shape)))
    exceptions = [value for value in parameters if value["dtype"] != "torch.float16"
                  or not value["device"].startswith("cuda:") or value["requires_grad"] or value["is_meta"]]
    causal = [value for value in attention if value["role"] == "causal"]
    return dict(schema="session6-durable-actual-model-before-guard-v3",
        configurations=configurations, actual_attention_classes=attention,
        model_training=model.training, training_module_exceptions=training,
        has_peft_config=hasattr(model, "peft_config"),
        peft_config_type=type(getattr(model, "peft_config", None)).__name__,
        parameter_count=len(parameters), parameter_exceptions=exceptions,
        dtype_device_requires_grad_counts=dict(Counter(
            str((value["dtype"], value["device"], value["requires_grad"])) for value in parameters)),
        predicates=dict(causal_config_eager=model.config._attn_implementation == "eager",
            causal_classes_eager=bool(causal) and all(value["class_name"] == "LlamaAttention" for value in causal),
            model_frozen=not model.training and not any(value["requires_grad"] for value in parameters),
            no_peft=not hasattr(model, "peft_config"), all_parameters_FP16_CUDA_nonmeta=not exceptions),
        tensor_values_copied=False, neural_forwards=0)


def identity(loaded, assets_reference, observed_runtime, *, diagnostic_path):
    import torch
    tokenizer, model, processor = loaded
    observation = diagnostic(loaded)
    c.seal(diagnostic_path, observation)
    c.require(observation["predicates"]["causal_classes_eager"],
              "Actual causal attention classes are not all native eager LlamaAttention; see durable observations")
    c.require(model.config._attn_implementation == "eager"
              and not model.training and not hasattr(model, "peft_config")
              and all(not p.requires_grad and p.dtype == torch.float16
                      and p.device.type == "cuda" for p in model.parameters()),
              "START must use frozen FP16 CUDA parameters and eager causal attention")
    c.require(not model.config.mm_use_im_start_end and not model.config.mm_use_im_patch_token
              and model.config.image_aspect_ratio == "pad",
              "Native pad/image-token policy differs")
    configuration = model.config.to_dict()
    effective = {k: v for k, v in configuration.items()
                 if k not in ("_name_or_path", "transformers_version")}
    value = dict(schema="session6-actual-untouched-START-model-identity-v1",
        native_assets=assets_reference, source_commit="c121f0432da27facab705978f83c4ada465e46fd",
        parameter_sha256=start.parameter_digest(model),
        buffers=buffer_manifest(model), effective_model_config=effective,
        generation_config=start.generation_configuration(model),
        tokenizer=dict(length=len(tokenizer), vocabulary_sha256=c.key(tokenizer.get_vocab()),
            special_tokens=tokenizer.special_tokens_map, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, model_max_length=tokenizer.model_max_length,
            padding_side=tokenizer.padding_side, class_name=type(tokenizer).__name__),
        processor=processor.to_dict(), runtime=observed_runtime,
        language_attention=model.config._attn_implementation,
        vision_attention=model.get_vision_tower().vision_tower.config._attn_implementation,
        training_updates=0, adapter_active=False)
    c.require(value["generation_config"]["num_beams"] == 1,
              "Actual inherited generation configuration violates frozen greedy policy")
    return value, configuration


def qualify(config, output, observed_runtime):
    import torch
    assets = start.verified_assets(config["native_assets"], config["cache_root"])
    loaded = start.compose(assets)
    first, first_config = identity(loaded, config["native_assets"], observed_runtime,
                                   diagnostic_path=output / "BEFORE_FIRST_IDENTITY_GUARD.json")
    c.seal(output / "FIRST_MODEL_IDENTITY.json", first)
    c.seal(output / "FIRST_MODEL_CONFIG.json", first_config)
    capsule = start.save_no_training_capsule(loaded[1], config["native_assets"], output / "capsule")
    del loaded
    gc.collect()
    torch.cuda.empty_cache()
    loaded = start.compose(assets, projector_path=output / "capsule/mm_projector.bin",
                           config_path=output / "capsule")
    second, second_config = identity(loaded, config["native_assets"], observed_runtime,
                                     diagnostic_path=output / "BEFORE_RELOADED_IDENTITY_GUARD.json")
    c.seal(output / "RELOADED_MODEL_IDENTITY.json", second)
    c.seal(output / "RELOADED_MODEL_CONFIG.json", second_config)
    c.require(first == second, "No-training model/tokenizer/processor/generation/buffer roundtrip changed")
    start.verified_assets(config["native_assets"], config["cache_root"])
    return dict(status="PASS_SCOPED_ACTUAL_START_MODEL_NO_TRAINING_ROUNDTRIP",
        model_identity=first, model_identity_sha256=c.key(first), capsule=capsule,
        named_parameter_roundtrip_equal=True, buffer_roundtrip_equal=True,
        tokenizer_processor_generation_config_roundtrip_equal=True,
        model_instances_constructed=2, training_updates=0, neural_forwards=0,
        official_benchmark_predictions=0, pixel_or_generation_qualified=False,
        full_bank_or_scorer_or_scientific_PASS=False,
        config_metadata_exclusions=["_name_or_path", "transformers_version"])


def run(binding_path):
    began = time.monotonic()
    binding = c.read(binding_path)
    config = c.checked(binding["configuration"])
    c.validate_config(config)
    c.validate_review(c.checked(binding["review"]), binding["configuration"], config)
    context = actual(binding, config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "MODEL_QUALIFICATION.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = output / ("attempts/" + context["job_id"] + "-r" + str(context["restart"]))
        attempt.mkdir(parents=True, exist_ok=False)
        try:
            free = shutil.disk_usage(output).free
            c.seal(attempt / "START.json", dict(actual=context, configuration=binding["configuration"],
                sources=config["sources"], UTC=c.now(), disk_free=free,
                CPU_preparation_inside_owned_GPU_allocation=True))
            c.require(free >= config["reserved_bytes"], "Insufficient accepted small-capsule storage")
            observed = runtime()
            c.seal(attempt / "ACTUAL_IMPORTED_RUNTIME.json", observed)
            result = qualify(config, attempt, observed)
            result.update(actual=context, elapsed_seconds=time.monotonic() - began,
                          configuration=binding["configuration"], UTC=c.now())
            return c.seal(attempt / "RESULT.json", result)
        except BaseException as error:
            c.seal(attempt / "FAILURE.json", dict(actual=context,
                configuration=binding["configuration"], error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(), UTC=c.now(),
                elapsed_seconds=time.monotonic() - began, qualification_PASS=False))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    print(c.canonical(run(parser.parse_args().binding)).decode())
