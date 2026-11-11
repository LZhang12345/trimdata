"""Four complete official banks on one roster-bound, Preempt-trained Tool1 baseline model (v2)."""

import argparse
import gc
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
import uuid


ROOT = Path("/home/luyangz/data_selection_multimodal/claude_coincide_eval_src_v3_20261030")
NATIVE = ROOT / ("collaboration/published-foundation-20261022/reference_sources/LLaVA/"
                 "c121f0432da27facab705978f83c4ada465e46fd")
CHECKPOINT_ROOT = Path("/data/hf_cache/project_checkpoints/DS-MultiModal/helper-tool1-baseline-train-v4-20261102")
OUTPUT_ROOT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-baseline-eval-v2-20261102")
COUNTS = {"POPE": 8910, "GQA": 12578, "TextVQA": 5000, "ScienceQA_image": 2017}
PRODUCER_MODEL_SHA = "b53089293b49a5a2d9fb8d433999105d06c70629f3be788bbf35d3089a728b11"
ROLE = "TOOL1_BASELINE_FULL_EPOCH_FOUR_OFFICIAL_BANK_EVALUATION"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def trainable_names(qualification):
    groups = qualification["observation"]["optimizer_inventory"]["original_groups"]
    names = [name for group in groups for name in group["names"]]
    require(len(names) == len(set(names)) == 452, "Exact native language LoRA and projector support required")
    require(sum(".lora_" in name for name in names) == 448
            and sum("mm_projector" in name for name in names) == 4,
            "Unexpected native adapter parameter support")
    return names


def portable_key(name):
    return name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")


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
                language_attention="eager", vision_attention="eager",
                training_xformers_patch_installed=False)


def bank_scope(bank, roster):
    rows = roster["records"]
    require(len(rows) == COUNTS[bank] and all(row["bank"] == bank for row in rows),
            "Exact full official bank roster required")
    require(len({row["semantic_key"] for row in rows}) == COUNTS[bank],
            "Every full-bank semantic QA key must be unique")
    return rows


def validate_producer(qualification, producer):
    """Bind the exact roster/ACK-declared Preempt-trained producer; horizon follows its actual K."""
    require(isinstance(producer, dict) and set(producer) == {"cell", "job_id", "K", "training_seed", "steps",
                                                              "model_source_sha256"}
            and isinstance(producer["cell"], str) and re.fullmatch(r"[a-z0-9][a-z0-9-]{2,60}", producer["cell"])
            and isinstance(producer["job_id"], str) and producer["job_id"].isdecimal()
            and type(producer["K"]) is int and type(producer["steps"]) is int
            and producer["steps"] == (producer["K"] + 3) // 4 // 32 >= 2
            and producer["model_source_sha256"] == PRODUCER_MODEL_SHA,
            "Exact roster-bound Preempt producer descriptor required")
    source = qualification["config"]
    observation = qualification["observation"]
    steps = producer["steps"]
    require(qualification["job_id"] == source["job_id"] == producer["job_id"]
            and qualification["status"] == "ACTUAL_FULL_NATIVE_BASELINE_EPOCH_NOT_EVALUATED"
            and qualification["full_epoch_training_complete"] is True
            and qualification["cell"] == source["cell"] == producer["cell"]
            and qualification["K"] == source["K"] == producer["K"] and qualification["steps"] == steps
            and source["role"] == "TOOL1_BASELINE_FULL_EPOCH_NOT_EVALUATED"
            and source["training_seed"] == producer["training_seed"]
            and source["model_source"]["sha256"] == PRODUCER_MODEL_SHA,
            "Actual own source-bound baseline producer required; not a probe or a Root accepted-state alias")
    require(observation["optimizer_boundaries"] == steps and observation["microbatches"] == 32 * steps
            and len(observation["losses"]) == 32 * steps
            and all(type(value) in (float, int) and math.isfinite(value) for value in observation["losses"]),
            "Actual native horizon and all finite microbatch losses required")
    events = observation["step_events"]
    require(len(events) == steps and [event["step"] for event in events] == list(range(1, steps + 1))
            and all(type(event["overflow"]) is bool for event in events)
            and all(event["microbatches_seen"] == event["step"] * 32 for event in events)
            and observation["overflow_boundaries"] == sum(event["overflow"] for event in events)
            and observation["non_overflow_boundaries"] == steps - observation["overflow_boundaries"],
            "Actual overflow and optimizer-boundary accounting required")
    return CHECKPOINT_ROOT / producer["cell"] / "training"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_bytes())
    job = os.environ.get("SLURM_JOB_ID", "")
    require(os.getuid() == 2708899 and job.isdecimal() and config["job_id"] == job
            and config["role"] == ROLE
            and Path(config["output"]) == OUTPUT_ROOT / ("job-" + job),
            "Actual owned inference qualification required")
    require(Path(__file__).absolute() == Path(config["model_source"]["path"])
            and digest(__file__) == config["model_source"]["sha256"], "Current inference source changed")
    qualification = config["qualification"]
    training = validate_producer(qualification, config["producer"])
    require(config["cell"] == config["producer"]["cell"], "Evaluation cell must be its producer cell")
    for relative, expected in config["eval_sources"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                and digest(ROOT / relative) == expected, "Frozen evaluator source changed: " + relative)
    for name, reference in qualification["output_files"].items():
        require("/" not in name and (training / name).stat().st_size == reference["bytes"]
                and digest(training / name) == reference["sha256"],
                "Actual own adapter artifact changed")
    assets = qualification["config"]["assets"]
    for role in ("vicuna", "clip336", "stage1_projector"):
        for reference in assets[role]["files"]:
            require(Path(reference["path"]).stat().st_size == reference["bytes"]
                    and digest(reference["path"]) == reference["sha256"], "Cached frozen model changed")
    require(not any(name == "llava" or name.startswith("llava.") for name in sys.modules),
            "Fresh native inference import required")
    sys.path[:0] = [str(ROOT), str(NATIVE)]
    import torch
    import peft
    from safetensors.torch import load_file
    require(torch.__version__ == "2.1.2+cu121" and torch.cuda.device_count() == 1
            and "L40S" in torch.cuda.get_device_name(), "Exact inference device/build required")
    require(metadata.version("peft") == "0.7.1" and metadata.version("transformers") == "4.37.2"
            and metadata.version("xformers") == "0.0.23.post1", "Native inference package versions differ")
    from experiments.root_babel_start_evaluation_v2_20261025 import worker as official
    runtime_reference = config["official_runtime_reference"]
    runtime_raw = Path(runtime_reference["path"]).read_bytes()
    require(len(runtime_raw) == runtime_reference["bytes"]
            and hashlib.sha256(runtime_raw).hexdigest() == runtime_reference["sha256"],
            "Frozen official runtime configuration changed")
    attestation = official.runtime(json.loads(runtime_raw), Path(config["output"]))
    from transformers.models.llama.modeling_llama import LlamaAttention
    original_forward = LlamaAttention.forward
    from experiments.session6_start_model_qualification_v4_eager_20261024 import model as stage1
    from experiments.copilot7_coincide_native_eval_v2_20261028 import model as matched
    from experiments.root_native_trained_loader_v4_20261026 import generation as native_generation
    from experiments.root_current_four_bank_eval_v1_20261027 import worker as core
    from experiments.session6_start_evaluation_v1_20261024 import model as identity_tools
    from experiments.copilot7_coincide_native_eval_v2_20261028 import scoring
    reference = config["reference_evaluator"]
    reference_raw = Path(reference["path"]).read_bytes()
    require(len(reference_raw) == reference["bytes"]
            and hashlib.sha256(reference_raw).hexdigest() == reference["sha256"],
            "Frozen bank/scoring reference changed")
    evaluator = json.loads(reference_raw)
    bank_cfg = {name: evaluator[name] for name in (
        *core.c.BANK_FIELDS, "official_join_input_qualification", "official_join_input_qualification_review")}
    bank_cfg["output"] = config["output"]
    require(bank_cfg["banks"] == list(COUNTS) and bank_cfg["counts"] == COUNTS,
            "Unchanged complete four-bank definitions required")
    bankdata = core.joined_bank_data(bank_cfg, Path(config["output"]))
    scopes = {bank: bank_scope(bank, value[0]) for bank, value in bankdata.items()}
    require(set(scopes) == set(COUNTS), "All four actual bank inputs are required")
    with official.offline():
        tokenizer, base, processor = stage1.compose(assets, config_path=training)
        names = trainable_names(qualification)
        adapter = json.loads((training / "adapter_config.json").read_bytes())
        require(adapter["r"] == 128 and adapter["lora_alpha"] == 256
                and adapter["lora_dropout"] == .05 and adapter["bias"] == "none",
                "Native saved LoRA configuration differs")
        model, injection = matched.attach_native_adapter(
            base, adapter, {"trainable": [{"name": name} for name in names]},
            lambda value: peft.PeftModel.from_pretrained(
                value, str(training), adapter_name="default", is_trainable=False, local_files_only=True))
        non_lora = torch.load(training / "non_lora_trainables.bin", map_location="cpu")
        current = model.state_dict()
        require(set(non_lora) == {name for name in names if "mm_projector" in name},
                "Exact trained projector support differs")
        incompatible = model.load_state_dict(non_lora, strict=False)
        require(not incompatible.unexpected_keys
                and set(incompatible.missing_keys) == set(current) - set(non_lora),
                "Projector loading touched unexpected state")
        model.to(device="cuda:0", dtype=torch.float16).eval().requires_grad_(False)
        lora = load_file(str(training / "adapter_model.safetensors"), device="cpu")
        require(set(lora) == {portable_key(name) for name in names if ".lora_" in name},
                "Exact portable LoRA support differs")
        expected = {name: non_lora[name] if name in non_lora else lora[portable_key(name)] for name in names}
        parameters = dict(model.named_parameters())
        for name, tensor in expected.items():
            require(tensor.dtype == parameters[name].dtype == torch.float16
                    and tensor.shape == parameters[name].shape
                    and bool(torch.isfinite(tensor).all())
                    and torch.equal(parameters[name].detach().cpu(), tensor),
                    "Loaded trainable tensor is not exact: " + name)
        layers = [module for module in model.modules() if hasattr(module, "lora_A")]
        require(len(layers) == 224 and all(not layer.merged and not layer.disable_adapters for layer in layers),
                "Inference adapters must remain active and unmerged")
        require(not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()),
                "Unqualified mutable normalization buffers")
        persistent = set(model.state_dict()) - set(parameters)
        require(not persistent, "Portable-only output requires separate proof for persistent buffers")
        attention = eager_attention(model, LlamaAttention, original_forward)
        loaded = (tokenizer, model, processor, 2048)
        parameter_sha = identity_tools.parameter_digest(model)
        auxiliary_identity = core.c.key(dict(
            tokenizer_vocabulary=tokenizer.get_vocab(), processor=processor.to_dict(),
            model_generation_config=identity_tools.generation_configuration(model)))
        bridge_installation = native_generation.install(model)
        current = dict(config["actual_actor"], pid=os.getpid(), node=socket.gethostname().split(".")[0],
                       process_nonce=uuid.uuid4().hex, phase="FULL_FOUR_OFFICIAL_BANKS")
        require(current["job_id"] == job, "Own actual child and parent allocation must match")
        reports = {}
        deadline = time.monotonic() + 39000
        with torch.inference_mode():
            for bank, (roster, inventory, targets, physical, snapshot) in bankdata.items():
                rows = scopes[bank]
                directory = Path(config["output"]) / bank
                directory.mkdir()
                function = dict(
                    schema="tool1-baseline-full-official-bank-function-v2",
                    bank=bank, role="TOOL1_BASELINE_FULL_EPOCH_OFFICIAL_BANK_EVALUATION", cell=config["cell"],
                    model_result=config["model_result_reference"], adapter_active=True, adapter_merged=False,
                    model_identity_sha256=parameter_sha, full_roster_sha256=core.c.key(roster),
                    scoped_roster_sha256=core.c.key(rows), source_configuration=reference,
                    model_generation_config=identity_tools.generation_configuration(model),
                    generation_API_policy=native_generation.POLICY,
                    decoding=bank_cfg["scienceqa_decoding"] if bank == "ScienceQA_image" else bank_cfg["decoding"])
                core.u.publish(directory / "FUNCTION.json", function)
                journal = core.generation.AuditedJournal(directory / "journal", function, rows)
                store = core.media.Store(physical, rows, inventory, snapshot)
                try:
                    core.generation.native(bank, bank_cfg, rows, journal, loaded, store, current, deadline)
                    require(len(journal.present) == COUNTS[bank] and not journal.missing(),
                            "Full actual native QA generation incomplete; preserve accepted records: " + bank)
                    for entry in official.entries(journal, rows):
                        core.generation.validate_entry(bank, entry, journal.roster[entry["semantic_key"]], function)
                    predictions = journal.export(directory / "PREDICTIONS.jsonl")
                    score = scoring.score(bank, roster, journal, targets)
                    reports[bank] = dict(full_bank_questions=COUNTS[bank], accepted_QA=len(journal.present),
                                         predictions=predictions, score=score, score_computed=True,
                                         full_bank_evaluation_complete=True)
                    core.u.publish(directory / "SCORE.json", reports[bank])
                finally:
                    journal.close()
                    store.close()
        bridge = native_generation.readout(model)
        require(bridge["calls"] == sum(COUNTS.values()),
                "Exactly28505 actual original PEFT/native bridge calls required")
        require(eager_attention(model, LlamaAttention, original_forward) == attention,
                "Actual inference changed the matched attention implementation")
        require(identity_tools.parameter_digest(model) == parameter_sha
                and core.c.key(dict(
                    tokenizer_vocabulary=tokenizer.get_vocab(), processor=processor.to_dict(),
                    model_generation_config=identity_tools.generation_configuration(model))) == auxiliary_identity,
                "Scoped generation changed model parameters, tokenizer, processor or generation configuration")
        report = dict(
            status="COMPLETE_TOOL1_BASELINE_FOUR_OFFICIAL_BANKS", cell=config["cell"],
            job_id=job, config=config, native_injection=injection,
            loaded_trainable_tensor_count=len(expected), active_unmerged_layers=len(layers),
            persistent_buffer_names=sorted(persistent), attention=attention,
            exact_saved_trainables_loaded=True, parameter_sha256=parameter_sha,
            model_and_auxiliary_identity_unchanged=True, banks=reports,
            native_generation_bridge=bridge, bridge_installation=bridge_installation,
            runtime_attestation=attestation,
            official_runtime_reference=runtime_reference,
            full_input_roster_QA=28505, actual_accepted_native_QA=28505,
            GQA_main="UNCHANGED_NATIVE_NORMALIZED_SCORE_NOT_BASIS_UNCONVERTED",
            score_computed=True, independently_reviewed=False,
            scientific_performance_measured=True, superiority_established=False)
        output = Path(config["output"])
        with (output / "RESULT.json").open("xb") as stream:
            stream.write(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"job_id": job, "status": report["status"], "output": str(output)}), flush=True)
        del model, base, expected, non_lora, lora
        gc.collect()
if __name__ == "__main__":
    main()
