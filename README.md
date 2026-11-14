# trimdata — baseline handoff for multimodal instruction-data selection

> **New agent or collaborator? Start with [`AGENTS.md`](AGENTS.md)** (orientation, map, lessons), then [`AGENT_CONTRACT.md`](AGENT_CONTRACT.md) (your task list).

Research code, baseline checkpoints and exact data selections for a study of instruction-data selection for
LLaVA-1.5-7B. This repository lets a collaborator finish the **baseline side** of the study without retraining runs
that are already complete.

* Checkpoints (LoRA adapters + projector, 46 complete one-epoch runs): **https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints**
* Selections (ordered record ids + metadata for 16 selections, plus the eligible pool; the exact training `records.json` is rebuilt byte-for-byte from the public LLaVA mixture with `scripts/reconstruct_records.py`): **https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections**
* **Agent contract for whoever continues this work: [`AGENT_CONTRACT.md`](AGENT_CONTRACT.md)** (machine-readable: [`agent_contract.json`](agent_contract.json)).
* Code: this repository (`src/` = byte-level snapshot of the executed sources, see `SOURCE_MANIFEST.json` and `PATHS.md`).

Both Hugging Face repositories are complete (checksums in each repo's `SHA256SUMS`).

## Setting
LLaVA-1.5-7B stage-2 (instruction tuning) with a LoRA fine-tune for **one natural epoch** on a subset of the
660,763 eligible records (whole conversations) of the 665,298-record `llava_v1_5_mix665k.json`; 4,535 records are
excluded by the project's eligibility filter (the eligible ids are in `pool/pool_record_ids.json` on the dataset repo). Selected-data
budgets: 5% K=33,032, **10% K=66,076 (anchor)**, 15% K=99,108, 20% K=132,147. Evaluation on four full official banks
(28,505 questions) plus an eight-benchmark extended suite.

## Setup
* Environment pins of the executed env: `env/requirements-native-env.txt` (Python 3.10.19, torch 2.1.2+cu121,
  transformers 4.37.2, peft 0.7.1, deepspeed 0.12.6, xformers 0.0.23.post1). The `llava/` package at commit
  `c121f0432da27facab705978f83c4ada465e46fd` is vendored under `src/collaboration/published-foundation-20261022/reference_sources/LLaVA/`;
  clone `haotian-liu/LLaVA` at that commit as well for its `scripts/` (evaluation helpers).
* **Train only with `scripts/train_portable.py` and evaluate with `scripts/eval_portable.py`** (see "Executed trainer vs.
  official LLaVA" below and `AGENT_CONTRACT.md` §3–§4).
* Data: `llava_v1_5_mix665k.json` and the image folders (COCO train2017, GQA, OCR-VQA, TextVQA train_images, VG
  parts 1/2) from the official LLaVA instructions. Each selection's training `records.json` (LLaVA format, relative image paths such as `coco/train2017/...`) is rebuilt with `scripts/reconstruct_records.py --source llava_v1_5_mix665k.json --selection <dir>` (sha256-checked against `selection_meta.json`).
* Evaluation data: LLaVA `eval.zip` (c121) and the benchmark images from their official sources (see `docs/EVAL_INPUTS.md`).

## Loading a checkpoint
```bash
pip install -U "huggingface_hub>=0.23"
hf download submissionpaper1234/trimdata-baseline-checkpoints --include "uniform/b10/sel1234-t17/*" --local-dir ckpts
```
```python
# inside a checkout of haotian-liu/LLaVA at commit c121f0432da27facab705978f83c4ada465e46fd (v1.2.x API)
from llava.model.builder import load_pretrained_model
path = "ckpts/uniform/b10/sel1234-t17"          # adapter_config.json, adapter_model.safetensors, non_lora_trainables.bin, config.json
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=path, model_base="lmsys/vicuna-7b-v1.5",
    model_name="llava-v1.5-7b-lora")             # the name must contain "llava" and "lora" so the LoRA branch is used
```
`config.json` / `adapter_config.json` are *portable* copies in which the cluster-local snapshot paths were replaced by
the Hub ids above; the byte-exact originals are kept as `config.original.json` / `adapter_config.original.json`.
`non_lora_trainables.bin` holds the trained `mm_projector` weights (projector LR 2e-5); the builder loads it before
attaching the LoRA weights. Our own evaluator kept the LoRA **unmerged** (PEFT 0.7.1) in fp16; the official builder
merges it (`merge_and_unload`) — re-evaluating a provided checkpoint is part of the calibration step below.

## Training recipe
| Item | Value (as executed) |
|---|---|
| Base | Vicuna-7B-v1.5 + CLIP ViT-L/14-336 (`mm_vision_select_layer=-2`, `mm_vision_select_feature=patch`) + stage-1 `mlp2x_gelu` projector |
| Code base | official LLaVA `llava/train/train.py` at commit `c121f0432da27facab705978f83c4ada465e46fd`, conversation template `v1` |
| Trainable | LoRA on all LLM linear layers (q,k,v,o,gate,up,down), `r=128`, `alpha=256`, `dropout=0.05`, `bias=none`; projector fully trained |
| Learning rate | 2e-4 (LoRA), 2e-5 (`mm_projector_lr`), AdamW (`adamw_torch`), weight decay 0 |
| Schedule | cosine, `warmup_ratio=0.03`, **one natural epoch** over the selected subset |
| Batch | 1 GPU, `per_device_train_batch_size=4`, `gradient_accumulation_steps=32` → global batch 128 |
| Steps | `natural_steps = (K + 3) // 4 // 32` (the last partial accumulation is dropped): 258 @5%, 516 @10%, 774 @15%, 1032 @20% |
| Sequence | `model_max_length=2048`, `image_aspect_ratio=pad`, `group_by_modality_length=True`, `lazy_preprocess=True` |
| Precision | `fp16=True`, `bf16=False`, `tf32=False` (NVIDIA L40S), DeepSpeed ZeRO-3 (`zero3.json`), gradient checkpointing |
| Attention | training: LLaVA's xformers patch (`llama_xformers_attn_monkey_patch`) with xformers 0.0.23.post1; the Tool1/Tool3 drivers additionally forced the CUTLASS kernels with deterministic torch/cuBLAS settings. Inference: HF eager. FlashAttention was never used |
| Seeds | training seed ∈ {17, 29, 43} sets both `seed` and `data_seed`; the selection seed only determines the subset |
| Data loader | `dataloader_num_workers=0`; images from the official LLaVA-665K image folders (coco, gqa, ocr_vqa, textvqa, vg) |

Exact argument construction: `src/experiments/helper_tool1_baseline_train_v4_20261102/model.py::arguments` (used by
the v4/v10 drivers) and `src/experiments/root_native_campaign_v7_20261024/config.py` (identical recipe, used for the
10% Uniform-sel1234 / source-sqrt / COINCIDE runs).

### Executed trainer vs. official LLaVA
Every provided run used the official c121 `llava/train/train.py` **plus three runtime changes** (no LLaVA file was edited):
1. **Attention:** LLaVA's own xformers patch (as in its `train_xformers.py`); the Tool1/Tool3 drivers also forced the
   xformers CUTLASS forward/backward kernels with `torch.use_deterministic_algorithms(True)`,
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off.
2. **Structured v1 masking** (`src/experiments/root_native_campaign_v7_20261024/structured_preprocess.py`): turn
   boundaries come from the conversation structure instead of splitting the rendered text, so a message that itself
   contains a role marker such as "USER:" is masked correctly. Prompts, token ids and truncation are unchanged.
3. **Optimizer groups** (`.../optimizer_groups.py`): LLaVA's trainer builds four parameter groups, one of which (LoRA
   without weight decay) is always empty. DeepSpeed 0.12.6 removes empty groups *after* the cosine scheduler has captured
   four base learning rates, and torch 2.1.2 zips them without a length check, so the projector weights would train at
   the LoRA rate 2e-4 instead of 2e-5. The empty group is removed before the scheduler is built.

`scripts/train_portable.py` re-applies exactly these changes by importing the executed modules unchanged from `src/`,
checks the LR ratio during training and the final step count. `scripts/reference_train_command.sh` only lists the
arguments for reading; **training with it, `train.py` or `train_mem.py` reproduces a different recipe.**

## Evaluation protocol
Core four banks (full official banks, 28,505 questions; all greedy: `conv_mode=vicuna_v1`, `temperature=0`,
`num_beams=1`, fp16, batch size 1, eager attention, LoRA unmerged):

| Bank | Questions | Input (LLaVA `eval.zip`, c121) | Generation | Score |
|---|---|---|---|---|
| POPE | 8,910 (adversarial/popular/random, COCO val2014) | `pope/llava_pope_test.jsonl` | `model_vqa_loader`, `max_new_tokens=128` | official `eval_pope.py` per category; **reported = pooled accuracy over all 8,910** |
| GQA | 12,578 (test-dev balanced) | `llava_gqa_testdev_balanced.jsonl` | `model_vqa_loader`, `max_new_tokens=128` | official GQA balanced test-dev accuracy after LLaVA's `convert_gqa_for_eval` (`pred.rstrip('.').lower()`); this **native normalized score** is the main GQA number (not the unconverted-basis score) |
| TextVQA | 5,000 (val, v0.5.1 with OCR tokens) | `textvqa/llava_textvqa_val_v051_ocr.jsonl` | `model_vqa_loader`, `max_new_tokens=128` | official `m4c_evaluator` TextVQA accuracy |
| ScienceQA-IMG | 2,017 (test, image subset) | `scienceqa/llava_test_CQM-A.json` | `model_vqa_science`, `single_pred_prompt=True`, `max_new_tokens=1024` | official `eval_science_qa.py`, image-question accuracy |

Core average = unweighted mean of the four bank scores. Input hashes are in `docs/EVAL_INPUTS.md`.
To reproduce these settings with the official generation modules use `scripts/eval_portable.py` (keeps the LoRA
unmerged, forces eager attention and fp16, and selects the LoRA loading branch regardless of the folder name).

Extended suite (scored with the same greedy decoding): MME perception/cognition (official `llava_mme.jsonl`,
official MME calculation, points of 2000/800), SEED-Bench image (14,233), MMMU val (official MMMU prompt rendered with the c121 conversation, `banks4.py`), MMBench-EN dev (local labels,
CircularEval, VLMEvalKit `can_infer` exact-match extraction),
LLaVA-Bench in-the-wild and MM-Vet (**GPT judge `gpt-4o-2024-08-06`, temperature 0, one run** — a deviation from the
original judges gpt-4-0314 / gpt-4-0613, so numbers are not comparable with published ones), VizWiz test 8,000 questions
scored locally against the publicly released answers (not EvalAI test-dev/std; strongly sensitive to the
"unanswerable" rate and to seed). Extended-suite code: `src/experiments/helper_tool1_bench4_*`, `helper_tool1_bench6_*`,
`helper_tool1_cells_v7_extsuite_20261110`, `helper_tool1_vizwiz_score_v1_20261101`; bank definitions and prompts
`helper_tool1_bench6_eval_v1_20261103/banks6.py`, `helper_tool1_bench4_eval_v1_20261104/banks4.py`; judge prompts and
settings `helper_tool1_bench4_judge_v13_20261105/judge4_v13.py`. Judge code reads the API key from the environment or a
local key file; no key is included.

## Baseline results (core four; from per-run result files)
Each row is a mean over the evaluated training seeds; per-run values in `results/baseline_core4_per_run.csv`.
"trained (complete)" counts uploaded complete checkpoints.

| Budget | K | Method | trained (complete) | evaluated | per-seed core avg | POPE | GQA | TextVQA | SQA-img | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| – | 0 | START (no instruction tuning; base model) | – | 1 | 25.77 | 63.88 | 2.77 | 4.10 | 32.33 | **25.77** |
| 5% | 33,038 | COINCIDE (adaptation; see notes) | 3 (t17, t29, t43) | 3 | 64.29 (t17) / 63.79 (t29) / 64.13 (t43) | 85.10 | 53.51 | 51.35 | 66.32 | **64.07** |
| 5% | 33,032 | DataTailor (official code) | 3 (t17, t29, t43) | 3 | 63.28 (t17) / 64.25 (t29) / 63.68 (t43) | 82.93 | 54.33 | 50.87 | 66.82 | **63.74** |
| 5% | 33,032 | Uniform random (matched-K) | 3 (t17, t29, t43) | 3 | 64.16 (t17) / 63.40 (t29) / 63.63 (t43) | 84.25 | 52.77 | 52.15 | 65.74 | **63.73** |
| 10% | 66,076 | COINCIDE (adaptation; see notes) | 3 (t17, t29, t43) | 3 | 65.77 (t17) / 65.16 (t29) / 65.27 (t43) | 85.23 | 55.85 | 54.13 | 66.40 | **65.40** |
| 10% | 66,071 | DataTailor (official code) | 3 (t17, t29, t43) | 3 | 65.19 (t17) / 66.15 (t29) / 65.65 (t43) | 83.71 | 57.68 | 53.60 | 67.67 | **65.66** |
| 10% | 66,076 | LESS-approx (not faithful LESS) | 3 (t17, t29, t43) | 3 | 63.56 (t17) / 64.01 (t29) / 63.29 (t43) | 82.95 | 50.05 | 53.58 | 67.91 | **63.62** |
| 10% | 66,076 | Source-sqrt (source-balanced random) | 3 (t17, t29, t43) | 3 | 64.93 (t17) / 64.65 (t29) / 65.32 (t43) | 84.90 | 54.86 | 53.23 | 66.90 | **64.97** |
| 10% | 66,076 | Uniform random sel1234 | 3 (t17, t29, t43) | 3 | 65.23 (t17) / 64.93 (t29) / 64.83 (t43) | 84.73 | 55.54 | 52.74 | 67.00 | **65.00** |
| 10% | 66,076 | Uniform random sel1235 | 3 (t17, t29, t43) | 3 | 65.22 (t17) / 65.46 (t29) / 65.11 (t43) | 84.98 | 55.48 | 53.76 | 66.85 | **65.26** |
| 10% | 66,076 | Uniform random sel1236 | 3 (t17, t29, t43) | 3 | 64.96 (t17) / 65.06 (t29) / 64.95 (t43) | 84.22 | 55.52 | 53.49 | 66.73 | **64.99** |
| 10% | 66,076 | Uniform random, all 3 selection sets | 9 (3 sel x t17/t29/t43) | 9 | see per-set rows | 84.64 | 55.51 | 53.33 | 66.86 | **65.08** |
| 10% | 66,076 | Uniform scored-random (control) | 3 (t17, t29, t43) | 3 | 65.25 (t17) / 65.13 (t29) / 64.85 (t43) | 84.98 | 55.18 | 53.08 | 67.08 | **65.08** |
| 15% | 99,108 | DataTailor (official code) | 2 (t29, t43) | 2 | 66.13 (t29) / 66.56 (t43) | 84.78 | 58.83 | 53.79 | 68.00 | **66.35** |
| 15% | 99,108 | Uniform random (matched-K) | 3 (t17, t29, t43) | 3 | 65.89 (t17) / 66.25 (t29) / 66.12 (t43) | 85.65 | 57.13 | 54.41 | 67.16 | **66.09** |
| 20% | 132,152 | COINCIDE (adaptation; see notes) | 3 (t17, t29, t43) | 1 | 66.75 (t17) | 85.76 | 58.90 | 55.85 | 66.48 | **66.75** |
| 20% | 132,147 | DataTailor (official code) | 3 (t17, t29, t43) | 3 | 67.32 (t17) / 67.55 (t29) / 67.22 (t43) | 85.28 | 59.91 | 54.87 | 69.39 | **67.36** |
| 20% | 132,147 | Uniform random (matched-K) | 2 (t17, t43) | 2 | 66.23 (t17) / 66.51 (t43) | 85.16 | 58.11 | 55.00 | 67.20 | **66.37** |

Extended suite (means as reported in our living results table; not recomputed for this release;
`results/baseline_extended_reported.csv`; *VizWiz in brackets = number of seeds):

| Budget | Method (seeds) | MME-P | MME-C | SEED-img | MMMU | MMBench | LLaVA-W | MM-Vet | VizWiz* |
|---|---|---|---|---|---|---|---|---|---|
| – | START (1) | 622.5 | 168.6 | 0.0 | 29.3 | 37.1 | 42.2 | 22.9 | 44.72 |
| 10% | Uniform sel1234 (3) | 1342.0 | 317.7 | 59.8 | 34.3 | 58.8 | 56.7 | 27.2 | 41.77 (3) |
| 10% | Uniform sel1235 (3) | 1358.2 | 303.1 | 60.1 | 35.0 | 58.3 | 60.0 | 26.5 | 42.74 (1) |
| 10% | Uniform sel1236 (3) | 1343.6 | 303.4 | 59.4 | 33.5 | 58.4 | 58.2 | 26.4 | 39.53 (2) |
| 10% | Source-sqrt (3) | 1370.9 | 314.6 | 59.2 | 34.4 | 59.3 | 59.4 | 27.9 | 40.51 (3) |
| 10% | COINCIDE (3) | 1413.1 | 293.6 | 60.1 | 33.9 | 59.2 | 61.3 | 28.5 | 41.56 (3) |
| 10% | DataTailor (3) | 1429.1 | 330.8 | 61.0 | 33.7 | 59.1 | 59.6 | 26.3 | 41.23 (1; t17) |
| 10% | LESS-approx (t17 only) | 1057.1 | 257.5 | 60.4 | 35.3 | 58.6 | 60.7 | 24.9 | 45.51 |
| 10% | Uniform scored-random (t17 only) | 1344.6 | 335.7 | 59.9 | 33.0 | 59.4 | 59.4 | 25.2 | 44.00 |
| 5% | Uniform (t17) | 1286.0 | 287.1 | 57.40 | 33.89 | 56.80 | 60.80 | 24.20 | 40.27 |
| 5% | DataTailor (t17) | 1314.4 | 305.0 | 60.51 | 32.11 | 57.99 | 63.30 | 26.50 | 45.09 |
| 5% | COINCIDE (t17) | 1332.7 | 289.6 | 57.48 | 34.44 | 56.29 | 61.90 | 29.80 | 35.85 |
| 15% | Uniform (t17) | 1386.9 | 318.9 | 60.49 | 34.56 | 58.59 | 61.40 | 28.10 | 43.89 |
| 15% | DataTailor (t43) | 1394.7 | 294.3 | 62.13 | 33.56 | 61.05 | 59.40 | 27.00 | 41.49 |
| 20% | Uniform (t17) | 1397.5 | 302.1 | 62.31 | 32.67 | 58.16 | 58.90 | 24.40 | 44.81 |
| 20% | DataTailor (t17) | 1402.2 | 338.9 | 61.93 | 34.00 | 60.97 | 62.10 | 28.00 | 43.93 |

## Coauthor plan
**Current assignment (2026-11-13): start with
[`docs/COAUTHOR_NEXT_WAVE_20261113.md`](docs/COAUTHOR_NEXT_WAVE_20261113.md).**
The authoritative task list, gates, tolerances, schedule for 8× A100 40GB and hand-back format are in
[`AGENT_CONTRACT.md`](AGENT_CONTRACT.md), section 1a (machine-readable:
[`agent_contract.json`](agent_contract.json)); scheduling order is [`AGENTS.md`](AGENTS.md) §3a.

Project-lead update, 2026-11-12: defer LLaVA-W and MM-Vet judge experiments while continuing core-four
and locally scored extended work. Deferred metrics remain blank.

| ID | Task |
|---|---|
| T0 | Existing T0a/T0b/T0c-local passed; verify compatible receipts instead of repeating. T0c-judged deferred |
| T12, T4, T9, T10 | New VisNec, ICONS, XMAS, CoIDO coverage in that order; 10%, 5%, 20%, training seed 17 first |
| T0d, T13 | Gated 13B calibration, then baseline selected-set transfer (5%/10% existing methods; 10% new methods) |
| T8 | Full-data reference only after manager ownership/live-continuation reconciliation |
| T1, T5, T2 | Lower-priority remaining evaluations and Uniform 20% t29; skip already complete banks/cells |
| T3, T7 | Cancelled/dropped; no new 15% jobs or source-sqrt 5%/20% jobs |
| T6, T11 (optional, later) | Faithful LESS 10% seed 17; PreSel adaptation proposal requires approval |

## Checkpoint index
Every folder has `provenance.json` (source run, file sha256, steps, K, selection reference, core-four scores and the
evaluation result it came from). Machine-readable: `index/checkpoints.csv`.

| HF path | Method | Budget | K | sel seed | train seed | steps | core avg | adapter sha256 |
|---|---|---|---|---|---|---|---|---|
| [`coincide/b05/selfixed-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/coincide/b05/selfixed-t17/) | COINCIDE (adaptation; see notes) | 5% | 33,038 | fixed | 17 | 258/258 | 64.29 | `8a45f40e798511c52e9603edf3e2ecc8d3ed691c147c4131a64ecf789cdc68bd` |
| [`coincide/b05/selfixed-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/coincide/b05/selfixed-t29/) | COINCIDE (adaptation; see notes) | 5% | 33,038 | fixed | 29 | 258/258 | 63.79 | `93d2efd35b441132e7916b28a752021eb1edf65ad12388465c9af72076a312f5` |
| [`coincide/b05/selfixed-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/coincide/b05/selfixed-t43/) | COINCIDE (adaptation; see notes) | 5% | 33,038 | fixed | 43 | 258/258 | 64.13 | `f0e4b41b6f7ea27d2f0e2256822ed5d7fe54e0c083322e647f9d601f2d318aae` |
| [`coincide/b10/sel1234-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/coincide/b10/sel1234-t17/) | COINCIDE (adaptation; see notes) | 10% | 66,076 | 1234 | 17 | 516/516 | 65.77 | `ec05e1cf2c4fa846e88eae164791492fe7bfe032d5991e7da6da0e761090e411` |
| [`coincide/b10/sel1234-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/coincide/b10/sel1234-t29/) | COINCIDE (adaptation; see notes) | 10% | 66,076 | 1234 | 29 | 516/516 | 65.16 | `f9aa53fb95e2b38a02160230b37acd24e7d7789aa61c75749dda0b7d2d5605d8` |
| [`coincide/b10/sel1234-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/coincide/b10/sel1234-t43/) | COINCIDE (adaptation; see notes) | 10% | 66,076 | 1234 | 43 | 516/516 | 65.27 | `73ab1cc5ab79cf5578e29e4400750c5d47b44c6121aeaa6c7abf683a7673d6a7` |
| [`coincide/b20/selfixed-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/coincide/b20/selfixed-t17/) | COINCIDE (adaptation; see notes) | 20% | 132,152 | fixed | 17 | 1032/1032 | 66.75 | `1797baeb7b8423412ab7b21e356fe83326ebe2606560e16a7b95a0e35e0e3955` |
| [`coincide/b20/selfixed-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/coincide/b20/selfixed-t29/) | COINCIDE (adaptation; see notes) | 20% | 132,152 | fixed | 29 | 1032/1032 | not evaluated | `cc5546c447b21d38fe8cc934a27598206fec55daf8c367644c439fb5a5f3dd91` |
| [`coincide/b20/selfixed-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/coincide/b20/selfixed-t43/) | COINCIDE (adaptation; see notes) | 20% | 132,152 | fixed | 43 | 1032/1032 | not evaluated | `a528b174ad59753de3f43cf95df3688bb76a069af34b5e7b502ebb0ffb70679a` |
| [`datatailor/b05/sel1234-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/datatailor/b05/sel1234-t17/) | DataTailor (official code) | 5% | 33,032 | 1234 | 17 | 258/258 | 63.28 | `9378fa787b7980d5ef8f8830ea6549c106085e902b15d02e4ec45a518ee17f1c` |
| [`datatailor/b05/sel1234-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/datatailor/b05/sel1234-t29/) | DataTailor (official code) | 5% | 33,032 | 1234 | 29 | 258/258 | 64.25 | `b53b47b456a633e6950167587bc4f904e858565caebf5f96beecff7704f1d7b3` |
| [`datatailor/b05/sel1234-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/datatailor/b05/sel1234-t43/) | DataTailor (official code) | 5% | 33,032 | 1234 | 43 | 258/258 | 63.68 | `85470fad67e9902b01548506d820aa1e0197d4dd5ddd039ac8be03e50201afe0` |
| [`datatailor/b10/sel1234-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/datatailor/b10/sel1234-t17/) | DataTailor (official code) | 10% | 66,071 | 1234 | 17 | 516/516 | 65.19 | `79cecc3957338d68fb5820f56faa601e2ce199117aae8597d082334f928cd259` |
| [`datatailor/b10/sel1234-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/datatailor/b10/sel1234-t29/) | DataTailor (official code) | 10% | 66,071 | 1234 | 29 | 516/516 | 66.15 | `087d8786ccbb29ccfa4fdaa4e560aced5c4f9cd9aa48ad47dc3d0777426f606f` |
| [`datatailor/b10/sel1234-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/datatailor/b10/sel1234-t43/) | DataTailor (official code) | 10% | 66,071 | 1234 | 43 | 516/516 | 65.65 | `983b4230a32fa7c7eb6e44cec1e385ab25e50583a35579d9fd630b5b964e773a` |
| [`datatailor/b15/sel1234-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/datatailor/b15/sel1234-t29/) | DataTailor (official code) | 15% | 99,108 | 1234 | 29 | 774/774 | 66.13 | `a09c128e80cb3b4963a235941b815c85210d5df3e266acbdf6fb1da3e9707768` |
| [`datatailor/b15/sel1234-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/datatailor/b15/sel1234-t43/) | DataTailor (official code) | 15% | 99,108 | 1234 | 43 | 774/774 | 66.56 | `c26aae47c32a99d212a6e8e645ac08e90f44bb6f171657aa62ee14106b0b4e99` |
| [`datatailor/b20/sel1234-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/datatailor/b20/sel1234-t17/) | DataTailor (official code) | 20% | 132,147 | 1234 | 17 | 1032/1032 | 67.32 | `6631556a24bac9aa64efac9ed634533edcf97397691749264b51a412c05e318c` |
| [`datatailor/b20/sel1234-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/datatailor/b20/sel1234-t29/) | DataTailor (official code) | 20% | 132,147 | 1234 | 29 | 1032/1032 | 67.55 | `711265aecb5122a614ce1a23ce7323fc466c283c70192ab2592364e9b29197a5` |
| [`datatailor/b20/sel1234-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/datatailor/b20/sel1234-t43/) | DataTailor (official code) | 20% | 132,147 | 1234 | 43 | 1032/1032 | 67.22 | `42f2edcf47ad5f4181658d8915e3b088788c8c22d99719e5057373f0c03c330d` |
| [`less_approx/b10/sel1234-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/less_approx/b10/sel1234-t17/) | LESS-approx (not faithful LESS) | 10% | 66,076 | 1234 | 17 | 516/516 | 63.56 | `d89b12855323aa537ef3f909a6806ea036614980ef26cf63926df8127b2c8c5c` |
| [`less_approx/b10/sel1234-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/less_approx/b10/sel1234-t29/) | LESS-approx (not faithful LESS) | 10% | 66,076 | 1234 | 29 | 516/516 | 64.01 | `42ebbe0905667a8fa2a9ab488be110fba6447ce4965f7fe4c178be76a2522130` |
| [`less_approx/b10/sel1234-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/less_approx/b10/sel1234-t43/) | LESS-approx (not faithful LESS) | 10% | 66,076 | 1234 | 43 | 516/516 | 63.29 | `dc3a6e7fedbc73a88280f1cb657efd434deefa9d189017b8c8284fdd29a7367f` |
| [`source_sqrt/b10/sel1234-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/source_sqrt/b10/sel1234-t17/) | Source-sqrt (source-balanced random) | 10% | 66,076 | 1234 | 17 | 516/516 | 64.93 | `42ef0a16fd66092e576d0fbc72732d9bf977e24dc29b671eebbca8261e2be908` |
| [`source_sqrt/b10/sel1234-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/source_sqrt/b10/sel1234-t29/) | Source-sqrt (source-balanced random) | 10% | 66,076 | 1234 | 29 | 516/516 | 64.65 | `f0cdb1759654ec77c65f4b5c884c7b133e6252085b00f1a8bd25aadd5257cd0e` |
| [`source_sqrt/b10/sel1234-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/source_sqrt/b10/sel1234-t43/) | Source-sqrt (source-balanced random) | 10% | 66,076 | 1234 | 43 | 516/516 | 65.32 | `a4dbde88a11735b1cfed1dc97430f8a375bbc08f1825fcfa22f4f301b0663dee` |
| [`uniform/b05/sel1235-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b05/sel1235-t17/) | Uniform random (matched-K) | 5% | 33,032 | 1235 | 17 | 258/258 | 64.16 | `55cbfe9a9b88c92b25d3c0080ffd5f6d7b5e73a2afc12d4799f140d0d28f13d3` |
| [`uniform/b05/sel1235-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b05/sel1235-t29/) | Uniform random (matched-K) | 5% | 33,032 | 1235 | 29 | 258/258 | 63.40 | `c10f9eb7996a191f938ee9efa49c2a13762a9ac2a8697cfbc0aeb10c0525cc76` |
| [`uniform/b05/sel1235-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b05/sel1235-t43/) | Uniform random (matched-K) | 5% | 33,032 | 1235 | 43 | 258/258 | 63.63 | `dbb8ce202922e16746fac653ea5e0da4cf955697c0579ee0827488cec46302cb` |
| [`uniform/b10/sel1234-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b10/sel1234-t17/) | Uniform random | 10% | 66,076 | 1234 | 17 | 516/516 | 65.23 | `fa45381c39658c43b8e732305b1db6821c7ce1ba7f5701726fcc670c37f3f320` |
| [`uniform/b10/sel1234-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b10/sel1234-t29/) | Uniform random | 10% | 66,076 | 1234 | 29 | 516/516 | 64.93 | `557207f647bd71ce2acde7711aa1dd3d46a8bcb5f31cc08313741c1a7bb9b4f6` |
| [`uniform/b10/sel1234-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b10/sel1234-t43/) | Uniform random | 10% | 66,076 | 1234 | 43 | 516/516 | 64.83 | `17571e3caf09d7a78528bf954d893c1f45c00e798b92a520b4c254be331bc1f5` |
| [`uniform/b10/sel1235-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b10/sel1235-t17/) | Uniform random | 10% | 66,076 | 1235 | 17 | 516/516 | 65.22 | `a6388122e9c1ca84ce8bfb569b01bcd322f7a544dbbe002578de355c4860e884` |
| [`uniform/b10/sel1235-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b10/sel1235-t29/) | Uniform random | 10% | 66,076 | 1235 | 29 | 516/516 | 65.46 | `2e7ed4f238ec468186ed1e86004959a4c71f197980345c7cef3a354300e1d9ce` |
| [`uniform/b10/sel1235-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b10/sel1235-t43/) | Uniform random | 10% | 66,076 | 1235 | 43 | 516/516 | 65.11 | `ef32289f01b54dc17311bdcfe90705947e81d8656773ce6365411517e33bee21` |
| [`uniform/b10/sel1236-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b10/sel1236-t17/) | Uniform random | 10% | 66,076 | 1236 | 17 | 516/516 | 64.96 | `c800bc8e6d973707367670eaeb62b48cd5cf4e54921cffbb54bc4a03f33c6d5d` |
| [`uniform/b10/sel1236-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b10/sel1236-t29/) | Uniform random | 10% | 66,076 | 1236 | 29 | 516/516 | 65.06 | `309d8239df68307aaf44c798952056fbaf841538e085cf89e9cd10b750652798` |
| [`uniform/b10/sel1236-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b10/sel1236-t43/) | Uniform random | 10% | 66,076 | 1236 | 43 | 516/516 | 64.95 | `1c84f38913d792ed80b3b45f0f2ed800e071a30735d3ad164bf7cd65e146b946` |
| [`uniform/b15/sel1235-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b15/sel1235-t17/) | Uniform random (matched-K) | 15% | 99,108 | 1235 | 17 | 774/774 | 65.89 | `7a6c91f6c856f0f3a082c9b88c72f1b8cded2b0b9c61a24c6a7f742590de97a3` |
| [`uniform/b15/sel1235-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b15/sel1235-t29/) | Uniform random (matched-K) | 15% | 99,108 | 1235 | 29 | 774/774 | 66.25 | `2506fc133a361a750ea72d13f8cdb3e15c5cfab6b852f0a772c55afb3c644d6b` |
| [`uniform/b15/sel1235-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b15/sel1235-t43/) | Uniform random (matched-K) | 15% | 99,108 | 1235 | 43 | 774/774 | 66.12 | `5d1efabdfcc46c6626b13601fc436339e33a63deb41ebf5dd2a77c7c4132c1b6` |
| [`uniform/b20/sel1235-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b20/sel1235-t17/) | Uniform random (matched-K) | 20% | 132,147 | 1235 | 17 | 1032/1032 | 66.23 | `d76c6c03ded93f779044585f6f224043fcc5ad12c17c0220ff7e5d42b7583678` |
| [`uniform/b20/sel1235-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform/b20/sel1235-t43/) | Uniform random (matched-K) | 20% | 132,147 | 1235 | 43 | 1032/1032 | 66.51 | `f8aa18d86127ecc11c815da83d493909c465c49850d34b3386cc3175bd668424` |
| [`uniform_scored_random/b10/sel1234-t17/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform_scored_random/b10/sel1234-t17/) | Uniform scored-random (control) | 10% | 66,076 | 1234 | 17 | 516/516 | 65.25 | `cc652c2e09d09f432f7ecdb183f910bf32a652998eba8d987102e6cf0dc245f9` |
| [`uniform_scored_random/b10/sel1234-t29/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform_scored_random/b10/sel1234-t29/) | Uniform scored-random (control) | 10% | 66,076 | 1234 | 29 | 516/516 | 65.13 | `cbcde30bc7f0c9796f436d30050876c7c48aa2d70fb3cc44e653004e71732081` |
| [`uniform_scored_random/b10/sel1234-t43/`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints/tree/main/uniform_scored_random/b10/sel1234-t43/) | Uniform scored-random (control) | 10% | 66,076 | 1234 | 43 | 516/516 | 64.85 | `8a2c89834ec428bb837454c9f681d701c7a6ee74d76a95983f916d192df3b8bc` |

## Selection index
Each selection folder on the dataset repo has `record_ids.json` (ordered pool ids `<index in llava_v1_5_mix665k.json>:<LLaVA id>`)
and `selection_meta.json` (K, seed, per-source counts, sha256 of the training `records.json`). The `records.json` text itself is
not re-hosted (it is verbatim LLaVA-665K content, including third-party code snippets that trip credential scanners);
`scripts/reconstruct_records.py` rebuilds it byte-identically from `llava_v1_5_mix665k.json` (HF `liuhaotian/LLaVA-Instruct-150K`,
revision `9d451dc7629cfe0469f6ae4432b765cd603d5fcb`, sha256 `ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8`). Machine-readable: `index/selections.csv`.

| Dataset path | Method | Budget | K | sel seed | records.json sha256 |
|---|---|---|---|---|---|
| [`selections/coincide/b05/selfixed`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/coincide/b05/selfixed) | COINCIDE (adaptation; see notes) | 5% | 33,038 | fixed | `24abed8530b7f2b7e89ae351d134f9baf0551d59273e31a02be9d9e87cb669ed` |
| [`selections/coincide/b10/sel1234`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/coincide/b10/sel1234) | COINCIDE (adaptation; see notes) | 10% | 66,076 | 1234 | `36d75b04009fe9ce4c8bed6b5be4c7da698a0327e81e59b373a64aae352debdf` |
| [`selections/coincide/b20/selfixed`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/coincide/b20/selfixed) | COINCIDE (adaptation; see notes) | 20% | 132,152 | fixed | `b38b714c95267e403689849a0c6efd992a9e3274161d93248a78f07fe7e200fa` |
| [`selections/datatailor/b05/sel1234`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/datatailor/b05/sel1234) | DataTailor (official code) | 5% | 33,032 | 1234 | `4ef48c0305a10150fcf33f1e491d975262a329a1e39423f2b5a47c31b14042db` |
| [`selections/datatailor/b10/sel1234`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/datatailor/b10/sel1234) | DataTailor (official code) | 10% | 66,071 | 1234 | `43a5eaf05dff650dc85d0564923af97273b6284921b81bcaa8419f1f7a79b88c` |
| [`selections/datatailor/b15/sel1234`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/datatailor/b15/sel1234) | DataTailor (official code) | 15% | 99,108 | 1234 | `06daec696605cfbb9133c343bfe7f9a932449ff1a6f083b0b803ade03850b9ee` |
| [`selections/datatailor/b20/sel1234`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/datatailor/b20/sel1234) | DataTailor (official code) | 20% | 132,147 | 1234 | `9f48baf83bbdcf6bb005518d1fcc205cd130e478ac85d4bc0689cd025feea82b` |
| [`selections/less_approx/b10/sel1234`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/less_approx/b10/sel1234) | LESS-approx (not faithful LESS) | 10% | 66,076 | 1234 | `23054b9f7f69bb63ec257549381dbcbac8e823a3ac7750a3c4b43c21771a03ce` |
| [`selections/source_sqrt/b10/sel1234`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/source_sqrt/b10/sel1234) | Source-sqrt (source-balanced random) | 10% | 66,076 | 1234 | `ce593e22db4f5c753ed4bd592b4d7a4260bb05681ac2ec6f9c0aec7c3a2c23fd` |
| [`selections/uniform/b05/sel1235`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/uniform/b05/sel1235) | Uniform random (matched-K) | 5% | 33,032 | 1235 | `c0b81ac856e5a0575ccfbf7370d39e5e9ff67a9615e21f0d389d7f1e2871c811` |
| [`selections/uniform/b10/sel1234`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/uniform/b10/sel1234) | Uniform random | 10% | 66,076 | 1234 | `563b8572a69730c229b309dde77bb146e15d098caf1d26bec7d1a83b787d45cd` |
| [`selections/uniform/b10/sel1235`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/uniform/b10/sel1235) | Uniform random | 10% | 66,076 | 1235 | `4b2104f8e32ababc98728823beb21f3083fd669be4507bdb9f35b3a191c3b459` |
| [`selections/uniform/b10/sel1236`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/uniform/b10/sel1236) | Uniform random | 10% | 66,076 | 1236 | `5573f3bb85a062460f36d93a8972e15bc0e2066a0bfd63d7b6979a40bcac8ae0` |
| [`selections/uniform/b15/sel1235`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/uniform/b15/sel1235) | Uniform random (matched-K) | 15% | 99,108 | 1235 | `4ec139524195f8fbff3d6cb2f01337ea23cf484a26c10063e34d7590c2be59a1` |
| [`selections/uniform/b20/sel1235`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/uniform/b20/sel1235) | Uniform random (matched-K) | 20% | 132,147 | 1235 | `9fce3a449186af63cae5ba24f1c4fbc309933217f50b0f88d7b546afb79da51f` |
| [`selections/uniform_scored_random/b10/sel1234`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections/tree/main/selections/uniform_scored_random/b10/sel1234) | Uniform scored-random (control) | 10% | 66,076 | 1234 | `7ec32ab31dcea9557d93a369ec799d88ce4273c922059018e8100918a9580a7e` |

## Not included / known gaps
* **Not uploaded:** base-model weights; any images (LLaVA-665K or benchmark); our own method's checkpoints and
  selections; optimizer/RNG/DeepSpeed states; incomplete or warm-up states (e.g. the ICONS warm-up adapter).
* **Uniform 20% training seed 29** — only a partial attempt (387/1032 steps) exists; not uploaded.
* **DataTailor 15% training seed 17** — only a partial attempt (215/774 steps) exists; not uploaded.
* **COINCIDE 20% training seeds 29 and 43** — trained (1032/1032 steps, uploaded) but **not evaluated**.
* **ICONS, VisNec, XMAS, CoIDO, PreSel** — no faithful run exists (ICONS has only a warm-up state); contract tasks T4, T9,
  T10 and optional T11.
* **Faithful LESS** — not run. `less_approx` uses count-sketch **SGD** gradient features (the LESS paper uses
  Adam-preconditioned features), a 3×Uniform candidate pre-filter and two benchmark-aligned dev targets (GQA-val,
  ScienceQA-val). It is therefore *not* LESS and is also not task-blind. Its set is shifted to short / yes-no answers
  and the models show a strong "yes" bias on POPE/MME.
* **Full-data (100%) reference** — no complete run exists (a partial run reached 2,675 / 5,162 steps on our cluster
  and is not released); contract task T8.
* **Historical allocation arms ("A-P", "A-M")** appearing in our internal tables are project-internal allocation
  variants, not published baselines; they are omitted here.
* **Source-sqrt** exists only at 10% (selection seed 1234). The 5%/15%/20% Uniform cells use selection seed 1235 with K
  matched to DataTailor's realised K; they are not nested with the 10% sel1235 set.
* **COINCIDE** is an adaptation: the authors' TinyLLaVA/SigLIP feature route and clustering (FAISS, 10,000 clusters,
  spherical k-means 50 iterations, temperature 0.1) applied to the full eligible LLaVA-665K pool; its realised K is
  66,076 / 33,038 / 132,152 at 10/5/20%. The selection is deterministic (no selection seed).
* **DataTailor** uses the official code (commit `d6fa5eb6bc2920a561c614abb6b841a128affb92`); its realised K at 10% is
  66,071 (five fewer than the nominal 66,076).
* Extended-suite coverage is incomplete (24 provided checkpoints; exact list in `AGENT_CONTRACT.md`, task T5). The 10% evaluation banks were used repeatedly
  during development; they are not a fresh held-out confirmation for anything.
* Timing and GPU-hour accounting for the provided runs are not part of this release; please record your own (see the results template).

## Licenses and attribution
Base-model lineage (no base weights are redistributed here): LLaVA-1.5-7B (`liuhaotian/llava-v1.5-7b` recipe) =
`lmsys/vicuna-7b-v1.5` (revision `3321f76e3f527bd14065daf69dad9344000a201d`, Llama 2 Community License) +
`openai/clip-vit-large-patch14-336` (revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`) + the stage-1 projector
`liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5`. The adapters are derivatives of Vicuna/Llama 2 and are
subject to the **Llama 2 Community License** and its acceptable-use policy, in addition to the LLaVA license terms.

Data: the selections are subsets of the LLaVA-1.5 instruction-tuning mixture `llava_v1_5_mix665k.json`
(LLaVA-Instruct / LLaVA-1.5 data, released by the LLaVA authors under **CC BY 4.0**; the original authors also ask
users to abide by the OpenAI terms of use for the GPT-generated portions). Component datasets (COCO, GQA, OCR-VQA,
TextVQA, Visual Genome, ShareGPT text) keep their own licenses. **No images are redistributed.**

Vendored upstream reference sources under `src/collaboration/*/reference_sources/` (LLaVA, COINCIDE, DataTailor)
keep their original licenses. The project-specific code in `src/` is research code released for reproduction.
