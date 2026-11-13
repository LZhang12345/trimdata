# Calibration report

Status refreshed on 2026-11-13: T0a passed for both checkpoints after full
fresh generation. T0b passed its 516-step retrain and full core evaluation.
Local T0c retry 2 passed all six metrics. Judge-dependent T0c remains deferred.
Final gate evidence is retained in `calibration/T0A_PASSED`,
`calibration/T0B_PASSED` and `calibration/T0C_LOCAL_PASSED`.

## Hardware and environment

- Instance hardware: eight NVIDIA A100-SXM4-40GB GPUs, confirmed by `nvidia-smi`.
- The project lead corrected the original 80GB assumption on 2026-11-12 and
  authorized continuing with the fixed recipe on 40GB.
- The shared environment has Python 3.13.18 and torch 2.14.1+cu130. It was not modified.
- Private native environment: Python 3.10.19, torch 2.1.2+cu121, xformers 0.0.23.post1,
  transformers 4.37.2, PEFT 0.7.1, DeepSpeed 0.12.6. Exact freeze is retained on the
  instance at `/opt/dlami/nvme/trimdata-work/control/native-freeze.txt`.
- The optimizer-group CPU test passes. A private SHA-verified CUDA 12.1 toolkit
  supplies the compiler required by the pinned dependencies.
- All 62 requirement pins match installed distribution versions exactly.
  Inventory, freeze and GPU/driver metadata are in `env/`.
- Extended evaluation alone uses a separate overlay: pandas 1.5.3,
  python-dateutil 2.8.2, pytz 2023.3. The original native pin list omits pandas,
  which the official c121 MMBench module imports. Original Python 3.10 wheels
  were verified against PyPI SHA256 and ZIP CRC before extraction. Wheel
  provenance is in `env/extended-overlay.json`; training retains all 62 native
  pins and excludes the overlay.

## Official scorer qualification (CPU, supplied predictions)

| Checkpoint | POPE | GQA normalized | TextVQA | ScienceQA-IMG |
|---|---:|---:|---:|---:|
| Uniform b10 sel1234-t17 | 84.769921 | 55.49 | 52.86 | 67.823500 |
| DataTailor b10 sel1234-t17 | 83.075196 | 57.52 | 53.65 | 66.534457 |

Each bank matches its published score to rounding precision (maximum absolute
delta <0.005 points). These checks allocate no GPU and do not qualify fresh
generation on A100 or the training implementation.

## Evaluator diagnosis and permitted retry

Uniform generation attempt 1 reported SDPA despite the wrapper passing the
required eager keyword. Stopped only its generation subprocess; the runner
retained the partial outputs and recorded 0.137182 failed-attempt GPU-hours.
No core score or gate pass is assigned to this attempt.

The pinned Transformers 4.37.2 loader passes an existing config through a
default getter that reports eager while the internal setting remains unset.
It therefore selects SDPA automatically. A CPU regression with the actual
vendored LLaVA class reproduced `LlamaSdpaAttention`, then verified that setting
the loaded config explicitly instantiates `LlamaAttention` without changing any
checkpoint file.

`eval_portable.py` now makes the intended eager setting explicit and checks all
instantiated attention layers before generation. Uniform attempt 2 and
DataTailor attempt 1 each report 32 eager attention layers and 224 active,
unmerged LoRA layers. The student training recipe and vendored sources are unchanged.

Training controller attempts 1 and 2 ended before allocating a GPU: the first
encountered an input-manifest creation race; the second stopped when T0a stopped.
Both appear as zero-cost setup failures in the CSV. Controller attempt 3
launched the first scientific T0b training run at 09:30 UTC on GPU2 using the
complete official image archives and reference early-loss checks.

## Fresh full-core evaluation (T0a)

| Checkpoint | POPE | GQA normalized | TextVQA | ScienceQA-IMG | Average | Average delta | GPU-hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform b10 sel1234-t17, attempt 2 | 84.680135 | 55.45 | 52.85 | 67.922657 | 65.225698 | −0.004302 | 2.912308 |
| DataTailor b10 sel1234-t17, attempt 1 | 83.063973 | 57.52 | 53.63 | 66.584036 | 65.199502 | +0.009502 | 2.409384 |

Both checkpoints answered all 28,505 questions and passed every per-bank
tolerance (±0.30 points) and average tolerance (±0.10 points). Answer agreement
with the supplied predictions ranges from 99.44% to 99.95% across banks.
Neither run raised the POPE yes-ratio alert. `control/T0A_PASSED` is present.
Completed attempt manifests are retained in `calibration/`; the results CSV
and run log have been refreshed from the instance.

## Training calibration (T0b), passed

The fixed recipe completed exactly 516 steps without changes on a single A100
40GB at 18:18:44 UTC. Training consumed 8.803642 GPU-hours. Observed peak GPU
memory was 31,973 MiB (31.2 GiB). Learning-rate group checks at steps 1, 10 and 100
passed; the step-10 LoRA learning rate is 0.000125.

| Step | Observed loss | Delta from reference |
|---|---:|---:|
| 10 | 1.0962 | +0.0001 |
| 20 | 1.0101 | +0.0001 |
| 30 | 0.8529 | −0.0002 |
| 40 | 0.8943 | +0.0001 |
| 50 | 0.8709 | +0.0257 |
| 60 | 0.9460 | +0.0001 |

All six checks pass the ±0.03 loss tolerance. Full core evaluation finished
at 20:45 UTC on November 12, consuming 2.451137 GPU-hours:

| Metric | Fresh score | Delta from contract reference | Answer agreement |
|---|---:|---:|---:|
| POPE | 84.646465 | +0.086465 | 98.65% |
| GQA normalized | 55.30 | −0.15 | 95.89% |
| TextVQA | 53.27 | −0.50 | 91.52% |
| ScienceQA-IMG | 67.426872 | +0.346872 | 96.98% |
| Core average | 65.160834 | −0.059166 | — |

The average passes ±0.30 and every bank passes ±1.0. No POPE yes-ratio alert.
The complete gate manifest is `calibration/T0B_PASSED`.

## Local extended scorer qualification and fresh T0c

| Metric | Supplied-prediction score | Delta from rounded reference |
|---|---:|---:|
| MME perception | 1360.194478 | −0.005522 |
| MME cognition | 311.785714 | −0.014286 |
| SEED image | 59.87 | 0 |
| MMMU validation | 32.777778 | −0.002222 |
| MMBench-EN dev CircularEval | 58.758503 | −0.001497 |
| VizWiz test | 43.43 | 0 |

CPU qualification attempt 1 stopped at missing pandas and consumed zero
GPU-hours. After the diagnosed overlay setup, attempt 2 passed all six scorers
using the full supplied predictions. Its exact outputs are in
`calibration/extended-cpu-scorer-a2.json`; the zero-cost setup failure is in the
CSV and run log.

MME uses the unchanged native text conversion and original calculation script.
SEED uses the unchanged official image-only converter, including its single
duplicate-ID last-wins behavior. MMMU uses the original validation gold file
and official parser/evaluator. MMBench uses exact `can_infer` extraction and
the primary TSV CircularEval score (691/1,176); explicit rotations
(683/1,176) are retained separately. VizWiz uses the pinned answer processor
and official accuracy class; its supplied unanswerable prediction rate is
28.15%.

Fresh local generation attempt 1 started on GPU6 at 18:59 UTC through
`eval_portable.py`. It completed MME perception generation, then failed at
19:17 UTC while loading MME cognition. The trace identifies three other CUDA
processes occupying 9.72/9.72/19.60 GiB on physical GPU6, leaving 27.81 MiB
free when the loader tried its first 250 MiB allocation. Failed cost:
0.301205 GPU-hours. No fresh score or numerical gate failure was assigned.

The diagnosed remedy was the single permitted fresh retry, initially on GPU4.
It confirms an empty device twice before every module reload, including CUDA
process ownership, and records each assignment. Only one GPU is used
concurrently; waits for an available device are excluded from GPU-hours.
Generation completed with actual eager attention confirmed. No single-model
40GB blocker was established; the complete training calibration peaked at
31.2 GiB. The failed attempt, retry state and diagnosis are retained in
`calibration/` and `provenance/`.

The complete retry passed at 22:21 UTC on November 12, consuming 2.848635
GPU-hours:

| Metric | Fresh score | Delta from reference | Allowed absolute delta |
|---|---:|---:|---:|
| MME perception | 1357.743497 | −2.456503 | 15 |
| MME cognition | 311.785714 | −0.014286 | 10 |
| SEED image | 59.94 | +0.07 | 0.5 |
| MMMU validation | 32.555556 | −0.224444 | 0.5 |
| MMBench-EN dev | 58.928571 | +0.168571 | 0.5 |
| VizWiz test | 43.48 | +0.05 | 1.0 |

All six local tolerances pass; evidence is `calibration/T0C_LOCAL_PASSED`.
LLaVA-W and MM-Vet generation/judging are deferred, so overall T0c remains
incomplete.

The official MMBench module's option/hint handling, integer index JSON
serialization and all ordered original/rotation inputs also passed a native
CPU check. Generation uses the unchanged official passes (4,377 original
rows plus 4,430 base rotations); scoring deduplicates by exact image bytes and
prompt, retaining the same 4,532-input union as the supplied run. This
additional generation work is included in the measured evaluation cost.

## Calibration gates

- T0a: passed (Uniform sel1234-t17 attempt 2 and DataTailor t17 attempt 1).
- T0b: passed (Uniform sel1235-t17); early checks and full evaluation passed.
- T0c local: passed on fresh retry 2 (MME-P, MME-C, SEED-img, MMMU,
  MMBench, VizWiz).
- T0c judged: deferred by the project lead (LLaVA-W, MM-Vet).

No deferred metric is reported as passed or assigned an invented score.
