# AGENTS.md — start here (for AI agents and new collaborators)

You are taking over part of a research study on **instruction-data selection for LLaVA-1.5-7B and 13B**. This file is the
entry point: what the project is, where everything is, what is done, what you are expected to do, and the practical
lessons learned so far. It is written so that an agent with no prior context can start correctly.

## 0. Read in this order
1. **This file** — orientation, map, status, **run order (§3a)**, lessons.
2. [`docs/COAUTHOR_NEXT_WAVE_20261113.md`](docs/COAUTHOR_NEXT_WAVE_20261113.md) — concise current assignment,
   missing-cell roster, ready versus gated work, exclusions and hand-back.
3. [`AGENT_CONTRACT.md`](AGENT_CONTRACT.md) — your **binding work order** for 8× A100 40GB. Read §1a first;
   it overrides older task-order text. Tasks T0–T13, calibration, protocol and reporting are defined here.
   Machine-readable: [`agent_contract.json`](agent_contract.json).
4. [`README.md`](README.md) — reference recipe, historical results and checkpoint/selection index.
5. Only then open `src/` (see the map below) for the exact code that produced each result.

If this file, the contract and the README disagree: **contract > README > this file**. Ask the project lead about any
remaining conflict instead of guessing.

Project-lead updates (2026-11-12): the instance has 40GB GPUs. LLaVA-W and MM-Vet judge experiments are
deferred; continue core-four and locally scored extended work with the corresponding calibration gates.
Before launching, read the coauthor results and run log and inspect live owned processes: detached controllers
may already be running. `results/coauthor/INSTANCE_STATUS.md` was referenced by the previous agent but was not
committed at the audited handover. If present locally, reconcile it; if absent, reconstruct current status from
the instance and record it rather than assuming no work is active or blocking all work on the missing file.

## 1. The study in one paragraph
Fine-tune LLaVA-1.5-7B (stage 2, LoRA, one natural epoch, fixed recipe) on a **subset** of the 660,763 eligible
conversations of `llava_v1_5_mix665k.json`, chosen by a data-selection method under a fixed **selected-data budget K**
(5% = 33,032; 10% = 66,076, the anchor; 20% = 132,147; 15% is historical only). Compare selection methods by the trained model's
scores on four full official benchmarks (POPE, GQA, TextVQA, ScienceQA-IMG; 28,505 questions) and an eight-benchmark
extended suite. Provided baselines: Uniform random (the random baseline), source-sqrt, COINCIDE, DataTailor,
LESS-approx (not faithful LESS). Still to run (your tasks): ICONS, VisNec, XMAS, CoIDO, a full-data reference, optionally faithful
LESS and PreSel. The project's own method is developed elsewhere and is **not** part
of this repository; your job is the baselines.

## 2. Map of everything
### GitHub (this repository)
| Path | What it is | When you need it |
|---|---|---|
| `AGENTS.md` | this orientation file | first |
| `AGENT_CONTRACT.md`, `agent_contract.json` | binding task list + protocol | before any action |
| `README.md` | recipe, eval protocol, results, indices, gaps, licenses | reference |
| `results/baseline_core4_per_run.csv` | per-run core-four scores of every provided checkpoint | calibration (T0), comparison |
| `results/baseline_extended_reported.csv` | extended-suite means reported so far | T5 |
| `results/RESULTS_TEMPLATE.csv` | **the exact CSV schema you must return** | outputs |
| `index/checkpoints.csv`, `index/selections.csv` | machine-readable indices of the HF artifacts (paths, K, seeds, sha256) | locating inputs |
| `scripts/reconstruct_records.py` | rebuilds a selection's training `records.json` byte-for-byte from the public LLaVA-665K json | before every training run |
| `scripts/verify_sha256.py` | checks downloaded files against `SHA256SUMS` | after every download |
| `scripts/train_portable.py` | **the only allowed training entry point**: official LLaVA c121 trainer + the three runtime changes our runs used (see lesson 19); single-GPU by default, aborts on wrong step count or wrong LR groups | every training run |
| `scripts/eval_portable.py` | runs an official `llava.eval.*` generation module with our inference settings (LoRA unmerged, eager attention, fp16) | every evaluation |
| `scripts/zero3.json` | DeepSpeed ZeRO-3 config (byte-identical to the executed one) | used by `train_portable.py` |
| `tests/test_optimizer_group_lr.py` | CPU-only demonstration of the optimizer-group bug and the fix (seconds) | sanity check after setup |
| `scripts/reference_train_command.sh` | read-only list of the training arguments; **do not train with it** | reference |
| `env/requirements-native-env.txt` | exact package pins of the executed environment (incl. `xformers==0.0.23.post1`) | setup |
| `docs/EVAL_INPUTS.md` | evaluation input files and hashes | evaluation |
| `PATHS.md` | cluster paths that appear inside `src/` and what to substitute | reading `src/` |
| `SOURCE_MANIFEST.json` | sha256 of every file in `src/` (and its pre-rename hash) | provenance |
| `src/` | byte-level snapshot of the code that actually ran (see below) | exact details |

**Inside `src/` (names are historical; `helper_toolN`, `copilotN`, `root`, `session`, `manager` are internal worker
labels, not meaningful):**
| Need | Look at |
|---|---|
| Training arguments (the recipe) | `src/experiments/helper_tool1_baseline_train_v4_20261102/model.py::arguments`; same recipe in `src/experiments/root_native_campaign_v7_20261024/config.py` |
| Training driver used for 5%/15%/20% and newer cells | `src/experiments/helper_tool1_baseline_train_v10_20261108/` |
| Core-four evaluator | `src/experiments/helper_tool1_baseline_eval_v4_20261105/` (+ `_v5_`, `_t3coin_v2_` variants for specific producers) |
| Training runtime changes reused by `train_portable.py` | `src/experiments/root_native_campaign_v7_20261024/{structured_preprocess,optimizer_groups,runtime_guards}.py`; deterministic CUTLASS attention: `root_native_deterministic_attention_v{1,2}_*` |
| Extended-suite bank definitions, prompts, judge | `src/experiments/helper_tool1_bench6_eval_v1_20261103/banks6.py` (MME, SEED), `helper_tool1_bench4_eval_v1_20261104/banks4.py` (MMMU, LLaVA-W, MM-Vet, MMBench), `helper_tool1_bench4_judge_v13_20261105/judge4_v13.py` (judge prompts/settings), `helper_tool1_vizwiz_score_v1_20261101/scorer.py`; drivers: `helper_tool1_bench4_*`, `helper_tool1_bench6_*`, `helper_tool1_cells_v7_extsuite_20261110` |
| Vendored LLaVA c121 (`llava/` package) | `src/collaboration/published-foundation-20261022/reference_sources/LLaVA/c121f0432da27facab705978f83c4ada465e46fd/` |
| Uniform / source-sqrt selector | `src/experiments/root_native_campaign_v7_20261024/selection.py`; matched-K Uniform: `helper_tool1_uniform_matchedk_build_v1_20261102` |
| COINCIDE (adapted) | `src/experiments/copilot7_coincide_native_*` and `src/collaboration/*/reference_sources/` (upstream code) |
| DataTailor (official) | `src/experiments/root_dt_formal_selection_v2_20261026`, `copilot7_datatailor_*`, upstream in `reference_sources/` |
| LESS-approx | `src/experiments/helper_tool2_less_approx_v2_20261105`, `helper_tool2_less_select_v1_20261106` |

Most drivers also contain cluster bookkeeping (Slurm headers, ownership/locking/receipt checks). You do not need it;
the scientific parts are the trainer arguments, the selectors and the scorers.

### Hugging Face
| Repo | Content | Layout |
|---|---|---|
| [`submissionpaper1234/trimdata-baseline-checkpoints`](https://huggingface.co/submissionpaper1234/trimdata-baseline-checkpoints) (model) | 46 complete one-epoch baseline runs: LoRA adapter, trained projector (`non_lora_trainables.bin`), portable + original configs, `trainer_state.json`, `provenance.json`; per-run evaluation outputs under `eval/` where available | `<method>/<bNN>/<selXXXX or selfixed>-t<seed>/`; root `index.csv`, `index.json`, `SHA256SUMS`, model card |
| [`submissionpaper1234/trimdata-baseline-selections`](https://huggingface.co/datasets/submissionpaper1234/trimdata-baseline-selections) (dataset) | 16 selections (ordered record ids + `selection_meta.json`), the eligible pool ids, `reconstruct_records.py` | `selections/<method>/<bNN>/<selXXXX or selfixed>/`, `pool/pool_record_ids.json`, `selections_index.csv`, `SHA256SUMS` |

Method folder names: `uniform`, `uniform_scored_random` (control), `source_sqrt`, `coincide`, `datatailor`,
`less_approx`. Budget folders: `b05`, `b10`, `b15`, `b20`. `selfixed` = deterministic method without a selection seed.

Download only what a task needs, e.g.
`hf download submissionpaper1234/trimdata-baseline-checkpoints --include "uniform/b10/sel1234-t17/*" --local-dir ckpts`.

## 3. Status at hand-over
* **Complete and provided:** 46 checkpoints (44 evaluated on the core four; COINCIDE 20% t29/t43 are not). Full 3-seed coverage at 10% for Uniform
  (3 selection sets × 3 training seeds), source-sqrt, COINCIDE, DataTailor, LESS-approx and Uniform scored-random;
  5% Uniform/DataTailor/COINCIDE ×3; 15% Uniform ×3, DataTailor t29/t43; 20% DataTailor ×3, COINCIDE ×3, Uniform t17/t43.
* **Evaluation outputs provided:** every evaluated checkpoint folder has `eval/core4/<bank>/PREDICTIONS.jsonl` and
  `SCORE.json`; 27 also have extended-suite predictions/scores (22 with all eight). Use them for calibration.
* **Your tasks:** run them in the order of §3a below (contract tasks T0–T13). **Plan amended 2026-11-13** — read `AGENT_CONTRACT.md` §1a first: coverage before seeds (seed 17 first), budgets 5/10/20% (15% dropped, T3 cancelled), new baselines go to all three budgets, and new task T13 = LLaVA-1.5-13B transfer.
* **Done by 2026-11-13 (`results/coauthor/`):** T0a/T0b/T0c-local passed; T1 core for COINCIDE 20% t29/t43; 8 of 24 T5 gaps.
* **Not yours:** the project's own method, the paper, any change of protocol.

## 3a. Run order (execute in this sequence)
Scheduling rule: **whenever a GPU becomes free, start the highest-priority item in the queue below whose dependencies are
met.** Never start an item before its gate passes. Items marked *[parallel]* may run alongside earlier items. Everything
is single-GPU except T13 (2 GPUs) and T8 (4–8 GPUs). Gates, tolerances and commands are in `AGENT_CONTRACT.md` §4–§6.

| P | Item | Contract task | Depends on | GPUs | Done when |
|---|---|---|---|---|---|
| 0–3 | Setup and calibration T0a/T0b/T0c-local — **done 2026-11-12** | T0 | — | — | passed |
| 4 | **Diagnose the 2026-11-13 failure bursts** (04:23 and 06:27 UTC; CUDA launch failures; T3 crash at step 460): `dmesg`/Xid, `nvidia-smi -q`, other processes on those GPUs. Record cause and remedy | — | — | 0 | cause in `RUN_LOG.md` |
| 5 | *[parallel]* Selection-side compute, in this order: **VisNec** (+ fidelity check) → ICONS → XMAS → CoIDO; produce 10%, 5%, 20% selections each | T12, T4, T9, T10 | Setup; P4 additionally required for long GPU work | up to 4 | `record_ids.json` + `selection_meta.json` per method × budget |
| 6 | **VisNec 10% t17**, then 5% t17, then 20% t17 + evaluation (required) | T12 | P4, VisNec selections | 1 each | 3 rows + `VISNEC_FIDELITY.md` |
| 7 | ICONS 10%, 5%, 20% t17 + evaluation | T4 | P4, ICONS selections | 1 each | 3 rows + `ICONS_FIDELITY.md` |
| 8 | XMAS 10%, 5%, 20% t17 + evaluation | T9 | P4, XMAS selections | 1 each | 3 rows + `XMAS_FIDELITY.md` |
| 9 | CoIDO 10%, 5%, 20% t17 + evaluation | T10 | P4, CoIDO selections | 1 each | 3 rows + `COIDO_FIDELITY.md` |
| 10 | **T0d 13B calibration — retrain only** (revised 2026-11-14; no HF checkpoint will be posted; see contract §5) | T0d | `selections/uniform/b05/sel1235`, 13B pins, reference losses/scores in contract §5 | 2 (retrain) | losses, average and banks within §5 tolerances |
| 11 | **13B transfer, seed 17:** Uniform, DataTailor, COINCIDE at 5% and 10%; then VisNec/ICONS/XMAS/CoIDO 10% | T13 | P10 | 2 per run | rows + checkpoints |
| 12 | Full-data reference (660,763 records, 5,162 steps, seed 17; multi-GPU allowed, global batch 128) + evaluation | T8 | P4 + manager ownership/live-continuation reconciliation; then ≥4 free GPUs | 4–8 | step 5,162; row |
| 13 | Seeds 29 and 43 for the P6–P9 cells (10% first) | T12, T4, T9, T10 | P6–P9 | 1 each | rows |
| 14 | Retry still-missing, non-15% T5 banks after diagnosis, then Uniform 20% t29; skip completed banks | T5, T2 | P4 | 1 each | rows |
| 15 | *(optional)* Faithful LESS 10% t17; PreSel-adapted plan (approval first) | T6, T11 | P4 | 1 each | rows + fidelity notes |
| — | ~~DataTailor 15% t17 (T3), source-sqrt 5/20% (T7)~~ — dropped 2026-11-13 | — | — | — | — |
| 16 | Hand-back: `results/coauthor/*`, fidelity notes, HF uploads, pull request | — | everything above that ran | 0 | PR opened |

Continuously: evaluate every new checkpoint (core four, then the six local extended metrics; judged banks deferred)
as soon as it finishes; log every attempt
(including failures and GPU-hours) in `RUN_LOG.md` the moment it ends. A failed run is retried once as a new attempt
after diagnosing the cause; a second failure → stop that item and ask.

Layout from 2026-11-13: after P4, up to 4 GPUs on selection-side compute (P5) and the rest on students as selections
finish (P6–P9); T13 pairs of GPUs once T0d passes; T8 when ≥4 GPUs are free together.
T13 is selected-set transfer, not a native 13B selector reproduction. Its two-GPU layout still requires qualification.
Blocked T13 or T8 does not block other ready work. Do not duplicate manager-owned full-data trajectories.

## 4. Lessons learned (read before running anything)
**Protocol and scoring**
1. **GQA main score = the native normalized score** produced after LLaVA's `convert_gqa_for_eval`
   (`pred.rstrip('.').lower()`), not the unconverted-basis score. Mixing the two silently shifts GQA by several points.
2. **Step count is a correctness check.** `natural_steps = (K + 3) // 4 // 32` at global batch 128. A run that stops
   early, resumes with a different batch layout, or ends at another step count is not comparable.
3. **Keep a method's realised K.** DataTailor yields 66,071 at 10% (not 66,076); COINCIDE yields 33,038 / 132,152 at
   5% / 20%. Never top up or trim to the nominal K, and never claim "same exact K" for them.
4. **Selected-data budget ≠ compute budget.** Do not change epochs or steps to equalise compute.
5. **Uniform *is* the random baseline.** Do not report it twice under two names. The three Uniform selection sets at
   10% are seed replicates of one method.
6. **Labels must match fidelity.** `less_approx` (count-sketch SGD gradients, 3×Uniform pre-filter, two benchmark-aligned
   targets) is *not* LESS; COINCIDE here is an adaptation of the authors' feature route to this pool. Any new deviation
   needs its own label.
7. **Evaluate LoRA unmerged** (PEFT 0.7.1, fp16, eager attention) to reproduce our numbers; the official builder merges
   the LoRA and only takes the LoRA branch if the folder name contains "llava" and "lora" (ours do not).
   `scripts/eval_portable.py` handles all three. T0a tells you whether anything else matters on your hardware.
8. **Extended-suite judge** is `gpt-4o-2024-08-06` (temperature 0, one run), not the original GPT-4 judges, so
   LLaVA-W / MM-Vet numbers are not comparable with published ones.

**Variance and interpretation**
9. Seed-to-seed spread of the core average at 10% is ≈0.2–1.0 points (e.g. DataTailor 65.19 / 66.15 / 65.65). Differences
   between methods of a few tenths are within noise; always report all seeds and paired comparisons.
10. **VizWiz is calibration-dominated.** The share of "unanswerable" predictions varies 6–31% across models and moves the
    score by up to ~10 points; record that rate with every VizWiz score.
11. **Yes-bias check.** LESS-approx selects short / yes-no answers and produces a strong "yes" bias (POPE yes-ratio 0.59–0.67
    vs ≈0.4; MME perception collapses). Check POPE yes-ratios when a selector favours short answers.
12. The 10% evaluation banks were used repeatedly during development; treat them as a reporting benchmark, not as a fresh
    held-out confirmation.

**Engineering**
13. **Verify base-model files before every job.** Our runs repeatedly failed within seconds because a shared cache cleanup
    removed Hugging Face snapshot files/links (e.g. the Vicuna `config.json`) under running pipelines. A 10-second
    pre-flight check that every base-model file exists and hashes correctly saves hours.
14. **One output directory per attempt.** A crashed attempt that leaves an empty output directory can make the next
    attempt fail on "directory exists". Use a fresh directory per attempt and keep failed ones (renamed) as evidence.
15. **Bad nodes happen.** If several jobs die at CUDA initialisation on the same node, exclude that node and retry; do not
    count it as a method failure (but do log it and its GPU-hours).
16. **Rebuild, do not re-host, training text.** Selections are distributed as ordered record ids; `reconstruct_records.py`
    rebuilds `records.json` byte-for-byte (sha256 in `selection_meta.json`). The raw LLaVA-665K text contains third-party
    code snippets that trip credential scanners, which is why it is not re-hosted.
17. **Resume only with full state.** If you checkpoint mid-epoch (preemptible hardware), resume must restore optimizer,
    scheduler, RNG and data-loader position; otherwise restart the run. Verify on one run before relying on it.
18. **Account for everything.** Record GPU-hours for selection, training, evaluation and failed attempts separately; the
    results template has columns for each.
19. **Never train with plain LLaVA `train.py`/`train_mem.py`.** Our runs added three runtime changes: (a) LLaVA's xformers
    attention patch with CUTLASS kernels and deterministic settings; (b) structured v1 label masking (the official
    `preprocess_v1` splits on text and mis-masks turns containing a role marker such as "USER:"); (c) removal of the
    empty LoRA-bias optimizer group before the scheduler is built. Without (c), DeepSpeed 0.12.6 drops that group after
    the cosine scheduler captured four learning rates, and with torch 2.1.2's silent `zip` the **projector weights
    train at 2e-4 instead of 2e-5** (reproduced in a CPU simulation). `scripts/train_portable.py` applies all three
    and aborts if the projector/LoRA LR ratio is not 0.1.
20. **Calibrate against the right producer.** The 10% Uniform-sel1234 / source-sqrt / COINCIDE runs came from an older
    driver that used xformers' default kernels; the Tool1-v4 runs (Uniform sel1235/1236, DataTailor, budget cells)
    forced deterministic CUTLASS, like `train_portable.py`. That is why T0b retrains `uniform/b10/sel1235-t17`.
21. **One GPU per run on A100s.** Run eight independent single-GPU jobs side by side. Launching the reference command on
    eight GPUs gives global batch 1,024 and a different step count; only the full-data reference may use data
    parallelism, with global batch kept at 128.

## 5. How to report back
Follow `AGENT_CONTRACT.md` §7: `results/coauthor/RESULTS.csv` (template schema, one row per run plus one per failed
attempt), `RUN_LOG.md`, `CALIBRATION_REPORT.md`, fidelity notes for ICONS/XMAS/CoIDO (and LESS/PreSel if run), and new checkpoints/selections uploaded
to a Hugging Face repo you control in the same layout. Open a pull request to this repository; do not commit weights,
tokens or API keys.
