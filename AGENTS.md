# AGENTS.md — start here (for AI agents and new collaborators)

You are taking over part of a research study on **instruction-data selection for LLaVA-1.5-7B**. This file is the
entry point: what the project is, where everything is, what is done, what you are expected to do, and the practical
lessons learned so far. It is written so that an agent with no prior context can start correctly.

## 0. Read in this order
1. **This file** — orientation, map, status, lessons.
2. [`AGENT_CONTRACT.md`](AGENT_CONTRACT.md) — your **binding work order** (scope, inputs, fixed protocol, calibration
   gate, tasks T0–T7, outputs, acceptance criteria, stop-and-ask rules). Machine-readable: [`agent_contract.json`](agent_contract.json).
3. [`README.md`](README.md) — full recipe, evaluation protocol, results tables, checkpoint/selection index, known gaps.
4. Only then open `src/` (see the map below) for the exact code that produced each result.

If this file, the contract and the README disagree: **contract > README > this file**. Ask the project lead about any
remaining conflict instead of guessing.

## 1. The study in one paragraph
Fine-tune LLaVA-1.5-7B (stage 2, LoRA, one natural epoch, fixed recipe) on a **subset** of the 660,763 eligible
conversations of `llava_v1_5_mix665k.json`, chosen by a data-selection method under a fixed **selected-data budget K**
(5% = 33,032; 10% = 66,076, the anchor; 15% = 99,108; 20% = 132,147). Compare selection methods by the trained model's
scores on four full official benchmarks (POPE, GQA, TextVQA, ScienceQA-IMG; 28,505 questions) and an eight-benchmark
extended suite. Baselines: Uniform random (the random baseline), source-sqrt, COINCIDE, DataTailor, LESS (only an
approximation exists so far), ICONS (not yet run). The project's own method is developed elsewhere and is **not** part
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
| `scripts/reference_train_command.sh`, `scripts/zero3.json` | the training recipe restated for the official LLaVA entry point (a reconstruction for convenience, not the executed driver) | training |
| `env/requirements-native-env.txt` | package pins of the executed environment | setup |
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
| Extended-suite evaluators / scorers / judge | `src/experiments/helper_tool1_bench4_*`, `helper_tool1_bench6_*`, `helper_tool1_cells_v7_extsuite_20261110`, `helper_tool1_vizwiz_score_v1_20261101` |
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
* **Complete and provided:** 46 checkpoints (43 evaluated on the core four). Full 3-seed coverage at 10% for Uniform
  (3 selection sets × 3 training seeds), source-sqrt, COINCIDE, DataTailor, LESS-approx and Uniform scored-random;
  5% Uniform/DataTailor/COINCIDE ×3; 15% Uniform ×3, DataTailor t29/t43; 20% DataTailor ×3, COINCIDE ×3, Uniform t17/t43.
* **Your tasks** (details and order in the contract): T0 calibration gate → T1 evaluate COINCIDE 20% t29/t43 → T2 Uniform
  20% t29 → T3 DataTailor 15% t17 → T4 faithful ICONS (10%, then 5%/20%) → T5 missing extended-suite scores →
  optional T6 faithful LESS, T7 source-sqrt at 5%/20%.
* **Not yours:** the project's own method, the paper, any change of protocol.

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
7. **Evaluate LoRA unmerged** (PEFT 0.7.1, fp16) to reproduce our numbers; the official builder merges the LoRA and can
   differ slightly. T0a tells you whether that matters on your hardware.
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

## 5. How to report back
Follow `AGENT_CONTRACT.md` §6: `results/coauthor/RESULTS.csv` (template schema, one row per run plus one per failed
attempt), `RUN_LOG.md`, `CALIBRATION_REPORT.md`, fidelity notes for ICONS/LESS, and new checkpoints/selections uploaded
to a Hugging Face repo you control in the same layout. Open a pull request to this repository; do not commit weights,
tokens or API keys.
