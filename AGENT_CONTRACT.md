# Agent contract — baseline completion for `trimdata`

This file is the **binding work order** for any agent (human or AI) that takes over the baseline side of this study.
Read it fully before acting. If anything here conflicts with `README.md`, this contract wins; if it conflicts with an
explicit instruction from the project lead, ask the lead. A machine-readable copy of the task list is in
[`agent_contract.json`](agent_contract.json).

## 1. Role and scope
* **You own:** the published baselines and controls listed in §5 — selection (only where stated), training,
  evaluation, results and cost accounting.
* **You do not own:** the project's own selection method (not in this repo), the paper text, or any change to the
  protocol in §3. Do not add new baselines, budgets, benchmarks or hyper-parameter variants without written approval.
* **Hardware assumed:** 8× NVIDIA A100. The provided runs were made on NVIDIA L40S (1 GPU per run); task T0 checks that
  your hardware reproduces them before anything else is spent.

## 2. Inputs (read-only; verify hashes before use)
| Input | Location | Verification |
|---|---|---|
| Code snapshot (selectors, trainer, evaluator, scorers) | this repository, `src/` | `SOURCE_MANIFEST.json` (sha256 per file) |
| Baseline checkpoints (46 complete one-epoch LoRA adapters + projector) | HF model repo `submissionpaper1234/trimdata-baseline-checkpoints` | `provenance.json` per folder, `SHA256SUMS`, `index/checkpoints.csv` |
| Baseline selections (16 selections + eligible pool) | HF dataset repo `submissionpaper1234/trimdata-baseline-selections` | `selection_meta.json` (records sha256), `SHA256SUMS`, `index/selections.csv` |
| Training data | `llava_v1_5_mix665k.json` (HF `liuhaotian/LLaVA-Instruct-150K`, revision `9d451dc7629cfe0469f6ae4432b765cd603d5fcb`, sha256 `ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8`) + official LLaVA image folders (coco/train2017, gqa, ocr_vqa, textvqa/train_images, vg) | rebuild each selection's `records.json` with `scripts/reconstruct_records.py`; it must match `selection_meta.json` byte-for-byte |
| Base model | `lmsys/vicuna-7b-v1.5` @ `3321f76e3f527bd14065daf69dad9344000a201d`, `openai/clip-vit-large-patch14-336` @ `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`, stage-1 projector `liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5` | see `README.md` → Licenses and attribution |
| Evaluation inputs | LLaVA `eval.zip` (commit `c121f0432da27facab705978f83c4ada465e46fd`) + official benchmark images | `docs/EVAL_INPUTS.md` |
| Reference scores | `results/baseline_core4_per_run.csv`, `results/baseline_extended_reported.csv` | — |

If any hash does not match, **stop** and report; do not substitute a "similar" file.

## 3. Fixed protocol (invariants — never change)
1. **Model / code:** LLaVA-1.5-7B stage-2 with the official `llava/train/train.py` at commit `c121f0432da27facab705978f83c4ada465e46fd`,
   conversation template `v1`.
2. **Recipe:** LoRA r=128, alpha=256, dropout 0.05 on all LLM linear layers; projector trained (LR 2e-5); LoRA LR 2e-4;
   AdamW, weight decay 0; cosine schedule, warm-up ratio 0.03; **global batch 128**; `model_max_length` 2048;
   `image_aspect_ratio=pad`; `group_by_modality_length=True`; fp16; DeepSpeed ZeRO-3; gradient checkpointing; eager
   attention. Training seed ∈ {17, 29, 43} sets both `seed` and `data_seed`.
3. **Exactly one natural epoch:** `natural_steps = (K + 3) // 4 // 32` → 258 (5%), 516 (10%), 774 (15%), 1032 (20%).
   A run is *complete* only if its final global step equals `natural_steps`. Multi-GPU is allowed only if the global
   batch stays 128 and the step count is unchanged; record the layout you used.
4. **Selected-data budget (K) is a data budget, not a compute budget.** Budgets: 5% K=33,032; 10% K=66,076;
   15% K=99,108; 20% K=132,147. Methods with a realised K (DataTailor 66,071 at 10%; COINCIDE 33,038 / 132,152 at 5% / 20%)
   keep it and report it. Never pad or truncate a method's selection to hit K exactly, and never change epochs or steps to
   match compute.
5. **Evaluation (core four, 28,505 questions, greedy, fp16, batch 1, LoRA unmerged):** POPE 8,910 (pooled accuracy);
   GQA test-dev balanced 12,578 (**main = native normalized score** after `convert_gqa_for_eval`, not the unconverted
   score); TextVQA val 5,000 (official m4c evaluator); ScienceQA-IMG 2,017 (official script). Core avg = unweighted mean
   of the four. Extended eight: MME-P, MME-C, SEED-img, MMMU val, MMBench-EN dev, LLaVA-W, MM-Vet (judge
   `gpt-4o-2024-08-06`, temperature 0, one run), VizWiz test (local scoring; also record the "unanswerable" rate).
   Details: `README.md` → Evaluation protocol.
6. **Baseline fidelity:** official code and published defaults. No tuning of any baseline's hyper-parameters, features,
   selection units or quotas. Any unavoidable adaptation gets a **new label** (e.g. `ICONS-adapted`) and a written
   description; never publish an adaptation under the original method name.
7. **No leakage:** never use evaluation-bank labels or answers for selection, early stopping, checkpoint choice or
   tuning. If a method needs target/validation tasks (ICONS, LESS), use only the sets the method prescribes and record them.
8. **Report everything:** every seed and every attempt, including failures and partial runs, with GPU-hours.

## 4. Gate (must pass before any other task)
**T0 — hardware calibration.**
* **T0a:** re-evaluate the provided `uniform/b10/sel1234-t17` and `datatailor/b10/sel1234-t17` with the core four.
  Pass if the core avg is within **±0.10** and each bank within **±0.30** of `results/baseline_core4_per_run.csv`
  (Uniform: 65.23 = POPE 84.77 / GQA 55.49 / TextVQA 52.86 / SQA-IMG 67.82). Evaluate with the LoRA unmerged
  (PEFT 0.7.1) as we did; if you also try the merged builder, report both.
* **T0b:** retrain `uniform/b10/sel1234-t17` from `selections/uniform/b10/sel1234` (K=66,076, 516 steps) and evaluate it.
  Pass if the final step is 516, the core avg is within **±0.30** of 65.23 and no bank is off by more than **±1.0**.
* If T0a or T0b fails: **stop** and report the deltas, environment and logs. Do not start T1 or later.

## 5. Tasks (priority order; do not start a task before its dependencies pass)
| ID | Task | Depends on | Inputs | Deliverable | Approx. A100 GPU-h |
|---|---|---|---|---|---|
| T0 | Calibration (above) | — | 2 checkpoints + 1 selection | 3 result rows + `CALIBRATION_REPORT.md` | ~12 |
| T1 | Evaluate COINCIDE 20% t29 and t43 (core four + extended eight); extended eight for COINCIDE 20% t17 | T0 | `coincide/b20/selfixed-t{17,29,43}` | 3 rows | ~8 |
| T2 | Train + evaluate **Uniform 20% t29** (`selections/uniform/b20/sel1235`, K=132,147, 1032 steps) | T0 | selection | checkpoint + row | ~18 |
| T3 | Train + evaluate **DataTailor 15% t17** (`selections/datatailor/b15/sel1234`, K=99,108, 774 steps) | T0 | selection | checkpoint + row | ~14 |
| T4 | **ICONS, faithful**: 10% first (3 training seeds), then 5% and 20% (3 seeds each). Official code at commit `b8ce8c86d7b098836a84ec2b40d259195b7f4494`: its warm-up, gradient features, influence-consensus selection and prescribed target tasks; then train with §3 | T0 | eligible pool ids, official repo | 3 selections + 9 checkpoints + 9 rows + `ICONS_FIDELITY.md` | ~150–250 |
| T5 | Extended eight for every provided checkpoint that lacks them (list: `README.md` → Coauthor plan, step 5) and for all new T2–T4 checkpoints | T0 | checkpoints | rows | ~3 per checkpoint |
| T6 (optional) | **Faithful LESS** (official pipeline, commit `8abf9628b9a814ac3045445eebc8ba3c908fdc78`, paper defaults) at 10%, 3 seeds; `less_approx` keeps its own label | T0 | pool, official repo | 1 selection + 3 checkpoints + rows + `LESS_FIDELITY.md` | ~100 |
| T7 (optional) | Source-sqrt at 5% and 20% (selector `src/experiments/root_native_campaign_v7_20261024/selection.py`, method `source_sqrt`, selection seed 1234), 3 seeds each | T0 | pool | 2 selections + 6 checkpoints + rows | ~90 |

Do **not** retrain anything that already has a complete checkpoint in the model repo. Each (method, budget, selection
seed, training seed) cell is trained successfully at most once; a repeat is allowed only as a documented retry of a
failed attempt.

## 6. Outputs (what you hand back)
1. **Results CSV:** `results/coauthor/RESULTS.csv` with exactly the columns of `results/RESULTS_TEMPLATE.csv`, one row per
   (method, budget, selection seed, training seed), plus **one row per failed attempt** (scores empty, cost filled).
   Scores in percent with two decimals; MME in points; `global_steps` and `natural_steps` filled; GPU-hours per stage.
2. **Checkpoints:** upload new adapters (`adapter_model.safetensors`, `adapter_config.json`, `non_lora_trainables.bin`,
   `config.json`) to an HF repo you control, in the layout `<method>/<bNN>/<selXXXX|selfixed>-t<seed>/`, each with a
   `provenance.json` (code commit, selection sha256, steps, K, hardware, scores). Put the repo id and revision in the CSV.
3. **Selections** (T4, T6, T7): `record_ids.json` (ordered `<index>:<LLaVA id>`) and `selection_meta.json` (K, seed,
   per-source counts, records sha256, code commit, every non-default setting), in the dataset-repo layout.
4. **Fidelity notes:** `ICONS_FIDELITY.md` / `LESS_FIDELITY.md` with the official commit, every setting used vs. the
   paper default, the target/validation sets, and every deviation with the label it caused.
5. **Run log:** `results/coauthor/RUN_LOG.md` with date, task ID, job, hardware, status (completed / failed / partial),
   failure cause and GPU-hours, including the T0 deltas.
6. **Raw evaluation outputs:** per-run prediction files for the core four (answer jsonl), uploaded next to each
   checkpoint so scores can be recomputed.

Hand-back: a pull request to this repository adding `results/coauthor/*` and the fidelity notes (no weights in Git).

## 7. Acceptance criteria (per row)
* Training complete (`global_steps == natural_steps`), selection hash matches, recipe unchanged (attach the exact
  launch command or config).
* Core four evaluated on the full banks (8,910 / 12,578 / 5,000 / 2,017 questions; no subsampling); GQA main score as in §3.5.
* Provenance and hashes present; failed attempts listed; no method label changed silently.

## 8. Stop and ask (do not decide alone)
* T0 fails, or any input hash does not match.
* A method's official code cannot run on this pool or model without changing its selection unit, features or defaults.
* A run needs a different global batch or step count.
* A result looks implausible (e.g. core avg < 60 at 10%, or > 2 points away from comparable baselines) before you have
  checked implementation and evaluation correctness.
* You want to add or drop any cell.

## 9. Prohibited
* Hyper-parameter search on baselines; changing K, epochs, steps, decoding or scorers.
* Using benchmark answers for selection or tuning; reporting only the best seed.
* Overwriting provided checkpoints or selections.
* Committing tokens, API keys or other credentials.
* Re-hosting LLaVA-665K text or images (rebuild from the official source instead).
