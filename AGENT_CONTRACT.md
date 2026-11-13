# Agent contract — baseline completion for `trimdata` (8× NVIDIA A100 40GB)

This file is the **binding work order** for any agent (human or AI) that takes over the baseline side of this study.
Read it fully before acting. Precedence: **this contract > `README.md` > `AGENTS.md`**. If it conflicts with an
explicit instruction from the project lead, ask the lead. Machine-readable copy: [`agent_contract.json`](agent_contract.json).

## 1. Role and scope
* **You own:** the published baselines, controls and reference rows in §6: selection (where stated), training,
  evaluation, results, cost accounting and hand-back.
* **You do not own:** the project's own selection method (not in this repository), the paper, or any change to the
  fixed protocol in §3. Do not add or drop baselines, budgets, seeds, benchmarks or hyper-parameter variants without
  written approval from the project lead.
* **Hardware:** 8× NVIDIA A100 40GB on one machine (corrected by the project lead on 2026-11-12). Continue with the
  fixed recipe unless 40GB memory becomes a demonstrated blocker. All provided runs were made on single NVIDIA L40S GPUs; the gates
  in §5 establish that your hardware reproduces them before any new result is produced.
* **Temporary judge deferral, project lead 2026-11-12:** defer LLaVA-W and MM-Vet generation/judging.
  Continue T0a/T0b and the six locally scored extended metrics. T0c remains incomplete overall;
  its local metrics can be calibrated independently. Core-only evaluation tasks require T0a,
  and locally scored extended tasks additionally require their local T0c calibration.
  Do not assign scores to deferred metrics or omit a selector's prescribed quality judge.

## 2. Inputs (read-only; verify every hash before use)
| Input | Location / identity | Verification |
|---|---|---|
| Code snapshot | this repository: `src/` (executed sources), `scripts/` (portable entry points) | `SOURCE_MANIFEST.json` |
| 46 provided checkpoints (LoRA adapter + projector + per-run evaluation outputs) | HF model repo `submissionpaper1234/trimdata-baseline-checkpoints` | `SHA256SUMS`, `provenance.json`, `index/checkpoints.csv` |
| 16 provided selections + eligible pool | HF dataset repo `submissionpaper1234/trimdata-baseline-selections` | `SHA256SUMS`, `selection_meta.json`, `index/selections.csv` |
| Training data | `llava_v1_5_mix665k.json`, HF `liuhaotian/LLaVA-Instruct-150K` @ `9d451dc7629cfe0469f6ae4432b765cd603d5fcb` (sha256 `ce959ce6…af6cf8`), plus the official LLaVA-1.5 image folders `coco/train2017`, `gqa/images`, `ocr_vqa/images`, `textvqa/train_images`, `vg/VG_100K{,_2}` | `scripts/reconstruct_records.py` must reproduce each selection's `records.json` sha256 |
| Base model | `lmsys/vicuna-7b-v1.5` @ `3321f76e3f527bd14065daf69dad9344000a201d`; `openai/clip-vit-large-patch14-336` @ `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`; stage-1 projector `liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5` @ `5414da88308e4287a29f2e9609256458afb0a981` | download with `--revision <sha>`; these are also the current `main` revisions |
| LLaVA code | `haotian-liu/LLaVA` @ `c121f0432da27facab705978f83c4ada465e46fd`. The `llava/` package is vendored under `src/collaboration/published-foundation-20261022/reference_sources/LLaVA/c121…/` and used by the portable scripts by default; clone the repo at this commit as well for its `scripts/` (e.g. `scripts/convert_gqa_for_eval.py`, `scripts/v1_5/eval/*.sh`) | `git rev-parse HEAD` |
| Environment | `env/requirements-native-env.txt` (Python 3.10.19, torch 2.1.2+cu121, transformers 4.37.2, peft 0.7.1, deepspeed 0.12.6, **xformers 0.0.23.post1**) | `pip freeze` diff in `CALIBRATION_REPORT.md` |
| Evaluation inputs | LLaVA `eval.zip` (c121 instructions) + official benchmark images; hashes in `docs/EVAL_INPUTS.md` | hashes |
| Reference scores | `results/baseline_core4_per_run.csv`, `results/baseline_extended_reported.csv`, per-run `eval/**/SCORE*.json` and `PREDICTIONS.jsonl` on the HF model repo | — |
| Official baseline code (pins) | ICONS `princetonvisualai/icons` @ `b8ce8c86d7b098836a84ec2b40d259195b7f4494`; LESS `princeton-nlp/LESS` @ `8abf9628b9a814ac3045445eebc8ba3c908fdc78`; XMAS `BigML-CS-UCLA/XMAS` @ `e534dd99e9ce9b7be345bd55e055c9b4e5c91c90` (fork of `NilayN7/XMAS` @ `4fda84bfc7acf387c092dc93d0d15b299fe197d8`; MIT); VisNec `DMK041218/VisNec` @ `d894598ef9ba3e03da847a47464185c6262d230f` (ECCV 2026; no license file in the repo: use for research reproduction only, do not copy its code into this repository); CoIDO `SuDIS-ZJU/CoIDO` @ `a91906703267741e73532b5b560593d899ffa252` (AGPL-3.0); PreSel `bardisafa/PreSel` @ `2dd6709de1fa5d687f88e55377118a38716144ca` (MIT) | record the commit you actually ran |

Hugging Face revisions at hand-over (2026-11-11): checkpoints `ec9d98a2b071f2873889ad7f6e6dc66a02c481e6`, selections `3074f2b1ca0aa15b4ac0dbced03afd637d84cc75`. Download with `--revision`
these values so later card edits cannot change your inputs.

If any hash does not match, **stop** and report; never substitute a "similar" file.

## 3. Fixed protocol (invariants — never change)
1. **Training entry point: `scripts/train_portable.py` only.** Our runs used the official LLaVA c121 trainer plus three
   runtime changes that plain `train.py` / `train_mem.py` / `scripts/reference_train_command.sh` do **not** have:
   (a) LLaVA's xformers attention patch with xformers CUTLASS kernels and deterministic torch/cuBLAS settings;
   (b) structured v1 label masking (official `preprocess_v1` mis-masks turns whose text contains a role marker);
   (c) removal of the empty LoRA-bias optimizer group before the LR scheduler is built — without it DeepSpeed 0.12.6
   misaligns the per-group learning rates and the **projector trains at 2e-4 instead of 2e-5** (reproduced in a CPU
   simulation of torch 2.1.2 + DeepSpeed's group removal). The script imports the executed modules unchanged from
   `src/` and aborts if the projector/LoRA learning-rate ratio is not 0.1 at steps 1, 10 and 100.
2. **Recipe:** LoRA r=128, alpha=256, dropout 0.05 on all LLM linear layers; projector trained (LR 2e-5); LoRA LR 2e-4;
   AdamW (`adamw_torch`), weight decay 0; cosine, warm-up ratio 0.03; **global batch 128**; `model_max_length` 2048;
   `image_aspect_ratio=pad`; `group_by_modality_length=True`; fp16 (no bf16/tf32); DeepSpeed ZeRO-3 (`scripts/zero3.json`);
   gradient checkpointing; `dataloader_num_workers=0`; conversation template `v1`. Training seed ∈ {17, 29, 43} sets
   both `seed` and `data_seed`. All of this is hard-coded in `train_portable.py`; do not edit it.
3. **One run per GPU.** Every provided run is single-GPU: per-device batch 4 × accumulation 32. Run up to eight
   independent runs side by side, one per GPU (`CUDA_VISIBLE_DEVICES=i`). Multi-GPU data parallelism is allowed
   **only** for T8 (full-data reference), with per-device × accumulation × world size = 128 (`torchrun ... --allow-multi-gpu`).
4. **Exactly one natural epoch.** A run is complete only if its final global step equals `natural_steps`
   (`train_portable.py --expected-steps` enforces it):

   | Budget | K | natural_steps |
   |---|---|---|
   | 5% | 33,032 (COINCIDE 33,038) | 258 |
   | 10% | 66,076 (DataTailor 66,071) | 516 |
   | 15% | 99,108 | 774 |
   | 20% | 132,147 (COINCIDE 132,152) | 1,032 |
   | 100% | 660,763 | 5,162 |
5. **Selected-data budget ≠ compute budget.** Keep each method's realised K (DataTailor 66,071 at 10%; COINCIDE
   33,038 / 132,152 at 5% / 20%); never pad or truncate a selection, never change epochs or steps.
6. **Evaluation (core four; 28,505 questions; full banks only):** run the official LLaVA c121 generation modules
   through `scripts/eval_portable.py` (LoRA **unmerged**, HF **eager** attention, fp16, greedy `temperature 0`, batch 1,
   `--conv-mode vicuna_v1`) and score with the official scripts unchanged:

   | Bank | n | Generation (via `eval_portable.py`) | Score |
   |---|---|---|---|
   | POPE (adversarial/popular/random) | 8,910 | `llava.eval.model_vqa_loader`, `pope/llava_pope_test.jsonl` | `eval_pope.py`; **report pooled accuracy = correct / 8,910** (also log the yes-ratio per category) |
   | GQA test-dev balanced | 12,578 | `llava.eval.model_vqa_loader`, `llava_gqa_testdev_balanced.jsonl` | LLaVA's `scripts/convert_gqa_for_eval.py` then the official GQA `eval.py` (as in LLaVA `scripts/v1_5/eval/gqa.sh`); **main = this native normalized score** |
   | TextVQA val (v0.5.1, OCR tokens) | 5,000 | `llava.eval.model_vqa_loader`, `textvqa/llava_textvqa_val_v051_ocr.jsonl` | `llava.eval.eval_textvqa` (m4c evaluator) |
   | ScienceQA-IMG test | 2,017 | `llava.eval.model_vqa_science --single-pred-prompt`, `scienceqa/llava_test_CQM-A.json` | `llava/eval/eval_science_qa.py`, image subset |

   Core average = unweighted mean of the four (percent, two decimals).
7. **Extended eight** (same loader settings): MME perception (/2000) and cognition (/800), SEED-Bench **image subset**
   (14,233; `model_vqa_loader`), MMMU **val** (900; official MMMU prompt rendered with the c121 conversation, see
   `src/experiments/helper_tool1_bench4_eval_v1_20261104/banks4.py`), MMBench-EN **dev** (local labels, **CircularEval**,
   VLMEvalKit `can_infer` exact-match extraction, no LLM extraction; `model_vqa_mmbench`), LLaVA-Bench-in-the-Wild and
   MM-Vet (`model_vqa`; judge **`gpt-4o-2024-08-06`, temperature 0, one run**, prompts in
   `src/experiments/helper_tool1_bench4_judge_v13_20261105/judge4_v13.py`), VizWiz test 8,000 (`model_vqa_loader`;
   local scoring with the public answers via `src/experiments/helper_tool1_vizwiz_score_v1_20261101/scorer.py`; also
   record the share of "unanswerable" predictions). MME/SEED bank definitions: `src/experiments/helper_tool1_bench6_eval_v1_20261103/banks6.py`.
   Never send any image or benchmark question to an API other than the two judged benchmarks; keep API keys in the
   environment only.
8. **Baseline fidelity:** official code at the pinned commit with its published defaults and prescribed target sets.
   No tuning of any baseline's hyper-parameters, features, selection units, quotas or targets. A bug fix to an official
   release is allowed only if it restores the paper's stated behaviour, is listed in the fidelity note, and does not
   change the method. Any unavoidable adaptation gets a **new label** (e.g. `XMAS-adapted`); never report an adaptation
   under the original name.
9. **No leakage:** never use evaluation-bank labels or answers for selection, early stopping, checkpoint choice or
   tuning. Methods that prescribe target/validation tasks (ICONS, LESS, XMAS if applicable) use only those, recorded.
10. **Report everything:** every seed and every attempt, including failures and partial runs, with GPU-hours.

## 4. Running on 8× A100 40GB
**Setup (once).** Create the environment from `env/requirements-native-env.txt` (CUDA 12.1 wheels); verify
`python -c "import xformers, torch; print(xformers.__version__, torch.__version__)"` → `0.0.23.post1 2.1.2+cu121`.
Download the three base assets at the pinned revisions to local directories (no symlinked caches that other jobs may
clean). Rebuild each needed `records.json` with `scripts/reconstruct_records.py` and check its sha256. Download only the
checkpoint folders a task needs (`hf download ... --include "<path>/*"`), then `python scripts/verify_sha256.py`.

**Training command (one GPU):**
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/train_portable.py \
  --data-path work/records/uniform-b10-sel1235.json --image-folder /data/llava665k \
  --model /models/vicuna-7b-v1.5 --vision-tower /models/clip-vit-large-patch14-336 \
  --projector /models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
  --output-dir runs/uniform-b10-sel1235-t17-a1 --seed 17 --expected-steps 516 2>&1 | tee logs/uniform-b10-sel1235-t17-a1.log
```
Use a **new output directory for every attempt** (`-a1`, `-a2`, …; the script refuses a non-empty one because LLaVA would
otherwise silently resume). Keep failed attempt directories as evidence. Do not resume a crashed run from a mid-run
checkpoint; restart it as a new attempt. Exit code 3 = wrong step count or missing outputs → the run is invalid.

**Evaluation command (one GPU per checkpoint):**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_portable.py llava.eval.model_vqa_loader \
  --model-path ckpts/uniform/b10/sel1234-t17 --model-base /models/vicuna-7b-v1.5 \
  --question-file eval/pope/llava_pope_test.jsonl --image-folder eval/pope/val2014 \
  --answers-file out/uniform-b10-sel1234-t17/pope.jsonl --temperature 0 --conv-mode vicuna_v1
```
Each checkpoint folder's `config.json` names the CLIP tower by its Hub id; have the pinned revision in your HF cache.

**Throughput planning.** Measured on one L40S: about 67–77 s per optimizer step, i.e. ≈10–11 h for a 10% run
(scale linearly with K). Measure A100 40GB throughput with T0b rather than assuming the L40S timing. Core-four
evaluation is roughly 2–4 GPU-h per checkpoint and the extended eight a similar amount (judge calls are API-bound).
Capacity is ≈190 GPU-h per day; the full roster in §6 is on the order of 900–1,300 GPU-h (estimates, not budgets).

**Suggested schedule** (keep every GPU busy, respect the gates):
| Phase | GPUs | Work |
|---|---|---|
| A (day 0) | 0–1 | T0a: core four for `uniform/b10/sel1234-t17` and `datatailor/b10/sel1234-t17` |
|  | 2 | T0b: retrain `uniform/b10/sel1235-t17` |
|  | 3 | T0c: extended eight for `uniform/b10/sel1234-t17` |
|  | 4–7 | selection-side compute of T4 ICONS, T9 XMAS, T10 CoIDO (warm-ups, proxies, gradients, features) — these use each method's own official procedure, not our recipe, so they need no gate |
| B (after T0a [+T0c]) | free GPUs | evaluation-only tasks T1 and T5 |
| C (after T0b) | free GPUs | training tasks T2, T3, then students of T4/T9/T10 as each selection is finalised; T8 when 4–8 GPUs are free together |
| D | free GPUs | optional T6, T7, T11 if approved |

## 5. Gates (must pass before dependent tasks)
* **T0a — evaluation calibration** (unlocks evaluation tasks T1, and the core-four part of T5/T2–T4/T8–T10 evaluation):
  re-evaluate the provided `uniform/b10/sel1234-t17` (POPE 84.77 / GQA 55.49 / TextVQA 52.86 / SQA-IMG 67.82, avg
  65.23) and `datatailor/b10/sel1234-t17` (83.08 / 57.52 / 53.65 / 66.53, avg 65.19) with §3.6. **Pass:** core average
  within ±0.10 and every bank within ±0.30 for both. Also report the answer-level agreement with each folder's
  `eval/core4/<bank>/PREDICTIONS.jsonl` (informational). If it fails with `eval_portable.py`, also try `--merged`, report
  both, and **stop**.
* **T0c — extended calibration** (unlocks the extended part of T1/T5 and of new rows): run the extended eight on
  `uniform/b10/sel1234-t17` and compare to its `eval/extended/*/SCORE*.json` (MME-P 1360.2, MME-C 311.8, SEED-img 59.87,
  MMMU 32.78, MMBench-EN dev 58.76, LLaVA-W 56.5, MM-Vet 26.6, VizWiz 43.43). **Pass:** MME-P ±15, MME-C ±10,
  SEED/MMMU/MMBench ±0.5, VizWiz ±1.0, LLaVA-W/MM-Vet ±2.0 (judge variance). Otherwise **stop** and report per bank.
* **T0b — training calibration** (unlocks every training task): retrain `uniform/b10/sel1235-t17` from
  `selections/uniform/b10/sel1235` (K=66,076, seed 17, 516 steps). This reference was produced by the same deterministic
  CUTLASS trainer configuration that `train_portable.py` reproduces.
  * **Early check (first ~1 h, abort on failure):** the log must show learning rate `0.000125` at step 10, the three-group
    LR check at steps 1/10/100, and losses at steps 10–60 within ±0.03 of the reference `trainer_state.json`
    (1.0961, 1.0100, 0.8531, 0.8942, 0.8452, 0.9459). Two consecutive deviations → stop and diagnose.
  * **Final:** global step 516; core average within ±0.30 of 65.22 (84.56 / 55.45 / 53.77 / 67.08); no bank off by
    more than ±1.0.
* On any gate failure: **stop** the dependent tasks and report the deltas, environment (`pip freeze`), GPU/driver and
  logs in `CALIBRATION_REPORT.md`. Tasks whose gate passed may continue.

## 6. Tasks
Execution sequence (binding): `AGENTS.md` §3a. Summary: T0 → T1 → T5 → T2 → T3 → T4 (10%) → T12 → T9 → T10 → lead
checkpoint → T8 → T4 (5%/20%) → approved extra budgets → optional T6, T7, T11. Selection-side compute of T4/T12/T9/T10 runs in
parallel from day 0.

| ID | Task | Gate | Inputs | Deliverable | Rough A100 GPU-h |
|---|---|---|---|---|---|
| T0 | Calibration T0a, T0b, T0c (§5) | — | 2 checkpoints, 1 selection | 3 calibration rows + `CALIBRATION_REPORT.md` | ~25 |
| T1 | Core four + extended eight for COINCIDE 20% t29 and t43; extended eight for COINCIDE 20% t17 | T0a, T0c | `coincide/b20/selfixed-t{17,29,43}` | 3 rows | ~15 |
| T2 | Train + evaluate **Uniform 20% t29** (`selections/uniform/b20/sel1235`, K=132,147, 1,032 steps) | T0b | selection | checkpoint + row | ~25 |
| T3 | Train + evaluate **DataTailor 15% t17** (`selections/datatailor/b15/sel1234`, K=99,108, 774 steps) | T0b | selection | checkpoint + row | ~20 |
| T4 | **ICONS, faithful** (official code + defaults: its warm-up, gradient features, influence-consensus selection over its prescribed target tasks). 10% × seeds 17/29/43 first; then 5% and 20% × 3 seeds | selection: none; students: T0b | pool ids, official repo | 3 selections, 9 checkpoints, rows, `ICONS_FIDELITY.md` | ~150–250 |
| T5 | Extended eight for every provided checkpoint that lacks it (list below) and for every new checkpoint from T2–T4, T8–T10 | T0a, T0c | checkpoints | rows | ~3 each |
| T12 | **VisNec, faithful (required)** — closest prior art to the project's visual-necessity filter. Official code `DMK041218/VisNec` @ `d894598ef9ba3e03da847a47464185c6262d230f`: compute VisNec scores with `scripts/cal_visnec.py` exactly as in `scripts/cal_visnec.sh` (Stage-1 model = Vicuna-7B-v1.5 + stage-1 projector, its bf16/batch settings) on the 660,763 eligible records; select with `scripts/task_distribution.py` unchanged except paths and `TOP_RATIO = 0.10` (20 MiniBatchKMeans clusters of all-MiniLM-L6-v2 instruction embeddings, random_state 42, top 10% per cluster by VisNec). Keep the realised K (sum of per-cluster int(n×0.10), slightly below 66,076) and report it; never top up. Deterministic selection (no selection seed). **Fidelity check first (cheap):** also run `TOP_RATIO = 0.15` on the full `llava_v1_5_mix665k.json` and report the overlap with the authors' released `llava_v1.5-7b-top15.json` (HF dataset `aaakiyasuqqqa/VisNec`); stop and ask if it is far below ~90%. Then 10% × seeds 17/29/43 with `train_portable.py` (natural steps of the realised K) + core/extended evaluation. Label `VisNec` (disclose: eligible-pool restriction and 10% ratio); any other deviation → `VisNec-adapted`. 5%/20% only if the lead approves at the P13 checkpoint | selection: none; students: T0b | pool ids, official repo, released top-15 subset | selection, 3 checkpoints, rows, `VISNEC_FIDELITY.md` | ~80–160 |
| T9 | **XMAS, faithful** (official code: proxy training with its released proxy config, cross-modal alignment trajectories over its checkpoints, clustering and stability sampling). 10% × seeds 17/29/43. Known release defects to fix and disclose (do not change the method): `cluster_and_sample.sh` defines `BUDGET` but passes `PERCENTAGE`; the stability path's `random.sample` ignores the CLI seed; the clustering script silently skips checkpoints whose metrics fail to load (must fail instead) | selection: none; students: T0b | pool ids, official repo | selection(s), 3 checkpoints, rows, `XMAS_FIDELITY.md` | ~120–200 |
| T10 | **CoIDO, faithful** (official code: CLIP features, ImageReward and CLIP scores, LLM quality judge, stage-1 scorer trained on the paper's 20% random sample, stage-2 selection). 10% × seeds 17/29/43. Record the LLM judge model/version/cost; the README notes the release omits the paper's similarity filtering — if the release differs from the paper, label the row `CoIDO-release` and describe it. AGPL-3.0: keep its code in its own repository/directory | selection: none; students: T0b | pool ids, official repo, LLM API | selection, 3 checkpoints, rows, `COIDO_FIDELITY.md` | ~100–150 + API |
| T8 | **Full-data reference**: all 660,763 eligible records (`pool/pool_record_ids.json`), seed 17, 5,162 steps; multi-GPU allowed (§3.3); record the layout | T0b | pool | checkpoint + row (core + extended) | ~80–120 |
| T6 (optional) | **Faithful LESS** (official pipeline at the pin, paper defaults: LoRA warm-up, Adam-preconditioned projected gradients, its target-task dev sets), 10% × 3 seeds; `less_approx` keeps its own label | students: T0b | pool, official repo | selection, 3 checkpoints, rows, `LESS_FIDELITY.md` | ~100 |
| T7 (optional) | **Source-sqrt** at 5% and 20% (selector `src/experiments/root_native_campaign_v7_20261024/selection.py`, method `source_sqrt`, selection seed 1234), 3 seeds each | T0b | pool | 2 selections, 6 checkpoints, rows | ~90 |
| T11 (optional, approval first) | **PreSel-adapted**: PreSel selects images before instructions exist; on this pool it is necessarily an adaptation. Write the adaptation plan (task/source mapping, image-level → conversation-level mapping, K) and get the lead's approval **before** spending compute; 10% × 3 seeds | approval, T0b | official repo | plan, selection, 3 checkpoints, rows, `PRESEL_FIDELITY.md` | ~100 |

**After T4/T12/T9/T10 have 10% results:** report them and ask the lead which one or two methods get 5%/20% runs. Do not
start budget curves for XMAS or CoIDO on your own.

**T5 list (from the HF repo contents, 2026-11-11).** Missing all eight extended benchmarks (19):
`coincide/b05/selfixed-t29,-t43`; `coincide/b20/selfixed-t17,-t29,-t43`; `datatailor/b05/sel1234-t29,-t43`;
`datatailor/b15/sel1234-t29`; `datatailor/b20/sel1234-t29,-t43`; `less_approx/b10/sel1234-t29,-t43`;
`uniform/b05/sel1235-t29,-t43`; `uniform/b15/sel1235-t29,-t43`; `uniform/b20/sel1235-t43`;
`uniform_scored_random/b10/sel1234-t29,-t43`. Missing only VizWiz (5): `datatailor/b10/sel1234-t29,-t43`;
`uniform/b10/sel1235-t29,-t43`; `uniform/b10/sel1236-t43`. (COINCIDE 20% t29/t43 are covered by T1.)

**Already complete — do not retrain:** every folder in the HF model repo with `complete_one_epoch=true`. Each
(method, budget, selection seed, training seed) cell is trained successfully at most once; a repeat is allowed only as a
documented retry of a failed attempt. A partial full-data run (2,675 / 5,162 steps) exists on our cluster; it is not
released and does not change T8.

## 7. Outputs (hand-back)
1. **`results/coauthor/RESULTS.csv`** with exactly the columns of `results/RESULTS_TEMPLATE.csv`: one row per
   (method, budget, selection seed, training seed) plus **one row per failed attempt** (scores empty, cost filled).
   Percent with two decimals; MME in points; `global_steps`, `natural_steps`, `gpu_type`, GPU-hours per stage, hashes.
2. **Checkpoints** in an HF repo you control, layout `<method>/<bNN>/<selXXXX|selfixed>-t<seed>/`, each with
   `adapter_model.safetensors`, `adapter_config.json`, `non_lora_trainables.bin`, `config.json`, `trainer_state.json`,
   `PORTABLE_RUN_SUMMARY.json`, `provenance.json` (code commit, selection sha256, steps, K, hardware, scores) and
   `eval/` with the raw prediction files of every bank.
3. **Selections** (T4, T6, T7, T9–T11): `record_ids.json` (ordered `<index>:<LLaVA id>`) and `selection_meta.json`
   (K, seed, per-source counts, records sha256, official commit, every non-default setting) in the dataset-repo layout.
4. **Fidelity notes** `<METHOD>_FIDELITY.md`: official commit, every setting vs. the paper default, target/validation
   sets, release bug fixes, every deviation and the label it caused, selection cost (GPU-h, API calls/$).
5. **`results/coauthor/RUN_LOG.md`**: date, task, command, GPU(s), status (completed / failed / partial), failure
   cause and GPU-hours for every attempt, including T0.
6. **`results/coauthor/CALIBRATION_REPORT.md`**: T0a/T0b/T0c deltas, prediction agreement, `pip freeze`, GPU/driver.

Hand-back = a pull request to this repository adding `results/coauthor/*` and the fidelity notes. Never commit weights,
images, tokens or API keys.

## 8. Acceptance criteria (per row)
* Training complete (`global_steps == natural_steps`, `PORTABLE_RUN_SUMMARY.json` present), selection hash matches,
  recipe unchanged (attach the exact command line).
* Core four on the full banks (8,910 / 12,578 / 5,000 / 2,017) with the GQA main score of §3.6; extended eight per §3.7.
* Provenance and hashes present; failed attempts listed; no method label changed silently.

## 9. Stop and ask (do not decide alone)
* Any gate fails, or any input hash does not match.
* An official method cannot run on this pool or model without changing its selection unit, features, defaults or
  target sets; or the release contradicts the paper in a way that changes the method.
* A run needs a different global batch, step count or GPU layout than §3.3–§3.4 allow.
* A result looks implausible (e.g. core average < 60 at 10%, or > 2 points from comparable baselines) before you have
  checked implementation and evaluation correctness; or POPE yes-ratio > 0.55 in any category.
* You want to add or drop any cell, budget or benchmark, or the projected cost of a task exceeds twice its estimate.

## 10. Prohibited
* Hyper-parameter search on baselines; changing K, epochs, steps, decoding, scorers or judge settings.
* Using benchmark answers for selection or tuning; reporting only the best seed.
* Overwriting provided checkpoints or selections; editing `src/` executed sources (put new code in new files).
* Training with plain `llava/train/train.py`, `train_mem.py` or `scripts/reference_train_command.sh`.
* Committing tokens, API keys or other credentials; re-hosting LLaVA-665K text or images.
