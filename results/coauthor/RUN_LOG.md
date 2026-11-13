# Coauthor run log

All times are recorded in UTC. GPU-hours are elapsed seconds multiplied by the
number of allocated GPUs, divided by 3,600. Setup and read-only checks use zero GPUs.

## 2026-11-12 — handover and instance inspection

- Verified all 780 entries of `SOURCE_MANIFEST.json`; zero missing or mismatched files.
- Connected to the project lead's instance using the SSH instructions in
  `../MechInterp/p4-instance-access.md`.
- Observed eight NVIDIA A100-SXM4-40GB GPUs. GPUs 0 and 7 were free during inspection;
  GPUs 1–6 had existing workloads and are not available to this campaign.
- Found no LLaVA assets in the inspected `/home/ubuntu` directories.
- Verified pinned training JSON, selection index, and checkpoint index download
  endpoints from the instance (HTTP 200).
- Found 6.5TB available at `/opt/dlami/nvme`; created a private campaign root at
  `/opt/dlami/nvme/trimdata-work` and a repository directory at `/home/ubuntu/trimdata`.
- Created an isolated bootstrap virtual environment and installed uv. The shared
  `/opt/pytorch` environment is not modified.
- Status: setup in progress. GPU-hours: 0.

## Project-lead instructions, 2026-11-12

- Correct the hardware assumption to 40GB; continue without changing the fixed
  recipe unless GPU memory becomes a demonstrated blocker.
- Use an isolated virtual environment.
- Defer judged benchmark experiments for now: LLaVA-W and MM-Vet. T0c is split
  into locally scored calibration and deferred judged calibration. Only locally
  scored extended metrics can pass their calibration while the judges are deferred.
- This deferral does not authorize substituting or omitting a selector's prescribed
  quality judge. CoIDO's selection prerequisites will be tracked separately.

## 2026-11-12 — isolated environment and verified inputs

- Native setup attempt 1 failed because `CUDA_HOME` was absent. No GPU was allocated.
  Installed the official SHA-verified CUDA 12.1 compiler/runtime/header components
  in the private work directory; setup attempt 2 completed successfully.
- Python 3.10.19; torch 2.1.2+cu121; xformers 0.0.23.post1; transformers 4.37.2;
  PEFT 0.7.1; DeepSpeed 0.12.6. The CPU optimizer test prints RAISES, PASSES, PASSES.
  Exact freeze retained at `control/native-freeze.txt` on the instance.
- HF inputs use the contract's pinned revisions and verified Hub/published hashes.
  Reconstructed Uniform 10% sel1235, Uniform 20% sel1235, and DataTailor 15% sel1234
  training records with their exact published hashes.
- Pinned official LLaVA c121 and ICONS/XMAS/CoIDO source repositories are staged.
  CoIDO remains in a separate upstream directory.
- Official evaluation archive hash matches. COCO val2014, TextVQA images,
  COCO train2017, and GQA labels have downloaded and passed archive CRC checks.
- Core staging attempt 1 hit an HTTP 503 in Stanford byte-range retrieval.
  Attempt 2 uses bounded retries against the same official archive. GPU-hours: 0.
- OCR staging attempt 1 used a tooling environment without Pillow and stopped
  with an import error. Attempt 2 used the native environment; all 7,968 required
  original OCR-VQA images downloaded and passed image validation. GPU-hours: 0.
  Both attempt manifests and logs are retained.

## 2026-11-12T07:15Z — CPU scorer qualification

- Ran unchanged official scorers against the complete published prediction banks
  for Uniform b10 sel1234-t17 and DataTailor b10 sel1234-t17.
- Uniform: POPE 84.769921, GQA 55.49, TextVQA 52.86, ScienceQA-IMG 67.823500.
- DataTailor: POPE 83.075196, GQA 57.52, TextVQA 53.65, ScienceQA-IMG 66.534457.
- All individual scores match the published two-decimal values to rounding
  precision. GQA uses the official normalized conversion.
- Both CPU attempts completed; zero GPU-hours. Exact commands and scoring outputs:
  `runs/{uniform,datatailor}-b10-sel1234-t17-scorecheck-a1/`.
- This is scorer qualification only. T0a fresh generation remains pending.
- Started a persistent input-gated calibration controller. It checks that a GPU
  is free twice before launching; no occupied GPU is allocated.

## 2026-11-12T07:21:28.393284+00:00 — T0b training attempt 1

- Status: failed; GPU: None; training GPU-hours: 0.
- Output: `/opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1235-t17-train-a1`; error: FileNotFoundError(2, 'No such file or directory').

## 2026-11-12T07:27:20.628583+00:00 — T0a uniform/b10/sel1234-t17 attempt 1

- Status: failed. GPU-hours: 0.137182.
- GPU: 5. Output: `/opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1234-t17-eval-a1`.
- Error: RuntimeError('Command exited -15; see /opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1234-t17-eval-a1/POPE/generation.log').
- Commands and raw outputs: `ATTEMPT.json`, per-bank generation/scoring logs.

## 2026-11-12T07:27:41.423550+00:00 — T0b training attempt 2

- Status: failed; GPU: None; training GPU-hours: 0.
- Output: `/opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1235-t17-train-a2`; error: RuntimeError('T0a stopped; dependent calibration work must be diagnosed').

## 2026-11-12T07:35:42.804683+00:00 — diagnosed evaluator repair and active retry

- Stopped Uniform T0a attempt 1 because it selected SDPA; failed GPU-hours: 0.137182. Checkpoints and vendored sources are unchanged.
- The pinned-loader CPU regression reproduces the error and passes with explicit eager config. eval_portable.py now checks actual attention classes before generation.
- Uniform attempt 2 and DataTailor attempt 1 run with 32 eager attention layers and 224 active unmerged LoRA layers. No T0a gate has passed yet.
- The first two T0b controller attempts allocated no GPU; their setup failures are recorded separately. Controller attempt 3 waits for exact images and retains the fixed training recipe.
- Stanford range retrieval became unavailable with HTTP 503. Full official archive downloads continue; no data are trimmed or substituted.
- Applied the three contract-authorized XMAS release fixes; syntax checks passed and the exact patch/hashes are retained. Released unique IDs cover every eligible conversation.
- Started full eligible pool reconstruction and original OCR image staging, with zero allocated GPUs.

## 2026-11-12T09:55:30.668547+00:00 — T0a datatailor/b10/sel1234-t17 attempt 1

- Status: completed. GPU-hours: 2.409384.
- GPU: 5. Output: `/opt/dlami/nvme/trimdata-work/runs/datatailor-b10-sel1234-t17-eval-a1`.
- Error: none.
- Commands and raw outputs: `ATTEMPT.json`, per-bank generation/scoring logs.

## 2026-11-12T10:25:41.192887+00:00 — T0a uniform/b10/sel1234-t17 attempt 2

- Status: completed. GPU-hours: 2.912308.
- GPU: 1. Output: `/opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1234-t17-eval-a2`.
- Error: none.
- Commands and raw outputs: `ATTEMPT.json`, per-bank generation/scoring logs.

## 2026-11-12T18:18:44.825673+00:00 — T0b training phase finished

- Uniform b10 sel1235-t17 attempt 3 completed exactly 516 steps; 8.803642 GPU-hours; peak 31,973 MiB. Early loss and LR checks passed. Final core evaluation remains pending.

## 2026-11-12 — Local extended scorer setup

- CPU qualification attempt 1 stopped at missing pandas; 0 GPU-hours. No GPU calibration ran or failed.
- Native environment has no pip. Extracted original SHA/CRC-verified Python 3.10 wheels for pandas 1.5.3, python-dateutil 2.8.2, pytz 2023.3 into a separate extended overlay; native training dependencies unchanged. Wheel hashes in control/extended-overlay.json. Retrying the diagnosed CPU setup check as attempt 2.

## 2026-11-12 — Local extended qualification and dispatch

- CPU scorer qualification attempt 2 passed all six metrics against supplied predictions (0 GPU-hours). MME-P 1360.194478; MME-C 311.785714; SEED 59.87; MMMU 32.777778; MMBench CircularEval 58.758503; VizWiz 43.43, unanswerable share 0.2815.
- Fresh local T0c generation attempt 1 launched on GPU6; gate pending. Benchmark judge generation/scoring remains deferred.
- Controller stages the exact 24 provided T1/T5 gap cells, waits for local T0c, then dispatches onto twice-confirmed free GPUs. New T0b/T2/T3 checkpoints wait for successful core evaluation. Failures retain per-attempt evidence and stop that item for diagnosis.
- Continuing without unavailable OCR image 689852649.jpg as instructed: unaffected cells proceed, no pool record was removed or image substituted.

## 2026-11-12T19:17:27.416492+00:00 — T0c uniform/b10/sel1234-t17 local extended attempt 1

- Status: failed; GPU 6; 0.301205 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1234-t17-extended-a1/ATTEMPT.json`.

## 2026-11-12T19:17:52.845051+00:00 — Local extended controller

- Status: failed; error: RuntimeError('Local T0c failed; dependent extended work remains stopped'); controller GPU-hours: 0.
- Child attempts retain their own output and cost records.

## 2026-11-12T19:28:48.522370+00:00 — Local T0c failure diagnosis and permitted retry

- Attempt 1 failed during MME cognition model loading on physical GPU6 after full MME perception generation. Failed cost 0.301205 GPU-hours; no fresh scores or gate pass assigned.
- Three other CUDA processes held 9.72/9.72/19.60 GiB when the model reload tried its first 250 MiB allocation. Their processes were left untouched. No single-model 40GB blocker was established.
- Fresh attempt 2 starts on empty GPU4. Availability and CUDA ownership are checked twice before every module; the run may wait or use another empty A100 at a module boundary, always one GPU concurrently. Unallocated waits are excluded from GPU-hours. One permitted scientific calibration retry; a second failure stops this item.

## 2026-11-12T20:45:50.520599+00:00 — T0b uniform/b10/sel1235-t17 attempt 3

- Status: completed. GPU-hours: 2.451137.
- GPU: 2. Output: `/opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1235-t17-eval-a3`.
- Error: none.
- Commands and raw outputs: `ATTEMPT.json`, per-bank generation/scoring logs.

## 2026-11-12T20:45:50.539590+00:00 — T0b training attempt 3

- Status: passed; GPU: 2; training GPU-hours: 8.803642483340276.
- Output: `/opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1235-t17-train-a3`; error: none.

## 2026-11-12T20:47:42.120458+00:00 — T2 training uniform/b20/sel1235-t29 attempt 1

- Status: training_failed; GPU: 2; training GPU-hours: 0.023413.
- Output: `/opt/dlami/nvme/trimdata-work/runs/uniform-b20-sel1235-t29-train-a1`; error: RuntimeError('Fixed trainer exited 1; see training.log').
- Exact command, input hashes, and measurements: `ATTEMPT.json`.

## 2026-11-12T20:47:57.893820+00:00 — T2/T3 controller

- Status: failed; error: RuntimeError('uniform/b20/sel1235-t29 failed; diagnose before retrying').
- Controller GPU-hours: 0; any child costs are recorded by their own attempts.

## 2026-11-12T21:03:01.301864+00:00 — T1 coincide/b20/selfixed-t43 attempt 1

- Status: completed. GPU-hours: 2.469041.
- GPU: 3. Output: `/opt/dlami/nvme/trimdata-work/runs/coincide-b20-selfixed-t43-eval-a1`.
- Error: none.
- Commands and raw outputs: `ATTEMPT.json`, per-bank generation/scoring logs.

## 2026-11-12T21:07:39.105014+00:00 — T1 coincide/b20/selfixed-t29 attempt 1

- Status: completed. GPU-hours: 2.546209.
- GPU: 0. Output: `/opt/dlami/nvme/trimdata-work/runs/coincide-b20-selfixed-t29-eval-a1`.
- Error: none.
- Commands and raw outputs: `ATTEMPT.json`, per-bank generation/scoring logs.

## 2026-11-12T22:21:38.620655+00:00 — T0c uniform/b10/sel1234-t17 local extended attempt 2

- Status: completed; GPU 7; 2.848635 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1234-t17-extended-a2/ATTEMPT.json`.

## 2026-11-12T23:27:47.439450+00:00 — T1/T5 datatailor/b05/sel1234-t29 local extended attempt 1

- Status: failed; GPU 2; 0.168619 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/datatailor-b05-sel1234-t29-extended-a1/ATTEMPT.json`.

## 2026-11-12T23:27:47.654747+00:00 — T1/T5 coincide/b20/selfixed-t29 local extended attempt 1

- Status: failed; GPU 2; 0.169915 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/coincide-b20-selfixed-t29-extended-a1/ATTEMPT.json`.

## 2026-11-12T23:27:47.753017+00:00 — T1/T5 coincide/b20/selfixed-t17 local extended attempt 1

- Status: failed; GPU 2; 0.195729 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/coincide-b20-selfixed-t17-extended-a1/ATTEMPT.json`.

## 2026-11-12T23:27:47.812501+00:00 — T1/T5 coincide/b20/selfixed-t43 local extended attempt 1

- Status: failed; GPU 2; 0.162327 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/coincide-b20-selfixed-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T02:22:37.846682+00:00 — T1/T5 coincide/b05/selfixed-t43 local extended attempt 1

- Status: completed; GPU 5; 2.796664 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/coincide-b05-selfixed-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T02:47:41.333258+00:00 — T1/T5 uniform/b15/sel1235-t29 local extended attempt 1

- Status: failed; GPU 3; 0.165572 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform-b15-sel1235-t29-extended-a1/ATTEMPT.json`.

## 2026-11-13T02:47:41.423965+00:00 — T1/T5 coincide/b05/selfixed-t29 local extended attempt 1

- Status: failed; GPU 3; 1.957505 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/coincide-b05-selfixed-t29-extended-a1/ATTEMPT.json`.

## 2026-11-13T02:47:41.525257+00:00 — T1/T5 datatailor/b05/sel1234-t43 local extended attempt 1

- Status: failed; GPU 3; 1.935910 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/datatailor-b05-sel1234-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T03:39:47.700287+00:00 — T1/T5 datatailor/b20/sel1234-t29 local extended attempt 1

- Status: completed; GPU 2; 2.832060 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/datatailor-b20-sel1234-t29-extended-a1/ATTEMPT.json`.

## 2026-11-13T04:03:30.345493+00:00 — T1/T5 datatailor/b10/sel1234-t29 local extended attempt 1

- Status: completed; GPU 3; 0.944800 GPU-hours.
- Banks: vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/datatailor-b10-sel1234-t29-extended-a1/ATTEMPT.json`.

## 2026-11-13T04:23:50.514019+00:00 — T1/T5 uniform/b05/sel1235-t29 local extended attempt 1

- Status: failed; GPU 6; 1.209995 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform-b05-sel1235-t29-extended-a1/ATTEMPT.json`.

## 2026-11-13T04:23:50.587299+00:00 — T1/T5 uniform/b20/sel1235-t43 local extended attempt 1

- Status: failed; GPU 6; 1.291737 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform-b20-sel1235-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T04:23:50.638440+00:00 — T1/T5 datatailor/b20/sel1234-t43 local extended attempt 1

- Status: failed; GPU 6; 1.284030 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/datatailor-b20-sel1234-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T04:23:50.800502+00:00 — T1/T5 uniform/b05/sel1235-t43 local extended attempt 1

- Status: failed; GPU 6; 1.229200 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform-b05-sel1235-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T04:23:50.889555+00:00 — T1/T5 less_approx/b10/sel1234-t29 local extended attempt 1

- Status: failed; GPU 6; 1.924883 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/less_approx-b10-sel1234-t29-extended-a1/ATTEMPT.json`.

## 2026-11-13T04:33:18.429394+00:00 — T1/T5 uniform/b10/sel1235-t29 local extended attempt 1

- Status: completed; GPU 2; 0.884554 GPU-hours.
- Banks: vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1235-t29-extended-a1/ATTEMPT.json`.

## 2026-11-13T04:37:17.091372+00:00 — T1/T5 datatailor/b10/sel1234-t43 local extended attempt 1

- Status: completed; GPU 1; 1.001228 GPU-hours.
- Banks: vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/datatailor-b10-sel1234-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T04:54:25.811792+00:00 — T1/T5 uniform/b10/sel1235-t43 local extended attempt 1

- Status: completed; GPU 7; 0.911946 GPU-hours.
- Banks: vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1235-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T04:59:12.633874+00:00 — T1/T5 datatailor/b15/sel1234-t29 local extended attempt 1

- Status: completed; GPU 0; 3.102257 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/datatailor-b15-sel1234-t29-extended-a1/ATTEMPT.json`.

## 2026-11-13T05:47:31.051042+00:00 — T1/T5 uniform/b10/sel1236-t43 local extended attempt 1

- Status: completed; GPU 7; 0.878139 GPU-hours.
- Banks: vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform-b10-sel1236-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T06:27:05.484609+00:00 — T1/T5 less_approx/b10/sel1234-t43 local extended attempt 1

- Status: failed; GPU 7; 2.940977 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/less_approx-b10-sel1234-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T06:27:05.824250+00:00 — T1/T5 uniform_scored_random/b10/sel1234-t29 local extended attempt 1

- Status: failed; GPU 5; 1.464908 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform_scored_random-b10-sel1234-t29-extended-a1/ATTEMPT.json`.

## 2026-11-13T06:27:05.983014+00:00 — T1/T5 uniform/b15/sel1235-t43 local extended attempt 1

- Status: failed; GPU 0; 2.024436 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform-b15-sel1235-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T06:27:06.360014+00:00 — T1/T5 uniform_scored_random/b10/sel1234-t43 local extended attempt 1

- Status: failed; GPU 2; 2.223339 GPU-hours.
- Banks: mme-perception, mme-cognition, seed-image, mmmu-val, mmbench-en-dev, vizwiz-test. Judges deferred.
- Commands, scores, hashes and errors: `/opt/dlami/nvme/trimdata-work/runs/uniform_scored_random-b10-sel1234-t43-extended-a1/ATTEMPT.json`.

## 2026-11-13T06:27:06.768883+00:00 — T3 training datatailor/b15/sel1234-t17 attempt 1

- Status: training_failed; GPU: 4; training GPU-hours: 9.680259.
- Output: `/opt/dlami/nvme/trimdata-work/runs/datatailor-b15-sel1234-t17-train-a1`; error: RuntimeError('Fixed trainer exited -6; see training.log').
- Exact command, input hashes, and measurements: `ATTEMPT.json`.

## 2026-11-13T18:51:17.143083+00:00 — Contract update 3ecf666 incorporated

- Required T12 VisNec added at P10: pinned official code, top-15 released-subset fidelity check, eligible-pool 10% selection preserving realised K, student seeds 17/29/43 and core/local-extended evaluation. VisNec 5%/20% requires P13 lead approval.
- Four updated work-order files merged with existing local changes; A100 40GB correction and benchmark judge deferral preserved. No commit, weights or credentials added.
- Refreshed outcomes: 8/24 provided local extended gaps completed; 16 failed first attempts. T3 failed after last logged step 460/774 with a CUDA launch failure. No campaign GPU child active at the update check.
- Scheduling reservations and passed-calibration readiness need correction before retries; correlated 06:27 CUDA failures need diagnosis. No retries launched during this update.
- Updated continuation plan, instance status, calibration report and preparation-only VISNEC_FIDELITY.md. Evidence: provenance/contract-update-20261113.json and provenance/status-20261113.json.
- Administrative update GPU-hours: 0; prior attempt costs retained unchanged in RESULTS.csv.
