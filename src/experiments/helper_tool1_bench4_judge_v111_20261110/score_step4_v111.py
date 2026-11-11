"""Registered BENCH4 GPT-judge step v1.11 for Tool1 (MM-Vet + LLaVA-Bench-in-the-Wild), judge kind only.

v1.11 (2026-11-10) is v1.10 (helper_tool1_bench4_judge_v110_20261109/score_step4_v110.py, unchanged) with one
custody-only change for the CELLS_V7 extended-suite producers t2-m1ur-b05 and t2-univn-b05 (Tool1 dispatch, 2026-11-10):
  * a ninth generation family, bench4-v3c7 = the CELLS_V7 cells outside CELLS_V3, CELLS_V5 and CELLS_V6, read through
    the pinned CELLS_V7 BENCH4 controller (helper_tool1_cells_v7_extsuite_20261110/prepare7_b4.py), whose roots, owner,
    attempts and banks are the v3 controller's own. Campaign roster order v1 + v3 + Tool2 v1..v4 + v3c5 + v3c6 + v3c7,
    so every cell's PRIMARY (rank 0) precedes any SUPPLEMENTARY.
  * the afterok plan mode accepts CELLS_V5, CELLS_V6 or CELLS_V7 generations, each bound to its own version's
    controller, plan schema and roster pin (generation_controller is recorded per cell).
  * version identity: SOURCE, CONTROL scoring-judge-v111, owner, statuses, comment, job prefix dsmm-t1-b4j-v111-; v1.10
    is retired by RETIRED.json (sealed by v1.11 `retire --version v1.10` after v1.10's own `withdraw` of its PARTIAL
    budget-stopped batch-0, whose PRIMARY scores stay valid evidence) and joins the foreign judge controls.
Judge policy file judge4_v13.py, prompts, models (PRIMARY gpt-4o-2024-08-06, SUPPLEMENTARY gpt-5.6-terra, no
CALIBRATION), request parameters, prices, the SAME shared append-only ledger, STOP_USD 80.00 / HARD_BUDGET_USD 81.29,
single live judge writer and every receipt schema are unchanged. Text below this paragraph is the v1.10 description.

Registered BENCH4 GPT-judge step v1.10 for Tool1 (MM-Vet + LLaVA-Bench-in-the-Wild), judge kind only.

v1.10 (2026-11-09) is v1.9 (helper_tool1_bench4_judge_v19_20261108/score_step4_v19.py, unchanged) with one custody-only
change for the CELLS_V6 extended-suite producer t2-m1ur-b20 (Tool2 request relayed by Tool1, 2026-11-09):
  * an eighth generation family, bench4-v3c6 = the CELLS_V6 cells outside CELLS_V3 and CELLS_V5 (t2-m1ur-b20), read
    through the pinned CELLS_V6 BENCH4 controller (helper_tool1_cells_v6_extsuite_20261109/prepare6_b4.py), whose
    roots, owner, attempts and banks are the v3 controller's own. Campaign roster order v1 + v3 + Tool2 v1..v4 + v3c5 +
    v3c6, so every cell's PRIMARY (rank 0) precedes any SUPPLEMENTARY.
  * the afterok plan mode accepts CELLS_V5 or CELLS_V6 generations, each bound to its own version's controller, plan
    schema and roster pin (generation_controller is recorded per cell).
  * version identity: SOURCE, CONTROL scoring-judge-v110, owner, statuses, comment, job prefix dsmm-t1-b4j-v110-; v1.9 is
    retired by RETIRED.json (sealed by v1.10 `retire --version v1.9` after v1.9's own `withdraw` of its PARTIAL
    budget-stopped batch-1; batch-0 was withdrawn before start) and joins the foreign judge controls.
Judge policy file judge4_v13.py, prompts, models (PRIMARY gpt-4o-2024-08-06, SUPPLEMENTARY gpt-5.6-terra, no
CALIBRATION), request parameters, prices, the SAME shared append-only ledger, STOP_USD 80.00 / HARD_BUDGET_USD 81.29,
single live judge writer and every receipt schema are unchanged. Text below this paragraph is the v1.9 description.

Registered BENCH4 GPT-judge step v1.9 for Tool1 (MM-Vet + LLaVA-Bench-in-the-Wild), judge kind only.

v1.9 (2026-11-08) is v1.8 (helper_tool1_bench4_judge_v18_20261107/score_step4_v18.py, unchanged) with one custody-only
change for the CELLS_V5 extended-suite view of the budget-curve producers (author dispatch via Tool1, 2026-11-08):
  * a seventh generation family, bench4-v3c5 = the CELLS_V5 cells outside CELLS_V3 (Tool1 trainer v10 t2-m1uvn-b05,
    Tool3 chain-v3 coincide-b05-s17), read through the pinned CELLS_V5 BENCH4 controller
    (helper_tool1_cells_v5_extsuite_20261108/prepare5_b4.py), whose control/output roots, owner, attempts and banks
    are the v3 controller's own; CELLS_V5's b6 cells are CELLS_V3 cells and stay in the bench4-v3 family. Campaign
    roster order v1 + v3 + Tool2 v1..v4 + v3c5, so every cell's PRIMARY (rank 0) precedes any SUPPLEMENTARY.
  * an afterok plan mode (`plan --afterok`) for CELLS_V5 generations only: the plan binds each cell's newest
    generation ACK made by prepare5_b4.py, its job and the one BENCH4 input inventory RESULT the generation plans bind;
    the request carries --dependency=afterok:<jobs>; the worker, after every bound job COMPLETED, binds the single
    complete ATTEMPT_RESULT of exactly each bound ACK (released with that inventory), pins its MM-Vet/LLaVA-W
    predictions (LATE_BINDING-<job>.json) and only then runs the unchanged judge_work. Ordinary plans are unchanged.
  * version identity: SOURCE, CONTROL scoring-judge-v19, owner, statuses, comment, job prefix dsmm-t1-b4j-v19-; v1.8
    is retired by RETIRED.json (sealed by v1.9 `retire --version v1.8`; its batch-0 is completely scored) and joins
    the foreign judge controls; live_plan_cells reads each plan's function cells (equal to its generation results).
Judge policy file judge4_v13.py, prompts, models (PRIMARY gpt-4o-2024-08-06, SUPPLEMENTARY gpt-5.6-terra, no
CALIBRATION), request parameters, prices, the SAME shared append-only ledger, STOP_USD 80.00 / HARD_BUDGET_USD 81.29,
single live judge writer and every receipt schema are unchanged. Text below this paragraph is the v1.8 description.

Registered BENCH4 GPT-judge step v1.8 for Tool1 (MM-Vet + LLaVA-Bench-in-the-Wild), judge kind only.

v1.8 (2026-11-07) is v1.7 (helper_tool1_bench4_judge_v17_20261107/score_step4_v17.py, unchanged) with one custody-only
change: a sixth generation family, tool2-extbench-v4 = the single cell of Tool2's sealed
collaboration/helper-tool2-baselines-20261101/EXTBENCH_GENERATION_MANIFEST_V4.json over its single pinned roster
extbench-v3d (t2-lessa-t17 = LESS-approx (not LESS): count-sketch SGD gradient features, 3x-Uniform candidate
pre-filter, 2 dev targets GQA-val/SQA-val (benchmark-aligned, not blind to the benchmark tasks); generation job 11077884, producer
11073967 kind preempt, HOME-sealed Tool2 TRAINING_RESULT), is judged with the identical policy. Author dispatch
(relayed, 2026-11-07): score ONE more Tool2 cell exactly as the V3 batch (judge v1.7 job 11078010). Changes only:
  * families = bench4-v1 + bench4-v3 + tool2-extbench-v1 + tool2-extbench-v2 + tool2-extbench-v3 (unchanged pins) +
    tool2-extbench-v4 (t2family.py v4 of helper_tool1_t2_extbench_scoring_v4_20261107, manifest V4, CELLS_T2 v3d,
    Tool2 v3d prepare.py, custody {"extbench-v3d": custody_t2-v3d.py} = the v3c custody module plus one ROOTS entry,
    the Tool2 preempt-train-v3 chain root). Campaign roster order v1 + v3 + Tool2 v1 + v2 + v3 + v4, so every cell's
    PRIMARY (rank 0) precedes any SUPPLEMENTARY (rank 100) in the spend reservation.
  * version identity: SOURCE, CONTROL scoring-judge-v18, owner, statuses, comment, job prefix dsmm-t1-b4j-v18-; v1.7
    is retired by RETIRED.json (sealed by v1.8 `retire --version v1.7`; its PARTIAL budget-stopped batch-0 is
    withdrawn by v1.7's own `withdraw`, keeping every SCORE file) and joins the foreign judge controls.
Judge policy file judge4_v13.py, prompts, models (PRIMARY gpt-4o-2024-08-06, SUPPLEMENTARY gpt-5.6-terra, no
CALIBRATION), request parameters, prices, the SAME shared append-only ledger, STOP_USD 80.00 / HARD_BUDGET_USD 81.29,
single live judge writer and every receipt schema are unchanged. Text below this paragraph is the v1.7 description.

Registered BENCH4 GPT-judge step v1.7 for Tool1 (MM-Vet + LLaVA-Bench-in-the-Wild), judge kind only.

v1.7 (2026-11-07) is v1.6 (helper_tool1_bench4_judge_v16_20261107/score_step4_v16.py, unchanged) with one custody-only
change: a fifth generation family, tool2-extbench-v3 = the 6 cells of Tool2's sealed
collaboration/helper-tool2-baselines-20261101/EXTBENCH_GENERATION_MANIFEST_V3.json over its two pinned rosters
extbench-v3b (t2-m1l-t43, t2-m1g-t43, t2-m1ud-t17, t2-unid-t17, t2-unisr-t17) and extbench-v3c (t2-p34-s1235-t17),
producer kinds general/preempt (HOME-sealed Tool2 TRAINING_RESULTs), is judged with the identical policy. Author
dispatch (relayed, 2026-11-07): Tool2 asked Tool1 to score 6 more cells exactly as the V2 batch (judge v1.6 job
11075657). Changes only:
  * families = bench4-v1 + bench4-v3 + tool2-extbench-v1 + tool2-extbench-v2 (unchanged pins) + tool2-extbench-v3
    (t2family.py v3 of helper_tool1_t2_extbench_scoring_v3_20261107, manifest V3, CELLS_T2 v3b/v3c, Tool2 v3b/v3c
    prepare.py). Campaign roster order v1 + v3 + Tool2 v1 + Tool2 v2 + Tool2 v3, so every cell's PRIMARY (rank 0)
    precedes any SUPPLEMENTARY (rank 100) in the spend reservation.
  * custody per roster version: the two v3 rosters were generated with different Tool2 custody modules (v3b: the v1
    custody_t2.py bytes; v3c: seeds 1235/1236 and the sealed selection-input policy identity), so a family's custody
    entry may map each roster version to its own pinned custody source; t2_family binds one custody module per roster
    version (checked against that roster's own prepare.py pin) and t2_generation validates each cell with its own
    roster version's custody. The v1/v2 families keep their single custody source for every roster version.
  * version identity: SOURCE, CONTROL scoring-judge-v17, owner, statuses, comment, job prefix dsmm-t1-b4j-v17-; v1.6
    is retired by RETIRED.json (sealed by v1.7 `retire --version v1.6`; its PARTIAL budget-stopped batch-0 is
    withdrawn by v1.6's own `withdraw`, keeping every SCORE file) and joins the foreign judge controls.
Judge policy file judge4_v13.py, prompts, models (PRIMARY gpt-4o-2024-08-06, SUPPLEMENTARY gpt-5.6-terra, no
CALIBRATION), request parameters, prices, the SAME shared append-only ledger, STOP_USD 80.00 / HARD_BUDGET_USD 81.29,
single live judge writer and every receipt schema are unchanged. Text below this paragraph is the v1.6 description.


v1.6 (2026-11-07) is v1.5 (helper_tool1_bench4_judge_v15_20261105/score_step4_v15.py, unchanged) with one custody-only
change: a fourth generation family, tool2-extbench-v2 = the 6 cells of Tool2's sealed
collaboration/helper-tool2-baselines-20261101/EXTBENCH_GENERATION_MANIFEST_V2.json over its single pinned roster
extbench-v4 (Tool1-trained DataTailor K=66,071 t17/29/43, Uniform sel1235 t29/t43, sel1236 t43; producer kind
tool1-baseline through custody_t2.py v4 and HOME-sealed snapshots of the producer /data RESULTs), is judged with the
identical policy. Author dispatch (relayed, 2026-11-07): Tool2 asked Tool1 to score 6 more cells exactly as its 16
cells on 2026-11-05 (judge v1.5 job 11059828). Changes only:
  * families = bench4-v1 + bench4-v3 + tool2-extbench-v1 (unchanged pins) + tool2-extbench-v2 (t2family.py v2 of
    helper_tool1_t2_extbench_scoring_v2_20261107, custody_t2.py v4, manifest V2, CELLS_T2 v4, Tool2 v4 prepare.py).
    Campaign roster order v1 + v3 + Tool2 v1 + Tool2 v2, so every cell's PRIMARY (rank 0) precedes any SUPPLEMENTARY
    (rank 100) in the spend reservation. Both Tool2 families use the same binding/validation code, selected by the
    family's own pinned module/manifest (T2_FAMILY_SOURCES) instead of the fixed third family.
  * version identity: SOURCE, CONTROL scoring-judge-v16, owner, statuses, comment, job prefix dsmm-t1-b4j-v16-; v1.5
    is retired by RETIRED.json (sealed by v1.6 `retire --version v1.5`) and joins the foreign judge controls.
Judge policy file judge4_v13.py, prompts, models (PRIMARY gpt-4o-2024-08-06, SUPPLEMENTARY gpt-5.6-terra, no
CALIBRATION), request parameters, prices, the SAME shared append-only ledger, STOP_USD 80.00 / HARD_BUDGET_USD 81.29,
single live judge writer and every receipt schema are unchanged. Text below this paragraph is the v1.5 description.


v1.5 (2026-11-05) is v1.4 (helper_tool1_bench4_judge_v14_20261105/score_step4_v14.py, unchanged) with one change: a
third generation family, Tool2's 16 extended-benchmark students (Tool2 custody), is judged with the identical policy.
Author dispatch (relayed, 2026-11-05): Tool2 asked Tool1 to score its 16 extbench students with Tool1's existing
reviewed scorers and GPT judge exactly as Tool1 scores its own baselines; add the 16 cells to the judge reservation
with PRIMARY first; SUPPLEMENTARY may become partial. Changes only:
  * families = bench4-v1 (CELLS_BENCH6 eligible) + bench4-v3 (CELLS_V3) + tool2-extbench-v1 (the cells of Tool2's
    sealed EXTBENCH_GENERATION_MANIFEST_V1.json over its three pinned CELLS_T2 rosters). Campaign roster order is
    v1 + v3 + Tool2, so the bound policy's campaign_runs gives every cell's PRIMARY (rank 0) before any SUPPLEMENTARY
    (rank 100) in the spend reservation. Tool2 cells are bound through t2family.py (helper_tool1_t2_extbench_scoring_
    v1_20261105, pinned): Tool2's single COMPLETE HOME ATTEMPT_RESULT-bench4-<completing job> found through Tool2's own
    ACK/RELEASE_INTENT, Tool1's validation semantics with the Tool2 custody basis (accepted_inference_files equal to
    the producer TRAINING_RESULT hashes via custody_t2.py) instead of prior parameter parity, manifest attempt exports
    byte-identical to the bank-level exports, and in the worker the manifest-pinned /data RESULT.json equal to the
    HOME ATTEMPT_RESULT. Bank inputs/labels are Tool1's own BENCH4 inventory (the RESULT every Tool2 release bound).
    An earlier Tool2 attempt's COMPLETE bench4 result (resume-stable journals) is accepted only if it names identical
    parameters, inference files and bank-level exports.
  * version identity: SOURCE, CONTROL scoring-judge-v15, owner, statuses, comment, job prefix dsmm-t1-b4j-v15-; v1.4
    is retired by RETIRED.json (sealed by v1.5 `retire --version v1.4`) and joins the foreign judge controls.
Judge policy file judge4_v13.py, prompts, models (PRIMARY gpt-4o-2024-08-06, SUPPLEMENTARY gpt-5.6-terra, no
CALIBRATION), request parameters, prices, the SAME shared append-only ledger, STOP_USD 80.00 / HARD_BUDGET_USD 81.29,
single live judge writer and every receipt schema are unchanged. Text below this paragraph is the v1.4 description.

v1.4 (2026-11-05) is v1.3 (helper_tool1_bench4_judge_v13_20261105/score_step4_v13.py, unchanged) with one policy
change: the CALIBRATION role (gpt-4-0613) is removed from the campaign, the spend reservation and every batch.
Author, 2026-11-05: "do whatever you think could be most efficient for us, focus on the coverage first."
Evidence: /v1/models inside judge jobs 11051994 (v1.2) and 11052460 (v1.3) did not list gpt-4-0613 (MODELS-<job>.json
bound by exact descriptors in every plan), so under v1.3 any batch holding a CALIBRATION cell ends ABORTED.
ACTIVE_ROLES = PRIMARY (gpt-4o-2024-08-06) -> SUPPLEMENTARY (gpt-5.6-terra): their ranks, prompts, models, request
parameters and run order are those of the exact bound v1.3 policy file judge4_v13.py (unchanged; STOP_USD 80.00,
HARD_BUDGET_USD 81.29) and the SAME shared ledger. v1.3 is retired by RETIRED.json (sealed by v1.4 `retire`); the
existing v1.1/v1.2 retirements sealed by v1.3 stay valid. Text below this paragraph is the v1.3 description.

v1.3 (2026-11-05) is v1.2 (helper_tool1_bench4_judge_v12_20261105/score_step4_v12.py, unchanged) with three changes:
  1. Budget. Author, 2026-11-05: "let's continue to use up all budget, do not save budget for me." The policy file
     judge4_v13.py is byte-identical to judge4_v11.py except STOP_USD = 80.00 (HARD_BUDGET_USD stays 81.29; the margin
     covers in-flight concurrent calls). The SAME shared append-only JUDGE_LEDGER is used, so cached calls replay free
     and every earlier paid call counts toward the stop. Prompts, models, roles, ranks and run ordering are unchanged.
  2. CELLS_V3 students. Generation results are bound per registered family: CELLS_BENCH6 cells through the BENCH4 v1
     control/owner/output roots (as before) and CELLS_V3 cells through the exact bound BENCH4 v3 controller
     (helper_tool1_bench4_eval_v3_20261104/prepare4.py) and its own CONTROL_ROOT, OWNER, OUTPUT_ROOT, MAX_ATTEMPTS and
     CELLS_V3 roster (custody3.check_cells), with identical validation semantics. The campaign roster used for the
     PRIMARY -> CALIBRATION -> SUPPLEMENTARY spend reservation is CELLS_BENCH6 eligible + CELLS_V3 registered cells.
  3. Single live writer. v1.3 plans, submits and releases only while every earlier judge version (v1.1, v1.2) carries
     an exact RETIRED.json naming exactly its existing batches, sealed by `retire` only with no live job of that
     version and every batch WITHDRAWN or completely scored; earlier non-withdrawn plans stay foreign (their cells are
     never rebound). One live dsmm-t1-b4j* job at a time and the ledger's exclusive flock are unchanged.
Text below this paragraph is the v1.2 description.

v1.2 (2026-11-05) is v1.1 (helper_tool1_bench4_judge_v11_20261104/score_step4_v11.py, unchanged and still bound by
its own plans) with one functional change: the reviewed route is chosen at plan time (--route general|preempt) exactly
as in helper_tool1_bench4_eval_v1_20261104/score_step4.py (ROUTES/check_route, per-route PLAN/OWNER_REVIEW/
INDEPENDENT_REVIEW files whose functions must be identical across routes, routes.py resources/expected_readback, the
route passed through the sbatch argv to the worker). The author authorized Babel General and Preempt for this
campaign (2026-11-05). Versioned identities: own SOURCE/CONTROL roots (scoring-judge-v12), owner, statuses, comment
and job prefix dsmm-t1-b4j-v12-; v1.1 plans are foreign judge controls (non-withdrawn v1.1 cells are never rebound).
The judge policy is the exact v1.1 judge4_v11.py file bound by hash; prompts, models, prices, the shared append-only
ledger/cache and the 70 USD stop are unchanged. Text below this paragraph is the v1.1 description.

Supersedes the never-run judge path of helper_tool1_bench4_eval_v1_20261104/score_step4.py (no v1 judge plan exists)
and the unbound draft helper_tool1_bench4_judge_v1_1_20261104; the v1 local (MMMU/MMBench) scoring path is unchanged
and stays in score_step4.py --kind local. Staged v1 sources used by released generation jobs are not edited: banks4.py,
CELLS_BENCH6.json, routes.py and the pinned official references are bound here by exact bytes; generation outputs are
bound by the exact ATTEMPT_RESULT descriptor and the exact PREDICTIONS.jsonl descriptor of each judge bank.

Author-approved final judge policy (judge4_v11.py): PRIMARY gpt-4o-2024-08-06 (temperature 0, one run, every cell,
both banks), official CALIBRATION gpt-4-0613 on MM-Vet for uniform-s17, coincide-s17, uniform-s29, coincide-s29,
uniform-s43, coincide-s43 (priority order), SUPPLEMENTARY gpt-5.6-terra (models-list-verified alias, returned
snapshot recorded and held fixed; every cell, both banks; parse failures recorded as missing). Spend order PRIMARY ->
CALIBRATION -> SUPPLEMENTARY across the whole 19-cell campaign via a priority reservation; one shared append-only
ledger/cache for every BENCH4 judge version (exclusive flock) stops every call that would cross 70 USD (hard author
budget 81.29 USD) and ends the attempt cleanly with a PARTIAL RESULT.

Runs in this stdlib process inside one owned Babel CPU+GPU allocation (Babel rule; no model is loaded) on the
reviewed route of its plan (v1.2: general/normal or preempt/preempt_qos, both --no-requeue). A batch plan binds,
per cell, the exact HOME
ATTEMPT_RESULT of the cell's BENCH4 generation with status COMPLETE_ALL_PLANNED_BANKS and the single inventory RESULT
every bound generation released with, so a plan can only be made after generation finished (no afterok).

  score_step4_v17.py retire      --version v1.3 --note "..."
  score_step4_v17.py plan        --batch B --route R --cell C [--cell C ...]
  score_step4_v17.py self-review --batch B --route R --note "..."
  score_step4_v17.py submit|release --batch B --attempt N --route R
  score_step4_v17.py withdraw    --batch B --route R --note "..."
A PARTIAL attempt (deadline or budget stop) or an interrupted one (TIMEOUT/PREEMPTED/NODE_FAIL) continues with
--attempt N+1 (cached calls replay free). A FAILED attempt, including an ABORTED RESULT (non-retryable API error,
foreign/changed snapshot, judge model not listed), continues only after an owner attempt-N/DIAGNOSIS-<job>.json and an
independent DIAGNOSIS_REVIEW-<job>.json bound to that job and this controller.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import time
from types import ModuleType


HOME = Path("/home/luyangz/data_selection_multimodal")
SOURCE = HOME / "experiments/helper_tool1_bench4_judge_v111_20261110"
CELLS5 = HOME / "experiments/helper_tool1_cells_v5_extsuite_20261108"
CELLS6 = HOME / "experiments/helper_tool1_cells_v6_extsuite_20261109"
CELLS7 = HOME / "experiments/helper_tool1_cells_v7_extsuite_20261110"
T2SCORE = HOME / "experiments/helper_tool1_t2_extbench_scoring_v1_20261105"
T2SCORE2 = HOME / "experiments/helper_tool1_t2_extbench_scoring_v2_20261107"
T2SCORE3 = HOME / "experiments/helper_tool1_t2_extbench_scoring_v3_20261107"
T2SCORE4 = HOME / "experiments/helper_tool1_t2_extbench_scoring_v4_20261107"
T2 = {"extbench-v1": HOME / "experiments/helper_tool2_extbench_v1_20261104",
      "extbench-v2": HOME / "experiments/helper_tool2_extbench_v2_20261105",
      "extbench-v3": HOME / "experiments/helper_tool2_extbench_v3_20261105",
      "extbench-v4": HOME / "experiments/helper_tool2_extbench_v4_20261106",
      "extbench-v3b": HOME / "experiments/helper_tool2_extbench_v3b_20261107",
      "extbench-v3c": HOME / "experiments/helper_tool2_extbench_v3c_20261107",
      "extbench-v3d": HOME / "experiments/helper_tool2_extbench_v3d_20261107"}
T2_CONTROL = HOME / "collaboration/helper-tool2-method-20261101"
T2_BASELINES = HOME / "collaboration/helper-tool2-baselines-20261101"
JUDGE_V13 = HOME / "experiments/helper_tool1_bench4_judge_v13_20261105"
EVAL = HOME / "experiments/helper_tool1_bench4_eval_v1_20261104"
EVAL_V3 = HOME / "experiments/helper_tool1_bench4_eval_v3_20261104"
BENCH6 = HOME / "experiments/helper_tool1_bench6_eval_v1_20261103"
V1 = HOME / "experiments/helper_tool1_vizwiz_eval_v1_20261102"
EVAL_CONTROL = HOME / "collaboration/helper-tool1-baseline-independent-20261101/bench4-eval-v1"
CONTROL = EVAL_CONTROL / "scoring-judge-v111"
LEDGER = EVAL_CONTROL / "scoring" / "judge-ledger" / "JUDGE_LEDGER.jsonl"
FOREIGN_JUDGE_CONTROLS = (EVAL_CONTROL / "scoring", EVAL_CONTROL / "scoring-v1_1", EVAL_CONTROL / "scoring-judge-v11",
                          EVAL_CONTROL / "scoring-judge-v12", EVAL_CONTROL / "scoring-judge-v13",
                          EVAL_CONTROL / "scoring-judge-v14", EVAL_CONTROL / "scoring-judge-v15",
                          EVAL_CONTROL / "scoring-judge-v16", EVAL_CONTROL / "scoring-judge-v17",
                          EVAL_CONTROL / "scoring-judge-v18", EVAL_CONTROL / "scoring-judge-v19",
                          EVAL_CONTROL / "scoring-judge-v110")
RETIRED_VERSIONS = {"v1.1": (EVAL_CONTROL / "scoring-judge-v11", "dsmm-t1-b4j-v11-"),
                    "v1.2": (EVAL_CONTROL / "scoring-judge-v12", "dsmm-t1-b4j-v12-"),
                    "v1.3": (EVAL_CONTROL / "scoring-judge-v13", "dsmm-t1-b4j-v13-"),
                    "v1.4": (EVAL_CONTROL / "scoring-judge-v14", "dsmm-t1-b4j-v14-"),
                    "v1.5": (EVAL_CONTROL / "scoring-judge-v15", "dsmm-t1-b4j-v15-"),
                    "v1.6": (EVAL_CONTROL / "scoring-judge-v16", "dsmm-t1-b4j-v16-"),
                    "v1.7": (EVAL_CONTROL / "scoring-judge-v17", "dsmm-t1-b4j-v17-"),
                    "v1.8": (EVAL_CONTROL / "scoring-judge-v18", "dsmm-t1-b4j-v18-"),
                    "v1.9": (EVAL_CONTROL / "scoring-judge-v19", "dsmm-t1-b4j-v19-"),
                    "v1.10": (EVAL_CONTROL / "scoring-judge-v110", "dsmm-t1-b4j-v110-")}
RETIRED_STATUS = "OWNER_RETIRED_BENCH4_JUDGE_VERSION_NO_LIVE_JOB_SUPERSEDED_BY_V111"
RETIRED_SCHEMA = "tool1-bench4-judge-version-retirement-v111"
# Retirements sealed by v1.3 (v1.1, v1.2), v1.4 (v1.3), v1.5 (v1.4), v1.6 (v1.5) and v1.7 (v1.6) remain exact and
# valid; v1.8 sealed v1.7's; v1.9 sealed v1.8's; v1.10 sealed v1.9's; v1.11 seals its own (v1.10).
ACCEPTED_RETIREMENTS = (("tool1-bench4-judge-version-retirement-v13",
                         "OWNER_RETIRED_BENCH4_JUDGE_VERSION_NO_LIVE_JOB_SUPERSEDED_BY_V13",
                         "Tool1-owned-BENCH4-registered-judge-v13"),
                        ("tool1-bench4-judge-version-retirement-v14",
                         "OWNER_RETIRED_BENCH4_JUDGE_VERSION_NO_LIVE_JOB_SUPERSEDED_BY_V14",
                         "Tool1-owned-BENCH4-registered-judge-v14"),
                        ("tool1-bench4-judge-version-retirement-v15",
                         "OWNER_RETIRED_BENCH4_JUDGE_VERSION_NO_LIVE_JOB_SUPERSEDED_BY_V15",
                         "Tool1-owned-BENCH4-registered-judge-v15"),
                        ("tool1-bench4-judge-version-retirement-v16",
                         "OWNER_RETIRED_BENCH4_JUDGE_VERSION_NO_LIVE_JOB_SUPERSEDED_BY_V16",
                         "Tool1-owned-BENCH4-registered-judge-v16"),
                        ("tool1-bench4-judge-version-retirement-v17",
                         "OWNER_RETIRED_BENCH4_JUDGE_VERSION_NO_LIVE_JOB_SUPERSEDED_BY_V17",
                         "Tool1-owned-BENCH4-registered-judge-v17"),
                        ("tool1-bench4-judge-version-retirement-v18",
                         "OWNER_RETIRED_BENCH4_JUDGE_VERSION_NO_LIVE_JOB_SUPERSEDED_BY_V18",
                         "Tool1-owned-BENCH4-registered-judge-v18"),
                        ("tool1-bench4-judge-version-retirement-v19",
                         "OWNER_RETIRED_BENCH4_JUDGE_VERSION_NO_LIVE_JOB_SUPERSEDED_BY_V19",
                         "Tool1-owned-BENCH4-registered-judge-v19"),
                        ("tool1-bench4-judge-version-retirement-v110",
                         "OWNER_RETIRED_BENCH4_JUDGE_VERSION_NO_LIVE_JOB_SUPERSEDED_BY_V110",
                         "Tool1-owned-BENCH4-registered-judge-v110"),
                        (RETIRED_SCHEMA, RETIRED_STATUS, "Tool1-owned-BENCH4-registered-judge-v111"))
RETIRE_COMMAND_VERSIONS = ("v1.10",)
ACTIVE_ROLES = ("PRIMARY", "SUPPLEMENTARY")
CALIBRATION_EVIDENCE = (EVAL_CONTROL / "scoring-judge-v12" / "judge-batch-0" / "attempt-0" / "MODELS-11051994.json",
                        EVAL_CONTROL / "scoring-judge-v13" / "judge-batch-0" / "attempt-0" / "MODELS-11052460.json")
SCORED_PREFIX = "COMPLETE_BENCH4_JUDGE_"
AUTHORIZATION = dict(
    quote="let's continue to use up all budget, do not save budget for me.", by="author (relayed by Tool1)",
    date="2026-11-05", effect="STOP_USD 70.00 -> 80.00 in judge4_v13.py; HARD_BUDGET_USD 81.29 unchanged; same "
                              "shared ledger (earlier spend counts, cached calls replay free)")
COVERAGE_AUTHORIZATION = dict(
    quote="do whatever you think could be most efficient for us, focus on the coverage first.",
    by="author (relayed by Tool1)", date="2026-11-05",
    effect="CALIBRATION (gpt-4-0613, unavailable) removed from campaign, reservation and batches; every student gets "
           "PRIMARY gpt-4o-2024-08-06 + SUPPLEMENTARY gpt-5.6-terra on MM-Vet and LLaVA-W; no extra judges or repeats")
OUTPUT_ROOT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-bench4-eval-v1-20261104")
INPUT_ROOT = Path("/data/hf_cache/project_artifacts/DS-MultiModal/helper-tool1-bench4-inputs-v1-20261104")
OWNER = "Tool1-owned-BENCH4-registered-judge-v111"
GENERATION_OWNER = "Tool1-owned-BENCH4-existing-student-evaluation-v1"
REVIEW_STATUS = "OWNER_REVIEWED_TOOL1_BENCH4_JUDGE_V111_NOT_INDEPENDENT_ACCEPTANCE"
INDEPENDENT_STATUS = "PASS_INDEPENDENT_EXACT_SOURCE_CONFIG_TOOL1_BENCH4_JUDGE_V111"
GENERATION_COMPLETE = "COMPLETE_ALL_PLANNED_BANKS"
BANK_COMPLETE = "COMPLETE_FULL_BANK_PREDICTIONS"
INPUT_STATUS = "COMPLETE_BENCH4_INPUT_INVENTORY_V1"
RESULT_STATUS = "COMPLETE_BENCH4_JUDGE_V111_SCORES"
PARTIAL_STATUS = "PARTIAL_BENCH4_JUDGE_V111_STOPPED_DEADLINE_OR_BUDGET"
ABORTED_STATUS = "ABORTED_BENCH4_JUDGE_V111_API_OR_MODEL_ERROR"
COMMENT_PREFIX = "dsmm-t1-b4-judge-v111:"
UID = 2708899
EXCLUDED = "babel-p9-20"
ROUTES = ("general", "preempt")
MINUTES = 240
CPUS = "2"
MEMORY = "16G"
MEMORY_MB = "16384M"
JOB_PREFIX = "dsmm-t1-b4j-v111-"
ANY_JUDGE_PREFIX = "dsmm-t1-b4j"
MAX_BATCHES = 12
MAX_ATTEMPTS = 6
GENERATION_ATTEMPTS = 8
MAX_EXPORT = 256 << 20
V5_GENERATION_SCHEMA = "tool1-bench4-eval-v3-custody-v5-plan-v1"
V6_GENERATION_SCHEMA = "tool1-bench4-eval-v3-custody-v6-plan-v1"
V7_GENERATION_SCHEMA = "tool1-bench4-eval-v3-custody-v7-plan-v1"
V5_FAMILIES = ("bench4-v3", "bench4-v3c5", "bench4-v3c6", "bench4-v3c7")
GENERATION_LIVE = ("PENDING", "RUNNING", "COMPLETED")
DEPENDENCY_ITEM = re.compile(r"afterok:([1-9][0-9]*)(?:_\*)?(?:\((unfulfilled|fulfilled)\))?")
BANKS = ("mmmu-val", "llava-wild", "mm-vet", "mmbench-en-dev")
JUDGE_BANKS = ("mm-vet", "llava-wild")
TERMINAL = ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
INTERRUPTED = ("TIMEOUT", "PREEMPTED", "NODE_FAIL")
STABLE = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
RESERVE_SECONDS = 600
DIAGNOSIS_SCHEMA = "tool1-bench4-failed-judge-v111-attempt-diagnosis"
DIAGNOSIS_STATUS = "OWNER_DIAGNOSED_FAILED_JUDGE_V111_ATTEMPT"
DIAGNOSIS_REVIEW_STATUS = "PASS_INDEPENDENT_FAILED_JUDGE_V111_DIAGNOSIS_TOOL1_BENCH4"
WITHDRAWN_STATUS = "OWNER_WITHDRAWN_BENCH4_JUDGE_V111_BATCH_NO_LIVE_JOB_NO_COMPLETE_RESULT"
HELPER = {
    "path": str(HOME / "experiments/helper_method_eval_bank_qual_v1_20261102/prepare.py"),
    "bytes": 51200, "sha256": "f0dc4f6a548eedc45d5ba3d92e0251328d33bb82b274d815f14b0aca25fa7357",
}
REF = EVAL / "reference"
SOURCES = {
    "judge4_v13.py": {"path": str(JUDGE_V13 / "judge4_v13.py"), "bytes": 44369,
                      "sha256": "639c5506f787ce9a9fe39a4fe25803a765452b8095d25c3c07b5af0d4b939a43"},
    "bench4_v3_prepare4.py": {"path": str(EVAL_V3 / "prepare4.py"), "bytes": 39588,
                              "sha256": "f03c756601f32c005d9c86e6799fc0e47857e8fd232e390696e5c5184dc2ee32"},
    "bench4_v5_prepare5_b4.py": {"path": str(CELLS5 / "prepare5_b4.py"), "bytes": 42211,
                                 "sha256": "46cabfd4b465726c9176ca4674e87d4acbc171e99bc43201aac7f1585b1bc62f"},
    "bench4_v6_prepare6_b4.py": {"path": str(CELLS6 / "prepare6_b4.py"), "bytes": 42210,
                                 "sha256": "e75bbbfc0c03851237f822688f8f92d03740f31c6f190ab5b242d5dfc28c1e16"},
    "bench4_v7_prepare7_b4.py": {"path": str(CELLS7 / "prepare7_b4.py"), "bytes": 42210,
                                 "sha256": "069be846b61df3b8269bb9b5940f9b32873e8dc998411bccf273adfd334389f3"},
    "CELLS_V7.json": {"path": str(CELLS7 / "CELLS_V7.json"), "bytes": 7796,
                      "sha256": "6276480cafcad44030510ede2de94ebf3420c83139e7054c3785bbb7c657bf8f"},
    "CELLS_V6.json": {"path": str(CELLS6 / "CELLS_V6.json"), "bytes": 4929,
                      "sha256": "68c29e8faf90f3bc7d7e8000ef8c7b8b6e934ab4cb28061890dde01c2ea77a00"},
    "CELLS_V5.json": {"path": str(CELLS5 / "CELLS_V5.json"), "bytes": 28870,
                      "sha256": "22bcace200cfa5bb29c2375105601b401d1a1b90af76f04d7dccfd2ff55ec54b"},
    "banks4.py": {"path": str(EVAL / "banks4.py"), "bytes": 30685,
                  "sha256": "8a3b14c9ce0676db1d7bed24f3d56069222d4816933546eb3e9487ae9e4aadbe"},
    "CELLS_BENCH6.json": {"path": str(BENCH6 / "CELLS_BENCH6.json"), "bytes": 84088,
                          "sha256": "1b38da4831510de4db37ef1498f69343768d0ed917ae85892f339a5674c82f85"},
    "routes.py": {"path": str(V1 / "routes.py"), "bytes": 4797,
                  "sha256": "5067bf831884699edcd0f5d86b5e9c1a3af65def6edf68aab4cb157c3d388709"},
    "mm-vet_evaluator.py": {"path": str(REF / "mmvet-acde843e/mm-vet_evaluator.py"), "bytes": 15232,
                            "sha256": "68cfd25dc1c83d9dbf695c86854749083fc3899398ef88ca2d70cea3ddd82cbf"},
    "rule.json": {"path": str(REF / "llava-c121/rule.json"), "bytes": 9098,
                  "sha256": "dd7345a44ce9f7a24e51a30ee030cdab32346f9d08e3359c80545ff1312b1e07"},
    "eval_gpt_review_bench.py": {"path": str(REF / "llava-c121/eval_gpt_review_bench.py"), "bytes": 4172,
                                 "sha256": "fd6a2c3ed99b0b2881e056160c03aab17f7ea5879b9e03fd993381b4d47e63f2"},
    "summarize_gpt_review.py": {"path": str(REF / "llava-c121/summarize_gpt_review.py"), "bytes": 2438,
                                "sha256": "e5d8856867207f3729877d233f74cb12240d5c998650f22b1aec0ba67c21740b"},
    "t2family.py": {"path": str(T2SCORE / "t2family.py"), "bytes": 12586, "sha256": "5dd4084bf612252c6d938da9d1b6abc6ff7289587d90ac578a634d835bf1bc36"},
    "custody_t2.py": {"path": str(T2["extbench-v1"] / "custody_t2.py"), "bytes": 5783,
                      "sha256": "6a7ebb8114b3d7a8ad25905dd526c9a13f2a235670e1397aeea6eef5356b54e0"},
    "EXTBENCH_GENERATION_MANIFEST_V1.json": {"path": str(T2_CONTROL / "EXTBENCH_GENERATION_MANIFEST_V1.json"),
                                             "bytes": 107546,
                                             "sha256": "95d61dc852e3f7cc2d740e104bb1fd73c4f0ce8af279b45b714de88917a48a1d"},
    "CELLS_T2-extbench-v1.json": {"path": str(T2["extbench-v1"] / "CELLS_T2.json"), "bytes": 7295,
                                  "sha256": "2ccc7993c08233d32b531f67857b22b7bd603291e8b6087c684ecd1dc1e2512e"},
    "CELLS_T2-extbench-v2.json": {"path": str(T2["extbench-v2"] / "CELLS_T2.json"), "bytes": 6405,
                                  "sha256": "9f070fb384b8cf209d1469ffe3799094d6b27949b143872f8a17fae6207a1ba8"},
    "CELLS_T2-extbench-v3.json": {"path": str(T2["extbench-v3"] / "CELLS_T2.json"), "bytes": 1312,
                                  "sha256": "8e6ab1cebe56e9c1bc6068cc3db20675f0cf2d32c4dd78a28a0d50ec227e8a81"},
    "prepare-extbench-v1.py": {"path": str(T2["extbench-v1"] / "prepare.py"), "bytes": 31176,
                               "sha256": "3ff68183845daed4f4f0e5390f0d91186f980aaf16df211402c4c5bd43e82a8c"},
    "prepare-extbench-v2.py": {"path": str(T2["extbench-v2"] / "prepare.py"), "bytes": 31176,
                               "sha256": "e46896c29f881aafb33e86f9a65b529f1fcc7e325d58d1a8cb8b6d3347907cdc"},
    "prepare-extbench-v3.py": {"path": str(T2["extbench-v3"] / "prepare.py"), "bytes": 31176,
                               "sha256": "6d8cd615567f7e71ea20f15af073e0a594b2a324e86cb606d8e77f42450df6bd"},
    "t2family-v2.py": {"path": str(T2SCORE2 / "t2family.py"), "bytes": 12796,
                       "sha256": "8ceaba8a08222dc855280497ead08615bd657c1382a933a073b21491f07a4c53"},
    "custody_t2-v4.py": {"path": str(T2["extbench-v4"] / "custody_t2.py"), "bytes": 7608,
                         "sha256": "ee5c9695013ad1e16dbe4788c6b707c16169cae03bee266b13bcc41a8b1a17d8"},
    "EXTBENCH_GENERATION_MANIFEST_V2.json": {"path": str(T2_BASELINES / "EXTBENCH_GENERATION_MANIFEST_V2.json"),
                                             "bytes": 35414,
                                             "sha256": "a3fa8a9a2d1a2ad77386b7be7df02ef9808a7a2221c3dfd17a6390f003782052"},
    "CELLS_T2-extbench-v4.json": {"path": str(T2["extbench-v4"] / "CELLS_T2.json"), "bytes": 7340,
                                  "sha256": "f750a01ecbbdccebb1cd1e6d659c9c799254b3e6db40a0f401b63f9c2b7307c4"},
    "prepare-extbench-v4.py": {"path": str(T2["extbench-v4"] / "prepare.py"), "bytes": 31176,
                               "sha256": "abfab33447e0835d0ae9caf1a7bb0966baf753f6453f438f5f9a4f43bfe1b890"},
    "t2family-v3.py": {"path": str(T2SCORE3 / "t2family.py"), "bytes": 13106,
                       "sha256": "8eb41d9ab8e09aa0d0e82c6ea399577ff6906fd62f8f26aabe2ad6bb23680f9a"},
    "custody_t2-v3b.py": {"path": str(T2["extbench-v3b"] / "custody_t2.py"), "bytes": 5783,
                          "sha256": "6a7ebb8114b3d7a8ad25905dd526c9a13f2a235670e1397aeea6eef5356b54e0"},
    "custody_t2-v3c.py": {"path": str(T2["extbench-v3c"] / "custody_t2.py"), "bytes": 6504,
                          "sha256": "61a4a4f4b45c01c9a35bf724e4db482f1c3affe189f58ceb776a26a794494db3"},
    "EXTBENCH_GENERATION_MANIFEST_V3.json": {"path": str(T2_BASELINES / "EXTBENCH_GENERATION_MANIFEST_V3.json"),
                                             "bytes": 31200,
                                             "sha256": "9bcc70141142169124f2c2d6e2d202dd08c8a7d24435d8006483295a7f01383b"},
    "CELLS_T2-extbench-v3b.json": {"path": str(T2["extbench-v3b"] / "CELLS_T2.json"), "bytes": 4757,
                                   "sha256": "7e68909807b4b46123c593570a10716496a3f809a24d94cb2630e77fb0dbe422"},
    "CELLS_T2-extbench-v3c.json": {"path": str(T2["extbench-v3c"] / "CELLS_T2.json"), "bytes": 2033,
                                   "sha256": "dc9d8449db8b37a12bc0f302faa8c8907886d345400fb50b1246a8c47117216e"},
    "prepare-extbench-v3b.py": {"path": str(T2["extbench-v3b"] / "prepare.py"), "bytes": 31179,
                                "sha256": "6b48559b23f8135256064106d971f2413ab6f04e37d734f509b7b74a4d8de3e2"},
    "prepare-extbench-v3c.py": {"path": str(T2["extbench-v3c"] / "prepare.py"), "bytes": 31179,
                                "sha256": "6d0ab4d3d3421429f72be6146587fe86b23ca8a26ba2419ad4e906a3cb66ca30"},
    "t2family-v4.py": {"path": str(T2SCORE4 / "t2family.py"), "bytes": 13372,
                       "sha256": "08b0195e0f1956e0790521fb7ecac01d3b234fd224fcd5c0097b3858689c97a6"},
    "custody_t2-v3d.py": {"path": str(T2["extbench-v3d"] / "custody_t2.py"), "bytes": 6605,
                          "sha256": "56137747f17d102fae0619d14f493654e8c9ec8c53c2fa573533261db0df8aeb"},
    "EXTBENCH_GENERATION_MANIFEST_V4.json": {"path": str(T2_BASELINES / "EXTBENCH_GENERATION_MANIFEST_V4.json"),
                                             "bytes": 5454,
                                             "sha256": "c7db2b4f6b8d3e148ff6a9a95697ad79980408aa7d63da3e9c205733cd7eaf09"},
    "CELLS_T2-extbench-v3d.json": {"path": str(T2["extbench-v3d"] / "CELLS_T2.json"), "bytes": 1988,
                                   "sha256": "94597245e4058c633d9e6e681d940206e8e220f74592df6d49ea40bc55282f93"},
    "prepare-extbench-v3d.py": {"path": str(T2["extbench-v3d"] / "prepare.py"), "bytes": 31179,
                                "sha256": "73457c80a2878012a2d7a958bf7d5406b9021ca9303af9ea0926941cc7c518c8"},
}
T2_AUTHORIZATION = dict(
    by="author dispatch to the Tool1 builder (relayed), 2026-11-05",
    quote="score helper Tool2's 16 extended-benchmark students with Tool1's existing reviewed scorers and GPT judge, "
          "exactly as Tool1 scores its own baselines ... Add the 16 cells to the judge reservation with PRIMARY first; "
          "SUPPLEMENTARY may become partial.",
    effect="tool2-extbench-v1 family appended to the campaign roster (PRIMARY rank 0 before every SUPPLEMENTARY); "
           "same policy, prompts, models, shared ledger and 80 USD stop")
T2V2_AUTHORIZATION = dict(
    by="author dispatch to the Tool1 builder (relayed), 2026-11-07",
    quote="Tool2 asked Tool1 to score 6 more extended-benchmark cells exactly as Tool1 scored Tool2's 16 cells on "
          "2026-11-05 (local job 11059827, judge v1.5 job 11059828)",
    effect="tool2-extbench-v2 family (manifest V2, roster extbench-v4) appended to the campaign roster (PRIMARY rank 0 "
           "before every SUPPLEMENTARY); same policy, prompts, models, shared ledger and 80 USD stop")
T2V3_AUTHORIZATION = dict(
    by="author dispatch to the Tool1 builder (relayed), 2026-11-07",
    quote="Tool2 asked Tool1 to score 6 more extended-benchmark cells (manifest V3: extbench-v3b t2-m1l-t43, t2-m1g-t43, "
          "t2-m1ud-t17, t2-unid-t17, t2-unisr-t17; extbench-v3c t2-p34-s1235-t17) exactly as Tool1 scored the V2 batch "
          "on 2026-11-07 (local job 11075656, judge v1.6 job 11075657)",
    effect="tool2-extbench-v3 family (manifest V3, rosters extbench-v3b/extbench-v3c) appended to the campaign roster "
           "(PRIMARY rank 0 before every SUPPLEMENTARY); same policy, prompts, models, shared ledger and 80 USD stop")
T2V4_AUTHORIZATION = dict(
    by="author dispatch to the Tool1 builder (relayed), 2026-11-07",
    quote="Score ONE more Tool2 extended-benchmark cell EXACTLY as Tool1 scored the V3 batch on 11-07 (local scoring v3 "
          "job 11078008, judge v1.7 job 11078010): t2-lessa-t17, generation job 11077884, manifest V4 (sha c7db2b4f), "
          "roster extbench-v3d; LESS-approx (not LESS): count-sketch SGD gradient features, 3x-Uniform candidate "
          "pre-filter, 2 dev targets GQA-val/SQA-val (benchmark-aligned, not blind to the benchmark tasks)",
    effect="tool2-extbench-v4 family (manifest V4, roster extbench-v3d) appended to the campaign roster (PRIMARY rank 0 "
           "before every SUPPLEMENTARY); same policy, prompts, models, shared ledger and 80 USD stop")
T2_TOOL2_MODEL4_SHA256 = "2941b0490f665f20f97a7ee01df7a8a0e03a083944d814af225519b9978972e1"
# Tool2 families: name -> (family module, custody module, manifest) source keys and the family's authorization.
# custody is one source key for every roster version of the family, or {roster version: source key}.
T2_FAMILY_SOURCES = {
    "tool2-extbench-v1": dict(family="t2family.py", custody="custody_t2.py",
                              manifest="EXTBENCH_GENERATION_MANIFEST_V1.json", authorization=T2_AUTHORIZATION),
    "tool2-extbench-v2": dict(family="t2family-v2.py", custody="custody_t2-v4.py",
                              manifest="EXTBENCH_GENERATION_MANIFEST_V2.json", authorization=T2V2_AUTHORIZATION),
    "tool2-extbench-v3": dict(family="t2family-v3.py",
                              custody={"extbench-v3b": "custody_t2-v3b.py", "extbench-v3c": "custody_t2-v3c.py"},
                              manifest="EXTBENCH_GENERATION_MANIFEST_V3.json", authorization=T2V3_AUTHORIZATION),
    "tool2-extbench-v4": dict(family="t2family-v4.py", custody={"extbench-v3d": "custody_t2-v3d.py"},
                              manifest="EXTBENCH_GENERATION_MANIFEST_V4.json", authorization=T2V4_AUTHORIZATION),
}
REFERENCE_ORIGINS = {
    "mm-vet_evaluator.py": "https://raw.githubusercontent.com/yuweihao/MM-Vet/"
                           "acde843e02dae30047bb456ee4a8ab28dee7e0ae/mm-vet_evaluator.py",
    "rule.json": "https://raw.githubusercontent.com/haotian-liu/LLaVA/c121f0432da27facab705978f83c4ada465e46fd/"
                 "llava/eval/table/rule.json",
    "eval_gpt_review_bench.py": "LLaVA c121f043 llava/eval/eval_gpt_review_bench.py",
    "summarize_gpt_review.py": "LLaVA c121f043 llava/eval/summarize_gpt_review.py",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def load_helpers():
    path = Path(HELPER["path"])
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = next_fd
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(HELPER["bytes"] + 1)
        after = os.fstat(stream.fileno())
    require(stat.S_ISREG(before.st_mode) and before.st_uid == UID
            and len(raw) == before.st_size == HELPER["bytes"]
            and hashlib.sha256(raw).hexdigest() == HELPER["sha256"]
            and all(getattr(before, field) == getattr(after, field) for field in STABLE),
            "Exact immutable executed metadata helper required")
    module = ModuleType("tool1_frozen_evaluation_metadata_helpers")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    return module


def bound_module(p, name, module_name):
    raw = p.checked(SOURCES[name])
    module = ModuleType(module_name)
    module.__file__ = SOURCES[name]["path"]
    exec(compile(raw, SOURCES[name]["path"], "exec", dont_inherit=True), vars(module))
    return module


def check_batch(batch):
    require(type(batch) is int and 0 <= batch < MAX_BATCHES, "Finite judge batch index required")
    return batch


def check_attempt(attempt):
    require(type(attempt) is int and 0 <= attempt < MAX_ATTEMPTS, "Finite judge attempt index required")
    return attempt


def check_route(route):
    require(isinstance(route, str) and route in ROUTES, "Reviewed route general or preempt required")
    return route


def plan_name(route):
    return "PLAN-%s.json" % check_route(route)


def review_name(route):
    return "OWNER_REVIEW-%s.json" % check_route(route)


def independent_name(route):
    return "INDEPENDENT_REVIEW-%s.json" % check_route(route)


def batch_folder(batch):
    return CONTROL / ("judge-batch-%d" % check_batch(batch))


def job_name(batch, attempt):
    return JOB_PREFIX + "b%d-a%d" % (check_batch(batch), check_attempt(attempt))


def comment(plan, attempt):
    return COMMENT_PREFIX + plan["function_identity"] + ":a%d" % attempt


def time_text(minutes):
    require(type(minutes) is int and 0 < minutes < 24 * 60, "Sub-day walltime required")
    return "%02d:%02d:00" % divmod(minutes, 60)


def private_directory(path):
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.stat(path, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Owned private control directory required: " + str(path))


def exact_ref(value):
    return isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"} and type(value["bytes"]) is int \
        and isinstance(value["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None


def read_ref(ref, limit=MAX_EXPORT):
    """Bounded, nonblocking, no-follow read of one pinned file with stable FD identity; bytes hashed once."""
    require(exact_ref(ref) and 0 < ref["bytes"] <= limit, "Bounded exact descriptor required")
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
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"]
            and all(getattr(before, k) == getattr(after, k) for k in STABLE), "Pinned file differs: " + ref["path"])
    return raw


def plain(ref):
    require(isinstance(ref, dict), "Descriptor required")
    return {"path": ref["path"], "bytes": ref["bytes"], "sha256": ref["sha256"]}


def jsonl(raw):
    require(raw.endswith(b"\n"), "Complete newline-terminated JSONL required")
    return [json.loads(line) for line in raw[:-1].split(b"\n")]


# ------------------------------------------------------------------------------------------ bound generation
def v1_family(p):
    roster = json.loads(p.checked(SOURCES["CELLS_BENCH6.json"]))
    return dict(name="bench4-v1", control=EVAL_CONTROL, owner=GENERATION_OWNER, output_root=OUTPUT_ROOT,
                attempts=GENERATION_ATTEMPTS, cells=sorted(roster["eligible"]))


def v3_family(p):
    """CELLS_V3 students, read only through the exact bound BENCH4 v3 controller's own roots, owner and roster."""
    v3 = bound_module(p, "bench4_v3_prepare4.py", "tool1_bench4_judge_v111_v3_controller")
    require(v3.COMPLETE_STATUS == GENERATION_COMPLETE and tuple(v3.BANKS) == BANKS
            and v3.SOURCES["banks4.py"] == SOURCES["banks4.py"], "BENCH4 v3 generation semantics differ")
    return dict(name="bench4-v3", control=Path(v3.CONTROL_ROOT), owner=v3.OWNER, output_root=Path(v3.OUTPUT_ROOT),
                attempts=v3.MAX_ATTEMPTS, cells=sorted(v3.bound_cells(p)["cells"]))


def t2_family(p, name):
    """Tool2 extbench students (Tool2 custody), read only through the pinned Tool2 rosters, manifest and controllers."""
    keys = T2_FAMILY_SOURCES[name]
    family = bound_module(p, keys["family"], "tool1_bench4_judge_v111_t2family_" + name[-2:])
    custody_keys = keys["custody"] if isinstance(keys["custody"], dict) else {
        version: keys["custody"] for version in family.ROSTER_VERSIONS}
    require(sorted(custody_keys) == sorted(family.ROSTER_VERSIONS), "Tool2 custody per roster version required: " + name)
    rosters, custodies = {}, {}
    for version in family.ROSTER_VERSIONS:
        custody = custodies[version] = bound_module(p, custody_keys[version], "tool1_bench4_judge_v111_t2custody_"
                                                    + name[-2:] + "_" + version[-3:])
        prepare = bound_module(p, "prepare-%s.py" % version, "tool1_bench4_judge_v111_t2prepare_" + version[-2:])
        cells_ref = SOURCES["CELLS_T2-%s.json" % version]
        require(Path(prepare.CONTROL_ROOT) == T2_CONTROL / version and prepare.OWNER == family.GENERATION_OWNER
                and prepare.COMPLETE_STATUS == GENERATION_COMPLETE == family.COMPLETE
                and str(prepare.OUTPUT_ROOTS["bench4"]) == family.output_root("bench4")
                and prepare.SOURCES["CELLS_T2.json"] == cells_ref and prepare.SOURCES["banks4.py"] == SOURCES["banks4.py"]
                and prepare.SOURCES["custody_t2.py"]["sha256"] == SOURCES[custody_keys[version]]["sha256"]
                and prepare.SOURCES["model4.py"]["sha256"] == T2_TOOL2_MODEL4_SHA256
                and tuple(prepare.BANKS4) == BANKS == family.GROUP_BANKS["bench4"],
                "Tool2 generation controller semantics differ: " + version)
        rosters[version] = custody.check_cells(json.loads(p.checked(cells_ref)))
    manifest = json.loads(p.checked(SOURCES[keys["manifest"]]))
    versions = family.check_manifest(manifest, rosters)
    return dict(name=name, control=T2_CONTROL, owner=family.GENERATION_OWNER,
                output_root=Path(family.output_root("bench4")), attempts=family.GENERATION_ATTEMPTS,
                cells=sorted(versions), module=family, custody=custodies, rosters=rosters, manifest=manifest,
                versions=versions)


def v5_family(p):
    """CELLS_V5 students outside CELLS_V3 (Tool1 trainer v10 / Tool3 chain-v3 producers), read only through the exact
    bound CELLS_V5 BENCH4 controller, whose roots, owner, attempts and banks are the v3 controller's own."""
    v5 = bound_module(p, "bench4_v5_prepare5_b4.py", "tool1_bench4_judge_v111_v5_controller")
    v3 = bound_module(p, "bench4_v3_prepare4.py", "tool1_bench4_judge_v111_v3_controller_for_v5")
    require(v5.COMPLETE_STATUS == GENERATION_COMPLETE and tuple(v5.BANKS) == BANKS
            and v5.SOURCES["banks4.py"] == SOURCES["banks4.py"]
            and v5.SOURCES["CELLS_V3.json"] == SOURCES["CELLS_V5.json"]
            and (str(v5.CONTROL_ROOT), v5.OWNER, str(v5.OUTPUT_ROOT), v5.MAX_ATTEMPTS)
            == (str(v3.CONTROL_ROOT), v3.OWNER, str(v3.OUTPUT_ROOT), v3.MAX_ATTEMPTS),
            "CELLS_V5 BENCH4 generation semantics differ")
    cells = set(v5.bound_cells(p)["cells"]) - set(v3.bound_cells(p)["cells"])
    return dict(name="bench4-v3c5", control=Path(v5.CONTROL_ROOT), owner=v5.OWNER, output_root=Path(v5.OUTPUT_ROOT),
                attempts=v5.MAX_ATTEMPTS, cells=sorted(cells))


def cells_v5(p):
    cells = json.loads(p.checked(SOURCES["CELLS_V5.json"]))
    require(cells.get("schema") == "tool1-baseline-cells-v5-extsuite-custody-v1" and isinstance(cells.get("cells"), dict),
            "Exact CELLS_V5 roster required")
    return sorted(cells["cells"])


def v6_family(p):
    """CELLS_V6 students outside CELLS_V3 and CELLS_V5 (Tool1 trainer v10 t2-m1ur-b20), read only through the exact bound
    CELLS_V6 BENCH4 controller, whose roots, owner, attempts and banks are the v3 controller's own."""
    v6 = bound_module(p, "bench4_v6_prepare6_b4.py", "tool1_bench4_judge_v111_v6_controller")
    v3 = bound_module(p, "bench4_v3_prepare4.py", "tool1_bench4_judge_v111_v3_controller_for_v6")
    require(v6.COMPLETE_STATUS == GENERATION_COMPLETE and tuple(v6.BANKS) == BANKS
            and v6.SOURCES["banks4.py"] == SOURCES["banks4.py"]
            and v6.SOURCES["CELLS_V3.json"] == SOURCES["CELLS_V6.json"]
            and (str(v6.CONTROL_ROOT), v6.OWNER, str(v6.OUTPUT_ROOT), v6.MAX_ATTEMPTS)
            == (str(v3.CONTROL_ROOT), v3.OWNER, str(v3.OUTPUT_ROOT), v3.MAX_ATTEMPTS),
            "CELLS_V6 BENCH4 generation semantics differ")
    cells = set(v6.bound_cells(p)["cells"]) - set(v3.bound_cells(p)["cells"]) - set(cells_v5(p))
    return dict(name="bench4-v3c6", control=Path(v6.CONTROL_ROOT), owner=v6.OWNER, output_root=Path(v6.OUTPUT_ROOT),
                attempts=v6.MAX_ATTEMPTS, cells=sorted(cells))


def cells_v6(p):
    cells = json.loads(p.checked(SOURCES["CELLS_V6.json"]))
    require(cells.get("schema") == "tool1-baseline-cells-v6-extsuite-custody-v1" and isinstance(cells.get("cells"), dict),
            "Exact CELLS_V6 roster required")
    return sorted(cells["cells"])


def v7_family(p):
    """CELLS_V7 students outside CELLS_V3, CELLS_V5 and CELLS_V6 (Tool1 trainer v10 t2-m1ur-b05, t2-univn-b05), read only
    through the exact bound CELLS_V7 BENCH4 controller, whose roots, owner, attempts and banks are the v3 controller's own."""
    v7 = bound_module(p, "bench4_v7_prepare7_b4.py", "tool1_bench4_judge_v111_v7_controller")
    v3 = bound_module(p, "bench4_v3_prepare4.py", "tool1_bench4_judge_v111_v3_controller_for_v7")
    require(v7.COMPLETE_STATUS == GENERATION_COMPLETE and tuple(v7.BANKS) == BANKS
            and v7.SOURCES["banks4.py"] == SOURCES["banks4.py"]
            and v7.SOURCES["CELLS_V3.json"] == SOURCES["CELLS_V7.json"]
            and (str(v7.CONTROL_ROOT), v7.OWNER, str(v7.OUTPUT_ROOT), v7.MAX_ATTEMPTS)
            == (str(v3.CONTROL_ROOT), v3.OWNER, str(v3.OUTPUT_ROOT), v3.MAX_ATTEMPTS),
            "CELLS_V7 BENCH4 generation semantics differ")
    cells = (set(v7.bound_cells(p)["cells"]) - set(v3.bound_cells(p)["cells"]) - set(cells_v5(p))
             - set(cells_v6(p)))
    return dict(name="bench4-v3c7", control=Path(v7.CONTROL_ROOT), owner=v7.OWNER, output_root=Path(v7.OUTPUT_ROOT),
                attempts=v7.MAX_ATTEMPTS, cells=sorted(cells))


def cells_v7(p):
    cells = json.loads(p.checked(SOURCES["CELLS_V7.json"]))
    require(cells.get("schema") == "tool1-baseline-cells-v7-extsuite-custody-v1" and isinstance(cells.get("cells"), dict),
            "Exact CELLS_V7 roster required")
    return sorted(cells["cells"])


def families(p):
    value = ((v1_family(p), v3_family(p)) + tuple(t2_family(p, name) for name in T2_FAMILY_SOURCES)
             + (v5_family(p), v6_family(p), v7_family(p)))
    names = [cell for family in value for cell in family["cells"]]
    require(len(set(names)) == len(names), "Generation families must not share cell names")
    return value


def t2_generation(p, family, cell, ref=None):
    """Tool2 cell: its single complete HOME bench4 ATTEMPT_RESULT (by the manifest completing job), validated with
    Tool1's semantics and the Tool2 custody basis; returns (result ref, release inputs_result ref, result)."""
    module = family["module"]
    roster = family["rosters"][family["versions"][cell]]
    producer = p.descriptor(Path(roster["cells"][cell]["producer"]["model_result"]))
    custody = family["custody"][family["versions"][cell]]
    files = custody.plan_custody(roster, cell, producer, json.loads(p.checked(producer)))["inference_files"]
    entry = module.manifest_entry(family["manifest"], cell, "bench4")
    found, inputs, _, _ = module.home_binding(p, family["control"] / family["versions"][cell], cell,
                                              entry["completing_job"], "bench4")
    require(ref is None or ref == found, "Tool2 generation binding changed: " + cell)
    result = json.loads(p.checked(found))
    module.validate_result(result, cell, "bench4", entry["completing_job"], files)
    module.manifest_consistent(entry, result)
    return found, inputs, result


def bound_generation(p, families_, cell, ref):
    """Validated generation result of a bound cell in its own family (Tool2 custody or Tool1 parity semantics)."""
    family = family_of(families_, cell)
    if family["name"] in T2_FAMILY_SOURCES:
        return t2_generation(p, family, cell, ref)[2]
    return validate_generation(json.loads(p.checked(ref)), cell, family["output_root"])


def family_of(families_, cell):
    found = [family for family in families_ if cell in family["cells"]]
    require(len(found) == 1, "Registered CELLS_BENCH6 eligible, CELLS_V3, Tool2 or CELLS_V5 cell required: %r" % (cell,))
    return found[0]


def campaign_cells(families_):
    return [cell for family in families_ for cell in family["cells"]]


def validate_generation(result, cell, output_root):
    """A complete, label-free, parity-checked BENCH4 generation result of exactly this cell (either family)."""
    require(isinstance(result, dict) and result.get("status") == GENERATION_COMPLETE and result.get("cell") == cell
            and result.get("planned_banks") == list(BANKS) and result.get("prior_parameter_parity") is True
            and result.get("labels_opened") is False
            and re.fullmatch(r"[0-9a-f]{64}", str(result.get("parameter_sha256"))) is not None
            and [b.get("protocol") for b in result.get("banks", [])] == list(BANKS),
            "Complete BENCH4 generation result with parameter parity required: " + cell)
    for report in result["banks"]:
        name, completion = report["protocol"], report.get("completion") or {}
        bank = str(Path(output_root) / name / cell)
        require(report.get("status") == BANK_COMPLETE and report.get("labels_opened") is False
                and completion.get("complete") is True and completion.get("missing") == 0
                and completion.get("accepted") == completion.get("items") == report["roster"]["items"],
                "Complete bank report required: " + name)
        for field, file in (("predictions", "PREDICTIONS.jsonl"), ("function", "FUNCTION.json")):
            require(exact_ref(report.get(field)) and report[field]["path"] == bank + "/" + file,
                    "Exact bank-level export descriptor required: " + name + " " + field)
    return result


def generation_binding(p, families_, cell):
    """The single complete generation ATTEMPT_RESULT of this cell and the inventory RESULT its release bound."""
    family = family_of(families_, cell)
    if family["name"] in T2_FAMILY_SOURCES:
        found, inputs, _ = t2_generation(p, family, cell)
        return found, inputs
    folder = family["control"] / cell
    found = []
    for attempt in range(family["attempts"]):
        ack_path = folder / ("attempt-%d" % attempt) / "ACK.json"
        if not ack_path.exists():
            continue
        ack = json.loads(p.checked(p.descriptor(ack_path)))
        require(ack.get("owner") == family["owner"] and ack.get("cell") == cell and ack.get("attempt") == attempt
                and re.fullmatch(r"[1-9]\d*", str(ack.get("job_id"))) is not None, "Exact generation ACK required")
        path = ack_path.parent / ("ATTEMPT_RESULT-" + ack["job_id"] + ".json")
        if path.exists():
            ref = p.descriptor(path)
            if json.loads(p.checked(ref)).get("status") == GENERATION_COMPLETE:
                release = json.loads(p.checked(p.descriptor(ack_path.parent / "RELEASE_INTENT.json")))
                found.append((ref, release["inputs_result"]))
    require(len(found) == 1, "Exactly one complete BENCH4 generation result required for " + cell)
    validate_generation(json.loads(p.checked(found[0][0])), cell, family["output_root"])
    return found[0]


def v5_generation_ack(p, families_, cell):
    """The newest acknowledged attempt of a CELLS_V5/CELLS_V6/CELLS_V7 cell's BENCH4 generation, made by that version's
    pinned BENCH4 controller (a cell is registered in exactly one of the three)."""
    family = family_of(families_, cell)
    in5, in6, in7 = cell in cells_v5(p), cell in cells_v6(p), cell in cells_v7(p)
    require(family["name"] in V5_FAMILIES and [in5, in6, in7].count(True) == 1,
            "Afterok judging needs a CELLS_V5, CELLS_V6 or CELLS_V7 generation: " + cell)
    schema, name, key, roster = ((V7_GENERATION_SCHEMA, "prepare7_b4.py", "bench4_v7_prepare7_b4.py", "CELLS_V7.json")
                                 if in7 else
                                 (V6_GENERATION_SCHEMA, "prepare6_b4.py", "bench4_v6_prepare6_b4.py", "CELLS_V6.json")
                                 if in6 else (V5_GENERATION_SCHEMA, "prepare5_b4.py", "bench4_v5_prepare5_b4.py",
                                              "CELLS_V5.json"))
    folder = family["control"] / cell
    attempts = [index for index in range(family["attempts"]) if (folder / ("attempt-%d" % index) / "ACK.json").exists()]
    require(attempts and attempts == list(range(len(attempts))), "Sequential acknowledged generation attempts required")
    ack_ref = p.descriptor(folder / ("attempt-%d" % attempts[-1]) / "ACK.json")
    ack = json.loads(p.checked(ack_ref))
    require(ack.get("owner") == family["owner"] and ack.get("cell") == cell and ack.get("attempt") == attempts[-1]
            and re.fullmatch(r"[1-9]\d*", str(ack.get("job_id"))) is not None, "Exact generation ACK required")
    plan = json.loads(p.checked(ack["plan"]))
    require(plan.get("schema") == schema and plan.get("cell") == cell
            and plan.get("sources", {}).get(name) == SOURCES[key]
            and plan["sources"].get("CELLS_V3.json") == SOURCES[roster],
            "Generation must be the pinned CELLS_V5/CELLS_V6/CELLS_V7 BENCH4 controller's own attempt")
    return ack_ref, ack, plan


def afterok_gate(p, families_, function, final=False):
    """Bound generation jobs live or COMPLETED (submit/release); all COMPLETED with exactly their own complete results
    released with the planned input inventory (worker)."""
    states, results = {}, {}
    for cell, ack_ref in sorted(function["generation_acks"].items()):
        ack = json.loads(p.checked(ack_ref))
        job = function["generation_jobs"][cell]
        require(ack["job_id"] == job and v5_generation_ack(p, families_, cell)[0] == ack_ref,
                "Bound generation identity differs or a newer attempt exists: " + cell)
        state, code = sacct_state(p, job)
        require(state in (("COMPLETED",) if final else GENERATION_LIVE),
                "Generation job state does not permit judging: %s %s" % (cell, state))
        states[cell] = dict(generation_job=job, state=state, exit_code=code)
        if state == "COMPLETED":
            found, inputs = generation_binding(p, families_, cell)
            require(found["path"] == str(Path(ack_ref["path"]).parent / ("ATTEMPT_RESULT-%s.json" % job))
                    and inputs == function["inventory_result"],
                    "The single complete generation result must be the bound job's own: " + cell)
            results[cell] = found
    return dict(states=states, generation_results=results)


def dependency_ok(value, plan, running):
    """afterok readback: (null) once satisfied, else per-job afterok:<job>(unfulfilled|fulfilled) of bound jobs only."""
    if value in ("(null)", plan["dependency"]):
        return True
    bound = set(plan["function"]["generation_jobs"].values())
    items = [DEPENDENCY_ITEM.fullmatch(item) for item in str(value).split(",")]
    return (bool(items) and all(match is not None and match[1] in bound for match in items)
            and not (running and any(match[2] == "unfulfilled" for match in items)))


def validate_inventory(result):
    require(result.get("status") == INPUT_STATUS and result.get("worker_returncode") == 0
            and sorted(result.get("banks", {})) == sorted(BANKS)
            and all(result["banks"][name]["status"] == "PASS" for name in BANKS)
            and all(exact_ref(result["banks"][name]["labels"]) for name in BANKS)
            and all(str(result["banks"][name]["labels"]["path"]).startswith(str(INPUT_ROOT) + "/") for name in BANKS),
            "Completed BENCH4 input inventory with scoring-only LABELS required")
    return result


def active_campaign(judge, campaign_cells):
    """The bound policy's campaign roster restricted to ACTIVE_ROLES (ranks and order unchanged)."""
    return [run for run in judge.campaign_runs(campaign_cells) if run["role"] in ACTIVE_ROLES]


def active_batch_runs(judge, cells, campaign_cells):
    """The bound policy's ordered batch runs restricted to ACTIVE_ROLES."""
    return [run for run in judge.batch_runs(list(cells), judge.campaign_runs(campaign_cells))
            if run["role"] in ACTIVE_ROLES]


def calibration_removal(p, judge):
    """Exact MODELS evidence that the removed CALIBRATION judge was not listed by the models endpoint."""
    evidence = []
    for path in CALIBRATION_EVIDENCE:
        ref = p.descriptor(path)
        listed = json.loads(p.checked(ref))
        require(listed.get("available", {}).get(judge.CALIBRATION) is False
                and listed["available"].get(judge.PRIMARY) is True
                and listed["available"].get(judge.SUPPLEMENTARY) is True,
                "MODELS evidence must show CALIBRATION unavailable and both active judges listed")
        evidence.append(ref)
    return dict(role="CALIBRATION", model=judge.CALIBRATION, removed_cells=list(judge.CALIBRATION_CELLS),
                active_roles=list(ACTIVE_ROLES), models_evidence=evidence, authorization=COVERAGE_AUTHORIZATION)


def judge_spec(judge, prompt, rules, cells, campaign_cells):
    campaign = active_campaign(judge, campaign_cells)
    models = {role: model for model, role in judge.ROLES.items() if role in ACTIVE_ROLES}
    return dict(
        version=judge.VERSION, cells=list(cells), runs=active_batch_runs(judge, cells, campaign_cells),
        campaign_cells=list(campaign_cells), campaign_runs=len(campaign), active_roles=list(ACTIVE_ROLES),
        models=models,
        supplementary_returned_pattern=judge.SUPPLEMENTARY_RETURNED,
        request_parameters={model: {bank: judge.request_parameters(model, bank) for bank in judge.JUDGE_BANKS}
                            for model in models.values()},
        temperature=judge.TEMPERATURE, calibration_cells=[],
        prices_usd_per_mtok=judge.PRICES_USD_PER_MTOK, price_policy=judge.PRICE_POLICY, stop_usd=judge.STOP_USD,
        hard_budget_usd=judge.HARD_BUDGET_USD, endpoint=judge.ENDPOINT, models_endpoint=judge.MODELS_ENDPOINT,
        key_file_path_only=judge.KEY_PATH, ledger=str(LEDGER), ledger_schema=judge.LEDGER_SCHEMA,
        repeat_policy=judge.REPEAT_POLICY, missing_policy=judge.MISSING_POLICY,
        prompt_hashes=judge.prompt_hashes(prompt, rules), mmvet_max_tokens=judge.MMVET_MAX_TOKENS,
        wild_max_tokens=judge.WILD_MAX_TOKENS, reasoning_max_completion_tokens=judge.REASONING_MAX_COMPLETION_TOKENS,
        concurrency=judge.CONCURRENCY,
        reserve_floor_usd={"%s|%s" % key: value for key, value in sorted(judge.RESERVE_FLOOR_USD.items())},
        reserve_observed_factor=judge.RESERVE_OBSERVED_FACTOR,
        spend_order="PRIMARY (rank 0) -> SUPPLEMENTARY (rank 100; CALIBRATION removed in v1.4) over campaign_cells: each call needs spend + call maximum + estimated cost of every "
                    "unfinished lower-rank campaign run <= stop_usd; each unfinished run needs spend + reservation + "
                    "its projected uncached cost <= stop_usd before it starts, else the attempt stops PARTIAL",
        official_judges={"mm-vet": "gpt-4-0613 temperature 0", "llava-wild": "gpt-4-0314 temperature 0.2"},
        deviation="Primary judge gpt-4o-2024-08-06 at temperature 0 for every cell (author-approved); official "
                  "gpt-4-0613 MM-Vet calibration removed in v1.4 (not listed by the models endpoint; author: coverage "
                  "first); supplementary newest judge "
                  "gpt-5.6-terra (reasoning model: default temperature/reasoning effort, max_completion_tokens instead "
                  "of the official max_tokens, final parse failures missing instead of the official 0.0/[-1, -1]).")


def make_plan(p, batch, route, cells, results, inventory_ref, acks=None):
    routes = bound_module(p, "routes.py", "tool1_bench4_judge_v111_routes")
    banks = bound_module(p, "banks4.py", "tool1_bench4_judge_v111_banks")
    judge = bound_module(p, "judge4_v13.py", "tool1_bench4_judge_v111_judge")
    families_ = families(p)
    campaign = campaign_cells(families_)
    require(tuple(banks.ORDER) == BANKS and tuple(judge.JUDGE_BANKS) == JUDGE_BANKS,
            "Bound bank protocol, judge and judge step disagree")
    if acks is not None:
        require(results is None and isinstance(acks, dict) and isinstance(cells, list) and cells
                and sorted(cells) == sorted(acks) and set(cells) <= set(campaign) and set(cells) <= set(cells_v5(p)) | set(cells_v6(p)) | set(cells_v7(p))
                and all(exact_ref(ref) for ref in acks.values())
                and all(family_of(families_, cell)["name"] in V5_FAMILIES for cell in cells),
                "Nonempty exact roster of CELLS_V5 generation ACKs required")
    else:
        require(isinstance(cells, list) and cells and sorted(cells) == sorted(results)
                and set(cells) <= set(campaign) and all(exact_ref(ref) for ref in results.values()),
                "Nonempty exact roster of generation results required")
    inventory = validate_inventory(json.loads(p.checked(inventory_ref)))
    predictions, t2_finals = {}, {}
    for cell in (cells if acks is None else ()):
        generation = bound_generation(p, families_, cell, results[cell])
        predictions[cell] = {r["protocol"]: plain(r["predictions"]) for r in generation["banks"]
                             if r["protocol"] in JUDGE_BANKS}
        family = family_of(families_, cell)
        if family["name"] in T2_FAMILY_SOURCES:
            t2_finals[cell] = family["module"].manifest_entry(family["manifest"], cell, "bench4")["final_result"]
    prompt = judge.mmvet_prompt(p.checked(SOURCES["mm-vet_evaluator.py"]))
    rules = judge.wild_rules(p.checked(SOURCES["rule.json"]), p.checked(SOURCES["eval_gpt_review_bench.py"]))
    function = dict(
        owner=OWNER, kind="judge", batch=batch, banks=list(JUDGE_BANKS), cells=list(cells),
        generation_results=results, predictions=predictions, inventory_result=inventory_ref,
        generation_families={cell: family_of(families_, cell)["name"] for cell in cells},
        tool2_manifest_final_results=t2_finals,
        tool2_custody_authorization={name: keys["authorization"] for name, keys in T2_FAMILY_SOURCES.items()},
        budget_authorization=AUTHORIZATION,
        calibration_removed=calibration_removal(p, judge),
        labels={name: plain(inventory["banks"][name]["labels"]) for name in JUDGE_BANKS},
        questions={name: plain(inventory["banks"][name]["questions"]) for name in JUDGE_BANKS},
        scorers={name: banks.protocol(name)["scoring"] for name in JUDGE_BANKS},
        judge=judge_spec(judge, prompt, rules, cells, campaign),
        bound_sources=SOURCES, reference_origins=dict(REFERENCE_ORIGINS), metadata_helper=HELPER, model_forwards=0,
        training_updates=0, used_for_selection_or_tuning=False,
        score_files=str(batch_folder(batch) / "cells" / "<cell>" / "SCORE_<protocol>__<judge>.json"),
        labels_decoded_only_inside_scorer=True)
    dependency = None
    if acks is not None:
        jobs = {cell: json.loads(p.checked(acks[cell]))["job_id"] for cell in sorted(acks)}
        function.update(generation_results=None, predictions=None, generation_acks=acks, generation_jobs=jobs,
                        generation_controller={cell: SOURCES["bench4_v7_prepare7_b4.py" if cell in cells_v7(p)
                                                             else "bench4_v6_prepare6_b4.py" if cell in cells_v6(p)
                                                             else "bench4_v5_prepare5_b4.py"] for cell in sorted(acks)},
                        cells_v5=SOURCES["CELLS_V5.json"], cells_v6=SOURCES["CELLS_V6.json"],
                        cells_v7=SOURCES["CELLS_V7.json"],
                        late_binding="worker binds the single complete ATTEMPT_RESULT-<job>.json of exactly each bound "
                                     "ACK (released with this input inventory) after every afterok generation job "
                                     "COMPLETED, and pins its MM-Vet/LLaVA-W predictions before any label or API use")
        dependency = "afterok:" + ":".join(jobs[cell] for cell in sorted(jobs))
    resource = routes.ROUTES[routes.check_route(check_route(route))]
    plan = dict(
        schema="tool1-bench4-registered-judge-plan-v111", owner=OWNER, kind="judge", batch=batch, route=route,
        function=function, function_identity=hashlib.sha256(p.canonical(function)).hexdigest(),
        sources=dict({"score_step4_v111.py": p.descriptor(SOURCE / "score_step4_v111.py"),
                      "score_run_v111.sbatch": p.descriptor(SOURCE / "score_run_v111.sbatch")}, **SOURCES),
        resource=dict(resource, gres=routes.GRES, GPUs=1, CPUs=int(CPUS), memory=MEMORY, minutes=MINUTES,
                      time=routes.time_text(MINUTES), exclude=routes.EXCLUDED),
        expected_readback=routes.expected_route_fields(route), automatic_requeue=False, independently_reviewed=False)
    if dependency is not None:
        plan["dependency"] = dependency
    return plan


def live_plan_cells(p, root, batch=None):
    """Cells bound by non-withdrawn judge plans under one control root (all versions use judge-batch-<B>)."""
    taken = set()
    for other in range(MAX_BATCHES):
        folder = root / ("judge-batch-%d" % other)
        if other == batch or (folder / "WITHDRAWN.json").exists():
            continue
        for path in sorted(folder.glob("PLAN-*.json")) if folder.exists() else ():
            taken |= set(json.loads(p.checked(p.descriptor(path)))["function"]["cells"])
    return taken


def make_plan_file(p, batch, route, cells, afterok=False):
    check_batch(batch)
    check_route(route)
    retired_versions(p)
    families_ = families(p)
    require(isinstance(cells, list) and cells and len(set(cells)) == len(cells)
            and set(cells) <= set(campaign_cells(families_)),
            "Explicit unique CELLS_BENCH6 eligible or CELLS_V3 registered cells required")
    require(not set(cells) & live_plan_cells(p, CONTROL, batch), "A cell is already bound by another v111 batch")
    for root in FOREIGN_JUDGE_CONTROLS:
        require(not set(cells) & live_plan_cells(p, root), "A cell is bound by a superseded judge version's plan")
    if afterok:
        acks, paths = {}, set()
        for cell in sorted(cells):
            ack_ref, ack, generation_plan = v5_generation_ack(p, families_, cell)
            require(sacct_state(p, ack["job_id"])[0] in GENERATION_LIVE, "Bound generation job is not live: " + cell)
            acks[cell] = ack_ref
            paths.add(generation_plan["function"]["inputs_inventory"]["result_path"])
        require(len(paths) == 1, "Every bound generation plan must bind the same BENCH4 input inventory run")
        plan = make_plan(p, batch, route, list(cells), None, p.descriptor(Path(paths.pop())), acks=acks)
    else:
        bindings = {cell: generation_binding(p, families_, cell) for cell in cells}
        inventories = {json.dumps(binding[1], sort_keys=True) for binding in bindings.values()}
        require(len(inventories) == 1, "Every bound generation must have released with the same input inventory")
        inventory_ref = next(iter(bindings.values()))[1]
        plan = make_plan(p, batch, route, list(cells), {cell: b[0] for cell, b in bindings.items()}, inventory_ref)
    private_directory(CONTROL)
    private_directory(batch_folder(batch))
    path = batch_folder(batch) / plan_name(route)
    require(not path.exists(), "Existing judge plan is immutable; use a new batch")
    for other in ROUTES:
        sibling = batch_folder(batch) / plan_name(other)
        if other != route and sibling.exists():
            require(json.loads(p.checked(p.descriptor(sibling)))["function"] == plan["function"],
                    "Both routes of one batch must bind the identical judge function")
    return p.seal(path, plan)


def replan(p, plan, route):
    function = plan["function"]
    return make_plan(p, plan["batch"], route, function["cells"], function["generation_results"],
                     function["inventory_result"], acks=function.get("generation_acks"))


def owner_review(p, batch, route, notes):
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete self-review notes required")
    folder = batch_folder(batch)
    plan_ref = p.descriptor(folder / plan_name(route))
    plan = json.loads(p.checked(plan_ref))
    require(plan == replan(p, plan, route), "Self-review must bind the exact plan")
    return p.seal(folder / review_name(route), dict(
        status=REVIEW_STATUS, independent=False, ai_review_disclosed=True, reviewer=OWNER, kind="judge", batch=batch,
        route=route, plan=plan_ref, sources=plan["sources"], function_identity=plan["function_identity"],
        blocking_findings=[], notes=notes))


def validate_independent(review, plan_ref, plan, batch, route):
    require(review.get("status") == INDEPENDENT_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER) and review.get("blocking_findings") == []
            and review.get("sources") == plan["sources"] and review.get("route") == route == plan["route"]
            and review.get("kind") == "judge" and review.get("batch") == batch and review.get("plan") == plan_ref,
            "Independent exact source/config PASS bound to this judge plan is required")


def contract(p, batch, route, independent=False):
    folder = batch_folder(batch)
    info = os.stat(folder, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == UID and not info.st_mode & 0o022,
            "Private owned judge control directory required")
    plan_ref = p.descriptor(folder / plan_name(route))
    plan = json.loads(p.checked(plan_ref))
    require(plan.get("kind") == "judge" and plan == replan(p, plan, route),
            "Exact source/config/route-bound judge plan required")
    review_ref = p.descriptor(folder / review_name(route))
    review = json.loads(p.checked(review_ref))
    require(review["status"] == REVIEW_STATUS and review["independent"] is False and review["plan"] == plan_ref
            and review["sources"] == plan["sources"] and review["blocking_findings"] == []
            and review["kind"] == "judge" and review["batch"] == batch and review["route"] == route,
            "Truthful exact private-owner review required; no independent PASS claimed")
    independent_ref = None
    if independent:
        independent_ref = p.descriptor(folder / independent_name(route))
        validate_independent(json.loads(p.checked(independent_ref)), plan_ref, plan, batch, route)
    return folder, plan_ref, plan, review_ref, independent_ref


def argv(batch, attempt, plan):
    resource = plan["resource"]
    return ["/usr/bin/sbatch", "--parsable", "--hold", "--no-requeue", "--export=NIL",
            "--account=" + resource["account"], "--partition=" + resource["partition"], "--qos=" + resource["qos"],
            "--gres=gpu:L40S:1", "--nodes=1", "--ntasks=1", "--cpus-per-task=" + CPUS,
            "--mem=" + MEMORY, "--time=" + time_text(MINUTES), "--exclude=" + EXCLUDED,
            "--job-name=" + job_name(batch, attempt), "--comment=" + comment(plan, attempt),
            "--chdir=" + str(HOME), "--input=/dev/null", "--output=/dev/null", "--error=/dev/null"] + (
            ["--dependency=" + plan["dependency"]] if "dependency" in plan else []) + [
            str(SOURCE / "score_run_v111.sbatch"), str(batch), str(attempt), check_route(plan["route"])]


def scheduler(p, batch, attempt, plan, job, held=False, running=False):
    raw = p.query(["/usr/bin/scontrol", "show", "job", "-o", job])
    row = p.fields(raw)
    expected = dict(
        plan["expected_readback"], JobId=job, UserId="luyangz(2708899)", JobName=job_name(batch, attempt),
        Comment=comment(plan, attempt), NumCPUs=CPUS, NumTasks="1", TimeLimit=time_text(MINUTES),
        Command=str(SOURCE / "score_run_v111.sbatch"), WorkDir=str(HOME),
        StdIn="/dev/null", StdOut="/dev/null", StdErr="/dev/null")
    dependency = (dependency_ok(row.get("Dependency"), plan, running) if "dependency" in plan
                  else row.get("Dependency") == "(null)")
    nodes = row.get("NumNodes") == "1" if running else row.get("NumNodes") in ("1", "1-1")
    require(plan["expected_readback"]["Requeue"] == "0" and all(row.get(k) == v for k, v in expected.items())
            and nodes and dependency and shlex.split(row.get("SubmitLine", "")) == argv(batch, attempt, plan),
            "Actual physical judge request/source/resources differ")
    if held:
        require(row.get("JobState") == "PENDING" and row.get("Reason") == "JobHeldUser"
                and row.get("RunTime") == "00:00:00", "Verified never-started held request required")
    if running:
        require(row.get("JobState") == "RUNNING" and row.get("Restarts") == "0"
                and row.get("NodeList") == socket.gethostname().split(".")[0]
                and p.valid_singleton_node(row["NodeList"]), "Actual owned running allocation required")
    tres = dict(value.split("=", 1) for value in row.get("AllocTRES" if running else "ReqTRES", "").split(",")
                if "=" in value)
    require(tres.get("cpu") == CPUS and tres.get("mem") in (MEMORY, MEMORY_MB) and tres.get("gres/gpu") == "1",
            "Actual total allocation/request resources differ")
    spool = subprocess.check_output(["/usr/bin/scontrol", "write", "batch_script", job, "-"],
                                    timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
    require(spool == p.checked(plan["sources"]["score_run_v111.sbatch"]), "Exact immutable bound spool required")
    return raw, spool


def sacct_state(p, job):
    value = p.query(["/usr/bin/sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State,ExitCode"]).strip().split("|")
    require(len(value) == 3 and value[0] == job, "Exact scheduler accounting record required")
    return value[1].split(" ")[0], value[2]


def predecessor_allows(p, folder, attempt):
    require(not any((folder / ("attempt-%d" % index)).exists() for index in range(attempt + 1, MAX_ATTEMPTS)),
            "Later judge attempt folders already exist")
    if attempt == 0:
        return None
    previous = folder / ("attempt-%d" % (attempt - 1))
    ack = json.loads(p.checked(p.descriptor(previous / "ACK.json")))
    state, code = sacct_state(p, ack["job_id"])
    require(state in TERMINAL, "Predecessor judge attempt still active")
    receipt = previous / ("RESULT-" + ack["job_id"] + ".json")
    diagnosis, status = None, None
    if receipt.exists():
        status = json.loads(p.checked(p.descriptor(receipt))).get("status")
        if status == ABORTED_STATUS and state == "FAILED":
            diagnosis = diagnosed_failure(p, previous, ack, code)
        else:
            require(status == PARTIAL_STATUS, "Batch already scored or its RESULT needs no successor")
    elif state == "FAILED":
        diagnosis = diagnosed_failure(p, previous, ack, code)
    else:
        require(state in INTERRUPTED, "A failed judge attempt needs diagnosis, not a blind successor")
    return dict(job_id=ack["job_id"], route=ack.get("route"), state=state, exit_code=code, result_status=status,
                partial_result=status == PARTIAL_STATUS, diagnosis=diagnosis)


def diagnosed_failure(p, previous, ack, code):
    """Owner DIAGNOSIS + independent DIAGNOSIS_REVIEW of a FAILED predecessor, bound to its job, ACK, scheduler
    state/exit code, a contained hashed worker-log excerpt and this controller."""
    job = ack["job_id"]
    diagnosis_ref = p.descriptor(previous / ("DIAGNOSIS-%s.json" % job))
    diagnosis = json.loads(p.checked(diagnosis_ref))
    evidence, scheduled = diagnosis.get("evidence") or {}, diagnosis.get("scheduler") or {}
    require(diagnosis.get("schema") == DIAGNOSIS_SCHEMA and diagnosis.get("status") == DIAGNOSIS_STATUS
            and diagnosis.get("owner") == OWNER and diagnosis.get("batch") == ack["batch"]
            and diagnosis.get("attempt") == ack["attempt"]
            and diagnosis.get("job_id") == job and diagnosis.get("ack") == p.descriptor(previous / "ACK.json")
            and scheduled.get("state") == "FAILED" and scheduled.get("exit_code") == code
            and diagnosis.get("controller") == p.descriptor(SOURCE / "score_step4_v111.py")
            and isinstance(diagnosis.get("cause"), str) and diagnosis["cause"].strip() != ""
            and isinstance(diagnosis.get("repair"), str) and diagnosis["repair"].strip() != ""
            and (evidence.get("worker_log") or {}).get("path") == str(previous / ("worker-" + job + ".log")),
            "Exact owner DIAGNOSIS bound to the FAILED judge predecessor and this controller required")
    excerpt = evidence.get("log_excerpt")
    require(isinstance(excerpt, str) and excerpt.strip() != "" and excerpt.encode() in p.checked(evidence["worker_log"])
            and evidence.get("log_excerpt_sha256") == hashlib.sha256(excerpt.encode()).hexdigest(),
            "The diagnosis excerpt must be contained in the exact worker log and hashed")
    review_ref = p.descriptor(previous / ("DIAGNOSIS_REVIEW-%s.json" % job))
    review = json.loads(p.checked(review_ref))
    require(review.get("status") == DIAGNOSIS_REVIEW_STATUS and review.get("independent") is True
            and review.get("reviewer") not in (None, "", OWNER) and review.get("blocking_findings") == []
            and review.get("diagnosis") == diagnosis_ref and review.get("job_id") == job
            and review.get("controller") == p.descriptor(SOURCE / "score_step4_v111.py"),
            "Independent PASS review bound to this exact judge DIAGNOSIS and this controller required")
    return dict(diagnosis=diagnosis_ref, review=review_ref)


def live_rows(p):
    rows = [line.split("|") for line in p.query(["/usr/bin/squeue", "-u", "luyangz", "-r", "-h", "-o",
                                                  "%i|%q|%k|%j|%T|%b"]).splitlines() if line.strip()]
    require(all(len(row) == 6 for row in rows), "Expanded queue census is incomplete or ambiguous")
    return rows


def planned_batches(root):
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.name.startswith("judge-batch-") and any(d.glob("PLAN-*.json")))


def retire(p, version, notes):
    """Seal RETIRED.json for an earlier judge version with no live job whose every batch is withdrawn or scored."""
    require(version in RETIRE_COMMAND_VERSIONS, "Earlier judge version v1.10 required (v1.1/v1.2 retired by v1.3, v1.3 by v1.4, v1.4 by v1.5, v1.5 by v1.6, v1.6 by v1.7, v1.7 by v1.8, v1.8 by v1.9, v1.9 by v1.10)")
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete retirement notes required")
    root, prefix = RETIRED_VERSIONS[version]
    with p.admission_lock():
        require(not (root / "RETIRED.json").exists(), "Existing retirement is immutable")
        rows = live_rows(p)
        require(not any(row[3].startswith(prefix) for row in rows), "A live job of this judge version exists")
        batches = []
        for name in planned_batches(root):
            folder = root / name
            withdrawn = folder / "WITHDRAWN.json"
            results, jobs = [], []
            for attempt in sorted(folder.glob("attempt-*")):
                if (attempt / "ACK.json").exists():
                    ack = json.loads(p.checked(p.descriptor(attempt / "ACK.json")))
                    state, code = sacct_state(p, ack["job_id"])
                    require(state in TERMINAL, "A judge job of this version is not terminal")
                    jobs.append(dict(attempt=attempt.name, job_id=ack["job_id"], state=state, exit_code=code))
                for path in sorted(attempt.glob("RESULT-*.json")):
                    ref = p.descriptor(path)
                    results.append(dict(result=ref, status=json.loads(p.checked(ref)).get("status")))
            scored = any(str(r["status"]).startswith(SCORED_PREFIX) for r in results)
            require(withdrawn.exists() or scored, "Every batch must be withdrawn or completely scored: " + name)
            batches.append(dict(batch=name, withdrawn=p.descriptor(withdrawn) if withdrawn.exists() else None,
                                results=results, jobs=jobs,
                                plans=[p.descriptor(x) for x in sorted(folder.glob("PLAN-*.json"))]))
        return p.seal(root / "RETIRED.json", dict(
            schema=RETIRED_SCHEMA, status=RETIRED_STATUS, version=version, control=str(root), job_prefix=prefix,
            batches=[b["batch"] for b in batches], evidence=batches, successor=OWNER,
            authorization=COVERAGE_AUTHORIZATION,
            notes=notes, controller=p.descriptor(SOURCE / "score_step4_v111.py")))


def retired_versions(p):
    """Every earlier judge version carries an exact retirement naming exactly its existing batches."""
    receipts = {}
    for version, (root, prefix) in sorted(RETIRED_VERSIONS.items()):
        require((root / "RETIRED.json").is_file(),
                "Exact retirement of earlier judge version %s naming exactly its batches required" % version)
        ref = p.descriptor(root / "RETIRED.json")
        receipt = json.loads(p.checked(ref))
        require((receipt.get("schema"), receipt.get("status"), receipt.get("successor")) in ACCEPTED_RETIREMENTS
                and receipt.get("version") == version and receipt.get("control") == str(root)
                and receipt.get("job_prefix") == prefix
                and receipt.get("batches") == planned_batches(root),
                "Exact retirement of earlier judge version %s naming exactly its batches required" % version)
        receipts[version] = ref
    return receipts


def withdraw(p, batch, route, notes):
    """Retire a batch with no live job and no complete RESULT so that its cells may be planned again."""
    require(isinstance(notes, list) and notes and all(isinstance(x, str) and x for x in notes),
            "Concrete withdrawal notes required")
    folder = batch_folder(batch)
    plan_ref = p.descriptor(folder / plan_name(route))
    plan = json.loads(p.checked(plan_ref))
    with p.admission_lock():
        rows = live_rows(p)
        prefix = job_name(batch, 0)[:-1]
        require(not any(plan["function_identity"] in row[2] or row[3].startswith(prefix) for row in rows),
                "A live attempt of this batch exists; never withdraw a batch with a live job")
        results = []
        for attempt in range(MAX_ATTEMPTS):
            for path in sorted((folder / ("attempt-%d" % attempt)).glob("RESULT-*.json")):
                status = json.loads(p.checked(p.descriptor(path))).get("status")
                require(status != RESULT_STATUS, "A complete batch is never withdrawn")
                results.append(dict(path=str(path), status=status))
        return p.seal(folder / "WITHDRAWN.json", dict(
            status=WITHDRAWN_STATUS, owner=OWNER, kind="judge", batch=batch, plan=plan_ref,
            function_identity=plan["function_identity"], cells=list(plan["function"]["cells"]),
            results=results, notes=notes, controller=p.descriptor(SOURCE / "score_step4_v111.py")))


def admit(p, batch, attempt, route, release=False):
    check_attempt(attempt)
    routes = bound_module(p, "routes.py", "tool1_bench4_judge_v111_routes_admit")
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, batch, route, independent=release)
    afolder = folder / ("attempt-%d" % attempt)
    with p.admission_lock():
        retired_versions(p)
        require(not (folder / "WITHDRAWN.json").exists(), "A withdrawn batch is never submitted or released")
        require(contract(p, batch, route, independent=release)
                == (folder, plan_ref, plan, review_ref, independent_ref),
                "Contract changed while acquiring shared admission lock")
        if not release:
            predecessor = predecessor_allows(p, folder, attempt)
            rows, gate = routes.actual_gate(p.query, route)
            require(not any(plan["function_identity"] in row[2] for row in rows),
                    "A live attempt of this judge batch already exists")
            require(not any(row[3].startswith(ANY_JUDGE_PREFIX) for row in rows),
                    "Only one live BENCH4 judge job of any version at a time (shared ledger and budget)")
            private_directory(afolder)
            require(not any((afolder / x).exists() for x in
                            ("INTENT.json", "ACK.json", "AMBIGUOUS.json", "FUNCTION_CLAIM.json")),
                    "Existing submission or ambiguous ACK: reconcile, never duplicate")
            generation_gate = (afterok_gate(p, families(p), plan["function"]) if "dependency" in plan else None)
            claim = p.seal(afolder / "FUNCTION_CLAIM.json", dict(
                owner=OWNER, kind="judge", batch=batch, attempt=attempt, route=route, function=plan["function"],
                function_identity=plan["function_identity"], plan=plan_ref, predecessor=predecessor,
                generation_gate=generation_gate))
            intent = p.seal(afolder / "INTENT.json", dict(
                owner=OWNER, kind="judge", batch=batch, attempt=attempt, route=route, plan=plan_ref, review=review_ref,
                claim=claim, argv=argv(batch, attempt, plan), route_gate=gate, expanded_requests=len(rows)))
            try:
                response = subprocess.run(argv(batch, attempt, plan), text=True, capture_output=True,
                                          timeout=45, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired) as error:
                p.seal(afolder / "AMBIGUOUS.json", dict(intent=intent, error=repr(error)))
                raise
            good = response.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;[\w.-]+)?\n?", response.stdout)
            if good:
                job = response.stdout.strip().split(";")[0]
                ack_ref = p.seal(afolder / "ACK.json", dict(owner=OWNER, kind="judge", batch=batch, attempt=attempt,
                                                           route=route, job_id=job, intent=intent, plan=plan_ref))
            p.seal(afolder / "SUBMISSION_RESULT.json", dict(
                intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
            require(good, "No unambiguous actual ACK; reconcile exact intent before another submission")
            raw, spool = scheduler(p, batch, attempt, plan, job, held=True)
            p.seal(afolder / "HELD_READBACK.json", dict(
                ACK=ack_ref, job_id=job, scheduler=raw, spool_base64=base64.b64encode(spool).decode()))
            return dict(kind="judge", batch=batch, attempt=attempt, route=route, job_id=job, state="HELD_VERIFIED")
        ack_ref = p.descriptor(afolder / "ACK.json")
        ack = json.loads(p.checked(ack_ref))
        require(ack["owner"] == OWNER and ack["plan"] == plan_ref and ack["kind"] == "judge" and ack["batch"] == batch
                and ack["attempt"] == attempt and ack["route"] == route, "Exact own ACK required")
        require(not (afolder / "RELEASE_INTENT.json").exists(), "Existing release requires reconciliation")
        job = ack["job_id"]
        scheduler(p, batch, attempt, plan, job, held=True)
        others = [row for row in live_rows(p) if row[3].startswith(ANY_JUDGE_PREFIX) and row[0] != job]
        require(not others, "Another live BENCH4 judge job exists; release one judge job at a time")
        if "dependency" in plan:
            p.seal(afolder / "RELEASE_GENERATION_GATE.json", dict(ACK=ack_ref, job_id=job,
                                                                 gate=afterok_gate(p, families(p), plan["function"])))
        intent = p.seal(afolder / "RELEASE_INTENT.json", dict(ACK=ack_ref, job_id=job,
                                                             independent_review=independent_ref))
        response = subprocess.run(["/usr/bin/scontrol", "release", job], text=True, capture_output=True,
                                  timeout=30, env=p.SLURM_ENV, stdin=subprocess.DEVNULL)
        p.seal(afolder / "RELEASE_RESULT.json", dict(
            intent=intent, returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        require(response.returncode == 0, "Release failed or ambiguous; preserve exact identity")
        raw, _ = scheduler(p, batch, attempt, plan, job)
        row = p.fields(raw)
        require(row.get("JobState") in ("PENDING", "RUNNING", "CONFIGURING")
                and row.get("Reason") not in ("JobHeldUser", "JobHeldAdmin", "DependencyNeverSatisfied"),
                "Released judge job must be eligible, dependency-pending or running")
        p.seal(afolder / "RELEASE_READBACK.json", dict(job_id=job, route=route, scheduler=raw))
        return dict(kind="judge", batch=batch, attempt=attempt, route=route, job_id=job,
                    state=row.get("JobState") + ":" + row.get("Reason", ""))


# ------------------------------------------------------------------------------------------------- judging
def publish_once(path, value):
    """Create-once SCORE file (O_EXCL + fsync); an existing file must hold the identical bytes."""
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    except FileExistsError:
        with os.fdopen(os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC), "rb") as stream:
            require(stream.read((32 << 20) + 1) == raw, "Existing SCORE file differs")
    else:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def bank_inputs(banks, generation, function, name, cache):
    """Exported predictions of one bank aligned with the inventory roster the generation journal was bound to."""
    report = [r for r in generation["banks"] if r["protocol"] == name][0]
    require(plain(report["predictions"]) == function["predictions"][generation["cell"]][name],
            "Generation predictions differ from the plan pin: " + name)
    if name not in cache:
        cache[name] = (jsonl(read_ref(function["questions"][name])), jsonl(read_ref(function["labels"][name])))
    rows, labels = cache[name]
    bank_function = json.loads(read_ref(report["function"], 1 << 20))
    require(bank_function.get("questions_sha256") == function["questions"][name]["sha256"]
            and bank_function.get("protocol") == name and bank_function.get("cell") == generation["cell"]
            and bank_function.get("full_roster_sha256") == report["roster"]["ordered_semantic_keys_sha256"],
            "Generation journal was bound to another roster: " + name)
    require(banks.validate_questions(name, rows, expected_items=len(rows)) == report["roster"],
            "Inventory roster differs from the generated roster: " + name)
    predictions = banks.parse_predictions(name, read_ref(report["predictions"]), rows)
    return report, rows, labels, predictions


def judge_units(judge, prompt, rules, run, inputs):
    labels, predictions = inputs
    if run["bank"] == "mm-vet":
        return [(run["cell"], l["key"], [{"role": "user", "content": judge.mmvet_question(prompt, l, p["text"])}])
                for l, p in zip(labels, predictions)]
    return [(run["cell"], l["question_id"], judge.wild_messages(rules, l, p["text"])) for l, p in zip(labels, predictions)]


def run_key(run):
    return (run["model"], run["bank"], run["cell"])


def score_judge_run(judge, client, ledger, run, inputs, prompt, rules, hashes, generation, function, folder):
    labels, predictions = inputs
    model, bank, cell = run_key(run)
    if bank == "mm-vet":
        summary, scores, evidence = judge.score_mmvet(client, model, cell, prompt, labels, predictions)
    else:
        summary, evidence = judge.score_wild(client, model, cell, rules, labels, predictions)
        scores = None
    for item in evidence:
        item.pop("cached", None)
    spec = function["judge"]
    cost = ledger.run_cost(model, bank, cell)
    document = dict(
        summary, owner=OWNER, version=spec["version"], cell=cell, protocol=bank, role=run["role"], rank=run["rank"],
        judge_model_requested=model, request_parameters=spec["request_parameters"][model][bank],
        temperature=judge.TEMPERATURE if run["role"] != "SUPPLEMENTARY" else None,
        repeat_policy=judge.REPEAT_POLICY, missing_policy=judge.MISSING_POLICY[run["role"]], prompt_hashes=hashes,
        prices_usd_per_mtok=judge.PRICES_USD_PER_MTOK, price_policy=judge.PRICE_POLICY, run_cost=cost,
        parameter_sha256=generation["parameter_sha256"], predictions=function["predictions"][cell][bank],
        labels=function["labels"][bank], scorer=function["scorers"][bank],
        official_judge=spec["official_judges"][bank], deviation=spec["deviation"], scientific_claims=False,
        independently_reviewed=False, held_out_claim=False)
    reference = publish_once(folder / ("SCORE_%s__%s.json" % (bank, model)), document)
    evidence_ref = publish_once(folder / ("JUDGE_EVIDENCE_%s__%s.json" % (bank, model)),
                                dict(cell=cell, protocol=bank, model=model, items=evidence))
    return dict(score=reference, evidence=evidence_ref, overall=summary.get("total", summary.get("overall")),
                missing=summary.get("missing"), judge_models_returned=summary["judge_models_returned"],
                cost=cost), scores


def time_left_seconds(text):
    match = re.fullmatch(r"(?:(\d+)-(\d+):|(\d+):)?(\d+):(\d+)", text.strip())
    require(match is not None, "Actual remaining Slurm time required")
    days, day_hours, hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    require(minutes < 60 and seconds < 60 and day_hours < 24, "Malformed remaining Slurm time")
    return (((days * 24 + day_hours + hours) * 60) + minutes) * 60 + seconds


def campaign_status(judge, ledger, campaign):
    done = {role: 0 for role in ("PRIMARY", "CALIBRATION", "SUPPLEMENTARY")}
    remaining = dict(done)
    estimate = 0.0
    for run in campaign:
        if run_key(run) in ledger.completed:
            done[run["role"]] += 1
        else:
            remaining[run["role"]] += 1
            estimate += judge.run_estimate(ledger, run["model"], run["bank"])
    return dict(completed_runs=done, remaining_runs=remaining, remaining_estimate_usd=estimate)


def judge_work(p, plan, folder, afolder, job, deadline, transport_factory=None, key_reader=None,
               models_reader=None):
    """All runs of one judge batch; returns the RESULT body. Key read only after every check and the first gate."""
    banks = bound_module(p, "banks4.py", "tool1_bench4_judge_v111_banks_worker")
    judge = bound_module(p, "judge4_v13.py", "tool1_bench4_judge_v111_worker")
    function = plan["function"]
    spec = function["judge"]
    prompt = judge.mmvet_prompt(p.checked(SOURCES["mm-vet_evaluator.py"]))
    rules = judge.wild_rules(p.checked(SOURCES["rule.json"]), p.checked(SOURCES["eval_gpt_review_bench.py"]))
    hashes = judge.prompt_hashes(prompt, rules)
    require(hashes == spec["prompt_hashes"], "Pinned judge prompts differ from the plan")
    campaign = active_campaign(judge, spec["campaign_cells"])
    runs = spec["runs"]
    require(runs == active_batch_runs(judge, function["cells"], spec["campaign_cells"])
            and all(run["role"] in ACTIVE_ROLES for run in runs), "Plan runs differ from the bound judge policy")
    cache, inputs, generations = {}, {}, {}
    families_ = families(p)
    for cell, ref in function["generation_results"].items():
        generations[cell] = bound_generation(p, families_, cell, ref)
        if cell in function["tool2_manifest_final_results"]:
            family = family_of(families_, cell)
            require(family["name"] in T2_FAMILY_SOURCES, "Tool2 manifest final RESULT bound for a non-Tool2 cell")
            family["module"].final_equals_home(read_ref(function["tool2_manifest_final_results"][cell], 2 << 20),
                                               family["module"].manifest_entry(family["manifest"], cell, "bench4"),
                                               generations[cell])
        for name in JUDGE_BANKS:
            _, rows, labels, predictions = bank_inputs(banks, generations[cell], function, name, cache)
            require([l["question_id"] for l in labels] == [r["native_id"] for r in rows], "Judge labels misaligned")
            inputs[(cell, name)] = (labels, predictions)
    private_directory(LEDGER.parent.parent)
    private_directory(LEDGER.parent)
    ledger = judge.Ledger(LEDGER, job)
    try:
        spend_before = ledger.spend()
        projections = {run_key(run): judge.project(run["model"], run["bank"], judge_units(
            judge, prompt, rules, run, inputs[(run["cell"], run["bank"])]), ledger) for run in runs}

        def gate(run, unavailable=()):
            reserve = judge.reservation(ledger, campaign, run, unavailable)
            spend = ledger.spend()
            return spend + reserve + projections[run_key(run)] <= judge.STOP_USD, spend, reserve

        base = dict(version=judge.VERSION, spend_before_usd=spend_before, stop_usd=judge.STOP_USD,
                    hard_budget_usd=judge.HARD_BUDGET_USD,
                    run_projections_usd=[dict(run, usd=projections[run_key(run)]) for run in runs])
        todo = [run for run in runs if run_key(run) not in ledger.completed]
        if todo:
            affordable, spend, reserve = gate(todo[0])
            if not affordable:
                return dict(base, status=PARTIAL_STATUS, runs=[], skipped=[], deferred=todo, budget_stop=True,
                            stop_reason="BUDGET: first unfinished run %s/%s/%s does not fit (spend %.4f, reservation "
                                        "%.4f)" % (run_key(todo[0]) + (spend, reserve)),
                            abort_reason=None, availability=None, agreements={}, spend_after_usd=spend,
                            spend_by_model_usd=ledger.by_model(), paid_calls=0, cached_calls=0,
                            torn_ledger_lines=ledger.torn, api_key_read=False,
                            campaign=campaign_status(judge, ledger, campaign))
        secret = (key_reader or judge.read_key)()
        availability = judge.model_availability((models_reader or judge.https_models)(secret))
        p.seal(afolder / ("MODELS-" + job + ".json"), dict(job_id=job, endpoint=judge.MODELS_ENDPOINT,
                                                           time=time.time(), **availability))
        unavailable = sorted(m for m, ok in availability["available"].items() if not ok)
        transport = (transport_factory or (lambda s: (lambda body: judge.https_transport(s, body))))(secret)
        client = judge.Client(ledger, transport, deadline=deadline)
        private_directory(folder / "cells")
        done, skipped, deferred, mmvet_scores = [], [], [], {}
        stopped, budget_stop, aborted = None, False, None
        if judge.PRIMARY in unavailable:
            aborted = "PRIMARY judge %s is not listed by the models endpoint" % judge.PRIMARY
        for position, run in enumerate(runs if aborted is None else []):
            if run["model"] in unavailable:
                skipped.append(dict(run, reason="MODEL_NOT_LISTED"))
                continue
            if run_key(run) not in ledger.completed:
                affordable, spend, reserve = gate(run, unavailable)
                if not affordable:
                    stopped, budget_stop = ("BUDGET: run %s/%s/%s does not fit (spend %.4f, reservation %.4f, "
                                            "projection %.4f)" % (run_key(run) + (spend, reserve,
                                                                                  projections[run_key(run)]))), True
                    deferred = [r for r in runs[position:] if r["model"] not in unavailable]
                    break
            client.reserve_usd = judge.reservation(ledger, campaign, run, unavailable)
            cell_folder = folder / "cells" / run["cell"]
            private_directory(cell_folder)
            try:
                record, scores = score_judge_run(judge, client, ledger, run, inputs[(run["cell"], run["bank"])],
                                                 prompt, rules, hashes, generations[run["cell"]], function, cell_folder)
            except judge.BudgetStop as error:
                stopped = str(error)
                budget_stop = "deadline" not in stopped.lower()
                deferred = [r for r in runs[position:] if r["model"] not in unavailable]
                break
            except judge.JudgeAbort as error:
                aborted = str(error)
                deferred = [r for r in runs[position:] if r["model"] not in unavailable]
                break
            ledger.run_complete(run["model"], run["bank"], run["cell"], record["score"], record["cost"]["usd"])
            done.append(dict(run, **record))
            if run["bank"] == "mm-vet":
                mmvet_scores[(run["role"], run["cell"])] = scores
        agreements = {}
        for (role_name, cell), scores in sorted(mmvet_scores.items()):
            if role_name == "PRIMARY" or ("PRIMARY", cell) not in mmvet_scores:
                continue
            other = spec["models"][role_name]
            name = "CALIBRATION_mm-vet.json" if role_name == "CALIBRATION" else "SUPPLEMENTARY_AGREEMENT_mm-vet.json"
            agreements["%s/%s" % (cell, role_name)] = publish_once(folder / "cells" / cell / name, dict(
                cell=cell, primary=spec["models"]["PRIMARY"], other=other, other_role=role_name,
                agreement=judge.judge_agreement(mmvet_scores[("PRIMARY", cell)], scores)))
        if aborted is not None or skipped:
            status = ABORTED_STATUS
        elif stopped is not None:
            status = PARTIAL_STATUS
        else:
            status = RESULT_STATUS
        return dict(base, status=status, runs=done, skipped=skipped, deferred=deferred, stop_reason=stopped,
                    budget_stop=budget_stop, abort_reason=aborted, availability=availability, agreements=agreements,
                    spend_after_usd=ledger.spend(), spend_by_model_usd=ledger.by_model(),
                    paid_calls=client.paid_calls, cached_calls=client.cached_calls, torn_ledger_lines=ledger.torn,
                    api_key_read=True, campaign=campaign_status(judge, ledger, campaign))
    finally:
        ledger.close()


def work(p, batch, attempt, route):
    check_attempt(attempt)
    folder, plan_ref, plan, review_ref, independent_ref = contract(p, batch, route, independent=True)
    afolder = folder / ("attempt-%d" % attempt)
    job = os.environ.get("SLURM_JOB_ID", "")
    ack_ref = p.descriptor(afolder / "ACK.json")
    ack = json.loads(p.checked(ack_ref))
    require(os.getuid() == UID and job.isdecimal() and ack["job_id"] == job and ack["kind"] == "judge"
            and ack["batch"] == batch and ack["attempt"] == attempt and ack["route"] == route
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0" and not os.environ.get("SLURM_ARRAY_JOB_ID"),
            "Actual unrestarted owned judge allocation required")
    release = json.loads(p.checked(p.descriptor(afolder / "RELEASE_INTENT.json")))
    require(release == dict(ACK=ack_ref, job_id=job, independent_review=independent_ref),
            "Actual independently reviewed release intent required before labels")
    raw, _ = scheduler(p, batch, attempt, plan, job, running=True)
    node = socket.gethostname().split(".")[0]
    p.validate_process(job, os.getpid(), node, os.environ.get("SLURM_STEP_ID"),
                       p.query(["/usr/bin/scontrol", "listpids", job]),
                       p.fields(p.query(["/usr/bin/scontrol", "show", "step", job + ".batch", "-o"])),
                       (Path("/proc") / str(os.getpid()) / "status").read_text())
    left = time_left_seconds(p.query(["/usr/bin/squeue", "-h", "-j", job, "-o", "%L"]))
    require(left > RESERVE_SECONDS + 300, "Too little allocation left for judging")
    actor = dict(job_id=job, pid=os.getpid(), node=node, plan=plan_ref, kind="judge", batch=batch, attempt=attempt,
                 route=route)
    p.seal(afolder / ("ACTOR_BEFORE-" + job + ".json"), actor)
    work_plan = plan
    if "dependency" in plan:
        families_ = families(p)
        gate = afterok_gate(p, families_, plan["function"], final=True)
        results = gate["generation_results"]
        require(sorted(results) == sorted(plan["function"]["cells"]),
                "Every bound cell needs its own complete generation result before labels or API use")
        predictions = {}
        for cell, ref in sorted(results.items()):
            generation = bound_generation(p, families_, cell, ref)
            predictions[cell] = {r["protocol"]: plain(r["predictions"]) for r in generation["banks"]
                                 if r["protocol"] in JUDGE_BANKS}
        work_plan = dict(plan, function=dict(plan["function"], generation_results=results, predictions=predictions))
        p.seal(afolder / ("LATE_BINDING-" + job + ".json"), dict(plan=plan_ref, gate=gate, predictions=predictions))
    body = judge_work(p, work_plan, folder, afolder, job, deadline=time.time() + left - RESERVE_SECONDS)
    result = dict(body, owner=OWNER, kind="judge", batch=batch, attempt=attempt, job_id=job, model_forwards=0,
                  training_updates=0, scheduler=raw)
    reference = p.seal(afolder / ("RESULT-" + job + ".json"), result)
    scheduler(p, batch, attempt, plan, job, running=True)
    p.seal(afolder / ("ACTOR_AFTER-" + job + ".json"), actor)
    print(json.dumps(dict(job_id=job, batch=batch, status=result["status"], result=reference), sort_keys=True),
          flush=True)
    require(result["status"] in (RESULT_STATUS, PARTIAL_STATUS), "Judge attempt aborted: " + result["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "self-review", "submit", "release", "worker", "withdraw",
                                            "retire"))
    parser.add_argument("--batch", type=int)
    parser.add_argument("--version", choices=RETIRE_COMMAND_VERSIONS)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--route", choices=ROUTES)
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--afterok", action="store_true")
    args = parser.parse_args()
    p = load_helpers()
    if args.command == "retire":
        require(args.batch is None and args.route is None and args.version, "retire takes --version and --note only")
        print(json.dumps(retire(p, args.version, args.note), sort_keys=True))
        return
    require(args.batch is not None and args.route is not None and args.version is None,
            "--batch and --route are required")
    if args.command == "worker":
        work(p, args.batch, args.attempt, args.route)
    elif args.command == "plan":
        print(json.dumps(make_plan_file(p, args.batch, args.route, args.cell, afterok=args.afterok), sort_keys=True))
    elif args.command == "self-review":
        print(json.dumps(owner_review(p, args.batch, args.route, args.note), sort_keys=True))
    elif args.command == "withdraw":
        print(json.dumps(withdraw(p, args.batch, args.route, args.note), sort_keys=True))
    else:
        print(json.dumps(admit(p, args.batch, args.attempt, args.route, release=args.command == "release"),
                         sort_keys=True))


if __name__ == "__main__":
    main()
