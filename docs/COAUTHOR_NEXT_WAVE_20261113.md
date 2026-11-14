# Coauthor agent: next execution wave (2026-11-13)

The project lead approved this assignment. Work on the `run` branch, not `main`.
The binding protocol is [AGENT_CONTRACT.md](../AGENT_CONTRACT.md), especially
section 1a. The scheduler order is [AGENTS.md](../AGENTS.md), section 3a.

## Agent entry points and precedence

- Claude Code: root [CLAUDE.md](../CLAUDE.md) points here and to the shared rules.
- Codex and other AGENTS-aware agents: root [AGENTS.md](../AGENTS.md) points here.
- This assignment is agent/model-independent. If your client does not load either
  entry file automatically, open them explicitly.
- Authority: explicit project-lead instructions, then contract section 1a, then
  the rest of the contract. This brief, the run order, JSON and README summarize
  those rules; if an unresolved conflict remains, do not guess or launch that cell.
- Priority is not a requirement to idle: choose the highest-priority ready cell,
  after checking passed gates, unique ownership and actual GPU availability.

## Start now

1. Fetch `origin/run`. Preserve local edits and reconcile them before pulling;
   never reset, force-push, or start a duplicate controller. Incorporate the
   coverage-first and model-identity amendments (`a1139d7`, `20d1653`) and the
   later execution clarifications. Record the actual HEAD in the run log.
2. Read the contract, run order, machine-readable contract and existing coauthor
   results. Reconcile live jobs and exact cell ownership before launching.
3. Diagnose the reported CUDA failure bursts before retries or new long GPU
   runs. Correlation is not proof of hardware failure: inspect application and
   controller logs, driver/Xid evidence, memory and concurrent GPU ownership.
   Record the evidence, remedy and qualification. Do not disturb other workloads.
   These instructions do not authorize a driver reset or killing other users'
   processes. If access prevents diagnosis, record the exact blocker and seek
   the instance owner's assistance; do not invent a cause.
4. Implement and qualify faithful selection routes, starting with VisNec and
   its released-subset fidelity check, then ICONS, XMAS and CoIDO. Independent,
   ready preparation may proceed in parallel; no dummy GPU work or premature
   student launch. Use the existing passed 7B calibration where compatible.

## Scientific roster

Training seed is 17 throughout this wave. Each student needs a complete core-four
evaluation, then the six locally scored extended metrics after their gate.
LLaVA-W and MM-Vet remain deferred. A blocked dependency does not block unrelated
ready cells.

| Order | Work | Training model | Selected-data budgets | Students | Readiness / novelty |
|---|---|---|---|---:|---|
| 1 | VisNec | 7B | 10%, then 5%, then 20% | 3 | New; official selection + fidelity first |
| 2 | ICONS | 7B | 10%, then 5%, then 20% | 3 | New; warm-up is not a finished student |
| 3 | XMAS | 7B | 10%, then 5%, then 20% | 3 | New; preserve disclosed release fixes |
| 4 | CoIDO | 7B | 10%, then 5%, then 20% | 3 | New; prescribed quality judge still required |
| Gate | T0d: evaluate our 13B reference and retrain that exact cell | 13B | Uniform 5%, selection seed 1235 | 1 | Intentional calibration repeat; counts as the T13 Uniform 5% row |
| 5 | Uniform | 13B | 10%, selection seed 1234 | 1 | New; T0d must pass |
| 6 | DataTailor | 13B | 5%, 10%, selection seed 1234 | 2 | New training-model transfer; T0d must pass |
| 7 | COINCIDE (existing adaptation label) | 13B | 5% fixed selection, 10% selection seed 1234 | 2 | New training-model transfer; T0d must pass |
| 8 | VisNec, ICONS, XMAS, CoIDO | 13B | 10% each | 4 | New selected-set transfer; selected sets + T0d required |
| Gated | T8 full-data reference | 7B | 100% | 1 | Reconcile manager ownership/live continuation first; do not duplicate |

These are missing cells relative to the audited `7bfc695` results, not permission
to ignore subsequent work. Check again immediately before launch. The primary
roster has 12 new 7B students, 9 new 13B cells and one intentional calibration
retrain. T8 is separate and ownership-gated.

## Completion and handoff gates

- A scientific cell is complete only with its exact selected-set identity, full
  natural-epoch checkpoint, full-bank scores, raw predictions, provenance and
  failed-inclusive costs. Missing extended banks remain explicitly incomplete.
- A core-four result may be reported while extended banks remain pending; do not
  relabel that as the entire evaluation suite completed.
- The T0d reference is a pending delivery, not a currently available artifact.
  Require its HF revision, checksums, training log/state, predictions and scores.
  Its calibration retrain is allowed to establish the gate; it does not depend
  on its own final PASS. Other T13 production cells do require that PASS.
- If only 13B is blocked, continue qualified 7B work. If all work is blocked,
  report exact attempted remedies and prerequisites instead of fabricating
  completion or waiting without a status record.

## Invariants and exclusions

- Our method variants run on Babel; you own baselines/controls, not those variants.
- Baselines use published fixed procedures/settings: no hyperparameter search,
  best-seed selection, hidden negative results or substitutions. Disclose all
  completed seeds and exact adaptation labels.
- Our best-variant/best-seed view is internal exploratory reporting. Its
  aggregation differs from fixed baseline rows; it is not a matched superiority
  estimate.
- Coverage first: seeds 29/43 and old extended-evaluation retries are later.
  Do not retrain already completed 7B Uniform/DataTailor/COINCIDE cells.
- No new 15% jobs or retries. Preserve all existing 15% results and failed attempts.
- Author-approved budgets do not override native-method feasibility. If an
  official route cannot support a requested setting, report it as blocked; do
  not invent a faithful reproduction.
- T13 transfers existing selected sets. It is not a native 13B selector rerun.
- Build LLaVA-1.5-13B from pinned Vicuna-13B-v1.5 + CLIP-336 + the 13B stage-1
  projector. Never initialize from instruction-tuned `liuhaotian/llava-v1.5-13b`.
- The authorized two-A100 layout (4 x 16 x 2 = global batch 128) is not yet
  qualified. L40S reserved memory alone does not prove single-A100 impossibility.
  T13 waits for the reference checkpoint, scores and passing T0d; continue ready
  7B work meanwhile. Do not silently relax the gate or change the recipe.

## Hand-back

Push fast-forward progress to `run`: cell identities, actual jobs/processes,
completed/running/blocked states, raw-result provenance, fidelity notes, and
failed-inclusive measured costs. Distinguish copied scores from fresh evaluation
and selected-set reuse from native selection. Update the coauthor results, run
log, calibration report and instance status. Upload qualified weights only to HF,
never Git. Do not claim preparation or a test PASS as a completed experiment.
Group results by base model as well as method, budget and both seeds. Keep the
existing CSV columns; every new row must identify `base_model=llava-1.5-7b` or
`base_model=llava-1.5-13b` in `notes`. Historical untagged rows are 7B. Never
overwrite a 7B row with a 13B result having the same method/budget/seeds.
