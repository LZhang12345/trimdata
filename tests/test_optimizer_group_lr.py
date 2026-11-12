"""CPU check of the optimizer-group fix used by scripts/train_portable.py (no GPU, no model download).

Emulates LLaVA's four optimizer groups, DeepSpeed 0.12.6's removal of empty groups after the scheduler is built and
torch 2.1.2's unchecked zip in LRScheduler.step. Expected output: without the fix the projector learning rate equals
the LoRA rate and the runtime check raises; with the fix it is 0.1x and the check passes.
Run: python tests/test_optimizer_group_lr.py
"""
import sys, types, math, inspect, textwrap, importlib.util
from pathlib import Path
import torch, transformers
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / "src"))
from experiments.root_native_campaign_v7_20261024 import optimizer_groups
spec = importlib.util.spec_from_file_location("tp", REPO / "scripts/train_portable.py"); tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
NAMES = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
         "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
         "base_model.model.model.mm_projector.0.weight", "base_model.model.model.mm_projector.0.bias"]

def callback_class():
    full = inspect.getsource(tp.build_trainer_class)
    src = textwrap.dedent(full[full.index("    class LearningRateGroupCheck"):full.index("    class PortableBaselineTrainer")])
    ns = {}; exec(src, {"TrainerCallback": transformers.TrainerCallback, "math": math, "MAIN_LR": tp.MAIN_LR, "PROJECTOR_LR": tp.PROJECTOR_LR}, ns)
    return ns["LearningRateGroupCheck"]

def old_torch_step(s):  # torch 2.1.2 LRScheduler.step: zip() without strict
    s.last_epoch += 1; values = s.get_lr()
    for g, v in zip(s.optimizer.param_groups, values): g["lr"] = v
    s._last_lr = [g["lr"] for g in s.optimizer.param_groups] if False else values

def run(prune, steps=10):
    named = [(n, torch.nn.Parameter(torch.zeros(2))) for n in NAMES]
    proj = lambda n: "mm_projector" in n; decay = lambda n: not n.endswith("bias")
    opt = torch.optim.AdamW([dict(params=[p for n, p in named if decay(n) and not proj(n)], weight_decay=0.0),
                             dict(params=[p for n, p in named if not decay(n) and not proj(n)], weight_decay=0.0),
                             dict(params=[p for n, p in named if decay(n) and proj(n)], weight_decay=0.0, lr=2e-5),
                             dict(params=[p for n, p in named if not decay(n) and proj(n)], weight_decay=0.0, lr=2e-5)], lr=2e-4)
    if prune:
        optimizer_groups.prune_before_scheduler(opt, named, scheduler=None, main_lr=2e-4, projector_lr=2e-5, weight_decay=0)
    sched = transformers.get_cosine_schedule_with_warmup(opt, 16, 516)
    opt.param_groups[:] = [pg for pg in opt.param_groups if pg["params"]]   # DeepSpeed 0.12.6 empty-group removal
    for _ in range(steps): old_torch_step(sched)
    proj_weight = named[2][1]
    proj_lr = [g["lr"] for g in opt.param_groups if any(p is proj_weight for p in g["params"])][0]
    try:
        callback_class()().on_step_end(None, types.SimpleNamespace(global_step=10), None, lr_scheduler=sched); verdict = "runtime check PASSES"
    except RuntimeError as e:
        verdict = "runtime check RAISES"
    print("prune=%-5s steps=%-2d: LoRA lr %.3g, projector-weight lr %.3g (should be %.3g) -> %s" % (prune, steps, opt.param_groups[0]["lr"], proj_lr, opt.param_groups[0]["lr"] / 10, verdict))

run(False); run(True)
run(True, steps=0)  # fp16 overflow skipped the first scheduler step: all rates 0 during warmup
