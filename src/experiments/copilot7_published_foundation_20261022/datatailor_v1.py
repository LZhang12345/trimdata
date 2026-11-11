"""Native DataTailor values and original-source-aware, whole-record selection."""
import ast
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


AUTHOR_FEATURE_SHA = "a29531ef51eb0d660b3e514917012ff1ab138f9d35a31f03af57375850f36e9f"
STARTS = (0, 57669, 80909, 157712, 240495, 249493, 315653, 364100, 450517, 522657, 602657, 624610)
ENDS = (57668, 80908, 157711, 240494, 249492, 315652, 364099, 450516, 522656, 602656, 624609, 665297)


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def native_values(states: torch.Tensor) -> dict[str, Any]:
    require(states.ndim == 2 and states.shape[0] > 0 and states.shape[1] > 0,
            "Expected one unpadded pre-final-norm token matrix")
    require(bool(torch.isfinite(states).all()), "Nonfinite native hidden states")
    with torch.no_grad():
        _, singular, _ = torch.svd(states.float())
        normalized = F.normalize(singular, dim=0, p=1)
        require(bool((normalized > 0).all()) and bool(torch.isfinite(normalized).all()),
                "Author entropy is undefined for a zero singular value; preserve the failure, do not add epsilon")
        entropy = 0.0
        for index in range(normalized.shape[0]):
            probability = normalized[index].item()
            entropy += -probability * math.log(probability)
        return dict(
            hidden=states[-1, :].detach().cpu().clone(),
            length=states.shape[0],
            informativeness=torch.tensor(entropy, dtype=torch.float32).item(),
            tau=torch.tensor(torch.max(normalized).item(), dtype=torch.float32).item(),
        )


def author_values(path: Path, states: torch.Tensor) -> dict[str, Any]:
    """Execute the unchanged author's per-record loop, not a second formula."""
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == AUTHOR_FEATURE_SHA, "Author feature bytes changed")
    tree = ast.parse(source)
    loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [value.id for value in node.target.elts if isinstance(value, ast.Name)] == ["i", "data_id"]
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "enumerate"
        and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Name)
        and node.iter.args[0].id == "data_ids"
    ]
    require(len(loops) == 1, "Exact author per-record feature loop was not identified")
    namespace = dict(
        torch=torch, F=F, math=math, data_ids=[0],
        hidden_states=states.unsqueeze(0), causal_mask=None,
        total_global_hidden_state={}, length=torch.zeros(1),
        informativeness=torch.zeros(1), tau=torch.zeros(1),
    )
    module = ast.fix_missing_locations(ast.Module(body=[loops[0]], type_ignores=[]))
    with torch.no_grad():
        exec(compile(module, str(path), "exec"), namespace)
    return dict(
        hidden=namespace["total_global_hidden_state"][0].clone(),
        length=int(namespace["length"][0].item()),
        informativeness=namespace["informativeness"][0].item(),
        tau=namespace["tau"][0].item(),
    )


def source_blocks(indices: list[int]) -> list[list[int]]:
    require(indices == sorted(set(indices)) and bool(indices), "Unique ordered original source indices required")
    require(all(type(index) is int and 0 <= index <= ENDS[-1] for index in indices),
            "Invalid original source index; shortened-pool offsets are not original source IDs")
    blocks = [[] for _ in STARTS]
    for position, index in enumerate(indices):
        block = int(np.searchsorted(ENDS, index, side="left"))
        blocks[block].append(position)
    return blocks


def select(
    records: list[dict[str, Any]],
    indices: list[int],
    informativeness: torch.Tensor,
    tau: torch.Tensor,
    uniqueness: torch.Tensor,
    representativeness: torch.Tensor,
    ratio: float = 0.15,
) -> dict[str, Any]:
    blocks = source_blocks(indices)
    count = len(indices)
    require(len(records) == count and 0 < ratio <= 1, "Whole-pool records and a valid ratio required")
    for name, value in dict(informativeness=informativeness, tau=tau, uniqueness=uniqueness,
                            representativeness=representativeness).items():
        require(value.device.type == "cpu" and value.dtype == torch.float32 and value.shape == (count,),
                "Native saved FP32 CPU score vector required: " + name)
        require(bool(torch.isfinite(value).all()), "Nonfinite native score: " + name)
    require(bool((tau > 0).all()), "Nonpositive native task-difficulty values")
    turns = torch.tensor([0] * count)
    for index, record in enumerate(records):
        conversation = record["conversations"]
        require(isinstance(conversation, list) and len(conversation) >= 2
                and len(conversation) % 2 == 0, "Complete paired human/assistant turns required")
        turns[index] = len(conversation) / 2

    weights = [len(block) * torch.mean(tau[block]) * torch.mean(tau[block])
               if block else torch.tensor(0.0) for block in blocks]
    total_weight = sum(weights, 0.0)
    require(bool(total_weight > 0), "No native allocation mass")
    multiplier = ratio * count / total_weight
    chosen, quotas = [], []
    for block, weight in zip(blocks, weights):
        quota = int(multiplier * weight)
        require(quota <= len(block), "Author source quota exceeds source population; no silent clipping/top-up")
        quotas.append(quota)
        if not block:
            continue
        ranks = [np.argsort(np.argsort(value[block])) / len(block)
                 for value in (informativeness, uniqueness, representativeness)]
        pairs = []
        for local, position in enumerate(block):
            score = (ranks[0][local] * turns[position] + ranks[1][local] + ranks[2][local]) / (turns[position] + 2)
            pairs.append((score, position))
        pairs.sort(key=lambda pair: pair[0], reverse=True)
        chosen.extend(position for _, position in pairs[:quota])
    chosen.sort()
    return dict(
        positions=chosen, source_indices=[indices[position] for position in chosen],
        count=len(chosen), source_quotas=quotas, source_counts=list(map(len, blocks)),
        ratio=ratio, top_up=False, whole_conversations=True,
        original_665298=indices == list(range(665298)),
        pool_adaptation=None if indices == list(range(665298)) else "Original-source blocks intersected with the complete declared protected pool",
        author_original_order_preserved=True,
    )


def random_k(indices: list[int], k: int, seed: int) -> list[int]:
    import random

    source_blocks(indices)
    require(type(k) is int and 0 < k <= len(indices), "Actual emitted native K required")
    require(seed in (17, 29, 43), "Declared paired seed required")
    return sorted(random.Random(seed).sample(range(len(indices)), k))
