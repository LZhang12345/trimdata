"""Preserve the author's FP32 allocation operation order exactly."""
from typing import Any

import numpy as np
import torch

from .datatailor_v1 import author_values, native_values, random_k, require, source_blocks


def select(
    records: list[dict[str, Any]],
    indices: list[int],
    informativeness: torch.Tensor,
    tau: torch.Tensor,
    uniqueness: torch.Tensor,
    representativeness: torch.Tensor,
    ratio: float = .15,
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
    weights = []
    total_weight = 0.
    for block in blocks:
        if block:
            mean = torch.mean(tau[block])
            weight = len(block) * (mean * mean)
        else:
            weight = torch.tensor(0.)
        weights.append(weight)
        total_weight += weight
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
    original = indices == list(range(665298))
    return dict(
        positions=chosen, source_indices=[indices[position] for position in chosen],
        count=len(chosen), source_quotas=quotas, source_counts=list(map(len, blocks)),
        ratio=ratio, top_up=False, whole_conversations=True, original_665298=original,
        pool_adaptation=None if original else "Original-source blocks intersected with the complete declared protected pool",
        author_original_order_preserved=True,
    )
