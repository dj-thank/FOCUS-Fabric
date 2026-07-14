"""Memory-matched single-family baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .codecs import HeadCodec, OperatorCodec, WeightedCoresetCodec
from .types import AttentionSummary


@dataclass
class StaticPage:
    heads: list[HeadCodec]
    dtype_bytes: int = 4

    def evaluate_batch(self, queries: Tensor) -> AttentionSummary:
        outputs: list[Tensor] = []
        masses: list[Tensor] = []
        for head, codec in enumerate(self.heads):
            result = codec.evaluate_batch(queries[head])
            outputs.append(result.summary.output)
            masses.append(result.summary.log_mass)
        return AttentionSummary(torch.stack(outputs), torch.stack(masses))

    def active_bytes(self) -> int:
        return sum(codec.active_bytes(self.dtype_bytes) for codec in self.heads)

    def report(self) -> dict[str, Any]:
        return {
            "active_bytes": self.active_bytes(),
            "codecs": [codec.metadata() for codec in self.heads],
        }


def compile_operator_only_matched(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    *,
    target_bytes_by_head: list[int],
    scale: float,
    dtype_bytes: int = 4,
    seed: int = 0,
    iterations: int = 12,
) -> StaticPage:
    heads: list[HeadCodec] = []
    for head in range(keys.shape[0]):
        candidates: list[OperatorCodec] = []
        for patches in (1, 2, 4, 8):
            if patches > queries.shape[1]:
                continue
            for rank in (1, 2, 4, 8):
                if rank > keys.shape[-1]:
                    continue
                candidates.append(
                    OperatorCodec.compile(
                        keys[head],
                        values[head],
                        queries[head],
                        patches=patches,
                        rank=rank,
                        scale=scale,
                        seed=seed + 1000 * head + 31 * patches + rank,
                        iterations=iterations,
                    )
                )
        target = target_bytes_by_head[head]
        feasible = [
            codec for codec in candidates if codec.active_bytes(dtype_bytes) <= target
        ]
        if feasible:
            chosen = max(feasible, key=lambda item: item.active_bytes(dtype_bytes))
        else:
            chosen = min(candidates, key=lambda item: item.active_bytes(dtype_bytes))
        heads.append(chosen)
    return StaticPage(heads, dtype_bytes)


def compile_coreset_matched(
    keys: Tensor,
    values: Tensor,
    *,
    target_bytes_by_head: list[int],
    queries: Tensor | None = None,
    scale: float,
    dtype_bytes: int = 4,
    seed: int = 0,
    iterations: int = 12,
) -> StaticPage:
    heads: list[HeadCodec] = []
    dimension = int(keys.shape[-1])
    floats_per_slot = 2 * dimension + 2
    for head in range(keys.shape[0]):
        slots = max(
            1,
            target_bytes_by_head[head] // (dtype_bytes * floats_per_slot),
        )
        slots = min(slots, int(keys.shape[1]) - 1)
        heads.append(
            WeightedCoresetCodec.compile(
                keys[head],
                values[head],
                slots=slots,
                scale=scale,
                seed=seed + 1009 * head,
                iterations=iterations,
                queries=(queries[head] if queries is not None else None),
                restarts=4 if queries is not None else 1,
            )
        )
    return StaticPage(heads, dtype_bytes)
