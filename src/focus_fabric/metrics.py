"""Metrics for attention approximation and sequence agreement."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .exact import exact_head_batch
from .types import AttentionSummary


def exact_multihead_batch(
    queries: Tensor, keys: Tensor, values: Tensor, scale: float
) -> AttentionSummary:
    if queries.ndim != 3 or keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("expected queries[H,Q,D] and matching keys/values[H,N,D]")
    outputs: list[Tensor] = []
    masses: list[Tensor] = []
    for head in range(keys.shape[0]):
        summary = exact_head_batch(queries[head], keys[head], values[head], scale)
        outputs.append(summary.output)
        masses.append(summary.log_mass)
    return AttentionSummary(torch.stack(outputs), torch.stack(masses))


def summary_metrics(
    approximation: AttentionSummary, exact: AttentionSummary
) -> dict[str, float]:
    difference = approximation.output.float() - exact.output.float()
    relative = torch.linalg.vector_norm(difference, dim=-1) / torch.linalg.vector_norm(
        exact.output.float(), dim=-1
    ).clamp_min(1e-5)
    mass_difference = approximation.log_mass.float() - exact.log_mass.float()
    return {
        "output_nmse": float(
            (
                difference.square().mean()
                / exact.output.float().square().mean().clamp_min(1e-8)
            ).item()
        ),
        "output_rmse": float(torch.sqrt(difference.square().mean()).item()),
        "output_relative_mean": float(relative.mean().item()),
        "output_relative_p95": float(torch.quantile(relative, 0.95).item()),
        "output_relative_max": float(relative.max().item()),
        "logmass_rmse": float(torch.sqrt(mass_difference.square().mean()).item()),
        "logmass_mae": float(mass_difference.abs().mean().item()),
    }


def token_agreement(left: Tensor, right: Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError("logit tensors must have identical shapes")
    left_token = left.argmax(dim=-1)
    right_token = right.argmax(dim=-1)
    difference = (left.float() - right.float()).abs()
    return {
        "argmax_token_agreement": float(
            (left_token == right_token).float().mean().item()
        ),
        "sequence_agreement": bool(torch.equal(left_token, right_token)),
        "max_logit_abs_error": float(difference.max().item()),
        "mean_logit_abs_error": float(difference.mean().item()),
    }
