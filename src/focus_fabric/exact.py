"""Exact attention reference operations."""
from __future__ import annotations

import torch
from torch import Tensor

from .types import AttentionSummary


def exact_head_summary(query: Tensor, keys: Tensor, values: Tensor, scale: float) -> AttentionSummary:
    if query.ndim != 1 or keys.ndim != 2 or values.shape != keys.shape:
        raise ValueError("expected query[D] and matching keys/values[N,D]")
    if keys.shape[-1] != query.shape[-1]:
        raise ValueError("query and key dimensions differ")
    if keys.shape[0] == 0:
        return AttentionSummary(torch.zeros_like(query), query.new_tensor(float("-inf")))
    scores = torch.mv(keys.float(), query.float()) * float(scale)
    log_mass = torch.logsumexp(scores, dim=0).to(query.dtype)
    probabilities = torch.softmax(scores, dim=0).to(values.dtype)
    output = torch.mv(values.transpose(0, 1), probabilities).to(query.dtype)
    return AttentionSummary(output, log_mass)


def exact_head_batch(queries: Tensor, keys: Tensor, values: Tensor, scale: float) -> AttentionSummary:
    if queries.ndim != 2 or keys.ndim != 2 or values.shape != keys.shape:
        raise ValueError("expected queries[Q,D] and matching keys/values[N,D]")
    if keys.shape[0] == 0:
        output = torch.zeros_like(queries)
        log_mass = torch.full((queries.shape[0],), -torch.inf, device=queries.device, dtype=queries.dtype)
        return AttentionSummary(output, log_mass)
    scores = torch.matmul(queries.float(), keys.float().transpose(0, 1)) * float(scale)
    log_mass = torch.logsumexp(scores, dim=-1).to(queries.dtype)
    probabilities = torch.softmax(scores, dim=-1).to(values.dtype)
    output = torch.matmul(probabilities, values).to(queries.dtype)
    return AttentionSummary(output, log_mass)


def exact_multihead_summary(query: Tensor, keys: Tensor, values: Tensor, scale: float) -> AttentionSummary:
    if query.ndim != 2 or keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("expected query[H,D] and matching keys/values[H,N,D]")
    if keys.shape[0] != query.shape[0] or keys.shape[-1] != query.shape[-1]:
        raise ValueError("head or feature dimension mismatch")
    if keys.shape[1] == 0:
        return AttentionSummary(
            torch.zeros_like(query),
            torch.full((query.shape[0],), -torch.inf, device=query.device, dtype=query.dtype),
        )
    scores = torch.einsum("hd,hnd->hn", query.float(), keys.float()) * float(scale)
    log_mass = torch.logsumexp(scores, dim=-1).to(query.dtype)
    probabilities = torch.softmax(scores, dim=-1).to(values.dtype)
    output = torch.einsum("hn,hnd->hd", probabilities, values).to(query.dtype)
    return AttentionSummary(output, log_mass)
