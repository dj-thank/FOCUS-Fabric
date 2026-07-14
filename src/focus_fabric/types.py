"""Shared tensor containers for FOCUS-Fabric.

All attention summaries are locally normalized. ``output`` is the value
expectation inside one disjoint memory region and ``log_mass`` is that region's
log unnormalized softmax mass. This representation composes exactly across
pages, unlike averaging page outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class AttentionSummary:
    output: Tensor
    log_mass: Tensor

    def to(self, *args: Any, **kwargs: Any) -> "AttentionSummary":
        return AttentionSummary(self.output.to(*args, **kwargs), self.log_mass.to(*args, **kwargs))


@dataclass(frozen=True)
class CodecEvaluation:
    summary: AttentionSummary
    proxy: Tensor
    valid: Tensor


@dataclass(frozen=True)
class CodecMetrics:
    name: str
    active_bytes: int
    byte_ratio: float
    estimated_flops: int
    flop_ratio: float
    output_nmse: float
    output_relative_mean: float
    logmass_rmse: float
    combined_mean: float
    combined_p95: float
    combined_max: float
    invalid_rate: float
    objective: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "active_bytes": self.active_bytes,
            "byte_ratio": self.byte_ratio,
            "estimated_flops": self.estimated_flops,
            "flop_ratio": self.flop_ratio,
            "output_nmse": self.output_nmse,
            "output_relative_mean": self.output_relative_mean,
            "logmass_rmse": self.logmass_rmse,
            "combined_mean": self.combined_mean,
            "combined_p95": self.combined_p95,
            "combined_max": self.combined_max,
            "invalid_rate": self.invalid_rate,
            "objective": self.objective,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RuntimeDecision:
    summary: AttentionSummary
    used_fallback: bool
    certificate_upper: float
    codec_name: str
    proxy: float


@dataclass
class FabricStats:
    queries: int = 0
    page_head_evaluations: int = 0
    fallback_decisions: int = 0
    fallback_tokens: int = 0
    archive_bytes_read: int = 0
    pages_compiled: int = 0
    pages_merged: int = 0
    compacted_tokens: int = 0
    compile_seconds: float = 0.0
    attention_seconds: float = 0.0
    invalid_codec_outputs: int = 0
    certificate_upper_sum: float = 0.0

    def merge_(self, other: "FabricStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict[str, float | int]:
        decisions = self.page_head_evaluations
        return {
            "queries": self.queries,
            "page_head_evaluations": decisions,
            "fallback_decisions": self.fallback_decisions,
            "fallback_rate": self.fallback_decisions / decisions if decisions else 0.0,
            "fallback_tokens": self.fallback_tokens,
            "archive_bytes_read": self.archive_bytes_read,
            "pages_compiled": self.pages_compiled,
            "pages_merged": self.pages_merged,
            "compacted_tokens": self.compacted_tokens,
            "compile_seconds": self.compile_seconds,
            "attention_seconds": self.attention_seconds,
            "invalid_codec_outputs": self.invalid_codec_outputs,
            "mean_certificate_upper": self.certificate_upper_sum / decisions if decisions else 0.0,
        }


def merge_summaries(summaries: list[AttentionSummary]) -> AttentionSummary:
    """Exactly combine summaries over disjoint token regions."""

    if not summaries:
        raise ValueError("at least one summary is required")
    outputs = torch.stack([item.output for item in summaries], dim=0)
    masses = torch.stack([item.log_mass for item in summaries], dim=0)
    merged_mass = torch.logsumexp(masses.float(), dim=0).to(masses.dtype)
    finite = torch.isfinite(merged_mass)
    weights = torch.zeros_like(masses)
    if finite.any():
        weights[:, finite] = torch.exp(masses[:, finite] - merged_mass[finite].unsqueeze(0))
    merged_output = torch.sum(weights.unsqueeze(-1) * outputs, dim=0)
    return AttentionSummary(merged_output, merged_mass)
