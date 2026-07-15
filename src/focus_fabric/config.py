"""Validated configuration for compilation and runtime."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class CompilerConfig:
    operator_patches: tuple[int, ...] = (2, 4)
    operator_ranks: tuple[int, ...] = (2, 4, 8)
    coreset_slots: tuple[int, ...] = (4, 8, 16)
    gaussian_clusters: tuple[int, ...] = (2, 4, 8)
    gaussian_ranks: tuple[int, ...] = (2, 4, 8)
    moment_ranks: tuple[int, ...] = (2, 4, 6)
    hybrid_exact_slots: tuple[int, ...] = (2, 4, 8)
    target_active_ratio: float = 0.30
    rate_lambda: float = 0.15
    latency_lambda: float = 0.04
    logmass_weight: float = 0.25
    certificate_alpha: float = 0.05
    certificate_tolerance: float = 0.18
    min_queries: int = 24
    max_queries: int = 256
    kmeans_iterations: int = 24
    coreset_restarts: int = 4
    dtype_bytes: int = 4
    seed: int = 0

    def validate(self) -> None:
        if not 0 < self.target_active_ratio <= 1:
            raise ValueError("target_active_ratio must be in (0,1]")
        if self.rate_lambda < 0 or self.latency_lambda < 0:
            raise ValueError("objective weights must be non-negative")
        if not 0 < self.certificate_alpha < 1:
            raise ValueError("certificate_alpha must be in (0,1)")
        if self.certificate_tolerance <= 0:
            raise ValueError("certificate_tolerance must be positive")
        if self.min_queries < 9 or self.max_queries < self.min_queries:
            raise ValueError("query calibration sizes are invalid")
        if self.kmeans_iterations < 1 or self.coreset_restarts < 1:
            raise ValueError("kmeans_iterations and coreset_restarts must be positive")
        if self.dtype_bytes not in (1, 2, 4, 8):
            raise ValueError("dtype_bytes must be 1, 2, 4, or 8")
        for sequence in (self.operator_patches, self.operator_ranks, self.coreset_slots, self.gaussian_clusters, self.gaussian_ranks, self.moment_ranks, self.hybrid_exact_slots):
            if not sequence or any(value <= 0 for value in sequence):
                raise ValueError("candidate grids must contain positive integers")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FabricConfig:
    n_heads: int
    head_dim: int
    hot_window: int = 64
    page_size: int = 64
    exact_fallback: bool = True
    retain_exact_archive: bool = True
    query_bank_size: int = 256
    compiler: CompilerConfig = field(default_factory=CompilerConfig)
    scale: float | None = None

    @property
    def attention_scale(self) -> float:
        return self.scale if self.scale is not None else self.head_dim ** -0.5

    def validate(self) -> None:
        if self.n_heads <= 0 or self.head_dim <= 0:
            raise ValueError("n_heads and head_dim must be positive")
        if self.hot_window < 1 or self.page_size < 2:
            raise ValueError("hot_window>=1 and page_size>=2 are required")
        if self.query_bank_size < self.compiler.min_queries:
            raise ValueError("query_bank_size is smaller than compiler.min_queries")
        if not self.retain_exact_archive:
            raise ValueError("public hierarchical reference requires retain_exact_archive=True")
        if not math.isfinite(self.attention_scale) or self.attention_scale <= 0:
            raise ValueError("attention scale must be finite and positive")
        self.compiler.validate()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["attention_scale"] = self.attention_scale
        return payload
