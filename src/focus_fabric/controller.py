"""Rate--distortion compiler and uncertainty-gated adaptive heads.

The compiler performs a strict three-way split of the observed query bank:
fit queries define local representations, selection queries rank candidate
codecs, and an untouched calibration split fits a marginal split-conformal
error certificate.  The cold exact K/V archive is never counted as active HBM;
it is the verifier and sparse fallback source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

from .certificate import ConformalCertificate
from .codecs import (
    ExactCodec,
    GaussianMixtureCodec,
    HeadCodec,
    HybridResidualCodec,
    MomentCodec,
    OperatorCodec,
    WeightedCoresetCodec,
)
from .config import CompilerConfig
from .exact import exact_head_batch, exact_head_summary
from .types import AttentionSummary, CodecMetrics, RuntimeDecision
from .utils import split_rows


@dataclass
class CandidateRecord:
    codec: HeadCodec
    metrics: CodecMetrics

    def as_dict(self) -> dict[str, Any]:
        return {"codec": self.codec.metadata(), "metrics": self.metrics.as_dict()}


@dataclass
class AdaptiveHead:
    codec: HeadCodec
    certificate: ConformalCertificate
    exact_keys: Tensor
    exact_values: Tensor
    scale: float
    tolerance: float
    dtype_bytes: int
    selected_metrics: CodecMetrics
    candidates: list[CodecMetrics] = field(default_factory=list)

    def evaluate(self, query: Tensor, *, exact_fallback: bool = True) -> RuntimeDecision:
        approximation = self.codec.evaluate(query)
        upper = float(self.certificate.upper(approximation.proxy).item())
        valid = bool(approximation.valid.item())
        use_fallback = bool(exact_fallback and (not valid or upper > self.tolerance))
        summary = (
            exact_head_summary(query, self.exact_keys, self.exact_values, self.scale)
            if use_fallback
            else approximation.summary
        )
        return RuntimeDecision(
            summary=summary,
            used_fallback=use_fallback,
            certificate_upper=upper,
            codec_name=self.codec.name,
            proxy=float(approximation.proxy.detach().float().item()),
        )

    @property
    def token_count(self) -> int:
        return int(self.exact_keys.shape[0])

    def active_bytes(self) -> int:
        # Five scalar certificate fields are retained for runtime and audit.
        return self.codec.active_bytes(self.dtype_bytes) + 5 * self.dtype_bytes

    def archive_bytes(self) -> int:
        return int((self.exact_keys.numel() + self.exact_values.numel()) * self.dtype_bytes)

    def report(self) -> dict[str, Any]:
        return {
            "selected": self.selected_metrics.as_dict(),
            "codec": self.codec.metadata(),
            "certificate": self.certificate.as_dict(),
            "tolerance": self.tolerance,
            "active_bytes": self.active_bytes(),
            "archive_bytes": self.archive_bytes(),
            "candidates": [item.as_dict() for item in self.candidates],
        }


def _augment_queries(
    keys: Tensor,
    queries: Tensor,
    *,
    minimum: int,
    maximum: int,
    seed: int,
) -> Tensor:
    """Construct a conservative query bank when deployment traces are sparse."""

    if queries.ndim != 2 or queries.shape[1] != keys.shape[1]:
        raise ValueError("queries must have shape [Q,D] matching keys")
    work = queries.detach()
    generator = torch.Generator(device=keys.device)
    generator.manual_seed(seed)
    if work.shape[0] > maximum:
        permutation = torch.randperm(
            work.shape[0], generator=generator, device=work.device
        )[:maximum]
        work = work.index_select(0, permutation)
    if work.shape[0] >= minimum:
        return work
    needed = minimum - int(work.shape[0])
    indices = torch.randint(
        keys.shape[0], (needed,), generator=generator, device=keys.device
    )
    base = keys.index_select(0, indices).float()
    scale = keys.float().std(dim=0, unbiased=False).mean().clamp_min(1e-3)
    noise = torch.randn(
        base.shape, generator=generator, device=base.device, dtype=torch.float32
    )
    synthetic = base + 0.05 * scale * noise
    return torch.cat([work, synthetic.to(keys.dtype)], dim=0)


def _combined_errors(
    approximation: AttentionSummary,
    exact: AttentionSummary,
    valid: Tensor,
    *,
    logmass_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    output_difference = approximation.output.float() - exact.output.float()
    output_relative = torch.linalg.vector_norm(output_difference, dim=-1) / torch.linalg.vector_norm(
        exact.output.float(), dim=-1
    ).clamp_min(1e-4)
    mass_relative = torch.abs(
        approximation.log_mass.float() - exact.log_mass.float()
    ) / (1.0 + torch.abs(exact.log_mass.float()))
    combined = output_relative + float(logmass_weight) * mass_relative
    combined = torch.where(valid, combined, torch.full_like(combined, 1e4))
    return combined, output_relative, mass_relative


def _measure_codec(
    codec: HeadCodec,
    queries: Tensor,
    exact: AttentionSummary,
    *,
    config: CompilerConfig,
    full_bytes: int,
    full_flops: int,
) -> CodecMetrics:
    result = codec.evaluate_batch(queries)
    combined, output_relative, _ = _combined_errors(
        result.summary,
        exact,
        result.valid,
        logmass_weight=config.logmass_weight,
    )
    difference = result.summary.output.float() - exact.output.float()
    output_nmse = difference.square().mean() / exact.output.float().square().mean().clamp_min(1e-8)
    logmass_rmse = torch.sqrt(
        (result.summary.log_mass.float() - exact.log_mass.float()).square().mean()
    )
    active_bytes = codec.active_bytes(config.dtype_bytes)
    byte_ratio = active_bytes / max(full_bytes, 1)
    estimated_flops = codec.estimated_flops()
    flop_ratio = estimated_flops / max(full_flops, 1)
    p95 = float(torch.quantile(combined, 0.95).item())
    invalid_rate = float((~result.valid).float().mean().item())
    budget_excess = max(0.0, byte_ratio - config.target_active_ratio)
    # The quadratic budget penalty makes a tiny accuracy improvement unable to
    # silently consume the entire cache.  Invalid numerical states are heavily
    # penalized because runtime fallback should be exceptional, not the norm.
    objective = (
        p95
        + config.rate_lambda * byte_ratio
        + config.latency_lambda * flop_ratio
        + 8.0 * budget_excess * budget_excess
        + 100.0 * invalid_rate
    )
    return CodecMetrics(
        name=codec.name,
        active_bytes=active_bytes,
        byte_ratio=float(byte_ratio),
        estimated_flops=estimated_flops,
        flop_ratio=float(flop_ratio),
        output_nmse=float(output_nmse.item()),
        output_relative_mean=float(output_relative.mean().item()),
        logmass_rmse=float(logmass_rmse.item()),
        combined_mean=float(combined.mean().item()),
        combined_p95=p95,
        combined_max=float(combined.max().item()),
        invalid_rate=invalid_rate,
        objective=float(objective),
        metadata=codec.metadata(),
    )


def _pareto_names(records: list[CandidateRecord]) -> set[str]:
    names: set[str] = set()
    for candidate in records:
        dominated = False
        for other in records:
            if other is candidate:
                continue
            weakly_better = (
                other.metrics.active_bytes <= candidate.metrics.active_bytes
                and other.metrics.combined_p95 <= candidate.metrics.combined_p95
                and other.metrics.estimated_flops <= candidate.metrics.estimated_flops
            )
            strictly_better = (
                other.metrics.active_bytes < candidate.metrics.active_bytes
                or other.metrics.combined_p95 < candidate.metrics.combined_p95
                or other.metrics.estimated_flops < candidate.metrics.estimated_flops
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            names.add(candidate.codec.name)
    return names


def compile_adaptive_head(
    keys: Tensor,
    values: Tensor,
    query_bank: Tensor,
    *,
    scale: float,
    config: CompilerConfig,
    seed: int = 0,
) -> AdaptiveHead:
    """Search and calibrate a heterogeneous representation for one page/head."""

    config.validate()
    if keys.ndim != 2 or values.shape != keys.shape or keys.shape[0] < 2:
        raise ValueError("keys/values must have matching shape [N,D], N>=2")
    queries = _augment_queries(
        keys,
        query_bank,
        minimum=config.min_queries,
        maximum=config.max_queries,
        seed=seed,
    )
    fit_queries, selection_queries, calibration_queries = split_rows(
        queries, (0.40, 0.30, 0.30), seed=seed
    )
    full_bytes = int(2 * keys.numel() * config.dtype_bytes)
    full_flops = int(4 * keys.shape[0] * keys.shape[1])
    exact_selection = exact_head_batch(selection_queries, keys, values, scale)

    factories: list[Callable[[], HeadCodec]] = []
    dimension = int(keys.shape[1])
    tokens = int(keys.shape[0])
    for patches in sorted(set(config.operator_patches)):
        if patches > fit_queries.shape[0]:
            continue
        for rank in sorted(set(config.operator_ranks)):
            if rank <= dimension:
                factories.append(
                    lambda patches=patches, rank=rank: OperatorCodec.compile(
                        keys,
                        values,
                        fit_queries,
                        patches=patches,
                        rank=rank,
                        scale=scale,
                        seed=seed + 101 * patches + rank,
                        iterations=config.kmeans_iterations,
                    )
                )
    for slots in sorted(set(config.coreset_slots)):
        if slots < tokens:
            factories.append(
                lambda slots=slots: WeightedCoresetCodec.compile(
                    keys,
                    values,
                    slots=slots,
                    scale=scale,
                    seed=seed + 2000 + slots,
                    iterations=config.kmeans_iterations,
                    queries=fit_queries,
                    restarts=config.coreset_restarts,
                )
            )
    for clusters in sorted(set(config.gaussian_clusters)):
        if clusters >= tokens:
            continue
        for rank in sorted(set(config.gaussian_ranks)):
            if rank <= dimension:
                factories.append(
                    lambda clusters=clusters, rank=rank: GaussianMixtureCodec.compile(
                        keys,
                        values,
                        clusters=clusters,
                        rank=rank,
                        scale=scale,
                        seed=seed + 3000 + clusters * 101 + rank,
                        iterations=config.kmeans_iterations,
                    )
                )
    for rank in sorted(set(config.moment_ranks)):
        if rank <= dimension:
            factories.append(
                lambda rank=rank: MomentCodec.compile(
                    keys, values, rank=rank, scale=scale
                )
            )
    hybrid_clusters = min(max(2, tokens // 16), 4)
    hybrid_rank = min(4, dimension)
    for exact_slots in sorted(set(config.hybrid_exact_slots)):
        if exact_slots < tokens - 1:
            factories.append(
                lambda exact_slots=exact_slots: HybridResidualCodec.compile_gaussian(
                    keys,
                    values,
                    fit_queries,
                    exact_slots=exact_slots,
                    clusters=hybrid_clusters,
                    rank=hybrid_rank,
                    scale=scale,
                    seed=seed + 4000 + exact_slots,
                    iterations=config.kmeans_iterations,
                )
            )

    records: list[CandidateRecord] = []
    failures: list[str] = []
    seen: set[str] = set()
    for factory in factories:
        try:
            codec = factory()
            if codec.name in seen:
                continue
            seen.add(codec.name)
            records.append(
                CandidateRecord(
                    codec,
                    _measure_codec(
                        codec,
                        selection_queries,
                        exact_selection,
                        config=config,
                        full_bytes=full_bytes,
                        full_flops=full_flops,
                    ),
                )
            )
        except (RuntimeError, ValueError, torch.linalg.LinAlgError) as error:
            failures.append(f"{type(error).__name__}: {error}")

    if not records:
        exact_codec = ExactCodec(keys, values, scale)
        records.append(
            CandidateRecord(
                exact_codec,
                _measure_codec(
                    exact_codec,
                    selection_queries,
                    exact_selection,
                    config=config,
                    full_bytes=full_bytes,
                    full_flops=full_flops,
                ),
            )
        )

    budget = full_bytes * config.target_active_ratio
    feasible = [item for item in records if item.metrics.active_bytes <= budget]
    pool = feasible if feasible else records
    selected = min(pool, key=lambda item: (item.metrics.objective, item.metrics.active_bytes))

    exact_calibration = exact_head_batch(calibration_queries, keys, values, scale)
    calibrated = selected.codec.evaluate_batch(calibration_queries)
    calibration_errors, _, _ = _combined_errors(
        calibrated.summary,
        exact_calibration,
        calibrated.valid,
        logmass_weight=config.logmass_weight,
    )
    certificate = ConformalCertificate.fit(
        calibration_errors,
        calibrated.proxy,
        alpha=config.certificate_alpha,
    )

    pareto = _pareto_names(records)
    audited_metrics: list[CodecMetrics] = []
    for record in records:
        metadata = dict(record.metrics.metadata)
        metadata["pareto"] = record.codec.name in pareto
        if failures:
            metadata["compiler_failures"] = failures
        audited_metrics.append(
            CodecMetrics(
                name=record.metrics.name,
                active_bytes=record.metrics.active_bytes,
                byte_ratio=record.metrics.byte_ratio,
                estimated_flops=record.metrics.estimated_flops,
                flop_ratio=record.metrics.flop_ratio,
                output_nmse=record.metrics.output_nmse,
                output_relative_mean=record.metrics.output_relative_mean,
                logmass_rmse=record.metrics.logmass_rmse,
                combined_mean=record.metrics.combined_mean,
                combined_p95=record.metrics.combined_p95,
                combined_max=record.metrics.combined_max,
                invalid_rate=record.metrics.invalid_rate,
                objective=record.metrics.objective,
                metadata=metadata,
            )
        )
    selected_metrics = next(item for item in audited_metrics if item.name == selected.codec.name)
    return AdaptiveHead(
        codec=selected.codec,
        certificate=certificate,
        exact_keys=keys.detach().contiguous(),
        exact_values=values.detach().contiguous(),
        scale=float(scale),
        tolerance=float(config.certificate_tolerance),
        dtype_bytes=int(config.dtype_bytes),
        selected_metrics=selected_metrics,
        candidates=sorted(audited_metrics, key=lambda item: item.objective),
    )
