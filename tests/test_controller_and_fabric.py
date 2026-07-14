from __future__ import annotations

import torch

from focus_fabric.config import CompilerConfig, FabricConfig
from focus_fabric.exact import exact_multihead_summary
from focus_fabric.fabric import MemoryFabricLayer
from focus_fabric.metrics import exact_multihead_batch, summary_metrics
from focus_fabric.page import FabricPage
from focus_fabric.synthetic import make_heterogeneous_case
from focus_fabric.types import AttentionSummary, FabricStats


def small_compiler(tolerance: float = 0.2) -> CompilerConfig:
    return CompilerConfig(
        operator_patches=(2,),
        operator_ranks=(2, 4),
        coreset_slots=(4, 8),
        gaussian_clusters=(2, 4),
        gaussian_ranks=(2, 4),
        moment_ranks=(2, 4),
        hybrid_exact_slots=(2, 4),
        min_queries=30,
        max_queries=72,
        kmeans_iterations=5,
        certificate_tolerance=tolerance,
        target_active_ratio=0.45,
    )


def evaluate_page(page: FabricPage, queries: torch.Tensor, fallback: bool) -> tuple[AttentionSummary, FabricStats]:
    outputs, masses = [], []
    stats = FabricStats()
    for index in range(queries.shape[1]):
        item = page.evaluate(queries[:, index], exact_fallback=fallback, stats=stats)
        outputs.append(item.output)
        masses.append(item.log_mass)
    return AttentionSummary(torch.stack(outputs, 1), torch.stack(masses, 1)), stats


def test_controller_produces_finite_heterogeneous_page() -> None:
    case = make_heterogeneous_case(
        tokens=96,
        dimension=12,
        train_queries=72,
        test_queries=40,
        ood_queries=16,
        seed=7,
    )
    page = FabricPage.compile(
        case.keys,
        case.values,
        case.query_train,
        start=0,
        end=96,
        level=0,
        scale=case.scale,
        config=small_compiler(),
        seed=3,
    )
    approximation, _ = evaluate_page(page, case.query_test, False)
    exact = exact_multihead_batch(case.query_test, case.keys, case.values, case.scale)
    metrics = summary_metrics(approximation, exact)
    assert torch.isfinite(approximation.output).all()
    assert torch.isfinite(approximation.log_mass).all()
    assert metrics["output_nmse"] < 0.25
    assert all(
        head.selected_metrics.objective
        <= min(candidate.objective for candidate in head.candidates) + 1e-12
        for head in page.heads
    )


def test_zero_tolerance_fallback_recovers_exact_page() -> None:
    case = make_heterogeneous_case(
        tokens=64,
        dimension=8,
        train_queries=48,
        test_queries=8,
        ood_queries=8,
        seed=9,
    )
    page = FabricPage.compile(
        case.keys,
        case.values,
        case.query_train,
        start=0,
        end=64,
        level=0,
        scale=case.scale,
        config=small_compiler(1e-12),
        seed=4,
    )
    query = case.query_ood[:, 0]
    stats = FabricStats()
    guarded = page.evaluate(query, exact_fallback=True, stats=stats)
    exact = exact_multihead_summary(query, case.keys, case.values, case.scale)
    assert stats.fallback_decisions == case.keys.shape[0]
    assert torch.allclose(guarded.output, exact.output, atol=2e-6, rtol=2e-6)
    assert torch.allclose(guarded.log_mass, exact.log_mass, atol=2e-6, rtol=2e-6)


def test_binary_counter_repeated_compaction() -> None:
    torch.manual_seed(3)
    config = FabricConfig(
        n_heads=2,
        head_dim=8,
        hot_window=8,
        page_size=8,
        query_bank_size=32,
        compiler=small_compiler(),
    )
    fabric = MemoryFabricLayer.create(config)
    for _ in range(48):
        query, key, value = torch.randn(3, 2, 8)
        fabric.append_and_attend(query, key, value)
    report = fabric.report(include_pages=False)
    assert report["pages_merged"] > 0
    assert len(report["page_levels"]) == len(set(report["page_levels"]))
    assert report["tokens"] == 48
