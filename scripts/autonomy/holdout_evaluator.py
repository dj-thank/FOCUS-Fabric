#!/usr/bin/env python3
"""Out-of-worktree randomized holdout evaluator for autonomous candidates.

The orchestrator invokes this file from the trusted root repository while
loading candidate code from an explicit source directory.  The seed is chosen
after the Codex run finishes, preventing direct tuning to the exact holdout
instances.  This is an anti-overfitting control, not a cryptographic sandbox:
a malicious candidate could still detect the evaluator at runtime, so external
human/reviewer validation remains required before publication.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


def load_candidate(source: Path) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"candidate source directory does not exist: {source}")
    sys.path.insert(0, str(source))


def evaluate(seed: int, cases: int) -> dict[str, Any]:
    from focus_fabric.baselines import compile_coreset_matched, compile_operator_only_matched
    from focus_fabric.config import CompilerConfig
    from focus_fabric.metrics import exact_multihead_batch, summary_metrics
    from focus_fabric.page import FabricPage
    from focus_fabric.synthetic import make_heterogeneous_case
    from focus_fabric.types import AttentionSummary, FabricStats

    compiler = CompilerConfig(
        operator_patches=(2, 4, 8),
        operator_ranks=(2, 4, 8),
        coreset_slots=(4, 8, 12, 16),
        gaussian_clusters=(2, 4),
        gaussian_ranks=(2, 4, 8),
        moment_ranks=(2, 4, 6, 8),
        hybrid_exact_slots=(2, 4),
        target_active_ratio=0.40,
        min_queries=36,
        max_queries=80,
        kmeans_iterations=6,
        certificate_tolerance=0.16,
        seed=seed,
    )
    torch.set_num_threads(1)
    results: list[dict[str, Any]] = []
    for index in range(cases):
        case_seed = (seed + 104729 * index) % (2**31 - 1)
        dimension = 8 + 2 * (index % 3)
        tokens = 72 + 8 * (index % 4)
        case = make_heterogeneous_case(
            tokens=tokens,
            dimension=dimension,
            train_queries=64,
            test_queries=40,
            ood_queries=20,
            seed=case_seed,
        )
        page = FabricPage.compile(
            case.keys,
            case.values,
            case.query_train,
            start=0,
            end=tokens,
            level=0,
            scale=case.scale,
            config=compiler,
            seed=case_seed ^ 0x5A5A5A,
        )
        target = [head.active_bytes() for head in page.heads]
        operator = compile_operator_only_matched(
            case.keys,
            case.values,
            case.query_train,
            target_bytes_by_head=target,
            scale=case.scale,
            seed=case_seed + 1,
            iterations=5,
        )
        coreset = compile_coreset_matched(
            case.keys,
            case.values,
            queries=case.query_train,
            target_bytes_by_head=target,
            scale=case.scale,
            seed=case_seed + 2,
            iterations=5,
        )

        outputs, masses = [], []
        stats = FabricStats()
        for query_index in range(case.query_test.shape[1]):
            summary = page.evaluate(
                case.query_test[:, query_index], exact_fallback=False, stats=stats
            )
            outputs.append(summary.output)
            masses.append(summary.log_mass)
        approximation = AttentionSummary(torch.stack(outputs, 1), torch.stack(masses, 1))
        exact = exact_multihead_batch(case.query_test, case.keys, case.values, case.scale)
        fabric_metrics = summary_metrics(approximation, exact)
        operator_metrics = summary_metrics(operator.evaluate_batch(case.query_test), exact)
        coreset_metrics = summary_metrics(coreset.evaluate_batch(case.query_test), exact)

        # Independently force the public exact verifier path on a shifted query.
        original_tolerances = [head.tolerance for head in page.heads]
        for head in page.heads:
            head.tolerance = 0.0
        guarded = page.evaluate(case.query_ood[:, 0], exact_fallback=True)
        exact_guarded = exact_multihead_batch(
            case.query_ood[:, :1], case.keys, case.values, case.scale
        )
        fallback_max_error = float(
            (guarded.output - exact_guarded.output[:, 0]).abs().max().item()
        )
        for head, tolerance in zip(page.heads, original_tolerances):
            head.tolerance = tolerance

        full_bytes = int(2 * case.keys.numel() * compiler.dtype_bytes)
        results.append(
            {
                "case": index,
                "seed": case_seed,
                "tokens": tokens,
                "dimension": dimension,
                "fabric_nmse": fabric_metrics["output_nmse"],
                "operator_nmse": operator_metrics["output_nmse"],
                "coreset_nmse": coreset_metrics["output_nmse"],
                "best_baseline_nmse": min(
                    operator_metrics["output_nmse"], coreset_metrics["output_nmse"]
                ),
                "active_ratio": page.active_bytes() / full_bytes,
                "fallback_max_abs_error": fallback_max_error,
                "invalid_outputs": stats.invalid_codec_outputs,
                "selected_codecs": [head.codec.name for head in page.heads],
            }
        )
    mean_fabric = sum(item["fabric_nmse"] for item in results) / len(results)
    mean_baseline = sum(item["best_baseline_nmse"] for item in results) / len(results)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "cases": cases,
        "mean_fabric_nmse": mean_fabric,
        "mean_best_single_family_nmse": mean_baseline,
        "relative_nmse": mean_fabric / max(mean_baseline, 1e-12),
        "max_fallback_abs_error": max(item["fallback_max_abs_error"] for item in results),
        "invalid_outputs": sum(item["invalid_outputs"] for item in results),
        "mean_active_ratio": sum(item["active_ratio"] for item in results) / len(results),
        "objective": mean_fabric + 0.04 * (sum(item["active_ratio"] for item in results) / len(results)),
        "single_family_reference": {
            "mean_best_nmse": mean_baseline,
            "fabric_to_best_ratio": mean_fabric / max(mean_baseline, 1e-12),
            "note": "Diagnostic only: heterogeneous compilation is not required to dominate every random field."
        },
        "passed": (
            max(item["fallback_max_abs_error"] for item in results) < 1e-5
            and sum(item["invalid_outputs"] for item in results) == 0
        ),
        "details": results,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_candidate(args.source)
    payload = evaluate(args.seed, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "details"}, indent=2))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
