#!/usr/bin/env python3
"""Reproducible CPU evidence suite for the FOCUS-Fabric release.

Evidence is partitioned into controlled attention fields, Q/K/V traces from the
bundled learned mechanism checkpoint, repeated online compaction, and
end-to-end sequential logit/token agreement.  CUDA and official benchmark
fields remain null when they are not actually measured.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def public_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT.resolve()))
    except (ValueError, OSError):
        return candidate.name if candidate.is_absolute() else str(candidate)


def public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(metadata)
    if "checkpoint" in sanitized:
        sanitized["checkpoint"] = public_path(str(sanitized["checkpoint"]))
    return sanitized

from focus_fabric.baselines import (
    compile_coreset_matched,
    compile_operator_only_matched,
)
from focus_fabric.config import CompilerConfig, FabricConfig
from focus_fabric.exact import exact_multihead_summary
from focus_fabric.fabric import MemoryFabricLayer
from focus_fabric.integration import generate_fabric, sequential_logits_fabric
from focus_fabric.metrics import exact_multihead_batch, summary_metrics, token_agreement
from focus_fabric.page import FabricPage
from focus_fabric.synthetic import make_heterogeneous_case
from focus_fabric.types import AttentionSummary, FabricStats
from focus_native import CacheConfig
from focus_native.generation import generate, sequential_logits
from focus_native.io import load_checkpoint


def compiler_config(*, tolerance: float = 0.18, fast: bool = False) -> CompilerConfig:
    return CompilerConfig(
        operator_patches=(2,) if fast else (2, 4),
        operator_ranks=(2, 4),
        coreset_slots=(4, 8) if fast else (4, 8, 16),
        gaussian_clusters=(2, 4),
        gaussian_ranks=(2, 4),
        moment_ranks=(2, 4),
        hybrid_exact_slots=(2, 4) if fast else (2, 4, 8),
        target_active_ratio=0.35 if fast else 0.30,
        rate_lambda=0.15,
        latency_lambda=0.04,
        certificate_alpha=0.05,
        certificate_tolerance=tolerance,
        min_queries=32 if fast else 48,
        max_queries=96 if fast else 144,
        kmeans_iterations=7 if fast else 10,
        seed=17,
    )


def evaluate_page(
    page: FabricPage, queries: Tensor, *, fallback: bool
) -> tuple[AttentionSummary, FabricStats]:
    outputs: list[Tensor] = []
    masses: list[Tensor] = []
    stats = FabricStats()
    for index in range(queries.shape[1]):
        summary = page.evaluate(
            queries[:, index], exact_fallback=fallback, stats=stats
        )
        outputs.append(summary.output)
        masses.append(summary.log_mass)
    return (
        AttentionSummary(torch.stack(outputs, 1), torch.stack(masses, 1)),
        stats,
    )


def certificate_coverage(
    page: FabricPage,
    queries: Tensor,
    exact: AttentionSummary,
    logmass_weight: float,
) -> dict[str, float]:
    covered = total = 0
    upper_values: list[float] = []
    error_values: list[float] = []
    for head_index, head in enumerate(page.heads):
        approximation = head.codec.evaluate_batch(queries[head_index])
        output_error = torch.linalg.vector_norm(
            approximation.summary.output.float() - exact.output[head_index].float(),
            dim=-1,
        ) / torch.linalg.vector_norm(
            exact.output[head_index].float(), dim=-1
        ).clamp_min(1e-4)
        mass_error = torch.abs(
            approximation.summary.log_mass.float()
            - exact.log_mass[head_index].float()
        ) / (1.0 + torch.abs(exact.log_mass[head_index].float()))
        error = output_error + logmass_weight * mass_error
        upper = head.certificate.upper(approximation.proxy)
        covered += int(((error <= upper) & approximation.valid).sum().item())
        total += int(error.numel())
        upper_values.extend(float(item) for item in upper)
        error_values.extend(float(item) for item in error)
    return {
        "coverage": covered / max(total, 1),
        "target_marginal_coverage": 1.0 - page.heads[0].certificate.alpha,
        "mean_upper": sum(upper_values) / max(len(upper_values), 1),
        "mean_error": sum(error_values) / max(len(error_values), 1),
    }


def median_seconds(function, repeats: int = 5) -> float:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        values.append(time.perf_counter() - started)
    return sorted(values)[len(values) // 2]


def synthetic_benchmark() -> dict[str, Any]:
    case = make_heterogeneous_case(
        tokens=192,
        dimension=16,
        train_queries=144,
        test_queries=128,
        ood_queries=48,
        seed=7,
    )
    config = compiler_config()
    started = time.perf_counter()
    page = FabricPage.compile(
        case.keys,
        case.values,
        case.query_train,
        start=0,
        end=case.keys.shape[1],
        level=0,
        scale=case.scale,
        config=config,
        seed=3,
    )
    compile_seconds = time.perf_counter() - started
    target_bytes = [head.active_bytes() for head in page.heads]
    operator = compile_operator_only_matched(
        case.keys,
        case.values,
        case.query_train,
        target_bytes_by_head=target_bytes,
        scale=case.scale,
        iterations=7,
    )
    coreset = compile_coreset_matched(
        case.keys,
        case.values,
        queries=case.query_train,
        target_bytes_by_head=target_bytes,
        scale=case.scale,
        iterations=7,
    )
    splits: dict[str, Any] = {}
    for split_name, queries in (
        ("in_distribution", case.query_test),
        ("distribution_shift", case.query_ood),
    ):
        exact = exact_multihead_batch(queries, case.keys, case.values, case.scale)
        approximate, raw_stats = evaluate_page(page, queries, fallback=False)
        guarded, guarded_stats = evaluate_page(page, queries, fallback=True)
        splits[split_name] = {
            "fabric_approx": {
                **summary_metrics(approximate, exact),
                **raw_stats.as_dict(),
            },
            "fabric_guarded": {
                **summary_metrics(guarded, exact),
                **guarded_stats.as_dict(),
            },
            "operator_only_memory_matched": summary_metrics(
                operator.evaluate_batch(queries), exact
            ),
            "coreset_memory_matched": summary_metrics(
                coreset.evaluate_batch(queries), exact
            ),
            "certificate": certificate_coverage(
                page, queries, exact, config.logmass_weight
            ),
        }
    full_bytes = int(2 * case.keys.numel() * config.dtype_bytes)
    return {
        "case": {
            "heads": case.keys.shape[0],
            "tokens": case.keys.shape[1],
            "dimension": case.keys.shape[2],
            "regimes": list(case.regimes),
        },
        "compile_seconds": compile_seconds,
        "memory": {
            "full_exact_bytes": full_bytes,
            "fabric_active_bytes": page.active_bytes(),
            "fabric_compression": full_bytes / page.active_bytes(),
            "operator_active_bytes": operator.active_bytes(),
            "coreset_active_bytes": coreset.active_bytes(),
        },
        "selected_codecs": [head.codec.name for head in page.heads],
        "selected_head_reports": [head.report() for head in page.heads],
        "splits": splits,
        "cpu_batch_median_seconds": {
            "full_exact": median_seconds(
                lambda: exact_multihead_batch(
                    case.query_test, case.keys, case.values, case.scale
                )
            ),
            "fabric_python_reference": median_seconds(
                lambda: evaluate_page(page, case.query_test, fallback=False)
            ),
            "operator_only": median_seconds(
                lambda: operator.evaluate_batch(case.query_test)
            ),
            "coreset": median_seconds(
                lambda: coreset.evaluate_batch(case.query_test)
            ),
        },
    }


def learned_focus_summary(
    model,
    layer_index: int,
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
) -> tuple[AttentionSummary, int]:
    attention = model.layers[layer_index].attn
    compiled = attention.compile_page(keys.unsqueeze(0), values.unsqueeze(0))
    output, mass, _, _ = attention.evaluate_page(queries.unsqueeze(0), compiled)
    active = sum(
        int(value.numel())
        for name, value in compiled.items()
        if name != "tail_ratio"
    ) * keys.element_size()
    return AttentionSummary(output[0], mass[0]), active


def checkpoint_trace_benchmark(checkpoint: Path) -> dict[str, Any]:
    model, _, metadata = load_checkpoint(checkpoint)
    model.eval()
    generator = torch.Generator().manual_seed(20260714)
    ids = torch.randint(
        0, model.config.vocab_size, (3, 192), generator=generator
    )
    with torch.no_grad():
        traces = model(ids, return_traces=True).traces
    assert traces is not None
    config = compiler_config(tolerance=0.10, fast=True)
    layers: dict[str, Any] = {}
    for layer_index in (0, model.config.n_layers - 1):
        trace = traces[layer_index]
        keys = trace["k"][0, :, :96].contiguous()
        values = trace["v"][0, :, :96].contiguous()
        query_train = torch.cat(
            [
                trace["q"][0, :, 96:144],
                trace["q"][1, :, 96:144],
            ],
            dim=1,
        ).contiguous()
        query_test = trace["q"][0, :, 144:192].contiguous()
        query_shift = (trace["q"][2, :, 144:192] * 2.5).contiguous()
        started = time.perf_counter()
        page = FabricPage.compile(
            keys,
            values,
            query_train,
            start=0,
            end=96,
            level=0,
            scale=model.config.head_dim ** -0.5,
            config=config,
            seed=41 + layer_index,
        )
        compile_seconds = time.perf_counter() - started
        target = [head.active_bytes() for head in page.heads]
        operator = compile_operator_only_matched(
            keys,
            values,
            query_train,
            target_bytes_by_head=target,
            scale=model.config.head_dim ** -0.5,
            iterations=6,
            seed=99 + layer_index,
        )
        coreset = compile_coreset_matched(
            keys,
            values,
            queries=query_train,
            target_bytes_by_head=target,
            scale=model.config.head_dim ** -0.5,
            iterations=6,
            seed=199 + layer_index,
        )
        split_payload: dict[str, Any] = {}
        for name, queries in (
            ("future_trace", query_test),
            ("scaled_shift", query_shift),
        ):
            exact = exact_multihead_batch(
                queries, keys, values, model.config.head_dim ** -0.5
            )
            approximate, _ = evaluate_page(page, queries, fallback=False)
            guarded, stats = evaluate_page(page, queries, fallback=True)
            learned, learned_bytes = learned_focus_summary(
                model, layer_index, keys, values, queries
            )
            split_payload[name] = {
                "fabric_approx": summary_metrics(approximate, exact),
                "fabric_guarded": {
                    **summary_metrics(guarded, exact),
                    **stats.as_dict(),
                },
                "legacy_learned_focus": summary_metrics(learned, exact),
                "operator_only_memory_matched": summary_metrics(
                    operator.evaluate_batch(queries), exact
                ),
                "coreset_memory_matched": summary_metrics(
                    coreset.evaluate_batch(queries), exact
                ),
                "certificate": certificate_coverage(
                    page, queries, exact, config.logmass_weight
                ),
                "legacy_focus_active_bytes": learned_bytes,
            }
        full_bytes = int(2 * keys.numel() * keys.element_size())
        layers[str(layer_index)] = {
            "compile_seconds": compile_seconds,
            "selected_codecs": [head.codec.name for head in page.heads],
            "memory": {
                "full_exact_bytes": full_bytes,
                "fabric_active_bytes": page.active_bytes(),
                "fabric_compression": full_bytes / page.active_bytes(),
                "operator_active_bytes": operator.active_bytes(),
                "coreset_active_bytes": coreset.active_bytes(),
            },
            "splits": split_payload,
        }
    return {
        "checkpoint": public_path(checkpoint),
        "metadata": public_metadata(metadata),
        "trace_source": (
            "bundled learned one-million-parameter symbolic mechanism checkpoint; "
            "random token IDs because the original symbolic tokenizer was not preserved"
        ),
        "layers": layers,
    }


def repeated_compaction_benchmark() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(8080)
    heads, dimension, tokens = 2, 12, 128
    config = FabricConfig(
        n_heads=heads,
        head_dim=dimension,
        hot_window=16,
        page_size=16,
        query_bank_size=48,
        compiler=compiler_config(tolerance=0.10, fast=True),
    )
    fabric = MemoryFabricLayer.create(config)
    keys: list[Tensor] = []
    values: list[Tensor] = []
    relative_errors: list[float] = []
    checkpoints: list[dict[str, Any]] = []
    projection = torch.randn(
        heads, 4, dimension, generator=generator
    ) * 0.15
    for position in range(tokens):
        latent = torch.randn(heads, 4, generator=generator)
        key = torch.einsum("hr,hrd->hd", latent, projection)
        value = torch.tanh(
            torch.randn(heads, dimension, generator=generator) + 0.2 * key
        )
        query = key + 0.1 * torch.randn(
            heads, dimension, generator=generator
        )
        keys.append(key)
        values.append(value)
        approximate = fabric.append_and_attend(query, key, value)
        exact = exact_multihead_summary(
            query,
            torch.stack(keys, 1),
            torch.stack(values, 1),
            dimension ** -0.5,
        ).output
        relative = torch.linalg.vector_norm(
            approximate - exact, dim=-1
        ) / torch.linalg.vector_norm(exact, dim=-1).clamp_min(1e-5)
        relative_errors.append(float(relative.mean().item()))
        if position + 1 in {32, 64, 96, 128}:
            checkpoints.append(fabric.report(include_pages=False))
    report = fabric.report(include_pages=False)
    return {
        "tokens": tokens,
        "mean_relative_attention_error": sum(relative_errors) / len(relative_errors),
        "p95_relative_attention_error": float(
            torch.quantile(torch.tensor(relative_errors), 0.95).item()
        ),
        "max_relative_attention_error": max(relative_errors),
        "final": report,
        "checkpoints": checkpoints,
    }


def end_to_end_benchmark(checkpoint: Path) -> dict[str, Any]:
    model, _, metadata = load_checkpoint(checkpoint)
    model.eval()
    generator = torch.Generator().manual_seed(424242)
    token_ids = torch.randint(
        0, model.config.vocab_size, (64,), generator=generator
    ).tolist()
    exact_logits, exact_report = sequential_logits(
        model, token_ids, CacheConfig(mode="exact")
    )
    config = FabricConfig(
        n_heads=model.config.n_heads,
        head_dim=model.config.head_dim,
        hot_window=16,
        page_size=16,
        query_bank_size=48,
        compiler=compiler_config(tolerance=0.08, fast=True),
    )
    started = time.perf_counter()
    fabric_logits, fabric_report = sequential_logits_fabric(
        model, token_ids, config
    )
    fabric_seconds = time.perf_counter() - started
    prompt = token_ids[:48]
    exact_generation = generate(
        model,
        prompt,
        CacheConfig(mode="exact"),
        max_new_tokens=8,
        eos_id=None,
    )
    fabric_generation = generate_fabric(
        model,
        prompt,
        config,
        max_new_tokens=8,
        eos_id=None,
    )
    exact_new = exact_generation.token_ids[len(prompt) :]
    fabric_new = fabric_generation.token_ids[len(prompt) :]
    return {
        "checkpoint": public_path(checkpoint),
        "metadata": public_metadata(metadata),
        "teacher_forced": {
            **token_agreement(fabric_logits, exact_logits),
            "tokens": len(token_ids),
            "fabric_seconds": fabric_seconds,
            "exact_cache": exact_report,
            "fabric_cache": fabric_report,
        },
        "free_running": {
            "generated_tokens": 8,
            "sequence_agreement": exact_new == fabric_new,
            "token_agreement": sum(
                left == right for left, right in zip(exact_new, fabric_new)
            )
            / 8,
            "exact_tokens": exact_new,
            "fabric_tokens": fabric_new,
            "exact_elapsed_seconds": exact_generation.elapsed_seconds,
            "fabric_elapsed_seconds": fabric_generation.elapsed_seconds,
            "fabric_cache": fabric_generation.cache_report,
        },
        "tokenizer_note": (
            "token IDs are used; the legacy checkpoint's original symbolic "
            "tokenizer was not preserved"
        ),
    }


def flatten_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, methods in payload["synthetic"]["splits"].items():
        for method, metrics in methods.items():
            if method == "certificate":
                continue
            row: dict[str, Any] = {
                "section": "synthetic",
                "split": split,
                "method": method,
            }
            row.update(
                {
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }
            )
            rows.append(row)
    for layer, layer_payload in payload["checkpoint_trace"]["layers"].items():
        for split, methods in layer_payload["splits"].items():
            for method, metrics in methods.items():
                if method in {"certificate", "legacy_focus_active_bytes"}:
                    continue
                row = {
                    "section": f"checkpoint_layer_{layer}",
                    "split": split,
                    "method": method,
                }
                row.update(
                    {
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    }
                )
                rows.append(row)
    return rows


def write_plot(payload: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    methods = [
        "fabric_approx",
        "operator_only_memory_matched",
        "coreset_memory_matched",
    ]
    labels = ["FOCUS-Fabric", "Operator-only", "KV coreset"]
    values = [
        payload["synthetic"]["splits"]["in_distribution"][method][
            "output_nmse"
        ]
        for method in methods
    ]
    figure = plt.figure(figsize=(8, 4.5))
    axis = figure.add_subplot(111)
    axis.bar(labels, values)
    axis.set_yscale("log")
    axis.set_ylabel("Attention output NMSE (log scale)")
    axis.set_title("Controlled heterogeneous attention, memory-matched")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "fabric_benchmark.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "focus-native-small",
    )
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "threads": args.threads,
        },
        "synthetic": synthetic_benchmark(),
        "checkpoint_trace": checkpoint_trace_benchmark(args.checkpoint),
        "repeated_compaction": repeated_compaction_benchmark(),
        "end_to_end": end_to_end_benchmark(args.checkpoint),
        "official_benchmarks": {
            "LongBench": None,
            "RULER": None,
            "BABILong": None,
            "LifeBench": None,
            "status": "not executed in this network-isolated CPU environment",
        },
        "gpu": {
            "kernel_correctness": None,
            "latency": None,
            "physical_hbm_bandwidth": None,
            "status": "not executed because CUDA/Triton are unavailable",
        },
    }
    payload["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = flatten_rows(payload)
    csv_path = args.output.with_suffix(".csv")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_plot(payload, args.output.with_suffix(".png"))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "elapsed_seconds": payload["elapsed_seconds"],
                "synthetic_selected": payload["synthetic"]["selected_codecs"],
                "synthetic_fabric_nmse": payload["synthetic"]["splits"][
                    "in_distribution"
                ]["fabric_approx"]["output_nmse"],
                "synthetic_operator_nmse": payload["synthetic"]["splits"][
                    "in_distribution"
                ]["operator_only_memory_matched"]["output_nmse"],
                "end_to_end_token_agreement": payload["end_to_end"][
                    "teacher_forced"
                ]["argmax_token_agreement"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
