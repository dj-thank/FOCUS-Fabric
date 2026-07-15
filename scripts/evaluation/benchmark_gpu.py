#!/usr/bin/env python3
"""CUDA/Triton correctness and latency harness; never fabricates GPU numbers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from focus_fabric.codecs import GaussianMixtureCodec
from focus_fabric.kernels.gaussian_eval import TRITON_AVAILABLE, gaussian_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "gpu_benchmark.json",
    )
    parser.add_argument("--groups", type=int, default=256)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not TRITON_AVAILABLE:
        payload = {
            "status": "not_executed",
            "reason": "CUDA or Triton is unavailable",
            "cuda_available": torch.cuda.is_available(),
            "triton_available": TRITON_AVAILABLE,
            "kernel_correctness": None,
            "latency_ms": None,
            "physical_hbm_bandwidth_gb_s": None,
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(17)
    g, c, d, r = args.groups, args.clusters, args.dimension, args.rank
    query = torch.randn(g, d, generator=generator, device=device, dtype=torch.float16)
    mean_keys = torch.randn(g, c, d, generator=generator, device=device, dtype=torch.float16)
    mean_values = torch.randn_like(mean_keys)
    log_counts = torch.randn(g, c, generator=generator, device=device, dtype=torch.float16)
    basis = torch.randn(g, c, d, r, generator=generator, device=device, dtype=torch.float16)
    variances = torch.rand(g, c, r, generator=generator, device=device, dtype=torch.float16)
    cross = torch.randn_like(basis)
    scale = d**-0.5
    reference = gaussian_eval(
        query, mean_keys, mean_values, log_counts, basis, variances, cross, scale, use_triton=False
    )
    candidate = gaussian_eval(
        query, mean_keys, mean_values, log_counts, basis, variances, cross, scale, use_triton=True
    )
    max_output_error = float((candidate[0] - reference[0]).abs().max().item())
    max_mass_error = float((candidate[1] - reference[1]).abs().max().item())
    for _ in range(args.warmup):
        gaussian_eval(query, mean_keys, mean_values, log_counts, basis, variances, cross, scale)
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(args.repeats):
        gaussian_eval(query, mean_keys, mean_values, log_counts, basis, variances, cross, scale)
    ended.record()
    torch.cuda.synchronize()
    milliseconds = started.elapsed_time(ended) / args.repeats
    # This is an algorithmic byte estimate, not a physical profiler counter.
    estimated_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (query, mean_keys, mean_values, log_counts, basis, variances, cross)
    ) + candidate[0].numel() * candidate[0].element_size() + candidate[1].numel() * candidate[1].element_size()
    payload = {
        "status": "executed",
        "device": torch.cuda.get_device_name(),
        "shapes": {"groups": g, "clusters": c, "dimension": d, "rank": r},
        "kernel_correctness": {
            "max_output_abs_error": max_output_error,
            "max_logmass_abs_error": max_mass_error,
        },
        "latency_ms": milliseconds,
        "estimated_effective_bandwidth_gb_s": estimated_bytes / (milliseconds * 1e6),
        "physical_hbm_bandwidth_gb_s": None,
        "physical_hbm_note": "Run Nsight Compute; this script does not relabel estimated tensor traffic as a hardware counter.",
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
