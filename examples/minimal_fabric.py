"""Minimal online attention-memory example."""
from __future__ import annotations

import torch

from focus_fabric.config import CompilerConfig, FabricConfig
from focus_fabric.fabric import MemoryFabricLayer


def main() -> None:
    torch.manual_seed(0)
    compiler = CompilerConfig(
        operator_patches=(2,),
        operator_ranks=(2,),
        coreset_slots=(4, 8),
        gaussian_clusters=(2,),
        gaussian_ranks=(2,),
        moment_ranks=(2,),
        hybrid_exact_slots=(2,),
        min_queries=24,
        max_queries=32,
        kmeans_iterations=4,
        certificate_tolerance=1.0,
    )
    config = FabricConfig(
        n_heads=2,
        head_dim=8,
        hot_window=8,
        page_size=8,
        query_bank_size=32,
        compiler=compiler,
    )
    layer = MemoryFabricLayer.create(config)
    for _ in range(64):
        query, key, value = torch.randn(3, config.n_heads, config.head_dim)
        output = layer.append_and_attend(query, key, value)
    report = layer.report(include_pages=False)
    print(output.shape)
    print({
        "tokens": report["tokens"],
        "page_levels": report["page_levels"],
        "active_compression": report["active_compression"],
        "fallback_rate": report["fallback_rate"],
    })


if __name__ == "__main__":
    main()
