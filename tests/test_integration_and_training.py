from __future__ import annotations

from pathlib import Path

import pytest
import torch

from focus_fabric.config import CompilerConfig, FabricConfig
from focus_fabric.integration import sequential_logits_fabric
from focus_fabric.training import NativeLossWeights, focus_native_loss
from focus_native import CacheConfig
from focus_native.config import ModelConfig
from focus_native.generation import sequential_logits
from focus_native.io import load_checkpoint
from focus_native.model import FocusTransformer

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/focus-native-small" / "model.safetensors"


def fast_fabric(model: FocusTransformer) -> FabricConfig:
    compiler = CompilerConfig(
        operator_patches=(2,),
        operator_ranks=(2,),
        coreset_slots=(4,),
        gaussian_clusters=(2,),
        gaussian_ranks=(2,),
        moment_ranks=(2,),
        hybrid_exact_slots=(2,),
        min_queries=18,
        max_queries=24,
        kmeans_iterations=3,
        certificate_tolerance=0.08,
        target_active_ratio=0.8,
    )
    return FabricConfig(
        n_heads=model.config.n_heads,
        head_dim=model.config.head_dim,
        hot_window=8,
        page_size=8,
        query_bank_size=24,
        compiler=compiler,
    )


def test_repaired_checkpoint_parallel_matches_exact_cache() -> None:
    if not CHECKPOINT.exists():
        pytest.skip("optional archived checkpoint is not redistributed in the public source repository")
    model, _, _ = load_checkpoint(ROOT / "checkpoints/focus-native-small")
    ids = torch.randint(0, model.config.vocab_size, (1, 20))
    with torch.no_grad():
        parallel = model(ids).logits[0]
        sequential, _ = sequential_logits(
            model, ids[0].tolist(), CacheConfig(mode="exact")
        )
    assert torch.allclose(parallel, sequential, atol=3e-5, rtol=3e-5)


def test_fabric_argmax_agrees_on_archived_checkpoint_trace() -> None:
    if not CHECKPOINT.exists():
        pytest.skip("optional archived checkpoint is not redistributed in the public source repository")
    model, _, _ = load_checkpoint(ROOT / "checkpoints/focus-native-small")
    ids = torch.randint(0, model.config.vocab_size, (32,)).tolist()
    exact, _ = sequential_logits(model, ids, CacheConfig(mode="exact"))
    approximate, _ = sequential_logits_fabric(model, ids, fast_fabric(model))
    assert torch.equal(exact.argmax(-1), approximate.argmax(-1))


def test_native_loss_backpropagates() -> None:
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=32,
        focus_patches=2,
        focus_rank=4,
        memory_code_head=3,
    )
    model = FocusTransformer(config)
    ids = torch.randint(0, 64, (2, 16))
    labels = ids.roll(-1, dims=1)
    result = focus_native_loss(
        model,
        ids,
        labels,
        cold_len=8,
        weights=NativeLossWeights(),
    )
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
