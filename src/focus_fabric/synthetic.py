"""Controlled heterogeneous attention fields for mechanism validation.

These cases are not language benchmarks.  They are falsifiable diagnostics for
the architectural claim that one memory representation need not dominate every
head: smooth response manifolds, clustered exponential families, diffuse
low-rank moments, and rare high-influence associations are generated together.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class HeterogeneousAttentionCase:
    keys: Tensor
    values: Tensor
    query_train: Tensor
    query_test: Tensor
    query_ood: Tensor
    regimes: tuple[str, ...]
    scale: float


def _orthogonal(dimension: int, rank: int, generator: torch.Generator) -> Tensor:
    matrix = torch.randn(dimension, rank, generator=generator)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis[:, :rank]


def make_heterogeneous_case(
    *,
    tokens: int = 256,
    dimension: int = 16,
    train_queries: int = 192,
    test_queries: int = 256,
    ood_queries: int = 96,
    seed: int = 7,
) -> HeterogeneousAttentionCase:
    if tokens < 32 or dimension < 8:
        raise ValueError("tokens>=32 and dimension>=8 are required")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    scale = dimension ** -0.5
    keys: list[Tensor] = []
    values: list[Tensor] = []
    train: list[Tensor] = []
    test: list[Tensor] = []
    ood: list[Tensor] = []

    # Head 0: smooth low-dimensional response field.
    rank0 = min(3, dimension)
    basis0 = _orthogonal(dimension, rank0, generator)
    value_map0 = torch.randn(rank0, dimension, generator=generator) / rank0**0.5
    latent0 = torch.randn(tokens, rank0, generator=generator)
    key0 = 0.9 * latent0 @ basis0.T + 0.025 * torch.randn(
        tokens, dimension, generator=generator
    )
    value0 = torch.tanh(latent0 @ value_map0) + 0.025 * torch.randn(
        tokens, dimension, generator=generator
    )

    def queries0(count: int, multiplier: float = 1.0) -> Tensor:
        latent = multiplier * torch.randn(count, rank0, generator=generator)
        return 0.9 * latent @ basis0.T + 0.025 * torch.randn(
            count, dimension, generator=generator
        )

    keys.append(key0)
    values.append(value0)
    train.append(queries0(train_queries))
    test.append(queries0(test_queries))
    ood.append(queries0(ood_queries, 3.4))

    # Head 1: four locally Gaussian populations with linear value tilt.
    clusters = 4
    centers = F.normalize(
        torch.randn(clusters, dimension, generator=generator), dim=-1
    ) * 2.2
    value_centers = torch.randn(clusters, dimension, generator=generator)
    labels = torch.arange(tokens) % clusters
    labels = labels[torch.randperm(tokens, generator=generator)]
    key1 = centers[labels] + 0.18 * torch.randn(
        tokens, dimension, generator=generator
    )
    local_map = torch.randn(
        clusters, dimension, dimension, generator=generator
    ) * 0.045
    delta = key1 - centers[labels]
    value1 = value_centers[labels] + torch.einsum(
        "nd,ndk->nk", delta, local_map[labels]
    )
    value1 = value1 + 0.04 * torch.randn(tokens, dimension, generator=generator)

    def queries1(count: int, shift: float = 0.0) -> Tensor:
        target = torch.randint(clusters, (count,), generator=generator)
        query = centers[target] + 0.30 * torch.randn(
            count, dimension, generator=generator
        )
        if shift:
            query = query + shift * F.normalize(
                torch.randn(count, dimension, generator=generator), dim=-1
            )
        return query

    keys.append(key1)
    values.append(value1)
    train.append(queries1(train_queries))
    test.append(queries1(test_queries))
    ood.append(queries1(ood_queries, 4.0))

    # Head 2: diffuse low-rank keys where many small contributions matter.
    rank2 = min(4, dimension)
    basis2 = _orthogonal(dimension, rank2, generator)
    latent2 = 0.30 * torch.randn(tokens, rank2, generator=generator)
    key2 = latent2 @ basis2.T + 0.012 * torch.randn(
        tokens, dimension, generator=generator
    )
    map2 = torch.randn(rank2, dimension, generator=generator) / rank2**0.5
    value2 = latent2 @ map2 + 0.06 * torch.randn(
        tokens, dimension, generator=generator
    )

    def queries2(count: int, multiplier: float = 1.0) -> Tensor:
        return multiplier * (
            0.30 * torch.randn(count, rank2, generator=generator) @ basis2.T
        )

    keys.append(key2)
    values.append(value2)
    train.append(queries2(train_queries))
    test.append(queries2(test_queries))
    ood.append(queries2(ood_queries, 5.0))

    # Head 3: compressible background plus rare high-salience associations.
    needle_count = min(8, tokens // 8)
    background_count = tokens - needle_count
    key3 = 0.28 * torch.randn(tokens, dimension, generator=generator)
    value3 = 0.22 * torch.randn(tokens, dimension, generator=generator)
    needle_keys = F.normalize(
        torch.randn(needle_count, dimension, generator=generator), dim=-1
    ) * 4.7
    needle_values = F.normalize(
        torch.randn(needle_count, dimension, generator=generator), dim=-1
    ) * 3.2
    key3[background_count:] = needle_keys
    value3[background_count:] = needle_values

    def queries3(count: int, shifted: bool = False) -> Tensor:
        query = 0.28 * torch.randn(count, dimension, generator=generator)
        target_count = max(1, count // 3)
        target = torch.randint(needle_count, (target_count,), generator=generator)
        query[:target_count] = needle_keys[target] + 0.08 * torch.randn(
            target_count, dimension, generator=generator
        )
        if shifted:
            query[target_count:] *= 6.0
        return query[torch.randperm(count, generator=generator)]

    keys.append(key3)
    values.append(value3)
    train.append(queries3(train_queries))
    test.append(queries3(test_queries))
    ood.append(queries3(ood_queries, True))

    return HeterogeneousAttentionCase(
        keys=torch.stack(keys),
        values=torch.stack(values),
        query_train=torch.stack(train),
        query_test=torch.stack(test),
        query_ood=torch.stack(ood),
        regimes=(
            "smooth_operator",
            "clustered_cumulant",
            "diffuse_moment",
            "rare_exact_residual",
        ),
        scale=scale,
    )
