"""Fused Gaussian-mixture evaluator with a CPU PyTorch reference.

Heterogeneous pages are grouped by codec family and static shape before GPU
dispatch.  This keeps representation-family branching outside the warp.  The
kernel is included and syntax-tested here; CUDA correctness and throughput are
explicitly left unclaimed in this CPU-only environment.
"""
from __future__ import annotations

import torch
from torch import Tensor

try:  # pragma: no cover - unavailable in CPU CI
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:  # pragma: no cover - requires CUDA/Triton

    @triton.jit
    def _gaussian_kernel(
        query_ptr,
        mean_key_ptr,
        mean_value_ptr,
        log_count_ptr,
        basis_ptr,
        variance_ptr,
        cross_ptr,
        output_ptr,
        logmass_ptr,
        scale,
        C: tl.constexpr,
        D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        group = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        mask = d < D
        q = tl.load(query_ptr + group * D + d, mask=mask, other=0.0).to(tl.float32)

        maximum = -float("inf")
        for cluster in tl.static_range(0, C):
            mean_key = tl.load(
                mean_key_ptr + (group * C + cluster) * D + d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            score = scale * tl.sum(tl.where(mask, q * mean_key, 0.0), axis=0)
            for rank in tl.static_range(0, R):
                basis = tl.load(
                    basis_ptr + ((group * C + cluster) * D + d) * R + rank,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                coordinate = tl.sum(tl.where(mask, q * basis, 0.0), axis=0)
                variance = tl.load(
                    variance_ptr + (group * C + cluster) * R + rank
                ).to(tl.float32)
                score += 0.5 * scale * scale * variance * coordinate * coordinate
            score += tl.load(log_count_ptr + group * C + cluster).to(tl.float32)
            maximum = tl.maximum(maximum, score)

        denominator = 0.0
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for cluster in tl.static_range(0, C):
            mean_key = tl.load(
                mean_key_ptr + (group * C + cluster) * D + d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            tilted = tl.load(
                mean_value_ptr + (group * C + cluster) * D + d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            score = scale * tl.sum(tl.where(mask, q * mean_key, 0.0), axis=0)
            for rank in tl.static_range(0, R):
                basis = tl.load(
                    basis_ptr + ((group * C + cluster) * D + d) * R + rank,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                coordinate = tl.sum(tl.where(mask, q * basis, 0.0), axis=0)
                variance = tl.load(
                    variance_ptr + (group * C + cluster) * R + rank
                ).to(tl.float32)
                score += 0.5 * scale * scale * variance * coordinate * coordinate
                cross = tl.load(
                    cross_ptr + ((group * C + cluster) * D + d) * R + rank,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                tilted += scale * cross * coordinate
            score += tl.load(log_count_ptr + group * C + cluster).to(tl.float32)
            weight = tl.exp(score - maximum)
            denominator += weight
            accumulator += weight * tilted
        tl.store(output_ptr + group * D + d, accumulator / denominator, mask=mask)
        tl.store(logmass_ptr + group, maximum + tl.log(denominator))


def torch_gaussian_eval(
    query: Tensor,
    mean_keys: Tensor,
    mean_values: Tensor,
    log_counts: Tensor,
    basis: Tensor,
    variances: Tensor,
    cross: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Reference evaluator for homogeneous codec batches.

    Shapes are ``query[G,D]``, means ``[G,C,D]``, basis/cross
    ``[G,C,D,R]``, and variances ``[G,C,R]``.
    """

    coordinates = torch.einsum("gd,gcdr->gcr", query.float(), basis.float())
    component = log_counts.float()
    component = component + float(scale) * torch.einsum(
        "gd,gcd->gc", query.float(), mean_keys.float()
    )
    component = component + 0.5 * float(scale) ** 2 * torch.sum(
        variances.float() * coordinates.square(), dim=-1
    )
    log_mass = torch.logsumexp(component, dim=-1)
    weights = torch.softmax(component, dim=-1)
    tilted = mean_values.float() + float(scale) * torch.einsum(
        "gcdr,gcr->gcd", cross.float(), coordinates
    )
    output = torch.sum(weights.unsqueeze(-1) * tilted, dim=1)
    return output.to(query.dtype), log_mass.to(query.dtype)


def gaussian_eval(
    query: Tensor,
    mean_keys: Tensor,
    mean_values: Tensor,
    log_counts: Tensor,
    basis: Tensor,
    variances: Tensor,
    cross: Tensor,
    scale: float,
    *,
    use_triton: bool = True,
) -> tuple[Tensor, Tensor]:
    groups, clusters, dimension = mean_keys.shape
    rank = basis.shape[-1]
    expected = {
        "query": (groups, dimension),
        "mean_values": (groups, clusters, dimension),
        "log_counts": (groups, clusters),
        "basis": (groups, clusters, dimension, rank),
        "variances": (groups, clusters, rank),
        "cross": (groups, clusters, dimension, rank),
    }
    actual = {
        "query": tuple(query.shape),
        "mean_values": tuple(mean_values.shape),
        "log_counts": tuple(log_counts.shape),
        "basis": tuple(basis.shape),
        "variances": tuple(variances.shape),
        "cross": tuple(cross.shape),
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} expected {shape}, got {actual[name]}")
    tensors = [query, mean_keys, mean_values, log_counts, basis, variances, cross]
    if not all(tensor.is_contiguous() for tensor in tensors):
        query, mean_keys, mean_values, log_counts, basis, variances, cross = [
            tensor.contiguous() for tensor in tensors
        ]
    if not (use_triton and TRITON_AVAILABLE and query.is_cuda):
        return torch_gaussian_eval(
            query,
            mean_keys,
            mean_values,
            log_counts,
            basis,
            variances,
            cross,
            scale,
        )
    if clusters > 16 or rank > 16 or dimension > 256:
        raise ValueError("reference Triton kernel supports C,R<=16 and D<=256")
    output = torch.empty_like(query)
    log_mass = torch.empty(groups, dtype=query.dtype, device=query.device)
    block_dimension = triton.next_power_of_2(dimension)
    _gaussian_kernel[(groups,)](
        query,
        mean_keys,
        mean_values,
        log_counts,
        basis,
        variances,
        cross,
        output,
        log_mass,
        float(scale),
        C=clusters,
        D=dimension,
        R=rank,
        BLOCK_D=block_dimension,
        num_warps=4 if block_dimension >= 64 else 2,
    )
    return output, log_mass
