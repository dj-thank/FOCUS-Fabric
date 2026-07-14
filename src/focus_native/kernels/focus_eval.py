"""Fused Triton evaluator for compiled FOCUS pages.

The kernel routes each (page, head) query to its nearest anchor and evaluates
both the low-rank output Taylor operator and quadratic log-mass operator in one
pass.  A pure PyTorch implementation is provided for CPU and correctness.
"""
from __future__ import annotations

import torch
from torch import Tensor

try:  # Triton is optional on CPU-only installations.
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on CPU-only CI
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _focus_page_kernel(
        query_ptr,
        anchor_ptr,
        basis_ptr,
        output0_ptr,
        logmass0_ptr,
        left_ptr,
        gradient_ptr,
        hessian_ptr,
        output_ptr,
        logmass_ptr,
        route_ptr,
        distance_ptr,
        H: tl.constexpr,
        M: tl.constexpr,
        D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        program = tl.program_id(0)
        page = program // H
        head = program - page * H
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        q = tl.load(query_ptr + head * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        best_distance = float("inf")
        best_patch = 0
        for patch in tl.static_range(0, M):
            anchor_offset = (head * M + patch) * D + d_offsets
            anchor = tl.load(anchor_ptr + anchor_offset, mask=d_mask, other=0.0).to(tl.float32)
            squared = tl.sum(tl.where(d_mask, (q - anchor) * (q - anchor), 0.0), axis=0)
            take = squared < best_distance
            best_distance = tl.where(take, squared, best_distance)
            best_patch = tl.where(take, patch, best_patch)

        anchor_offset = (head * M + best_patch) * D + d_offsets
        anchor = tl.load(anchor_ptr + anchor_offset, mask=d_mask, other=0.0).to(tl.float32)
        delta = q - anchor

        reduced = tl.zeros((R,), dtype=tl.float32)
        r_offsets = tl.arange(0, R)
        # D is deliberately small (head dimensions 16--256); static unrolling
        # gives the compiler a simple reduction for every low-rank coordinate.
        for rank_index in tl.static_range(0, R):
            basis_offset = ((head * M + best_patch) * D + d_offsets) * R + rank_index
            basis_value = tl.load(basis_ptr + basis_offset, mask=d_mask, other=0.0).to(tl.float32)
            coordinate = tl.sum(tl.where(d_mask, basis_value * delta, 0.0), axis=0)
            reduced = tl.where(r_offsets == rank_index, coordinate, reduced)

        page_head_patch = (page * H + head) * M + best_patch
        base_output_offset = page_head_patch * D + d_offsets
        result = tl.load(output0_ptr + base_output_offset, mask=d_mask, other=0.0).to(tl.float32)
        gradient = tl.load(gradient_ptr + base_output_offset, mask=d_mask, other=0.0).to(tl.float32)
        linear_mass = tl.sum(tl.where(d_mask, gradient * delta, 0.0), axis=0)

        for rank_index in tl.static_range(0, R):
            left_offset = (page_head_patch * D + d_offsets) * R + rank_index
            left_value = tl.load(left_ptr + left_offset, mask=d_mask, other=0.0).to(tl.float32)
            coordinate = tl.sum(tl.where(r_offsets == rank_index, reduced, 0.0), axis=0)
            result += left_value * coordinate

        quadratic = 0.0
        for row in tl.static_range(0, R):
            row_value = tl.sum(tl.where(r_offsets == row, reduced, 0.0), axis=0)
            for column in tl.static_range(0, R):
                column_value = tl.sum(tl.where(r_offsets == column, reduced, 0.0), axis=0)
                hessian_offset = page_head_patch * R * R + row * R + column
                hessian_value = tl.load(hessian_ptr + hessian_offset).to(tl.float32)
                quadratic += row_value * hessian_value * column_value

        mass = tl.load(logmass0_ptr + page_head_patch).to(tl.float32)
        mass += linear_mass + 0.5 * quadratic
        output_offset = (page * H + head) * D + d_offsets
        tl.store(output_ptr + output_offset, result, mask=d_mask)
        tl.store(logmass_ptr + page * H + head, mass)
        tl.store(route_ptr + page * H + head, best_patch)
        tl.store(distance_ptr + page * H + head, tl.sqrt(best_distance))


def _torch_focus_page_eval(
    query: Tensor,
    anchors: Tensor,
    basis: Tensor,
    output0: Tensor,
    logmass0: Tensor,
    left: Tensor,
    gradient: Tensor,
    hessian: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # query [H,D], page fields [P,H,M,...]
    distances = torch.linalg.vector_norm(query[:, None, :] - anchors, dim=-1)
    route = distances.argmin(dim=-1)
    selected_distance = distances.gather(1, route[:, None]).squeeze(1)
    pages = output0.shape[0]
    head_index = torch.arange(query.shape[0], device=query.device)
    selected_anchor = anchors[head_index, route]
    selected_basis = basis[head_index, route]
    delta = query - selected_anchor
    reduced = torch.einsum("hdr,hd->hr", selected_basis, delta)

    page_index = torch.arange(pages, device=query.device)[:, None]
    heads = head_index[None, :]
    routes = route[None, :].expand(pages, -1)
    base = output0[page_index, heads, routes]
    selected_left = left[page_index, heads, routes]
    selected_gradient = gradient[page_index, heads, routes]
    selected_hessian = hessian[page_index, heads, routes]
    output = base + torch.einsum("phdr,hr->phd", selected_left, reduced)
    mass = logmass0[page_index, heads, routes]
    mass = mass + torch.einsum("phd,hd->ph", selected_gradient, delta)
    mass = mass + 0.5 * torch.einsum("hr,phrs,hs->ph", reduced, selected_hessian, reduced)
    return output, mass, route.expand(pages, -1), selected_distance.expand(pages, -1)


def focus_page_eval(
    query: Tensor,
    anchors: Tensor,
    basis: Tensor,
    output0: Tensor,
    logmass0: Tensor,
    left: Tensor,
    gradient: Tensor,
    hessian: Tensor,
    *,
    use_triton: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate all pages for one query.

    Shapes are ``query[H,D]``, ``anchors[H,M,D]``, ``basis[H,M,D,R]`` and
    page-first compiled tensors ``[P,H,M,...]``.
    """
    tensors = [query, anchors, basis, output0, logmass0, left, gradient, hessian]
    if not all(tensor.is_contiguous() for tensor in tensors):
        tensors = [tensor.contiguous() for tensor in tensors]
        query, anchors, basis, output0, logmass0, left, gradient, hessian = tensors
    pages, heads, patches, dimension = output0.shape
    rank = basis.shape[-1]
    expected = {
        "query": (heads, dimension),
        "anchors": (heads, patches, dimension),
        "basis": (heads, patches, dimension, rank),
        "logmass0": (pages, heads, patches),
        "left": (pages, heads, patches, dimension, rank),
        "gradient": (pages, heads, patches, dimension),
        "hessian": (pages, heads, patches, rank, rank),
    }
    actual = {
        "query": tuple(query.shape), "anchors": tuple(anchors.shape),
        "basis": tuple(basis.shape), "logmass0": tuple(logmass0.shape),
        "left": tuple(left.shape), "gradient": tuple(gradient.shape),
        "hessian": tuple(hessian.shape),
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} expected {shape}, got {actual[name]}")

    if not (use_triton and TRITON_AVAILABLE and query.is_cuda):
        return _torch_focus_page_eval(
            query, anchors, basis, output0, logmass0, left, gradient, hessian
        )
    if rank <= 0 or rank > 32 or rank & (rank - 1):
        raise ValueError("Triton reference kernel requires power-of-two rank<=32")
    if dimension <= 0 or dimension > 256:
        raise ValueError("Triton reference kernel supports head_dim<=256")
    output = torch.empty((pages, heads, dimension), device=query.device, dtype=query.dtype)
    logmass = torch.empty((pages, heads), device=query.device, dtype=query.dtype)
    route = torch.empty((pages, heads), device=query.device, dtype=torch.int32)
    distance = torch.empty((pages, heads), device=query.device, dtype=query.dtype)
    block_d = triton.next_power_of_2(dimension)
    _focus_page_kernel[(pages * heads,)](
        query, anchors, basis, output0, logmass0, left, gradient, hessian,
        output, logmass, route, distance,
        H=heads, M=patches, D=dimension, R=rank, BLOCK_D=block_d,
        num_warps=4 if block_d >= 64 else 2,
    )
    return output, logmass, route.to(torch.long), distance
