"""Deterministic numerical helpers."""
from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def finite_sample_quantile(values: Tensor, coverage: float) -> Tensor:
    flat = values.detach().float().flatten()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return torch.tensor(float("inf"), device=values.device)
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0,1]")
    rank = math.ceil((flat.numel() + 1) * coverage)
    rank = min(max(rank, 1), flat.numel())
    return torch.sort(flat).values[rank - 1]


def deterministic_kmeans(points: Tensor, clusters: int, *, iterations: int = 25, seed: int = 0) -> tuple[Tensor, Tensor]:
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("points must have shape [N,D] with N>0")
    n = points.shape[0]
    clusters = min(max(int(clusters), 1), n)
    work = points.detach().float()
    generator = torch.Generator(device=work.device)
    generator.manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=generator, device=work.device).item())
    centers = [work[first]]
    min_squared = torch.full((n,), float("inf"), device=work.device)
    for _ in range(1, clusters):
        distance = torch.sum((work - centers[-1]) ** 2, dim=-1)
        min_squared = torch.minimum(min_squared, distance)
        total = min_squared.sum()
        if not torch.isfinite(total) or total <= 1e-12:
            index = int(torch.argmax(min_squared).item())
        else:
            index = int(torch.multinomial(min_squared / total, 1, generator=generator).item())
        centers.append(work[index])
    center_tensor = torch.stack(centers)
    assignment = torch.full((n,), -1, dtype=torch.long, device=work.device)
    for _ in range(iterations):
        distances = torch.cdist(work, center_tensor)
        new_assignment = distances.argmin(dim=-1)
        new_centers = center_tensor.clone()
        for cluster in range(clusters):
            members = work[new_assignment == cluster]
            if members.numel():
                new_centers[cluster] = members.mean(dim=0)
            else:
                farthest = distances.min(dim=-1).values.argmax()
                new_centers[cluster] = work[farthest]
        converged = torch.equal(new_assignment, assignment) or torch.allclose(new_centers, center_tensor, rtol=1e-5, atol=1e-6)
        assignment, center_tensor = new_assignment, new_centers
        if converged:
            break
    return center_tensor.to(points.dtype), assignment


def principal_basis(points: Tensor, rank: int) -> tuple[Tensor, Tensor, float]:
    if points.ndim != 2:
        raise ValueError("points must be a matrix")
    dimension = points.shape[1]
    rank = min(max(int(rank), 1), dimension)
    work = points.detach().float()
    if work.shape[0] <= 1 or torch.allclose(work, work[:1].expand_as(work)):
        basis = torch.eye(dimension, device=work.device)[:, :rank]
        values = torch.zeros(rank, device=work.device)
        return basis.to(points.dtype), values.to(points.dtype), 0.0
    centered = work - work.mean(dim=0, keepdim=True)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    available = min(rank, vh.shape[0])
    basis = vh[:available].transpose(0, 1)
    if available < rank:
        candidate = torch.cat([basis, torch.eye(dimension, device=work.device)], dim=1)
        basis, _ = torch.linalg.qr(candidate, mode="reduced")
        basis = basis[:, :rank]
    variances_all = singular.square() / max(work.shape[0] - 1, 1)
    values = torch.zeros(rank, device=work.device)
    values[: min(rank, variances_all.numel())] = variances_all[:rank]
    discarded = float(variances_all[rank:].sum().item()) if variances_all.numel() > rank else 0.0
    return basis.to(points.dtype), values.to(points.dtype), discarded


def split_rows(points: Tensor, fractions: Iterable[float], *, seed: int) -> list[Tensor]:
    fractions = list(fractions)
    if not fractions or any(value <= 0 for value in fractions):
        raise ValueError("fractions must be positive")
    total = sum(fractions)
    normalized = [value / total for value in fractions]
    n = points.shape[0]
    generator = torch.Generator(device=points.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(n, generator=generator, device=points.device)
    boundaries = [0]
    cumulative = 0.0
    for fraction in normalized[:-1]:
        cumulative += fraction
        boundaries.append(round(cumulative * n))
    boundaries.append(n)
    return [points[permutation[boundaries[i] : boundaries[i + 1]]] for i in range(len(fractions))]


def tensor_numel(items: Iterable[Tensor]) -> int:
    return sum(int(item.numel()) for item in items)
