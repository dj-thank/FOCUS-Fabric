"""Heterogeneous per-head KV-cache representations.

FOCUS-Fabric deliberately avoids assuming that one compression family is
universally optimal.  The compiler can choose, independently for every
page/head, among:

* a local response operator (FOCUS),
* a weighted sparse KV coreset,
* a low-rank Gaussian/cumulant mixture,
* a merge-friendly second-order moment state, and
* a semiparametric hybrid with a small exact residual set.

Every codec returns both the locally normalized value expectation and the log
unnormalized attention mass.  This is essential: page outputs can only be
composed correctly when their softmax normalizers are retained.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from .exact import exact_head_batch, exact_head_summary
from .types import AttentionSummary, CodecEvaluation, merge_summaries
from .utils import deterministic_kmeans, principal_basis, tensor_numel


class HeadCodec(ABC):
    """Interface for a single attention head over one disjoint token region."""

    name: str
    token_count: int
    dimension: int
    scale: float

    @abstractmethod
    def evaluate(self, query: Tensor) -> CodecEvaluation:
        raise NotImplementedError

    def evaluate_batch(self, queries: Tensor) -> CodecEvaluation:
        if queries.ndim != 2 or queries.shape[-1] != self.dimension:
            raise ValueError("queries must have shape [Q,D]")
        evaluations = [self.evaluate(query) for query in queries]
        return CodecEvaluation(
            AttentionSummary(
                torch.stack([item.summary.output for item in evaluations]),
                torch.stack([item.summary.log_mass for item in evaluations]),
            ),
            torch.stack([item.proxy for item in evaluations]),
            torch.stack([item.valid for item in evaluations]),
        )

    @abstractmethod
    def active_numel(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def estimated_flops(self) -> int:
        raise NotImplementedError

    def active_bytes(self, dtype_bytes: int) -> int:
        return self.active_numel() * int(dtype_bytes)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tokens": int(self.token_count),
            "dimension": int(self.dimension),
        }


@dataclass
class ExactCodec(HeadCodec):
    keys: Tensor
    values: Tensor
    scale: float
    name: str = "exact"

    def __post_init__(self) -> None:
        if self.keys.ndim != 2 or self.values.shape != self.keys.shape:
            raise ValueError("exact codec expects matching keys/values [N,D]")
        self.token_count = int(self.keys.shape[0])
        self.dimension = int(self.keys.shape[1])

    def evaluate(self, query: Tensor) -> CodecEvaluation:
        summary = exact_head_summary(query, self.keys, self.values, self.scale)
        zero = query.new_zeros(())
        valid = torch.ones((), dtype=torch.bool, device=query.device)
        return CodecEvaluation(summary, zero, valid)

    def evaluate_batch(self, queries: Tensor) -> CodecEvaluation:
        summary = exact_head_batch(queries, self.keys, self.values, self.scale)
        return CodecEvaluation(
            summary,
            torch.zeros(queries.shape[0], dtype=queries.dtype, device=queries.device),
            torch.ones(queries.shape[0], dtype=torch.bool, device=queries.device),
        )

    def active_numel(self) -> int:
        return int(self.keys.numel() + self.values.numel())

    def estimated_flops(self) -> int:
        return int(4 * self.token_count * self.dimension)


@dataclass
class WeightedCoresetCodec(HeadCodec):
    centroids: Tensor
    values: Tensor
    log_weights: Tensor
    radii: Tensor
    scale: float
    token_count: int
    compile_restarts: int = 1
    compile_score: float = float("nan")
    name: str = "weighted_coreset"

    def __post_init__(self) -> None:
        if self.centroids.ndim != 2 or self.values.shape != self.centroids.shape:
            raise ValueError("coreset centroids and values must match")
        if self.log_weights.shape != (self.centroids.shape[0],):
            raise ValueError("invalid coreset weights")
        self.dimension = self.centroids.shape[1]

    @classmethod
    def _compile_once(
        cls,
        keys: Tensor,
        values: Tensor,
        *,
        slots: int,
        scale: float,
        seed: int,
        iterations: int,
    ) -> "WeightedCoresetCodec":
        centers, assignment = deterministic_kmeans(keys, slots, iterations=iterations, seed=seed)
        clusters = centers.shape[0]
        mean_keys = torch.empty_like(centers)
        mean_values = torch.empty_like(centers)
        counts = torch.empty(clusters, device=keys.device, dtype=torch.float32)
        radii = torch.empty(clusters, device=keys.device, dtype=torch.float32)
        for index in range(clusters):
            mask = assignment == index
            member_keys = keys[mask]
            member_values = values[mask]
            if member_keys.shape[0] == 0:  # defensive; k-means repairs empty clusters
                member_keys = centers[index : index + 1]
                member_values = values[:1]
            # A centroid key alone is not sufficient: the associated value and
            # multiplicity are the first two moments of the original cluster.
            mean_keys[index] = member_keys.mean(dim=0)
            mean_values[index] = member_values.mean(dim=0)
            counts[index] = member_keys.shape[0]
            radii[index] = torch.linalg.vector_norm(
                member_keys.float() - mean_keys[index].float(), dim=-1
            ).max()
        return cls(
            mean_keys.contiguous(),
            mean_values.contiguous(),
            counts.log().to(keys.dtype),
            radii.to(keys.dtype),
            float(scale),
            int(keys.shape[0]),
            name=f"weighted_coreset_s{clusters}",
        )

    @classmethod
    def compile(
        cls,
        keys: Tensor,
        values: Tensor,
        *,
        slots: int,
        scale: float,
        seed: int,
        iterations: int,
        queries: Tensor | None = None,
        restarts: int = 1,
    ) -> "WeightedCoresetCodec":
        """Compile a weighted KV coreset with query-aware multi-start selection.

        K-means is non-convex and a single unlucky initialization can erase a
        rare retrieval route.  When a trace bank is available, independent
        restarts are selected by exact attention response and log-mass error,
        not merely Euclidean key inertia.
        """

        restarts = max(int(restarts), 1)
        candidates = [
            cls._compile_once(
                keys,
                values,
                slots=slots,
                scale=scale,
                seed=seed + 7919 * restart,
                iterations=iterations,
            )
            for restart in range(restarts)
        ]
        if queries is None or queries.shape[0] == 0 or len(candidates) == 1:
            selected = candidates[0]
            selected.compile_restarts = restarts
            return selected
        exact = exact_head_batch(queries, keys, values, scale)
        denominator = exact.output.float().square().mean().clamp_min(1e-8)
        mass_scale = (1.0 + exact.log_mass.float().abs().mean()).square()
        scored: list[tuple[float, WeightedCoresetCodec]] = []
        for candidate in candidates:
            approximation = candidate.evaluate_batch(queries).summary
            output_nmse = (
                (approximation.output.float() - exact.output.float()).square().mean()
                / denominator
            )
            mass_mse = (
                (approximation.log_mass.float() - exact.log_mass.float()).square().mean()
                / mass_scale
            )
            scored.append((float((output_nmse + 0.25 * mass_mse).item()), candidate))
        score, selected = min(scored, key=lambda item: item[0])
        selected.compile_restarts = restarts
        selected.compile_score = score
        return selected

    def evaluate(self, query: Tensor) -> CodecEvaluation:
        scores = torch.mv(self.centroids.float(), query.float()) * self.scale + self.log_weights.float()
        mass = torch.logsumexp(scores, dim=0).to(query.dtype)
        probabilities = torch.softmax(scores, dim=0).to(self.values.dtype)
        output = torch.mv(self.values.transpose(0, 1), probabilities).to(query.dtype)
        proxy = self.scale * torch.linalg.vector_norm(query.float()) * torch.max(self.radii.float())
        valid = torch.isfinite(output).all() & torch.isfinite(mass)
        return CodecEvaluation(AttentionSummary(output, mass), proxy.to(query.dtype), valid)

    def evaluate_batch(self, queries: Tensor) -> CodecEvaluation:
        scores = torch.matmul(queries.float(), self.centroids.float().transpose(0, 1)) * self.scale
        scores = scores + self.log_weights.float().unsqueeze(0)
        mass = torch.logsumexp(scores, dim=-1).to(queries.dtype)
        probabilities = torch.softmax(scores, dim=-1).to(self.values.dtype)
        output = torch.matmul(probabilities, self.values).to(queries.dtype)
        proxy = (
            self.scale
            * torch.linalg.vector_norm(queries.float(), dim=-1)
            * torch.max(self.radii.float())
        ).to(queries.dtype)
        valid = torch.isfinite(output).all(dim=-1) & torch.isfinite(mass)
        return CodecEvaluation(AttentionSummary(output, mass), proxy, valid)

    def active_numel(self) -> int:
        return tensor_numel((self.centroids, self.values, self.log_weights, self.radii))

    def estimated_flops(self) -> int:
        slots = self.centroids.shape[0]
        return int(slots * (4 * self.dimension + 8))

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "slots": int(self.centroids.shape[0]),
            "compile_restarts": self.compile_restarts,
            "compile_score": self.compile_score,
        }


@dataclass
class OperatorCodec(HeadCodec):
    """Piecewise low-rank local operator for attention response and mass."""

    anchors: Tensor              # [M,D]
    basis: Tensor                # [M,D,R]
    output0: Tensor              # [M,D]
    logmass0: Tensor             # [M]
    left: Tensor                 # [M,D,R], J @ basis
    gradient: Tensor             # [M,D]
    hessian_small: Tensor        # [M,R,R]
    jacobian_tail: Tensor        # [M]
    hessian_tail: Tensor         # [M]
    support_radius: Tensor       # [M]
    local_curvature: Tensor      # [M]
    scale: float
    token_count: int
    name: str = "operator"

    def __post_init__(self) -> None:
        if self.anchors.ndim != 2 or self.basis.ndim != 3:
            raise ValueError("operator anchors/basis must have shapes [M,D] and [M,D,R]")
        if self.basis.shape[:2] != self.anchors.shape:
            raise ValueError("operator basis is incompatible with anchors")
        self.dimension = int(self.anchors.shape[-1])

    @classmethod
    def compile(
        cls,
        keys: Tensor,
        values: Tensor,
        queries: Tensor,
        *,
        patches: int,
        rank: int,
        scale: float,
        seed: int,
        iterations: int,
    ) -> "OperatorCodec":
        if queries.ndim != 2 or queries.shape[-1] != keys.shape[-1]:
            raise ValueError("operator query bank must have shape [Q,D]")
        routing_queries = queries if queries.shape[0] else keys
        anchors, assignment = deterministic_kmeans(
            routing_queries, patches, iterations=iterations, seed=seed
        )
        patch_count = int(anchors.shape[0])
        dimension = int(keys.shape[-1])
        rank = min(max(int(rank), 1), dimension)

        basis = torch.empty(patch_count, dimension, rank, device=keys.device, dtype=keys.dtype)
        output0 = torch.empty(patch_count, dimension, device=keys.device, dtype=values.dtype)
        logmass0 = torch.empty(patch_count, device=keys.device, dtype=keys.dtype)
        left = torch.empty(patch_count, dimension, rank, device=keys.device, dtype=values.dtype)
        gradient = torch.empty(patch_count, dimension, device=keys.device, dtype=keys.dtype)
        hessian_small = torch.empty(patch_count, rank, rank, device=keys.device, dtype=keys.dtype)
        jacobian_tail = torch.empty(patch_count, device=keys.device, dtype=keys.dtype)
        hessian_tail = torch.empty(patch_count, device=keys.device, dtype=keys.dtype)
        support_radius = torch.empty(patch_count, device=keys.device, dtype=keys.dtype)
        local_curvature = torch.empty(patch_count, device=keys.device, dtype=keys.dtype)

        kf = keys.float()
        vf = values.float()
        for patch in range(patch_count):
            anchor = anchors[patch].float()
            scores = torch.mv(kf, anchor) * float(scale)
            probabilities = torch.softmax(scores, dim=0)
            response = torch.mv(vf.transpose(0, 1), probabilities)
            mean_key = torch.mv(kf.transpose(0, 1), probabilities)
            weighted_cross = torch.matmul(vf.transpose(0, 1), probabilities[:, None] * kf)
            jacobian = float(scale) * (
                weighted_cross - response[:, None] * mean_key[None, :]
            )
            centered_keys = kf - mean_key
            covariance = torch.matmul(
                (probabilities[:, None] * centered_keys).transpose(0, 1),
                centered_keys,
            )
            hessian = float(scale) ** 2 * covariance

            # Joint subspace: output sensitivity and normalizer curvature both
            # influence the retained directions.  Normalization prevents one
            # term from dominating purely by units.
            j_scale = torch.linalg.matrix_norm(jacobian).clamp_min(1e-6)
            h_scale = torch.linalg.matrix_norm(hessian).clamp_min(1e-6)
            gram = (jacobian.transpose(0, 1) @ jacobian) / j_scale.square()
            gram = gram + (hessian.transpose(0, 1) @ hessian) / h_scale.square()
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(gram)
                local_basis = eigenvectors[:, torch.argsort(eigenvalues, descending=True)[:rank]]
            except RuntimeError:
                local_basis, _, _ = principal_basis(routing_queries - anchors[patch], rank)
                local_basis = local_basis.float()
            if local_basis.shape[1] < rank:
                candidate = torch.cat(
                    [local_basis, torch.eye(dimension, device=keys.device)], dim=1
                )
                local_basis, _ = torch.linalg.qr(candidate, mode="reduced")
                local_basis = local_basis[:, :rank]

            reduced_left = jacobian @ local_basis
            reduced_hessian = local_basis.transpose(0, 1) @ hessian @ local_basis
            reconstructed_j = reduced_left @ local_basis.transpose(0, 1)
            reconstructed_h = local_basis @ reduced_hessian @ local_basis.transpose(0, 1)

            members = routing_queries[assignment == patch]
            if members.shape[0]:
                distances = torch.linalg.vector_norm(members.float() - anchor, dim=-1)
                radius = torch.quantile(distances, 0.95).clamp_min(1e-4)
                # Curvature proxy from Jacobian variation sampled within the
                # patch.  This is intentionally cheap and later calibrated by
                # split conformal prediction.
                sample = members[: min(8, members.shape[0])]
                variations: list[Tensor] = []
                for point in sample:
                    p_scores = torch.mv(kf, point.float()) * float(scale)
                    p_prob = torch.softmax(p_scores, dim=0)
                    p_out = torch.mv(vf.transpose(0, 1), p_prob)
                    p_mean = torch.mv(kf.transpose(0, 1), p_prob)
                    p_cross = torch.matmul(vf.transpose(0, 1), p_prob[:, None] * kf)
                    p_jacobian = float(scale) * (p_cross - p_out[:, None] * p_mean[None, :])
                    denominator = torch.linalg.vector_norm(point.float() - anchor).clamp_min(1e-4)
                    variations.append(torch.linalg.matrix_norm(p_jacobian - jacobian) / denominator)
                curvature = torch.stack(variations).max() if variations else jacobian.new_tensor(0.0)
            else:
                radius = routing_queries.float().std().clamp_min(1e-3)
                curvature = jacobian.new_tensor(0.0)

            basis[patch] = local_basis.to(keys.dtype)
            output0[patch] = response.to(values.dtype)
            logmass0[patch] = torch.logsumexp(scores, dim=0).to(keys.dtype)
            left[patch] = reduced_left.to(values.dtype)
            gradient[patch] = (float(scale) * mean_key).to(keys.dtype)
            hessian_small[patch] = reduced_hessian.to(keys.dtype)
            jacobian_tail[patch] = torch.linalg.matrix_norm(
                jacobian - reconstructed_j, ord=2
            ).to(keys.dtype)
            hessian_tail[patch] = torch.linalg.matrix_norm(
                hessian - reconstructed_h, ord=2
            ).to(keys.dtype)
            support_radius[patch] = radius.to(keys.dtype)
            local_curvature[patch] = curvature.to(keys.dtype)

        return cls(
            anchors=anchors.contiguous(),
            basis=basis.contiguous(),
            output0=output0.contiguous(),
            logmass0=logmass0.contiguous(),
            left=left.contiguous(),
            gradient=gradient.contiguous(),
            hessian_small=hessian_small.contiguous(),
            jacobian_tail=jacobian_tail.contiguous(),
            hessian_tail=hessian_tail.contiguous(),
            support_radius=support_radius.contiguous(),
            local_curvature=local_curvature.contiguous(),
            scale=float(scale),
            token_count=int(keys.shape[0]),
            name=f"operator_m{patch_count}_r{rank}",
        )

    def _route(self, queries: Tensor) -> tuple[Tensor, Tensor]:
        distances = torch.cdist(queries.float(), self.anchors.float())
        route = distances.argmin(dim=-1)
        selected_distance = distances.gather(1, route[:, None]).squeeze(1)
        return route, selected_distance

    def evaluate_batch(self, queries: Tensor) -> CodecEvaluation:
        route, distance = self._route(queries)
        anchors = self.anchors.index_select(0, route)
        selected_basis = self.basis.index_select(0, route)
        delta = queries - anchors
        reduced = torch.einsum("qdr,qd->qr", selected_basis, delta)
        output = self.output0.index_select(0, route) + torch.einsum(
            "qdr,qr->qd", self.left.index_select(0, route), reduced
        )
        log_mass = self.logmass0.index_select(0, route)
        log_mass = log_mass + torch.sum(
            self.gradient.index_select(0, route) * delta, dim=-1
        )
        log_mass = log_mass + 0.5 * torch.einsum(
            "qr,qrs,qs->q",
            reduced,
            self.hessian_small.index_select(0, route),
            reduced,
        )
        jacobian_tail = self.jacobian_tail.index_select(0, route).float()
        hessian_tail = self.hessian_tail.index_select(0, route).float()
        curvature = self.local_curvature.index_select(0, route).float()
        support = self.support_radius.index_select(0, route).float().clamp_min(1e-4)
        normalized = distance / support
        proxy = (
            jacobian_tail * distance
            + 0.5 * (curvature + hessian_tail) * distance.square()
            + torch.relu(normalized - 1.0).square()
        )
        valid = torch.isfinite(output).all(dim=-1) & torch.isfinite(log_mass)
        return CodecEvaluation(
            AttentionSummary(output.to(queries.dtype), log_mass.to(queries.dtype)),
            proxy.to(queries.dtype),
            valid,
        )

    def evaluate(self, query: Tensor) -> CodecEvaluation:
        result = self.evaluate_batch(query.unsqueeze(0))
        return CodecEvaluation(
            AttentionSummary(result.summary.output[0], result.summary.log_mass[0]),
            result.proxy[0],
            result.valid[0],
        )

    def active_numel(self) -> int:
        return tensor_numel(
            (
                self.anchors,
                self.basis,
                self.output0,
                self.logmass0,
                self.left,
                self.gradient,
                self.hessian_small,
                self.jacobian_tail,
                self.hessian_tail,
                self.support_radius,
                self.local_curvature,
            )
        )

    def estimated_flops(self) -> int:
        patches, dimension, rank = self.basis.shape
        # Routing to every anchor plus one selected low-rank response.
        return int(3 * patches * dimension + 4 * dimension * rank + 2 * rank * rank)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "patches": int(self.anchors.shape[0]),
            "rank": int(self.basis.shape[-1]),
        }


@dataclass
class GaussianMixtureCodec(HeadCodec):
    """Low-rank cumulant approximation to clustered keys and values."""

    mean_keys: Tensor          # [C,D]
    mean_values: Tensor        # [C,D]
    log_counts: Tensor         # [C]
    basis: Tensor              # [C,D,R]
    variances: Tensor          # [C,R]
    cross: Tensor              # [C,D,R] = Cov(value, projected key)
    tail_variance: Tensor      # [C]
    cluster_radius: Tensor     # [C]
    scale: float
    token_count: int
    name: str = "gaussian_mixture"

    def __post_init__(self) -> None:
        if self.mean_keys.ndim != 2 or self.mean_values.shape != self.mean_keys.shape:
            raise ValueError("Gaussian means must have matching [C,D] shapes")
        self.dimension = int(self.mean_keys.shape[-1])

    @classmethod
    def compile(
        cls,
        keys: Tensor,
        values: Tensor,
        *,
        clusters: int,
        rank: int,
        scale: float,
        seed: int,
        iterations: int,
    ) -> "GaussianMixtureCodec":
        centers, assignment = deterministic_kmeans(
            keys, clusters, iterations=iterations, seed=seed
        )
        cluster_count = int(centers.shape[0])
        dimension = int(keys.shape[-1])
        rank = min(max(int(rank), 1), dimension)
        mean_keys = torch.empty(cluster_count, dimension, device=keys.device, dtype=keys.dtype)
        mean_values = torch.empty_like(mean_keys, dtype=values.dtype)
        log_counts = torch.empty(cluster_count, device=keys.device, dtype=keys.dtype)
        bases = torch.empty(cluster_count, dimension, rank, device=keys.device, dtype=keys.dtype)
        variances = torch.empty(cluster_count, rank, device=keys.device, dtype=keys.dtype)
        cross = torch.empty(cluster_count, dimension, rank, device=keys.device, dtype=values.dtype)
        tails = torch.empty(cluster_count, device=keys.device, dtype=keys.dtype)
        radii = torch.empty(cluster_count, device=keys.device, dtype=keys.dtype)

        for cluster in range(cluster_count):
            mask = assignment == cluster
            member_keys = keys[mask]
            member_values = values[mask]
            if member_keys.shape[0] == 0:
                member_keys = centers[cluster : cluster + 1]
                member_values = values[:1]
            kf = member_keys.float()
            vf = member_values.float()
            mean_key = kf.mean(dim=0)
            mean_value = vf.mean(dim=0)
            centered_keys = kf - mean_key
            centered_values = vf - mean_value
            local_basis, _, discarded = principal_basis(centered_keys, rank)
            local_basis = local_basis.float()
            projected = centered_keys @ local_basis
            denominator = max(int(member_keys.shape[0]), 1)
            covariance_projected = projected.transpose(0, 1) @ projected / denominator
            # Diagonalize covariance inside the retained subspace.
            eigenvalues, rotation = torch.linalg.eigh(covariance_projected)
            order = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[order].clamp_min(0)
            local_basis = local_basis @ rotation[:, order]
            projected = centered_keys @ local_basis
            local_cross = centered_values.transpose(0, 1) @ projected / denominator
            residual = centered_keys - projected @ local_basis.transpose(0, 1)
            residual_variance = residual.square().sum(dim=-1).mean()

            mean_keys[cluster] = mean_key.to(keys.dtype)
            mean_values[cluster] = mean_value.to(values.dtype)
            log_counts[cluster] = math.log(float(member_keys.shape[0]))
            bases[cluster] = local_basis.to(keys.dtype)
            variances[cluster] = eigenvalues[:rank].to(keys.dtype)
            cross[cluster] = local_cross.to(values.dtype)
            tails[cluster] = float(discarded) + float(residual_variance.item())
            radii[cluster] = torch.linalg.vector_norm(centered_keys, dim=-1).max().to(keys.dtype)

        return cls(
            mean_keys=mean_keys.contiguous(),
            mean_values=mean_values.contiguous(),
            log_counts=log_counts.contiguous(),
            basis=bases.contiguous(),
            variances=variances.contiguous(),
            cross=cross.contiguous(),
            tail_variance=tails.contiguous(),
            cluster_radius=radii.contiguous(),
            scale=float(scale),
            token_count=int(keys.shape[0]),
            name=f"gaussian_c{cluster_count}_r{rank}",
        )

    def evaluate_batch(self, queries: Tensor) -> CodecEvaluation:
        qf = queries.float()
        coordinates = torch.einsum("qd,cdr->qcr", qf, self.basis.float())
        linear = self.scale * torch.einsum("qd,cd->qc", qf, self.mean_keys.float())
        quadratic = 0.5 * self.scale**2 * torch.sum(
            self.variances.float().unsqueeze(0) * coordinates.square(), dim=-1
        )
        component_mass = self.log_counts.float().unsqueeze(0) + linear + quadratic
        log_mass = torch.logsumexp(component_mass, dim=-1)
        weights = torch.softmax(component_mass, dim=-1)
        tilted_values = self.mean_values.float().unsqueeze(0) + self.scale * torch.einsum(
            "cdr,qcr->qcd", self.cross.float(), coordinates
        )
        output = torch.sum(weights.unsqueeze(-1) * tilted_values, dim=1)

        qnorm = torch.linalg.vector_norm(qf, dim=-1, keepdim=True)
        per_cluster_proxy = (
            self.scale * qnorm * self.cluster_radius.float().unsqueeze(0)
            + 0.5
            * self.scale**2
            * qnorm.square()
            * self.tail_variance.float().unsqueeze(0)
        )
        proxy = torch.sum(weights * per_cluster_proxy, dim=-1)
        valid = torch.isfinite(output).all(dim=-1) & torch.isfinite(log_mass)
        return CodecEvaluation(
            AttentionSummary(output.to(queries.dtype), log_mass.to(queries.dtype)),
            proxy.to(queries.dtype),
            valid,
        )

    def evaluate(self, query: Tensor) -> CodecEvaluation:
        result = self.evaluate_batch(query.unsqueeze(0))
        return CodecEvaluation(
            AttentionSummary(result.summary.output[0], result.summary.log_mass[0]),
            result.proxy[0],
            result.valid[0],
        )

    def active_numel(self) -> int:
        return tensor_numel(
            (
                self.mean_keys,
                self.mean_values,
                self.log_counts,
                self.basis,
                self.variances,
                self.cross,
                self.tail_variance,
                self.cluster_radius,
            )
        )

    def estimated_flops(self) -> int:
        clusters, dimension, rank = self.basis.shape
        return int(clusters * (5 * dimension * rank + 4 * dimension + 8 * rank))

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "clusters": int(self.mean_keys.shape[0]),
            "rank": int(self.basis.shape[-1]),
        }


@dataclass
class MomentCodec(HeadCodec):
    """Second-order exponential moment state in a low-rank key subspace."""

    mean_key: Tensor
    basis: Tensor
    z_sum: Tensor
    zz_sum: Tensor
    value_sum: Tensor
    vz_sum: Tensor
    vzz_sum: Tensor
    max_projected_norm: Tensor
    max_residual_norm: Tensor
    scale: float
    token_count: int
    name: str = "moment_state"

    def __post_init__(self) -> None:
        self.dimension = int(self.mean_key.numel())

    @classmethod
    def compile(
        cls,
        keys: Tensor,
        values: Tensor,
        *,
        rank: int,
        scale: float,
    ) -> "MomentCodec":
        mean_key = keys.float().mean(dim=0)
        centered = keys.float() - mean_key
        basis, _, _ = principal_basis(centered, rank)
        basis = basis.float()
        z = centered @ basis
        residual = centered - z @ basis.transpose(0, 1)
        vf = values.float()
        return cls(
            mean_key=mean_key.to(keys.dtype),
            basis=basis.to(keys.dtype),
            z_sum=z.sum(dim=0).to(keys.dtype),
            zz_sum=torch.einsum("nr,ns->rs", z, z).to(keys.dtype),
            value_sum=vf.sum(dim=0).to(values.dtype),
            vz_sum=torch.einsum("nd,nr->dr", vf, z).to(values.dtype),
            vzz_sum=torch.einsum("nd,nr,ns->drs", vf, z, z).to(values.dtype),
            max_projected_norm=torch.linalg.vector_norm(z, dim=-1).max().to(keys.dtype),
            max_residual_norm=torch.linalg.vector_norm(residual, dim=-1).max().to(keys.dtype),
            scale=float(scale),
            token_count=int(keys.shape[0]),
            name=f"moment_r{basis.shape[1]}_o2",
        )

    def evaluate_batch(self, queries: Tensor) -> CodecEvaluation:
        qf = queries.float()
        coordinate = self.scale * torch.matmul(qf, self.basis.float())
        denominator = torch.full(
            (queries.shape[0],),
            float(self.token_count),
            dtype=torch.float32,
            device=queries.device,
        )
        denominator = denominator + torch.matmul(coordinate, self.z_sum.float())
        denominator = denominator + 0.5 * torch.einsum(
            "qr,rs,qs->q", coordinate, self.zz_sum.float(), coordinate
        )
        numerator = self.value_sum.float().unsqueeze(0).expand(queries.shape[0], -1).clone()
        numerator = numerator + torch.einsum("dr,qr->qd", self.vz_sum.float(), coordinate)
        numerator = numerator + 0.5 * torch.einsum(
            "drs,qr,qs->qd", self.vzz_sum.float(), coordinate, coordinate
        )
        positive = denominator > 1e-6
        safe_denominator = denominator.clamp_min(1e-6)
        output = numerator / safe_denominator.unsqueeze(-1)
        base = self.scale * torch.mv(qf, self.mean_key.float())
        log_mass = base + torch.log(safe_denominator)

        maximum_argument = (
            torch.linalg.vector_norm(coordinate, dim=-1)
            * self.max_projected_norm.float()
        )
        capped = maximum_argument.clamp(max=12.0)
        taylor_remainder = torch.exp(capped) * maximum_argument.pow(3) / 6.0
        projection_error = (
            self.scale
            * torch.linalg.vector_norm(qf, dim=-1)
            * self.max_residual_norm.float()
        )
        proxy = taylor_remainder + projection_error
        valid = positive & torch.isfinite(output).all(dim=-1) & torch.isfinite(log_mass)
        return CodecEvaluation(
            AttentionSummary(output.to(queries.dtype), log_mass.to(queries.dtype)),
            proxy.to(queries.dtype),
            valid,
        )

    def evaluate(self, query: Tensor) -> CodecEvaluation:
        result = self.evaluate_batch(query.unsqueeze(0))
        return CodecEvaluation(
            AttentionSummary(result.summary.output[0], result.summary.log_mass[0]),
            result.proxy[0],
            result.valid[0],
        )

    def active_numel(self) -> int:
        return tensor_numel(
            (
                self.mean_key,
                self.basis,
                self.z_sum,
                self.zz_sum,
                self.value_sum,
                self.vz_sum,
                self.vzz_sum,
                self.max_projected_norm,
                self.max_residual_norm,
            )
        )

    def estimated_flops(self) -> int:
        dimension, rank = self.basis.shape
        return int(3 * dimension * rank * rank + 4 * dimension * rank + 4 * rank * rank)

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "rank": int(self.basis.shape[-1]), "order": 2}


@dataclass
class HybridResidualCodec(HeadCodec):
    """Compressed background plus a small exact high-influence residual set."""

    base: HeadCodec
    exact_keys: Tensor
    exact_values: Tensor
    scale: float
    token_count: int
    name: str = "hybrid_residual"

    def __post_init__(self) -> None:
        if self.exact_keys.ndim != 2 or self.exact_values.shape != self.exact_keys.shape:
            raise ValueError("hybrid exact residual must have matching [E,D] K/V")
        self.dimension = int(self.exact_keys.shape[-1] if self.exact_keys.numel() else self.base.dimension)
        self.name = f"hybrid_{self.base.name}_e{self.exact_keys.shape[0]}"

    @staticmethod
    def influence_scores(keys: Tensor, values: Tensor, queries: Tensor, scale: float) -> Tensor:
        if queries.shape[0] == 0:
            centered = keys.float() - keys.float().mean(dim=0, keepdim=True)
            return torch.linalg.vector_norm(centered, dim=-1)
        exact = exact_head_batch(queries, keys, values, scale)
        logits = torch.matmul(queries.float(), keys.float().transpose(0, 1)) * float(scale)
        probabilities = torch.softmax(logits, dim=-1)
        value_novelty = torch.linalg.vector_norm(
            values.float().unsqueeze(0) - exact.output.float().unsqueeze(1), dim=-1
        )
        forward_influence = (probabilities * value_novelty).max(dim=0).values
        peak_attention = probabilities.max(dim=0).values
        centered = keys.float() - keys.float().mean(dim=0, keepdim=True)
        leverage = torch.linalg.vector_norm(centered, dim=-1)
        leverage = leverage / leverage.mean().clamp_min(1e-6)
        # Future influence is dominant; peak mass and geometric leverage protect
        # rare keys that a finite query bank may not fully exercise.
        return forward_influence + 0.20 * peak_attention + 0.03 * leverage

    @classmethod
    def compile_gaussian(
        cls,
        keys: Tensor,
        values: Tensor,
        queries: Tensor,
        *,
        exact_slots: int,
        clusters: int,
        rank: int,
        scale: float,
        seed: int,
        iterations: int,
    ) -> "HybridResidualCodec":
        exact_slots = min(max(int(exact_slots), 1), max(int(keys.shape[0]) - 2, 1))
        scores = cls.influence_scores(keys, values, queries, scale)
        selected = torch.topk(scores, exact_slots).indices
        keep = torch.ones(keys.shape[0], dtype=torch.bool, device=keys.device)
        keep[selected] = False
        background_keys = keys[keep]
        background_values = values[keep]
        if background_keys.shape[0] == 0:
            background_keys = keys[:1]
            background_values = values[:1]
        base = GaussianMixtureCodec.compile(
            background_keys,
            background_values,
            clusters=min(int(clusters), int(background_keys.shape[0])),
            rank=rank,
            scale=scale,
            seed=seed,
            iterations=iterations,
        )
        return cls(
            base=base,
            exact_keys=keys.index_select(0, selected).contiguous(),
            exact_values=values.index_select(0, selected).contiguous(),
            scale=float(scale),
            token_count=int(keys.shape[0]),
        )

    def evaluate_batch(self, queries: Tensor) -> CodecEvaluation:
        base = self.base.evaluate_batch(queries)
        exact = exact_head_batch(queries, self.exact_keys, self.exact_values, self.scale)
        masses = torch.stack([base.summary.log_mass, exact.log_mass], dim=0)
        outputs = torch.stack([base.summary.output, exact.output], dim=0)
        merged_mass = torch.logsumexp(masses.float(), dim=0).to(queries.dtype)
        weights = torch.exp(masses.float() - merged_mass.float().unsqueeze(0)).to(outputs.dtype)
        merged_output = torch.sum(weights.unsqueeze(-1) * outputs, dim=0)
        valid = base.valid & torch.isfinite(merged_output).all(dim=-1) & torch.isfinite(merged_mass)
        return CodecEvaluation(
            AttentionSummary(merged_output, merged_mass), base.proxy, valid
        )

    def evaluate(self, query: Tensor) -> CodecEvaluation:
        base = self.base.evaluate(query)
        exact = exact_head_summary(query, self.exact_keys, self.exact_values, self.scale)
        merged = merge_summaries([base.summary, exact])
        valid = base.valid & torch.isfinite(merged.output).all() & torch.isfinite(merged.log_mass)
        return CodecEvaluation(merged, base.proxy, valid)

    def active_numel(self) -> int:
        return self.base.active_numel() + int(self.exact_keys.numel() + self.exact_values.numel())

    def estimated_flops(self) -> int:
        return self.base.estimated_flops() + int(4 * self.exact_keys.numel())

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "base": self.base.metadata(),
            "exact_slots": int(self.exact_keys.shape[0]),
        }
