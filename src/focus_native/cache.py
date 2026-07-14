"""Hierarchical runtime cache for FOCUS-Native autoregressive decoding.

The active tier contains exact hot K/V plus compact query-conditioned page
operators.  Raw page K/V are retained in an archive tier so unsupported
queries can fall back to exact attention and adjacent pages can be recompiled
into progressively larger pages.  This is deliberately a reference
implementation: correctness and instrumentation are prioritized over speed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .config import CacheConfig
from .kernels.focus_eval import focus_page_eval

if TYPE_CHECKING:  # pragma: no cover
    from .model import FocusSelfAttention


@dataclass
class RuntimeStats:
    """Mutable cache telemetry collected during one generation run."""

    queries: int = 0
    page_evaluations: int = 0
    route_decisions: int = 0
    fallback_decisions: int = 0
    fallback_tokens_fetched: int = 0
    pages_compiled: int = 0
    pages_merged: int = 0
    compacted_tokens: int = 0
    compaction_seconds: float = 0.0
    attention_seconds: float = 0.0
    archive_bytes_read: int = 0

    def as_dict(self) -> dict[str, float | int]:
        fallback_rate = (
            self.fallback_decisions / self.route_decisions
            if self.route_decisions
            else 0.0
        )
        return {
            "queries": self.queries,
            "page_evaluations": self.page_evaluations,
            "route_decisions": self.route_decisions,
            "fallback_decisions": self.fallback_decisions,
            "fallback_rate": fallback_rate,
            "fallback_tokens_fetched": self.fallback_tokens_fetched,
            "pages_compiled": self.pages_compiled,
            "pages_merged": self.pages_merged,
            "compacted_tokens": self.compacted_tokens,
            "compaction_seconds": self.compaction_seconds,
            "attention_seconds": self.attention_seconds,
            "archive_bytes_read": self.archive_bytes_read,
        }


@dataclass
class CompiledPage:
    """One functional page and its cold exact archive payload."""

    start: int
    end: int
    level: int
    compiled: dict[str, Tensor]
    archive_keys: Tensor
    archive_values: Tensor

    @property
    def token_count(self) -> int:
        return self.end - self.start

    def active_numel(self) -> int:
        # tail_ratio is a compiler diagnostic, not a per-page runtime field.
        return sum(
            value.numel()
            for name, value in self.compiled.items()
            if name != "tail_ratio"
        )

    def archive_numel(self) -> int:
        return self.archive_keys.numel() + self.archive_values.numel()


@dataclass
class LayerCache:
    """Per-layer exact, sliding-window, or hierarchical FOCUS cache."""

    config: CacheConfig
    n_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype
    archive_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    hot_keys: list[Tensor] = field(default_factory=list)
    hot_values: list[Tensor] = field(default_factory=list)
    pages: list[CompiledPage] = field(default_factory=list)
    next_position: int = 0
    stats: RuntimeStats = field(default_factory=RuntimeStats)

    def __post_init__(self) -> None:
        self.config.validate()

    @classmethod
    def create(
        cls,
        config: CacheConfig,
        n_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
        archive_device: torch.device | str = "cpu",
    ) -> "LayerCache":
        return cls(
            config=config,
            n_heads=n_heads,
            head_dim=head_dim,
            device=torch.device(device),
            dtype=dtype,
            archive_device=torch.device(archive_device),
        )

    def reset(self) -> None:
        self.hot_keys.clear()
        self.hot_values.clear()
        self.pages.clear()
        self.next_position = 0
        self.stats = RuntimeStats()

    def _stack_hot(self) -> tuple[Tensor, Tensor]:
        if not self.hot_keys:
            empty = torch.empty(
                self.n_heads, 0, self.head_dim, device=self.device, dtype=self.dtype
            )
            return empty, empty.clone()
        return torch.stack(self.hot_keys, dim=1), torch.stack(self.hot_values, dim=1)

    @torch.no_grad()
    def _compile_archive_page(
        self,
        attention: "FocusSelfAttention",
        archive_keys: Tensor,
        archive_values: Tensor,
        *,
        start: int,
        end: int,
        level: int,
    ) -> CompiledPage:
        t0 = time.perf_counter()
        keys = archive_keys.to(device=self.device, dtype=self.dtype, non_blocking=False)
        values = archive_values.to(device=self.device, dtype=self.dtype, non_blocking=False)
        compiled = attention.compile_page(keys.unsqueeze(0), values.unsqueeze(0))
        runtime_compiled = {
            name: value.detach().contiguous()
            for name, value in compiled.items()
            if name != "tail_ratio"
        }
        self.stats.pages_compiled += 1
        self.stats.compaction_seconds += time.perf_counter() - t0
        return CompiledPage(
            start=start,
            end=end,
            level=level,
            compiled=runtime_compiled,
            archive_keys=archive_keys.contiguous(),
            archive_values=archive_values.contiguous(),
        )

    @torch.no_grad()
    def _compact_one_page(self, attention: "FocusSelfAttention") -> None:
        size = self.config.page_size
        keys = torch.stack(self.hot_keys[:size], dim=1)
        values = torch.stack(self.hot_values[:size], dim=1)
        del self.hot_keys[:size]
        del self.hot_values[:size]
        archive_keys = keys.to(self.archive_device).contiguous()
        archive_values = values.to(self.archive_device).contiguous()
        start = self.pages[-1].end if self.pages else 0
        page = self._compile_archive_page(
            attention,
            archive_keys,
            archive_values,
            start=start,
            end=start + size,
            level=0,
        )
        self.pages.append(page)
        self.stats.compacted_tokens += size
        self._merge_equal_levels(attention)

    @torch.no_grad()
    def _merge_equal_levels(self, attention: "FocusSelfAttention") -> None:
        # Binary-counter compaction bounds active page count by O(log N).
        while len(self.pages) >= 2 and self.pages[-1].level == self.pages[-2].level:
            right = self.pages.pop()
            left = self.pages.pop()
            if left.end != right.start:
                raise RuntimeError("attempted to merge non-contiguous cache pages")
            archive_keys = torch.cat([left.archive_keys, right.archive_keys], dim=1)
            archive_values = torch.cat([left.archive_values, right.archive_values], dim=1)
            merged = self._compile_archive_page(
                attention,
                archive_keys,
                archive_values,
                start=left.start,
                end=right.end,
                level=left.level + 1,
            )
            self.pages.append(merged)
            self.stats.pages_merged += 1

    @torch.no_grad()
    def append(self, key: Tensor, value: Tensor, attention: "FocusSelfAttention") -> None:
        """Append one token's [H,D] key/value and compact when needed."""
        if key.shape != (self.n_heads, self.head_dim) or value.shape != key.shape:
            raise ValueError(
                f"expected K/V {(self.n_heads, self.head_dim)}, got {tuple(key.shape)} / {tuple(value.shape)}"
            )
        self.hot_keys.append(key.detach().to(self.device, self.dtype).contiguous())
        self.hot_values.append(value.detach().to(self.device, self.dtype).contiguous())
        self.next_position += 1

        if self.config.mode == "sliding":
            while len(self.hot_keys) > self.config.hot_window:
                self.hot_keys.pop(0)
                self.hot_values.pop(0)
        elif self.config.mode == "focus":
            while len(self.hot_keys) >= self.config.hot_window + self.config.page_size:
                self._compact_one_page(attention)

    @staticmethod
    def _exact_summary(query: Tensor, keys: Tensor, values: Tensor, scale: float) -> tuple[Tensor, Tensor]:
        # query [H,D], keys/values [H,N,D]
        if keys.shape[1] == 0:
            output = torch.zeros_like(query)
            logmass = torch.full(
                (query.shape[0],), -torch.inf, device=query.device, dtype=query.dtype
            )
            return output, logmass
        scores = torch.einsum("hd,hnd->hn", query, keys) * scale
        logmass = torch.logsumexp(scores.float(), dim=-1).to(query.dtype)
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        output = torch.einsum("hn,hnd->hd", probabilities, values)
        return output, logmass

    @torch.no_grad()
    def _page_summary(
        self,
        query: Tensor,
        page: CompiledPage,
        attention: "FocusSelfAttention",
    ) -> tuple[Tensor, Tensor]:
        q = query.unsqueeze(0).unsqueeze(2)  # [1,H,1,D]
        output, logmass, route, distance = attention.evaluate_page(q, page.compiled)
        output = output[0, :, 0]
        logmass = logmass[0, :, 0]
        route = route[0, :, 0]
        distance = distance[0, :, 0]

        radii = attention.focus_radius.gather(1, route.unsqueeze(-1)).squeeze(-1)
        unsupported = distance > radii
        self.stats.page_evaluations += 1
        self.stats.route_decisions += self.n_heads
        self.stats.fallback_decisions += int(unsupported.sum().item())

        if self.config.exact_fallback and unsupported.any():
            # Transfer and evaluate only unsupported heads.  This makes the
            # reference implementation's archive-byte telemetry match the
            # actual tensor payload rather than assuming an idealized sparse
            # fetch while copying every head.
            selected = unsupported.nonzero(as_tuple=False).flatten()
            archive_index = selected.to(page.archive_keys.device)
            archive_keys = page.archive_keys.index_select(0, archive_index).to(
                self.device, self.dtype
            )
            archive_values = page.archive_values.index_select(0, archive_index).to(
                self.device, self.dtype
            )
            exact_output, exact_logmass = self._exact_summary(
                query.index_select(0, selected), archive_keys, archive_values, attention.scale
            )
            output = output.clone()
            logmass = logmass.clone()
            output.index_copy_(0, selected, exact_output)
            logmass.index_copy_(0, selected, exact_logmass)
            selected_heads = selected.numel()
            self.stats.fallback_tokens_fetched += selected_heads * page.token_count
            self.stats.archive_bytes_read += (
                archive_keys.numel() + archive_values.numel()
            ) * self.config.dtype_bytes
        return output, logmass

    @torch.no_grad()
    def _functional_summaries(
        self, query: Tensor, attention: "FocusSelfAttention"
    ) -> tuple[Tensor, Tensor]:
        """Evaluate every active page through the fused operator ABI.

        On CUDA this dispatches the Triton kernel; on CPU it uses the exact
        same packed tensor interface through the PyTorch reference.  Packing
        is performed here for clarity.  A serving implementation should retain
        these page-first tensors and update them only after compaction.
        """
        if not self.pages:
            return (
                torch.empty(0, self.n_heads, self.head_dim, device=self.device, dtype=self.dtype),
                torch.empty(0, self.n_heads, device=self.device, dtype=self.dtype),
            )
        fields = {
            name: torch.cat([page.compiled[name] for page in self.pages], dim=0).contiguous()
            for name in (
                "output0", "logmass0", "left", "gradient", "hessian_small"
            )
        }
        output, logmass, route, distance = focus_page_eval(
            query.contiguous(),
            attention.focus_anchors.contiguous(),
            attention.basis().contiguous(),
            fields["output0"],
            fields["logmass0"],
            fields["left"],
            fields["gradient"],
            fields["hessian_small"],
        )
        pages = len(self.pages)
        expanded_radii = attention.focus_radius.unsqueeze(0).expand(pages, -1, -1)
        radii = expanded_radii.gather(2, route.unsqueeze(-1)).squeeze(-1)
        unsupported = distance > radii
        self.stats.page_evaluations += pages
        self.stats.route_decisions += pages * self.n_heads
        self.stats.fallback_decisions += int(unsupported.sum().item())

        if self.config.exact_fallback and unsupported.any():
            output = output.clone()
            logmass = logmass.clone()
            for page_index, page in enumerate(self.pages):
                selected = unsupported[page_index].nonzero(as_tuple=False).flatten()
                if selected.numel() == 0:
                    continue
                archive_index = selected.to(page.archive_keys.device)
                archive_keys = page.archive_keys.index_select(0, archive_index).to(
                    self.device, self.dtype
                )
                archive_values = page.archive_values.index_select(0, archive_index).to(
                    self.device, self.dtype
                )
                exact_output, exact_logmass = self._exact_summary(
                    query.index_select(0, selected),
                    archive_keys,
                    archive_values,
                    attention.scale,
                )
                output[page_index].index_copy_(0, selected, exact_output)
                logmass[page_index].index_copy_(0, selected, exact_logmass)
                selected_heads = selected.numel()
                self.stats.fallback_tokens_fetched += selected_heads * page.token_count
                self.stats.archive_bytes_read += (
                    archive_keys.numel() + archive_values.numel()
                ) * self.config.dtype_bytes
        return output, logmass

    @torch.no_grad()
    def attend(self, query: Tensor, attention: "FocusSelfAttention") -> Tensor:
        """Return one-token attention output [H,D] over the current cache."""
        if query.shape != (self.n_heads, self.head_dim):
            raise ValueError(f"expected query {(self.n_heads, self.head_dim)}, got {tuple(query.shape)}")
        t0 = time.perf_counter()
        self.stats.queries += 1
        hot_keys, hot_values = self._stack_hot()
        hot_output, hot_logmass = self._exact_summary(
            query, hot_keys, hot_values, attention.scale
        )
        if self.config.mode == "focus" and self.pages:
            page_outputs, page_logmasses = self._functional_summaries(query, attention)
            outputs = torch.cat([page_outputs, hot_output.unsqueeze(0)], dim=0)
            logmasses = torch.cat([page_logmasses, hot_logmass.unsqueeze(0)], dim=0)
        else:
            outputs = hot_output.unsqueeze(0)
            logmasses = hot_logmass.unsqueeze(0)
        merged_logmass = torch.logsumexp(logmasses.float(), dim=0).to(query.dtype)
        mixture = torch.exp(logmasses - merged_logmass.unsqueeze(0)).unsqueeze(-1)
        merged = torch.sum(mixture * outputs, dim=0)
        self.stats.attention_seconds += time.perf_counter() - t0
        return merged

    @torch.no_grad()
    def append_and_attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention: "FocusSelfAttention",
    ) -> Tensor:
        # Self-attention includes the current token, so append before querying.
        self.append(key, value, attention)
        return self.attend(query, attention)

    def active_numel(self) -> int:
        hot = (len(self.hot_keys) + len(self.hot_values)) * self.n_heads * self.head_dim
        return hot + sum(page.active_numel() for page in self.pages)

    def archive_numel(self) -> int:
        return sum(page.archive_numel() for page in self.pages)

    def exact_equivalent_numel(self) -> int:
        return self.next_position * self.n_heads * self.head_dim * 2

    def memory_report(self) -> dict[str, float | int]:
        active_bytes = self.active_numel() * self.config.dtype_bytes
        archive_bytes = self.archive_numel() * self.config.dtype_bytes
        exact_bytes = self.exact_equivalent_numel() * self.config.dtype_bytes
        return {
            "tokens": self.next_position,
            "hot_tokens": len(self.hot_keys),
            "pages": len(self.pages),
            "max_page_level": max((page.level for page in self.pages), default=-1),
            "page_levels": [page.level for page in self.pages],
            "active_bytes": active_bytes,
            "archive_bytes": archive_bytes,
            "exact_kv_bytes": exact_bytes,
            "active_compression": exact_bytes / active_bytes if active_bytes else 1.0,
            **self.stats.as_dict(),
        }


@dataclass
class ModelCache:
    """Collection of per-layer caches with aggregate telemetry."""

    layers: list[LayerCache]

    @classmethod
    def create(
        cls,
        *,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        config: CacheConfig,
        device: torch.device | str,
        dtype: torch.dtype,
        archive_device: torch.device | str = "cpu",
    ) -> "ModelCache":
        return cls([
            LayerCache.create(
                config=config,
                n_heads=n_heads,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
                archive_device=archive_device,
            )
            for _ in range(n_layers)
        ])

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def memory_report(self) -> dict[str, Any]:
        reports = [layer.memory_report() for layer in self.layers]
        numeric_sum_keys = [
            "active_bytes", "archive_bytes", "exact_kv_bytes", "queries",
            "page_evaluations", "route_decisions", "fallback_decisions",
            "fallback_tokens_fetched", "pages_compiled", "pages_merged",
            "compacted_tokens", "compaction_seconds", "attention_seconds",
            "archive_bytes_read",
        ]
        aggregate: dict[str, Any] = {
            key: sum(float(report[key]) for report in reports) for key in numeric_sum_keys
        }
        exact_bytes = aggregate["exact_kv_bytes"]
        active_bytes = aggregate["active_bytes"]
        aggregate["active_compression"] = exact_bytes / active_bytes if active_bytes else 1.0
        aggregate["fallback_rate"] = (
            aggregate["fallback_decisions"] / aggregate["route_decisions"]
            if aggregate["route_decisions"]
            else 0.0
        )
        aggregate["layers"] = reports
        return aggregate
