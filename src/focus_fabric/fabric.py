"""Online hierarchical FOCUS-Fabric runtime.

Pages form a binary-counter hierarchy.  When two adjacent pages at the same
level merge, the new representation is recompiled from their exact cold
archives—not from already approximated states.  This intentionally trades an
off-hot-path O(N) archive for bounded compaction drift, exact verification, and
sparse recoverability.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import time
from typing import Any, Deque

import torch
from torch import Tensor

from .config import FabricConfig
from .exact import exact_multihead_summary
from .page import FabricPage
from .types import AttentionSummary, FabricStats, merge_summaries


@dataclass
class MemoryFabricLayer:
    config: FabricConfig
    device: torch.device
    dtype: torch.dtype
    hot_keys: list[Tensor] = field(default_factory=list)
    hot_values: list[Tensor] = field(default_factory=list)
    query_bank: Deque[Tensor] = field(default_factory=deque)
    pages: list[FabricPage] = field(default_factory=list)
    next_position: int = 0
    hot_start: int = 0
    stats: FabricStats = field(default_factory=FabricStats)

    def __post_init__(self) -> None:
        self.config.validate()
        if self.query_bank.maxlen is None:
            self.query_bank = deque(self.query_bank, maxlen=self.config.query_bank_size)

    @classmethod
    def create(
        cls,
        config: FabricConfig,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "MemoryFabricLayer":
        return cls(
            config=config,
            device=torch.device(device),
            dtype=dtype,
            query_bank=deque(maxlen=config.query_bank_size),
        )

    def reset(self) -> None:
        self.hot_keys.clear()
        self.hot_values.clear()
        self.query_bank.clear()
        self.pages.clear()
        self.next_position = 0
        self.hot_start = 0
        self.stats = FabricStats()

    def _query_tensor(self) -> Tensor:
        if self.query_bank:
            return torch.stack(list(self.query_bank), dim=1)  # [H,Q,D]
        return torch.empty(
            self.config.n_heads,
            0,
            self.config.head_dim,
            device=self.device,
            dtype=self.dtype,
        )

    def _stack_hot(self) -> tuple[Tensor, Tensor]:
        if not self.hot_keys:
            empty = torch.empty(
                self.config.n_heads,
                0,
                self.config.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            return empty, empty.clone()
        return torch.stack(self.hot_keys, dim=1), torch.stack(self.hot_values, dim=1)

    def observe_query(self, query: Tensor) -> None:
        expected = (self.config.n_heads, self.config.head_dim)
        if query.shape != expected:
            raise ValueError(f"expected query {expected}, got {tuple(query.shape)}")
        self.query_bank.append(query.detach().to(self.device, self.dtype).contiguous())

    def append(self, key: Tensor, value: Tensor) -> None:
        expected = (self.config.n_heads, self.config.head_dim)
        if key.shape != expected or value.shape != expected:
            raise ValueError(
                f"expected keys/values {expected}, got {tuple(key.shape)} / {tuple(value.shape)}"
            )
        self.hot_keys.append(key.detach().to(self.device, self.dtype).contiguous())
        self.hot_values.append(value.detach().to(self.device, self.dtype).contiguous())
        self.next_position += 1
        while len(self.hot_keys) >= self.config.hot_window + self.config.page_size:
            self._compact_oldest()

    def _compile_page(
        self,
        keys: Tensor,
        values: Tensor,
        *,
        start: int,
        end: int,
        level: int,
    ) -> FabricPage:
        started = time.perf_counter()
        page = FabricPage.compile(
            keys,
            values,
            self._query_tensor(),
            start=start,
            end=end,
            level=level,
            scale=self.config.attention_scale,
            config=self.config.compiler,
            seed=self.config.compiler.seed + 104729 * start + 1009 * level,
        )
        self.stats.pages_compiled += 1
        self.stats.compile_seconds += time.perf_counter() - started
        return page

    def _compact_oldest(self) -> None:
        size = self.config.page_size
        keys = torch.stack(self.hot_keys[:size], dim=1)
        values = torch.stack(self.hot_values[:size], dim=1)
        del self.hot_keys[:size]
        del self.hot_values[:size]
        start = self.hot_start
        end = start + size
        self.hot_start = end
        self.pages.append(
            self._compile_page(keys, values, start=start, end=end, level=0)
        )
        self.stats.compacted_tokens += size
        self._merge_equal_levels()

    def _merge_equal_levels(self) -> None:
        while len(self.pages) >= 2 and self.pages[-1].level == self.pages[-2].level:
            right = self.pages.pop()
            left = self.pages.pop()
            if left.end != right.start:
                raise RuntimeError("only adjacent pages can be merged")
            keys = torch.cat([left.exact_keys(), right.exact_keys()], dim=1)
            values = torch.cat([left.exact_values(), right.exact_values()], dim=1)
            self.pages.append(
                self._compile_page(
                    keys,
                    values,
                    start=left.start,
                    end=right.end,
                    level=left.level + 1,
                )
            )
            self.stats.pages_merged += 1

    def attend(self, query: Tensor) -> Tensor:
        expected = (self.config.n_heads, self.config.head_dim)
        if query.shape != expected:
            raise ValueError(f"expected query {expected}, got {tuple(query.shape)}")
        started = time.perf_counter()
        self.stats.queries += 1
        summaries: list[AttentionSummary] = [
            page.evaluate(
                query,
                exact_fallback=self.config.exact_fallback,
                stats=self.stats,
            )
            for page in self.pages
        ]
        hot_keys, hot_values = self._stack_hot()
        summaries.append(
            exact_multihead_summary(
                query, hot_keys, hot_values, self.config.attention_scale
            )
        )
        merged = merge_summaries(summaries)
        self.stats.attention_seconds += time.perf_counter() - started
        return merged.output

    def append_and_attend(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # Decoder self-attention includes the current token in the causal prefix.
        self.observe_query(query)
        self.append(key, value)
        return self.attend(query)

    def active_bytes(self) -> int:
        scalar = self.config.compiler.dtype_bytes
        hot = (
            (len(self.hot_keys) + len(self.hot_values))
            * self.config.n_heads
            * self.config.head_dim
            * scalar
        )
        pages = sum(page.active_bytes() for page in self.pages)
        query_reservoir = (
            len(self.query_bank)
            * self.config.n_heads
            * self.config.head_dim
            * scalar
        )
        return int(hot + pages + query_reservoir)

    def archive_bytes(self) -> int:
        return sum(page.archive_bytes() for page in self.pages)

    def exact_equivalent_bytes(self) -> int:
        return int(
            self.next_position
            * self.config.n_heads
            * self.config.head_dim
            * 2
            * self.config.compiler.dtype_bytes
        )

    def report(
        self,
        *,
        include_pages: bool = True,
        include_candidates: bool = False,
    ) -> dict[str, Any]:
        active = self.active_bytes()
        exact = self.exact_equivalent_bytes()
        representations: Counter[str] = Counter()
        for page in self.pages:
            representations.update(page.representations())
        payload: dict[str, Any] = {
            "tokens": self.next_position,
            "hot_tokens": len(self.hot_keys),
            "query_bank_tokens": len(self.query_bank),
            "pages": len(self.pages),
            "page_levels": [page.level for page in self.pages],
            "max_page_level": max((page.level for page in self.pages), default=-1),
            "representations": dict(representations),
            "active_bytes": active,
            "archive_bytes": self.archive_bytes(),
            "exact_kv_bytes": exact,
            "active_compression": exact / active if active else 1.0,
            **self.stats.as_dict(),
        }
        if include_pages:
            payload["page_reports"] = [
                page.report(include_candidates=include_candidates) for page in self.pages
            ]
        return payload
