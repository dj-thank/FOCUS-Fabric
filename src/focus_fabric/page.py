"""Compiled heterogeneous cache pages."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .config import CompilerConfig
from .controller import AdaptiveHead, compile_adaptive_head
from .types import AttentionSummary, FabricStats


@dataclass
class FabricPage:
    start: int
    end: int
    level: int
    heads: list[AdaptiveHead]
    dtype_bytes: int

    @property
    def token_count(self) -> int:
        return int(self.end - self.start)

    @classmethod
    def compile(
        cls,
        keys: Tensor,
        values: Tensor,
        query_bank: Tensor,
        *,
        start: int,
        end: int,
        level: int,
        scale: float,
        config: CompilerConfig,
        seed: int,
    ) -> "FabricPage":
        """Compile ``keys/values[H,N,D]`` using ``query_bank[H,Q,D]``."""

        if keys.ndim != 3 or values.shape != keys.shape:
            raise ValueError("page keys/values must have matching shape [H,N,D]")
        if query_bank.ndim != 3 or query_bank.shape[0] != keys.shape[0]:
            raise ValueError("query_bank must have shape [H,Q,D]")
        if end - start != keys.shape[1]:
            raise ValueError("page position range is inconsistent with token count")
        heads = [
            compile_adaptive_head(
                keys[head],
                values[head],
                query_bank[head],
                scale=scale,
                config=config,
                seed=seed + 10007 * head,
            )
            for head in range(keys.shape[0])
        ]
        return cls(start, end, level, heads, config.dtype_bytes)

    def evaluate(
        self,
        query: Tensor,
        *,
        exact_fallback: bool,
        stats: FabricStats | None = None,
    ) -> AttentionSummary:
        if query.ndim != 2 or query.shape[0] != len(self.heads):
            raise ValueError("query must have shape [H,D]")
        outputs: list[Tensor] = []
        masses: list[Tensor] = []
        for head_index, head in enumerate(self.heads):
            decision = head.evaluate(query[head_index], exact_fallback=exact_fallback)
            outputs.append(decision.summary.output)
            masses.append(decision.summary.log_mass)
            if stats is not None:
                stats.page_head_evaluations += 1
                stats.certificate_upper_sum += decision.certificate_upper
                if decision.used_fallback:
                    stats.fallback_decisions += 1
                    stats.fallback_tokens += head.token_count
                    stats.archive_bytes_read += head.archive_bytes()
                if not torch.isfinite(decision.summary.output).all() or not torch.isfinite(
                    decision.summary.log_mass
                ):
                    stats.invalid_codec_outputs += 1
        return AttentionSummary(torch.stack(outputs), torch.stack(masses))

    def exact_keys(self) -> Tensor:
        return torch.stack([head.exact_keys for head in self.heads], dim=0)

    def exact_values(self) -> Tensor:
        return torch.stack([head.exact_values for head in self.heads], dim=0)

    def active_bytes(self) -> int:
        return sum(head.active_bytes() for head in self.heads)

    def archive_bytes(self) -> int:
        return sum(head.archive_bytes() for head in self.heads)

    def representations(self) -> Counter[str]:
        families: Counter[str] = Counter()
        for head in self.heads:
            name = head.codec.name
            if name.startswith("weighted_coreset"):
                family = "coreset"
            elif name.startswith("operator"):
                family = "operator"
            elif name.startswith("gaussian"):
                family = "gaussian"
            elif name.startswith("moment"):
                family = "moment"
            elif name.startswith("hybrid"):
                family = "hybrid"
            else:
                family = name.split("_")[0]
            families[family] += 1
        return families

    def report(self, *, include_candidates: bool = False) -> dict[str, Any]:
        head_reports: list[dict[str, Any]] = []
        for index, head in enumerate(self.heads):
            report = head.report()
            if not include_candidates:
                report.pop("candidates", None)
            report["head"] = index
            head_reports.append(report)
        return {
            "start": self.start,
            "end": self.end,
            "level": self.level,
            "tokens": self.token_count,
            "active_bytes": self.active_bytes(),
            "archive_bytes": self.archive_bytes(),
            "representations": dict(self.representations()),
            "heads": head_reports,
        }
