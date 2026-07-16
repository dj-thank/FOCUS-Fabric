"""Configuration for the archived FOCUS-Native demonstrator.

The development export omitted this module.  The restored package makes the
checkpoint architecture explicit.  Bundled weights are suitable for numerical
Q/K/V and cache-mechanism experiments; the original symbolic tokenizer was not
preserved, so no natural-language capability claim is made for them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 1058
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 384
    max_seq_len: int = 2048
    focus_patches: int = 4
    focus_rank: int = 8
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    memory_code_enabled: bool = False
    memory_code_head: int = 3
    memory_code_scale: float = 8.0
    memory_code_seed: int = 20260417
    memory_router_active: bool = False
    memory_router_temperature: float = 12.0
    memory_router_min_temperature: float = 1.0
    memory_router_max_temperature: float = 64.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> None:
        for name in (
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "d_ff",
            "max_seq_len",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.focus_patches <= 0 or not 0 < self.focus_rank <= self.head_dim:
            raise ValueError("invalid FOCUS patch count/rank")
        if not 0 <= self.memory_code_head < self.n_heads:
            raise ValueError("memory_code_head is out of range")
        if self.memory_router_active and not self.memory_code_enabled:
            raise ValueError("memory router requires memory_code_enabled")
        if not (
            0 < self.memory_router_min_temperature
            <= self.memory_router_temperature
            <= self.memory_router_max_temperature
        ):
            raise ValueError("invalid memory-router temperature bounds")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CacheConfig:
    mode: Literal["exact", "sliding", "focus"] = "focus"
    hot_window: int = 64
    page_size: int = 64
    exact_fallback: bool = True
    dtype_bytes: int = 4

    def validate(self) -> None:
        if self.mode not in {"exact", "sliding", "focus"}:
            raise ValueError(f"unsupported cache mode: {self.mode}")
        if self.hot_window <= 0 or self.page_size <= 0:
            raise ValueError("hot_window and page_size must be positive")
        if self.dtype_bytes not in (1, 2, 4, 8):
            raise ValueError("dtype_bytes must be 1, 2, 4, or 8")
