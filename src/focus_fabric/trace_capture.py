"""Capture the Q/K/V tensors actually presented to PyTorch SDPA.

This is a narrow integration hook for research traces.  It captures tensors at
the scaled-dot-product-attention boundary, after architecture-specific
projection and positional transforms.  It does not claim support for custom
FlashAttention/flex-attention kernels that bypass
``torch.nn.functional.scaled_dot_product_attention``.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Any, Iterator

import torch
from torch import Tensor
import torch.nn.functional as F


_CAPTURE_LOCK = threading.Lock()


@dataclass(frozen=True)
class SDPATrace:
    query: Tensor
    key: Tensor
    value: Tensor
    is_causal: bool
    enable_gqa: bool

    @property
    def query_heads(self) -> int:
        return int(self.query.shape[-3])

    @property
    def key_value_heads(self) -> int:
        return int(self.key.shape[-3])

    def expanded_kv(self) -> tuple[Tensor, Tensor]:
        """Repeat grouped K/V heads to query-head cardinality when necessary."""

        if self.query_heads == self.key_value_heads:
            return self.key, self.value
        if self.query_heads % self.key_value_heads:
            raise ValueError(
                f"query heads {self.query_heads} are not divisible by KV heads {self.key_value_heads}"
            )
        repeat = self.query_heads // self.key_value_heads
        return self.key.repeat_interleave(repeat, dim=-3), self.value.repeat_interleave(repeat, dim=-3)


@contextmanager
def capture_sdpa_traces(
    *,
    detach: bool = True,
    move_to_cpu: bool = True,
    clone: bool = True,
) -> Iterator[list[SDPATrace]]:
    """Temporarily intercept calls to PyTorch SDPA.

    The patch is process-global and therefore protected by a non-reentrant lock.
    Use it only around a single-threaded model forward.  The original function
    is restored even when the forward raises.
    """

    acquired = _CAPTURE_LOCK.acquire(blocking=False)
    if not acquired:
        raise RuntimeError("another SDPA capture is already active in this process")
    original = F.scaled_dot_product_attention
    traces: list[SDPATrace] = []

    def materialize(tensor: Tensor) -> Tensor:
        result = tensor.detach() if detach else tensor
        result = result.cpu() if move_to_cpu else result
        return result.clone() if clone else result

    def wrapper(query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        is_causal = bool(kwargs.get("is_causal", False))
        enable_gqa = bool(kwargs.get("enable_gqa", False))
        # is_causal may be supplied positionally: (attn_mask, dropout_p,
        # is_causal, scale, enable_gqa).
        if len(args) >= 3:
            is_causal = bool(args[2])
        if len(args) >= 5:
            enable_gqa = bool(args[4])
        traces.append(
            SDPATrace(
                materialize(query),
                materialize(key),
                materialize(value),
                is_causal=is_causal,
                enable_gqa=enable_gqa,
            )
        )
        return original(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = wrapper
    try:
        yield traces
    finally:
        F.scaled_dot_product_attention = original
        _CAPTURE_LOCK.release()
