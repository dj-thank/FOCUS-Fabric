"""Lossless speculative decoding with a compressed-cache drafter.

The reference accepts a backend-independent prefix-to-logits oracle.  A
production engine should use cache snapshots and batched block verification,
but the accept/reject invariant is identical: every emitted token is the exact
model's greedy token, even when the compressed drafter is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor


LogitsOracle = Callable[[Sequence[int]], Tensor]


@dataclass(frozen=True)
class VerifiedDecodeResult:
    token_ids: list[int]
    generated_tokens: int
    proposed_tokens: int
    accepted_draft_tokens: int
    corrected_tokens: int
    draft_calls: int
    verifier_calls: int

    @property
    def draft_acceptance_rate(self) -> float:
        return (
            self.accepted_draft_tokens / self.proposed_tokens
            if self.proposed_tokens
            else 0.0
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "generated_tokens": self.generated_tokens,
            "proposed_tokens": self.proposed_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "corrected_tokens": self.corrected_tokens,
            "draft_calls": self.draft_calls,
            "verifier_calls": self.verifier_calls,
            "draft_acceptance_rate": self.draft_acceptance_rate,
        }


def _greedy(logits: Tensor) -> int:
    if logits.ndim != 1:
        raise ValueError("oracle must return one logit vector")
    return int(torch.argmax(logits).item())


def greedy_decode(
    oracle: LogitsOracle,
    prompt: Sequence[int],
    *,
    max_new_tokens: int,
    eos_id: int | None = None,
) -> list[int]:
    prefix = [int(token) for token in prompt]
    for _ in range(max_new_tokens):
        token = _greedy(oracle(prefix))
        prefix.append(token)
        if eos_id is not None and token == eos_id:
            break
    return prefix


def verified_block_decode(
    draft_oracle: LogitsOracle,
    exact_oracle: LogitsOracle,
    prompt: Sequence[int],
    *,
    max_new_tokens: int,
    block_size: int = 4,
    eos_id: int | None = None,
) -> VerifiedDecodeResult:
    """Emit the exact greedy sequence while measuring draft usefulness."""

    if block_size <= 0 or max_new_tokens < 0:
        raise ValueError("block_size must be positive and max_new_tokens non-negative")
    prefix = [int(token) for token in prompt]
    prompt_length = len(prefix)
    proposed = accepted = corrected = draft_calls = verifier_calls = 0
    while len(prefix) - prompt_length < max_new_tokens:
        remaining = max_new_tokens - (len(prefix) - prompt_length)
        proposal_count = min(block_size, remaining)
        draft_prefix = list(prefix)
        block: list[int] = []
        for _ in range(proposal_count):
            token = _greedy(draft_oracle(draft_prefix))
            draft_calls += 1
            proposed += 1
            block.append(token)
            draft_prefix.append(token)
            if eos_id is not None and token == eos_id:
                break

        mismatch = False
        for token in block:
            exact_token = _greedy(exact_oracle(prefix))
            verifier_calls += 1
            if exact_token == token:
                prefix.append(token)
                accepted += 1
            else:
                prefix.append(exact_token)
                corrected += 1
                mismatch = True
            if eos_id is not None and prefix[-1] == eos_id:
                return VerifiedDecodeResult(
                    prefix,
                    len(prefix) - prompt_length,
                    proposed,
                    accepted,
                    corrected,
                    draft_calls,
                    verifier_calls,
                )
            if mismatch or len(prefix) - prompt_length >= max_new_tokens:
                break
    return VerifiedDecodeResult(
        prefix,
        len(prefix) - prompt_length,
        proposed,
        accepted,
        corrected,
        draft_calls,
        verifier_calls,
    )
