from __future__ import annotations

import torch

from focus_fabric.agent_memory import MemoryKind, TrajectoryLedger
from focus_fabric.verified import greedy_decode, verified_block_decode


def test_semantic_compaction_preserves_governance_and_provenance() -> None:
    ledger = TrajectoryLedger()
    policy = ledger.append(
        MemoryKind.POLICY,
        "Never publish an unsupported benchmark claim.",
        priority=100,
    )
    evidence = ledger.append(MemoryKind.EVIDENCE, "results/run-001.json")
    decision = ledger.append(
        MemoryKind.DECISION,
        "Use heterogeneous routing.",
        priority=95,
        evidence_ids=(evidence.record_id,),
    )
    for index in range(20):
        ledger.append(
            MemoryKind.TOOL_RESULT,
            f"tool output {index}",
            priority=index % 3,
        )
    snapshot = ledger.compact(active_budget=5, capsule_group_size=4)
    snapshot.verify()
    active_ids = {record.record_id for record in snapshot.active}
    assert policy.record_id in active_ids
    assert decision.record_id in active_ids
    assert evidence.record_id in active_ids
    assert snapshot.compacted_ids


def test_verified_drafter_emits_exact_greedy_sequence() -> None:
    vocabulary = 7

    def exact(prefix):
        logits = torch.zeros(vocabulary)
        logits[(sum(prefix) + 2) % vocabulary] = 4
        return logits

    def imperfect(prefix):
        logits = exact(prefix).clone()
        return torch.roll(logits, 1) if len(prefix) % 3 == 0 else logits

    prompt = [1, 2]
    expected = greedy_decode(exact, prompt, max_new_tokens=20)
    result = verified_block_decode(
        imperfect,
        exact,
        prompt,
        max_new_tokens=20,
        block_size=5,
    )
    assert result.token_ids == expected
    assert result.corrected_tokens > 0
