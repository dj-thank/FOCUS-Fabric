"""Protected semantic memory with provenance-preserving compaction."""
from focus_fabric.agent_memory import MemoryKind, TrajectoryLedger

ledger = TrajectoryLedger()
policy = ledger.append(
    MemoryKind.POLICY,
    "Never publish a benchmark value without an immutable artifact.",
    priority=100,
)
evidence = ledger.append(MemoryKind.EVIDENCE, "results/fabric_benchmark.json")
ledger.append(
    MemoryKind.DECISION,
    "Retain exact fallback under query shift.",
    priority=95,
    evidence_ids=(evidence.record_id,),
)
for index in range(20):
    ledger.append(MemoryKind.TOOL_RESULT, f"transient output {index}")

snapshot = ledger.compact(active_budget=5)
snapshot.verify()
assert policy.record_id in {record.record_id for record in snapshot.active}
print(snapshot.root_digest, len(snapshot.active), len(snapshot.archive))
