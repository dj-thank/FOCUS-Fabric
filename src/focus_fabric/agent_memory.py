"""Typed, provenance-preserving compaction for long-horizon agent traces.

Numerical KV compression and semantic agent memory solve different failure
modes.  This module refuses to let an opaque summarizer decide that a standing
policy, unresolved constraint, commitment, or its supporting evidence may be
forgotten.  Records are immutable and hash chained; protected classes survive
compaction verbatim; extractive capsules retain source manifests; and the
append-only archive remains the source of truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Sequence


class MemoryKind(str, Enum):
    POLICY = "policy"
    CONSTRAINT = "constraint"
    GOAL = "goal"
    DECISION = "decision"
    COMMITMENT = "commitment"
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PROGRESS = "progress"
    EVIDENCE = "evidence"
    CAPSULE = "capsule"


PROTECTED_KINDS = {
    MemoryKind.POLICY,
    MemoryKind.CONSTRAINT,
    MemoryKind.GOAL,
    MemoryKind.COMMITMENT,
}


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryRecord:
    sequence: int
    kind: MemoryKind
    content: str
    priority: int = 0
    parent_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    resolved: bool = False
    previous_digest: str = "GENESIS"
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    record_id: str = ""
    digest: str = ""

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        kind: MemoryKind,
        content: str,
        priority: int = 0,
        parent_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        resolved: bool = False,
        previous_digest: str = "GENESIS",
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> "MemoryRecord":
        if not content.strip():
            raise ValueError("memory content cannot be empty")
        body = {
            "sequence": int(sequence),
            "kind": kind.value,
            "content": content,
            "priority": int(priority),
            "parent_ids": list(parent_ids),
            "evidence_ids": list(evidence_ids),
            "resolved": bool(resolved),
            "previous_digest": previous_digest,
            "metadata": metadata or {},
        }
        digest = _digest(body)
        record_id = f"m{sequence:08d}-{digest[:12]}"
        return cls(
            sequence=sequence,
            kind=kind,
            content=content,
            priority=priority,
            parent_ids=tuple(parent_ids),
            evidence_ids=tuple(evidence_ids),
            resolved=resolved,
            previous_digest=previous_digest,
            metadata=dict(metadata or {}),
            record_id=record_id,
            digest=digest,
        )

    def verify(self) -> bool:
        body = {
            "sequence": self.sequence,
            "kind": self.kind.value,
            "content": self.content,
            "priority": self.priority,
            "parent_ids": list(self.parent_ids),
            "evidence_ids": list(self.evidence_ids),
            "resolved": self.resolved,
            "previous_digest": self.previous_digest,
            "metadata": self.metadata,
        }
        return self.digest == _digest(body) and self.record_id == (
            f"m{self.sequence:08d}-{self.digest[:12]}"
        )


@dataclass(frozen=True)
class CompactionSnapshot:
    active: tuple[MemoryRecord, ...]
    archive: tuple[MemoryRecord, ...]
    compacted_ids: tuple[str, ...]
    protected_ids: tuple[str, ...]
    root_digest: str

    def verify(self) -> None:
        by_id = {record.record_id: record for record in self.archive}
        if len(by_id) != len(self.archive):
            raise ValueError("duplicate record IDs in archive")
        previous = "GENESIS"
        for record in self.archive:
            if not record.verify():
                raise ValueError(f"record digest invalid: {record.record_id}")
            if record.previous_digest != previous:
                raise ValueError(f"hash chain broken at {record.record_id}")
            previous = record.digest
        expected_root = self.archive[-1].digest if self.archive else "GENESIS"
        if self.root_digest != expected_root:
            raise ValueError("root digest does not match archive head")
        active_ids = {record.record_id for record in self.active}
        missing = set(self.protected_ids) - active_ids
        if missing:
            raise ValueError(f"protected records were compacted away: {sorted(missing)}")
        for record in self.active:
            if not record.verify():
                raise ValueError(f"active record digest invalid: {record.record_id}")
            if record.kind == MemoryKind.CAPSULE:
                source_manifest = record.metadata.get("source_ids", "")
                if not isinstance(source_manifest, str) or not source_manifest:
                    raise ValueError("capsule has no source manifest")
                source_ids = source_manifest.split(",")
                if any(source_id not in by_id for source_id in source_ids):
                    raise ValueError("capsule references an unknown source record")
                expected_digest = _digest([by_id[source].digest for source in source_ids])
                if record.metadata.get("source_digest") != expected_digest:
                    raise ValueError("capsule source digest does not match archive")


@dataclass
class TrajectoryLedger:
    """Append-only typed event ledger with deterministic safe compaction."""

    records: list[MemoryRecord] = field(default_factory=list)

    @property
    def root_digest(self) -> str:
        return self.records[-1].digest if self.records else "GENESIS"

    def append(
        self,
        kind: MemoryKind,
        content: str,
        *,
        priority: int = 0,
        parent_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        resolved: bool = False,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> MemoryRecord:
        known = {record.record_id for record in self.records}
        missing = (set(parent_ids) | set(evidence_ids)) - known
        if missing:
            raise ValueError(f"record references unknown dependencies: {sorted(missing)}")
        record = MemoryRecord.create(
            sequence=len(self.records),
            kind=kind,
            content=content,
            priority=priority,
            parent_ids=parent_ids,
            evidence_ids=evidence_ids,
            resolved=resolved,
            previous_digest=self.root_digest,
            metadata=metadata,
        )
        self.records.append(record)
        return record

    def _protected_ids(self) -> set[str]:
        protected = {
            record.record_id
            for record in self.records
            if (record.kind in PROTECTED_KINDS and not record.resolved)
            or record.priority >= 90
        }
        by_id = {record.record_id: record for record in self.records}
        changed = True
        while changed:
            changed = False
            for record_id in tuple(protected):
                record = by_id[record_id]
                for dependency in (*record.parent_ids, *record.evidence_ids):
                    if dependency not in protected:
                        protected.add(dependency)
                        changed = True
        return protected

    def compact(
        self,
        *,
        active_budget: int,
        capsule_group_size: int = 8,
    ) -> CompactionSnapshot:
        if active_budget < 1 or capsule_group_size < 1:
            raise ValueError("budgets must be positive")
        protected = self._protected_ids()
        unprotected = [
            record for record in self.records if record.record_id not in protected
        ]
        keep_slots = max(0, active_budget - len(protected))
        recent = sorted(
            unprotected,
            key=lambda item: (item.priority, item.sequence),
            reverse=True,
        )[:keep_slots]
        recent_ids = {record.record_id for record in recent}
        compacted = [
            record for record in unprotected if record.record_id not in recent_ids
        ]

        capsules: list[MemoryRecord] = []
        for start in range(0, len(compacted), capsule_group_size):
            group = compacted[start : start + capsule_group_size]
            source_ids = [record.record_id for record in group]
            capsules.append(
                MemoryRecord.create(
                    sequence=-(start // capsule_group_size + 1),
                    kind=MemoryKind.CAPSULE,
                    content="\n".join(
                        f"[{record.kind.value}] {record.content}" for record in group
                    ),
                    priority=max((record.priority for record in group), default=0),
                    parent_ids=source_ids,
                    metadata={
                        "source_ids": ",".join(source_ids),
                        "source_digest": _digest(
                            [record.digest for record in group]
                        ),
                    },
                )
            )
        protected_records = [
            record for record in self.records if record.record_id in protected
        ]
        active = tuple(
            sorted([*protected_records, *recent], key=lambda item: item.sequence)
            + capsules
        )
        snapshot = CompactionSnapshot(
            active=active,
            archive=tuple(self.records),
            compacted_ids=tuple(record.record_id for record in compacted),
            protected_ids=tuple(sorted(protected)),
            root_digest=self.root_digest,
        )
        snapshot.verify()
        return snapshot

    def to_jsonl(self) -> str:
        return "\n".join(
            _canonical_json({**asdict(record), "kind": record.kind.value})
            for record in self.records
        )
