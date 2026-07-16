# FOCUS-Fabric agent contract

This repository is a scientific systems project, not a demo-generation task.
Every Codex session and subagent must follow these rules.

## Non-negotiable invariants

1. **Never convert an unmeasured quantity into a result.** CUDA latency, HBM
   bandwidth, official benchmark scores, model intelligence, and SOTA status
   remain `null` until a reproducible artifact contains the measurement.
2. **Exact attention is the source of truth.** Compressed states may draft,
   route, or accelerate; unsupported queries must remain recoverable from the
   exact archive or a deterministic regeneration backend.
3. **Preserve the softmax normalizer.** A page representation that emits only
   a value vector and discards log attention mass is mathematically invalid for
   cross-page composition.
4. **Never tune on the test split.** Representation fitting, selection,
   conformal calibration, and final evaluation use disjoint query sets.
5. **No recursive approximation drift.** Hierarchical merges in the public
   reference recompile from exact cold archives.  Archive-free merge operators
   require a separate, explicitly experimental path and drift test.
6. **Governance memory is not ordinary text.** Policies, constraints,
   commitments, unresolved goals, and their evidence must survive semantic
   compaction verbatim with provenance.
7. **A performance claim needs a claim-ledger entry** identifying command,
   environment, artifact path, SHA-256, metric selector, and allowed wording.
8. **Do not weaken tests or thresholds to make a hypothesis pass.** A failed
   hypothesis is useful evidence and must be logged as such.
9. **Keep changes hypothesis-local.** No opportunistic refactors unrelated to
   the active experiment.
10. **Do not publish or push automatically.** Promotion to the release branch
    is gated; external publication remains an explicit operator action.

## Required workflow

- Read `docs/ARCHITECTURE.md`, `docs/WEAKNESS_AUDIT.md`, and the active
  hypothesis record before editing.
- Delegate independent read-heavy work to specialized subagents.  Avoid
  parallel write-heavy edits to the same files.
- Add or update tests before claiming a mechanism works.
- Run `make gate` before returning a completion result.
- Put raw measurements under `results/`; do not hand-edit generated metrics.
- Update `docs/CLAIMS_LEDGER.json` only through
  `scripts/autonomy/validate_claims.py --refresh-digests`.
- Return failures, uncertainty, and negative results in the final structured
  response.  Do not hide them in prose.

## Repository map

- `src/focus_fabric/`: heterogeneous memory, certificates, semantic memory,
  verified decoding, and model integration.
- `src/focus_native/`: reconstructed FOCUS-Native mechanism demonstrator.
- `scripts/benchmark_fabric.py`: CPU evidence suite.
- `scripts/autonomy/`: Codex orchestration, gates, drift and claim auditing.
- `autonomy/hypotheses.json`: machine-readable research backlog.
- `references/literature_2026-07.json`: curated design evidence.
- `docs/`: architecture, threat model, limitations, claims, and release policy.
