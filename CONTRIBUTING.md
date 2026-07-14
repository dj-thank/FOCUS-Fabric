# Contributing

FOCUS-Fabric accepts falsifiable mechanism improvements, reproductions, negative results, and systems integrations.

Before opening a change:

1. Read `AGENTS.md`, `docs/CLAIMS.md`, and `docs/WEAKNESS_AUDIT.md`.
2. Add or update an entry in `autonomy/hypotheses.json` for material architecture experiments.
3. Keep fit, model-selection, calibration, and final-test data disjoint.
4. Include a memory-matched baseline and exact fallback check.
5. Add tests for every changed numerical invariant.
6. Store measurements in JSON and register public quantitative statements in `docs/CLAIMS_LEDGER.json`.
7. Run `make gate`.

Do not refresh evidence hashes to conceal an unexplained metric change. Document failures that materially influenced the design.

For GPU changes, provide hardware/driver/Triton versions, PyTorch differential tests, p50/p95 latency, and physical HBM counters when making bandwidth claims.
