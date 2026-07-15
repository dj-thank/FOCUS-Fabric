# Claims and Non-Claims

## Supported, narrowly scoped claims

The committed evidence supports the following statements **for the exact configurations in the artifacts**:

1. A heterogeneous per-head compiler can select multiple representation families and evaluate finite attention summaries containing both value output and log-mass.
2. On the controlled four-regime attention field, active state is compressed 11.452× and in-distribution output NMSE is 5.11794387e-05.
3. On that field, the Fabric NMSE is below the committed memory-matched operator and query-aware coreset baselines.
4. Forced exact fallback reproduces exact page attention within floating-point tolerance in tests and randomized holdout.
5. In-distribution split-conformal coverage is 0.9688 for a 0.95 target; shifted coverage is lower and is not hidden.
6. The repaired symbolic checkpoint runs through the Fabric path with 1.0000 teacher-forced argmax agreement on the committed 64-token case and exact agreement on the committed 8-token greedy generation.
7. Binary-counter compaction runs repeatedly without invalid codec outputs in the committed 128-token controlled case.
8. The typed semantic ledger retained all protected records and rejected prose-only policy promotion in its deterministic adversarial substrate benchmark.
9. Codex autonomy artifacts implement worktree isolation, allowed-file checks, deterministic gates, hash-chained events, and a post-hoc randomized root-vs-candidate evaluator. Execute mode itself was not run in this container because Codex CLI was absent.

These statements are machine-linked in `CLAIMS_LEDGER.json`.

## Explicitly unsupported

This release does **not** establish:

- a model with superior general intelligence;
- superior natural-language benchmark quality;
- official LongBench, RULER, BABILong, LifeBench, or tool-agent scores;
- GPU speedup, physical HBM reduction, or production throughput;
- exactness without a cold source of truth;
- universal robustness to distribution shift;
- million-token stability;
- privacy, confidentiality, or authorization properties;
- autonomous publication safety.

## Comparison wording

“Better” may be used only with the exact named baseline, metric, configuration, and artifact. For example, “lower controlled output NMSE than the committed memory-matched coreset” is valid. “Better long-context LLM” is not.

“Exact” refers only to one of:

- algebraic merge of exact disjoint summaries up to floating-point rounding;
- a forced fallback that reads exact K/V;
- the greedy verified decoder relative to its exact oracle;
- checkpoint exact-cache parity within stated numerical tolerance.

It must not be used for an approximate codec alone.
