# Evaluation Contract

## Evidence classes

Results are separated because they answer different questions.

1. **Unit and invariant tests** — algebraic merge, exact fallback, checkpoint parity, training gradients, semantic-memory integrity, verified decode, and drift state transitions.
2. **Controlled heterogeneous attention** — tests whether a per-head portfolio can beat memory-matched single-family codecs on a deliberately mixed field.
3. **Randomized holdout** — discovers initialization/regime overfitting after implementation decisions.
4. **Learned Q/K/V traces** — uses actual projections from the repaired archived checkpoint, but random token IDs and a small symbolic model.
5. **Repeated compaction** — measures error and telemetry across binary-counter merges.
6. **End-to-end token/logit agreement** — compares exact cache and Fabric inside the same model.
7. **Semantic-memory substrate** — tests structural retention and provenance under repeated noise/injection.
8. **External official suites** — LongBench, RULER, BABILong, long tool trajectories, and GPU profiling; not executed in this environment.

These classes must not be collapsed into one “model quality” number.

## Controlled benchmark

Configuration: four heads, 192 tokens/head, dimension 16, 144 compile queries, 128 in-distribution test queries, and 48 shifted test queries. The head regimes are smooth low-rank, clustered/cumulant, diffuse moment, and rare-residual.

Memory accounting includes every active codec tensor and certificate scalars. The cold exact archive is reported separately. Memory-matched baselines receive approximately the same per-head active byte allowance.

### Current accepted results

| Metric | Fabric | Operator-only | Weighted coreset |
|---|---:|---:|---:|
| Active bytes | 8,584 | 7,356 | 8,296 |
| In-distribution output NMSE | 5.11794387e-05 | 0.0877658129 | 0.000440167583 |
| Shift output NMSE | 0.00169266551 | 0.161744982 | 0.00381286792 |

Exact active KV is 98,304 bytes, giving 11.452× active compression. Guarded shift NMSE is 0.00147937366 with fallback rate 0.255.

Certificate coverage is 0.9688 in distribution and 0.8073 under shift. The latter is deliberately retained as a failure signal.

## Randomized holdout

The retained suite contains three independently seeded runs totaling 11 controlled attention cases. `results/randomized_holdout_suite.json` is a deterministic aggregation of the three raw artifacts. All three runs passed the safety condition, the worst run-level Fabric-to-best-single-family NMSE ratio was 0.098802554956204, forced exact fallback had maximum absolute error 0.0, and no invalid outputs occurred. The suite is a mechanism stress test only; it cannot substitute for natural-language or official long-context benchmarks.


The trusted evaluator chooses cases after an autonomous code change. It compares root and candidate source on the same seed and verifies forced exact fallback. The committed self-test reports:

- mean Fabric NMSE: 3.15046594e-05;
- mean best memory-matched single-family NMSE: 0.000353266912;
- Fabric/reference ratio: 0.089181;
- maximum forced-fallback absolute error: 0.0e+00;
- invalid outputs: 0.

The historical pre-fix value is documented but not placed in the accepted claim ledger because it is not an accepted artifact.

## Learned checkpoint traces

The trace cache consists of 128 tokens/head with 64 future test queries. It is not an official language benchmark.

| Layer | Active compression | Fabric future NMSE | FOCUS-Native operator NMSE | Guarded future NMSE | Guarded fallback |
|---:|---:|---:|---:|---:|---:|
| 0 | 13.114× | 0.000161036529 | 0.00034484925 | 7.03984697e-05 | 0.042 |
| 3 | 12.629× | 0.00013939575 | 2.1833268e-05 | 6.95370836e-05 | 0.052 |

Layer 3 is an important counterexample: the unguarded FOCUS-Native operator is more accurate than the selected Fabric approximation. The portfolio is not guaranteed to dominate every head/layer. Guarding reduces error by reading exact state.

## Repeated compaction

The accepted controlled run processes 128 tokens and performs 4 page merges. Mean, p95, and maximum relative attention errors are 0.008742, 0.019830, and 0.028922. Active compression is 2.029× after including the bounded query reservoir. Fallback rate is 0.112.

## End-to-end model path

Teacher-forced token-ID evaluation covers 64 positions. Argmax agreement is 1.0000; maximum and mean logit absolute errors are 0.131632 and 0.009616. Free-running greedy agreement is 1.0000 over 8 generated tokens.

The Fabric path is much slower in this Python CPU reference. Compilation time is counted. No speedup claim follows from token equality.

## Semantic-memory benchmark

The structural benchmark runs 20 seeds, 20 compaction cycles/seed, and 25 noise events/cycle. It measures protected retention, hash-chain validation, and whether injected prose can become a policy record without typed authority.

It does not measure whether an LLM can answer questions, complete tools, or infer habits from the resulting memory.

## Official suite runner

`scripts/evaluation/run_official_benchmarks.py` accepts local JSON/JSONL rows and invokes the same backend in `exact` and `focus` mode. It records text, optional token IDs, latency, fallback, active/exact bytes, archive traffic, generic diagnostic EM/F1, and exact-vs-Focus agreement.

Generic EM/F1 is explicitly not the official aggregate. Predictions must be passed to each suite's official scorer.

Required release matrix for a production claim:

- LongBench and LongBench v2 by task and context length;
- RULER all 13 tasks and configurable lengths;
- BABILong by task, context size, and supporting-fact count;
- long-form generation with token agreement and semantic grading;
- long tool-use/CoT with task success, policy retention, evidence accuracy, and compaction count;
- full KV, sliding window, strongest eviction/quantization baselines, the FOCUS-Native operator, and Fabric;
- at least three seeds or confidence intervals;
- GPU p50/p95, prefill/decode split, archive I/O, and Nsight physical counters.

## Reproduction

```bash
make gate
make benchmark
make agent-memory
make gpu-benchmark
```

Every accepted scalar used in prose is registered in `docs/CLAIMS_LEDGER.json` with a JSON path and SHA-256 of its source artifact.
