# Publication Status — 2026-07-17

## Decision

**Ready as a `0.2.1` unsigned research-preview candidate; not ready as a production inference engine or superior-LLM claim.**

The `0.2.1` candidate contains an installable package, repaired checkpoint mechanism, heterogeneous memory compiler, online hierarchy, fallback and monitoring controls, semantic agent ledger, verified greedy decoder, FOCUS-Native loss, optional Triton kernel, Hugging Face SDPA trace collector, Codex experiment orchestrator, tests, CI, evidence artifacts, and a machine-verifiable claim ledger. A locally retained pre-publication `0.2.0` sdist candidate was quarantined after archive inspection found excluded `.safetensors` checkpoint payloads. It must not be published. The successor pipeline exports exact clean Git `HEAD` to an isolated build directory, excludes common model-weight suffixes recursively and case-insensitively, and verifies wheel, sdist, clean-target imports, and generated source-ZIP members. Candidate archives are explicitly labeled; final generation fetches canonical GitHub `origin/main` and requires `HEAD` to match the fetched commit. The exact `0.2.1` source commit, manifest, Python-distribution hashes, and source-ZIP checksums are fixed only when the reviewed release assets are built.

## Evidence classes

| Evidence class | Status | Retained artifact |
|---|---|---|
| Controlled heterogeneous attention | Complete on CPU | `results/fabric_benchmark.json` |
| Randomized hidden holdout | Three retained post-hoc seeds; complete on CPU | `results/randomized_holdout_suite.json` |
| Learned local checkpoint Q/K/V traces | Complete for archived symbolic mechanism checkpoint | `results/fabric_benchmark.json` |
| Repeated online compaction | Complete for the committed 128-token controlled run | `results/fabric_benchmark.json` |
| End-to-end token/logit agreement | Complete for the committed symbolic token-ID run | `results/fabric_benchmark.json` |
| Typed agent-memory substrate | Complete for deterministic structural attacks | `results/agent_memory_benchmark.json` |
| Codex orchestration | Public dry-run retained; local H001 live cycle completed, but the unpromoted candidate is rejected as holdout-insensitive under the strengthened policy | Public: `results/autonomy_dry_run.json`; detailed live report remains ignored local audit data |
| External pretrained model | SDPA trace collector implemented; no external weights were available locally | no score claimed |
| Official LongBench/RULER/BABILong/LifeBench | Not executed | explicit `null` fields |
| CUDA/Triton correctness, latency, HBM | Not executed | `results/gpu_benchmark.json` |

## Blocking gates for stronger wording

A stronger release title such as “production-ready,” “faster,” “nearly lossless on long-context LLMs,” or “superior model” requires all of the following:

1. integration with named modern pretrained checkpoints and tokenizer-preserving generation;
2. official benchmark scoring across retrieval, reasoning, long decoding, agent state tracking, and tool-use trajectories;
3. CUDA kernel parity, p50/p95 latency, achieved bandwidth, allocation, and physical HBM counter measurements on named hardware;
4. million-token and repeated-compaction stress tests with drift/fallback accounting;
5. external reproduction from the source distribution;
6. signed provenance and human publication authorization.

## Deliberately retained negative evidence

- The CPU reference path is slower than vectorized exact attention.
- Split-conformal coverage degrades under distribution shift.
- A randomized holdout exposed catastrophic single-seed K-means instability; the compiler was redesigned with query-aware multi-start selection.
- On one learned layer, the FOCUS-Native operator has lower unguarded NMSE than the selected Fabric codec; the portfolio is not universally dominant.
- The public reversible hierarchy retains an O(N) exact cold source of truth.
- The original symbolic tokenizer was not present in the archived FOCUS-Native export.
- A live autonomous candidate improved the fixed public benchmark but produced no measurable effect in any of four paired randomized holdout cases; it was not promoted, and the evaluator now rejects this failure mode.
- The pre-publication `0.2.0` sdist candidate included local checkpoint weights despite the intended public exclusion; it is quarantined rather than relabeled as publishable evidence.
