# Weakness Audit

This document is intentionally adversarial. A release is useful only if its failure surfaces are easier to find than its claims.

## A. Evidence and model scale

| Weakness | Observed consequence | Mitigation in this release | Remaining work |
|---|---|---|---|
| No external production LLM weights were available | Learned-trace results come from a ~1M-parameter symbolic checkpoint, not a modern instruction model | Trace source and tokenizer loss are stated in every evidence artifact | Integrate Llama/Qwen/Gemma-class checkpoints and publish per-layer traces |
| Original symbolic tokenizer was absent from the old export | Natural-language capability cannot be reproduced from the checkpoint | A compatibility byte tokenizer repairs the package API but is explicitly non-equivalent | Recover original mapping or retrain from a fully versioned tokenizer |
| No official LongBench/RULER/BABILong data run | No official task score exists | Backend-neutral paired exact/FOCUS runner emits predictions for official scorers | Run official suites at multiple lengths and budgets on the same model |
| End-to-end generation test is short | 64 teacher-forced tokens and 8 free-running tokens can miss late divergence | Exact token and logit telemetry is committed; verified decode is available | Evaluate thousands of generated tokens, long CoT, and tool trajectories |

## B. Numerical approximation

### Distribution shift invalidates naive certificate interpretation

In distribution, empirical certificate coverage was 0.9688 for a 0.95 target. Under the deliberately shifted query set it fell to 0.8073. This is not a contradiction: split conformal coverage assumes exchangeability. It is direct evidence that a static certificate cannot be marketed as a universal per-query guarantee.

Mitigation: exact fallback, sparse audits, and `DriftSentinel`. Remaining weakness: the sentinel is a control mechanism, not proof against abrupt adversarial shifts before sufficient audits accumulate.

### Fallback can become the dominant cost

Shift fallback reached 25.52%. A learned-trace scaled-shift case can reach still higher rates. Exactness is recovered at the cost of cold-memory traffic; a system that hides this traffic would misreport performance.

Mitigation: archive bytes are counted from actual selected-head tensors. Remaining work: asynchronous prefetch, residual-stream regeneration, and I/O-aware routing.

### Log-normalizer error can matter even when output NMSE is low

A codec can match its local value output while misestimating mass, changing its weight when pages are merged. The benchmark therefore records both quantities and the compiler preserves `log_mass`. Remaining work: add global merged-output loss across multiple simultaneous pages, not only isolated-page selection.

### Exact archive is still O(N)

The active path compresses memory, but reversibility keeps exact cold K/V in the public reference. This shifts, rather than erases, information storage. Residual regeneration may reduce archive size, but it was not integrated or measured in this environment.

### Recompilation cost is high

The CPU Python reference recompiles candidates, runs multiple K-means restarts, and performs exact selection/calibration. It is intentionally correctness-first and slower than vectorized exact attention. The cost should be moved to asynchronous compaction/sleep cycles or amortized kernels.

### Randomized holdout exposed optimizer instability

Before query-aware multi-start selection, a single unlucky K-means seed produced mean Fabric NMSE about 0.0524 on the retained holdout. After the repair it became 3.1504659e-05. Three retained seeds totaling 11 controlled cases all passed, but this still does not establish robustness outside the synthetic generator family. The remaining risk is broader hyperparameter and regime overfitting; promotion therefore compares root and candidate on a seed generated only after Codex finishes.

### Local selection is not global allocation

Current selection is per page/head under a local ratio target. It does not solve a global constrained optimization over all layers, heads, decode phases, or device tiers. A Lagrangian/knapsack controller and online dual updates are preregistered work.

## C. Repeated compaction

The hierarchy avoids recursive approximation by rebuilding from exact archives. This controls drift but increases compile work and requires archive availability. The 128-token controlled run had mean relative attention error 0.008742, maximum 0.028922, and 4 merges. It is far too short to establish million-token stability.

No power-loss recovery, distributed archive consistency, or concurrent compaction protocol is implemented. A production design needs copy-on-write page manifests, checksummed atomic commits, and replay testing.

## D. GPU and systems performance

CUDA and Triton were unavailable. Consequently:

- fused-kernel correctness on GPU is unmeasured;
- p50/p95 kernel and end-to-end latency are unmeasured;
- occupancy and register pressure are unknown;
- physical HBM counters are unknown;
- PCIe/CXL/NVMe fallback behavior is unknown.

`results/gpu_benchmark.json` contains explicit `null` fields rather than estimates relabeled as measurements. The Python reference was slower than exact attention and must not be used to claim speedup.

## E. Semantic/tool memory

The typed ledger preserves protected records structurally, but it does not decide whether a newly proposed statement deserves `policy` status. Only trusted code may assign protected types. Text inside a tool result cannot promote itself, but a compromised caller can still misclassify it.

Hash chaining detects mutation but does not prove truth, freshness, or authorization. Extractive capsules retain sources yet may still overwhelm retrieval or omit unprotected but later-relevant details. Long-horizon reasoning over the ledger has not been evaluated on LifeBench, MemoryAgentBench, AppWorld, SWE-EVO, or comparable interactive suites.

## F. Verified decode

Greedy sequence equality is guaranteed only when the verifier oracle itself is the exact target model and every proposed token is checked in order. Sampling distributions require rejection-sampling/speculative-decoding math, not the greedy proof. The reference implementation rebuilds prefixes, so it demonstrates correctness but not acceleration.

## G. Autonomous Codex operation

- Codex CLI was not installed in this container, so only dry-run planning—not a live autonomous code mutation—can be executed here.
- An agent could optimize the public benchmark; the post-hoc randomized holdout reduces accidental overfitting but is not secure against a deliberately evasive candidate.
- An agent can modify allowed production code and tests. Scope enforcement, immutable root evaluator, claim hashes, and review agents raise the bar but do not replace external review.
- Autonomous repositories accumulate duplicated patterns and benchmark-specific hacks. The pipeline includes drift scans and cleanup hypotheses, but long-term architectural entropy remains an open risk.
- `--auto-promote` is opt-in. Publication, release signing, and deployment must remain separately authorized.

## H. Security and privacy

Exact archives may contain secrets, personal data, or copyrighted context. No encryption-at-rest, retention policy, redaction, tenant isolation, or secure deletion is included. Tool outputs and benchmark files are untrusted input. Running arbitrary external backends requires OS-level sandboxing beyond Python validation.

## I. Claims that are prohibited

The current artifacts do not support claims of:

- superior general intelligence;
- official benchmark leadership;
- production-scale natural-language quality;
- GPU or HBM speedup;
- million-token correctness;
- universal conformal guarantees;
- archive-free exactness;
- security against a malicious host or model.
