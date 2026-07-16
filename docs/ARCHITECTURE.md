# Architecture

## 1. Design objective

FOCUS-Fabric treats long-context memory as a constrained control problem rather than a token-retention heuristic. For every disjoint page and KV head, it must approximate two functions of a future query \(q\):

\[
F_B(q)=\frac{\sum_{i\in B}e^{\beta q^\top k_i}v_i}{\sum_{i\in B}e^{\beta q^\top k_i}},
\qquad
s_B(q)=\log\sum_{i\in B}e^{\beta q^\top k_i}.
\]

The pair is an `AttentionSummary(output, log_mass)`. Summaries from disjoint regions are combined exactly by `logsumexp` and mass-weighted output mixing. This prevents the common but incorrect operation of averaging independently normalized page outputs.

## 2. Memory planes

### Exact truth plane

- A recent hot window remains exact.
- Every compacted page retains a cold exact K/V archive in the public reference implementation.
- An alternative backend may store residual-stream checkpoints and regenerate K/V, but the public code does not silently pretend that archive-free reconstruction exists.
- Exact fallback is performed per page and head.

### Active heterogeneous plane

The rate–distortion compiler considers:

- **Operator codec**: local low-rank Jacobian for \(F_B\) and reduced Hessian for \(s_B\).
- **Weighted coreset**: centroid K/V, multiplicity, and radius. Compilation uses query-aware multi-start selection because one K-means initialization proved catastrophically unstable under randomized holdout.
- **Gaussian/cumulant codec**: mixture mass via first/second key cumulants and value–key cross-covariance.
- **Moment state**: merge-friendly second-order projected exponential moments.
- **Hybrid residual codec**: a smooth Gaussian state plus a small exact set chosen using attention residual and leverage.

No family is assumed dominant. A page can use different codecs across heads, and separate layers can produce unrelated mixtures.

### Hierarchical compaction plane

New exact tokens accumulate behind a hot window. Full pages are compiled at level 0. Two adjacent pages of the same level are merged like a binary counter. Crucially, a merged page is rebuilt from the two pages' exact archives, not from their compressed outputs. This makes repeated compaction expensive off-path but avoids unbounded approximation-on-approximation drift.

## 3. Compiler and objective

Query traces are divided into disjoint fitting, selection, and conformal-calibration sets. Candidate construction may use the fitting split. Selection uses exact attention on the selection split. Certificates see only the calibration split.

For candidate \(c\), the reference objective is:

\[
J(c)=E_{95}(c)+\lambda_R\,R(c)+\lambda_F\,C(c)+P_{budget}+P_{invalid},
\]

where \(E_{95}\) is the 95th percentile of relative output error plus a weighted normalized log-mass error, \(R\) is active-byte ratio, and \(C\) is an estimated compute ratio. All candidate metrics and Pareto status are retained for audit.

This is a local controller, not yet a globally optimal layer/head budget allocator. Global allocation is a registered autonomous-research hypothesis.

## 4. Error control

Each selected codec emits an error proxy. Split conformal calibration fits a scalar multiplier to the observed error/proxy ratios. At runtime:

1. Evaluate the active codec.
2. Convert its proxy into a calibrated marginal upper bound.
3. If the bound exceeds tolerance, or the codec result is invalid, read exact K/V for that page/head.
4. Merge the resulting exact or approximate summary with all other disjoint regions.

The certificate is marginal under exchangeability. It is not a mathematical per-query proof. A `DriftSentinel` therefore schedules sparse exact audits, maintains a finite-window Hoeffding interval for miscoverage, and requests strict fallback plus recompilation when the lower confidence bound exceeds the target.

## 5. FOCUS-Native training

`focus_native_loss` exposes both an exact teacher path and differentiable compressed-prefix student path. Its terms include:

- exact next-token loss;
- compressed-path next-token loss;
- suffix logit distillation;
- attention-output NMSE;
- log-normalizer MSE;
- discarded Jacobian singular-energy ratio;
- route-distance penalty.

The archived checkpoint predates the complete heterogeneous controller. This loss is supplied for new training runs; no claim is made that the archived model was trained with every term.

## 6. Semantic agent memory

Numerical KV preservation does not ensure that policies, unresolved goals, evidence, or commitments survive context summarization. `TrajectoryLedger` supplies a separate typed control plane:

- append-only hash-chained records;
- protected classes (`policy`, `constraint`, `goal`, `commitment`);
- transitive preservation of cited evidence;
- extractive capsules with source ID and source-digest manifests;
- deterministic validation after every compaction.

The hash chain detects mutation; it does not establish external truth or signer identity. Deployments needing authenticity must sign roots and protect keys outside the model workspace.

## 7. Verified generation

`verified_block_decode` allows any compressed-memory oracle to propose greedy token blocks. An exact oracle verifies tokens sequentially. A mismatch emits the exact token and drops the unverified suffix. By induction over the emitted prefix, the resulting sequence is identical to exact greedy decoding. The current backend-neutral implementation rebuilds prefixes and is for correctness testing; a production engine needs cache snapshots and rollback.

## 8. GPU path

The release includes a homogeneous-batch fused Triton ABI for Gaussian codecs and a PyTorch reference. Heterogeneous pages are intended to be grouped by codec shape before dispatch. CUDA/Triton were unavailable in the build environment, so the kernel is syntax-checked but not runtime-validated here.

## 9. Codex research control plane

Each hypothesis is preregistered with allowed files, expected mechanism, falsifier, resource budget, and a paired-holdout policy. `run_codex_loop.py` creates a separate Git worktree, invokes `codex exec` with JSONL events and an output schema, rejects out-of-scope changes, runs deterministic gates, and then invokes a trusted post-hoc randomized evaluator outside the worktree. Promotion requires safety invariants, matched case/seed evidence, no aggregate or case-level holdout regression, a measurable holdout effect, and an evidence-score improvement. The event ledger is hash-chained.
