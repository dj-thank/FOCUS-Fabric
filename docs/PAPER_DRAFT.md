# FOCUS-Fabric: Heterogeneous, Reversible, and Evidence-Gated Memory for Long-Context Decoders

**Public research draft — 2026-07-14**

**Project creator:** [dj-thank](https://github.com/dj-thank)

## Abstract

Long-context decoder inference is commonly treated as a uniform key–value (KV)
retention problem. That abstraction is incomplete: different heads, layers,
pages, and future query regimes exhibit different compressibility, and a
locally accurate value estimate is not composable unless its softmax mass is
also preserved. We present **FOCUS-Fabric**, a correctness-first research
architecture that compiles each old-context page/head into one member of a
heterogeneous representation portfolio: a local low-rank attention-response
operator, a weighted KV coreset, a Gaussian/cumulant state, a projected moment
state, or a smooth state with an exact sparse residual. Candidate fitting,
selection, and split-conformal calibration use disjoint query traces. A cold
exact archive remains the source of truth for sparse fallback, repeated
compaction rebuilds from exact state rather than recursively compressing
approximations, and disjoint page responses compose through their value output
and log-normalizer.

The release also separates numerical attention memory from long-horizon agent
memory. Policies, constraints, unresolved goals, commitments, and evidence are
stored as typed hash-chained records and are not eligible for ordinary prose
compaction. An exact greedy verifier can correct blocks proposed by a
compressed-memory drafter. Finally, a Codex research harness preregisters
hypotheses, isolates changes in Git worktrees, enforces file scope, runs
reproducible gates, evaluates post-hoc randomized holdouts, and maps every
quantitative public claim to a hashed artifact.

In a controlled four-regime attention field, FOCUS-Fabric used 8,584 active
bytes versus 98,304 bytes for exact KV (11.452× active compression) and obtained
5.1179e-05 in-distribution output NMSE. Memory-matched operator-only and
weighted-coreset references obtained 8.7766e-02 and 4.4017e-04, respectively.
Across three retained randomized runs totaling 11 controlled cases, every run
passed the safety gate; the worst run-level Fabric-to-best-single-family NMSE
ratio was 0.0988, forced exact fallback had zero measured error, and no invalid
outputs occurred. A repaired approximately one-million-parameter symbolic
checkpoint preserved 100% argmax agreement over 64 teacher-forced positions and
8 free-running greedy tokens in the accepted CPU run. These results validate
mechanisms, not superior general language-model intelligence, production GPU
speed, or official long-context benchmark leadership.

## 1. Problem statement

For one disjoint old-context block \(B\), scaled dot-product attention can be
written as two query-dependent functions:

\[
F_B(q)=\frac{\sum_{i\in B}\exp(\beta q^\top k_i)v_i}
               {\sum_{i\in B}\exp(\beta q^\top k_i)},
\qquad
s_B(q)=\log\sum_{i\in B}\exp(\beta q^\top k_i).
\]

A useful compressed block must estimate both \(F_B\) and \(s_B\). Given
summaries for disjoint blocks \(A\) and \(B\), their exact composition is

\[
s_{A\cup B}=\operatorname{logaddexp}(s_A,s_B),
\]

\[
F_{A\cup B}=e^{s_A-s_{A\cup B}}F_A
            +e^{s_B-s_{A\cup B}}F_B.
\]

This identity is the algebraic contract of the implementation. A cache that
stores only independently normalized value vectors cannot, in general, recover
the correct global attention output.

We seek a runtime representation \(c\) that minimizes held-out distortion while
respecting active-memory and compute budgets:

\[
J(c)=Q_{0.95}(E_c)+\lambda_R R_c+\lambda_C C_c
     +P_{\text{budget}}+P_{\text{invalid}},
\]

where \(E_c\) combines relative value-output error and normalized log-mass
error, \(R_c\) is active byte ratio, and \(C_c\) is an estimated evaluation-cost
ratio. The objective is local in the public implementation; globally coupled
layer/device allocation remains future work.

## 2. Why one representation family is insufficient

A single approximation family embeds a strong and usually hidden assumption
about future attention geometry. Local Taylor operators work well when the
query-to-response map is smooth and locally low rank. Vector-quantized KV
coresets work well when a small number of representative key/value atoms
preserve mass. Gaussian or cumulant states work well for clustered key
populations with locally approximately exponential tilting. Moment states are
merge-friendly when many diffuse contributions live in a low-dimensional
subspace. Rare identifiers and high-leverage facts may instead require sparse
exact residuals.

The release therefore treats FOCUS operators as one codec, not as a universal
replacement for KV. The compiler preserves all candidate metrics and Pareto
status, so a chosen representation can be audited rather than inferred from an
opaque policy.

## 3. Representation portfolio

### 3.1 Local response operator

At anchor \(a_j\), the codec stores \(F(a_j)\), \(s(a_j)\), a low-rank
right-subspace approximation of the Jacobian of \(F\), the gradient of \(s\),
and a reduced Hessian for \(s\). With \(\delta=q-a_j\):

\[
\widehat F(q)=F(a_j)+L_jB_j^\top\delta,
\]

\[
\widehat s(q)=s(a_j)+g_j^\top\delta+
\tfrac12(B_j^\top\delta)^\top H_j(B_j^\top\delta).
\]

Discarded Jacobian/Hessian spectral norms, anchor distance, and support radius
feed the error proxy.

### 3.2 Weighted KV coreset

Keys are clustered with query-aware multi-start selection. Each slot stores a
mean key, mean value, log multiplicity, and cluster radius. The approximated
score includes log multiplicity, preserving aggregate mass. Multi-start
selection was added only after a randomized holdout exposed a catastrophic
single-initialization failure.

### 3.3 Gaussian/cumulant mixture

For each cluster, the codec stores mean key/value, count, a low-rank covariance
basis and variances, value–key cross covariance, discarded variance, and a
radius. A second-order cumulant expansion estimates component log mass; the
cross covariance estimates the exponentially tilted value. This is a
semiparametric local model rather than an assumption that all attention is
Gaussian.

### 3.4 Projected moment state

The moment codec stores sufficient statistics through second order in a
projected key coordinate system, including value–coordinate cross moments.
Its evaluation uses a truncated exponential expansion. It is compact and
structurally merge-friendly, but may produce invalid denominators outside its
trusted regime; invalid outputs fail closed to exact state.

### 3.5 Exact-residual hybrid

A smooth Gaussian state models the background while a small exact set retains
high attention-residual and high-leverage tokens. The two summaries are
composed with the exact softmax-mass identity. The hybrid makes a deliberate
trade: spend a few exact slots on discontinuous or identity-sensitive events
rather than increasing the order of a global smooth approximation.

## 4. Compilation and statistical control

Query traces are split into fitting, model-selection, and calibration subsets.
Candidate construction sees only the fitting split. Exact attention on the
selection split determines the rate–distortion objective. A split-conformal
multiplier is fit on calibration error/proxy ratios. For a runtime proxy
\(p(q)\), the stored upper estimate is

\[
\widehat U(q)=\widehat q_{1-\alpha}(E/(p+\epsilon))\,(p(q)+\epsilon).
\]

Under exchangeability this yields a finite-sample marginal statement for the
chosen scalar nonconformity score. It is not a conditional, adversarial, or
per-query proof. The accepted benchmark deliberately records coverage falling
from 0.9688 in distribution to 0.8073 under query shift. This failure motivates
sparse exact audits and a drift sentinel that requests strict fallback and
recompilation when a confidence bound on miscoverage exceeds target.

## 5. Online memory hierarchy

Recent tokens remain exact. Once the hot window exceeds a page boundary, the
oldest exact page is compiled. Two adjacent pages at the same level are merged
as in a binary counter. The public reference rebuilds the parent from the two
exact child archives. It does not compress a compressed state, avoiding
unbounded approximation-on-approximation drift. This makes the exact archive
\(O(N)\), shifts compilation off the hot path, and explicitly separates active
HBM pressure from total information retention.

An alternative deployment may regenerate K/V from residual-stream checkpoints,
but this backend is not implemented or claimed by the release.

## 6. Native training objective

For future training runs, an exact teacher and a differentiable
compressed-prefix student are optimized jointly:

\[
\mathcal L=\lambda_e\mathcal L_{\text{LM,exact}}
+\lambda_f\mathcal L_{\text{LM,fabric}}
+\lambda_d\|z_f-z_e\|_2^2
+\lambda_o E_{\text{attn-output}}
+\lambda_m E_{\text{log-mass}}
+\lambda_t E_{\text{Jacobian-tail}}
+\lambda_r E_{\text{route-distance}}.
\]

The supplied function backpropagates through the legacy differentiable FOCUS
path. The archived checkpoint predates the complete heterogeneous controller,
so no claim is made that it was trained with this full objective.

## 7. Semantic memory for tool-using agents

Numerical cache fidelity does not guarantee preservation of standing policy,
commitments, unresolved goals, or evidence. `TrajectoryLedger` therefore uses
an append-only hash chain with typed records. Policy, constraint, goal, and
commitment classes survive ordinary compaction verbatim; evidence dependencies
are preserved transitively. Extractive capsules include source IDs and a digest
manifest. Untrusted tool prose cannot self-promote into a protected class
because authority is carried by the typed write operation, not by textual
content.

This substrate detects mutation and provenance loss; it does not prove that a
record is true, authorized by a real person, or fresh. Production use requires
external signatures, access control, privacy policy, and trusted time.

## 8. Verified decoding

A compressed-memory oracle can propose a block of greedy tokens. An exact
oracle verifies each token in prefix order. Matching tokens are accepted; a
mismatch emits the exact token and discards the unverified suffix. Induction on
the emitted prefix proves equality with exact greedy decoding, provided the
verifier is the exact target model. The current backend rebuilds prefixes and
exists for correctness testing. Efficient deployment needs cache snapshots,
rollback, and batched verification. Sampling requires standard speculative
rejection logic rather than the greedy proof.

## 9. Autonomous research harness

The repository contains a Codex control plane with the following invariants:

1. Each hypothesis includes a falsifier and allowed file scope.
2. A fresh Git worktree isolates the candidate.
3. `codex exec` emits JSONL events and a schema-constrained result.
4. Out-of-scope modifications reject the run.
5. Syntax, tests, CPU evidence, claim integrity, and drift gates execute before
   comparison.
6. Only after the agent finishes does the trusted root generate a randomized
   holdout seed and evaluate root and candidate on identical cases.
7. Promotion requires exactness constraints, no randomized holdout regression,
   and a minimum public-evidence improvement; auto-promotion is opt-in.
8. Events are hash chained and quantitative prose is backed by JSON Pointer,
   artifact SHA-256, and permitted wording.

The harness reduces accidental benchmark overfitting and scope drift. It is not
a security boundary against a malicious process with host access, and it does
not substitute for independent scientific review.

## 10. Experimental evidence

### 10.1 Controlled heterogeneous field

The accepted case has four heads, 192 tokens per head, head dimension 16, 144
compile queries, 128 in-distribution queries, and 48 shifted queries. The heads
are constructed to express smooth low-rank, clustered/cumulant, diffuse
moment, and rare-residual regimes.

| Metric | FOCUS-Fabric | Operator-only | Weighted coreset |
|---|---:|---:|---:|
| Active bytes | 8,584 | 7,356 | 8,296 |
| In-distribution output NMSE | 5.11794e-05 | 8.77658e-02 | 4.40168e-04 |
| Shift output NMSE | 1.69267e-03 | 1.61745e-01 | 3.81287e-03 |

Exact active KV is 98,304 bytes. Guarded shifted NMSE is 1.47937e-03 with a
0.2552 fallback rate. The Python reference is slower than vectorized exact
attention and supplies no latency claim.

### 10.2 Post-hoc randomized suite

Three retained seeds produce 11 controlled cases. All runs pass the declared
safety condition. Weighted mean Fabric NMSE is 4.02260e-05, but this aggregate
is dominated by heterogeneous baseline difficulty and should not be read as an
external effect size. The more conservative statistic is the worst run-level
Fabric-to-best-single-family NMSE ratio, 0.0988026. Forced exact fallback has
zero measured maximum absolute error and invalid outputs are zero.

The suite is particularly important because its first version falsified the
single-start clustering implementation. The architecture was changed in
response; the failed value is documented but excluded from accepted claims.

### 10.3 Learned checkpoint traces

On actual Q/K/V projections from the repaired symbolic checkpoint, layer 0
future-trace NMSE is 1.61037e-04 for Fabric and 3.44849e-04 for legacy learned
FOCUS. At layer 3, legacy FOCUS is better unguarded: 2.18333e-05 versus
1.39396e-04. This counterexample is retained because the portfolio is not
expected to dominate every local field. Exact guarding reduces layer-3 Fabric
error at the cost of archive reads.

### 10.4 Repeated compaction and model path

The accepted 128-token controlled compaction run performs four page merges and
has mean/p95/maximum relative attention error 0.008742/0.019830/0.028922. The
repaired symbolic model has 100% argmax agreement over 64 teacher-forced
positions and 100% free-running agreement over 8 greedy tokens in the accepted
run. Its maximum logit absolute difference is 0.131632. The original symbolic
tokenizer is unavailable, so these are token-ID mechanism tests, not natural
language results.

### 10.5 Agent-memory substrate

Across 20 seeds and 20 compaction cycles per seed, protected-record retention
and hash-chain verification are 100%, while untrusted injected prose is
promoted to policy in 0% of cases. This validates structural rules only; it does
not measure agent task success or reasoning.

## 11. Limitations and negative evidence

- No production-scale pretrained LLM was integrated in this environment.
- LongBench, RULER, BABILong, LifeBench, long natural-language tool use, and
  long CoT were not executed; their result fields are null.
- CUDA/Triton runtime correctness, p50/p95 latency, occupancy, and physical HBM
  traffic were not measured.
- The exact archive remains linear in context size.
- Static conformal calibration fails under distribution shift; the sentinel
  needs enough audits before it can react.
- Fallback can dominate cold-memory traffic.
- The CPU compiler is expensive and not a serving implementation.
- The semantic hash chain is not an authenticated signature.
- The Codex pipeline was dry-run in this container because the Codex executable
  was absent; execute mode is supplied but not falsely marked as exercised.
- The architecture has not established superior general intelligence, official
  benchmark leadership, production speedup, or million-token correctness.

## 12. Reproducibility and artifacts

All accepted values resolve through `docs/CLAIMS_LEDGER.json` to immutable
result paths and SHA-256 digests. Core commands are:

```bash
pip install -e '.[dev]'
make gate
make benchmark
make agent-memory
make holdout
make autonomy-dry-run
```

The release preserves raw JSON, CSV, plots, checkpoint metadata, tests, the
Codex hypothesis backlog, and an explicit not-executed artifact for GPU
measurements.

## 13. Conclusion

FOCUS-Fabric's main contribution is not a claim that one new cache replaces all
others. It is a falsifiable systems thesis: long-context memory should be
heterogeneous, composable through value and mass, reversible through an exact
truth plane, trainable, monitored for shift, and separated from typed semantic
governance. The release demonstrates that this design can outperform
memory-matched single-family references in controlled heterogeneous fields and
can preserve short token trajectories in a repaired model. The evidence is
promising but deliberately bounded. The next decisive test is a GPU-integrated
7B–8B model evaluated on official long-context and long-horizon agent suites
with physical memory and latency counters.

## References

A machine-readable bibliography with adoption decisions and caveats is in
`references/literature_2026-07.json`. Closest and most influential works include:

1. *CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for
   Resource-Efficient Long-Context LLM Inference*, arXiv:2606.24467.
2. *PolyKV: Heterogeneous Retention and Allocation for KV Cache Compression*,
   arXiv:2606.15157.
3. *Understanding the Physics of Key-Value Cache Compression for LLMs through
   Attention Dynamics*, arXiv:2603.01426.
4. *Information-Aware KV Cache Compression for Long Reasoning*,
   arXiv:2606.26875.
5. *Fast KV Compaction via Attention Matching*, arXiv:2602.16284.
6. *Nectar: Neural Estimation of Cached-Token Attention via Regression*,
   arXiv:2605.09778.
7. *The Residual Stream Is All You Need: On the Redundancy of the KV Cache in
   Transformer Inference*, arXiv:2603.19664.
8. *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663.
9. *ATLAS: Learning to Optimally Memorize the Context at Test Time*,
   arXiv:2505.23735.
10. *Agentic Context Engineering: Evolving Contexts for Self-Improving Language
    Models*, arXiv:2510.04618.
11. *Memory in the Age of AI Agents*, arXiv:2512.13564.
12. *Fast Inference from Transformers via Speculative Decoding*,
    arXiv:2211.17192.
