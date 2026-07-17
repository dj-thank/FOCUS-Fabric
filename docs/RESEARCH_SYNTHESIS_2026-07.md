# Research Synthesis — 2026-07-14

## Thesis

The initial single-family prototype assumed that old attention could be represented by one family of query-conditioned local operators. The literature and our randomized falsification both reject that universal assumption. The revised thesis is:

> Long-context intelligence needs a **portfolio of reversible memory representations**, selected by measured future-query behavior, guarded by an exact source of truth, and paired with a separately typed semantic control plane for agent commitments and provenance.

The release does not claim that this thesis is globally optimal. It turns the thesis into falsifiable code and gates.

## Synthesis by field

### 1. KV compression and routing geometry

CompressKV and PolyKV motivate head/layer heterogeneity rather than uniform policy. The attention-dynamics analysis distinguishes token retention from actual semantic accessibility and reports failure cliffs at extreme compression. InfoKV argues that current attention is backward-looking and that uncertainty/forward influence identifies distant-future value. Fast Attention Matching and Nectar both make the normalizer explicit: preserving a locally normalized value vector without its mass is insufficient.

**Adopted:** heterogeneous codecs; output plus log-mass; exact fallback; forward-residual scoring; measured distribution shift; no “percentage retained equals memory preserved” claim.

**Rejected:** one global eviction rule; one fixed cache budget; attention-score-only importance; fixed-context approximation without an OOD escape path.

### 2. Numerical approximation and statistics

The operator codec uses local Taylor models, the Gaussian codec uses cumulant expansion, the moment state uses projected second-order exponential moments, and the coreset uses vector quantization. A rate–distortion objective decides which error/byte/compute trade-off is acceptable. Split conformal prediction calibrates each codec's proxy on a disjoint split; Hoeffding monitoring audits whether marginal coverage is drifting.

A critical empirical correction came from randomized holdout: one K-means initialization could silently choose a disastrous route. Query-aware multi-start selection was therefore added. This is a direct example of scientific infrastructure improving the architecture, not merely documenting it.

### 3. Test-time learning and “sleep” consolidation

Titans, ATLAS, MesaNet, and sleep-like offline recurrence all treat memory as an updateable computational object rather than a passive token list. They suggest a continuum between fast online writes, slower consolidation, and exact episodic storage.

**Adopted now:** expensive page compilation occurs off the hottest path; native loss makes compressed-prefix behavior trainable; compiler cost is explicit.

**Deferred:** learned fast weights, offline recurrent passes, conjugate-gradient inner solvers, and joint end-to-end training at useful scale. These are hypotheses, not silently claimed features.

### 4. Databases, storage systems, and control theory

The binary-counter hierarchy borrows the spirit of LSM-tree compaction: immutable units are merged in the background. Unlike approximate recursive summaries, the merge source is the exact archive. The controller behaves like a safety system: calibrated estimates are accepted only inside a budget; otherwise it degrades to a slower exact mode. Drift monitoring converts occasional exact audits into a state transition—normal, warning, or strict.

Missing production pieces are explicit: atomic manifests, crash recovery, asynchronous I/O, multi-device placement, and signed archive roots.

### 5. Agent memory, provenance, and security

ACE shows why iterative prose rewriting can collapse context; the broader agent-memory taxonomy separates memory forms/functions and emphasizes trustworthiness. FOCUS-Fabric therefore does not use numerical KV compression as a substitute for semantic memory governance.

Policies, constraints, goals, commitments, decisions, and evidence are typed records. Protected records and transitive evidence survive compaction verbatim. Capsules are extractive and carry source IDs and digests. Untrusted tool text cannot self-promote into a protected type; classification authority remains outside the text channel.

This is closer to event sourcing than to a chat summary. It improves auditability but does not prove that a record is true. Digital signatures, access control, privacy policy, and trusted time remain deployment responsibilities.

### 6. Speculative execution and verification

The compressed fabric may be a strong drafter even when exact token agreement is not guaranteed. Speculative decoding motivates separating proposal quality from correctness. The supplied greedy verifier emits only tokens checked against the exact oracle, making the final greedy sequence exact by construction. Production acceleration still requires batched verification, cache rollback, and GPU kernels.

### 7. Autonomous software science

Current Codex supports non-interactive execution, JSONL events, output schemas, sandbox policies, project `AGENTS.md`, and specialized agents. Those capabilities are used as a harness—not as evidence that autonomous output is correct.

Every experiment is preregistered with a falsifier and allowed-file scope, executed in a worktree, reviewed by deterministic gates, then tested on post-hoc randomized cases chosen after the code change. Claims are paths into immutable JSON plus SHA-256. Repository entropy is treated as a recurring maintenance target.

## Architecture decisions and alternatives

| Decision | Chosen design | Alternative not chosen | Reason |
|---|---|---|---|
| Old-context representation | Per-head heterogeneous portfolio | Universal operator cache | Empirical regimes and randomized holdout refuted universality |
| Safety | Exact cold truth + sparse fallback | Irreversible delete-only cache | Rare facts and OOD queries need recovery |
| Error statistic | Output + log-mass | Output-only | Cross-page softmax weights depend on mass |
| Calibration | Split fit/select/calibrate | Reuse training traces for all stages | Prevent optimistic error estimates |
| Compaction merge | Recompile from exact archive | Compress compressed state | Avoid recursive approximation drift |
| Semantic memory | Typed hash-chained ledger | Free-form rolling summary | Policies/evidence require provenance and protection |
| Generation | Optional exact verifier | Trust approximate cache logits | Proposal quality and final correctness are distinct |
| Autonomy | Worktree + external randomized holdout | Agent edits main branch and self-scores | Reduces scope drift and public-benchmark overfitting |

## Preregistered high-value next experiments

1. **Forward-influence residual selection** on real 7B–8B Q/K/V traces and long reasoning.
2. **Sleep consolidation** that optimizes page states across several future-query windows.
3. **Sequentially valid drift control** replacing a static exchangeable conformal multiplier.
4. **Global layer/head/page/device budget allocation** with dual updates under measured HBM and latency constraints.
5. **Residual-stream exact archive backend** to reduce cold storage while preserving token identity.
6. **Long tool trajectory evaluation** combining numerical cache telemetry with policy/evidence retention and task completion.
7. **Grouped Triton/CUDA kernels** with Nsight physical counters and end-to-end generation profiling.

## Source catalog

Machine-readable details, adoption decisions, and caveats are in [`references/literature_2026-07.json`](../references/literature_2026-07.json). Important primary sources include:

- https://arxiv.org/abs/2606.24467
- https://arxiv.org/abs/2606.15157
- https://arxiv.org/abs/2603.01426
- https://arxiv.org/abs/2606.26875
- https://arxiv.org/abs/2602.16284
- https://arxiv.org/abs/2605.09778
- https://arxiv.org/abs/2603.19664
- https://arxiv.org/abs/2605.26099
- https://arxiv.org/abs/2510.04618
- https://arxiv.org/abs/2512.13564
