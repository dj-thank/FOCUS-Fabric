# FOCUS-Fabric 2026.07

**A heterogeneous, uncertainty-gated memory fabric for Transformer KV state, long-horizon agent memory, and verified generation.**

FOCUS-Fabricは、旧FOCUS-KV/FOCUS-Nativeを監査・再構築した研究リリースです。単一の圧縮方式を全head・layer・pageへ強制せず、各領域のattention場に応じて異なる記憶表現を選び、分布外ではexact archiveへ戻ります。

> **公開上の位置づけ**: 実行可能な研究プロトタイプです。既存の大規模一般LLMより優れた自然言語能力、公式LongBench/RULER/BABILongスコア、GPU高速化、物理HBM帯域削減は未実証であり、主張しません。

## What changed

旧FOCUS作用素を一候補へ降格し、次の六つの制御面を統合しました。

1. **Heterogeneous numerical memory** — local response operator、query-aware multi-start weighted KV coreset、Gaussian/cumulant state、mergeable moment state、exact-residual hybridをpage/head単位で選択。
2. **Rate–distortion compiler** — fit / model-selection / conformal-calibrationを分離し、誤差・active bytes・推定FLOPsのPareto集合から選択。
3. **Reversible hierarchy** — exact hot KV、binary-counter compaction、cold exact archive。mergeは近似状態を再圧縮せずarchiveから再コンパイルし、再帰誤差を抑制。
4. **Uncertainty control** — attention outputだけでなくlog-normalizerも保持し、split-conformal upper bound、sparse exact fallback、drift sentinelを実装。
5. **Agent semantic memory** — policy、constraint、goal、commitment、evidenceを不可逆要約から保護するtyped hash-chain ledgerと、source manifest付きextractive capsule。
6. **Verified generation and autonomous science** — compressed drafterをexact verifierで訂正するlossless greedy path、Codex worktree実験、外部乱数holdout、claim ledger、CI gate。

## Measured evidence in this CPU environment

### Controlled heterogeneous attention

| Metric | Result |
|---|---:|
| Exact KV bytes | 98,304 |
| Active Fabric bytes | 8,584 |
| Active compression | **11.452×** |
| Fabric ID output NMSE | **5.1179439e-05** |
| Memory-matched operator NMSE | 0.087765813 |
| Memory-matched coreset NMSE | 0.00044016758 |
| ID conformal coverage | 0.9688 (target 0.95) |
| Shift guarded NMSE | 0.0014793737 |
| Shift fallback rate | 0.2552 |

Selected representations were heterogeneous: `hybrid_gaussian_c4_r4_e2, weighted_coreset_s4, gaussian_c2_r4, hybrid_gaussian_c4_r4_e8`.

### Archived checkpoint trace and generation

The reported ~1M-parameter symbolic mechanism checkpoint was repaired and loaded during the local study. Because its original training corpus, tokenizer mapping, and redistribution provenance are incomplete, the weight binaries are **not committed to this public GitHub repository**. The retained JSON evidence, architecture metadata, hashes, and loader code are provided for auditability; these checkpoint-derived figures are numerical Q/K/V and token-ID mechanism experiments only.

| Metric | Result |
|---|---:|
| Teacher-forced argmax agreement, 64 tokens | **1.0000** |
| Free-running greedy agreement, 8 generated tokens | **1.0000** |
| Max logit absolute error | 0.131632 |
| Repeated-compaction mean attention relative error | 0.008742 |
| Repeated-compaction maximum relative error | 0.028922 |
| Invalid codec outputs | 0 |

The Python reference is **not a speed implementation**: in this CPU run the compilation/evaluation path was slower than vectorized exact attention. Triton code is included but CUDA was unavailable, so GPU latency and physical HBM fields remain `null`.

### Adversarial agent-memory substrate

Across 20 seeds × 20 repeated compactions, protected-record retention was 100.0%, hash-chain verification succeeded in 100.0%, and injected text was promoted into a policy record in 0.0% of cases. This validates structural preservation only; it is not an LLM tool-reasoning score.

### Randomized holdout that found and fixed a real failure

The post-hoc evaluator initially exposed catastrophic single-seed K-means instability. Query-aware multi-start selection reduced the original retained holdout's mean Fabric NMSE to **3.1504659e-05**, versus 0.00035326691 for its best memory-matched single-family reference. Across three retained seeds and 11 controlled randomized cases, every run passed the safety gate; the worst run-level Fabric/best-single-family NMSE ratio was **0.0988026**, exact forced fallback had max error **0.0**, and invalid outputs were **0**. These are controlled attention-field results, not language-model benchmark scores.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make gate
```

Reproduce the committed CPU evidence:

```bash
make benchmark
make agent-memory
make gpu-benchmark   # writes an explicit not_executed record without CUDA/Triton
```

Plan autonomous experiments without changing the repository:

```bash
make autonomy-dry-run
```

With an authenticated current Codex CLI installed, run one isolated hypothesis:

```bash
python scripts/autonomy/run_codex_loop.py --mode execute --max-hypotheses 1
```

Automatic promotion is disabled unless `--auto-promote` is explicitly supplied. Even then, deterministic tests, claim integrity, a post-hoc randomized holdout, exactness constraints, and a minimum public-evidence improvement must all pass.

## Repository map

- [`docs/PAPER_DRAFT.md`](docs/PAPER_DRAFT.md) — publication-style method and evidence draft.

- `src/focus_fabric/` — new heterogeneous fabric, certificates, semantic ledger, verified decode, native loss, Triton ABI.
- `src/focus_native/` — repaired legacy Transformer mechanism and optional local checkpoint loader.
- `scripts/benchmark_fabric.py` — controlled, learned-trace, repeated-compaction, and end-to-end benchmark.
- `scripts/evaluation/` — agent-memory, GPU, and official-dataset runners.
- `scripts/autonomy/` — Codex orchestration, randomized holdout, claim and drift gates.
- `docs/` — architecture, weakness audit, research synthesis, evaluation contract, model card, threat model, and claims.
- `results/` — immutable evidence artifacts referenced by `docs/CLAIMS_LEDGER.json`.

## Read first

- [Architecture](docs/ARCHITECTURE.md)
- [Weakness audit](docs/WEAKNESS_AUDIT.md)
- [Research synthesis, July 2026](docs/RESEARCH_SYNTHESIS_2026-07.md)
- [Codex autonomous operation](docs/CODEX_AUTONOMY.md)
- [Evaluation contract](docs/EVALUATION.md)
- [Claims and non-claims](docs/CLAIMS.md)
- [Model card](docs/MODEL_CARD.md)
- [Limitations](docs/LIMITATIONS.md)
- [Publication status](docs/PUBLICATION_STATUS.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)

## License

Apache-2.0 for repository code and documentation. The historical checkpoint weight binaries are intentionally excluded because their original training-data and redistribution provenance is incomplete; see [`checkpoints/README.md`](checkpoints/README.md).
