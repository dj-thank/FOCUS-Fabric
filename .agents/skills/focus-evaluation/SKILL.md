---
name: focus-evaluation
description: Evaluate a cache or memory candidate without leakage, hidden memory, or unsupported performance claims.
---

# Evaluation protocol

- Keep fit, selection, conformal calibration, and final test query sets disjoint.
- Report attention output and log-mass errors; either alone is insufficient.
- Count query reservoirs, routing metadata, certificates, exact residuals, and
  hot K/V in active memory.
- Report cold archive separately rather than pretending it disappeared.
- Compare against memory-matched operator-only and KV-coreset baselines.
- Test in-distribution, shifted, rare-retrieval, repeated-compaction, and
  end-to-end token agreement paths.
- Mark CUDA, HBM, and official benchmark fields `null` if not measured.
- Store JSON/CSV raw artifacts and SHA-256 digests before writing prose.
