---
name: focus-research
description: Run a falsifiable FOCUS-Fabric architecture experiment from literature synthesis through gated evidence.
---

# FOCUS research cycle

1. Read `autonomy/hypotheses.json` and select exactly one pending hypothesis.
2. Ask `research_scout` and `memory_redteam` to analyze prior art and failure
   modes in parallel. Wait for both and preserve their negative findings.
3. Write an experiment contract containing independent variable, controls,
   memory budget, metrics, disconfirming threshold, and immutable test split.
4. Ask `architecture_scientist` to implement the smallest complete mechanism.
   Use `kernel_engineer` only when the reference behavior is already tested.
5. Ask `benchmark_adversary` to add counterexamples and run all gates.
6. Ask `reproducibility_auditor` and `claim_auditor` for final read-only review.
7. Record raw outputs and hashes. Promote only if every hard gate passes and the
   predeclared objective improves without a safety regression.

Never merge two hypotheses into one experiment after seeing results.
