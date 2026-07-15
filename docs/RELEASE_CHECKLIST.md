# Release Checklist

Status date: **2026-07-14**. Checked items were executed or directly inspected in the retained CPU release environment. Open items are publication or hardware gates, not hidden successes.

## Code and packaging

- [x] `python -m compileall -q src scripts tests`
- [x] `pytest -q`
- [x] editable install succeeds with `--no-deps` against the retained environment
- [x] wheel and source distribution build successfully
- [x] clean-target wheel import succeeds without importing the source tree
- [x] `twine check dist/*`
- [x] no secrets, absolute private paths, or transient worktrees in the release tree
- [x] license, citation, contribution, and security files present

## Evidence

- [x] controlled benchmark regenerated on the release source
- [x] randomized holdout passes on at least three post-hoc seeds
- [x] learned-trace and end-to-end artifacts identify checkpoint provenance
- [ ] official benchmark predictions scored by official LongBench/RULER/BABILong/LifeBench tools
- [ ] GPU correctness and latency measured on named hardware
- [ ] physical HBM counters captured separately from estimated traffic
- [x] all quantitative claims resolve to immutable JSON paths and matching SHA-256
- [x] negative/failed results retained where they changed the design

## Safety and integrity

- [x] distribution-shift coverage and fallback cost reported
- [x] exact archive assumptions disclosed
- [x] semantic-memory protected classes reviewed
- [x] prompt/tool-output injection substrate tests run
- [ ] Codex-generated candidate code executed in an isolated runner in this environment
- [ ] release manifest signed by an external identity
- [ ] human authorization for public hosting and deployment

## Capability wording

- [x] no general-intelligence superiority claim
- [x] no official benchmark claim without official scorer output
- [x] no GPU speed claim from CPU timing or estimated bytes
- [x] no universal conformal guarantee
- [x] comparisons name metric, baseline, budget, model, and context length

## Publication decision

The code, CPU evidence, claim ledger, source distribution, wheel, and unsigned release archive are suitable for a **research preview**. A production or model-capability release remains blocked on the open official-benchmark, GPU, external-signature, and human-authorization gates.
