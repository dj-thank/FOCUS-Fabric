# Release Checklist

Status date: **2026-07-17**. Checked packaging items describe the retained 2026-07-14 `0.2.0` release snapshot unless stated otherwise. Post-release `main` changes are not silently folded into that release; a new release must choose a new identity and fixed source commit before regenerating metadata.

## Code and packaging

- [x] `python -m compileall -q src scripts tests`
- [x] `pytest -q`
- [x] editable install succeeds with `--no-deps` against the retained environment
- [x] wheel and source distribution build successfully
- [x] clean-target wheel import succeeds without importing the source tree
- [x] `twine check dist/*`
- [x] no secrets, absolute private paths, or transient worktrees in the release tree
- [x] release generation enumerates Git-tracked regular files only and binds metadata generation to a clean HEAD commit
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
- [x] Codex-generated candidate code executed in an isolated Windows worktree; the run completed all gates, remained unpromoted, and the strengthened paired holdout rejects it as insensitive
- [ ] release manifest signed by an external identity
- [ ] human authorization for public hosting and deployment
- [ ] choose a new release identity/date/commit and regenerate manifest/checksums in a clean checkout for post-`0.2.0` `main`

## Capability wording

- [x] no general-intelligence superiority claim
- [x] no official benchmark claim without official scorer output
- [x] no GPU speed claim from CPU timing or estimated bytes
- [x] no universal conformal guarantee
- [x] comparisons name metric, baseline, budget, model, and context length

## Publication decision

The retained `0.2.0` code, CPU evidence, claim ledger, source distribution, wheel, and unsigned archive remain suitable for a **research preview**. Current `main` is a post-release development state, not a rebuilt archive. A new public artifact requires a distinct release identity plus the still-open official-benchmark, GPU, external-signature, and human-authorization decisions appropriate to its wording.
