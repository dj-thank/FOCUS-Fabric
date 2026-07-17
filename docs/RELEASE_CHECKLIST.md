# Release Checklist

Status date: **2026-07-17**. The successor identity `FOCUS_Fabric_2026-07_0.2.1_release` / `0.2.1` was published as an [unsigned GitHub research preview](https://github.com/dj-thank/FOCUS-Fabric/releases/tag/v0.2.1) from fixed source commit `069351b0a586487961f3d7c54fb3c94bb70c32cc`. The supplied pre-publication `0.2.0` sdist remains quarantined and is not an authorized public artifact.

## Code and packaging

- [x] `python -m compileall -q src scripts tests`
- [x] `pytest -q`
- [x] editable install succeeds with `--no-deps` against the retained environment
- [x] wheel and source distribution build successfully
- [x] clean-target wheel import succeeds without importing the source tree
- [x] `twine check dist/*`
- [x] no secrets, absolute private paths, or transient worktrees in the release tree
- [x] release generation enumerates Git-tracked regular files only and binds metadata generation to a clean HEAD commit
- [x] Python distributions are rebuilt from an isolated export of exact clean Git `HEAD`, excluding stale/untracked build state, and recursive model-weight exclusions are applied
- [x] wheel and sdist members are independently checked for unsafe paths, links, excluded weight payloads, required package files, and exact metadata version
- [x] license, citation, contribution, and security files present
- [x] all nine GitHub Release assets were downloaded after publication and matched the locally verified files byte-for-byte

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
- [x] operator authorization recorded in [PR #1](https://github.com/dj-thank/FOCUS-Fabric/pull/1#issuecomment-4997600158), explicitly distinguished from independent review and cryptographic signing
- [ ] deployment authorization beyond source/release publication
- [x] choose a distinct `0.2.1` release identity and date
- [x] merge [PR #1](https://github.com/dj-thank/FOCUS-Fabric/pull/1), fetch `origin/main`, and build the final manifest/checksums from exact commit `069351b0a586487961f3d7c54fb3c94bb70c32cc`
- [x] publish the verified artifacts under the [`v0.2.1` GitHub release](https://github.com/dj-thank/FOCUS-Fabric/releases/tag/v0.2.1)

## Capability wording

- [x] no general-intelligence superiority claim
- [x] no official benchmark claim without official scorer output
- [x] no GPU speed claim from CPU timing or estimated bytes
- [x] no universal conformal guarantee
- [x] comparisons name metric, baseline, budget, model, and context length

## Publication decision

The retained CPU evidence and claim ledger remain suitable for the carefully scoped **research preview** wording used by the published `v0.2.1` release. The pre-publication `0.2.0` sdist itself remains unsuitable because it contains excluded local checkpoint weights. Publication does not close the still-open official-benchmark, GPU, external-signature, deployment, independent-reproduction, or stronger-capability gates.
