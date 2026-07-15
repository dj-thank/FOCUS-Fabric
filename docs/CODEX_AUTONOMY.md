# Codex Autonomous Research Pipeline

## Purpose

The pipeline automates **hypothesis execution**, not truth. It is designed to make autonomous code changes cheap while making unsupported promotion expensive.

## Requirements

- A current authenticated Codex CLI available as `codex`.
- Git repository with a clean working tree.
- Python environment satisfying `pyproject.toml`.
- Sufficient CPU/GPU/dataset resources for the selected hypothesis.

The build container used for this release did not have the Codex executable, so `execute` mode was not run here. `dry-run` is executable and records the exact planned commands.

## Repository instructions

`AGENTS.md` is the root contract. It states invariants that must remain true:

- no unsupported measurements;
- output and log-mass are inseparable;
- exact fallback/archive accounting is explicit;
- fit, selection, calibration, and test data remain separate;
- protected semantic memory cannot be summarized away;
- tests cannot be weakened to manufacture improvement;
- publication is never automatic.

`.codex/agents/*.toml` defines specialized research, architecture, kernel, benchmark, reproducibility, claim, and memory-red-team roles. `.agents/skills/` supplies reusable workflows.

## Lifecycle

### 1. Preregistration

`autonomy/hypotheses.json` defines:

- mechanism and predicted effect;
- falsifier;
- allowed files;
- resource budget;
- priority and status.

The orchestrator refuses an unknown or non-pending hypothesis.

### 2. Isolation

A clean Git worktree and branch are created per hypothesis. The agent may edit only declared paths. Changed files are checked after the run; an out-of-scope path blocks the candidate.

### 3. Structured Codex execution

The command uses non-interactive `codex exec`, workspace-write sandboxing, JSONL event output, an output-last-message file, and `autonomy/schemas/agent_result.schema.json`. The prompt includes the preregistration, invariants, baseline digest, allowed scope, and required gates.

The event stream is preserved. A separate hash-chained automation ledger records start, tool completion, gate results, evidence comparison, and promotion decision.

### 4. Deterministic gates

`autonomy/gates.json` currently requires:

1. source/script/test compilation;
2. unit tests;
3. candidate benchmark generation;
4. claim-ledger verification;
5. repository drift scan.

A candidate that changes expected evidence must update the candidate artifact, not overwrite the accepted baseline.

### 5. External randomized holdout

After Codex finishes, the trusted root orchestrator generates a seed. `scripts/autonomy/holdout_evaluator.py` is executed twice—once with root source and once with worktree source—on the same unseen cases. The candidate cannot edit the trusted evaluator from its worktree scope.

Promotion requires:

- exact forced-fallback error below tolerance;
- no invalid outputs;
- randomized objective no worse than 2% relative to root;
- public evidence score improved by at least 0.5%;
- end-to-end token agreement not below baseline;
- repeated-compaction invalid outputs equal zero.

This is an anti-overfitting measure, not a cryptographic defense against malicious code. Independent review and external infrastructure remain mandatory for release.

### 6. Promotion

Default behavior is evidence generation only. `--auto-promote` is required to commit and fast-forward the root branch. Release tagging, signing, publishing, and deployment are outside the automatic loop.

## Commands

```bash
# List/plan the first two pending hypotheses.
python scripts/autonomy/run_codex_loop.py --mode dry-run --max-hypotheses 2

# Execute one hypothesis in a worktree; do not merge it.
python scripts/autonomy/run_codex_loop.py --mode execute --max-hypotheses 1

# Execute and permit promotion only if every gate passes.
python scripts/autonomy/run_codex_loop.py --mode execute --max-hypotheses 1 --auto-promote
```

Use a particular hypothesis:

```bash
python scripts/autonomy/run_codex_loop.py --mode execute --hypothesis H003
```

## Failure handling

A failed Codex command, schema violation, scope violation, test failure, missing artifact, holdout regression, or evidence regression produces a non-promoted result. Worktrees are preserved only long enough to record diagnostics, then removed. The hypothesis remains pending unless a human changes its status with a reason.

## Entropy control

Autonomous agents copy local patterns. Drift therefore compounds. The pipeline includes:

- `detect_drift.py` for missing tests, claim phrases, stale benchmark evidence, and oversized Python modules;
- claim hashes to stop documentation from drifting away from measurements;
- specialized cleanup hypotheses;
- review agents with adversarial roles;
- immutable architecture and weakness documents as repository-visible system knowledge.

## Security notes

Never use `--dangerously-bypass-approvals-and-sandbox` on a normal host. Run external benchmark backends and candidate kernels in an OS/container boundary with restricted credentials and network access. The supplied workspace scope check is not an OS security boundary.
