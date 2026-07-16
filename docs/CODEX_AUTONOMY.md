# Codex Autonomous Research Pipeline

## What this automates

The pipeline lets Codex investigate one preregistered FOCUS-Fabric hypothesis in an isolated Git worktree. It automates the research loop, not the truth of the result and not publication.

The safe default is:

- diagnose every prerequisite before spending a run;
- execute only a hypothesis with a machine-readable evaluator;
- keep all candidate changes in a separate worktree;
- run deterministic gates and a post-hoc holdout from trusted root files;
- preserve negative results;
- make no commit, merge, push, tag, or release.

`--auto-promote` changes only the commit and fast-forward merge step. Push, release, and publication always remain manual.

## Current executable scope

`H001-forward-influence-routing` is the only live-executable hypothesis. Its promotion contract is explicit:

- primary metric: in-distribution `output_nmse`;
- at least 5% relative improvement;
- exactly matched Fabric active bytes;
- no teacher-forced token-agreement regression;
- free-running sequence agreement remains true;
- repeated compaction produces zero invalid codec outputs;
- randomized holdout safety passes.

H002-H004 remain useful preregistrations, but preflight blocks live execution until each has a dedicated evaluator. They are not silently judged with H001's metric.

## Why the runner bootstraps Codex explicitly

Codex project config is loaded only for a trusted exact project path. A dynamically created Git worktree has a different path, so it cannot safely rely on inherited trust.

Every live command therefore:

- discovers and probes the installed Codex Desktop runtimes instead of trusting the first PATH shim;
- uses `gpt-5.6-sol` for the root and the checked-in Sol/Terra specialist profiles;
- registers every `.codex/agents/*.toml` profile through CLI config overrides;
- enables multi-agent execution with bounded depth and concurrency;
- ignores user config and enables only the checked-in `SubagentStart`/`SubagentStop`
  recorder;
- explicitly selects the native Windows `elevated` sandbox implementation,
  because ignoring user config would otherwise discard that machine-level
  selector even when `workspace-write` is requested;
- writes role/model lifecycle evidence to an orchestrator-owned directory outside
  the candidate worktree, where the candidate sandbox cannot forge it;
- sets Codex workspace network access to false;
- uses `workspace-write`, approval policy `never`, JSONL output, an output schema, and an ephemeral session.

The last-message JSON and raw JSONL stream are stored below ignored `autonomy/state/`. Public candidate evidence belongs below `results/experiments/<hypothesis>/`.

## Windows commands

Run from the repository root in PowerShell:

```powershell
# 1. Verify CLI version, login, model catalog, venv, Git state, profiles, and evaluator.
.\scripts\autonomy\run_cycle.cmd --mode preflight --hypothesis H001-forward-influence-routing

# 2. Render the exact isolated plan without invoking an agent.
.\scripts\autonomy\run_cycle.cmd --mode dry-run --hypothesis H001-forward-influence-routing

# 3. Run H001 once in a separate worktree. No commit or merge is made.
.\scripts\autonomy\run_cycle.cmd --mode execute --hypothesis H001-forward-influence-routing

# 4. Optional: permit commit + ff-only merge only after every gate passes.
.\scripts\autonomy\run_cycle.cmd --mode execute --hypothesis H001-forward-influence-routing --auto-promote
```

If preflight reports that Codex is not authenticated for the current Windows user, run the current Codex CLI's device login once, then repeat preflight:

```powershell
$Preflight = .\scripts\autonomy\run_cycle.cmd --mode preflight --hypothesis H001-forward-influence-routing | ConvertFrom-Json
$Codex = $Preflight.codex.path
& $Codex login --device-auth
```

This deliberately uses the executable path resolved by preflight; a stale or broken
PATH shim is not used.

The `.cmd` wrapper works under PowerShell's default script policy and always runs `.venv\Scripts\python.exe`. This prevents global Python or a missing PATH-level `pytest` from changing the gate environment. `run_cycle.ps1` is also provided for environments that permit PowerShell scripts.

Equivalent Make targets are `autonomy-preflight`, `autonomy-dry-run`, and `autonomy-execute` when GNU Make is available.

## Execution lifecycle

1. **Preflight** probes a working Codex executable, login state, current model catalog, supported CLI options, project venv, clean Git state, required artifacts, specialist profiles, and the selected evaluator. On native Windows it also runs a model-free `codex sandbox` check that must create a temporary workspace sentinel; parsing `--help` alone is not accepted as proof of write access.
2. **Preregistration** loads the hypothesis, controls, falsifier, primary metric, allowed paths, and baseline SHA-256. The outer runner writes the complete `experiment.json` and records its byte SHA-256 before Codex starts. The same bytes are required after Codex, after every host-side candidate process, before acceptance, and before staging; the parsed payload must also exactly equal the generated contract.
3. **Isolation** creates `../.focus-fabric-worktrees/<hypothesis>-<timestamp>` and records its starting commit.
4. **Agent run** injects the checked-in specialist roles. The agent is forbidden to commit or alter Git history. Trusted lifecycle hooks record each required role's start, stop, and actual model outside the candidate worktree.
5. **Validation** checks the final JSON schema/status, its hash, exact agreement between self-reported and Git-observed changes, history immutability, and exact file/directory boundaries. The same scope check runs again after gates and evidence generation.
6. **Trusted gates** load `autonomy/gates.json` and the original test suite from the root checkout, while importing candidate source. Python commands are pinned to the project interpreter and receive a credential-free environment. Plotting is fixed to the headless `Agg` backend. After Codex and after every host-side candidate process, the runner verifies the root HEAD, status, baseline digest, and byte-level digest of every tracked root file before continuing.
7. **Holdout and decision** compare root and candidate on the same seed generated only after the agent run, then apply the H001 metric contract.
8. **Retention** leaves the branch and worktree intact for human inspection. With no `--auto-promote`, even an accepted candidate stays uncommitted. Automatic promotion is also blocked if an existing tracked test was changed or deleted; new tests may be promoted only with the candidate implementation. When promotion is explicitly enabled, the runner fixes the staged Git tree before commit, disables repository hooks with a fresh random nonexistent hooks path for each Git operation, verifies the commit's single parent, tree, and exact diff paths, and merges that verified commit hash rather than a mutable branch name. The merged root HEAD, clean status, and tracked-file byte digest must then match the validated candidate.

Root ledger and run reports are written to ignored `autonomy/state/` and `results/autonomy_runs/`, so one run does not make the next preflight dirty.

## Failure behavior

Codex command failure, model/config mismatch, unavailable native workspace-write sandbox, agent `failed` or `blocked` status, missing or invalid result JSON, experiment-contract mismatch, self-report mismatch, agent-created commit, scope escape, trusted-root mutation, deterministic gate failure, holdout regression, or primary-metric failure all produce a non-promoted result. The worktree is preserved so the failure can be audited rather than erased.

The root lock is released in a `finally` block. A stale lock after a machine crash must be inspected before manual removal; never delete it while another cycle may still be running.

## Security boundary

The inner Codex process is sandboxed and has workspace network access disabled. Both Codex and gate subprocesses start from small environment allowlists: Codex retains only the paths needed to discover the existing ChatGPT login, while API-key-like and unrelated credentials are not inherited. The one-off hook-trust bypass applies only to the explicit checked-in lifecycle recorder; user/global hooks remain ignored.

This is still not a VM or container security boundary. Candidate Python is executed on the host during tests and benchmarks. Root-integrity snapshots detect persistent writes to tracked project files; they do not prevent transient writes, access to unrelated host files, or tampering outside the tracked tree. Use a dedicated VM/container for adversarial or third-party hypotheses, and never use `--dangerously-bypass-approvals-and-sandbox` on a normal workstation.
