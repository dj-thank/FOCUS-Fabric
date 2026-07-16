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
- uses `gpt-5.6-sol` for the root and a checked-in `gpt-5.6-luna` default for
  every bounded specialist task;
- registers every `.codex/agents/*.toml` profile through CLI config overrides;
- enables multi-agent execution with bounded depth and concurrency;
- requires every specialist spawn to use `fork_turns="none"`, an exact role task
  name, and a self-contained message carrying the checked-in role contract;
- ignores user config and does not enable command hooks;
- explicitly selects the native Windows `elevated` sandbox implementation,
  because ignoring user config would otherwise discard that machine-level
  selector even when `workspace-write` is requested;
- snapshots the host-owned Codex session directory before execution, then
  verifies only newly created child-session metadata bound to the parent thread;
- requires an exact role path, `task_complete`, the actual `gpt-5.6-luna`
  runtime model in both model fields, and the OpenAI provider for every required
  role; task names and display nicknames are never accepted as model evidence;
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

The `.cmd` wrapper works under PowerShell's default script policy and always runs `.venv\Scripts\python.exe`. The outer runner makes that same interpreter the inner Codex shell's default, pins `PYTHONPATH` to the candidate worktree's `src`, and disables package-index access. This prevents accidental global-Python fallback or root-checkout imports; explicitly launching another interpreter remains out of contract rather than technically impossible. H001 also requires an authorized local `checkpoints/focus-native-small/model.safetensors`: preflight requires the documented SHA-256, exposes only its parent path as `FOCUS_CHECKPOINT`, and never copies the excluded weight into the Git worktree. `run_cycle.ps1` is also provided for environments that permit PowerShell scripts.

Equivalent Make targets are `autonomy-preflight`, `autonomy-dry-run`, and `autonomy-execute` when GNU Make is available.

## Execution lifecycle

1. **Preflight** probes a working Codex executable, login state, current model catalog, supported CLI options, project venv, clean Git state, required artifacts, specialist profiles, bounded host-session metadata access, and the selected evaluator. The non-redistributed local checkpoint must match the public digest in `checkpoints/README.md`. On native Windows the same model-free `codex sandbox` check runs the project interpreter, imports `torch` and `pytest`, reads and hashes that checkpoint, and creates a temporary workspace sentinel. Parsing `--help` alone is not accepted as proof of a usable research runtime.
2. **Preregistration** loads the hypothesis, controls, falsifier, primary metric, allowed paths, baseline SHA-256, and trusted local-checkpoint path/digest. The outer runner writes the complete `experiment.json` and records its byte SHA-256 before Codex starts. The same bytes are required after Codex, after every host-side candidate process, before acceptance, and before staging; the parsed payload must also exactly equal the generated contract.
3. **Isolation** creates `../.focus-fabric-worktrees/<hypothesis>-<timestamp>` and records its starting commit.
4. **Agent run** injects the checked-in default and specialist role contracts, the project venv on `PATH`, its absolute path as `FOCUS_PYTHON`, and only the candidate `src` on `PYTHONPATH`. The prompt supplies platform-neutral Python gate commands; GNU Make is optional, and package installation is forbidden. The agent is forbidden to commit or alter Git history. After Codex exits, the outer runner binds the CLI's `thread.started` id to newly created host session records and requires a completed runtime record for every role and configured model.
5. **Validation** checks the final JSON schema/status, its hash, exact agreement between self-reported and Git-observed changes, history immutability, and exact file/directory boundaries. The same scope check runs again after gates and evidence generation.
6. **Trusted gates** load `autonomy/gates.json` and the original test suite from the root checkout, while importing candidate source. Python commands are pinned to the project interpreter and receive a credential-free environment. The CPU evidence gate receives the already verified root checkpoint explicitly; no ignored weight is inferred from the candidate worktree. Plotting is fixed to the headless `Agg` backend. After Codex and after every host-side candidate process, the runner verifies the root HEAD, status, baseline digest, local-checkpoint digest, and byte-level digest of every tracked root file before continuing.
7. **Holdout and decision** compare root and candidate on the same seed generated only after the agent run, then apply the H001 metric contract.
8. **Retention** leaves the branch and worktree intact for human inspection. With no `--auto-promote`, even an accepted candidate stays uncommitted. Automatic promotion is also blocked if an existing tracked test was changed or deleted; new tests may be promoted only with the candidate implementation. When promotion is explicitly enabled, the runner fixes the staged Git tree before commit, disables repository hooks with a fresh random nonexistent hooks path for each Git operation, verifies the commit's single parent, tree, and exact diff paths, and merges that verified commit hash rather than a mutable branch name. The merged root HEAD, clean status, and tracked-file byte digest must then match the validated candidate.

Root ledger and run reports are written to ignored `autonomy/state/` and `results/autonomy_runs/`, so one run does not make the next preflight dirty.

## Failure behavior

Codex command failure, missing or ambiguous parent thread metadata, role/model/provider mismatch, unavailable native workspace-write sandbox, agent `failed` or `blocked` status, missing or invalid result JSON, experiment-contract mismatch, self-report mismatch, agent-created commit, scope escape, trusted-root mutation, deterministic gate failure, holdout regression, or primary-metric failure all produce a non-promoted result. The worktree is preserved so the failure can be audited rather than erased.

The root lock is released in a `finally` block. A stale lock after a machine crash must be inspected before manual removal; never delete it while another cycle may still be running.

## Security boundary

The inner Codex process is sandboxed and has workspace network access disabled. Both Codex and gate subprocesses start from small environment allowlists: Codex retains only the paths needed to discover the existing ChatGPT login and host session metadata, while API-key-like and unrelated credentials are not inherited. The shared project venv is exposed as an executable dependency runtime, but candidate imports are forced to the isolated worktree and the agent contract forbids package-manager mutations. Command hooks remain disabled. The verifier reads only the minimal session metadata needed for parent binding, role path, provider, runtime model, and completion; it does not copy transcript content into public evidence. Old session paths are snapshotted before the run and cannot satisfy a new run.

The session files are host runtime records, not cryptographic backend attestations,
and their wire format may evolve. The parser therefore fails closed when the
required fields are absent or inconsistent. It rejects links and Windows
junctions, bounds file count and matching-record size, and publishes only
anonymous role counts rather than host session identifiers. The candidate
sandbox cannot write to this directory; a model-free Windows A/B probe confirmed
that the same process can write inside its candidate worktree but receives
access denied for the host-owned evidence location.

The outer runner can attest the child path, parent binding, provider, runtime
model, and terminal completion record. Codex encrypts the actual inter-agent task
payload in its local transcript, so the runner cannot independently attest the
exact self-contained message body; the root prompt and checked-in role contracts
remain the enforceable instruction layer for that part.

This is still not a VM or container security boundary. Candidate Python is executed on the host during tests and benchmarks. Root-integrity snapshots detect persistent writes to tracked project files; they do not prevent transient writes, access to unrelated host files, or tampering outside the tracked tree. Use a dedicated VM/container for adversarial or third-party hypotheses, and never use `--dangerously-bypass-approvals-and-sandbox` on a normal workstation.
