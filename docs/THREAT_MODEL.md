# Threat Model

## Assets

- exact context/KV or residual archive;
- protected agent policies, constraints, goals, commitments, and evidence;
- benchmark evidence and claim ledger;
- Codex credentials, Git history, and release signing keys;
- GPU/host execution environment.

## Adversaries and failures

### Untrusted context or tool output

An input may contain instructions to erase policy, fabricate measurements, exfiltrate archive content, or classify itself as trusted memory.

Control: text cannot assign its own `MemoryKind`; protected typing is an API authority. Capsules preserve source provenance. Remaining risk: a compromised caller can still misclassify records.

### Distribution-shifted queries

A query may fall outside calibration support while appearing numerically finite.

Control: certificate threshold, exact fallback, sparse exact audits, drift state. Remaining risk: abrupt attacks can precede enough audits; proxies may be confidently wrong.

### Poisoned compiler traces

An attacker may shape the query bank so the selected codec preserves attacker routes and forgets benign ones.

Control: bounded reservoir, split selection/calibration, randomized holdout, exact archive. Remaining work: robust sampling, tenant isolation, influence diagnostics, and signed trace provenance.

### Autonomous code overfitting or evidence fabrication

Codex may tune to public metrics, weaken tests, alter evidence, or add unsupported prose.

Control: preregistered scope, clean worktree, immutable root holdout evaluator, post-hoc random seed, deterministic gates, claim-path SHA-256, specialist reviewers, opt-in promotion. Remaining risk: workspace scope is not an OS sandbox and candidate code can attempt evaluator detection.

### Archive exfiltration

Exact fallback necessarily reads sensitive historical state.

Control supplied: byte telemetry only. Missing: encryption, access control, redaction, tenant boundaries, retention, secure deletion, and data-loss prevention. Production use without these is unsafe.

### Artifact tampering

A result file or semantic record may be edited after measurement.

Control: SHA-256 claim ledger and record hash chains. Remaining risk: local hashes do not provide trusted identity/time. Sign release manifests and store transparency logs externally.

### Kernel memory safety or numerical fault

A fused kernel may read out of bounds, overflow, or silently disagree with the reference.

Control: shape checks, PyTorch oracle, explicit GPU correctness harness. GPU execution is untested in this environment. Use compute sanitizers, randomized differential tests, and fail-closed dispatch before deployment.

## Trust boundaries

- Python type checks are not a security sandbox.
- Git worktrees isolate changes logically, not from the host.
- Codex workspace-write sandbox does not replace a dedicated VM/container for untrusted code.
- Exact archive and release credentials must not be accessible to candidate agent processes unless strictly needed.

## Safe deployment baseline

A production integrator should add: process/container isolation, read-only benchmark/evaluator mounts, signed manifests, secrets outside the repo, authenticated archives, encryption, audit logs, resource quotas, timeouts, kernel sanitization, and human release approval.
