#!/usr/bin/env python3
"""Gated Codex research loop for FOCUS-Fabric.

The orchestrator is intentionally conservative: every hypothesis is isolated
in a git worktree, Codex receives a predeclared experiment contract, raw JSONL
is retained, deterministic gates run after the agent exits, and the candidate
is promoted only when hard safety constraints and the declared objective pass.
A dry-run mode validates the entire plan without requiring Codex credentials.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
import secrets
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKTREE_ROOT = ROOT.parent / ".focus-fabric-worktrees"


class PipelineError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EventLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.previous = "GENESIS"
        if path.exists():
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                self.previous = json.loads(lines[-1])["digest"]

    def append(self, event: str, payload: dict[str, Any]) -> None:
        body = {
            "timestamp": now(),
            "event": event,
            "payload": payload,
            "previous_digest": self.previous,
        }
        digest = hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()
        record = {**body, "digest": digest}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(record) + "\n")
        self.previous = digest


@dataclass(frozen=True)
class Hypothesis:
    identifier: str
    title: str
    rationale: str
    controls: tuple[str, ...]
    primary_metric: str
    disconfirming_condition: str
    allowed_files: tuple[str, ...]
    status: str

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Hypothesis":
        return cls(
            identifier=payload["id"],
            title=payload["title"],
            rationale=payload["rationale"],
            controls=tuple(payload["controls"]),
            primary_metric=payload["primary_metric"],
            disconfirming_condition=payload["disconfirming_condition"],
            allowed_files=tuple(payload["allowed_files"]),
            status=payload["status"],
        )


def load_hypotheses(path: Path) -> tuple[dict[str, Any], list[Hypothesis]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise PipelineError("unsupported hypothesis schema")
    return document, [Hypothesis.from_json(item) for item in document["hypotheses"]]


def run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 300,
    environment: dict[str, str] | None = None,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(environment or {})
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and completed.returncode:
        raise PipelineError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    return completed


def git_output(args: Iterable[str], cwd: Path) -> str:
    return run(["git", *args], cwd=cwd).stdout.strip()


def assert_git_repository(root: Path) -> None:
    if run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, check=False).returncode:
        raise PipelineError("execute mode requires a git repository")


def assert_clean(root: Path) -> None:
    status = git_output(["status", "--porcelain"], root)
    if status:
        raise PipelineError("root worktree must be clean before autonomous execution")


def slug(identifier: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in identifier).strip("-")


def prompt_for(hypothesis: Hypothesis, baseline_digest: str) -> str:
    allowed = "\n".join(f"- {item}" for item in hypothesis.allowed_files)
    controls = "\n".join(f"- {item}" for item in hypothesis.controls)
    return f"""You are the root research agent for hypothesis {hypothesis.identifier}.

Read AGENTS.md and use the focus-research skill. Spawn research_scout and
memory_redteam first, wait for both, then use architecture_scientist. Use
kernel_engineer only if a tested reference requires a GPU path. After changes,
spawn benchmark_adversary, reproducibility_auditor, and claim_auditor. Preserve
negative results.

TITLE
{hypothesis.title}

RATIONALE
{hypothesis.rationale}

CONTROLS
{controls}

PRIMARY METRIC
{hypothesis.primary_metric}

PREDECLARED DISCONFIRMING CONDITION
{hypothesis.disconfirming_condition}

ALLOWED PATH PREFIXES
{allowed}

BASELINE ARTIFACT SHA-256
{baseline_digest}

Requirements:
1. Create `autonomy/state/{hypothesis.identifier}.experiment.json` before code
   changes with independent variable, controls, split discipline, metrics,
   stopping rule, and disconfirming threshold.
2. Do not modify files outside allowed prefixes except the experiment record
   and candidate result artifacts.
3. Run `make gate`. Do not weaken gates or edit baseline results.
4. Finish with structured JSON matching the supplied schema. Include failures,
   risks, negative results, and exact evidence paths.
"""


def path_allowed(path: str, hypothesis: Hypothesis) -> bool:
    always = (
        f"autonomy/state/{hypothesis.identifier}.",
        "results/candidate_",
        "results/experiments/",
    )
    prefixes = (*hypothesis.allowed_files, *always)
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in prefixes)


def changed_files(worktree: Path) -> list[str]:
    output = git_output(["status", "--porcelain"], worktree)
    paths: list[str] = []
    for line in output.splitlines():
        if not line:
            continue
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value)
    return sorted(paths)


def run_gates(worktree: Path, ledger: EventLedger) -> list[dict[str, Any]]:
    gate_config = json.loads((worktree / "autonomy/gates.json").read_text(encoding="utf-8"))
    reports: list[dict[str, Any]] = []
    for gate in gate_config["commands"]:
        completed = run(
            list(gate["command"]),
            cwd=worktree,
            timeout=int(gate.get("timeout_seconds", 300)),
            environment={str(k): str(v) for k, v in gate.get("environment", {}).items()},
            check=False,
        )
        report = {
            "name": gate["name"],
            "command": gate["command"],
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        reports.append(report)
        ledger.append("gate.completed", report)
        if completed.returncode:
            break
    return reports


def deep_get(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for component in path.split("."):
        current = current[component]
    return current


def evidence_score(payload: dict[str, Any]) -> float:
    synthetic = payload["synthetic"]
    id_metrics = synthetic["splits"]["in_distribution"]["fabric_approx"]
    shift = synthetic["splits"]["distribution_shift"]["fabric_guarded"]
    memory = synthetic["memory"]
    end = payload["end_to_end"]["teacher_forced"]
    rate = memory["fabric_active_bytes"] / memory["full_exact_bytes"]
    disagreement = 1.0 - end["argmax_token_agreement"]
    return (
        float(id_metrics["output_nmse"])
        + 0.25 * float(shift["output_nmse"])
        + 0.04 * float(rate)
        + 10.0 * disagreement
    )


def compare_candidate(root: Path, worktree: Path) -> dict[str, Any]:
    baseline_path = root / "results/fabric_benchmark.json"
    candidate_path = worktree / "results/candidate_benchmark.json"
    if not candidate_path.exists():
        return {"passed": False, "reason": "candidate benchmark missing"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_score = evidence_score(baseline)
    candidate_score = evidence_score(candidate)
    baseline_agreement = baseline["end_to_end"]["teacher_forced"]["argmax_token_agreement"]
    candidate_agreement = candidate["end_to_end"]["teacher_forced"]["argmax_token_agreement"]
    candidate_free = candidate["end_to_end"]["free_running"]["sequence_agreement"]
    hard_pass = (
        candidate_agreement + 1e-12 >= baseline_agreement
        and bool(candidate_free)
        and candidate["repeated_compaction"]["final"]["invalid_codec_outputs"] == 0
    )
    relative_improvement = (baseline_score - candidate_score) / max(abs(baseline_score), 1e-12)
    return {
        "passed": bool(hard_pass and relative_improvement >= 0.005),
        "hard_safety_passed": bool(hard_pass),
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "relative_improvement": relative_improvement,
        "baseline_sha256": sha256_file(baseline_path),
        "candidate_sha256": sha256_file(candidate_path),
    }


def run_external_holdout(
    root: Path,
    worktree: Path,
    hypothesis: Hypothesis,
    ledger: EventLedger,
) -> dict[str, Any]:
    """Evaluate root and candidate code on the same post-hoc random cases."""

    seed = secrets.randbits(31)
    evaluator = root / "scripts/autonomy/holdout_evaluator.py"
    candidate_output = (
        worktree / "results/experiments" / hypothesis.identifier / "holdout.json"
    )
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="focus-holdout-") as temporary:
        baseline_output = Path(temporary) / "baseline.json"
        environment = {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            # The evaluator prepends --source itself; an inherited PYTHONPATH
            # must not accidentally import the root package for the candidate.
            "PYTHONPATH": "",
        }
        baseline_run = run(
            [
                sys.executable,
                str(evaluator),
                "--source",
                str(root / "src"),
                "--seed",
                str(seed),
                "--cases",
                "4",
                "--output",
                str(baseline_output),
            ],
            cwd=root,
            timeout=900,
            environment=environment,
            check=False,
        )
        candidate_run = run(
            [
                sys.executable,
                str(evaluator),
                "--source",
                str(worktree / "src"),
                "--seed",
                str(seed),
                "--cases",
                "4",
                "--output",
                str(candidate_output),
            ],
            cwd=worktree,
            timeout=900,
            environment=environment,
            check=False,
        )
        if not baseline_output.exists() or not candidate_output.exists():
            report = {
                "passed": False,
                "reason": "holdout evaluator did not produce both artifacts",
                "baseline_returncode": baseline_run.returncode,
                "candidate_returncode": candidate_run.returncode,
            }
            ledger.append("holdout.failed", report)
            return report
        baseline = json.loads(baseline_output.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_output.read_text(encoding="utf-8"))
    baseline_objective = float(baseline["objective"])
    candidate_objective = float(candidate["objective"])
    # The randomized holdout is a non-regression gate.  The public benchmark
    # supplies the improvement requirement; the post-hoc cases prevent a
    # narrowly overfit improvement from being promoted.
    relative_change = (candidate_objective - baseline_objective) / max(
        abs(baseline_objective), 1e-12
    )
    report = {
        "passed": bool(
            baseline.get("passed")
            and candidate.get("passed")
            and relative_change <= 0.02
        ),
        "seed": seed,
        "baseline_objective": baseline_objective,
        "candidate_objective": candidate_objective,
        "relative_change": relative_change,
        "candidate_artifact": str(candidate_output.relative_to(worktree)),
        "candidate_sha256": sha256_file(candidate_output),
        "baseline_safety_passed": bool(baseline.get("passed")),
        "candidate_safety_passed": bool(candidate.get("passed")),
        "tolerance": "candidate objective may regress by at most 2% on randomized holdout",
    }
    ledger.append("holdout.completed", report)
    return report

def acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise PipelineError(f"autonomy lock already exists: {path}") from error
    os.write(descriptor, f"pid={os.getpid()} time={now()}\n".encode())
    return descriptor


def dry_run_plan(root: Path, hypotheses: list[Hypothesis], codex: str) -> dict[str, Any]:
    baseline = root / "results/fabric_benchmark.json"
    digest = sha256_file(baseline) if baseline.exists() else "MISSING"
    plans = []
    for hypothesis in hypotheses:
        plans.append(
            {
                "hypothesis": hypothesis.identifier,
                "branch": f"autonomy/{slug(hypothesis.identifier)}-TIMESTAMP",
                "codex_command": [
                    codex,
                    "exec",
                    "--json",
                    "--sandbox",
                    "workspace-write",
                    "--ask-for-approval",
                    "never",
                    "--output-schema",
                    "autonomy/schemas/agent_result.schema.json",
                    "--output-last-message",
                    f"results/experiments/{hypothesis.identifier}/agent-result.json",
                    "-",
                ],
                "prompt_sha256": hashlib.sha256(
                    prompt_for(hypothesis, digest).encode("utf-8")
                ).hexdigest(),
                "allowed_files": list(hypothesis.allowed_files),
                "gates": json.loads((root / "autonomy/gates.json").read_text())["commands"],
                "external_holdout": {
                    "evaluator": "scripts/autonomy/holdout_evaluator.py",
                    "seed_timing": "generated after the Codex run",
                    "policy": "candidate safety must pass and randomized objective may regress by at most 2% versus root baseline"
                },
            }
        )
    return {
        "mode": "dry-run",
        "timestamp": now(),
        "codex_available": shutil.which(codex) is not None,
        "baseline_sha256": digest,
        "plans": plans,
    }


def execute_one(
    root: Path,
    hypothesis: Hypothesis,
    *,
    codex: str,
    worktree_root: Path,
    auto_promote: bool,
    ledger: EventLedger,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    branch = f"autonomy/{slug(hypothesis.identifier)}-{timestamp}"
    worktree = worktree_root / f"{slug(hypothesis.identifier)}-{timestamp}"
    baseline = root / "results/fabric_benchmark.json"
    baseline_digest = sha256_file(baseline)
    run(["git", "worktree", "add", "-b", branch, str(worktree), "HEAD"], cwd=root)
    experiment_dir = worktree / "results" / "experiments" / hypothesis.identifier
    experiment_dir.mkdir(parents=True, exist_ok=True)
    event_path = experiment_dir / "codex-events.jsonl"
    result_path = experiment_dir / "agent-result.json"
    prompt = prompt_for(hypothesis, baseline_digest)
    command = [
        codex,
        "exec",
        "--json",
        "--sandbox",
        "workspace-write",
        "--ask-for-approval",
        "never",
        "--output-schema",
        "autonomy/schemas/agent_result.schema.json",
        "--output-last-message",
        str(result_path.relative_to(worktree)),
        "-",
    ]
    ledger.append(
        "codex.started",
        {"hypothesis": hypothesis.identifier, "branch": branch, "worktree": str(worktree)},
    )
    completed = run(command, cwd=worktree, timeout=7200, input_text=prompt, check=False)
    event_path.write_text(completed.stdout, encoding="utf-8")
    ledger.append(
        "codex.completed",
        {
            "hypothesis": hypothesis.identifier,
            "returncode": completed.returncode,
            "events_sha256": sha256_file(event_path),
            "stderr_tail": completed.stderr[-2000:],
        },
    )
    if completed.returncode:
        return {
            "hypothesis": hypothesis.identifier,
            "status": "codex_failed",
            "branch": branch,
            "worktree": str(worktree),
            "returncode": completed.returncode,
        }
    changes = changed_files(worktree)
    forbidden = [path for path in changes if not path_allowed(path, hypothesis)]
    if forbidden:
        ledger.append("scope.failed", {"hypothesis": hypothesis.identifier, "forbidden": forbidden})
        return {
            "hypothesis": hypothesis.identifier,
            "status": "scope_failed",
            "forbidden_files": forbidden,
            "branch": branch,
            "worktree": str(worktree),
        }
    gate_reports = run_gates(worktree, ledger)
    gates_passed = all(report["returncode"] == 0 for report in gate_reports)
    holdout = (
        run_external_holdout(root, worktree, hypothesis, ledger)
        if gates_passed
        else {"passed": False, "reason": "deterministic gates failed"}
    )
    comparison = compare_candidate(root, worktree) if gates_passed else {
        "passed": False,
        "reason": "one or more gates failed",
    }
    if not holdout.get("passed"):
        comparison = {**comparison, "passed": False, "holdout_blocked": True}
    result = {
        "hypothesis": hypothesis.identifier,
        "status": "accepted" if comparison.get("passed") else "rejected",
        "branch": branch,
        "worktree": str(worktree),
        "changed_files": changes,
        "gates": gate_reports,
        "holdout": holdout,
        "comparison": comparison,
    }
    (experiment_dir / "orchestrator-result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if comparison.get("passed"):
        run(["git", "add", "-A"], cwd=worktree)
        run(
            [
                "git",
                "commit",
                "-m",
                f"experiment: {hypothesis.identifier}",
            ],
            cwd=worktree,
        )
        if auto_promote:
            assert_clean(root)
            run(["git", "merge", "--ff-only", branch], cwd=root)
            result["promoted"] = True
    ledger.append("hypothesis.completed", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "execute"), default="dry-run")
    parser.add_argument("--hypothesis", action="append", default=[])
    parser.add_argument("--max-hypotheses", type=int, default=1)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--worktree-root", type=Path, default=DEFAULT_WORKTREE_ROOT)
    parser.add_argument("--auto-promote", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "autonomy_dry_run.json",
    )
    args = parser.parse_args()
    document, hypotheses = load_hypotheses(ROOT / "autonomy/hypotheses.json")
    selected = [item for item in hypotheses if item.status == "pending"]
    if args.hypothesis:
        selected = [item for item in selected if item.identifier in set(args.hypothesis)]
    selected = selected[: args.max_hypotheses]
    if not selected:
        raise PipelineError("no pending hypotheses matched")
    if args.mode == "dry-run":
        report = dry_run_plan(ROOT, selected, args.codex)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if shutil.which(args.codex) is None:
        raise PipelineError(f"Codex executable not found: {args.codex}")
    assert_git_repository(ROOT)
    assert_clean(ROOT)
    lock_path = ROOT / "autonomy/state/pipeline.lock"
    descriptor = acquire_lock(lock_path)
    ledger = EventLedger(ROOT / "autonomy/state/events.jsonl")
    try:
        args.worktree_root.mkdir(parents=True, exist_ok=True)
        results = [
            execute_one(
                ROOT,
                hypothesis,
                codex=args.codex,
                worktree_root=args.worktree_root,
                auto_promote=args.auto_promote,
                ledger=ledger,
            )
            for hypothesis in selected
        ]
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (PipelineError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"autonomy pipeline failed: {error}", file=sys.stderr)
        raise SystemExit(2)
