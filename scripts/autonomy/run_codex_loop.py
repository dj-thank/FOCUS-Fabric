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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import secrets
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKTREE_ROOT = ROOT.parent / ".focus-fabric-worktrees"
SUPPORTED_EVALUATORS = {"fabric_benchmark_v1"}
MAX_CODEX_SESSION_FILES = 100_000
MAX_CODEX_SESSION_BYTES = 16 * 1024 * 1024
TRUSTED_CHECKPOINT_RELATIVE_PATH = Path(
    "checkpoints/focus-native-small/model.safetensors"
)
TRUSTED_CHECKPOINT_SHA256 = (
    "348e4d7699060add3a155b961e2998bcbf5ff071b14272ce8699e21507a0631a"
)
REQUIRED_RESEARCH_ROLES = {
    "research_scout",
    "memory_redteam",
    "architecture_scientist",
    "benchmark_adversary",
    "reproducibility_auditor",
    "claim_auditor",
}


class PipelineError(RuntimeError):
    pass


class RootIntegrityError(PipelineError):
    pass


class PromotionIntegrityError(PipelineError):
    pass


class CandidateIntegrityError(PipelineError):
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


def assert_file_sha256(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise CandidateIntegrityError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise CandidateIntegrityError(
            f"{label} changed: expected {expected_sha256}, observed {actual}"
        )


def trusted_checkpoint_path(root: Path) -> Path:
    """Resolve the fixed checkpoint path without accepting link escapes."""

    resolved_root = root.resolve()
    current = root
    is_junction = getattr(os.path, "isjunction", lambda _: False)
    for part in TRUSTED_CHECKPOINT_RELATIVE_PATH.parts:
        current = current / part
        if current.is_symlink() or is_junction(current):
            raise PipelineError(
                f"trusted checkpoint path must not contain links or junctions: {current}"
            )
    checkpoint = current.resolve(strict=False)
    if not checkpoint.is_relative_to(resolved_root):
        raise PipelineError(f"trusted checkpoint resolves outside repository: {checkpoint}")
    return checkpoint


def trusted_checkpoint_file(root: Path) -> Path:
    """Return the authorized local checkpoint only when its bytes are exact."""

    checkpoint = trusted_checkpoint_path(root)
    if not checkpoint.is_file():
        raise PipelineError(f"trusted checkpoint is missing: {checkpoint}")
    observed = sha256_file(checkpoint)
    if observed != TRUSTED_CHECKPOINT_SHA256:
        raise PipelineError(
            "trusted checkpoint digest mismatch: "
            f"expected {TRUSTED_CHECKPOINT_SHA256}, observed {observed}"
        )
    return checkpoint


def exact_nonnegative_integer(value: Any, label: str) -> int:
    """Parse a JSON scalar without silently truncating fractional byte counts."""

    if isinstance(value, bool):
        raise PipelineError(f"{label} must be a non-negative integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise PipelineError(f"{label} must be a finite integer-valued number")
        parsed = int(value)
    elif isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        parsed = int(value)
    else:
        raise PipelineError(f"{label} must be a non-negative integer")
    if parsed < 0:
        raise PipelineError(f"{label} must be a non-negative integer")
    return parsed


def finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Require a real finite JSON number within an optional closed interval."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PipelineError(f"{label} must be a finite number")
    if minimum is not None and parsed < minimum:
        raise PipelineError(f"{label} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise PipelineError(f"{label} must be <= {maximum}")
    return parsed


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
class HoldoutPolicy:
    """Preregistered paired-holdout acceptance policy."""

    cases: int = 4
    max_aggregate_regression: float = 0.02
    max_case_regression: float = 0.02
    min_effect_absolute: float = 1e-8
    min_effect_relative: float = 1e-6
    min_effect_cases: int = 1

    def __post_init__(self) -> None:
        cases = exact_nonnegative_integer(self.cases, "holdout cases")
        min_effect_cases = exact_nonnegative_integer(
            self.min_effect_cases, "minimum changed holdout cases"
        )
        max_aggregate_regression = finite_number(
            self.max_aggregate_regression,
            "max aggregate holdout regression",
            minimum=0.0,
        )
        max_case_regression = finite_number(
            self.max_case_regression,
            "max per-case holdout regression",
            minimum=0.0,
        )
        min_effect_absolute = finite_number(
            self.min_effect_absolute,
            "minimum absolute holdout effect",
            minimum=0.0,
        )
        min_effect_relative = finite_number(
            self.min_effect_relative,
            "minimum relative holdout effect",
            minimum=0.0,
        )
        if cases < 1:
            raise PipelineError("holdout cases must be positive")
        if not 1 <= min_effect_cases <= cases:
            raise PipelineError("minimum changed holdout cases must be within holdout cases")
        if min_effect_absolute == 0.0 and min_effect_relative == 0.0:
            raise PipelineError("at least one minimum holdout effect must be positive")
        # Direct constructors are used by trusted tests; reject values that only
        # compare equal after lossy parsing rather than silently normalizing them.
        if (
            cases != self.cases
            or min_effect_cases != self.min_effect_cases
            or max_aggregate_regression != self.max_aggregate_regression
            or max_case_regression != self.max_case_regression
            or min_effect_absolute != self.min_effect_absolute
            or min_effect_relative != self.min_effect_relative
        ):
            raise PipelineError("holdout policy fields must use canonical numeric types")

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> "HoldoutPolicy":
        values = {} if payload is None else payload
        if not isinstance(values, dict):
            raise PipelineError("holdout_policy must be an object")
        allowed = {
            "cases",
            "max_aggregate_regression",
            "max_case_regression",
            "min_effect_absolute",
            "min_effect_relative",
            "min_effect_cases",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise PipelineError(f"unknown holdout_policy fields: {unknown}")
        policy = cls(
            cases=exact_nonnegative_integer(values.get("cases", 4), "holdout cases"),
            max_aggregate_regression=finite_number(
                values.get("max_aggregate_regression", 0.02),
                "max aggregate holdout regression",
                minimum=0.0,
            ),
            max_case_regression=finite_number(
                values.get("max_case_regression", 0.02),
                "max per-case holdout regression",
                minimum=0.0,
            ),
            min_effect_absolute=finite_number(
                values.get("min_effect_absolute", 1e-8),
                "minimum absolute holdout effect",
                minimum=0.0,
            ),
            min_effect_relative=finite_number(
                values.get("min_effect_relative", 1e-6),
                "minimum relative holdout effect",
                minimum=0.0,
            ),
            min_effect_cases=exact_nonnegative_integer(
                values.get("min_effect_cases", 1), "minimum changed holdout cases"
            ),
        )
        return policy

    def to_json(self) -> dict[str, int | float]:
        return {
            "cases": self.cases,
            "max_aggregate_regression": self.max_aggregate_regression,
            "max_case_regression": self.max_case_regression,
            "min_effect_absolute": self.min_effect_absolute,
            "min_effect_relative": self.min_effect_relative,
            "min_effect_cases": self.min_effect_cases,
        }


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
    evaluator: str | None = None
    holdout_policy: HoldoutPolicy = HoldoutPolicy()

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
            evaluator=payload.get("evaluator"),
            holdout_policy=HoldoutPolicy.from_json(payload.get("holdout_policy")),
        )


@dataclass(frozen=True)
class CodexRuntime:
    path: Path
    version: str


@dataclass(frozen=True)
class RootSnapshot:
    head: str
    status: str
    tracked_tree_sha256: str
    baseline_sha256: str
    checkpoint_sha256: str


def codex_candidates(requested: str) -> list[Path]:
    """Return Codex executables in precedence order without trusting PATH alone."""

    candidates: list[Path] = []
    configured = os.environ.get("FOCUS_CODEX")
    if configured:
        candidates.append(Path(configured).expanduser())
    if requested != "codex":
        explicit = Path(requested).expanduser()
        resolved = shutil.which(requested)
        candidates.append(Path(resolved) if resolved else explicit)
    else:
        if os.name == "nt":
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                desktop_bin = Path(local_app_data) / "OpenAI" / "Codex" / "bin"
                if desktop_bin.is_dir():
                    desktop_runtimes = sorted(
                        desktop_bin.rglob("codex.exe"),
                        key=lambda item: item.stat().st_mtime_ns,
                        reverse=True,
                    )
                    candidates.extend(desktop_runtimes)
        resolved = shutil.which(requested)
        if resolved:
            candidates.append(Path(resolved))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate.resolve(strict=False)))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_codex_runtime(requested: str) -> CodexRuntime:
    failures: list[str] = []
    for candidate in codex_candidates(requested):
        try:
            completed = run(
                [str(candidate), "--version"],
                cwd=ROOT,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"{candidate}: {type(error).__name__}")
            continue
        version = (completed.stdout or completed.stderr).strip()
        if completed.returncode == 0 and version:
            return CodexRuntime(path=candidate, version=version.splitlines()[0])
        failures.append(f"{candidate}: exit {completed.returncode}")
    detail = "; ".join(failures) if failures else "no candidates discovered"
    raise PipelineError(f"no working Codex executable ({detail})")


def native_sandbox_overrides(platform_name: str) -> list[str]:
    """Return the OS-level sandbox implementation required by isolated exec.

    ``--ignore-user-config`` is intentional for reproducibility, but on native
    Windows it also removes the implementation selector that makes
    ``workspace-write`` writable.  Keep that selector explicit in every
    generated command instead of relying on a machine-local default.
    """

    if platform_name == "nt":
        return ["-c", 'windows.sandbox="elevated"']
    return []


def resolved_repository_child(root: Path, path: Path, *, label: str) -> Path:
    """Resolve a path and reject links or junctions that escape the repository."""

    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or not resolved_path.is_relative_to(resolved_root):
        raise PipelineError(f"{label} resolves outside repository: {resolved_path}")
    return resolved_path


def build_codex_exec_command(
    runtime: CodexRuntime,
    *,
    schema_path: Path,
    result_path: Path,
    project_root: Path = ROOT,
    shell_environment: dict[str, str] | None = None,
) -> list[str]:
    """Build a noninteractive command using options accepted by current Codex."""

    if shell_environment is None:
        shell_environment = candidate_codex_environment(ROOT, project_root)
    shell_keys = (
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "FOCUS_PYTHON",
        "FOCUS_CHECKPOINT",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PIP_NO_INDEX",
        "PIP_DISABLE_PIP_VERSION_CHECK",
    )
    missing_shell_keys = [key for key in shell_keys if key not in shell_environment]
    if missing_shell_keys:
        raise PipelineError(
            "candidate shell environment is incomplete: " + ", ".join(missing_shell_keys)
        )
    shell_table = ",".join(
        f"{key}={json.dumps(shell_environment[key])}" for key in shell_keys
    )

    command = [
        str(runtime.path),
        "--ask-for-approval",
        "never",
        "exec",
        "--strict-config",
        "--ignore-user-config",
        "--enable",
        "multi_agent",
        "-m",
        project_root_model(project_root),
        "-c",
        'model_reasoning_effort="xhigh"',
        "-c",
        "agents.max_threads=6",
        "-c",
        "agents.max_depth=1",
        "-c",
        "agents.job_max_runtime_seconds=1800",
        "-c",
        "agents.interrupt_message=true",
        "-c",
        "sandbox_workspace_write.network_access=false",
        *native_sandbox_overrides(os.name),
        "-c",
        "allow_login_shell=false",
        "-c",
        'shell_environment_policy.inherit="core"',
        "-c",
        f"shell_environment_policy.set={{{shell_table}}}",
        "--json",
        "--color",
        "never",
        "--sandbox",
        "workspace-write",
        "--ephemeral",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(result_path),
        "-",
    ]
    profile_directory = project_root / ".codex" / "agents"
    profiles = sorted(profile_directory.glob("*.toml"))
    if not profiles:
        raise PipelineError(f"no project agent profiles found: {profile_directory}")
    insertion_point = command.index("--json")
    overrides: list[str] = []
    for profile in profiles:
        content = profile.read_text(encoding="utf-8")
        name_match = re.search(r'^name\s*=\s*"([a-z0-9_]+)"\s*$', content, re.MULTILINE)
        description_match = re.search(
            r'^description\s*=\s*"([^"]+)"\s*$', content, re.MULTILINE
        )
        if not name_match or not description_match:
            raise PipelineError(f"invalid agent profile metadata: {profile}")
        name = name_match.group(1)
        description = description_match.group(1)
        overrides.extend(
            [
                "-c",
                f"agents.{name}.description={json.dumps(description)}",
                "-c",
                f"agents.{name}.config_file={json.dumps(str(profile.resolve()))}",
            ]
        )
    command[insertion_point:insertion_point] = overrides
    return command


def probe_codex_workspace_write(
    runtime: CodexRuntime,
    root: Path,
    *,
    environment: dict[str, str],
    platform_name: str = os.name,
    python_executable: Path | None = None,
    checkpoint_file: Path | None = None,
) -> tuple[bool, str]:
    """Prove the native sandbox can run project deps and write its workspace."""

    if platform_name != "nt":
        return True, "native Windows workspace-write probe not required"

    state_path = root / "autonomy" / "state"
    state_root = resolved_repository_child(
        root,
        state_path,
        label="sandbox state directory",
    )
    state_root.mkdir(parents=True, exist_ok=True)
    # Re-resolve after creation so a raced junction replacement cannot move
    # the probe or its cleanup outside the repository boundary.
    state_root = resolved_repository_child(
        root,
        state_path,
        label="sandbox state directory",
    )
    probe_root = resolved_repository_child(
        root,
        state_root / f"preflight-sandbox-{secrets.token_hex(8)}",
        label="sandbox probe directory",
    )
    if not probe_root.is_relative_to(state_root):
        raise PipelineError(f"sandbox probe directory escaped state root: {probe_root}")
    runtime_python = Path(python_executable or sys.executable).resolve()
    if not runtime_python.is_file():
        return False, f"project runtime is missing: {runtime_python}"
    probe_root.mkdir(parents=False, exist_ok=False)
    sentinel = probe_root / "workspace-write.ok"
    checkpoint_probe = ""
    if checkpoint_file is not None:
        checkpoint_probe = (
            "checkpoint=Path("
            + json.dumps(str(checkpoint_file.resolve()))
            + "); "
            + "observed=hashlib.sha256(checkpoint.read_bytes()).hexdigest(); "
            + "assert observed=="
            + json.dumps(TRUSTED_CHECKPOINT_SHA256)
            + ", observed; "
        )
    command = [
        str(runtime.path),
        "sandbox",
        *native_sandbox_overrides(platform_name),
        "-P",
        ":workspace",
        "-C",
        str(probe_root),
        str(runtime_python),
        "-c",
        (
            "from pathlib import Path; import hashlib, pytest, torch; "
            + checkpoint_probe
            + "Path('workspace-write.ok').write_text('ok', encoding='ascii')"
        ),
    ]
    try:
        completed = run(
            command,
            cwd=root,
            timeout=30,
            environment=environment,
            inherit_environment=False,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-1000:]
            return False, f"native sandbox command failed: {detail or completed.returncode}"
        if not sentinel.is_file():
            return False, "native sandbox did not create the workspace-write sentinel"
        if sentinel.read_text(encoding="ascii") != "ok":
            return False, "native sandbox created an invalid workspace-write sentinel"
        detail = "project runtime and workspace-write sentinel verified"
        if checkpoint_file is not None:
            detail += "; trusted checkpoint readable"
        return True, detail
    finally:
        if os.path.lexists(probe_root):
            cleanup_root = resolved_repository_child(
                root,
                probe_root,
                label="sandbox probe cleanup directory",
            )
            cleanup_state_root = resolved_repository_child(
                root,
                state_path,
                label="sandbox state cleanup directory",
            )
            is_junction = getattr(os.path, "isjunction", lambda _: False)
            if (
                cleanup_root != probe_root
                or not cleanup_root.is_relative_to(cleanup_state_root)
                or not cleanup_root.name.startswith("preflight-sandbox-")
                or probe_root.is_symlink()
                or is_junction(probe_root)
            ):
                raise PipelineError(f"unsafe sandbox probe cleanup path: {probe_root}")
            shutil.rmtree(cleanup_root)


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
    inherit_environment: bool = True,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy() if inherit_environment else safe_subprocess_environment()
    env.update(environment or {})
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
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


def safe_subprocess_environment() -> dict[str, str]:
    """Return a small host environment without credentials or API tokens."""

    allowlisted = {
        "COMSPEC",
        "LANG",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "WINDIR",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowlisted
    }


def codex_cli_environment() -> dict[str, str]:
    """Keep ChatGPT-managed auth discovery while excluding token-like secrets."""

    environment = safe_subprocess_environment()
    for key in ("APPDATA", "CODEX_HOME", "HOME", "HOMEDRIVE", "HOMEPATH", "USERPROFILE"):
        if key in os.environ:
            environment[key] = os.environ[key]
    return environment


def candidate_codex_environment(
    root: Path,
    worktree: Path,
    *,
    checkpoint_file: Path | None = None,
) -> dict[str, str]:
    """Pin inner Codex tools to the project venv and candidate source tree."""

    resolved_root = root.resolve()
    resolved_venv = (resolved_root / ".venv").resolve()
    runtime_python = Path(sys.executable).resolve()
    try:
        in_project_venv = runtime_python.is_relative_to(resolved_venv)
    except ValueError:
        in_project_venv = False
    if not in_project_venv or not runtime_python.is_file():
        raise PipelineError(
            f"candidate runtime must use the project venv: {runtime_python}"
        )
    scripts_directory = runtime_python.parent
    checkpoint = trusted_checkpoint_path(resolved_root)
    if checkpoint_file is not None and checkpoint_file.resolve() != checkpoint:
        raise PipelineError(
            "candidate runtime checkpoint differs from the trusted repository path"
        )
    environment = codex_cli_environment()
    inherited_path = environment.get("PATH", "")
    environment.update(
        {
            "PATH": os.pathsep.join(
                item for item in (str(scripts_directory), inherited_path) if item
            ),
            "VIRTUAL_ENV": str(resolved_venv),
            "FOCUS_PYTHON": str(runtime_python),
            "FOCUS_CHECKPOINT": str(checkpoint.parent),
            "PYTHONPATH": str((worktree / "src").resolve()),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    return environment


def codex_session_root(environment: dict[str, str]) -> Path:
    """Locate host-owned Codex session metadata without trusting the candidate."""

    configured = environment.get("CODEX_HOME")
    if configured:
        return (Path(configured).expanduser() / "sessions").resolve(strict=False)
    user_home = environment.get("USERPROFILE") or environment.get("HOME")
    if not user_home:
        raise PipelineError("Codex session root is unavailable: no CODEX_HOME or user home")
    return (Path(user_home).expanduser() / ".codex" / "sessions").resolve(strict=False)


def snapshot_codex_session_files(session_root: Path) -> frozenset[Path]:
    """Snapshot existing session logs so an old matching record cannot be replayed."""

    return frozenset(iter_codex_session_files(session_root))


def iter_codex_session_files(session_root: Path) -> Iterator[Path]:
    """Yield bounded regular session files without following links or junctions."""

    if not session_root.is_dir():
        return
    is_junction = getattr(os.path, "isjunction", lambda _: False)
    if session_root.is_symlink() or is_junction(session_root):
        raise PipelineError(f"Codex session root must not be a link: {session_root}")
    resolved_root = session_root.resolve()
    count = 0
    for current, directory_names, file_names in os.walk(
        session_root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        safe_directories: list[str] = []
        for name in sorted(directory_names):
            child = current_path / name
            if child.is_symlink() or is_junction(child):
                continue
            try:
                resolved_child = child.resolve()
            except OSError:
                continue
            if resolved_child.is_relative_to(resolved_root):
                safe_directories.append(name)
        directory_names[:] = safe_directories
        for name in sorted(file_names):
            if not name.endswith(".jsonl"):
                continue
            path = current_path / name
            if path.is_symlink() or is_junction(path) or not path.is_file():
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if not resolved.is_relative_to(resolved_root):
                continue
            count += 1
            if count > MAX_CODEX_SESSION_FILES:
                raise PipelineError(
                    "Codex session metadata exceeds the bounded file-count limit"
                )
            yield resolved


def codex_thread_id(events_jsonl: str) -> str:
    """Extract the parent thread id only from Codex's top-level JSONL envelope."""

    identifiers: set[str] = set()
    for line in events_jsonl.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("type") != "thread.started":
            continue
        identifier = str(record.get("thread_id", ""))
        if re.fullmatch(r"[A-Za-z0-9-]+", identifier):
            identifiers.add(identifier)
    if len(identifiers) != 1:
        raise PipelineError(
            "Codex JSONL must contain exactly one valid thread.started identifier"
        )
    return next(iter(identifiers))


def git_command(args: Iterable[str], cwd: Path) -> list[str]:
    safe_directory = cwd.resolve().as_posix()
    return ["git", "-c", f"safe.directory={safe_directory}", *args]


def git_output(args: Iterable[str], cwd: Path) -> str:
    return run(git_command(args, cwd), cwd=cwd).stdout.strip()


def assert_git_repository(root: Path) -> None:
    if run(
        git_command(["rev-parse", "--is-inside-work-tree"], root),
        cwd=root,
        check=False,
    ).returncode:
        raise PipelineError("execute mode requires a git repository")


def assert_clean(root: Path) -> None:
    status = git_output(["status", "--porcelain"], root)
    if status:
        raise PipelineError("root worktree must be clean before autonomous execution")


def tracked_tree_sha256(root: Path) -> str:
    """Hash every tracked path's actual bytes, independent of index status flags."""

    raw_paths = run(git_command(["ls-files", "-z", "--"], root), cwd=root).stdout
    digest = hashlib.sha256()
    for relative in sorted(path for path in raw_paths.split("\0") if path):
        path = root / relative
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode("utf-8"))
        elif path.is_file():
            digest.update(b"file\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise RootIntegrityError(f"tracked root path is missing or invalid: {relative}")
        digest.update(b"\0")
    return digest.hexdigest()


def capture_root_snapshot(root: Path) -> RootSnapshot:
    baseline = root / "results/fabric_benchmark.json"
    try:
        checkpoint = trusted_checkpoint_path(root)
        checkpoint_sha256 = (
            sha256_file(checkpoint) if checkpoint.is_file() else "MISSING"
        )
    except (OSError, PipelineError) as error:
        raise RootIntegrityError(f"trusted checkpoint path is invalid: {error}") from error
    return RootSnapshot(
        head=git_output(["rev-parse", "HEAD"], root),
        status=git_output(["status", "--porcelain"], root),
        tracked_tree_sha256=tracked_tree_sha256(root),
        baseline_sha256=sha256_file(baseline),
        checkpoint_sha256=checkpoint_sha256,
    )


def assert_root_unchanged(root: Path, snapshot: RootSnapshot) -> None:
    current = capture_root_snapshot(root)
    differences = [
        field
        for field in (
            "head",
            "status",
            "tracked_tree_sha256",
            "baseline_sha256",
            "checkpoint_sha256",
        )
        if getattr(current, field) != getattr(snapshot, field)
    ]
    if differences:
        raise RootIntegrityError(
            "trusted root changed during candidate execution: " + ", ".join(differences)
        )


def capture_clean_candidate_commit(
    worktree: Path,
    *,
    expected_parent: str,
    expected_tree: str,
    expected_paths: Iterable[str],
) -> tuple[str, str]:
    status = git_output(["status", "--porcelain"], worktree)
    if status:
        raise PromotionIntegrityError("candidate worktree is dirty after commit")
    commit = git_output(["rev-parse", "HEAD"], worktree)
    parents = git_output(["rev-list", "--parents", "-n", "1", commit], worktree).split()
    if parents != [commit, expected_parent]:
        raise PromotionIntegrityError(
            f"candidate commit has unexpected ancestry: {parents}"
        )
    tree = git_output(["rev-parse", f"{commit}^{{tree}}"], worktree)
    if tree != expected_tree:
        raise PromotionIntegrityError(
            f"candidate commit tree differs from staged tree: {tree} != {expected_tree}"
        )
    committed_paths = set(
        git_output(
            [
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                "--no-renames",
                expected_parent,
                commit,
                "--",
            ],
            worktree,
        ).splitlines()
    )
    if committed_paths != set(expected_paths):
        raise PromotionIntegrityError(
            "candidate commit paths differ from validated paths: "
            f"committed={sorted(committed_paths)}, validated={sorted(expected_paths)}"
        )
    return commit, tracked_tree_sha256(worktree)


def assert_promotion_postcondition(
    root: Path,
    *,
    expected_head: str,
    expected_tree_sha256: str,
) -> None:
    actual_head = git_output(["rev-parse", "HEAD"], root)
    status = git_output(["status", "--porcelain"], root)
    actual_tree_sha256 = tracked_tree_sha256(root)
    differences: list[str] = []
    if actual_head != expected_head:
        differences.append("head")
    if status:
        differences.append("status")
    if actual_tree_sha256 != expected_tree_sha256:
        differences.append("tracked_tree_sha256")
    if differences:
        raise PromotionIntegrityError(
            "promotion postcondition failed: " + ", ".join(differences)
        )


def random_nonexistent_hooks_path(root: Path) -> Path:
    for _ in range(8):
        candidate = (
            root
            / "autonomy/state/disabled-git-hooks"
            / secrets.token_hex(16)
        )
        if not candidate.exists():
            return candidate
    raise PromotionIntegrityError("could not allocate a nonexistent Git hooks path")


@contextmanager
def root_integrity_guard(
    root: Path, snapshot: RootSnapshot | None
) -> Iterator[None]:
    try:
        yield
    finally:
        if snapshot is not None:
            assert_root_unchanged(root, snapshot)


def project_root_model(root: Path) -> str:
    config = root / ".codex/config.toml"
    match = re.search(
        r'^model\s*=\s*"([^"]+)"\s*$',
        config.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise PipelineError(f"project config has no model: {config}")
    return match.group(1)


def project_agent_models(root: Path) -> dict[str, str]:
    models: dict[str, str] = {}
    for profile in sorted((root / ".codex" / "agents").glob("*.toml")):
        content = profile.read_text(encoding="utf-8")
        name_match = re.search(
            r'^name\s*=\s*"([a-z0-9_]+)"\s*$', content, re.MULTILINE
        )
        model_match = re.search(
            r'^model\s*=\s*"([^"]+)"\s*$',
            content,
            re.MULTILINE,
        )
        if not name_match or not model_match:
            raise PipelineError(f"agent profile has incomplete routing metadata: {profile}")
        name = name_match.group(1)
        if name in models:
            raise PipelineError(f"duplicate agent profile name: {name}")
        models[name] = model_match.group(1)
    return models


def project_model_ids(root: Path) -> list[str]:
    return sorted({project_root_model(root), *project_agent_models(root).values()})


def preflight_report(
    root: Path,
    hypotheses: list[Hypothesis],
    requested_codex: str,
) -> dict[str, Any]:
    """Inspect every local prerequisite and fail closed before live execution."""

    blockers: list[str] = []
    warnings: list[str] = []
    try:
        git_status = git_output(["status", "--porcelain"], root)
        git_repository = True
    except (PipelineError, OSError, subprocess.TimeoutExpired) as error:
        git_status = ""
        git_repository = False
        blockers.append(f"git repository unavailable: {error}")
    root_clean = git_repository and not git_status
    if not root_clean:
        blockers.append("root worktree must be clean before execute")

    expected_venv = (root / ".venv").resolve()
    running_python = Path(sys.executable).resolve()
    try:
        venv_ready = running_python.is_relative_to(expected_venv)
    except ValueError:
        venv_ready = False
    if not venv_ready:
        blockers.append(f"pipeline must run with the project venv: {running_python}")

    required_files = [
        root / "results/fabric_benchmark.json",
        root / "autonomy/gates.json",
        root / "autonomy/schemas/agent_result.schema.json",
        root / ".codex/config.toml",
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        blockers.append("missing required files: " + ", ".join(missing_files))
    baseline_sha256 = (
        sha256_file(root / "results/fabric_benchmark.json")
        if (root / "results/fabric_benchmark.json").is_file()
        else None
    )
    checkpoint_error: str | None = None
    try:
        checkpoint_path = trusted_checkpoint_path(root)
        checkpoint_observed_sha256 = (
            sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
        )
    except (OSError, PipelineError) as error:
        checkpoint_path = root / TRUSTED_CHECKPOINT_RELATIVE_PATH
        checkpoint_observed_sha256 = None
        checkpoint_error = f"unreadable ({error})"
    checkpoint_ready = checkpoint_observed_sha256 == TRUSTED_CHECKPOINT_SHA256
    if not checkpoint_ready:
        checkpoint_detail = (
            checkpoint_error
            or (
                "missing"
                if checkpoint_observed_sha256 is None
                else f"digest {checkpoint_observed_sha256}"
            )
        )
        blockers.append(
            "authorized local checkpoint unavailable or invalid: " + checkpoint_detail
        )

    unsupported = [
        item.identifier for item in hypotheses if item.evaluator not in SUPPORTED_EVALUATORS
    ]
    if unsupported:
        blockers.append(
            "no automated evaluator configured for: " + ", ".join(unsupported)
        )

    runtime: CodexRuntime | None = None
    login_detail = "not checked"
    logged_in = False
    command_compatible = False
    catalog_models: list[str] = []
    required_models: list[str] = []
    model_catalog_ready = False
    bootstrap_ready = False
    workspace_write_ready = False
    workspace_write_detail = "not checked"
    session_metadata_ready = False
    session_metadata_detail = "not checked"
    try:
        runtime = resolve_codex_runtime(requested_codex)
        codex_environment = codex_cli_environment()
        try:
            session_root = codex_session_root(codex_environment)
            session_count = len(snapshot_codex_session_files(session_root))
            session_metadata_ready = True
            session_metadata_detail = (
                f"host session metadata accessible ({session_count} existing files)"
            )
        except (PipelineError, OSError) as error:
            session_metadata_detail = str(error)
            blockers.append(
                "Codex host session metadata is unavailable: " + str(error)
            )
        login = run(
            [str(runtime.path), "login", "status"],
            cwd=root,
            timeout=30,
            environment=codex_environment,
            inherit_environment=False,
            check=False,
        )
        login_detail = (login.stdout or login.stderr).strip()[-1000:]
        logged_in = login.returncode == 0 and "logged in" in login_detail.lower()
        if not logged_in:
            blockers.append("Codex is not logged in for the current Windows user")

        help_result = run(
            [str(runtime.path), "exec", "--help"],
            cwd=root,
            timeout=30,
            environment=codex_environment,
            inherit_environment=False,
            check=False,
        )
        required_options = (
            "--strict-config",
            "--json",
            "--sandbox",
            "--output-schema",
            "--output-last-message",
        )
        command_compatible = help_result.returncode == 0 and all(
            option in help_result.stdout for option in required_options
        )
        if not command_compatible:
            blockers.append("Codex exec does not expose the required pipeline options")

        model_result = run(
            [str(runtime.path), "debug", "models"],
            cwd=root,
            timeout=30,
            environment=codex_environment,
            inherit_environment=False,
            check=False,
        )
        if model_result.returncode == 0:
            model_payload = json.loads(model_result.stdout)
            catalog_models = sorted(
                {
                    str(item["slug"])
                    for item in model_payload.get("models", [])
                    if isinstance(item, dict) and item.get("slug")
                }
            )
        required_models = project_model_ids(root)
        missing_models = sorted(set(required_models) - set(catalog_models))
        model_catalog_ready = bool(catalog_models) and not missing_models
        if not model_catalog_ready:
            blockers.append(
                "configured Codex model(s) unavailable: "
                + (", ".join(missing_models) if missing_models else "catalog unreadable")
            )

        bootstrap_probe = build_codex_exec_command(
            runtime,
            schema_path=root / "autonomy/schemas/agent_result.schema.json",
            result_path=Path("autonomy/state/preflight-agent-result.json"),
            project_root=root,
        )
        bootstrap_probe[-1] = "--help"
        bootstrap_result = run(
            bootstrap_probe,
            cwd=root,
            timeout=30,
            environment=codex_environment,
            inherit_environment=False,
            check=False,
        )
        bootstrap_ready = bootstrap_result.returncode == 0
        if not bootstrap_ready:
            blockers.append(
                "explicit Codex agent bootstrap was rejected: "
                + bootstrap_result.stderr.strip()[-1000:]
            )

        workspace_write_ready, workspace_write_detail = probe_codex_workspace_write(
            runtime,
            root,
            environment=codex_environment,
            checkpoint_file=(checkpoint_path if checkpoint_ready else None),
        )
        if not workspace_write_ready:
            blockers.append(
                "Codex workspace-write sandbox is unavailable: "
                + workspace_write_detail
            )
    except (PipelineError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        blockers.append(f"Codex runtime preflight failed: {error}")

    dry_run_requirements = (
        git_repository,
        venv_ready,
        not missing_files,
        runtime is not None,
        command_compatible,
        model_catalog_ready,
        bootstrap_ready,
    )
    ready_for_dry_run = all(dry_run_requirements)
    ready_for_execute = (
        ready_for_dry_run
        and root_clean
        and logged_in
        and workspace_write_ready
        and checkpoint_ready
        and session_metadata_ready
        and not unsupported
    )
    if ready_for_dry_run and not ready_for_execute:
        warnings.append("dry-run is available, but live execute remains blocked")
    return {
        "mode": "preflight",
        "timestamp": now(),
        "root": str(root),
        "python": str(running_python),
        "project_venv": venv_ready,
        "git_repository": git_repository,
        "root_clean": root_clean,
        "baseline_sha256": baseline_sha256,
        "trusted_checkpoint": {
            "path": TRUSTED_CHECKPOINT_RELATIVE_PATH.as_posix(),
            "expected_sha256": TRUSTED_CHECKPOINT_SHA256,
            "observed_sha256": checkpoint_observed_sha256,
            "ready": checkpoint_ready,
        },
        "codex": (
            {"path": str(runtime.path), "version": runtime.version}
            if runtime is not None
            else None
        ),
        "logged_in": logged_in,
        "login_detail": login_detail,
        "command_compatible": command_compatible,
        "agent_bootstrap": "explicit-cli-overrides",
        "bootstrap_ready": bootstrap_ready,
        "workspace_write_ready": workspace_write_ready,
        "workspace_write_detail": workspace_write_detail,
        "session_metadata_ready": session_metadata_ready,
        "session_metadata_detail": session_metadata_detail,
        "required_models": required_models,
        "catalog_models": catalog_models,
        "model_catalog_ready": model_catalog_ready,
        "selected_hypotheses": [item.identifier for item in hypotheses],
        "unsupported_hypotheses": unsupported,
        "ready_for_dry_run": ready_for_dry_run,
        "ready_for_execute": ready_for_execute,
        "blockers": blockers,
        "warnings": warnings,
    }


def slug(identifier: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "-" for character in identifier
    ).strip("-")


def experiment_contract_for(
    hypothesis: Hypothesis, baseline_digest: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "hypothesis_id": hypothesis.identifier,
        "baseline_sha256": baseline_digest,
        "trusted_local_checkpoint": {
            "path": TRUSTED_CHECKPOINT_RELATIVE_PATH.as_posix(),
            "sha256": TRUSTED_CHECKPOINT_SHA256,
            "redistribution": "prohibited; authorized local evaluation input only",
        },
        "independent_variable": hypothesis.rationale,
        "controls": list(hypothesis.controls),
        "split_discipline": (
            "Use the committed benchmark contract for the primary comparison; "
            "the outer runner generates the randomized holdout seed only after "
            "the Codex implementation phase ends."
        ),
        "primary_metric": hypothesis.primary_metric,
        "holdout_policy": hypothesis.holdout_policy.to_json(),
        "stopping_rule": (
            "Stop after one bounded Codex execution. The outer evaluator applies "
            "the declared threshold without tuning or retrying on holdout results."
        ),
        "disconfirming_condition": hypothesis.disconfirming_condition,
    }


def prompt_for(hypothesis: Hypothesis, baseline_digest: str) -> str:
    allowed_rules = (
        *hypothesis.allowed_files,
        "results/candidate_benchmark.json",
        "results/candidate_benchmark.csv",
        "results/candidate_benchmark.png",
        f"results/experiments/{hypothesis.identifier}/",
    )
    allowed = "\n".join(f"- {item}" for item in allowed_rules)
    controls = "\n".join(f"- {item}" for item in hypothesis.controls)
    experiment_contract = experiment_contract_for(hypothesis, baseline_digest)
    return f"""You are the root research agent for hypothesis {hypothesis.identifier}.

Read AGENTS.md and follow this preregistered research protocol. For every
`spawn_agent` call, set `fork_turns="none"`, use the exact role below as
`task_name`, and put all needed experiment context plus that role's checked-in
`.codex/agents/<role>.toml` instructions in the message. A task name or nickname
is not model evidence; the outer runner verifies host runtime metadata. Spawn
research_scout and memory_redteam first, wait for both, then use
architecture_scientist. Use
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
1. The outer runner created this experiment contract before starting Codex. Read
   it from `results/experiments/{hypothesis.identifier}/experiment.json`; do not
   modify, delete, or replace it:
{json.dumps(experiment_contract, indent=2, ensure_ascii=False)}
2. Do not modify files outside allowed prefixes except the experiment record
   and candidate result artifacts.
3. Do not commit, amend, rebase, or otherwise change Git history. The outer
   orchestrator owns history and promotion.
4. Use the injected project runtime (`python` on PATH, also named by
   `FOCUS_PYTHON`) and the candidate source injected through `PYTHONPATH`.
   Do not run pip, install, uninstall, or otherwise mutate the shared runtime.
   Run these platform-neutral gate equivalents; `make` is optional and must not
   be treated as a prerequisite:
   - `python -m compileall -q src scripts tests`
   - `python -m pytest -q`
   - `python scripts/autonomy/validate_claims.py`
   - `python scripts/autonomy/detect_drift.py`
   `FOCUS_CHECKPOINT` names the hash-verified, authorized local checkpoint
   directory outside this Git worktree. Read it in place; never copy, modify,
   redistribute, or add it to Git. Generate the candidate benchmark in
   PowerShell with
   `python scripts/benchmark_fabric.py --threads 1 --checkpoint "$env:FOCUS_CHECKPOINT" --output results/candidate_benchmark.json`
   (on POSIX, use `"$FOCUS_CHECKPOINT"`).
   Do not weaken gates or edit baseline results.
5. Finish with structured JSON matching the supplied schema. Include failures,
   risks, negative results, and exact evidence paths. `changed_files` must list
   every agent-created Git-observed changed path exactly, including any candidate
   benchmark, but not the orchestrator-owned experiment contract or
   `autonomy/state/` files.
"""


def normalize_repo_path(path: str) -> str:
    raw = str(path).replace("\\", "/")
    if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        raise PipelineError(f"unsafe repository path: {path}")
    parts = raw.split("/")
    if ".." in parts:
        raise PipelineError(f"unsafe repository path: {path}")
    normalized = "/".join(part for part in parts if part not in {"", "."})
    if not normalized:
        raise PipelineError(f"unsafe repository path: {path}")
    return normalized


def path_allowed(path: str, hypothesis: Hypothesis) -> bool:
    try:
        normalized = normalize_repo_path(path)
    except PipelineError:
        return False
    rules = (
        *hypothesis.allowed_files,
        "results/candidate_benchmark.json",
        "results/candidate_benchmark.csv",
        "results/candidate_benchmark.png",
        f"results/experiments/{hypothesis.identifier}/",
    )
    for rule in rules:
        normalized_rule = rule.replace("\\", "/")
        if normalized_rule.endswith("/"):
            if normalized.startswith(normalized_rule):
                return True
        elif normalized == normalized_rule:
            return True
    return False


def changed_files(worktree: Path) -> list[str]:
    tracked = git_output(
        ["diff", "--name-only", "--no-renames", "HEAD", "--"], worktree
    )
    untracked = git_output(
        ["ls-files", "--others", "--exclude-standard", "--"], worktree
    )
    return sorted(
        {
            item.replace("\\", "/")
            for output in (tracked, untracked)
            for item in output.splitlines()
            if item.strip()
        }
    )


def candidate_blob_map(worktree: Path, paths: Iterable[str]) -> dict[str, str | None]:
    blobs: dict[str, str | None] = {}
    for item in sorted(set(paths)):
        normalized = normalize_repo_path(item)
        path = worktree / normalized
        if path.is_symlink():
            raise CandidateIntegrityError(f"candidate symlink is not allowed: {normalized}")
        if not path.exists():
            blobs[normalized] = None
            continue
        if not path.is_file():
            raise CandidateIntegrityError(f"candidate path is not a file: {normalized}")
        blobs[normalized] = git_output(
            ["hash-object", "--no-filters", "--", normalized],
            worktree,
        )
    return blobs


def staged_blob_map(worktree: Path, paths: Iterable[str]) -> dict[str, str | None]:
    blobs: dict[str, str | None] = {}
    for item in sorted(set(paths)):
        normalized = normalize_repo_path(item)
        entry = git_output(["ls-files", "--stage", "--", normalized], worktree)
        if not entry:
            blobs[normalized] = None
            continue
        fields = entry.split(maxsplit=3)
        if len(fields) != 4 or fields[2] != "0":
            raise PromotionIntegrityError(f"invalid staged entry for {normalized}: {entry}")
        blobs[normalized] = fields[1]
    return blobs


def stage_validated_paths(worktree: Path, paths: Iterable[str]) -> None:
    """Stage exact file bytes with plumbing, bypassing clean filters and hooks."""

    for item in sorted(set(paths)):
        normalized = normalize_repo_path(item)
        path = worktree / normalized
        if not path.exists():
            run(
                git_command(
                    ["update-index", "--force-remove", "--", normalized], worktree
                ),
                cwd=worktree,
            )
            continue
        if path.is_symlink() or not path.is_file():
            raise PromotionIntegrityError(
                f"only regular files can be promoted: {normalized}"
            )
        blob = git_output(
            ["hash-object", "-w", "--no-filters", "--", normalized],
            worktree,
        )
        tracked_entry = git_output(["ls-tree", "HEAD", "--", normalized], worktree)
        mode = tracked_entry.split(maxsplit=1)[0] if tracked_entry else "100644"
        if mode not in {"100644", "100755"}:
            raise PromotionIntegrityError(
                f"unsupported Git file mode for {normalized}: {mode}"
            )
        run(
            git_command(
                [
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    f"{mode},{blob},{normalized}",
                ],
                worktree,
            ),
            cwd=worktree,
        )


def orchestrator_mutable_paths(hypothesis: Hypothesis) -> set[str]:
    experiment_root = f"results/experiments/{hypothesis.identifier}"
    return {
        "results/candidate_benchmark.json",
        "results/candidate_benchmark.csv",
        "results/candidate_benchmark.png",
        f"{experiment_root}/holdout-baseline.json",
        f"{experiment_root}/holdout-candidate.json",
        f"{experiment_root}/orchestrator-result.json",
    }


def assert_candidate_path_set_unchanged(
    initial_immutable_paths: Iterable[str],
    final_changes: Iterable[str],
    hypothesis: Hypothesis,
) -> None:
    before = set(initial_immutable_paths)
    after = set(final_changes) - orchestrator_mutable_paths(hypothesis)
    if after != before:
        raise CandidateIntegrityError(
            "candidate path set changed after gates: "
            f"before={sorted(before)}, after={sorted(after)}"
        )


def validate_candidate_state(
    worktree: Path,
    hypothesis: Hypothesis,
    start_head: str,
) -> list[str]:
    current_head = git_output(["rev-parse", "HEAD"], worktree)
    if current_head != start_head:
        raise PipelineError(
            f"candidate changed Git history: start={start_head}, current={current_head}"
        )
    changes = changed_files(worktree)
    forbidden = [path for path in changes if not path_allowed(path, hypothesis)]
    if forbidden:
        raise PipelineError("candidate changed forbidden paths: " + ", ".join(forbidden))
    return changes


def existing_tracked_test_changes(root: Path, changes: list[str]) -> list[str]:
    tracked_tests = set(
        git_output(["ls-files", "--", "tests"], root).splitlines()
    )
    return sorted(path for path in changes if path in tracked_tests)


def pinned_python_command(command: list[str]) -> list[str]:
    if not command:
        raise PipelineError("gate command must not be empty")
    executable = command[0].lower()
    if executable in {"python", "python3", "py"}:
        return [sys.executable, *command[1:]]
    if executable in {"pytest", "pytest.exe"}:
        return [sys.executable, "-m", "pytest", *command[1:]]
    raise PipelineError(f"gate executable is not allowlisted: {command[0]}")


def run_gates(
    worktree: Path,
    ledger: EventLedger,
    trusted_root: Path,
    root_snapshot: RootSnapshot | None = None,
    contract_path: Path | None = None,
    contract_sha256: str | None = None,
    checkpoint_file: Path | None = None,
) -> list[dict[str, Any]]:
    gate_config = json.loads(
        (trusted_root / "autonomy/gates.json").read_text(encoding="utf-8")
    )
    runtime_tmp = worktree / "autonomy/state/tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    matplotlib_config = worktree / "autonomy/state/matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for gate in gate_config["commands"]:
        requested_command = [str(item) for item in gate["command"]]
        command = pinned_python_command(requested_command)
        if gate["name"] == "cpu-evidence":
            verified_checkpoint = trusted_checkpoint_file(trusted_root)
            if (
                checkpoint_file is None
                or checkpoint_file.resolve() != verified_checkpoint
            ):
                raise PipelineError(
                    "cpu-evidence gate requires the verified trusted checkpoint"
                )
            command.extend(["--checkpoint", str(verified_checkpoint.parent)])
            for suffix in ("json", "csv", "png"):
                (worktree / f"results/candidate_benchmark.{suffix}").unlink(
                    missing_ok=True
                )
        if command[1:3] == ["-m", "pytest"] and "--basetemp" not in command:
            command.extend(
                ["--basetemp", str(worktree / "autonomy/state/pytest-candidate")]
            )
        environment = {
            str(k): str(v) for k, v in gate.get("environment", {}).items()
        }
        environment["PYTHONPATH"] = str(worktree / "src")
        environment["TEMP"] = str(runtime_tmp)
        environment["TMP"] = str(runtime_tmp)
        environment["MPLBACKEND"] = "Agg"
        environment["MPLCONFIGDIR"] = str(matplotlib_config)
        with root_integrity_guard(trusted_root, root_snapshot):
            completed = run(
                command,
                cwd=worktree,
                timeout=int(gate.get("timeout_seconds", 300)),
                environment=environment,
                inherit_environment=False,
                check=False,
            )
        if contract_path is not None and contract_sha256 is not None:
            assert_file_sha256(contract_path, contract_sha256, "experiment contract")
        report = {
            "name": gate["name"],
            "requested_command": requested_command,
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        reports.append(report)
        ledger.append("gate.completed", report)
        if completed.returncode:
            break
    return reports


def run_trusted_root_tests(
    root: Path,
    worktree: Path,
    ledger: EventLedger,
    root_snapshot: RootSnapshot | None = None,
    contract_path: Path | None = None,
    contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the immutable root test suite against the candidate source tree."""

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-c",
        str(root / "pyproject.toml"),
        str(root / "tests"),
        "--basetemp",
        str(worktree / "autonomy/state/pytest-trusted"),
    ]
    runtime_tmp = worktree / "autonomy/state/tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    with root_integrity_guard(root, root_snapshot):
        completed = run(
            command,
            cwd=worktree,
            timeout=1200,
            environment={
                "PYTHONPATH": str(worktree / "src"),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "TEMP": str(runtime_tmp),
                "TMP": str(runtime_tmp),
            },
            inherit_environment=False,
            check=False,
        )
    if contract_path is not None and contract_sha256 is not None:
        assert_file_sha256(contract_path, contract_sha256, "experiment contract")
    report = {
        "name": "trusted-root-tests",
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    ledger.append("gate.completed", report)
    return report


def validate_agent_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineError(f"agent result missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_types: dict[str, type] = {
        "status": str,
        "summary": str,
        "changed_files": list,
        "evidence": list,
        "risks": list,
        "negative_results": list,
        "next_hypotheses": list,
    }
    missing = [key for key in required_types if key not in payload]
    if missing:
        raise PipelineError(f"agent result missing required field(s): {', '.join(missing)}")
    extra = sorted(set(payload) - set(required_types))
    if extra:
        raise PipelineError(f"agent result has unexpected field(s): {', '.join(extra)}")
    for key, expected_type in required_types.items():
        if not isinstance(payload[key], expected_type):
            raise PipelineError(f"agent result field has wrong type: {key}")
    for key in (
        "changed_files",
        "evidence",
        "risks",
        "negative_results",
        "next_hypotheses",
    ):
        if not all(isinstance(item, str) for item in payload[key]):
            raise PipelineError(f"agent result field must contain only strings: {key}")
    if payload["status"] not in {"completed", "failed", "blocked"}:
        raise PipelineError(f"agent result has invalid status: {payload['status']}")
    return payload


def validate_reported_changes(
    payload: dict[str, Any],
    actual_changes: list[str],
    ignored_paths: Iterable[str] = (),
) -> None:
    reported = {normalize_repo_path(str(item)) for item in payload["changed_files"]}
    actual = {normalize_repo_path(item) for item in actual_changes}
    ignored = {normalize_repo_path(item) for item in ignored_paths}
    reported = {item for item in reported if not item.startswith("autonomy/state/")}
    actual = {item for item in actual if not item.startswith("autonomy/state/")}
    reported -= ignored
    actual -= ignored
    if reported != actual:
        missing = sorted(actual - reported)
        overstated = sorted(reported - actual)
        raise PipelineError(
            "agent changed_files does not match Git: "
            f"unreported={missing}, not_changed={overstated}"
        )


def validate_experiment_contract(
    worktree: Path,
    hypothesis: Hypothesis,
    baseline_digest: str,
) -> dict[str, Any]:
    path = (
        worktree
        / "results"
        / "experiments"
        / hypothesis.identifier
        / "experiment.json"
    )
    if not path.is_file():
        raise PipelineError(f"experiment contract missing: {path.relative_to(worktree)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PipelineError("experiment contract must be a JSON object")
    expected = experiment_contract_for(hypothesis, baseline_digest)
    if payload != expected:
        changed = sorted(
            key
            for key in set(payload) | set(expected)
            if payload.get(key) != expected.get(key)
        )
        raise PipelineError(
            "experiment contract differs from the preregistered payload: "
            + ", ".join(changed)
        )
    return payload


def subagent_role_evidence(
    session_root: Path,
    previous_paths: frozenset[Path],
    parent_thread_id: str,
    expected_models: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Verify completed roles and actual models from host-owned runtime metadata."""

    role_sessions: dict[str, list[dict[str, Any]]] = {
        role: [] for role in expected_models
    }
    for resolved_path in iter_codex_session_files(session_root):
        if resolved_path in previous_paths:
            continue
        try:
            before_stat = resolved_path.stat()
            with resolved_path.open(encoding="utf-8") as handle:
                first_line = handle.readline(1_000_001)
                if not first_line.endswith("\n"):
                    continue
                first_record = json.loads(first_line)
                if (
                    not isinstance(first_record, dict)
                    or first_record.get("type") != "session_meta"
                ):
                    continue
                metadata = first_record.get("payload")
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("parent_thread_id") != parent_thread_id:
                    continue
                if before_stat.st_size > MAX_CODEX_SESSION_BYTES:
                    raise PipelineError(
                        "matching Codex session metadata exceeds the size limit"
                    )
                source = metadata.get("source")
                if not isinstance(source, dict):
                    continue
                subagent = source.get("subagent")
                if not isinstance(subagent, dict):
                    continue
                spawn = subagent.get("thread_spawn")
                if not isinstance(spawn, dict):
                    continue
                agent_path = str(spawn.get("agent_path", ""))
                if (
                    metadata.get("thread_source") != "subagent"
                    or str(metadata.get("agent_path", "")) != agent_path
                    or spawn.get("parent_thread_id") != parent_thread_id
                    or spawn.get("depth") != 1
                ):
                    continue
                matching_roles = [
                    role for role in expected_models if agent_path == f"/root/{role}"
                ]
                if len(matching_roles) != 1:
                    continue
                role = matching_roles[0]
                runtime_models: set[str] = set()
                collaboration_models: set[str] = set()
                terminal_record: tuple[str, str] = ("session_meta", "")
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    payload = record.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    record_type = str(record.get("type", ""))
                    terminal_record = (record_type, str(payload.get("type", "")))
                    if record_type == "turn_context":
                        model = payload.get("model")
                        if isinstance(model, str) and model:
                            runtime_models.add(model)
                        collaboration = payload.get("collaboration_mode")
                        if isinstance(collaboration, dict):
                            settings = collaboration.get("settings")
                            if isinstance(settings, dict):
                                collaboration_model = settings.get("model")
                                if (
                                    isinstance(collaboration_model, str)
                                    and collaboration_model
                                ):
                                    collaboration_models.add(collaboration_model)
            after_stat = resolved_path.stat()
        except (OSError, json.JSONDecodeError):
            continue
        if (before_stat.st_size, before_stat.st_mtime_ns) != (
            after_stat.st_size,
            after_stat.st_mtime_ns,
        ):
            raise PipelineError("Codex session metadata changed during verification")
        completed = terminal_record == ("event_msg", "task_complete")
        session_id = str(metadata.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9-]+", session_id):
            continue
        role_sessions[role].append(
            {
                "session_id": session_id,
                "provider": str(metadata.get("model_provider", "")),
                "runtime_models": runtime_models,
                "collaboration_models": collaboration_models,
                "completed": completed,
            }
        )

    report: dict[str, dict[str, Any]] = {}
    for role, expected_model in sorted(expected_models.items()):
        sessions = role_sessions[role]
        completed_sessions = [item for item in sessions if item["completed"]]
        observed_models = sorted(
            {
                model
                for item in sessions
                for model in (
                    item["runtime_models"] | item["collaboration_models"]
                )
            }
        )
        observed_providers = sorted({item["provider"] for item in sessions})
        model_routed = bool(completed_sessions) and all(
            item["runtime_models"] == {expected_model}
            and item["collaboration_models"] == {expected_model}
            for item in completed_sessions
        )
        provider_routed = bool(completed_sessions) and all(
            item["provider"] == "openai" for item in completed_sessions
        )
        completed_ids = sorted(item["session_id"] for item in completed_sessions)
        incomplete_ids = sorted(
            item["session_id"] for item in sessions if not item["completed"]
        )
        report[role] = {
            "expected_model": expected_model,
            "completed_session_ids": completed_ids,
            "incomplete_session_ids": incomplete_ids,
            "observed_models": observed_models,
            "observed_providers": observed_providers,
            "model_routed": model_routed,
            "provider_routed": provider_routed,
            "passed": bool(completed_ids and model_routed and provider_routed),
        }
    return report


def compare_candidate(
    root: Path,
    worktree: Path,
    hypothesis: Hypothesis,
) -> dict[str, Any]:
    if hypothesis.evaluator != "fabric_benchmark_v1":
        return {
            "passed": False,
            "reason": f"no automated evaluator configured: {hypothesis.evaluator or 'none'}",
            "primary_metric": hypothesis.primary_metric,
        }
    baseline_path = root / "results/fabric_benchmark.json"
    candidate_path = worktree / "results/candidate_benchmark.json"
    if not candidate_path.exists():
        return {"passed": False, "reason": "candidate benchmark missing"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    try:
        baseline_metric_value = baseline["synthetic"]["splits"]["in_distribution"][
            "fabric_approx"
        ]["output_nmse"]
        candidate_metric_value = candidate["synthetic"]["splits"]["in_distribution"][
            "fabric_approx"
        ]["output_nmse"]
        baseline_bytes_value = baseline["synthetic"]["memory"]["fabric_active_bytes"]
        candidate_bytes_value = candidate["synthetic"]["memory"]["fabric_active_bytes"]
        baseline_agreement_value = baseline["end_to_end"]["teacher_forced"][
            "argmax_token_agreement"
        ]
        candidate_agreement_value = candidate["end_to_end"]["teacher_forced"][
            "argmax_token_agreement"
        ]
        candidate_free = candidate["end_to_end"]["free_running"]["sequence_agreement"]
        invalid_outputs_value = candidate["repeated_compaction"]["final"][
            "invalid_codec_outputs"
        ]
    except (KeyError, TypeError) as error:
        raise PipelineError(f"benchmark artifact has invalid shape: {error}") from error

    baseline_metric = finite_number(
        baseline_metric_value, "baseline output_nmse", minimum=0.0
    )
    candidate_metric = finite_number(
        candidate_metric_value, "candidate output_nmse", minimum=0.0
    )
    baseline_active_bytes = exact_nonnegative_integer(
        baseline_bytes_value, "baseline fabric_active_bytes"
    )
    candidate_active_bytes = exact_nonnegative_integer(
        candidate_bytes_value, "candidate fabric_active_bytes"
    )
    if baseline_active_bytes == 0 or candidate_active_bytes == 0:
        raise PipelineError("fabric_active_bytes must be positive")
    active_bytes_matched = candidate_active_bytes == baseline_active_bytes
    baseline_agreement = finite_number(
        baseline_agreement_value,
        "baseline argmax_token_agreement",
        minimum=0.0,
        maximum=1.0,
    )
    candidate_agreement = finite_number(
        candidate_agreement_value,
        "candidate argmax_token_agreement",
        minimum=0.0,
        maximum=1.0,
    )
    if not isinstance(candidate_free, bool):
        raise PipelineError("candidate sequence_agreement must be a boolean")
    invalid_outputs = exact_nonnegative_integer(
        invalid_outputs_value, "candidate invalid_codec_outputs"
    )
    hard_pass = (
        candidate_agreement + 1e-12 >= baseline_agreement
        and candidate_free is True
        and invalid_outputs == 0
    )
    relative_improvement = (baseline_metric - candidate_metric) / max(
        abs(baseline_metric), 1e-12
    )
    return {
        "passed": bool(hard_pass and active_bytes_matched and relative_improvement >= 0.05),
        "hard_safety_passed": bool(hard_pass),
        "primary_metric": hypothesis.primary_metric,
        "baseline_metric": baseline_metric,
        "candidate_metric": candidate_metric,
        "relative_improvement": relative_improvement,
        "baseline_active_bytes": baseline_active_bytes,
        "candidate_active_bytes": candidate_active_bytes,
        "active_bytes_matched": active_bytes_matched,
        "baseline_sha256": sha256_file(baseline_path),
        "candidate_sha256": sha256_file(candidate_path),
    }


def write_public_orchestrator_result(path: Path, result: dict[str, Any]) -> None:
    """Write candidate evidence without claiming an outer merge already happened."""

    public_keys = {
        "hypothesis",
        "status",
        "branch",
        "agent_changed_files",
        "changed_files",
        "promotion_requested",
        "promotion_eligible",
        "promotion_blocked",
        "protected_test_changes",
    }
    public_result = {
        key: value
        for key, value in result.items()
        if key in public_keys
    }
    private_role_evidence = result.get("role_evidence", {})
    public_result["role_evidence"] = {
        role: {
            "expected_model": item.get("expected_model"),
            "completed_count": len(item.get("completed_session_ids", [])),
            "incomplete_count": len(item.get("incomplete_session_ids", [])),
            "observed_models": item.get("observed_models", []),
            "observed_providers": item.get("observed_providers", []),
            "model_routed": item.get("model_routed", False),
            "provider_routed": item.get("provider_routed", False),
            "passed": item.get("passed", False),
        }
        for role, item in sorted(private_role_evidence.items())
        if isinstance(role, str) and isinstance(item, dict)
    }
    public_result["gates"] = [
        {
            "name": report.get("name"),
            "requested_command": report.get("requested_command"),
            "returncode": report.get("returncode"),
        }
        for report in result.get("gates", [])
    ]
    holdout_keys = {
        "passed",
        "reason",
        "seed",
        "baseline_objective",
        "candidate_objective",
        "relative_change",
        "aggregate_relative_change",
        "max_case_relative_change",
        "effect_detected",
        "changed_cases",
        "required_changed_cases",
        "rejection_reasons",
        "policy",
        "candidate_artifact",
        "candidate_sha256",
        "baseline_artifact",
        "baseline_sha256",
        "baseline_safety_passed",
        "candidate_safety_passed",
        "baseline_returncode",
        "candidate_returncode",
        "tolerance",
    }
    public_result["holdout"] = {
        key: value
        for key, value in result.get("holdout", {}).items()
        if key in holdout_keys
    }
    comparison_keys = {
        "passed",
        "reason",
        "hard_safety_passed",
        "primary_metric",
        "baseline_metric",
        "candidate_metric",
        "relative_improvement",
        "baseline_active_bytes",
        "candidate_active_bytes",
        "active_bytes_matched",
        "baseline_sha256",
        "candidate_sha256",
        "holdout_blocked",
    }
    public_result["comparison"] = {
        key: value
        for key, value in result.get("comparison", {}).items()
        if key in comparison_keys
    }
    public_result["promotion_outcome_scope"] = (
        "Actual commit/merge outcome is recorded only in the parent run report and ledger."
    )
    path.write_text(json.dumps(public_result, indent=2), encoding="utf-8")


def validate_holdout_artifact(
    payload: Any,
    policy: HoldoutPolicy,
    *,
    seed: int,
    label: str,
) -> dict[str, Any]:
    """Validate trusted evaluator output before comparing paired cases."""

    if not isinstance(payload, dict):
        raise PipelineError(f"{label} holdout artifact must be an object")
    schema_version = exact_nonnegative_integer(
        payload.get("schema_version"), f"{label} holdout schema_version"
    )
    if schema_version != 2:
        raise PipelineError(f"{label} holdout schema_version must be 2")
    artifact_seed = exact_nonnegative_integer(payload.get("seed"), f"{label} holdout seed")
    if artifact_seed != seed:
        raise PipelineError(
            f"{label} holdout seed mismatch: expected {seed}, observed {artifact_seed}"
        )
    artifact_cases = exact_nonnegative_integer(
        payload.get("cases"), f"{label} holdout cases"
    )
    if artifact_cases != policy.cases:
        raise PipelineError(
            f"{label} holdout cases mismatch: expected {policy.cases}, "
            f"observed {artifact_cases}"
        )
    safety = payload.get("passed")
    if not isinstance(safety, bool):
        raise PipelineError(f"{label} holdout passed must be a boolean")
    objective = finite_number(
        payload.get("objective"), f"{label} holdout objective", minimum=0.0
    )
    details = payload.get("details")
    if not isinstance(details, list) or len(details) != policy.cases:
        raise PipelineError(
            f"{label} holdout details must contain exactly {policy.cases} cases"
        )

    normalized_details: list[dict[str, int | float]] = []
    seen_cases: set[int] = set()
    for position, item in enumerate(details):
        if not isinstance(item, dict):
            raise PipelineError(f"{label} holdout details[{position}] must be an object")
        case_id = exact_nonnegative_integer(
            item.get("case"), f"{label} holdout details[{position}].case"
        )
        if case_id in seen_cases:
            raise PipelineError(f"{label} holdout contains duplicate case {case_id}")
        seen_cases.add(case_id)
        case_seed = exact_nonnegative_integer(
            item.get("seed"), f"{label} holdout details[{position}].seed"
        )
        expected_case_seed = (seed + 104729 * case_id) % (2**31 - 1)
        if case_seed != expected_case_seed:
            raise PipelineError(
                f"{label} holdout case {case_id} seed mismatch: "
                f"expected {expected_case_seed}, observed {case_seed}"
            )
        fabric_nmse = finite_number(
            item.get("fabric_nmse"),
            f"{label} holdout details[{position}].fabric_nmse",
            minimum=0.0,
        )
        active_ratio = finite_number(
            item.get("active_ratio"),
            f"{label} holdout details[{position}].active_ratio",
            minimum=0.0,
        )
        case_objective = finite_number(
            item.get("objective"),
            f"{label} holdout details[{position}].objective",
            minimum=0.0,
        )
        recomputed = fabric_nmse + 0.04 * active_ratio
        if not math.isclose(case_objective, recomputed, rel_tol=1e-10, abs_tol=1e-12):
            raise PipelineError(
                f"{label} holdout case {case_id} objective does not match its metrics"
            )
        normalized_details.append(
            {
                "case": case_id,
                "seed": case_seed,
                "objective": case_objective,
            }
        )
    expected_cases = set(range(policy.cases))
    if seen_cases != expected_cases:
        raise PipelineError(
            f"{label} holdout case ids mismatch: expected {sorted(expected_cases)}, "
            f"observed {sorted(seen_cases)}"
        )
    normalized_details.sort(key=lambda item: int(item["case"]))
    recomputed_objective = sum(
        float(item["objective"]) for item in normalized_details
    ) / policy.cases
    if not math.isclose(objective, recomputed_objective, rel_tol=1e-10, abs_tol=1e-12):
        raise PipelineError(f"{label} aggregate holdout objective does not match details")
    return {
        "objective": objective,
        "passed": safety,
        "details": normalized_details,
    }


def compare_paired_holdout(
    baseline_payload: Any,
    candidate_payload: Any,
    policy: HoldoutPolicy,
    *,
    seed: int,
) -> dict[str, Any]:
    """Apply the preregistered effect and non-regression policy case by case."""

    baseline = validate_holdout_artifact(
        baseline_payload, policy, seed=seed, label="baseline"
    )
    candidate = validate_holdout_artifact(
        candidate_payload, policy, seed=seed, label="candidate"
    )
    baseline_details = baseline["details"]
    candidate_details = candidate["details"]
    baseline_pairing = [
        (item["case"], item["seed"]) for item in baseline_details
    ]
    candidate_pairing = [
        (item["case"], item["seed"]) for item in candidate_details
    ]
    if baseline_pairing != candidate_pairing:
        raise PipelineError("candidate holdout case/seed pairing differs from baseline")

    baseline_objective = float(baseline["objective"])
    candidate_objective = float(candidate["objective"])
    aggregate_relative_change = (
        candidate_objective - baseline_objective
    ) / max(abs(baseline_objective), 1e-12)
    case_relative_changes: list[float] = []
    changed_cases = 0
    for baseline_case, candidate_case in zip(baseline_details, candidate_details):
        baseline_case_objective = float(baseline_case["objective"])
        candidate_case_objective = float(candidate_case["objective"])
        difference = candidate_case_objective - baseline_case_objective
        relative_change = difference / max(abs(baseline_case_objective), 1e-12)
        case_relative_changes.append(relative_change)
        effect_floor = max(
            policy.min_effect_absolute,
            policy.min_effect_relative * abs(baseline_case_objective),
        )
        if abs(difference) >= effect_floor:
            changed_cases += 1

    max_case_relative_change = max(case_relative_changes)
    effect_detected = changed_cases >= policy.min_effect_cases
    rejection_reasons: list[str] = []
    if baseline["passed"] is not True:
        rejection_reasons.append("baseline_safety")
    if candidate["passed"] is not True:
        rejection_reasons.append("candidate_safety")
    if aggregate_relative_change > policy.max_aggregate_regression:
        rejection_reasons.append("aggregate_regression")
    if max_case_relative_change > policy.max_case_regression:
        rejection_reasons.append("case_regression")
    if not effect_detected:
        rejection_reasons.append("insensitive")
    return {
        "passed": not rejection_reasons,
        "baseline_objective": baseline_objective,
        "candidate_objective": candidate_objective,
        "relative_change": aggregate_relative_change,
        "aggregate_relative_change": aggregate_relative_change,
        "max_case_relative_change": max_case_relative_change,
        "case_relative_changes": case_relative_changes,
        "baseline_safety_passed": baseline["passed"],
        "candidate_safety_passed": candidate["passed"],
        "effect_detected": effect_detected,
        "changed_cases": changed_cases,
        "required_changed_cases": policy.min_effect_cases,
        "rejection_reasons": rejection_reasons,
        "policy": policy.to_json(),
    }


def run_external_holdout(
    root: Path,
    worktree: Path,
    hypothesis: Hypothesis,
    ledger: EventLedger,
    root_snapshot: RootSnapshot | None = None,
    contract_path: Path | None = None,
    contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate root and candidate code on the same post-hoc random cases."""

    seed = secrets.randbits(31)
    policy = hypothesis.holdout_policy
    evaluator = root / "scripts/autonomy/holdout_evaluator.py"
    evidence_dir = worktree / "results/experiments" / hypothesis.identifier
    evidence_dir.mkdir(parents=True, exist_ok=True)
    baseline_output = evidence_dir / "holdout-baseline.json"
    candidate_output = evidence_dir / "holdout-candidate.json"
    baseline_output.unlink(missing_ok=True)
    candidate_output.unlink(missing_ok=True)
    runtime_tmp = worktree / "autonomy/state/tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": "",
        "TEMP": str(runtime_tmp),
        "TMP": str(runtime_tmp),
    }
    with root_integrity_guard(root, root_snapshot):
        baseline_run = run(
            [
                sys.executable,
                str(evaluator),
                "--source",
                str(root / "src"),
                "--seed",
                str(seed),
                "--cases",
                str(policy.cases),
                "--output",
                str(baseline_output),
            ],
            cwd=root,
            timeout=900,
            environment=environment,
            inherit_environment=False,
            check=False,
        )
    if contract_path is not None and contract_sha256 is not None:
        assert_file_sha256(contract_path, contract_sha256, "experiment contract")
    with root_integrity_guard(root, root_snapshot):
        candidate_run = run(
            [
                sys.executable,
                str(evaluator),
                "--source",
                str(worktree / "src"),
                "--seed",
                str(seed),
                "--cases",
                str(policy.cases),
                "--output",
                str(candidate_output),
            ],
            cwd=worktree,
            timeout=900,
            environment=environment,
            inherit_environment=False,
            check=False,
        )
    if contract_path is not None and contract_sha256 is not None:
        assert_file_sha256(contract_path, contract_sha256, "experiment contract")
    if (
        baseline_run.returncode != 0
        or candidate_run.returncode != 0
        or not baseline_output.exists()
        or not candidate_output.exists()
    ):
        report = {
            "passed": False,
            "reason": "holdout evaluator failed or did not produce fresh artifacts",
            "baseline_returncode": baseline_run.returncode,
            "candidate_returncode": candidate_run.returncode,
            "baseline_stderr_tail": baseline_run.stderr[-1000:],
            "candidate_stderr_tail": candidate_run.stderr[-1000:],
        }
        ledger.append("holdout.failed", report)
        return report
    baseline = json.loads(baseline_output.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_output.read_text(encoding="utf-8"))
    paired = compare_paired_holdout(baseline, candidate, policy, seed=seed)
    report = {
        **paired,
        "seed": seed,
        "candidate_artifact": str(candidate_output.relative_to(worktree)),
        "candidate_sha256": sha256_file(candidate_output),
        "baseline_artifact": str(baseline_output.relative_to(worktree)),
        "baseline_sha256": sha256_file(baseline_output),
        "baseline_returncode": baseline_run.returncode,
        "candidate_returncode": candidate_run.returncode,
        "tolerance": (
            "paired randomized holdout requires measurable effect, no case above the "
            "preregistered regression bound, and aggregate non-regression"
        ),
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


def dry_run_plan(
    root: Path,
    hypotheses: list[Hypothesis],
    runtime: CodexRuntime,
) -> dict[str, Any]:
    baseline = root / "results/fabric_benchmark.json"
    digest = sha256_file(baseline) if baseline.exists() else "MISSING"
    plans = []
    for hypothesis in hypotheses:
        result_path = Path(
            f"autonomy/state/runs/{hypothesis.identifier}/agent-result.json"
        )
        plans.append(
            {
                "hypothesis": hypothesis.identifier,
                "branch": f"autonomy/{slug(hypothesis.identifier)}-TIMESTAMP",
                "codex_command": build_codex_exec_command(
                    runtime,
                    schema_path=root / "autonomy/schemas/agent_result.schema.json",
                    result_path=result_path,
                    project_root=root,
                ),
                "automated_evaluator": hypothesis.evaluator,
                "executable": hypothesis.evaluator in SUPPORTED_EVALUATORS,
                "prompt_sha256": hashlib.sha256(
                    prompt_for(hypothesis, digest).encode("utf-8")
                ).hexdigest(),
                "allowed_files": list(hypothesis.allowed_files),
                "gates": json.loads((root / "autonomy/gates.json").read_text())["commands"],
                "external_holdout": {
                    "evaluator": "scripts/autonomy/holdout_evaluator.py",
                    "seed_timing": "generated after the Codex run",
                    "policy": hypothesis.holdout_policy.to_json(),
                },
            }
        )
    return {
        "mode": "dry-run",
        "timestamp": now(),
        "codex": {"path": str(runtime.path), "version": runtime.version},
        "baseline_sha256": digest,
        "plans": plans,
    }


def root_integrity_failure(
    error: RootIntegrityError,
    *,
    hypothesis: Hypothesis,
    branch: str,
    worktree: Path,
    ledger: EventLedger,
) -> dict[str, Any]:
    result = {
        "hypothesis": hypothesis.identifier,
        "status": "root_integrity_failed",
        "branch": branch,
        "worktree": str(worktree),
        "error": str(error),
        "promoted": False,
    }
    ledger.append("root_integrity.failed", result)
    return result


def execute_one(
    root: Path,
    hypothesis: Hypothesis,
    *,
    runtime: CodexRuntime,
    worktree_root: Path,
    auto_promote: bool,
    ledger: EventLedger,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    branch = f"autonomy/{slug(hypothesis.identifier)}-{timestamp}"
    worktree = worktree_root / f"{slug(hypothesis.identifier)}-{timestamp}"
    host_codex_environment = codex_cli_environment()
    session_root = codex_session_root(host_codex_environment)
    previous_session_paths = snapshot_codex_session_files(session_root)
    checkpoint_file = trusted_checkpoint_file(root)
    baseline = root / "results/fabric_benchmark.json"
    baseline_digest = sha256_file(baseline)
    root_snapshot = capture_root_snapshot(root)
    run(
        git_command(
            ["worktree", "add", "-b", branch, str(worktree), "HEAD"], root
        ),
        cwd=root,
    )
    codex_environment = candidate_codex_environment(
        root, worktree, checkpoint_file=checkpoint_file
    )
    start_head = git_output(["rev-parse", "HEAD"], worktree)
    experiment_dir = worktree / "results" / "experiments" / hypothesis.identifier
    experiment_dir.mkdir(parents=True, exist_ok=True)
    contract_path = experiment_dir / "experiment.json"
    preregistered_contract = experiment_contract_for(hypothesis, baseline_digest)
    contract_path.write_text(
        json.dumps(preregistered_contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    contract_sha256 = sha256_file(contract_path)
    private_run_dir = worktree / "autonomy" / "state" / "runs" / hypothesis.identifier
    private_run_dir.mkdir(parents=True, exist_ok=True)
    event_path = private_run_dir / "codex-events.jsonl"
    result_path = private_run_dir / "agent-result.json"
    prompt = prompt_for(hypothesis, baseline_digest)
    command = build_codex_exec_command(
        runtime,
        schema_path=worktree / "autonomy/schemas/agent_result.schema.json",
        result_path=result_path.relative_to(worktree),
        project_root=worktree,
        shell_environment=codex_environment,
    )
    ledger.append(
        "experiment_contract.preregistered",
        {
            "hypothesis": hypothesis.identifier,
            "path": str(contract_path.relative_to(worktree)),
            "sha256": contract_sha256,
        },
    )
    ledger.append(
        "codex.started",
        {"hypothesis": hypothesis.identifier, "branch": branch, "worktree": str(worktree)},
    )
    try:
        with root_integrity_guard(root, root_snapshot):
            completed = run(
                command,
                cwd=worktree,
                timeout=7200,
                environment=codex_environment,
                inherit_environment=False,
                input_text=prompt,
                check=False,
            )
    except RootIntegrityError as error:
        return root_integrity_failure(
            error,
            hypothesis=hypothesis,
            branch=branch,
            worktree=worktree,
            ledger=ledger,
        )
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
    try:
        assert_root_unchanged(root, root_snapshot)
        assert_file_sha256(contract_path, contract_sha256, "experiment contract")
    except CandidateIntegrityError as error:
        ledger.append(
            "experiment_contract.failed",
            {"hypothesis": hypothesis.identifier, "error": str(error)},
        )
        return {
            "hypothesis": hypothesis.identifier,
            "status": "experiment_contract_failed",
            "branch": branch,
            "worktree": str(worktree),
            "error": str(error),
        }
    except RootIntegrityError as error:
        return root_integrity_failure(
            error,
            hypothesis=hypothesis,
            branch=branch,
            worktree=worktree,
            ledger=ledger,
        )
    if completed.returncode:
        return {
            "hypothesis": hypothesis.identifier,
            "status": "codex_failed",
            "branch": branch,
            "worktree": str(worktree),
            "returncode": completed.returncode,
        }
    current_head = git_output(["rev-parse", "HEAD"], worktree)
    if current_head != start_head:
        ledger.append(
            "history.failed",
            {
                "hypothesis": hypothesis.identifier,
                "start_head": start_head,
                "current_head": current_head,
            },
        )
        return {
            "hypothesis": hypothesis.identifier,
            "status": "history_mutation_failed",
            "branch": branch,
            "worktree": str(worktree),
            "start_head": start_head,
            "current_head": current_head,
        }
    try:
        agent_result = validate_agent_result(result_path)
    except (PipelineError, json.JSONDecodeError) as error:
        ledger.append(
            "agent_result.failed",
            {"hypothesis": hypothesis.identifier, "error": str(error)},
        )
        return {
            "hypothesis": hypothesis.identifier,
            "status": "agent_result_invalid",
            "branch": branch,
            "worktree": str(worktree),
            "error": str(error),
        }
    ledger.append(
        "agent_result.validated",
        {
            "hypothesis": hypothesis.identifier,
            "status": agent_result["status"],
            "sha256": sha256_file(result_path),
        },
    )
    if agent_result["status"] != "completed":
        return {
            "hypothesis": hypothesis.identifier,
            "status": f"agent_{agent_result['status']}",
            "branch": branch,
            "worktree": str(worktree),
            "summary": agent_result["summary"],
            "negative_results": agent_result["negative_results"],
        }
    configured_models = project_agent_models(worktree)
    expected_models = {
        role: configured_models[role]
        for role in REQUIRED_RESEARCH_ROLES
        if role in configured_models
    }
    try:
        parent_thread_id = codex_thread_id(completed.stdout)
        role_evidence = subagent_role_evidence(
            session_root,
            previous_session_paths,
            parent_thread_id,
            expected_models,
        )
    except PipelineError as error:
        ledger.append(
            "agent_roles.failed",
            {"hypothesis": hypothesis.identifier, "error": str(error)},
        )
        return {
            "hypothesis": hypothesis.identifier,
            "status": "agent_routing_failed",
            "branch": branch,
            "worktree": str(worktree),
            "error": str(error),
            "missing_roles": sorted(REQUIRED_RESEARCH_ROLES),
        }
    missing_roles = sorted(
        role for role in REQUIRED_RESEARCH_ROLES if not role_evidence.get(role, {}).get("passed")
    )
    ledger.append(
        "agent_roles.observed",
        {
            "hypothesis": hypothesis.identifier,
            "evidence": role_evidence,
            "missing": missing_roles,
        },
    )
    if missing_roles:
        return {
            "hypothesis": hypothesis.identifier,
            "status": "agent_routing_failed",
            "branch": branch,
            "worktree": str(worktree),
            "role_evidence": role_evidence,
            "missing_roles": missing_roles,
        }
    try:
        changes = validate_candidate_state(worktree, hypothesis, start_head)
    except PipelineError as error:
        ledger.append(
            "scope.failed",
            {"hypothesis": hypothesis.identifier, "error": str(error)},
        )
        return {
            "hypothesis": hypothesis.identifier,
            "status": "candidate_validation_failed",
            "error": str(error),
            "branch": branch,
            "worktree": str(worktree),
        }
    try:
        validate_reported_changes(
            agent_result,
            changes,
            ignored_paths=(str(contract_path.relative_to(worktree)),),
        )
    except PipelineError as error:
        ledger.append(
            "agent_result.failed",
            {"hypothesis": hypothesis.identifier, "error": str(error)},
        )
        return {
            "hypothesis": hypothesis.identifier,
            "status": "change_report_mismatch",
            "branch": branch,
            "worktree": str(worktree),
            "error": str(error),
            "changed_files": changes,
        }
    try:
        experiment_contract = validate_experiment_contract(
            worktree, hypothesis, baseline_digest
        )
    except (PipelineError, json.JSONDecodeError) as error:
        ledger.append(
            "experiment_contract.failed",
            {"hypothesis": hypothesis.identifier, "error": str(error)},
        )
        return {
            "hypothesis": hypothesis.identifier,
            "status": "experiment_contract_failed",
            "branch": branch,
            "worktree": str(worktree),
            "error": str(error),
        }
    ledger.append(
        "experiment_contract.validated",
        {
            "hypothesis": hypothesis.identifier,
            "sha256": hashlib.sha256(canonical(experiment_contract).encode("utf-8")).hexdigest(),
        },
    )
    agent_changes = [
        path
        for path in changes
        if path != contract_path.relative_to(worktree).as_posix()
    ]
    immutable_candidate_paths = sorted(
        (
            set(agent_changes)
            | {contract_path.relative_to(worktree).as_posix()}
        )
        - orchestrator_mutable_paths(hypothesis)
    )
    tested_candidate_blobs = candidate_blob_map(
        worktree, immutable_candidate_paths
    )
    try:
        gate_reports = run_gates(
            worktree,
            ledger,
            root,
            root_snapshot,
            contract_path,
            contract_sha256,
            checkpoint_file=checkpoint_file,
        )
        gates_passed = all(report["returncode"] == 0 for report in gate_reports)
        if gates_passed:
            trusted_report = run_trusted_root_tests(
                root,
                worktree,
                ledger,
                root_snapshot,
                contract_path,
                contract_sha256,
            )
            gate_reports.append(trusted_report)
            gates_passed = trusted_report["returncode"] == 0
        if gates_passed:
            try:
                holdout = run_external_holdout(
                    root,
                    worktree,
                    hypothesis,
                    ledger,
                    root_snapshot,
                    contract_path,
                    contract_sha256,
                )
            except CandidateIntegrityError:
                raise
            except (PipelineError, json.JSONDecodeError) as error:
                holdout = {
                    "passed": False,
                    "reason": f"invalid holdout evidence: {error}",
                }
                ledger.append("holdout.failed", holdout)
            assert_root_unchanged(root, root_snapshot)
            assert_file_sha256(contract_path, contract_sha256, "experiment contract")
            try:
                comparison = compare_candidate(root, worktree, hypothesis)
            except (PipelineError, json.JSONDecodeError) as error:
                comparison = {
                    "passed": False,
                    "reason": f"invalid benchmark evidence: {error}",
                }
                ledger.append("comparison.failed", comparison)
        else:
            holdout = {"passed": False, "reason": "deterministic gates failed"}
            comparison = {"passed": False, "reason": "one or more gates failed"}
        assert_root_unchanged(root, root_snapshot)
        assert_file_sha256(contract_path, contract_sha256, "experiment contract")
        observed_candidate_blobs = candidate_blob_map(
            worktree, immutable_candidate_paths
        )
        if observed_candidate_blobs != tested_candidate_blobs:
            raise CandidateIntegrityError(
                "candidate code/evidence changed while host gates were running"
            )
    except RootIntegrityError as error:
        return root_integrity_failure(
            error,
            hypothesis=hypothesis,
            branch=branch,
            worktree=worktree,
            ledger=ledger,
        )
    except CandidateIntegrityError as error:
        result = {
            "hypothesis": hypothesis.identifier,
            "status": "candidate_integrity_failed",
            "branch": branch,
            "worktree": str(worktree),
            "error": str(error),
            "promoted": False,
        }
        ledger.append("candidate_integrity.failed", result)
        return result
    if not holdout.get("passed"):
        comparison = {**comparison, "passed": False, "holdout_blocked": True}
    result = {
        "hypothesis": hypothesis.identifier,
        "status": "accepted" if comparison.get("passed") else "rejected",
        "branch": branch,
        "worktree": str(worktree),
        "agent_changed_files": agent_changes,
        "changed_files": [],
        "role_evidence": role_evidence,
        "gates": gate_reports,
        "holdout": holdout,
        "comparison": comparison,
        "promotion_requested": auto_promote,
        "promotion_eligible": bool(comparison.get("passed")),
    }
    result["promoted"] = False
    orchestrator_result = experiment_dir / "orchestrator-result.json"
    write_public_orchestrator_result(orchestrator_result, result)
    try:
        final_changes = validate_candidate_state(worktree, hypothesis, start_head)
    except PipelineError as error:
        result["status"] = "post_gate_validation_failed"
        result["error"] = str(error)
        write_public_orchestrator_result(orchestrator_result, result)
        ledger.append("scope.failed", result)
        return result
    try:
        assert_file_sha256(contract_path, contract_sha256, "experiment contract")
        assert_candidate_path_set_unchanged(
            immutable_candidate_paths, final_changes, hypothesis
        )
        if candidate_blob_map(worktree, immutable_candidate_paths) != tested_candidate_blobs:
            raise CandidateIntegrityError(
                "candidate code/evidence changed after successful gates"
            )
    except CandidateIntegrityError as error:
        result["status"] = "candidate_integrity_failed"
        result["error"] = str(error)
        write_public_orchestrator_result(orchestrator_result, result)
        ledger.append("candidate_integrity.failed", result)
        return result
    result["changed_files"] = final_changes
    write_public_orchestrator_result(orchestrator_result, result)
    if comparison.get("passed") and auto_promote:
        protected_tests = existing_tracked_test_changes(root, final_changes)
        if protected_tests:
            result["status"] = "accepted_requires_review"
            result["promotion_blocked"] = "existing tracked tests changed"
            result["protected_test_changes"] = protected_tests
            write_public_orchestrator_result(orchestrator_result, result)
            ledger.append("promotion.blocked", result)
            return result
        final_blob_contract = candidate_blob_map(worktree, final_changes)
        stage_validated_paths(worktree, final_changes)
        staged = set(
            git_output(
                ["diff", "--cached", "--name-only", "--no-renames", "HEAD", "--"],
                worktree,
            ).splitlines()
        )
        if staged != set(final_changes):
            raise PipelineError(
                "staged candidate paths differ from validated paths: "
                f"staged={sorted(staged)}, validated={final_changes}"
            )
        staged_final_blobs = staged_blob_map(worktree, final_changes)
        if staged_final_blobs != final_blob_contract:
            raise PromotionIntegrityError(
                "staged candidate content differs from the final validated bytes"
            )
        staged_immutable_blobs = {
            path: staged_final_blobs[path]
            for path in immutable_candidate_paths
        }
        if staged_immutable_blobs != tested_candidate_blobs:
            raise PromotionIntegrityError(
                "staged agent content differs from the content that passed gates"
            )
        expected_tree = git_output(["write-tree"], worktree)
        commit_hooks_path = random_nonexistent_hooks_path(root)
        if commit_hooks_path.exists():
            raise PromotionIntegrityError("commit hooks path unexpectedly exists")
        run(
            git_command(
                [
                    "-c",
                    f"core.hooksPath={commit_hooks_path}",
                    "-c",
                    "commit.gpgSign=false",
                    "commit",
                    "-m",
                    f"experiment: {hypothesis.identifier}",
                ],
                worktree,
            ),
            cwd=worktree,
        )
        try:
            candidate_commit, candidate_tree_sha256 = capture_clean_candidate_commit(
                worktree,
                expected_parent=start_head,
                expected_tree=expected_tree,
                expected_paths=final_changes,
            )
        except PromotionIntegrityError as error:
            result["status"] = "accepted_requires_review"
            result["promotion_blocked"] = str(error)
            ledger.append("promotion.blocked", result)
            return result
        assert_root_unchanged(root, root_snapshot)
        assert_clean(root)
        merge_hooks_path = random_nonexistent_hooks_path(root)
        if merge_hooks_path.exists():
            raise PromotionIntegrityError("merge hooks path unexpectedly exists")
        run(
            git_command(
                [
                    "-c",
                    f"core.hooksPath={merge_hooks_path}",
                    "-c",
                    "merge.verifySignatures=false",
                    "merge",
                    "--ff-only",
                    candidate_commit,
                ],
                root,
            ),
            cwd=root,
        )
        try:
            assert_promotion_postcondition(
                root,
                expected_head=candidate_commit,
                expected_tree_sha256=candidate_tree_sha256,
            )
        except PromotionIntegrityError as error:
            result["status"] = "promotion_postcondition_failed"
            result["error"] = str(error)
            result["promoted"] = (
                git_output(["rev-parse", "HEAD"], root) == candidate_commit
            )
            ledger.append("promotion.postcondition_failed", result)
            return result
        result["promoted"] = True
    elif not auto_promote:
        assert_root_unchanged(root, root_snapshot)
    ledger.append("hypothesis.completed", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("preflight", "dry-run", "execute"), default="preflight"
    )
    parser.add_argument("--hypothesis", action="append", default=[])
    parser.add_argument("--max-hypotheses", type=int, default=1)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--worktree-root", type=Path, default=DEFAULT_WORKTREE_ROOT)
    parser.add_argument("--auto-promote", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    _, hypotheses = load_hypotheses(ROOT / "autonomy/hypotheses.json")
    selected = [item for item in hypotheses if item.status == "pending"]
    if args.hypothesis:
        selected = [item for item in selected if item.identifier in set(args.hypothesis)]
    selected = selected[: args.max_hypotheses]
    if not selected:
        raise PipelineError("no pending hypotheses matched")
    if args.auto_promote and args.max_hypotheses != 1:
        raise PipelineError("auto-promote requires --max-hypotheses 1")
    preflight = preflight_report(ROOT, selected, args.codex)
    if args.mode == "preflight":
        report = preflight
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    if not preflight["ready_for_dry_run"]:
        raise PipelineError("preflight blocked: " + "; ".join(preflight["blockers"]))
    codex_info = preflight["codex"]
    if codex_info is None:
        raise PipelineError("preflight did not resolve a Codex runtime")
    runtime = CodexRuntime(Path(codex_info["path"]), codex_info["version"])
    if args.mode == "dry-run":
        report = dry_run_plan(ROOT, selected, runtime)
        output = args.output or ROOT / "results/autonomy_runs/dry-run.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if not preflight["ready_for_execute"]:
        raise PipelineError("execute blocked: " + "; ".join(preflight["blockers"]))
    assert_git_repository(ROOT)
    assert_clean(ROOT)
    lock_path = ROOT / "autonomy/state/pipeline.lock"
    descriptor = acquire_lock(lock_path)
    try:
        ledger = EventLedger(ROOT / "autonomy/state/events.jsonl")
        args.worktree_root.mkdir(parents=True, exist_ok=True)
        results = [
            execute_one(
                ROOT,
                hypothesis,
                runtime=runtime,
                worktree_root=args.worktree_root,
                auto_promote=args.auto_promote,
                ledger=ledger,
            )
            for hypothesis in selected
        ]
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
    output = args.output or ROOT / "results/autonomy_runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (PipelineError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"autonomy pipeline failed: {error}", file=sys.stderr)
        raise SystemExit(2)
