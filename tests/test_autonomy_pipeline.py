from __future__ import annotations

import importlib.util
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/autonomy/run_codex_loop.py"
HOOK_PATH = ROOT / "scripts/autonomy/record_subagent_event.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("focus_autonomy_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_hook_recorder():
    spec = importlib.util.spec_from_file_location("focus_subagent_hook", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_subagent_session(
    path: Path,
    *,
    parent_thread_id: str,
    role: str,
    model: str,
    session_id: str,
    provider: str = "openai",
    completed: bool = True,
    spawn_parent_thread_id: str | None = None,
    metadata_agent_path: str | None = None,
    thread_source: str = "subagent",
    trailing_event: bool = False,
) -> None:
    agent_path = f"/root/{role}"
    records = [
        {
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "parent_thread_id": parent_thread_id,
                "thread_source": thread_source,
                "agent_path": metadata_agent_path or agent_path,
                "model_provider": provider,
                "source": {
                    "subagent": {
                        "thread_spawn": {
                            "parent_thread_id": (
                                spawn_parent_thread_id or parent_thread_id
                            ),
                            "depth": 1,
                            "agent_path": agent_path,
                        }
                    }
                },
            },
        },
        {
            "type": "turn_context",
            "payload": {
                "model": model,
                "collaboration_mode": {"settings": {"model": model}},
            },
        },
    ]
    if completed:
        records.append({"type": "event_msg", "payload": {"type": "task_complete"}})
    if trailing_event:
        records.append({"type": "event_msg", "payload": {"type": "token_count"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_codex_runtime_skips_a_broken_path_shim(monkeypatch, tmp_path: Path) -> None:
    runner = load_runner()
    broken = tmp_path / "codex.cmd"
    working = tmp_path / "codex.exe"
    broken.touch()
    working.touch()

    monkeypatch.setattr(
        runner,
        "codex_candidates",
        lambda requested: [broken, working],
    )

    def fake_run(command, **kwargs):
        if Path(command[0]) == broken:
            return subprocess.CompletedProcess(command, 1, "", "missing target")
        return subprocess.CompletedProcess(command, 0, "codex-cli 0.130.0\n", "")

    monkeypatch.setattr(runner, "run", fake_run)

    runtime = runner.resolve_codex_runtime("codex")

    assert runtime.path == working
    assert runtime.version == "codex-cli 0.130.0"


def test_windows_codex_candidates_prefer_newest_desktop_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    runner = load_runner()
    bin_dir = tmp_path / "OpenAI" / "Codex" / "bin"
    legacy = bin_dir / "codex.exe"
    current = bin_dir / "3135b80b111fd431" / "codex.exe"
    legacy.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    legacy.touch()
    current.touch()
    os.utime(legacy, (1, 1))
    os.utime(current, (2, 2))

    monkeypatch.delenv("FOCUS_CODEX", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(runner.shutil, "which", lambda _: str(tmp_path / "broken.cmd"))

    candidates = runner.codex_candidates("codex")

    assert candidates[:2] == [current, legacy]


def test_codex_exec_command_uses_current_noninteractive_syntax(tmp_path: Path) -> None:
    runner = load_runner()
    runtime = runner.CodexRuntime(tmp_path / "codex.exe", "codex-cli 0.144.2")

    command = runner.build_codex_exec_command(
        runtime,
        schema_path=tmp_path / "schema.json",
        result_path=tmp_path / "result.json",
    )

    assert command[:4] == [str(runtime.path), "--ask-for-approval", "never", "exec"]
    assert "--strict-config" in command
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--ask-for-approval" not in command[4:]
    assert any(
        item.startswith("agents.research_scout.config_file=") for item in command
    )
    assert any(item.startswith("agents.default.config_file=") for item in command)
    assert "sandbox_workspace_write.network_access=false" in command
    if os.name == "nt":
        assert 'windows.sandbox="elevated"' in command
    assert "allow_login_shell=false" in command
    assert 'shell_environment_policy.inherit="core"' in command
    enabled = [command[index + 1] for index, item in enumerate(command) if item == "--enable"]
    assert enabled == ["multi_agent"]
    assert "--dangerously-bypass-hook-trust" not in command
    assert not any(item.startswith("hooks.") for item in command)
    assert command[-1] == "-"


def test_native_sandbox_overrides_are_explicit_on_windows() -> None:
    runner = load_runner()

    assert runner.native_sandbox_overrides("nt") == [
        "-c",
        'windows.sandbox="elevated"',
    ]
    assert runner.native_sandbox_overrides("posix") == []


def test_windows_workspace_write_probe_requires_a_real_sentinel(
    monkeypatch, tmp_path: Path
) -> None:
    runner = load_runner()
    runtime = runner.CodexRuntime(tmp_path / "codex.exe", "codex-cli 0.144.2")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner, "run", fake_run)

    ready, detail = runner.probe_codex_workspace_write(
        runtime,
        tmp_path,
        environment={"PATH": ""},
        platform_name="nt",
    )

    assert ready is False
    assert "sentinel" in detail
    assert calls and 'windows.sandbox="elevated"' in calls[0]


def test_windows_workspace_write_probe_accepts_a_sandbox_created_sentinel(
    monkeypatch, tmp_path: Path
) -> None:
    runner = load_runner()
    runtime = runner.CodexRuntime(tmp_path / "codex.exe", "codex-cli 0.144.2")

    def fake_run(command, **kwargs):
        working_root = Path(command[command.index("-C") + 1])
        (working_root / "workspace-write.ok").write_text("ok", encoding="ascii")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner, "run", fake_run)

    ready, detail = runner.probe_codex_workspace_write(
        runtime,
        tmp_path,
        environment={"PATH": ""},
        platform_name="nt",
    )

    assert ready is True
    assert detail == "workspace-write sentinel created"
    state_root = tmp_path / "autonomy" / "state"
    assert list(state_root.glob("preflight-sandbox-*")) == []


def test_resolved_repository_child_rejects_an_external_target(tmp_path: Path) -> None:
    runner = load_runner()
    repository = tmp_path / "repository"
    external = tmp_path / "external"
    repository.mkdir()
    external.mkdir()

    with pytest.raises(runner.PipelineError, match="outside repository"):
        runner.resolved_repository_child(
            repository,
            external,
            label="sandbox state directory",
        )


def test_gate_commands_are_pinned_to_current_python(monkeypatch, tmp_path: Path) -> None:
    runner = load_runner()
    (tmp_path / "autonomy").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "autonomy" / "gates.json").write_text(
        json.dumps(
            {
                "commands": [
                    {"name": "compile", "command": ["python", "-m", "compileall", "src"]},
                    {"name": "tests", "command": ["pytest", "-q"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner, "run", fake_run)
    ledger = runner.EventLedger(tmp_path / "state" / "events.jsonl")

    reports = runner.run_gates(tmp_path, ledger, tmp_path)

    assert len(reports) == 2
    assert calls[0][0][0] == sys.executable
    assert calls[1][0][:3] == [sys.executable, "-m", "pytest"]
    assert "--basetemp" in calls[1][0]
    assert calls[0][1]["environment"]["PYTHONPATH"] == str(tmp_path / "src")
    assert calls[1][1]["environment"]["PYTHONPATH"] == str(tmp_path / "src")
    assert calls[0][1]["environment"]["MPLBACKEND"] == "Agg"
    assert Path(calls[0][1]["environment"]["MPLCONFIGDIR"]) == (
        tmp_path / "autonomy/state/matplotlib"
    )


def test_gate_rejects_any_preregistered_contract_byte_change(
    monkeypatch, tmp_path: Path
) -> None:
    runner = load_runner()
    (tmp_path / "autonomy").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "autonomy/gates.json").write_text(
        json.dumps(
            {
                "commands": [
                    {"name": "compile", "command": ["python", "-m", "compileall", "src"]}
                ]
            }
        ),
        encoding="utf-8",
    )
    contract = tmp_path / "results/experiments/H001/experiment.json"
    contract.parent.mkdir(parents=True)
    contract.write_text('{"locked":true}\n', encoding="utf-8")
    contract_sha256 = runner.sha256_file(contract)

    def mutating_run(command, **kwargs):
        contract.write_text('{"locked": true}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner, "run", mutating_run)
    ledger = runner.EventLedger(tmp_path / "state/events.jsonl")

    with pytest.raises(runner.CandidateIntegrityError, match="experiment contract changed"):
        runner.run_gates(
            tmp_path,
            ledger,
            tmp_path,
            contract_path=contract,
            contract_sha256=contract_sha256,
        )


def test_allowed_file_paths_honor_file_and_directory_boundaries() -> None:
    runner = load_runner()
    hypothesis = runner.Hypothesis(
        identifier="H001",
        title="title",
        rationale="rationale",
        controls=(),
        primary_metric="metric",
        disconfirming_condition="condition",
        allowed_files=("src/focus_fabric/codecs.py", "tests/"),
        status="pending",
    )

    assert runner.path_allowed("src/focus_fabric/codecs.py", hypothesis)
    assert not runner.path_allowed("src/focus_fabric/codecs.py.evil", hypothesis)
    assert runner.path_allowed("tests/test_codec.py", hypothesis)
    assert runner.path_allowed("results/candidate_benchmark.csv", hypothesis)
    assert runner.path_allowed("results/candidate_benchmark.png", hypothesis)
    assert not runner.path_allowed("tests-escape/test_codec.py", hypothesis)
    assert not runner.path_allowed("../src/focus_fabric/codecs.py", hypothesis)


def test_agent_result_requires_complete_schema_and_completed_status(tmp_path: Path) -> None:
    runner = load_runner()
    result_path = tmp_path / "agent-result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "summary": "done",
                "changed_files": ["tests/test_codec.py"],
                "evidence": ["pytest passed"],
                "risks": [],
                "negative_results": [],
                "next_hypotheses": [],
            }
        ),
        encoding="utf-8",
    )

    payload = runner.validate_agent_result(result_path)

    assert payload["status"] == "completed"

    payload.pop("negative_results")
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.PipelineError, match="negative_results"):
        runner.validate_agent_result(result_path)

    payload["negative_results"] = ["falsifier triggered"]
    payload["status"] = "blocked"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.validate_agent_result(result_path)["status"] == "blocked"


def test_h001_evaluator_uses_declared_metric_and_matched_active_bytes(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    root = tmp_path / "root"
    worktree = tmp_path / "candidate"
    (root / "results").mkdir(parents=True)
    (worktree / "results").mkdir(parents=True)
    baseline = json.loads((ROOT / "results/fabric_benchmark.json").read_text(encoding="utf-8"))
    candidate = copy.deepcopy(baseline)
    candidate["synthetic"]["splits"]["in_distribution"]["fabric_approx"][
        "output_nmse"
    ] *= 0.94
    (root / "results/fabric_benchmark.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    (worktree / "results/candidate_benchmark.json").write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    hypothesis = runner.Hypothesis(
        identifier="H001-forward-influence-routing",
        title="title",
        rationale="rationale",
        controls=(),
        primary_metric="synthetic.in_distribution.output_nmse",
        disconfirming_condition="condition",
        allowed_files=("results/",),
        status="pending",
        evaluator="fabric_benchmark_v1",
    )

    report = runner.compare_candidate(root, worktree, hypothesis)

    assert report["passed"] is True
    assert report["primary_metric"] == hypothesis.primary_metric
    assert report["relative_improvement"] == pytest.approx(0.06)
    assert report["active_bytes_matched"] is True

    candidate["synthetic"]["memory"]["fabric_active_bytes"] += 1
    (worktree / "results/candidate_benchmark.json").write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    report = runner.compare_candidate(root, worktree, hypothesis)
    assert report["passed"] is False
    assert report["active_bytes_matched"] is False


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("synthetic", "splits", "in_distribution", "fabric_approx", "output_nmse"), float("-inf")),
        (("synthetic", "memory", "fabric_active_bytes"), 8584.5),
        (("end_to_end", "teacher_forced", "argmax_token_agreement"), float("nan")),
        (("end_to_end", "free_running", "sequence_agreement"), "false"),
        (("repeated_compaction", "final", "invalid_codec_outputs"), False),
    ],
)
def test_h001_evaluator_rejects_malformed_or_nonfinite_evidence(
    tmp_path: Path, path: tuple[str, ...], bad_value: object
) -> None:
    runner = load_runner()
    root = tmp_path / "root"
    worktree = tmp_path / "candidate"
    (root / "results").mkdir(parents=True)
    (worktree / "results").mkdir(parents=True)
    baseline = json.loads((ROOT / "results/fabric_benchmark.json").read_text(encoding="utf-8"))
    candidate = copy.deepcopy(baseline)
    candidate["synthetic"]["splits"]["in_distribution"]["fabric_approx"][
        "output_nmse"
    ] *= 0.94
    target = candidate
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = bad_value
    (root / "results/fabric_benchmark.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    (worktree / "results/candidate_benchmark.json").write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    hypothesis = runner.Hypothesis(
        identifier="H001-forward-influence-routing",
        title="title",
        rationale="rationale",
        controls=(),
        primary_metric="synthetic.in_distribution.output_nmse",
        disconfirming_condition="condition",
        allowed_files=("results/",),
        status="pending",
        evaluator="fabric_benchmark_v1",
    )

    with pytest.raises(runner.PipelineError):
        runner.compare_candidate(root, worktree, hypothesis)


def test_trusted_root_tests_run_against_candidate_source(monkeypatch, tmp_path: Path) -> None:
    runner = load_runner()
    root = tmp_path / "root"
    worktree = tmp_path / "candidate"
    (root / "tests").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (worktree / "src").mkdir(parents=True)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "trusted tests passed", "")

    monkeypatch.setattr(runner, "run", fake_run)
    ledger = runner.EventLedger(tmp_path / "state" / "events.jsonl")

    report = runner.run_trusted_root_tests(root, worktree, ledger)

    assert report["returncode"] == 0
    assert captured["command"][:3] == [sys.executable, "-m", "pytest"]
    assert str(root / "tests") in captured["command"]
    assert "--basetemp" in captured["command"]
    assert captured["kwargs"]["cwd"] == worktree
    assert captured["kwargs"]["environment"]["PYTHONPATH"] == str(worktree / "src")
    assert captured["kwargs"]["inherit_environment"] is False


def test_holdout_rejects_failed_runs_and_removes_stale_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    runner = load_runner()
    root = tmp_path / "root"
    worktree = tmp_path / "candidate"
    evidence_dir = worktree / "results/experiments/H001"
    evidence_dir.mkdir(parents=True)
    for name in ("holdout-baseline.json", "holdout-candidate.json"):
        (evidence_dir / name).write_text(
            json.dumps({"objective": 0.0, "passed": True}), encoding="utf-8"
        )
    hypothesis = runner.Hypothesis(
        identifier="H001",
        title="title",
        rationale="rationale",
        controls=(),
        primary_metric="metric",
        disconfirming_condition="condition",
        allowed_files=(),
        status="pending",
        evaluator="fabric_benchmark_v1",
    )
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "", "trusted evaluator failed"
        ),
    )
    ledger = runner.EventLedger(root / "autonomy/state/events.jsonl")

    report = runner.run_external_holdout(root, worktree, hypothesis, ledger)

    assert report["passed"] is False
    assert report["baseline_returncode"] == 1
    assert report["candidate_returncode"] == 1
    assert not (evidence_dir / "holdout-baseline.json").exists()
    assert not (evidence_dir / "holdout-candidate.json").exists()


def test_public_orchestrator_result_omits_host_paths_and_actual_merge_state(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    output = tmp_path / "orchestrator-result.json"
    private_path = str(tmp_path / "private-worktree")
    runner.write_public_orchestrator_result(
        output,
        {
            "hypothesis": "H001",
            "status": "accepted",
            "worktree": private_path,
            "promoted": True,
            "promotion_requested": True,
            "promotion_eligible": True,
            "role_evidence": {
                "research_scout": {
                    "expected_model": "gpt-5.6-luna",
                    "completed_session_ids": ["private-session-id"],
                    "incomplete_session_ids": ["private-incomplete-id"],
                    "observed_models": ["gpt-5.6-luna"],
                    "observed_providers": ["openai"],
                    "model_routed": True,
                    "provider_routed": True,
                    "passed": True,
                }
            },
            "gates": [
                {
                    "name": "tests",
                    "requested_command": ["pytest", "-q"],
                    "command": [private_path, "-m", "pytest"],
                    "returncode": 0,
                    "stdout_tail": private_path,
                    "stderr_tail": private_path,
                }
            ],
            "holdout": {"passed": True},
            "comparison": {"passed": True},
        },
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert private_path not in serialized
    assert "worktree" not in payload
    assert "promoted" not in payload
    assert "private-session-id" not in serialized
    assert "private-incomplete-id" not in serialized
    assert payload["role_evidence"]["research_scout"]["completed_count"] == 1
    assert payload["role_evidence"]["research_scout"]["incomplete_count"] == 1
    assert payload["gates"] == [
        {"name": "tests", "requested_command": ["pytest", "-q"], "returncode": 0}
    ]


def test_changed_files_uses_head_diff_without_rename_collapsing(
    monkeypatch, tmp_path: Path
) -> None:
    runner = load_runner()
    calls = []

    def fake_git_output(args, cwd):
        calls.append(args)
        if args[0] == "diff":
            return "src/old.py\nsrc/new.py"
        return "tests/test_new.py"

    monkeypatch.setattr(runner, "git_output", fake_git_output)

    changes = runner.changed_files(tmp_path)

    assert changes == ["src/new.py", "src/old.py", "tests/test_new.py"]
    assert "--no-renames" in calls[0]


def test_gate_environment_drops_unrelated_host_secrets(monkeypatch) -> None:
    runner = load_runner()
    monkeypatch.setenv("SYSTEMROOT", r"C:\\Windows")
    monkeypatch.setenv("PATH", r"C:\\Windows\\System32")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")

    environment = runner.safe_subprocess_environment()

    assert environment["SYSTEMROOT"] == r"C:\\Windows"
    assert "PATH" in environment
    assert "OPENAI_API_KEY" not in environment

    monkeypatch.setenv("USERPROFILE", r"C:\\Users\\rambo")
    codex_environment = runner.codex_cli_environment()
    assert codex_environment["USERPROFILE"] == r"C:\\Users\\rambo"
    assert "OPENAI_API_KEY" not in codex_environment


def test_preflight_fails_closed_when_codex_is_not_logged_in(monkeypatch) -> None:
    runner = load_runner()
    # The trusted outer gate deliberately runs a linked candidate worktree with
    # the root checkout's interpreter.  Make this preflight unit test assert its
    # own venv premise instead of inheriting whichever checkout pytest uses.
    monkeypatch.setattr(
        runner.sys,
        "executable",
        str(runner.ROOT / ".venv" / "Scripts" / "python.exe"),
    )
    runtime = runner.CodexRuntime(ROOT / "fake-codex.exe", "codex-cli 0.144.2")
    hypothesis = runner.Hypothesis(
        identifier="H001-forward-influence-routing",
        title="title",
        rationale="rationale",
        controls=(),
        primary_metric="synthetic.in_distribution.output_nmse",
        disconfirming_condition="condition",
        allowed_files=("src/focus_fabric/codecs.py",),
        status="pending",
        evaluator="fabric_benchmark_v1",
    )
    monkeypatch.setattr(runner, "resolve_codex_runtime", lambda _: runtime)
    monkeypatch.setattr(runner, "git_output", lambda args, cwd: "")
    monkeypatch.setattr(
        runner,
        "probe_codex_workspace_write",
        lambda *args, **kwargs: (True, "workspace-write sentinel created"),
    )

    def fake_run(command, **kwargs):
        if command[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(command, 1, "", "Not logged in")
        if command[1:3] == ["exec", "--help"]:
            return subprocess.CompletedProcess(
                command,
                0,
                "--strict-config --json --sandbox --output-schema --output-last-message",
                "",
            )
        if command[1:3] == ["debug", "models"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "models": [
                            {"slug": "gpt-5.6-sol"},
                            {"slug": "gpt-5.6-luna"},
                        ]
                    }
                ),
                "",
            )
        if command[-1] == "--help" and "--ignore-user-config" in command:
            return subprocess.CompletedProcess(command, 0, "validated", "")
        raise AssertionError(command)

    monkeypatch.setattr(runner, "run", fake_run)

    report = runner.preflight_report(ROOT, [hypothesis], "codex")

    assert report["ready_for_dry_run"] is True
    assert report["ready_for_execute"] is False
    assert report["workspace_write_ready"] is True
    assert report["session_metadata_ready"] is True
    assert any("logged in" in blocker.lower() for blocker in report["blockers"])

    monkeypatch.setattr(
        runner,
        "probe_codex_workspace_write",
        lambda *args, **kwargs: (False, "native sandbox denied the sentinel"),
    )
    report = runner.preflight_report(ROOT, [hypothesis], "codex")
    assert report["workspace_write_ready"] is False
    assert any("workspace-write" in blocker for blocker in report["blockers"])

    monkeypatch.setattr(
        runner,
        "probe_codex_workspace_write",
        lambda *args, **kwargs: (True, "workspace-write sentinel created"),
    )

    unsupported = runner.Hypothesis(
        identifier="H002",
        title="title",
        rationale="rationale",
        controls=(),
        primary_metric="unsupported_metric",
        disconfirming_condition="condition",
        allowed_files=("tests/",),
        status="pending",
    )
    report = runner.preflight_report(ROOT, [unsupported], "codex")
    assert report["ready_for_dry_run"] is True
    assert report["ready_for_execute"] is False


def test_change_report_rejects_parent_path_aliases() -> None:
    runner = load_runner()
    payload = {"changed_files": ["../src/focus_fabric/codecs.py"]}

    with pytest.raises(runner.PipelineError, match="unsafe"):
        runner.validate_reported_changes(
            payload,
            ["src/focus_fabric/codecs.py"],
        )


def test_experiment_contract_locks_preregistered_fields(tmp_path: Path) -> None:
    runner = load_runner()
    hypothesis = runner.Hypothesis(
        identifier="H001",
        title="title",
        rationale="rationale",
        controls=("committed baseline",),
        primary_metric="output_nmse",
        disconfirming_condition="less than five percent improvement",
        allowed_files=(),
        status="pending",
        evaluator="fabric_benchmark_v1",
    )
    contract_path = tmp_path / "results/experiments/H001/experiment.json"
    contract_path.parent.mkdir(parents=True)
    payload = runner.experiment_contract_for(hypothesis, "abc123")
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    assert runner.validate_experiment_contract(tmp_path, hypothesis, "abc123") == payload

    payload["controls"] = ["easier control"]
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.PipelineError, match="controls"):
        runner.validate_experiment_contract(tmp_path, hypothesis, "abc123")


def test_root_integrity_guard_fails_when_tracked_snapshot_changes(
    monkeypatch, tmp_path: Path
) -> None:
    runner = load_runner()
    original = runner.RootSnapshot("head", "", "tree-a", "baseline")
    changed = runner.RootSnapshot("head", "", "tree-b", "baseline")
    monkeypatch.setattr(runner, "capture_root_snapshot", lambda root: changed)

    with pytest.raises(runner.RootIntegrityError, match="tracked_tree_sha256"):
        with runner.root_integrity_guard(tmp_path, original):
            pass


def test_promotion_postcondition_requires_head_clean_status_and_matching_tree(
    monkeypatch, tmp_path: Path
) -> None:
    runner = load_runner()
    state = {"head": "candidate-commit", "status": "", "tree": "tree-digest"}

    def fake_git_output(args, cwd):
        if args[:2] == ["rev-parse", "HEAD"]:
            return state["head"]
        if args[:2] == ["status", "--porcelain"]:
            return state["status"]
        raise AssertionError(args)

    monkeypatch.setattr(runner, "git_output", fake_git_output)
    monkeypatch.setattr(runner, "tracked_tree_sha256", lambda root: state["tree"])

    runner.assert_promotion_postcondition(
        tmp_path,
        expected_head="candidate-commit",
        expected_tree_sha256="tree-digest",
    )

    state["status"] = " M README.md"
    with pytest.raises(runner.PromotionIntegrityError, match="status"):
        runner.assert_promotion_postcondition(
            tmp_path,
            expected_head="candidate-commit",
            expected_tree_sha256="tree-digest",
        )


def test_candidate_commit_must_match_fixed_parent_tree_and_paths(
    monkeypatch, tmp_path: Path
) -> None:
    runner = load_runner()
    state = {"parents": "candidate parent"}

    def fake_git_output(args, cwd):
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args[:2] == ["rev-parse", "HEAD"]:
            return "candidate"
        if args[:3] == ["rev-list", "--parents", "-n"]:
            return state["parents"]
        if args[:2] == ["rev-parse", "candidate^{tree}"]:
            return "fixed-tree"
        if args[0] == "diff-tree":
            return "results/experiments/H001/experiment.json\nsrc/code.py"
        raise AssertionError(args)

    monkeypatch.setattr(runner, "git_output", fake_git_output)
    monkeypatch.setattr(runner, "tracked_tree_sha256", lambda root: "actual-tree")

    assert runner.capture_clean_candidate_commit(
        tmp_path,
        expected_parent="parent",
        expected_tree="fixed-tree",
        expected_paths=["src/code.py", "results/experiments/H001/experiment.json"],
    ) == ("candidate", "actual-tree")

    state["parents"] = "injected parent candidate"
    with pytest.raises(runner.PromotionIntegrityError, match="ancestry"):
        runner.capture_clean_candidate_commit(
            tmp_path,
            expected_parent="parent",
            expected_tree="fixed-tree",
            expected_paths=["src/code.py", "results/experiments/H001/experiment.json"],
        )


def test_plumbing_stage_binds_exact_regular_file_bytes(tmp_path: Path) -> None:
    runner = load_runner()
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    git("init")
    git("config", "user.name", "Autonomy Test")
    git("config", "user.email", "autonomy@example.invalid")
    (repo / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "deleted.txt").write_text("delete me\n", encoding="utf-8")
    git("add", "--", "tracked.py", "deleted.txt")
    git("commit", "-m", "fixture")

    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "new.json").write_text('{"new":true}\n', encoding="utf-8")
    (repo / "deleted.txt").unlink()
    paths = ["tracked.py", "new.json", "deleted.txt"]
    expected = runner.candidate_blob_map(repo, paths)

    runner.stage_validated_paths(repo, paths)

    assert runner.staged_blob_map(repo, paths) == expected


def test_subagent_session_evidence_requires_new_completed_runtime_metadata(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    session_root = tmp_path / "sessions"
    old_path = session_root / "2026/07/17/old.jsonl"
    write_subagent_session(
        old_path,
        parent_thread_id="root-thread",
        role="research_scout",
        model="gpt-5.6-luna",
        session_id="old-forged-session",
    )
    previous_paths = runner.snapshot_codex_session_files(session_root)
    write_subagent_session(
        session_root / "2026/07/17/research.jsonl",
        parent_thread_id="root-thread",
        role="research_scout",
        model="gpt-5.6-luna",
        session_id="research-session",
    )
    write_subagent_session(
        session_root / "2026/07/17/memory.jsonl",
        parent_thread_id="root-thread",
        role="memory_redteam",
        model="gpt-5.6-sol",
        session_id="memory-session",
    )
    write_subagent_session(
        session_root / "2026/07/17/benchmark.jsonl",
        parent_thread_id="root-thread",
        role="benchmark_adversary",
        model="gpt-5.6-luna",
        session_id="benchmark-session",
        completed=False,
    )
    write_subagent_session(
        session_root / "2026/07/17/claim.jsonl",
        parent_thread_id="root-thread",
        role="claim_auditor",
        model="gpt-5.6-luna",
        session_id="claim-session",
        spawn_parent_thread_id="different-parent",
    )
    write_subagent_session(
        session_root / "2026/07/17/repro.jsonl",
        parent_thread_id="root-thread",
        role="reproducibility_auditor",
        model="gpt-5.6-luna",
        session_id="repro-session",
        trailing_event=True,
    )

    evidence = runner.subagent_role_evidence(
        session_root,
        previous_paths,
        "root-thread",
        {
            "research_scout": "gpt-5.6-luna",
            "memory_redteam": "gpt-5.6-luna",
            "benchmark_adversary": "gpt-5.6-luna",
            "claim_auditor": "gpt-5.6-luna",
            "reproducibility_auditor": "gpt-5.6-luna",
        },
    )

    assert evidence["research_scout"]["passed"] is True
    assert evidence["research_scout"]["completed_session_ids"] == [
        "research-session"
    ]
    assert evidence["memory_redteam"]["passed"] is False
    assert evidence["memory_redteam"]["observed_models"] == ["gpt-5.6-sol"]
    assert evidence["memory_redteam"]["completed_session_ids"] == ["memory-session"]
    assert evidence["benchmark_adversary"]["passed"] is False
    assert evidence["benchmark_adversary"]["incomplete_session_ids"] == [
        "benchmark-session"
    ]
    assert evidence["claim_auditor"]["passed"] is False
    assert evidence["claim_auditor"]["completed_session_ids"] == []
    assert evidence["reproducibility_auditor"]["passed"] is False
    assert evidence["reproducibility_auditor"]["incomplete_session_ids"] == [
        "repro-session"
    ]


def test_session_snapshot_skips_junction_directories(
    monkeypatch, tmp_path: Path
) -> None:
    runner = load_runner()
    session_root = tmp_path / "sessions"
    junction = session_root / "junction"
    write_subagent_session(
        junction / "escaped.jsonl",
        parent_thread_id="root-thread",
        role="research_scout",
        model="gpt-5.6-luna",
        session_id="escaped-session",
    )
    real_isjunction = getattr(runner.os.path, "isjunction", lambda _: False)
    monkeypatch.setattr(
        runner.os.path,
        "isjunction",
        lambda path: Path(path) == junction or real_isjunction(path),
        raising=False,
    )

    paths = runner.snapshot_codex_session_files(session_root)

    assert paths == frozenset()


def test_session_metadata_scan_is_bounded(monkeypatch, tmp_path: Path) -> None:
    runner = load_runner()
    session_root = tmp_path / "sessions"
    for index in range(2):
        write_subagent_session(
            session_root / f"session-{index}.jsonl",
            parent_thread_id="root-thread",
            role="research_scout",
            model="gpt-5.6-luna",
            session_id=f"session-{index}",
        )
    monkeypatch.setattr(runner, "MAX_CODEX_SESSION_FILES", 1)

    with pytest.raises(runner.PipelineError, match="file-count limit"):
        runner.snapshot_codex_session_files(session_root)

    monkeypatch.setattr(runner, "MAX_CODEX_SESSION_FILES", 100)
    monkeypatch.setattr(runner, "MAX_CODEX_SESSION_BYTES", 1)
    with pytest.raises(runner.PipelineError, match="size limit"):
        runner.subagent_role_evidence(
            session_root,
            frozenset(),
            "root-thread",
            {"research_scout": "gpt-5.6-luna"},
        )


def test_codex_thread_id_comes_from_the_cli_jsonl_envelope() -> None:
    runner = load_runner()

    thread_id = runner.codex_thread_id(
        '{"type":"thread.started","thread_id":"root-thread"}\n'
        '{"type":"item.completed","item":{"type":"agent_message",'
        '"text":"not trusted as a thread id"}}\n'
    )

    assert thread_id == "root-thread"
    with pytest.raises(runner.PipelineError, match="thread.started"):
        runner.codex_thread_id('{"type":"turn.completed"}\n')


def test_subagent_hook_records_only_minimal_lifecycle_fields(tmp_path: Path) -> None:
    hook = load_hook_recorder()

    event_dir = tmp_path / "orchestrator-owned-events"
    destination = hook.record_event(
        {
            "hook_event_name": "SubagentStart",
            "cwd": str(tmp_path),
            "agent_id": "agent-1",
            "agent_type": "research_scout",
            "model": "gpt-5.6-terra",
            "permission_mode": "dontAsk",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "transcript_path": "must-not-be-recorded",
        },
        event_dir,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert destination.parent == event_dir.resolve()
    assert payload["agent_type"] == "research_scout"
    assert "transcript_path" not in payload

    with pytest.raises(FileExistsError):
        hook.record_event(
            {
                "hook_event_name": "SubagentStart",
                "agent_id": "agent-1",
                "agent_type": "research_scout",
                "model": "gpt-5.6-terra",
            },
            event_dir,
        )


def test_candidate_state_is_rechecked_after_generated_gates(monkeypatch, tmp_path: Path) -> None:
    runner = load_runner()
    hypothesis = runner.Hypothesis(
        identifier="H001",
        title="title",
        rationale="rationale",
        controls=(),
        primary_metric="metric",
        disconfirming_condition="condition",
        allowed_files=("src/focus_fabric/codecs.py", "tests/"),
        status="pending",
    )
    monkeypatch.setattr(runner, "git_output", lambda args, cwd: "abc123")
    monkeypatch.setattr(
        runner,
        "changed_files",
        lambda _: ["src/focus_fabric/codecs.py", "docs/escaped.md"],
    )

    with pytest.raises(runner.PipelineError, match="docs/escaped.md"):
        runner.validate_candidate_state(tmp_path, hypothesis, "abc123")


def test_candidate_cannot_add_a_new_allowed_path_after_gates() -> None:
    runner = load_runner()
    hypothesis = runner.Hypothesis(
        identifier="H001",
        title="title",
        rationale="rationale",
        controls=(),
        primary_metric="metric",
        disconfirming_condition="condition",
        allowed_files=("src/focus_fabric/codecs.py", "tests/"),
        status="pending",
    )
    initial = [
        "results/experiments/H001/experiment.json",
        "src/focus_fabric/codecs.py",
    ]
    final = [
        *initial,
        "tests/test_late_injection.py",
        "results/candidate_benchmark.json",
        "results/experiments/H001/holdout-baseline.json",
        "results/experiments/H001/holdout-candidate.json",
        "results/experiments/H001/orchestrator-result.json",
    ]

    with pytest.raises(runner.CandidateIntegrityError, match="path set changed"):
        runner.assert_candidate_path_set_unchanged(initial, final, hypothesis)


def test_accepted_candidate_is_not_committed_without_auto_promote(
    monkeypatch, tmp_path: Path
) -> None:
    runner = load_runner()
    root = tmp_path / "root"
    worktree_root = tmp_path / "worktrees"
    (root / "results").mkdir(parents=True)
    (root / "autonomy" / "state").mkdir(parents=True)
    session_root = tmp_path / "codex-home" / "sessions"
    session_root.mkdir(parents=True)
    (root / "results" / "fabric_benchmark.json").write_text(
        (ROOT / "results" / "fabric_benchmark.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runtime = runner.CodexRuntime(tmp_path / "codex.exe", "codex-cli 0.144.2")
    hypothesis = runner.Hypothesis(
        identifier="H001-forward-influence-routing",
        title="title",
        rationale="rationale",
        controls=(),
        primary_metric="synthetic.in_distribution.output_nmse",
        disconfirming_condition="condition",
        allowed_files=("src/focus_fabric/codecs.py", "tests/"),
        status="pending",
        evaluator="fabric_benchmark_v1",
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "git" and "worktree" in command and "add" in command:
            worktree = Path(command[-2])
            profile_dir = worktree / ".codex" / "agents"
            profile_dir.mkdir(parents=True)
            (worktree / ".codex/config.toml").write_text(
                'model = "gpt-5.6-sol"\n', encoding="utf-8"
            )
            for role in sorted({*runner.REQUIRED_RESEARCH_ROLES, "default"}):
                (profile_dir / f"{role}.toml").write_text(
                    f'name = "{role}"\n'
                    f'description = "{role} profile"\n'
                    'model = "gpt-5.6-luna"\n'
                    'developer_instructions = "bounded role"\n',
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(command, 0, "", "")
        if Path(command[0]) == runtime.path:
            worktree = kwargs["cwd"]
            experiment = (
                worktree
                / "results/experiments/H001-forward-influence-routing/experiment.json"
            )
            assert experiment.is_file()
            assert json.loads(experiment.read_text(encoding="utf-8")) == (
                runner.experiment_contract_for(
                    hypothesis,
                    runner.sha256_file(root / "results/fabric_benchmark.json"),
                )
            )
            result_path = (
                worktree
                / "autonomy/state/runs/H001-forward-influence-routing/agent-result.json"
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "summary": "candidate ready",
                        "changed_files": ["src/focus_fabric/codecs.py"],
                        "evidence": ["tests passed"],
                        "risks": [],
                        "negative_results": [],
                        "next_hypotheses": [],
                    }
                ),
                encoding="utf-8",
            )
            for index, role in enumerate(sorted(runner.REQUIRED_RESEARCH_ROLES)):
                write_subagent_session(
                    session_root / f"session-{index}.jsonl",
                    parent_thread_id="root-thread",
                    role=role,
                    model="gpt-5.6-luna",
                    session_id=f"agent-{index}",
                )
            return subprocess.CompletedProcess(
                command,
                0,
                '{"type":"thread.started","thread_id":"root-thread"}\n',
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "codex_session_root", lambda _: session_root)
    monkeypatch.setattr(runner, "git_output", lambda args, cwd: "abc123")
    monkeypatch.setattr(
        runner,
        "changed_files",
        lambda _: [
            "results/experiments/H001-forward-influence-routing/experiment.json",
            "src/focus_fabric/codecs.py",
        ],
    )

    def passing_gates(*args, **kwargs):
        return [{"returncode": 0}]

    def passing_trusted_tests(*args, **kwargs):
        return {"returncode": 0}

    def passing_holdout(*args, **kwargs):
        return {"passed": True}

    monkeypatch.setattr(runner, "run_gates", passing_gates)
    monkeypatch.setattr(runner, "run_trusted_root_tests", passing_trusted_tests)
    monkeypatch.setattr(runner, "run_external_holdout", passing_holdout)
    monkeypatch.setattr(
        runner,
        "compare_candidate",
        lambda root, worktree, hypothesis: {"passed": True},
    )
    ledger = runner.EventLedger(root / "autonomy/state/events.jsonl")

    result = runner.execute_one(
        root,
        hypothesis,
        runtime=runtime,
        worktree_root=worktree_root,
        auto_promote=False,
        ledger=ledger,
    )

    assert result["status"] == "accepted"
    assert result["promoted"] is False
    assert result["agent_changed_files"] == ["src/focus_fabric/codecs.py"]
    assert not any(command[0] == "git" and "commit" in command for command in commands)
    assert not any(command[0] == "git" and "merge" in command for command in commands)
    events = [
        json.loads(line)["event"]
        for line in (root / "autonomy/state/events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events.index("experiment_contract.preregistered") < events.index(
        "codex.started"
    )
