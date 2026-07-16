from pathlib import Path
import importlib.util
import subprocess

import pytest

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "release" / "build_release.py"
_SPEC = importlib.util.spec_from_file_location("build_release", _MODULE_PATH)
build_release = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(build_release)


def _git_result(records: list[tuple[str, str]], *, root: Path):
    payload = b"".join(
        f"{mode} {'0' * 40} 0\t{path}".encode() + b"\0"
        for mode, path in records
    )

    def run(command, **kwargs):
        assert kwargs["cwd"] == root.resolve()
        if "ls-files" in command:
            return subprocess.CompletedProcess(command, 0, stdout=payload, stderr=b"")
        raise AssertionError(command)

    return run


def test_release_files_uses_tracked_files_and_generated_contract(tmp_path, monkeypatch):
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("untracked", encoding="utf-8")
    manifest = tmp_path / "RELEASE_MANIFEST.json"
    manifest.write_text("{}", encoding="utf-8")
    checksums = tmp_path / "SHA256SUMS"
    checksums.write_text("", encoding="utf-8")
    monkeypatch.setattr(build_release, "ROOT", tmp_path)
    monkeypatch.setattr(
        build_release.subprocess,
        "run",
        _git_result(
            [("100644", "tracked.txt"), ("100644", "RELEASE_MANIFEST.json"), ("100644", "SHA256SUMS")],
            root=tmp_path,
        ),
    )

    assert [p.name for p in build_release.release_files()] == ["tracked.txt"]
    assert {p.name for p in build_release.release_files(include_generated=True)} == {
        "tracked.txt", "RELEASE_MANIFEST.json", "SHA256SUMS"
    }


def test_release_files_rejects_missing_tracked_file(tmp_path, monkeypatch):
    monkeypatch.setattr(build_release, "ROOT", tmp_path)
    monkeypatch.setattr(
        build_release.subprocess, "run", _git_result([("100644", "missing.txt")], root=tmp_path)
    )
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        build_release.release_files()


def test_release_files_rejects_tracked_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(build_release, "ROOT", tmp_path)
    monkeypatch.setattr(
        build_release.subprocess, "run", _git_result([("120000", "link.txt")], root=tmp_path)
    )
    with pytest.raises(RuntimeError, match="non-regular"):
        build_release.release_files()


def test_release_files_rejects_worktree_link_for_regular_index_entry(
    tmp_path, monkeypatch
):
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked", encoding="utf-8")
    monkeypatch.setattr(build_release, "ROOT", tmp_path)
    monkeypatch.setattr(
        build_release.subprocess,
        "run",
        _git_result([("100644", "tracked.txt")], root=tmp_path),
    )
    original_is_symlink = Path.is_symlink

    def is_symlink(path):
        return path == tracked or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)

    with pytest.raises(RuntimeError, match="linked tracked"):
        build_release.release_files()


def test_release_generation_rejects_dirty_tracked_source(tmp_path, monkeypatch):
    monkeypatch.setattr(build_release, "ROOT", tmp_path)

    def run(command, **kwargs):
        assert kwargs["cwd"] == tmp_path.resolve()
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"")

    monkeypatch.setattr(build_release.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="must be clean"):
        build_release.assert_clean_tracked_source()


def test_source_commit_is_required_and_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(build_release, "ROOT", tmp_path)

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="not-a-commit\n", stderr="")

    monkeypatch.setattr(build_release.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="malformed"):
        build_release.source_commit()


def test_source_tree_is_required_and_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(build_release, "ROOT", tmp_path)

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="not-a-tree\n", stderr="")

    monkeypatch.setattr(build_release.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="malformed"):
        build_release.source_tree()


def test_retained_release_identity_cannot_be_rebound() -> None:
    with pytest.raises(RuntimeError, match="already bound"):
        build_release.assert_release_identity("f" * 40)

    build_release.assert_release_identity(
        "aa96cd5e8c6d730380c425bdc7b43cd751316c90"
    )


def test_manifest_build_rejects_rebound_retained_identity(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    results = tmp_path / "results"
    docs.mkdir()
    results.mkdir()
    (docs / "CLAIMS_LEDGER.json").write_text('{"claims": []}', encoding="utf-8")
    (results / "fabric_benchmark.json").write_text(
        '{"environment": {"torch": "test", "cuda_available": false}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(build_release, "ROOT", tmp_path)
    monkeypatch.setattr(build_release, "release_files", lambda: [])
    monkeypatch.setattr(build_release, "source_commit", lambda: "a" * 40)
    monkeypatch.setattr(build_release, "source_tree", lambda: "f" * 40)

    with pytest.raises(RuntimeError, match="already bound"):
        build_release.build_manifest()
