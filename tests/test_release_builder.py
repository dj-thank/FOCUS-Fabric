from pathlib import Path
import importlib.util
import subprocess
import zipfile

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


def test_release_files_excludes_model_weights_case_insensitively(
    tmp_path, monkeypatch
):
    names = [
        "model.safetensors",
        "model.SAFETENSORS",
        "model.pt",
        "model.pth",
        "model.ckpt",
        "model.bin",
        "model.onnx",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"private")
    safe = tmp_path / "metadata.json"
    safe.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(build_release, "ROOT", tmp_path)
    monkeypatch.setattr(
        build_release.subprocess,
        "run",
        _git_result(
            [("100644", name) for name in [*names, "metadata.json"]],
            root=tmp_path,
        ),
    )

    assert build_release.release_files() == [safe]


def test_release_zip_verifier_rejects_forbidden_payload(tmp_path):
    archive_path = tmp_path / "release.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("release/checkpoints/model.BIN", b"private")

    with pytest.raises(RuntimeError, match="forbidden release payload"):
        build_release.verify_zip_archive(archive_path, release_name="release")


def test_build_zip_uses_captured_commit_payloads(tmp_path, monkeypatch):
    root = tmp_path / "FOCUS-Fabric"
    root.mkdir()
    monkeypatch.setattr(build_release, "ROOT", root)
    output = tmp_path / "outputs" / "release" / "release.zip"

    build_release.build_zip(
        output,
        release_name="release",
        payloads=[("tracked.txt", 0o100644, b"committed bytes")],
    )

    with zipfile.ZipFile(output) as archive:
        assert archive.read("release/tracked.txt") == b"committed bytes"


def test_release_output_is_limited_to_dedicated_outputs(tmp_path, monkeypatch):
    root = tmp_path / "FOCUS-Fabric"
    root.mkdir()
    monkeypatch.setattr(build_release, "ROOT", root)

    with pytest.raises(RuntimeError, match="dedicated outputs directory"):
        build_release.validated_release_output(tmp_path / "private.zip")

    allowed = tmp_path / "outputs" / "release" / "release.zip"
    assert build_release.validated_release_output(allowed) == allowed.absolute()


def test_generated_metadata_path_rejects_links(tmp_path, monkeypatch):
    root = tmp_path / "FOCUS-Fabric"
    root.mkdir()
    manifest = root / "RELEASE_MANIFEST.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(build_release, "ROOT", root)
    original_is_symlink = Path.is_symlink

    def is_symlink(path):
        return path == manifest or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)

    with pytest.raises(RuntimeError, match="generated metadata path"):
        build_release.assert_generated_metadata_path(manifest)


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


def test_retained_release_identity_cannot_be_rebound(monkeypatch) -> None:
    monkeypatch.setattr(build_release, "RELEASE_NAME", "FOCUS_Fabric_2026-07_public_release")
    monkeypatch.setattr(build_release, "VERSION", "0.2.0")
    monkeypatch.setattr(build_release, "RELEASE_DATE", build_release.date(2026, 7, 14))

    with pytest.raises(RuntimeError, match="already bound"):
        build_release.assert_release_identity("f" * 40)

    build_release.assert_release_identity(
        "aa96cd5e8c6d730380c425bdc7b43cd751316c90"
    )


def test_current_release_identity_is_unique() -> None:
    assert build_release.VERSION == "0.2.1"
    assert build_release.RELEASE_NAME == "FOCUS_Fabric_2026-07_0.2.1_release"
    assert build_release.RELEASE_DATE == build_release.date(2026, 7, 17)
    build_release.assert_release_identity("f" * 40)


def test_final_release_requires_main_at_origin_main(monkeypatch) -> None:
    monkeypatch.setattr(build_release, "source_commit", lambda: "a" * 40)

    def git_text(*arguments):
        if arguments == ("remote", "get-url", "origin"):
            return "https://github.com/dj-thank/FOCUS-Fabric.git"
        if arguments == ("fetch", "--quiet", "origin", "main"):
            return ""
        if arguments == ("symbolic-ref", "--quiet", "--short", "HEAD"):
            return "agent/release-0.2.1"
        if arguments == ("rev-parse", "--verify", "FETCH_HEAD"):
            return "a" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(build_release, "_git_text", git_text)

    with pytest.raises(RuntimeError, match="main branch"):
        build_release.assert_final_publication_source()


def test_final_release_requires_head_to_match_origin_main(monkeypatch) -> None:
    monkeypatch.setattr(build_release, "source_commit", lambda: "a" * 40)

    def git_text(*arguments):
        if arguments == ("remote", "get-url", "origin"):
            return "https://github.com/dj-thank/FOCUS-Fabric.git"
        if arguments == ("fetch", "--quiet", "origin", "main"):
            return ""
        if arguments == ("symbolic-ref", "--quiet", "--short", "HEAD"):
            return "main"
        if arguments == ("rev-parse", "--verify", "FETCH_HEAD"):
            return "b" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(build_release, "_git_text", git_text)

    with pytest.raises(RuntimeError, match="origin/main"):
        build_release.assert_final_publication_source()


def test_final_release_accepts_main_at_origin_main(monkeypatch) -> None:
    monkeypatch.setattr(build_release, "source_commit", lambda: "a" * 40)

    def git_text(*arguments):
        if arguments == ("remote", "get-url", "origin"):
            return "https://github.com/dj-thank/FOCUS-Fabric.git"
        if arguments == ("fetch", "--quiet", "origin", "main"):
            return ""
        if arguments == ("symbolic-ref", "--quiet", "--short", "HEAD"):
            return "main"
        if arguments == ("rev-parse", "--verify", "FETCH_HEAD"):
            return "a" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(build_release, "_git_text", git_text)

    build_release.assert_final_publication_source()


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
    monkeypatch.setattr(build_release, "RELEASE_NAME", "FOCUS_Fabric_2026-07_public_release")
    monkeypatch.setattr(build_release, "VERSION", "0.2.0")
    monkeypatch.setattr(build_release, "RELEASE_DATE", build_release.date(2026, 7, 14))

    with pytest.raises(RuntimeError, match="already bound"):
        build_release.build_manifest()


def test_manifest_records_distribution_hashes_without_local_paths(
    tmp_path, monkeypatch
):
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
    report = {
        "source": {"git_commit": "a" * 40},
        "clean_target_wheel_import": True,
        "artifacts": [
            {
                "kind": "wheel",
                "path": "C:/private/dist/focus_fabric-0.2.1-py3-none-any.whl",
                "name": "focus-fabric",
                "version": "0.2.1",
                "members": 35,
                "bytes": 123,
                "sha256": "1" * 64,
            }
        ]
    }

    manifest = build_release.build_manifest(distribution_report=report)

    artifact = manifest["artifacts"]["python_distributions"][0]
    assert artifact["filename"] == "focus_fabric-0.2.1-py3-none-any.whl"
    assert "path" not in artifact
    assert "private" not in str(manifest)
    assert manifest["artifacts"]["python_distribution_source"] == {
        "git_commit": "a" * 40
    }
    assert manifest["local_verification"]["wheel_and_sdist"] == "passed"


def test_manifest_uses_captured_commit_payloads_instead_of_worktree(
    tmp_path, monkeypatch
):
    docs = tmp_path / "docs"
    results = tmp_path / "results"
    docs.mkdir()
    results.mkdir()
    (docs / "CLAIMS_LEDGER.json").write_text(
        '{"claims": ["uncommitted"]}', encoding="utf-8"
    )
    (results / "fabric_benchmark.json").write_text(
        '{"environment": {"torch": "uncommitted", "cuda_available": true}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(build_release, "ROOT", tmp_path)
    source_payloads = [
        ("docs/CLAIMS_LEDGER.json", 0o100644, b'{"claims": []}'),
        (
            "results/fabric_benchmark.json",
            0o100644,
            b'{"environment": {"torch": "committed", "cuda_available": false}}',
        ),
    ]

    manifest = build_release.build_manifest(
        source_payloads=source_payloads,
        source_commit_value="a" * 40,
        source_tree_value="b" * 40,
    )

    assert manifest["claim_count"] == 0
    assert manifest["environment"]["torch"] == "committed"
    assert manifest["source"] == {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
    }
