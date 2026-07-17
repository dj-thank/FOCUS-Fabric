from __future__ import annotations

from io import BytesIO
import importlib.util
from pathlib import Path
import tarfile

import pytest


_MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "release" / "build_distributions.py"
)
_SPEC = importlib.util.spec_from_file_location("build_distributions", _MODULE_PATH)
build_distributions = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(build_distributions)


def test_extract_git_archive_copies_regular_source(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.tar"
    with tarfile.open(archive_path, "w") as archive:
        payload = b"[project]\nname='demo'\n"
        info = tarfile.TarInfo("pyproject.toml")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    target = tmp_path / "source"

    build_distributions.extract_git_archive(archive_path, target)

    assert (target / "pyproject.toml").read_bytes() == payload


def test_extract_git_archive_rejects_link(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("linked")
        info.type = tarfile.SYMTYPE
        info.linkname = "../private"
        archive.addfile(info)

    with pytest.raises(RuntimeError, match="non-regular Git archive member"):
        build_distributions.extract_git_archive(archive_path, tmp_path / "source")


def test_output_directory_must_stay_in_workspace(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "FOCUS-Fabric"
    root.mkdir(parents=True)
    monkeypatch.setattr(build_distributions, "ROOT", root)

    with pytest.raises(RuntimeError, match="dedicated repository dist"):
        build_distributions.validated_output_directory(tmp_path / "outside")

    allowed = root / "dist"
    assert build_distributions.validated_output_directory(allowed) == allowed.absolute()

    with pytest.raises(RuntimeError, match="dedicated repository dist"):
        build_distributions.validated_output_directory(root / ".git")
