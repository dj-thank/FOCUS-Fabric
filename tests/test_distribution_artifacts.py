from __future__ import annotations

from email.message import Message
from email.generator import BytesGenerator
from io import BytesIO
import importlib.util
from pathlib import Path
import re
import tarfile
import zipfile

import pytest
import focus_fabric


_MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "release" / "verify_distributions.py"
)
_SPEC = importlib.util.spec_from_file_location("verify_distributions", _MODULE_PATH)
verify_distributions = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(verify_distributions)


def _metadata(*, version: str = "0.2.1") -> bytes:
    message = Message()
    message["Metadata-Version"] = "2.4"
    message["Name"] = "focus-fabric"
    message["Version"] = version
    output = BytesIO()
    BytesGenerator(output, maxheaderlen=0).flatten(message)
    return output.getvalue()


def _write_wheel(path: Path, *, extra: dict[str, bytes] | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("focus_fabric/__init__.py", b"")
        archive.writestr("focus_native/__init__.py", b"")
        archive.writestr("focus_fabric-0.2.1.dist-info/METADATA", _metadata())
        for name, payload in (extra or {}).items():
            archive.writestr(name, payload)


def _write_sdist(
    path: Path,
    *,
    extra: dict[str, bytes] | None = None,
    link: str | None = None,
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in {
            "focus_fabric-0.2.1/PKG-INFO": _metadata(),
            "focus_fabric-0.2.1/src/focus_fabric/__init__.py": b"",
            "focus_fabric-0.2.1/src/focus_native/__init__.py": b"",
            **(extra or {}),
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
        if link is not None:
            info = tarfile.TarInfo("focus_fabric-0.2.1/linked")
            info.type = tarfile.SYMTYPE
            info.linkname = link
            archive.addfile(info)


def test_distribution_set_accepts_clean_wheel_and_sdist(tmp_path: Path) -> None:
    wheel = tmp_path / "focus_fabric-0.2.1-py3-none-any.whl"
    sdist = tmp_path / "focus_fabric-0.2.1.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    report = verify_distributions.verify_distribution_set(
        tmp_path, expected_version="0.2.1"
    )

    assert {artifact["kind"] for artifact in report["artifacts"]} == {
        "wheel",
        "sdist",
    }
    assert all(artifact["version"] == "0.2.1" for artifact in report["artifacts"])
    assert all(len(artifact["sha256"]) == 64 for artifact in report["artifacts"])


def test_runtime_version_matches_project_metadata() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert match is not None
    assert focus_fabric.__version__ == match.group(1) == "0.2.1"


@pytest.mark.parametrize(
    "member",
    [
        "checkpoints/model.safetensors",
        "checkpoints/model.pt",
        "checkpoints/model.pth",
        "checkpoints/model.ckpt",
        "checkpoints/model.bin",
        "checkpoints/model.onnx",
        "checkpoints/MODEL.SAFETENSORS",
    ],
)
@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distribution_set_rejects_weight_payloads(
    tmp_path: Path, member: str, kind: str
) -> None:
    wheel = tmp_path / "focus_fabric-0.2.1-py3-none-any.whl"
    sdist = tmp_path / "focus_fabric-0.2.1.tar.gz"
    _write_wheel(wheel, extra={member: b"private"} if kind == "wheel" else None)
    _write_sdist(
        sdist,
        extra={f"focus_fabric-0.2.1/{member}": b"private"}
        if kind == "sdist"
        else None,
    )

    with pytest.raises(
        verify_distributions.DistributionVerificationError,
        match="forbidden weight payload",
    ):
        verify_distributions.verify_distribution_set(
            tmp_path, expected_version="0.2.1"
        )


def test_distribution_set_rejects_path_escape(tmp_path: Path) -> None:
    wheel = tmp_path / "focus_fabric-0.2.1-py3-none-any.whl"
    sdist = tmp_path / "focus_fabric-0.2.1.tar.gz"
    _write_wheel(wheel, extra={"../escape.txt": b"escape"})
    _write_sdist(sdist)

    with pytest.raises(
        verify_distributions.DistributionVerificationError,
        match="unsafe archive path",
    ):
        verify_distributions.verify_distribution_set(
            tmp_path, expected_version="0.2.1"
        )


def test_distribution_set_rejects_tar_links(tmp_path: Path) -> None:
    wheel = tmp_path / "focus_fabric-0.2.1-py3-none-any.whl"
    sdist = tmp_path / "focus_fabric-0.2.1.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist, link="../../private")

    with pytest.raises(
        verify_distributions.DistributionVerificationError,
        match="non-regular tar member",
    ):
        verify_distributions.verify_distribution_set(
            tmp_path, expected_version="0.2.1"
        )


def test_distribution_set_rejects_wrong_metadata_version(tmp_path: Path) -> None:
    wheel = tmp_path / "focus_fabric-0.2.1-py3-none-any.whl"
    sdist = tmp_path / "focus_fabric-0.2.1.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("focus_fabric/__init__.py", b"")
        archive.writestr("focus_native/__init__.py", b"")
        archive.writestr("focus_fabric-0.2.1.dist-info/METADATA", _metadata(version="0.2.0"))

    with pytest.raises(
        verify_distributions.DistributionVerificationError,
        match="metadata version",
    ):
        verify_distributions.verify_distribution_set(
            tmp_path, expected_version="0.2.1"
        )


def test_distribution_set_rejects_missing_package_payload(tmp_path: Path) -> None:
    wheel = tmp_path / "focus_fabric-0.2.1-py3-none-any.whl"
    sdist = tmp_path / "focus_fabric-0.2.1.tar.gz"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("focus_fabric-0.2.1.dist-info/METADATA", _metadata())
    _write_sdist(sdist)

    with pytest.raises(
        verify_distributions.DistributionVerificationError,
        match="missing required package payload",
    ):
        verify_distributions.verify_distribution_set(
            tmp_path, expected_version="0.2.1"
        )


def test_distribution_set_rejects_symlink_artifact(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    dist_dir = tmp_path / "dist"
    real_dir.mkdir()
    dist_dir.mkdir()
    real_wheel = real_dir / "outside.whl"
    _write_wheel(real_wheel)
    linked_wheel = dist_dir / "focus_fabric-0.2.1-py3-none-any.whl"
    try:
        linked_wheel.symlink_to(real_wheel)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    _write_sdist(dist_dir / "focus_fabric-0.2.1.tar.gz")

    with pytest.raises(
        verify_distributions.DistributionVerificationError,
        match="regular file inside dist directory",
    ):
        verify_distributions.verify_distribution_set(
            dist_dir, expected_version="0.2.1"
        )
