#!/usr/bin/env python3
"""Fail closed when Python release archives contain unsafe or private payloads."""
from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PROJECT_NAME = "focus-fabric"
ARCHIVE_NAME = "focus_fabric"
FORBIDDEN_WEIGHT_SUFFIXES = {
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".onnx",
}


class DistributionVerificationError(RuntimeError):
    """Raised when a distribution archive violates the public release contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_project_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _validate_member_name(name: str) -> PurePosixPath:
    if not name or "\0" in name or "\\" in name:
        raise DistributionVerificationError(f"unsafe archive path: {name!r}")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise DistributionVerificationError(f"unsafe archive path: {name!r}")
    raw_parts = name.rstrip("/").split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise DistributionVerificationError(f"unsafe archive path: {name!r}")
    relative = PurePosixPath(*raw_parts)
    if relative.suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES:
        raise DistributionVerificationError(
            f"forbidden weight payload in public distribution: {name}"
        )
    return relative


def _validate_metadata(
    payload: bytes,
    *,
    expected_version: str,
    archive: Path,
) -> tuple[str, str]:
    metadata = BytesParser(policy=policy.default).parsebytes(payload)
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or _canonical_project_name(name) != PROJECT_NAME:
        raise DistributionVerificationError(
            f"unexpected metadata project name in {archive.name}: {name!r}"
        )
    if version != expected_version:
        raise DistributionVerificationError(
            f"metadata version in {archive.name} is {version!r}, "
            f"expected {expected_version!r}"
        )
    return name, version


def _verify_wheel(path: Path, *, expected_version: str) -> dict[str, object]:
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise DistributionVerificationError(f"wheel CRC check failed: {path.name}")
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise DistributionVerificationError(
                    f"duplicate wheel member name: {path.name}"
                )
            for member in members:
                _validate_member_name(member.filename)
                mode = (member.external_attr >> 16) & 0xFFFF
                if mode and stat.S_ISLNK(mode):
                    raise DistributionVerificationError(
                        f"non-regular wheel member: {member.filename}"
                    )
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise DistributionVerificationError(
                    f"wheel must contain exactly one METADATA file: {path.name}"
                )
            required_payload = {
                "focus_fabric/__init__.py",
                "focus_native/__init__.py",
            }
            missing_payload = sorted(required_payload - set(names))
            if missing_payload:
                raise DistributionVerificationError(
                    f"missing required package payload in {path.name}: {missing_payload}"
                )
            name, version = _validate_metadata(
                archive.read(metadata_names[0]),
                expected_version=expected_version,
                archive=path,
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise DistributionVerificationError(f"cannot read wheel: {path.name}") from exc
    return {
        "kind": "wheel",
        "path": str(path.resolve()),
        "name": name,
        "version": version,
        "members": len(members),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _verify_sdist(path: Path, *, expected_version: str) -> dict[str, object]:
    allowed_roots = {
        f"{ARCHIVE_NAME}-{expected_version}",
        f"{PROJECT_NAME}-{expected_version}",
    }
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise DistributionVerificationError(
                    f"duplicate sdist member name: {path.name}"
                )
            roots: set[str] = set()
            for member in members:
                relative = _validate_member_name(member.name)
                roots.add(relative.parts[0])
                if not member.isdir() and not member.isfile():
                    raise DistributionVerificationError(
                        f"non-regular tar member: {member.name}"
                    )
            if len(roots) != 1 or not roots.issubset(allowed_roots):
                raise DistributionVerificationError(
                    f"sdist members are outside the expected root: {sorted(roots)}"
                )
            expected_root = next(iter(roots))
            required_payload = {
                f"{expected_root}/src/focus_fabric/__init__.py",
                f"{expected_root}/src/focus_native/__init__.py",
            }
            missing_payload = sorted(required_payload - set(names))
            if missing_payload:
                raise DistributionVerificationError(
                    f"missing required package payload in {path.name}: {missing_payload}"
                )
            metadata_name = f"{expected_root}/PKG-INFO"
            metadata_members = [member for member in members if member.name == metadata_name]
            if len(metadata_members) != 1:
                raise DistributionVerificationError(
                    f"sdist must contain {metadata_name}: {path.name}"
                )
            extracted = archive.extractfile(metadata_members[0])
            if extracted is None:
                raise DistributionVerificationError(
                    f"cannot read sdist metadata: {path.name}"
                )
            name, version = _validate_metadata(
                extracted.read(),
                expected_version=expected_version,
                archive=path,
            )
    except (OSError, tarfile.TarError) as exc:
        raise DistributionVerificationError(f"cannot read sdist: {path.name}") from exc
    return {
        "kind": "sdist",
        "path": str(path.resolve()),
        "name": name,
        "version": version,
        "members": len(members),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_distribution_set(
    dist_dir: Path,
    *,
    expected_version: str,
) -> dict[str, object]:
    """Verify the exact wheel/sdist pair for one version and reject stale siblings."""

    requested_dist_dir = dist_dir.absolute()
    if requested_dist_dir.is_symlink() or not requested_dist_dir.is_dir():
        raise DistributionVerificationError(
            "distribution directory must be a real directory"
        )
    dist_dir = requested_dist_dir.resolve(strict=True)
    expected = {
        dist_dir / f"{ARCHIVE_NAME}-{expected_version}-py3-none-any.whl",
        dist_dir / f"{ARCHIVE_NAME}-{expected_version}.tar.gz",
    }
    candidates = set(dist_dir.glob(f"{ARCHIVE_NAME}-*.whl")) | set(
        dist_dir.glob(f"{ARCHIVE_NAME}-*.tar.gz")
    )
    missing = sorted(path.name for path in expected - candidates)
    unexpected = sorted(path.name for path in candidates - expected)
    if missing or unexpected:
        raise DistributionVerificationError(
            f"distribution set mismatch; missing={missing}, unexpected={unexpected}"
        )
    wheel = dist_dir / f"{ARCHIVE_NAME}-{expected_version}-py3-none-any.whl"
    sdist = dist_dir / f"{ARCHIVE_NAME}-{expected_version}.tar.gz"
    for artifact in (wheel, sdist):
        if artifact.is_symlink() or not artifact.is_file():
            raise DistributionVerificationError(
                f"artifact must be a regular file inside dist directory: {artifact.name}"
            )
        resolved_artifact = artifact.resolve(strict=True)
        if resolved_artifact.parent != dist_dir:
            raise DistributionVerificationError(
                f"artifact must be a regular file inside dist directory: {artifact.name}"
            )
    return {
        "project": PROJECT_NAME,
        "version": expected_version,
        "artifacts": [
            _verify_wheel(wheel, expected_version=expected_version),
            _verify_sdist(sdist, expected_version=expected_version),
        ],
    }


def project_version(project_root: Path = ROOT) -> str:
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        pyproject,
        flags=re.MULTILINE,
    )
    if match is None:
        raise DistributionVerificationError("cannot resolve project version")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--version", default=project_version())
    args = parser.parse_args()
    try:
        report = verify_distribution_set(
            args.dist_dir,
            expected_version=args.version,
        )
    except DistributionVerificationError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"passed": True, **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
