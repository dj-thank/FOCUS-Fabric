#!/usr/bin/env python3
"""Build wheel and sdist from the exact clean Git HEAD, then verify them."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def _git_command(*arguments: str) -> list[str]:
    return ["git", "-c", f"safe.directory={ROOT.resolve()}", *arguments]


def _git_text(*arguments: str) -> str:
    try:
        result = subprocess.run(
            _git_command(*arguments),
            cwd=ROOT.resolve(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot query Git state: {' '.join(arguments)}") from exc
    return result.stdout.strip()


def assert_clean_tracked_source() -> None:
    for arguments in (
        ("diff", "--quiet", "--no-ext-diff", "--"),
        ("diff", "--cached", "--quiet", "--no-ext-diff", "--"),
    ):
        try:
            result = subprocess.run(
                _git_command(*arguments),
                cwd=ROOT.resolve(),
                check=False,
                capture_output=True,
            )
        except OSError as exc:
            raise RuntimeError("cannot verify distribution source cleanliness") from exc
        if result.returncode == 1:
            raise RuntimeError("tracked source must be clean before distribution build")
        if result.returncode != 0:
            raise RuntimeError("cannot verify distribution source cleanliness")


def source_commit() -> str:
    commit = _git_text("rev-parse", "--verify", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("distribution source commit is malformed")
    return commit


def validated_output_directory(output_dir: Path) -> Path:
    """Restrict replaceable build outputs to this checkout's generated dist dir."""

    requested = output_dir.absolute()
    expected = ROOT.resolve() / "dist"
    resolved = requested.resolve(strict=False)
    if requested.is_symlink() or resolved != expected:
        raise RuntimeError(
            "distribution output must be the dedicated repository dist directory"
        )
    return expected


def extract_git_archive(archive_path: Path, target: Path) -> None:
    """Extract only regular, relative members from a Git-generated tar archive."""

    target.mkdir(parents=True, exist_ok=False)
    root = target.resolve()
    try:
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive.getmembers():
                name = member.name
                relative = PurePosixPath(name)
                if (
                    not name
                    or "\\" in name
                    or relative.is_absolute()
                    or any(part in {"", ".", ".."} for part in relative.parts)
                ):
                    raise RuntimeError(f"unsafe Git archive member: {name!r}")
                destination = (root / Path(*relative.parts)).resolve(strict=False)
                try:
                    destination.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(f"unsafe Git archive member: {name!r}") from exc
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise RuntimeError(f"non-regular Git archive member: {name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"cannot read Git archive member: {name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as handle:
                    shutil.copyfileobj(extracted, handle)
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError("cannot extract Git source archive") from exc


def _load_verifier():
    module_path = Path(__file__).with_name("verify_distributions.py")
    spec = importlib.util.spec_from_file_location("verify_distributions", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Python distribution verifier")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier


def assert_clean_target_import(wheel: Path, *, work_root: Path) -> None:
    """Import both packages from an isolated wheel target, never from the source tree."""

    with tempfile.TemporaryDirectory(prefix="focus-fabric-import-", dir=work_root) as temporary:
        temporary_root = Path(temporary)
        target = temporary_root / "site"
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--no-index",
                    "--target",
                    str(target),
                    str(wheel),
                ],
                cwd=temporary_root,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import pathlib,sys; "
                        "target=pathlib.Path(sys.argv[1]).resolve(); "
                        "sys.path.insert(0,str(target)); "
                        "import focus_fabric,focus_native; "
                        "paths=[pathlib.Path(focus_fabric.__file__).resolve(),"
                        "pathlib.Path(focus_native.__file__).resolve()]; "
                        "assert all(path.is_relative_to(target) for path in paths),paths"
                    ),
                    str(target),
                ],
                cwd=temporary_root,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("clean-target wheel import failed") from exc


def build_verified_distributions(
    output_dir: Path,
    *,
    replace: bool = False,
) -> dict[str, object]:
    """Build from Git HEAD in an isolated workspace directory and publish after checks."""

    assert_clean_tracked_source()
    commit = source_commit()
    output = validated_output_directory(output_dir)
    if output.exists():
        if not output.is_dir() or output.is_symlink():
            raise RuntimeError("distribution output must be a real directory")
        if any(output.iterdir()) and not replace:
            raise RuntimeError("distribution output is not empty; pass --replace")
    output.parent.mkdir(parents=True, exist_ok=True)

    work_root = ROOT.parent / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    verifier = _load_verifier()
    with tempfile.TemporaryDirectory(prefix="focus-fabric-dist-", dir=work_root) as temporary:
        temporary_root = Path(temporary)
        git_archive = temporary_root / "source.tar"
        source_root = temporary_root / "source"
        staged_dist = temporary_root / "dist"
        try:
            subprocess.run(
                _git_command(
                    "archive",
                    "--format=tar",
                    f"--output={git_archive}",
                    commit,
                ),
                cwd=ROOT.resolve(),
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("cannot archive exact Git HEAD") from exc
        extract_git_archive(git_archive, source_root)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--no-isolation",
                    "--outdir",
                    str(staged_dist),
                    str(source_root),
                ],
                cwd=ROOT.resolve(),
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("Python distribution build failed") from exc

        version = verifier.project_version(source_root)
        verifier.verify_distribution_set(staged_dist, expected_version=version)
        artifacts = [
            staged_dist / f"focus_fabric-{version}-py3-none-any.whl",
            staged_dist / f"focus_fabric-{version}.tar.gz",
        ]
        assert_clean_target_import(artifacts[0], work_root=work_root)
        try:
            subprocess.run(
                [sys.executable, "-m", "twine", "check", *(str(path) for path in artifacts)],
                cwd=ROOT.resolve(),
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("twine verification failed") from exc

        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(staged_dist, output)

    final_report = verifier.verify_distribution_set(output, expected_version=version)
    return {
        "passed": True,
        "source": {"git_commit": commit},
        "clean_target_wheel_import": True,
        **final_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        report = build_verified_distributions(
            args.output_dir,
            replace=args.replace,
        )
    except RuntimeError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
