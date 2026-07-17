#!/usr/bin/env python3
"""Build an auditable, unsigned FOCUS-Fabric research release archive."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.2.1"
RELEASE_NAME = "FOCUS_Fabric_2026-07_0.2.1_release"
RELEASE_DATE = date(2026, 7, 17)
ALLOWED_ORIGIN_URLS = {
    "https://github.com/dj-thank/FOCUS-Fabric.git",
    "git@github.com:dj-thank/FOCUS-Fabric.git",
}
BOUND_RELEASE_TREES = {
    (
        "FOCUS_Fabric_2026-07_public_release",
        "0.2.0",
        "2026-07-14",
    ): "aa96cd5e8c6d730380c425bdc7b43cd751316c90",
}

EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "build",
    "focus_fabric.egg-info",
    "worktrees",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".onnx",
}
GENERATED = {"RELEASE_MANIFEST.json", "SHA256SUMS"}
ReleasePayload = tuple[str, int, bytes]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_python_distributions(dist_dir: Path) -> dict[str, object]:
    """Run the archive verifier and return its hash-bearing report."""

    module_path = Path(__file__).with_name("verify_distributions.py")
    spec = importlib.util.spec_from_file_location("verify_distributions", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Python distribution verifier")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    try:
        return verifier.verify_distribution_set(
            dist_dir,
            expected_version=VERSION,
        )
    except verifier.DistributionVerificationError as exc:
        raise RuntimeError("Python distribution verification failed") from exc


def build_python_distributions(dist_dir: Path) -> dict[str, object]:
    """Rebuild distributions from exact Git HEAD before accepting their hashes."""

    module_path = Path(__file__).with_name("build_distributions.py")
    spec = importlib.util.spec_from_file_location("build_distributions", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Python distribution builder")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    try:
        return builder.build_verified_distributions(dist_dir, replace=True)
    except RuntimeError as exc:
        raise RuntimeError("Python distribution rebuild failed") from exc


def assert_twine_distributions(dist_dir: Path) -> None:
    artifacts = [
        dist_dir / f"focus_fabric-{VERSION}-py3-none-any.whl",
        dist_dir / f"focus_fabric-{VERSION}.tar.gz",
    ]
    try:
        subprocess.run(
            [sys.executable, "-m", "twine", "check", *(str(path) for path in artifacts)],
            cwd=ROOT.resolve(),
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("twine verification failed") from exc


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


def source_commit() -> str:
    """Return the exact release source commit or fail closed."""

    try:
        result = subprocess.run(
            _git_command("rev-parse", "--verify", "HEAD"),
            cwd=ROOT.resolve(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve release source commit") from exc
    commit = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("release source commit is malformed")
    return commit


def assert_final_publication_source() -> None:
    """Require final assets to come from the fetched GitHub main commit."""

    origin_url = _git_text("remote", "get-url", "origin")
    if origin_url not in ALLOWED_ORIGIN_URLS:
        raise RuntimeError("final release origin is not the canonical GitHub repository")
    _git_text("fetch", "--quiet", "origin", "main")
    branch = _git_text("symbolic-ref", "--quiet", "--short", "HEAD")
    if branch != "main":
        raise RuntimeError(
            "final release generation requires the reviewed main branch; "
            "use --candidate for a pre-merge artifact"
        )
    head = source_commit()
    remote_main = _git_text("rev-parse", "--verify", "FETCH_HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", remote_main) is None:
        raise RuntimeError("origin/main commit is malformed")
    if head != remote_main:
        raise RuntimeError(
            "final release HEAD must exactly match fetched origin/main; "
            "fetch and merge the reviewed PR before building"
        )


def source_tree(commit: str | None = None) -> str:
    """Return the committed tree object used to bind a release identity."""

    try:
        result = subprocess.run(
            _git_command("rev-parse", "--verify", f"{commit or 'HEAD'}^{{tree}}"),
            cwd=ROOT.resolve(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve release source tree") from exc
    tree = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", tree) is None:
        raise RuntimeError("release source tree is malformed")
    return tree


def assert_release_identity(tree: str, *, release_name: str | None = None) -> None:
    """Prevent a retained release name/version/date from being rebound."""

    identity = (release_name or RELEASE_NAME, VERSION, str(RELEASE_DATE))
    bound_tree = BOUND_RELEASE_TREES.get(identity)
    if bound_tree is not None and tree != bound_tree:
        raise RuntimeError(
            "release identity is already bound to another source tree; "
            "choose a new release name, version, and date"
        )


def assert_clean_tracked_source() -> None:
    """Bind generated metadata to HEAD rather than uncommitted tracked bytes."""

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
            raise RuntimeError("cannot verify release source cleanliness") from exc
        if result.returncode == 1:
            raise RuntimeError("tracked release source must be clean before generation")
        if result.returncode != 0:
            raise RuntimeError("cannot verify release source cleanliness")


def _tracked_entries() -> list[tuple[int, Path]]:
    """Return tracked index entries, rejecting unsafe or non-regular entries."""
    root = ROOT.resolve()
    command = _git_command("ls-files", "--stage", "-z")
    try:
        result = subprocess.run(
            command, cwd=root, check=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot enumerate tracked release files") from exc

    entries: list[tuple[int, Path]] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode_text, _object_id, stage = header.decode("ascii").split(" ", 2)
            relative_text = raw_path.decode("utf-8")
            relative = Path(relative_text)
            mode = int(mode_text, 8)
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("malformed git index entry") from exc
        if relative.is_absolute() or relative.anchor or ".." in relative.parts:
            raise RuntimeError(f"tracked path escapes release root: {relative_text}")
        if stage != "0":
            raise RuntimeError(f"unmerged tracked release path: {relative_text}")
        if mode not in (0o100644, 0o100755):
            raise RuntimeError(f"non-regular tracked release path: {relative_text}")
        unresolved = root / relative
        current = root
        is_junction = getattr(os.path, "isjunction", lambda _path: False)
        for part in relative.parts:
            current = current / part
            if current.is_symlink() or is_junction(current):
                raise RuntimeError(f"linked tracked release path: {relative_text}")
        path = unresolved.resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"tracked path escapes release root: {relative_text}") from exc
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"tracked release path is missing or unsafe: {relative_text}")
        entries.append((mode, path))
    return entries


def release_files(*, include_generated: bool = False) -> list[Path]:
    files: list[Path] = []
    for _mode, path in _tracked_entries():
        relative = path.relative_to(ROOT.resolve())
        if not _include_release_path(relative, include_generated=include_generated):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def _include_release_path(relative: Path, *, include_generated: bool) -> bool:
    if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts):
        return False
    if relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if not include_generated and relative.as_posix() in GENERATED:
        return False
    return True


def committed_release_payloads(commit: str) -> list[ReleasePayload]:
    """Capture immutable release bytes from one exact Git commit."""

    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("release source commit is malformed")
    work_root = ROOT.parent / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    payloads: list[ReleasePayload] = []
    with tempfile.TemporaryDirectory(prefix="focus-fabric-source-", dir=work_root) as temporary:
        archive_path = Path(temporary) / "source.tar"
        try:
            subprocess.run(
                _git_command(
                    "archive",
                    "--format=tar",
                    f"--output={archive_path}",
                    commit,
                ),
                cwd=ROOT.resolve(),
                check=True,
            )
            with tarfile.open(archive_path, "r:") as archive:
                for member in archive.getmembers():
                    if member.isdir():
                        continue
                    relative_posix = PurePosixPath(member.name)
                    if (
                        not member.name
                        or "\\" in member.name
                        or relative_posix.is_absolute()
                        or any(part in {"", ".", ".."} for part in relative_posix.parts)
                        or not member.isfile()
                    ):
                        raise RuntimeError(
                            f"unsafe committed release member: {member.name!r}"
                        )
                    relative = Path(*relative_posix.parts)
                    if not _include_release_path(relative, include_generated=False):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(
                            f"cannot read committed release member: {member.name}"
                        )
                    mode = 0o100755 if member.mode & 0o111 else 0o100644
                    payloads.append((relative.as_posix(), mode, extracted.read()))
        except (OSError, subprocess.CalledProcessError, tarfile.TarError) as exc:
            raise RuntimeError("cannot capture committed release source") from exc
    return sorted(payloads, key=lambda item: item[0])


def build_manifest(
    *,
    release_name: str | None = None,
    status: str = "unsigned_research_preview",
    distribution_report: dict[str, object] | None = None,
    source_payloads: list[ReleasePayload] | None = None,
    source_commit_value: str | None = None,
    source_tree_value: str | None = None,
) -> dict[str, object]:
    if source_payloads is None:
        claim_ledger = json.loads(
            (ROOT / "docs" / "CLAIMS_LEDGER.json").read_text(encoding="utf-8")
        )
        benchmark = json.loads(
            (ROOT / "results" / "fabric_benchmark.json").read_text(encoding="utf-8")
        )
        files = release_files()
        file_records = [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ]
    else:
        payload_by_path = {path: data for path, _mode, data in source_payloads}
        try:
            claim_ledger = json.loads(
                payload_by_path["docs/CLAIMS_LEDGER.json"].decode("utf-8")
            )
            benchmark = json.loads(
                payload_by_path["results/fabric_benchmark.json"].decode("utf-8")
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("committed release evidence is missing or malformed") from exc
        file_records = [
            {
                "path": path,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for path, _mode, data in source_payloads
        ]
    commit = source_commit_value or source_commit()
    if source_tree_value is not None:
        tree = source_tree_value
    elif source_commit_value is not None:
        tree = source_tree(commit)
    else:
        tree = source_tree()
    selected_release_name = release_name or RELEASE_NAME
    assert_release_identity(tree, release_name=selected_release_name)
    python_distributions: list[dict[str, object]] = []
    python_distribution_source: dict[str, str] | None = None
    if distribution_report is not None:
        distribution_source = distribution_report.get("source", {})
        distribution_commit = str(distribution_source.get("git_commit", ""))
        if distribution_commit != commit:
            raise RuntimeError(
                "Python distributions are not bound to the release source commit"
            )
        python_distribution_source = {"git_commit": distribution_commit}
        for artifact in distribution_report.get("artifacts", []):
            digest = str(artifact["sha256"])
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise RuntimeError("distribution SHA-256 is malformed")
            python_distributions.append({
                "kind": artifact["kind"],
                "filename": Path(str(artifact["path"])).name,
                "name": artifact["name"],
                "version": artifact["version"],
                "members": artifact["members"],
                "bytes": artifact["bytes"],
                "sha256": digest,
            })
    manifest = {
        "schema_version": 1,
        "release": selected_release_name,
        "version": VERSION,
        "date": str(RELEASE_DATE),
        "status": status,
        "source": {"git_commit": commit, "git_tree": tree},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": benchmark["environment"]["torch"],
            "cuda_available": benchmark["environment"]["cuda_available"],
        },
        "local_verification": {
            "compileall": "passed",
            "pytest": "passed",
            "claim_ledger": "passed",
            "drift_audit": "passed",
            "wheel_and_sdist": "passed" if distribution_report is not None else "not_evaluated",
            "twine_check": "passed" if distribution_report is not None else "not_evaluated",
            "clean_target_wheel_import": (
                "passed"
                if distribution_report is not None
                and distribution_report.get("clean_target_wheel_import") is True
                else "not_evaluated"
            ),
        },
        "unexecuted_external_gates": {
            "official_longbench": True,
            "official_ruler": True,
            "official_babilong": True,
            "official_lifebench": True,
            "cuda_triton": True,
            "physical_hbm_counters": True,
            "codex_execute_mode": True,
            "external_signature": True,
            "human_publication_authorization": True,
        },
        "claim_count": len(claim_ledger.get("claims", [])),
        "artifacts": {
            "python_distribution_source": python_distribution_source,
            "python_distributions": python_distributions,
        },
        "evidence": {
            "controlled_benchmark": "results/fabric_benchmark.json",
            "randomized_holdout": "results/randomized_holdout_suite.json",
            "agent_memory": "results/agent_memory_benchmark.json",
            "codex_dry_run": "results/autonomy_dry_run.json",
            "gpu_status": "results/gpu_benchmark.json",
        },
        "files": file_records,
    }
    return manifest


def assert_generated_metadata_path(path: Path) -> None:
    expected = ROOT.resolve() / path.name
    is_junction = getattr(os.path, "isjunction", lambda _path: False)
    if (
        path.name not in GENERATED
        or path.absolute() != expected
        or path.is_symlink()
        or is_junction(path)
        or (path.exists() and not path.is_file())
    ):
        raise RuntimeError(f"unsafe generated metadata path: {path}")


def write_metadata(
    *,
    release_name: str | None = None,
    status: str = "unsigned_research_preview",
    dist_dir: Path | None = None,
) -> list[ReleasePayload]:
    assert_clean_tracked_source()
    commit = source_commit()
    tree = source_tree(commit)
    source_payloads = committed_release_payloads(commit)
    selected_dist_dir = (dist_dir or ROOT / "dist").resolve()
    build_report = build_python_distributions(selected_dist_dir)
    distribution_report = verify_python_distributions(selected_dist_dir)
    distribution_report["source"] = build_report["source"]
    distribution_report["clean_target_wheel_import"] = build_report[
        "clean_target_wheel_import"
    ]
    assert_twine_distributions(selected_dist_dir)
    manifest = build_manifest(
        release_name=release_name,
        status=status,
        distribution_report=distribution_report,
        source_payloads=source_payloads,
        source_commit_value=commit,
        source_tree_value=tree,
    )
    manifest_path = ROOT / "RELEASE_MANIFEST.json"
    assert_generated_metadata_path(manifest_path)
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_payload: ReleasePayload = (
        "RELEASE_MANIFEST.json",
        0o100644,
        manifest_bytes,
    )
    checksum_path = ROOT / "SHA256SUMS"
    assert_generated_metadata_path(checksum_path)
    lines = [
        f"{hashlib.sha256(data).hexdigest()}  {path}"
        for path, _mode, data in [*source_payloads, manifest_payload]
    ]
    checksum_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    checksum_path.write_bytes(checksum_bytes)
    checksum_payload: ReleasePayload = ("SHA256SUMS", 0o100644, checksum_bytes)
    return [*source_payloads, manifest_payload, checksum_payload]


def verify_zip_archive(
    output: Path,
    *,
    release_name: str,
    expected_members: set[str] | None = None,
) -> None:
    """Inspect the generated source ZIP rather than trusting construction alone."""

    try:
        with zipfile.ZipFile(output) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("release ZIP CRC verification failed")
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise RuntimeError("duplicate release ZIP member")
            prefix = f"{release_name}/"
            for member in members:
                name = member.filename
                if (
                    not name.startswith(prefix)
                    or "\\" in name
                    or "\0" in name
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                ):
                    raise RuntimeError(f"unsafe release ZIP path: {name!r}")
                if Path(name).suffix.lower() in EXCLUDED_SUFFIXES:
                    raise RuntimeError(f"forbidden release payload: {name}")
                mode = (member.external_attr >> 16) & 0xFFFF
                if mode and stat.S_ISLNK(mode):
                    raise RuntimeError(f"linked release ZIP member: {name}")
            if expected_members is not None and set(names) != expected_members:
                raise RuntimeError("release ZIP member set does not match tracked source")
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError("cannot verify generated release ZIP") from exc


def validated_release_output(output: Path) -> Path:
    """Restrict replaceable ZIP outputs to a dedicated outputs directory."""

    outputs_root_path = ROOT.parent / "outputs"
    if outputs_root_path.is_symlink():
        raise RuntimeError("dedicated outputs directory cannot be a symlink")
    outputs_root = outputs_root_path.resolve(strict=False)
    requested = output.absolute()
    if requested.is_symlink():
        raise RuntimeError("release output cannot be a symlink")
    resolved = requested.parent.resolve(strict=False) / requested.name
    try:
        relative = resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise RuntimeError("release output must be inside the dedicated outputs directory") from exc
    if not relative.parts or resolved.suffix.lower() != ".zip":
        raise RuntimeError("release output must be a ZIP inside the dedicated outputs directory")
    if resolved.exists() and not resolved.is_file():
        raise RuntimeError("release output must be a regular file")
    return resolved


def build_zip(
    output: Path,
    *,
    release_name: str | None = None,
    payloads: list[ReleasePayload] | None = None,
) -> Path:
    output = validated_release_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    prefix = release_name or RELEASE_NAME
    if payloads is None:
        selected_payloads: list[ReleasePayload] = []
        for mode, path in _tracked_entries():
            relative = path.relative_to(ROOT.resolve())
            if _include_release_path(relative, include_generated=True):
                selected_payloads.append((relative.as_posix(), mode, path.read_bytes()))
    else:
        selected_payloads = list(payloads)
    selected_payloads.sort(key=lambda item: item[0])
    expected_members = {f"{prefix}/{path}" for path, _mode, _data in selected_payloads}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, mode, data in selected_payloads:
            info = zipfile.ZipInfo(
                f"{prefix}/{path}",
                date_time=(RELEASE_DATE.year, RELEASE_DATE.month, RELEASE_DATE.day, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = mode << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    verify_zip_archive(
        output,
        release_name=prefix,
        expected_members=expected_members,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.candidate:
        release_name = f"{RELEASE_NAME}_candidate"
        release_status = "unsigned_research_preview_candidate"
    else:
        assert_final_publication_source()
        release_name = RELEASE_NAME
        release_status = "unsigned_research_preview"
    output = args.output or (
        ROOT.parent / "outputs" / release_name / f"{release_name}.zip"
    )
    payloads = write_metadata(
        release_name=release_name,
        status=release_status,
        dist_dir=args.dist_dir,
    )
    output = build_zip(output, release_name=release_name, payloads=payloads)
    archive_digest = sha256(output)
    digest_path = output.with_suffix(output.suffix + ".sha256")
    if digest_path.is_symlink() or (digest_path.exists() and not digest_path.is_file()):
        raise RuntimeError("release checksum sidecar must be a regular file")
    digest_path.write_text(f"{archive_digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({
        "release": release_name,
        "status": release_status,
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": archive_digest,
        "sha256_file": str(digest_path),
        "manifest": str(ROOT / "RELEASE_MANIFEST.json"),
        "checksums": str(ROOT / "SHA256SUMS"),
    }, indent=2))


if __name__ == "__main__":
    main()
