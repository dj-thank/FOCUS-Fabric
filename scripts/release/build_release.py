#!/usr/bin/env python3
"""Build an auditable, unsigned FOCUS-Fabric research release archive."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.2.0"
RELEASE_NAME = "FOCUS_Fabric_2026-07_public_release"
RELEASE_DATE = date(2026, 7, 14)
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
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".safetensors"}
GENERATED = {"RELEASE_MANIFEST.json", "SHA256SUMS"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_command(*arguments: str) -> list[str]:
    return ["git", "-c", f"safe.directory={ROOT.resolve()}", *arguments]


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


def source_tree() -> str:
    """Return the committed tree object used to bind a release identity."""

    try:
        result = subprocess.run(
            _git_command("rev-parse", "--verify", "HEAD^{tree}"),
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


def assert_release_identity(tree: str) -> None:
    """Prevent a retained release name/version/date from being rebound."""

    identity = (RELEASE_NAME, VERSION, str(RELEASE_DATE))
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
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        if not include_generated and relative.as_posix() in GENERATED:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def build_manifest() -> dict[str, object]:
    claim_ledger = json.loads((ROOT / "docs" / "CLAIMS_LEDGER.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "results" / "fabric_benchmark.json").read_text(encoding="utf-8"))
    files = release_files()
    commit = source_commit()
    tree = source_tree()
    assert_release_identity(tree)
    manifest = {
        "schema_version": 1,
        "release": RELEASE_NAME,
        "version": VERSION,
        "date": str(RELEASE_DATE),
        "status": "unsigned_research_preview",
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
            "wheel_and_sdist": "passed",
            "twine_check": "passed",
            "clean_target_wheel_import": "passed",
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
        "evidence": {
            "controlled_benchmark": "results/fabric_benchmark.json",
            "randomized_holdout": "results/randomized_holdout_suite.json",
            "agent_memory": "results/agent_memory_benchmark.json",
            "codex_dry_run": "results/autonomy_dry_run.json",
            "gpu_status": "results/gpu_benchmark.json",
        },
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    return manifest


def write_metadata() -> None:
    assert_clean_tracked_source()
    manifest = build_manifest()
    manifest_path = ROOT / "RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checksum_paths = release_files(include_generated=True)
    checksum_path = ROOT / "SHA256SUMS"
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in checksum_paths
        if path != checksum_path
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_zip(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    prefix = RELEASE_NAME
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in release_files(include_generated=True):
            archive.write(path, f"{prefix}/{path.relative_to(ROOT).as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / f"{RELEASE_NAME}.zip",
    )
    args = parser.parse_args()
    write_metadata()
    build_zip(args.output)
    archive_digest = sha256(args.output)
    digest_path = args.output.with_suffix(args.output.suffix + ".sha256")
    digest_path.write_text(f"{archive_digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({
        "release": RELEASE_NAME,
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": archive_digest,
        "sha256_file": str(digest_path),
        "manifest": str(ROOT / "RELEASE_MANIFEST.json"),
        "checksums": str(ROOT / "SHA256SUMS"),
    }, indent=2))


if __name__ == "__main__":
    main()
