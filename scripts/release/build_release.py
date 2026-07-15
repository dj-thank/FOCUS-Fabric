#!/usr/bin/env python3
"""Build an auditable, unsigned FOCUS-Fabric research release archive."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import platform
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.2.0"
RELEASE_NAME = "FOCUS_Fabric_2026-07_public_release"

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


def release_files(*, include_generated: bool = False) -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
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
    return {
        "schema_version": 1,
        "release": RELEASE_NAME,
        "version": VERSION,
        "date": str(date(2026, 7, 14)),
        "status": "unsigned_research_preview",
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


def write_metadata() -> None:
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
