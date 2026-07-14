#!/usr/bin/env python3
"""Verify that every quantitative public claim resolves to an immutable artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "CLAIMS_LEDGER.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def resolve(payload: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError("metric_pointer must be a JSON Pointer")
    current = payload
    for raw in pointer.lstrip("/").split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        current = current[int(key)] if isinstance(current, list) else current[key]
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-digests", action="store_true")
    args = parser.parse_args()
    if not LEDGER.exists():
        print("claim ledger missing", file=sys.stderr)
        raise SystemExit(1)
    document = json.loads(LEDGER.read_text(encoding="utf-8"))
    failures: list[str] = []
    changed = False
    for claim in document.get("claims", []):
        artifact = ROOT / claim["artifact"]
        if not artifact.exists():
            failures.append(f"{claim['id']}: artifact missing: {artifact}")
            continue
        actual_digest = digest(artifact)
        if args.refresh_digests:
            claim["sha256"] = actual_digest
            changed = True
        elif claim.get("sha256") != actual_digest:
            failures.append(f"{claim['id']}: SHA-256 mismatch")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        try:
            value = resolve(payload, claim["metric_pointer"])
        except (KeyError, IndexError, ValueError, TypeError) as error:
            failures.append(f"{claim['id']}: metric pointer invalid: {error}")
            continue
        expected = claim["value"]
        tolerance = float(claim.get("tolerance", 0.0))
        if isinstance(expected, (int, float)) and isinstance(value, (int, float)):
            if abs(float(value) - float(expected)) > tolerance:
                failures.append(
                    f"{claim['id']}: expected {expected} +/- {tolerance}, found {value}"
                )
        elif value != expected:
            failures.append(f"{claim['id']}: expected {expected!r}, found {value!r}")
        if not claim.get("allowed_wording"):
            failures.append(f"{claim['id']}: no allowed wording")
    if changed:
        LEDGER.write_text(json.dumps(document, indent=2), encoding="utf-8")
    report = {
        "claims": len(document.get("claims", [])),
        "failures": failures,
        "passed": not failures,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
