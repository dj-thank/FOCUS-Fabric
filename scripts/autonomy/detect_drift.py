#!/usr/bin/env python3
"""Detect evidence, repository, and scientific-claim drift."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    failures: list[str] = []
    benchmark_path = ROOT / "results/fabric_benchmark.json"
    if not benchmark_path.exists():
        failures.append("baseline benchmark is missing")
    else:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        cuda = bool(benchmark["environment"]["cuda_available"])
        gpu = benchmark["gpu"]
        if not cuda and any(
            gpu.get(key) is not None
            for key in ("kernel_correctness", "latency", "physical_hbm_bandwidth")
        ):
            failures.append("GPU metrics are non-null in a CPU-only artifact")
        if benchmark["official_benchmarks"]["status"].startswith("not executed"):
            for key in ("LongBench", "RULER", "BABILong", "LifeBench"):
                if benchmark["official_benchmarks"].get(key) is not None:
                    failures.append(f"official score {key} is non-null despite not-executed status")
        if benchmark["end_to_end"]["teacher_forced"]["argmax_token_agreement"] < 0.99:
            failures.append("baseline end-to-end token agreement fell below 0.99")
        if benchmark["repeated_compaction"]["final"]["invalid_codec_outputs"]:
            failures.append("baseline contains invalid codec outputs")

    # Strong public superlatives are forbidden unless a dedicated claim record
    # names an external comparison and its artifact.  This catches accidental
    # drift introduced by autonomous prose generation.
    forbidden = re.compile(
        r"\b(SOTA|state[- ]of[- ]the[- ]art|best[- ]in[- ]class|universally superior)\b",
        re.IGNORECASE,
    )
    for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line) and "NOT CLAIMED" not in line:
                failures.append(f"unsupported superlative: {path.relative_to(ROOT)}:{line_number}")
    print(json.dumps({"passed": not failures, "failures": failures}, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
