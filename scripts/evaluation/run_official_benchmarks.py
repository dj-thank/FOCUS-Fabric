#!/usr/bin/env python3
"""Backend-neutral exact-vs-FOCUS runner for LongBench/RULER/BABILong.

The script intentionally does not download datasets or pretend that generic
exact-match is each suite's official aggregate score.  It emits per-example
predictions, token agreement, latency and cache telemetry so the suite's own
scorer can be run on the same outputs.

Backend contract: the configured command receives one JSON object on stdin and
must return one JSON object on stdout with at least ``text``.  Recommended
fields are ``token_ids``, ``prefill_seconds``, ``decode_seconds``,
``fallback_rate``, ``active_bytes``, ``exact_kv_bytes`` and
``archive_bytes_read``.  The request contains ``mode`` (``exact`` or
``focus``), ``prompt``, and ``max_new_tokens``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def token_f1(prediction: str, references: list[str]) -> float:
    predicted = normalized(prediction).split()
    best = 0.0
    for reference in references:
        target = normalized(reference).split()
        overlap = sum((Counter(predicted) & Counter(target)).values())
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(target) if target else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best = max(best, score)
    return best


def exact_match(prediction: str, references: list[str]) -> float:
    value = normalized(prediction)
    return float(any(value == normalized(reference) for reference in references))


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        for key in ("data", "examples", "samples"):
            if isinstance(payload.get(key), list):
                return payload[key]
        raise ValueError("JSON benchmark file contains no recognized sample list")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_from_row(row: dict[str, Any], suite: str, index: int) -> dict[str, Any]:
    prompt = row.get("prompt") or row.get("input") or row.get("text")
    if suite == "longbench" and not prompt:
        context = row.get("context", "")
        question = row.get("question", "")
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    if not isinstance(prompt, str):
        raise ValueError(f"sample {index} has no textual prompt")
    raw_answers = (
        row.get("answers")
        or row.get("outputs")
        or row.get("target")
        or row.get("answer")
        or []
    )
    if isinstance(raw_answers, str):
        answers = [raw_answers]
    elif isinstance(raw_answers, list):
        answers = [str(item) for item in raw_answers]
    else:
        answers = [str(raw_answers)]
    identifier = str(row.get("id") or row.get("_id") or f"{suite}-{index:06d}")
    return {
        "id": identifier,
        "prompt": prompt,
        "answers": answers,
        "task": row.get("dataset") or row.get("task") or suite,
        "metadata": {
            key: row[key]
            for key in ("length", "all_classes", "category")
            if key in row
        },
    }


def invoke(command: list[str], request: dict[str, Any], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        input=json.dumps(request),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"backend failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    response = json.loads(completed.stdout)
    if not isinstance(response.get("text"), str):
        raise ValueError("backend response must contain string field 'text'")
    response.setdefault("wall_seconds", time.perf_counter() - started)
    return response


def sequence_agreement(left: Any, right: Any) -> tuple[float | None, bool | None]:
    if not isinstance(left, list) or not isinstance(right, list):
        return None, None
    length = max(len(left), len(right))
    if length == 0:
        return 1.0, True
    matches = sum(a == b for a, b in zip(left, right))
    return matches / length, left == right


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("longbench", "ruler", "babilong"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--backend-command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    command = shlex.split(args.backend_command)
    if not command:
        raise ValueError("backend command is empty")
    rows = load_rows(args.data)
    if args.limit > 0:
        rows = rows[: args.limit]
    samples = [sample_from_row(row, args.suite, index) for index, row in enumerate(rows)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.output.with_suffix(".jsonl")
    details: list[dict[str, Any]] = []
    with detail_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            base_request = {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "max_new_tokens": args.max_new_tokens,
            }
            exact = invoke(command, {**base_request, "mode": "exact"}, args.timeout)
            focus = invoke(command, {**base_request, "mode": "focus"}, args.timeout)
            token_rate, sequence_equal = sequence_agreement(
                exact.get("token_ids"), focus.get("token_ids")
            )
            detail = {
                **sample,
                "exact": exact,
                "focus": focus,
                "metrics": {
                    "exact_match_exact": exact_match(exact["text"], sample["answers"]),
                    "exact_match_focus": exact_match(focus["text"], sample["answers"]),
                    "token_f1_exact": token_f1(exact["text"], sample["answers"]),
                    "token_f1_focus": token_f1(focus["text"], sample["answers"]),
                    "focus_vs_exact_token_agreement": token_rate,
                    "focus_vs_exact_sequence_agreement": sequence_equal,
                },
            }
            details.append(detail)
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    aggregate = {
        "suite": args.suite,
        "samples": len(details),
        "data": str(args.data),
        "detail_jsonl": str(detail_path),
        "proxy_metrics_not_official_aggregate": {
            "exact_match_exact": mean(item["metrics"]["exact_match_exact"] for item in details),
            "exact_match_focus": mean(item["metrics"]["exact_match_focus"] for item in details),
            "token_f1_exact": mean(item["metrics"]["token_f1_exact"] for item in details),
            "token_f1_focus": mean(item["metrics"]["token_f1_focus"] for item in details),
        },
        "focus_vs_exact": {
            "token_agreement": mean(
                item["metrics"]["focus_vs_exact_token_agreement"]
                for item in details
                if item["metrics"]["focus_vs_exact_token_agreement"] is not None
            ),
            "sequence_agreement": mean(
                float(item["metrics"]["focus_vs_exact_sequence_agreement"])
                for item in details
                if item["metrics"]["focus_vs_exact_sequence_agreement"] is not None
            ),
            "mean_fallback_rate": mean(
                float(item["focus"].get("fallback_rate", 0.0)) for item in details
            ),
            "mean_archive_bytes_read": mean(
                float(item["focus"].get("archive_bytes_read", 0.0)) for item in details
            ),
        },
        "official_scoring_status": (
            "Predictions are emitted for the suite's official scorer; the generic "
            "EM/F1 values above are diagnostics, not an official leaderboard score."
        ),
    }
    args.output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
