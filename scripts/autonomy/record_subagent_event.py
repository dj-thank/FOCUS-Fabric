#!/usr/bin/env python3
"""Record a minimal, private Codex subagent lifecycle event."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


def record_event(payload: dict[str, Any], output_dir: Path) -> Path:
    event = str(payload.get("hook_event_name", ""))
    if event not in {"SubagentStart", "SubagentStop"}:
        raise ValueError(f"unsupported hook event: {event}")
    agent_id = str(payload["agent_id"])
    agent_type = str(payload["agent_type"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", agent_id):
        raise ValueError("invalid agent_id")
    if not re.fullmatch(r"[a-z0-9_]+", agent_type):
        raise ValueError("invalid agent_type")
    destination = output_dir.resolve() / f"{event}-{agent_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "hook_event_name": event,
        "agent_id": agent_id,
        "agent_type": agent_type,
        "model": str(payload.get("model", "")),
        "permission_mode": str(payload.get("permission_mode", "")),
        "session_id": str(payload.get("session_id", "")),
        "turn_id": str(payload.get("turn_id", "")),
    }
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("hook input must be a JSON object")
    record_event(payload, args.output_dir)


if __name__ == "__main__":
    main()
