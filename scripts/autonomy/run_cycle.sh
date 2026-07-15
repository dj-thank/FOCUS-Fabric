#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
MODE="${1:-dry-run}"
shift || true
exec python scripts/autonomy/run_codex_loop.py --mode "$MODE" "$@"
