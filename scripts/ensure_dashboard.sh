#!/usr/bin/env bash
# Ensure the Claude-lane observability dashboard is running on 127.0.0.1:8799.
# Idempotent: if the port is already listening, do nothing (never spawns a 2nd
# server). Otherwise start it detached so it survives the calling process.
# Reads/ignores any JSON on stdin (so it works as a Stop/SessionStart hook).
set -u
ROOT="/Users/williamdalmaso/Desktop/iknow_v1_claude"
PORT=8799
LOG="$ROOT/data/runs/observability/dashboard.log"

# Already up? nothing to do.
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  exit 0
fi

mkdir -p "$ROOT/data/runs/observability"
cd "$ROOT" || exit 0
nohup python3 src/claude_observability_server.py --host 127.0.0.1 --port "$PORT" >>"$LOG" 2>&1 &
disown 2>/dev/null || true
exit 0
