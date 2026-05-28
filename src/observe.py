from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVENT_LOG = ROOT / "data" / "runs" / "observability" / "events.jsonl"

SCHEMA = "iknow.observe/2"
LEVELS = ("debug", "info", "milestone", "warning", "error")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str | None:
    """Short git commit for run correlation; ``None`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def append_event(
    kind: str,
    message: str,
    details: dict | None = None,
    *,
    level: str = "info",
    actor: str = "claude",
    book_id: str | None = None,
    run_id: str | None = None,
    phase: str | None = None,
    commit: str | None = None,
) -> dict:
    """Append a structured observability event.

    Backward compatible: ``append_event(kind, message, details)`` still works and
    still produces an event readable by the legacy dashboard (it carries the
    original ``timestamp``/``kind``/``message``/``details`` keys). The optional
    keyword fields add severity and run correlation for the richer Claude-lane
    dashboard.
    """
    if level not in LEVELS:
        level = "info"
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "schema": SCHEMA,
        "timestamp": utc_now(),
        "level": level,
        "actor": actor,
        "kind": kind,
        "book_id": book_id,
        "run_id": run_id,
        "phase": phase,
        "git_commit": commit if commit is not None else git_commit(),
        "message": message,
        "details": details or {},
    }
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")
    return event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append an iknow v1 observability event.")
    parser.add_argument("kind")
    parser.add_argument("message")
    parser.add_argument("--level", default="info", choices=LEVELS)
    parser.add_argument("--actor", default="claude")
    parser.add_argument("--book", dest="book_id", default=None)
    parser.add_argument("--run", dest="run_id", default=None)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--details", default="{}", help="JSON object with event details.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details = json.loads(args.details)
    if not isinstance(details, dict):
        raise SystemExit("--details must be a JSON object")
    event = append_event(
        args.kind,
        args.message,
        details,
        level=args.level,
        actor=args.actor,
        book_id=args.book_id,
        run_id=args.run_id,
        phase=args.phase,
    )
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
