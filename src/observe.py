from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVENT_LOG = ROOT / "data" / "runs" / "observability" / "events.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_event(kind: str, message: str, details: dict | None = None) -> dict:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": utc_now(),
        "kind": kind,
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
    parser.add_argument("--details", default="{}", help="JSON object with event details.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details = json.loads(args.details)
    if not isinstance(details, dict):
        raise SystemExit("--details must be a JSON object")
    print(json.dumps(append_event(args.kind, args.message, details), indent=2))


if __name__ == "__main__":
    main()

