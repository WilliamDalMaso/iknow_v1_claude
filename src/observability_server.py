from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
EVENT_LOG = ROOT / "data" / "runs" / "observability" / "events.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_event_log() -> None:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not EVENT_LOG.exists():
        append_event("system", "Observability log initialized", {"status": "active"})


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


def read_events(limit: int = 200) -> list[dict]:
    ensure_event_log()
    rows = []
    with EVENT_LOG.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(
                    {
                        "timestamp": utc_now(),
                        "kind": "log_error",
                        "message": "Unreadable event row",
                        "details": {"raw": line},
                    }
                )
    return rows[-limit:]


def dashboard_html() -> bytes:
    return b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>iknow v1 Observability</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #000; color: #fff; }
    main { max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }
    header { display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; border-bottom: 1px solid #333; padding-bottom: 20px; }
    h1 { font-size: 28px; margin: 0 0 8px; letter-spacing: 0; }
    p { color: #cfcfcf; margin: 0; }
    .status { border: 1px solid #fff; padding: 10px 12px; min-width: 180px; text-align: center; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 22px 0; }
    .metric { border: 1px solid #333; padding: 14px; background: #080808; min-height: 86px; }
    .metric span { display: block; color: #aaa; font-size: 12px; text-transform: uppercase; }
    .metric strong { display: block; font-size: 24px; margin-top: 8px; }
    .events { border-top: 1px solid #333; }
    .event { display: grid; grid-template-columns: 180px 140px 1fr; gap: 14px; padding: 14px 0; border-bottom: 1px solid #1f1f1f; }
    .time, .kind, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .time { color: #aaa; font-size: 12px; }
    .kind { color: #fff; font-size: 12px; text-transform: uppercase; }
    .message { font-size: 15px; }
    pre { white-space: pre-wrap; margin: 8px 0 0; color: #bdbdbd; font-size: 12px; }
    a { color: #fff; }
    @media (max-width: 780px) {
      header { display: block; }
      .status { margin-top: 16px; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .event { grid-template-columns: 1fr; gap: 6px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <section>
      <h1>iknow v1 Observability</h1>
      <p>Local build/process monitor. Black and white by design. Auto-refreshes every two seconds.</p>
    </section>
    <div class="status" id="connection">CONNECTING</div>
  </header>

  <section class="grid" aria-label="Metrics">
    <div class="metric"><span>Total events</span><strong id="total">0</strong></div>
    <div class="metric"><span>Latest kind</span><strong id="latestKind">-</strong></div>
    <div class="metric"><span>Last update</span><strong id="lastUpdate">-</strong></div>
    <div class="metric"><span>Phase focus</span><strong>Phase 1</strong></div>
  </section>

  <section class="events" id="events" aria-label="Event stream"></section>
</main>
<script>
  const eventsEl = document.getElementById("events");
  const connectionEl = document.getElementById("connection");
  const totalEl = document.getElementById("total");
  const latestKindEl = document.getElementById("latestKind");
  const lastUpdateEl = document.getElementById("lastUpdate");

  function renderDetails(details) {
    if (!details || Object.keys(details).length === 0) return "";
    return `<pre>${escapeHtml(JSON.stringify(details, null, 2))}</pre>`;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  async function refresh() {
    try {
      const response = await fetch("/api/events", { cache: "no-store" });
      const events = await response.json();
      connectionEl.textContent = "LIVE";
      totalEl.textContent = events.length;
      const latest = events[events.length - 1];
      latestKindEl.textContent = latest ? latest.kind : "-";
      lastUpdateEl.textContent = latest ? latest.timestamp.split("T")[1].replace("Z", "") : "-";
      eventsEl.innerHTML = events.slice().reverse().map(event => `
        <article class="event">
          <div class="time">${escapeHtml(event.timestamp)}</div>
          <div class="kind">${escapeHtml(event.kind)}</div>
          <div>
            <div class="message">${escapeHtml(event.message)}</div>
            ${renderDetails(event.details)}
          </div>
        </article>
      `).join("");
    } catch (error) {
      connectionEl.textContent = "OFFLINE";
    }
  }

  refresh();
  setInterval(refresh, 2000);
</script>
</body>
</html>"""


class ObservabilityHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.respond(HTTPStatus.OK, "text/html; charset=utf-8", dashboard_html())
            return
        if parsed.path == "/api/events":
            payload = json.dumps(read_events(), ensure_ascii=True).encode("utf-8")
            self.respond(HTTPStatus.OK, "application/json; charset=utf-8", payload)
            return
        self.respond(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/events":
            self.respond(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
            event = append_event(
                str(payload.get("kind", "note")),
                str(payload.get("message", "")),
                payload.get("details") if isinstance(payload.get("details"), dict) else {},
            )
            self.respond(HTTPStatus.CREATED, "application/json; charset=utf-8", json.dumps(event).encode("utf-8"))
        except json.JSONDecodeError:
            self.respond(HTTPStatus.BAD_REQUEST, "application/json; charset=utf-8", b'{"error":"invalid json"}')

    def log_message(self, format: str, *args: object) -> None:
        return

    def respond(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local iknow v1 observability dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_event_log()
    server = ThreadingHTTPServer((args.host, args.port), ObservabilityHandler)
    append_event("system", "Observability server started", {"host": args.host, "port": args.port})
    print(f"Serving iknow v1 observability at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
