from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "runs"
EVENT_LOG = ROOT / "data" / "runs" / "observability" / "events.jsonl"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


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


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if limit is not None and len(rows) >= limit:
                    break
    except OSError:
        return []
    return rows


def phase1_run_summaries() -> list[dict]:
    summaries: list[dict] = []
    if not RUNS_DIR.exists():
        return summaries
    for run_dir in sorted(RUNS_DIR.glob("*/phase1_v0")):
        if not run_dir.is_dir():
            continue
        manifest = read_json(run_dir / "source_manifest.json")
        inventory = read_jsonl(run_dir / "page_inventory.jsonl")
        canonical = read_json(run_dir / "canonical_reading_order.json")
        status_counts: dict[str, int] = {}
        flag_counts: dict[str, int] = {}
        for row in inventory:
            status = str(row.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            for flag in row.get("review_flags", []):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        summaries.append(
            {
                "book_id": manifest.get("book_id") or run_dir.parent.name,
                "run_id": manifest.get("run_id") or run_dir.name,
                "created_at": manifest.get("created_at", ""),
                "page_count": manifest.get("page_count", len(inventory)),
                "object_count": canonical.get("object_count", 0),
                "status_counts": status_counts,
                "flag_counts": flag_counts,
                "review_flags": canonical.get("review_flags", []),
                "audit_url": f"/runs/{run_dir.parent.name}/{run_dir.name}/phase1_audit.html",
                "manifest_url": f"/runs/{run_dir.parent.name}/{run_dir.name}/source_manifest.json",
            }
        )
    return summaries


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
    .panel { border-top: 1px solid #333; padding-top: 20px; margin-top: 24px; }
    .run { border: 1px solid #333; background: #080808; padding: 16px; margin: 12px 0; }
    .run h3 { margin: 0 0 10px; font-size: 18px; }
    .run-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .run-stat { border-top: 1px solid #252525; padding-top: 8px; }
    .run-stat span { display: block; color: #aaa; font-size: 11px; text-transform: uppercase; }
    .run-stat strong { display: block; margin-top: 4px; }
    .links { margin-top: 12px; display: flex; gap: 14px; flex-wrap: wrap; }
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
      .run-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
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
    <div class="metric"><span>Phase 1 runs</span><strong id="phase1Runs">0</strong></div>
  </section>

  <section class="panel" aria-label="Phase 1 runs">
    <h2>Phase 1 Runs</h2>
    <div id="runs"></div>
  </section>

  <section class="panel" aria-label="Process events">
    <h2>Process Events</h2>
    <div class="events" id="events" aria-label="Event stream"></div>
  </section>
</main>
<script>
  const eventsEl = document.getElementById("events");
  const runsEl = document.getElementById("runs");
  const connectionEl = document.getElementById("connection");
  const totalEl = document.getElementById("total");
  const latestKindEl = document.getElementById("latestKind");
  const lastUpdateEl = document.getElementById("lastUpdate");
  const phase1RunsEl = document.getElementById("phase1Runs");

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

  function countMap(value) {
    if (!value || Object.keys(value).length === 0) return "-";
    return Object.entries(value).map(([key, count]) => `${escapeHtml(key)}: ${escapeHtml(count)}`).join("<br>");
  }

  function renderRuns(runs) {
    phase1RunsEl.textContent = runs.length;
    if (!runs.length) {
      runsEl.innerHTML = "<p>No Phase 1 runs detected yet.</p>";
      return;
    }
    runsEl.innerHTML = runs.slice().reverse().map(run => `
      <article class="run">
        <h3>${escapeHtml(run.book_id)} / ${escapeHtml(run.run_id)}</h3>
        <div class="run-grid">
          <div class="run-stat"><span>Created</span><strong>${escapeHtml(run.created_at || "-")}</strong></div>
          <div class="run-stat"><span>Pages</span><strong>${escapeHtml(run.page_count)}</strong></div>
          <div class="run-stat"><span>Objects</span><strong>${escapeHtml(run.object_count)}</strong></div>
          <div class="run-stat"><span>Review flags</span><strong>${escapeHtml((run.review_flags || []).length)}</strong></div>
          <div class="run-stat"><span>Page status</span><strong>${countMap(run.status_counts)}</strong></div>
          <div class="run-stat"><span>Flag counts</span><strong>${countMap(run.flag_counts)}</strong></div>
        </div>
        <div class="links">
          <a href="${escapeHtml(run.audit_url)}">Open audit</a>
          <a href="${escapeHtml(run.manifest_url)}">Manifest JSON</a>
        </div>
      </article>
    `).join("");
  }

  async function refresh() {
    try {
      const [eventsResponse, runsResponse] = await Promise.all([
        fetch("/api/events", { cache: "no-store" }),
        fetch("/api/phase1-runs", { cache: "no-store" })
      ]);
      const events = await eventsResponse.json();
      const runs = await runsResponse.json();
      connectionEl.textContent = "LIVE";
      totalEl.textContent = events.length;
      renderRuns(runs);
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
        if parsed.path == "/api/phase1-runs":
            payload = json.dumps(phase1_run_summaries(), ensure_ascii=True).encode("utf-8")
            self.respond(HTTPStatus.OK, "application/json; charset=utf-8", payload)
            return
        if parsed.path.startswith("/runs/"):
            self.serve_run_artifact(parsed.path)
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

    def serve_run_artifact(self, request_path: str) -> None:
        parts = [part for part in request_path.split("/") if part]
        if len(parts) != 4 or parts[0] != "runs":
            self.respond(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found")
            return
        _, book_id, run_id, filename = parts
        allowed = {"phase1_audit.html", "source_manifest.json"}
        if filename not in allowed:
            self.respond(HTTPStatus.FORBIDDEN, "text/plain; charset=utf-8", b"Forbidden")
            return
        candidate = (RUNS_DIR / book_id / run_id / filename).resolve()
        runs_root = RUNS_DIR.resolve()
        if not str(candidate).startswith(str(runs_root)) or not candidate.exists():
            self.respond(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found")
            return
        content_type = "text/html; charset=utf-8" if filename.endswith(".html") else "application/json; charset=utf-8"
        self.respond(HTTPStatus.OK, content_type, candidate.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local iknow v1 observability dashboard.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_event_log()
    server = ReusableThreadingHTTPServer((args.host, args.port), ObservabilityHandler)
    append_event("system", "Observability server started", {"host": args.host, "port": args.port})
    print(f"Serving iknow v1 observability at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
