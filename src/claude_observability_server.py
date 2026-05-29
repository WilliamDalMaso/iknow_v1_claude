"""Claude-lane observability dashboard.

A thorough, friendly local monitor for iknow_v1_claude. Runs on the Claude-lane
port 8799 so it never collides with the legacy/Codex dashboard on 8765; both can
run at once. Dependency-free (standard library only).

Improvements over the legacy dashboard:
  - severity levels (debug/info/milestone/warning/error) with color accents;
  - run correlation (book_id / run_id / phase / git_commit) per event;
  - client-side search and faceted filters (level, actor, book, phase);
  - optional grouping of events by run;
  - aggregate summary (issues, distinct runs/books/actors);
  - pause + keyboard shortcuts for friendlier inspection;
  - graceful handling of legacy flat events.

CLI:
    python3 src/claude_observability_server.py --host 127.0.0.1 --port 8799
"""
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
EVENT_LOG = RUNS_DIR / "observability" / "events.jsonl"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8799  # Claude-lane port; 8765 belongs to the legacy/Codex dashboard.
LEVELS = ("debug", "info", "milestone", "warning", "error")


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_event(raw: dict) -> dict:
    """Fill defaults so legacy flat events and new structured events render uniformly."""
    level = raw.get("level", "info")
    if level not in LEVELS:
        level = "info"
    details = raw.get("details")
    return {
        "schema": raw.get("schema", "legacy"),
        "timestamp": raw.get("timestamp", ""),
        "level": level,
        "actor": raw.get("actor") or "unknown",
        "kind": raw.get("kind") or "note",
        "book_id": raw.get("book_id"),
        "run_id": raw.get("run_id"),
        "phase": raw.get("phase"),
        "git_commit": raw.get("git_commit"),
        "message": raw.get("message", ""),
        "details": details if isinstance(details, dict) else {},
    }


def read_events(limit: int = 500) -> list[dict]:
    if not EVENT_LOG.exists():
        return []
    rows: list[dict] = []
    with EVENT_LOG.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(normalize_event(json.loads(line)))
            except json.JSONDecodeError:
                rows.append(
                    normalize_event(
                        {"level": "error", "kind": "log_error", "message": "Unreadable event row",
                         "details": {"raw": line}, "timestamp": utc_now()}
                    )
                )
    return rows[-limit:]


def summarize_events(events: list[dict]) -> dict:
    by_level = {level: 0 for level in LEVELS}
    actors: set[str] = set()
    books: set[str] = set()
    runs: set[str] = set()
    for event in events:
        by_level[event["level"]] = by_level.get(event["level"], 0) + 1
        actors.add(event["actor"])
        if event.get("book_id"):
            books.add(str(event["book_id"]))
        if event.get("book_id") and event.get("run_id"):
            runs.add(f"{event['book_id']}/{event['run_id']}")
    return {
        "total": len(events),
        "by_level": by_level,
        "issues": by_level.get("warning", 0) + by_level.get("error", 0),
        "actors": sorted(actors),
        "books": sorted(books),
        "runs": sorted(runs),
        "latest": events[-1]["timestamp"] if events else "",
    }


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict]:
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
    except OSError:
        return []
    return rows


def phase1_run_summaries() -> list[dict]:
    summaries: list[dict] = []
    if not RUNS_DIR.exists():
        return summaries
    for run_dir in sorted(RUNS_DIR.glob("*/phase1_v*")):
        if not run_dir.is_dir():
            continue
        manifest = read_json(run_dir / "source_manifest.json")
        inventory = read_jsonl(run_dir / "page_inventory.jsonl")
        safety = read_json(run_dir / "post_adoption_canonical_safety_report.json")
        summaries.append(
            {
                "book_id": manifest.get("book_id") or run_dir.parent.name,
                "run_id": manifest.get("run_id") or run_dir.name,
                "created_at": manifest.get("created_at", ""),
                "page_count": manifest.get("page_count", len(inventory)),
                "safe_for_downstream": safety.get("safe_for_downstream"),
                "warning_count": safety.get("warning_count"),
                "audit_url": f"/runs/{run_dir.parent.name}/{run_dir.name}/phase1_audit.html",
            }
        )
    return summaries


RUN_STREAMS = {
    "main_paragraph_candidates.jsonl": "main_paragraph",
    "structure_candidates.jsonl": "structure",
    "page_artifacts_candidates.jsonl": "page_artifact",
    "unknown_objects.jsonl": "unknown",
}


def latest_run_dir(book: str | None = None, run: str | None = None) -> Path | None:
    if book and run:
        candidate = RUNS_DIR / book / run
        return candidate if candidate.is_dir() else None
    runs = [p for p in sorted(RUNS_DIR.glob("*/phase1_v*")) if p.is_dir()]
    return runs[-1] if runs else None


def page_object_breakdown(run_dir: Path) -> list[dict]:
    """Per-page object map that makes cross-page (spanning) paragraphs explicit.

    A paragraph spanning pages is attributed to its START page, so a page that
    only continues a prior paragraph shows fewer 'starts'. This surfaces both
    'starts' (objects beginning on the page) and 'continued_from' (objects begun
    earlier that flow into the page), explaining per-page object counts.
    """
    objs: list[dict] = []
    for fname, otype in RUN_STREAMS.items():
        for o in read_jsonl(run_dir / fname):
            pg = o.get("page_number")
            if pg is None:
                continue
            bbox = o.get("bbox") or {}
            spans = sorted({int(p) for p in (bbox.get("page_numbers") or [pg]) if p is not None}) or [int(pg)]
            text = o.get("clean_text") or o.get("raw_text") or o.get("text") or ""
            objs.append({"id": o.get("object_id"), "type": otype, "start_page": int(pg), "spans": spans, "preview": text[:140]})
    pages = sorted({p for o in objs for p in o["spans"]})
    out: list[dict] = []
    for pg in pages:
        starts = [o for o in objs if o["start_page"] == pg]
        continued = [
            {"id": o["id"], "type": o["type"], "from_page": o["start_page"], "preview": o["preview"]}
            for o in objs if pg in o["spans"] and o["start_page"] < pg
        ]
        out.append({
            "page": pg,
            "object_count": len(starts),
            "starts": [
                {"id": o["id"], "type": o["type"], "preview": o["preview"],
                 "spans_to": [p for p in o["spans"] if p > pg]}
                for o in starts
            ],
            "continued_from": continued,
        })
    return out


def dashboard_html() -> bytes:
    return DASHBOARD.encode("utf-8")


DASHBOARD = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>iknow · Claude lane observability</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0c0f14; --panel: #141921; --line: #232a34; --text: #e7ecf3; --muted: #8b97a7;
      --accent: #5cc8ff;
      --debug: #6b7686; --info: #5cc8ff; --milestone: #7ee787; --warning: #f0b86e; --error: #ff7b72;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
    main { max-width: 1240px; margin: 0 auto; padding: 24px 20px 64px; }
    header { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap; }
    h1 { font-size: 22px; margin: 0 0 4px; }
    .sub { color: var(--muted); font-size: 13px; margin: 0; }
    .live { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--milestone); box-shadow: 0 0 8px var(--milestone); }
    .dot.off { background: var(--error); box-shadow: 0 0 8px var(--error); }
    .cards { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin: 20px 0; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
    .card span { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
    .card strong { display: block; font-size: 22px; margin-top: 6px; }
    .card.warn strong { color: var(--warning); } .card.err strong { color: var(--error); }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin-bottom: 18px; }
    .toolbar input[type=search], .toolbar select { background: #0c0f14; color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 7px 9px; font-size: 13px; }
    .toolbar input[type=search] { flex: 1; min-width: 200px; }
    .toolbar label { font-size: 13px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
    .toolbar button { background: #0c0f14; color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 7px 12px; font-size: 13px; cursor: pointer; }
    .toolbar button:hover { border-color: var(--accent); }
    .hint { color: var(--muted); font-size: 12px; }
    h2 { font-size: 15px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; border-top: 1px solid var(--line); padding-top: 16px; margin: 26px 0 12px; }
    .runs { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }
    .run { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .run h3 { margin: 0 0 8px; font-size: 14px; }
    .run .kv { display: flex; justify-content: space-between; font-size: 13px; padding: 3px 0; color: var(--muted); }
    .run .kv b { color: var(--text); font-weight: 600; }
    .badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; border: 1px solid var(--line); }
    .badge.no { color: var(--warning); border-color: var(--warning); }
    .badge.yes { color: var(--milestone); border-color: var(--milestone); }
    .group { border: 1px solid var(--line); border-radius: 8px; margin: 10px 0; overflow: hidden; }
    .group > summary { cursor: pointer; padding: 10px 14px; background: var(--panel); font-size: 14px; display: flex; justify-content: space-between; }
    .event { display: grid; grid-template-columns: 92px 84px 1fr; gap: 12px; padding: 11px 14px; border-top: 1px solid var(--line); border-left: 3px solid var(--info); }
    .event.debug { border-left-color: var(--debug); } .event.info { border-left-color: var(--info); }
    .event.milestone { border-left-color: var(--milestone); } .event.warning { border-left-color: var(--warning); }
    .event.error { border-left-color: var(--error); }
    .event .time { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; color: var(--muted); }
    .tags { display: flex; flex-direction: column; gap: 3px; }
    .tag { font-size: 10px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
    .tag.lvl-milestone { color: var(--milestone); } .tag.lvl-warning { color: var(--warning); } .tag.lvl-error { color: var(--error); }
    .msg { font-size: 14px; }
    .meta { font-size: 11px; color: var(--muted); margin-top: 3px; display: flex; gap: 10px; flex-wrap: wrap; }
    pre { white-space: pre-wrap; margin: 6px 0 0; color: var(--muted); font-size: 12px; font-family: ui-monospace, Menlo, Consolas, monospace; }
    .empty { color: var(--muted); padding: 24px; text-align: center; border: 1px dashed var(--line); border-radius: 8px; }
    .pom-obj { display: flex; gap: 10px; align-items: baseline; padding: 7px 14px; border-top: 1px solid var(--line); font-size: 13px; }
    .pom-text { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .span-to { color: var(--milestone); white-space: nowrap; }
    .span-from { color: var(--warning); white-space: nowrap; }
    a { color: var(--accent); }
    @media (max-width: 900px) { .cards { grid-template-columns: repeat(3, 1fr); } .event { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>iknow · Claude lane observability</h1>
      <p class="sub" id="sub">Local monitor &mdash; loading&hellip;</p>
    </div>
    <div class="live"><span class="dot" id="dot"></span><span id="liveText">connecting</span></div>
  </header>

  <section class="cards">
    <div class="card"><span>Events</span><strong id="m-total">0</strong></div>
    <div class="card"><span>Milestones</span><strong id="m-milestone">0</strong></div>
    <div class="card warn"><span>Warnings</span><strong id="m-warning">0</strong></div>
    <div class="card err"><span>Errors</span><strong id="m-error">0</strong></div>
    <div class="card"><span>Runs</span><strong id="m-runs">0</strong></div>
    <div class="card"><span>Books</span><strong id="m-books">0</strong></div>
  </section>

  <div class="toolbar">
    <input type="search" id="q" placeholder="Search message, kind, details&hellip;  ( / to focus )">
    <select id="f-level"><option value="">all levels</option></select>
    <select id="f-actor"><option value="">all actors</option></select>
    <select id="f-book"><option value="">all books</option></select>
    <select id="f-phase"><option value="">all phases</option></select>
    <label><input type="checkbox" id="f-issues"> issues only</label>
    <label><input type="checkbox" id="f-group" checked> group by run</label>
    <label><input type="checkbox" id="f-pause"> pause</label>
    <button id="clear">clear</button>
    <span class="hint">keys: / search · p pause · r refresh</span>
  </div>

  <section>
    <h2>Phase 1 runs</h2>
    <div class="runs" id="runs"></div>
  </section>

  <section>
    <h2>Per-page object map <span class="hint" id="pomRun"></span></h2>
    <p class="hint">A paragraph spanning a page break is attributed to the page it starts on, so a page may show fewer "starts" than paragraphs you see. "continues to / continued from" markers make spanning explicit.</p>
    <div id="pageObjects"></div>
  </section>

  <section>
    <h2>Events <span class="hint" id="shown"></span></h2>
    <div id="events"></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const STATE = { events: [], runs: [], meta: {} };

function esc(v){return String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));}
function opts(sel, values){
  const cur = sel.value;
  const base = sel.querySelector("option").outerHTML;
  sel.innerHTML = base + values.map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join("");
  if (values.includes(cur)) sel.value = cur;
}
function uniq(arr){return [...new Set(arr)].sort();}

function passesFilters(e){
  const q = $("q").value.trim().toLowerCase();
  if ($("f-level").value && e.level !== $("f-level").value) return false;
  if ($("f-actor").value && e.actor !== $("f-actor").value) return false;
  if ($("f-book").value && (e.book_id||"") !== $("f-book").value) return false;
  if ($("f-phase").value && (e.phase||"") !== $("f-phase").value) return false;
  if ($("f-issues").checked && !(e.level==="warning"||e.level==="error")) return false;
  if (q){
    const hay = (e.message+" "+e.kind+" "+JSON.stringify(e.details)).toLowerCase();
    if (!hay.includes(q)) return false;
  }
  return true;
}

function eventHtml(e){
  const t = e.timestamp ? esc(e.timestamp.split("T")[1]||e.timestamp).replace("Z","") : "—";
  const meta = [];
  if (e.book_id) meta.push(`book: ${esc(e.book_id)}`);
  if (e.run_id) meta.push(`run: ${esc(e.run_id)}`);
  if (e.phase) meta.push(`phase: ${esc(e.phase)}`);
  if (e.git_commit) meta.push(`@${esc(e.git_commit)}`);
  const det = Object.keys(e.details||{}).length ? `<pre>${esc(JSON.stringify(e.details,null,2))}</pre>` : "";
  return `<article class="event ${esc(e.level)}">
    <div class="time">${t}</div>
    <div class="tags"><span class="tag lvl-${esc(e.level)}">${esc(e.level)}</span><span class="tag">${esc(e.actor)}</span><span class="tag">${esc(e.kind)}</span></div>
    <div><div class="msg">${esc(e.message)}</div>${meta.length?`<div class="meta">${meta.map(m=>`<span>${m}</span>`).join("")}</div>`:""}${det}</div>
  </article>`;
}

function render(){
  const filtered = STATE.events.filter(passesFilters);
  $("shown").textContent = `(${filtered.length} of ${STATE.events.length})`;
  const host = $("events");
  if (!filtered.length){ host.innerHTML = `<div class="empty">No events match the current filters.</div>`; return; }
  const ordered = filtered.slice().reverse();
  if ($("f-group").checked){
    const groups = {};
    for (const e of ordered){
      const key = e.book_id ? `${e.book_id} / ${e.run_id||"—"}` : "unattributed";
      (groups[key] = groups[key]||[]).push(e);
    }
    host.innerHTML = Object.entries(groups).map(([k,list])=>
      `<details class="group" open><summary><span>${esc(k)}</span><span class="hint">${list.length}</span></summary>${list.map(eventHtml).join("")}</details>`
    ).join("");
  } else {
    host.innerHTML = ordered.map(eventHtml).join("");
  }
}

function renderRuns(){
  const host = $("runs");
  if (!STATE.runs.length){ host.innerHTML = `<div class="empty">No Phase 1 runs detected.</div>`; return; }
  host.innerHTML = STATE.runs.slice().reverse().map(r=>{
    const safe = r.safe_for_downstream;
    const badge = safe===true ? `<span class="badge yes">safe</span>` : safe===false ? `<span class="badge no">blocked</span>` : `<span class="badge">—</span>`;
    return `<article class="run"><h3>${esc(r.book_id)} / ${esc(r.run_id)}</h3>
      <div class="kv"><span>downstream</span>${badge}</div>
      <div class="kv"><span>pages</span><b>${esc(r.page_count)}</b></div>
      <div class="kv"><span>warnings</span><b>${esc(r.warning_count??"—")}</b></div>
      <div class="kv"><span>created</span><b>${esc(r.created_at||"—")}</b></div>
      <div class="kv"><a href="${esc(r.audit_url)}">open audit ↗</a></div></article>`;
  }).join("");
}

function applySummary(s){
  $("m-total").textContent = s.total;
  $("m-milestone").textContent = s.by_level.milestone||0;
  $("m-warning").textContent = s.by_level.warning||0;
  $("m-error").textContent = s.by_level.error||0;
  $("m-runs").textContent = s.runs.length;
  $("m-books").textContent = s.books.length;
  opts($("f-actor"), s.actors);
  opts($("f-book"), s.books);
}

async function refresh(){
  try {
    const [ev, rs, sm] = await Promise.all([
      fetch("/api/events",{cache:"no-store"}).then(r=>r.json()),
      fetch("/api/phase1-runs",{cache:"no-store"}).then(r=>r.json()),
      fetch("/api/summary",{cache:"no-store"}).then(r=>r.json()),
    ]);
    STATE.events = ev; STATE.runs = rs;
    $("dot").classList.remove("off"); $("liveText").textContent = $("f-pause").checked ? "paused" : "live";
    opts($("f-level"), ["debug","info","milestone","warning","error"]);
    opts($("f-phase"), uniq(ev.map(e=>e.phase).filter(Boolean)));
    applySummary(sm);
    renderRuns(); render();
  } catch(e){
    $("dot").classList.add("off"); $("liveText").textContent = "offline";
  }
}

["q","f-level","f-actor","f-book","f-phase","f-issues","f-group"].forEach(id=>{
  $(id).addEventListener("input", render);
});
$("clear").addEventListener("click", ()=>{
  ["q","f-level","f-actor","f-book","f-phase"].forEach(id=>$(id).value="");
  $("f-issues").checked=false; render();
});
document.addEventListener("keydown", e=>{
  if (e.target.tagName==="INPUT"||e.target.tagName==="SELECT") { if(e.key==="Escape") e.target.blur(); return; }
  if (e.key==="/"){ e.preventDefault(); $("q").focus(); }
  if (e.key==="p"){ $("f-pause").checked = !$("f-pause").checked; }
  if (e.key==="r"){ refresh(); }
});

function renderPageObjects(data){
  const host=$("pageObjects");
  $("pomRun").textContent = data.run ? `(${data.run})` : "";
  if(!data.pages || !data.pages.length){ host.innerHTML=`<div class="empty">No run objects found.</div>`; return; }
  host.innerHTML = data.pages.map(p=>{
    const starts=(p.starts||[]).map(s=>`<div class="pom-obj"><span class="tag">${esc(s.type)}</span>${(s.spans_to&&s.spans_to.length)?`<span class="span-to">&rarr; continues to p${s.spans_to.join(",")}</span>`:""}<span class="pom-text">${esc(s.preview)}</span></div>`).join("");
    const cont=(p.continued_from||[]).map(c=>`<div class="pom-obj"><span class="span-from">&#8617; continued from p${c.from_page}</span><span class="pom-text">${esc(c.preview)}</span></div>`).join("");
    const meta=`${p.object_count} start${p.object_count===1?"":"s"}${p.continued_from.length?` &middot; +${p.continued_from.length} continued`:""}`;
    return `<details class="group"><summary><span>Page ${p.page}</span><span class="hint">${meta}</span></summary>${starts}${cont}</details>`;
  }).join("");
}
function loadPageObjects(){
  fetch("/api/page-objects",{cache:"no-store"}).then(r=>r.json()).then(renderPageObjects).catch(()=>{});
}

fetch("/api/meta",{cache:"no-store"}).then(r=>r.json()).then(m=>{
  STATE.meta=m; $("sub").innerHTML = `Claude-lane monitor on <code>${esc(m.host)}:${esc(m.port)}</code> &middot; schema <code>${esc(m.schema)}</code> &middot; 8765 is the legacy/Codex dashboard.`;
}).catch(()=>{});

refresh();
loadPageObjects();
setInterval(()=>{ if(!$("f-pause").checked) refresh(); }, 2000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.respond(HTTPStatus.OK, "text/html; charset=utf-8", dashboard_html())
        elif path == "/api/events":
            self.json(read_events())
        elif path == "/api/summary":
            self.json(summarize_events(read_events()))
        elif path == "/api/phase1-runs":
            self.json(phase1_run_summaries())
        elif path == "/api/page-objects":
            from urllib.parse import parse_qs
            q = parse_qs(urlparse(self.path).query)
            run_dir = latest_run_dir(q.get("book", [None])[0], q.get("run", [None])[0])
            self.json({
                "run": run_dir.parent.name + "/" + run_dir.name if run_dir else None,
                "pages": page_object_breakdown(run_dir) if run_dir else [],
            })
        elif path == "/api/meta":
            self.json({"host": self.server.server_address[0], "port": self.server.server_address[1], "schema": "iknow.observe/2"})
        elif path.startswith("/runs/"):
            self.serve_run_artifact(path)
        else:
            self.respond(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found")

    def json(self, payload: object) -> None:
        self.respond(HTTPStatus.OK, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=True).encode("utf-8"))

    def log_message(self, *args: object) -> None:
        return

    def respond(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_run_artifact(self, request_path: str) -> None:
        parts = [p for p in request_path.split("/") if p]
        if len(parts) < 4 or parts[0] != "runs":
            self.respond(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found")
            return
        _, book_id, run_id, *artifact_parts = parts
        allowed_top = len(artifact_parts) == 1 and artifact_parts[0] in {"phase1_audit.html", "source_manifest.json"}
        allowed_img = (
            len(artifact_parts) == 2 and artifact_parts[0] == "page_images"
            and artifact_parts[1].startswith("page_") and artifact_parts[1].endswith(".jpg")
        )
        if not (allowed_top or allowed_img):
            self.respond(HTTPStatus.FORBIDDEN, "text/plain; charset=utf-8", b"Forbidden")
            return
        candidate = (RUNS_DIR / book_id / run_id / Path(*artifact_parts)).resolve()
        if not str(candidate).startswith(str(RUNS_DIR.resolve())) or not candidate.exists():
            self.respond(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found")
            return
        ctype = {".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8", ".jpg": "image/jpeg"}.get(candidate.suffix, "application/octet-stream")
        self.respond(HTTPStatus.OK, ctype, candidate.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Claude-lane observability dashboard.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ReusableThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving Claude-lane observability at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
