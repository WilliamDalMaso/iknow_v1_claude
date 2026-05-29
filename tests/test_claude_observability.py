"""Contract tests for the Claude-lane observability writer and server helpers.

Pure functions only -- no socket is bound, so these run anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import claude_observability_server as srv
import observe


def test_append_event_is_structured_and_backward_compatible(tmp_path, monkeypatch):
    log = tmp_path / "events.jsonl"
    monkeypatch.setattr(observe, "EVENT_LOG", log)
    # Positional call (legacy style) still works.
    event = observe.append_event("note", "hello", {"k": 1})
    assert event["schema"] == "iknow.observe/2"
    assert event["level"] == "info"
    assert event["actor"] == "claude"
    # Legacy-readable keys are present.
    for key in ("timestamp", "kind", "message", "details"):
        assert key in event
    written = json.loads(log.read_text(encoding="utf-8").strip())
    assert written["message"] == "hello"


def test_append_event_structured_fields(tmp_path, monkeypatch):
    log = tmp_path / "events.jsonl"
    monkeypatch.setattr(observe, "EVENT_LOG", log)
    event = observe.append_event(
        "phase", "started", {"x": 2},
        level="milestone", actor="claude", book_id="douglass_narrative",
        run_id="phase1_v3", phase="A", commit="abc1234",
    )
    assert event["level"] == "milestone"
    assert event["book_id"] == "douglass_narrative"
    assert event["run_id"] == "phase1_v3"
    assert event["phase"] == "A"
    assert event["git_commit"] == "abc1234"


def test_invalid_level_falls_back_to_info(tmp_path, monkeypatch):
    monkeypatch.setattr(observe, "EVENT_LOG", tmp_path / "events.jsonl")
    event = observe.append_event("note", "msg", level="bogus")
    assert event["level"] == "info"


def test_normalize_legacy_event_fills_defaults():
    norm = srv.normalize_event({"timestamp": "t", "kind": "system", "message": "hi"})
    assert norm["schema"] == "legacy"
    assert norm["level"] == "info"
    assert norm["actor"] == "unknown"
    assert norm["book_id"] is None
    assert norm["details"] == {}


def test_normalize_preserves_structured_event():
    raw = {"schema": "iknow.observe/2", "timestamp": "2026-05-28T00:00:00Z",
           "level": "error", "actor": "claude",
           "book_id": "b", "run_id": "r", "phase": "C", "git_commit": "deadbee",
           "kind": "k", "message": "m", "details": {"a": 1}}
    norm = srv.normalize_event(raw)
    assert norm == raw


def test_summarize_events_counts_levels_and_facets():
    events = [
        srv.normalize_event({"level": "milestone", "actor": "claude", "book_id": "b1", "run_id": "r1"}),
        srv.normalize_event({"level": "warning", "actor": "claude", "book_id": "b1", "run_id": "r1"}),
        srv.normalize_event({"level": "error", "actor": "codex", "book_id": "b2", "run_id": "r2"}),
        srv.normalize_event({"level": "info"}),
    ]
    summary = srv.summarize_events(events)
    assert summary["total"] == 4
    assert summary["by_level"]["warning"] == 1
    assert summary["by_level"]["error"] == 1
    assert summary["issues"] == 2
    assert summary["actors"] == ["claude", "codex", "unknown"]
    assert summary["books"] == ["b1", "b2"]
    assert summary["runs"] == ["b1/r1", "b2/r2"]


def test_page_object_breakdown_marks_spanning(tmp_path):
    run = tmp_path / "phase1_vX"
    run.mkdir()
    (run / "main_paragraph_candidates.jsonl").write_text(
        json.dumps({"object_id": "a", "page_number": 6, "clean_text": "A beloved friend",
                    "bbox": {"page_numbers": [6, 7]}}) + "\n"
        + json.dumps({"object_id": "b", "page_number": 7, "clean_text": "It was at once"}) + "\n",
        encoding="utf-8",
    )
    (run / "page_artifacts_candidates.jsonl").write_text(
        json.dumps({"object_id": "h6", "page_number": 6, "clean_text": "vi PREFACE"}) + "\n", encoding="utf-8")
    by_page = {p["page"]: p for p in srv.page_object_breakdown(run)}
    # page 6: paragraph 'a' (spans to 7) + header start here
    assert by_page[6]["object_count"] == 2
    a = next(s for s in by_page[6]["starts"] if s["id"] == "a")
    assert a["spans_to"] == [7]
    # page 7: only 'b' starts here; 'a' is continued from page 6
    assert by_page[7]["object_count"] == 1
    assert by_page[7]["continued_from"][0]["from_page"] == 6
    assert by_page[7]["continued_from"][0]["id"] == "a"


def test_read_events_handles_unreadable_rows(tmp_path, monkeypatch):
    log = tmp_path / "events.jsonl"
    log.write_text('{"level":"info","message":"ok"}\nnot json\n', encoding="utf-8")
    monkeypatch.setattr(srv, "EVENT_LOG", log)
    events = srv.read_events()
    assert len(events) == 2
    assert events[1]["level"] == "error"
    assert events[1]["kind"] == "log_error"
