"""Contract tests for the Phase 1 reproducibility manifest tool.

These are self-contained: they build a temporary run directory so they do not
depend on git-ignored local run output, and therefore run in any clone/CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import baseline_manifest as bm


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "page_images").mkdir(parents=True)
    (run / "clean_objects.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    (run / "validation_report.json").write_text('{"ok": true}\n', encoding="utf-8")
    (run / "page_images" / "page_0001.png").write_bytes(b"\x89PNG\r\n")
    return run


def test_compute_and_verify_match(tmp_path):
    run = _make_run(tmp_path)
    manifest = tmp_path / "manifest.json"
    payload = bm.write_manifest(run, manifest, label="test")
    assert payload["file_count"] == 3
    result = bm.verify_manifest(run, manifest)
    assert bm.manifest_matches(result)


def test_changed_file_detected(tmp_path):
    run = _make_run(tmp_path)
    manifest = tmp_path / "manifest.json"
    bm.write_manifest(run, manifest, label="test")
    (run / "clean_objects.jsonl").write_text('{"id": 2}\n', encoding="utf-8")
    result = bm.verify_manifest(run, manifest)
    assert result["changed"] == ["clean_objects.jsonl"]
    assert not bm.manifest_matches(result)


def test_missing_and_unexpected_detected(tmp_path):
    run = _make_run(tmp_path)
    manifest = tmp_path / "manifest.json"
    bm.write_manifest(run, manifest, label="test")
    (run / "validation_report.json").unlink()
    (run / "extra.json").write_text("{}\n", encoding="utf-8")
    result = bm.verify_manifest(run, manifest)
    assert result["missing"] == ["validation_report.json"]
    assert result["unexpected"] == ["extra.json"]


def test_relative_posix_keys(tmp_path):
    run = _make_run(tmp_path)
    entries = bm.compute_manifest(run)
    assert "page_images/page_0001.png" in entries
    assert all(not key.startswith("/") for key in entries)


# --- normalized content fingerprint -----------------------------------------

def test_canonical_content_strips_volatile_json_keys(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    a = run / "report.json"
    a.write_text('{"created_at": "2026-05-28T00:00:00Z", "run_id": "x", "value": 7}\n', encoding="utf-8")
    first = bm.canonical_content(a, run)
    a.write_text('{"created_at": "2026-05-29T11:22:33Z", "run_id": "y", "value": 7}\n', encoding="utf-8")
    second = bm.canonical_content(a, run)
    assert first == second  # volatile fields excluded


def test_canonical_content_neutralizes_text_timestamps_and_paths(tmp_path):
    run = tmp_path / "phase1_vX"
    run.mkdir()
    html = run / "phase1_audit.html"
    html.write_text(f"built 2026-05-28T17:43:59Z at {run.resolve()}/page_images/p1.jpg", encoding="utf-8")
    out = bm.canonical_content(html, run).decode("utf-8")
    assert "<ts>" in out and "<run_dir>" in out
    assert "2026-05-28T17:43:59Z" not in out


def test_canonical_content_does_not_corrupt_words_colliding_with_run_id(tmp_path):
    # Regression: a bare run_id that is a substring of real content (e.g. "ca"
    # inside "candidate") must NOT be mangled. The run_id is neutralized only as
    # a path segment, so two runs whose ids collide with content still match.
    run_ca = tmp_path / "ca"
    run_ok = tmp_path / "zz"
    for run in (run_ca, run_ok):
        run.mkdir()
        (run / "phase1_audit.html").write_text("main_paragraph_candidates: 262\n", encoding="utf-8")
    out_ca = bm.canonical_content(run_ca / "phase1_audit.html", run_ca)
    out_ok = bm.canonical_content(run_ok / "phase1_audit.html", run_ok)
    assert b"candidates" in out_ca  # not corrupted into "<run_id>ndidates"
    assert out_ca == out_ok  # run-id choice does not change normalized content


def test_canonical_content_neutralizes_run_id_as_path_segment(tmp_path):
    run = tmp_path / "myrun"
    run.mkdir()
    (run / "phase1_audit.html").write_text("see data/runs/book/myrun/page_images/p1.jpg\n", encoding="utf-8")
    out = bm.canonical_content(run / "phase1_audit.html", run).decode("utf-8")
    assert "/myrun/" not in out and "<run_id>" in out


def test_content_fingerprint_changed_detected(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "clean_objects.jsonl").write_text('{"text": "a"}\n', encoding="utf-8")
    manifest = tmp_path / "fp.json"
    bm.write_content_fingerprint(run, manifest, label="t")
    assert bm.manifest_matches(bm.verify_content_fingerprint(run, manifest))
    (run / "clean_objects.jsonl").write_text('{"text": "b"}\n', encoding="utf-8")
    result = bm.verify_content_fingerprint(run, manifest)
    assert result["changed"] == ["clean_objects.jsonl"]
