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
