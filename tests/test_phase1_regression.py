"""Golden-output regression for the Douglass phase1_v3 pipeline.

Re-runs the deterministic Phase 1 extraction and asserts the output reproduces
the committed *normalized* fingerprint (timestamps, run_id, and absolute paths
neutralized). This is the guard that lets the monolith be refactored safely:
any behavior drift surfaces as a fingerprint mismatch.

It is inherently local-only because the source PDF lives under git-ignored
data/books/. It skips cleanly when the PDF, baseline, or pdfplumber is absent,
so a fresh clone / CI stays green.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import baseline_manifest as bm

PDF = ROOT / "data" / "books" / "douglass_narrative.pdf"
BASELINE = ROOT / "docs" / "baselines" / "douglass_narrative" / "phase1_v3" / "normalized_manifest.json"


@pytest.mark.skipif(not PDF.exists(), reason="Douglass PDF not present (git-ignored); local-only regression")
@pytest.mark.skipif(not BASELINE.exists(), reason="normalized baseline fingerprint missing")
def test_douglass_v3_reproduces_normalized_baseline():
    try:
        import phase1_extract
    except Exception as exc:  # pdfplumber or other extraction dep unavailable
        pytest.skip(f"phase1_extract unavailable: {exc}")

    run_id = "_regression_check"
    out_dir = ROOT / "data" / "runs" / "douglass_narrative" / run_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        phase1_extract.run_phase1(PDF, "douglass_narrative", run_id)
        result = bm.verify_content_fingerprint(out_dir, BASELINE)
        assert bm.manifest_matches(result), (
            f"Phase 1 output drifted from baseline: "
            f"missing={result['missing']} unexpected={result['unexpected']} changed={result['changed']}"
        )
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir)
