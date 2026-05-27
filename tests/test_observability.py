from __future__ import annotations

from src.observability_server import dashboard_html


def test_dashboard_contains_phase_1_focus() -> None:
    html = dashboard_html().decode("utf-8")
    assert "iknow v1 Observability" in html
    assert "Phase 1" in html
    assert "/api/events" in html

