"""Bbox/span bucketing helpers for Phase 1 diagnostics.

Extracted verbatim from ``phase1_extract.py`` (decomposition step 5). Pure
range/severity bucketing with no project state. ``phase1_extract`` re-imports
these names so existing call sites are unchanged.
"""
from __future__ import annotations

from typing import Any


def bbox_height(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    top = value.get("top")
    bottom = value.get("bottom")
    if top is None or bottom is None:
        return None
    return max(0.0, float(bottom) - float(top))


def first_unique(values: list[Any], limit: int) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def source_line_count_range(count: int) -> str:
    if count <= 3:
        return "1-3"
    if count <= 6:
        return "4-6"
    if count <= 10:
        return "7-10"
    return "11+"


def page_height_ratio_range(ratio: float | None) -> str:
    if ratio is None:
        return "unknown"
    if ratio < 0.18:
        return "lt_18_percent"
    if ratio < 0.28:
        return "18_to_28_percent"
    if ratio < 0.40:
        return "28_to_40_percent"
    return "gte_40_percent"


def bbox_span_severity(line_count: int, page_height_ratio: float | None, word_count: int) -> str:
    if page_height_ratio is not None and page_height_ratio >= 0.40:
        return "high"
    if line_count >= 11:
        return "high"
    if page_height_ratio is not None and page_height_ratio >= 0.28:
        return "medium"
    if word_count > 170:
        return "medium"
    return "low"
