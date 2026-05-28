"""Line-geometry helpers for Phase 1 (margins, indents, heights, gaps).

Extracted verbatim from ``phase1_extract.py`` (decomposition step 3). Pure
functions over line records; the only dependency is ``classify_line``.
``phase1_extract`` re-imports these names so existing call sites are unchanged.
"""
from __future__ import annotations

from typing import Any

from phase1.text_utils import classify_line


def normalize_line_records(raw_lines: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_index, value in enumerate(raw_lines, start=1):
        if isinstance(value, dict):
            text = str(value.get("text", ""))
            records.append(
                {
                    "line_index": line_index,
                    "text": text,
                    "x0": value.get("x0"),
                    "x1": value.get("x1"),
                    "top": value.get("top"),
                    "bottom": value.get("bottom"),
                }
            )
        else:
            records.append({"line_index": line_index, "text": str(value), "x0": None, "x1": None, "top": None, "bottom": None})
    return records


def body_left_margin(line_records: list[dict[str, Any]], page_number: int) -> float | None:
    candidates: list[float] = []
    for record in line_records:
        text = record["text"]
        if not text.strip() or record.get("x0") is None:
            continue
        line_type, _, _ = classify_line(text, page_number)
        if line_type == "paragraph_line":
            candidates.append(float(record["x0"]))
    if not candidates:
        return None
    return min(candidates)


def starts_new_indented_paragraph(record: dict[str, Any], left_margin: float | None) -> bool:
    if left_margin is None or record.get("x0") is None:
        return False
    return float(record["x0"]) - left_margin >= 12.0


def line_height(record: dict[str, Any]) -> float | None:
    if record.get("top") is None or record.get("bottom") is None:
        return None
    return max(0.0, float(record["bottom"]) - float(record["top"]))


def line_gap(previous: dict[str, Any], current: dict[str, Any]) -> float | None:
    if previous.get("bottom") is None or current.get("top") is None:
        return None
    return float(current["top"]) - float(previous["bottom"])
