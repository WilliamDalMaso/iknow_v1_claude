"""Paragraph-text and cross-page provenance heuristics for Phase 1.

Extracted verbatim from ``phase1_extract.py`` (decomposition step 4). Pure
functions over text and candidate rows (only ``re`` and the shared
``TERMINAL_PUNCTUATION`` constant). ``phase1_extract`` re-imports these names
(constant included) so existing call sites are unchanged.
"""
from __future__ import annotations

import re
from typing import Any

TERMINAL_PUNCTUATION = set(".?!;:'\")”’]")


def paragraph_text_looks_incomplete(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[-1] not in TERMINAL_PUNCTUATION:
        return True
    return bool(re.search(r"\b(and|or|than|to|of|with|from|for|in|by|the|a|an|new|must|enough)$", stripped, re.IGNORECASE))


def paragraph_text_looks_like_continuation(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    if stripped[0].islower():
        return True
    if stripped[0] in ",;:)]}”’":
        return True
    first_token = stripped.split()[0].strip(".,;:!?\"“”‘’")
    return first_token.istitle() and len(first_token) <= 12


def combine_bbox_for_cross_page(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_bbox = first.get("bbox") or {}
    second_bbox = second.get("bbox") or {}
    x0_values = [value for value in [first_bbox.get("x0"), second_bbox.get("x0")] if value is not None]
    x1_values = [value for value in [first_bbox.get("x1"), second_bbox.get("x1")] if value is not None]
    page_numbers = []
    for source in [first_bbox.get("page_numbers"), [first.get("page_number")], second_bbox.get("page_numbers"), [second.get("page_number")]]:
        for page_number in source or []:
            if page_number is not None and page_number not in page_numbers:
                page_numbers.append(page_number)
    return {
        "x0": min(x0_values) if x0_values else None,
        "x1": max(x1_values) if x1_values else None,
        "top": first_bbox.get("top"),
        "bottom": second_bbox.get("bottom"),
        "cross_page": True,
        "page_numbers": page_numbers,
    }


def page_numbers_for_candidate(row: dict[str, Any]) -> list[int]:
    pages = []
    bbox = row.get("bbox") or {}
    for page_number in bbox.get("page_numbers") or []:
        if isinstance(page_number, int) and page_number not in pages:
            pages.append(page_number)
    for line_id in row.get("source_line_ids", []):
        match = re.search(r":p(\d{4}):line\d+", str(line_id))
        if match:
            page_number = int(match.group(1))
            if page_number not in pages:
                pages.append(page_number)
    if not pages and row.get("page_number") is not None:
        pages.append(int(row["page_number"]))
    return sorted(pages)


def source_line_page_and_index(line_id: str) -> tuple[int | None, int | None]:
    match = re.search(r":p(\d{4}):line(\d+)", str(line_id))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))
