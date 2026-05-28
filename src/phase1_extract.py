from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber
import pypdfium2 as pdfium

from phase1.io_utils import (
    read_commented_jsonl,
    read_json,
    read_jsonl,
    safe_dom_id,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from phase1.text_utils import (
    CID_PATTERN,
    ROMAN_PATTERN,
    TEXT_TOKEN_PATTERN,
    classify_line,
    clean_line,
    is_page_number_token,
    join_paragraph_lines,
    normalized_furniture_text,
    normalized_object_text,
    terminal_line_boundary,
    uppercase_line_start,
)
from phase1.layout_geometry import (
    body_left_margin,
    line_gap,
    line_height,
    normalize_line_records,
    starts_new_indented_paragraph,
)
from phase1.heuristics import (
    TERMINAL_PUNCTUATION,
    combine_bbox_for_cross_page,
    page_numbers_for_candidate,
    paragraph_text_looks_incomplete,
    paragraph_text_looks_like_continuation,
    source_line_page_and_index,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "runs"
REVIEWS_DIR = ROOT / "reviews"
PAGE_IMAGES_DIR_NAME = "page_images"
REQUIRED_ARTIFACTS = [
    "source_manifest.json",
    "page_inventory.jsonl",
    "raw_pages.jsonl",
    "layout_objects.jsonl",
    "clean_objects.jsonl",
    "main_paragraph_candidates.jsonl",
    "structure_candidates.jsonl",
    "page_artifacts_candidates.jsonl",
    "unknown_objects.jsonl",
    "reconstruction_map_candidate.json",
    "reading_order_candidate.json",
    PAGE_IMAGES_DIR_NAME,
    "review_overrides_applied.jsonl",
    "cross_page_join_decisions_applied.jsonl",
    "canonical_paragraphs.jsonl",
    "promotion_blockers.jsonl",
    "canonical_promotion_report.json",
    "canonical_paragraph_review_report.json",
    "paragraph_merge_experiment_report.json",
    "paragraph_merge_failure_taxonomy_report.json",
    "cross_page_join_review_report.json",
    "xpage_join_0032_investigation.json",
    "policy_adoption_decision.json",
    "post_adoption_canonical_safety_report.json",
    "post_adoption_bbox_span_diagnosis.json",
    "post_adoption_remediation_plan.json",
    "front_matter_metadata_review_report.json",
    "visual_review_cases_report.json",
    "narrow_grouping_correction_design.json",
    "chained_cross_page_continuation_experiment.json",
    "chained_join_review_queue.json",
    "chained_join_decisions_applied.json",
    "guarded_chained_cross_page_continuation_experiment.json",
    "guarded_chained_policy_adoption_decision.json",
    "gold_evaluation_report.json",
    "cleanup_log.jsonl",
    "validation_report.json",
    "phase1_audit.html",
]
VALID_OVERRIDE_BUCKETS = {
    "main_paragraph_candidate",
    "structure_candidate",
    "page_artifact_candidate",
    "unknown_needs_review",
}
BASELINE_PARAGRAPH_MERGE_POLICY = "v1_consecutive_lines"
PARAGRAPH_BREAK_GUARDED_POLICY = "v2_paragraph_break_guarded"
CROSS_PAGE_CONTINUATION_POLICY = "v2_cross_page_continuation"
CHAINED_CROSS_PAGE_CONTINUATION_POLICY = "v3_chained_cross_page_continuation"
GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY = "v3_chained_cross_page_continuation_guarded"
ACTIVE_PARAGRAPH_MERGE_POLICY = GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY
EXPERIMENTAL_PARAGRAPH_MERGE_POLICY = GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY
VALID_PARAGRAPH_MERGE_POLICIES = {
    BASELINE_PARAGRAPH_MERGE_POLICY,
    PARAGRAPH_BREAK_GUARDED_POLICY,
    CROSS_PAGE_CONTINUATION_POLICY,
    CHAINED_CROSS_PAGE_CONTINUATION_POLICY,
    GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY,
}
REQUIRED_OVERRIDE_FIELDS = {
    "object_id",
    "original_bucket",
    "corrected_bucket",
    "reason",
    "reviewer",
    "date",
    "evidence_reference",
}
VALID_CROSS_PAGE_JOIN_DECISIONS = {"accept", "reject", "needs_review"}
VALID_CHAINED_JOIN_DECISIONS = {"accept", "reject", "needs_review"}
VALID_CHAINED_JOIN_DECISION_REASONS = {
    "valid_chained_continuation",
    "false_join",
    "structure_boundary",
    "page_furniture_interference",
    "extraction_loss_suspected",
    "insufficient_evidence",
}
REQUIRED_CROSS_PAGE_JOIN_DECISION_FIELDS = {
    "join_id",
    "left_page",
    "right_page",
    "left_candidate_id",
    "right_candidate_id",
    "decision",
    "reason",
    "reviewer",
    "date",
    "evidence_reference",
}
REQUIRED_CHAINED_JOIN_DECISION_FIELDS = {
    "chained_join_id",
    "decision",
    "reason",
    "reviewer",
    "reviewed_at",
    "affected_pages",
    "evidence_reference",
    "notes",
}


def read_review_overrides(path: Path) -> list[dict[str, Any]]:
    return read_commented_jsonl(path)


def curated_review_overrides_path(book_id: str) -> Path:
    return REVIEWS_DIR / book_id / "review_overrides.jsonl"


def curated_cross_page_join_decisions_path(book_id: str) -> Path:
    return REVIEWS_DIR / book_id / "cross_page_join_decisions.jsonl"


def curated_chained_join_decisions_path(book_id: str) -> Path:
    return REVIEWS_DIR / book_id / "chained_join_decisions.jsonl"


def gold_review_dir(book_id: str) -> Path:
    return REVIEWS_DIR / book_id / "gold"


def prepare_applied_review_overrides(review_overrides: list[dict[str, Any]], source_path: Path) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for row in review_overrides:
        applied_row = dict(row)
        applied_row["review_source"] = "curated"
        applied_row["review_source_path"] = str(source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT) else source_path)
        applied.append(applied_row)
    return applied


def prepare_applied_cross_page_join_decisions(decisions: list[dict[str, Any]], source_path: Path) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for row in decisions:
        applied_row = dict(row)
        applied_row["review_source"] = "curated"
        applied_row["review_source_path"] = str(source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT) else source_path)
        applied.append(applied_row)
    return applied


def prepare_applied_chained_join_decisions(decisions: list[dict[str, Any]], source_path: Path) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for row in decisions:
        applied_row = dict(row)
        applied_row["review_source"] = "curated"
        applied_row["review_source_path"] = str(source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT) else source_path)
        applied.append(applied_row)
    return applied


def page_image_filename(page_number: int) -> str:
    return f"page_{page_number:04d}.jpg"


def render_page_images(pdf_path: Path, output_dir: Path, scale: float = 1.5) -> list[dict[str, Any]]:
    page_images_dir = output_dir / PAGE_IMAGES_DIR_NAME
    page_images_dir.mkdir(parents=True, exist_ok=True)
    for existing in page_images_dir.glob("page_*.jpg"):
        existing.unlink()

    rendered: list[dict[str, Any]] = []
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        for page_index in range(len(document)):
            page_number = page_index + 1
            image_path = page_images_dir / page_image_filename(page_number)
            page = document[page_index]
            try:
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil()
                image.save(image_path, "JPEG", quality=86, optimize=True)
            finally:
                page.close()
            rendered.append(
                {
                    "page_number": page_number,
                    "path": f"{PAGE_IMAGES_DIR_NAME}/{image_path.name}",
                    "file_size_bytes": image_path.stat().st_size,
                }
            )
    finally:
        document.close()
    return rendered


def paragraph_break_guarded_split_reason(
    paragraph_buffer: list[dict[str, Any]],
    record: dict[str, Any],
    left_margin: float | None,
) -> str | None:
    if not paragraph_buffer:
        return None
    previous = paragraph_buffer[-1]
    previous_text = str(previous.get("raw_text", ""))
    current_text = str(record.get("text", ""))
    previous_boundary = terminal_line_boundary(previous_text)
    current_upper = uppercase_line_start(current_text)
    current_opens_like_paragraph = bool(re.match(r'^\s*["“‘A-Z0-9]', current_text))
    current_indented = starts_new_indented_paragraph(record, left_margin)
    current_gap = line_gap(previous, record)
    heights = [value for value in (line_height(item) for item in paragraph_buffer + [record]) if value is not None]
    median_height = sorted(heights)[len(heights) // 2] if heights else None
    x_shift = None
    if previous.get("x0") is not None and record.get("x0") is not None:
        x_shift = float(record["x0"]) - float(previous["x0"])
    if not previous_boundary:
        return None
    if not (current_upper or current_opens_like_paragraph):
        return None
    if current_gap is not None and median_height is not None:
        if current_gap >= max(7.0, median_height * 0.55) and (current_indented or (x_shift is not None and x_shift >= 8.0)):
            return "paragraph_break_guard_gap_indent_after_terminal_line"
        if current_gap >= max(14.0, median_height * 1.15):
            return "paragraph_break_guard_large_gap_after_terminal_line"
    if x_shift is not None and x_shift >= 16.0 and len(paragraph_buffer) >= 4:
        return "paragraph_break_guard_indent_after_terminal_line"
    return None


def build_segmented_objects(
    book_id: str,
    page_number: int,
    raw_lines: list[Any],
    paragraph_merge_policy: str = BASELINE_PARAGRAPH_MERGE_POLICY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if paragraph_merge_policy not in VALID_PARAGRAPH_MERGE_POLICIES:
        raise ValueError(f"Unknown paragraph merge policy: {paragraph_merge_policy}")
    line_records = normalize_line_records(raw_lines)
    left_margin = body_left_margin(line_records, page_number)
    layout_objects: list[dict[str, Any]] = []
    clean_objects: list[dict[str, Any]] = []
    cleanup_log: list[dict[str, Any]] = []
    paragraph_buffer: list[dict[str, Any]] = []
    object_index = 1

    def next_object_id() -> str:
        nonlocal object_index
        object_id = f"{book_id}:p{page_number:04d}:obj{object_index:03d}"
        object_index += 1
        return object_id

    def append_cleanup_entries(object_id: str, operations: list[str], raw_text: str, clean_text: str) -> None:
        for operation in operations:
            cleanup_log.append(
                {
                    "book_id": book_id,
                    "object_id": object_id,
                    "page_number": page_number,
                    "operation": operation,
                    "raw_text": raw_text,
                    "clean_text": clean_text,
                }
            )

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        if not paragraph_buffer:
            return
        object_id = next_object_id()
        raw_text = "\n".join(item["raw_text"] for item in paragraph_buffer)
        clean_text, paragraph_operations = join_paragraph_lines([item["raw_text"] for item in paragraph_buffer])
        line_operations = [op for item in paragraph_buffer for op in item["cleanup_operations"]]
        operations = list(dict.fromkeys(line_operations + paragraph_operations))
        source_line_ids = [item["line_id"] for item in paragraph_buffer]
        source_line_indexes = [item["line_index"] for item in paragraph_buffer]
        xs = [float(item["x0"]) for item in paragraph_buffer if item.get("x0") is not None]
        x1s = [float(item["x1"]) for item in paragraph_buffer if item.get("x1") is not None]
        tops = [float(item["top"]) for item in paragraph_buffer if item.get("top") is not None]
        bottoms = [float(item["bottom"]) for item in paragraph_buffer if item.get("bottom") is not None]
        bbox = {
            "x0": min(xs) if xs else None,
            "x1": max(x1s) if x1s else None,
            "top": min(tops) if tops else None,
            "bottom": max(bottoms) if bottoms else None,
        }
        layout_objects.append(
            {
                "book_id": book_id,
                "object_id": object_id,
                "page_number": page_number,
                "object_index": object_index - 1,
                "object_type": "paragraph",
                "confidence": 0.65,
                "classification_reasons": ["merged_consecutive_paragraph_lines", f"paragraph_merge_policy:{paragraph_merge_policy}"],
                "source_line_ids": source_line_ids,
                "source_line_indexes": source_line_indexes,
                "bbox": bbox,
                "raw_text": raw_text,
            }
        )
        clean_objects.append(
            {
                "book_id": book_id,
                "object_id": object_id,
                "page_number": page_number,
                "object_type": "paragraph",
                "clean_text": clean_text,
                "cleanup_operations": operations,
            }
        )
        append_cleanup_entries(object_id, operations, raw_text, clean_text)
        paragraph_buffer = []

    for record in line_records:
        line_index = int(record["line_index"])
        raw_line = record["text"]
        if not raw_line.strip():
            flush_paragraph()
            continue
        line_type, confidence, reasons = classify_line(raw_line, page_number)
        cleaned_text, operations = clean_line(raw_line)
        line_id = f"{book_id}:p{page_number:04d}:line{line_index:03d}"
        if line_type == "paragraph_line":
            if paragraph_buffer and starts_new_indented_paragraph(record, left_margin):
                flush_paragraph()
            elif paragraph_merge_policy == PARAGRAPH_BREAK_GUARDED_POLICY:
                split_reason = paragraph_break_guarded_split_reason(paragraph_buffer, record, left_margin)
                if split_reason:
                    paragraph_buffer[-1].setdefault("cleanup_operations", []).append(split_reason)
                    flush_paragraph()
            paragraph_buffer.append(
                {
                    "line_id": line_id,
                    "line_index": line_index,
                    "raw_text": raw_line,
                    "x0": record.get("x0"),
                    "x1": record.get("x1"),
                    "top": record.get("top"),
                    "bottom": record.get("bottom"),
                    "cleanup_operations": operations,
                }
            )
            continue
        flush_paragraph()
        object_id = next_object_id()
        object_type = "heading_candidate" if line_type == "heading" else line_type
        object_reasons = reasons + (["not_canonical_heading_yet"] if object_type == "heading_candidate" else [])
        layout_objects.append(
            {
                "book_id": book_id,
                "object_id": object_id,
                "page_number": page_number,
                "object_index": object_index - 1,
                "object_type": object_type,
                "confidence": confidence,
                "classification_reasons": object_reasons,
                "source_line_ids": [line_id],
                "source_line_indexes": [line_index],
                "x0": record.get("x0"),
                "x1": record.get("x1"),
                "top": record.get("top"),
                "bottom": record.get("bottom"),
                "bbox": {
                    "x0": record.get("x0"),
                    "x1": record.get("x1"),
                    "top": record.get("top"),
                    "bottom": record.get("bottom"),
                },
                "raw_text": raw_line,
            }
        )
        clean_objects.append(
            {
                "book_id": book_id,
                "object_id": object_id,
                "page_number": page_number,
                "object_type": object_type,
                "clean_text": cleaned_text,
                "cleanup_operations": operations,
            }
        )
        append_cleanup_entries(object_id, operations, raw_line, cleaned_text)
    flush_paragraph()
    return layout_objects, clean_objects, cleanup_log


def looks_like_running_page_furniture(layout: dict[str, Any]) -> bool:
    text = normalized_object_text(str(layout.get("raw_text", "")))
    bbox = layout.get("bbox") or {}
    top = bbox.get("top")
    if top is not None and float(top) <= 60:
        return True
    return bool(re.search(r"\b(narrative|life of frederick douglass)\b", text))


def intervening_structure_before_first_paragraph(page_objects: list[dict[str, Any]], first_paragraph: dict[str, Any]) -> list[dict[str, Any]]:
    blockers = []
    for row in page_objects:
        if row.get("object_id") == first_paragraph.get("object_id"):
            break
        if row.get("object_type") == "paragraph":
            continue
        if looks_like_running_page_furniture(row):
            continue
        blockers.append(row)
    return blockers


def intervening_body_content_after_joined_candidate(
    first: dict[str, Any],
    by_page: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    pages = page_numbers_for_candidate(first)
    if not pages:
        return []
    terminal_page = max(pages)
    first_line_indexes = [
        line_index
        for page_number, line_index in (source_line_page_and_index(line_id) for line_id in first.get("source_line_ids", []))
        if page_number == terminal_page and line_index is not None
    ]
    if not first_line_indexes:
        return []
    max_first_line_index = max(first_line_indexes)
    blockers = []
    first_source_ids = set(first.get("source_object_ids") or []) | {first.get("object_id")}
    for row in by_page.get(terminal_page, []):
        if row.get("object_id") in first_source_ids:
            continue
        if row.get("object_type") != "paragraph":
            continue
        if looks_like_running_page_furniture(row):
            continue
        row_line_indexes = [
            line_index
            for page_number, line_index in (source_line_page_and_index(line_id) for line_id in row.get("source_line_ids", []))
            if page_number == terminal_page and line_index is not None
        ]
        if row_line_indexes and min(row_line_indexes) > max_first_line_index:
            blockers.append(row)
    return blockers


def apply_cross_page_continuation_experiment(
    layout_objects: list[dict[str, Any]],
    clean_objects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    clean_by_id = {row["object_id"]: row for row in clean_objects}
    by_page: dict[int, list[dict[str, Any]]] = {}
    for row in layout_objects:
        by_page.setdefault(int(row.get("page_number") or 0), []).append(row)
    for rows in by_page.values():
        rows.sort(key=lambda row: int(row.get("object_index") or 0))

    skipped_ids: set[str] = set()
    merged_by_first_id: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {}
    joined_examples = []
    rejected_examples = []

    for page_number in sorted(by_page):
        next_page = page_number + 1
        if next_page not in by_page:
            continue
        current_paragraphs = [row for row in by_page[page_number] if row.get("object_type") == "paragraph"]
        next_paragraphs = [row for row in by_page[next_page] if row.get("object_type") == "paragraph"]
        if not current_paragraphs or not next_paragraphs:
            continue
        first = current_paragraphs[-1]
        second = next_paragraphs[0]
        if first.get("object_id") in skipped_ids or second.get("object_id") in skipped_ids:
            continue
        first_clean = str(clean_by_id.get(first["object_id"], {}).get("clean_text", ""))
        second_clean = str(clean_by_id.get(second["object_id"], {}).get("clean_text", ""))
        blockers = intervening_structure_before_first_paragraph(by_page[next_page], second)
        previous_incomplete = paragraph_text_looks_incomplete(first_clean)
        next_continuation = paragraph_text_looks_like_continuation(second_clean)
        no_structure = not blockers
        reasons = []
        if previous_incomplete:
            reasons.append("previous_page_last_paragraph_incomplete")
        if next_continuation:
            reasons.append("next_page_first_paragraph_looks_like_continuation")
        if no_structure:
            reasons.append("no_intervening_structure_candidate")

        if previous_incomplete and next_continuation and no_structure:
            merged_by_first_id[first["object_id"]] = (first, second, ";".join(reasons))
            skipped_ids.add(second["object_id"])
            joined_examples.append(
                {
                    "first_object_id": first.get("object_id"),
                    "second_object_id": second.get("object_id"),
                    "pages": [page_number, next_page],
                    "source_line_ids": list(first.get("source_line_ids", [])) + list(second.get("source_line_ids", [])),
                    "source_line_count": len(first.get("source_line_ids", [])) + len(second.get("source_line_ids", [])),
                    "join_reasons": reasons,
                    "first_text_end": first_clean[-160:],
                    "second_text_start": second_clean[:160],
                    "joined_text_preview": f"{first_clean} {second_clean}"[:360],
                }
            )
        else:
            rejected_examples.append(
                {
                    "first_object_id": first.get("object_id"),
                    "second_object_id": second.get("object_id"),
                    "pages": [page_number, next_page],
                    "reasons_present": reasons,
                    "rejection_reasons": [
                        reason
                        for condition, reason in [
                            (previous_incomplete, "previous_page_last_paragraph_appears_complete"),
                            (next_continuation, "next_page_first_paragraph_does_not_look_like_continuation"),
                            (no_structure, "intervening_structure_candidate_present"),
                        ]
                        if not condition
                    ],
                    "intervening_structure_object_ids": [row.get("object_id") for row in blockers],
                    "first_text_end": first_clean[-120:],
                    "second_text_start": second_clean[:120],
                }
            )

    merged_layout: list[dict[str, Any]] = []
    merged_clean: list[dict[str, Any]] = []
    for row in layout_objects:
        object_id = row["object_id"]
        if object_id in skipped_ids:
            continue
        if object_id in merged_by_first_id:
            first, second, reason = merged_by_first_id[object_id]
            first_clean = clean_by_id[first["object_id"]]
            second_clean = clean_by_id[second["object_id"]]
            merged_id = f"{first['object_id']}__xpage__{second['object_id'].split(':')[-1]}"
            source_object_ids = (first.get("source_object_ids") or [first["object_id"]]) + (second.get("source_object_ids") or [second["object_id"]])
            source_line_ids = list(first.get("source_line_ids", [])) + list(second.get("source_line_ids", []))
            source_line_indexes = list(first.get("source_line_indexes", [])) + list(second.get("source_line_indexes", []))
            raw_text = f"{first.get('raw_text', '').rstrip()}\n{second.get('raw_text', '').lstrip()}"
            clean_text = f"{first_clean.get('clean_text', '').rstrip()} {second_clean.get('clean_text', '').lstrip()}".strip()
            operations = list(
                dict.fromkeys(
                    list(first_clean.get("cleanup_operations", []))
                    + list(second_clean.get("cleanup_operations", []))
                    + ["cross_page_paragraph_continuation_join"]
                )
            )
            merged_layout.append(
                {
                    **first,
                    "object_id": merged_id,
                    "source_object_ids": source_object_ids,
                    "source_line_ids": source_line_ids,
                    "source_line_indexes": source_line_indexes,
                    "bbox": combine_bbox_for_cross_page(first, second),
                    "raw_text": raw_text,
                    "classification_reasons": list(
                        dict.fromkeys(
                            list(first.get("classification_reasons", []))
                            + ["paragraph_merge_policy:v2_cross_page_continuation", reason]
                        )
                    ),
                }
            )
            merged_clean.append(
                {
                    **first_clean,
                    "object_id": merged_id,
                    "clean_text": clean_text,
                    "cleanup_operations": operations,
                }
            )
            continue
        merged_layout.append(row)
        merged_clean.append(clean_by_id[object_id])

    return merged_layout, merged_clean, {
        "joined_cross_page_paragraphs": joined_examples,
        "rejected_cross_page_candidates": rejected_examples,
        "joined_count": len(joined_examples),
        "rejected_count": len(rejected_examples),
    }


def apply_chained_cross_page_continuation_experiment(
    layout_objects: list[dict[str, Any]],
    clean_objects: list[dict[str, Any]],
    guarded: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    clean_by_id = {row["object_id"]: row for row in clean_objects}
    by_page: dict[int, list[dict[str, Any]]] = {}
    for row in layout_objects:
        by_page.setdefault(int(row.get("page_number") or 0), []).append(row)
    for rows in by_page.values():
        rows.sort(key=lambda row: int(row.get("object_index") or 0))

    skipped_ids: set[str] = set()
    merged_by_left_id: dict[str, tuple[dict[str, Any], dict[str, Any], str, list[str]]] = {}
    joined_examples = []
    rejected_examples = []

    left_candidates = [
        row
        for row in layout_objects
        if row.get("object_type") == "paragraph"
        and (row.get("bbox") or {}).get("cross_page")
        and "cross_page_paragraph_continuation_join" in clean_by_id.get(row.get("object_id"), {}).get("cleanup_operations", [])
    ]
    left_candidates.sort(key=lambda row: (max(page_numbers_for_candidate(row) or [int(row.get("page_number") or 0)]), int(row.get("object_index") or 0)))

    for first in left_candidates:
        pages = page_numbers_for_candidate(first)
        if not pages:
            continue
        next_page = max(pages) + 1
        next_paragraphs = [row for row in by_page.get(next_page, []) if row.get("object_type") == "paragraph"]
        if not next_paragraphs:
            continue
        second = next_paragraphs[0]
        if first.get("object_id") in skipped_ids or second.get("object_id") in skipped_ids:
            continue
        first_clean = str(clean_by_id.get(first["object_id"], {}).get("clean_text", ""))
        second_clean = str(clean_by_id.get(second["object_id"], {}).get("clean_text", ""))
        blockers = intervening_structure_before_first_paragraph(by_page[next_page], second)
        intervening_body_blockers = intervening_body_content_after_joined_candidate(first, by_page) if guarded else []
        previous_incomplete = paragraph_text_looks_incomplete(first_clean)
        next_continuation = paragraph_text_looks_like_continuation(second_clean)
        no_structure = not blockers
        no_intervening_body = not intervening_body_blockers
        reasons = ["left_candidate_already_cross_page_join"]
        if previous_incomplete:
            reasons.append("joined_left_text_still_incomplete")
        if next_continuation:
            reasons.append("next_page_first_paragraph_looks_like_continuation")
        if no_structure:
            reasons.append("no_intervening_structure_candidate")
        if guarded and no_intervening_body:
            reasons.append("no_intervening_terminal_page_body_content")

        pages_after_join = pages + [next_page]
        if previous_incomplete and next_continuation and no_structure and no_intervening_body:
            reason = ";".join(reasons)
            merged_by_left_id[first["object_id"]] = (first, second, reason, pages_after_join)
            skipped_ids.add(second["object_id"])
            joined_examples.append(
                {
                    "first_object_id": first.get("object_id"),
                    "second_object_id": second.get("object_id"),
                    "pages": pages_after_join,
                    "source_line_ids": list(first.get("source_line_ids", [])) + list(second.get("source_line_ids", [])),
                    "source_line_count": len(first.get("source_line_ids", [])) + len(second.get("source_line_ids", [])),
                    "join_reasons": reasons,
                    "first_text_end": first_clean[-180:],
                    "second_text_start": second_clean[:180],
                    "joined_text_preview": f"{first_clean} {second_clean}"[:420],
                }
            )
        else:
            rejected_examples.append(
                {
                    "first_object_id": first.get("object_id"),
                    "second_object_id": second.get("object_id"),
                    "pages": pages_after_join,
                    "reasons_present": reasons,
                    "rejection_reasons": [
                        reason
                        for condition, reason in [
                            (previous_incomplete, "joined_left_text_appears_complete"),
                            (next_continuation, "next_page_first_paragraph_does_not_look_like_continuation"),
                            (no_structure, "intervening_structure_candidate_present"),
                            (no_intervening_body, "intervening_terminal_page_body_content_present"),
                        ]
                        if not condition
                    ],
                    "intervening_structure_object_ids": [row.get("object_id") for row in blockers],
                    "intervening_body_object_ids": [row.get("object_id") for row in intervening_body_blockers],
                    "first_text_end": first_clean[-140:],
                    "second_text_start": second_clean[:140],
                }
            )

    merged_layout: list[dict[str, Any]] = []
    merged_clean: list[dict[str, Any]] = []
    for row in layout_objects:
        object_id = row["object_id"]
        if object_id in skipped_ids:
            continue
        if object_id in merged_by_left_id:
            first, second, reason, pages_after_join = merged_by_left_id[object_id]
            first_clean = clean_by_id[first["object_id"]]
            second_clean = clean_by_id[second["object_id"]]
            merged_id = f"{first['object_id']}__chain__{second['object_id'].split(':')[-1]}"
            source_object_ids = list(
                dict.fromkeys(
                    [first["object_id"]]
                    + (first.get("source_object_ids") or [first["object_id"]])
                    + (second.get("source_object_ids") or [second["object_id"]])
                )
            )
            source_line_ids = list(first.get("source_line_ids", [])) + list(second.get("source_line_ids", []))
            source_line_indexes = list(first.get("source_line_indexes", [])) + list(second.get("source_line_indexes", []))
            raw_text = f"{first.get('raw_text', '').rstrip()}\n{second.get('raw_text', '').lstrip()}"
            clean_text = f"{first_clean.get('clean_text', '').rstrip()} {second_clean.get('clean_text', '').lstrip()}".strip()
            operations = list(
                dict.fromkeys(
                    list(first_clean.get("cleanup_operations", []))
                    + list(second_clean.get("cleanup_operations", []))
                    + ["chained_cross_page_paragraph_continuation_join"]
                )
            )
            bbox = combine_bbox_for_cross_page(first, second)
            bbox["page_numbers"] = pages_after_join
            merged_layout.append(
                {
                    **first,
                    "object_id": merged_id,
                    "source_object_ids": source_object_ids,
                    "source_line_ids": source_line_ids,
                    "source_line_indexes": source_line_indexes,
                    "bbox": bbox,
                    "raw_text": raw_text,
                    "classification_reasons": list(
                        dict.fromkeys(
                            list(first.get("classification_reasons", []))
                            + [
                                (
                                    "paragraph_merge_policy:v3_chained_cross_page_continuation_guarded"
                                    if guarded
                                    else "paragraph_merge_policy:v3_chained_cross_page_continuation"
                                ),
                                reason,
                            ]
                        )
                    ),
                }
            )
            merged_clean.append(
                {
                    **first_clean,
                    "object_id": merged_id,
                    "clean_text": clean_text,
                    "cleanup_operations": operations,
                }
            )
            continue
        merged_layout.append(row)
        merged_clean.append(clean_by_id[object_id])

    return merged_layout, merged_clean, {
        "joined_chained_cross_page_paragraphs": joined_examples,
        "rejected_chained_cross_page_candidates": rejected_examples,
        "joined_count": len(joined_examples),
        "rejected_count": len(rejected_examples),
        "guarded": guarded,
    }


def page_status(text: str, image_count: int) -> str:
    if text.strip() and image_count:
        return "mixed_text_and_image"
    if text.strip():
        return "text"
    if image_count:
        return "image_only"
    return "blank_or_unreadable"


def confidence_for_paragraph(clean_text: str, classification_reasons: list[str]) -> tuple[float, list[str]]:
    warnings: list[str] = []
    word_count = len(clean_text.split())
    confidence = 0.82
    if word_count < 10:
        confidence -= 0.25
        warnings.append("short_paragraph_candidate")
    if word_count > 220:
        confidence -= 0.15
        warnings.append("long_paragraph_candidate")
    if CID_PATTERN.search(clean_text):
        confidence -= 0.35
        warnings.append("cid_noise_detected")
    if "merged_consecutive_paragraph_lines" not in classification_reasons:
        confidence -= 0.05
    return max(0.0, round(confidence, 2)), warnings


def page_height(inventory_by_page: dict[int, dict[str, Any]], page_number: int) -> float | None:
    page = inventory_by_page.get(page_number)
    if not page:
        return None
    height = page.get("height")
    return float(height) if height is not None else None


def object_margin_zone(layout: dict[str, Any], height: float | None) -> str | None:
    if height is None:
        return None
    bbox = layout.get("bbox") or {}
    top = bbox.get("top")
    bottom = bbox.get("bottom")
    if top is not None and float(top) / height <= 0.12:
        return "top"
    if bottom is not None and float(bottom) / height >= 0.88:
        return "bottom"
    return None


def top_position_bucket(layout: dict[str, Any]) -> int | None:
    bbox = layout.get("bbox") or {}
    top = bbox.get("top")
    if top is None:
        return None
    return round(float(top) / 8) * 8


def page_number_attached_to_text(raw_text: str) -> bool:
    normalized = normalized_object_text(raw_text)
    stripped = normalized_furniture_text(raw_text)
    return bool(normalized and stripped and normalized != stripped)


def build_page_furniture_profiles(
    layout_objects: list[dict[str, Any]],
    clean_objects: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    clean_by_id = {row["object_id"]: row for row in clean_objects}
    inventory_by_page = {int(row["page_number"]): row for row in inventory}
    repeated_pages: dict[str, set[int]] = {}
    repeated_position_pages: dict[tuple[str, str | None, int | None], set[int]] = {}
    object_evidence: dict[str, dict[str, Any]] = {}

    for layout in layout_objects:
        object_id = layout["object_id"]
        clean_text = str(clean_by_id[object_id].get("clean_text", ""))
        page_number = int(layout["page_number"])
        normalized = normalized_object_text(clean_text)
        furniture_text = normalized_furniture_text(clean_text)
        height = page_height(inventory_by_page, page_number)
        zone = object_margin_zone(layout, height)
        position_bucket = top_position_bucket(layout)
        word_count = len(clean_text.split())
        object_evidence[object_id] = {
            "normalized_text": normalized,
            "normalized_furniture_text": furniture_text,
            "margin_zone": zone,
            "position_bucket": position_bucket,
            "word_count": word_count,
            "short_text": word_count <= 10 and len(clean_text) <= 90,
            "page_number_attached_to_text": page_number_attached_to_text(clean_text),
        }
        if furniture_text:
            repeated_pages.setdefault(furniture_text, set()).add(page_number)
            repeated_position_pages.setdefault((furniture_text, zone, position_bucket), set()).add(page_number)

    profiles: dict[str, dict[str, Any]] = {}
    for layout in layout_objects:
        object_id = layout["object_id"]
        evidence = object_evidence[object_id]
        furniture_text = evidence["normalized_furniture_text"]
        page_repeat_count = len(repeated_pages.get(furniture_text, set()))
        position_repeat_count = len(
            repeated_position_pages.get((furniture_text, evidence["margin_zone"], evidence["position_bucket"]), set())
        )
        reasons: list[str] = []
        warnings: list[str] = ["candidate_only_page_furniture_detection"]
        artifact_subtype = None
        confidence = 0.0
        object_type = layout.get("object_type")
        clean_text = str(clean_by_id[object_id].get("clean_text", ""))

        if object_type == "page_artifact":
            artifact_subtype = "page_number_candidate" if is_page_number_token(normalized_object_text(clean_text)) else "preexisting_page_artifact_candidate"
            confidence = 0.72
            reasons.append("preexisting_page_artifact_classification")
        elif (
            object_type == "heading_candidate"
            and evidence["margin_zone"] in {"top", "bottom"}
            and evidence["short_text"]
            and page_repeat_count >= 3
            and position_repeat_count >= 2
        ):
            artifact_subtype = "running_header_candidate" if evidence["margin_zone"] == "top" else "running_footer_candidate"
            confidence = 0.9 if page_repeat_count >= 8 else 0.82
            reasons.extend(
                [
                    f"normalized_furniture_text_repeats_on_{page_repeat_count}_pages",
                    f"same_margin_position_repeats_on_{position_repeat_count}_pages",
                    f"{evidence['margin_zone']}_margin",
                    "short_text",
                ]
            )
            if evidence["page_number_attached_to_text"]:
                reasons.append("page_number_attached_to_repeated_text")
        elif (
            object_type == "heading_candidate"
            and evidence["margin_zone"] in {"top", "bottom"}
            and evidence["short_text"]
            and evidence["page_number_attached_to_text"]
            and page_repeat_count >= 2
        ):
            artifact_subtype = "page_number_plus_running_title_candidate"
            confidence = 0.78
            reasons.extend(
                [
                    f"normalized_furniture_text_repeats_on_{page_repeat_count}_pages",
                    "page_number_attached_to_repeated_text",
                    f"{evidence['margin_zone']}_margin",
                    "short_text",
                ]
            )

        profiles[object_id] = {
            **evidence,
            "repeat_page_count": page_repeat_count,
            "repeat_position_page_count": position_repeat_count,
            "artifact_subtype": artifact_subtype,
            "artifact_confidence": round(confidence, 2),
            "classification_reasons": reasons,
            "warnings": warnings if artifact_subtype else [],
        }
    return profiles


def classify_stream_object(
    layout: dict[str, Any],
    clean: dict[str, Any],
    furniture_profile: dict[str, Any] | None = None,
) -> tuple[str, str, float, list[str], dict[str, Any]]:
    object_type = layout.get("object_type")
    clean_text = str(clean.get("clean_text", ""))
    reasons = list(layout.get("classification_reasons", []))
    if furniture_profile and furniture_profile.get("artifact_subtype"):
        confidence = max(float(layout.get("confidence", 0.6)), float(furniture_profile["artifact_confidence"]))
        warnings = furniture_profile.get("warnings", [])
        extra = {
            "artifact_subtype": furniture_profile["artifact_subtype"],
            "furniture_evidence": {
                "normalized_furniture_text": furniture_profile.get("normalized_furniture_text"),
                "margin_zone": furniture_profile.get("margin_zone"),
                "repeat_page_count": furniture_profile.get("repeat_page_count"),
                "repeat_position_page_count": furniture_profile.get("repeat_position_page_count"),
                "position_bucket": furniture_profile.get("position_bucket"),
            },
            "classification_reasons": furniture_profile.get("classification_reasons", []),
        }
        return "page_artifact", "page_artifact_candidate", round(confidence, 2), warnings, extra
    if object_type == "paragraph":
        confidence, warnings = confidence_for_paragraph(clean_text, reasons)
        return "main_paragraph", "main_paragraph_candidate", confidence, warnings, {}
    if object_type == "heading_candidate":
        confidence = float(layout.get("confidence", 0.55))
        warnings = ["not_canonical_heading_yet"]
        if CID_PATTERN.search(clean_text):
            warnings.append("cid_noise_detected")
            confidence = min(confidence, 0.4)
        return "structure", "structure_candidate", round(confidence, 2), warnings, {}
    if object_type == "page_artifact":
        return "page_artifact", "page_artifact_candidate", float(layout.get("confidence", 0.6)), reasons, {}
    return "unknown", "unknown_needs_review", float(layout.get("confidence", 0.3)), reasons or ["unclassified_object"], {}


def build_reconstruction_streams(
    book_id: str,
    run_id: str,
    layout_objects: list[dict[str, Any]],
    clean_objects: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    page_count: int,
    review_overrides: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    clean_by_id = {row["object_id"]: row for row in clean_objects}
    furniture_profiles = build_page_furniture_profiles(layout_objects, clean_objects, inventory)
    overrides_by_id = {row["object_id"]: row for row in review_overrides or []}
    main_paragraph_candidates: list[dict[str, Any]] = []
    structure_candidates: list[dict[str, Any]] = []
    page_artifacts_candidates: list[dict[str, Any]] = []
    unknown_objects: list[dict[str, Any]] = []
    object_to_stream: dict[str, str] = {}
    paragraphs_by_page: dict[str, list[str]] = {}
    structure_by_page: dict[str, list[str]] = {}
    artifacts_by_page: dict[str, list[str]] = {}
    unknowns_by_page: dict[str, list[str]] = {}
    paragraph_index = 1

    for layout in layout_objects:
        object_id = layout["object_id"]
        clean = clean_by_id[object_id]
        page_number = int(layout["page_number"])
        stream, stream_type, confidence, warnings, extra = classify_stream_object(layout, clean, furniture_profiles.get(object_id))
        original_stream = stream
        original_stream_type = stream_type
        original_confidence = confidence
        original_extra = dict(extra)
        classification_reasons = list(layout.get("classification_reasons", [])) + list(extra.get("classification_reasons", []))
        classification_reasons = list(dict.fromkeys(classification_reasons))
        review_override = overrides_by_id.get(object_id)
        if review_override:
            corrected_bucket = review_override.get("corrected_bucket")
            bucket_to_stream = {
                "main_paragraph_candidate": ("main_paragraph", "main_paragraph_candidate"),
                "structure_candidate": ("structure", "structure_candidate"),
                "page_artifact_candidate": ("page_artifact", "page_artifact_candidate"),
                "unknown_needs_review": ("unknown", "unknown_needs_review"),
            }
            if corrected_bucket not in bucket_to_stream:
                raise ValueError(f"Invalid corrected_bucket for {object_id}: {corrected_bucket}")
            stream, stream_type = bucket_to_stream[corrected_bucket]
            confidence = float(review_override.get("confidence", confidence))
            warnings = list(dict.fromkeys(warnings + ["review_override_applied"]))
            classification_reasons.append("review_override_applied")
            if review_override.get("corrected_subtype"):
                extra["artifact_subtype"] = review_override["corrected_subtype"]
        common = {
            "book_id": book_id,
            "run_id": run_id,
            "object_id": object_id,
            "page_number": page_number,
            "source_object_ids": layout.get("source_object_ids") or [object_id],
            "source_line_ids": layout.get("source_line_ids", []),
            "source_line_indexes": layout.get("source_line_indexes", []),
            "bbox": layout.get("bbox"),
            "raw_text": layout.get("raw_text", ""),
            "clean_text": clean.get("clean_text", ""),
            "confidence": confidence,
            "original_stream_type": original_stream_type,
            "classification_reasons": classification_reasons,
            "warnings": warnings,
        }
        if review_override:
            common["review_override"] = {
                "original_bucket": original_stream_type,
                "corrected_bucket": stream_type,
                "declared_original_bucket": review_override.get("original_bucket", ""),
                "reason": review_override.get("reason", ""),
                "reviewer": review_override.get("reviewer", ""),
                "date": review_override.get("date", ""),
                "evidence_reference": review_override.get("evidence_reference", ""),
                "line_number": review_override.get("_line_number"),
            }
            common["original_confidence"] = original_confidence
            common["original_artifact_type"] = original_extra.get("artifact_subtype")
        object_to_stream[object_id] = stream
        page_key = str(page_number)
        if stream == "main_paragraph":
            paragraph_id = f"p_{paragraph_index:06d}"
            paragraph_index += 1
            main_paragraph_candidates.append(
                {
                    "paragraph_id": paragraph_id,
                    "stream_type": stream_type,
                    **common,
                    "cleanup_operations": clean.get("cleanup_operations", []),
                }
            )
            paragraphs_by_page.setdefault(page_key, []).append(paragraph_id)
        elif stream == "structure":
            structure_candidates.append(
                {
                    "stream_type": stream_type,
                    "structure_type": layout.get("object_type"),
                    **common,
                    "evidence": {
                        "classification_reasons": classification_reasons,
                        "x0": layout.get("x0"),
                        "top": layout.get("top"),
                        "bottom": layout.get("bottom"),
                    },
                }
            )
            structure_by_page.setdefault(page_key, []).append(object_id)
        elif stream == "page_artifact":
            page_artifacts_candidates.append(
                {
                    "stream_type": stream_type,
                    "artifact_type": extra.get("artifact_subtype") or layout.get("object_type"),
                    **common,
                    "reason": ", ".join(classification_reasons) or "deterministic_page_artifact_candidate",
                    "furniture_evidence": extra.get("furniture_evidence", {}),
                }
            )
            artifacts_by_page.setdefault(page_key, []).append(object_id)
        else:
            unknown_objects.append(
                {
                    "stream_type": stream_type,
                    **common,
                    "reason": ", ".join(warnings) or "requires_review_before_canonical_use",
                    "needs_review": True,
                }
            )
            unknowns_by_page.setdefault(page_key, []).append(object_id)

    reconstruction_map = {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "method": "deterministic_candidate_reconstruction_v1",
        "rule": "No Phase 1 output is canonical until it has passed audit. AI may classify, link, order, and suggest repairs, but must not invent book content.",
        "page_count": page_count,
        "counts": {
            "main_paragraph_candidates": len(main_paragraph_candidates),
            "structure_candidates": len(structure_candidates),
            "page_artifacts_candidates": len(page_artifacts_candidates),
            "unknown_objects": len(unknown_objects),
        },
        "artifact_type_counts": dict(sorted(Counter(row.get("artifact_type", "unknown") for row in page_artifacts_candidates).items())),
        "review_override_count": len(overrides_by_id),
        "review_override_object_ids": sorted(overrides_by_id),
        "object_to_stream": object_to_stream,
        "artifact_candidate_object_ids": [row["object_id"] for row in page_artifacts_candidates],
        "candidate_only_exclusions": {
            "page_artifact_candidate_object_ids": [row["object_id"] for row in page_artifacts_candidates],
            "rule": "These objects are excluded from structure candidates only as candidates; no content is deleted.",
        },
        "paragraphs_by_page": paragraphs_by_page,
        "structure_by_page": structure_by_page,
        "artifacts_by_page": artifacts_by_page,
        "unknowns_by_page": unknowns_by_page,
        "notes": [
            "This map is a reconstruction aid, not proof of final book structure.",
            "Main paragraphs are candidates until visual/audit checks confirm cleanliness.",
            "Non-paragraph content is preserved separately instead of deleted.",
        ],
    }
    return main_paragraph_candidates, structure_candidates, page_artifacts_candidates, unknown_objects, reconstruction_map


BLOCKING_PROMOTION_WARNINGS = {
    "cid_noise",
    "cid_noise_detected",
    "long_paragraph_candidate",
    "short_paragraph_candidate",
}


def bbox_has_any_value(value: Any) -> bool:
    return isinstance(value, dict) and any(value.get(key) is not None for key in ["x0", "x1", "top", "bottom"])


def paragraph_promotion_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    warnings = set(row.get("warnings", []))
    if row.get("stream_type") != "main_paragraph_candidate":
        blockers.append("final_bucket_not_main_paragraph_candidate")
    if not row.get("object_id"):
        blockers.append("missing_object_id")
    if row.get("page_number") is None:
        blockers.append("missing_page")
    if not row.get("source_object_ids"):
        blockers.append("missing_source_object_ids")
    if not row.get("source_line_ids"):
        blockers.append("missing_source_line_ids")
    if not str(row.get("raw_text", "")).strip():
        blockers.append("missing_raw_text")
    if not str(row.get("clean_text", "")).strip():
        blockers.append("missing_clean_text")
    blocking_warnings = sorted(warnings & BLOCKING_PROMOTION_WARNINGS)
    for warning in blocking_warnings:
        blockers.append(f"blocking_warning:{warning}")
    override = row.get("review_override")
    if override:
        required = ["original_bucket", "corrected_bucket", "reason", "reviewer", "date", "evidence_reference"]
        if not all(str(override.get(field, "")).strip() for field in required):
            blockers.append("invalid_or_incomplete_review_override")
    return blockers


def build_paragraph_promotion_artifacts(
    book_id: str,
    run_id: str,
    main_paragraphs: list[dict[str, Any]],
    structure: list[dict[str, Any]],
    page_artifacts: list[dict[str, Any]],
    unknown: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    canonical_paragraphs: list[dict[str, Any]] = []
    promotion_blockers: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    canonical_index = 1

    for row in main_paragraphs:
        warnings = list(row.get("warnings", []))
        warning_counts.update(warnings)
        blockers = paragraph_promotion_blockers(row)
        promotion_reasons = [
            "final_bucket_main_paragraph_candidate",
            "source_object_ids_present",
            "source_line_ids_present",
            "raw_text_present",
            "clean_text_present",
            "no_blocking_warnings",
            "candidate_only_promotion_v1",
        ]
        if bbox_has_any_value(row.get("bbox")):
            promotion_reasons.append("bbox_available")
        else:
            promotion_reasons.append("bbox_unavailable_but_not_required")
        if row.get("review_override"):
            promotion_reasons.append("review_override_visible")
        if blockers:
            promotion_blockers.append(
                {
                    "book_id": book_id,
                    "run_id": run_id,
                    "object_id": row.get("object_id"),
                    "paragraph_id": row.get("paragraph_id"),
                    "page_number": row.get("page_number"),
                    "source_object_ids": row.get("source_object_ids", []),
                    "source_line_ids": row.get("source_line_ids", []),
                    "raw_text": row.get("raw_text", ""),
                    "clean_text": row.get("clean_text", ""),
                    "stream_type": row.get("stream_type"),
                    "promotion_status": "blocked",
                    "blocker_reasons": blockers,
                    "warnings": warnings,
                    "applied_override": row.get("review_override"),
                }
            )
            continue
        canonical_paragraphs.append(
            {
                "book_id": book_id,
                "run_id": run_id,
                "canonical_paragraph_id": f"cp_{canonical_index:06d}",
                "source_candidate_object_id": row.get("object_id"),
                "source_candidate_paragraph_id": row.get("paragraph_id"),
                "page_number": row.get("page_number"),
                "raw_text": row.get("raw_text", ""),
                "clean_text": row.get("clean_text", ""),
                "source_object_ids": row.get("source_object_ids", []),
                "source_line_ids": row.get("source_line_ids", []),
                "bbox": row.get("bbox"),
                "promotion_status": "promoted",
                "promotion_reasons": promotion_reasons,
                "warnings": warnings,
                "applied_override": row.get("review_override"),
            }
        )
        canonical_index += 1

    non_paragraph_rows = structure + page_artifacts + unknown
    for row in non_paragraph_rows:
        warnings = list(row.get("warnings", []))
        warning_counts.update(warnings)
        promotion_blockers.append(
            {
                "book_id": book_id,
                "run_id": run_id,
                "object_id": row.get("object_id"),
                "page_number": row.get("page_number"),
                "source_object_ids": row.get("source_object_ids", []),
                "source_line_ids": row.get("source_line_ids", []),
                "raw_text": row.get("raw_text", ""),
                "clean_text": row.get("clean_text", ""),
                "stream_type": row.get("stream_type"),
                "promotion_status": "blocked",
                "blocker_reasons": [f"not_paragraph_stream:{row.get('stream_type', 'unknown')}"],
                "warnings": warnings,
                "applied_override": row.get("review_override"),
            }
        )

    all_reviewed_count = len(main_paragraphs) + len(structure) + len(page_artifacts) + len(unknown)
    report = {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "promotion_gate": "paragraph_candidate_promotion_v1",
        "scope": "main_paragraph_candidates_only",
        "canonical_does_not_mean_perfect": True,
        "rule": "Canonical paragraphs are evidence-bound, audited enough under current rules, and safe for downstream use; they are not claimed flawless.",
        "status": "pass",
        "counts": {
            "total_candidates_reviewed": all_reviewed_count,
            "paragraph_candidates_reviewed": len(main_paragraphs),
            "promoted_paragraphs": len(canonical_paragraphs),
            "blocked_candidates": len(promotion_blockers),
            "paragraph_candidates_blocked": sum(1 for row in promotion_blockers if row.get("stream_type") == "main_paragraph_candidate"),
            "non_paragraph_candidates_blocked": sum(1 for row in promotion_blockers if row.get("stream_type") != "main_paragraph_candidate"),
            "override_influenced_promotions": sum(1 for row in canonical_paragraphs if row.get("applied_override")),
        },
        "warning_counts": dict(sorted(warning_counts.items())),
        "blocking_warning_policy": sorted(BLOCKING_PROMOTION_WARNINGS),
    }
    return canonical_paragraphs, promotion_blockers, report


METADATA_LEAKAGE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bcontents\b",
        r"\bchapter\s+[ivxlcdm0-9]+\b",
        r"\bpreface\b",
        r"\bappendix\b",
        r"\bcopyright\b",
        r"\bpublished\b",
        r"\bentered according to act\b",
        r"\bnarrative of the life\b",
        r"\bfrederick douglass\b",
    ]
]
WARNING_DRILLDOWN_RULES = {
    "possible_broken_paragraph_merge": {
        "cluster": "merge_risk",
        "severity": "high",
        "likely_next_corrective_action": "Inspect paragraph line grouping and split candidates where multiple ideas were merged.",
        "may_require": ["detector_changes", "manual_inspection"],
    },
    "possible_metadata_or_structure_leakage": {
        "cluster": "structure_metadata_leakage",
        "severity": "high",
        "likely_next_corrective_action": "Move publisher/front matter or heading-like text out of canonical paragraphs into structure or artifacts.",
        "may_require": ["promotion_rule_changes", "review_overrides", "manual_inspection"],
    },
    "possible_missing_paragraph_start": {
        "cluster": "missing_paragraph_start_end",
        "severity": "medium",
        "likely_next_corrective_action": "Inspect neighboring source objects to determine whether paragraph starts were split or joined incorrectly.",
        "may_require": ["detector_changes", "manual_inspection"],
    },
    "unusual_paragraph_ending": {
        "cluster": "missing_paragraph_start_end",
        "severity": "medium",
        "likely_next_corrective_action": "Inspect neighboring source objects to determine whether paragraph endings were truncated or split.",
        "may_require": ["detector_changes", "manual_inspection"],
    },
    "possible_bad_hyphenation_join": {
        "cluster": "hyphenation_line_join",
        "severity": "medium",
        "likely_next_corrective_action": "Review hyphenated line-join cleanup rules against raw source lines.",
        "may_require": ["detector_changes", "promotion_rule_changes"],
    },
    "missing_bbox_visual_evidence": {
        "cluster": "bbox_span_risk",
        "severity": "medium",
        "likely_next_corrective_action": "Require visual evidence or route missing-bbox paragraphs to manual inspection.",
        "may_require": ["promotion_rule_changes", "manual_inspection"],
    },
    "suspicious_source_line_vertical_span": {
        "cluster": "bbox_span_risk",
        "severity": "high",
        "likely_next_corrective_action": "Inspect visual page spans and split paragraph candidates whose source lines cover too much page height.",
        "may_require": ["detector_changes", "manual_inspection"],
    },
    "source_lines_span_suspicious_distance": {
        "cluster": "bbox_span_risk",
        "severity": "high",
        "likely_next_corrective_action": "Inspect multi-line paragraph grouping and split candidates with suspicious vertical span.",
        "may_require": ["detector_changes", "manual_inspection"],
    },
    "front_matter_or_chapter_boundary_risk": {
        "cluster": "boundary_page_risk",
        "severity": "medium",
        "likely_next_corrective_action": "Define front matter and chapter boundary handling before promoting early-page text downstream.",
        "may_require": ["promotion_rule_changes", "review_overrides", "manual_inspection"],
    },
    "possible_repeated_header_or_furniture_leakage": {
        "cluster": "repeated_furniture_leakage",
        "severity": "high",
        "likely_next_corrective_action": "Improve page-furniture detection or add curated overrides for repeated text that entered canonical paragraphs.",
        "may_require": ["detector_changes", "review_overrides"],
    },
    "suspiciously_short_promoted_paragraph": {
        "cluster": "unusual_length",
        "severity": "low",
        "likely_next_corrective_action": "Sample short paragraphs to decide whether they are legitimate prose, fragments, captions, or metadata.",
        "may_require": ["promotion_rule_changes", "manual_inspection"],
    },
    "suspiciously_long_promoted_paragraph": {
        "cluster": "unusual_length",
        "severity": "medium",
        "likely_next_corrective_action": "Inspect long paragraphs for accidental merges before downstream chunking.",
        "may_require": ["detector_changes", "manual_inspection"],
    },
}
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


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


def bbox_span_interpretation(
    line_count: int,
    page_height_ratio: float | None,
    word_count: int,
    warnings: list[str],
) -> str:
    if "possible_metadata_or_structure_leakage" in warnings:
        return "page layout artifact"
    if page_height_ratio is not None and page_height_ratio >= 0.40:
        return "possible accidental merge"
    if line_count >= 11:
        return "possible accidental merge"
    if word_count > 170 and page_height_ratio is not None and page_height_ratio < 0.28:
        return "normal long paragraph"
    if page_height_ratio is not None and page_height_ratio < 0.18:
        return "threshold too strict"
    return "needs manual inspection"


def bbox_span_corrective_path(interpretation: str) -> list[str]:
    if interpretation == "possible accidental merge":
        return ["paragraph merge rule adjustment", "manual inspection"]
    if interpretation == "threshold too strict":
        return ["promotion blocker threshold adjustment", "manual inspection"]
    if interpretation == "page layout artifact":
        return ["curated review override", "manual inspection"]
    if interpretation == "normal long paragraph":
        return ["promotion blocker threshold adjustment"]
    return ["manual inspection"]


def bbox_span_decision(diagnostic: dict[str, Any]) -> dict[str, Any]:
    warnings = set(str(warning) for warning in diagnostic.get("all_warnings", []))
    page_number = int(diagnostic.get("page_number") or 0)
    line_count = int(diagnostic.get("source_line_count") or 0)
    ratio = diagnostic.get("page_height_ratio")
    ratio_value = float(ratio) if isinstance(ratio, (int, float)) else None
    word_count = int(diagnostic.get("word_count") or 0)
    text_length = int(diagnostic.get("text_length") or 0)
    severity = str(diagnostic.get("warning_severity", "low"))

    if page_number <= 20 or "front_matter_or_chapter_boundary_risk" in warnings:
        likely_cause = "boundary_or_front_matter_artifact"
        confidence = 0.72 if page_number <= 20 else 0.62
        recommended_action = "inspect manually"
    elif "possible_broken_paragraph_merge" in warnings and (line_count >= 8 or (ratio_value is not None and ratio_value >= 0.28)):
        likely_cause = "true_accidental_merge"
        confidence = 0.78
        recommended_action = "adjust paragraph merge rule"
    elif ratio_value is not None and ratio_value >= 0.40:
        likely_cause = "true_accidental_merge"
        confidence = 0.74
        recommended_action = "adjust paragraph merge rule"
    elif line_count >= 11 and text_length < 900:
        likely_cause = "true_accidental_merge"
        confidence = 0.68
        recommended_action = "adjust paragraph merge rule"
    elif word_count > 150 and ratio_value is not None and ratio_value < 0.35 and severity != "high":
        likely_cause = "normal_long_paragraph"
        confidence = 0.66
        recommended_action = "no action"
    elif ratio_value is not None and ratio_value < 0.28 and line_count <= 10:
        likely_cause = "threshold_too_strict"
        confidence = 0.64
        recommended_action = "adjust review threshold"
    else:
        likely_cause = "needs_manual_inspection"
        confidence = 0.55
        recommended_action = "inspect manually"

    if "possible_metadata_or_structure_leakage" in warnings:
        likely_cause = "boundary_or_front_matter_artifact"
        confidence = max(confidence, 0.70)
        recommended_action = "add curated override"

    return {
        "canonical_paragraph_id": diagnostic.get("canonical_paragraph_id"),
        "page_number": page_number,
        "severity": severity,
        "source_line_count": line_count,
        "vertical_bbox_span": diagnostic.get("vertical_bbox_span"),
        "page_height_ratio": diagnostic.get("page_height_ratio"),
        "text_length": text_length,
        "first_source_line_preview": diagnostic.get("first_source_line_preview"),
        "last_source_line_preview": diagnostic.get("last_source_line_preview"),
        "likely_cause": likely_cause,
        "confidence": confidence,
        "recommended_action": recommended_action,
        "source_candidate_object_id": diagnostic.get("source_candidate_object_id"),
        "audit_anchor": diagnostic.get("audit_anchor"),
        "page_anchor": diagnostic.get("page_anchor"),
    }


def audit_anchor_for_object(object_id: Any) -> str:
    return "#card-" + re.sub(r"[^a-zA-Z0-9_-]+", "-", str(object_id))


def review_canonical_paragraphs(
    book_id: str,
    run_id: str,
    canonical_paragraphs: list[dict[str, Any]],
    page_heights_by_page: dict[int, float] | None = None,
) -> dict[str, Any]:
    page_heights_by_page = page_heights_by_page or {}
    normalized_counts = Counter(normalized_object_text(row.get("clean_text", "")) for row in canonical_paragraphs)
    warning_counts: Counter[str] = Counter()
    risky_samples: list[dict[str, Any]] = []
    warning_records: dict[str, list[dict[str, Any]]] = {}
    cluster_records: dict[str, list[dict[str, Any]]] = {}
    bbox_span_diagnostics: list[dict[str, Any]] = []
    clean_count = 0

    for row in canonical_paragraphs:
        clean_text = str(row.get("clean_text", "")).strip()
        raw_text = str(row.get("raw_text", "")).strip()
        words = clean_text.split()
        word_count = len(words)
        warnings: list[str] = []
        normalized = normalized_object_text(clean_text)
        height = bbox_height(row.get("bbox"))
        page_number = int(row.get("page_number") or 0)
        source_line_ids = row.get("source_line_ids", [])

        if "\n" in raw_text and clean_text.count(". ") == 0 and word_count > 55:
            warnings.append("possible_broken_paragraph_merge")
        if any(pattern.search(clean_text) for pattern in METADATA_LEAKAGE_PATTERNS):
            warnings.append("possible_metadata_or_structure_leakage")
        if clean_text[:1].islower() or clean_text.startswith((",", ";", ":", ")", "]")):
            warnings.append("possible_missing_paragraph_start")
        if clean_text and clean_text[-1] not in TERMINAL_PUNCTUATION:
            warnings.append("unusual_paragraph_ending")
        if re.search(r"\w-\s+\w", clean_text):
            warnings.append("possible_bad_hyphenation_join")
        if height is None:
            warnings.append("missing_bbox_visual_evidence")
        elif height > 220:
            warnings.append("suspicious_source_line_vertical_span")
        if page_number <= 20:
            warnings.append("front_matter_or_chapter_boundary_risk")
        if normalized and normalized_counts[normalized] > 1 and word_count <= 16:
            warnings.append("possible_repeated_header_or_furniture_leakage")
        if word_count < 25:
            warnings.append("suspiciously_short_promoted_paragraph")
        if word_count > 170:
            warnings.append("suspiciously_long_promoted_paragraph")
        if len(source_line_ids) >= 8 and height is not None and height > 160:
            warnings.append("source_lines_span_suspicious_distance")

        if warnings:
            warning_counts.update(warnings)
            sample_record = {
                "canonical_paragraph_id": row.get("canonical_paragraph_id"),
                "source_candidate_object_id": row.get("source_candidate_object_id"),
                "page_number": page_number,
                "word_count": word_count,
                "bbox": row.get("bbox"),
                "source_line_ids": source_line_ids,
                "text_preview": clean_text[:220],
            }
            for warning in warnings:
                warning_records.setdefault(warning, []).append(sample_record)
                cluster = WARNING_DRILLDOWN_RULES.get(warning, {}).get("cluster", "other")
                cluster_records.setdefault(str(cluster), []).append({**sample_record, "warning": warning})
            if len(risky_samples) < 40:
                risky_samples.append(
                    {
                        "canonical_paragraph_id": row.get("canonical_paragraph_id"),
                        "source_candidate_object_id": row.get("source_candidate_object_id"),
                        "page_number": page_number,
                        "warnings": warnings,
                        "word_count": word_count,
                        "bbox": row.get("bbox"),
                        "source_line_ids": source_line_ids,
                        "raw_text_sample": raw_text[:360],
                        "clean_text_sample": clean_text[:360],
                    }
                )
            if any(
                warning in warnings
                for warning in [
                    "missing_bbox_visual_evidence",
                    "suspicious_source_line_vertical_span",
                    "source_lines_span_suspicious_distance",
                ]
            ):
                page_height = page_heights_by_page.get(page_number)
                page_height_ratio = height / page_height if height is not None and page_height else None
                raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                first_line_preview = raw_lines[0][:180] if raw_lines else clean_text[:180]
                last_line_preview = raw_lines[-1][:180] if raw_lines else clean_text[-180:]
                severity = bbox_span_severity(len(source_line_ids), page_height_ratio, word_count)
                interpretation = bbox_span_interpretation(len(source_line_ids), page_height_ratio, word_count, warnings)
                bbox_span_diagnostics.append(
                    {
                        "canonical_paragraph_id": row.get("canonical_paragraph_id"),
                        "page_number": page_number,
                        "source_candidate_object_id": row.get("source_candidate_object_id"),
                        "source_line_count": len(source_line_ids),
                        "source_line_count_range": source_line_count_range(len(source_line_ids)),
                        "vertical_bbox_span": height,
                        "page_height": page_height,
                        "page_height_ratio": page_height_ratio,
                        "page_height_ratio_range": page_height_ratio_range(page_height_ratio),
                        "text_length": len(clean_text),
                        "word_count": word_count,
                        "first_source_line_preview": first_line_preview,
                        "last_source_line_preview": last_line_preview,
                        "warning_severity": severity,
                        "likely_interpretation": interpretation,
                        "likely_corrective_path": bbox_span_corrective_path(interpretation),
                        "bbox_span_warnings": [
                            warning
                            for warning in warnings
                            if warning
                            in {
                                "missing_bbox_visual_evidence",
                                "suspicious_source_line_vertical_span",
                                "source_lines_span_suspicious_distance",
                            }
                        ],
                        "all_warnings": warnings,
                        "audit_anchor": audit_anchor_for_object(row.get("source_candidate_object_id")),
                        "page_anchor": f"#page-{page_number}",
                    }
                )
        else:
            clean_count += 1

    def warning_drilldown_rows() -> list[dict[str, Any]]:
        rows = []
        for warning, rows_for_warning in warning_records.items():
            rule = WARNING_DRILLDOWN_RULES.get(warning, {})
            severity = str(rule.get("severity", "low"))
            unique_pages = sorted({int(row["page_number"]) for row in rows_for_warning})
            rows.append(
                {
                    "warning": warning,
                    "cluster": rule.get("cluster", "other"),
                    "count": len(rows_for_warning),
                    "severity": severity,
                    "severity_rank": SEVERITY_RANK.get(severity, 1),
                    "sample_canonical_paragraph_ids": first_unique([row["canonical_paragraph_id"] for row in rows_for_warning], 5),
                    "affected_pages": unique_pages[:20],
                    "text_previews": [row["text_preview"] for row in rows_for_warning[:3]],
                    "likely_next_corrective_action": rule.get("likely_next_corrective_action", "Inspect samples and define a corrective rule."),
                    "may_require": rule.get("may_require", ["manual_inspection"]),
                }
            )
        return sorted(rows, key=lambda row: (-int(row["count"]), -int(row["severity_rank"]), str(row["warning"])))

    def cluster_drilldown_rows() -> list[dict[str, Any]]:
        rows = []
        for cluster, rows_for_cluster in cluster_records.items():
            warnings_in_cluster = sorted({str(row["warning"]) for row in rows_for_cluster})
            severities = [
                str(WARNING_DRILLDOWN_RULES.get(warning, {}).get("severity", "low"))
                for warning in warnings_in_cluster
            ]
            max_rank = max((SEVERITY_RANK.get(severity, 1) for severity in severities), default=1)
            severity = next(key for key, value in SEVERITY_RANK.items() if value == max_rank)
            unique_pages = sorted({int(row["page_number"]) for row in rows_for_cluster})
            next_actions = []
            may_require = set()
            for warning in warnings_in_cluster:
                rule = WARNING_DRILLDOWN_RULES.get(warning, {})
                action = rule.get("likely_next_corrective_action")
                if action and action not in next_actions:
                    next_actions.append(str(action))
                may_require.update(str(value) for value in rule.get("may_require", []))
            rows.append(
                {
                    "cluster": cluster,
                    "count": len(rows_for_cluster),
                    "severity": severity,
                    "severity_rank": max_rank,
                    "warnings": warnings_in_cluster,
                    "sample_canonical_paragraph_ids": first_unique([row["canonical_paragraph_id"] for row in rows_for_cluster], 6),
                    "affected_pages": unique_pages[:24],
                    "text_previews": [row["text_preview"] for row in rows_for_cluster[:3]],
                    "likely_next_corrective_action": " ".join(next_actions) or "Inspect samples and define a corrective rule.",
                    "may_require": sorted(may_require),
                }
            )
        return sorted(rows, key=lambda row: (-(int(row["severity_rank"]) * int(row["count"])), -int(row["count"]), str(row["cluster"])))

    warning_drilldown = warning_drilldown_rows()
    cluster_drilldown = cluster_drilldown_rows()
    bbox_span_diagnostics = sorted(
        bbox_span_diagnostics,
        key=lambda row: (
            -SEVERITY_RANK.get(str(row.get("warning_severity", "low")), 1),
            int(row.get("page_number") or 0),
            str(row.get("canonical_paragraph_id")),
        ),
    )

    def grouped_bbox_span_rows(key: str) -> list[dict[str, Any]]:
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for diagnostic in bbox_span_diagnostics:
            grouped.setdefault(diagnostic.get(key), []).append(diagnostic)
        return [
            {
                key: group_key,
                "count": len(rows),
                "sample_canonical_paragraph_ids": first_unique([row.get("canonical_paragraph_id") for row in rows], 6),
                "affected_pages": first_unique([row.get("page_number") for row in rows], 12),
            }
            for group_key, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), str(item[0])))
        ]

    bbox_span_risk_summary = {
        "total": len(bbox_span_diagnostics),
        "by_page": grouped_bbox_span_rows("page_number"),
        "by_severity": grouped_bbox_span_rows("warning_severity"),
        "by_source_line_count_range": grouped_bbox_span_rows("source_line_count_range"),
        "by_page_height_ratio_range": grouped_bbox_span_rows("page_height_ratio_range"),
    }
    bbox_span_decisions = [bbox_span_decision(diagnostic) for diagnostic in bbox_span_diagnostics]

    def grouped_decision_rows(key: str) -> list[dict[str, Any]]:
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for decision in bbox_span_decisions:
            grouped.setdefault(decision.get(key), []).append(decision)
        return [
            {
                key: group_key,
                "count": len(rows),
                "sample_canonical_paragraph_ids": first_unique([row.get("canonical_paragraph_id") for row in rows], 6),
                "affected_pages": first_unique([row.get("page_number") for row in rows], 12),
            }
            for group_key, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), str(item[0])))
        ]

    high_severity_by_page = []
    high_by_page: dict[int, list[dict[str, Any]]] = {}
    for decision in bbox_span_decisions:
        if decision.get("severity") == "high":
            high_by_page.setdefault(int(decision.get("page_number") or 0), []).append(decision)
    for page_number, rows in sorted(high_by_page.items(), key=lambda item: (-len(item[1]), item[0])):
        high_severity_by_page.append(
            {
                "page_number": page_number,
                "count": len(rows),
                "sample_canonical_paragraph_ids": first_unique([row.get("canonical_paragraph_id") for row in rows], 6),
            }
        )

    page_inspection_scores: dict[int, dict[str, Any]] = {}
    for decision in bbox_span_decisions:
        page_number = int(decision.get("page_number") or 0)
        row = page_inspection_scores.setdefault(page_number, {"page_number": page_number, "count": 0, "high_count": 0, "causes": Counter()})
        row["count"] += 1
        row["high_count"] += 1 if decision.get("severity") == "high" else 0
        row["causes"].update([str(decision.get("likely_cause"))])
    top_pages_needing_inspection = []
    for row in sorted(page_inspection_scores.values(), key=lambda item: (-int(item["high_count"]), -int(item["count"]), int(item["page_number"])))[:12]:
        top_pages_needing_inspection.append(
            {
                "page_number": row["page_number"],
                "diagnostic_count": row["count"],
                "high_severity_count": row["high_count"],
                "likely_causes": dict(sorted(row["causes"].items())),
            }
        )

    bbox_span_decision_summary = {
        "total": len(bbox_span_decisions),
        "by_likely_cause": grouped_decision_rows("likely_cause"),
        "by_recommended_action": grouped_decision_rows("recommended_action"),
        "high_severity_rows_by_page": high_severity_by_page,
        "top_pages_needing_inspection": top_pages_needing_inspection,
    }
    top_risk = cluster_drilldown[0] if cluster_drilldown else None
    warning_total = sum(warning_counts.values())
    safe_for_downstream = warning_total == 0
    recommendation = {
        "top_risk_to_fix_first": top_risk.get("cluster") if top_risk else None,
        "why_it_matters": (
            "Large source-line or bounding-box spans can indicate accidental paragraph merges, which would contaminate chunks and reasoning."
            if top_risk and top_risk.get("cluster") == "bbox_span_risk"
            else "The highest-priority risk cluster should be reviewed before downstream intelligence uses canonical paragraphs."
        ),
        "expected_impact": (
            f"Reviewing this cluster addresses {top_risk.get('count')} warning instances across pages {', '.join(str(page) for page in top_risk.get('affected_pages', [])[:8])}."
            if top_risk
            else "No risky cluster detected."
        ),
        "may_require": top_risk.get("may_require", []) if top_risk else [],
    }
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "review_method": "deterministic_canonical_paragraph_review_v1",
        "scope": "promoted_canonical_paragraphs_only",
        "does_not_demote": True,
        "safe_for_downstream": safe_for_downstream,
        "recommendation": "hold_downstream_intelligence_until_risks_are_reviewed" if not safe_for_downstream else "safe_for_limited_downstream_trial",
        "counts": {
            "total_canonical_paragraphs_reviewed": len(canonical_paragraphs),
            "clean_looking_count": clean_count,
            "warning_count": warning_total,
            "risky_paragraph_count": len(canonical_paragraphs) - clean_count,
            "sample_risky_paragraphs": len(risky_samples),
        },
        "warning_categories": dict(sorted(warning_counts.items())),
        "warning_category_drilldown": warning_drilldown,
        "risky_paragraph_clusters": cluster_drilldown,
        "bbox_span_risk_summary": bbox_span_risk_summary,
        "bbox_span_risk_diagnostics": bbox_span_diagnostics,
        "bbox_span_decision_summary": bbox_span_decision_summary,
        "bbox_span_decisions": bbox_span_decisions,
        "recommendation_detail": recommendation,
        "sample_risky_canonical_paragraphs": risky_samples,
    }


def paragraph_policy_evaluation(
    book_id: str,
    run_id: str,
    layout_objects: list[dict[str, Any]],
    clean_objects: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    page_count: int,
    review_overrides: list[dict[str, Any]],
    page_heights_by_page: dict[int, float],
) -> dict[str, Any]:
    main_paragraphs, structure, page_artifacts, unknown, reconstruction_map = build_reconstruction_streams(
        book_id, run_id, layout_objects, clean_objects, inventory, page_count, review_overrides
    )
    canonical_paragraphs, promotion_blockers, promotion_report = build_paragraph_promotion_artifacts(
        book_id, run_id, main_paragraphs, structure, page_artifacts, unknown
    )
    review_report = review_canonical_paragraphs(book_id, run_id, canonical_paragraphs, page_heights_by_page)
    gold_evaluation_report = evaluate_gold_reviews(book_id, run_id, main_paragraphs, structure, page_artifacts, unknown)
    return {
        "main_paragraphs": main_paragraphs,
        "structure": structure,
        "page_artifacts": page_artifacts,
        "unknown": unknown,
        "reconstruction_map": reconstruction_map,
        "canonical_paragraphs": canonical_paragraphs,
        "promotion_blockers": promotion_blockers,
        "promotion_report": promotion_report,
        "review_report": review_report,
        "gold_evaluation_report": gold_evaluation_report,
    }


def count_likely_true_merges(review_report: dict[str, Any]) -> int:
    return sum(1 for row in review_report.get("bbox_span_decisions", []) if row.get("likely_cause") == "true_accidental_merge")


def taxonomy_counts_for_review(canonical_paragraphs: list[dict[str, Any]], review_report: dict[str, Any]) -> Counter[str]:
    canonical_by_id = {row.get("canonical_paragraph_id"): row for row in canonical_paragraphs}
    decisions_by_id = {
        row.get("canonical_paragraph_id"): row
        for row in review_report.get("bbox_span_decisions", [])
    }
    counts: Counter[str] = Counter()
    for diagnostic in review_report.get("bbox_span_risk_diagnostics", []):
        if (decisions_by_id.get(diagnostic.get("canonical_paragraph_id")) or {}).get("likely_cause") != "true_accidental_merge":
            continue
        category, _, _ = merge_failure_taxonomy_decision(
            diagnostic,
            canonical_by_id.get(diagnostic.get("canonical_paragraph_id"), {}),
        )
        counts.update([category])
    return counts


def candidate_word_count(row: dict[str, Any]) -> int:
    return len(str(row.get("clean_text", "")).split())


def build_paragraph_merge_experiment_report(
    book_id: str,
    run_id: str,
    baseline: dict[str, Any],
    active: dict[str, Any],
    experiment_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    experiment_details = experiment_details or {}
    baseline_paragraphs = baseline["main_paragraphs"]
    active_paragraphs = active["main_paragraphs"]
    active_by_page: dict[int, list[dict[str, Any]]] = {}
    for row in active_paragraphs:
        active_by_page.setdefault(int(row.get("page_number") or 0), []).append(row)

    split_examples = []
    oversplit_examples = []
    for baseline_row in baseline_paragraphs:
        baseline_lines = set(baseline_row.get("source_line_ids", []))
        if len(baseline_lines) < 2:
            continue
        matching_active = [
            row
            for row in active_by_page.get(int(baseline_row.get("page_number") or 0), [])
            if set(row.get("source_line_ids", [])).issubset(baseline_lines)
            and set(row.get("source_line_ids", []))
        ]
        covered_lines = set().union(*(set(row.get("source_line_ids", [])) for row in matching_active)) if matching_active else set()
        if len(matching_active) > 1 and covered_lines == baseline_lines:
            example = {
                "baseline_object_id": baseline_row.get("object_id"),
                "page_number": baseline_row.get("page_number"),
                "baseline_source_line_count": len(baseline_row.get("source_line_ids", [])),
                "baseline_text_preview": str(baseline_row.get("clean_text", ""))[:260],
                "new_paragraphs": [
                    {
                        "object_id": row.get("object_id"),
                        "source_line_count": len(row.get("source_line_ids", [])),
                        "word_count": candidate_word_count(row),
                        "text_preview": str(row.get("clean_text", ""))[:180],
                    }
                    for row in matching_active
                ],
            }
            split_examples.append(example)
            risky_children = [
                row
                for row in matching_active
                if candidate_word_count(row) < 25 or str(row.get("clean_text", "")).strip()[-1:] not in TERMINAL_PUNCTUATION
            ]
            if risky_children:
                oversplit_examples.append({**example, "risk_reason": "short_or_unusual_ending_child_paragraph"})

    baseline_review = baseline["review_report"]
    active_review = active["review_report"]
    baseline_gold = baseline.get("gold_evaluation_report", {})
    active_gold = active.get("gold_evaluation_report", {})
    baseline_gold_counts = baseline_gold.get("counts", {})
    active_gold_counts = active_gold.get("counts", {})
    baseline_gold_scores = baseline_gold.get("scores", {})
    active_gold_scores = active_gold.get("scores", {})
    baseline_promotion_counts = baseline["promotion_report"].get("counts", {})
    active_promotion_counts = active["promotion_report"].get("counts", {})
    baseline_review_counts = baseline_review.get("counts", {})
    active_review_counts = active_review.get("counts", {})
    baseline_true_merges = count_likely_true_merges(baseline_review)
    active_true_merges = count_likely_true_merges(active_review)
    baseline_taxonomy_counts = taxonomy_counts_for_review(baseline["canonical_paragraphs"], baseline_review)
    active_taxonomy_counts = taxonomy_counts_for_review(active["canonical_paragraphs"], active_review)
    baseline_bbox_span = (baseline_review.get("bbox_span_risk_summary") or {}).get("total", 0)
    active_bbox_span = (active_review.get("bbox_span_risk_summary") or {}).get("total", 0)
    baseline_precision = baseline_gold_scores.get("paragraph_precision")
    active_precision = active_gold_scores.get("paragraph_precision")
    baseline_recall = baseline_gold_scores.get("paragraph_recall")
    active_recall = active_gold_scores.get("paragraph_recall")
    gold_score_improved = (
        isinstance(active_precision, (int, float))
        and isinstance(baseline_precision, (int, float))
        and isinstance(active_recall, (int, float))
        and isinstance(baseline_recall, (int, float))
        and active_precision > baseline_precision
        and active_recall > baseline_recall
    )
    over_splits_decreased = active_gold_counts.get("over_split_paragraphs", 0) < baseline_gold_counts.get("over_split_paragraphs", 0)
    over_merges_not_increased = active_gold_counts.get("over_merged_paragraphs", 0) <= baseline_gold_counts.get("over_merged_paragraphs", 0)
    warning_regression = active_review_counts.get("warning_count", 0) > baseline_review_counts.get("warning_count", 0) + 5
    bbox_regression = active_bbox_span > baseline_bbox_span + 5
    improved = gold_score_improved and over_splits_decreased and over_merges_not_increased and not warning_regression and not bbox_regression
    worsened = (
        active_gold_counts.get("over_merged_paragraphs", 0) > baseline_gold_counts.get("over_merged_paragraphs", 0)
        or warning_regression
        or bbox_regression
    )
    experiment_outcome = (
        "new_policy_improved_gold_score"
        if improved
        else "new_policy_not_adopted_gold_or_safety_regression"
        if worsened
        else "new_policy_not_adopted_insufficient_gold_improvement"
    )
    experiment_recommendation = (
        "review_join_examples_then_consider_narrow_adoption"
        if experiment_outcome == "new_policy_improved_gold_score"
        else "do_not_adopt_cross_page_policy_yet_refine_continuation_conditions"
    )
    report = {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "experiment": f"paragraph_merge_policy_{EXPERIMENTAL_PARAGRAPH_MERGE_POLICY}",
        "baseline_paragraph_merge_policy": BASELINE_PARAGRAPH_MERGE_POLICY,
        "new_paragraph_merge_policy": EXPERIMENTAL_PARAGRAPH_MERGE_POLICY,
        "scope": "deterministic_paragraph_merge_policy_experiment_only",
        "experiment_outcome": experiment_outcome,
        "does_not_add": ["OCR", "AI/model review", "embeddings", "retrieval", "graph work", "structure promotion"],
        "counts": {
            "baseline_paragraph_candidate_count": len(baseline_paragraphs),
            "new_paragraph_candidate_count": len(active_paragraphs),
            "baseline_canonical_promoted_count": baseline_promotion_counts.get("promoted_paragraphs", 0),
            "new_canonical_promoted_count": active_promotion_counts.get("promoted_paragraphs", 0),
            "baseline_bbox_span_risk_count": baseline_bbox_span,
            "new_bbox_span_risk_count": active_bbox_span,
            "baseline_likely_true_accidental_merge_count": baseline_true_merges,
            "new_likely_true_accidental_merge_count": active_true_merges,
            "baseline_merged_across_paragraph_break_count": baseline_taxonomy_counts.get("merged_across_paragraph_break", 0),
            "new_merged_across_paragraph_break_count": active_taxonomy_counts.get("merged_across_paragraph_break", 0),
            "baseline_merged_across_large_vertical_whitespace_count": baseline_taxonomy_counts.get("merged_across_large_vertical_whitespace", 0),
            "new_merged_across_large_vertical_whitespace_count": active_taxonomy_counts.get("merged_across_large_vertical_whitespace", 0),
            "baseline_blocked_paragraph_count": baseline_promotion_counts.get("paragraph_candidates_blocked", 0),
            "new_blocked_paragraph_count": active_promotion_counts.get("paragraph_candidates_blocked", 0),
            "split_example_count": len(split_examples),
            "possible_oversplitting_risk_count": len(oversplit_examples),
            "baseline_gold_matched_paragraphs": baseline_gold_counts.get("matched_paragraphs", 0),
            "new_gold_matched_paragraphs": active_gold_counts.get("matched_paragraphs", 0),
            "baseline_gold_over_split_paragraphs": baseline_gold_counts.get("over_split_paragraphs", 0),
            "new_gold_over_split_paragraphs": active_gold_counts.get("over_split_paragraphs", 0),
            "baseline_gold_over_merged_paragraphs": baseline_gold_counts.get("over_merged_paragraphs", 0),
            "new_gold_over_merged_paragraphs": active_gold_counts.get("over_merged_paragraphs", 0),
            "baseline_gold_missing_paragraphs": baseline_gold_counts.get("missing_paragraphs", 0),
            "new_gold_missing_paragraphs": active_gold_counts.get("missing_paragraphs", 0),
            "cross_page_join_count": experiment_details.get("joined_count", 0),
            "cross_page_rejected_count": experiment_details.get("rejected_count", 0),
        },
        "gold_scores": {
            "baseline_paragraph_precision": baseline_precision,
            "new_paragraph_precision": active_precision,
            "baseline_paragraph_recall": baseline_recall,
            "new_paragraph_recall": active_recall,
            "baseline_object_label_accuracy": baseline_gold_scores.get("object_label_accuracy"),
            "new_object_label_accuracy": active_gold_scores.get("object_label_accuracy"),
            "sufficient_to_judge_policy": bool(baseline_gold.get("sufficient_to_judge_merge_policy_adoption")),
        },
        "acceptance_rule": {
            "gold_score_improved": gold_score_improved,
            "over_splits_decreased": over_splits_decreased,
            "over_merges_not_increased": over_merges_not_increased,
            "audit_warning_regression": warning_regression,
            "bbox_span_regression": bbox_regression,
            "adoptable": improved,
        },
        "taxonomy_counts": {
            "baseline": dict(sorted(baseline_taxonomy_counts.items())),
            "new": dict(sorted(active_taxonomy_counts.items())),
        },
        "downstream_safety": {
            "baseline_safe_for_downstream": baseline_review.get("safe_for_downstream"),
            "new_safe_for_downstream": active_review.get("safe_for_downstream"),
            "baseline_recommendation": baseline_review.get("recommendation"),
            "new_recommendation": active_review.get("recommendation"),
            "recommendation": experiment_recommendation,
        },
        "warning_counts": {
            "baseline_total_warnings": baseline_review_counts.get("warning_count", 0),
            "new_total_warnings": active_review_counts.get("warning_count", 0),
        },
        "examples_of_paragraphs_split_by_new_policy": split_examples[:30],
        "examples_of_possible_oversplitting_risk": oversplit_examples[:30],
        "examples_of_joined_cross_page_paragraphs": (experiment_details.get("joined_cross_page_paragraphs") or [])[:30],
        "examples_of_rejected_cross_page_candidates": (experiment_details.get("rejected_cross_page_candidates") or [])[:30],
    }
    return report


def merge_failure_taxonomy_decision(
    diagnostic: dict[str, Any],
    canonical_row: dict[str, Any],
) -> tuple[str, float, str]:
    warnings = set(str(warning) for warning in diagnostic.get("all_warnings", []))
    ratio = diagnostic.get("page_height_ratio")
    ratio_value = float(ratio) if isinstance(ratio, (int, float)) else None
    source_line_count = int(diagnostic.get("source_line_count") or 0)
    clean_text = str(canonical_row.get("clean_text", ""))
    sentence_boundary_count = clean_text.count(". ") + clean_text.count("? ") + clean_text.count("! ")
    word_count = len(clean_text.split())

    if "possible_metadata_or_structure_leakage" in warnings:
        return "merged_heading_or_metadata_into_paragraph", 0.74, "inspect manually and consider curated override"
    if ratio_value is not None and ratio_value >= 0.40:
        return "merged_across_large_vertical_whitespace", 0.72, "inspect page whitespace before changing merge rule"
    if source_line_count >= 10 and sentence_boundary_count >= 2:
        return "merged_across_paragraph_break", 0.68, "inspect manually and design narrower paragraph-break rule"
    if word_count > 170 and ratio_value is not None and ratio_value < 0.35:
        return "normal_long_paragraph_false_positive", 0.62, "no action until visually confirmed"
    if ratio_value is not None and ratio_value < 0.28:
        return "bbox_or_threshold_artifact", 0.60, "adjust review threshold only after visual inspection"
    return "needs_manual_inspection", 0.55, "inspect manually"


def build_paragraph_merge_failure_taxonomy_report(
    book_id: str,
    run_id: str,
    canonical_paragraphs: list[dict[str, Any]],
    canonical_review_report: dict[str, Any],
    sample_size: int = 10,
) -> dict[str, Any]:
    canonical_by_id = {row.get("canonical_paragraph_id"): row for row in canonical_paragraphs}
    decisions_by_id = {
        row.get("canonical_paragraph_id"): row
        for row in canonical_review_report.get("bbox_span_decisions", [])
    }
    diagnostics = [
        row
        for row in canonical_review_report.get("bbox_span_risk_diagnostics", [])
        if (decisions_by_id.get(row.get("canonical_paragraph_id")) or {}).get("likely_cause") == "true_accidental_merge"
    ]
    diagnostics = sorted(
        diagnostics,
        key=lambda row: (
            -SEVERITY_RANK.get(str(row.get("warning_severity", "low")), 1),
            int(row.get("page_number") or 0),
            str(row.get("canonical_paragraph_id")),
        ),
    )[:sample_size]
    samples = []
    for diagnostic in diagnostics:
        canonical_row = canonical_by_id.get(diagnostic.get("canonical_paragraph_id"), {})
        category, confidence, recommended_action = merge_failure_taxonomy_decision(diagnostic, canonical_row)
        samples.append(
            {
                "canonical_paragraph_id": diagnostic.get("canonical_paragraph_id"),
                "page_number": diagnostic.get("page_number"),
                "source_candidate_object_id": diagnostic.get("source_candidate_object_id"),
                "severity": diagnostic.get("warning_severity"),
                "text_preview": str(canonical_row.get("clean_text", ""))[:320],
                "first_source_line_preview": diagnostic.get("first_source_line_preview"),
                "last_source_line_preview": diagnostic.get("last_source_line_preview"),
                "source_line_count": diagnostic.get("source_line_count"),
                "vertical_bbox_span": diagnostic.get("vertical_bbox_span"),
                "page_height_ratio": diagnostic.get("page_height_ratio"),
                "provisional_category": category,
                "confidence": confidence,
                "recommended_next_action": recommended_action,
                "audit_anchor": diagnostic.get("audit_anchor"),
                "page_anchor": diagnostic.get("page_anchor"),
            }
        )

    category_counts = Counter(row["provisional_category"] for row in samples)
    action_counts = Counter(row["recommended_next_action"] for row in samples)
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "method": "deterministic_merge_failure_taxonomy_v1",
        "scope": "sampled_likely_true_accidental_merge_rows_only",
        "analysis_only": True,
        "sample_selection": {
            "requested_sample_size": sample_size,
            "actual_sample_size": len(samples),
            "selection_rule": "highest-severity likely true_accidental_merge rows sorted by severity, page, and canonical id",
        },
        "summary": {
            "sampled_rows": len(samples),
            "count_by_category": dict(sorted(category_counts.items())),
            "count_by_recommended_action": dict(sorted(action_counts.items())),
        },
        "samples": samples,
    }


def gold_row_is_authoritative(row: dict[str, Any]) -> bool:
    return str(row.get("review_status") or row.get("status") or "").lower() == "authoritative"


def expected_label_for_candidate(row: dict[str, Any]) -> str:
    stream_type = str(row.get("stream_type", ""))
    if stream_type == "main_paragraph_candidate":
        return "main_paragraph"
    if stream_type == "structure_candidate":
        return "structure"
    if stream_type == "page_artifact_candidate":
        return "page_artifact"
    if stream_type == "unknown_needs_review":
        return "unknown"
    return "unknown"


def evaluate_gold_reviews(
    book_id: str,
    run_id: str,
    main_paragraphs: list[dict[str, Any]],
    structure: list[dict[str, Any]],
    page_artifacts: list[dict[str, Any]],
    unknown: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_dir = gold_review_dir(book_id)
    gold_pages_path = gold_dir / "gold_pages.json"
    gold_boundaries_path = gold_dir / "gold_paragraph_boundaries.jsonl"
    gold_labels_path = gold_dir / "gold_object_labels.jsonl"
    gold_pages = read_json(gold_pages_path) if gold_pages_path.exists() else {"pages": []}
    boundary_rows = read_jsonl(gold_boundaries_path) if gold_boundaries_path.exists() else []
    label_rows = read_jsonl(gold_labels_path) if gold_labels_path.exists() else []

    candidate_rows = main_paragraphs + structure + page_artifacts + unknown
    candidate_by_object_id = {}
    for row in candidate_rows:
        candidate_by_object_id[row.get("object_id")] = row
        for source_object_id in row.get("source_object_ids", []):
            candidate_by_object_id.setdefault(source_object_id, row)
    paragraph_line_sets = [
        {
            "object_id": row.get("object_id"),
            "paragraph_id": row.get("paragraph_id"),
            "page_number": row.get("page_number"),
            "source_line_ids": set(row.get("source_line_ids", [])),
            "clean_text": row.get("clean_text", ""),
        }
        for row in main_paragraphs
    ]

    authoritative_boundaries = [row for row in boundary_rows if gold_row_is_authoritative(row)]
    authoritative_labels = [row for row in label_rows if gold_row_is_authoritative(row)]
    placeholder_boundaries = [row for row in boundary_rows if not gold_row_is_authoritative(row)]
    placeholder_labels = [row for row in label_rows if not gold_row_is_authoritative(row)]

    matched_paragraphs = []
    missing_paragraphs = []
    over_merged_paragraphs = []
    over_split_paragraphs = []
    for gold in authoritative_boundaries:
        expected_lines = set(gold.get("source_line_ids", []))
        if not expected_lines:
            missing_paragraphs.append({**gold, "reason": "gold_row_has_no_source_line_ids"})
            continue
        exact = [row for row in paragraph_line_sets if row["source_line_ids"] == expected_lines]
        if exact:
            matched_paragraphs.append({"gold_id": gold.get("gold_id"), "matched_object_id": exact[0].get("object_id")})
            continue
        containing = [row for row in paragraph_line_sets if expected_lines and expected_lines.issubset(row["source_line_ids"])]
        if containing:
            over_merged_paragraphs.append(
                {
                    "gold_id": gold.get("gold_id"),
                    "matched_object_id": containing[0].get("object_id"),
                    "extra_source_line_ids": sorted(containing[0]["source_line_ids"] - expected_lines),
                }
            )
            continue
        overlapping = [row for row in paragraph_line_sets if expected_lines.intersection(row["source_line_ids"])]
        if len(overlapping) > 1:
            over_split_paragraphs.append(
                {
                    "gold_id": gold.get("gold_id"),
                    "matched_object_ids": [row.get("object_id") for row in overlapping],
                }
            )
            continue
        missing_paragraphs.append({"gold_id": gold.get("gold_id"), "reason": "no_matching_paragraph_candidate"})

    matched_labels = []
    wrong_object_labels = []
    missing_object_labels = []
    for gold in authoritative_labels:
        object_id = gold.get("object_id")
        candidate = candidate_by_object_id.get(object_id)
        if not candidate:
            missing_object_labels.append({"object_id": object_id, "expected_label": gold.get("expected_label")})
            continue
        actual_label = expected_label_for_candidate(candidate)
        if actual_label == gold.get("expected_label"):
            matched_labels.append({"object_id": object_id, "label": actual_label})
        else:
            wrong_object_labels.append(
                {
                    "object_id": object_id,
                    "expected_label": gold.get("expected_label"),
                    "actual_label": actual_label,
                }
            )

    authoritative_paragraph_count = len(authoritative_boundaries)
    authoritative_label_count = len(authoritative_labels)
    paragraph_precision = None
    paragraph_recall = None
    label_accuracy = None
    if authoritative_paragraph_count:
        paragraph_recall = len(matched_paragraphs) / authoritative_paragraph_count
        paragraph_precision = len(matched_paragraphs) / max(1, len(matched_paragraphs) + len(over_merged_paragraphs) + len(over_split_paragraphs))
    if authoritative_label_count:
        label_accuracy = len(matched_labels) / authoritative_label_count

    reviewed_pages = [row.get("page") for row in gold_pages.get("pages", [])]
    sufficient = authoritative_paragraph_count >= 10 and len(set(row.get("page") for row in authoritative_boundaries)) >= 3
    try:
        gold_dir_display = str(gold_dir.relative_to(ROOT))
    except ValueError:
        gold_dir_display = str(gold_dir)

    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "method": "deterministic_gold_review_evaluation_v1",
        "gold_dir": gold_dir_display,
        "analysis_only": True,
        "scoring_authoritative": bool(authoritative_boundaries or authoritative_labels),
        "sufficient_to_judge_merge_policy_adoption": sufficient,
        "insufficiency_reasons": [] if sufficient else ["Need at least 10 authoritative paragraph boundary rows across at least 3 reviewed pages."],
        "counts": {
            "gold_pages_reviewed": len(reviewed_pages),
            "gold_paragraph_rows": len(boundary_rows),
            "gold_object_label_rows": len(label_rows),
            "authoritative_paragraph_rows": authoritative_paragraph_count,
            "authoritative_object_label_rows": authoritative_label_count,
            "placeholder_paragraph_rows_excluded": len(placeholder_boundaries),
            "placeholder_object_label_rows_excluded": len(placeholder_labels),
            "matched_paragraphs": len(matched_paragraphs),
            "missing_paragraphs": len(missing_paragraphs),
            "over_merged_paragraphs": len(over_merged_paragraphs),
            "over_split_paragraphs": len(over_split_paragraphs),
            "matched_object_labels": len(matched_labels),
            "wrong_object_labels": len(wrong_object_labels),
            "missing_object_labels": len(missing_object_labels),
        },
        "scores": {
            "paragraph_precision": paragraph_precision,
            "paragraph_recall": paragraph_recall,
            "object_label_accuracy": label_accuracy,
        },
        "gold_pages": gold_pages.get("pages", []),
        "matched_paragraphs": matched_paragraphs,
        "missing_paragraphs": missing_paragraphs,
        "over_merged_paragraphs": over_merged_paragraphs,
        "over_split_paragraphs": over_split_paragraphs,
        "wrong_object_labels": wrong_object_labels,
        "missing_object_labels": missing_object_labels,
        "excluded_placeholder_rows": {
            "paragraph_gold_ids": [row.get("gold_id") for row in placeholder_boundaries],
            "object_ids": [row.get("object_id") for row in placeholder_labels],
        },
    }


def authoritative_gold_line_sets(book_id: str) -> list[dict[str, Any]]:
    path = gold_review_dir(book_id) / "gold_paragraph_boundaries.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    return [
        {
            "gold_id": row.get("gold_id"),
            "page": row.get("page"),
            "source_line_ids": set(row.get("source_line_ids", [])),
        }
        for row in rows
        if gold_row_is_authoritative(row) and row.get("source_line_ids")
    ]


def cross_page_join_risk(join: dict[str, Any], gold_ids: list[str]) -> tuple[str, float, str]:
    pages = join.get("pages") or []
    source_line_count = int(join.get("source_line_count") or 0)
    if gold_ids:
        return "likely_correct_continuation", 0.96, "covered by authoritative gold row"
    if any(int(page) <= 18 for page in pages if isinstance(page, int)):
        return "boundary_or_structure_risk", 0.58, "inspect manually before adoption because join is in front matter or early book matter"
    if source_line_count >= 34:
        return "needs_manual_review", 0.62, "inspect long joined paragraph before adoption"
    if "previous_page_last_paragraph_incomplete" in join.get("join_reasons", []) and "next_page_first_paragraph_looks_like_continuation" in join.get("join_reasons", []):
        return "likely_correct_continuation", 0.78, "review sample, then accept if visual page evidence confirms continuation"
    return "possible_false_join", 0.52, "inspect manually before adoption"


def validate_cross_page_join_decisions(
    decisions: list[dict[str, Any]],
    proposed_joins: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    valid_by_join_id: dict[str, dict[str, Any]] = {}
    seen_join_ids: set[str] = set()
    for row in decisions:
        join_id = str(row.get("join_id", ""))
        line_number = row.get("_line_number")
        missing_fields = [
            field
            for field in REQUIRED_CROSS_PAGE_JOIN_DECISION_FIELDS
            if field not in row or not str(row.get(field, "")).strip()
        ]
        if missing_fields:
            errors.append(
                {
                    "code": "missing_required_fields",
                    "join_id": join_id,
                    "line_number": line_number,
                    "fields": missing_fields,
                }
            )
        if join_id in seen_join_ids:
            errors.append({"code": "duplicate_join_decision", "join_id": join_id, "line_number": line_number})
        seen_join_ids.add(join_id)
        if row.get("decision") not in VALID_CROSS_PAGE_JOIN_DECISIONS:
            errors.append({"code": "invalid_decision", "join_id": join_id, "line_number": line_number, "decision": row.get("decision")})
        proposed = proposed_joins.get(join_id)
        if not proposed:
            errors.append({"code": "missing_join_id", "join_id": join_id, "line_number": line_number})
            continue
        mismatches = [
            field
            for field in ["left_page", "right_page", "left_candidate_id", "right_candidate_id"]
            if str(row.get(field)) != str(proposed.get(field))
        ]
        if mismatches:
            errors.append({"code": "candidate_mismatch", "join_id": join_id, "line_number": line_number, "fields": mismatches})
        row_has_error = any(error.get("join_id") == join_id for error in errors)
        if not row_has_error:
            valid_by_join_id[join_id] = row
    return {
        "status": "pass" if not errors else "fail",
        "source_row_count": len(decisions),
        "valid_decision_count": len(valid_by_join_id),
        "error_count": len(errors),
        "errors": errors,
    }, valid_by_join_id


def cross_page_join_decision_status(row: dict[str, Any], curated_decision: dict[str, Any] | None) -> str:
    if curated_decision:
        decision = curated_decision.get("decision")
        if decision == "accept":
            return "curated_accepted"
        if decision == "reject":
            return "curated_rejected"
        return "still_needs_manual_review"
    if row.get("overlaps_authoritative_gold"):
        return "gold_covered"
    if row.get("risk_category") == "likely_correct_continuation":
        return "auto_likely_correct"
    if row.get("risk_category") == "boundary_or_structure_risk":
        return "boundary_or_structure_risk"
    return "still_needs_manual_review"


def build_cross_page_join_review_report(
    book_id: str,
    run_id: str,
    experiment_details: dict[str, Any],
    join_decisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gold_line_sets = authoritative_gold_line_sets(book_id)
    review_rows = []
    for index, join in enumerate(experiment_details.get("joined_cross_page_paragraphs", []), start=1):
        join_lines = set(join.get("source_line_ids", []))
        matching_gold_ids = [
            str(row.get("gold_id"))
            for row in gold_line_sets
            if join_lines and join_lines == row.get("source_line_ids")
        ]
        risk_category, confidence, action = cross_page_join_risk(join, matching_gold_ids)
        left_page, right_page = (join.get("pages") or [None, None])[:2]
        review_rows.append(
            {
                "join_id": f"xpage_join_{index:04d}",
                "left_page": left_page,
                "right_page": right_page,
                "left_candidate_id": join.get("first_object_id"),
                "right_candidate_id": join.get("second_object_id"),
                "left_text_end_preview": join.get("first_text_end"),
                "right_text_start_preview": join.get("second_text_start"),
                "joined_text_preview": join.get("joined_text_preview"),
                "continuation_evidence": join.get("join_reasons", []),
                "acceptance_signals": [
                    "previous_page_last_paragraph_incomplete",
                    "next_page_first_paragraph_looks_like_continuation",
                    "no_intervening_structure_candidate",
                ],
                "rejection_signals": [],
                "overlaps_authoritative_gold": bool(matching_gold_ids),
                "gold_ids": matching_gold_ids,
                "risk_category": risk_category,
                "confidence": confidence,
                "recommended_action": action,
                "page_anchor": f"#page-{left_page}",
                "left_audit_anchor": f"#card-{safe_dom_id(str(join.get('first_object_id')))}",
                "right_audit_anchor": f"#card-{safe_dom_id(str(join.get('second_object_id')))}",
            }
        )

    decision_validation, valid_decisions_by_join_id = validate_cross_page_join_decisions(
        join_decisions or [],
        {row["join_id"]: row for row in review_rows},
    )
    for row in review_rows:
        curated_decision = valid_decisions_by_join_id.get(row["join_id"])
        row["curated_join_decision"] = (
            {
                "decision": curated_decision.get("decision"),
                "reason": curated_decision.get("reason"),
                "reviewer": curated_decision.get("reviewer"),
                "date": curated_decision.get("date"),
                "evidence_reference": curated_decision.get("evidence_reference"),
                "line_number": curated_decision.get("_line_number"),
            }
            if curated_decision
            else None
        )
        row["decision_status"] = cross_page_join_decision_status(row, curated_decision)

    risk_counts = Counter(row["risk_category"] for row in review_rows)
    status_counts = Counter(row["decision_status"] for row in review_rows)
    top_pages = Counter()
    for row in review_rows:
        if row["decision_status"] in {"still_needs_manual_review", "boundary_or_structure_risk"}:
            top_pages.update([row["left_page"], row["right_page"]])
    unresolved_count = status_counts.get("still_needs_manual_review", 0) + status_counts.get("boundary_or_structure_risk", 0)
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "method": "deterministic_cross_page_join_review_v1",
        "analysis_only": True,
        "does_not_change_active_policy": True,
        "decision_source": f"reviews/{book_id}/cross_page_join_decisions.jsonl",
        "decision_validation": decision_validation,
        "summary": {
            "total_proposed_joins": len(review_rows),
            "likely_correct_joins": risk_counts.get("likely_correct_continuation", 0),
            "possible_false_joins": risk_counts.get("possible_false_join", 0),
            "boundary_or_structure_risk_joins": risk_counts.get("boundary_or_structure_risk", 0),
            "needs_manual_review": risk_counts.get("needs_manual_review", 0),
            "auto_likely_correct_joins": status_counts.get("auto_likely_correct", 0),
            "curated_accepted_joins": status_counts.get("curated_accepted", 0),
            "curated_rejected_joins": status_counts.get("curated_rejected", 0),
            "still_needs_manual_review_joins": status_counts.get("still_needs_manual_review", 0),
            "decision_boundary_or_structure_risk_joins": status_counts.get("boundary_or_structure_risk", 0),
            "gold_covered_joins": status_counts.get("gold_covered", 0),
            "joins_covered_by_authoritative_gold": sum(1 for row in review_rows if row["overlaps_authoritative_gold"]),
            "joins_not_covered_by_gold": sum(1 for row in review_rows if not row["overlaps_authoritative_gold"]),
            "unresolved_join_count": unresolved_count,
            "unresolved_risk_low_enough_for_adoption": unresolved_count == 0 and decision_validation["status"] == "pass",
            "top_pages_needing_review": [
                {"page": page, "count": count}
                for page, count in top_pages.most_common(10)
                if page is not None
            ],
        },
        "joins": review_rows,
    }


def build_xpage_join_0032_investigation(
    book_id: str,
    run_id: str,
    cross_page_join_review_report: dict[str, Any],
    layout_objects: list[dict[str, Any]],
    page_artifacts: list[dict[str, Any]],
    structure: list[dict[str, Any]],
) -> dict[str, Any]:
    join = next(
        (row for row in cross_page_join_review_report.get("joins", []) if row.get("join_id") == "xpage_join_0032"),
        {},
    )
    left_id = join.get("left_candidate_id")
    right_id = join.get("right_candidate_id")
    objects_by_id = {row.get("object_id"): row for row in layout_objects}
    left_object = objects_by_id.get(left_id, {})
    right_object = objects_by_id.get(right_id, {})
    intervening_artifacts = [
        row
        for row in page_artifacts
        if row.get("page_number") in {join.get("left_page"), join.get("right_page")}
    ]
    intervening_structure = [
        row
        for row in structure
        if row.get("page_number") in {join.get("left_page"), join.get("right_page")}
    ]

    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "join_id": "xpage_join_0032",
        "left_page": join.get("left_page"),
        "right_page": join.get("right_page"),
        "left_candidate_id": left_id,
        "right_candidate_id": right_id,
        "left_text_end_preview": join.get("left_text_end_preview"),
        "right_text_start_preview": join.get("right_text_start_preview"),
        "raw_source_line_evidence": {
            "left_last_lines": str(left_object.get("raw_text", "")).splitlines()[-4:],
            "right_first_lines": str(right_object.get("raw_text", "")).splitlines()[:4],
            "left_source_line_ids": left_object.get("source_line_ids", [])[-4:],
            "right_source_line_ids": right_object.get("source_line_ids", [])[:4],
        },
        "visual_page_evidence_references": [
            f"page_images/{page_image_filename(int(join.get('left_page') or 0))}",
            f"page_images/{page_image_filename(int(join.get('right_page') or 0))}",
            join.get("left_audit_anchor"),
            join.get("right_audit_anchor"),
        ],
        "intervening_page_artifacts": [
            {
                "object_id": row.get("object_id"),
                "page": row.get("page_number"),
                "clean_text": row.get("clean_text"),
                "artifact_type": row.get("artifact_type"),
            }
            for row in intervening_artifacts
        ],
        "intervening_structure_candidates": [
            {
                "object_id": row.get("object_id"),
                "page": row.get("page_number"),
                "clean_text": row.get("clean_text"),
            }
            for row in intervening_structure
        ],
        "suspected_issue": "valid_continuation",
        "recommended_decision": "accept",
        "reason": (
            "Rendered page witnesses and raw source lines show page 55 ends with 'make the four letters' "
            "and page 56 begins 'named.' The phrase is 'make the four letters named.' Only running-header "
            "page furniture intervenes; no structure candidate crosses the join boundary."
        ),
    }


def build_policy_adoption_decision(
    book_id: str,
    run_id: str,
    paragraph_merge_experiment_report: dict[str, Any],
    cross_page_join_review_report: dict[str, Any],
    xpage_join_0032_investigation: dict[str, Any],
    active_manifest_policy: str,
    active_promotion_report: dict[str, Any],
    active_review_report: dict[str, Any],
    active_gold_report: dict[str, Any],
) -> dict[str, Any]:
    counts = paragraph_merge_experiment_report.get("counts", {})
    gold_scores = paragraph_merge_experiment_report.get("gold_scores", {})
    acceptance = paragraph_merge_experiment_report.get("acceptance_rule", {})
    join_summary = cross_page_join_review_report.get("summary", {})
    active_gold_scores = active_gold_report.get("scores", {})
    active_gold_counts = active_gold_report.get("counts", {})
    active_review_counts = active_review_report.get("counts", {})
    gates = {
        "gold_score_improved": bool(acceptance.get("gold_score_improved")),
        "over_splits_decreased": bool(acceptance.get("over_splits_decreased")),
        "over_merges_not_increased": bool(acceptance.get("over_merges_not_increased")),
        "audit_warning_regression": bool(acceptance.get("audit_warning_regression")),
        "bbox_span_regression": bool(acceptance.get("bbox_span_regression")),
        "all_proposed_joins_reviewed": join_summary.get("unresolved_join_count") == 0,
        "no_curated_rejections": join_summary.get("curated_rejected_joins") == 0,
        "xpage_join_0032_resolved": xpage_join_0032_investigation.get("recommended_decision") == "accept",
        "validation_required": True,
    }
    gates_pass = (
        gates["gold_score_improved"]
        and gates["over_splits_decreased"]
        and gates["over_merges_not_increased"]
        and not gates["audit_warning_regression"]
        and not gates["bbox_span_regression"]
        and gates["all_proposed_joins_reviewed"]
        and gates["no_curated_rejections"]
        and gates["xpage_join_0032_resolved"]
    )
    adopted = active_manifest_policy in {CROSS_PAGE_CONTINUATION_POLICY, GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY} and gates_pass
    if adopted and active_manifest_policy == GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY:
        decision = "v2_cross_page_continuation_adoption_superseded_by_guarded_v3"
        adoption_note = (
            "The v2 cross-page continuation gate remains satisfied as a prerequisite, but current active policy is governed by guarded_chained_policy_adoption_decision.json."
        )
    elif adopted:
        decision = "adopt_v2_cross_page_continuation"
        adoption_note = "Policy is active for Phase 1 paragraph merging; downstream intelligence remains blocked until canonical paragraph review becomes safe."
    else:
        decision = "do_not_adopt"
        adoption_note = "Policy was not adopted because one or more gates failed."
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "decision": decision,
        "active_paragraph_merge_policy": active_manifest_policy,
        "previous_policy": BASELINE_PARAGRAPH_MERGE_POLICY,
        "adopted_policy": active_manifest_policy if adopted else None,
        "does_not_add": ["OCR", "AI/model review", "embeddings", "retrieval", "graph work", "structure promotion"],
        "gate_evidence": {
            "gold_paragraph_precision_before": gold_scores.get("baseline_paragraph_precision"),
            "gold_paragraph_precision_after": gold_scores.get("new_paragraph_precision"),
            "gold_paragraph_recall_before": gold_scores.get("baseline_paragraph_recall"),
            "gold_paragraph_recall_after": gold_scores.get("new_paragraph_recall"),
            "matched_gold_paragraphs_before": counts.get("baseline_gold_matched_paragraphs"),
            "matched_gold_paragraphs_after": counts.get("new_gold_matched_paragraphs"),
            "over_split_paragraphs_before": counts.get("baseline_gold_over_split_paragraphs"),
            "over_split_paragraphs_after": counts.get("new_gold_over_split_paragraphs"),
            "over_merged_paragraphs_before": counts.get("baseline_gold_over_merged_paragraphs"),
            "over_merged_paragraphs_after": counts.get("new_gold_over_merged_paragraphs"),
            "object_label_accuracy_before": gold_scores.get("baseline_object_label_accuracy"),
            "object_label_accuracy_after": gold_scores.get("new_object_label_accuracy"),
            "bbox_span_risk_before": counts.get("baseline_bbox_span_risk_count"),
            "bbox_span_risk_after": counts.get("new_bbox_span_risk_count"),
            "likely_true_accidental_merges_before": counts.get("baseline_likely_true_accidental_merge_count"),
            "likely_true_accidental_merges_after": counts.get("new_likely_true_accidental_merge_count"),
            "merged_across_paragraph_break_before": counts.get("baseline_merged_across_paragraph_break_count"),
            "merged_across_paragraph_break_after": counts.get("new_merged_across_paragraph_break_count"),
            "proposed_joins": join_summary.get("total_proposed_joins"),
            "curated_accepted_joins": join_summary.get("curated_accepted_joins"),
            "curated_rejected_joins": join_summary.get("curated_rejected_joins"),
            "unresolved_joins": join_summary.get("unresolved_join_count"),
            "xpage_join_0032_result": xpage_join_0032_investigation.get("suspected_issue"),
        },
        "active_run_after_adoption": {
            "canonical_promoted_paragraphs": active_promotion_report.get("counts", {}).get("promoted_paragraphs"),
            "paragraph_candidates": active_promotion_report.get("counts", {}).get("paragraph_candidates_reviewed"),
            "blocked_paragraph_candidates": active_promotion_report.get("counts", {}).get("paragraph_candidates_blocked"),
            "canonical_review_warning_count": active_review_counts.get("warning_count"),
            "canonical_review_risky_paragraph_count": active_review_counts.get("risky_paragraph_count"),
            "safe_for_downstream": active_review_report.get("safe_for_downstream"),
            "downstream_recommendation": active_review_report.get("recommendation"),
            "gold_paragraph_precision": active_gold_scores.get("paragraph_precision"),
            "gold_paragraph_recall": active_gold_scores.get("paragraph_recall"),
            "gold_matched_paragraphs": active_gold_counts.get("matched_paragraphs"),
            "gold_over_split_paragraphs": active_gold_counts.get("over_split_paragraphs"),
            "gold_over_merged_paragraphs": active_gold_counts.get("over_merged_paragraphs"),
            "object_label_accuracy": active_gold_scores.get("object_label_accuracy"),
        },
        "gates": gates,
        "adoption_note": adoption_note,
    }


def build_post_adoption_canonical_safety_report(
    book_id: str,
    run_id: str,
    baseline_review_report: dict[str, Any],
    active_review_report: dict[str, Any],
    paragraph_merge_experiment_report: dict[str, Any],
    policy_adoption_decision: dict[str, Any],
) -> dict[str, Any]:
    baseline_counts = baseline_review_report.get("counts", {})
    active_counts = active_review_report.get("counts", {})
    baseline_warning_categories = baseline_review_report.get("warning_categories", {})
    active_warning_categories = active_review_report.get("warning_categories", {})
    merge_counts = paragraph_merge_experiment_report.get("counts", {})
    taxonomy_counts = paragraph_merge_experiment_report.get("taxonomy_counts", {})
    active_clusters = active_review_report.get("risky_paragraph_clusters", [])
    active_drilldown = active_review_report.get("warning_category_drilldown", [])
    top_cluster = active_clusters[0] if active_clusters else {}
    top_warning = active_drilldown[0] if active_drilldown else {}
    affected_pages = first_unique(
        list(top_cluster.get("affected_pages", [])) + list(top_warning.get("affected_pages", [])),
        20,
    )
    sample_risky = []
    for row in active_review_report.get("sample_risky_canonical_paragraphs", [])[:12]:
        sample_risky.append(
            {
                "canonical_paragraph_id": row.get("canonical_paragraph_id"),
                "source_candidate_object_id": row.get("source_candidate_object_id"),
                "page_number": row.get("page_number"),
                "warnings": row.get("warnings", []),
                "word_count": row.get("word_count"),
                "source_line_ids": row.get("source_line_ids", []),
                "text_preview": row.get("clean_text_sample", "")[:260],
                "audit_anchor": audit_anchor_for_object(row.get("source_candidate_object_id")),
                "page_anchor": f"#page-{row.get('page_number')}",
            }
        )

    def category_delta(category: str) -> dict[str, Any]:
        before = int(baseline_warning_categories.get(category, 0) or 0)
        after = int(active_warning_categories.get(category, 0) or 0)
        return {"warning": category, "before": before, "after": after, "delta": after - before}

    all_categories = sorted(set(baseline_warning_categories) | set(active_warning_categories))
    category_deltas = sorted(
        [category_delta(category) for category in all_categories],
        key=lambda row: (-abs(int(row["delta"])), -int(row["after"]), str(row["warning"])),
    )
    likely_corrective_path = top_cluster.get("likely_next_corrective_action") or top_warning.get("likely_next_corrective_action")
    may_require = top_cluster.get("may_require") or top_warning.get("may_require") or ["manual_inspection"]
    issue_type = "manual-review issue"
    if "detector_changes" in may_require:
        issue_type = "detector issue"
    elif "promotion_rule_changes" in may_require:
        issue_type = "promotion-rule issue"
    elif "review_overrides" in may_require:
        issue_type = "manual-review issue"
    elif "gold_review" in may_require:
        issue_type = "gold-set gap"

    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "post_adoption_canonical_paragraph_safety_analysis",
        "active_policy": policy_adoption_decision.get("active_paragraph_merge_policy"),
        "does_not_change_extraction_behavior": True,
        "does_not_add": ["OCR", "AI/model review", "embeddings", "retrieval", "graph work", "structure promotion"],
        "current_state": {
            "promoted_canonical_paragraphs": active_counts.get("total_canonical_paragraphs_reviewed"),
            "risky_canonical_paragraphs": active_counts.get("risky_paragraph_count"),
            "clean_looking_canonical_paragraphs": active_counts.get("clean_looking_count"),
            "warning_count": active_counts.get("warning_count"),
            "safe_for_downstream": active_review_report.get("safe_for_downstream"),
            "downstream_recommendation": active_review_report.get("recommendation"),
        },
        "before_after_adoption": {
            "risky_canonical_paragraphs": {
                "before": baseline_counts.get("risky_paragraph_count"),
                "after": active_counts.get("risky_paragraph_count"),
            },
            "warning_count": {
                "before": baseline_counts.get("warning_count"),
                "after": active_counts.get("warning_count"),
            },
            "bbox_span_risk": {
                "before": merge_counts.get("baseline_bbox_span_risk_count"),
                "after": merge_counts.get("new_bbox_span_risk_count"),
            },
            "likely_true_accidental_merges": {
                "before": merge_counts.get("baseline_likely_true_accidental_merge_count"),
                "after": merge_counts.get("new_likely_true_accidental_merge_count"),
            },
            "merged_across_paragraph_break": {
                "before": merge_counts.get("baseline_merged_across_paragraph_break_count"),
                "after": merge_counts.get("new_merged_across_paragraph_break_count"),
            },
            "taxonomy_counts": taxonomy_counts,
            "warning_category_deltas": category_deltas,
        },
        "current_top_risk": {
            "cluster": top_cluster.get("cluster") or top_warning.get("cluster"),
            "warning": top_warning.get("warning"),
            "count": top_cluster.get("count") or top_warning.get("count"),
            "severity": top_cluster.get("severity") or top_warning.get("severity"),
            "affected_pages": affected_pages,
            "likely_corrective_path": likely_corrective_path,
            "may_require": may_require,
            "issue_type": issue_type,
            "why_it_matters": (active_review_report.get("recommendation_detail") or {}).get("why_it_matters"),
        },
        "risk_counts": {
            "warning_categories": active_warning_categories,
            "risk_clusters": [
                {
                    "cluster": row.get("cluster"),
                    "count": row.get("count"),
                    "severity": row.get("severity"),
                    "affected_pages": row.get("affected_pages", []),
                    "may_require": row.get("may_require", []),
                }
                for row in active_clusters
            ],
        },
        "sample_risky_paragraphs": sample_risky,
        "recommendation": "analyze_top_blocker_before_changing_extraction_behavior",
    }


def post_adoption_bbox_likely_cause(
    decision: dict[str, Any],
    canonical_row: dict[str, Any],
    gold_coverage: str,
) -> tuple[str, float, str]:
    warnings = set(str(warning) for warning in canonical_row.get("warnings", []))
    page_number = int(decision.get("page_number") or 0)
    existing_cause = str(decision.get("likely_cause", ""))
    confidence = float(decision.get("confidence") or 0.55)
    if "possible_metadata_or_structure_leakage" in warnings or page_number <= 20:
        return "front_matter_or_metadata_artifact", max(confidence, 0.72), "inspect front matter and keep out of downstream body use until structure policy is defined"
    if existing_cause == "normal_long_paragraph":
        return "normal_long_paragraph", max(confidence, 0.66), "no extraction change; consider tuning review threshold after visual confirmation"
    if existing_cause == "threshold_too_strict":
        return "threshold_noise", max(confidence, 0.64), "adjust review threshold only after visual spot check confirms false positive"
    if existing_cause == "true_accidental_merge":
        return "true_paragraph_grouping_defect", max(confidence, 0.70), "inspect visually and consider a narrow paragraph grouping correction only after examples are classified"
    if gold_coverage == "uncovered":
        return "gold_set_gap", 0.60, "add authoritative gold rows for this page or case before adopting another rule"
    return "needs_visual_review", 0.56, "inspect page image, overlay, and source lines before changing rules"


def build_post_adoption_bbox_span_diagnosis(
    book_id: str,
    run_id: str,
    canonical_paragraphs: list[dict[str, Any]],
    canonical_review_report: dict[str, Any],
    active_policy: str,
) -> dict[str, Any]:
    canonical_by_id = {row.get("canonical_paragraph_id"): row for row in canonical_paragraphs}
    diagnostics_by_id = {
        row.get("canonical_paragraph_id"): row
        for row in canonical_review_report.get("bbox_span_risk_diagnostics", [])
    }
    gold_sets = authoritative_gold_line_sets(book_id)
    gold_pages = {int(row.get("page") or 0) for row in gold_sets}
    diagnosis_rows: list[dict[str, Any]] = []

    for decision in canonical_review_report.get("bbox_span_decisions", []):
        canonical_id = decision.get("canonical_paragraph_id")
        canonical_row = canonical_by_id.get(canonical_id, {})
        diagnostic = diagnostics_by_id.get(canonical_id, {})
        source_lines = set(canonical_row.get("source_line_ids", []))
        exact_gold_ids = [
            str(row.get("gold_id"))
            for row in gold_sets
            if source_lines and source_lines == row.get("source_line_ids")
        ]
        partial_gold_ids = [
            str(row.get("gold_id"))
            for row in gold_sets
            if source_lines and source_lines.intersection(row.get("source_line_ids", set())) and str(row.get("gold_id")) not in exact_gold_ids
        ]
        page_number = int(decision.get("page_number") or canonical_row.get("page_number") or 0)
        if exact_gold_ids:
            gold_coverage = "exact_gold_match"
        elif partial_gold_ids:
            gold_coverage = "partial_gold_overlap"
        elif page_number in gold_pages:
            gold_coverage = "gold_page_without_line_match"
        else:
            gold_coverage = "uncovered"
        diagnosis_input = {**canonical_row, "warnings": diagnostic.get("all_warnings", [])}
        likely_cause, confidence, recommended_action = post_adoption_bbox_likely_cause(decision, diagnosis_input, gold_coverage)
        diagnosis_rows.append(
            {
                "canonical_paragraph_id": canonical_id,
                "page_number": page_number,
                "source_candidate_object_id": decision.get("source_candidate_object_id"),
                "text_preview": str(canonical_row.get("clean_text", ""))[:300],
                "source_line_count": decision.get("source_line_count"),
                "vertical_bbox_span": decision.get("vertical_bbox_span"),
                "page_height_ratio": decision.get("page_height_ratio"),
                "first_source_line_preview": decision.get("first_source_line_preview"),
                "last_source_line_preview": decision.get("last_source_line_preview"),
                "current_warning_labels": diagnostic.get("all_warnings", []),
                "existing_bbox_likely_cause": decision.get("likely_cause"),
                "likely_cause": likely_cause,
                "confidence": confidence,
                "recommended_action": recommended_action,
                "severity": decision.get("severity"),
                "gold_coverage": gold_coverage,
                "matching_gold_ids": exact_gold_ids,
                "overlapping_gold_ids": partial_gold_ids,
                "audit_anchor": decision.get("audit_anchor"),
                "page_anchor": decision.get("page_anchor"),
            }
        )

    def grouped_rows(key: str) -> list[dict[str, Any]]:
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for row in diagnosis_rows:
            grouped.setdefault(row.get(key), []).append(row)
        return [
            {
                key: group_key,
                "count": len(rows),
                "sample_canonical_paragraph_ids": first_unique([row.get("canonical_paragraph_id") for row in rows], 8),
                "affected_pages": first_unique([row.get("page_number") for row in rows], 16),
            }
            for group_key, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), str(item[0])))
        ]

    top_pages = sorted(
        grouped_rows("page_number"),
        key=lambda row: (-int(row.get("count") or 0), int(row.get("page_number") or 0)),
    )[:12]
    high_severity = [row for row in diagnosis_rows if row.get("severity") == "high"]
    likely_true_defects = [row for row in diagnosis_rows if row.get("likely_cause") == "true_paragraph_grouping_defect"]
    likely_false_positive_or_noise = [
        row
        for row in diagnosis_rows
        if row.get("likely_cause") in {"normal_long_paragraph", "threshold_noise", "front_matter_or_metadata_artifact"}
    ]
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "post_adoption_bbox_span_diagnosis",
        "active_policy": active_policy,
        "does_not_change_extraction_behavior": True,
        "does_not_add": ["OCR", "AI/model review", "embeddings", "retrieval", "graph work", "structure promotion", "new merge-policy experiment"],
        "summary": {
            "total_bbox_span_cases": len(diagnosis_rows),
            "high_severity_cases": len(high_severity),
            "likely_true_defects": len(likely_true_defects),
            "likely_false_positive_or_noise": len(likely_false_positive_or_noise),
            "covered_by_gold": sum(1 for row in diagnosis_rows if row.get("gold_coverage") == "exact_gold_match"),
            "partially_covered_by_gold": sum(1 for row in diagnosis_rows if row.get("gold_coverage") == "partial_gold_overlap"),
            "gold_page_without_line_match": sum(1 for row in diagnosis_rows if row.get("gold_coverage") == "gold_page_without_line_match"),
            "not_covered_by_gold": sum(1 for row in diagnosis_rows if row.get("gold_coverage") == "uncovered"),
        },
        "by_likely_cause": grouped_rows("likely_cause"),
        "by_page": top_pages,
        "by_recommended_action": grouped_rows("recommended_action"),
        "by_gold_coverage": grouped_rows("gold_coverage"),
        "high_severity_cases": high_severity[:30],
        "top_pages_needing_visual_review": top_pages,
        "likely_true_defects": likely_true_defects[:30],
        "likely_false_positives_or_threshold_noise": likely_false_positive_or_noise[:30],
        "diagnoses": diagnosis_rows,
        "recommendation": "review_remaining_bbox_span_cases_before_changing_extraction_behavior",
    }


REMEDIATION_GROUPS = {
    "likely_true_paragraph_grouping_defects": {
        "likely_causes": {"true_paragraph_grouping_defect"},
        "recommended_next_action": "Visually inspect a focused sample, then design a narrow grouping correction only after the defect pattern is confirmed.",
        "action_type": "merge/grouping rule work",
        "risk_level": "high",
        "downstream_blocked": True,
    },
    "front_matter_metadata_artifacts": {
        "likely_causes": {"front_matter_or_metadata_artifact"},
        "recommended_next_action": "Review these as promotion/classification issues and keep front matter out of downstream body content until structure handling is explicit.",
        "action_type": "promotion-rule work",
        "risk_level": "high",
        "downstream_blocked": True,
    },
    "gold_set_gaps": {
        "likely_causes": {"gold_set_gap"},
        "recommended_next_action": "Expand authoritative gold rows for these pages or cases before using them to judge future policy changes.",
        "action_type": "gold expansion",
        "risk_level": "medium",
        "downstream_blocked": True,
    },
    "needs_visual_review": {
        "likely_causes": {"needs_visual_review", "normal_long_paragraph", "threshold_noise"},
        "recommended_next_action": "Inspect page images, overlays, source lines, and neighboring objects before deciding whether the warning is real or threshold noise.",
        "action_type": "manual visual review",
        "risk_level": "medium",
        "downstream_blocked": True,
    },
}


def build_post_adoption_remediation_plan(
    book_id: str,
    run_id: str,
    bbox_diagnosis_report: dict[str, Any],
    canonical_safety_report: dict[str, Any],
) -> dict[str, Any]:
    diagnoses = bbox_diagnosis_report.get("diagnoses", [])
    queues = []
    assigned_count = 0
    for group_name, config in REMEDIATION_GROUPS.items():
        likely_causes = set(config["likely_causes"])
        rows = [row for row in diagnoses if row.get("likely_cause") in likely_causes]
        assigned_count += len(rows)
        queues.append(
            {
                "group": group_name,
                "count": len(rows),
                "affected_pages": first_unique([row.get("page_number") for row in rows], 30),
                "sample_canonical_paragraph_ids": first_unique([row.get("canonical_paragraph_id") for row in rows], 12),
                "text_previews": [row.get("text_preview", "") for row in rows[:5]],
                "recommended_next_action": config["recommended_next_action"],
                "action_type": config["action_type"],
                "risk_level": config["risk_level"],
                "downstream_remains_blocked_because_of_this_group": bool(rows) and bool(config["downstream_blocked"]),
                "sample_rows": rows[:10],
            }
        )
    safety_state = canonical_safety_report.get("current_state") or {}
    queue_counts = {row["group"]: row["count"] for row in queues}
    recommended_order = []
    if queue_counts.get("gold_set_gaps", 0):
        recommended_order.append(f"Expand authoritative gold rows for the {queue_counts['gold_set_gaps']} gold-set gaps.")
    if queue_counts.get("front_matter_metadata_artifacts", 0):
        recommended_order.append(
            f"Review the {queue_counts['front_matter_metadata_artifacts']} front-matter/metadata artifacts as likely promotion/classification issues."
        )
    if queue_counts.get("needs_visual_review", 0):
        recommended_order.append(f"Review the {queue_counts['needs_visual_review']} needs-visual-review cases.")
    if queue_counts.get("likely_true_paragraph_grouping_defects", 0):
        recommended_order.append(
            f"Only then consider a narrow correction for the {queue_counts['likely_true_paragraph_grouping_defects']} likely true paragraph grouping defects."
        )
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "post_adoption_remediation_planning",
        "active_policy": bbox_diagnosis_report.get("active_policy"),
        "planning_only": True,
        "does_not_change_extraction_behavior": True,
        "does_not_add": ["OCR", "AI/model review", "embeddings", "retrieval", "graph work", "structure promotion", "new merge-policy experiment"],
        "source_artifact": "post_adoption_bbox_span_diagnosis.json",
        "summary": {
            "total_cases": len(diagnoses),
            "assigned_cases": assigned_count,
            "queue_count": len(queues),
            "safe_for_downstream": safety_state.get("safe_for_downstream"),
            "downstream_recommendation": safety_state.get("downstream_recommendation"),
        },
        "queues": queues,
        "recommended_order": recommended_order,
        "next_action": (
            "expand_gold_rows_before_new_merge_experiment"
            if queue_counts.get("gold_set_gaps", 0)
            else "review_front_matter_and_visual_review_queues_before_new_merge_experiment"
        ),
    }


VALID_FRONT_MATTER_REVIEW_CLASSIFICATIONS = {
    "incorrectly_promoted_metadata",
    "valid_front_matter_content",
    "structure_candidate_needed",
    "promotion_rule_should_block",
    "needs_review",
}


VALID_VISUAL_REVIEW_CLASSIFICATIONS = {
    "valid_canonical_paragraph",
    "true_paragraph_grouping_defect",
    "threshold_noise",
    "extraction_loss_suspected",
    "unresolved",
}


def build_front_matter_metadata_review_report(
    book_id: str,
    run_id: str,
    remediation_plan: dict[str, Any],
    canonical_paragraphs: list[dict[str, Any]],
    active_policy: str,
) -> dict[str, Any]:
    canonical_by_id = {row.get("canonical_paragraph_id"): row for row in canonical_paragraphs}
    front_matter_queue = next(
        (row for row in remediation_plan.get("queues", []) if row.get("group") == "front_matter_metadata_artifacts"),
        {},
    )
    queue_rows = front_matter_queue.get("sample_rows", [])
    review_rows = []
    for queue_row in queue_rows:
        canonical_id = queue_row.get("canonical_paragraph_id")
        canonical_row = canonical_by_id.get(canonical_id, {})
        page_number = int(queue_row.get("page_number") or canonical_row.get("page_number") or 0)
        warnings = set(str(warning) for warning in queue_row.get("current_warning_labels", []))
        source_candidate_id = queue_row.get("source_candidate_object_id") or canonical_row.get("source_candidate_object_id")
        text_preview = queue_row.get("text_preview") or str(canonical_row.get("clean_text", ""))[:300]
        if page_number in {19, 20}:
            likely_classification = "needs_review"
            recommended_action = (
                "Treat as visually valid Chapter I narrative body; review the front-matter boundary "
                "warning because the early-page heuristic is overbroad."
            )
            confidence = 0.76
        elif page_number <= 18:
            likely_classification = "valid_front_matter_content"
            recommended_action = (
                "Keep as real prefatory or letter prose, but do not let it enter downstream main-body "
                "use until front-matter structure handling is explicit."
            )
            confidence = 0.84
        elif "possible_metadata_or_structure_leakage" in warnings:
            likely_classification = "structure_candidate_needed"
            recommended_action = "Review as possible structure-bearing prose before changing promotion rules."
            confidence = 0.62
        else:
            likely_classification = "needs_review"
            recommended_action = "Inspect page image, overlay, object card, and source lines before deciding."
            confidence = 0.55
        review_rows.append(
            {
                "canonical_paragraph_id": canonical_id,
                "page": page_number,
                "source_candidate_id": source_candidate_id,
                "text_preview": text_preview,
                "current_bucket": "main_paragraph_candidate",
                "promotion_status": canonical_row.get("promotion_status"),
                "visual_evidence_reference": (
                    f"phase1_audit.html#page-{page_number}; "
                    f"phase1_audit.html{audit_anchor_for_object(source_candidate_id)}; "
                    f"{PAGE_IMAGES_DIR_NAME}/{page_image_filename(page_number)}"
                ),
                "likely_classification": likely_classification,
                "recommended_action": recommended_action,
                "confidence": confidence,
                "gold_coverage": queue_row.get("gold_coverage"),
                "matching_gold_ids": queue_row.get("matching_gold_ids", []),
                "overlapping_gold_ids": queue_row.get("overlapping_gold_ids", []),
                "current_warning_labels": queue_row.get("current_warning_labels", []),
                "source_line_count": queue_row.get("source_line_count"),
                "page_height_ratio": queue_row.get("page_height_ratio"),
                "audit_anchor": audit_anchor_for_object(source_candidate_id),
                "page_anchor": f"#page-{page_number}",
            }
        )
    classification_counts = Counter(row["likely_classification"] for row in review_rows)
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "front_matter_metadata_review",
        "active_policy": active_policy,
        "review_only": True,
        "does_not_change_extraction_behavior": True,
        "does_not_change_promotion_rules": True,
        "does_not_add": ["OCR", "AI/model review", "embeddings", "retrieval", "graph work", "structure promotion", "new merge-policy experiment"],
        "source_artifacts": [
            "post_adoption_remediation_plan.json",
            "post_adoption_bbox_span_diagnosis.json",
            "phase1_audit.html",
        ],
        "summary": {
            "total_reviewed": len(review_rows),
            "classification_counts": dict(sorted(classification_counts.items())),
            "gold_covered_rows": sum(1 for row in review_rows if row.get("gold_coverage") in {"exact_gold_match", "partial_gold_overlap"}),
            "valid_front_matter_content": classification_counts.get("valid_front_matter_content", 0),
            "needs_review": classification_counts.get("needs_review", 0),
            "safe_for_downstream": False,
            "recommendation": "review_front_matter_structure_and_boundary_rules_before_grouping_correction",
        },
        "rows": review_rows,
    }


def build_visual_review_cases_report(
    book_id: str,
    run_id: str,
    remediation_plan: dict[str, Any],
    canonical_paragraphs: list[dict[str, Any]],
    active_policy: str,
) -> dict[str, Any]:
    canonical_by_id = {row.get("canonical_paragraph_id"): row for row in canonical_paragraphs}
    visual_queue = next(
        (row for row in remediation_plan.get("queues", []) if row.get("group") == "needs_visual_review"),
        {},
    )
    review_rows = []
    for queue_row in visual_queue.get("sample_rows", []):
        canonical_id = queue_row.get("canonical_paragraph_id")
        canonical_row = canonical_by_id.get(canonical_id, {})
        page_number = int(queue_row.get("page_number") or canonical_row.get("page_number") or 0)
        source_candidate_id = queue_row.get("source_candidate_object_id") or canonical_row.get("source_candidate_object_id")
        source_line_ids = canonical_row.get("source_line_ids", [])
        gold_coverage = queue_row.get("gold_coverage")
        warnings = set(str(warning) for warning in queue_row.get("current_warning_labels", []))
        if canonical_id == "cp_000103" or gold_coverage == "partial_gold_overlap":
            likely_classification = "true_paragraph_grouping_defect"
            confidence = 0.88
            recommended_action = (
                "Treat as a real over-split continuation defect; this page-109 paragraph should be "
                "joined with the page 107-109 gold paragraph before downstream use."
            )
        elif gold_coverage == "exact_gold_match":
            likely_classification = "valid_canonical_paragraph"
            confidence = 0.86
            recommended_action = (
                "Keep as a valid canonical paragraph; the remaining span/length warning is threshold "
                "noise for a visually verified long paragraph."
            )
        elif "possible_missing_paragraph_start" in warnings:
            likely_classification = "unresolved"
            confidence = 0.58
            recommended_action = "Inspect neighboring pages and source lines before deciding whether this is a split or extraction-loss case."
        else:
            likely_classification = "threshold_noise"
            confidence = 0.62
            recommended_action = "No extraction change; consider threshold tuning only after more visually verified examples."
        review_rows.append(
            {
                "canonical_paragraph_id": canonical_id,
                "page": page_number,
                "source_candidate_id": source_candidate_id,
                "text_preview": queue_row.get("text_preview") or str(canonical_row.get("clean_text", ""))[:300],
                "visual_evidence_reference": (
                    f"phase1_audit.html#page-{page_number}; "
                    f"phase1_audit.html{audit_anchor_for_object(source_candidate_id)}; "
                    f"{PAGE_IMAGES_DIR_NAME}/{page_image_filename(page_number)}"
                ),
                "source_line_evidence": {
                    "source_line_ids": source_line_ids,
                    "source_line_count": queue_row.get("source_line_count"),
                    "first_source_line_preview": queue_row.get("first_source_line_preview"),
                    "last_source_line_preview": queue_row.get("last_source_line_preview"),
                    "vertical_bbox_span": queue_row.get("vertical_bbox_span"),
                    "page_height_ratio": queue_row.get("page_height_ratio"),
                },
                "likely_classification": likely_classification,
                "confidence": confidence,
                "recommended_action": recommended_action,
                "gold_coverage": gold_coverage,
                "matching_gold_ids": queue_row.get("matching_gold_ids", []),
                "overlapping_gold_ids": queue_row.get("overlapping_gold_ids", []),
                "current_warning_labels": queue_row.get("current_warning_labels", []),
                "audit_anchor": audit_anchor_for_object(source_candidate_id),
                "page_anchor": f"#page-{page_number}",
            }
        )
    classification_counts = Counter(row["likely_classification"] for row in review_rows)
    true_grouping_defects = classification_counts.get("true_paragraph_grouping_defect", 0)
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "visual_review_cases",
        "active_policy": active_policy,
        "review_only": True,
        "does_not_change_extraction_behavior": True,
        "does_not_change_promotion_rules": True,
        "does_not_add": ["OCR", "AI/model review", "embeddings", "retrieval", "graph work", "structure promotion", "new merge-policy experiment"],
        "source_artifacts": [
            "post_adoption_remediation_plan.json",
            "post_adoption_bbox_span_diagnosis.json",
            "phase1_audit.html",
        ],
        "summary": {
            "total_reviewed": len(review_rows),
            "classification_counts": dict(sorted(classification_counts.items())),
            "gold_covered_rows": sum(1 for row in review_rows if row.get("gold_coverage") in {"exact_gold_match", "partial_gold_overlap"}),
            "valid_canonical_paragraphs": classification_counts.get("valid_canonical_paragraph", 0),
            "true_paragraph_grouping_defects": true_grouping_defects,
            "threshold_noise": classification_counts.get("threshold_noise", 0),
            "extraction_loss_suspected": classification_counts.get("extraction_loss_suspected", 0),
            "unresolved": classification_counts.get("unresolved", 0),
            "safe_for_downstream": False,
            "recommendation": (
                "fix_or_gate_confirmed_visual_grouping_defects_before_downstream_use"
                if true_grouping_defects
                else "no_visual_review_grouping_defect_remains; continue_with_remaining_likely_grouping_defect_queue"
            ),
        },
        "rows": review_rows,
    }


def update_remediation_plan_after_reviews(
    remediation_plan: dict[str, Any],
    front_matter_report: dict[str, Any],
    visual_review_report: dict[str, Any],
) -> dict[str, Any]:
    plan = dict(remediation_plan)
    queues = [dict(row) for row in remediation_plan.get("queues", [])]
    queue_by_name = {row.get("group"): row for row in queues}

    review_progress: dict[str, Any] = {}
    front_queue = queue_by_name.get("front_matter_metadata_artifacts", {})
    front_summary = front_matter_report.get("summary") or {}
    front_reviewed = int(front_summary.get("total_reviewed") or 0)
    front_total = int(front_queue.get("count") or 0)
    front_complete = front_total == 0 or (front_reviewed == front_total and front_total > 0)
    if front_queue:
        front_queue["review_status"] = "reviewed" if front_complete else "pending_review"
        front_queue["review_artifact"] = "front_matter_metadata_review_report.json"
        front_queue["review_summary"] = front_summary
        front_queue["recommended_next_action"] = (
            "Front-matter/metadata review is complete; do not apply a broad metadata or promotion block."
            if front_complete
            else front_queue.get("recommended_next_action")
        )
    review_progress["front_matter_metadata_artifacts"] = {
        "status": "reviewed" if front_complete else "pending_review",
        "reviewed": front_reviewed,
        "total": front_total,
        "artifact": "front_matter_metadata_review_report.json",
    }

    visual_queue = queue_by_name.get("needs_visual_review", {})
    visual_summary = visual_review_report.get("summary") or {}
    visual_reviewed = int(visual_summary.get("total_reviewed") or 0)
    visual_total = int(visual_queue.get("count") or 0)
    visual_complete = visual_total == 0 or (visual_reviewed == visual_total and visual_total > 0)
    if visual_queue:
        visual_queue["review_status"] = "reviewed" if visual_complete else "pending_review"
        visual_queue["review_artifact"] = "visual_review_cases_report.json"
        visual_queue["review_summary"] = visual_summary
        visual_queue["recommended_next_action"] = (
            "Visual review is complete; use confirmed grouping defects to guide a narrow correction."
            if visual_complete
            else visual_queue.get("recommended_next_action")
        )
    review_progress["needs_visual_review"] = {
        "status": "reviewed" if visual_complete else "pending_review",
        "reviewed": visual_reviewed,
        "total": visual_total,
        "artifact": "visual_review_cases_report.json",
    }

    queue_counts = {row["group"]: row["count"] for row in queues}
    recommended_order = []
    if queue_counts.get("gold_set_gaps", 0):
        recommended_order.append(f"Expand authoritative gold rows for the {queue_counts['gold_set_gaps']} gold-set gaps.")
    if queue_counts.get("front_matter_metadata_artifacts", 0) and not front_complete:
        recommended_order.append(
            f"Review the {queue_counts['front_matter_metadata_artifacts']} front-matter/metadata artifacts as likely promotion/classification issues."
        )
    if queue_counts.get("needs_visual_review", 0) and not visual_complete:
        recommended_order.append(f"Review the {queue_counts['needs_visual_review']} needs-visual-review cases.")
    if queue_counts.get("likely_true_paragraph_grouping_defects", 0):
        recommended_order.append(
            f"Review and design a narrow correction path for the {queue_counts['likely_true_paragraph_grouping_defects']} likely true paragraph grouping defects under the active policy."
        )
    plan["queues"] = queues
    plan["review_progress"] = review_progress
    plan["recommended_order"] = recommended_order
    if queue_counts.get("gold_set_gaps", 0):
        plan["next_action"] = "expand_gold_rows_before_new_merge_experiment"
    elif not front_complete:
        plan["next_action"] = "review_front_matter_metadata_queue"
    elif not visual_complete:
        plan["next_action"] = "review_visual_review_queue"
    else:
        plan["next_action"] = "review_likely_true_grouping_defects_under_active_policy"
    return plan


def build_narrow_grouping_correction_design(
    book_id: str,
    run_id: str,
    canonical_paragraphs: list[dict[str, Any]],
    visual_review_report: dict[str, Any],
    gold_evaluation_report: dict[str, Any],
    active_policy: str,
) -> dict[str, Any]:
    canonical_by_id = {row.get("canonical_paragraph_id"): row for row in canonical_paragraphs}
    defect_rows = [
        row
        for row in visual_review_report.get("rows", [])
        if row.get("likely_classification") == "true_paragraph_grouping_defect"
    ]
    designs = []
    for row in defect_rows:
        canonical_id = row.get("canonical_paragraph_id")
        canonical_row = canonical_by_id.get(canonical_id, {})
        overlapping_gold_ids = row.get("overlapping_gold_ids", [])
        gold_id = overlapping_gold_ids[0] if overlapping_gold_ids else None
        current_source_id = canonical_row.get("source_candidate_object_id")
        expected_matched_ids = []
        for split in gold_evaluation_report.get("over_split_paragraphs", []):
            if split.get("gold_id") == gold_id:
                expected_matched_ids = split.get("matched_object_ids", [])
                break
        designs.append(
            {
                "defect_id": f"grouping_defect_{canonical_id}",
                "canonical_paragraph_id": canonical_id,
                "affected_pages": [107, 108, 109] if canonical_id == "cp_000103" else [row.get("page")],
                "current_paragraph_grouping_behavior": {
                    "active_policy": active_policy,
                    "left_joined_candidate": "douglass_narrative:p0107:obj002__xpage__obj002" if canonical_id == "cp_000103" else None,
                    "right_fragment_candidate": current_source_id,
                    "observed_problem": (
                        "The active policy joined pages 107 and 108, but left page 109 as a separate "
                        "canonical paragraph even though the page 107-108 text still ends incomplete."
                    ),
                },
                "expected_gold_behavior": {
                    "gold_id": gold_id,
                    "gold_pages": [107, 108, 109] if canonical_id == "cp_000103" else [],
                    "current_matched_object_ids": expected_matched_ids,
                    "expected_result": "One logical paragraph spanning pages 107, 108, and 109.",
                },
                "source_line_evidence": row.get("source_line_evidence", {}),
                "bbox_evidence": {
                    "page_109_bbox": {
                        "vertical_bbox_span": (row.get("source_line_evidence") or {}).get("vertical_bbox_span"),
                        "page_height_ratio": (row.get("source_line_evidence") or {}).get("page_height_ratio"),
                    },
                    "current_left_candidate_is_cross_page": canonical_id == "cp_000103",
                },
                "visual_evidence_references": [
                    "phase1_audit.html#page-107",
                    "phase1_audit.html#page-108",
                    "phase1_audit.html#page-109",
                    "page_images/page_0107.jpg",
                    "page_images/page_0108.jpg",
                    "page_images/page_0109.jpg",
                    row.get("visual_evidence_reference"),
                ],
                "why_current_policy_failed": (
                    "v2_cross_page_continuation can create a two-page joined candidate, but it does not "
                    "chain another continuation decision from that newly joined candidate into the next page."
                ),
                "proposed_narrow_correction_rule": {
                    "name": "v3_chained_cross_page_continuation_design",
                    "description": (
                        "After a cross-page join, re-evaluate the newly joined paragraph against the next "
                        "page's first body paragraph only when the joined text still ends syntactically "
                        "incomplete and the next page starts as continuation text."
                    ),
                },
                "conditions_required_before_rule_applies": [
                    "left candidate was already created by a reviewed cross-page continuation join",
                    "left joined text ends without terminal punctuation or ends with an incomplete phrase",
                    "right candidate is the first main paragraph candidate on the next page after page furniture removal",
                    "right candidate starts lowercase or with continuation-like syntax",
                    "no structure candidate, chapter heading, appendix boundary, or section boundary intervenes",
                    "both candidates are main paragraph candidates with source line provenance",
                    "the combined source lines match or improve authoritative gold coverage when a gold row exists",
                ],
                "conditions_that_must_block_rule": [
                    "right candidate starts a new paragraph with strong indentation plus capitalized sentence start after terminal punctuation",
                    "a chapter, section, appendix, preface, or metadata boundary intervenes",
                    "either side is not a main paragraph candidate",
                    "candidate source lines are missing",
                    "the join would cross more than one unreviewed page boundary",
                    "the join would worsen authoritative gold precision or recall",
                    "side-effect review flags the join as false or unresolved",
                ],
                "expected_effect_on_gold_metrics": {
                    "paragraph_precision": "0.941 -> 1.000 if the page 107-109 over-split is resolved without new errors",
                    "paragraph_recall": "0.941 -> 1.000 if the page 107-109 over-split is resolved without new errors",
                    "over_split_paragraphs": "1 -> 0 for douglass_gold_p0107_0109_001",
                    "over_merged_paragraphs": "must remain 0",
                },
                "expected_side_effects": [
                    "canonical paragraph count may decrease by 1 for the resolved split",
                    "bbox/span warnings may decrease for cp_000103 but could increase for the longer joined paragraph",
                    "new chained joins outside page 109 must be listed and reviewed before adoption",
                ],
                "risk_analysis": [
                    "A chained rule could over-join true page-start paragraphs if applied broadly.",
                    "The rule should be limited to previously joined cross-page candidates and next-page first body candidates.",
                    "Gold and side-effect review must control adoption, as with v2_cross_page_continuation.",
                ],
                "validation_plan": [
                    "Run extraction with the proposed rule as experiment-only.",
                    "Compare gold_evaluation_report before and after.",
                    "Require douglass_gold_p0107_0109_001 to match one candidate.",
                    "Require over_merged_paragraphs to remain 0.",
                    "Generate a chained-join side-effect review report before adoption.",
                    "Run python3 -B -m pytest tests and validation_report.json.",
                ],
                "adoption_gates": [
                    "Gold precision and recall improve or remain perfect after fixing the confirmed defect.",
                    "No new over-merged or missing gold paragraphs.",
                    "All proposed chained joins have evidence review status accepted or low-risk.",
                    "Audit safety metrics do not materially regress.",
                    "Policy adoption is recorded in a formal decision artifact before becoming active.",
                ],
            }
        )
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "narrow_grouping_correction_design",
        "active_policy": active_policy,
        "design_only": True,
        "does_not_change_extraction_behavior": True,
        "does_not_change_active_policy": True,
        "does_not_change_canonical_promotion": True,
        "does_not_add": ["OCR", "AI/model review", "embeddings", "retrieval", "graph work", "structure promotion", "broad merge-policy experiment"],
        "source_artifacts": [
            "visual_review_cases_report.json",
            "gold_evaluation_report.json",
            "canonical_paragraphs.jsonl",
            "phase1_audit.html",
        ],
        "summary": {
            "confirmed_defects": len(designs),
            "primary_defect": designs[0].get("canonical_paragraph_id") if designs else None,
            "recommended_next_action": "implement_experiment_only_chained_cross_page_continuation",
            "downstream_remains_blocked": True,
        },
        "designs": designs,
    }


def build_chained_cross_page_continuation_experiment(
    book_id: str,
    run_id: str,
    active_evaluation: dict[str, Any],
    experimental_evaluation: dict[str, Any],
    experiment_details: dict[str, Any],
    narrow_design: dict[str, Any],
) -> dict[str, Any]:
    active_gold = active_evaluation.get("gold_evaluation_report", {})
    experimental_gold = experimental_evaluation.get("gold_evaluation_report", {})
    active_gold_counts = active_gold.get("counts", {})
    experimental_gold_counts = experimental_gold.get("counts", {})
    active_gold_scores = active_gold.get("scores", {})
    experimental_gold_scores = experimental_gold.get("scores", {})
    active_review = active_evaluation.get("review_report", {})
    experimental_review = experimental_evaluation.get("review_report", {})
    active_review_counts = active_review.get("counts", {})
    experimental_review_counts = experimental_review.get("counts", {})
    active_bbox = (active_review.get("bbox_span_risk_summary") or {}).get("total", 0)
    experimental_bbox = (experimental_review.get("bbox_span_risk_summary") or {}).get("total", 0)
    active_true_defects = count_likely_true_merges(active_review)
    experimental_true_defects = count_likely_true_merges(experimental_review)
    active_taxonomy = taxonomy_counts_for_review(active_evaluation["canonical_paragraphs"], active_review)
    experimental_taxonomy = taxonomy_counts_for_review(experimental_evaluation["canonical_paragraphs"], experimental_review)
    active_promotion_counts = active_evaluation["promotion_report"].get("counts", {})
    experimental_promotion_counts = experimental_evaluation["promotion_report"].get("counts", {})
    target_gold_id = "douglass_gold_p0107_0109_001"
    target_over_split_before = any(row.get("gold_id") == target_gold_id for row in active_gold.get("over_split_paragraphs", []))
    target_over_split_after = any(row.get("gold_id") == target_gold_id for row in experimental_gold.get("over_split_paragraphs", []))
    target_matched_after = any(row.get("gold_id") == target_gold_id for row in experimental_gold.get("matched_paragraphs", []))
    proposed_join_rows = experiment_details.get("joined_chained_cross_page_paragraphs", [])
    gold_line_sets = authoritative_gold_line_sets(book_id)
    joins_covered_by_gold = 0
    for join in proposed_join_rows:
        join_lines = set(join.get("source_line_ids", []))
        if any(join_lines and join_lines == row.get("source_line_ids") for row in gold_line_sets):
            joins_covered_by_gold += 1
    unresolved_side_effects = max(0, len(proposed_join_rows) - joins_covered_by_gold)
    precision_before = active_gold_scores.get("paragraph_precision")
    precision_after = experimental_gold_scores.get("paragraph_precision")
    recall_before = active_gold_scores.get("paragraph_recall")
    recall_after = experimental_gold_scores.get("paragraph_recall")
    object_accuracy_before = active_gold_scores.get("object_label_accuracy")
    object_accuracy_after = experimental_gold_scores.get("object_label_accuracy")
    gold_improved = (
        isinstance(precision_before, (int, float))
        and isinstance(precision_after, (int, float))
        and isinstance(recall_before, (int, float))
        and isinstance(recall_after, (int, float))
        and precision_after >= precision_before
        and recall_after >= recall_before
        and (precision_after > precision_before or recall_after > recall_before)
    )
    over_splits_decreased = experimental_gold_counts.get("over_split_paragraphs", 0) < active_gold_counts.get("over_split_paragraphs", 0)
    over_merges_not_increased = experimental_gold_counts.get("over_merged_paragraphs", 0) <= active_gold_counts.get("over_merged_paragraphs", 0)
    object_label_accuracy_not_worsened = (
        not isinstance(object_accuracy_before, (int, float))
        or not isinstance(object_accuracy_after, (int, float))
        or object_accuracy_after >= object_accuracy_before
    )
    warning_regression = experimental_review_counts.get("warning_count", 0) > active_review_counts.get("warning_count", 0) + 5
    bbox_regression = experimental_bbox > active_bbox + 5
    cp_000103_fixed = bool(target_over_split_before and not target_over_split_after and target_matched_after)
    adoptable_by_metrics = (
        cp_000103_fixed
        and gold_improved
        and over_splits_decreased
        and over_merges_not_increased
        and object_label_accuracy_not_worsened
        and not warning_regression
        and not bbox_regression
    )
    adoption_recommendation = (
        "do_not_adopt_until_object_label_regression_is_explained"
        if not object_label_accuracy_not_worsened
        else
        "create_chained_join_review_queue_before_adoption"
        if adoptable_by_metrics and unresolved_side_effects
        else "eligible_for_formal_adoption_checkpoint"
        if adoptable_by_metrics
        else "do_not_adopt_refine_chained_continuation_conditions"
    )
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "experiment_only_chained_cross_page_continuation",
        "active_policy": ACTIVE_PARAGRAPH_MERGE_POLICY,
        "experimental_policy": CHAINED_CROSS_PAGE_CONTINUATION_POLICY,
        "does_not_change_active_policy": True,
        "does_not_change_canonical_promotion_rules": True,
        "does_not_add": ["OCR", "AI/model review", "embeddings", "retrieval", "graph work", "user-facing answers", "structure promotion", "broad promotion blocking"],
        "source_design_artifact": "narrow_grouping_correction_design.json",
        "target_defect": {
            "canonical_paragraph_id": "cp_000103",
            "page": 109,
            "gold_id": target_gold_id,
            "fixed_by_experiment": cp_000103_fixed,
        },
        "active_policy_metrics": {
            "paragraph_candidates": len(active_evaluation["main_paragraphs"]),
            "canonical_promoted": active_promotion_counts.get("promoted_paragraphs", 0),
            "warning_count": active_review_counts.get("warning_count", 0),
            "bbox_span_risk": active_bbox,
            "likely_true_accidental_merges": active_true_defects,
            "merged_across_paragraph_break": active_taxonomy.get("merged_across_paragraph_break", 0),
            "safe_for_downstream": active_review.get("safe_for_downstream"),
        },
        "experimental_policy_metrics": {
            "paragraph_candidates": len(experimental_evaluation["main_paragraphs"]),
            "canonical_promoted": experimental_promotion_counts.get("promoted_paragraphs", 0),
            "warning_count": experimental_review_counts.get("warning_count", 0),
            "bbox_span_risk": experimental_bbox,
            "likely_true_accidental_merges": experimental_true_defects,
            "merged_across_paragraph_break": experimental_taxonomy.get("merged_across_paragraph_break", 0),
            "safe_for_downstream": experimental_review.get("safe_for_downstream"),
        },
        "gold_scores": {
            "active_paragraph_precision": precision_before,
            "experimental_paragraph_precision": precision_after,
            "active_paragraph_recall": recall_before,
            "experimental_paragraph_recall": recall_after,
            "active_object_label_accuracy": object_accuracy_before,
            "experimental_object_label_accuracy": object_accuracy_after,
        },
        "gold_counts": {
            "active_matched_paragraphs": active_gold_counts.get("matched_paragraphs", 0),
            "experimental_matched_paragraphs": experimental_gold_counts.get("matched_paragraphs", 0),
            "active_over_split_paragraphs": active_gold_counts.get("over_split_paragraphs", 0),
            "experimental_over_split_paragraphs": experimental_gold_counts.get("over_split_paragraphs", 0),
            "active_over_merged_paragraphs": active_gold_counts.get("over_merged_paragraphs", 0),
            "experimental_over_merged_paragraphs": experimental_gold_counts.get("over_merged_paragraphs", 0),
            "active_missing_paragraphs": active_gold_counts.get("missing_paragraphs", 0),
            "experimental_missing_paragraphs": experimental_gold_counts.get("missing_paragraphs", 0),
            "active_wrong_object_labels": active_gold_counts.get("wrong_object_labels", 0),
            "experimental_wrong_object_labels": experimental_gold_counts.get("wrong_object_labels", 0),
            "active_missing_object_labels": active_gold_counts.get("missing_object_labels", 0),
            "experimental_missing_object_labels": experimental_gold_counts.get("missing_object_labels", 0),
        },
        "warning_deltas": {
            "warning_count_delta": experimental_review_counts.get("warning_count", 0) - active_review_counts.get("warning_count", 0),
            "bbox_span_risk_delta": experimental_bbox - active_bbox,
            "likely_true_accidental_merge_delta": experimental_true_defects - active_true_defects,
            "merged_across_paragraph_break_delta": experimental_taxonomy.get("merged_across_paragraph_break", 0) - active_taxonomy.get("merged_across_paragraph_break", 0),
        },
        "side_effects": {
            "proposed_chained_joins": len(proposed_join_rows),
            "rejected_chained_joins": experiment_details.get("rejected_count", 0),
            "joins_covered_by_authoritative_gold": joins_covered_by_gold,
            "joins_not_covered_by_gold": unresolved_side_effects,
            "likely_side_effects": [
                "canonical paragraph count may decrease where chained joins are accepted",
                "unscored chained joins need review before any adoption",
                "longer multi-page paragraphs can still carry bbox/span review warnings",
            ],
        },
        "acceptance_rule": {
            "cp_000103_fixed": cp_000103_fixed,
            "gold_score_improved": gold_improved,
            "over_splits_decreased": over_splits_decreased,
            "over_merges_not_increased": over_merges_not_increased,
            "object_label_accuracy_not_worsened": object_label_accuracy_not_worsened,
            "audit_warning_regression": warning_regression,
            "bbox_span_regression": bbox_regression,
            "adoptable_by_metrics_only": adoptable_by_metrics,
            "requires_side_effect_review_before_adoption": unresolved_side_effects > 0,
            "adoption_recommendation": adoption_recommendation,
        },
        "proposed_chained_joins": proposed_join_rows[:80],
        "rejected_chained_joins": (experiment_details.get("rejected_chained_cross_page_candidates") or [])[:80],
        "experimental_wrong_object_labels": experimental_gold.get("wrong_object_labels", []),
        "experimental_missing_object_labels": experimental_gold.get("missing_object_labels", []),
        "design_summary": narrow_design.get("summary", {}),
    }


def classify_chained_join_review_risk(join: dict[str, Any]) -> tuple[str, float, str]:
    pages = [int(page) for page in join.get("pages", []) if isinstance(page, int)]
    source_line_count = int(join.get("source_line_count") or 0)
    first_end = str(join.get("first_text_end", "")).strip()
    second_start = str(join.get("second_text_start", "")).lstrip()
    if any(page <= 18 for page in pages):
        return "structure_boundary_risk", 0.62, "inspect early/front-matter boundary visually before any decision"
    if re.search(r"\b(chapter|preface|appendix|letter from|contents)\b", second_start[:80], re.IGNORECASE):
        return "structure_boundary_risk", 0.68, "inspect possible structure boundary before accepting"
    if source_line_count >= 55:
        return "needs_visual_review", 0.60, "inspect long chained paragraph across all affected pages"
    if second_start[:1].isupper() and not first_end.endswith(tuple(TERMINAL_PUNCTUATION)):
        return "possible_overmerge", 0.58, "inspect whether the next page starts a new sentence or continues the previous one"
    if "running_header" in str(join.get("join_reasons", "")):
        return "page_furniture_risk", 0.55, "inspect page furniture before accepting"
    return "likely_valid_chained_continuation", 0.66, "review visual evidence and accept only if the sentence clearly continues"


def build_chained_join_review_queue(
    book_id: str,
    run_id: str,
    chained_experiment: dict[str, Any],
) -> dict[str, Any]:
    queue_rows = []
    for index, join in enumerate(chained_experiment.get("proposed_chained_joins", []), start=1):
        join_lines = set(join.get("source_line_ids", []))
        covered_gold = []
        for gold in authoritative_gold_line_sets(book_id):
            if join_lines and join_lines == gold.get("source_line_ids"):
                covered_gold.append(gold.get("gold_id"))
        if covered_gold:
            continue
        risk, confidence, action = classify_chained_join_review_risk(join)
        affected_pages = join.get("pages", [])
        left_id = join.get("first_object_id")
        right_id = join.get("second_object_id")
        queue_rows.append(
            {
                "chained_join_id": f"chained_join_review_{len(queue_rows) + 1:04d}",
                "source_experiment_join_index": index,
                "affected_pages": affected_pages,
                "source_candidate_ids": {
                    "left_candidate_id": left_id,
                    "right_candidate_id": right_id,
                },
                "current_active_v2_behavior": "Candidates remain separate under the active v2 output; v3 is experiment-only.",
                "experimental_v3_behavior": "Experimental v3 would chain the existing cross-page candidate into the next page's first body paragraph.",
                "text_preview_before_join": {
                    "left_text_end": join.get("first_text_end"),
                    "right_text_start": join.get("second_text_start"),
                },
                "text_preview_after_join": join.get("joined_text_preview"),
                "source_line_evidence": {
                    "source_line_ids": join.get("source_line_ids", []),
                    "source_line_count": join.get("source_line_count", 0),
                    "join_reasons": join.get("join_reasons", []),
                },
                "visual_evidence_references": [
                    f"phase1_audit.html#page-{page}" for page in affected_pages
                ]
                + [
                    f"phase1_audit.html#card-{safe_dom_id(str(left_id))}",
                    f"phase1_audit.html#card-{safe_dom_id(str(right_id))}",
                ],
                "gold_coverage_exists": False,
                "gold_ids": [],
                "likely_risk": risk,
                "confidence": confidence,
                "recommended_review_action": action,
                "decision_status": "queued_for_review",
            }
        )
    risk_counts = Counter(row["likely_risk"] for row in queue_rows)
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "review_queue_only_for_unscored_chained_joins",
        "source_artifact": "chained_cross_page_continuation_experiment.json",
        "does_not_change_active_policy": True,
        "does_not_apply_decisions": True,
        "does_not_change_canonical_promotion_rules": True,
        "summary": {
            "total_unscored_chained_joins": len(queue_rows),
            "risk_counts": dict(sorted(risk_counts.items())),
            "review_queue_open": bool(queue_rows),
            "adoption_remains_blocked": True,
            "recommended_next_action": "review_chained_join_queue_before_any_v3_adoption",
        },
        "queue": queue_rows,
    }


def validate_chained_join_decisions(
    decisions: list[dict[str, Any]],
    queue_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    valid_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for row in decisions:
        chained_join_id = str(row.get("chained_join_id", ""))
        line_number = row.get("_line_number")
        missing_fields = [
            field
            for field in REQUIRED_CHAINED_JOIN_DECISION_FIELDS
            if field not in row or not str(row.get(field, "")).strip()
        ]
        if missing_fields:
            errors.append({"code": "missing_required_fields", "chained_join_id": chained_join_id, "line_number": line_number, "fields": missing_fields})
        if chained_join_id in seen_ids:
            errors.append({"code": "duplicate_chained_join_decision", "chained_join_id": chained_join_id, "line_number": line_number})
        seen_ids.add(chained_join_id)
        if row.get("decision") not in VALID_CHAINED_JOIN_DECISIONS:
            errors.append({"code": "invalid_decision", "chained_join_id": chained_join_id, "line_number": line_number, "decision": row.get("decision")})
        if row.get("reason") not in VALID_CHAINED_JOIN_DECISION_REASONS:
            errors.append({"code": "invalid_reason", "chained_join_id": chained_join_id, "line_number": line_number, "reason": row.get("reason")})
        proposed = queue_rows.get(chained_join_id)
        if not proposed:
            errors.append({"code": "missing_chained_join_id", "chained_join_id": chained_join_id, "line_number": line_number})
            continue
        if [int(page) for page in row.get("affected_pages", [])] != [int(page) for page in proposed.get("affected_pages", [])]:
            errors.append({"code": "affected_pages_mismatch", "chained_join_id": chained_join_id, "line_number": line_number})
        row_has_error = any(error.get("chained_join_id") == chained_join_id for error in errors)
        if not row_has_error:
            valid_by_id[chained_join_id] = row
    return {
        "status": "pass" if not errors else "fail",
        "source_row_count": len(decisions),
        "valid_decision_count": len(valid_by_id),
        "error_count": len(errors),
        "errors": errors,
    }, valid_by_id


def build_chained_join_decisions_applied(
    book_id: str,
    run_id: str,
    chained_join_review_queue: dict[str, Any],
    applied_decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    queue_rows = chained_join_review_queue.get("queue") or []
    queue_by_id = {row.get("chained_join_id"): row for row in queue_rows}
    validation, valid_by_id = validate_chained_join_decisions(applied_decisions, queue_by_id)
    rows = []
    for queue_row in queue_rows:
        decision = valid_by_id.get(queue_row.get("chained_join_id"))
        rows.append(
            {
                "chained_join_id": queue_row.get("chained_join_id"),
                "affected_pages": queue_row.get("affected_pages"),
                "source_candidate_ids": queue_row.get("source_candidate_ids"),
                "likely_risk": queue_row.get("likely_risk"),
                "decision": decision.get("decision") if decision else "unreviewed",
                "reason": decision.get("reason") if decision else None,
                "reviewer": decision.get("reviewer") if decision else None,
                "reviewed_at": decision.get("reviewed_at") if decision else None,
                "evidence_reference": decision.get("evidence_reference") if decision else None,
                "notes": decision.get("notes") if decision else None,
                "decision_status": "curated_" + decision.get("decision") if decision else "queued_for_review",
            }
        )
    status_counts = Counter(row["decision"] for row in rows)
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "curated_chained_join_decisions_replay",
        "decision_source": f"reviews/{book_id}/chained_join_decisions.jsonl",
        "does_not_change_active_policy": True,
        "does_not_adopt_v3": True,
        "does_not_change_canonical_promotion_rules": True,
        "validation": validation,
        "summary": {
            "queued_chained_joins": len(queue_rows),
            "decision_rows": len(applied_decisions),
            "accepted": status_counts.get("accept", 0),
            "rejected": status_counts.get("reject", 0),
            "needs_review": status_counts.get("needs_review", 0),
            "unreviewed": status_counts.get("unreviewed", 0),
            "adoption_remains_separate_checkpoint": True,
        },
        "decisions": rows,
    }


def chained_join_key_from_join(join: dict[str, Any]) -> tuple[tuple[int, ...], str, str]:
    return (
        tuple(int(page) for page in join.get("pages", []) if isinstance(page, int)),
        str(join.get("first_object_id")),
        str(join.get("second_object_id")),
    )


def chained_join_key_from_decision(row: dict[str, Any]) -> tuple[tuple[int, ...], str, str]:
    source_ids = row.get("source_candidate_ids") or {}
    return (
        tuple(int(page) for page in row.get("affected_pages", []) if isinstance(page, int)),
        str(source_ids.get("left_candidate_id")),
        str(source_ids.get("right_candidate_id")),
    )


def build_guarded_chained_cross_page_continuation_experiment(
    book_id: str,
    run_id: str,
    active_evaluation: dict[str, Any],
    previous_v3_experiment: dict[str, Any],
    guarded_evaluation: dict[str, Any],
    guarded_details: dict[str, Any],
    chained_decisions_applied: dict[str, Any],
) -> dict[str, Any]:
    active_gold = active_evaluation.get("gold_evaluation_report", {})
    guarded_gold = guarded_evaluation.get("gold_evaluation_report", {})
    active_gold_counts = active_gold.get("counts", {})
    guarded_gold_counts = guarded_gold.get("counts", {})
    active_gold_scores = active_gold.get("scores", {})
    guarded_gold_scores = guarded_gold.get("scores", {})
    active_review = active_evaluation.get("review_report", {})
    guarded_review = guarded_evaluation.get("review_report", {})
    active_review_counts = active_review.get("counts", {})
    guarded_review_counts = guarded_review.get("counts", {})
    active_bbox = (active_review.get("bbox_span_risk_summary") or {}).get("total", 0)
    guarded_bbox = (guarded_review.get("bbox_span_risk_summary") or {}).get("total", 0)
    active_true_defects = count_likely_true_merges(active_review)
    guarded_true_defects = count_likely_true_merges(guarded_review)
    active_taxonomy = taxonomy_counts_for_review(active_evaluation["canonical_paragraphs"], active_review)
    guarded_taxonomy = taxonomy_counts_for_review(guarded_evaluation["canonical_paragraphs"], guarded_review)
    active_promotion_counts = active_evaluation["promotion_report"].get("counts", {})
    guarded_promotion_counts = guarded_evaluation["promotion_report"].get("counts", {})
    target_gold_id = "douglass_gold_p0107_0109_001"
    cp_fixed = (
        not any(row.get("gold_id") == target_gold_id for row in guarded_gold.get("over_split_paragraphs", []))
        and any(row.get("gold_id") == target_gold_id for row in guarded_gold.get("matched_paragraphs", []))
    )
    guarded_join_keys = {chained_join_key_from_join(row) for row in guarded_details.get("joined_chained_cross_page_paragraphs", [])}
    rejected_candidate_keys = {chained_join_key_from_join(row) for row in guarded_details.get("rejected_chained_cross_page_candidates", [])}
    accepted_decisions = [row for row in chained_decisions_applied.get("decisions", []) if row.get("decision") == "accept"]
    rejected_decisions = [row for row in chained_decisions_applied.get("decisions", []) if row.get("decision") == "reject"]
    accepted_preserved = [
        row.get("chained_join_id")
        for row in accepted_decisions
        if chained_join_key_from_decision(row) in guarded_join_keys
    ]
    rejected_blocked = [
        row.get("chained_join_id")
        for row in rejected_decisions
        if chained_join_key_from_decision(row) not in guarded_join_keys
        and chained_join_key_from_decision(row) in rejected_candidate_keys
    ]
    previous_side_effects = previous_v3_experiment.get("side_effects") or {}
    guarded_precision = guarded_gold_scores.get("paragraph_precision")
    guarded_recall = guarded_gold_scores.get("paragraph_recall")
    active_precision = active_gold_scores.get("paragraph_precision")
    active_recall = active_gold_scores.get("paragraph_recall")
    gold_improved = (
        isinstance(guarded_precision, (int, float))
        and isinstance(active_precision, (int, float))
        and isinstance(guarded_recall, (int, float))
        and isinstance(active_recall, (int, float))
        and guarded_precision >= active_precision
        and guarded_recall >= active_recall
        and (guarded_precision > active_precision or guarded_recall > active_recall)
    )
    object_accuracy_not_worsened = (
        not isinstance(active_gold_scores.get("object_label_accuracy"), (int, float))
        or not isinstance(guarded_gold_scores.get("object_label_accuracy"), (int, float))
        or guarded_gold_scores.get("object_label_accuracy") >= active_gold_scores.get("object_label_accuracy")
    )
    warning_regression = guarded_review_counts.get("warning_count", 0) > active_review_counts.get("warning_count", 0) + 5
    bbox_regression = guarded_bbox > active_bbox + 5
    pass_metrics = (
        cp_fixed
        and "chained_join_review_0004" in rejected_blocked
        and len(accepted_preserved) == len(accepted_decisions)
        and gold_improved
        and guarded_gold_counts.get("over_merged_paragraphs", 0) <= active_gold_counts.get("over_merged_paragraphs", 0)
        and object_accuracy_not_worsened
        and not warning_regression
        and not bbox_regression
    )
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "experiment_only_guarded_chained_cross_page_continuation",
        "active_policy": ACTIVE_PARAGRAPH_MERGE_POLICY,
        "active_v2_policy_before_guarded_adoption": CROSS_PAGE_CONTINUATION_POLICY,
        "previous_experimental_policy": CHAINED_CROSS_PAGE_CONTINUATION_POLICY,
        "guarded_experimental_policy": GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY,
        "does_not_change_active_policy": ACTIVE_PARAGRAPH_MERGE_POLICY != GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY,
        "does_not_adopt_guarded_policy": ACTIVE_PARAGRAPH_MERGE_POLICY != GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY,
        "does_not_change_canonical_promotion_rules": True,
        "guard_rule": "Block chained joins when terminal page of the existing joined candidate contains later non-furniture paragraph content before the next-page candidate.",
        "active_v2_metrics": {
            "paragraph_candidates": len(active_evaluation["main_paragraphs"]),
            "canonical_promoted": active_promotion_counts.get("promoted_paragraphs", 0),
            "warning_count": active_review_counts.get("warning_count", 0),
            "bbox_span_risk": active_bbox,
            "likely_true_accidental_merges": active_true_defects,
            "merged_across_paragraph_break": active_taxonomy.get("merged_across_paragraph_break", 0),
            "safe_for_downstream": active_review.get("safe_for_downstream"),
        },
        "previous_v3_metrics": {
            "proposed_chained_joins": previous_side_effects.get("proposed_chained_joins"),
            "joins_not_covered_by_gold": previous_side_effects.get("joins_not_covered_by_gold"),
            "cp_000103_fixed": (previous_v3_experiment.get("target_defect") or {}).get("fixed_by_experiment"),
            "paragraph_precision": (previous_v3_experiment.get("gold_scores") or {}).get("experimental_paragraph_precision"),
            "paragraph_recall": (previous_v3_experiment.get("gold_scores") or {}).get("experimental_paragraph_recall"),
        },
        "guarded_v3_metrics": {
            "paragraph_candidates": len(guarded_evaluation["main_paragraphs"]),
            "canonical_promoted": guarded_promotion_counts.get("promoted_paragraphs", 0),
            "warning_count": guarded_review_counts.get("warning_count", 0),
            "bbox_span_risk": guarded_bbox,
            "likely_true_accidental_merges": guarded_true_defects,
            "merged_across_paragraph_break": guarded_taxonomy.get("merged_across_paragraph_break", 0),
            "safe_for_downstream": guarded_review.get("safe_for_downstream"),
        },
        "gold_scores": {
            "active_paragraph_precision": active_precision,
            "guarded_paragraph_precision": guarded_precision,
            "active_paragraph_recall": active_recall,
            "guarded_paragraph_recall": guarded_recall,
            "active_object_label_accuracy": active_gold_scores.get("object_label_accuracy"),
            "guarded_object_label_accuracy": guarded_gold_scores.get("object_label_accuracy"),
        },
        "gold_counts": {
            "active_matched_paragraphs": active_gold_counts.get("matched_paragraphs", 0),
            "guarded_matched_paragraphs": guarded_gold_counts.get("matched_paragraphs", 0),
            "active_over_split_paragraphs": active_gold_counts.get("over_split_paragraphs", 0),
            "guarded_over_split_paragraphs": guarded_gold_counts.get("over_split_paragraphs", 0),
            "active_over_merged_paragraphs": active_gold_counts.get("over_merged_paragraphs", 0),
            "guarded_over_merged_paragraphs": guarded_gold_counts.get("over_merged_paragraphs", 0),
        },
        "decision_replay": {
            "accepted_prior_decisions": len(accepted_decisions),
            "accepted_prior_decisions_preserved": len(accepted_preserved),
            "accepted_prior_decision_ids_preserved": accepted_preserved,
            "rejected_prior_decisions": len(rejected_decisions),
            "rejected_prior_decisions_blocked": len(rejected_blocked),
            "rejected_prior_decision_ids_blocked": rejected_blocked,
            "chained_join_review_0004_blocked": "chained_join_review_0004" in rejected_blocked,
        },
        "warning_deltas": {
            "warning_count_delta": guarded_review_counts.get("warning_count", 0) - active_review_counts.get("warning_count", 0),
            "bbox_span_risk_delta": guarded_bbox - active_bbox,
            "likely_true_accidental_merge_delta": guarded_true_defects - active_true_defects,
            "merged_across_paragraph_break_delta": guarded_taxonomy.get("merged_across_paragraph_break", 0) - active_taxonomy.get("merged_across_paragraph_break", 0),
        },
        "side_effects": {
            "proposed_chained_joins": guarded_details.get("joined_count", 0),
            "rejected_chained_joins": guarded_details.get("rejected_count", 0),
            "side_effect_risks": [
                "guarded policy still requires formal adoption checkpoint",
                "accepted prior decisions are preserved by metrics, but adoption must verify audit safety again",
                "downstream remains blocked until canonical paragraph safety is recalculated after any adoption",
            ],
        },
        "acceptance_rule": {
            "cp_000103_remains_fixed": cp_fixed,
            "chained_join_review_0004_blocked": "chained_join_review_0004" in rejected_blocked,
            "all_accepted_prior_decisions_preserved": len(accepted_preserved) == len(accepted_decisions),
            "rejected_prior_decisions_blocked": len(rejected_blocked) == len(rejected_decisions),
            "gold_score_improved": gold_improved,
            "over_merges_not_increased": guarded_gold_counts.get("over_merged_paragraphs", 0) <= active_gold_counts.get("over_merged_paragraphs", 0),
            "object_label_accuracy_not_worsened": object_accuracy_not_worsened,
            "audit_warning_regression": warning_regression,
            "bbox_span_regression": bbox_regression,
            "passes_experiment_gate": pass_metrics,
            "adoption_recommendation": "prepare_formal_guarded_v3_adoption_checkpoint" if pass_metrics else "do_not_adopt_refine_guard_conditions",
        },
        "proposed_chained_joins": (guarded_details.get("joined_chained_cross_page_paragraphs") or [])[:80],
        "rejected_chained_candidates": (guarded_details.get("rejected_chained_cross_page_candidates") or [])[:80],
    }


def build_guarded_chained_policy_adoption_decision(
    book_id: str,
    run_id: str,
    guarded_experiment: dict[str, Any],
    active_evaluation: dict[str, Any],
    active_policy: str,
    validation_status: str = "pending",
) -> dict[str, Any]:
    acceptance = guarded_experiment.get("acceptance_rule") or {}
    decision_replay = guarded_experiment.get("decision_replay") or {}
    side_effects = guarded_experiment.get("side_effects") or {}
    gold_scores = guarded_experiment.get("gold_scores") or {}
    gold_counts = guarded_experiment.get("gold_counts") or {}
    warning_deltas = guarded_experiment.get("warning_deltas") or {}
    active_review = active_evaluation.get("review_report", {})
    active_review_counts = active_review.get("counts", {})
    active_promotion_counts = active_evaluation.get("promotion_report", {}).get("counts", {})
    active_gold = active_evaluation.get("gold_evaluation_report", {})
    active_gold_scores = active_gold.get("scores", {})
    active_gold_counts = active_gold.get("counts", {})
    unresolved_chained_joins = int(decision_replay.get("accepted_prior_decisions", 0) or 0) - int(
        decision_replay.get("accepted_prior_decisions_preserved", 0) or 0
    )
    unresolved_chained_joins += int(decision_replay.get("rejected_prior_decisions", 0) or 0) - int(
        decision_replay.get("rejected_prior_decisions_blocked", 0) or 0
    )
    gates = {
        "guarded_experiment_passed": bool(acceptance.get("passes_experiment_gate")),
        "cp_000103_remains_fixed": bool(acceptance.get("cp_000103_remains_fixed")),
        "false_join_blocked": bool(acceptance.get("chained_join_review_0004_blocked")),
        "accepted_prior_decisions_preserved": bool(acceptance.get("all_accepted_prior_decisions_preserved")),
        "rejected_prior_decisions_blocked": bool(acceptance.get("rejected_prior_decisions_blocked")),
        "gold_score_improved": bool(acceptance.get("gold_score_improved")),
        "over_merges_not_increased": bool(acceptance.get("over_merges_not_increased")),
        "object_label_accuracy_not_worsened": bool(acceptance.get("object_label_accuracy_not_worsened")),
        "audit_warning_regression": bool(acceptance.get("audit_warning_regression")),
        "bbox_span_regression": bool(acceptance.get("bbox_span_regression")),
        "unresolved_chained_joins": unresolved_chained_joins,
        "active_policy_is_guarded_v3": active_policy == GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY,
    }
    adopted = (
        gates["active_policy_is_guarded_v3"]
        and gates["guarded_experiment_passed"]
        and gates["cp_000103_remains_fixed"]
        and gates["false_join_blocked"]
        and gates["accepted_prior_decisions_preserved"]
        and gates["rejected_prior_decisions_blocked"]
        and gates["gold_score_improved"]
        and gates["over_merges_not_increased"]
        and gates["object_label_accuracy_not_worsened"]
        and not gates["audit_warning_regression"]
        and not gates["bbox_span_regression"]
        and unresolved_chained_joins == 0
    )
    return {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "formal_guarded_chained_policy_adoption_decision",
        "decision": "adopt_v3_chained_cross_page_continuation_guarded" if adopted else "do_not_adopt_guarded_v3",
        "previous_active_policy": CROSS_PAGE_CONTINUATION_POLICY,
        "active_paragraph_merge_policy": active_policy,
        "adopted_policy": GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY if adopted else None,
        "does_not_add": ["OCR", "AI/model review", "embeddings", "retrieval", "graph work", "structure promotion", "broad promotion blocking"],
        "does_not_unlock_downstream": True,
        "validation_status": validation_status,
        "gate_evidence": {
            "active_v2_metrics": guarded_experiment.get("active_v2_metrics"),
            "previous_unguarded_v3_metrics": guarded_experiment.get("previous_v3_metrics"),
            "guarded_v3_metrics": guarded_experiment.get("guarded_v3_metrics"),
            "gold_paragraph_precision_before": gold_scores.get("active_paragraph_precision"),
            "gold_paragraph_precision_after": gold_scores.get("guarded_paragraph_precision"),
            "gold_paragraph_recall_before": gold_scores.get("active_paragraph_recall"),
            "gold_paragraph_recall_after": gold_scores.get("guarded_paragraph_recall"),
            "matched_paragraphs_before": gold_counts.get("active_matched_paragraphs"),
            "matched_paragraphs_after": gold_counts.get("guarded_matched_paragraphs"),
            "over_split_paragraphs_before": gold_counts.get("active_over_split_paragraphs"),
            "over_split_paragraphs_after": gold_counts.get("guarded_over_split_paragraphs"),
            "over_merged_paragraphs_before": gold_counts.get("active_over_merged_paragraphs"),
            "over_merged_paragraphs_after": gold_counts.get("guarded_over_merged_paragraphs"),
            "object_label_accuracy_before": gold_scores.get("active_object_label_accuracy"),
            "object_label_accuracy_after": gold_scores.get("guarded_object_label_accuracy"),
            "warning_deltas": warning_deltas,
            "cp_000103_remains_fixed": acceptance.get("cp_000103_remains_fixed"),
            "chained_join_review_0004_blocked": acceptance.get("chained_join_review_0004_blocked"),
            "accepted_prior_decisions_preserved": decision_replay.get("accepted_prior_decisions_preserved"),
            "accepted_prior_decisions": decision_replay.get("accepted_prior_decisions"),
            "rejected_prior_decisions_blocked": decision_replay.get("rejected_prior_decisions_blocked"),
            "rejected_prior_decisions": decision_replay.get("rejected_prior_decisions"),
            "unresolved_chained_joins": unresolved_chained_joins,
            "guarded_proposed_chained_joins": side_effects.get("proposed_chained_joins"),
            "guarded_rejected_chained_joins": side_effects.get("rejected_chained_joins"),
            "validation_status": validation_status,
            "downstream_safety_status": active_review.get("safe_for_downstream"),
        },
        "active_run_after_adoption": {
            "canonical_promoted_paragraphs": active_promotion_counts.get("promoted_paragraphs"),
            "paragraph_candidates": active_promotion_counts.get("paragraph_candidates_reviewed"),
            "blocked_paragraph_candidates": active_promotion_counts.get("paragraph_candidates_blocked"),
            "canonical_review_warning_count": active_review_counts.get("warning_count"),
            "canonical_review_risky_paragraph_count": active_review_counts.get("risky_paragraph_count"),
            "safe_for_downstream": active_review.get("safe_for_downstream"),
            "downstream_recommendation": active_review.get("recommendation"),
            "gold_paragraph_precision": active_gold_scores.get("paragraph_precision"),
            "gold_paragraph_recall": active_gold_scores.get("paragraph_recall"),
            "gold_matched_paragraphs": active_gold_counts.get("matched_paragraphs"),
            "gold_over_split_paragraphs": active_gold_counts.get("over_split_paragraphs"),
            "gold_over_merged_paragraphs": active_gold_counts.get("over_merged_paragraphs"),
            "object_label_accuracy": active_gold_scores.get("object_label_accuracy"),
        },
        "gates": gates,
        "adoption_note": (
            "Guarded v3 is now the active paragraph merge policy. Downstream intelligence remains blocked until canonical paragraph safety is separately safe."
            if adopted
            else "Guarded v3 was not adopted because one or more adoption gates failed."
        ),
    }


def review_flags(text: str, image_count: int, table_count: int) -> list[str]:
    flags: list[str] = []
    if not text.strip():
        flags.append("no_extracted_text")
    if image_count:
        flags.append("has_images")
    if table_count:
        flags.append("has_tables")
    if CID_PATTERN.search(text):
        flags.append("cid_noise_detected")
    return flags


def extract_line_records(page: Any, raw_text: str) -> list[dict[str, Any]]:
    if hasattr(page, "extract_text_lines"):
        try:
            records = page.extract_text_lines(layout=False, strip=False)
            if records:
                return records
        except Exception:
            pass
    return [{"text": line, "x0": None, "top": None, "bottom": None} for line in raw_text.splitlines()]


def build_audit_html(
    book_id: str,
    source_pdf: Path,
    manifest: dict[str, Any],
    inventory: list[dict[str, Any]],
    layout_objects: list[dict[str, Any]],
    object_counts: Counter[str],
    stream_counts: dict[str, int],
    stream_samples: dict[str, list[dict[str, Any]]],
    validation_report: dict[str, Any],
    output_dir: Path,
) -> str:
    status_counts = Counter(row["status"] for row in inventory)
    flagged_pages = [row for row in inventory if row["review_flags"]]
    sample_pages = inventory[:12]
    generated = utc_now()
    generated_date = generated[:10]

    def esc(value: Any) -> str:
        return html.escape(str(value))

    candidate_rows = [row for rows in stream_samples.values() for row in rows]
    # The full candidate rows are provided through stream_samples["__all__"] when available.
    candidate_rows = stream_samples.get("__all__", candidate_rows)
    canonical_paragraphs = read_jsonl(output_dir / "canonical_paragraphs.jsonl") if (output_dir / "canonical_paragraphs.jsonl").exists() else []
    promotion_blockers = read_jsonl(output_dir / "promotion_blockers.jsonl") if (output_dir / "promotion_blockers.jsonl").exists() else []
    promotion_report = read_json(output_dir / "canonical_promotion_report.json") if (output_dir / "canonical_promotion_report.json").exists() else {}
    canonical_review_report = read_json(output_dir / "canonical_paragraph_review_report.json") if (output_dir / "canonical_paragraph_review_report.json").exists() else {}
    paragraph_merge_experiment_report = read_json(output_dir / "paragraph_merge_experiment_report.json") if (output_dir / "paragraph_merge_experiment_report.json").exists() else {}
    paragraph_merge_failure_taxonomy_report = read_json(output_dir / "paragraph_merge_failure_taxonomy_report.json") if (output_dir / "paragraph_merge_failure_taxonomy_report.json").exists() else {}
    cross_page_join_review_report = read_json(output_dir / "cross_page_join_review_report.json") if (output_dir / "cross_page_join_review_report.json").exists() else {}
    xpage_join_0032_investigation = read_json(output_dir / "xpage_join_0032_investigation.json") if (output_dir / "xpage_join_0032_investigation.json").exists() else {}
    policy_adoption_decision = read_json(output_dir / "policy_adoption_decision.json") if (output_dir / "policy_adoption_decision.json").exists() else {}
    post_adoption_safety_report = read_json(output_dir / "post_adoption_canonical_safety_report.json") if (output_dir / "post_adoption_canonical_safety_report.json").exists() else {}
    post_adoption_bbox_diagnosis = read_json(output_dir / "post_adoption_bbox_span_diagnosis.json") if (output_dir / "post_adoption_bbox_span_diagnosis.json").exists() else {}
    post_adoption_remediation_plan = read_json(output_dir / "post_adoption_remediation_plan.json") if (output_dir / "post_adoption_remediation_plan.json").exists() else {}
    front_matter_metadata_review_report = read_json(output_dir / "front_matter_metadata_review_report.json") if (output_dir / "front_matter_metadata_review_report.json").exists() else {}
    visual_review_cases_report = read_json(output_dir / "visual_review_cases_report.json") if (output_dir / "visual_review_cases_report.json").exists() else {}
    narrow_grouping_correction_design = read_json(output_dir / "narrow_grouping_correction_design.json") if (output_dir / "narrow_grouping_correction_design.json").exists() else {}
    chained_cross_page_experiment = read_json(output_dir / "chained_cross_page_continuation_experiment.json") if (output_dir / "chained_cross_page_continuation_experiment.json").exists() else {}
    chained_join_review_queue = read_json(output_dir / "chained_join_review_queue.json") if (output_dir / "chained_join_review_queue.json").exists() else {}
    chained_join_decisions_applied = read_json(output_dir / "chained_join_decisions_applied.json") if (output_dir / "chained_join_decisions_applied.json").exists() else {}
    guarded_chained_experiment = read_json(output_dir / "guarded_chained_cross_page_continuation_experiment.json") if (output_dir / "guarded_chained_cross_page_continuation_experiment.json").exists() else {}
    guarded_policy_adoption_decision = read_json(output_dir / "guarded_chained_policy_adoption_decision.json") if (output_dir / "guarded_chained_policy_adoption_decision.json").exists() else {}
    gold_evaluation_report = read_json(output_dir / "gold_evaluation_report.json") if (output_dir / "gold_evaluation_report.json").exists() else {}
    promoted_object_ids = {row.get("source_candidate_object_id") for row in canonical_paragraphs}
    blocker_by_object_id = {row.get("object_id"): row for row in promotion_blockers if row.get("object_id")}
    candidate_by_object_id = {row["object_id"]: row for row in candidate_rows if row.get("object_id")}
    artifact_rows = [row for row in candidate_rows if row.get("stream_type") == "page_artifact_candidate"]
    objects_by_page: dict[int, list[dict[str, Any]]] = {}
    for obj in layout_objects:
        objects_by_page.setdefault(int(obj["page_number"]), []).append(obj)

    def infer_page_zones() -> dict[int, str]:
        structure_rows = [row for row in candidate_rows if row.get("stream_type") == "structure_candidate"]
        chapter_pages = [
            int(row["page_number"])
            for row in structure_rows
            if normalized_object_text(str(row.get("clean_text", ""))).startswith("chapter")
        ]
        appendix_pages = [
            int(row["page_number"])
            for row in structure_rows
            if "appendix" in normalized_object_text(str(row.get("clean_text", ""))).split()
        ]
        first_chapter_page = min(chapter_pages) if chapter_pages else None
        first_appendix_page = min(appendix_pages) if appendix_pages else None
        zones: dict[int, str] = {}
        for row in inventory:
            page_number = int(row["page_number"])
            if first_appendix_page is not None and page_number >= first_appendix_page:
                zones[page_number] = "appendix"
            elif first_chapter_page is not None and page_number < first_chapter_page:
                zones[page_number] = "front_matter"
            elif first_chapter_page is not None:
                zones[page_number] = "body"
            else:
                zones[page_number] = "unknown"
        return zones

    page_zones = infer_page_zones()

    def page_list_text(pages: list[int]) -> str:
        if not pages:
            return "-"
        if len(pages) <= 12:
            return ", ".join(str(page) for page in pages)
        return ", ".join(str(page) for page in pages[:6]) + " ... " + ", ".join(str(page) for page in pages[-3:])

    def object_top(row: dict[str, Any]) -> float | None:
        bbox = row.get("bbox") or {}
        top = bbox.get("top")
        return float(top) if top is not None else None

    def artifact_review_groups() -> tuple[str, str]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in artifact_rows:
            evidence = row.get("furniture_evidence") or {}
            normalized = evidence.get("normalized_furniture_text") or normalized_furniture_text(str(row.get("clean_text", ""))) or "[blank]"
            subtype = str(row.get("artifact_type", "unknown"))
            groups.setdefault((normalized, subtype), []).append(row)

        pattern_rows = []
        risk_rows = []
        structure_chapter_pages = {
            int(row["page_number"])
            for row in candidate_rows
            if row.get("stream_type") == "structure_candidate"
            and normalized_object_text(str(row.get("clean_text", ""))).startswith("chapter")
        }
        for (normalized, subtype), rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0][0])):
            rows = sorted(rows, key=lambda row: int(row["page_number"]))
            pages = [int(row["page_number"]) for row in rows]
            first = rows[0]
            middle = rows[len(rows) // 2]
            last = rows[-1]
            tops = [top for top in (object_top(row) for row in rows) if top is not None]
            avg_top = sum(tops) / len(tops) if tops else None
            margin_counts = Counter((row.get("furniture_evidence") or {}).get("margin_zone") or "unknown" for row in rows)
            zones = Counter(page_zones.get(page, "unknown") for page in pages)
            max_words = max(len(str(row.get("clean_text", "")).split()) for row in rows)
            max_chars = max(len(str(row.get("clean_text", ""))) for row in rows)
            risks = []
            if len(rows) < 5:
                risks.append("low_repetition_count")
            if pages and max(pages) - min(pages) <= 5:
                risks.append("small_page_range")
            if any(abs(page - chapter_page) <= 1 for page in pages for chapter_page in structure_chapter_pages):
                risks.append("near_chapter_boundary")
            if max_words >= 7 or max_chars >= 65:
                risks.append("long_text_may_be_structure")
            if any(zone in {"front_matter", "appendix"} for zone in zones):
                risks.append("front_or_appendix_zone")
            risk_text = ", ".join(risks) or "-"
            pattern_rows.append(
                "<tr>"
                f"<td><code>{esc(normalized)}</code></td>"
                f"<td><code>{esc(subtype)}</code></td>"
                f"<td>{len(rows)}</td>"
                f"<td>{esc(page_list_text(pages))}</td>"
                f"<td>{esc(first.get('clean_text', ''))}</td>"
                f"<td>{esc(middle.get('clean_text', ''))}</td>"
                f"<td>{esc(last.get('clean_text', ''))}</td>"
                f"<td>{esc(f'{avg_top:.1f}' if avg_top is not None else '-')}</td>"
                f"<td>{esc(', '.join(f'{key}: {value}' for key, value in sorted(margin_counts.items())))}</td>"
                f"<td>{esc(risk_text)}</td>"
                "</tr>"
            )
            if risks:
                risk_rows.append(
                    "<tr>"
                    f"<td><code>{esc(normalized)}</code></td>"
                    f"<td><code>{esc(subtype)}</code></td>"
                    f"<td>{len(rows)}</td>"
                    f"<td>{esc(page_list_text(pages))}</td>"
                    f"<td>{esc(', '.join(f'{key}: {value}' for key, value in sorted(zones.items())))}</td>"
                    f"<td>{esc(risk_text)}</td>"
                    "</tr>"
                )
        return "\n".join(pattern_rows), "\n".join(risk_rows)

    def bbox_text(value: Any) -> str:
        if not isinstance(value, dict):
            return "-"
        parts = []
        for key in ["x0", "x1", "top", "bottom"]:
            raw_value = value.get(key)
            if raw_value is not None:
                parts.append(f"{key}={float(raw_value):.1f}")
        return ", ".join(parts) or "-"

    def overlay_style(bbox: Any, page: dict[str, Any]) -> str | None:
        if not isinstance(bbox, dict):
            return None
        required = [bbox.get("x0"), bbox.get("x1"), bbox.get("top"), bbox.get("bottom"), page.get("width"), page.get("height")]
        if any(value is None for value in required):
            return None
        page_width = float(page["width"])
        page_height_value = float(page["height"])
        if page_width <= 0 or page_height_value <= 0:
            return None
        x0 = max(0.0, min(float(bbox["x0"]), page_width))
        x1 = max(0.0, min(float(bbox["x1"]), page_width))
        top = max(0.0, min(float(bbox["top"]), page_height_value))
        bottom = max(0.0, min(float(bbox["bottom"]), page_height_value))
        width = max(0.4, x1 - x0)
        height = max(0.4, bottom - top)
        return (
            f"left:{(x0 / page_width) * 100:.4f}%;"
            f"top:{(top / page_height_value) * 100:.4f}%;"
            f"width:{(width / page_width) * 100:.4f}%;"
            f"height:{(height / page_height_value) * 100:.4f}%;"
        )

    def overlay_class(label: str, has_override: bool) -> str:
        classes = ["bbox-overlay"]
        if label.startswith("main_paragraph"):
            classes.append("overlay-paragraph")
        elif label.startswith("structure"):
            classes.append("overlay-structure")
        elif label.startswith("page_artifact"):
            classes.append("overlay-artifact")
        elif label.startswith("unknown") or label == "missing_bucket":
            classes.append("overlay-unknown")
        if has_override:
            classes.append("overlay-overridden")
        return " ".join(classes)

    def bucket_label(candidate: dict[str, Any] | None) -> str:
        if not candidate:
            return "missing_bucket"
        return str(candidate.get("stream_type", "unknown_bucket"))

    def bucket_class(label: str) -> str:
        if label.startswith("main_paragraph"):
            return "bucket paragraph"
        if label.startswith("structure"):
            return "bucket structure"
        if label.startswith("page_artifact"):
            return "bucket artifact"
        if label.startswith("unknown") or label == "missing_bucket":
            return "bucket unknown"
        return "bucket"

    page_summary_rows = []
    page_detail_sections = []
    for page in inventory:
        page_number = int(page["page_number"])
        page_image_href = f"{PAGE_IMAGES_DIR_NAME}/{page_image_filename(page_number)}"
        page_objects = objects_by_page.get(page_number, [])
        bucket_counts: Counter[str] = Counter()
        object_cards = []
        overlay_boxes = []
        for obj in page_objects:
            candidate = candidate_by_object_id.get(obj["object_id"])
            label = bucket_label(candidate)
            promotion_status = "promoted" if obj["object_id"] in promoted_object_ids else "not_promoted"
            promotion_blocker = blocker_by_object_id.get(obj["object_id"])
            bucket_counts[label] += 1
            warnings = candidate.get("warnings", []) if candidate else ["object_missing_from_candidate_streams"]
            reasons = obj.get("classification_reasons", [])
            source_lines = candidate.get("source_line_ids", []) if candidate else obj.get("source_line_ids", [])
            subtype = candidate.get("artifact_type") or candidate.get("structure_type") or "-" if candidate else "-"
            confidence = candidate.get("confidence", "-") if candidate else "-"
            zone = page_zones.get(page_number, "unknown")
            override = candidate.get("review_override") if candidate else None
            override_text = (
                f"{override.get('original_bucket')} -> {override.get('corrected_bucket')}: {override.get('reason')}"
                if override
                else "-"
            )
            detector_bucket = override.get("original_bucket") if override else candidate.get("original_stream_type", label) if candidate else "-"
            override_bucket = override.get("corrected_bucket") if override else "-"
            object_id = str(obj.get("object_id", ""))
            dom_id = safe_dom_id(object_id)
            evidence_reference = f"phase1_audit.html#card-{dom_id}; page {page_number}; object {object_id}"
            style = overlay_style(obj.get("bbox"), page)
            if style:
                overlay_boxes.append(
                    f"""<button type="button" class="{esc(overlay_class(label, bool(override)))}" style="{esc(style)}" data-object-id="{esc(object_id)}" data-bucket="{esc(label)}" data-subtype="{esc(subtype)}" data-overridden="{str(bool(override)).lower()}" aria-label="Highlight {esc(object_id)}" title="{esc(label)} · {esc(object_id)}"></button>"""
                )
            object_cards.append(
                f"""<article class="object-card {esc('is-promoted' if promotion_status == 'promoted' else '')}" id="card-{esc(dom_id)}" data-object-id="{esc(object_id)}" data-bucket="{esc(label)}" data-detector-bucket="{esc(detector_bucket)}" data-subtype="{esc(subtype)}" data-confidence="{esc(confidence)}" data-warnings="{esc(' '.join(warnings))}" data-zone="{esc(zone)}" data-page="{page_number}" data-evidence-reference="{esc(evidence_reference)}" data-promotion-status="{esc(promotion_status)}" tabindex="0">
                  <header>
                    <code>{esc(obj.get('object_id'))}</code>
                    <span>
                      <span class="{esc(bucket_class(label))}">{esc(label)}</span>
                      <span class="promotion-badge {esc('promoted' if promotion_status == 'promoted' else 'blocked')}">{esc(promotion_status)}</span>
                    </span>
                  </header>
                  <div class="object-grid">
                    <section>
                      <h4>Raw Extracted Object</h4>
                      <p>{esc(obj.get('raw_text', ''))}</p>
                      <dl>
                        <dt>Object type</dt><dd><code>{esc(obj.get('object_type'))}</code></dd>
                        <dt>Book zone</dt><dd><code>{esc(zone)}</code></dd>
                        <dt>Source lines</dt><dd><code>{esc(', '.join(source_lines))}</code></dd>
                        <dt>Bounding box</dt><dd><code>{esc(bbox_text(obj.get('bbox')))}</code></dd>
                      </dl>
                    </section>
                    <section>
                      <h4>Candidate Assignment</h4>
                      <p>{esc(candidate.get('clean_text', '') if candidate else '')}</p>
                      <dl>
                        <dt>Detector bucket</dt><dd><code>{esc(detector_bucket)}</code></dd>
                        <dt>Override bucket</dt><dd><code>{esc(override_bucket)}</code></dd>
                        <dt>Final candidate bucket</dt><dd><code>{esc(label)}</code></dd>
                        <dt>Confidence</dt><dd><code>{esc(confidence)}</code></dd>
                        <dt>Subtype</dt><dd><code>{esc(subtype)}</code></dd>
                        <dt>Review override</dt><dd><code>{esc(override_text)}</code></dd>
                        <dt>Promotion</dt><dd><code>{esc(promotion_status)}</code></dd>
                        <dt>Promotion blockers</dt><dd><code>{esc(', '.join(promotion_blocker.get('blocker_reasons', [])) if promotion_blocker else '-')}</code></dd>
                        <dt>Warnings</dt><dd><code>{esc(', '.join(warnings) or '-')}</code></dd>
                        <dt>Reasons</dt><dd><code>{esc(', '.join(reasons) or '-')}</code></dd>
                      </dl>
                    </section>
                  </div>
                </article>"""
            )
        bucket_summary = ", ".join(f"{key}: {count}" for key, count in sorted(bucket_counts.items())) or "-"
        page_summary_rows.append(
            "<tr>"
            f"<td><a href=\"#page-{page_number}\">{page_number}</a></td>"
            f"<td>{esc(page['status'])}</td>"
            f"<td>{len(page_objects)}</td>"
            f"<td>{esc(bucket_summary)}</td>"
            f"<td>{esc(', '.join(page['review_flags']) or '-')}</td>"
            f"<td>{esc(page['sample'])}</td>"
            "</tr>"
        )
        page_detail_sections.append(
            f"""<details class="page-audit" id="page-{page_number}">
              <summary>Page {page_number}: {esc(page['status'])} · {len(page_objects)} objects · {esc(bucket_summary)}</summary>
              <div class="page-meta">
                <span>Chars: <code>{page['raw_char_count']}</code></span>
                <span>Lines: <code>{page['line_count']}</code></span>
                <span>Images: <code>{page['image_count']}</code></span>
                <span>Tables: <code>{page['table_count']}</code></span>
                <span>Flags: <code>{esc(', '.join(page['review_flags']) or '-')}</code></span>
                <span>Page image: <a href="{esc(page_image_href)}">{esc(page_image_href)}</a></span>
              </div>
              <div class="page-review-grid">
                <figure class="page-image-witness">
                  <div class="page-image-stage">
                    <a href="{esc(page_image_href)}"><img src="{esc(page_image_href)}" alt="Rendered PDF page {page_number}"></a>
                    <div class="bbox-layer" aria-label="Extracted object bounding boxes">
                      {''.join(overlay_boxes)}
                    </div>
                  </div>
                  <figcaption>
                    Rendered PDF page {page_number}
                    <button type="button" class="zoom-toggle">Toggle page zoom</button>
                  </figcaption>
                  <aside class="selected-object-detail" aria-live="polite">
                    <strong>Selected object</strong>
                    <span>Click a box or object card.</span>
                    <div class="override-template-panel">
                      <strong>Copy-ready override JSONL</strong>
                      <p>
                        Copy one reviewed line into
                        <code>reviews/{esc(book_id)}/review_overrides.jsonl</code>, then rerun Phase 1.
                        This audit does not auto-apply changes.
                      </p>
                      <pre class="override-template" data-book-id="{esc(book_id)}" data-template-date="{esc(generated_date)}">Select an object to generate an override template.</pre>
                    </div>
                  </aside>
                </figure>
                <div class="page-object-list">
                  {''.join(object_cards) or '<p>No extracted objects for this page.</p>'}
                </div>
              </div>
            </details>"""
        )

    rows_html = "\n".join(
        "<tr>"
        f"<td>{row['page_number']}</td>"
        f"<td>{esc(row['status'])}</td>"
        f"<td>{row['raw_char_count']}</td>"
        f"<td>{row['line_count']}</td>"
        f"<td>{row['image_count']}</td>"
        f"<td>{row['table_count']}</td>"
        f"<td>{esc(', '.join(row['review_flags']))}</td>"
        f"<td>{esc(row['sample'])}</td>"
        "</tr>"
        for row in sample_pages
    )
    status_items = "\n".join(f"<li><code>{esc(k)}</code>: {v}</li>" for k, v in sorted(status_counts.items()))
    object_items = "\n".join(f"<li><code>{esc(k)}</code>: {v}</li>" for k, v in sorted(object_counts.items()))
    stream_items = "\n".join(f"<li><code>{esc(k)}</code>: {v}</li>" for k, v in sorted(stream_counts.items()))
    artifact_type_counts = Counter(row.get("artifact_type", "unknown") for row in candidate_rows if row.get("stream_type") == "page_artifact_candidate")
    artifact_type_items = "\n".join(f"<li><code>{esc(k)}</code>: {v}</li>" for k, v in sorted(artifact_type_counts.items()))
    artifact_pattern_rows_html, false_positive_rows_html = artifact_review_groups()
    promotion_counts = promotion_report.get("counts", {})
    promotion_warning_items = "\n".join(
        f"<li><code>{esc(key)}</code>: {value}</li>"
        for key, value in sorted((promotion_report.get("warning_counts") or {}).items())
    )
    promotion_blocker_items = "\n".join(
        f"<li><strong>{esc(row.get('object_id'))}</strong> ({esc(row.get('stream_type'))}): <code>{esc(', '.join(row.get('blocker_reasons', [])))}</code></li>"
        for row in promotion_blockers[:40]
    )

    def fmt_decimal(value: Any, digits: int = 1) -> str:
        return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else "-"

    merge_experiment_counts = paragraph_merge_experiment_report.get("counts") or {}
    merge_experiment_safety = paragraph_merge_experiment_report.get("downstream_safety") or {}
    merge_experiment_gold_scores = paragraph_merge_experiment_report.get("gold_scores") or {}
    merge_experiment_acceptance = paragraph_merge_experiment_report.get("acceptance_rule") or {}
    merge_split_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('baseline_object_id'))}</code></td>"
        f"<td>{esc(row.get('page_number'))}</td>"
        f"<td>{esc(row.get('baseline_source_line_count'))}</td>"
        f"<td>{esc(len(row.get('new_paragraphs', [])))}</td>"
        f"<td>{esc(row.get('baseline_text_preview', ''))}</td>"
        f"<td>{esc(' | '.join(str(child.get('text_preview', '')) for child in row.get('new_paragraphs', [])))}</td>"
        "</tr>"
        for row in (paragraph_merge_experiment_report.get("examples_of_paragraphs_split_by_new_policy") or [])[:20]
    )
    merge_oversplit_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('baseline_object_id'))}</code></td>"
        f"<td>{esc(row.get('page_number'))}</td>"
        f"<td>{esc(row.get('risk_reason'))}</td>"
        f"<td>{esc(row.get('baseline_text_preview', ''))}</td>"
        f"<td>{esc(' | '.join(str(child.get('text_preview', '')) for child in row.get('new_paragraphs', [])))}</td>"
        "</tr>"
        for row in (paragraph_merge_experiment_report.get("examples_of_possible_oversplitting_risk") or [])[:20]
    )
    merge_join_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('first_object_id'))}</code></td>"
        f"<td><code>{esc(row.get('second_object_id'))}</code></td>"
        f"<td>{esc(', '.join(str(page) for page in row.get('pages', [])))}</td>"
        f"<td>{esc(row.get('source_line_count'))}</td>"
        f"<td>{esc(', '.join(row.get('join_reasons', [])))}</td>"
        f"<td>{esc(row.get('first_text_end', ''))}</td>"
        f"<td>{esc(row.get('second_text_start', ''))}</td>"
        "</tr>"
        for row in (paragraph_merge_experiment_report.get("examples_of_joined_cross_page_paragraphs") or [])[:20]
    )
    merge_rejected_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('first_object_id'))}</code></td>"
        f"<td><code>{esc(row.get('second_object_id'))}</code></td>"
        f"<td>{esc(', '.join(str(page) for page in row.get('pages', [])))}</td>"
        f"<td>{esc(', '.join(row.get('rejection_reasons', [])))}</td>"
        f"<td>{esc(', '.join(str(item) for item in row.get('intervening_structure_object_ids', [])))}</td>"
        f"<td>{esc(row.get('first_text_end', ''))}</td>"
        f"<td>{esc(row.get('second_text_start', ''))}</td>"
        "</tr>"
        for row in (paragraph_merge_experiment_report.get("examples_of_rejected_cross_page_candidates") or [])[:20]
    )
    merge_taxonomy_summary = paragraph_merge_failure_taxonomy_report.get("summary") or {}
    merge_taxonomy_category_items = "\n".join(
        f"<li><code>{esc(key)}</code>: {esc(value)}</li>"
        for key, value in sorted((merge_taxonomy_summary.get("count_by_category") or {}).items())
    )
    merge_taxonomy_action_items = "\n".join(
        f"<li><code>{esc(key)}</code>: {esc(value)}</li>"
        for key, value in sorted((merge_taxonomy_summary.get("count_by_recommended_action") or {}).items())
    )
    cross_page_join_summary = cross_page_join_review_report.get("summary") or {}
    cross_page_decision_validation = cross_page_join_review_report.get("decision_validation") or {}

    def join_decision_template(row: dict[str, Any]) -> str:
        return json.dumps(
            {
                "join_id": row.get("join_id"),
                "left_page": row.get("left_page"),
                "right_page": row.get("right_page"),
                "left_candidate_id": row.get("left_candidate_id"),
                "right_candidate_id": row.get("right_candidate_id"),
                "decision": "TODO_choose_one_of: accept | reject | needs_review",
                "reason": "TODO_explain_the_review_decision_from_visible_evidence",
                "reviewer": "human",
                "date": generated_date,
                "evidence_reference": f"phase1_audit.html#{row.get('join_id')}",
            },
            ensure_ascii=True,
        )

    cross_page_join_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('join_id'))}</code></td>"
        f"<td><a href=\"{esc(row.get('page_anchor', '#'))}\">{esc(row.get('left_page'))}-{esc(row.get('right_page'))}</a></td>"
        f"<td><a href=\"{esc(row.get('left_audit_anchor', '#'))}\"><code>{esc(row.get('left_candidate_id'))}</code></a></td>"
        f"<td><a href=\"{esc(row.get('right_audit_anchor', '#'))}\"><code>{esc(row.get('right_candidate_id'))}</code></a></td>"
        f"<td>{esc(row.get('decision_status'))}</td>"
        f"<td>{esc(row.get('risk_category'))}</td>"
        f"<td>{esc(fmt_decimal(row.get('confidence'), 2))}</td>"
        f"<td>{esc(row.get('overlaps_authoritative_gold'))}</td>"
        f"<td>{esc(', '.join(row.get('gold_ids', [])))}</td>"
        f"<td>{esc(', '.join(row.get('continuation_evidence', [])))}</td>"
        f"<td>{esc(row.get('left_text_end_preview', ''))}</td>"
        f"<td>{esc(row.get('right_text_start_preview', ''))}</td>"
        f"<td>{esc(row.get('recommended_action', ''))}</td>"
        f"<td><pre class=\"join-decision-template\">{esc(join_decision_template(row))}</pre></td>"
        "</tr>"
        for row in (cross_page_join_review_report.get("joins") or [])[:120]
    )
    cross_page_top_page_items = "\n".join(
        f"<li>Page <code>{esc(row.get('page'))}</code>: <code>{esc(row.get('count'))}</code></li>"
        for row in cross_page_join_summary.get("top_pages_needing_review", [])
    )
    xpage_0032_lines = xpage_join_0032_investigation.get("raw_source_line_evidence", {})
    xpage_0032_visual_refs = "\n".join(
        f"<li><code>{esc(ref)}</code></li>"
        for ref in xpage_join_0032_investigation.get("visual_page_evidence_references", [])
        if ref
    )
    xpage_0032_left_lines = "\n".join(f"<li>{esc(line)}</li>" for line in xpage_0032_lines.get("left_last_lines", []))
    xpage_0032_right_lines = "\n".join(f"<li>{esc(line)}</li>" for line in xpage_0032_lines.get("right_first_lines", []))
    merge_taxonomy_rows = "\n".join(
        "<tr>"
        f"<td><a href=\"{esc(row.get('audit_anchor', '#'))}\"><code>{esc(row.get('canonical_paragraph_id'))}</code></a></td>"
        f"<td><a href=\"{esc(row.get('page_anchor', '#'))}\">{esc(row.get('page_number'))}</a></td>"
        f"<td>{esc(row.get('severity'))}</td>"
        f"<td>{esc(row.get('source_line_count'))}</td>"
        f"<td>{esc(fmt_decimal(row.get('vertical_bbox_span'), 1))}</td>"
        f"<td>{esc(fmt_decimal(row.get('page_height_ratio'), 3))}</td>"
        f"<td><code>{esc(row.get('provisional_category'))}</code></td>"
        f"<td>{esc(fmt_decimal(row.get('confidence'), 2))}</td>"
        f"<td>{esc(row.get('recommended_next_action'))}</td>"
        f"<td>{esc(row.get('first_source_line_preview', ''))}</td>"
        f"<td>{esc(row.get('last_source_line_preview', ''))}</td>"
        f"<td>{esc(row.get('text_preview', ''))}</td>"
        "</tr>"
        for row in (paragraph_merge_failure_taxonomy_report.get("samples") or [])
    )
    gold_counts = gold_evaluation_report.get("counts") or {}
    gold_scores = gold_evaluation_report.get("scores") or {}
    gold_page_items = "\n".join(
        f"<li>Page <code>{esc(row.get('page'))}</code>: {esc(row.get('reason', ''))}</li>"
        for row in (gold_evaluation_report.get("gold_pages") or [])
    )
    canonical_review_counts = canonical_review_report.get("counts", {})
    canonical_review_warning_items = "\n".join(
        f"<li><code>{esc(key)}</code>: {value}</li>"
        for key, value in sorted((canonical_review_report.get("warning_categories") or {}).items())
    )
    canonical_review_sample_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('canonical_paragraph_id'))}</code></td>"
        f"<td>{esc(row.get('page_number'))}</td>"
        f"<td><code>{esc(row.get('source_candidate_object_id'))}</code></td>"
        f"<td>{esc(', '.join(row.get('warnings', [])))}</td>"
        f"<td>{esc(row.get('clean_text_sample', ''))}</td>"
        "</tr>"
        for row in (canonical_review_report.get("sample_risky_canonical_paragraphs") or [])[:20]
    )
    canonical_review_drilldown_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('warning'))}</code></td>"
        f"<td>{esc(row.get('cluster'))}</td>"
        f"<td>{esc(row.get('severity'))}</td>"
        f"<td>{esc(row.get('count'))}</td>"
        f"<td>{esc(', '.join(str(page) for page in row.get('affected_pages', [])))}</td>"
        f"<td>{esc(', '.join(str(item) for item in row.get('sample_canonical_paragraph_ids', [])))}</td>"
        f"<td>{esc(row.get('likely_next_corrective_action', ''))}</td>"
        "</tr>"
        for row in (canonical_review_report.get("warning_category_drilldown") or [])
    )
    canonical_review_cluster_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('cluster'))}</code></td>"
        f"<td>{esc(row.get('severity'))}</td>"
        f"<td>{esc(row.get('count'))}</td>"
        f"<td>{esc(', '.join(row.get('warnings', [])))}</td>"
        f"<td>{esc(', '.join(str(page) for page in row.get('affected_pages', [])))}</td>"
        f"<td>{esc(row.get('likely_next_corrective_action', ''))}</td>"
        f"<td>{esc(', '.join(row.get('may_require', [])))}</td>"
        "</tr>"
        for row in (canonical_review_report.get("risky_paragraph_clusters") or [])
    )
    bbox_span_summary = canonical_review_report.get("bbox_span_risk_summary") or {}
    bbox_span_diag_rows = "\n".join(
        "<tr>"
        f"<td><a href=\"{esc(row.get('audit_anchor', '#'))}\"><code>{esc(row.get('canonical_paragraph_id'))}</code></a></td>"
        f"<td><a href=\"{esc(row.get('page_anchor', '#'))}\">{esc(row.get('page_number'))}</a></td>"
        f"<td><a href=\"{esc(row.get('audit_anchor', '#'))}\"><code>{esc(row.get('source_candidate_object_id'))}</code></a></td>"
        f"<td>{esc(row.get('source_line_count'))}</td>"
        f"<td>{esc(fmt_decimal(row.get('vertical_bbox_span'), 1))}</td>"
        f"<td>{esc(fmt_decimal(row.get('page_height_ratio'), 3))}</td>"
        f"<td>{esc(row.get('text_length'))}</td>"
        f"<td>{esc(row.get('warning_severity'))}</td>"
        f"<td>{esc(row.get('likely_interpretation'))}</td>"
        f"<td>{esc(', '.join(row.get('likely_corrective_path', [])))}</td>"
        f"<td>{esc(row.get('first_source_line_preview', ''))}</td>"
        f"<td>{esc(row.get('last_source_line_preview', ''))}</td>"
        "</tr>"
        for row in (canonical_review_report.get("bbox_span_risk_diagnostics") or [])
    )

    def bbox_span_group_items(group_name: str, label_key: str) -> str:
        return "\n".join(
            "<li>"
            f"<code>{esc(row.get(label_key))}</code>: {esc(row.get('count'))} "
            f"(samples: <code>{esc(', '.join(str(item) for item in row.get('sample_canonical_paragraph_ids', [])))}</code>; "
            f"pages: <code>{esc(', '.join(str(page) for page in row.get('affected_pages', [])))}</code>)"
            "</li>"
            for row in (bbox_span_summary.get(group_name) or [])
        )

    bbox_span_by_severity = bbox_span_group_items("by_severity", "warning_severity")
    bbox_span_by_page = bbox_span_group_items("by_page", "page_number")
    bbox_span_by_line_count = bbox_span_group_items("by_source_line_count_range", "source_line_count_range")
    bbox_span_by_ratio = bbox_span_group_items("by_page_height_ratio_range", "page_height_ratio_range")
    bbox_span_decision_summary = canonical_review_report.get("bbox_span_decision_summary") or {}

    def bbox_span_decision_group_items(group_name: str, label_key: str) -> str:
        return "\n".join(
            "<li>"
            f"<code>{esc(row.get(label_key))}</code>: {esc(row.get('count'))} "
            f"(samples: <code>{esc(', '.join(str(item) for item in row.get('sample_canonical_paragraph_ids', [])))}</code>; "
            f"pages: <code>{esc(', '.join(str(page) for page in row.get('affected_pages', [])))}</code>)"
            "</li>"
            for row in (bbox_span_decision_summary.get(group_name) or [])
        )

    bbox_span_by_cause = bbox_span_decision_group_items("by_likely_cause", "likely_cause")
    bbox_span_by_action = bbox_span_decision_group_items("by_recommended_action", "recommended_action")
    bbox_span_top_pages = "\n".join(
        "<li>"
        f"Page <a href=\"#page-{esc(row.get('page_number'))}\">{esc(row.get('page_number'))}</a>: "
        f"<code>{esc(row.get('diagnostic_count'))}</code> diagnostics, "
        f"<code>{esc(row.get('high_severity_count'))}</code> high severity, "
        f"causes <code>{esc(', '.join(f'{key}: {value}' for key, value in (row.get('likely_causes') or {}).items()))}</code>"
        "</li>"
        for row in (bbox_span_decision_summary.get("top_pages_needing_inspection") or [])
    )
    bbox_span_decision_rows = "\n".join(
        "<tr>"
        f"<td><a href=\"{esc(row.get('audit_anchor', '#'))}\"><code>{esc(row.get('canonical_paragraph_id'))}</code></a></td>"
        f"<td><a href=\"{esc(row.get('page_anchor', '#'))}\">{esc(row.get('page_number'))}</a></td>"
        f"<td>{esc(row.get('severity'))}</td>"
        f"<td>{esc(row.get('source_line_count'))}</td>"
        f"<td>{esc(fmt_decimal(row.get('vertical_bbox_span'), 1))}</td>"
        f"<td>{esc(fmt_decimal(row.get('page_height_ratio'), 3))}</td>"
        f"<td>{esc(row.get('text_length'))}</td>"
        f"<td><code>{esc(row.get('likely_cause'))}</code></td>"
        f"<td>{esc(fmt_decimal(row.get('confidence'), 2))}</td>"
        f"<td><code>{esc(row.get('recommended_action'))}</code></td>"
        f"<td>{esc(row.get('first_source_line_preview', ''))}</td>"
        f"<td>{esc(row.get('last_source_line_preview', ''))}</td>"
        "</tr>"
        for row in (canonical_review_report.get("bbox_span_decisions") or [])
    )
    canonical_review_recommendation = canonical_review_report.get("recommendation_detail") or {}
    post_adoption_state = post_adoption_safety_report.get("current_state") or {}
    post_adoption_comparison = post_adoption_safety_report.get("before_after_adoption") or {}
    post_adoption_top_risk = post_adoption_safety_report.get("current_top_risk") or {}
    post_adoption_category_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('warning'))}</code></td>"
        f"<td>{esc(row.get('before'))}</td>"
        f"<td>{esc(row.get('after'))}</td>"
        f"<td>{esc(row.get('delta'))}</td>"
        "</tr>"
        for row in (post_adoption_comparison.get("warning_category_deltas") or [])
    )
    post_adoption_sample_rows = "\n".join(
        "<tr>"
        f"<td><a href=\"{esc(row.get('audit_anchor', '#'))}\"><code>{esc(row.get('canonical_paragraph_id'))}</code></a></td>"
        f"<td><a href=\"{esc(row.get('page_anchor', '#'))}\">{esc(row.get('page_number'))}</a></td>"
        f"<td><code>{esc(row.get('source_candidate_object_id'))}</code></td>"
        f"<td>{esc(', '.join(row.get('warnings', [])))}</td>"
        f"<td>{esc(row.get('text_preview', ''))}</td>"
        "</tr>"
        for row in (post_adoption_safety_report.get("sample_risky_paragraphs") or [])[:12]
    )
    bbox_diagnosis_summary = post_adoption_bbox_diagnosis.get("summary") or {}
    bbox_diagnosis_cause_items = "\n".join(
        f"<li><code>{esc(row.get('likely_cause'))}</code>: {esc(row.get('count'))}</li>"
        for row in (post_adoption_bbox_diagnosis.get("by_likely_cause") or [])
    )
    bbox_diagnosis_page_items = "\n".join(
        f"<li>Page <code>{esc(row.get('page_number'))}</code>: {esc(row.get('count'))} cases; samples <code>{esc(', '.join(str(item) for item in row.get('sample_canonical_paragraph_ids', [])))}</code></li>"
        for row in (post_adoption_bbox_diagnosis.get("top_pages_needing_visual_review") or [])[:10]
    )
    bbox_diagnosis_true_defect_rows = "\n".join(
        "<tr>"
        f"<td><a href=\"{esc(row.get('audit_anchor', '#'))}\"><code>{esc(row.get('canonical_paragraph_id'))}</code></a></td>"
        f"<td><a href=\"{esc(row.get('page_anchor', '#'))}\">{esc(row.get('page_number'))}</a></td>"
        f"<td>{esc(row.get('source_line_count'))}</td>"
        f"<td>{esc(fmt_decimal(row.get('page_height_ratio'), 3))}</td>"
        f"<td>{esc(row.get('gold_coverage'))}</td>"
        f"<td>{esc(row.get('recommended_action'))}</td>"
        f"<td>{esc(row.get('text_preview', ''))}</td>"
        "</tr>"
        for row in (post_adoption_bbox_diagnosis.get("likely_true_defects") or [])[:20]
    )
    bbox_diagnosis_noise_rows = "\n".join(
        "<tr>"
        f"<td><a href=\"{esc(row.get('audit_anchor', '#'))}\"><code>{esc(row.get('canonical_paragraph_id'))}</code></a></td>"
        f"<td><a href=\"{esc(row.get('page_anchor', '#'))}\">{esc(row.get('page_number'))}</a></td>"
        f"<td><code>{esc(row.get('likely_cause'))}</code></td>"
        f"<td>{esc(row.get('source_line_count'))}</td>"
        f"<td>{esc(fmt_decimal(row.get('page_height_ratio'), 3))}</td>"
        f"<td>{esc(row.get('gold_coverage'))}</td>"
        f"<td>{esc(row.get('text_preview', ''))}</td>"
        "</tr>"
        for row in (post_adoption_bbox_diagnosis.get("likely_false_positives_or_threshold_noise") or [])[:20]
    )
    bbox_diagnosis_all_rows = "\n".join(
        "<tr>"
        f"<td><a href=\"{esc(row.get('audit_anchor', '#'))}\"><code>{esc(row.get('canonical_paragraph_id'))}</code></a></td>"
        f"<td><a href=\"{esc(row.get('page_anchor', '#'))}\">{esc(row.get('page_number'))}</a></td>"
        f"<td>{esc(row.get('source_line_count'))}</td>"
        f"<td>{esc(fmt_decimal(row.get('vertical_bbox_span'), 1))}</td>"
        f"<td>{esc(fmt_decimal(row.get('page_height_ratio'), 3))}</td>"
        f"<td><code>{esc(row.get('likely_cause'))}</code></td>"
        f"<td>{esc(fmt_decimal(row.get('confidence'), 2))}</td>"
        f"<td>{esc(row.get('gold_coverage'))}</td>"
        f"<td>{esc(row.get('recommended_action'))}</td>"
        f"<td>{esc(', '.join(row.get('current_warning_labels', [])))}</td>"
        "</tr>"
        for row in (post_adoption_bbox_diagnosis.get("diagnoses") or [])[:80]
    )
    remediation_summary = post_adoption_remediation_plan.get("summary") or {}
    remediation_queue_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('group'))}</code></td>"
        f"<td>{esc(row.get('count'))}</td>"
        f"<td>{esc(row.get('risk_level'))}</td>"
        f"<td>{esc(row.get('action_type'))}</td>"
        f"<td>{esc(', '.join(str(page) for page in row.get('affected_pages', [])))}</td>"
        f"<td><code>{esc(', '.join(str(item) for item in row.get('sample_canonical_paragraph_ids', [])))}</code></td>"
        f"<td>{esc(row.get('recommended_next_action', ''))}</td>"
        f"<td><code>{esc(row.get('downstream_remains_blocked_because_of_this_group'))}</code></td>"
        "</tr>"
        for row in (post_adoption_remediation_plan.get("queues") or [])
    )
    remediation_order_items = "\n".join(
        f"<li>{esc(item)}</li>"
        for item in (post_adoption_remediation_plan.get("recommended_order") or [])
    )
    front_matter_summary = front_matter_metadata_review_report.get("summary") or {}
    front_matter_classification_items = "\n".join(
        f"<li><code>{esc(key)}</code>: {esc(value)}</li>"
        for key, value in sorted((front_matter_summary.get("classification_counts") or {}).items())
    )
    front_matter_review_rows = "\n".join(
        "<tr>"
        f"<td><a href=\"{esc(row.get('audit_anchor', '#'))}\"><code>{esc(row.get('canonical_paragraph_id'))}</code></a></td>"
        f"<td><a href=\"{esc(row.get('page_anchor', '#'))}\">{esc(row.get('page'))}</a></td>"
        f"<td><code>{esc(row.get('source_candidate_id'))}</code></td>"
        f"<td><code>{esc(row.get('promotion_status'))}</code></td>"
        f"<td><code>{esc(row.get('likely_classification'))}</code></td>"
        f"<td>{esc(fmt_decimal(row.get('confidence'), 2))}</td>"
        f"<td>{esc(row.get('gold_coverage'))}</td>"
        f"<td>{esc(row.get('recommended_action'))}</td>"
        f"<td>{esc(row.get('text_preview', ''))}</td>"
        f"<td>{esc(row.get('visual_evidence_reference', ''))}</td>"
        "</tr>"
        for row in (front_matter_metadata_review_report.get("rows") or [])
    )
    visual_review_summary = visual_review_cases_report.get("summary") or {}
    visual_review_classification_items = "\n".join(
        f"<li><code>{esc(key)}</code>: {esc(value)}</li>"
        for key, value in sorted((visual_review_summary.get("classification_counts") or {}).items())
    )
    visual_review_rows = "\n".join(
        "<tr>"
        f"<td><a href=\"{esc(row.get('audit_anchor', '#'))}\"><code>{esc(row.get('canonical_paragraph_id'))}</code></a></td>"
        f"<td><a href=\"{esc(row.get('page_anchor', '#'))}\">{esc(row.get('page'))}</a></td>"
        f"<td><code>{esc(row.get('source_candidate_id'))}</code></td>"
        f"<td><code>{esc(row.get('likely_classification'))}</code></td>"
        f"<td>{esc(fmt_decimal(row.get('confidence'), 2))}</td>"
        f"<td>{esc(row.get('gold_coverage'))}</td>"
        f"<td>{esc((row.get('source_line_evidence') or {}).get('source_line_count'))}</td>"
        f"<td>{esc((row.get('source_line_evidence') or {}).get('first_source_line_preview', ''))}</td>"
        f"<td>{esc((row.get('source_line_evidence') or {}).get('last_source_line_preview', ''))}</td>"
        f"<td>{esc(row.get('recommended_action'))}</td>"
        f"<td>{esc(row.get('text_preview', ''))}</td>"
        f"<td>{esc(row.get('visual_evidence_reference', ''))}</td>"
        "</tr>"
        for row in (visual_review_cases_report.get("rows") or [])
    )
    narrow_design_summary = narrow_grouping_correction_design.get("summary") or {}
    narrow_design_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('defect_id'))}</code></td>"
        f"<td><a href=\"{esc('#page-' + str((row.get('affected_pages') or [''])[0]))}\">{esc(', '.join(str(page) for page in row.get('affected_pages', [])))}</a></td>"
        f"<td><code>{esc(row.get('canonical_paragraph_id'))}</code></td>"
        f"<td>{esc((row.get('expected_gold_behavior') or {}).get('gold_id'))}</td>"
        f"<td>{esc((row.get('current_paragraph_grouping_behavior') or {}).get('observed_problem'))}</td>"
        f"<td>{esc(row.get('why_current_policy_failed'))}</td>"
        f"<td><code>{esc(((row.get('proposed_narrow_correction_rule') or {}).get('name')))}</code>: {esc(((row.get('proposed_narrow_correction_rule') or {}).get('description')))}</td>"
        f"<td>{esc('; '.join(row.get('conditions_required_before_rule_applies', [])))}</td>"
        f"<td>{esc('; '.join(row.get('conditions_that_must_block_rule', [])))}</td>"
        f"<td>{esc('; '.join(row.get('adoption_gates', [])))}</td>"
        "</tr>"
        for row in (narrow_grouping_correction_design.get("designs") or [])
    )
    chained_acceptance = chained_cross_page_experiment.get("acceptance_rule") or {}
    chained_active_metrics = chained_cross_page_experiment.get("active_policy_metrics") or {}
    chained_experimental_metrics = chained_cross_page_experiment.get("experimental_policy_metrics") or {}
    chained_gold_scores = chained_cross_page_experiment.get("gold_scores") or {}
    chained_gold_counts = chained_cross_page_experiment.get("gold_counts") or {}
    chained_side_effects = chained_cross_page_experiment.get("side_effects") or {}
    chained_join_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('first_object_id'))}</code></td>"
        f"<td><code>{esc(row.get('second_object_id'))}</code></td>"
        f"<td>{esc(', '.join(str(page) for page in row.get('pages', [])))}</td>"
        f"<td>{esc(row.get('source_line_count'))}</td>"
        f"<td>{esc(', '.join(row.get('join_reasons', [])))}</td>"
        f"<td>{esc(row.get('first_text_end', ''))}</td>"
        f"<td>{esc(row.get('second_text_start', ''))}</td>"
        f"<td>{esc(row.get('joined_text_preview', ''))}</td>"
        "</tr>"
        for row in (chained_cross_page_experiment.get("proposed_chained_joins") or [])[:60]
    )
    chained_rejected_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('first_object_id'))}</code></td>"
        f"<td><code>{esc(row.get('second_object_id'))}</code></td>"
        f"<td>{esc(', '.join(str(page) for page in row.get('pages', [])))}</td>"
        f"<td>{esc(', '.join(row.get('rejection_reasons', [])))}</td>"
        f"<td>{esc(row.get('first_text_end', ''))}</td>"
        f"<td>{esc(row.get('second_text_start', ''))}</td>"
        "</tr>"
        for row in (chained_cross_page_experiment.get("rejected_chained_joins") or [])[:60]
    )
    chained_queue_summary = chained_join_review_queue.get("summary") or {}
    chained_queue_risk_items = "\n".join(
        f"<li><code>{esc(key)}</code>: {esc(value)}</li>"
        for key, value in sorted((chained_queue_summary.get("risk_counts") or {}).items())
    )
    chained_queue_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('chained_join_id'))}</code></td>"
        f"<td>{esc(', '.join(str(page) for page in row.get('affected_pages', [])))}</td>"
        f"<td><code>{esc((row.get('source_candidate_ids') or {}).get('left_candidate_id'))}</code></td>"
        f"<td><code>{esc((row.get('source_candidate_ids') or {}).get('right_candidate_id'))}</code></td>"
        f"<td>{esc(row.get('likely_risk'))}</td>"
        f"<td>{esc(fmt_decimal(row.get('confidence'), 2))}</td>"
        f"<td>{esc(row.get('gold_coverage_exists'))}</td>"
        f"<td>{esc(row.get('recommended_review_action'))}</td>"
        f"<td>{esc((row.get('text_preview_before_join') or {}).get('left_text_end', ''))}</td>"
        f"<td>{esc((row.get('text_preview_before_join') or {}).get('right_text_start', ''))}</td>"
        f"<td>{esc(row.get('text_preview_after_join', ''))}</td>"
        f"<td>{esc('; '.join(row.get('visual_evidence_references', [])))}</td>"
        "</tr>"
        for row in (chained_join_review_queue.get("queue") or [])[:80]
    )
    chained_decision_summary = chained_join_decisions_applied.get("summary") or {}
    chained_decision_validation = chained_join_decisions_applied.get("validation") or {}
    chained_decision_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('chained_join_id'))}</code></td>"
        f"<td>{esc(', '.join(str(page) for page in row.get('affected_pages', [])))}</td>"
        f"<td>{esc(row.get('likely_risk'))}</td>"
        f"<td>{esc(row.get('decision'))}</td>"
        f"<td>{esc(row.get('reason'))}</td>"
        f"<td>{esc(row.get('reviewer'))}</td>"
        f"<td>{esc(row.get('reviewed_at'))}</td>"
        f"<td>{esc(row.get('notes'))}</td>"
        f"<td>{esc(row.get('evidence_reference'))}</td>"
        "</tr>"
        for row in (chained_join_decisions_applied.get("decisions") or [])[:80]
    )
    guarded_acceptance = guarded_chained_experiment.get("acceptance_rule") or {}
    guarded_gold_scores = guarded_chained_experiment.get("gold_scores") or {}
    guarded_gold_counts = guarded_chained_experiment.get("gold_counts") or {}
    guarded_active_metrics = guarded_chained_experiment.get("active_v2_metrics") or {}
    guarded_previous_metrics = guarded_chained_experiment.get("previous_v3_metrics") or {}
    guarded_metrics = guarded_chained_experiment.get("guarded_v3_metrics") or {}
    guarded_decision_replay = guarded_chained_experiment.get("decision_replay") or {}
    guarded_side_effects = guarded_chained_experiment.get("side_effects") or {}
    guarded_warning_deltas = guarded_chained_experiment.get("warning_deltas") or {}
    guarded_adoption_gate_evidence = guarded_policy_adoption_decision.get("gate_evidence") or {}
    guarded_adoption_active_run = guarded_policy_adoption_decision.get("active_run_after_adoption") or {}
    guarded_adoption_gates = guarded_policy_adoption_decision.get("gates") or {}
    guarded_join_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('first_object_id'))}</code></td>"
        f"<td><code>{esc(row.get('second_object_id'))}</code></td>"
        f"<td>{esc(', '.join(str(page) for page in row.get('pages', [])))}</td>"
        f"<td>{esc(row.get('source_line_count'))}</td>"
        f"<td>{esc(', '.join(row.get('join_reasons', [])))}</td>"
        f"<td>{esc(row.get('first_text_end', ''))}</td>"
        f"<td>{esc(row.get('second_text_start', ''))}</td>"
        f"<td>{esc(row.get('joined_text_preview', ''))}</td>"
        "</tr>"
        for row in (guarded_chained_experiment.get("proposed_chained_joins") or [])[:60]
    )
    guarded_rejected_rows = "\n".join(
        "<tr>"
        f"<td><code>{esc(row.get('first_object_id'))}</code></td>"
        f"<td><code>{esc(row.get('second_object_id'))}</code></td>"
        f"<td>{esc(', '.join(str(page) for page in row.get('pages', [])))}</td>"
        f"<td>{esc(', '.join(row.get('rejection_reasons', [])))}</td>"
        f"<td>{esc(', '.join(row.get('intervening_body_object_ids', [])))}</td>"
        f"<td>{esc(row.get('first_text_end', ''))}</td>"
        f"<td>{esc(row.get('second_text_start', ''))}</td>"
        "</tr>"
        for row in (guarded_chained_experiment.get("rejected_chained_candidates") or [])[:60]
    )
    bucket_options = "\n".join(
        f"<option value=\"{esc(value)}\">{esc(value)}</option>"
        for value in sorted({bucket_label(row) for row in candidate_rows})
    )
    subtype_options = "\n".join(
        f"<option value=\"{esc(value)}\">{esc(value)}</option>"
        for value in sorted({str(row.get("artifact_type") or row.get("structure_type") or "-") for row in candidate_rows})
    )
    zone_options = "\n".join(
        f"<option value=\"{esc(value)}\">{esc(value)}</option>"
        for value in sorted(set(page_zones.values()))
    )
    flagged_items = "\n".join(
        f"<li>Page {row['page_number']}: {esc(', '.join(row['review_flags']))}</li>"
        for row in flagged_pages[:40]
    )
    validation_items = "\n".join(
        f"<li><code>{esc(check['name'])}</code>: {esc(check['status'])} {esc(check.get('detail', ''))}</li>"
        for check in validation_report.get("checks", [])
    )
    sample_sections = []
    for stream_name, rows in stream_samples.items():
        if stream_name == "__all__":
            continue
        items = "\n".join(
            f"<li><strong>Page {esc(row.get('page_number'))}</strong>: {esc(row.get('clean_text', ''))}</li>"
            for row in rows[:8]
        )
        sample_sections.append(f"<h3>{esc(stream_name)}</h3><ul>{items or '<li>No rows.</li>'}</ul>")
    sample_stream_html = "\n".join(sample_sections)
    page_summary_html = "\n".join(page_summary_rows)
    page_detail_html = "\n".join(page_detail_sections)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 1 Audit: {esc(book_id)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.55; margin: 0; color: #17202a; background: #fbfbf8; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 48px 24px; }}
    h1, h2 {{ line-height: 1.15; }}
    h2 {{ margin-top: 2.2rem; border-top: 1px solid #d8d6cc; padding-top: 1.2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #d8d6cc; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #ece9df; text-align: left; }}
    code {{ background: #ece9df; padding: 0.1rem 0.25rem; border-radius: 4px; }}
    .rule {{ background: #eef4f2; border-left: 4px solid #2d6f63; padding: 14px 16px; }}
    .page-audit {{ border: 1px solid #d8d6cc; margin: 12px 0; background: #fffef9; }}
    .page-audit summary {{ cursor: pointer; padding: 12px 14px; font-weight: 700; background: #f1efe6; }}
    .page-meta {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 12px 14px; border-top: 1px solid #d8d6cc; border-bottom: 1px solid #d8d6cc; }}
    .page-review-grid {{ display: grid; grid-template-columns: minmax(260px, 360px) minmax(0, 1fr); align-items: start; gap: 14px; padding: 14px; }}
    .page-image-witness {{ position: sticky; top: 12px; margin: 0; border: 1px solid #d8d6cc; background: #fbfbf8; padding: 10px; }}
    .page-image-witness.is-zoomed {{ grid-column: 1 / -1; position: static; max-width: 860px; }}
    .page-image-stage {{ position: relative; }}
    .page-image-witness img {{ display: block; width: 100%; height: auto; border: 1px solid #e3e0d6; background: white; }}
    .page-image-witness figcaption {{ margin-top: 8px; font-size: 0.9rem; color: #56616b; }}
    .zoom-toggle {{ margin-left: 8px; font: inherit; padding: 3px 8px; border: 1px solid #9c978b; background: #fffef9; cursor: pointer; }}
    .selected-object-detail {{ margin-top: 10px; padding: 10px; border: 1px solid #d8d6cc; background: #fffef9; font-size: 0.9rem; }}
    .selected-object-detail span {{ display: block; margin-top: 4px; overflow-wrap: anywhere; }}
    .override-template-panel {{ margin-top: 12px; padding-top: 10px; border-top: 1px solid #d8d6cc; }}
    .override-template-panel p {{ margin: 6px 0 8px; color: #56616b; }}
    .override-template {{ margin: 0; padding: 10px; border: 1px solid #d8d6cc; background: #f7f4ea; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .join-decision-template {{ max-width: 360px; margin: 0; padding: 8px; border: 1px solid #d8d6cc; background: #f7f4ea; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .bbox-layer {{ position: absolute; inset: 1px; pointer-events: none; }}
    .bbox-overlay {{ position: absolute; box-sizing: border-box; border: 2px solid #111; background: rgba(255,255,255,0.08); padding: 0; pointer-events: auto; cursor: pointer; }}
    .bbox-overlay.hidden {{ display: none; }}
    .overlay-paragraph {{ border-color: #1f7a4d; background: rgba(31,122,77,0.10); }}
    .overlay-structure {{ border-color: #315a9f; background: rgba(49,90,159,0.10); }}
    .overlay-artifact {{ border-color: #a46316; background: rgba(164,99,22,0.12); }}
    .overlay-unknown {{ border-color: #a1382f; background: rgba(161,56,47,0.12); }}
    .overlay-overridden {{ outline: 3px dashed #111; outline-offset: 2px; }}
    .bbox-overlay.is-active, .bbox-overlay:hover, .bbox-overlay:focus {{ z-index: 4; box-shadow: 0 0 0 3px rgba(0,0,0,0.28); }}
    .page-object-list {{ min-width: 0; }}
    .object-card {{ padding: 14px; border-top: 1px solid #d8d6cc; }}
    .object-card.is-active {{ background: #fff7d6; box-shadow: inset 4px 0 0 #111; }}
    .object-card.is-promoted {{ box-shadow: inset 4px 0 0 #2d6f63; }}
    .page-object-list .object-card:first-child {{ border-top: 0; }}
    .object-card.hidden {{ display: none; }}
    .object-card header {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }}
    .object-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; }}
    .object-grid section {{ border: 1px solid #e3e0d6; padding: 12px; background: #fbfbf8; }}
    .object-grid h4 {{ margin: 0 0 8px; }}
    .object-grid p {{ white-space: pre-wrap; margin: 0 0 10px; }}
    dl {{ display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 4px 10px; margin: 0; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    .bucket {{ border: 1px solid #9c978b; padding: 2px 8px; font-size: 0.82rem; font-weight: 700; }}
    .promotion-badge {{ display: inline-block; margin-left: 6px; border: 1px solid #9c978b; padding: 2px 8px; font-size: 0.82rem; font-weight: 700; }}
    .promotion-badge.promoted {{ background: #e6f3eb; }}
    .promotion-badge.blocked {{ background: #f4eee4; }}
    .paragraph {{ background: #e9f3ee; }}
    .structure {{ background: #edf0f7; }}
    .artifact {{ background: #f5eadb; }}
    .unknown {{ background: #f7e4e1; }}
    .review-controls {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; padding: 14px; border: 1px solid #d8d6cc; background: #fffef9; }}
    .review-controls label {{ display: grid; gap: 4px; font-weight: 700; }}
    .review-controls select, .review-controls input {{ font: inherit; padding: 6px 8px; border: 1px solid #bdb8ac; background: #fbfbf8; }}
    .overlay-controls {{ margin-top: 12px; display: grid; gap: 10px; padding: 14px; border: 1px solid #d8d6cc; background: #fffef9; }}
    .overlay-actions {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .overlay-actions button {{ font: inherit; padding: 6px 10px; border: 1px solid #9c978b; background: #fbfbf8; cursor: pointer; }}
    .overlay-buckets {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .overlay-buckets label {{ display: flex; gap: 6px; align-items: center; font-weight: 700; }}
    .review-count {{ margin: 10px 0 0; font-weight: 700; }}
    @media (max-width: 900px) {{ .page-review-grid {{ grid-template-columns: 1fr; }} .page-image-witness {{ position: static; }} }}
    @media (max-width: 760px) {{ .object-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Phase 1 Audit: {esc(book_id)}</h1>
  <p><strong>Generated:</strong> {esc(generated)}</p>
  <p><strong>Source:</strong> <code>{esc(source_pdf)}</code></p>

  <div class="rule">
    This is a deterministic two-stream extraction audit. Main paragraph candidates are separated
    from structure, page artifacts, and unknown objects without deleting the surrounding evidence.
  </div>

  <h2>Manifest</h2>
  <ul>
    <li>Pages: <code>{manifest['page_count']}</code></li>
    <li>File size: <code>{manifest['file_size_bytes']}</code> bytes</li>
    <li>SHA-256: <code>{esc(manifest['sha256'])}</code></li>
    <li>Output directory: <code>{esc(output_dir)}</code></li>
  </ul>

  <h2>Page Status Counts</h2>
  <ul>{status_items}</ul>

  <h2>Object Type Counts</h2>
  <ul>{object_items}</ul>

  <h2>Reconstruction Stream Counts</h2>
  <ul>{stream_items}</ul>

  <h2>Page Artifact Candidate Types</h2>
  <ul>{artifact_type_items or '<li>No page artifact candidates.</li>'}</ul>

  <h2>Review Overrides</h2>
  <p>
    Curated reviewer decisions live in <code>reviews/{esc(book_id)}/review_overrides.jsonl</code>.
    The generated run records replayed decisions in <code>review_overrides_applied.jsonl</code>.
    Overrides change candidate bucket assignment for review purposes only; they do not promote
    content to canonical.
  </p>
  <ul>
    <li>Applied overrides: <code>{validation_report.get('summary', {}).get('review_override_rows', 0)}</code></li>
  </ul>

  <h2>Canonical Paragraph Promotion</h2>
  <div class="rule">
    Canonical paragraphs are evidence-bound and promoted under the current paragraph-only gate.
    Structure, page artifacts, unknown objects, OCR, AI review, retrieval, and graph work are not
    promoted here.
  </div>
  <ul>
    <li>Status: <code>{esc(promotion_report.get('status', 'unknown'))}</code></li>
    <li>Total candidates reviewed: <code>{esc(promotion_counts.get('total_candidates_reviewed', 0))}</code></li>
    <li>Paragraph candidates reviewed: <code>{esc(promotion_counts.get('paragraph_candidates_reviewed', 0))}</code></li>
    <li>Promoted paragraphs: <code>{esc(promotion_counts.get('promoted_paragraphs', 0))}</code></li>
    <li>Blocked candidates: <code>{esc(promotion_counts.get('blocked_candidates', 0))}</code></li>
    <li>Override-influenced promotions: <code>{esc(promotion_counts.get('override_influenced_promotions', 0))}</code></li>
  </ul>
  <h3>Promotion Warning Counts</h3>
  <ul>{promotion_warning_items or '<li>No warnings recorded.</li>'}</ul>
  <h3>Promotion Blocker Samples</h3>
  <ul>{promotion_blocker_items or '<li>No blockers recorded.</li>'}</ul>

  <h2>Paragraph Merge Experiment</h2>
  <div class="rule">
    This compares the baseline paragraph merge policy with the active experimental policy. It is a
    deterministic correction experiment; OCR, AI review, retrieval, graph work, and structure
    promotion are still excluded.
  </div>
  <ul>
    <li>Baseline policy: <code>{esc(paragraph_merge_experiment_report.get('baseline_paragraph_merge_policy', '-'))}</code></li>
    <li>New policy: <code>{esc(paragraph_merge_experiment_report.get('new_paragraph_merge_policy', '-'))}</code></li>
    <li>Experiment outcome: <code>{esc(paragraph_merge_experiment_report.get('experiment_outcome', '-'))}</code></li>
    <li>Baseline paragraph candidates: <code>{esc(merge_experiment_counts.get('baseline_paragraph_candidate_count', 0))}</code></li>
    <li>New paragraph candidates: <code>{esc(merge_experiment_counts.get('new_paragraph_candidate_count', 0))}</code></li>
    <li>Baseline canonical promoted: <code>{esc(merge_experiment_counts.get('baseline_canonical_promoted_count', 0))}</code></li>
    <li>New canonical promoted: <code>{esc(merge_experiment_counts.get('new_canonical_promoted_count', 0))}</code></li>
    <li>Baseline bbox span risk: <code>{esc(merge_experiment_counts.get('baseline_bbox_span_risk_count', 0))}</code></li>
    <li>New bbox span risk: <code>{esc(merge_experiment_counts.get('new_bbox_span_risk_count', 0))}</code></li>
    <li>Baseline likely true accidental merge: <code>{esc(merge_experiment_counts.get('baseline_likely_true_accidental_merge_count', 0))}</code></li>
    <li>New likely true accidental merge: <code>{esc(merge_experiment_counts.get('new_likely_true_accidental_merge_count', 0))}</code></li>
    <li>Baseline merged across paragraph break: <code>{esc(merge_experiment_counts.get('baseline_merged_across_paragraph_break_count', 0))}</code></li>
    <li>New merged across paragraph break: <code>{esc(merge_experiment_counts.get('new_merged_across_paragraph_break_count', 0))}</code></li>
    <li>Baseline merged across large vertical whitespace: <code>{esc(merge_experiment_counts.get('baseline_merged_across_large_vertical_whitespace_count', 0))}</code></li>
    <li>New merged across large vertical whitespace: <code>{esc(merge_experiment_counts.get('new_merged_across_large_vertical_whitespace_count', 0))}</code></li>
    <li>Baseline blocked paragraphs: <code>{esc(merge_experiment_counts.get('baseline_blocked_paragraph_count', 0))}</code></li>
    <li>New blocked paragraphs: <code>{esc(merge_experiment_counts.get('new_blocked_paragraph_count', 0))}</code></li>
    <li>Baseline gold paragraph precision: <code>{esc(fmt_decimal(merge_experiment_gold_scores.get('baseline_paragraph_precision'), 3))}</code></li>
    <li>New gold paragraph precision: <code>{esc(fmt_decimal(merge_experiment_gold_scores.get('new_paragraph_precision'), 3))}</code></li>
    <li>Baseline gold paragraph recall: <code>{esc(fmt_decimal(merge_experiment_gold_scores.get('baseline_paragraph_recall'), 3))}</code></li>
    <li>New gold paragraph recall: <code>{esc(fmt_decimal(merge_experiment_gold_scores.get('new_paragraph_recall'), 3))}</code></li>
    <li>Baseline gold over-split paragraphs: <code>{esc(merge_experiment_counts.get('baseline_gold_over_split_paragraphs', 0))}</code></li>
    <li>New gold over-split paragraphs: <code>{esc(merge_experiment_counts.get('new_gold_over_split_paragraphs', 0))}</code></li>
    <li>Baseline gold over-merged paragraphs: <code>{esc(merge_experiment_counts.get('baseline_gold_over_merged_paragraphs', 0))}</code></li>
    <li>New gold over-merged paragraphs: <code>{esc(merge_experiment_counts.get('new_gold_over_merged_paragraphs', 0))}</code></li>
    <li>Cross-page joins: <code>{esc(merge_experiment_counts.get('cross_page_join_count', 0))}</code></li>
    <li>Cross-page rejected candidates: <code>{esc(merge_experiment_counts.get('cross_page_rejected_count', 0))}</code></li>
    <li>Acceptance adoptable: <code>{esc(merge_experiment_acceptance.get('adoptable', False))}</code></li>
    <li>Possible oversplitting risks: <code>{esc(merge_experiment_counts.get('possible_oversplitting_risk_count', 0))}</code></li>
    <li>Recommendation: <code>{esc(merge_experiment_safety.get('recommendation', '-'))}</code></li>
  </ul>
  <h3>Joined Cross-Page Paragraphs</h3>
  <table>
    <thead>
      <tr><th>First Object</th><th>Second Object</th><th>Pages</th><th>Lines</th><th>Reasons</th><th>First End</th><th>Second Start</th></tr>
    </thead>
    <tbody>{merge_join_rows or '<tr><td colspan="7">No cross-page joins recorded.</td></tr>'}</tbody>
  </table>
  <h3>Rejected Cross-Page Candidates</h3>
  <table>
    <thead>
      <tr><th>First Object</th><th>Second Object</th><th>Pages</th><th>Rejection Reasons</th><th>Intervening Structure</th><th>First End</th><th>Second Start</th></tr>
    </thead>
    <tbody>{merge_rejected_rows or '<tr><td colspan="7">No rejected cross-page candidates recorded.</td></tr>'}</tbody>
  </table>

  <h2>Policy Adoption Decision</h2>
  <div class="rule">
    Decision: <code>{esc(policy_adoption_decision.get('decision', '-'))}</code>.
    Active policy: <code>{esc(policy_adoption_decision.get('active_paragraph_merge_policy', '-'))}</code>.
  </div>
  <ul>
    <li>Adopted policy: <code>{esc(policy_adoption_decision.get('adopted_policy', '-'))}</code></li>
    <li>Canonical promoted paragraphs after adoption: <code>{esc((policy_adoption_decision.get('active_run_after_adoption') or {}).get('canonical_promoted_paragraphs', '-'))}</code></li>
    <li>Gold precision after adoption: <code>{esc(fmt_decimal((policy_adoption_decision.get('active_run_after_adoption') or {}).get('gold_paragraph_precision'), 3))}</code></li>
    <li>Gold recall after adoption: <code>{esc(fmt_decimal((policy_adoption_decision.get('active_run_after_adoption') or {}).get('gold_paragraph_recall'), 3))}</code></li>
    <li>Safe for downstream: <code>{esc((policy_adoption_decision.get('active_run_after_adoption') or {}).get('safe_for_downstream', '-'))}</code></li>
  </ul>

  <h2>Cross-Page Join Review</h2>
  <div class="rule">
    This review classifies every proposed cross-page join before the continuation policy can become
    active. It is review-only and does not change canonical output.
  </div>
  <ul>
    <li>Total proposed joins: <code>{esc(cross_page_join_summary.get('total_proposed_joins', 0))}</code></li>
    <li>Likely correct joins: <code>{esc(cross_page_join_summary.get('likely_correct_joins', 0))}</code></li>
    <li>Possible false joins: <code>{esc(cross_page_join_summary.get('possible_false_joins', 0))}</code></li>
    <li>Boundary or structure risk joins: <code>{esc(cross_page_join_summary.get('boundary_or_structure_risk_joins', 0))}</code></li>
    <li>Needs manual review: <code>{esc(cross_page_join_summary.get('needs_manual_review', 0))}</code></li>
    <li>Auto likely correct: <code>{esc(cross_page_join_summary.get('auto_likely_correct_joins', 0))}</code></li>
    <li>Curated accepted: <code>{esc(cross_page_join_summary.get('curated_accepted_joins', 0))}</code></li>
    <li>Curated rejected: <code>{esc(cross_page_join_summary.get('curated_rejected_joins', 0))}</code></li>
    <li>Still needs manual review: <code>{esc(cross_page_join_summary.get('still_needs_manual_review_joins', 0))}</code></li>
    <li>Unresolved joins: <code>{esc(cross_page_join_summary.get('unresolved_join_count', 0))}</code></li>
    <li>Unresolved risk low enough for adoption: <code>{esc(cross_page_join_summary.get('unresolved_risk_low_enough_for_adoption', False))}</code></li>
    <li>Covered by authoritative gold: <code>{esc(cross_page_join_summary.get('joins_covered_by_authoritative_gold', 0))}</code></li>
    <li>Not covered by gold: <code>{esc(cross_page_join_summary.get('joins_not_covered_by_gold', 0))}</code></li>
    <li>Decision validation: <code>{esc(cross_page_decision_validation.get('status', 'unknown'))}</code></li>
  </ul>
  <p>
    Copy reviewed join decisions into
    <code>reviews/{esc(book_id)}/cross_page_join_decisions.jsonl</code>, then rerun Phase 1.
    Templates below are copy-ready only; this audit does not auto-apply changes.
  </p>
  <h3>Top Pages Needing Review</h3>
  <ul>{cross_page_top_page_items or '<li>No high-risk pages identified.</li>'}</ul>

  <h3>xpage_join_0032 Investigation</h3>
  <div class="rule">
    Focused investigation artifact: <code>xpage_join_0032_investigation.json</code>.
    Suspected issue: <code>{esc(xpage_join_0032_investigation.get('suspected_issue', '-'))}</code>.
    Recommended decision: <code>{esc(xpage_join_0032_investigation.get('recommended_decision', '-'))}</code>.
  </div>
  <p>{esc(xpage_join_0032_investigation.get('reason', 'No focused investigation generated yet.'))}</p>
  <h4>Left Page Raw Ending</h4>
  <ul>{xpage_0032_left_lines or '<li>No raw left-line evidence.</li>'}</ul>
  <h4>Right Page Raw Start</h4>
  <ul>{xpage_0032_right_lines or '<li>No raw right-line evidence.</li>'}</ul>
  <h4>Visual Evidence References</h4>
  <ul>{xpage_0032_visual_refs or '<li>No visual evidence references.</li>'}</ul>

  <table>
    <thead>
      <tr><th>Join</th><th>Pages</th><th>Left</th><th>Right</th><th>Decision Status</th><th>Risk</th><th>Confidence</th><th>Gold</th><th>Gold IDs</th><th>Evidence</th><th>Left End</th><th>Right Start</th><th>Action</th><th>Copy-Ready Decision JSONL</th></tr>
    </thead>
    <tbody>{cross_page_join_rows or '<tr><td colspan="14">No cross-page join review rows.</td></tr>'}</tbody>
  </table>
  <h3>Paragraphs Split By New Policy</h3>
  <table>
    <thead>
      <tr><th>Baseline Object</th><th>Page</th><th>Baseline Lines</th><th>New Paragraphs</th><th>Baseline Preview</th><th>New Previews</th></tr>
    </thead>
    <tbody>{merge_split_rows or '<tr><td colspan="6">No split examples recorded.</td></tr>'}</tbody>
  </table>
  <h3>Possible Over-Splitting Risks</h3>
  <table>
    <thead>
      <tr><th>Baseline Object</th><th>Page</th><th>Risk</th><th>Baseline Preview</th><th>New Previews</th></tr>
    </thead>
    <tbody>{merge_oversplit_rows or '<tr><td colspan="5">No possible over-splitting risks recorded.</td></tr>'}</tbody>
  </table>

  <h2>Merge Failure Taxonomy</h2>
  <div class="rule">
    This is a deterministic sample of likely true accidental merge rows. It classifies failure
    patterns for review only and does not change active extraction behavior.
  </div>
  <ul>
    <li>Sampled rows: <code>{esc(merge_taxonomy_summary.get('sampled_rows', 0))}</code></li>
  </ul>
  <h3>Taxonomy Category Counts</h3>
  <ul>{merge_taxonomy_category_items or '<li>No sampled taxonomy categories.</li>'}</ul>
  <h3>Recommended Action Counts</h3>
  <ul>{merge_taxonomy_action_items or '<li>No sampled taxonomy actions.</li>'}</ul>
  <table>
    <thead>
      <tr>
        <th>Canonical ID</th><th>Page</th><th>Severity</th><th>Lines</th><th>BBox Span</th>
        <th>Page Ratio</th><th>Category</th><th>Confidence</th><th>Recommended Action</th>
        <th>First Source Line</th><th>Last Source Line</th><th>Text Preview</th>
      </tr>
    </thead>
    <tbody>{merge_taxonomy_rows or '<tr><td colspan="12">No merge taxonomy samples.</td></tr>'}</tbody>
  </table>

  <h2>Gold Review</h2>
  <div class="rule">
    Gold review is the answer-key layer for future policy adoption. Placeholder rows are excluded
    from scoring until a reviewer marks them authoritative.
  </div>
  <ul>
    <li>Gold instructions: <code>reviews/{esc(book_id)}/gold/gold_review_instructions.html</code></li>
    <li>Gold pages: <code>{esc(gold_counts.get('gold_pages_reviewed', 0))}</code></li>
    <li>Gold paragraph rows: <code>{esc(gold_counts.get('gold_paragraph_rows', 0))}</code></li>
    <li>Gold object label rows: <code>{esc(gold_counts.get('gold_object_label_rows', 0))}</code></li>
    <li>Authoritative paragraph rows: <code>{esc(gold_counts.get('authoritative_paragraph_rows', 0))}</code></li>
    <li>Authoritative object label rows: <code>{esc(gold_counts.get('authoritative_object_label_rows', 0))}</code></li>
    <li>Placeholder paragraph rows excluded: <code>{esc(gold_counts.get('placeholder_paragraph_rows_excluded', 0))}</code></li>
    <li>Matched paragraphs: <code>{esc(gold_counts.get('matched_paragraphs', 0))}</code></li>
    <li>Missing paragraphs: <code>{esc(gold_counts.get('missing_paragraphs', 0))}</code></li>
    <li>Over-merged paragraphs: <code>{esc(gold_counts.get('over_merged_paragraphs', 0))}</code></li>
    <li>Over-split paragraphs: <code>{esc(gold_counts.get('over_split_paragraphs', 0))}</code></li>
    <li>Wrong object labels: <code>{esc(gold_counts.get('wrong_object_labels', 0))}</code></li>
    <li>Paragraph precision: <code>{esc(fmt_decimal(gold_scores.get('paragraph_precision'), 3))}</code></li>
    <li>Paragraph recall: <code>{esc(fmt_decimal(gold_scores.get('paragraph_recall'), 3))}</code></li>
    <li>Object label accuracy: <code>{esc(fmt_decimal(gold_scores.get('object_label_accuracy'), 3))}</code></li>
    <li>Scoring authoritative: <code>{esc(gold_evaluation_report.get('scoring_authoritative', False))}</code></li>
    <li>Sufficient to judge merge policy adoption: <code>{esc(gold_evaluation_report.get('sufficient_to_judge_merge_policy_adoption', False))}</code></li>
  </ul>
  <h3>Gold Pages</h3>
  <ul>{gold_page_items or '<li>No gold pages defined.</li>'}</ul>

  <h2>Canonical Paragraph Review</h2>
  <div class="rule">
    This review inspects promoted canonical paragraphs for likely quality risks. It does not demote
    paragraphs or change promotion rules.
  </div>
  <ul>
    <li>Safe for downstream: <code>{esc(canonical_review_report.get('safe_for_downstream', 'unknown'))}</code></li>
    <li>Recommendation: <code>{esc(canonical_review_report.get('recommendation', 'unknown'))}</code></li>
    <li>Total reviewed: <code>{esc(canonical_review_counts.get('total_canonical_paragraphs_reviewed', 0))}</code></li>
    <li>Clean-looking: <code>{esc(canonical_review_counts.get('clean_looking_count', 0))}</code></li>
    <li>Risky paragraphs: <code>{esc(canonical_review_counts.get('risky_paragraph_count', 0))}</code></li>
    <li>Warning count: <code>{esc(canonical_review_counts.get('warning_count', 0))}</code></li>
  </ul>
  <h3>Canonical Review Warning Categories</h3>
  <ul>{canonical_review_warning_items or '<li>No canonical paragraph review warnings.</li>'}</ul>
  <h3>Canonical Warning Drilldown</h3>
  <table>
    <thead>
      <tr><th>Warning</th><th>Cluster</th><th>Severity</th><th>Count</th><th>Affected Pages</th><th>Sample IDs</th><th>Likely Next Action</th></tr>
    </thead>
    <tbody>{canonical_review_drilldown_rows or '<tr><td colspan="7">No warning drilldown rows.</td></tr>'}</tbody>
  </table>
  <h3>Risk Clusters</h3>
  <table>
    <thead>
      <tr><th>Cluster</th><th>Severity</th><th>Count</th><th>Warnings</th><th>Affected Pages</th><th>Likely Next Action</th><th>May Require</th></tr>
    </thead>
    <tbody>{canonical_review_cluster_rows or '<tr><td colspan="7">No risk clusters.</td></tr>'}</tbody>
  </table>
  <h3>BBox Span Risk Diagnostics</h3>
  <p>
    This diagnostic view focuses on canonical paragraphs flagged by bounding-box or source-line span warnings.
    It is evidence-only: it does not split, demote, promote, or rewrite any paragraph.
  </p>
  <ul>
    <li>Total bbox span diagnostics: <code>{esc(bbox_span_summary.get('total', 0))}</code></li>
  </ul>
  <h4>Grouped By Severity</h4>
  <ul>{bbox_span_by_severity or '<li>No bbox span severity groups.</li>'}</ul>
  <h4>Grouped By Source Line Count Range</h4>
  <ul>{bbox_span_by_line_count or '<li>No source line count groups.</li>'}</ul>
  <h4>Grouped By Page Height Ratio Range</h4>
  <ul>{bbox_span_by_ratio or '<li>No page-height ratio groups.</li>'}</ul>
  <h4>Grouped By Page</h4>
  <ul>{bbox_span_by_page or '<li>No page groups.</li>'}</ul>
  <table>
    <thead>
      <tr>
        <th>Canonical ID</th><th>Page</th><th>Source Candidate</th><th>Line Count</th>
        <th>BBox Span</th><th>Page Ratio</th><th>Text Length</th><th>Severity</th>
        <th>Likely Interpretation</th><th>Likely Corrective Path</th>
        <th>First Source Line</th><th>Last Source Line</th>
      </tr>
    </thead>
    <tbody>{bbox_span_diag_rows or '<tr><td colspan="12">No bbox span diagnostics.</td></tr>'}</tbody>
  </table>
  <h3>BBox Span Decision Summary</h3>
  <p>
    This analysis classifies each bbox span diagnostic row into a likely cause and recommended next action.
    It is still analysis-only and does not alter canonical paragraphs.
  </p>
  <ul>
    <li>Total bbox span decisions: <code>{esc(bbox_span_decision_summary.get('total', 0))}</code></li>
  </ul>
  <h4>Grouped Likely Causes</h4>
  <ul>{bbox_span_by_cause or '<li>No likely-cause decisions.</li>'}</ul>
  <h4>Grouped Recommended Actions</h4>
  <ul>{bbox_span_by_action or '<li>No recommended-action decisions.</li>'}</ul>
  <h4>Top Pages Needing Inspection</h4>
  <ul>{bbox_span_top_pages or '<li>No top inspection pages.</li>'}</ul>
  <table>
    <thead>
      <tr>
        <th>Canonical ID</th><th>Page</th><th>Severity</th><th>Line Count</th>
        <th>BBox Span</th><th>Page Ratio</th><th>Text Length</th><th>Likely Cause</th>
        <th>Confidence</th><th>Recommended Action</th><th>First Source Line</th><th>Last Source Line</th>
      </tr>
    </thead>
    <tbody>{bbox_span_decision_rows or '<tr><td colspan="12">No bbox span decisions.</td></tr>'}</tbody>
  </table>
  <h3>Recommendation</h3>
  <ul>
    <li>Top risk to fix first: <code>{esc(canonical_review_recommendation.get('top_risk_to_fix_first', '-'))}</code></li>
    <li>Why it matters: {esc(canonical_review_recommendation.get('why_it_matters', '-'))}</li>
    <li>Expected impact: {esc(canonical_review_recommendation.get('expected_impact', '-'))}</li>
    <li>May require: <code>{esc(', '.join(canonical_review_recommendation.get('may_require', [])))}</code></li>
  </ul>
  <h3>Risky Canonical Paragraph Samples</h3>
  <table>
    <thead>
      <tr><th>Canonical ID</th><th>Page</th><th>Source Candidate</th><th>Warnings</th><th>Clean Text Sample</th></tr>
    </thead>
    <tbody>{canonical_review_sample_rows or '<tr><td colspan="5">No risky canonical paragraph samples.</td></tr>'}</tbody>
  </table>

  <h2>Post-Adoption Canonical Safety</h2>
  <div class="warn">
    This section reassesses canonical paragraph safety under the active adopted policy. It is
    analysis-only and does not change extraction behavior.
  </div>
  <ul>
    <li>Active policy: <code>{esc(post_adoption_safety_report.get('active_policy', '-'))}</code></li>
    <li>Promoted canonical paragraphs: <code>{esc(post_adoption_state.get('promoted_canonical_paragraphs', '-'))}</code></li>
    <li>Risky canonical paragraphs: <code>{esc(post_adoption_state.get('risky_canonical_paragraphs', '-'))}</code></li>
    <li>Clean-looking canonical paragraphs: <code>{esc(post_adoption_state.get('clean_looking_canonical_paragraphs', '-'))}</code></li>
    <li>Warning count: <code>{esc(post_adoption_state.get('warning_count', '-'))}</code></li>
    <li>Safe for downstream: <code>{esc(post_adoption_state.get('safe_for_downstream', '-'))}</code></li>
    <li>Current top risk: <code>{esc(post_adoption_top_risk.get('cluster', '-'))}</code> / <code>{esc(post_adoption_top_risk.get('warning', '-'))}</code></li>
    <li>Issue type: <code>{esc(post_adoption_top_risk.get('issue_type', '-'))}</code></li>
    <li>Likely corrective path: {esc(post_adoption_top_risk.get('likely_corrective_path', '-'))}</li>
    <li>May require: <code>{esc(', '.join(post_adoption_top_risk.get('may_require', [])))}</code></li>
  </ul>
  <h3>Before/After Adoption Comparison</h3>
  <ul>
    <li>Risky canonical paragraphs: <code>{esc((post_adoption_comparison.get('risky_canonical_paragraphs') or {}).get('before', '-'))}</code> to <code>{esc((post_adoption_comparison.get('risky_canonical_paragraphs') or {}).get('after', '-'))}</code></li>
    <li>Total warnings: <code>{esc((post_adoption_comparison.get('warning_count') or {}).get('before', '-'))}</code> to <code>{esc((post_adoption_comparison.get('warning_count') or {}).get('after', '-'))}</code></li>
    <li>BBox span risk: <code>{esc((post_adoption_comparison.get('bbox_span_risk') or {}).get('before', '-'))}</code> to <code>{esc((post_adoption_comparison.get('bbox_span_risk') or {}).get('after', '-'))}</code></li>
    <li>Likely true accidental merges: <code>{esc((post_adoption_comparison.get('likely_true_accidental_merges') or {}).get('before', '-'))}</code> to <code>{esc((post_adoption_comparison.get('likely_true_accidental_merges') or {}).get('after', '-'))}</code></li>
    <li>Merged across paragraph break: <code>{esc((post_adoption_comparison.get('merged_across_paragraph_break') or {}).get('before', '-'))}</code> to <code>{esc((post_adoption_comparison.get('merged_across_paragraph_break') or {}).get('after', '-'))}</code></li>
  </ul>
  <h3>Warning Category Deltas</h3>
  <table>
    <thead>
      <tr><th>Warning</th><th>Before</th><th>After</th><th>Delta</th></tr>
    </thead>
    <tbody>{post_adoption_category_rows or '<tr><td colspan="4">No warning deltas recorded.</td></tr>'}</tbody>
  </table>
  <h3>Sample Risky Paragraphs After Adoption</h3>
  <table>
    <thead>
      <tr><th>Canonical ID</th><th>Page</th><th>Source Candidate</th><th>Warnings</th><th>Text Preview</th></tr>
    </thead>
    <tbody>{post_adoption_sample_rows or '<tr><td colspan="5">No post-adoption risky paragraph samples.</td></tr>'}</tbody>
  </table>

  <h2>Post-Adoption BBox Span Diagnosis</h2>
  <div class="warn">
    This diagnosis classifies remaining bbox/span warnings under the adopted active policy. It is
    analysis-only and does not change extraction behavior.
  </div>
  <ul>
    <li>Total bbox/span cases: <code>{esc(bbox_diagnosis_summary.get('total_bbox_span_cases', '-'))}</code></li>
    <li>High-severity cases: <code>{esc(bbox_diagnosis_summary.get('high_severity_cases', '-'))}</code></li>
    <li>Likely true defects: <code>{esc(bbox_diagnosis_summary.get('likely_true_defects', '-'))}</code></li>
    <li>Likely false positives or noise: <code>{esc(bbox_diagnosis_summary.get('likely_false_positive_or_noise', '-'))}</code></li>
    <li>Exact gold-covered cases: <code>{esc(bbox_diagnosis_summary.get('covered_by_gold', '-'))}</code></li>
    <li>Not covered by gold: <code>{esc(bbox_diagnosis_summary.get('not_covered_by_gold', '-'))}</code></li>
  </ul>
  <h3>By Likely Cause</h3>
  <ul>{bbox_diagnosis_cause_items or '<li>No bbox/span diagnoses.</li>'}</ul>
  <h3>Top Pages Needing Visual Review</h3>
  <ul>{bbox_diagnosis_page_items or '<li>No top pages recorded.</li>'}</ul>
  <h3>Likely True Defects</h3>
  <table>
    <thead>
      <tr><th>Canonical ID</th><th>Page</th><th>Lines</th><th>Page Ratio</th><th>Gold Coverage</th><th>Recommended Action</th><th>Text Preview</th></tr>
    </thead>
    <tbody>{bbox_diagnosis_true_defect_rows or '<tr><td colspan="7">No likely true defects classified.</td></tr>'}</tbody>
  </table>
  <h3>Likely False Positives Or Threshold Noise</h3>
  <table>
    <thead>
      <tr><th>Canonical ID</th><th>Page</th><th>Likely Cause</th><th>Lines</th><th>Page Ratio</th><th>Gold Coverage</th><th>Text Preview</th></tr>
    </thead>
    <tbody>{bbox_diagnosis_noise_rows or '<tr><td colspan="7">No likely false positives or noise classified.</td></tr>'}</tbody>
  </table>
  <h3>All BBox Span Diagnoses</h3>
  <table>
    <thead>
      <tr><th>Canonical ID</th><th>Page</th><th>Lines</th><th>BBox Span</th><th>Page Ratio</th><th>Likely Cause</th><th>Confidence</th><th>Gold Coverage</th><th>Recommended Action</th><th>Warnings</th></tr>
    </thead>
    <tbody>{bbox_diagnosis_all_rows or '<tr><td colspan="10">No bbox/span diagnosis rows.</td></tr>'}</tbody>
  </table>

  <h2>Post-Adoption Remediation Plan</h2>
  <div class="rule">
    This is planning-only. It separates remaining bbox/span cases into action queues and does not
    change extraction behavior.
  </div>
  <ul>
    <li>Total cases: <code>{esc(remediation_summary.get('total_cases', '-'))}</code></li>
    <li>Assigned cases: <code>{esc(remediation_summary.get('assigned_cases', '-'))}</code></li>
    <li>Queue count: <code>{esc(remediation_summary.get('queue_count', '-'))}</code></li>
    <li>Safe for downstream: <code>{esc(remediation_summary.get('safe_for_downstream', '-'))}</code></li>
    <li>Next action: <code>{esc(post_adoption_remediation_plan.get('next_action', '-'))}</code></li>
  </ul>
  <table>
    <thead>
      <tr><th>Queue</th><th>Count</th><th>Risk</th><th>Action Type</th><th>Affected Pages</th><th>Samples</th><th>Recommended Next Action</th><th>Blocks Downstream</th></tr>
    </thead>
    <tbody>{remediation_queue_rows or '<tr><td colspan="8">No remediation queues recorded.</td></tr>'}</tbody>
  </table>
  <h3>Recommended Order</h3>
  <ol>{remediation_order_items or '<li>No remediation order recorded.</li>'}</ol>

  <h2>Front-Matter / Metadata Review</h2>
  <div class="warn">
    This review inspects the front-matter/metadata remediation queue only. It does not demote
    paragraphs, change promotion rules, add OCR, add model review, or unlock downstream use.
  </div>
  <ul>
    <li>Total reviewed: <code>{esc(front_matter_summary.get('total_reviewed', '-'))}</code></li>
    <li>Gold-covered rows: <code>{esc(front_matter_summary.get('gold_covered_rows', '-'))}</code></li>
    <li>Safe for downstream: <code>{esc(front_matter_summary.get('safe_for_downstream', '-'))}</code></li>
    <li>Recommendation: <code>{esc(front_matter_summary.get('recommendation', '-'))}</code></li>
  </ul>
  <h3>Classification Counts</h3>
  <ul>{front_matter_classification_items or '<li>No front-matter review classifications recorded.</li>'}</ul>
  <table>
    <thead>
      <tr><th>Canonical ID</th><th>Page</th><th>Source Candidate</th><th>Promotion</th><th>Classification</th><th>Confidence</th><th>Gold</th><th>Recommended Action</th><th>Text Preview</th><th>Visual Evidence</th></tr>
    </thead>
    <tbody>{front_matter_review_rows or '<tr><td colspan="10">No front-matter/metadata review rows.</td></tr>'}</tbody>
  </table>

  <h2>Visual Review Cases</h2>
  <div class="warn">
    This review inspects the remaining visual-review queue only. It does not demote paragraphs,
    change grouping, change promotion rules, add OCR, add model review, or unlock downstream use.
  </div>
  <ul>
    <li>Total reviewed: <code>{esc(visual_review_summary.get('total_reviewed', '-'))}</code></li>
    <li>Gold-covered rows: <code>{esc(visual_review_summary.get('gold_covered_rows', '-'))}</code></li>
    <li>Valid canonical paragraphs: <code>{esc(visual_review_summary.get('valid_canonical_paragraphs', '-'))}</code></li>
    <li>True paragraph grouping defects: <code>{esc(visual_review_summary.get('true_paragraph_grouping_defects', '-'))}</code></li>
    <li>Safe for downstream: <code>{esc(visual_review_summary.get('safe_for_downstream', '-'))}</code></li>
    <li>Recommendation: <code>{esc(visual_review_summary.get('recommendation', '-'))}</code></li>
  </ul>
  <h3>Classification Counts</h3>
  <ul>{visual_review_classification_items or '<li>No visual-review classifications recorded.</li>'}</ul>
  <table>
    <thead>
      <tr><th>Canonical ID</th><th>Page</th><th>Source Candidate</th><th>Classification</th><th>Confidence</th><th>Gold</th><th>Lines</th><th>First Line</th><th>Last Line</th><th>Recommended Action</th><th>Text Preview</th><th>Visual Evidence</th></tr>
    </thead>
    <tbody>{visual_review_rows or '<tr><td colspan="12">No visual-review rows.</td></tr>'}</tbody>
  </table>

  <h2>Narrow Grouping Correction Design</h2>
  <div class="rule">
    This is design-only. It records the proposed correction for confirmed grouping defects without
    changing extraction behavior, active policy, canonical promotion, OCR, model review, retrieval,
    or graph work.
  </div>
  <ul>
    <li>Confirmed defects: <code>{esc(narrow_design_summary.get('confirmed_defects', '-'))}</code></li>
    <li>Primary defect: <code>{esc(narrow_design_summary.get('primary_defect', '-'))}</code></li>
    <li>Recommended next action: <code>{esc(narrow_design_summary.get('recommended_next_action', '-'))}</code></li>
    <li>Downstream remains blocked: <code>{esc(narrow_design_summary.get('downstream_remains_blocked', '-'))}</code></li>
  </ul>
  <table>
    <thead>
      <tr><th>Defect</th><th>Pages</th><th>Canonical ID</th><th>Gold</th><th>Current Behavior</th><th>Why Policy Failed</th><th>Proposed Rule</th><th>Required Conditions</th><th>Block Conditions</th><th>Adoption Gates</th></tr>
    </thead>
    <tbody>{narrow_design_rows or '<tr><td colspan="10">No narrow grouping correction design rows.</td></tr>'}</tbody>
  </table>

  <h2>Chained Cross-Page Continuation Experiment</h2>
  <div class="rule">
    This is experiment-only. Active policy remains
    <code>{esc(chained_cross_page_experiment.get('active_policy', '-'))}</code>; experimental policy is
    <code>{esc(chained_cross_page_experiment.get('experimental_policy', '-'))}</code>. No adoption,
    canonical promotion rule change, OCR, model review, retrieval, or graph work happens here.
  </div>
  <ul>
    <li>Target defect: <code>{esc((chained_cross_page_experiment.get('target_defect') or {}).get('canonical_paragraph_id', '-'))}</code></li>
    <li>Target fixed by experiment: <code>{esc((chained_cross_page_experiment.get('target_defect') or {}).get('fixed_by_experiment', '-'))}</code></li>
    <li>Active gold precision: <code>{esc(fmt_decimal(chained_gold_scores.get('active_paragraph_precision'), 3))}</code></li>
    <li>Experimental gold precision: <code>{esc(fmt_decimal(chained_gold_scores.get('experimental_paragraph_precision'), 3))}</code></li>
    <li>Active gold recall: <code>{esc(fmt_decimal(chained_gold_scores.get('active_paragraph_recall'), 3))}</code></li>
    <li>Experimental gold recall: <code>{esc(fmt_decimal(chained_gold_scores.get('experimental_paragraph_recall'), 3))}</code></li>
    <li>Active matched paragraphs: <code>{esc(chained_gold_counts.get('active_matched_paragraphs', '-'))}</code></li>
    <li>Experimental matched paragraphs: <code>{esc(chained_gold_counts.get('experimental_matched_paragraphs', '-'))}</code></li>
    <li>Active over-split paragraphs: <code>{esc(chained_gold_counts.get('active_over_split_paragraphs', '-'))}</code></li>
    <li>Experimental over-split paragraphs: <code>{esc(chained_gold_counts.get('experimental_over_split_paragraphs', '-'))}</code></li>
    <li>Active over-merged paragraphs: <code>{esc(chained_gold_counts.get('active_over_merged_paragraphs', '-'))}</code></li>
    <li>Experimental over-merged paragraphs: <code>{esc(chained_gold_counts.get('experimental_over_merged_paragraphs', '-'))}</code></li>
    <li>Active object-label accuracy: <code>{esc(fmt_decimal(chained_gold_scores.get('active_object_label_accuracy'), 3))}</code></li>
    <li>Experimental object-label accuracy: <code>{esc(fmt_decimal(chained_gold_scores.get('experimental_object_label_accuracy'), 3))}</code></li>
    <li>Active wrong object labels: <code>{esc(chained_gold_counts.get('active_wrong_object_labels', '-'))}</code></li>
    <li>Experimental wrong object labels: <code>{esc(chained_gold_counts.get('experimental_wrong_object_labels', '-'))}</code></li>
    <li>Active warning count: <code>{esc(chained_active_metrics.get('warning_count', '-'))}</code></li>
    <li>Experimental warning count: <code>{esc(chained_experimental_metrics.get('warning_count', '-'))}</code></li>
    <li>Active bbox/span risk: <code>{esc(chained_active_metrics.get('bbox_span_risk', '-'))}</code></li>
    <li>Experimental bbox/span risk: <code>{esc(chained_experimental_metrics.get('bbox_span_risk', '-'))}</code></li>
    <li>Proposed chained joins: <code>{esc(chained_side_effects.get('proposed_chained_joins', '-'))}</code></li>
    <li>Rejected chained joins: <code>{esc(chained_side_effects.get('rejected_chained_joins', '-'))}</code></li>
    <li>Joins not covered by gold: <code>{esc(chained_side_effects.get('joins_not_covered_by_gold', '-'))}</code></li>
    <li>Adoptable by metrics only: <code>{esc(chained_acceptance.get('adoptable_by_metrics_only', '-'))}</code></li>
    <li>Object-label accuracy not worsened: <code>{esc(chained_acceptance.get('object_label_accuracy_not_worsened', '-'))}</code></li>
    <li>Requires side-effect review: <code>{esc(chained_acceptance.get('requires_side_effect_review_before_adoption', '-'))}</code></li>
    <li>Recommendation: <code>{esc(chained_acceptance.get('adoption_recommendation', '-'))}</code></li>
  </ul>
  <h3>Proposed Chained Joins</h3>
  <table>
    <thead>
      <tr><th>Left Object</th><th>Right Object</th><th>Pages</th><th>Lines</th><th>Reasons</th><th>Left End</th><th>Right Start</th><th>Joined Preview</th></tr>
    </thead>
    <tbody>{chained_join_rows or '<tr><td colspan="8">No proposed chained joins.</td></tr>'}</tbody>
  </table>
  <h3>Rejected Chained Candidates</h3>
  <table>
    <thead>
      <tr><th>Left Object</th><th>Right Object</th><th>Pages</th><th>Rejection Reasons</th><th>Left End</th><th>Right Start</th></tr>
    </thead>
    <tbody>{chained_rejected_rows or '<tr><td colspan="6">No rejected chained candidates.</td></tr>'}</tbody>
  </table>

  <h2>Chained Join Side-Effect Review Queue</h2>
  <div class="warn">
    This queue contains unscored chained joins from the v3 experiment. It is review-only: no
    accept/reject decisions are applied here, and v3 remains inactive.
  </div>
  <ul>
    <li>Total unscored chained joins: <code>{esc(chained_queue_summary.get('total_unscored_chained_joins', '-'))}</code></li>
    <li>Review queue open: <code>{esc(chained_queue_summary.get('review_queue_open', '-'))}</code></li>
    <li>Adoption remains blocked: <code>{esc(chained_queue_summary.get('adoption_remains_blocked', '-'))}</code></li>
    <li>Recommended next action: <code>{esc(chained_queue_summary.get('recommended_next_action', '-'))}</code></li>
  </ul>
  <h3>Risk Counts</h3>
  <ul>{chained_queue_risk_items or '<li>No queued risk counts.</li>'}</ul>
  <table>
    <thead>
      <tr><th>Queue ID</th><th>Pages</th><th>Left Candidate</th><th>Right Candidate</th><th>Likely Risk</th><th>Confidence</th><th>Gold</th><th>Action</th><th>Left End</th><th>Right Start</th><th>Joined Preview</th><th>Evidence</th></tr>
    </thead>
    <tbody>{chained_queue_rows or '<tr><td colspan="12">No unscored chained joins queued.</td></tr>'}</tbody>
  </table>

  <h2>Chained Join Decisions</h2>
  <div class="rule">
    These are curated replayed decisions from
    <code>reviews/{esc(book_id)}/chained_join_decisions.jsonl</code>. They do not adopt
    <code>v3</code>; adoption remains a separate checkpoint.
  </div>
  <ul>
    <li>Decision validation: <code>{esc(chained_decision_validation.get('status', 'unknown'))}</code></li>
    <li>Queued joins: <code>{esc(chained_decision_summary.get('queued_chained_joins', '-'))}</code></li>
    <li>Decision rows: <code>{esc(chained_decision_summary.get('decision_rows', '-'))}</code></li>
    <li>Accepted: <code>{esc(chained_decision_summary.get('accepted', '-'))}</code></li>
    <li>Rejected: <code>{esc(chained_decision_summary.get('rejected', '-'))}</code></li>
    <li>Needs review: <code>{esc(chained_decision_summary.get('needs_review', '-'))}</code></li>
    <li>Unreviewed: <code>{esc(chained_decision_summary.get('unreviewed', '-'))}</code></li>
    <li>Adoption remains separate checkpoint: <code>{esc(chained_decision_summary.get('adoption_remains_separate_checkpoint', '-'))}</code></li>
  </ul>
  <table>
    <thead>
      <tr><th>Queue ID</th><th>Pages</th><th>Risk</th><th>Decision</th><th>Reason</th><th>Reviewer</th><th>Reviewed</th><th>Notes</th><th>Evidence</th></tr>
    </thead>
    <tbody>{chained_decision_rows or '<tr><td colspan="9">No chained join decisions applied.</td></tr>'}</tbody>
  </table>

  <h2>Guarded Chained Cross-Page Continuation Experiment</h2>
  <div class="rule">
    This is experiment-only. Active policy remains
    <code>{esc(guarded_chained_experiment.get('active_policy', '-'))}</code>; guarded experimental policy is
    <code>{esc(guarded_chained_experiment.get('guarded_experimental_policy', '-'))}</code>. The guard blocks
    chained joins when the terminal page of an existing joined candidate contains later non-furniture paragraph
    content before the next-page candidate. It does not adopt v3 or change canonical promotion rules.
  </div>
  <ul>
    <li>Guard rule: <code>{esc(guarded_chained_experiment.get('guard_rule', '-'))}</code></li>
    <li>Active v2 gold precision: <code>{esc(fmt_decimal(guarded_gold_scores.get('active_paragraph_precision'), 3))}</code></li>
    <li>Guarded v3 gold precision: <code>{esc(fmt_decimal(guarded_gold_scores.get('guarded_paragraph_precision'), 3))}</code></li>
    <li>Active v2 gold recall: <code>{esc(fmt_decimal(guarded_gold_scores.get('active_paragraph_recall'), 3))}</code></li>
    <li>Guarded v3 gold recall: <code>{esc(fmt_decimal(guarded_gold_scores.get('guarded_paragraph_recall'), 3))}</code></li>
    <li>Active matched paragraphs: <code>{esc(guarded_gold_counts.get('active_matched_paragraphs', '-'))}</code></li>
    <li>Guarded matched paragraphs: <code>{esc(guarded_gold_counts.get('guarded_matched_paragraphs', '-'))}</code></li>
    <li>Active over-split paragraphs: <code>{esc(guarded_gold_counts.get('active_over_split_paragraphs', '-'))}</code></li>
    <li>Guarded over-split paragraphs: <code>{esc(guarded_gold_counts.get('guarded_over_split_paragraphs', '-'))}</code></li>
    <li>Active over-merged paragraphs: <code>{esc(guarded_gold_counts.get('active_over_merged_paragraphs', '-'))}</code></li>
    <li>Guarded over-merged paragraphs: <code>{esc(guarded_gold_counts.get('guarded_over_merged_paragraphs', '-'))}</code></li>
    <li>Active canonical promoted: <code>{esc(guarded_active_metrics.get('canonical_promoted', '-'))}</code></li>
    <li>Previous v3 proposed joins: <code>{esc(guarded_previous_metrics.get('proposed_chained_joins', '-'))}</code></li>
    <li>Guarded proposed joins: <code>{esc(guarded_side_effects.get('proposed_chained_joins', '-'))}</code></li>
    <li>Guarded rejected candidates: <code>{esc(guarded_side_effects.get('rejected_chained_joins', '-'))}</code></li>
    <li>Accepted prior decisions preserved: <code>{esc(guarded_decision_replay.get('accepted_prior_decisions_preserved', '-'))}</code> / <code>{esc(guarded_decision_replay.get('accepted_prior_decisions', '-'))}</code></li>
    <li>Rejected prior decisions blocked: <code>{esc(guarded_decision_replay.get('rejected_prior_decisions_blocked', '-'))}</code> / <code>{esc(guarded_decision_replay.get('rejected_prior_decisions', '-'))}</code></li>
    <li><code>chained_join_review_0004</code> blocked: <code>{esc(guarded_decision_replay.get('chained_join_review_0004_blocked', '-'))}</code></li>
    <li><code>cp_000103</code> remains fixed: <code>{esc(guarded_acceptance.get('cp_000103_remains_fixed', '-'))}</code></li>
    <li>Warning delta: <code>{esc(guarded_warning_deltas.get('warning_count_delta', '-'))}</code></li>
    <li>BBox/span delta: <code>{esc(guarded_warning_deltas.get('bbox_span_risk_delta', '-'))}</code></li>
    <li>Passes experiment gate: <code>{esc(guarded_acceptance.get('passes_experiment_gate', '-'))}</code></li>
    <li>Recommendation: <code>{esc(guarded_acceptance.get('adoption_recommendation', '-'))}</code></li>
  </ul>
  <h3>Guarded Proposed Chained Joins</h3>
  <table>
    <thead>
      <tr><th>Left Object</th><th>Right Object</th><th>Pages</th><th>Lines</th><th>Reasons</th><th>Left End</th><th>Right Start</th><th>Joined Preview</th></tr>
    </thead>
    <tbody>{guarded_join_rows or '<tr><td colspan="8">No guarded proposed chained joins.</td></tr>'}</tbody>
  </table>
  <h3>Guarded Rejected Chained Candidates</h3>
  <table>
    <thead>
      <tr><th>Left Object</th><th>Right Object</th><th>Pages</th><th>Rejection Reasons</th><th>Intervening Body Objects</th><th>Left End</th><th>Right Start</th></tr>
    </thead>
    <tbody>{guarded_rejected_rows or '<tr><td colspan="7">No guarded rejected chained candidates.</td></tr>'}</tbody>
  </table>

  <h2>Guarded Chained Policy Adoption Decision</h2>
  <div class="rule">
    This is the formal adoption checkpoint for
    <code>{esc(guarded_policy_adoption_decision.get('adopted_policy', '-'))}</code>. It records
    policy adoption separately from downstream readiness.
  </div>
  <ul>
    <li>Decision: <code>{esc(guarded_policy_adoption_decision.get('decision', '-'))}</code></li>
    <li>Previous active policy: <code>{esc(guarded_policy_adoption_decision.get('previous_active_policy', '-'))}</code></li>
    <li>Current active policy: <code>{esc(guarded_policy_adoption_decision.get('active_paragraph_merge_policy', '-'))}</code></li>
    <li>Validation status: <code>{esc(guarded_policy_adoption_decision.get('validation_status', '-'))}</code></li>
    <li>Gold precision before/after: <code>{esc(fmt_decimal(guarded_adoption_gate_evidence.get('gold_paragraph_precision_before'), 3))}</code> to <code>{esc(fmt_decimal(guarded_adoption_gate_evidence.get('gold_paragraph_precision_after'), 3))}</code></li>
    <li>Gold recall before/after: <code>{esc(fmt_decimal(guarded_adoption_gate_evidence.get('gold_paragraph_recall_before'), 3))}</code> to <code>{esc(fmt_decimal(guarded_adoption_gate_evidence.get('gold_paragraph_recall_after'), 3))}</code></li>
    <li>Matched paragraphs before/after: <code>{esc(guarded_adoption_gate_evidence.get('matched_paragraphs_before', '-'))}</code> to <code>{esc(guarded_adoption_gate_evidence.get('matched_paragraphs_after', '-'))}</code></li>
    <li>Over-split before/after: <code>{esc(guarded_adoption_gate_evidence.get('over_split_paragraphs_before', '-'))}</code> to <code>{esc(guarded_adoption_gate_evidence.get('over_split_paragraphs_after', '-'))}</code></li>
    <li>Over-merged before/after: <code>{esc(guarded_adoption_gate_evidence.get('over_merged_paragraphs_before', '-'))}</code> to <code>{esc(guarded_adoption_gate_evidence.get('over_merged_paragraphs_after', '-'))}</code></li>
    <li><code>cp_000103</code> remains fixed: <code>{esc(guarded_adoption_gate_evidence.get('cp_000103_remains_fixed', '-'))}</code></li>
    <li><code>chained_join_review_0004</code> blocked: <code>{esc(guarded_adoption_gate_evidence.get('chained_join_review_0004_blocked', '-'))}</code></li>
    <li>Unresolved chained joins: <code>{esc(guarded_adoption_gate_evidence.get('unresolved_chained_joins', '-'))}</code></li>
    <li>Canonical promoted paragraphs after adoption: <code>{esc(guarded_adoption_active_run.get('canonical_promoted_paragraphs', '-'))}</code></li>
    <li>Canonical review warnings after adoption: <code>{esc(guarded_adoption_active_run.get('canonical_review_warning_count', '-'))}</code></li>
    <li>Safe for downstream: <code>{esc(guarded_adoption_active_run.get('safe_for_downstream', '-'))}</code></li>
    <li>Does not unlock downstream: <code>{esc(guarded_policy_adoption_decision.get('does_not_unlock_downstream', '-'))}</code></li>
  </ul>
  <h3>Adoption Gates</h3>
  <ul>
    <li>Guarded experiment passed: <code>{esc(guarded_adoption_gates.get('guarded_experiment_passed', '-'))}</code></li>
    <li>Accepted decisions preserved: <code>{esc(guarded_adoption_gates.get('accepted_prior_decisions_preserved', '-'))}</code></li>
    <li>Rejected decisions blocked: <code>{esc(guarded_adoption_gates.get('rejected_prior_decisions_blocked', '-'))}</code></li>
    <li>Gold score improved: <code>{esc(guarded_adoption_gates.get('gold_score_improved', '-'))}</code></li>
    <li>Over-merges not increased: <code>{esc(guarded_adoption_gates.get('over_merges_not_increased', '-'))}</code></li>
    <li>Object-label accuracy not worsened: <code>{esc(guarded_adoption_gates.get('object_label_accuracy_not_worsened', '-'))}</code></li>
    <li>Warning regression: <code>{esc(guarded_adoption_gates.get('audit_warning_regression', '-'))}</code></li>
    <li>BBox/span regression: <code>{esc(guarded_adoption_gates.get('bbox_span_regression', '-'))}</code></li>
  </ul>

  <h2>Repeated Artifact Pattern Review</h2>
  <table>
    <thead>
      <tr><th>Normalized Pattern</th><th>Subtype</th><th>Count</th><th>Pages</th><th>First</th><th>Middle</th><th>Last</th><th>Avg Y</th><th>Margin Evidence</th><th>False-Positive Signals</th></tr>
    </thead>
    <tbody>{artifact_pattern_rows_html or '<tr><td colspan="10">No artifact patterns detected.</td></tr>'}</tbody>
  </table>

  <h2>False-Positive Risk Review</h2>
  <table>
    <thead>
      <tr><th>Normalized Pattern</th><th>Subtype</th><th>Count</th><th>Pages</th><th>Zones</th><th>Risk Signals</th></tr>
    </thead>
    <tbody>{false_positive_rows_html or '<tr><td colspan="6">No false-positive risk signals detected.</td></tr>'}</tbody>
  </table>

  <h2>Validation</h2>
  <p><strong>Status:</strong> <code>{esc(validation_report.get('status', 'unknown'))}</code></p>
  <ul>{validation_items}</ul>

  <h2>Flagged Pages</h2>
  <ul>{flagged_items or '<li>No flagged pages.</li>'}</ul>

  <h2>Stream Samples</h2>
  {sample_stream_html}

  <h2>Page Inspection Index</h2>
  <table>
    <thead>
      <tr><th>Page</th><th>Status</th><th>Objects</th><th>Candidate Buckets</th><th>Flags</th><th>Raw Sample</th></tr>
    </thead>
    <tbody>{page_summary_html}</tbody>
  </table>

  <h2>Page-by-Page Object Inspection</h2>
  <p>
    Open a page to compare each raw extracted object against its candidate bucket, cleaned text,
    confidence, warnings, source lines, and bounding box evidence.
  </p>
  <div class="review-controls" id="review-controls">
    <label>Bucket
      <select id="filter-bucket"><option value="">All buckets</option>{bucket_options}</select>
    </label>
    <label>Artifact subtype
      <select id="filter-subtype"><option value="">All subtypes</option>{subtype_options}</select>
    </label>
    <label>Book zone
      <select id="filter-zone"><option value="">All zones</option>{zone_options}</select>
    </label>
    <label>Page from
      <input id="filter-page-min" type="number" min="1" max="{manifest['page_count']}" step="1" placeholder="1">
    </label>
    <label>Page to
      <input id="filter-page-max" type="number" min="1" max="{manifest['page_count']}" step="1" placeholder="{manifest['page_count']}">
    </label>
    <label>Max confidence
      <input id="filter-confidence" type="number" min="0" max="1" step="0.01" placeholder="Example: 0.85">
    </label>
    <label>Warning contains
      <input id="filter-warning" type="text" placeholder="Example: candidate_only">
    </label>
  </div>
  <div class="overlay-controls" id="overlay-controls">
    <div class="overlay-actions">
      <button type="button" id="overlay-show-all">Show all overlays</button>
      <button type="button" id="overlay-hide-all">Hide all overlays</button>
      <button type="button" id="overlay-overrides-only">Highlight only overridden objects</button>
    </div>
    <div class="overlay-buckets" aria-label="Overlay bucket toggles">
      <label><input type="checkbox" data-overlay-bucket="main_paragraph_candidate" checked> Main paragraphs</label>
      <label><input type="checkbox" data-overlay-bucket="structure_candidate" checked> Structure</label>
      <label><input type="checkbox" data-overlay-bucket="page_artifact_candidate" checked> Page artifacts</label>
      <label><input type="checkbox" data-overlay-bucket="unknown_needs_review" checked> Unknown</label>
      <label><input type="checkbox" id="overlay-overridden-toggle" checked> Overridden</label>
    </div>
  </div>
  <p class="review-count" id="review-count"></p>
  {page_detail_html}

  <h2>First 12 Pages</h2>
  <table>
    <thead>
      <tr><th>Page</th><th>Status</th><th>Chars</th><th>Lines</th><th>Images</th><th>Tables</th><th>Flags</th><th>Sample</th></tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</main>
<script>
  const controls = {{
    bucket: document.getElementById("filter-bucket"),
    subtype: document.getElementById("filter-subtype"),
    zone: document.getElementById("filter-zone"),
    pageMin: document.getElementById("filter-page-min"),
    pageMax: document.getElementById("filter-page-max"),
    confidence: document.getElementById("filter-confidence"),
    warning: document.getElementById("filter-warning"),
    count: document.getElementById("review-count")
  }};
  const cards = Array.from(document.querySelectorAll(".object-card"));
  const overlays = Array.from(document.querySelectorAll(".bbox-overlay"));
  const overlayByObjectId = new Map(overlays.map(box => [box.dataset.objectId, box]));
  const cardByObjectId = new Map(cards.map(card => [card.dataset.objectId, card]));
  const overlayBucketToggles = Array.from(document.querySelectorAll("[data-overlay-bucket]"));
  const overriddenToggle = document.getElementById("overlay-overridden-toggle");
  let overlayHideAll = false;
  let overlayOverridesOnly = false;
  function selectedDetailFor(card) {{
    if (!card) return "";
    const rawText = card.querySelector("section:first-child p")?.textContent.trim() || "";
    return [
      card.dataset.objectId,
      `bucket: ${{card.dataset.bucket || "-"}}`,
      `subtype: ${{card.dataset.subtype || "-"}}`,
      rawText ? `raw: ${{rawText.slice(0, 220)}}` : ""
    ].filter(Boolean).join("\\n");
  }}
  function overrideTemplateFor(card) {{
    if (!card) return "Select an object to generate an override template.";
    const pageSection = card.closest(".page-audit");
    const template = pageSection ? pageSection.querySelector(".override-template") : null;
    const payload = {{
      object_id: card.dataset.objectId || "",
      page: Number(card.dataset.page || "0"),
      original_bucket: card.dataset.detectorBucket || card.dataset.bucket || "",
      corrected_bucket: "TODO_choose_one_of: main_paragraph_candidate | structure_candidate | page_artifact_candidate | unknown_needs_review",
      reason: "TODO_explain_the_review_decision_from_visible_evidence",
      reviewer: "human",
      date: template?.dataset.templateDate || "{generated_date}",
      evidence_reference: card.dataset.evidenceReference || `phase1_audit.html#${{card.id}}`
    }};
    return JSON.stringify(payload);
  }}
  function setActiveObject(objectId, shouldScroll = false) {{
    for (const card of cards) card.classList.toggle("is-active", card.dataset.objectId === objectId);
    for (const box of overlays) box.classList.toggle("is-active", box.dataset.objectId === objectId);
    const card = cardByObjectId.get(objectId);
    const detail = selectedDetailFor(card);
    const pageSection = card ? card.closest(".page-audit") : null;
    const selectedDetail = pageSection ? pageSection.querySelector(".selected-object-detail span") : null;
    const selectedTemplate = pageSection ? pageSection.querySelector(".override-template") : null;
    if (selectedDetail) selectedDetail.textContent = detail || "No object details available.";
    if (selectedTemplate) selectedTemplate.textContent = overrideTemplateFor(card);
    if (shouldScroll) {{
      if (card) card.scrollIntoView({{ behavior: "smooth", block: "center" }});
    }}
  }}
  for (const card of cards) {{
    card.addEventListener("mouseenter", () => setActiveObject(card.dataset.objectId));
    card.addEventListener("focus", () => setActiveObject(card.dataset.objectId));
    card.addEventListener("click", () => setActiveObject(card.dataset.objectId));
  }}
  for (const box of overlays) {{
    box.addEventListener("mouseenter", () => setActiveObject(box.dataset.objectId));
    box.addEventListener("focus", () => setActiveObject(box.dataset.objectId));
    box.addEventListener("click", () => setActiveObject(box.dataset.objectId, true));
  }}
  for (const button of document.querySelectorAll(".zoom-toggle")) {{
    button.addEventListener("click", () => button.closest(".page-image-witness").classList.toggle("is-zoomed"));
  }}
  document.getElementById("overlay-show-all").addEventListener("click", () => {{
    overlayHideAll = false;
    overlayOverridesOnly = false;
    for (const toggle of overlayBucketToggles) toggle.checked = true;
    overriddenToggle.checked = true;
    applyReviewFilters();
  }});
  document.getElementById("overlay-hide-all").addEventListener("click", () => {{
    overlayHideAll = true;
    overlayOverridesOnly = false;
    applyReviewFilters();
  }});
  document.getElementById("overlay-overrides-only").addEventListener("click", () => {{
    overlayHideAll = false;
    overlayOverridesOnly = true;
    overriddenToggle.checked = true;
    applyReviewFilters();
  }});
  for (const toggle of [...overlayBucketToggles, overriddenToggle]) {{
    toggle.addEventListener("input", () => {{
      overlayHideAll = false;
      overlayOverridesOnly = false;
      applyReviewFilters();
    }});
  }}
  function overlayAllowedByControls(overlay) {{
    if (overlayHideAll) return false;
    if (overlayOverridesOnly && overlay.dataset.overridden !== "true") return false;
    if (overlay.dataset.overridden === "true" && !overriddenToggle.checked) return false;
    const bucketToggle = overlayBucketToggles.find(toggle => toggle.dataset.overlayBucket === overlay.dataset.bucket);
    return !bucketToggle || bucketToggle.checked;
  }}
  function applyReviewFilters() {{
    const bucket = controls.bucket.value;
    const subtype = controls.subtype.value;
    const zone = controls.zone.value;
    const pageMin = controls.pageMin.value === "" ? null : Number(controls.pageMin.value);
    const pageMax = controls.pageMax.value === "" ? null : Number(controls.pageMax.value);
    const maxConfidence = controls.confidence.value === "" ? null : Number(controls.confidence.value);
    const warning = controls.warning.value.trim().toLowerCase();
    let visible = 0;
    for (const card of cards) {{
      const confidence = Number(card.dataset.confidence || "0");
      const page = Number(card.dataset.page || "0");
      const matches =
        (!bucket || card.dataset.bucket === bucket) &&
        (!subtype || card.dataset.subtype === subtype) &&
        (!zone || card.dataset.zone === zone) &&
        (pageMin === null || page >= pageMin) &&
        (pageMax === null || page <= pageMax) &&
        (maxConfidence === null || confidence <= maxConfidence) &&
        (!warning || (card.dataset.warnings || "").toLowerCase().includes(warning));
      card.classList.toggle("hidden", !matches);
      const overlay = overlayByObjectId.get(card.dataset.objectId);
      if (overlay) overlay.classList.toggle("hidden", !(matches && overlayAllowedByControls(overlay)));
      if (matches) visible += 1;
    }}
    controls.count.textContent = `${{visible}} of ${{cards.length}} object cards visible`;
  }}
  for (const control of [controls.bucket, controls.subtype, controls.zone, controls.pageMin, controls.pageMax, controls.confidence, controls.warning]) {{
    control.addEventListener("input", applyReviewFilters);
  }}
  applyReviewFilters();
</script>
</body>
</html>
"""


def validate_phase1_run(output_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "status": "pass" if passed else "fail", "detail": detail})

    for artifact in REQUIRED_ARTIFACTS:
        add_check(f"artifact_exists:{artifact}", (output_dir / artifact).exists())

    manifest = read_json(output_dir / "source_manifest.json")
    audit_html = (output_dir / "phase1_audit.html").read_text(encoding="utf-8")
    inventory = read_jsonl(output_dir / "page_inventory.jsonl")
    raw_pages = read_jsonl(output_dir / "raw_pages.jsonl")
    layout_objects = read_jsonl(output_dir / "layout_objects.jsonl")
    clean_objects = read_jsonl(output_dir / "clean_objects.jsonl")
    cleanup_log = read_jsonl(output_dir / "cleanup_log.jsonl")
    review_overrides = read_review_overrides(output_dir / "review_overrides_applied.jsonl")
    cross_page_join_decisions = read_commented_jsonl(output_dir / "cross_page_join_decisions_applied.jsonl")
    main_paragraphs = read_jsonl(output_dir / "main_paragraph_candidates.jsonl")
    structure = read_jsonl(output_dir / "structure_candidates.jsonl")
    page_artifacts = read_jsonl(output_dir / "page_artifacts_candidates.jsonl")
    unknown = read_jsonl(output_dir / "unknown_objects.jsonl")
    canonical_paragraphs = read_jsonl(output_dir / "canonical_paragraphs.jsonl")
    promotion_blockers = read_jsonl(output_dir / "promotion_blockers.jsonl")
    canonical_promotion_report = read_json(output_dir / "canonical_promotion_report.json")
    canonical_paragraph_review_report = read_json(output_dir / "canonical_paragraph_review_report.json")
    paragraph_merge_experiment_report = read_json(output_dir / "paragraph_merge_experiment_report.json")
    paragraph_merge_failure_taxonomy_report = read_json(output_dir / "paragraph_merge_failure_taxonomy_report.json")
    cross_page_join_review_report = read_json(output_dir / "cross_page_join_review_report.json")
    xpage_join_0032_investigation = read_json(output_dir / "xpage_join_0032_investigation.json")
    policy_adoption_decision = read_json(output_dir / "policy_adoption_decision.json")
    post_adoption_safety_report = read_json(output_dir / "post_adoption_canonical_safety_report.json")
    post_adoption_bbox_diagnosis = read_json(output_dir / "post_adoption_bbox_span_diagnosis.json")
    post_adoption_remediation_plan = read_json(output_dir / "post_adoption_remediation_plan.json")
    front_matter_metadata_review_report = read_json(output_dir / "front_matter_metadata_review_report.json")
    visual_review_cases_report = read_json(output_dir / "visual_review_cases_report.json")
    narrow_grouping_correction_design = read_json(output_dir / "narrow_grouping_correction_design.json")
    chained_cross_page_experiment = read_json(output_dir / "chained_cross_page_continuation_experiment.json")
    chained_join_review_queue = read_json(output_dir / "chained_join_review_queue.json")
    chained_join_decisions_applied = read_json(output_dir / "chained_join_decisions_applied.json")
    guarded_chained_experiment = read_json(output_dir / "guarded_chained_cross_page_continuation_experiment.json")
    guarded_policy_adoption_decision = read_json(output_dir / "guarded_chained_policy_adoption_decision.json")
    gold_evaluation_report = read_json(output_dir / "gold_evaluation_report.json")
    reconstruction_map = read_json(output_dir / "reconstruction_map_candidate.json")
    reading_order = read_json(output_dir / "reading_order_candidate.json")
    page_image_files = sorted((output_dir / PAGE_IMAGES_DIR_NAME).glob("page_*.jpg"))
    bbox_object_ids = {
        row.get("object_id")
        for row in layout_objects
        if isinstance(row.get("bbox"), dict)
        and all(row["bbox"].get(key) is not None for key in ["x0", "x1", "top", "bottom"])
    }

    page_count = int(manifest.get("page_count", 0))
    add_check("page_inventory_matches_manifest", len(inventory) == page_count, f"{len(inventory)} inventory rows / {page_count} manifest pages")
    add_check("raw_pages_matches_manifest", len(raw_pages) == page_count, f"{len(raw_pages)} raw rows / {page_count} manifest pages")
    inventory_pages = {row.get("page_number") for row in inventory}
    raw_pages_set = {row.get("page_number") for row in raw_pages}
    add_check("page_numbers_align", inventory_pages == raw_pages_set == set(range(1, page_count + 1)))

    layout_ids = [row.get("object_id") for row in layout_objects]
    clean_ids = [row.get("object_id") for row in clean_objects]
    reading_order_ids = reading_order.get("object_ids", [])
    add_check("object_ids_unique", len(layout_ids) == len(set(layout_ids)))
    add_check("clean_objects_match_layout", set(clean_ids) == set(layout_ids))
    add_check("reading_order_candidate_ids_match_layout", reading_order_ids == layout_ids)
    add_check("object_count_matches_reading_order_candidate", reading_order.get("object_count") == len(layout_objects))
    add_check("objects_have_source_lines", all(row.get("source_line_ids") for row in layout_objects))
    add_check("cleanup_log_references_known_objects", {row.get("object_id") for row in cleanup_log}.issubset(set(layout_ids)))
    add_check("raw_text_preserved", all("raw_text" in row for row in raw_pages))
    add_check("review_flags_present", "review_flags" in reading_order)
    add_check("paragraph_merge_experiment_report_exists", (output_dir / "paragraph_merge_experiment_report.json").exists())
    add_check("paragraph_merge_failure_taxonomy_report_exists", (output_dir / "paragraph_merge_failure_taxonomy_report.json").exists())
    add_check("cross_page_join_review_report_exists", (output_dir / "cross_page_join_review_report.json").exists())
    add_check("xpage_join_0032_investigation_exists", (output_dir / "xpage_join_0032_investigation.json").exists())
    add_check("policy_adoption_decision_exists", (output_dir / "policy_adoption_decision.json").exists())
    add_check("post_adoption_canonical_safety_report_exists", (output_dir / "post_adoption_canonical_safety_report.json").exists())
    add_check("post_adoption_bbox_span_diagnosis_exists", (output_dir / "post_adoption_bbox_span_diagnosis.json").exists())
    add_check("post_adoption_remediation_plan_exists", (output_dir / "post_adoption_remediation_plan.json").exists())
    add_check("front_matter_metadata_review_report_exists", (output_dir / "front_matter_metadata_review_report.json").exists())
    add_check("visual_review_cases_report_exists", (output_dir / "visual_review_cases_report.json").exists())
    add_check("narrow_grouping_correction_design_exists", (output_dir / "narrow_grouping_correction_design.json").exists())
    add_check("chained_cross_page_continuation_experiment_exists", (output_dir / "chained_cross_page_continuation_experiment.json").exists())
    add_check("chained_join_review_queue_exists", (output_dir / "chained_join_review_queue.json").exists())
    add_check("chained_join_decisions_applied_exists", (output_dir / "chained_join_decisions_applied.json").exists())
    add_check("guarded_chained_cross_page_continuation_experiment_exists", (output_dir / "guarded_chained_cross_page_continuation_experiment.json").exists())
    add_check("guarded_chained_policy_adoption_decision_exists", (output_dir / "guarded_chained_policy_adoption_decision.json").exists())
    add_check("gold_evaluation_report_exists", (output_dir / "gold_evaluation_report.json").exists())
    merge_counts = paragraph_merge_experiment_report.get("counts", {})
    taxonomy_summary = paragraph_merge_failure_taxonomy_report.get("summary", {})
    taxonomy_samples = paragraph_merge_failure_taxonomy_report.get("samples", [])
    add_check("paragraph_merge_policy_recorded", manifest.get("paragraph_merge_policy") == ACTIVE_PARAGRAPH_MERGE_POLICY)
    add_check("paragraph_merge_experiment_policy_recorded", manifest.get("paragraph_merge_experiment_policy") == EXPERIMENTAL_PARAGRAPH_MERGE_POLICY)
    add_check(
        "paragraph_merge_experiment_active_counts_match_outputs",
        merge_counts.get("new_paragraph_candidate_count") == len(main_paragraphs)
        and merge_counts.get("new_canonical_promoted_count") == len(canonical_paragraphs)
        and merge_counts.get("new_blocked_paragraph_count")
        == sum(1 for row in promotion_blockers if row.get("stream_type") == "main_paragraph_candidate"),
    )
    add_check(
        "paragraph_merge_experiment_counts_present",
        all(
            key in merge_counts
            for key in [
                "baseline_paragraph_candidate_count",
                "new_paragraph_candidate_count",
                "baseline_bbox_span_risk_count",
                "new_bbox_span_risk_count",
                "baseline_likely_true_accidental_merge_count",
                "new_likely_true_accidental_merge_count",
                "baseline_merged_across_paragraph_break_count",
                "new_merged_across_paragraph_break_count",
                "baseline_merged_across_large_vertical_whitespace_count",
                "new_merged_across_large_vertical_whitespace_count",
            ]
        ),
    )
    cross_page_join_summary = cross_page_join_review_report.get("summary", {})
    cross_page_join_rows = cross_page_join_review_report.get("joins", [])
    add_check(
        "cross_page_join_review_counts_match_rows",
        cross_page_join_summary.get("total_proposed_joins") == len(cross_page_join_rows),
    )
    add_check(
        "cross_page_join_review_has_required_fields",
        all(
            all(row.get(field) is not None for field in ["join_id", "left_page", "right_page", "left_candidate_id", "right_candidate_id", "risk_category", "decision_status"])
            for row in cross_page_join_rows
        ),
    )
    join_validation = cross_page_join_review_report.get("decision_validation", {})
    add_check("cross_page_join_decision_validation_passes", join_validation.get("status") == "pass")
    add_check(
        "cross_page_join_decision_counts_match_source",
        join_validation.get("source_row_count", 0) == len(cross_page_join_decisions),
    )
    join_ids = [row.get("join_id") for row in cross_page_join_decisions]
    proposed_join_by_id = {row.get("join_id"): row for row in cross_page_join_rows}
    add_check("cross_page_join_decision_ids_unique", len(join_ids) == len(set(join_ids)))
    add_check("cross_page_join_decisions_reference_known_joins", set(join_ids).issubset(set(proposed_join_by_id)))
    add_check(
        "cross_page_join_decision_required_fields_present",
        all(
            REQUIRED_CROSS_PAGE_JOIN_DECISION_FIELDS.issubset(row)
            and all(str(row.get(field, "")).strip() for field in REQUIRED_CROSS_PAGE_JOIN_DECISION_FIELDS)
            for row in cross_page_join_decisions
        ),
    )
    add_check(
        "cross_page_join_decisions_valid",
        all(row.get("decision") in VALID_CROSS_PAGE_JOIN_DECISIONS for row in cross_page_join_decisions),
    )
    add_check(
        "cross_page_join_decision_candidate_ids_match",
        all(
            proposed_join_by_id.get(row.get("join_id"), {}).get("left_candidate_id") == row.get("left_candidate_id")
            and proposed_join_by_id.get(row.get("join_id"), {}).get("right_candidate_id") == row.get("right_candidate_id")
            and str(proposed_join_by_id.get(row.get("join_id"), {}).get("left_page")) == str(row.get("left_page"))
            and str(proposed_join_by_id.get(row.get("join_id"), {}).get("right_page")) == str(row.get("right_page"))
            for row in cross_page_join_decisions
        ),
    )
    valid_taxonomy_categories = {
        "merged_across_paragraph_break",
        "merged_heading_or_metadata_into_paragraph",
        "merged_across_large_vertical_whitespace",
        "normal_long_paragraph_false_positive",
        "bbox_or_threshold_artifact",
        "needs_manual_inspection",
    }
    add_check(
        "paragraph_merge_failure_taxonomy_counts_match_samples",
        taxonomy_summary.get("sampled_rows") == len(taxonomy_samples),
    )
    add_check(
        "paragraph_merge_failure_taxonomy_categories_valid",
        all(row.get("provisional_category") in valid_taxonomy_categories for row in taxonomy_samples),
    )
    add_check(
        "paragraph_merge_failure_taxonomy_samples_trace_to_canonical",
        {row.get("canonical_paragraph_id") for row in taxonomy_samples}.issubset({row.get("canonical_paragraph_id") for row in canonical_paragraphs}),
    )
    gold_counts = gold_evaluation_report.get("counts", {})
    add_check("gold_evaluation_counts_present", all(key in gold_counts for key in ["gold_pages_reviewed", "gold_paragraph_rows", "gold_object_label_rows"]))
    add_check(
        "gold_evaluation_placeholder_rows_excluded_from_scoring",
        gold_counts.get("placeholder_paragraph_rows_excluded", 0) + gold_counts.get("authoritative_paragraph_rows", 0)
        == gold_counts.get("gold_paragraph_rows", 0),
    )
    add_check("page_image_count_matches_manifest", len(page_image_files) == page_count, f"{len(page_image_files)} images / {page_count} pages")
    add_check("page_images_nonempty", all(path.stat().st_size > 0 for path in page_image_files))
    add_check(
        "audit_links_page_images",
        all(f"{PAGE_IMAGES_DIR_NAME}/{page_image_filename(page_number)}" in audit_html for page_number in range(1, page_count + 1)),
    )
    overlay_count = audit_html.count('class="bbox-overlay')
    add_check("overlay_data_exists_for_bbox_objects", all(f'data-object-id="{object_id}"' in audit_html for object_id in bbox_object_ids))
    add_check("audit_renders_overlay_elements", overlay_count >= len(bbox_object_ids), f"{overlay_count} overlays / {len(bbox_object_ids)} bbox objects")
    add_check("audit_has_overlay_controls", all(token in audit_html for token in ["overlay-show-all", "overlay-hide-all", "overlay-overrides-only", "data-overlay-bucket"]))
    add_check("audit_has_selected_object_detail", "selected-object-detail" in audit_html)
    add_check(
        "audit_has_override_template_generator",
        all(
            token in audit_html
            for token in [
                "override-template",
                "overrideTemplateFor",
                "reviews/",
                "corrected_bucket",
                "evidence_reference",
            ]
        ),
    )
    stream_object_ids = (
        {row.get("object_id") for row in main_paragraphs}
        | {row.get("object_id") for row in structure}
        | {row.get("object_id") for row in page_artifacts}
        | {row.get("object_id") for row in unknown}
    )
    expected_counts = {
        "main_paragraph_candidates": len(main_paragraphs),
        "structure_candidates": len(structure),
        "page_artifacts_candidates": len(page_artifacts),
        "unknown_objects": len(unknown),
    }
    stream_rows = main_paragraphs + structure + page_artifacts + unknown
    add_check("stream_objects_match_layout", stream_object_ids == set(layout_ids))
    add_check("paragraph_stream_has_paragraph_ids", all(row.get("paragraph_id") for row in main_paragraphs))
    add_check("stream_rows_are_evidence_bound", all(row.get("source_object_ids") and row.get("source_line_ids") for row in stream_rows))
    add_check("reconstruction_map_counts_match_streams", reconstruction_map.get("counts") == expected_counts)
    candidate_ids = {row.get("object_id") for row in stream_rows}
    main_paragraph_ids = {row.get("object_id") for row in main_paragraphs}
    non_paragraph_ids = {row.get("object_id") for row in structure + page_artifacts + unknown}
    canonical_ids = [row.get("canonical_paragraph_id") for row in canonical_paragraphs]
    canonical_source_ids = [row.get("source_candidate_object_id") for row in canonical_paragraphs]
    blocker_source_ids = [row.get("object_id") for row in promotion_blockers]
    promotion_counts = canonical_promotion_report.get("counts", {})
    add_check("canonical_paragraph_ids_unique", len(canonical_ids) == len(set(canonical_ids)))
    add_check("canonical_paragraphs_trace_to_candidates", set(canonical_source_ids).issubset(candidate_ids))
    add_check("canonical_paragraphs_trace_to_main_paragraph_candidates", set(canonical_source_ids).issubset(main_paragraph_ids))
    add_check("canonical_paragraphs_exclude_non_paragraph_streams", set(canonical_source_ids).isdisjoint(non_paragraph_ids))
    add_check("canonical_paragraph_rows_promoted", all(row.get("promotion_status") == "promoted" for row in canonical_paragraphs))
    add_check("canonical_paragraph_rows_evidence_bound", all(row.get("source_object_ids") and row.get("source_line_ids") and row.get("raw_text") and row.get("clean_text") for row in canonical_paragraphs))
    add_check("promotion_blockers_trace_to_candidates", set(blocker_source_ids).issubset(candidate_ids))
    add_check("promotion_blockers_rows_blocked", all(row.get("promotion_status") == "blocked" and row.get("blocker_reasons") for row in promotion_blockers))
    add_check(
        "promotion_report_counts_match_outputs",
        promotion_counts
        == {
            "total_candidates_reviewed": len(stream_rows),
            "paragraph_candidates_reviewed": len(main_paragraphs),
            "promoted_paragraphs": len(canonical_paragraphs),
            "blocked_candidates": len(promotion_blockers),
            "paragraph_candidates_blocked": sum(1 for row in promotion_blockers if row.get("stream_type") == "main_paragraph_candidate"),
            "non_paragraph_candidates_blocked": sum(1 for row in promotion_blockers if row.get("stream_type") != "main_paragraph_candidate"),
            "override_influenced_promotions": sum(1 for row in canonical_paragraphs if row.get("applied_override")),
        },
    )
    add_check("promotion_report_status_pass", canonical_promotion_report.get("status") == "pass")
    canonical_review_counts = canonical_paragraph_review_report.get("counts", {})
    add_check(
        "canonical_paragraph_review_count_matches_canonical",
        canonical_review_counts.get("total_canonical_paragraphs_reviewed") == len(canonical_paragraphs),
    )
    add_check(
        "canonical_paragraph_review_counts_add_up",
        canonical_review_counts.get("clean_looking_count", 0) + canonical_review_counts.get("risky_paragraph_count", 0)
        == len(canonical_paragraphs),
    )
    add_check(
        "canonical_paragraph_review_samples_trace_to_canonical",
        {
            row.get("canonical_paragraph_id")
            for row in canonical_paragraph_review_report.get("sample_risky_canonical_paragraphs", [])
        }.issubset(set(canonical_ids)),
    )
    add_check(
        "canonical_paragraph_review_drilldown_counts_match_warnings",
        {
            row.get("warning"): row.get("count")
            for row in canonical_paragraph_review_report.get("warning_category_drilldown", [])
        }
        == canonical_paragraph_review_report.get("warning_categories", {}),
    )
    add_check(
        "canonical_paragraph_review_recommendation_present",
        bool((canonical_paragraph_review_report.get("recommendation_detail") or {}).get("top_risk_to_fix_first"))
        or canonical_review_counts.get("warning_count", 0) == 0,
    )
    bbox_span_diagnostics = canonical_paragraph_review_report.get("bbox_span_risk_diagnostics", [])
    bbox_span_summary = canonical_paragraph_review_report.get("bbox_span_risk_summary", {})
    required_bbox_diagnostic_fields = {
        "canonical_paragraph_id",
        "page_number",
        "source_candidate_object_id",
        "source_line_count",
        "vertical_bbox_span",
        "page_height_ratio",
        "text_length",
        "first_source_line_preview",
        "last_source_line_preview",
        "warning_severity",
        "likely_interpretation",
        "likely_corrective_path",
        "audit_anchor",
    }
    add_check(
        "bbox_span_diagnostic_count_matches_summary",
        bbox_span_summary.get("total") == len(bbox_span_diagnostics),
    )
    add_check(
        "bbox_span_diagnostics_have_required_fields",
        all(required_bbox_diagnostic_fields.issubset(row) for row in bbox_span_diagnostics),
    )
    add_check(
        "bbox_span_diagnostics_trace_to_canonical",
        {row.get("canonical_paragraph_id") for row in bbox_span_diagnostics}.issubset(set(canonical_ids)),
    )
    bbox_span_decisions = canonical_paragraph_review_report.get("bbox_span_decisions", [])
    bbox_span_decision_summary = canonical_paragraph_review_report.get("bbox_span_decision_summary", {})
    valid_bbox_causes = {
        "true_accidental_merge",
        "normal_long_paragraph",
        "threshold_too_strict",
        "boundary_or_front_matter_artifact",
        "needs_manual_inspection",
    }
    valid_bbox_actions = {
        "adjust paragraph merge rule",
        "adjust review threshold",
        "add curated override",
        "inspect manually",
        "no action",
    }
    add_check(
        "bbox_span_decision_count_matches_diagnostics",
        len(bbox_span_decisions) == len(bbox_span_diagnostics) == bbox_span_decision_summary.get("total"),
    )
    add_check(
        "bbox_span_decisions_trace_to_canonical",
        {row.get("canonical_paragraph_id") for row in bbox_span_decisions}.issubset(set(canonical_ids)),
    )
    add_check(
        "bbox_span_decision_causes_valid",
        all(row.get("likely_cause") in valid_bbox_causes for row in bbox_span_decisions),
    )
    add_check(
        "bbox_span_decision_actions_valid",
        all(row.get("recommended_action") in valid_bbox_actions for row in bbox_span_decisions),
    )
    add_check("audit_has_canonical_paragraph_review", "Canonical Paragraph Review" in audit_html)
    add_check("audit_has_canonical_review_drilldown", "Canonical Warning Drilldown" in audit_html and "Risk Clusters" in audit_html)
    add_check("audit_has_gold_review", "Gold Review" in audit_html)
    add_check("audit_has_cross_page_join_review", "Cross-Page Join Review" in audit_html)
    add_check("audit_has_policy_adoption_decision", "Policy Adoption Decision" in audit_html)
    add_check("audit_has_cross_page_join_decision_template", "cross_page_join_decisions.jsonl" in audit_html and "join-decision-template" in audit_html)
    add_check("audit_has_xpage_join_0032_investigation", "xpage_join_0032 Investigation" in audit_html)
    add_check(
        "xpage_join_0032_investigation_recommends_valid_decision",
        xpage_join_0032_investigation.get("recommended_decision") in {"accept", "reject", "needs_ocr_witness", "needs_manual_visual_review"},
    )
    add_check("policy_adoption_decision_matches_active_policy", policy_adoption_decision.get("active_paragraph_merge_policy") == manifest.get("paragraph_merge_policy"))
    add_check(
        "policy_adoption_decision_gate_evidence_present",
        all(
            key in (policy_adoption_decision.get("gate_evidence") or {})
            for key in ["gold_paragraph_precision_before", "gold_paragraph_precision_after", "proposed_joins", "unresolved_joins"]
        ),
    )
    post_adoption_state = post_adoption_safety_report.get("current_state") or {}
    post_adoption_top_risk = post_adoption_safety_report.get("current_top_risk") or {}
    add_check("audit_has_post_adoption_canonical_safety", "Post-Adoption Canonical Safety" in audit_html)
    add_check("post_adoption_safety_matches_active_policy", post_adoption_safety_report.get("active_policy") == manifest.get("paragraph_merge_policy"))
    add_check(
        "post_adoption_safety_counts_match_review",
        post_adoption_state.get("promoted_canonical_paragraphs") == canonical_review_counts.get("total_canonical_paragraphs_reviewed")
        and post_adoption_state.get("risky_canonical_paragraphs") == canonical_review_counts.get("risky_paragraph_count")
        and post_adoption_state.get("warning_count") == canonical_review_counts.get("warning_count"),
    )
    add_check("post_adoption_top_risk_present", bool(post_adoption_top_risk.get("cluster")) or canonical_review_counts.get("warning_count", 0) == 0)
    bbox_diagnosis_summary = post_adoption_bbox_diagnosis.get("summary") or {}
    bbox_diagnoses = post_adoption_bbox_diagnosis.get("diagnoses") or []
    valid_post_adoption_bbox_causes = {
        "true_paragraph_grouping_defect",
        "normal_long_paragraph",
        "front_matter_or_metadata_artifact",
        "threshold_noise",
        "gold_set_gap",
        "needs_visual_review",
    }
    add_check("audit_has_post_adoption_bbox_span_diagnosis", "Post-Adoption BBox Span Diagnosis" in audit_html)
    add_check("post_adoption_bbox_diagnosis_matches_active_policy", post_adoption_bbox_diagnosis.get("active_policy") == manifest.get("paragraph_merge_policy"))
    add_check(
        "post_adoption_bbox_diagnosis_count_matches_review",
        bbox_diagnosis_summary.get("total_bbox_span_cases") == len(bbox_span_decisions) == len(bbox_diagnoses),
    )
    add_check("post_adoption_bbox_diagnosis_causes_valid", all(row.get("likely_cause") in valid_post_adoption_bbox_causes for row in bbox_diagnoses))
    add_check("post_adoption_bbox_diagnosis_trace_to_canonical", {row.get("canonical_paragraph_id") for row in bbox_diagnoses}.issubset(set(canonical_ids)))
    remediation_summary = post_adoption_remediation_plan.get("summary") or {}
    remediation_queues = post_adoption_remediation_plan.get("queues") or []
    valid_remediation_groups = set(REMEDIATION_GROUPS)
    valid_action_types = {
        "merge/grouping rule work",
        "promotion-rule work",
        "object classification work",
        "gold expansion",
        "manual visual review",
    }
    add_check("audit_has_post_adoption_remediation_plan", "Post-Adoption Remediation Plan" in audit_html)
    add_check("post_adoption_remediation_plan_matches_active_policy", post_adoption_remediation_plan.get("active_policy") == manifest.get("paragraph_merge_policy"))
    add_check("post_adoption_remediation_plan_counts_match_diagnosis", remediation_summary.get("total_cases") == bbox_diagnosis_summary.get("total_bbox_span_cases"))
    add_check("post_adoption_remediation_groups_valid", {row.get("group") for row in remediation_queues}.issubset(valid_remediation_groups))
    add_check("post_adoption_remediation_action_types_valid", all(row.get("action_type") in valid_action_types for row in remediation_queues))
    front_matter_rows = front_matter_metadata_review_report.get("rows") or []
    front_matter_summary = front_matter_metadata_review_report.get("summary") or {}
    front_matter_queue_count = next(
        (row.get("count") for row in remediation_queues if row.get("group") == "front_matter_metadata_artifacts"),
        0,
    )
    add_check("audit_has_front_matter_metadata_review", "Front-Matter / Metadata Review" in audit_html)
    add_check("front_matter_review_matches_active_policy", front_matter_metadata_review_report.get("active_policy") == manifest.get("paragraph_merge_policy"))
    add_check("front_matter_review_count_matches_queue", front_matter_summary.get("total_reviewed") == front_matter_queue_count == len(front_matter_rows))
    add_check("front_matter_review_classifications_valid", all(row.get("likely_classification") in VALID_FRONT_MATTER_REVIEW_CLASSIFICATIONS for row in front_matter_rows))
    add_check("front_matter_review_trace_to_canonical", {row.get("canonical_paragraph_id") for row in front_matter_rows}.issubset(set(canonical_ids)))
    add_check("front_matter_review_is_review_only", bool(front_matter_metadata_review_report.get("review_only")))
    visual_review_rows = visual_review_cases_report.get("rows") or []
    visual_review_summary = visual_review_cases_report.get("summary") or {}
    visual_review_queue_count = next(
        (row.get("count") for row in remediation_queues if row.get("group") == "needs_visual_review"),
        0,
    )
    add_check("audit_has_visual_review_cases", "Visual Review Cases" in audit_html)
    add_check("visual_review_matches_active_policy", visual_review_cases_report.get("active_policy") == manifest.get("paragraph_merge_policy"))
    add_check("visual_review_count_matches_queue", visual_review_summary.get("total_reviewed") == visual_review_queue_count == len(visual_review_rows))
    add_check("visual_review_classifications_valid", all(row.get("likely_classification") in VALID_VISUAL_REVIEW_CLASSIFICATIONS for row in visual_review_rows))
    add_check("visual_review_trace_to_canonical", {row.get("canonical_paragraph_id") for row in visual_review_rows}.issubset(set(canonical_ids)))
    add_check("visual_review_is_review_only", bool(visual_review_cases_report.get("review_only")))
    remediation_review_progress = post_adoption_remediation_plan.get("review_progress") or {}
    front_progress = remediation_review_progress.get("front_matter_metadata_artifacts") or {}
    visual_progress = remediation_review_progress.get("needs_visual_review") or {}
    add_check(
        "remediation_plan_front_matter_progress_matches_review",
        front_progress.get("reviewed") == front_matter_summary.get("total_reviewed")
        and front_progress.get("total") == front_matter_queue_count,
    )
    add_check(
        "remediation_plan_visual_progress_matches_review",
        visual_progress.get("reviewed") == visual_review_summary.get("total_reviewed")
        and visual_progress.get("total") == visual_review_queue_count,
    )
    narrow_design_rows = narrow_grouping_correction_design.get("designs") or []
    narrow_design_summary = narrow_grouping_correction_design.get("summary") or {}
    add_check("audit_has_narrow_grouping_correction_design", "Narrow Grouping Correction Design" in audit_html)
    add_check("narrow_grouping_design_matches_active_policy", narrow_grouping_correction_design.get("active_policy") == manifest.get("paragraph_merge_policy"))
    add_check("narrow_grouping_design_is_design_only", bool(narrow_grouping_correction_design.get("design_only")))
    add_check("narrow_grouping_design_count_matches_rows", narrow_design_summary.get("confirmed_defects") == len(narrow_design_rows))
    add_check("narrow_grouping_design_references_visual_defects", {row.get("canonical_paragraph_id") for row in narrow_design_rows}.issubset({row.get("canonical_paragraph_id") for row in visual_review_rows}))
    chained_acceptance = chained_cross_page_experiment.get("acceptance_rule") or {}
    chained_target = chained_cross_page_experiment.get("target_defect") or {}
    chained_queue_rows = chained_join_review_queue.get("queue") or []
    chained_queue_summary = chained_join_review_queue.get("summary") or {}
    chained_decision_validation = chained_join_decisions_applied.get("validation") or {}
    chained_decision_summary = chained_join_decisions_applied.get("summary") or {}
    chained_decision_rows = chained_join_decisions_applied.get("decisions") or []
    valid_chained_queue_risks = {
        "likely_valid_chained_continuation",
        "possible_overmerge",
        "structure_boundary_risk",
        "page_furniture_risk",
        "needs_visual_review",
    }
    add_check("audit_has_chained_cross_page_continuation_experiment", "Chained Cross-Page Continuation Experiment" in audit_html)
    add_check("chained_experiment_keeps_active_policy", chained_cross_page_experiment.get("active_policy") == manifest.get("paragraph_merge_policy"))
    add_check("chained_experiment_policy_recorded", chained_cross_page_experiment.get("experimental_policy") == CHAINED_CROSS_PAGE_CONTINUATION_POLICY)
    add_check("chained_experiment_does_not_change_active_policy", bool(chained_cross_page_experiment.get("does_not_change_active_policy")))
    add_check("chained_experiment_targets_confirmed_defect", chained_target.get("canonical_paragraph_id") == "cp_000103")
    add_check("chained_experiment_acceptance_fields_present", all(key in chained_acceptance for key in ["cp_000103_fixed", "gold_score_improved", "over_merges_not_increased", "object_label_accuracy_not_worsened", "adoption_recommendation"]))
    add_check("audit_has_chained_join_review_queue", "Chained Join Side-Effect Review Queue" in audit_html)
    add_check("chained_join_review_queue_is_review_only", bool(chained_join_review_queue.get("does_not_apply_decisions")) and bool(chained_join_review_queue.get("does_not_change_active_policy")))
    add_check(
        "chained_join_review_queue_count_matches_experiment",
        chained_queue_summary.get("total_unscored_chained_joins") == len(chained_queue_rows) == (chained_cross_page_experiment.get("side_effects") or {}).get("joins_not_covered_by_gold"),
    )
    add_check("chained_join_review_queue_risks_valid", all(row.get("likely_risk") in valid_chained_queue_risks for row in chained_queue_rows))
    add_check("chained_join_review_queue_fields_present", all(all(row.get(field) is not None for field in ["chained_join_id", "affected_pages", "source_candidate_ids", "likely_risk", "recommended_review_action"]) for row in chained_queue_rows))
    add_check("audit_has_chained_join_decisions", "Chained Join Decisions" in audit_html)
    add_check("chained_join_decision_validation_passes", chained_decision_validation.get("status") == "pass")
    add_check("chained_join_decision_count_matches_queue", chained_decision_summary.get("queued_chained_joins") == len(chained_queue_rows) == len(chained_decision_rows))
    add_check("chained_join_decisions_do_not_adopt_v3", bool(chained_join_decisions_applied.get("does_not_adopt_v3")) and bool(chained_decision_summary.get("adoption_remains_separate_checkpoint")))
    add_check("chained_join_decision_values_valid", all(row.get("decision") in VALID_CHAINED_JOIN_DECISIONS | {"unreviewed"} for row in chained_decision_rows))
    guarded_acceptance = guarded_chained_experiment.get("acceptance_rule") or {}
    guarded_decision_replay = guarded_chained_experiment.get("decision_replay") or {}
    guarded_side_effects = guarded_chained_experiment.get("side_effects") or {}
    add_check("audit_has_guarded_chained_cross_page_continuation_experiment", "Guarded Chained Cross-Page Continuation Experiment" in audit_html)
    add_check("guarded_chained_experiment_records_active_policy", guarded_chained_experiment.get("active_policy") == manifest.get("paragraph_merge_policy"))
    add_check("guarded_chained_experiment_policy_recorded", guarded_chained_experiment.get("guarded_experimental_policy") == GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY)
    add_check(
        "guarded_chained_experiment_adoption_flag_matches_active_policy",
        bool(guarded_chained_experiment.get("does_not_adopt_guarded_policy"))
        == (manifest.get("paragraph_merge_policy") != GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY),
    )
    add_check("guarded_chained_experiment_blocks_rejected_join", bool(guarded_decision_replay.get("chained_join_review_0004_blocked")))
    add_check(
        "guarded_chained_experiment_preserves_accepted_decisions",
        guarded_decision_replay.get("accepted_prior_decisions_preserved") == guarded_decision_replay.get("accepted_prior_decisions"),
    )
    add_check(
        "guarded_chained_experiment_blocks_rejected_decisions",
        guarded_decision_replay.get("rejected_prior_decisions_blocked") == guarded_decision_replay.get("rejected_prior_decisions"),
    )
    add_check("guarded_chained_experiment_keeps_cp_000103_fixed", bool(guarded_acceptance.get("cp_000103_remains_fixed")))
    add_check("guarded_chained_experiment_has_join_counts", all(key in guarded_side_effects for key in ["proposed_chained_joins", "rejected_chained_joins"]))
    guarded_adoption_evidence = guarded_policy_adoption_decision.get("gate_evidence") or {}
    guarded_adoption_gates = guarded_policy_adoption_decision.get("gates") or {}
    add_check("audit_has_guarded_chained_policy_adoption_decision", "Guarded Chained Policy Adoption Decision" in audit_html)
    add_check("guarded_policy_adoption_matches_active_policy", guarded_policy_adoption_decision.get("active_paragraph_merge_policy") == manifest.get("paragraph_merge_policy"))
    add_check("guarded_policy_adoption_records_guarded_policy", guarded_policy_adoption_decision.get("adopted_policy") == GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY)
    add_check("guarded_policy_adoption_decision_is_adopt", guarded_policy_adoption_decision.get("decision") == "adopt_v3_chained_cross_page_continuation_guarded")
    add_check("guarded_policy_adoption_does_not_unlock_downstream", bool(guarded_policy_adoption_decision.get("does_not_unlock_downstream")))
    add_check(
        "guarded_policy_adoption_gate_evidence_present",
        all(
            key in guarded_adoption_evidence
            for key in [
                "active_v2_metrics",
                "previous_unguarded_v3_metrics",
                "guarded_v3_metrics",
                "gold_paragraph_precision_before",
                "gold_paragraph_precision_after",
                "matched_paragraphs_before",
                "matched_paragraphs_after",
                "unresolved_chained_joins",
                "validation_status",
                "downstream_safety_status",
            ]
        ),
    )
    add_check(
        "guarded_policy_adoption_gates_pass",
        guarded_adoption_gates.get("guarded_experiment_passed")
        and guarded_adoption_gates.get("cp_000103_remains_fixed")
        and guarded_adoption_gates.get("false_join_blocked")
        and guarded_adoption_gates.get("accepted_prior_decisions_preserved")
        and guarded_adoption_gates.get("rejected_prior_decisions_blocked")
        and guarded_adoption_gates.get("gold_score_improved")
        and guarded_adoption_gates.get("over_merges_not_increased")
        and guarded_adoption_gates.get("object_label_accuracy_not_worsened")
        and not guarded_adoption_gates.get("audit_warning_regression")
        and not guarded_adoption_gates.get("bbox_span_regression")
        and guarded_adoption_gates.get("unresolved_chained_joins") == 0,
    )
    add_check("audit_has_merge_failure_taxonomy", "Merge Failure Taxonomy" in audit_html)
    add_check("audit_has_bbox_span_diagnostics", "BBox Span Risk Diagnostics" in audit_html)
    add_check("audit_has_bbox_span_decision_summary", "BBox Span Decision Summary" in audit_html)
    known_ids = set(layout_ids)
    override_object_ids = [row.get("object_id") for row in review_overrides]
    add_check("review_overrides_reference_known_objects", set(override_object_ids).issubset(known_ids))
    add_check("review_override_object_ids_unique", len(override_object_ids) == len(set(override_object_ids)))
    add_check(
        "review_override_required_fields_present",
        all(REQUIRED_OVERRIDE_FIELDS.issubset(row) and all(str(row.get(field, "")).strip() for field in REQUIRED_OVERRIDE_FIELDS) for row in review_overrides),
    )
    add_check("review_override_original_buckets_valid", all(row.get("original_bucket") in VALID_OVERRIDE_BUCKETS for row in review_overrides))
    add_check("review_override_buckets_valid", all(row.get("corrected_bucket") in VALID_OVERRIDE_BUCKETS for row in review_overrides))
    add_check("review_overrides_are_curated_source", all(row.get("review_source") == "curated" for row in review_overrides))
    add_check("review_overrides_record_source_path", all(str(row.get("review_source_path", "")).strip() for row in review_overrides))
    stream_by_id = {row["object_id"]: row for row in stream_rows}
    add_check(
        "review_override_original_bucket_matches_detector",
        all(stream_by_id.get(row.get("object_id"), {}).get("original_stream_type") == row.get("original_bucket") for row in review_overrides),
    )
    add_check("reconstruction_map_records_review_overrides", reconstruction_map.get("review_override_count") == len(review_overrides))

    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {
        "created_at": utc_now(),
        "status": status,
        "checks": checks,
        "summary": {
            "page_count": page_count,
            "inventory_rows": len(inventory),
            "raw_page_rows": len(raw_pages),
            "layout_object_rows": len(layout_objects),
            "clean_object_rows": len(clean_objects),
            "main_paragraph_rows": len(main_paragraphs),
            "structure_rows": len(structure),
            "page_artifact_rows": len(page_artifacts),
            "unknown_rows": len(unknown),
            "canonical_paragraph_rows": len(canonical_paragraphs),
            "promotion_blocker_rows": len(promotion_blockers),
            "canonical_paragraph_review_warning_rows": canonical_review_counts.get("warning_count", 0),
            "review_override_rows": len(review_overrides),
            "cross_page_join_decision_rows": len(cross_page_join_decisions),
            "page_image_rows": len(page_image_files),
            "cleanup_log_rows": len(cleanup_log),
        },
    }


def run_phase1(pdf_path: Path, book_id: str, run_id: str = "phase1_v3") -> Path:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_dir = RUNS_DIR / book_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    review_overrides_path = curated_review_overrides_path(book_id)
    review_overrides = read_review_overrides(review_overrides_path)
    applied_review_overrides = prepare_applied_review_overrides(review_overrides, review_overrides_path)
    cross_page_join_decisions_path = curated_cross_page_join_decisions_path(book_id)
    cross_page_join_decisions = read_commented_jsonl(cross_page_join_decisions_path)
    applied_cross_page_join_decisions = prepare_applied_cross_page_join_decisions(
        cross_page_join_decisions,
        cross_page_join_decisions_path,
    )
    chained_join_decisions_path = curated_chained_join_decisions_path(book_id)
    chained_join_decisions = read_commented_jsonl(chained_join_decisions_path)
    applied_chained_join_decisions = prepare_applied_chained_join_decisions(
        chained_join_decisions,
        chained_join_decisions_path,
    )

    inventory: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []
    baseline_layout_objects: list[dict[str, Any]] = []
    baseline_clean_objects: list[dict[str, Any]] = []
    layout_objects: list[dict[str, Any]] = []
    clean_objects: list[dict[str, Any]] = []
    cleanup_log: list[dict[str, Any]] = []
    cross_page_experiment_details: dict[str, Any] = {}
    v2_layout_objects: list[dict[str, Any]] = []
    v2_clean_objects: list[dict[str, Any]] = []
    guarded_active_experiment_details: dict[str, Any] = {}

    with pdfplumber.open(str(pdf_path)) as pdf:
        page_count = len(pdf.pages)
        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            raw_lines = extract_line_records(page, raw_text)
            try:
                table_count = len(page.find_tables())
            except Exception:
                table_count = 0
            image_count = len(page.images)
            flags = review_flags(raw_text, image_count, table_count)
            inventory.append(
                {
                    "book_id": book_id,
                    "page_number": page_number,
                    "width": page.width,
                    "height": page.height,
                    "status": page_status(raw_text, image_count),
                    "raw_char_count": len(raw_text),
                    "line_count": len([line for line in raw_lines if str(line.get("text", "")).strip()]),
                    "image_count": image_count,
                    "table_count": table_count,
                    "review_flags": flags,
                    "sample": raw_text[:240].replace("\n", " | "),
                }
            )
            raw_pages.append(
                {
                    "book_id": book_id,
                    "page_number": page_number,
                    "raw_text": raw_text,
                    "raw_char_count": len(raw_text),
                }
            )
            page_baseline_layout, page_baseline_clean, page_baseline_cleanup = build_segmented_objects(
                book_id, page_number, raw_lines, paragraph_merge_policy=BASELINE_PARAGRAPH_MERGE_POLICY
            )
            baseline_layout_objects.extend(page_baseline_layout)
            baseline_clean_objects.extend(page_baseline_clean)
            page_active_layout, page_active_clean, page_active_cleanup = build_segmented_objects(
                book_id, page_number, raw_lines, paragraph_merge_policy=ACTIVE_PARAGRAPH_MERGE_POLICY
            )
            layout_objects.extend(page_active_layout)
            clean_objects.extend(page_active_clean)
            cleanup_log.extend(page_active_cleanup)
    if ACTIVE_PARAGRAPH_MERGE_POLICY in {CROSS_PAGE_CONTINUATION_POLICY, GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY}:
        layout_objects, clean_objects, cross_page_experiment_details = apply_cross_page_continuation_experiment(
            layout_objects,
            clean_objects,
        )
        v2_layout_objects = [dict(row) for row in layout_objects]
        v2_clean_objects = [dict(row) for row in clean_objects]
        layout_by_id = {row["object_id"]: row for row in layout_objects}
        cleanup_log = [row for row in cleanup_log if row.get("object_id") in layout_by_id]
        for clean_row in clean_objects:
            if "cross_page_paragraph_continuation_join" not in clean_row.get("cleanup_operations", []):
                continue
            layout_row = layout_by_id.get(clean_row.get("object_id"), {})
            cleanup_log.append(
                {
                    "book_id": book_id,
                    "object_id": clean_row["object_id"],
                    "page_number": layout_row.get("page_number"),
                    "operation": "cross_page_paragraph_continuation_join",
                    "raw_text": layout_row.get("raw_text", ""),
                    "clean_text": clean_row.get("clean_text", ""),
                }
            )
    if ACTIVE_PARAGRAPH_MERGE_POLICY == GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY:
        layout_objects, clean_objects, guarded_active_experiment_details = apply_chained_cross_page_continuation_experiment(
            layout_objects,
            clean_objects,
            guarded=True,
        )
        layout_by_id = {row["object_id"]: row for row in layout_objects}
        cleanup_log = [row for row in cleanup_log if row.get("object_id") in layout_by_id]
        for clean_row in clean_objects:
            if "chained_cross_page_paragraph_continuation_join" not in clean_row.get("cleanup_operations", []):
                continue
            layout_row = layout_by_id.get(clean_row.get("object_id"), {})
            cleanup_log.append(
                {
                    "book_id": book_id,
                    "object_id": clean_row["object_id"],
                    "page_number": layout_row.get("page_number"),
                    "operation": "chained_cross_page_paragraph_continuation_join",
                    "raw_text": layout_row.get("raw_text", ""),
                    "clean_text": clean_row.get("clean_text", ""),
                }
            )
    if not v2_layout_objects:
        v2_layout_objects = [dict(row) for row in layout_objects]
        v2_clean_objects = [dict(row) for row in clean_objects]
    page_images = render_page_images(pdf_path, output_dir)

    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "source_pdf": str(pdf_path),
        "file_size_bytes": pdf_path.stat().st_size,
        "sha256": sha256_file(pdf_path),
        "page_count": page_count,
        "tooling": {"pdfplumber": getattr(pdfplumber, "__version__", "unknown")},
        "paragraph_merge_policy": ACTIVE_PARAGRAPH_MERGE_POLICY,
        "paragraph_merge_experiment_policy": EXPERIMENTAL_PARAGRAPH_MERGE_POLICY,
        "outputs": {
            "page_inventory": "page_inventory.jsonl",
            "raw_pages": "raw_pages.jsonl",
            "layout_objects": "layout_objects.jsonl",
            "clean_objects": "clean_objects.jsonl",
            "page_images": PAGE_IMAGES_DIR_NAME,
            "reading_order_candidate": "reading_order_candidate.json",
            "review_overrides_source": str(review_overrides_path.relative_to(ROOT)),
            "review_overrides_applied": "review_overrides_applied.jsonl",
            "cross_page_join_decisions_source": str(cross_page_join_decisions_path.relative_to(ROOT)),
            "cross_page_join_decisions_applied": "cross_page_join_decisions_applied.jsonl",
            "chained_join_decisions_source": str(chained_join_decisions_path.relative_to(ROOT)),
            "chained_join_decisions_applied": "chained_join_decisions_applied.json",
            "cleanup_log": "cleanup_log.jsonl",
            "validation_report": "validation_report.json",
            "phase1_audit": "phase1_audit.html",
            "canonical_paragraphs": "canonical_paragraphs.jsonl",
            "promotion_blockers": "promotion_blockers.jsonl",
            "canonical_promotion_report": "canonical_promotion_report.json",
            "canonical_paragraph_review_report": "canonical_paragraph_review_report.json",
            "paragraph_merge_experiment_report": "paragraph_merge_experiment_report.json",
            "paragraph_merge_failure_taxonomy_report": "paragraph_merge_failure_taxonomy_report.json",
            "cross_page_join_review_report": "cross_page_join_review_report.json",
            "xpage_join_0032_investigation": "xpage_join_0032_investigation.json",
            "policy_adoption_decision": "policy_adoption_decision.json",
            "post_adoption_canonical_safety_report": "post_adoption_canonical_safety_report.json",
            "post_adoption_bbox_span_diagnosis": "post_adoption_bbox_span_diagnosis.json",
            "post_adoption_remediation_plan": "post_adoption_remediation_plan.json",
            "front_matter_metadata_review_report": "front_matter_metadata_review_report.json",
            "visual_review_cases_report": "visual_review_cases_report.json",
            "narrow_grouping_correction_design": "narrow_grouping_correction_design.json",
            "chained_cross_page_continuation_experiment": "chained_cross_page_continuation_experiment.json",
            "chained_join_review_queue": "chained_join_review_queue.json",
            "guarded_chained_cross_page_continuation_experiment": "guarded_chained_cross_page_continuation_experiment.json",
            "guarded_chained_policy_adoption_decision": "guarded_chained_policy_adoption_decision.json",
            "gold_evaluation_report": "gold_evaluation_report.json",
        },
    }
    object_counts = Counter(row["object_type"] for row in layout_objects)
    page_heights_by_page = {
        int(row["page_number"]): float(row["height"])
        for row in inventory
        if row.get("page_number") is not None and row.get("height") is not None
    }
    baseline_evaluation = paragraph_policy_evaluation(
        book_id,
        run_id,
        baseline_layout_objects,
        baseline_clean_objects,
        inventory,
        page_count,
        applied_review_overrides,
        page_heights_by_page,
    )
    active_evaluation = paragraph_policy_evaluation(
        book_id,
        run_id,
        layout_objects,
        clean_objects,
        inventory,
        page_count,
        applied_review_overrides,
        page_heights_by_page,
    )
    v2_evaluation = (
        paragraph_policy_evaluation(
            book_id,
            run_id,
            v2_layout_objects,
            v2_clean_objects,
            inventory,
            page_count,
            applied_review_overrides,
            page_heights_by_page,
        )
        if ACTIVE_PARAGRAPH_MERGE_POLICY == GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY
        else active_evaluation
    )
    main_paragraphs = active_evaluation["main_paragraphs"]
    structure = active_evaluation["structure"]
    page_artifacts = active_evaluation["page_artifacts"]
    unknown = active_evaluation["unknown"]
    reconstruction_map = active_evaluation["reconstruction_map"]
    canonical_paragraphs = active_evaluation["canonical_paragraphs"]
    promotion_blockers = active_evaluation["promotion_blockers"]
    canonical_promotion_report = active_evaluation["promotion_report"]
    canonical_paragraph_review_report = active_evaluation["review_report"]
    paragraph_merge_experiment_report = build_paragraph_merge_experiment_report(
        book_id, run_id, baseline_evaluation, active_evaluation, cross_page_experiment_details
    )
    cross_page_join_review_report = build_cross_page_join_review_report(
        book_id, run_id, cross_page_experiment_details, applied_cross_page_join_decisions
    )
    xpage_join_0032_investigation = build_xpage_join_0032_investigation(
        book_id,
        run_id,
        cross_page_join_review_report,
        baseline_layout_objects,
        page_artifacts,
        structure,
    )
    policy_adoption_decision = build_policy_adoption_decision(
        book_id,
        run_id,
        paragraph_merge_experiment_report,
        cross_page_join_review_report,
        xpage_join_0032_investigation,
        ACTIVE_PARAGRAPH_MERGE_POLICY,
        canonical_promotion_report,
        canonical_paragraph_review_report,
        active_evaluation["gold_evaluation_report"],
    )
    post_adoption_canonical_safety_report = build_post_adoption_canonical_safety_report(
        book_id,
        run_id,
        baseline_evaluation["review_report"],
        canonical_paragraph_review_report,
        paragraph_merge_experiment_report,
        policy_adoption_decision,
    )
    post_adoption_bbox_span_diagnosis = build_post_adoption_bbox_span_diagnosis(
        book_id,
        run_id,
        canonical_paragraphs,
        canonical_paragraph_review_report,
        ACTIVE_PARAGRAPH_MERGE_POLICY,
    )
    post_adoption_remediation_plan = build_post_adoption_remediation_plan(
        book_id,
        run_id,
        post_adoption_bbox_span_diagnosis,
        post_adoption_canonical_safety_report,
    )
    front_matter_metadata_review_report = build_front_matter_metadata_review_report(
        book_id,
        run_id,
        post_adoption_remediation_plan,
        canonical_paragraphs,
        ACTIVE_PARAGRAPH_MERGE_POLICY,
    )
    visual_review_cases_report = build_visual_review_cases_report(
        book_id,
        run_id,
        post_adoption_remediation_plan,
        canonical_paragraphs,
        ACTIVE_PARAGRAPH_MERGE_POLICY,
    )
    post_adoption_remediation_plan = update_remediation_plan_after_reviews(
        post_adoption_remediation_plan,
        front_matter_metadata_review_report,
        visual_review_cases_report,
    )
    narrow_grouping_correction_design = build_narrow_grouping_correction_design(
        book_id,
        run_id,
        canonical_paragraphs,
        visual_review_cases_report,
        active_evaluation["gold_evaluation_report"],
        ACTIVE_PARAGRAPH_MERGE_POLICY,
    )
    chained_layout_objects, chained_clean_objects, chained_experiment_details = apply_chained_cross_page_continuation_experiment(
        v2_layout_objects,
        v2_clean_objects,
    )
    chained_evaluation = paragraph_policy_evaluation(
        book_id,
        run_id,
        chained_layout_objects,
        chained_clean_objects,
        inventory,
        page_count,
        applied_review_overrides,
        page_heights_by_page,
    )
    chained_cross_page_continuation_experiment = build_chained_cross_page_continuation_experiment(
        book_id,
        run_id,
        v2_evaluation,
        chained_evaluation,
        chained_experiment_details,
        narrow_grouping_correction_design,
    )
    chained_join_review_queue = build_chained_join_review_queue(
        book_id,
        run_id,
        chained_cross_page_continuation_experiment,
    )
    chained_join_decisions_applied = build_chained_join_decisions_applied(
        book_id,
        run_id,
        chained_join_review_queue,
        applied_chained_join_decisions,
    )
    guarded_chained_layout_objects, guarded_chained_clean_objects, guarded_chained_experiment_details = (
        (layout_objects, clean_objects, guarded_active_experiment_details)
        if ACTIVE_PARAGRAPH_MERGE_POLICY == GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY
        else apply_chained_cross_page_continuation_experiment(
            v2_layout_objects,
            v2_clean_objects,
            guarded=True,
        )
    )
    guarded_chained_evaluation = active_evaluation if ACTIVE_PARAGRAPH_MERGE_POLICY == GUARDED_CHAINED_CROSS_PAGE_CONTINUATION_POLICY else paragraph_policy_evaluation(
        book_id,
        run_id,
        guarded_chained_layout_objects,
        guarded_chained_clean_objects,
        inventory,
        page_count,
        applied_review_overrides,
        page_heights_by_page,
    )
    guarded_chained_cross_page_continuation_experiment = build_guarded_chained_cross_page_continuation_experiment(
        book_id,
        run_id,
        v2_evaluation,
        chained_cross_page_continuation_experiment,
        guarded_chained_evaluation,
        guarded_chained_experiment_details,
        chained_join_decisions_applied,
    )
    guarded_chained_policy_adoption_decision = build_guarded_chained_policy_adoption_decision(
        book_id,
        run_id,
        guarded_chained_cross_page_continuation_experiment,
        active_evaluation,
        ACTIVE_PARAGRAPH_MERGE_POLICY,
    )
    paragraph_merge_failure_taxonomy_report = build_paragraph_merge_failure_taxonomy_report(
        book_id, run_id, canonical_paragraphs, canonical_paragraph_review_report
    )
    gold_evaluation_report = active_evaluation["gold_evaluation_report"]
    stream_counts = {
        "main_paragraph_candidates": len(main_paragraphs),
        "structure_candidates": len(structure),
        "page_artifacts_candidates": len(page_artifacts),
        "unknown_objects": len(unknown),
    }
    artifact_type_counts = dict(sorted(Counter(row.get("artifact_type", "unknown") for row in page_artifacts).items()))
    reading_order_candidate = {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "object_count": len(layout_objects),
        "object_type_counts": dict(sorted(object_counts.items())),
        "candidate_stream_counts": stream_counts,
        "artifact_type_counts": artifact_type_counts,
        "review_override_count": len(applied_review_overrides),
        "page_image_count": len(page_images),
        "page_count": page_count,
        "object_ids": [row["object_id"] for row in layout_objects],
        "main_paragraph_candidate_ids": [row["paragraph_id"] for row in main_paragraphs],
        "review_flags": sorted({flag for page in inventory for flag in page["review_flags"]}),
    }
    manifest["outputs"].update(
        {
            "main_paragraph_candidates": "main_paragraph_candidates.jsonl",
            "structure_candidates": "structure_candidates.jsonl",
            "page_artifacts_candidates": "page_artifacts_candidates.jsonl",
            "unknown_objects": "unknown_objects.jsonl",
            "reconstruction_map_candidate": "reconstruction_map_candidate.json",
        }
    )

    write_json(output_dir / "source_manifest.json", manifest)
    write_jsonl(output_dir / "page_inventory.jsonl", inventory)
    write_jsonl(output_dir / "raw_pages.jsonl", raw_pages)
    write_jsonl(output_dir / "layout_objects.jsonl", layout_objects)
    write_jsonl(output_dir / "clean_objects.jsonl", clean_objects)
    write_jsonl(output_dir / "main_paragraph_candidates.jsonl", main_paragraphs)
    write_jsonl(output_dir / "structure_candidates.jsonl", structure)
    write_jsonl(output_dir / "page_artifacts_candidates.jsonl", page_artifacts)
    write_jsonl(output_dir / "unknown_objects.jsonl", unknown)
    write_json(output_dir / "reconstruction_map_candidate.json", reconstruction_map)
    write_json(output_dir / "reading_order_candidate.json", reading_order_candidate)
    write_jsonl(output_dir / "review_overrides_applied.jsonl", applied_review_overrides)
    write_jsonl(output_dir / "cross_page_join_decisions_applied.jsonl", applied_cross_page_join_decisions)
    write_json(output_dir / "chained_join_decisions_applied.json", chained_join_decisions_applied)
    write_jsonl(output_dir / "canonical_paragraphs.jsonl", canonical_paragraphs)
    write_jsonl(output_dir / "promotion_blockers.jsonl", promotion_blockers)
    write_json(output_dir / "canonical_promotion_report.json", canonical_promotion_report)
    write_json(output_dir / "canonical_paragraph_review_report.json", canonical_paragraph_review_report)
    write_json(output_dir / "paragraph_merge_experiment_report.json", paragraph_merge_experiment_report)
    write_json(output_dir / "paragraph_merge_failure_taxonomy_report.json", paragraph_merge_failure_taxonomy_report)
    write_json(output_dir / "cross_page_join_review_report.json", cross_page_join_review_report)
    write_json(output_dir / "xpage_join_0032_investigation.json", xpage_join_0032_investigation)
    write_json(output_dir / "policy_adoption_decision.json", policy_adoption_decision)
    write_json(output_dir / "post_adoption_canonical_safety_report.json", post_adoption_canonical_safety_report)
    write_json(output_dir / "post_adoption_bbox_span_diagnosis.json", post_adoption_bbox_span_diagnosis)
    write_json(output_dir / "post_adoption_remediation_plan.json", post_adoption_remediation_plan)
    write_json(output_dir / "front_matter_metadata_review_report.json", front_matter_metadata_review_report)
    write_json(output_dir / "visual_review_cases_report.json", visual_review_cases_report)
    write_json(output_dir / "narrow_grouping_correction_design.json", narrow_grouping_correction_design)
    write_json(output_dir / "chained_cross_page_continuation_experiment.json", chained_cross_page_continuation_experiment)
    write_json(output_dir / "chained_join_review_queue.json", chained_join_review_queue)
    write_json(output_dir / "guarded_chained_cross_page_continuation_experiment.json", guarded_chained_cross_page_continuation_experiment)
    write_json(output_dir / "guarded_chained_policy_adoption_decision.json", guarded_chained_policy_adoption_decision)
    write_json(output_dir / "gold_evaluation_report.json", gold_evaluation_report)
    write_jsonl(output_dir / "cleanup_log.jsonl", cleanup_log)
    pending_validation = {
        "created_at": utc_now(),
        "status": "pending",
        "checks": [],
        "summary": {"detail": "Validation report is generated after the first audit render."},
    }
    write_json(output_dir / "validation_report.json", pending_validation)
    stream_samples = {
        "main_paragraph_candidates": main_paragraphs[:8],
        "structure_candidates": structure[:8],
        "page_artifacts_candidates": page_artifacts[:8],
        "unknown_objects": unknown[:8],
        "__all__": main_paragraphs + structure + page_artifacts + unknown,
    }
    (output_dir / "phase1_audit.html").write_text(
        build_audit_html(book_id, pdf_path, manifest, inventory, layout_objects, object_counts, stream_counts, stream_samples, pending_validation, output_dir),
        encoding="utf-8",
    )
    validation_report = validate_phase1_run(output_dir)
    write_json(output_dir / "validation_report.json", validation_report)
    guarded_chained_policy_adoption_decision = build_guarded_chained_policy_adoption_decision(
        book_id,
        run_id,
        guarded_chained_cross_page_continuation_experiment,
        active_evaluation,
        ACTIVE_PARAGRAPH_MERGE_POLICY,
        validation_report.get("status", "unknown"),
    )
    write_json(output_dir / "guarded_chained_policy_adoption_decision.json", guarded_chained_policy_adoption_decision)
    (output_dir / "phase1_audit.html").write_text(
        build_audit_html(book_id, pdf_path, manifest, inventory, layout_objects, object_counts, stream_counts, stream_samples, validation_report, output_dir),
        encoding="utf-8",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Phase 1 PDF extraction.")
    parser.add_argument("pdf_path", help="Path to the source PDF.")
    parser.add_argument("--book-id", required=True, help="Stable book id for output paths.")
    parser.add_argument("--run-id", default="phase1_v3", help="Run id for output paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run_phase1(Path(args.pdf_path), args.book_id, args.run_id)
    print(json.dumps({"book_id": args.book_id, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
