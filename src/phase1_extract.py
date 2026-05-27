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


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "runs"
CID_PATTERN = re.compile(r"\(cid:\d+\)")
ROMAN_PATTERN = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
TEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
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
    "review_overrides.jsonl",
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
REQUIRED_OVERRIDE_FIELDS = {
    "object_id",
    "original_bucket",
    "corrected_bucket",
    "reason",
    "reviewer",
    "date",
    "evidence_reference",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_review_overrides_template(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "# Add one JSON object per line. Lines starting with # are ignored.\n"
        "# Required: object_id, original_bucket, corrected_bucket, reason, reviewer, date, evidence_reference.\n"
        "# Optional: corrected_subtype, confidence.\n",
        encoding="utf-8",
    )


def read_review_overrides(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        row["_line_number"] = line_number
        rows.append(row)
    return rows


def clean_line(raw: str) -> tuple[str, list[str]]:
    operations: list[str] = []
    text = raw.rstrip()
    if text != raw:
        operations.append("rstrip")
    collapsed = re.sub(r"\s+", " ", text).strip()
    if collapsed != text:
        operations.append("collapse_whitespace")
    text = collapsed
    if CID_PATTERN.search(text):
        operations.append("flag_cid_noise")
    return text, operations


def normalized_object_text(text: str) -> str:
    return " ".join(TEXT_TOKEN_PATTERN.findall(text.lower()))


def is_page_number_token(value: str) -> bool:
    return value.isdigit() or bool(ROMAN_PATTERN.fullmatch(value))


def normalized_furniture_text(text: str) -> str:
    tokens = normalized_object_text(text).split()
    if len(tokens) > 1 and is_page_number_token(tokens[0]):
        tokens = tokens[1:]
    if len(tokens) > 1 and is_page_number_token(tokens[-1]):
        tokens = tokens[:-1]
    return " ".join(tokens)


def classify_line(line: str, page_number: int) -> tuple[str, float, list[str]]:
    clean = line.strip()
    reasons: list[str] = []
    if not clean:
        return "blank", 1.0, reasons
    if CID_PATTERN.search(clean):
        reasons.append("cid_noise")
        return "unknown", 0.4, reasons
    if clean.isdigit() or ROMAN_PATTERN.fullmatch(clean):
        return "page_artifact", 0.7, ["possible_page_number"]
    upper_ratio = sum(1 for char in clean if char.isupper()) / max(1, sum(1 for char in clean if char.isalpha()))
    word_count = len(clean.split())
    if word_count <= 8 and upper_ratio >= 0.75:
        return "heading", 0.75, ["short_uppercase_line"]
    if page_number <= 12 and word_count <= 12 and upper_ratio >= 0.5:
        return "heading", 0.55, ["front_matter_heading_candidate"]
    return "paragraph_line", 0.55, reasons


def join_paragraph_lines(lines: list[str]) -> tuple[str, list[str]]:
    operations = ["merge_paragraph_lines"] if len(lines) > 1 else []
    cleaned_lines = [clean_line(line)[0] for line in lines]
    if not cleaned_lines:
        return "", operations
    text = cleaned_lines[0]
    for next_line in cleaned_lines[1:]:
        if text.endswith("-") and next_line[:1].islower():
            text = text[:-1] + next_line
            operations.append("join_hyphenated_line_break")
        else:
            text = f"{text} {next_line}"
    if CID_PATTERN.search(text):
        operations.append("flag_cid_noise")
    return text.strip(), list(dict.fromkeys(operations))


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
                    "top": value.get("top"),
                    "bottom": value.get("bottom"),
                }
            )
        else:
            records.append({"line_index": line_index, "text": str(value), "x0": None, "top": None, "bottom": None})
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


def build_segmented_objects(book_id: str, page_number: int, raw_lines: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
        tops = [float(item["top"]) for item in paragraph_buffer if item.get("top") is not None]
        bottoms = [float(item["bottom"]) for item in paragraph_buffer if item.get("bottom") is not None]
        bbox = {
            "x0": min(xs) if xs else None,
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
                "classification_reasons": ["merged_consecutive_paragraph_lines"],
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
            paragraph_buffer.append(
                {
                    "line_id": line_id,
                    "line_index": line_index,
                    "raw_text": raw_line,
                    "x0": record.get("x0"),
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
                "top": record.get("top"),
                "bottom": record.get("bottom"),
                "bbox": {
                    "x0": record.get("x0"),
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
            "source_object_ids": [object_id],
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

    def esc(value: Any) -> str:
        return html.escape(str(value))

    candidate_rows = [row for rows in stream_samples.values() for row in rows]
    # The full candidate rows are provided through stream_samples["__all__"] when available.
    candidate_rows = stream_samples.get("__all__", candidate_rows)
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
        for key in ["x0", "top", "bottom"]:
            raw_value = value.get(key)
            if raw_value is not None:
                parts.append(f"{key}={float(raw_value):.1f}")
        return ", ".join(parts) or "-"

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
        page_objects = objects_by_page.get(page_number, [])
        bucket_counts: Counter[str] = Counter()
        object_cards = []
        for obj in page_objects:
            candidate = candidate_by_object_id.get(obj["object_id"])
            label = bucket_label(candidate)
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
            object_cards.append(
                f"""<article class="object-card" data-bucket="{esc(label)}" data-subtype="{esc(subtype)}" data-confidence="{esc(confidence)}" data-warnings="{esc(' '.join(warnings))}" data-zone="{esc(zone)}" data-page="{page_number}">
                  <header>
                    <code>{esc(obj.get('object_id'))}</code>
                    <span class="{esc(bucket_class(label))}">{esc(label)}</span>
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
                        <dt>Confidence</dt><dd><code>{esc(confidence)}</code></dd>
                        <dt>Subtype</dt><dd><code>{esc(subtype)}</code></dd>
                        <dt>Review override</dt><dd><code>{esc(override_text)}</code></dd>
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
              </div>
              {''.join(object_cards) or '<p>No extracted objects for this page.</p>'}
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
    .object-card {{ padding: 14px; border-top: 1px solid #d8d6cc; }}
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
    .paragraph {{ background: #e9f3ee; }}
    .structure {{ background: #edf0f7; }}
    .artifact {{ background: #f5eadb; }}
    .unknown {{ background: #f7e4e1; }}
    .review-controls {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; padding: 14px; border: 1px solid #d8d6cc; background: #fffef9; }}
    .review-controls label {{ display: grid; gap: 4px; font-weight: 700; }}
    .review-controls select, .review-controls input {{ font: inherit; padding: 6px 8px; border: 1px solid #bdb8ac; background: #fbfbf8; }}
    .review-count {{ margin: 10px 0 0; font-weight: 700; }}
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
    Persistent reviewer decisions live in <code>review_overrides.jsonl</code>. Overrides change
    candidate bucket assignment for review purposes only; they do not promote content to canonical.
  </p>
  <ul>
    <li>Applied overrides: <code>{validation_report.get('summary', {}).get('review_override_rows', 0)}</code></li>
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
    inventory = read_jsonl(output_dir / "page_inventory.jsonl")
    raw_pages = read_jsonl(output_dir / "raw_pages.jsonl")
    layout_objects = read_jsonl(output_dir / "layout_objects.jsonl")
    clean_objects = read_jsonl(output_dir / "clean_objects.jsonl")
    cleanup_log = read_jsonl(output_dir / "cleanup_log.jsonl")
    review_overrides = read_review_overrides(output_dir / "review_overrides.jsonl")
    main_paragraphs = read_jsonl(output_dir / "main_paragraph_candidates.jsonl")
    structure = read_jsonl(output_dir / "structure_candidates.jsonl")
    page_artifacts = read_jsonl(output_dir / "page_artifacts_candidates.jsonl")
    unknown = read_jsonl(output_dir / "unknown_objects.jsonl")
    reconstruction_map = read_json(output_dir / "reconstruction_map_candidate.json")
    reading_order = read_json(output_dir / "reading_order_candidate.json")

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
            "review_override_rows": len(review_overrides),
            "cleanup_log_rows": len(cleanup_log),
        },
    }


def run_phase1(pdf_path: Path, book_id: str, run_id: str = "phase1_v3") -> Path:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_dir = RUNS_DIR / book_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    review_overrides_path = output_dir / "review_overrides.jsonl"
    ensure_review_overrides_template(review_overrides_path)
    review_overrides = read_review_overrides(review_overrides_path)

    inventory: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []
    layout_objects: list[dict[str, Any]] = []
    clean_objects: list[dict[str, Any]] = []
    cleanup_log: list[dict[str, Any]] = []

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
            page_layout_objects, page_clean_objects, page_cleanup_log = build_segmented_objects(book_id, page_number, raw_lines)
            layout_objects.extend(page_layout_objects)
            clean_objects.extend(page_clean_objects)
            cleanup_log.extend(page_cleanup_log)

    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "source_pdf": str(pdf_path),
        "file_size_bytes": pdf_path.stat().st_size,
        "sha256": sha256_file(pdf_path),
        "page_count": page_count,
        "tooling": {"pdfplumber": getattr(pdfplumber, "__version__", "unknown")},
        "outputs": {
            "page_inventory": "page_inventory.jsonl",
            "raw_pages": "raw_pages.jsonl",
            "layout_objects": "layout_objects.jsonl",
            "clean_objects": "clean_objects.jsonl",
            "reading_order_candidate": "reading_order_candidate.json",
            "review_overrides": "review_overrides.jsonl",
            "cleanup_log": "cleanup_log.jsonl",
            "validation_report": "validation_report.json",
            "phase1_audit": "phase1_audit.html",
        },
    }
    object_counts = Counter(row["object_type"] for row in layout_objects)
    main_paragraphs, structure, page_artifacts, unknown, reconstruction_map = build_reconstruction_streams(
        book_id, run_id, layout_objects, clean_objects, inventory, page_count, review_overrides
    )
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
        "review_override_count": len(review_overrides),
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
