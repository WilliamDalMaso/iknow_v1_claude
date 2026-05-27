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
REQUIRED_ARTIFACTS = [
    "source_manifest.json",
    "page_inventory.jsonl",
    "raw_pages.jsonl",
    "layout_objects.jsonl",
    "clean_objects.jsonl",
    "canonical_reading_order.json",
    "cleanup_log.jsonl",
    "validation_report.json",
    "phase1_audit.html",
]


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
    object_counts: Counter[str],
    validation_report: dict[str, Any],
    output_dir: Path,
) -> str:
    status_counts = Counter(row["status"] for row in inventory)
    flagged_pages = [row for row in inventory if row["review_flags"]]
    sample_pages = inventory[:12]
    generated = utc_now()

    def esc(value: Any) -> str:
        return html.escape(str(value))

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
    flagged_items = "\n".join(
        f"<li>Page {row['page_number']}: {esc(', '.join(row['review_flags']))}</li>"
        for row in flagged_pages[:40]
    )
    validation_items = "\n".join(
        f"<li><code>{esc(check['name'])}</code>: {esc(check['status'])} {esc(check.get('detail', ''))}</li>"
        for check in validation_report.get("checks", [])
    )
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
  </style>
</head>
<body>
<main>
  <h1>Phase 1 Audit: {esc(book_id)}</h1>
  <p><strong>Generated:</strong> {esc(generated)}</p>
  <p><strong>Source:</strong> <code>{esc(source_pdf)}</code></p>

  <div class="rule">
    This is a deterministic extraction audit. It is designed to reveal extraction risks before
    retrieval, reasoning, or graph construction begins.
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

  <h2>Validation</h2>
  <p><strong>Status:</strong> <code>{esc(validation_report.get('status', 'unknown'))}</code></p>
  <ul>{validation_items}</ul>

  <h2>Flagged Pages</h2>
  <ul>{flagged_items or '<li>No flagged pages.</li>'}</ul>

  <h2>First 12 Pages</h2>
  <table>
    <thead>
      <tr><th>Page</th><th>Status</th><th>Chars</th><th>Lines</th><th>Images</th><th>Tables</th><th>Flags</th><th>Sample</th></tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</main>
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
    canonical = read_json(output_dir / "canonical_reading_order.json")

    page_count = int(manifest.get("page_count", 0))
    add_check("page_inventory_matches_manifest", len(inventory) == page_count, f"{len(inventory)} inventory rows / {page_count} manifest pages")
    add_check("raw_pages_matches_manifest", len(raw_pages) == page_count, f"{len(raw_pages)} raw rows / {page_count} manifest pages")
    inventory_pages = {row.get("page_number") for row in inventory}
    raw_pages_set = {row.get("page_number") for row in raw_pages}
    add_check("page_numbers_align", inventory_pages == raw_pages_set == set(range(1, page_count + 1)))

    layout_ids = [row.get("object_id") for row in layout_objects]
    clean_ids = [row.get("object_id") for row in clean_objects]
    canonical_ids = canonical.get("object_ids", [])
    add_check("object_ids_unique", len(layout_ids) == len(set(layout_ids)))
    add_check("clean_objects_match_layout", set(clean_ids) == set(layout_ids))
    add_check("canonical_ids_match_layout", canonical_ids == layout_ids)
    add_check("object_count_matches_canonical", canonical.get("object_count") == len(layout_objects))
    add_check("objects_have_source_lines", all(row.get("source_line_ids") for row in layout_objects))
    add_check("cleanup_log_references_known_objects", {row.get("object_id") for row in cleanup_log}.issubset(set(layout_ids)))
    add_check("raw_text_preserved", all("raw_text" in row for row in raw_pages))
    add_check("review_flags_present", "review_flags" in canonical)

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
            "cleanup_log_rows": len(cleanup_log),
        },
    }


def run_phase1(pdf_path: Path, book_id: str, run_id: str = "phase1_v1") -> Path:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_dir = RUNS_DIR / book_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

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
            "canonical_reading_order": "canonical_reading_order.json",
            "cleanup_log": "cleanup_log.jsonl",
            "validation_report": "validation_report.json",
            "phase1_audit": "phase1_audit.html",
        },
    }
    object_counts = Counter(row["object_type"] for row in layout_objects)
    canonical = {
        "book_id": book_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "object_count": len(layout_objects),
        "object_type_counts": dict(sorted(object_counts.items())),
        "page_count": page_count,
        "object_ids": [row["object_id"] for row in layout_objects],
        "review_flags": sorted({flag for page in inventory for flag in page["review_flags"]}),
    }

    write_json(output_dir / "source_manifest.json", manifest)
    write_jsonl(output_dir / "page_inventory.jsonl", inventory)
    write_jsonl(output_dir / "raw_pages.jsonl", raw_pages)
    write_jsonl(output_dir / "layout_objects.jsonl", layout_objects)
    write_jsonl(output_dir / "clean_objects.jsonl", clean_objects)
    write_json(output_dir / "canonical_reading_order.json", canonical)
    write_jsonl(output_dir / "cleanup_log.jsonl", cleanup_log)
    pending_validation = {
        "created_at": utc_now(),
        "status": "pending",
        "checks": [],
        "summary": {"detail": "Validation report is generated after the first audit render."},
    }
    write_json(output_dir / "validation_report.json", pending_validation)
    (output_dir / "phase1_audit.html").write_text(
        build_audit_html(book_id, pdf_path, manifest, inventory, object_counts, pending_validation, output_dir),
        encoding="utf-8",
    )
    validation_report = validate_phase1_run(output_dir)
    write_json(output_dir / "validation_report.json", validation_report)
    (output_dir / "phase1_audit.html").write_text(
        build_audit_html(book_id, pdf_path, manifest, inventory, object_counts, validation_report, output_dir),
        encoding="utf-8",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Phase 1 PDF extraction.")
    parser.add_argument("pdf_path", help="Path to the source PDF.")
    parser.add_argument("--book-id", required=True, help="Stable book id for output paths.")
    parser.add_argument("--run-id", default="phase1_v1", help="Run id for output paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run_phase1(Path(args.pdf_path), args.book_id, args.run_id)
    print(json.dumps({"book_id": args.book_id, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
