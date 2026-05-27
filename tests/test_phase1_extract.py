from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.phase1_extract import (
    CID_PATTERN,
    build_audit_html,
    build_reconstruction_streams,
    build_segmented_objects,
    classify_line,
    clean_line,
    join_paragraph_lines,
    page_status,
)


def inventory_row(page_number: int = 1) -> dict:
    return {
        "book_id": "book",
        "page_number": page_number,
        "width": 400,
        "height": 600,
        "status": "text",
        "raw_char_count": 32,
        "line_count": 3,
        "image_count": 0,
        "table_count": 0,
        "review_flags": [],
        "sample": "sample",
    }


def test_clean_line_flags_cid_noise() -> None:
    text, operations = clean_line("Title (cid:3)  text  ")
    assert text == "Title (cid:3) text"
    assert "rstrip" in operations
    assert "collapse_whitespace" in operations
    assert "flag_cid_noise" in operations
    assert CID_PATTERN.search(text)


def test_page_status_distinguishes_image_only() -> None:
    assert page_status("", 1) == "image_only"
    assert page_status("Readable", 0) == "text"
    assert page_status("Readable", 2) == "mixed_text_and_image"
    assert page_status("", 0) == "blank_or_unreadable"


def test_classify_line_detects_heading_and_page_artifact() -> None:
    assert classify_line("PREFACE", 5)[0] == "heading"
    assert classify_line("vi", 6)[0] == "page_artifact"
    assert classify_line("This is a normal sentence from a paragraph.", 20)[0] == "paragraph_line"


def test_join_paragraph_lines_merges_and_dehyphenates() -> None:
    text, operations = join_paragraph_lines(["This is a hyphen-", "ated word."])
    assert text == "This is a hyphenated word."
    assert "merge_paragraph_lines" in operations
    assert "join_hyphenated_line_break" in operations


def test_build_segmented_objects_merges_paragraph_lines() -> None:
    layout, clean, cleanup = build_segmented_objects(
        "book",
        3,
        [
            "CHAPTER I",
            "This is the first line",
            "of a paragraph.",
            "ANOTHER HEADING",
        ],
    )
    assert [row["object_type"] for row in layout] == ["heading_candidate", "paragraph", "heading_candidate"]
    paragraph = clean[1]
    assert paragraph["clean_text"] == "This is the first line of a paragraph."
    assert "merge_paragraph_lines" in paragraph["cleanup_operations"]
    assert any(row["operation"] == "merge_paragraph_lines" for row in cleanup)


def test_build_segmented_objects_splits_indented_paragraphs() -> None:
    layout, clean, _ = build_segmented_objects(
        "book",
        5,
        [
            {"text": "First paragraph starts here", "x0": 64.8, "top": 10, "bottom": 20},
            {"text": "and continues here.", "x0": 43.2, "top": 24, "bottom": 34},
            {"text": "Second paragraph starts here", "x0": 64.9, "top": 38, "bottom": 48},
            {"text": "and continues here.", "x0": 43.3, "top": 52, "bottom": 62},
        ],
    )
    assert [row["object_type"] for row in layout] == ["paragraph", "paragraph"]
    assert clean[0]["clean_text"] == "First paragraph starts here and continues here."
    assert clean[1]["clean_text"] == "Second paragraph starts here and continues here."
    assert layout[0]["bbox"]["x0"] == 43.2
    assert layout[0]["bbox"]["top"] == 10
    assert layout[0]["bbox"]["bottom"] == 34


def test_build_reconstruction_streams_buckets_every_object_once() -> None:
    layout, clean, _ = build_segmented_objects(
        "book",
        3,
        [
            "CHAPTER I",
            "This is the first line",
            "of a paragraph.",
            "iv",
        ],
    )
    paragraphs, structure, artifacts, unknown, reconstruction_map = build_reconstruction_streams(
        "book", "phase1_v3", layout, clean, [inventory_row(3)], page_count=3
    )
    assert len(paragraphs) == 1
    assert len(structure) == 1
    assert len(artifacts) == 1
    assert unknown == []
    assert paragraphs[0]["stream_type"] == "main_paragraph_candidate"
    assert paragraphs[0]["source_object_ids"] == [paragraphs[0]["object_id"]]
    assert structure[0]["stream_type"] == "structure_candidate"
    assert artifacts[0]["stream_type"] == "page_artifact_candidate"
    assert reconstruction_map["counts"] == {
        "main_paragraph_candidates": 1,
        "structure_candidates": 1,
        "page_artifacts_candidates": 1,
        "unknown_objects": 0,
    }


def test_repeated_margin_heading_becomes_page_artifact_candidate() -> None:
    layout = []
    clean = []
    for page_number, printed_page in [(20, "2"), (22, "4"), (24, "6")]:
        page_layout, page_clean, _ = build_segmented_objects(
            "book",
            page_number,
            [
                {"text": f"{printed_page} NARRATIVE OF THE", "x0": 42, "top": 37, "bottom": 49},
                {"text": "This is a real paragraph line.", "x0": 64, "top": 80, "bottom": 92},
            ],
        )
        layout.extend(page_layout)
        clean.extend(page_clean)

    paragraphs, structure, artifacts, unknown, reconstruction_map = build_reconstruction_streams(
        "book",
        "phase1_v3",
        layout,
        clean,
        [inventory_row(20), inventory_row(22), inventory_row(24)],
        page_count=24,
    )

    assert len(paragraphs) == 3
    assert len(structure) == 0
    assert len(artifacts) == 3
    assert unknown == []
    assert {row["artifact_type"] for row in artifacts} == {"running_header_candidate"}
    assert all("page_number_attached_to_repeated_text" in row["classification_reasons"] for row in artifacts)
    assert reconstruction_map["candidate_only_exclusions"]["page_artifact_candidate_object_ids"] == [
        row["object_id"] for row in artifacts
    ]


def test_build_audit_html_contains_page_object_inspection() -> None:
    layout, clean, _ = build_segmented_objects(
        "book",
        1,
        [
            "CHAPTER I",
            "This is the first line",
            "of a paragraph.",
        ],
    )
    paragraphs, structure, artifacts, unknown, _ = build_reconstruction_streams(
        "book", "phase1_v3", layout, clean, [inventory_row(1)], page_count=1
    )
    stream_samples = {
        "main_paragraph_candidates": paragraphs[:8],
        "structure_candidates": structure[:8],
        "page_artifacts_candidates": artifacts[:8],
        "unknown_objects": unknown[:8],
        "__all__": paragraphs + structure + artifacts + unknown,
    }
    html = build_audit_html(
        "book",
        Path("book.pdf"),
        {"page_count": 1, "file_size_bytes": 10, "sha256": "abc"},
        [
            {
                "page_number": 1,
                "status": "text",
                "raw_char_count": 32,
                "line_count": 3,
                "image_count": 0,
                "table_count": 0,
                "review_flags": [],
                "sample": "CHAPTER I | This is the first line",
            }
        ],
        layout,
        Counter(row["object_type"] for row in layout),
        {
            "main_paragraph_candidates": len(paragraphs),
            "structure_candidates": len(structure),
            "page_artifacts_candidates": len(artifacts),
            "unknown_objects": len(unknown),
        },
        stream_samples,
        {"status": "pass", "checks": []},
        Path("out"),
    )
    assert "Page Inspection Index" in html
    assert "Page-by-Page Object Inspection" in html
    assert "Raw Extracted Object" in html
    assert "Candidate Assignment" in html
    assert "main_paragraph_candidate" in html
    assert "Repeated Artifact Pattern Review" in html
    assert "False-Positive Risk Review" in html
    assert "filter-bucket" in html
    assert "filter-page-min" in html
