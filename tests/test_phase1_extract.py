from __future__ import annotations

from src.phase1_extract import CID_PATTERN, build_segmented_objects, classify_line, clean_line, join_paragraph_lines, page_status


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
