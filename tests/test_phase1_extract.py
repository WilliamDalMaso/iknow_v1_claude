from __future__ import annotations

from src.phase1_extract import CID_PATTERN, classify_line, clean_line, page_status


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

