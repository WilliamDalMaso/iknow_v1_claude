"""Pure text/line helpers and the shared text-pattern constants for Phase 1.

Extracted verbatim from ``phase1_extract.py`` (decomposition step 2). These are
deterministic string operations with no project state or IO. ``phase1_extract``
re-imports the names (constants included) so existing call sites are unchanged.
"""
from __future__ import annotations

import re

CID_PATTERN = re.compile(r"\(cid:\d+\)")
ROMAN_PATTERN = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
TEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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


def terminal_line_boundary(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped[-1] in ".!?;:"


def uppercase_line_start(text: str) -> bool:
    stripped = text.lstrip()
    return bool(stripped) and stripped[0].isupper()
