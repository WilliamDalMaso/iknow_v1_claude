from __future__ import annotations

from collections import Counter
import json
from tempfile import TemporaryDirectory
from pathlib import Path

from src.phase1_extract import (
    CID_PATTERN,
    build_audit_html,
    build_paragraph_promotion_artifacts,
    build_reconstruction_streams,
    build_segmented_objects,
    build_paragraph_merge_failure_taxonomy_report,
    classify_line,
    clean_line,
    join_paragraph_lines,
    page_status,
    review_canonical_paragraphs,
    validate_phase1_run,
    write_jsonl,
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


def test_span_guarded_merge_policy_splits_likely_accidental_merge() -> None:
    raw_lines = [
        {"text": "First paragraph begins with a normal sentence.", "x0": 43.0, "x1": 340.0, "top": 10.0, "bottom": 20.0},
        {"text": "It continues until this sentence ends.", "x0": 43.0, "x1": 330.0, "top": 24.0, "bottom": 34.0},
        {"text": "Second paragraph begins after a visual gap.", "x0": 43.0, "x1": 350.0, "top": 58.0, "bottom": 68.0},
        {"text": "It continues here.", "x0": 43.0, "x1": 180.0, "top": 72.0, "bottom": 82.0},
    ]

    baseline_layout, _, _ = build_segmented_objects("book", 22, raw_lines)
    guarded_layout, guarded_clean, _ = build_segmented_objects(
        "book", 22, raw_lines, paragraph_merge_policy="v2_span_guarded"
    )

    assert [row["object_type"] for row in baseline_layout] == ["paragraph"]
    assert [row["object_type"] for row in guarded_layout] == ["paragraph", "paragraph"]
    assert guarded_clean[0]["clean_text"] == "First paragraph begins with a normal sentence. It continues until this sentence ends."
    assert guarded_clean[1]["clean_text"] == "Second paragraph begins after a visual gap. It continues here."


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
            {"text": "CHAPTER I", "x0": 40, "x1": 120, "top": 20, "bottom": 34},
            {"text": "This is the first line", "x0": 60, "x1": 180, "top": 50, "bottom": 64},
            {"text": "of a paragraph.", "x0": 40, "x1": 140, "top": 66, "bottom": 80},
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
    assert "review_overrides.jsonl" in html
    assert "review_overrides_applied.jsonl" in html
    assert "page_images/page_0001.jpg" in html
    assert "Rendered PDF page 1" in html
    assert "bbox-overlay" in html
    assert "data-object-id=\"book:p0001:obj001\"" in html
    assert "overlay-show-all" in html
    assert "overlay-hide-all" in html
    assert "selected-object-detail" in html
    assert "Detector bucket" in html
    assert "Override bucket" in html
    assert "Final candidate bucket" in html
    assert "Canonical Paragraph Promotion" in html
    assert "data-promotion-status" in html
    assert "Copy-ready override JSONL" in html
    assert "overrideTemplateFor" in html
    assert "data-detector-bucket" in html
    assert "corrected_bucket" in html
    assert "evidence_reference" in html
    assert "reviews/book/review_overrides.jsonl" in html


def test_paragraph_promotion_promotes_only_evidence_bound_paragraphs() -> None:
    layout, clean, _ = build_segmented_objects(
        "book",
        1,
        [
            "CHAPTER I",
            "This is a real paragraph line",
            "with enough words to pass the simple promotion gate.",
            "iv",
        ],
    )
    paragraphs, structure, artifacts, unknown, _ = build_reconstruction_streams(
        "book", "phase1_v3", layout, clean, [inventory_row(1)], page_count=1
    )

    canonical, blockers, report = build_paragraph_promotion_artifacts(
        "book", "phase1_v3", paragraphs, structure, artifacts, unknown
    )

    assert len(canonical) == 1
    assert canonical[0]["promotion_status"] == "promoted"
    assert canonical[0]["source_candidate_object_id"] == paragraphs[0]["object_id"]
    assert canonical[0]["source_object_ids"] == [paragraphs[0]["object_id"]]
    assert {row["stream_type"] for row in blockers} == {"structure_candidate", "page_artifact_candidate"}
    assert report["counts"]["total_candidates_reviewed"] == 3
    assert report["counts"]["paragraph_candidates_reviewed"] == 1
    assert report["counts"]["promoted_paragraphs"] == 1
    assert report["counts"]["blocked_candidates"] == 2


def test_canonical_paragraph_review_flags_risky_promoted_rows() -> None:
    canonical = [
        {
            "book_id": "book",
            "run_id": "phase1_v3",
            "canonical_paragraph_id": "cp_000001",
            "source_candidate_object_id": "book:p0001:obj001",
            "page_number": 1,
            "raw_text": "preface material",
            "clean_text": "preface material",
            "source_object_ids": ["book:p0001:obj001"],
            "source_line_ids": ["book:p0001:line001"],
            "bbox": {"x0": 40, "x1": 120, "top": 20, "bottom": 35},
            "promotion_status": "promoted",
        }
    ]

    report = review_canonical_paragraphs("book", "phase1_v3", canonical)

    assert report["counts"]["total_canonical_paragraphs_reviewed"] == 1
    assert report["safe_for_downstream"] is False
    assert report["sample_risky_canonical_paragraphs"][0]["canonical_paragraph_id"] == "cp_000001"
    assert "possible_metadata_or_structure_leakage" in report["warning_categories"]
    assert report["warning_category_drilldown"][0]["count"] >= 1
    assert report["risky_paragraph_clusters"]
    assert report["recommendation_detail"]["top_risk_to_fix_first"]


def test_canonical_paragraph_review_adds_bbox_span_diagnostics() -> None:
    raw_lines = [f"line {index} continues the same promoted paragraph" for index in range(1, 13)]
    canonical = [
        {
            "book_id": "book",
            "run_id": "phase1_v3",
            "canonical_paragraph_id": "cp_000002",
            "source_candidate_object_id": "book:p0022:obj003",
            "page_number": 22,
            "raw_text": "\n".join(raw_lines),
            "clean_text": " ".join(raw_lines),
            "source_object_ids": ["book:p0022:obj003"],
            "source_line_ids": [f"book:p0022:line{index:03d}" for index in range(1, 13)],
            "bbox": {"x0": 40, "x1": 360, "top": 40, "bottom": 330},
            "promotion_status": "promoted",
        }
    ]

    report = review_canonical_paragraphs("book", "phase1_v3", canonical, {22: 600})

    diagnostics = report["bbox_span_risk_diagnostics"]
    assert report["bbox_span_risk_summary"]["total"] == 1
    assert diagnostics[0]["canonical_paragraph_id"] == "cp_000002"
    assert diagnostics[0]["source_line_count"] == 12
    assert diagnostics[0]["page_height_ratio"] == 290 / 600
    assert diagnostics[0]["warning_severity"] == "high"
    assert diagnostics[0]["likely_interpretation"] == "possible accidental merge"
    assert diagnostics[0]["likely_corrective_path"] == ["paragraph merge rule adjustment", "manual inspection"]
    assert diagnostics[0]["audit_anchor"] == "#card-book-p0022-obj003"
    assert report["bbox_span_risk_summary"]["by_source_line_count_range"][0]["source_line_count_range"] == "11+"
    assert report["bbox_span_decision_summary"]["total"] == 1
    assert report["bbox_span_decisions"][0]["likely_cause"] == "true_accidental_merge"
    assert report["bbox_span_decisions"][0]["recommended_action"] == "adjust paragraph merge rule"
    assert report["bbox_span_decision_summary"]["by_likely_cause"][0]["likely_cause"] == "true_accidental_merge"


def test_merge_failure_taxonomy_samples_likely_true_merges() -> None:
    canonical = [
            {
                "canonical_paragraph_id": "cp_000010",
                "clean_text": "First sentence ends. Second paragraph probably starts here. It continues with enough text to inspect.",
            }
    ]
    review_report = {
        "bbox_span_decisions": [
            {
                "canonical_paragraph_id": "cp_000010",
                "likely_cause": "true_accidental_merge",
            }
        ],
        "bbox_span_risk_diagnostics": [
            {
                "canonical_paragraph_id": "cp_000010",
                "page_number": 30,
                "source_candidate_object_id": "book:p0030:obj002",
                "warning_severity": "high",
                "source_line_count": 12,
                "vertical_bbox_span": 210.0,
                "page_height_ratio": 0.34,
                "first_source_line_preview": "First sentence ends.",
                "last_source_line_preview": "Second paragraph probably starts here.",
                "all_warnings": ["source_lines_span_suspicious_distance"],
                "audit_anchor": "#card-book-p0030-obj002",
                "page_anchor": "#page-30",
            }
        ],
    }

    report = build_paragraph_merge_failure_taxonomy_report("book", "phase1_v3", canonical, review_report)

    assert report["summary"]["sampled_rows"] == 1
    assert report["samples"][0]["canonical_paragraph_id"] == "cp_000010"
    assert report["samples"][0]["provisional_category"] == "merged_across_paragraph_break"
    assert report["summary"]["count_by_category"]["merged_across_paragraph_break"] == 1


def test_review_override_moves_candidate_bucket() -> None:
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
    overridden_object_id = layout[0]["object_id"]

    paragraphs, structure, artifacts, unknown, reconstruction_map = build_reconstruction_streams(
        "book",
        "phase1_v3",
        layout,
        clean,
        [inventory_row(20), inventory_row(22), inventory_row(24)],
        page_count=24,
        review_overrides=[
            {
                "object_id": overridden_object_id,
                "original_bucket": "page_artifact_candidate",
                "corrected_bucket": "structure_candidate",
                "reason": "test override",
                "reviewer": "test",
                "date": "2026-05-27",
                "evidence_reference": "unit test repeated header sample",
            }
        ],
    )

    assert len(paragraphs) == 3
    assert len(structure) == 1
    assert len(artifacts) == 2
    assert unknown == []
    assert structure[0]["object_id"] == overridden_object_id
    assert structure[0]["review_override"]["original_bucket"] == "page_artifact_candidate"
    assert structure[0]["review_override"]["corrected_bucket"] == "structure_candidate"
    assert structure[0]["review_override"]["declared_original_bucket"] == "page_artifact_candidate"
    assert structure[0]["review_override"]["evidence_reference"] == "unit test repeated header sample"
    assert reconstruction_map["review_override_count"] == 1


def test_validation_rejects_malformed_review_override_row() -> None:
    layout, clean, cleanup = build_segmented_objects(
        "book",
        1,
        [
            "This is a real paragraph line.",
        ],
    )
    paragraphs, structure, artifacts, unknown, reconstruction_map = build_reconstruction_streams(
        "book",
        "phase1_v3",
        layout,
        clean,
        [inventory_row(1)],
        page_count=1,
    )
    malformed_overrides = [
        {
            "object_id": layout[0]["object_id"],
            "original_bucket": "structure_candidate",
            "corrected_bucket": "not_a_bucket",
            "reason": "bad row should fail multiple checks",
            "reviewer": "test",
            "date": "2026-05-27",
            "review_source": "curated",
            "review_source_path": "reviews/book/review_overrides.jsonl",
        },
        {
            "object_id": layout[0]["object_id"],
            "original_bucket": "main_paragraph_candidate",
            "corrected_bucket": "structure_candidate",
            "reason": "duplicate object id",
            "reviewer": "test",
            "date": "2026-05-27",
            "evidence_reference": "unit test",
            "review_source": "curated",
            "review_source_path": "reviews/book/review_overrides.jsonl",
        },
        {
            "object_id": "book:p0001:obj999",
            "original_bucket": "main_paragraph_candidate",
            "corrected_bucket": "structure_candidate",
            "reason": "missing object id",
            "reviewer": "test",
            "date": "2026-05-27",
            "evidence_reference": "unit test",
            "review_source": "curated",
            "review_source_path": "reviews/book/review_overrides.jsonl",
        },
    ]

    with TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        (output_dir / "source_manifest.json").write_text(
            json.dumps(
                {
                    "page_count": 1,
                    "paragraph_merge_policy": "v1_consecutive_lines",
                    "paragraph_merge_experiment_policy": "v2_span_guarded",
                }
            ),
            encoding="utf-8",
        )
        write_jsonl(output_dir / "page_inventory.jsonl", [inventory_row(1)])
        write_jsonl(output_dir / "raw_pages.jsonl", [{"page_number": 1, "raw_text": "This is a real paragraph line."}])
        write_jsonl(output_dir / "layout_objects.jsonl", layout)
        write_jsonl(output_dir / "clean_objects.jsonl", clean)
        write_jsonl(output_dir / "main_paragraph_candidates.jsonl", paragraphs)
        write_jsonl(output_dir / "structure_candidates.jsonl", structure)
        write_jsonl(output_dir / "page_artifacts_candidates.jsonl", artifacts)
        write_jsonl(output_dir / "unknown_objects.jsonl", unknown)
        canonical, blockers, report = build_paragraph_promotion_artifacts(
            "book", "phase1_v3", paragraphs, structure, artifacts, unknown
        )
        write_jsonl(output_dir / "canonical_paragraphs.jsonl", canonical)
        write_jsonl(output_dir / "promotion_blockers.jsonl", blockers)
        (output_dir / "canonical_promotion_report.json").write_text(json.dumps(report), encoding="utf-8")
        review_report = review_canonical_paragraphs("book", "phase1_v3", canonical)
        (output_dir / "canonical_paragraph_review_report.json").write_text(json.dumps(review_report), encoding="utf-8")
        (output_dir / "paragraph_merge_experiment_report.json").write_text(
            json.dumps(
                {
                    "counts": {
                        "baseline_paragraph_candidate_count": len(paragraphs),
                        "new_paragraph_candidate_count": len(paragraphs),
                        "baseline_canonical_promoted_count": len(canonical),
                        "new_canonical_promoted_count": len(canonical),
                        "baseline_bbox_span_risk_count": 0,
                        "new_bbox_span_risk_count": 0,
                        "baseline_likely_true_accidental_merge_count": 0,
                        "new_likely_true_accidental_merge_count": 0,
                        "baseline_blocked_paragraph_count": sum(1 for row in blockers if row.get("stream_type") == "main_paragraph_candidate"),
                        "new_blocked_paragraph_count": sum(1 for row in blockers if row.get("stream_type") == "main_paragraph_candidate"),
                    }
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "paragraph_merge_failure_taxonomy_report.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "sampled_rows": 0,
                        "count_by_category": {},
                        "count_by_recommended_action": {},
                    },
                    "samples": [],
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "reconstruction_map_candidate.json").write_text(json.dumps(reconstruction_map), encoding="utf-8")
        (output_dir / "reading_order_candidate.json").write_text(
            json.dumps({"object_ids": [row["object_id"] for row in layout]}),
            encoding="utf-8",
        )
        write_jsonl(output_dir / "review_overrides_applied.jsonl", malformed_overrides)
        write_jsonl(output_dir / "cleanup_log.jsonl", cleanup)
        (output_dir / "validation_report.json").write_text("{}", encoding="utf-8")
        (output_dir / "phase1_audit.html").write_text("<html></html>", encoding="utf-8")

        report = validate_phase1_run(output_dir)

    assert report["status"] == "fail"
    failed = {row["name"] for row in report["checks"] if row["status"] == "fail"}
    assert "review_overrides_reference_known_objects" in failed
    assert "review_override_object_ids_unique" in failed
    assert "review_override_required_fields_present" in failed
    assert "review_override_buckets_valid" in failed
    assert "review_override_original_bucket_matches_detector" in failed
