"""Benchmark failures must be visible without loading models or using Docker."""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_ocr_candidates import main, selected_cases, summarize


def test_missing_media_produces_failure_report_before_model_initialisation(tmp_path):
    report = tmp_path / "report.json"
    result = main(["--media-root", str(tmp_path / "missing-media"), "--output", str(report)])
    measured = json.loads(report.read_text(encoding="utf-8"))
    assert result == 1
    assert measured["status"] == "FAIL"
    assert measured["expected_cases"] == len(measured["missing_fixtures"]) >= 13
    assert set(measured["missing_fixtures"].values()) == {"MISSING_FIXTURE"}


def test_unknown_case_cannot_become_an_empty_success():
    with pytest.raises(ValueError, match="Unknown case IDs"):
        selected_cases({"slides": {"existing": {}}}, ["missing"])


def test_ocr_error_keeps_failed_slide_and_regions_in_denominator():
    rows = [
        {"seconds": 1, "expected_regions": 2, "metrics": {
            "all_regions_detected": True, "detected_regions": 2, "expected_regions": 2,
            "visible_text_missing": False,
            "recognition": {"exact_normalised": True, "character_errors": 0, "expected_characters": 10},
        }},
        {"expected_regions": 3, "error": "OCR_ERROR"},
    ]
    report = summarize(rows, expected_count=2)
    assert report["expected_slides"] == 2
    assert report["measured_slides"] == report["detection_slides"] == 1
    assert report["errors"] == 1
    assert report["detection_regions"] == 2
    assert report["expected_regions"] == 5


def test_report_cannot_overwrite_a_post_artifact(tmp_path):
    output = tmp_path / "1234567890123456789" / "metadata" / "manifest.json"
    with pytest.raises(SystemExit) as failure:
        main(["--media-root", str(tmp_path), "--output", str(output)])
    assert failure.value.code == 2
    assert not output.exists()
