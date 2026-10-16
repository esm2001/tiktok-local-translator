"""Protect independent ground truth and the meaning of regression measurements."""

import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "translator"))

from regression_metrics import fixture_failures, load_dataset, ocr_score, text_score, translation_score
from src.ocr.editorial import editorial_lines


def rectangle(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


@pytest.fixture
def two_lines():
    return {
        "expected_text": "You never admit\nyou're wrong.",
        "expected_regions": [
            {"box": [.1, .1, .9, .3], "text": "You never admit"},
            {"box": [.1, .5, .9, .7], "text": "you're wrong."},
        ],
    }


def test_detecting_both_regions_does_not_hide_character_errors(two_lines):
    first, second = rectangle(20, 10, 180, 30), rectangle(20, 50, 180, 70)
    result = ocr_score(two_lines, {
        "detected_polygons": [first, second],
        "recognized_lines": [
            {"polygon": first, "text": "You never admlt"},
            {"polygon": second, "text": "you're wr0ng."},
        ],
    }, (100, 200))
    assert result["all_regions_detected"] is True
    assert result["detected_regions"] == 2
    assert result["recognition"]["character_errors"] == 2
    assert result["recognition"]["exact_normalised"] is False


def test_missing_second_detector_region_remains_a_detection_failure(two_lines):
    first = rectangle(20, 10, 180, 30)
    result = ocr_score(two_lines, {
        "detected_polygons": [first],
        "recognized_lines": [{"polygon": first, "text": "You never admit"}],
    }, (100, 200))
    assert result["detected_regions"] == 1
    assert result["all_regions_detected"] is False
    assert result["regions"][1]["actual"] == ""
    assert result["regions"][1]["detected"] is False
    assert result["recognition"]["cer"] > 0


def test_detector_evidence_without_recognition_is_not_success(two_lines):
    result = ocr_score(two_lines, {
        "detected_polygons": [rectangle(20, 10, 180, 30), rectangle(20, 50, 180, 70)],
        "recognized_lines": [],
    }, (100, 200))
    assert result["all_regions_detected"] is True
    assert result["visible_text_missing"] is True
    assert result["recognition"]["cer"] == 1


def test_split_polygons_jointly_detect_and_recognize_one_line(two_lines):
    left, right = rectangle(20, 10, 96, 30), rectangle(96, 10, 180, 30)
    second = rectangle(20, 50, 180, 70)
    # Recognition is deliberately supplied out of order; spatial order wins.
    result = ocr_score(two_lines, {
        "detected_polygons": [right, left, second],
        "recognized_lines": [
            {"polygon": right, "text": "admit"},
            {"polygon": second, "text": "you're wrong."},
            {"polygon": left, "text": "You never"},
        ],
    }, (100, 200))
    assert result["detected_regions"] == 2
    assert result["regions"][0]["actual"] == "You never admit"
    assert result["recognition"]["exact_normalised"] is True


def test_duplicate_detector_polygons_cannot_inflate_region_count(two_lines):
    first, second = rectangle(20, 10, 180, 30), rectangle(20, 50, 180, 70)
    result = ocr_score(two_lines, {
        "detected_polygons": [first, first, second, second],
        "recognized_lines": [
            {"polygon": first, "text": "You never admit"},
            {"polygon": second, "text": "you're wrong."},
        ],
    }, (100, 200))
    assert result["detected_regions"] == result["expected_regions"] == 2
    assert result["recognition"]["exact_normalised"] is True


def test_duplicate_recognition_is_scored_as_extra_characters(two_lines):
    first, second = rectangle(20, 10, 180, 30), rectangle(20, 50, 180, 70)
    result = ocr_score(two_lines, {
        "detected_polygons": [first, second],
        "recognized_lines": [
            {"polygon": first, "text": "You never admit"},
            {"polygon": first, "text": "You never admit"},
            {"polygon": second, "text": "you're wrong."},
        ],
    }, (100, 200))
    assert result["all_regions_detected"] is True
    assert result["recognition"]["exact_normalised"] is False
    assert result["recognition"]["character_errors"] > 0


def test_repeated_words_in_separate_regions_are_not_deduplicated():
    case = {"expected_text": "No\nNo", "expected_regions": [
        {"box": [.1, .1, .4, .3], "text": "No"},
        {"box": [.1, .5, .4, .7], "text": "No"},
    ]}
    first, second = rectangle(20, 10, 80, 30), rectangle(20, 50, 80, 70)
    result = ocr_score(case, {"detected_polygons": [first, second], "recognized_lines": [
        {"polygon": first, "text": "No"}, {"polygon": second, "text": "No"},
    ]}, (100, 200))
    assert result["recognition"]["exact_normalised"] is True
    assert [region["actual"] for region in result["regions"]] == ["No", "No"]


def test_clause_presence_cannot_certify_a_translation_with_reversed_meaning():
    case = {"translation_checks": [["admites"], ["equivocado"]]}
    faithful = translation_score(case, "Nunca admites que estás equivocado.")
    reversed_meaning = translation_score(case, "Siempre admites que estás equivocado.")
    assert faithful["all_clause_checks_passed"] is True
    assert reversed_meaning["all_clause_checks_passed"] is True
    assert reversed_meaning["semantic_human_review"] == "PENDING"


def test_normalized_character_score_does_not_claim_spacing_accuracy():
    assert text_score("comfort zone.", "comfortzone")["exact_normalised"] is True
    assert text_score("wrong", "wr0ng")["character_errors"] == 1


def test_local_ground_truth_has_consistent_text_and_valid_independent_regions():
    dataset = load_dataset()
    assert len(dataset["slides"]) >= 13
    for identifier, case in dataset["slides"].items():
        assert identifier == f"{case['post_id']}_{case['position']:02d}"
        assert case["expected_text"] == "\n".join(case["expected_lines"])
        assert [region["text"] for region in case["expected_regions"]] == case["expected_lines"]
        assert case["expected_regions"]
        image = Path(case["image"])
        assert not image.is_absolute() and ".." not in image.parts
        for region in case["expected_regions"]:
            left, top, right, bottom = region["box"]
            assert 0 <= left < right <= 1
            assert 0 <= top < bottom <= 1
        if case["source_expected"] is True:
            evidence = case["source_evidence"]
            assert evidence.get("file") or evidence.get("image_url")


def test_printed_typo_and_semantically_ambiguous_title_remain_explicit():
    cases = load_dataset()["slides"]
    typo = cases["7682913895333383456_06"]
    assert "Recources" in typo["expected_text"]
    assert any("Resources" in variant for variant in typo["reconstruction_variants"])
    ambiguous = cases["7682913895333383456_05"]
    assert "Teach someone simpler." in ambiguous["expected_text"]
    assert any(fragment["text"] == "Teach someone simpler." for fragment in ambiguous["ambiguous_fragments"])
    assert "Teach someone simpler." in ambiguous["expected_translation"]


def test_missing_dataset_is_a_failure_instead_of_an_empty_success(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.json")


def test_missing_fixture_is_reported_even_when_other_fixture_exists(tmp_path):
    dataset = {"slides": {
        "present": {"image": "post/original/01.jpg"},
        "missing": {"image": "post/original/02.jpg"},
        "directory": {"image": "post/original/03.jpg"},
    }}
    original = tmp_path / "post" / "original"
    original.mkdir(parents=True)
    (original / "01.jpg").write_bytes(b"local file presence fixture")
    (original / "03.jpg").mkdir()
    assert fixture_failures(dataset, tmp_path) == {
        "missing": "MISSING_FIXTURE", "directory": "MISSING_FIXTURE",
    }


def test_absent_media_root_reports_every_case_as_a_failure(tmp_path):
    dataset = load_dataset()
    failures = fixture_failures(dataset, tmp_path / "not-downloaded")
    assert set(failures) == set(dataset["slides"])
    assert set(failures.values()) == {"MISSING_FIXTURE"}


def test_altered_fixture_at_same_filename_fails_hash_validation(tmp_path):
    image = tmp_path / 'post/original/01.jpg'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'independently reviewed photograph')
    dataset = {'slides': {'reviewed': {'image': 'post/original/01.jpg',
                                     'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}}}
    assert fixture_failures(dataset, tmp_path) == {}
    image.write_bytes(b'another photograph with the same filename')
    assert fixture_failures(dataset, tmp_path) == {'reviewed': 'FIXTURE_HASH_MISMATCH'}


def test_small_legitimate_subtitle_and_short_no_are_preserved():
    lines = [
        {"text": "Do what you can", "confidence": .99, "polygon": rectangle(200, 100, 800, 180)},
        {"text": "Stay with your choices", "confidence": .99, "polygon": rectangle(200, 200, 800, 280)},
        {"text": "An important subtitle", "confidence": .99, "polygon": rectangle(200, 500, 800, 520)},
        {"text": "No", "confidence": .99, "polygon": rectangle(470, 960, 530, 975)},
    ]
    kept, ignored = editorial_lines(lines, (1000, 1000))
    assert kept == lines
    assert ignored == []
