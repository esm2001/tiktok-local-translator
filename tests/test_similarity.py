"""Regression checks against the seven real carousel images supplied locally.

Positive cases use an actual overlay-free crop of slide 01; no text is rendered
and no generated background is counted as a recovered source. Downloaded media
is deliberately not copied into the repository. Tests skip on another machine
until a local carousel fixture is provided with TIKTOK_TEST_POST.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "translator"))
from src.image.cleaning import build_text_mask, make_clean_image, read_image, write_image
from src.image.similarity import compare_candidate


REAL_POST = Path(os.environ.get("TIKTOK_TEST_POST", str(PROJECT.parent / "TikTokTraducidos" / "7659483315203804449")))


@unittest.skipUnless((REAL_POST / "original" / "01.jpg").exists(), "Local real carousel fixture is unavailable")
class RealCarouselSimilarityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="tiktok_similarity_")
        cls.temp = Path(cls.workspace.name)
        cls.original = REAL_POST / "original" / "01.jpg"
        cls.image = read_image(cls.original)
        # Annotated bounds include both actual overlay lines and the emoji.
        # Paddle polygons from a measured baseline are preferable when available.
        cls.polygons = [
            [[145, 593], [1040, 593], [1040, 694], [145, 694]],
            [[195, 690], [1000, 690], [1000, 792], [195, 792]],
        ]
        baseline = PROJECT / ".cache" / "baseline.json"
        if baseline.exists():
            data = json.loads(baseline.read_text(encoding="utf-8"))
            rows = data.get("images", data.get("slides", [])) if isinstance(data, dict) else data
            if rows and rows[0].get("polygons"):
                cls.polygons = rows[0]["polygons"]
            elif rows and rows[0].get("detected_polygons"):
                cls.polygons = rows[0]["detected_polygons"]
        cls.mask = cls.temp / "01_MASK.png"
        write_image(cls.mask, build_text_mask(cls.image.shape, cls.polygons))
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        positives = [r for r in cls.results if r["expected"]]
        negatives = [r for r in cls.results if not r["expected"]]
        print("\nReal-image similarity benchmark: " + json.dumps({
            "positive_acceptance": f"{sum(r['accepted'] for r in positives)}/{len(positives)}",
            "false_positives": f"{sum(r['accepted'] for r in negatives)}/{len(negatives)}",
            "seconds": round(sum(r["seconds"] for r in cls.results), 3),
            "cases": cls.results,
        }, ensure_ascii=False))
        cls.workspace.cleanup()

    def compare(self, name: str, image_path: Path, expected: bool):
        started = time.perf_counter()
        result = compare_candidate(self.original, image_path, self.mask)
        self.results.append({
            "case": name, "expected": expected, "accepted": result["accepted"],
            "score": result["score"], "reason": result["reason"],
            "seconds": round(time.perf_counter() - started, 3),
        })
        return result

    def test_crop_scale_jpeg_and_small_colour_variation(self):
        crop = self.image[820:].copy()  # Actual lower portrait pixels, with no text overlay.
        variants = {
            "background_crop": (crop, 95),
            "crop_scaled": (cv2.resize(crop, (819, round(crop.shape[0] * 0.7)), interpolation=cv2.INTER_AREA), 90),
            "crop_jpeg": (crop, 35),
            "crop_colour": (np.clip(crop.astype(np.float32) * 0.94 + 8, 0, 255).astype(np.uint8), 90),
        }
        for name, (variant, quality) in variants.items():
            with self.subTest(name=name):
                path = self.temp / f"{name}.jpg"
                write_image(path, variant, [cv2.IMWRITE_JPEG_QUALITY, quality])
                result = self.compare(name, path, True)
                self.assertTrue(result["accepted"], result)
                self.assertGreaterEqual(result["metrics"]["background_coverage"], 0.45)

    def test_other_slides_are_rejected(self):
        for position in range(2, 8):
            with self.subTest(position=position):
                result = self.compare(f"different_slide_{position}", REAL_POST / "original" / f"{position:02d}.jpg", False)
                self.assertFalse(result["accepted"], result)

    def test_full_resolution_homography_maps_actual_crop_coordinates(self):
        scale = .7
        crop = cv2.resize(self.image[820:], (819, round((self.image.shape[0] - 820) * scale)), interpolation=cv2.INTER_AREA)
        path = self.temp / "homography_crop.jpg"
        write_image(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result = self.compare("full_resolution_homography", path, True)
        self.assertTrue(result["accepted"], result)
        matrix = np.asarray(result["metrics"]["homography_full_resolution"], dtype=np.float64)
        points = np.asarray([[[200, 80]], [[550, 230]], [[600, 380]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
        expected = points.reshape(-1, 2) / scale + [0, 820]
        self.assertLess(float(np.linalg.norm(projected - expected, axis=1).max()), 5.0)

    def test_original_with_overlay_is_rejected(self):
        result = self.compare("original_with_overlay", self.original, False)
        self.assertFalse(result["accepted"], result)
        self.assertEqual(result["reason"], "ORIGINAL_TEXT_STILL_PRESENT")

    def test_overlay_copy_stays_rejected_after_resize_jpeg_and_colour_change(self):
        variants = {
            "overlay_resized_jpeg": cv2.resize(self.image, (819, 1040), interpolation=cv2.INTER_AREA),
            "overlay_colour": np.clip(self.image.astype(np.float32) * 0.94 + 8, 0, 255).astype(np.uint8),
        }
        for name, variant in variants.items():
            with self.subTest(name=name):
                path = self.temp / f"{name}.jpg"
                write_image(path, variant, [cv2.IMWRITE_JPEG_QUALITY, 35])
                result = self.compare(name, path, False)
                self.assertFalse(result["accepted"], result)
                self.assertEqual(result["reason"], "ORIGINAL_TEXT_STILL_PRESENT")

    def test_small_matching_crop_has_insufficient_coverage(self):
        path = self.temp / "small_background_crop.jpg"
        write_image(path, self.image[:550])
        result = self.compare("insufficient_crop_coverage", path, False)
        self.assertFalse(result["accepted"], result)
        self.assertEqual(result["reason"], "INSUFFICIENT_BACKGROUND_COVERAGE")

    def test_comparing_existing_clean_only_returns_diagnostics_without_exporting_it(self):
        result = self.compare("supplied_existing_clean", REAL_POST / "clean" / "01_CLEAN.jpg", False)
        self.assertGreater(result['score'], 0)
        self.assertNotIn('downloaded', result)

    def test_comparing_clean_does_not_create_or_export_another_clean(self):
        clean = self.temp / "01_CLEAN.jpg"
        mask = self.temp / "01_SAVED_MASK.png"
        info = make_clean_image(self.original, clean, self.polygons, mask)
        self.assertGreater(info["mask_pixels"], 0)
        np.testing.assert_array_equal(read_image(self.mask, cv2.IMREAD_GRAYSCALE), read_image(mask, cv2.IMREAD_GRAYSCALE))
        before = clean.read_bytes()
        from unittest.mock import patch
        with patch('src.image.similarity.cv2.inpaint') as inpaint:
            result = self.compare("telea_fallback_copy", clean, False)
        inpaint.assert_not_called()
        self.assertGreater(result['score'], 0)
        self.assertNotIn('downloaded', result)
        self.assertEqual(clean.read_bytes(), before)

    def test_tiny_thumbnail_still_receives_a_visual_score(self):
        path = self.temp / "thumbnail.jpg"
        write_image(path, cv2.resize(self.image, (240, 305)))
        result = self.compare("tiny_thumbnail", path, False)
        self.assertGreater(result['score'], 0)
        self.assertTrue(result['metrics']['thumbnail'])

    def test_missing_and_invalid_mask_return_diagnostics(self):
        result = compare_candidate(self.original, self.temp / "missing.jpg", self.mask)
        self.assertEqual(result["reason"], "UNREADABLE_IMAGE")
        tiny_mask = self.temp / "invalid_mask.png"
        write_image(tiny_mask, np.zeros((8, 8), dtype=np.uint8))
        result = compare_candidate(self.original, self.original, tiny_mask)
        self.assertEqual(result["reason"], "MASK_SIZE_MISMATCH")

    def test_cleaning_preserves_existing_telea_algorithm(self):
        mask = build_text_mask(self.image.shape, self.polygons)
        height, width = self.image.shape[:2]
        pad = max(5, round(min(height, width) * 0.008))
        expected = cv2.inpaint(self.image, mask, max(4, pad // 2), cv2.INPAINT_TELEA)
        expected_file = self.temp / "expected.jpg"
        output_file = self.temp / "actual.jpg"
        write_image(expected_file, expected, [cv2.IMWRITE_JPEG_QUALITY, 96])
        make_clean_image(self.original, output_file, self.polygons)
        self.assertEqual(expected_file.read_bytes(), output_file.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
