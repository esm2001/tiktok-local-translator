"""RAW recovery: OCR ranks candidates without blocking a web download."""

import hashlib
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "translator"))
from src.config import Settings
from src.image.cleaning import build_text_mask, read_image, write_image
from src.recovery import recover_slide, candidate_overlay_check, prepare_reverse_query, revalidate_saved_source, download_candidate

REAL_POST = Path(os.environ.get("TIKTOK_TEST_POST", str(PROJECT.parent / "TikTokTraducidos" / "7659483315203804449")))


def find_archived_overlay(post: Path = REAL_POST) -> Path | None:
    original_location = post / "translated" / "01_ES.jpg"
    if original_location.is_file():
        return original_location
    archive_root = post.parent / ".cache" / "legacy" / post.name
    archived = list(archive_root.glob("*/translated/01_ES.jpg")) if archive_root.is_dir() else []
    return max(archived, key=lambda path: path.stat().st_mtime_ns) if archived else None


@unittest.skipUnless((REAL_POST / "original" / "01.jpg").exists(), "Local real carousel fixture is unavailable")
class RealCandidateRecoveryTests(unittest.TestCase):
    def setUp(self):
        overlay = find_archived_overlay()
        if overlay is None:
            self.skipTest("Archived overlay fixture is unavailable in the post and its legacy archive")
        temporary = tempfile.TemporaryDirectory(prefix="tiktok_recovery_")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.settings = Settings(output_base=self.base)
        for folder in ("original", "clean", "metadata/masks", "recovered"):
            (self.base / folder).mkdir(parents=True)
        self.original = self.base / "original" / "01.jpg"
        shutil.copyfile(REAL_POST / "original" / "01.jpg", self.original)
        shutil.copyfile(REAL_POST / "clean" / "01_CLEAN.jpg", self.base / "clean" / "01_CLEAN.jpg")
        baseline = PROJECT / ".cache" / "baseline.json"
        if baseline.exists():
            self.polygons = json.loads(baseline.read_text(encoding="utf-8"))["slides"][0]["polygons"]
        else:
            self.polygons = [[[145, 593], [1040, 593], [1040, 792], [145, 792]]]
        image = read_image(self.original)
        mask = self.base / "metadata" / "masks" / "01_MASK.png"
        write_image(mask, build_text_mask(image.shape, self.polygons))
        self.slide = {"position": 1, "original": "original/01.jpg", "clean": "clean/01_CLEAN.jpg", "ocr": {"mask": "metadata/masks/01_MASK.png"}}
        self.good = self.base / "actual_background_crop.jpg"
        # Strong but compressed positive scores below the archived-overlay false
        # visual match, so recovery must reject that first and try this candidate.
        write_image(self.good, image[820:], [cv2.IMWRITE_JPEG_QUALITY, 10])
        self.overlay = overlay
        self.source_ocr = Mock(side_effect=self.ocr_fixture)

    def ocr_fixture(self, path, settings):
        polygons = self.polygons if read_image(path).shape[0] > 1000 else []
        return {"detected_polygons": polygons, "recognized_lines": []}

    @staticmethod
    def candidate(name, path):
        return {"image_url": f"https://example.test/images/{name}.jpg", "source_url": f"https://example.test/{name}", "fixture": str(path)}

    @staticmethod
    def download_fixture(candidate, target):
        if candidate.get("fail"):
            return {"candidate": candidate, "error": "HTTP failure"}
        shutil.copyfile(candidate["fixture"], target)
        return {"candidate": candidate, "path": target, "bytes": target.stat().st_size}

    def recover(self, candidates):
        with patch("src.recovery.download_candidate", side_effect=self.download_fixture), patch("src.recovery.ocr_image", self.source_ocr):
            return recover_slide(self.base, self.slide, {"provider": "google_lens", "status": "CANDIDATES", "candidates": candidates}, self.settings)

    def test_web_image_with_text_is_downloaded_unchanged_when_only_candidate(self):
        result = self.recover([self.candidate("archived_overlay", self.overlay)])
        self.assertEqual(result["status"], "MATCH", result)
        self.assertTrue(result["contains_text"])
        self.source_ocr.assert_called_once()
        self.assertEqual((self.base / result["downloaded"]).read_bytes(), self.overlay.read_bytes())

    def test_next_verified_candidate_is_used_after_overlay_rejection(self):
        result = self.recover([self.candidate("archived_overlay", self.overlay), self.candidate("actual_crop", self.good)])
        self.assertEqual(result["status"], "MATCH", result)
        self.assertTrue(result["image_url"].endswith("/actual_crop.jpg"))
        self.assertGreater(result["candidates"][0]["ranking_penalties"]["text"], 0)
        self.assertEqual(self.source_ocr.call_count, 2)
        destination = self.base / result["downloaded"]
        self.assertEqual(destination.read_bytes(), self.good.read_bytes())
        self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(), result["sha256"])

    def test_failed_candidate_ocr_still_saves_raw_source(self):
        self.source_ocr.side_effect = RuntimeError("Candidate OCR unavailable")
        result = self.recover([self.candidate("actual_crop", self.good)])
        self.assertEqual(result["status"], "MATCH", result)
        self.assertIn("error", result["validation"]["source_ocr"])
        self.assertEqual(result["fallback"], self.slide["clean"])

    def test_failed_download_does_not_prevent_next_match(self):
        broken = {**self.candidate("failed_download", self.overlay), "fail": True}
        result = self.recover([broken, self.candidate("actual_crop", self.good)])
        self.assertEqual(result["status"], "MATCH", result)
        self.assertEqual(result["candidates"][0]["reason"], "DOWNLOAD_FAILED")
        self.source_ocr.assert_called_once()

    def test_low_similarity_only_candidate_is_saved_with_low_confidence(self):
        result = self.recover([self.candidate("different_photo", REAL_POST / "original" / "02.jpg")])
        self.assertEqual(result["status"], "MATCH", result)
        self.assertEqual(result["confidence"], "LOW")
        self.source_ocr.assert_called_once()

    def test_captcha_is_reported_without_candidate_downloads(self):
        with patch("src.recovery.download_candidate") as downloader:
            result = recover_slide(self.base, self.slide, {"provider": "google_lens", "status": "CAPTCHA", "candidates": []}, self.settings)
        self.assertEqual(result["status"], "CAPTCHA")
        self.assertEqual(result["fallback"], self.slide["clean"])
        downloader.assert_not_called()

    def test_fourth_verified_candidate_is_not_lost_after_three_overlay_rejections(self):
        candidates = [self.candidate(f'overlay_{index}', self.overlay) for index in range(3)]
        candidates.append(self.candidate('actual_crop', self.good))
        result = self.recover(candidates)
        self.assertEqual(result['status'], 'MATCH', result)
        self.assertTrue(result['image_url'].endswith('/actual_crop.jpg'))
        self.assertEqual(self.source_ocr.call_count, 4)
        self.assertEqual(result['candidates_checked'], 4)

    def saved_source(self, image):
        destination = self.base/'recovered/01_SOURCE.jpg'
        shutil.copyfile(image, destination)
        original_hash = hashlib.sha256(self.original.read_bytes()).hexdigest()
        self.slide['original_sha256'] = original_hash
        prior = {'original_sha256': original_hash, 'reverse_search': {'status': 'MATCH',
            'downloaded': 'recovered/01_SOURCE.jpg', 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'score': .99, 'matcher_version': 'obsolete', 'source_url': 'https://fixture.example/photo.jpg',
            'image_url': 'https://fixture.example/photo.jpg', 'source': 'web_download', 'raw_download': True, 'modified': False}}
        return prior,destination

    def test_saved_source_is_revalidated_after_version_change_without_a_download(self):
        prior,destination = self.saved_source(self.good)
        before = destination.read_bytes()
        with patch('src.recovery.download_candidate') as downloader, patch('src.recovery.ocr_image', self.source_ocr):
            result = revalidate_saved_source(self.base, prior, self.slide, self.settings)
        self.assertEqual(result['status'], 'MATCH', result)
        self.assertTrue(result['cache_revalidated'])
        self.assertNotEqual(result['matcher_version'], 'obsolete')
        self.assertEqual(destination.read_bytes(), before)
        downloader.assert_not_called()
        self.source_ocr.assert_not_called()
        self.assertEqual(prior['reverse_search']['matcher_version'], 'obsolete')

    def test_saved_source_hash_change_is_rejected_before_candidate_ocr(self):
        prior,destination = self.saved_source(self.good)
        shutil.copyfile(self.overlay, destination)
        with patch('src.recovery.ocr_image', self.source_ocr):
            result = revalidate_saved_source(self.base, prior, self.slide, self.settings)
        self.assertIsNone(result)
        self.source_ocr.assert_not_called()

    def test_saved_raw_captioned_source_reuses_bytes_without_running_ocr(self):
        prior,destination = self.saved_source(self.overlay)
        with patch('src.recovery.ocr_image', self.source_ocr):
            result = revalidate_saved_source(self.base, prior, self.slide, self.settings)
        self.assertEqual(result['status'], 'MATCH')
        self.source_ocr.assert_not_called()
        self.assertEqual(destination.read_bytes(), self.overlay.read_bytes())

    def test_unknown_download_extension_cannot_silently_skip_paddle_detection(self):
        from src.ocr.paddle import ocr_image

        candidate = self.base / "downloaded_candidate.img"
        shutil.copyfile(self.overlay, candidate)

        def paddle_input_sampler(image):
            # Match the installed Paddle ImageBatchSampler contract: a .img path
            # silently yields an empty batch, whereas decoded arrays are accepted.
            if isinstance(image, str) and Path(image).suffix == ".img":
                return []
            self.assertTrue(isinstance(image, np.ndarray) or Path(image).suffix in {".jpg", ".png", ".webp"})
            return [{"dt_polys": self.polygons, "rec_texts": ["Editorial overlay"], "rec_scores": [.99]}]

        engine = Mock()
        engine.predict.side_effect = paddle_input_sampler
        with patch("src.ocr.paddle.get_ocr", return_value=engine):
            result = ocr_image(candidate, self.settings)
        self.assertTrue(result["detected_polygons"], "Unknown suffix must be decoded before Paddle sampling")
        engine.predict.assert_called_once()

    def test_empty_paddle_batch_cannot_be_treated_as_overlay_free_image(self):
        from src.ocr.paddle import ocr_image

        engine = Mock()
        engine.predict.return_value = []
        with patch("src.ocr.paddle.get_ocr", return_value=engine):
            with self.assertRaisesRegex(RuntimeError, "no ha procesado"):
                ocr_image(self.original, self.settings)

    def test_overlay_fixture_can_be_read_after_legacy_archive_migration(self):
        post = self.base / "fixture_outputs" / REAL_POST.name
        archive = post.parent / ".cache" / "legacy" / post.name / "migration-test" / "translated"
        archive.mkdir(parents=True)
        destination = archive / "01_ES.jpg"
        shutil.copyfile(self.overlay, destination)
        self.assertFalse((post / "translated").exists())
        self.assertEqual(find_archived_overlay(post), destination)


class OverlayEvidenceTests(unittest.TestCase):
    def test_primary_source_ocr_preserves_original_slide_adaptive_and_crop_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            mask = Path(folder)/'mask.png'
            write_image(mask, np.zeros((300, 300), dtype=np.uint8))
            settings = Settings(ocr_adaptive_fallback=True, ocr_max_crop_retries=6)
            with patch('src.recovery.ocr_image', return_value={'detected_polygons': [], 'recognized_lines': []}) as ocr:
                candidate_overlay_check(Path('original.jpg'), Path('candidate.jpg'), mask,
                                        {'metrics': {'homography_full_resolution': np.eye(3).tolist()}}, settings)
            source_settings = ocr.call_args.args[1]
            self.assertFalse(source_settings.ocr_adaptive_fallback)
            self.assertEqual(source_settings.ocr_max_crop_retries, 0)
            self.assertTrue(settings.ocr_adaptive_fallback)
            self.assertEqual(settings.ocr_max_crop_retries, 6)

    def test_text_query_uses_editorial_context_without_background_equations(self):
        slide = {'position': 1, 'ocr': {'lines': [{'text': 'background = x^2', 'confidence': .99}],
                                      'editorial_lines': [{'text': 'Nikola Tesla', 'confidence': .99},
                                                          {'text': 'Who invented the future?', 'confidence': .99}]}}
        query = prepare_reverse_query(Path('.'), slide, 'text_assisted', 5)
        self.assertEqual(query['text_query'], 'Nikola Tesla Who invented the future?')

    def test_english_ocr_diagonal_wire_symbols_are_preserved_as_noise_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            mask = Path(folder) / 'mask.png'
            write_image(mask, np.full((300, 300), 255, dtype=np.uint8))
            wire = [[30, 20], [200, 190], [185, 205], [15, 35]]
            comparison = {'metrics': {'homography_full_resolution': np.eye(3).tolist()}}
            for text, confidence in [('LRU', .573), ('m', .704), ('000000300030000', .502)]:
                with self.subTest(text=text), patch('src.recovery.ocr_image', return_value={
                        'detected_polygons': [wire], 'recognized_lines': [{'polygon': wire, 'text': text, 'confidence': confidence}]}):
                    result = candidate_overlay_check(Path('original.jpg'), Path('candidate.jpg'), mask, comparison, Settings())
                self.assertTrue(result['accepted'], result)
                self.assertEqual(result['ignored_detection_regions'][0]['text'], text)

    def test_rotated_words_and_reliable_short_labels_still_reject_captioned_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            mask = Path(folder) / 'mask.png'
            write_image(mask, np.full((300, 300), 255, dtype=np.uint8))
            polygon = [[30, 20], [200, 190], [185, 205], [15, 35]]
            comparison = {'metrics': {'homography_full_resolution': np.eye(3).tolist()}}
            for text, confidence in [('Editorial caption', .55), ('LR', .98)]:
                with self.subTest(text=text), patch('src.recovery.ocr_image', return_value={
                        'detected_polygons': [polygon], 'recognized_lines': [{'polygon': polygon, 'text': text, 'confidence': confidence}]}):
                    result = candidate_overlay_check(Path('original.jpg'), Path('candidate.jpg'), mask, comparison, Settings())
                self.assertFalse(result['accepted'], result)

    def test_tesla_diagonal_detections_without_characters_are_not_overlay_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            mask = Path(folder) / 'mask.png'
            write_image(mask, np.full((300, 300), 255, dtype=np.uint8))
            false_wire = [[30, 20], [200, 190], [185, 205], [15, 35]]
            comparison = {'metrics': {'homography_full_resolution': np.eye(3).tolist()}}
            with patch('src.recovery.ocr_image', return_value={'detected_polygons': [false_wire], 'recognized_lines': []}):
                result = candidate_overlay_check(Path('original.jpg'), Path('candidate.jpg'), mask, comparison, Settings())
            self.assertTrue(result['accepted'], result)
            self.assertEqual(len(result['ignored_detection_regions']), 1)

    def test_unrecognized_horizontal_caption_stays_uncertain_and_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            mask = Path(folder) / 'mask.png'
            write_image(mask, np.full((300, 300), 255, dtype=np.uint8))
            caption = [[10, 80], [250, 80], [250, 130], [10, 130]]
            comparison = {'metrics': {'homography_full_resolution': np.eye(3).tolist()}}
            with patch('src.recovery.ocr_image', return_value={'detected_polygons': [caption], 'recognized_lines': []}):
                result = candidate_overlay_check(Path('original.jpg'), Path('candidate.jpg'), mask, comparison, Settings())
            self.assertFalse(result['accepted'])
            self.assertTrue(result['overlaid_regions'][0]['uncertain_detection'])

    def test_crop_query_uses_only_unmasked_original_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base/'original').mkdir()
            (base/'metadata/masks').mkdir(parents=True)
            original = np.random.default_rng(7).integers(0,256,(1000, 700, 3),dtype=np.uint8)
            mask = np.zeros((1000, 700), dtype=np.uint8)
            mask[400:650] = 255
            write_image(base/'original/01.jpg', original)
            write_image(base/'metadata/masks/01.png', mask)
            slide = {'position': 1, 'original': 'original/01.jpg', 'clean': 'clean/01.jpg', 'ocr': {'mask': 'metadata/masks/01.png'}}
            for index in (1, 2):
                query = prepare_reverse_query(base, slide, f'clean_crop_{index}', index+2)
                self.assertFalse(query.get('skipped'), query)
                x0,y0,x1,y1 = query['crop_box']
                self.assertFalse(np.any(mask[y0:y1,x0:x1]))
                self.assertTrue((base/query['query_file']).is_file())


class RecoveryTimeBudgetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        (self.base/'metadata').mkdir()
        self.slide = {'position':1,'original':'original/01.jpg','clean':'clean/01.jpg','ocr':{'mask':'mask.png'}}
        self.settings = Settings(candidate_workers=1)
        self.clock = [0.0]
        self.time_patch = patch('src.recovery.time.monotonic',side_effect=lambda:self.clock[0])
        self.time_patch.start();self.addCleanup(self.time_patch.stop)
        self.search = {'status':'CANDIDATES','_validation_deadline':10.0,'candidates':[
            {'image_url':f'https://fixture.example/{i}.jpg'} for i in range(3)]}

    def download(self,candidate,target,deadline=None):
        write_image(target, np.random.default_rng(5).integers(0,256,(100,100,3),dtype=np.uint8))
        return {'candidate':candidate,'path':target,'bytes':target.stat().st_size}

    def test_expired_budget_skips_all_downloads_and_preserves_full_candidate_audit(self):
        self.clock[0] = 11.0
        with patch('src.recovery.download_candidate') as download, patch('src.recovery.compare_candidate') as matcher:
            result = recover_slide(self.base,self.slide,self.search,self.settings)
        self.assertEqual(result['status'],'TIMEOUT')
        self.assertTrue(result['budget_exhausted'])
        self.assertEqual(len(result['candidates']),3)
        self.assertTrue(all(c['reason']=='TIME_BUDGET_EXHAUSTED' for c in result['candidates']))
        download.assert_not_called();matcher.assert_not_called()
        self.assertFalse((self.base/'recovered/01_SOURCE.jpg').exists())

    def test_queued_downloads_do_not_start_after_first_download_consumes_budget(self):
        def download(*args,**kwargs):
            result = self.download(*args,**kwargs)
            self.clock[0] = 11.0
            return result
        with patch('src.recovery.download_candidate',side_effect=download) as transfer, patch('src.recovery.compare_candidate') as matcher:
            result = recover_slide(self.base,self.slide,self.search,self.settings)
        self.assertEqual(result['status'],'MATCH')
        self.assertEqual(len(result['candidates']),3)
        transfer.assert_called_once()

    def test_expired_ranking_does_not_discard_already_downloaded_bytes(self):
        def compare(*args, **kwargs):
            self.clock[0] = 11.0
            return {'accepted':False, 'score':.54, 'reason':'BACKGROUND_DIFFERS', 'width':100, 'height':100, 'metrics':{}}
        with patch('src.recovery.download_candidate', side_effect=self.download), patch('src.recovery.compare_candidate', side_effect=compare), patch('src.recovery.candidate_overlay_check', side_effect=RuntimeError('OCR unavailable')):
            result = recover_slide(self.base, self.slide, self.search, self.settings)
        self.assertEqual(result['status'], 'MATCH')
        self.assertEqual(result['confidence'], 'LOW')
        self.assertTrue((self.base / result['downloaded']).is_file())

    def test_source_ocr_after_deadline_still_exports_raw_source(self):
        def check(*args, **kwargs):
            self.clock[0] = 11.0
            return {'accepted':False, 'candidate_text_area_ratio':.2, 'contains_text':True}
        comparison = {'accepted':False, 'score':.54, 'reason':'BACKGROUND_DIFFERS', 'width':100, 'height':100, 'metrics':{}}
        with patch('src.recovery.download_candidate',side_effect=self.download), patch('src.recovery.compare_candidate',return_value=comparison), patch('src.recovery.candidate_overlay_check',side_effect=check):
            result = recover_slide(self.base,self.slide,self.search,self.settings)
        self.assertEqual(result['status'],'MATCH')
        self.assertTrue(result['raw_download'])
        self.assertFalse(result['modified'])
        self.assertIsNotNone(result['downloaded'])

    def test_candidate_redirect_is_not_followed_after_total_budget_expires(self):
        response = Mock()
        response.is_redirect = True
        response.status_code = 302
        response.headers = {'Location':'https://fixture.example/redirected.jpg'}
        def get(*args,**kwargs):
            self.clock[0] = 11.0
            manager = Mock();manager.__enter__ = Mock(return_value=response);manager.__exit__ = Mock(return_value=False)
            return manager
        with patch('src.recovery.public_image_url'), patch('src.recovery.requests.get',side_effect=get) as fetch:
            result = download_candidate(self.search['candidates'][0],self.base/'candidate.jpg',deadline=10.0)
        self.assertTrue(result['budget_exhausted'])
        fetch.assert_called_once()
        self.assertFalse((self.base/'candidate.jpg').exists())

    def test_partial_candidate_stream_is_removed_when_deadline_expires(self):
        response = Mock()
        response.is_redirect = False
        response.status_code = 200
        response.headers = {'Content-Type':'image/jpeg'}
        chunks = iter([b'first',b'second'])
        def read(*args,**kwargs):
            chunk = next(chunks)
            if chunk==b'second':
                self.clock[0] = 11.0
            return chunk
        response.raw.read1.side_effect = read
        manager = Mock();manager.__enter__ = Mock(return_value=response);manager.__exit__ = Mock(return_value=False)
        with patch('src.recovery.public_image_url'), patch('src.recovery.requests.get',return_value=manager) as fetch:
            result = download_candidate(self.search['candidates'][0],self.base/'candidate.jpg',deadline=10.0)
        self.assertTrue(result['budget_exhausted'])
        self.assertEqual(response.raw.read1.call_count,2)
        self.assertEqual(fetch.call_args.kwargs['timeout'],(7,10))
        self.assertFalse((self.base/'candidate.jpg').exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
