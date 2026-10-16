"""Local pipeline regressions: real photographs/masks, mocked external services."""

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "translator"))

from src.config import Settings
from src.output.manifest import read_json, write_json
from src.pipeline import completed_cache, reusable_source, run_pipeline, translation_retry_due, recover_with_strategy
from src.tiktok.downloader import DownloadError

REAL_POST = Path(os.environ.get("TIKTOK_TEST_POST", str(PROJECT.parent / "TikTokTraducidos" / "7659483315203804449")))
POST_ID = "7659483315203804449"
URL = f"https://www.tiktok.com/@socratesdiedforthis/photo/{POST_ID}"


@unittest.skipUnless((REAL_POST / "original" / "01.jpg").exists(), "Local real carousel fixture is unavailable")
class RealImagePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tiktok_pipeline_")
        self.addCleanup(self.temporary.cleanup)
        self.settings = Settings(output_base=Path(self.temporary.name) / "outputs")
        self.base = self.settings.output_base / POST_ID
        self.manifest_path = self.base / "metadata" / "manifest.json"
        self.source = REAL_POST / "original" / "01.jpg"
        baseline_path = PROJECT / ".cache" / "baseline.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["slides"][0]
            self.ocr_data = {"detected_polygons": baseline["polygons"], "recognized_lines": [
                line if isinstance(line, dict) else {"text": line, "confidence": .99} for line in baseline["lines"]
            ]}
        else:
            self.ocr_data = {
                "detected_polygons": [[[145, 593], [1040, 593], [1040, 792], [145, 792]]],
                "recognized_lines": [{"text": "Your biggest red flag =", "confidence": .99}, {"text": "Your philosopher.", "confidence": .99}],
            }
        self.download = self.start_patch("src.pipeline.download_images", side_effect=self.fake_download)
        self.resolve = self.start_patch("src.pipeline.resolve_tiktok_url", side_effect=lambda url: url)
        self.ocr = self.start_patch("src.pipeline.ocr_image", side_effect=lambda *args: copy.deepcopy(self.ocr_data))
        self.translator_class = self.start_patch("src.pipeline.OllamaTranslator")
        self.translator_class.return_value.metrics = {"eval_count": 20}
        self.translator_class.return_value.translate.return_value = {
            "1": {"original": "Your biggest red flag = Your philosopher.", "spanish": "Tu mayor señal de alarma = tu filósofo."}
        }
        self.start_patch("src.pipeline.logger")
        # These tests isolate scheduling/cache decisions. Fresh image verification
        # has its own real-photo tests in test_recovery, including failed hashes.
        self.source_revalidation = self.start_patch('src.pipeline.revalidate_saved_source', return_value=None)
        self.reverse = Mock(return_value={1: {"provider": "google_lens", "status": "CANDIDATES",
            "candidates": [{"image_url": "https://fixture.example/background.jpg"}]}})
        self.recovery = self.start_patch('src.pipeline.recover_slide', side_effect=self.fake_recovery)

    def start_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def fake_download(self, url, directory, cookies):
        destination = directory / "01.jpg"
        shutil.copyfile(self.source, destination)
        return [destination], "Local fixture; gallery-dl mocked."

    def fake_verified_source(self, base, slide):
        from src.image.cleaning import read_image

        # A recorded web response stands in for the downloader. No pixel edits or
        # re-encoding occur at recovery, even in this scheduling/cache fixture.
        fixture = PROJECT / '.cache/recovery-public-probe/nietzsche-03-candidate-01.jpg'
        if not fixture.is_file():
            fixture = REAL_POST / 'original/02.jpg'
        relative = f"recovered/{slide['position']:02d}_SOURCE.jpg"
        path = base / relative
        shutil.copyfile(fixture, path)
        candidate = read_image(path)
        return {'provider': 'google_lens', 'status': 'MATCH', 'score': .95,
            'downloaded': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_url': 'https://example.com/public-photo.jpg',
            'image_url': 'https://example.com/public-photo.jpg', 'source': 'web_download',
            'raw_download': True, 'modified': False,
            'resolution': {'width': candidate.shape[1], 'height': candidate.shape[0]},
            'validation': {'source_ocr': {'accepted': True, 'ranking_only': True}}, 'error': '', 'candidates': []}

    def fake_recovery(self, base, slide, search, settings):
        if search['status'] == 'CANDIDATES' and search.get('candidates'):
            return self.fake_verified_source(base, slide)
        return {'provider': 'google_lens', 'status': search['status'],
            'downloaded': None, 'score': 0, 'raw_download': False, 'modified': False,
            'error': search.get('error', ''), 'candidates': []}

    def prepare_source_cache(self, positive_positions=(1,)):
        def two_downloads(url, directory, cookies):
            paths = []
            for index in (1, 2):
                destination = directory / f'{index:02d}.jpg'
                shutil.copyfile(REAL_POST / 'original' / destination.name, destination)
                paths.append(destination)
            return paths, 'Two real photographs; downloader mocked.'

        def recovery(base, slide, search, settings):
            if search['status'] == 'CANDIDATES':
                return self.fake_verified_source(base, slide)
            return {'provider': 'google_lens', 'status': search['status'],
                'downloaded': None, 'score': 0, 'fallback': slide['clean'],
                'error': search.get('error', ''), 'candidates': []}

        self.download.side_effect = two_downloads
        self.reverse.side_effect = lambda base, slides: {
            slide['position']: {'status': 'CANDIDATES' if slide['position'] in positive_positions else 'ERROR',
                'error': '' if slide['position'] in positive_positions else 'Temporary Lens timeout', 'candidates': []}
            for slide in slides}
        self.recovery = self.start_patch('src.pipeline.recover_slide', side_effect=recovery)
        self.run_expected('OK' if set(positive_positions) == {1, 2} else 'INCOMPLETE')
        return read_json(self.manifest_path)

    def run_success(self):
        return self.run_expected('OK')

    def run_expected(self, expected):
        result = run_pipeline(URL, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result["status"], expected, result)
        return result

    def test_complete_cache_makes_no_external_calls(self):
        self.run_success()
        for mock in (self.download, self.resolve, self.ocr, self.translator_class, self.reverse):
            mock.reset_mock()
        result = run_pipeline(URL, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result["status"], "OK_CACHED", result)
        for mock in (self.download, self.resolve, self.ocr, self.translator_class, self.reverse):
            mock.assert_not_called()

    def test_exhausted_negative_cache_expires_and_retries_without_redownload(self):
        self.reverse.return_value = {1: {'status': 'NOT_FOUND', 'candidates': []}}
        self.run_expected('INCOMPLETE')
        manifest = read_json(self.manifest_path)
        recovery = manifest['slides'][0]['reverse_search']
        self.assertEqual(recovery['status'], 'NOT_FOUND')
        self.assertTrue(recovery['strategy_exhausted'])
        self.assertFalse(completed_cache(self.base, manifest, self.settings, True))
        recovery['searched_at'] = time.time()-self.settings.reverse_negative_ttl-1
        self.assertFalse(completed_cache(self.base, manifest, self.settings, True))
        write_json(self.manifest_path, manifest)
        for service in (self.download, self.ocr, self.translator_class, self.reverse):
            service.reset_mock()
        self.run_expected('INCOMPLETE')
        retried = read_json(self.manifest_path)['slides'][0]['reverse_search']
        self.assertEqual([a['query']['kind'] for a in retried['attempts']],
                         ['clean_full', 'original_full', 'clean_crop_1', 'clean_crop_2', 'google_images', 'text_assisted'])
        self.assertEqual(self.reverse.call_count, sum(a['status'] != 'SKIPPED' for a in retried['attempts']))
        self.assertEqual(self.reverse.call_count, retried['queries_attempted'])
        self.download.assert_not_called()
        self.ocr.assert_not_called()
        self.translator_class.assert_not_called()

    def test_low_score_raw_candidate_is_saved_without_additional_searches(self):
        from src.recovery import recover_slide
        fixture = REAL_POST / 'original/02.jpg'
        attempted = []
        def reverse(base, slides):
            query = copy.deepcopy(slides[0]['reverse_query'])
            attempted.append(query['kind'])
            return {1: {'provider': 'google_lens', 'status': 'CANDIDATES',
                        'candidates': [{'image_url': 'https://fixture.example/low-score.jpg'}]}}
        def download(candidate, target, deadline=None):
            shutil.copyfile(fixture, target)
            return {'candidate': candidate, 'path': target, 'bytes': target.stat().st_size,
                    'http_status': 200, 'content_type': 'image/jpeg'}
        self.reverse.side_effect = reverse
        self.recovery.side_effect = recover_slide
        with patch('src.recovery.download_candidate', side_effect=download), patch('src.recovery.ocr_image',
                   return_value={'detected_polygons': [], 'recognized_lines': []}), patch('src.recovery.compare_candidate',
                   return_value={'accepted': False, 'score': .54, 'metrics': {}}):
            self.run_success()
        recovery = read_json(self.manifest_path)['slides'][0]['reverse_search']
        self.assertEqual(attempted, ['clean_full'])
        self.assertEqual(recovery['status'], 'MATCH')
        self.assertEqual(recovery['confidence'], 'LOW')
        self.assertEqual([attempt['status'] for attempt in recovery['attempts']], ['MATCH'])
        self.assertEqual(recovery['candidates_checked'], 1)
        self.assertTrue(recovery['candidates'][0]['accepted'])
        self.assertFalse(recovery['candidates'][0]['visual_accepted'])
        self.assertTrue(recovery['raw_download'])
        self.assertFalse(recovery['modified'])
        self.assertEqual((self.base/recovery['downloaded']).read_bytes(), fixture.read_bytes())

    def test_captcha_tries_available_queries_and_keeps_clean_separate(self):
        self.reverse.return_value = {1: {'provider': 'google_lens', 'status': 'CAPTCHA',
                                        'error': 'Manual CAPTCHA required.', 'candidates': []}}
        self.run_expected('INCOMPLETE')
        slide = read_json(self.manifest_path)['slides'][0]
        self.assertEqual(self.reverse.call_count, slide['reverse_search']['queries_attempted'])
        self.assertEqual([entry['query']['kind'] for entry in slide['reverse_search']['attempts']],
                         ['clean_full', 'original_full', 'clean_crop_1', 'clean_crop_2', 'google_images', 'text_assisted'])
        self.assertEqual(slide['reverse_search']['status'], 'CAPTCHA')
        self.assertFalse(slide['reverse_search']['strategy_exhausted'])
        self.assertIsNone(slide['best_image'])
        self.assertTrue((self.base / slide['clean']).is_file())

    def test_refinement_upgrade_repeats_detector_and_retranslates_without_redownload(self):
        polygon = [[146, 776], [918, 780], [918, 901], [145, 897]]
        self.source = REAL_POST / 'original' / '07.jpg'
        self.ocr_data = {'detected_polygons': [polygon], 'recognized_lines': [
            {'text': 'Which one are your', 'confidence': .899, 'polygon': polygon}
        ]}
        self.run_success()
        previous = read_json(self.manifest_path)
        previous['signature'].pop('ocr_refinement', None)
        write_json(self.manifest_path, previous)
        original_hash = previous['slides'][0]['original_sha256']
        clean = self.base / previous['slides'][0]['clean']
        mask = self.base / previous['slides'][0]['ocr']['mask']
        clean_bytes, mask_bytes = clean.read_bytes(), mask.read_bytes()
        self.ocr.reset_mock()
        self.download.reset_mock()
        self.translator_class.reset_mock()
        self.translator_class.return_value.translate.return_value = {
            '1': {'original': 'Which one are you?', 'spanish': '¿Cuál eres tú?'}
        }
        self.ocr_data['recognized_lines'][0].update(text='Which one are you?', confidence=.953)
        self.run_success()
        self.ocr.assert_called_once()
        cached = run_pipeline(URL, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(cached['status'], 'OK_CACHED', cached)
        self.ocr.assert_called_once()
        self.download.assert_not_called()
        self.translator_class.return_value.translate.assert_called_once()
        current = read_json(self.manifest_path)
        self.assertEqual(current['slides'][0]['original_sha256'], original_hash)
        self.assertEqual(current['slides'][0]['ocr']['polygons'], [polygon])
        self.assertEqual(current['slides'][0]['ocr']['lines'][0]['text'], 'Which one are you?')
        self.assertNotEqual(current['translation_input_sha256'], previous['translation_input_sha256'])
        self.assertEqual(clean.read_bytes(), clean_bytes)
        self.assertEqual(mask.read_bytes(), mask_bytes)

    def test_refinement_version_upgrade_repeats_detection_even_with_confident_cached_lines(self):
        self.run_success()
        previous = read_json(self.manifest_path)
        previous['signature'].pop('ocr_refinement', None)
        write_json(self.manifest_path, previous)
        self.ocr.reset_mock()
        self.translator_class.reset_mock()
        with patch('src.ocr.paddle.get_ocr') as model:
            self.run_success()
        self.ocr.assert_called_once()
        model.assert_not_called()
        self.translator_class.assert_called_once()

    def test_old_manifest_without_polygons_repeats_ocr_once_and_keeps_download(self):
        original_folder = self.base / "original"
        original_folder.mkdir(parents=True)
        shutil.copyfile(self.source, original_folder / "01.jpg")
        write_json(self.manifest_path, {"post_id": POST_ID, "images": [{"position": 1, "ocr_lines": ["Old OCR"]}]})
        self.run_success()
        self.download.assert_not_called()
        self.ocr.assert_called_once()
        self.ocr.reset_mock()
        result = run_pipeline(URL, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result["status"], "OK_CACHED", result)
        self.ocr.assert_not_called()

    def test_missing_mask_reuses_saved_ocr_and_translation(self):
        self.run_success()
        manifest = read_json(self.manifest_path)
        (self.base / manifest["slides"][0]["ocr"]["mask"]).unlink()
        self.ocr.reset_mock()
        self.translator_class.reset_mock()
        result = run_pipeline(URL, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result["status"], "OK", result)
        self.ocr.assert_not_called()
        self.translator_class.assert_not_called()
        self.assertTrue((self.base / manifest["slides"][0]["ocr"]["mask"]).is_file())

    def test_no_text_keeps_web_raw_and_does_not_contact_ollama(self):
        from src.translation.ollama import OllamaTranslator

        self.ocr_data = {"detected_polygons": [], "recognized_lines": []}
        self.translator_class.side_effect = OllamaTranslator
        with patch("src.translation.ollama.requests.Session") as http:
            self.run_success()
            http.assert_not_called()
        manifest = read_json(self.manifest_path)
        self.assertIn("OCR_NO_TEXT", manifest["slides"][0]["warnings"])
        self.assertEqual(manifest["slides"][0]["best_image"], 'recovered/01_SOURCE.jpg')
        self.assertTrue(manifest['slides'][0]['reverse_search']['raw_download'])
        self.assertFalse(manifest['slides'][0]['reverse_search']['modified'])
        self.assertFalse((self.base / "translated").exists())
        self.assertTrue((self.base / "traducciones.txt").is_file())

    def test_reverse_callback_failure_keeps_carousel_translation(self):
        self.reverse.side_effect = RuntimeError("Google Lens CAPTCHA or timeout")
        result = run_pipeline(URL, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result["status"], "INCOMPLETE", result)
        manifest = read_json(self.manifest_path)
        self.assertNotEqual(manifest["slides"][0]["reverse_search"]["status"], "MATCH")
        self.assertIsNone(manifest["slides"][0]["best_image"])
        self.assertTrue((self.base / manifest['slides'][0]['clean']).is_file())
        self.translator_class.return_value.translate.assert_called_once()

    def test_force_download_failure_preserves_previous_valid_result(self):
        self.run_success()
        previous = {p: p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        self.download.side_effect = DownloadError("Network unavailable")
        result = run_pipeline(URL, force=True, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result["status"], "ERROR", result)
        for path, contents in previous.items():
            self.assertEqual(path.read_bytes(), contents, path)

    def test_force_translation_failure_preserves_previous_valid_result(self):
        self.run_success()
        previous = {p: p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        self.source = REAL_POST / "original" / "02.jpg"
        self.translator_class.return_value.translate.side_effect = RuntimeError("Ollama unavailable during refresh")
        result = run_pipeline(URL, force=True, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result["status"], "ERROR", result)
        for path, contents in previous.items():
            self.assertEqual(path.read_bytes(), contents, path)

    def test_force_no_images_preserves_previous_valid_result(self):
        self.run_success()
        previous = {p: p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        self.download.side_effect = None
        self.download.return_value = ([], "Video or no photo media")
        result = run_pipeline(URL, force=True, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result["status"], "SKIPPED_NO_IMAGES", result)
        for path, contents in previous.items():
            self.assertEqual(path.read_bytes(), contents, path)

    def test_force_incomplete_backup_never_replaces_previous_result(self):
        self.run_success()
        previous = {p: p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        with patch("src.pipeline.shutil.copytree", side_effect=PermissionError("Snapshot copy failed")):
            result = run_pipeline(URL, force=True, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result["status"], "ERROR", result)
        for path, contents in previous.items():
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.read_bytes(), contents, path)

    def test_transient_reverse_failure_is_retried_without_ocr_or_translation(self):
        self.reverse.side_effect = RuntimeError("Temporary Google Lens timeout")
        self.run_expected('INCOMPLETE')
        manifest = read_json(self.manifest_path)
        self.assertFalse(manifest["reverse_search_completed"])
        self.ocr.reset_mock()
        self.translator_class.reset_mock()
        self.reverse.reset_mock()
        self.reverse.side_effect = None
        result = run_pipeline(URL, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result["status"], "OK", result)
        recovery = read_json(self.manifest_path)['slides'][0]['reverse_search']
        self.assertEqual(self.reverse.call_count, recovery['queries_attempted'])
        self.assertEqual([a['query']['kind'] for a in recovery['attempts']], ['clean_full'])
        self.assertEqual(recovery['status'], 'MATCH')
        self.assertFalse(recovery['strategy_exhausted'])
        self.ocr.assert_not_called()
        self.translator_class.assert_not_called()
        self.assertTrue(read_json(self.manifest_path)['translation_cache_hit'])

    def test_corrupted_clean_is_repaired_without_ocr_or_translation(self):
        self.run_success()
        manifest = read_json(self.manifest_path)
        path = self.base / manifest["slides"][0]["clean"]
        expected = path.read_bytes()
        path.write_bytes(b"Damaged JPEG")
        self.ocr.reset_mock()
        self.translator_class.reset_mock()
        self.run_success()
        self.assertEqual(path.read_bytes(), expected)
        self.ocr.assert_not_called()
        self.translator_class.assert_not_called()

    def test_cache_rejects_corrupted_clean_or_mask(self):
        self.run_success()
        manifest = read_json(self.manifest_path)
        for key in ("clean", "mask"):
            with self.subTest(artifact=key):
                relative = manifest["slides"][0]["clean"] if key == "clean" else manifest["slides"][0]["ocr"]["mask"]
                path = self.base / relative
                original = path.read_bytes()
                path.write_bytes(b"This is not a valid image")
                self.assertFalse(completed_cache(self.base, manifest, self.settings, True))
                path.write_bytes(original)

    def test_cache_rejects_recovered_image_that_changed(self):
        self.run_success()
        manifest = read_json(self.manifest_path)
        manifest["slides"][0]["reverse_search"] = self.fake_verified_source(self.base, manifest['slides'][0])
        recovered = self.base / manifest['slides'][0]['reverse_search']['downloaded']
        self.assertTrue(completed_cache(self.base, manifest, self.settings, True))
        shutil.copyfile(REAL_POST / "original" / "02.jpg", recovered)
        self.assertFalse(completed_cache(self.base, manifest, self.settings, True))

    def test_partial_reverse_retry_keeps_verified_source_and_only_searches_pending_slide(self):
        previous = self.prepare_source_cache()
        source = self.base / previous['slides'][0]['reverse_search']['downloaded']
        source_bytes = source.read_bytes()
        self.assertFalse(previous['reverse_search_completed'])
        for mocked in (self.download, self.ocr, self.translator_class, self.reverse, self.recovery):
            mocked.reset_mock()
        self.reverse.side_effect = lambda base, slides: {
            slide['position']: {'status': 'NOT_FOUND', 'candidates': []} for slide in slides}
        self.run_expected('INCOMPLETE')
        self.assertEqual([s['position'] for s in self.reverse.call_args.args[1]], [2])
        self.assertGreater(self.recovery.call_count, 1)
        self.assertTrue(all(call.args[1]['position'] == 2 for call in self.recovery.call_args_list))
        for mocked in (self.download, self.ocr, self.translator_class):
            mocked.assert_not_called()
        current = read_json(self.manifest_path)
        self.assertFalse(current['reverse_search_completed'])
        self.assertTrue(current['slides'][0]['reverse_search']['cache_hit'])
        self.assertEqual(current['slides'][0]['best_image'], 'recovered/01_SOURCE.jpg')
        self.assertEqual(source.read_bytes(), source_bytes)
        self.assertNotIn('cache_hit', previous['slides'][0]['reverse_search'])
        # The returned cached record must also own its nested OCR evidence.
        cached = reusable_source(self.base, previous['slides'][0], current['slides'][0],
            previous['signature'], current['signature'], self.settings)
        cached['validation']['source_ocr']['accepted'] = False
        self.assertTrue(previous['slides'][0]['reverse_search']['validation']['source_ocr']['accepted'])

    def test_source_file_hash_change_requeries_and_discards_old_source(self):
        previous = self.prepare_source_cache()
        source = self.base / previous['slides'][0]['reverse_search']['downloaded']
        shutil.copyfile(REAL_POST / 'original' / '02.jpg', source)
        self.reverse.reset_mock()
        self.recovery.reset_mock()
        self.reverse.side_effect = lambda base, slides: {
            slide['position']: {'status': 'NOT_FOUND', 'candidates': []} for slide in slides}
        self.run_expected('INCOMPLETE')
        self.assertEqual([s['position'] for s in self.reverse.call_args.args[1]], [1, 2])
        self.assertGreater(self.recovery.call_count, 2)
        self.assertTrue(all([call.args[1]['position'] for call in self.recovery.call_args_list][offset:offset+2] == [1, 2]
                            for offset in range(0, self.recovery.call_count, 2)))
        self.assertFalse(source.exists())
        self.assertIsNone(read_json(self.manifest_path)['slides'][0]['best_image'])

    def test_undecodable_source_with_matching_recorded_hash_is_not_reusable(self):
        previous = self.prepare_source_cache()
        slide = previous['slides'][0]
        source = self.base / slide['reverse_search']['downloaded']
        source.write_bytes(b'Not an image, even though the manifest digest matches')
        slide['reverse_search']['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        write_json(self.manifest_path, previous)
        self.assertIsNone(reusable_source(self.base, slide, slide, previous['signature'],
            previous['signature'], self.settings))
        self.reverse.reset_mock()
        self.recovery.reset_mock()
        self.reverse.side_effect = lambda base, slides: {
            row['position']: {'status': 'NOT_FOUND', 'candidates': []} for row in slides}
        self.run_expected('INCOMPLETE')
        self.assertEqual([s['position'] for s in self.reverse.call_args.args[1]], [1, 2])
        self.assertFalse(source.exists())

    def test_force_refresh_ignores_verified_source_cache(self):
        self.prepare_source_cache()
        for mocked in (self.download, self.ocr, self.reverse, self.recovery):
            mocked.reset_mock()
        self.reverse.side_effect = lambda base, slides: {
            slide['position']: {'status': 'NOT_FOUND', 'candidates': []} for slide in slides}
        result = run_pipeline(URL, force=True, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result['status'], 'INCOMPLETE', result)
        self.download.assert_called_once()
        self.assertEqual(self.ocr.call_count, 2)
        self.assertEqual([s['position'] for s in self.reverse.call_args.args[1]], [1, 2])
        self.assertGreater(self.recovery.call_count, 2)
        self.assertTrue(all([call.args[1]['position'] for call in self.recovery.call_args_list][offset:offset+2] == [1, 2]
                            for offset in range(0, self.recovery.call_count, 2)))
        self.assertFalse((self.base / 'recovered' / '01_SOURCE.jpg').exists())

    def test_translation_model_change_keeps_all_sources_without_browser_or_candidate_ocr(self):
        self.prepare_source_cache(positive_positions=(1, 2))
        settings = replace(self.settings, ollama_translation_model='different-local-model')
        for mocked in (self.download, self.ocr, self.translator_class, self.reverse, self.recovery):
            mocked.reset_mock()
        result = run_pipeline(URL, settings=settings, reverse_callback=self.reverse)
        self.assertEqual(result['status'], 'OK', result)
        for mocked in (self.download, self.ocr, self.reverse, self.recovery):
            mocked.assert_not_called()
        self.translator_class.assert_called_once()
        current = read_json(self.manifest_path)
        self.assertTrue(current['reverse_search_completed'])
        self.assertTrue(all(s['reverse_search']['cache_hit'] for s in current['slides']))
        self.assertTrue(all(s['best_image'].startswith('recovered/') for s in current['slides']))

    def test_discarded_candidate_download_error_does_not_invalidate_verified_match(self):
        previous = self.prepare_source_cache(positive_positions=(1, 2))
        previous['slides'][0]['reverse_search']['candidates'] = [
            {'accepted': False, 'reason': 'DOWNLOAD_FAILED', 'error': 'A discarded URL timed out'}]
        write_json(self.manifest_path, previous)
        settings = replace(self.settings, ollama_translation_model='different-local-model')
        self.reverse.reset_mock()
        self.recovery.reset_mock()
        result = run_pipeline(URL, settings=settings, reverse_callback=self.reverse)
        self.assertEqual(result['status'], 'OK', result)
        self.reverse.assert_not_called()
        self.recovery.assert_not_called()
        self.assertTrue(read_json(self.manifest_path)['reverse_search_completed'])

    def test_untrusted_source_metadata_or_ocr_version_requires_new_recovery(self):
        baseline = self.prepare_source_cache()
        original_source = (self.base / 'recovered' / '01_SOURCE.jpg').read_bytes()
        for defect in ('path', 'raw_download', 'modified', 'source', 'resolution', 'ocr_version', 'mask_hash'):
            with self.subTest(defect=defect):
                manifest = copy.deepcopy(baseline)
                slide = manifest['slides'][0]
                recovery = slide['reverse_search']
                if defect == 'path':
                    recovery['downloaded'] = 'original/01.jpg'
                    recovery['sha256'] = slide['original_sha256']
                elif defect == 'raw_download':
                    recovery['raw_download'] = False
                elif defect == 'modified':
                    recovery['modified'] = True
                elif defect == 'source':
                    recovery['source'] = 'clean'
                elif defect == 'resolution':
                    recovery['resolution']['height'] += 1
                elif defect == 'ocr_version':
                    manifest['signature']['ocr'] = 'old-ocr-version'
                else:
                    slide['ocr']['mask_sha256'] = 'invalidated-mask-hash'
                (self.base / 'recovered' / '01_SOURCE.jpg').write_bytes(original_source)
                write_json(self.manifest_path, manifest)
                self.reverse.reset_mock()
                self.recovery.reset_mock()
                self.reverse.side_effect = lambda base, slides: {
                    s['position']: {'status': 'NOT_FOUND', 'candidates': []} for s in slides}
                self.run_expected('INCOMPLETE')
                self.assertEqual([s['position'] for s in self.reverse.call_args.args[1]], [1, 2])
                self.assertGreater(self.recovery.call_count, 2)
                self.assertTrue(all([call.args[1]['position'] for call in self.recovery.call_args_list][offset:offset+2] == [1, 2]
                                    for offset in range(0, self.recovery.call_count, 2)))

    def test_failed_candidate_validation_does_not_mark_reverse_search_complete(self):
        self.reverse.return_value = {1: {'status': 'CANDIDATES', 'candidates': []}}
        with patch('src.pipeline.recover_slide', side_effect=RuntimeError('Candidate validation unavailable')):
            self.run_expected('INCOMPLETE')
        manifest = read_json(self.manifest_path)
        self.assertFalse(manifest['reverse_search_completed'])
        self.assertEqual(manifest['slides'][0]['reverse_search']['status'], 'ERROR')

    def test_reused_source_without_reverse_callback_keeps_search_incomplete(self):
        self.prepare_source_cache()
        self.recovery.reset_mock()
        settings = replace(self.settings, ollama_translation_model='different-local-model')
        result = run_pipeline(URL, settings=settings)
        self.assertEqual(result['status'], 'INCOMPLETE', result)
        current = read_json(self.manifest_path)
        self.assertFalse(current['reverse_search_completed'])
        self.assertEqual(current['slides'][0]['best_image'], 'recovered/01_SOURCE.jpg')
        self.assertEqual([c.args[1]['position'] for c in self.recovery.call_args_list], [2])

    def test_refresh_without_web_source_removes_obsolete_recovered_source(self):
        recovered = self.base / "recovered" / "01_SOURCE.jpg"

        def previous_match(base, slide, search, settings):
            return self.fake_verified_source(base, slide)

        with patch("src.pipeline.recover_slide", side_effect=previous_match):
            self.run_success()
        self.assertTrue(recovered.is_file())
        self.reverse.return_value = {1: {'status': 'NOT_FOUND', 'candidates': []}}
        result = run_pipeline(URL, force=True, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result["status"], "INCOMPLETE", result)
        self.assertFalse(recovered.exists(), "Old SOURCE cannot remain when the current run has no downloaded web image")

    def test_low_score_and_source_ocr_annotations_do_not_invalidate_raw_cache(self):
        previous = self.prepare_source_cache(positive_positions=(1, 2))
        recovery = previous['slides'][0]['reverse_search']
        recovery['score'] = .12
        recovery['confidence'] = 'LOW'
        recovery['contains_text'] = True
        recovery['validation']['source_ocr']['accepted'] = False
        write_json(self.manifest_path, previous)
        for mocked in (self.download, self.ocr, self.reverse, self.recovery):
            mocked.reset_mock()
        result = run_pipeline(URL, settings=self.settings, reverse_callback=self.reverse)
        self.assertEqual(result['status'], 'OK_CACHED', result)
        for mocked in (self.download, self.ocr, self.reverse, self.recovery):
            mocked.assert_not_called()


class TranslationRetryCacheTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(translation_retry_ttl=60)
        self.manifest = {'translation_generated_at': time.time(), 'translations': {
            '1': {'fallback_to_english': True, 'retry_recommended': True}}}

    def test_transient_failure_can_reuse_fresh_result_until_retry_ttl(self):
        self.assertFalse(translation_retry_due(self.manifest, self.settings))

    def test_transient_failure_is_retried_after_ttl(self):
        self.manifest['translation_generated_at'] = time.time()-61
        self.assertTrue(translation_retry_due(self.manifest, self.settings))

    def test_real_ambiguous_fragment_preserved_in_english_does_not_retry_forever(self):
        self.manifest['translation_generated_at'] = 0
        self.manifest['translations']['1']['retry_recommended'] = False
        self.assertFalse(translation_retry_due(self.manifest, self.settings))

    def test_invalid_generation_timestamp_does_not_permanently_cache_a_transient_failure(self):
        for value in (None, 'invalid', float('nan'), float('inf')):
            with self.subTest(value=value):
                self.manifest['translation_generated_at'] = value
                self.assertTrue(translation_retry_due(self.manifest, self.settings))


class ReverseStrategyExhaustionTests(unittest.TestCase):
    def strategy(self, statuses, *, expire_after=None, geometric_skips=False):
        # Fresh state per case prevents a prior negative cache from hiding a
        # transient failure; only discovery/recovery boundaries are isolated.
        slide = {'position':1,'clean':'clean/01.jpg','timings':{},'warnings':[]}
        clock = [0.0]
        calls = []
        answers = iter(statuses)
        def query(base, slide, kind, index):
            result = {'kind':kind,'attempt_index':index}
            if geometric_skips and kind.startswith('clean_crop_'):
                result.update(skipped=True,reason='NO_SIGNIFICANT_TEXT_FREE_CROP')
            return result
        def callback(base, slides):
            calls.append(slides[0]['reverse_query']['kind'])
            status = next(answers)
            if expire_after and len(calls)==expire_after:
                clock[0] = 301.0
            return {1:{'status':status,'candidates':[]}}
        def recover(base, slide, search, settings):
            return {'status':search['status'],'candidates':[],'score':0,'downloaded':None}
        with patch('src.pipeline.prepare_reverse_query',side_effect=query), patch('src.pipeline.recover_slide',side_effect=recover), patch('src.pipeline.time.perf_counter',side_effect=lambda:clock[0]):
            recover_with_strategy(Path('.'),[slide],Settings(),callback,Mock())
        return slide['reverse_search'],calls

    def test_transient_failure_in_final_round_never_means_strategy_completed(self):
        for status in ('TIMEOUT','CAPTCHA','ERROR','UNAVAILABLE','DISABLED'):
            with self.subTest(status=status):
                result,calls = self.strategy(['NOT_FOUND']*5+[status])
                self.assertEqual(len(calls),6)
                self.assertEqual(result['status'],status)
                self.assertFalse(result['strategy_exhausted'])
                self.assertFalse(result.get('budget_exhausted',False))

    def test_prior_error_then_final_not_found_still_requires_another_search(self):
        result,calls = self.strategy(['ERROR']+['NOT_FOUND']*5)
        self.assertEqual(len(calls),6)
        self.assertEqual(result['status'],'NOT_FOUND')
        self.assertFalse(result['strategy_exhausted'])

    def test_completed_negative_strategy_including_geometric_skips_can_be_cached(self):
        result,calls = self.strategy(['NOT_FOUND']*4,geometric_skips=True)
        self.assertEqual(calls,['clean_full','original_full','google_images','text_assisted'])
        self.assertTrue(result['strategy_exhausted'])
        self.assertEqual([a['status'] for a in result['attempts']],['NOT_FOUND','NOT_FOUND','SKIPPED','SKIPPED','NOT_FOUND','NOT_FOUND'])

    def test_slow_prior_round_does_not_prevent_google_images_and_text_fallbacks(self):
        result,calls = self.strategy(['NOT_FOUND']*6,expire_after=4)
        self.assertEqual(len(calls),6)
        self.assertEqual(calls[-2:], ['google_images', 'text_assisted'])
        self.assertEqual(len(result['attempts']),6)
        self.assertEqual(result['status'],'NOT_FOUND')
        self.assertFalse(result.get('budget_exhausted',False))
        self.assertTrue(result['strategy_exhausted'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
