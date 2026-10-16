"""Exercise image URL discovery without Google, a browser, or external calls."""

import hashlib
import json
import sys
import tempfile
import time
import unittest
import importlib.util
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "launcher"))
from reverse_search.google_lens import GoogleLensProvider, PARSER_VERSION, extract_candidates, extract_source_urls


class LensParserTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('playwright'), 'Playwright browser runtime is optional')
    def test_async_lens_upload_modal_is_waited_for_before_selector_detection(self):
        """Actual browser regression for the modal race observed on Google."""
        from playwright.sync_api import sync_playwright
        fixture = '''<input type="file" name="ai_attachment" onchange="document.body.dataset.wrongUpload='true'">
        <button aria-label="Search by image" onclick="setTimeout(() => {
            const input=document.createElement('input');input.type='file';
            input.name='encoded_image';
            input.onchange=()=>location.href='/search?q=fixture';document.body.appendChild(input);
        }, 200)">Camera</button>'''
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True, channel='chrome')
            except Exception:
                try:
                    browser = playwright.chromium.launch(headless=True)
                except Exception as exc:
                    self.skipTest(f'Optional browser executable unavailable: {exc}')
            with tempfile.TemporaryDirectory() as directory:
                provider = GoogleLensProvider(Path(directory), headless=True, timeout_seconds=5, interactive_timeout=0)
                provider._context = browser.new_context()
                provider._page = provider._context.new_page()
                # Every request is fulfilled locally; no image leaves the machine.
                provider._page.route('**/*', lambda route: route.fulfill(body=fixture, content_type='text/html'))
                image_file = Path(directory)/'slide.jpg'
                image_file.write_bytes(b'fixture')
                try:
                    status,error = provider._upload(image_file)
                    self.assertIsNone(status, error)
                    self.assertIn('/search', provider._page.url)
                finally:
                    provider._context.close()
                    browser.close()

    def test_image_result_decodes_full_url_and_publisher(self):
        image = "https://images.example.org/photo.jpg?size=1920&quality=90"
        source = "https://publisher.example.org/portrait"
        link = "https://www.google.com/imgres?" + urlencode({"imgurl": image, "imgrefurl": source})
        result = extract_candidates(f'<a href="{link.replace("&", "&amp;")}"><img src="https://encrypted-tbn0.gstatic.com/thumb"></a>')
        self.assertEqual(result[0]["image_url"], image)
        self.assertEqual(result[0]["source_url"], source)
        self.assertIn("https://encrypted-tbn0.gstatic.com/thumb", result[0]["alternate_urls"])

    def test_payload_dimensions_filter_thumbnails_and_deduplicate(self):
        markup = '["https://photos.example.org/full.jpg",900,1200], ["https://photos.example.org/tiny.jpg",80,90], ["https://photos.example.org/full.jpg",900,1200]'
        self.assertEqual(len(extract_candidates(markup)), 1)
        self.assertTrue(extract_candidates(markup)[0]["image_url"].endswith("full.jpg"))

    def test_unicode_escaped_payload_is_decoded(self):
        markup = r'["https:\/\/images.example.org\/photo.jpg?x\u003d1\u0026y\u003d2",700,900]'
        # Google script strings contain escaped slashes/unicode, not escaped
        # wrapping quotes after the JavaScript payload is rendered to HTML.
        markup = markup.replace('\\"', '"')
        self.assertEqual(extract_candidates(markup)[0]["image_url"], "https://images.example.org/photo.jpg?x=1&y=2")

    def test_source_metadata_and_srcset(self):
        result = extract_candidates('<meta property="og:image" content="/full.webp"><img src="https://encrypted-tbn0.gstatic.com/tiny"><img srcset="/small.jpg 200w, /large.jpg 1200w">', "https://publisher.example.org/article")
        self.assertEqual(result[0]["image_url"], "https://publisher.example.org/full.webp")
        self.assertTrue(any(candidate["image_url"].endswith("large.jpg") for candidate in result))
        self.assertFalse(any("gstatic" in candidate["image_url"] for candidate in result))

    def test_private_urls_and_query_image_are_not_candidates(self):
        urls = ["http://127.0.0.1/private.jpg", "http://192.168.1.1/private.jpg", "http://localhost/private.jpg", "file:///C:/photo.jpg", "https://lens.google.com/upload.jpg", "https://cdn.tiktok.com/slide.jpg"]
        markup = "".join(f'<meta property="og:image" content="{url}">' for url in urls)
        self.assertEqual(extract_candidates(markup, "https://publisher.example.org/article"), [])

    def test_sources_ignore_google_navigation_and_unwrap_redirect(self):
        markup = '<a href="https://accounts.google.com/login">Sign in</a><a href="https://about.example.org/help">Footer</a><a href="/url?q=https%3A%2F%2Fpublisher.example.org%2Farticle"><img src="thumb.jpg">Result</a><a href="https://publisher.example.org/article"><img src="thumb.jpg">Duplicate</a><a href="http://10.0.0.1/local"><img src="thumb.jpg">Bad</a>'
        self.assertEqual(extract_source_urls(markup), ["https://publisher.example.org/article"])

    def test_bing_json_result_attribute_keeps_full_image_and_page(self):
        payload = '{&quot;murl&quot;:&quot;https://images.example.org/full.png&quot;,&quot;purl&quot;:&quot;https://publisher.example.org/article&quot;,&quot;turl&quot;:&quot;https://tse1.mm.bing.net/thumb&quot;}'
        result = extract_candidates(f'<a class="iusc" m="{payload}"><img src="https://tse1.mm.bing.net/thumb"></a>')
        self.assertEqual(result[0]['image_url'], 'https://images.example.org/full.png')
        self.assertEqual(result[0]['source_url'], 'https://publisher.example.org/article')
        self.assertIn('https://tse1.mm.bing.net/thumb', result[0]['alternate_urls'])

    def test_googleusercontent_full_image_remains_downloadable_candidate(self):
        result = extract_candidates('["https://blogger.googleusercontent.com/img/photo.jpg",900,1200]')
        self.assertEqual(result[0]['image_url'], 'https://blogger.googleusercontent.com/img/photo.jpg')

    def test_unrelated_bing_cards_try_commons_and_preserve_its_original_media_urls(self):
        from unittest.mock import patch
        markup = '<a m="{&quot;murl&quot;:&quot;https://images.example.org/unrelated.jpg&quot;,&quot;purl&quot;:&quot;https://publisher.example.org/movie&quot;,&quot;t&quot;:&quot;Unrelated movie&quot;}"></a>'
        with tempfile.TemporaryDirectory() as directory:
            provider = GoogleLensProvider(Path(directory))
            original = {'image_url': 'https://upload.wikimedia.org/wikipedia/commons/original.jpg',
                        'source_url': 'https://commons.wikimedia.org/wiki/File:Hegel.jpg'}
            with patch('reverse_search.google_lens._source_page', side_effect=[('', 'https://google.com'), (markup, 'https://bing.com')]), \
                 patch.object(provider, '_wikimedia_text_search', return_value={'provider': 'wikimedia_text_search',
                    'status': 'CANDIDATES', 'candidates': [original], 'diagnostics': {'textual_search_attempts': [{'provider': 'wikimedia_text_search'}]}}) as commons:
                result = provider._http_text_search('Hegel portrait painting')
            commons.assert_called_once_with('Hegel portrait painting')
            self.assertEqual(result['provider'], 'wikimedia_text_search')
            self.assertEqual(result['candidates'][0], original)
            self.assertEqual(len(result['diagnostics']['textual_search_attempts']), 3)

    def test_commons_failure_still_returns_available_bing_image_urls(self):
        from unittest.mock import patch
        markup = '<a m="{&quot;murl&quot;:&quot;https://images.example.org/unrelated.jpg&quot;,&quot;purl&quot;:&quot;https://publisher.example.org/movie&quot;,&quot;t&quot;:&quot;Unrelated movie&quot;}"></a>'
        with tempfile.TemporaryDirectory() as directory:
            provider = GoogleLensProvider(Path(directory))
            with patch('reverse_search.google_lens._source_page', side_effect=[('', 'https://google.com'), (markup, 'https://bing.com')]), \
                 patch.object(provider, '_wikimedia_text_search', return_value={'status': 'UNAVAILABLE',
                    'candidates': [], 'diagnostics': {'textual_search_attempts': []}}):
                result = provider._http_text_search('Hegel portrait painting')
            self.assertEqual(result['status'], 'CANDIDATES')
            self.assertEqual(result['candidates'][0]['image_url'], 'https://images.example.org/unrelated.jpg')

    def test_captcha_text_fallback_uses_public_http_without_starting_lens_browser(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            provider = GoogleLensProvider(Path(directory))
            provider._captcha_blocked = True
            provider._start = lambda: self.fail('Must not restart Lens after a CAPTCHA.')
            candidate = {'image_url': 'https://images.example.org/Hegel.jpg'}
            with patch.object(provider, '_http_text_search', return_value={'provider': 'bing_images_http',
                    'status': 'CANDIDATES', 'candidates': [candidate], 'diagnostics': {}}) as public_search:
                result = provider.search_text('Hegel portrait', force=True)
            public_search.assert_called_once_with('Hegel portrait')
            self.assertEqual(result['candidates'], [candidate])
            self.assertEqual(result['diagnostics']['lens_session_status'], 'CAPTCHA')

    def test_cached_candidates_do_not_start_browser_and_force_bypasses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.jpg"
            clean.write_bytes(b"test-clean")
            provider = GoogleLensProvider(root / "browser" / "google_lens")
            provider.cache_dir.mkdir(parents=True)
            digest = hashlib.sha256(clean.read_bytes()).hexdigest()
            result = {"provider": "google_lens", "status": "CANDIDATES", "candidates": [{"image_url": "https://example.org/full.jpg", "source_url": "", "source_domain": "example.org"}]}
            (provider.cache_dir / f"{digest}.json").write_text(json.dumps({"image_sha256": digest, "parser_version": PARSER_VERSION, "created_at": time.time(), "result": result}), encoding="utf-8")
            provider._start = lambda: (_ for _ in ()).throw(RuntimeError("browser must stay closed"))
            self.assertTrue(provider.search(clean)["cache_hit"])
            refreshed = provider.search(clean, force=True)
            self.assertEqual(refreshed['status'], 'CANDIDATES')
            self.assertEqual(refreshed['fresh_search_status'], 'UNAVAILABLE')

    def test_expired_and_wrong_hash_cache_are_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.jpg"
            clean.write_bytes(b"test-clean")
            provider = GoogleLensProvider(root / "browser" / "google_lens")
            provider.cache_dir.mkdir(parents=True)
            digest = hashlib.sha256(clean.read_bytes()).hexdigest()
            cache = provider.cache_dir / f"{digest}.json"
            provider._start = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
            for cached_hash, created in [("wrong", time.time()), (digest, time.time() - 4000)]:
                cache.write_text(json.dumps({"image_sha256": cached_hash, "parser_version": PARSER_VERSION, "created_at": created, "result": {"status": "NOT_FOUND", "candidates": []}}), encoding="utf-8")
                self.assertEqual(provider.search(clean)["status"], "UNAVAILABLE")

    def test_parser_upgrade_invalidates_old_results_even_when_image_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_file = root/'clean.jpg'
            image_file.write_bytes(b'fixture')
            digest = hashlib.sha256(image_file.read_bytes()).hexdigest()
            provider = GoogleLensProvider(root/'browser/google_lens')
            provider.cache_dir.mkdir(parents=True)
            (provider.cache_dir/f'{digest}.json').write_text(json.dumps({
                'image_sha256': digest, 'parser_version': 2, 'created_at': time.time(),
                'result': {'status': 'CANDIDATES', 'candidates': [{'image_url': 'https://fixture.example/photo.jpg'}]}}), encoding='utf-8')
            provider._start = lambda: (_ for _ in ()).throw(RuntimeError('offline after parser upgrade'))
            result = provider.search(image_file)
            self.assertEqual(result['status'], 'UNAVAILABLE')
            self.assertFalse(result.get('cache_hit', False))

    def test_parser_three_positive_cache_survives_upgrade_and_failed_force_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'slide.jpg'
            image.write_bytes(b'fixture')
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            provider = GoogleLensProvider(root / 'browser/google_lens')
            provider.cache_dir.mkdir(parents=True)
            target = provider.cache_dir / f'{digest}.json'
            target.write_text(json.dumps({'image_sha256': digest, 'parser_version': 3, 'created_at': time.time(),
                'result': {'provider': 'google_lens', 'status': 'CANDIDATES',
                    'candidates': [{'image_url': 'https://images.example.org/source.png'}]}}), encoding='utf-8')
            before = target.read_bytes()
            provider._start = lambda: (_ for _ in ()).throw(RuntimeError('browser unavailable'))
            result = provider.search(image, force=True)
            self.assertEqual(result['status'], 'CANDIDATES')
            self.assertTrue(result['cache_hit'])
            self.assertEqual(result['fresh_search_status'], 'UNAVAILABLE')
            self.assertIn('browser unavailable', result['fresh_search_error'])
            self.assertEqual(target.read_bytes(), before)

    def test_captcha_reuses_valid_google_urls_without_another_browser_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'slide.jpg'
            image.write_bytes(b'fixture')
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            provider = GoogleLensProvider(root / 'browser/google_lens')
            provider.cache_dir.mkdir(parents=True)
            (provider.cache_dir / f'{digest}.json').write_text(json.dumps({'image_sha256': digest,
                'parser_version': 3, 'created_at': time.time(), 'result': {'provider': 'google_lens',
                    'status': 'CANDIDATES', 'candidates': [{'image_url': 'https://images.example.org/source.jpg'}]}}), encoding='utf-8')
            provider._captcha_blocked = True
            provider._start = lambda: self.fail('CAPTCHA must not trigger another browser search.')
            result = provider.search(image, force=True)
            self.assertEqual(result['status'], 'CANDIDATES')
            self.assertTrue(result['cache_hit'])
            self.assertEqual(result['fresh_search_status'], 'CAPTCHA')

    def test_unresolved_captcha_skips_remaining_images_without_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.jpg"
            clean.write_bytes(b"clean")
            provider = GoogleLensProvider(root / "browser" / "google_lens")
            provider._captcha_blocked = True
            provider._start = lambda: self.fail("Must not open another browser")
            self.assertEqual(provider.search(clean)["status"], "CAPTCHA")
            self.assertFalse(provider.cache_dir.exists())

    def test_expired_global_budget_starts_no_browser_for_image_or_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/'slide.jpg'
            image.write_bytes(b'fixture')
            provider = GoogleLensProvider(root/'browser/google_lens')
            provider.set_deadline(time.monotonic()-1)
            provider._start = lambda: self.fail('No query may start after the global deadline.')
            self.assertEqual(provider.search(image)['status'], 'TIMEOUT')
            self.assertEqual(provider.search_text('Nikola Tesla')['status'], 'TIMEOUT')
            self.assertFalse(provider.cache_dir.exists())


if __name__ == "__main__":
    unittest.main()
