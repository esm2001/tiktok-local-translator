"""Regression checks for byte-identical web recovery and download fallbacks."""
import base64
import hashlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'translator'))
from src.config import Settings
from src.recovery import download_candidate, recover_slide, revalidate_saved_source


@pytest.fixture
def tmp_path():
    # The machine's shared pytest temp root can have inherited denied ACLs.
    cache = Path(__file__).resolve().parents[1] / '.cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test-raw-recovery-', dir=cache) as folder:
        yield Path(folder)


def image_bytes(extension='.png'):
    image = np.random.default_rng(9).integers(0, 256, (100, 150, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(extension, image)
    assert ok
    return encoded.tobytes()


def response(raw, content_type='image/png', status=200):
    result = Mock()
    result.status_code = status
    result.is_redirect = False
    result.headers = {'Content-Type': content_type, 'Content-Length': str(len(raw))}
    result.raw = None
    result.iter_content.return_value = iter([raw])
    result.encoding = 'utf-8'
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


@pytest.mark.parametrize('extension,content_type', [('.jpg', 'image/jpeg'), ('.png', 'image/png'), ('.webp', 'image/webp')])
def test_download_preserves_original_format_and_every_byte(tmp_path, extension, content_type):
    raw = image_bytes(extension)
    candidate = {'image_url': 'https://images.example.test/photo' + extension}
    with patch('src.recovery.public_image_url'), patch('src.recovery.requests.get', return_value=response(raw, content_type)):
        item = download_candidate(candidate, tmp_path / 'candidate.jpg')
    assert 'error' not in item
    assert item['path'].suffix == extension
    assert item['path'].read_bytes() == raw
    assert item['http_status'] == 200
    assert item['content_type'] == content_type


@pytest.mark.parametrize('content_type', ['text/html', 'image/jpeg', 'application/octet-stream', ''])
def test_html_never_becomes_an_image_file(tmp_path, content_type):
    raw = b'<!DOCTYPE html><html><body>access denied</body></html>' * 10
    with patch('src.recovery.public_image_url'), patch('src.recovery.requests.get', side_effect=lambda *a, **k: response(raw, content_type)):
        result = download_candidate({'image_url': 'https://example.test/photo.jpg'}, tmp_path / 'candidate.jpg')
    assert result.get('error')
    assert not list(tmp_path.glob('candidate.*'))


def test_failed_direct_url_extracts_full_resolution_image_and_uses_referer(tmp_path):
    raw = image_bytes('.png')
    seen = []
    def get(url, **kwargs):
        seen.append((url, kwargs.get('headers', {})))
        if url == 'https://publisher.example.test/article':
            return response(b'<meta property="og:image" content="/full.png">', 'text/html')
        if url == 'https://publisher.example.test/full.png':
            assert kwargs['headers']['Referer'] == 'https://publisher.example.test/article'
            return response(raw)
        return response(b'Forbidden', 'text/plain', 403)
    with patch('src.recovery.public_image_url'), patch('src.recovery.requests.get', side_effect=get):
        result = download_candidate({'image_url': 'https://cdn.example.test/broken.jpg',
            'source_url': 'https://publisher.example.test/article'}, tmp_path / 'candidate.jpg')
    assert result['path'].read_bytes() == raw
    assert result['candidate']['image_url'] == 'https://publisher.example.test/full.png'
    assert any(row['http_status'] == 403 for row in result['download_attempts'])


def test_valid_playwright_fallback_writes_original_response_bytes(tmp_path):
    raw = image_bytes('.webp')
    candidate = {'image_url': 'https://example.test/blocked.webp', 'browser_download': {
        'url': 'https://example.test/source.webp', 'http_status': 200, 'content_type': 'image/webp',
        'data_base64': base64.b64encode(raw).decode()}}
    with patch('src.recovery.public_image_url'), patch('src.recovery.requests.get') as get:
        result = download_candidate(candidate, tmp_path / 'candidate.jpg')
    get.assert_not_called()
    assert result['path'].suffix == '.webp'
    assert result['path'].read_bytes() == raw
    assert result['transport'] == 'playwright'


def recover_fixture(tmp_path, scores, text_ratios=None, ocr_fails=False):
    (tmp_path / 'metadata').mkdir(exist_ok=True)
    raw = image_bytes('.png')
    candidates = [{'image_url': f'https://example.test/{i}.png'} for i in range(len(scores))]
    def download(candidate, target):
        target.write_bytes(raw)
        return {'candidate': candidate, 'path': target, 'bytes': len(raw), 'http_status': 200, 'content_type': 'image/png'}
    comparisons = [{'accepted': False, 'score': score, 'reason': 'BACKGROUND_DIFFERS',
        'width': 150, 'height': 100, 'metrics': {}} for score in scores]
    checks = [{'accepted': ratio == 0, 'contains_text': ratio > 0, 'candidate_text_area_ratio': ratio}
        for ratio in (text_ratios or [0] * len(scores))]
    with patch('src.recovery.download_candidate', side_effect=download), \
         patch('src.recovery.compare_candidate', side_effect=comparisons), \
         patch('src.recovery.candidate_overlay_check', side_effect=RuntimeError('OCR failed') if ocr_fails else checks), \
         patch('src.recovery.write_image') as writer, patch('src.recovery.cv2.inpaint') as inpainter:
        result = recover_slide(tmp_path, {'position': 1, 'original': 'original/01.jpg',
            'clean': 'clean/01_CLEAN.jpg', 'ocr': {'mask': 'metadata/01.png'}},
            {'status': 'CANDIDATES', 'candidates': candidates}, Settings(output_base=tmp_path))
    writer.assert_not_called()
    inpainter.assert_not_called()
    assert (tmp_path / result['downloaded']).read_bytes() == raw
    assert result['sha256'] == hashlib.sha256(raw).hexdigest()
    assert result['modified'] is False
    assert result['raw_download'] is True
    assert result['source'] == 'web_download'
    assert result['downloaded'] == 'recovered/01_SOURCE.png'
    return result


def test_mediocre_candidates_always_save_highest_score(tmp_path):
    result = recover_fixture(tmp_path, [.31, .54, .48])
    assert result['status'] == 'MATCH'
    assert result['image_url'] == 'https://example.test/1.png'
    assert result['confidence'] == 'LOW'


def test_caption_penalty_prefers_similar_clean_web_candidate_without_editing(tmp_path):
    result = recover_fixture(tmp_path, [.88, .84], [.2, 0])
    assert result['image_url'] == 'https://example.test/1.png'
    assert result['contains_text'] is False


def test_only_captioned_candidate_is_saved_raw_and_reports_text(tmp_path):
    result = recover_fixture(tmp_path, [.54], [.2])
    assert result['contains_text'] is True
    assert result['status'] == 'MATCH'


def test_optional_candidate_ocr_failure_does_not_block_recovery(tmp_path):
    result = recover_fixture(tmp_path, [.54], ocr_fails=True)
    assert result['status'] == 'MATCH'
    assert result['validation']['source_ocr']['error'] == 'OCR failed'


def test_cache_without_explicit_raw_provenance_is_not_reused(tmp_path):
    original_hash = 'same-input'
    slide = {'position': 1, 'original_sha256': original_hash}
    prior = {'original_sha256': original_hash, 'reverse_search': {'status': 'MATCH',
        'downloaded': 'recovered/01_SOURCE.jpg', 'source_url': 'https://example.test/photo.jpg'}}
    audit = {}
    result = revalidate_saved_source(tmp_path, prior, slide, Settings(output_base=tmp_path), audit=audit)
    assert result is None
    assert audit['reason'] == 'NO_PRIOR_RAW_WEB_SOURCE'
