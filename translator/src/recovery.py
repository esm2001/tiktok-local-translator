"""Download web image bytes unchanged; visual/OCR checks only rank candidates."""
import hashlib
import copy
import base64
import io
import ipaddress
import json
import logging
import re
import shutil
import socket
import tempfile
import time
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

import requests
import cv2
import numpy as np
from PIL import Image

from src.config import Settings
from src.image.cleaning import read_image, write_image
from src.image.similarity import compare_candidate
from src.ocr.paddle import ocr_image
from src.ocr.editorial import unreliable_diagonal

LOGGER = logging.getLogger(__name__)
RECOVERY_VERSION = 'raw-web-ranking-v7'
QUERY_STRATEGY_VERSION = 'clean-original-crops-contextual-images-text-v4'
REVERSE_QUERY_KINDS = ('clean_full', 'original_full', 'clean_crop_1', 'clean_crop_2', 'google_images', 'text_assisted')
MAX_IMAGE_BYTES = 25_000_000
IMAGE_SUFFIXES = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif', 'BMP': '.bmp',
                  'TIFF': '.tiff', 'AVIF': '.avif'}


def _largest_clear_rectangles(mask: np.ndarray, limit: int = 2) -> list[list[int]]:
    """Find large wholly unmasked rectangles without quadratic image scans."""
    height, width = mask.shape
    scale = min(1.0, 160 / max(height, width))
    small = cv2.resize(mask, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    blocked = small > 0
    rectangles = []
    for _ in range(limit):
        heights = np.zeros(blocked.shape[1], dtype=int)
        best = None
        best_area = 0
        for row in range(blocked.shape[0]):
            heights = np.where(blocked[row], 0, heights + 1)
            stack = []
            for col in range(blocked.shape[1] + 1):
                current = int(heights[col]) if col < len(heights) else 0
                start = col
                while stack and stack[-1][1] > current:
                    left, tall = stack.pop()
                    start = left
                    wide = col - left
                    # Thin strips offer little information to Lens.
                    if min(wide / blocked.shape[1], tall / blocked.shape[0]) < .12:
                        continue
                    area = wide * tall
                    if area > best_area:
                        best_area = area
                        best = [left, row - tall + 1, col, row + 1]
                if not stack or stack[-1][1] < current:
                    stack.append((start, current))
        if best is None or best_area < blocked.size * .045:
            break
        x0, y0, x1, y1 = best
        # Conservative rounding and the dilated input ensure crops exclude text.
        box = [int(np.ceil(x0 * width / blocked.shape[1])), int(np.ceil(y0 * height / blocked.shape[0])),
               int(np.floor(x1 * width / blocked.shape[1])), int(np.floor(y1 * height / blocked.shape[0]))]
        rectangles.append(box)
        blocked[y0:y1, x0:x1] = True
    return rectangles


def prepare_reverse_query(base: Path, slide: dict, kind: str, attempt_index: int) -> dict:
    """Prepare image queries in the CPU backend; Windows needs only Playwright."""
    if kind not in REVERSE_QUERY_KINDS:
        raise ValueError('Unknown reverse image query kind.')
    query = {'kind': kind, 'attempt_index': attempt_index, 'strategy_version': QUERY_STRATEGY_VERSION}
    if kind in {'clean_full', 'original_full'}:
        query['query_file'] = slide['clean' if kind == 'clean_full' else 'original']
    elif kind.startswith('clean_crop_'):
        original = read_image(base / slide['original'])
        mask = read_image(base / slide['ocr']['mask'], cv2.IMREAD_GRAYSCALE)
        # The original supplies intact visual information outside the text mask.
        # Using CLEAN here would retain inpainting artifacts at crop edges.
        mask = cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8))
        boxes = _largest_clear_rectangles(mask, limit=6)
        ranked = []
        for box in boxes:
            x0,y0,x1,y1 = box
            crop = original[y0:y1,x0:x1]
            gray = cv2.cvtColor(cv2.resize(crop, (min(320,x1-x0), min(320,y1-y0))), cv2.COLOR_BGR2GRAY)
            contrast = float(gray.std())
            edge_density = float(np.mean(cv2.Canny(gray, 40, 120)>0))
            # A large uniform margin is a poor visual search query. Score intact
            # structure and area; this is query prioritisation, never acceptance.
            if contrast < 3 and edge_density < .003:
                continue
            visual_score = ((x1-x0)*(y1-y0))**.5 * (min(contrast/30,1) + min(edge_density/.12,1))
            ranked.append((visual_score,box,contrast,edge_density))
        ranked.sort(reverse=True)
        index = int(kind.rsplit('_', 1)[1]) - 1
        if index >= len(ranked):
            return {**query, 'skipped': True, 'reason': 'NO_SIGNIFICANT_TEXT_FREE_CROP'}
        _,box,contrast,edge_density = ranked[index]
        folder = base / 'metadata' / 'reverse_queries'
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{slide['position']:02d}_{kind}.jpg"
        x0, y0, x1, y1 = box
        write_image(target, original[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        query.update(query_file=target.relative_to(base).as_posix(), crop_box=box, crop_pixels=(x1-x0)*(y1-y0),
                     crop_contrast=round(contrast,3), crop_edge_density=round(edge_density,4))
    else:
        ocr = slide.get('ocr', {})
        lines = ocr.get('editorial_lines', ocr.get('lines', []))
        texts = [str(line.get('text', '')).strip() for line in lines if float(line.get('confidence', 0)) >= .7]
        # Proper names tend to be more useful than a complete editorial paragraph.
        names = [t for t in texts if re.fullmatch(r'[A-ZÀ-Ý][a-zà-ÿ]+(?: [A-ZÀ-Ý][a-zà-ÿ]+){0,2}', t)
                 and t.lower() not in {'you', 'your', 'he', 'she', 'they', 'we', 'it', 'this', 'that', 'follow', 'which'}]
        background = [str(line.get('text', '')).strip() for line in ocr.get('background_lines', [])
                      if float(line.get('confidence', 0)) >= .5 and str(line.get('text', '')).strip()]
        # Titles/captions belonging to the underlying photograph can identify its
        # source even when editorial text is generic. This does not change OCR.
        if background:
            # A photo's original caption is more useful than the added call to
            # action. Keep it alone, trimming a fragmented OCR word before a title.
            context = ' '.join(background)
            title = re.search(r'\b(?:giving|holding|presenting)\b.+', context, re.I)
            if title and len(context[:title.start()].split()) <= 2:
                context = title.group()
        elif kind == 'google_images' and names:
            context = ' '.join(names) + ' portrait painting'
        elif kind == 'google_images':
            context = 'philosopher portrait painting'
        else:
            context = ' '.join(names + [t for t in texts if t not in names])
        query['text_query'] = context[:240]
        if not query['text_query']:
            return {**query, 'skipped': True, 'reason': 'NO_RELIABLE_SEARCH_TEXT'}
    if query.get('query_file'):
        query['query_sha256'] = hashlib.sha256((base / query['query_file']).read_bytes()).hexdigest()
    return query


def candidate_overlay_check(original: Path, candidate: Path, mask: Path, comparison: dict, settings: Settings,
                            *, adaptive_fallback: bool = False, crop_retries: int = 0) -> dict:
    """Measure caption area for ranking only. Never reject or edit a web image."""
    # Paired real-source measurements kept every detector region and gate
    # decision, while grayscale fallback doubled warm source-validation time.
    # Original-slide OCR retains the caller's complete adaptive configuration.
    data = ocr_image(candidate, replace(settings, ocr_adaptive_fallback=adaptive_fallback,
                                        ocr_max_crop_retries=crop_retries))
    original_mask = read_image(mask, cv2.IMREAD_GRAYSCALE)
    matrix_value = comparison.get('metrics', {}).get('homography_full_resolution')
    matrix = np.asarray(matrix_value, dtype=np.float64) if matrix_value is not None else None
    try:
        candidate_shape = read_image(candidate).shape[:2]
    except (RuntimeError, OSError):
        candidate_shape = original_mask.shape
    text_pixels = np.zeros(candidate_shape, dtype=np.uint8)
    minimum_height = max(12, min(original_mask.shape) * .012)
    overlays = []
    ignored = []
    recognized = data.get('recognized_lines', [])
    evidence = []
    represented = set()
    for line in recognized:
        polygon = line.get('polygon', [])
        if not polygon:
            continue
        represented.add(tuple(map(tuple, polygon)))
        text = str(line.get('text', ''))
        # The English recognizer reads coil segments as LRU, m or runs of 0s.
        # Preserve ordinary words and reliable rotated text, but do not turn a
        # low-confidence diagonal symbol into proof of an editorial caption.
        if unreliable_diagonal(line):
            ignored.append({'polygon': polygon, 'text': text, 'confidence': line.get('confidence', 0),
                            'reason': 'LOW_CONFIDENCE_DIAGONAL_SYMBOLS'})
            continue
        evidence.append((polygon, line))
    # Detector boxes alone are not proof of text. Tesla's diagonal coil wires
    # produce many false boxes with no characters recognized at all.
    for polygon in data['detected_polygons']:
        if tuple(map(tuple, polygon)) not in represented:
            points = np.asarray(polygon, dtype=float)
            edge = points[1] - points[0]
            angle = abs(float(np.degrees(np.arctan2(edge[1], edge[0]))))
            angle = min(angle, abs(180-angle))
            tall = float(np.linalg.norm(points[-1]-points[0]))
            wide = float(np.linalg.norm(edge))
            if angle <= 20 and wide / max(tall, 1) >= 2.5:
                evidence.append((polygon, {'text': '', 'confidence': 0, 'uncertain_detection': True}))
            else:
                ignored.append({'polygon': polygon, 'reason': 'DETECTION_WITHOUT_TEXT_EVIDENCE'})
    for polygon, line in evidence:
        candidate_points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        cv2.fillPoly(text_pixels, [np.round(candidate_points).astype(np.int32)], 255)
        if matrix is None:
            points = candidate_points.copy()
            points[:, 0] *= original_mask.shape[1] / candidate_shape[1]
            points[:, 1] *= original_mask.shape[0] / candidate_shape[0]
        else:
            points = cv2.perspectiveTransform(candidate_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        if not np.isfinite(points).all():
            continue
        if float(points[:, 1].max() - points[:, 1].min()) < minimum_height:
            continue
        polygon_mask = np.zeros(original_mask.shape, dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [np.round(points).astype(np.int32)], 255)
        area = np.count_nonzero(polygon_mask)
        overlap = np.count_nonzero((polygon_mask > 0) & (original_mask > 0)) / max(1, area)
        if overlap >= .20 and area >= 150:
            overlays.append({'polygon': np.round(points).astype(int).tolist(), 'mask_overlap': round(overlap, 3),
                             'text': line.get('text', ''), 'confidence': line.get('confidence', 0),
                             'uncertain_detection': line.get('uncertain_detection', False)})
    area_ratio = np.count_nonzero(text_pixels) / max(1, text_pixels.size)
    return {'accepted': not overlays, 'contains_text': bool(evidence),
            'candidate_text_area_ratio': round(float(area_ratio), 6),
            'ranking_only': True, 'detected_regions': len(data['detected_polygons']),
            'recognized_regions': len(recognized), 'ignored_detection_regions': ignored, 'overlaid_regions': overlays,
            'diagnostics': data.get('diagnostics', {})}


def revalidate_saved_source(base: Path, prior: dict, slide: dict, settings: Settings, audit: dict | None = None) -> dict | None:
    """Reuse only hash-intact files with explicit raw web-download provenance."""
    recovery = prior.get('reverse_search', {})
    def rejected(reason):
        if audit is not None:
            audit.update(status='REJECTED', reason=reason)
        return None
    if recovery.get('status') != 'MATCH' or recovery.get('raw_download') is not True or recovery.get('modified') is not False:
        return rejected('NO_PRIOR_RAW_WEB_SOURCE')
    if recovery.get('source') != 'web_download' or not recovery.get('image_url'):
        return rejected('WEB_PROVENANCE_MISSING')
    if not prior.get('original_sha256') or prior['original_sha256'] != slide.get('original_sha256'):
        return rejected('ORIGINAL_INPUT_CHANGED')
    relative = recovery.get('downloaded', '')
    if not re.fullmatch(rf"recovered/{int(slide['position']):02d}_SOURCE\.(jpg|png|webp|gif|bmp|tiff|avif)", relative):
        return rejected('SOURCE_METADATA_INCOMPLETE_OR_UNMANAGED')
    source = base / relative
    try:
        if not source.resolve().is_relative_to(base.resolve()) or not source.is_file():
            return rejected('SOURCE_FILE_MISSING_OR_OUTSIDE_POST')
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != recovery.get('sha256'):
            return rejected('SOURCE_HASH_CHANGED')
        _inspect_image_bytes(raw)
        output = copy.deepcopy(recovery)
        output.update(cache_hit=True, cache_revalidated=True, matcher_version=RECOVERY_VERSION,
                      strategy_version=QUERY_STRATEGY_VERSION, error='')
        output['timings'] = {'candidate_download': 0, 'candidate_matching': 0, 'source_ocr': 0}
        output['revalidation'] = {'reason': 'RAW_PROVENANCE_AND_HASH_VERIFIED'}
        if audit is not None:
            audit.update(status='MATCH', reason='RAW_PROVENANCE_AND_HASH_VERIFIED')
        return output
    except Exception as exc:
        if audit is not None:
            audit.update(status='ERROR', reason='SAVED_SOURCE_REVALIDATION_FAILED', error=str(exc))
        return None


def public_image_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in {'http', 'https'} or not parts.hostname or parts.username or parts.password:
        raise ValueError('URL de candidato no válida.')
    host = parts.hostname.lower()
    if any(host == domain or host.endswith('.' + domain) for domain in ['tiktok.com', 'tiktokcdn.com', 'tiktokcdn-us.com', 'tiktokv.com']):
        raise ValueError('Se descarta una copia procedente de TikTok.')
    addresses = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == 'https' else 80), type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError('Los candidatos deben ser imágenes públicas de Internet.')
    return url


def _budget_expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _download_chunks(response, deadline=None):
    # read1 yields after an available socket read, so deadline checks need not
    # wait for a full 64 KiB chunk on a slowly streaming publisher connection.
    read1 = getattr(getattr(response, 'raw', None), 'read1', None)
    if callable(read1):
        while True:
            if _budget_expired(deadline):
                raise TimeoutError('Candidate validation total time budget exhausted.')
            chunk = read1(65536, decode_content=True)
            if not chunk:
                break
            yield chunk
    else:
        chunks = iter(response.iter_content(65536))
        while True:
            if _budget_expired(deadline):
                raise TimeoutError('Candidate validation total time budget exhausted.')
            try:
                yield next(chunks)
            except StopIteration:
                break


def _inspect_image_bytes(raw: bytes) -> dict:
    if not 128 <= len(raw) <= MAX_IMAGE_BYTES:
        raise ValueError(f'Image size is not reasonable: {len(raw)} bytes.')
    with Image.open(io.BytesIO(raw)) as image:
        format_name = image.format
        width, height = image.size
        if min(width, height) < 16 or width * height > 80_000_000:
            raise ValueError(f'Image dimensions are not reasonable: {width}x{height}.')
        image.verify()
    # verify() checks structure; load() also catches truncated pixel data.
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
    if format_name not in IMAGE_SUFFIXES:
        raise ValueError(f'Unsupported image format: {format_name}.')
    return {'extension': IMAGE_SUFFIXES[format_name], 'format': format_name,
            'width': int(width), 'height': int(height)}


class _PageImages(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.metadata = []
        self.images = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == 'meta':
            key = (attributes.get('property') or attributes.get('name') or '').lower()
            if key in {'og:image', 'og:image:secure_url', 'twitter:image', 'twitter:image:src'}:
                self.metadata.append(attributes.get('content', ''))
        elif tag in {'img', 'source'}:
            srcset = []
            for value in attributes.get('srcset', attributes.get('data-srcset', '')).split(','):
                fields = value.strip().split()
                if fields:
                    size = float(fields[1][:-1]) if len(fields) > 1 and re.fullmatch(r'[0-9.]+[wx]', fields[1]) else 0
                    srcset.append((size, fields[0]))
            self.images.extend(value for _, value in sorted(srcset, reverse=True))
            for key in ('data-original', 'data-large', 'data-src', 'src'):
                if attributes.get(key):
                    self.images.append(attributes[key])
        elif tag == 'a' and re.search(r'\.(jpe?g|png|webp|avif)([?#]|$)', attributes.get('href', ''), re.I):
            self.metadata.append(attributes['href'])


def _page_image_urls(markup: str, base_url: str) -> list[str]:
    parser = _PageImages()
    parser.feed(markup)
    values = parser.metadata + parser.images
    # Several publishers put the original file only in JSON-LD or JS payloads.
    values += re.findall(r'(?:"(?:contentUrl|image|url)"\s*:\s*")([^"<>]+\.(?:jpe?g|png|webp|avif)(?:\?[^"<>]*)?)', markup, re.I)
    output = []
    for value in values:
        value = urljoin(base_url, value.replace('\\/', '/'))
        if value.startswith(('http://', 'https://')) and value not in output:
            output.append(value)
    return output[:20]


def _candidate_urls(candidate: dict) -> list[str]:
    values = [candidate.get('image_url', '')]
    values.extend(candidate.get('alternate_urls') or [])
    for key in ('original_url', 'full_image_url', 'thumbnail_url'):
        if candidate.get(key):
            values.append(candidate[key])
    output = []
    for value in values:
        if not isinstance(value, str):
            continue
        query = parse_qs(urlsplit(value).query)
        for key in ('imgurl', 'mediaurl', 'url'):
            if query.get(key) and query[key][0].startswith(('http://', 'https://')):
                original = query[key][0]
                if original not in output:
                    output.append(original)
        if value.startswith(('http://', 'https://')) and value not in output:
            output.append(value)
    return output


def download_candidate(candidate: dict, target: Path, deadline: float | None = None) -> dict:
    """Try actual image URLs and publisher fallbacks; never encode image pixels."""
    attempts = []
    target.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(candidate['_download_cache_dir']) if candidate.get('_download_cache_dir') else None

    def finish(raw, url, content_type, http_status=200, transport='requests', cache_hit=False):
        info = _inspect_image_bytes(raw)
        path = target.with_suffix(info['extension'])
        path.write_bytes(raw)
        selected = {key: value for key, value in candidate.items() if not key.startswith('_') and key != 'browser_download'}
        selected['image_url'] = url
        result = {'path': path, 'candidate': selected, 'bytes': len(raw), 'http_status': http_status,
                  'content_type': content_type, 'transport': transport, 'cache_hit': cache_hit,
                  'download_attempts': attempts, **info}
        if cache_dir is not None and not cache_hit:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(url.encode()).hexdigest()
                (cache_dir / f'{digest}.bin').write_bytes(raw)
                (cache_dir / f'{digest}.json').write_text(json.dumps({'url': url, 'sha256': hashlib.sha256(raw).hexdigest(),
                    'content_type': content_type, 'http_status': 200, 'created_at': time.time()}), encoding='utf-8')
            except OSError:
                LOGGER.debug('Image download cache unavailable.', exc_info=True)
        LOGGER.info('[DOWNLOAD] GET %s | HTTP %s | Content-Type: %s | Bytes: %s%s',
                    url, http_status, content_type, len(raw), ' | cache' if cache_hit else '')
        return result

    def fetch(url, referer='', allow_html=False):
        for _ in range(5):
            if _budget_expired(deadline):
                raise TimeoutError('Candidate download time budget exhausted.')
            public_image_url(url)
            headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'image/avif,image/webp,image/*,*/*;q=0.8'}
            if referer:
                headers['Referer'] = referer
            remaining = max(.001, deadline - time.monotonic()) if deadline is not None else 15
            evidence = {'url': url, 'referer': referer}
            attempts.append(evidence)
            try:
                with requests.get(url, stream=True, allow_redirects=False,
                                  timeout=(min(7, remaining), min(15, remaining)), headers=headers) as response:
                    content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
                    evidence.update(http_status=response.status_code, content_type=content_type)
                    LOGGER.info('[DOWNLOAD] GET %s | HTTP %s | Content-Type: %s', url, response.status_code, content_type or '(missing)')
                    if response.is_redirect:
                        url = urljoin(url, response.headers['Location'])
                        continue
                    if response.status_code != 200:
                        raise ValueError(f'HTTP {response.status_code}')
                    image_response = content_type.startswith('image/')
                    html_response = allow_html and content_type in {'text/html', 'application/xhtml+xml'}
                    if not image_response and not html_response:
                        raise ValueError(f'Expected image/*, received {content_type or "no Content-Type"}.')
                    maximum = MAX_IMAGE_BYTES if image_response else 2_000_000
                    if int(response.headers.get('Content-Length', '0') or 0) > maximum:
                        raise ValueError('Response is too large.')
                    chunks = []
                    size = 0
                    for chunk in _download_chunks(response, deadline):
                        size += len(chunk)
                        if size > maximum:
                            raise ValueError('Response is too large.')
                        chunks.append(chunk)
                    raw = b''.join(chunks)
                    evidence['bytes'] = len(raw)
                    if html_response:
                        return {'html': raw.decode(response.encoding or 'utf-8', errors='replace'), 'url': url}
                    return finish(raw, url, content_type)
            except Exception as exc:
                evidence['error'] = str(exc)
                raise
        raise ValueError('Too many candidate redirects.')

    urls = _candidate_urls(candidate)
    source = str(candidate.get('source_url') or '')
    # A Windows Playwright fetch supplies exactly the original HTTP response bytes.
    browser = candidate.get('browser_download') or {}
    if browser:
        try:
            url = str(browser.get('url') or candidate.get('image_url') or '')
            public_image_url(url)
            content_type = str(browser.get('content_type') or '').lower().split(';')[0]
            if browser.get('http_status') != 200 or not content_type.startswith('image/'):
                raise ValueError('Browser fallback is not an HTTP 200 image response.')
            raw = base64.b64decode(browser['data_base64'], validate=True)
            attempts.extend(browser.get('download_attempts') or [])
            return finish(raw, url, content_type, transport='playwright')
        except Exception as exc:
            attempts.append({'transport': 'playwright', 'error': str(exc)})

    for url in urls:
        if cache_dir is not None:
            try:
                digest = hashlib.sha256(url.encode()).hexdigest()
                meta = json.loads((cache_dir / f'{digest}.json').read_text(encoding='utf-8'))
                raw = (cache_dir / f'{digest}.bin').read_bytes()
                if (meta.get('url') == url and meta.get('http_status') == 200
                        and str(meta.get('content_type', '')).startswith('image/')
                        and hashlib.sha256(raw).hexdigest() == meta.get('sha256')):
                    public_image_url(url)
                    return finish(raw, url, meta['content_type'], cache_hit=True)
            except (OSError, ValueError, KeyError):
                pass
        try:
            return fetch(url, source)
        except Exception:
            pass

    # A Lens result may point to a page instead of a file. Extract social images,
    # original image anchors and largest srcsets before trying another candidate.
    pages = [source] if source else []
    pages.extend(url for url in urls if url not in pages)
    for page_url in pages[:3]:
        if _budget_expired(deadline):
            break
        try:
            page = fetch(page_url, source, allow_html=True)
            if 'html' not in page:
                return page
            for image_url in _page_image_urls(page['html'], page['url']):
                if _budget_expired(deadline):
                    break
                try:
                    return fetch(image_url, page['url'])
                except Exception:
                    pass
        except Exception:
            pass

    # Retry a hotlink endpoint with its own origin if publisher Referer failed.
    for url in urls:
        if _budget_expired(deadline):
            break
        parts = urlsplit(url)
        for referer in (f'{parts.scheme}://{parts.netloc}/', ''):
            if referer == source:
                continue
            try:
                return fetch(url, referer)
            except Exception:
                pass
    return {'candidate': {key: value for key, value in candidate.items() if not key.startswith('_') and key != 'browser_download'},
            'error': 'No downloadable image found. ' + '; '.join(str(row.get('error', '')) for row in attempts[-4:]),
            'download_attempts': attempts, 'budget_exhausted': _budget_expired(deadline)}


def recover_slide(base: Path, slide: dict, search: dict, settings: Settings) -> dict:
    """Save the highest-ranked downloadable candidate regardless of confidence."""
    result = {'provider': search.get('provider', 'google_lens'), 'status': search.get('status', 'NOT_FOUND'),
        'source_url': None, 'source_domain': None, 'score': 0.0, 'downloaded': None,
        'fallback': slide['clean'], 'error': search.get('error', ''), 'candidates': [],
        'query': search.get('query', slide.get('reverse_query', {})), 'cache_hit': bool(search.get('cache_hit', False)),
        'diagnostics': search.get('diagnostics', {}), 'raw_download': False, 'modified': False,
        'candidates_checked': 0, 'best_candidate': None,
        'timings': {'candidate_download': 0, 'candidate_matching': 0, 'source_ocr': 0}}
    candidates = search.get('candidates', [])[:settings.max_candidates]
    deadline = search.get('_validation_deadline')
    LOGGER.info('[LENS] slide %s | query=%s | Google results=%s', slide['position'], result['query'], len(candidates))
    if _budget_expired(deadline) and candidates:
        result.update(status='TIMEOUT', budget_exhausted=True, error='Candidate download deadline expired before any image was downloaded.')
        result['candidates'] = [{'image_url': item.get('image_url'), 'source_url': item.get('source_url'),
            'accepted': False, 'score': 0, 'reason': 'TIME_BUDGET_EXHAUSTED'} for item in candidates]
        result['candidates_checked'] = len(candidates)
        return result
    if not candidates:
        if result['status'] == 'CANDIDATES':
            result['status'] = 'NOT_FOUND'
        return result
    (base / 'metadata').mkdir(parents=True, exist_ok=True)
    recovered = base / 'recovered'
    recovered.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='candidates-', dir=base / 'metadata') as temporary:
        folder = Path(temporary)
        started = time.perf_counter()
        def download(item):
            index, candidate = item
            if _budget_expired(deadline):
                return {'candidate': candidate, 'error': 'Candidate download deadline expired.', 'budget_exhausted': True}
            candidate = {**candidate, '_download_cache_dir': str(settings.output_base / '.cache' / 'web_downloads')}
            target = folder / f'{index:02d}.jpg'
            return download_candidate(candidate, target, deadline=deadline) if deadline is not None else download_candidate(candidate, target)
        with ThreadPoolExecutor(max_workers=settings.candidate_workers) as pool:
            downloaded = list(pool.map(download, enumerate(candidates)))
        result['timings']['candidate_download'] = round(time.perf_counter() - started, 3)
        usable = []
        for index, item in enumerate(downloaded, 1):
            data = {'candidate': index, 'image_url': item['candidate'].get('image_url'),
                    'source_url': item['candidate'].get('source_url'), 'download_attempts': item.get('download_attempts', [])}
            if item.get('error'):
                data.update(accepted=False, score=0, reason='DOWNLOAD_FAILED', error=item['error'])
                result['candidates'].append(data)
                LOGGER.info('[LENS] candidate %s: page=%s image=%s DOWNLOAD_FAILED %s',
                            index, data['source_url'], data['image_url'], data['error'])
                continue
            # Decode validation does not transform bytes. This also validates mocked
            # transports and future provider implementations before they enter RAW.
            try:
                info = _inspect_image_bytes(item['path'].read_bytes())
                item.update(info)
            except Exception as exc:
                data.update(accepted=False, score=0, reason='INVALID_IMAGE_BYTES', error=str(exc))
                result['candidates'].append(data)
                continue
            started = time.perf_counter()
            try:
                comparison = compare_candidate(base / slide['original'], item['path'], base / slide['ocr']['mask'],
                    settings.candidate_min_dimension, settings.match_threshold, clean=base / slide['clean'])
            except Exception as exc:
                comparison = {'accepted': False, 'score': 0, 'width': info['width'], 'height': info['height'],
                              'reason': 'VISUAL_RANKING_FAILED', 'metrics': {'error': str(exc)}}
            result['timings']['candidate_matching'] += time.perf_counter() - started
            data.update(comparison)
            data['visual_score'] = float(comparison.get('score', 0))
            data['visual_accepted'] = comparison.get('accepted', False)
            data['accepted'] = True  # Decoded web candidates are eligible at every score.
            data['width'], data['height'] = info['width'], info['height']
            data.update(http_status=item.get('http_status'), content_type=item.get('content_type'), bytes=item['bytes'])
            started = time.perf_counter()
            try:
                check = candidate_overlay_check(base / slide['original'], item['path'], base / slide['ocr']['mask'], comparison, settings)
                data.setdefault('metrics', {})['source_ocr'] = check
                data['candidate_text_area_ratio'] = check['candidate_text_area_ratio']
                data['contains_text'] = check['contains_text']
            except Exception as exc:
                # OCR is optional ranking evidence, never a gate to saving an image.
                data.setdefault('metrics', {})['source_ocr'] = {'ranking_only': True, 'error': str(exc)}
                data['candidate_text_area_ratio'] = 0.0
                data['contains_text'] = None
            result['timings']['source_ocr'] += time.perf_counter() - started
            text_penalty = min(.25, data['candidate_text_area_ratio'] * 1.2)
            if data['metrics'].get('source_ocr', {}).get('overlaid_regions'):
                text_penalty = max(text_penalty, .12)
            thumb = bool(item['candidate'].get('thumbnail')) or min(info['width'], info['height']) < settings.candidate_min_dimension
            data['thumbnail'] = thumb
            data['score'] = round(max(0.0, data['visual_score'] - text_penalty - (.04 if thumb else 0)), 4)
            data['reason'] = 'WEB_IMAGE_RANKED'
            data['ranking_penalties'] = {'text': round(text_penalty, 4), 'thumbnail': .04 if thumb else 0}
            usable.append((data, item))
            result['candidates'].append(data)
            LOGGER.info('[LENS] candidate %s: page=%s image=%s resolution=%sx%s visual_score=%.4f score=%.4f text_area=%.4f',
                index, data['source_url'], data['image_url'], data['width'], data['height'], data['visual_score'], data['score'], data['candidate_text_area_ratio'])
        result['candidates_checked'] = len(result['candidates'])
        if not usable:
            result.update(status='TIMEOUT' if _budget_expired(deadline) else 'DOWNLOAD_FAILED', error='Google returned candidates but none could be downloaded as a valid HTTP 200 image.',
                          budget_exhausted=_budget_expired(deadline))
            return result
        # Score is the primary key. Resolution/fullness only break ties; a LOW
        # match remains eligible and is written unchanged when it is the best.
        usable.sort(key=lambda pair: (pair[0]['score'], pair[0]['width'] * pair[0]['height'],
                                      pair[0].get('metrics', {}).get('background_coverage', 0)), reverse=True)
        for ranking, (data, _) in enumerate(usable, 1):
            data['ranking'] = ranking
        data, item = usable[0]
        destination = recovered / f"{slide['position']:02d}_SOURCE{item['extension']}"
        # Exactly one raw source per slide; remove only this slide's stale managed extension.
        for old in recovered.glob(f"{slide['position']:02d}_SOURCE.*"):
            if old != destination and old.suffix.lower() in IMAGE_SUFFIXES.values():
                old.unlink()
        shutil.copyfile(item['path'], destination)
        selected = item['candidate']
        raw = destination.read_bytes()
        confidence = 'HIGH' if data['visual_score'] >= .9 else 'MEDIUM' if data['visual_score'] >= .7 else 'LOW'
        result.update(status='MATCH', source='web_download', raw_download=True, modified=False,
            source_url=selected.get('source_url') or selected['image_url'], image_url=selected['image_url'],
            source_domain=urlsplit(selected.get('source_url') or selected['image_url']).hostname,
            score=data['score'], visual_score=data['visual_score'], ranking=1, confidence=confidence,
            contains_text=data['contains_text'], candidate_text_area_ratio=data['candidate_text_area_ratio'],
            resolution={'width': data['width'], 'height': data['height']}, downloaded=f'recovered/{destination.name}',
            sha256=hashlib.sha256(raw).hexdigest(), download_sha256=hashlib.sha256(raw).hexdigest(),
            http_status=item.get('http_status'), content_type=item.get('content_type'), bytes=len(raw),
            download_transport=item.get('transport', 'requests'), download_cache_hit=bool(item.get('cache_hit')),
            validation=data['metrics'], best_candidate=data, error='')
        result['timings'] = {name: round(value, 3) for name, value in result['timings'].items()}
        LOGGER.info('[LENS] slide %s SELECTED candidate %s | confidence=%s | raw_download=true modified=false',
                    slide['position'], data['candidate'], confidence)
        LOGGER.info('SAVED: %s | Bytes: %s | SHA256: %s', destination, len(raw), result['sha256'])
        return result
