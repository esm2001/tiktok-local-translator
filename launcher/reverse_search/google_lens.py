"""Google Lens's public web UI, isolated from the translation pipeline.

This is browser automation, not an official Google Lens API. Google can change
the interface. Failed searches retain CLEAN for inspection and try other sources.
Playwright and a browser are imported/launched only when an uncached search
actually needs them. One persistent, dedicated profile serves all slides.
"""

from __future__ import annotations

import hashlib
import base64
import html
import ipaddress
import json
import logging
import re
import socket
import time
import uuid
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .base import ReverseSearchProvider

LOGGER = logging.getLogger(__name__)
PARSER_VERSION = 4
MATCH_CACHE_SECONDS = 7 * 24 * 3600
NOT_FOUND_CACHE_SECONDS = 3600
IMAGE_EXTENSIONS = re.compile(r"\.(?:jpe?g|png|webp|gif|bmp|avif)(?:$|[?#])", re.I)
DIMENSIONED_IMAGE = re.compile(
    r'\[\s*"(https?://[^"<>\s]+)"\s*,\s*(\d{2,5})\s*,\s*(\d{2,5})\s*\]', re.I
)


def _decode_markup(value: str) -> str:
    value = html.unescape(value).replace("\\/", "/")
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda match: chr(int(match[1], 16)), value)


def _url(value: str, base: str = "") -> str:
    value = html.unescape(value).strip()
    if value.startswith("//"):
        value = "https:" + value
    elif base:
        value = urljoin(base, value)
    try:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username:
            return ""
        host = parts.hostname.lower().rstrip(".")
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            return ""
        try:
            address = ipaddress.ip_address(host)
            if not address.is_global:
                return ""
        except ValueError:
            if "." not in host:
                return ""
        return value
    except (ValueError, TypeError):
        return ""


def _is_google(host: str) -> bool:
    host = host.lower()
    return (host == "google.com" or host.endswith(".google.com")
            or bool(re.search(r"(^|\.)google\.[a-z.]+$", host))
            or host.endswith((".gstatic.com", ".googleusercontent.com")))


def _image_url(value: str, base: str = "") -> str:
    value = _url(value, base)
    if not value:
        return ""
    parts = urlsplit(value)
    host = parts.hostname or ""
    google_image_host = (host.endswith('.googleusercontent.com') or host == 'googleusercontent.com'
                         or host.startswith('encrypted-tbn') and host.endswith('.gstatic.com'))
    if (_is_google(host) and not google_image_host) or host == "tiktok.com" or host.endswith((".tiktok.com", ".tiktokcdn.com", ".tiktokcdn-us.com")):
        return ""
    if "/favicon" in parts.path.lower():
        return ""
    return value


def _source_link(value: str, base: str = "https://www.google.com") -> str:
    value = _url(value, base)
    if not value:
        return ""
    parts = urlsplit(value)
    if _is_google(parts.hostname or ""):
        query = parse_qs(parts.query)
        for key in ("imgrefurl", "url", "q"):
            if query.get(key):
                target = _url(query[key][0])
                if target and not _is_google(urlsplit(target).hostname or ""):
                    return target
        return ""
    return value


class _Markup(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.image_links: list[str] = []
        self.metadata_images: list[str] = []
        self.images: list[str] = []
        self.result_thumbnails: list[tuple[str, str]] = []
        self.result_payloads: list[dict] = []
        self._anchors: list[str] = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "a":
            if attributes.get('m'):
                try:
                    payload = json.loads(attributes['m'])
                    if isinstance(payload, dict):
                        self.result_payloads.append(payload)
                except (ValueError, TypeError):
                    pass
            self._anchors.append(attributes.get("href", ""))
            if attributes.get("href"):
                self.links.append(attributes["href"])
        if tag == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            if key in {"og:image", "og:image:secure_url", "twitter:image", "twitter:image:src"}:
                self.metadata_images.append(attributes.get("content", ""))
        if tag == "img":
            if self._anchors and self._anchors[-1]:
                self.image_links.append(self._anchors[-1])
                self.result_thumbnails.append((self._anchors[-1], attributes.get('src') or attributes.get('data-src') or ''))
            for key in ("src", "data-src", "data-original"):
                if attributes.get(key):
                    self.images.append(attributes[key])
            srcset = []
            for item in attributes.get("srcset", "").split(","):
                fields = item.strip().split()
                if fields:
                    dimension = float(fields[1][:-1]) if len(fields) > 1 and re.fullmatch(r'[0-9.]+[wx]', fields[1]) else 0
                    srcset.append((dimension, fields[0]))
            self.images.extend(value for _, value in sorted(srcset))

    def handle_endtag(self, tag):
        if tag == "a" and self._anchors:
            self._anchors.pop()


def extract_candidates(markup: str, source_url: str = "", max_candidates: int = 12) -> list[dict]:
    """Parse image-result links, dimensioned payloads, or a source page's images.

    Full image URLs are preferred. Google proxies/thumbnails remain alternatives
    when an original endpoint fails, and become candidates when no full URLs exist.
    The backend validates the response and ranks images without a score gate.
    """
    raw_markup = markup
    markup = _decode_markup(markup)
    parser = _Markup()
    # HTMLParser decodes attribute entities itself. Decoding &quot; before parsing
    # would corrupt Bing's JSON m="{&quot;murl&quot;:...}" result attributes.
    parser.feed(raw_markup)
    candidates: list[dict] = []
    seen: set[str] = set()

    def add(image_value: str, reference: str = ""):
        image_value = _image_url(image_value, source_url)
        if not image_value or image_value in seen or len(candidates) >= max_candidates:
            return
        reference = _source_link(reference, source_url) if reference else ""
        seen.add(image_value)
        candidates.append({
            "image_url": image_value,
            "source_url": reference,
            "source_domain": urlsplit(reference or image_value).hostname or "",
        })

    for link in parser.links:
        parsed = urlsplit(urljoin("https://www.google.com", link))
        query = parse_qs(parsed.query)
        if query.get("imgurl"):
            add(query["imgurl"][0], (query.get("imgrefurl") or [""])[0])
    for payload in parser.result_payloads:
        if payload.get('murl'):
            add(payload['murl'], payload.get('purl', ''))
            matching = next((item for item in candidates if item['image_url'] == _image_url(payload['murl'], source_url)), None)
            if matching is not None and _image_url(str(payload.get('turl', ''))):
                matching.setdefault('alternate_urls', []).append(payload['turl'])
            if matching is not None and payload.get('t'):
                matching['title'] = str(payload['t'])[:400]
    for match in DIMENSIONED_IMAGE.finditer(markup):
        if min(int(match[2]), int(match[3])) >= 256 and max(int(match[2]), int(match[3])) >= 400:
            add(match[1], source_url)
            matching = next((item for item in candidates if item['image_url'] == _image_url(match[1], source_url)), None)
            if matching is not None:
                # Google payloads store height, width in this order.
                matching.update(height=int(match[2]), width=int(match[3]))
    if source_url:
        for value in parser.metadata_images:
            add(value, source_url)
        # Source pages sometimes omit social metadata. Largest srcset entries
        # come first so a page logo does not consume the entire candidate budget.
        for value in reversed(parser.images):
            if IMAGE_EXTENSIONS.search(value) and not (urlsplit(value).hostname or '').startswith('encrypted-tbn'):
                add(value, source_url)
    full_results = bool(candidates)
    for link, thumbnail in parser.result_thumbnails:
        thumbnail = _image_url(thumbnail, source_url)
        if not thumbnail:
            continue
        query = parse_qs(urlsplit(urljoin('https://www.google.com', link)).query)
        full = (query.get('imgurl') or [''])[0]
        existing = next((item for item in candidates if item['image_url'] == full), None)
        if existing is not None:
            existing.setdefault('alternate_urls', []).append(thumbnail)
        elif not full_results:
            add(thumbnail, _source_link(link))
            if candidates:
                candidates[-1]['thumbnail'] = True
    if not candidates and not source_url:
        # Lens sometimes exposes only a dimensioned proxy or result thumbnail.
        for match in DIMENSIONED_IMAGE.finditer(markup):
            add(match[1], '')
            if candidates:
                candidates[-1].setdefault('thumbnail', min(int(match[2]), int(match[3])) < 400)
        for value in parser.images:
            host = urlsplit(value).hostname or ''
            if host.startswith('encrypted-tbn') or host.endswith('.googleusercontent.com'):
                add(value, '')
                if candidates:
                    candidates[-1].setdefault('thumbnail', host.startswith('encrypted-tbn'))
    return candidates


def extract_source_urls(markup: str, max_sources: int = 3) -> list[str]:
    parser = _Markup()
    parser.feed(_decode_markup(markup))
    result = []
    seen = set()
    # Plain navigation/footer links must not finish a search prematurely.
    result_links = parser.image_links + [link for link in parser.links if "imgrefurl=" in link]
    for link in result_links:
        value = _source_link(link)
        if not value or value in seen or IMAGE_EXTENSIONS.search(value):
            continue
        # Image-search cards link directly to a publisher page. Exclude
        # advertising and Google navigation links before following anything.
        if (urlsplit(value).hostname or "").endswith((".doubleclick.net", ".googleadservices.com")):
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= max_sources:
            break
    return result


def _public_endpoint(value: str) -> bool:
    value = _url(value)
    if not value:
        return False
    parsed = urlsplit(value)
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        return bool(addresses) and all(ipaddress.ip_address(item[4][0]).is_global for item in addresses)
    except (OSError, ValueError):
        return False


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


def _source_page(value: str, timeout: float, deadline: float | None = None) -> tuple[str, str]:
    """Bounded public HTML fetch; inspect each redirect before following it."""
    from urllib.error import HTTPError

    opener = build_opener(_NoRedirect())
    for _ in range(4):
        if deadline is not None and time.monotonic() >= deadline:
            return '', value
        if not _public_endpoint(value):
            return "", value
        request = Request(value, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"})
        try:
            remaining = max(.001, deadline-time.monotonic()) if deadline is not None else timeout
            with opener.open(request, timeout=min(timeout, remaining)) as response:
                if "text/html" not in response.headers.get("Content-Type", "").lower():
                    return "", value
                body = response.read(2_000_001)
                if len(body) > 2_000_000:
                    return "", value
                charset = response.headers.get_content_charset() or "utf-8"
                return body.decode(charset, errors="replace"), value
        except HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
                value = urljoin(value, exc.headers["Location"])
                continue
            return "", value
        except (OSError, ValueError):
            return "", value
    return "", value


class GoogleLensProvider(ReverseSearchProvider):
    provider = "google_lens"

    def __init__(self, profile_dir: Path, headless: bool = False,
                 timeout_seconds: float = 45, max_candidates: int = 12,
                 channel: str | None = None, interactive_timeout: float = 120,
                 chrome_path: str | None = None):
        self.profile_dir = Path(profile_dir).resolve()
        self.headless = headless
        self.timeout_seconds = max(5, float(timeout_seconds))
        self.max_candidates = max(1, min(30, int(max_candidates)))
        self.channel = channel
        self.interactive_timeout = max(0, float(interactive_timeout))
        self.chrome_path = chrome_path
        cache_base = self.profile_dir.parent.parent if self.profile_dir.parent.name == "browser" else self.profile_dir.parent
        self.cache_dir = cache_base / "reverse_search"
        self._playwright = None
        self._context = None
        self._page = None
        self._captcha_attempted = False
        self._captcha_blocked = False
        self._deadline = None

    def set_deadline(self, deadline: float):
        self._deadline = float(deadline)
        if self._page is not None and not self._page.is_closed() and not self._budget_expired():
            self._page.set_default_timeout(self._timeout_ms(10))
            self._page.set_default_navigation_timeout(self._timeout_ms(self.timeout_seconds))

    def reset_for_post(self):
        """Allow the next TikTok its own consent/CAPTCHA check without reopening Chrome."""
        self._captcha_attempted = False
        self._captcha_blocked = False

    def _budget_expired(self):
        return self._deadline is not None and time.monotonic() >= self._deadline

    def _timeout_ms(self, seconds: float):
        if self._budget_expired():
            raise TimeoutError('Reverse search total time budget exhausted.')
        remaining = self._deadline-time.monotonic() if self._deadline is not None else seconds
        return max(1, min(seconds, remaining)*1000)

    def _start(self):
        if self._context is not None:
            if self._page is not None and not self._page.is_closed():
                return
            self.close()
        from playwright.sync_api import sync_playwright

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        options = {"headless": self.headless, "locale": "en-US", "viewport": {"width": 1360, "height": 900},
                   'timeout': self._timeout_ms(self.timeout_seconds)}
        if self.chrome_path:
            options["executable_path"] = self.chrome_path
        elif self.channel:
            options["channel"] = self.channel
        try:
            self._context = self._playwright.chromium.launch_persistent_context(str(self.profile_dir), **options)
        except Exception as exc:
            if not (self.channel or self.chrome_path):
                raise
            LOGGER.info("[lens] Chrome no disponible; probando Chromium de Playwright.")
            options.pop("channel", None)
            options.pop("executable_path", None)
            options['timeout'] = self._timeout_ms(self.timeout_seconds)
            self._context = self._playwright.chromium.launch_persistent_context(str(self.profile_dir), **options)
        self._context.set_default_timeout(self._timeout_ms(min(self.timeout_seconds, 10)))
        self._context.set_default_navigation_timeout(self._timeout_ms(self.timeout_seconds))
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()

    def close(self):
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                LOGGER.debug("[lens] No se pudo cerrar el navegador.", exc_info=True)
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                LOGGER.debug("[lens] No se pudo cerrar Playwright.", exc_info=True)
        self._context = self._playwright = self._page = None

    def _captcha(self) -> bool:
        if "/sorry/" in self._page.url:
            return True
        text = self._page.locator("body").inner_text(timeout=self._timeout_ms(3)).lower()
        if "unusual traffic" in text or "tráfico inusual" in text or "not a robot" in text:
            return True
        return self._page.locator('iframe[src*="recaptcha"][title*="challenge"], iframe[title*="reCAPTCHA"]').count() > 0

    def _wait_for_human(self, reason: str, captcha: bool = False) -> bool:
        if self.headless or self.interactive_timeout <= 0 or (captcha and self._captcha_attempted):
            return False
        if captcha:
            self._captcha_attempted = True
        LOGGER.warning("[lens] %s Resuélvelo en la ventana del navegador (máximo %.0f s).", reason, self.interactive_timeout)
        deadline = time.monotonic() + self.interactive_timeout
        if self._deadline is not None:
            deadline = min(deadline, self._deadline)
        while time.monotonic() < deadline:
            self._page.wait_for_timeout(min(1000, max(1, (deadline-time.monotonic())*1000)))
            if self._budget_expired():
                return False
            if captcha and not self._captcha():
                return True
            if not captcha and "consent.google." not in self._page.url:
                return True
        return False

    def _consent(self):
        # Persist Google's own consent choice; no cookies are copied from the
        # user's everyday browser. Prefer rejecting optional cookies.
        for label in (r"^(Reject all|Rechazar todo)$",):
            button = self._page.get_by_role("button", name=re.compile(label, re.I))
            if button.count() and button.first.is_visible():
                button.first.click(timeout=self._timeout_ms(10))
                self._page.wait_for_timeout(700)
                break
        if "consent.google." in self._page.url:
            return self._wait_for_human("Google necesita tu consentimiento.")
        return True

    def _upload(self, clean_path: Path):
        self._page.goto("https://www.google.com/?hl=en", wait_until="domcontentloaded")
        if not self._consent():
            return "UNAVAILABLE", "Google necesita completar el consentimiento en el navegador."
        if self._captcha() and not self._wait_for_human("Google ha mostrado un CAPTCHA.", captcha=True):
            self._captcha_blocked = True
            return "CAPTCHA", "CAPTCHA pendiente; se probarán fuentes web en caché y otros métodos."
        # Google now also mounts AI Mode attachment selectors on its home page.
        # Only Lens's encoded_image input is a reverse-search upload target.
        file_input = self._page.locator('input[type="file"][name="encoded_image"]')
        if not file_input.count():
            camera = self._page.get_by_role("button", name=re.compile(r"Search by image|Buscar por imagen|Búsqueda por imagen", re.I))
            if not camera.count():
                camera = self._page.locator('[aria-label="Search by image"], [aria-label="Buscar por imagen"], [aria-label="Búsqueda por imagen"]')
            if not camera.count():
                return "UNAVAILABLE", "No se ha encontrado el botón de Google Lens; la interfaz puede haber cambiado."
            camera.first.click(timeout=self._timeout_ms(10))
            file_input = self._page.locator('input[type="file"][name="encoded_image"]')
            # Lens mounts its upload modal asynchronously. An immediate count()
            # misses a real selector that appears shortly after the camera click.
            try:
                file_input.first.wait_for(state='attached', timeout=self._timeout_ms(min(8, self.timeout_seconds)))
            except Exception:
                pass
        if file_input.count():
            file_input.first.set_input_files(str(clean_path), timeout=self._timeout_ms(10))
        else:
            upload = self._page.get_by_text(re.compile(r"^(upload a file|subir un archivo|sube un archivo)$", re.I))
            if not upload.count():
                return "UNAVAILABLE", "Google Lens no ofrece el selector de archivo esperado."
            with self._page.expect_file_chooser(timeout=self._timeout_ms(self.timeout_seconds)) as chooser:
                upload.first.click()
            chooser.value.set_files(str(clean_path))
        self._page.wait_for_url(re.compile(r"https://(?:lens\.google\.[^/]+/|(?:www\.)?google\.[^/]+/(?:search|sorry/))"),
                                timeout=self._timeout_ms(self.timeout_seconds), wait_until="domcontentloaded")
        return None, None

    def _results(self):
        deadline = time.monotonic() + self.timeout_seconds
        if self._deadline is not None:
            deadline = min(deadline, self._deadline)
        while time.monotonic() < deadline:
            if self._captcha():
                if not self._wait_for_human("Google ha mostrado un CAPTCHA.", captcha=True):
                    self._captcha_blocked = True
                    return "CAPTCHA", [], "CAPTCHA pendiente; se probarán fuentes web en caché y otros métodos."
                deadline = min(time.monotonic() + self.timeout_seconds, self._deadline or float('inf'))
            markup = self._page.content()
            candidates = extract_candidates(markup, max_candidates=self.max_candidates)
            sources = extract_source_urls(markup)
            if candidates or sources:
                # Only public publisher pages surfaced by Lens are inspected;
                # og:image/full image URLs then go to the backend for checking.
                for source in sources[:3]:
                    if len(candidates) >= self.max_candidates:
                        break
                    if self._budget_expired():
                        return 'TIMEOUT', [], 'Se agotó el tiempo de Lens; se probarán otras fuentes web.'
                    body, final_url = _source_page(source, self._timeout_ms(min(8, self.timeout_seconds))/1000,
                                                  deadline=self._deadline)
                    if not body:
                        body, final_url = self._browser_source_page(source)
                    existing = {candidate["image_url"] for candidate in candidates}
                    for candidate in extract_candidates(body, final_url, self.max_candidates):
                        if candidate["image_url"] not in existing:
                            candidates.append(candidate)
                            existing.add(candidate["image_url"])
                        else:
                            item = next(item for item in candidates if item['image_url'] == candidate['image_url'])
                            if not item.get('source_url'):
                                item.update(source_url=candidate['source_url'], source_domain=candidate['source_domain'])
                        if len(candidates) >= self.max_candidates:
                            break
                return ("CANDIDATES" if candidates else "NOT_FOUND"), candidates, None
            text = self._page.locator("body").inner_text(timeout=self._timeout_ms(3)).lower()
            if any(message in text for message in ("no results found", "no matching images", "no se han encontrado resultados")):
                return "NOT_FOUND", [], None
            self._page.wait_for_timeout(700)
        return ("TIMEOUT" if self._budget_expired() else "UNAVAILABLE"), [], "Lens no devolvió enlaces utilizables a tiempo; se probarán otras fuentes web."

    def _browser_source_page(self, url: str) -> tuple[str, str]:
        """Reuse Lens cookies/headers when a publisher needs browser context."""
        if self._context is None or not _public_endpoint(url) or self._budget_expired():
            return '', url
        try:
            response = self._context.request.get(url, headers={'Accept': 'text/html', 'Referer': self._page.url},
                timeout=self._timeout_ms(8), max_redirects=0)
            try:
                if response.status != 200 or 'text/html' not in response.headers.get('content-type', '').lower():
                    return '', url
                body = response.body()
                if len(body) > 2_000_000:
                    return '', url
                return body.decode('utf-8', errors='replace'), response.url
            finally:
                response.dispose()
        except Exception:
            return '', url

    def prepare_downloads(self, result: dict) -> dict:
        """Bridge only failed public endpoints through the existing browser.

        A streaming HTTP preflight reads a few bytes and closes. Successful public
        URLs are downloaded by the backend. Endpoints blocked for ordinary HTTP
        requests use the browser request context and pass unchanged response bytes.
        """
        import requests
        if self._budget_expired():
            return result
        started = time.monotonic()
        bridge_deadline = min(started + 60, self._deadline or float('inf'))
        for candidate in result.get('candidates', []):
            if time.monotonic() >= bridge_deadline:
                break
            urls = [candidate.get('image_url', '')] + list(candidate.get('alternate_urls') or [])
            source = str(candidate.get('source_url') or '')
            ordinary_ok = False
            attempts = []
            for url in urls[:3]:
                if not _public_endpoint(url) or time.monotonic() >= bridge_deadline:
                    continue
                headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'image/*'}
                if source:
                    headers['Referer'] = source
                try:
                    timeout = max(.1, min(4, bridge_deadline - time.monotonic()))
                    with requests.get(url, stream=True, allow_redirects=False, timeout=timeout, headers=headers) as response:
                        content_type = response.headers.get('Content-Type', '').lower().split(';')[0]
                        attempts.append({'url': url, 'http_status': response.status_code, 'content_type': content_type,
                                         'transport': 'windows_requests_preflight'})
                        if response.status_code == 200 and content_type.startswith('image/'):
                            # A MIME header alone is insufficient when a proxy sends HTML.
                            prefix = next(response.iter_content(128), b'')
                            if not prefix.lstrip().lower().startswith((b'<!doctype', b'<html', b'<head', b'<body')) and len(prefix) >= 16:
                                ordinary_ok = True
                                break
                except Exception as exc:
                    attempts.append({'url': url, 'error': str(exc), 'transport': 'windows_requests_preflight'})
            if ordinary_ok:
                continue
            if self._context is None:
                try:
                    self._start()
                except Exception as exc:
                    LOGGER.info('[DOWNLOAD] Browser fallback unavailable: %s', exc)
                    break
            if source and time.monotonic() < bridge_deadline:
                markup, final_url = _source_page(source, min(4, max(.1, bridge_deadline-time.monotonic())), deadline=bridge_deadline)
                if not markup:
                    markup, final_url = self._browser_source_page(source)
                if markup:
                    for resolved in extract_candidates(markup, final_url, max_candidates=6):
                        if resolved['image_url'] not in urls:
                            urls.append(resolved['image_url'])
            for initial_url in urls[:10]:
                if time.monotonic() >= bridge_deadline:
                    break
                url = initial_url
                for _ in range(4):
                    if not _public_endpoint(url) or time.monotonic() >= bridge_deadline:
                        break
                    response = None
                    evidence = {'url': url, 'transport': 'playwright', 'referer': source or self._page.url}
                    attempts.append(evidence)
                    try:
                        response = self._context.request.get(url, headers={'Accept': 'image/*',
                            'Referer': source or self._page.url}, timeout=min(self._timeout_ms(8),
                            max(1, (bridge_deadline-time.monotonic())*1000)), max_redirects=0)
                        content_type = response.headers.get('content-type', '').lower().split(';')[0]
                        evidence.update(http_status=response.status, content_type=content_type)
                        if response.status in {301, 302, 303, 307, 308} and response.headers.get('location'):
                            url = urljoin(url, response.headers['location'])
                            continue
                        if response.status != 200 or not content_type.startswith('image/'):
                            break
                        if int(response.headers.get('content-length', '0') or 0) > 25_000_000:
                            raise ValueError('Browser image exceeds 25 MB.')
                        raw = response.body()
                        if not 128 <= len(raw) <= 25_000_000:
                            raise ValueError('Browser image size is not reasonable.')
                        # PIL is optional on the Windows launcher. Backend always
                        # verifies and fully decodes before writing recovered.
                        try:
                            import io
                            from PIL import Image
                            with Image.open(io.BytesIO(raw)) as image:
                                image.verify()
                        except ImportError:
                            if raw.lstrip().lower().startswith((b'<html', b'<!doctype')):
                                raise ValueError('Browser response contains HTML.')
                        candidate['browser_download'] = {'data_base64': base64.b64encode(raw).decode('ascii'),
                            'url': response.url, 'http_status': 200, 'content_type': content_type,
                            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'download_attempts': attempts}
                        candidate['alternate_urls'] = list(dict.fromkeys(urls[1:]))[:12]
                        LOGGER.info('[DOWNLOAD] browser GET %s | HTTP 200 | Content-Type: %s | Bytes: %s',
                                    response.url, content_type, len(raw))
                        break
                    except Exception as exc:
                        evidence['error'] = str(exc)
                        break
                    finally:
                        if response is not None:
                            response.dispose()
                if candidate.get('browser_download'):
                    break
            if not candidate.get('browser_download'):
                candidate['alternate_urls'] = list(dict.fromkeys(urls[1:]))[:12]
        result.setdefault('timings', {})['browser_download_fallback'] = round(time.monotonic()-started, 3)
        return result

    def _read_cache(self, image_hash: str):
        try:
            data = json.loads((self.cache_dir / f"{image_hash}.json").read_text(encoding="utf-8"))
            result = data["result"]
            ttl = MATCH_CACHE_SECONDS if result.get("status") == "CANDIDATES" else NOT_FOUND_CACHE_SECONDS
            if (data.get("image_sha256") == image_hash and data.get("parser_version") in {3, PARSER_VERSION}
                    and result.get("status") in {"CANDIDATES", "NOT_FOUND"}
                    and 0 <= time.time() - data["created_at"] <= ttl):
                result["cache_hit"] = True
                return result
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return None

    def _cached_discovery_fallback(self, result: dict, cached: dict | None) -> dict:
        """A failed refresh cannot discard valid URLs from a prior Google search."""
        if (cached and cached.get('status') == 'CANDIDATES' and cached.get('candidates')
                and result.get('status') in {'CAPTCHA', 'UNAVAILABLE', 'ERROR', 'TIMEOUT', 'NOT_FOUND'}):
            fallback = dict(cached)
            fallback.update(cache_hit=True, fresh_search_status=result.get('status'),
                fresh_search_error=result.get('error') or result.get('status'),
                fresh_search_diagnostics=result.get('diagnostics', {}), timings=result.get('timings', {}))
            LOGGER.warning('[LENS] Refresh %s; reusing %s verified URLs from cache (provider=%s). %s',
                           result.get('status'), len(fallback['candidates']), fallback.get('provider'), fallback['fresh_search_error'])
            return fallback
        return result

    def _original_discovery_cache(self, clean_path: Path) -> dict | None:
        """CLEAN can change with its mask; the same slide's ORIGINAL stays stable."""
        if clean_path.parent.name != 'clean' or not re.fullmatch(r'\d+_CLEAN', clean_path.stem, re.I):
            return None
        position = clean_path.stem.split('_', 1)[0]
        original_folder = clean_path.parent.parent / 'original'
        for original in sorted(original_folder.glob(position + '.*')):
            if not original.is_file() or not IMAGE_EXTENSIONS.search(original.name):
                continue
            digest = hashlib.sha256(original.read_bytes()).hexdigest()
            result = self._read_cache(digest)
            if result and result.get('status') == 'CANDIDATES' and result.get('candidates'):
                result.update(cache_fallback_from='original_full', discovery_query_file=str(original),
                              discovery_query_sha256=digest)
                return result
        return None

    def _write_cache(self, image_hash: str, result: dict):
        if result["status"] not in {"CANDIDATES", "NOT_FOUND"}:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            target = self.cache_dir / f"{image_hash}.json"
            temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_text(json.dumps({"image_sha256": image_hash, "parser_version": PARSER_VERSION,
                                             "created_at": time.time(), "result": result}, ensure_ascii=False), encoding="utf-8")
            temporary.replace(target)
        except OSError:
            LOGGER.debug("[lens] No se pudo guardar la caché.", exc_info=True)

    def _failure_diagnostics(self, query_hash: str) -> dict:
        """Keep evidence of UI/parsing failures in the ignored local cache."""
        if self._page is None or self._page.is_closed():
            return {'stage': 'browser_closed'}
        try:
            folder = self.cache_dir/'debug'
            folder.mkdir(parents=True, exist_ok=True)
            stem = f'{query_hash[:16]}-{int(time.time())}'
            html_path = folder/f'{stem}.html'
            screenshot = folder/f'{stem}.png'
            html_path.write_text(self._page.content(), encoding='utf-8')
            self._page.screenshot(path=str(screenshot), full_page=False, timeout=5000)
            parsed = urlsplit(self._page.url)
            return {'stage': 'upload_or_results', 'page_host': parsed.hostname, 'page_path': parsed.path,
                    'page_text': self._page.locator('body').inner_text(timeout=3000)[:1200],
                    'html_file': str(html_path.relative_to(self.cache_dir)),
                    'screenshot_file': str(screenshot.relative_to(self.cache_dir))}
        except Exception as exc:
            return {'stage': 'diagnostic_capture_failed', 'error': str(exc)[:300]}

    def search(self, clean_path: Path, force: bool = False) -> dict:
        started = time.monotonic()
        result = {"provider": self.provider, "status": "ERROR", "candidates": []}
        image_hash = 'unhashed'
        cached = None
        refresh_fallback = None
        try:
            clean_path = Path(clean_path).resolve(strict=True)
            image_hash = hashlib.sha256(clean_path.read_bytes()).hexdigest()
            cached = self._read_cache(image_hash)
            refresh_fallback = cached if cached and cached.get('status') == 'CANDIDATES' else self._original_discovery_cache(clean_path)
            if self._budget_expired() and not self._captcha_blocked:
                return self._cached_discovery_fallback({**result, 'status': 'TIMEOUT', 'error': 'Se agotó el tiempo de búsqueda visual.',
                        'timings': {'reverse_search': round(time.monotonic()-started, 3)}}, refresh_fallback)
            if not force and cached:
                cached["timings"] = {"reverse_search": round(time.monotonic() - started, 3)}
                return cached
            if self._captcha_blocked:
                result.update(status="CAPTCHA", error="CAPTCHA anterior pendiente; búsquedas restantes omitidas en esta sesión.")
            else:
                try:
                    self._start()
                except Exception as exc:
                    self.close()
                    result.update(status="TIMEOUT" if self._budget_expired() else "UNAVAILABLE", error=f"No se pudo iniciar Playwright/Chrome: {exc}. Instala las dependencias del launcher y Chromium si hace falta.")
                    result["timings"] = {"reverse_search": round(time.monotonic() - started, 3)}
                    return self._cached_discovery_fallback(result, refresh_fallback)
                status, error = self._upload(clean_path)
                if status:
                    result.update(status=status, error=error)
                else:
                    status, candidates, error = self._results()
                    result.update(status=status, candidates=candidates)
                    if error:
                        result["error"] = error
            result["timings"] = {"reverse_search": round(time.monotonic() - started, 3)}
            if result['status'] in {'UNAVAILABLE', 'ERROR', 'NOT_FOUND'}:
                result['diagnostics'] = self._failure_diagnostics(image_hash)
            result = self._cached_discovery_fallback(result, refresh_fallback)
            if not result.get('cache_hit'):
                self._write_cache(image_hash, result)
        except Exception as exc:
            LOGGER.debug("[lens] Búsqueda fallida.", exc_info=True)
            message = str(exc)
            unavailable = any(word in message.lower() for word in ("timeout", "net::err_", "browser has been closed", "context or browser has been closed"))
            result.update(status="TIMEOUT" if self._budget_expired() else ("UNAVAILABLE" if unavailable else "ERROR"), error=f"Búsqueda visual fallida: {message}")
            if result['status'] != 'TIMEOUT':
                result['diagnostics'] = self._failure_diagnostics(image_hash)
            result["timings"] = {"reverse_search": round(time.monotonic() - started, 3)}
            # Closing a browser once must not poison every later query/slide.
            if 'closed' in message.lower() or 'detached' in message.lower():
                self.close()
        return self._cached_discovery_fallback(result, refresh_fallback)

    def _wikimedia_text_search(self, text: str) -> dict:
        """Public Commons image search; return original media URLs and dimensions."""
        import requests
        search_text = re.sub(r'\s+portrait painting$', '', text, flags=re.I).strip()
        if search_text.lower() == 'philosopher':
            search_text = 'philosopher portrait painting'
        endpoint = 'https://commons.wikimedia.org/w/api.php'
        params = {'action': 'query', 'generator': 'search', 'gsrnamespace': 6,
                  'gsrsearch': search_text, 'gsrlimit': self.max_candidates,
                  'prop': 'imageinfo', 'iiprop': 'url|size|mime', 'format': 'json'}
        evidence = {'provider': 'wikimedia_text_search', 'query': search_text, 'url': endpoint}
        try:
            if self._budget_expired() or not _public_endpoint(endpoint):
                raise TimeoutError('No time remaining for public Commons search.')
            timeout = self._timeout_ms(12) / 1000
            response = requests.get(endpoint, params=params, allow_redirects=False, timeout=(min(5, timeout), timeout),
                headers={'User-Agent': 'TikTokImageTool/2.0 (local image source recovery)', 'Accept': 'application/json'})
            evidence.update(http_status=response.status_code, url=response.url)
            if response.status_code != 200:
                raise ValueError(f'Commons HTTP {response.status_code}')
            data = response.json()
            if data.get('error'):
                raise ValueError(str(data['error']))
            candidates = []
            pages = sorted(data.get('query', {}).get('pages', {}).values(), key=lambda page: page.get('index', 9999))
            for page in pages:
                for info in page.get('imageinfo', [])[:1]:
                    url = _image_url(info.get('url', ''))
                    if not url or not str(info.get('mime', '')).startswith('image/'):
                        continue
                    source = info.get('descriptionurl') or 'https://commons.wikimedia.org/wiki/' + page['title'].replace(' ', '_')
                    candidates.append({'image_url': url, 'source_url': source, 'source_domain': 'commons.wikimedia.org',
                        'width': int(info.get('width', 0)), 'height': int(info.get('height', 0)),
                        'title': page.get('title', ''), 'discovery_provider': 'wikimedia_text_search'})
            evidence['candidates'] = len(candidates)
            for index, candidate in enumerate(candidates, 1):
                LOGGER.info('[LENS] Commons candidate %s page=%s image=%s resolution=%sx%s', index,
                            candidate['source_url'], candidate['image_url'], candidate['width'], candidate['height'])
            return {'provider': 'wikimedia_text_search', 'status': 'CANDIDATES' if candidates else 'NOT_FOUND',
                    'candidates': candidates, 'diagnostics': {'textual_search_attempts': [evidence]}}
        except Exception as exc:
            evidence['error'] = str(exc)
            LOGGER.info('[LENS] Commons search failed: %s', exc)
            return {'provider': 'wikimedia_text_search', 'status': 'UNAVAILABLE', 'candidates': [],
                    'diagnostics': {'textual_search_attempts': [evidence]}, 'error': str(exc)}

    def _http_text_search(self, text: str) -> dict:
        """Use public image-search pages when the Lens browser session is blocked."""
        attempts = []
        engines = [
            ('google_images_http', 'https://www.google.com/search?' + urlencode({'q': text, 'udm': '2', 'hl': 'en'})),
            ('bing_images_http', 'https://www.bing.com/images/search?' + urlencode({'q': text, 'form': 'HDRSC2', 'first': '1'})),
        ]
        for provider, search_url in engines:
            if self._budget_expired():
                break
            started = time.monotonic()
            LOGGER.info('[LENS] textual fallback %s query=%s url=%s', provider, text, search_url)
            markup, final_url = _source_page(search_url, self._timeout_ms(12)/1000, deadline=self._deadline)
            blocked = bool(re.search(r'(?:unusual traffic|/sorry/|g-recaptcha|tr[a?]fico inusual)', markup, re.I))
            if blocked:
                candidates = []
            else:
                # Search pages are never treated as publisher pages: only result
                # payloads/imgres images, not page logos, become candidates.
                candidates = extract_candidates(markup, max_candidates=self.max_candidates)
            attempts.append({'provider': provider, 'query': text, 'url': search_url, 'final_url': final_url,
                'blocked': blocked, 'markup_bytes': len(markup.encode('utf-8')), 'candidates': len(candidates),
                'seconds': round(time.monotonic()-started, 3)})
            for index, candidate in enumerate(candidates, 1):
                LOGGER.info('[LENS] %s candidate %s page=%s image=%s', provider, index,
                            candidate.get('source_url'), candidate.get('image_url'))
            if candidates:
                tokens = [token.casefold() for token in re.findall(r"[A-Za-z?-?]{3,}", text)
                          if token.casefold() not in {'the', 'and', 'to', 'are', 'one', 'you', 'your', 'which', 'follow',
                                                     'portrait', 'painting', 'giving', 'holding', 'presenting'}]
                relevant = any(any(token in (' '.join(str(candidate.get(key, '')) for key in ('title', 'image_url', 'source_url'))).casefold()
                    for token in tokens) for candidate in candidates)
                # Some Bing pages populate unrelated discovery cards. Preserve
                # them as a last resort, but search actual image metadata first.
                if provider == 'bing_images_http' and not relevant:
                    commons = self._wikimedia_text_search(text)
                    attempts.extend(commons.get('diagnostics', {}).get('textual_search_attempts', []))
                    if commons.get('candidates'):
                        merged = commons['candidates'] + [candidate for candidate in candidates
                            if candidate['image_url'] not in {item['image_url'] for item in commons['candidates']}]
                        return {'provider': 'wikimedia_text_search', 'status': 'CANDIDATES',
                                'candidates': merged[:self.max_candidates],
                                'diagnostics': {'textual_search_attempts': attempts, 'bing_query_supported': False}, 'text_query': text}
                return {'provider': provider, 'status': 'CANDIDATES', 'candidates': candidates,
                        'diagnostics': {'textual_search_attempts': attempts}, 'text_query': text}
        if not self._budget_expired():
            commons = self._wikimedia_text_search(text)
            attempts.extend(commons.get('diagnostics', {}).get('textual_search_attempts', []))
            if commons.get('candidates'):
                return {'provider': 'wikimedia_text_search', 'status': 'CANDIDATES', 'candidates': commons['candidates'],
                        'diagnostics': {'textual_search_attempts': attempts}, 'text_query': text}
        return {'provider': 'google_images_http', 'status': 'TIMEOUT' if self._budget_expired() else 'NOT_FOUND',
                'candidates': [], 'error': 'Public Google/Bing image-search pages returned no downloadable image URLs.',
                'diagnostics': {'textual_search_attempts': attempts}, 'text_query': text}

    def search_text(self, text: str, force: bool = False) -> dict:
        """Google Images first, then public Google/Bing HTTP discovery if blocked."""
        started = time.monotonic()
        result = {'provider': self.provider, 'status': 'ERROR', 'candidates': []}
        text = ' '.join(str(text).split())[:240]
        digest = hashlib.sha256(('text-assisted-v1:' + text).encode('utf-8')).hexdigest()
        cached = self._read_cache(digest)
        if not text:
            return {**result, 'status': 'NOT_FOUND', 'error': 'No reliable OCR search text.'}
        if self._budget_expired():
            return self._cached_discovery_fallback({**result, 'status': 'TIMEOUT', 'error': 'Image-search time budget exhausted.',
                'timings': {'reverse_search': round(time.monotonic()-started, 3)}}, cached)
        if not force and cached:
            cached['timings'] = {'reverse_search': round(time.monotonic()-started, 3)}
            return cached
        if self._captcha_blocked:
            # Do not submit another Lens query or try to solve its CAPTCHA.
            result = self._http_text_search(text)
            result['diagnostics']['lens_session_status'] = 'CAPTCHA'
        else:
            try:
                self._start()
                self._page.goto('https://www.google.com/search?' + urlencode({'q': text, 'udm': '2', 'hl': 'en'}), wait_until='domcontentloaded')
                if not self._consent():
                    result.update(status='UNAVAILABLE', error='Google requires manual consent.')
                else:
                    status, candidates, error = self._results()
                    result.update(status=status, candidates=candidates)
                    if error:
                        result['error'] = error
            except Exception as exc:
                result.update(status='TIMEOUT' if self._budget_expired() else 'UNAVAILABLE', error=f'Text-assisted discovery failed: {exc}')
                if 'closed' in str(exc).lower() or 'detached' in str(exc).lower():
                    self.close()
            if result.get('status') != 'CANDIDATES' and not self._budget_expired():
                browser_result = result
                result = self._http_text_search(text)
                result['diagnostics']['browser_status'] = browser_result.get('status')
                result['diagnostics']['browser_error'] = browser_result.get('error', '')
        result['timings'] = {'reverse_search': round(time.monotonic()-started, 3)}
        result = self._cached_discovery_fallback(result, cached)
        if not result.get('cache_hit'):
            self._write_cache(digest, result)
        return result
