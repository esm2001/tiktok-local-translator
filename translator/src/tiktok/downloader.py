import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import requests

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


class DownloadError(RuntimeError):
    pass


def validate_tiktok_url(url: str) -> str:
    url = (url or '').strip()
    try:
        parts = urlsplit(url)
        host = (parts.hostname or '').lower()
        valid = parts.scheme in {'http', 'https'} and (host == 'tiktok.com' or host.endswith('.tiktok.com'))
        if not valid or parts.username or parts.password or parts.port not in {None, 80, 443}:
            raise ValueError
    except ValueError:
        raise ValueError('Pega un enlace http(s) válido de TikTok (tiktok.com, vm.tiktok.com o vt.tiktok.com).') from None
    return url


def extract_post_id(url: str) -> str | None:
    match = re.search(r'/(?:video|photo)/(\d{15,25})(?:[/?#]|$)', urlsplit(url).path + '/')
    return match.group(1) if match else None


def resolve_tiktok_url(url: str) -> str:
    # A canonical post URL needs no HTTP visit; this also permits offline cache hits.
    if extract_post_id(url):
        return url
    try:
        with requests.Session() as session:
            current = url
            for _ in range(8):
                response = session.get(current, headers={'User-Agent': USER_AGENT}, allow_redirects=False, timeout=25)
                if response.is_redirect:
                    from urllib.parse import urljoin
                    current = validate_tiktok_url(urljoin(current, response.headers['Location']))
                    continue
                response.raise_for_status()
                if not extract_post_id(current):
                    raise DownloadError('El enlace no identifica un post. Puede haberse eliminado o ser privado.')
                return current
    except requests.RequestException as exc:
        raise DownloadError('TikTok no permite resolver el enlace. Comprueba si es público o requiere cookies.') from exc
    raise DownloadError('TikTok redirige demasiadas veces; prueba el enlace completo del post.')


def gallery_command(url: str, output_dir: Path, cookies_file: Path) -> list[str]:
    command = ['gallery-dl', '--config-ignore', '--no-mtime', '-D', str(output_dir), '-f', '{num:>02}.{extension}']
    for kind, value in [('photos', 'true'), ('videos', 'false'), ('audio', 'false'), ('covers', 'false')]:
        command += ['-o', f'extractor.tiktok.{kind}={value}']
    command += ['-o', 'extractor.retries=2', '-o', 'extractor.timeout=25']
    if cookies_file.is_file():
        command += ['--cookies', str(cookies_file)]
    return command + [url]


def download_images(url: str, original_dir: Path, cookies_file: Path) -> tuple[list[Path], str]:
    original_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(gallery_command(url, original_dir, cookies_file), text=True, capture_output=True, timeout=240)
    except FileNotFoundError as exc:
        raise DownloadError('gallery-dl no está instalado en el servicio translator.') from exc
    except subprocess.TimeoutExpired as exc:
        raise DownloadError('TikTok no respondió a tiempo al descargar las fotos.') from exc
    detail = (proc.stderr or proc.stdout or '').strip()[-2000:]
    images = sorted((p for p in original_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p: (int(re.search(r'\d+', p.stem).group()) if re.search(r'\d+', p.stem) else 99999, p.name))
    # A failed download must not silently mark a partial carousel complete.
    if proc.returncode:
        raise DownloadError('gallery-dl no pudo completar la descarga. El post puede ser privado, eliminado o requerir cookies. ' + detail[-900:])
    return images, detail
