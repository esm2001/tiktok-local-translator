"""Audit exported RAW files and TXT on disk; optionally compare fresh web bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

from PIL import Image
import requests

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'translator'))
from src.pipeline import validate_outputs


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('post', type=Path)
    parser.add_argument('--web', action='store_true', help='GET each selected URL again and compare its exact bytes.')
    parser.add_argument('--report', type=Path, default=PROJECT / '.cache/raw-validation/report.json')
    args = parser.parse_args()
    base = args.post.resolve()
    manifest = json.loads((base / 'metadata/manifest.json').read_text(encoding='utf-8'))
    report = {'post': str(base), 'url': manifest['resolved_url'], 'validation': validate_outputs(base, manifest['slides']),
              'timings': manifest.get('timings', {}), 'slides': [], 'web_check_requested': args.web}

    def inspect(slide):
        recovered = slide.get('recovered', {})
        legacy = slide.get('reverse_search', {})
        relative = recovered.get('file') or legacy.get('downloaded')
        row = {'slide': slide['position'], 'file': relative, 'url': recovered.get('url') or legacy.get('image_url'),
               'engine': slide.get('translation', {}).get('engine'),
               'translation': slide.get('translation', {}).get('text'), 'score': recovered.get('score'),
               'contains_text': recovered.get('contains_text'), 'errors': []}
        if not relative:
            row['errors'].append('MISSING_WEB_IMAGE')
            return row
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            row['errors'].append('OUTSIDE_POST')
            return row
        raw = path.read_bytes()
        row.update(bytes=len(raw), sha256=digest(raw))
        row['raw_provenance'] = recovered.get('source') == 'web_download' and recovered.get('raw_download') is True and recovered.get('modified') is False
        row['download_hash_matches'] = row['sha256'] == recovered.get('sha256') == legacy.get('sha256')
        snapshot_key = hashlib.sha256(str(row['url']).encode()).hexdigest()
        snapshot_root = base.parent / '.cache/web_downloads'
        snapshot = snapshot_root / (snapshot_key + '.bin')
        snapshot_meta = snapshot_root / (snapshot_key + '.json')
        if snapshot.is_file() and snapshot_meta.is_file():
            captured = json.loads(snapshot_meta.read_text(encoding='utf-8'))
            row['response_snapshot_matches'] = (snapshot.read_bytes() == raw
                and captured.get('http_status') == 200 and captured.get('sha256') == row['sha256']
                and str(captured.get('content_type', '')).startswith('image/'))
        row['differs_from_clean'] = raw != (base / slide['clean']).read_bytes()
        for key in ('raw_provenance', 'download_hash_matches', 'differs_from_clean'):
            if not row[key]:
                row['errors'].append(key.upper())
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            row.update(format=image.format, resolution=list(image.size))
        if not row['translation'] or slide['translation'].get('fallback_to_english'):
            row['errors'].append('NO_SPANISH_TRANSLATION')
        if args.web:
            try:
                selected = legacy.get('best_candidate', {}) or {}
                attempts = selected.get('download_attempts', [])
                successful = [attempt for attempt in attempts if attempt.get('url') == row['url']
                              and attempt.get('http_status') == 200
                              and str(attempt.get('content_type', '')).startswith('image/')]
                source = (successful[-1].get('referer', '') if successful else selected.get('source_url', ''))
                headers = {'User-Agent': 'Mozilla/5.0',
                           'Accept': 'image/*' if legacy.get('download_transport') == 'playwright'
                           else 'image/avif,image/webp,image/*,*/*;q=0.8'}
                if source:
                    headers['Referer'] = source
                response = requests.get(row['url'], headers=headers, timeout=(7, 25))
                row['web'] = {'status': response.status_code, 'content_type': response.headers.get('Content-Type'),
                              'bytes': len(response.content), 'sha256': digest(response.content),
                              'exact_bytes': response.content == raw, 'url': response.url}
                if response.status_code != 200 or not str(row['web']['content_type']).startswith('image/') or not row['web']['exact_bytes']:
                    if row.get('response_snapshot_matches'):
                        row['web']['note'] = 'Fresh response differs or is unavailable; file exactly matches the captured HTTP 200 image response.'
                    else:
                        row['errors'].append('FRESH_WEB_COMPARE_FAILED')
            except Exception as exc:
                row['errors'].append('FRESH_WEB_GET_FAILED: ' + str(exc))
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        report['slides'] = list(pool.map(inspect, manifest['slides']))
    report['spanish_count'] = sum(bool(s.get('translation', {}).get('text')) and not s.get('translation', {}).get('fallback_to_english') for s in manifest['slides'])
    report['status'] = 'PASS' if report['validation']['complete'] and all(not row['errors'] for row in report['slides']) else 'FAIL'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # A separate QA sheet allows visual review without modifying recovered files.
    rows = report['slides']
    sheet = Image.new('RGB', (280 * 4, 330 * ((len(rows) + 3) // 4)), '#dddddd')
    for index, row in enumerate(rows):
        if row.get('file'):
            with Image.open(base / row['file']) as image:
                thumbnail = image.convert('RGB')
                thumbnail.thumbnail((274, 324))
                sheet.paste(thumbnail, ((index % 4) * 280, (index // 4) * 330))
    sheet.save(args.report.with_suffix('.png'))
    print(json.dumps({key: report[key] for key in ('status', 'validation', 'spanish_count', 'timings')}, ensure_ascii=False))
    print('Report: ' + str(args.report.resolve()))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
