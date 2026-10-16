"""Run one real Lens query, allowing the user to solve Google's CAPTCHA.

No CAPTCHA solver or evasion is used. Close the window or Ctrl+C to stop.
The dedicated profile and reports remain under the ignored project .cache.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'launcher'))
from reverse_search.google_lens import GoogleLensProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('post_id')
    parser.add_argument('position', type=int)
    parser.add_argument('--output-base', type=Path, default=PROJECT.parent/'TikTokTraducidos')
    parser.add_argument('--kind', choices=['original_full', 'clean_full'], default='original_full')
    parser.add_argument('--interactive-timeout', type=float, default=180)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--hold-open', type=float, default=300,
                        help='Keep the dedicated window available after a failure for manual inspection.')
    args = parser.parse_args()
    if not args.post_id.isascii() or not args.post_id.isdecimal() or args.position < 1:
        parser.error('Use a numeric post ID and a positive slide position.')
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    post = args.output_base/args.post_id
    slide = next((s for s in json.loads((post/'metadata/manifest.json').read_text(encoding='utf-8'))['slides']
                  if s['position'] == args.position), None)
    if slide is None:
        parser.error('The local manifest has no such slide.')
    image = (post/slide['original' if args.kind == 'original_full' else 'clean']).resolve()
    if not image.is_relative_to(post.resolve()) or not image.is_file():
        parser.error('The managed image must exist inside the selected local post.')
    folder = PROJECT/'.cache/recovery-diagnosis'
    folder.mkdir(parents=True, exist_ok=True)
    report = folder/f'lens-manual-{args.post_id}-{args.position:02d}-{args.kind}.json'
    print(f'Lens: {args.post_id}, slide {args.position}, query {args.kind}.', flush=True)
    print('Si Google muestra un CAPTCHA, resuélvelo manualmente en la ventana dedicada.', flush=True)
    with GoogleLensProvider(PROJECT/'.cache/browser/google_lens', channel='chrome',
                            headless=args.headless, timeout_seconds=30,
                            interactive_timeout=max(0, args.interactive_timeout)) as provider:
        result = provider.search(image, force=True)
        result['query'] = {'kind': args.kind, 'post_id': args.post_id, 'position': args.position}
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.headless and result['status'] != 'CANDIDATES' and args.hold_open > 0:
            print(f'La ventana dedicada queda disponible {args.hold_open:.0f}s para revisión manual. Cierra esa ventana para terminar.', flush=True)
            deadline = time.monotonic()+args.hold_open
            while time.monotonic()<deadline and provider._page is not None and not provider._page.is_closed():
                provider._page.wait_for_timeout(1000)
    result['query'] = {'kind': args.kind, 'post_id': args.post_id, 'position': args.position}
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Report: {report.relative_to(PROJECT)}', flush=True)
    return 0 if result['status'] == 'CANDIDATES' else 2


if __name__ == '__main__':
    raise SystemExit(main())
