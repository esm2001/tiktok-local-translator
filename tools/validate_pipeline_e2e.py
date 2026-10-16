"""Isolated real pipeline validation with an explicitly offline Lens replay.

Downloads use real gallery-dl/TikTok; OCR, CLEAN, source validation and Ollama
are production code. Only source discovery and fixture transfer are replayed.
This cannot measure live Lens discovery or certify translation semantics.
Windows: python tools/validate_pipeline_e2e.py --container tiktok-translator
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'translator'))
sys.path.insert(0, str(PROJECT / 'tools'))
from regression_metrics import fixture_failures, load_dataset, media_root, ocr_score, text_score, translation_score


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_inputs(dataset_path, holdout_path=None):
    dataset = load_dataset(dataset_path)
    if holdout_path:
        holdout = load_dataset(holdout_path)
        overlap = set(dataset['slides']) & set(holdout['slides'])
        if overlap:
            raise ValueError('Holdout overlaps regression cases: ' + ', '.join(sorted(overlap)))
        if dataset['media_root'] != holdout['media_root']:
            raise ValueError('Regression and holdout must use the same media root.')
        dataset = {**dataset, 'slides': {**dataset['slides'], **holdout['slides']}}
    return dataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_memory():
    import resource
    current = None
    status = Path('/proc/self/status')
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                current = round(int(line.split()[1]) / 1024, 3)
                break
    return {'current_rss_mb': current,
            'process_peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
            'scope': 'Pipeline worker only; remote Ollama and browser RSS are excluded.'}


def safe_sandbox(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    allowed = root / '.cache' / 'regression' / 'e2e'
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError('Pipeline output must be a new run under media/.cache/regression/e2e/.')
    if output.exists():
        raise ValueError('E2E output already exists; use a fresh run to test real downloads.')
    return output


def existing_snapshot(root, post_ids):
    """Protect every existing user artifact, not only the original photographs."""
    root = Path(root)
    return {path.relative_to(root).as_posix(): digest(path)
            for identifier in sorted(set(post_ids))
            for path in sorted((Path(root) / identifier).rglob('*')) if path.is_file()}


def clone_existing_outputs(root, source, destination, post_ids):
    """Copy prior evidence into a new confined run; never migrate it in place."""
    started = time.perf_counter()
    root = Path(root).resolve()
    allowed = root / '.cache/regression/e2e'
    source = Path(source)
    source = (source if source.is_absolute() else root / source).resolve()
    destination = safe_sandbox(root, destination)
    if source == allowed or not source.is_relative_to(allowed) or not source.is_dir():
        raise ValueError('Previous pipeline output must exist inside media/.cache/regression/e2e/.')
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Previous evidence and the new sandbox must be independent directories.')
    identifiers = sorted(set(post_ids))
    for identifier in identifiers:
        if not re.fullmatch(r'\d{15,25}', identifier):
            raise ValueError('Invalid post identifier for evidence clone.')
        post = source / identifier
        manifest_path = post / 'metadata/manifest.json'
        if not manifest_path.is_file():
            raise ValueError('Previous manifest missing for ' + identifier)
        for path in [post, *post.rglob('*')]:
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ValueError('Previous evidence contains a link or escaped path.')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('post_id') != identifier or not manifest.get('resolved_url'):
            raise ValueError('Previous manifest identity is invalid for ' + identifier)
        if not (post / 'original').is_dir():
            raise ValueError('Previous originals missing for ' + identifier)
        if not any(path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
                   for path in (post / 'original').iterdir()):
            raise ValueError('Previous originals are empty for ' + identifier)
    before = existing_snapshot(source, identifiers)
    for identifier in identifiers:
        shutil.copytree(source / identifier, destination / identifier)
    if before != existing_snapshot(destination, identifiers) or before != existing_snapshot(source, identifiers):
        raise ValueError('Evidence clone hash verification failed.')
    return {'source': str(source), 'post_ids': identifiers, 'file_count': len(before),
            'clone_matches_prior_evidence': True, 'prior_hashes': before,
            'clone_and_hash_seconds': round(time.perf_counter() - started, 6)}


def known_urls(dataset, root):
    identifiers = list(dict.fromkeys(case['post_id'] for case in dataset['slides'].values()))
    urls = []
    for identifier in identifiers:
        manifest_path = root / identifier / 'metadata/manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
        explicit = {case.get('canonical_url') for case in dataset['slides'].values()
                    if case['post_id'] == identifier and case.get('canonical_url')}
        if len(explicit) > 1:
            raise ValueError('Conflicting canonical URLs for ' + identifier)
        url = manifest.get('resolved_url') or next(iter(explicit), '')
        if not re.search(r'/(?:photo|video)/' + re.escape(identifier) + r'(?:[/?#]|$)', url):
            raise ValueError('No verified canonical TikTok URL for ' + identifier)
        urls.append(url)
    return urls


class FixtureReplay:
    """Replay only discovery/transfer; matching and source OCR are never mocked."""
    def __init__(self, dataset, root):
        self.cases = {(case['post_id'], int(case['position'])): case
                      for case in dataset['slides'].values()}
        self.root = Path(root)
        self.files = {}
        self.calls = []
        self.transfers = []

    def candidate(self, identifier, position, name, path, evidence):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            raise ValueError('Replay fixture missing or outside media root: ' + str(path))
        url = f'https://fixtures.invalid/e2e/{identifier}/{position}/{name}.jpg'
        self.files[url] = {'path': path, 'sha256': digest(path)}
        return {'image_url': url, 'source_url': evidence.get('image_url', url),
                'fixture_provenance': evidence.get('kind', 'captioned_negative'),
                'fixture_sha256': self.files[url]['sha256']}

    def search(self, base, slides):
        answers = {}
        for slide in slides:
            case = self.cases.get((base.name, int(slide['position'])))
            query = slide['reverse_query']
            candidates = []
            if case and case.get('source_expected') is True:
                evidence = case['source_evidence']
                if query['kind'] == 'clean_full':
                    candidates = [self.candidate(base.name, slide['position'], 'captioned_negative',
                                                 self.root / case['image'], {})]
                elif query['kind'] == 'original_full':
                    candidates = [self.candidate(base.name, slide['position'], 'verified_source',
                                                 self.root / evidence['file'], evidence)]
            answer = {'provider': 'offline_fixture_replay', 'status': 'CANDIDATES' if candidates else 'NOT_FOUND',
                      'candidates': candidates, 'query': query,
                      'diagnostics': {'live_lens': False, 'discovery_and_transfer_replayed': True}}
            answers[slide['position']] = answer
            self.calls.append({'post_id': base.name, 'position': slide['position'], 'query': query,
                               'candidate_count': len(candidates)})
        return answers

    def download(self, candidate, target, deadline=None):
        fixture = self.files.get(candidate.get('image_url'))
        try:
            if fixture is None:
                raise ValueError('Unregistered replay candidate.')
            if digest(fixture['path']) != fixture['sha256']:
                raise ValueError('Replay fixture changed during validation.')
            size = fixture['path'].stat().st_size
            if size > 15_000_000:
                raise ValueError('Replay fixture exceeds the production 15 MB limit.')
            shutil.copyfile(fixture['path'], target)
            self.transfers.append({'image_url': candidate['image_url'], 'bytes': size,
                                   'sha256': digest(target), 'transport': 'local_fixture_copy'})
            return {'path': target, 'candidate': candidate, 'bytes': size}
        except Exception as exc:
            target.unlink(missing_ok=True)
            return {'candidate': candidate, 'error': str(exc)}


@contextmanager
def bounded_gallery_download(seconds):
    """Keep the production gallery-dl command, with this validation's budget."""
    original = subprocess.run

    def run(command, *args, **kwargs):
        if command and command[0] == 'gallery-dl':
            kwargs['timeout'] = min(kwargs.get('timeout', seconds), seconds)
        return original(command, *args, **kwargs)

    with patch('subprocess.run', run):
        yield


def score_manifest(manifest, dataset, base, revalidation=False, prior_manifest=None):
    from src.image.cleaning import read_image
    expected = {int(case['position']): (identifier, case) for identifier, case in dataset['slides'].items()
                if case['post_id'] == manifest['post_id']}
    if revalidation and prior_manifest is None:
        raise ValueError('Prior manifest is required to attribute historical discovery evidence.')
    prior_slides = {slide['position']: slide for slide in (prior_manifest or {}).get('slides', [])}
    rows = []
    for slide in manifest.get('slides', []):
        if slide['position'] not in expected:
            rows.append({'position': slide['position'], 'ground_truth': 'NOT_AVAILABLE'})
            continue
        identifier, case = expected[slide['position']]
        data = slide['ocr']
        recovery = slide.get('reverse_search', {})
        no_overlay = bool(case.get('expected_editorial_empty'))
        reconstruction = ({'status': 'NO_EDITORIAL_OVERLAY_NOT_SCORED'} if no_overlay else
                          min((text_score(variant, data.get('reconstructed', '')) for variant in
                               [case['expected_text'], *case.get('reconstruction_variants', [])]),
                              key=lambda score: score['cer']))
        translation = ({'status': 'NO_EDITORIAL_OVERLAY_NOT_SCORED'} if no_overlay else
                       translation_score(case, slide.get('translation', {}).get('spanish', '')))
        rows.append({'case': identifier, 'position': slide['position'],
                     'cohort': 'HOLDOUT' if 'fresh_carousel' in case.get('tags', []) else 'REGRESSION',
                     'expected_editorial_empty': bool(case.get('expected_editorial_empty')),
                     'observed_raw_lines': len(data.get('lines', [])),
                     'observed_editorial_lines': len(data.get('editorial_lines', [])),
                     'observed_background_lines': len(data.get('background_lines', [])),
                     'downloaded_original_matches_fixture': slide['original_sha256'] == case['image_sha256'],
                     'ocr': ocr_score(case, {'detected_polygons': data['polygons'],
                                            'recognized_lines': data['lines']}, read_image(base / slide['original']).shape),
                     'reconstruction': reconstruction, 'translation_clause_proxy': translation,
                     'translation_fallback': slide.get('translation', {}).get('fallback_to_english', False),
                     'uncertain_fragments': data.get('uncertain_fragments', []),
                     'warnings': slide.get('warnings', []),
                     'source_expected': case.get('source_expected'), 'source_status': recovery.get('status'),
                     'source_cache_revalidated': recovery.get('cache_revalidated', False),
                     'source_cache_hit': recovery.get('cache_hit', False),
                     'queries_attempted': recovery.get('queries_attempted', 0),
                     'source_sha256': recovery.get('sha256'), 'source_validation': recovery.get('validation', {}),
                     'negative_candidate_rejected': any(not c.get('accepted') for attempt in recovery.get('attempts', [])
                                                       if attempt['query']['kind'] == 'clean_full'
                                                       for c in attempt.get('candidates', []))})
        if revalidation:
            row = rows[-1]
            historical = prior_slides.get(slide['position'], {}).get('reverse_search', {})
            row['prior_negative_candidate_rejected'] = any(
                not candidate.get('accepted') for attempt in historical.get('attempts', [])
                if attempt['query']['kind'] == 'clean_full' for candidate in attempt.get('candidates', []))
            row['negative_candidate_rejected'] = None
            row['prior_queries_attempted'] = historical.get('queries_attempted', 0)
            row['queries_attempted'] = 0
            row['backend_local_attempt_records'] = len(recovery.get('attempts', []))
            row['discovery_evidence_scope'] = 'PRIOR_RUN_ONLY_NO_DISCOVERY_THIS_RUN'
    missing = sorted(set(expected) - {slide['position'] for slide in manifest.get('slides', [])})
    return {'expected_slides': len(expected), 'actual_slides': len(manifest.get('slides', [])),
            'missing_positions': missing, 'cases': rows,
            'semantic_human_review': 'PENDING'}


def source_pointers(manifest):
    return {str(slide['position']): {'best_image': slide.get('best_image'),
            'downloaded': slide.get('reverse_search', {}).get('downloaded'),
            'sha256': slide.get('reverse_search', {}).get('sha256')}
            for slide in manifest.get('slides', [])}


def raw_ocr_evidence(manifest):
    """Ignore rebuilt masks/editorial classification while preserving raw OCR."""
    return {str(slide['position']): {key: slide.get('ocr', {}).get(key)
                                   for key in ('lines', 'polygons', 'primary_lines')}
            for slide in manifest.get('slides', [])}


@contextmanager
def count_pipeline_work(pipeline):
    """Count calls while preserving production methods and their return values."""
    import src.recovery as recovery
    counts = {'download': 0, 'ocr': 0, 'source_ocr': 0, 'translation': 0, 'ollama_chat': 0}

    def counted(name, function):
        def run(*args, **kwargs):
            counts[name] += 1
            return function(*args, **kwargs)
        return run

    with ExitStack() as stack:
        for owner, name, key in [(pipeline, 'download_images', 'download'), (pipeline, 'ocr_image', 'ocr'),
                                 (recovery, 'ocr_image', 'source_ocr'),
                                 (pipeline.OllamaTranslator, 'translate', 'translation'),
                                 (pipeline.OllamaTranslator, '_chat', 'ollama_chat')]:
            stack.enter_context(patch.object(owner, name, counted(key, getattr(owner, name))))
        yield counts


@contextmanager
def no_network_downloads(pipeline):
    """Fail loudly if a saved-evidence revalidation attempts any new transfer."""
    import src.recovery as recovery
    attempts = []

    def blocked(kind):
        def run(*args, **kwargs):
            attempts.append(kind)
            raise RuntimeError('Unexpected network operation during saved-evidence revalidation: ' + kind)
        return run

    with ExitStack() as stack:
        for owner, name, kind in [(pipeline, 'download_images', 'gallery_download'),
                                 (pipeline, 'resolve_tiktok_url', 'url_resolution'),
                                 (recovery, 'download_candidate', 'candidate_download')]:
            stack.enter_context(patch.object(owner, name, blocked(kind)))
        yield attempts


def cached_repeat(pipeline, url, settings, progress, replay, base, manifest, reverse_callback):
    """Use run_pipeline again; snapshots and counters supplement its own timing."""
    before = existing_snapshot(settings.output_base, [base.name])
    before_queries, before_transfers = len(replay.calls), len(replay.transfers)
    started = time.perf_counter()
    with count_pipeline_work(pipeline) as counts:
        result = pipeline.run_pipeline(url, force=False, settings=settings, progress=progress, reverse_callback=reverse_callback)
    pipeline_wall_seconds = round(time.perf_counter() - started, 6)
    after = existing_snapshot(settings.output_base, [base.name])
    new_manifest = json.loads((base / 'metadata/manifest.json').read_text(encoding='utf-8'))
    record = {'seconds': round(time.perf_counter() - started, 6), 'result_status': result.get('status'),
              'pipeline_wall_seconds': pipeline_wall_seconds,
              'pipeline_reported_seconds': result.get('timings', {}).get('total'),
              'seconds_scope': 'Includes post-run artifact hashing and manifest reading; pipeline timings are separate.',
              'cached': result.get('cached'), 'additional_stage_calls': counts,
              'additional_replay_queries': len(replay.calls) - before_queries,
              'additional_fixture_transfers': len(replay.transfers) - before_transfers,
              'managed_artifacts_unchanged': before == after,
              'source_pointers_unchanged': source_pointers(manifest) == source_pointers(new_manifest),
              'source_pointers': source_pointers(new_manifest)}
    record['passed'] = (record['cached'] is True and record['result_status'] == 'OK_CACHED'
                        and not any(counts.values()) and record['additional_replay_queries'] == 0
                        and record['additional_fixture_transfers'] == 0 and record['managed_artifacts_unchanged']
                        and record['source_pointers_unchanged'])
    return record


def worker(args, dataset, root):
    from src.config import Settings
    import src.pipeline as pipeline
    from src.tiktok.downloader import download_images, extract_post_id, resolve_tiktok_url, validate_tiktok_url
    from src.ocr.paddle import OCR_ENGINE_VERSION, OCR_REFINEMENT_VERSION
    urls = args.url or known_urls(dataset, root)
    sandbox = safe_sandbox(root, args.pipeline_output or root / '.cache/regression/e2e' / uuid.uuid4().hex / 'posts')
    if not args.download_only:
        missing = fixture_failures(dataset, root)
        for identifier, case in dataset['slides'].items():
            if case.get('source_expected') is True and not (root / case['source_evidence']['file']).is_file():
                missing[identifier] = 'MISSING_SOURCE_FIXTURE'
        if missing:
            save(args.output, {'status': 'FAIL', 'error': 'MISSING_FIXTURES', 'missing_fixtures': missing})
            return 1
    protected = {case['post_id'] for case in dataset['slides'].values()}
    snapshot = existing_snapshot(root, protected)
    cloned = clone_existing_outputs(root, args.revalidate_existing, sandbox, protected) if args.revalidate_existing else None
    settings = Settings(output_base=sandbox, ollama_model=args.model, ollama_translation_model=args.translation_model,
                        cpu_threads=args.cpu_threads)
    report = {'status': 'RUNNING',
              'scope': ('REAL_PIPELINE_REVALIDATION_OF_CLONED_PRIOR_EVIDENCE' if cloned else
                        'REAL_PHOTO_DOWNLOAD_ONLY' if args.download_only else
                        'REAL_PIPELINE_WITH_OFFLINE_DISCOVERY_AND_CANDIDATE_TRANSFER_REPLAY'),
              'live_lens': False, 'gallery_dl_real_network': not bool(cloned), 'download_timeout_seconds': args.download_timeout_seconds,
              'pipeline_output': str(sandbox), 'dataset_sha256': digest(args.dataset), 'posts': [],
              'holdout_sha256': digest(args.holdout) if args.holdout else None,
              'expected_cohorts': {'regression': sum('fresh_carousel' not in case.get('tags', []) for case in dataset['slides'].values()),
                                   'holdout': sum('fresh_carousel' in case.get('tags', []) for case in dataset['slides'].values()),
                                   'editorial_text': sum(not case.get('expected_editorial_empty') for case in dataset['slides'].values()),
                                   'no_editorial_overlay': sum(bool(case.get('expected_editorial_empty')) for case in dataset['slides'].values())},
              'runtime': {'python': sys.version.split()[0], 'opencv': __import__('cv2').__version__,
                          'gallery_dl': importlib.metadata.version('gallery-dl'),
                          'ocr_engine': OCR_ENGINE_VERSION, 'ocr_refinement': OCR_REFINEMENT_VERSION,
                          'model': settings.ollama_model, 'translation_model': settings.translation_model,
                          'pipeline_signature': pipeline.pipeline_signature(settings)},
              'effective_settings': {'cpu_threads': settings.cpu_threads,
                                     'ocr_adaptive_fallback': settings.ocr_adaptive_fallback,
                                     'ocr_max_crop_retries': settings.ocr_max_crop_retries,
                                     'reverse_max_queries': settings.reverse_max_queries,
                                     'reverse_total_budget_seconds': settings.reverse_total_budget,
                                     'reverse_negative_ttl_seconds': settings.reverse_negative_ttl,
                                     'translation_timeout_seconds': settings.translation_timeout_seconds,
                                     'translation_budget_seconds': settings.translation_budget_seconds,
                                     'translation_retry_ttl_seconds': settings.translation_retry_ttl,
                                     'match_threshold': settings.match_threshold,
                                     'candidate_min_dimension': settings.candidate_min_dimension,
                                     'source_ocr_policy': 'PRIMARY_ONLY_NO_ADAPTIVE_CROPS'},
              'limitations': ['Live Lens discovery and public candidate HTTP downloads are not measured.',
                              'Normalized OCR/reconstruction CER ignores punctuation, spaces and case.',
                              'Translation clause checks are a proxy; semantic review is pending.',
                              'Unknown sources are unlabelled; NOT_FOUND is not proof of nonexistence.']}
    replay = FixtureReplay(dataset, root)
    network_attempts = []
    if cloned:
        report['prior_evidence'] = {key: value for key, value in cloned.items() if key != 'prior_hashes'}
        report['limitations'].append('Revalidation reuses prior real downloads; no fresh TikTok or discovery request is attempted.')
    save(args.output, report)
    started = time.perf_counter()

    def progress(stage, message, current=0, total=0):
        print(json.dumps({'stage': stage, 'message': message, 'current': current, 'total': total}, ensure_ascii=False), flush=True)

    try:
        for url in urls:
            url = validate_tiktok_url(url)
            print('E2E_START ' + url, flush=True)
            tick = time.perf_counter()
            if args.download_only:
                resolved = resolve_tiktok_url(url)
                identifier = extract_post_id(resolved)
                if not identifier:
                    raise ValueError('Resolved URL has no post identifier.')
                target = sandbox / identifier / 'original'
                try:
                    with bounded_gallery_download(args.download_timeout_seconds):
                        images, detail = download_images(resolved, target, settings.cookies_file)
                    entry = {'status': 'DOWNLOADED' if images else 'SKIPPED_NO_IMAGES', 'post_id': identifier,
                             'source_url': url, 'resolved_url': resolved, 'images_found': len(images),
                             'output_dir': str(target.parent), 'downloader_info': detail,
                             'images': [{'file': str(p), 'sha256': digest(p)} for p in images],
                             'ground_truth': 'NOT_REVIEWED'}
                except Exception as exc:
                    entry = {'status': 'ERROR', 'source_url': url, 'post_id': identifier, 'error': str(exc)}
            else:
                reverse_callback = None if cloned else replay.search
                with ExitStack() as stack:
                    if cloned:
                        attempts = stack.enter_context(no_network_downloads(pipeline))
                        calls = stack.enter_context(count_pipeline_work(pipeline))
                    else:
                        stack.enter_context(bounded_gallery_download(args.download_timeout_seconds))
                        stack.enter_context(patch('src.recovery.download_candidate', replay.download))
                    entry = pipeline.run_pipeline(url, force=not bool(cloned), settings=settings,
                                                  progress=progress, reverse_callback=reverse_callback)
                    if cloned:
                        network_attempts.extend(attempts)
                        entry['revalidation_stage_calls'] = calls
                        identifier = entry.get('post_id') or extract_post_id(url)
                        prior_base = Path(cloned['source']) / identifier
                        prior_manifest = json.loads((prior_base / 'metadata/manifest.json').read_text(encoding='utf-8'))
                        entry['prior_source_pointers'] = source_pointers(prior_manifest)
                manifest_path = Path(entry.get('manifest', ''))
                if entry.get('status') in {'OK', 'OK_CACHED'} and manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                    entry['evaluation'] = score_manifest(manifest, dataset, manifest_path.parents[1], revalidation=bool(cloned),
                                                         prior_manifest=prior_manifest if cloned else None)
                    entry['text_only_ollama'] = manifest.get('ollama_receives_images') is False
                    entry['no_image_text_rendering'] = manifest.get('image_text_rendering') is False
                    entry['ollama_metrics'] = manifest.get('ollama_metrics', {})
                    if cloned:
                        entry['prior_source_pointers_preserved'] = entry['prior_source_pointers'] == source_pointers(manifest)
                        entry['prior_raw_ocr_unchanged'] = raw_ocr_evidence(prior_manifest) == raw_ocr_evidence(manifest)
                        prefix = identifier + '/original/'
                        original_hashes = {key: value for key, value in cloned['prior_hashes'].items() if key.startswith(prefix)}
                        current_hashes = {key: value for key, value in existing_snapshot(sandbox, [identifier]).items()
                                          if key.startswith(prefix)}
                        entry['prior_original_files_unchanged'] = original_hashes == current_hashes
                        entry['prior_original_sha256'] = original_hashes
                    with ExitStack() as stack:
                        if cloned:
                            attempts = stack.enter_context(no_network_downloads(pipeline))
                        else:
                            stack.enter_context(bounded_gallery_download(args.download_timeout_seconds))
                            stack.enter_context(patch('src.recovery.download_candidate', replay.download))
                        entry['cached_repeat'] = cached_repeat(pipeline, url, settings, progress, replay,
                                                               manifest_path.parents[1], manifest, reverse_callback)
                        if cloned:
                            network_attempts.extend(attempts)
            entry['validation_seconds'] = round(time.perf_counter() - tick, 3)
            entry['worker_memory_after_post'] = worker_memory()
            report['posts'].append(entry)
            report['replay_queries'] = replay.calls
            report['replayed_transfers'] = replay.transfers
            save(args.output, report)
            print('E2E_RESULT ' + json.dumps({key: entry.get(key) for key in ('post_id', 'status', 'images_found', 'recovered_images', 'error')}), flush=True)
    except Exception as exc:
        report.update(error='VALIDATION_TOOL_ERROR', detail=str(exc))
    finally:
        after = existing_snapshot(root, protected)
        report['existing_artifacts_unchanged'] = snapshot == after
        report['fixture_failures_after_run'] = {} if args.download_only else fixture_failures(dataset, root)
        report['worker_memory'] = worker_memory()
        report['protected_file_count'] = len(snapshot)
        report['changed_existing_artifacts'] = sorted(key for key in snapshot.keys() | after.keys() if snapshot.get(key) != after.get(key))
        report['seconds'] = round(time.perf_counter() - started, 3)
        processed_ids = {post.get('post_id') for post in report['posts']}
        scored_cases = {row.get('case') for post in report['posts']
                        for row in post.get('evaluation', {}).get('cases', []) if row.get('case')}
        report['dataset_coverage'] = {'expected_post_ids': sorted(protected),
                                      'missing_post_ids': sorted(protected - processed_ids),
                                      'expected_slide_count': len(dataset['slides']),
                                      'scored_slide_count': len(scored_cases),
                                      'missing_cases': sorted(set(dataset['slides']) - scored_cases)}
        report['complete_dataset_coverage'] = (not report['dataset_coverage']['missing_post_ids']
                                              and not report['dataset_coverage']['missing_cases']
                                              and all(not post.get('evaluation', {}).get('missing_positions')
                                                      for post in report['posts']))
        if cloned:
            report['prior_evidence_unchanged'] = cloned['prior_hashes'] == existing_snapshot(Path(cloned['source']), protected)
            report['unexpected_network_attempts'] = network_attempts
        good_status = {'DOWNLOADED'} if args.download_only else {'OK', 'OK_CACHED'}
        report['status'] = ('MEASURED' if report['posts'] and all(post['status'] in good_status for post in report['posts'])
                            and (args.download_only or all(post.get('cached_repeat', {}).get('passed') for post in report['posts']))
                            and report['existing_artifacts_unchanged'] and not report['fixture_failures_after_run']
                            and (not cloned or (report['prior_evidence_unchanged'] and not network_attempts
                                                and report['complete_dataset_coverage']
                                                and all(post.get('prior_source_pointers_preserved')
                                                        and post.get('prior_raw_ocr_unchanged')
                                                        and post.get('prior_original_files_unchanged')
                                                        and post.get('revalidation_stage_calls', {}).get('ocr') == 0
                                                        and post.get('revalidation_stage_calls', {}).get('download') == 0
                                                        for post in report['posts'])))
                            and not report.get('error') else 'FAIL')
        save(args.output, report)
    return 0 if report['status'] == 'MEASURED' else 1


def docker_host(args):
    remote = '/tmp/tiktok-e2e-' + uuid.uuid4().hex
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix('.docker.log')

    def command(parts):
        result = subprocess.run(parts, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])

    try:
        command(['docker', 'exec', args.container, 'mkdir', '-p', remote + '/tools'])
        for source, target in [(Path(__file__), remote + '/tools/validate_pipeline_e2e.py'),
                               (PROJECT / 'tools/regression_metrics.py', remote + '/tools/regression_metrics.py'),
                               (args.dataset, remote + '/expected.json')]:
            command(['docker', 'cp', str(source), args.container + ':' + target])
        if args.holdout:
            command(['docker', 'cp', str(args.holdout), args.container + ':' + remote + '/holdout.json'])
        arguments = ['docker', 'exec', '-e', 'PYTHONPATH=/app:' + remote + '/tools', args.container, 'python', '-u',
                     remote + '/tools/validate_pipeline_e2e.py', '--dataset', remote + '/expected.json',
                     '--media-root', args.container_media_root, '--output', remote + '/report.json',
                     '--model', args.model, '--translation-model', args.translation_model,
                     '--cpu-threads', str(args.cpu_threads), '--download-timeout-seconds', str(args.download_timeout_seconds)]
        if args.download_only:
            arguments.append('--download-only')
        if args.holdout:
            arguments.extend(['--holdout', remote + '/holdout.json'])
        if args.revalidate_existing:
            arguments.extend(['--revalidate-existing', args.revalidate_existing])
        for url in args.url or []:
            arguments.extend(['--url', url])
        with log.open('w', encoding='utf-8') as handle:
            completed = subprocess.run(arguments, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout_seconds)
        command(['docker', 'cp', args.container + ':' + remote + '/report.json', str(args.output)])
        report = json.loads(args.output.read_text(encoding='utf-8'))
        report['transport'] = {'container': args.container, 'workdir': remote, 'log': str(log)}
        save(args.output, report)
        print(json.dumps({'status': report['status'], 'seconds': report['seconds'],
                          'existing_artifacts_unchanged': report['existing_artifacts_unchanged'],
                          'posts': [{k: post.get(k) for k in ('post_id', 'status', 'images_found', 'recovered_images', 'error')}
                                    for post in report['posts']]}, ensure_ascii=False), flush=True)
        return completed.returncode
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        save(args.output, {'status': 'FAIL', 'error': 'DOCKER_TRANSPORT_ERROR', 'detail': str(exc), 'log': str(log),
                           'container_workdir': remote, 'warning': 'Inspect container worker after a host transport timeout.'})
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=PROJECT / 'tests/regression/expected.json')
    parser.add_argument('--holdout', type=Path, help='Optional independent holdout JSON; cohorts remain distinct in the report.')
    parser.add_argument('--media-root', type=Path)
    parser.add_argument('--output', type=Path, default=PROJECT / '.cache/regression/e2e/report.json')
    parser.add_argument('--pipeline-output', type=Path, help='Linux only: a fresh sandbox under media/.cache/regression/e2e/.')
    parser.add_argument('--revalidate-existing', help='Prior sandbox path relative to media root; clone it, reuse real downloads and disable discovery/transfers.')
    parser.add_argument('--container')
    parser.add_argument('--container-media-root', default='/data/tiktok')
    parser.add_argument('--url', action='append', help='Specific real URL; otherwise use the two labelled manifest URLs.')
    parser.add_argument('--download-only', action='store_true', help='Real photo download probe without OCR/Ollama.')
    parser.add_argument('--model', default=os.getenv('OLLAMA_MODEL', 'gemma3:4b'))
    parser.add_argument('--translation-model', default=os.getenv('OLLAMA_TRANSLATION_MODEL', ''))
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--download-timeout-seconds', type=int, default=90)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if not 1 <= args.cpu_threads <= 32 or min(args.download_timeout_seconds, args.timeout_seconds) <= 0:
        parser.error('CPU threads must be 1..32 and timeouts must be positive.')
    if args.revalidate_existing and args.download_only:
        parser.error('--revalidate-existing cannot be used with --download-only.')
    try:
        dataset = load_inputs(args.dataset, args.holdout)
    except ValueError as exc:
        parser.error(str(exc))
    root = media_root(dataset, args.media_root)
    if args.revalidate_existing:
        previous = Path(args.revalidate_existing)
        previous = (previous if previous.is_absolute() else root / previous).resolve()
        if args.output.resolve().is_relative_to(previous):
            parser.error('Reports cannot overwrite the prior evidence being revalidated.')
    if args.output.resolve().is_relative_to(root):
        relative = args.output.resolve().relative_to(root)
        if relative.parts and relative.parts[0].isdecimal():
            parser.error('Reports cannot overwrite existing post artifacts.')
    if args.container:
        if args.pipeline_output:
            parser.error('--pipeline-output is Linux only; the Docker worker selects a fresh sandbox.')
        return docker_host(args)
    if not sys.platform.startswith('linux'):
        parser.error('Use --container tiktok-translator on Windows; production Paddle and RSS checks require Linux.')
    try:
        return worker(args, dataset, root)
    except Exception as exc:
        save(args.output, {'status': 'FAIL', 'error': 'PREFLIGHT_ERROR', 'detail': str(exc)})
        return 1


if __name__ == '__main__':
    if os.name == 'nt' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(main())
