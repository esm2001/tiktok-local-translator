"""Measure real labelled source pairs and hard negatives without using Lens.

This is candidate validation accuracy, not internet source discovery/recovery.
Downloaded copyright material stays in the local media directory and temporary
variants are removed. Run with the same OpenCV/Paddle runtime as production.
"""
import argparse
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'translator'))
from regression_metrics import fixture_failures, load_dataset, media_root


def preflight_cases(dataset, media):
    """Authenticate originals and identify inputs before importing the runtime.

    Source evidence has no expected hash in this dataset. Its presence is
    checked here; measured hashes remain part of each calibration row.
    """
    failures = {key: [reason] for key, reason in fixture_failures(dataset, media).items()}
    prepared = []
    for key, expected in dataset['slides'].items():
        if expected.get('source_expected') is not True:
            continue
        problems = failures.setdefault(key, [])
        post = media / expected['post_id']
        source_file = (expected.get('source_evidence') or {}).get('file')
        source = media / source_file if isinstance(source_file, str) and source_file else None
        if source is None or not source.is_file():
            problems.append('MISSING_SOURCE_FIXTURE')
        manifest = post / 'metadata/manifest.json'
        if not manifest.is_file():
            problems.append('MISSING_RECOVERY_MANIFEST')
            continue
        try:
            slides = json.loads(manifest.read_text(encoding='utf-8'))['slides']
            matching = [slide for slide in slides if slide['position'] == expected['position']]
            if len(matching) != 1:
                problems.append('MANIFEST_SLIDE_NOT_UNIQUE')
                continue
            slide = matching[0]
            original = post / slide['original']
            if original.resolve() != (media / expected['image']).resolve():
                problems.append('MANIFEST_ORIGINAL_MISMATCH')
            mask, clean = post / slide['ocr']['mask'], post / slide['clean']
        except (OSError, ValueError, KeyError, TypeError):
            problems.append('INVALID_RECOVERY_MANIFEST')
            continue
        prepared.append((key, expected, post, original, mask, clean, source))
    failures = {key: reasons for key, reasons in failures.items() if reasons}
    if not any(case.get('source_expected') is True for case in dataset['slides'].values()):
        failures['_dataset'] = ['NO_LABELLED_SOURCES']
    return prepared, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=PROJECT/'tests/regression/expected.json')
    parser.add_argument('--media-root', type=Path)
    parser.add_argument('--output', type=Path, default=PROJECT/'.cache/recovery-diagnosis/calibration.json')
    parser.add_argument('--with-source-ocr', action='store_true')
    parser.add_argument('--primary-source-ocr', action='store_true',
                        help='Explicitly select the production source OCR without grayscale fallback.')
    parser.add_argument('--adaptive-source-ocr', action='store_true',
                        help='Measure the previous source OCR with grayscale fallback enabled.')
    parser.add_argument('--compare-source-ocr-modes', action='store_true')
    parser.add_argument('--compare-source-crop-budgets', action='store_true')
    parser.add_argument('--source-crop-retries', type=int, default=0)
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset)
    media = media_root(dataset, args.media_root)
    prepared, failures = preflight_cases(dataset, media)
    if failures:
        report = {'status': 'FAIL', 'scope': 'offline_candidate_validation_not_lens_discovery',
                  'error': 'FIXTURE_PREFLIGHT_FAILED', 'fixture_failures': failures,
                  'expected_original_cases': len(dataset['slides']),
                  'source_integrity_basis': 'Source file presence checked; no expected SOURCE hash is validated.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 1
    import cv2
    import numpy as np
    from src.config import Settings
    from src.image.cleaning import read_image, write_image
    from src.image.similarity import compare_candidate
    from src.recovery import RECOVERY_VERSION, candidate_overlay_check
    from src.ocr import paddle

    settings = Settings()
    settings = replace(settings, ocr_adaptive_fallback=args.adaptive_source_ocr and not args.primary_source_ocr,
                       ocr_max_crop_retries=6 if args.compare_source_crop_budgets else args.source_crop_retries)
    rows = []
    source_mode_comparisons = []
    source_crop_comparisons = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='recovery-regression-') as temporary:
        folder = Path(temporary)
        for key, expected, post, original, mask, clean, source in prepared:
            inputs = {'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                      'mask_sha256': hashlib.sha256(mask.read_bytes()).hexdigest(),
                      'clean_sha256': hashlib.sha256(clean.read_bytes()).hexdigest()}
            cases = [('verified_source', source, True), ('original_with_text', original, False), ('clean_copy', clean, False)]
            other = next(p for p in sorted((post/'original').glob('*.jpg')) if p != original)
            cases.append(('different_photograph', other, False))
            image = read_image(source)
            for variant, pixels, quality in [
                ('jpeg_recompressed', image, 65),
                ('minor_colour_shift', np.clip(image.astype(np.float32)*.97+4, 0, 255).astype(np.uint8), 90),
            ]:
                target = folder/f'{key}-{variant}.jpg'
                write_image(target, pixels, [cv2.IMWRITE_JPEG_QUALITY, quality])
                cases.append((variant, target, True))
            archives = list((media/'.cache/legacy'/expected['post_id']).glob(f"*/translated/{expected['position']:02d}_ES.jpg"))
            if archives:
                cases.append(('archived_translated_overlay', archives[0], False))
            for name, candidate, label in cases:
                tick = time.perf_counter()
                result = compare_candidate(original, candidate, mask, settings.candidate_min_dimension,
                                           settings.match_threshold, clean=clean)
                visual_accepted = result['accepted']
                check = None
                if args.with_source_ocr and result['accepted']:
                    ocr_started = time.perf_counter()
                    check = candidate_overlay_check(original, candidate, mask, result, settings,
                                                    adaptive_fallback=settings.ocr_adaptive_fallback,
                                                    crop_retries=settings.ocr_max_crop_retries)
                    check['seconds'] = round(time.perf_counter()-ocr_started, 4)
                    result['metrics']['source_ocr'] = check
                    if not check['accepted']:
                        result.update(accepted=False, reason='OVERLAY_TEXT_DETECTED')
                pair_seconds = time.perf_counter()-tick
                if args.compare_source_crop_budgets and check is not None:
                    crop_started = time.perf_counter()
                    no_crop = candidate_overlay_check(original, candidate, mask, result,
                                                      replace(settings, ocr_max_crop_retries=0))
                    no_crop['seconds'] = round(time.perf_counter()-crop_started, 4)
                    source_crop_comparisons.append({'slide': key, 'case': name, 'with_crops': check,
                        'without_crops': no_crop, 'same_gate_decision': check['accepted'] == no_crop['accepted'],
                        'same_detected_regions': check['detected_regions'] == no_crop['detected_regions']})
                if args.compare_source_ocr_modes and check is not None:
                    mode_started = time.perf_counter()
                    other_check = candidate_overlay_check(original, candidate, mask, result, settings,
                                                          adaptive_fallback=not settings.ocr_adaptive_fallback,
                                                          crop_retries=settings.ocr_max_crop_retries)
                    other_check['seconds'] = round(time.perf_counter()-mode_started, 4)
                    adaptive_check = check if settings.ocr_adaptive_fallback else other_check
                    primary_check = other_check if settings.ocr_adaptive_fallback else check
                    source_mode_comparisons.append({'slide': key, 'case': name,
                        'adaptive': adaptive_check, 'primary': primary_check,
                        'primary_seconds': primary_check['seconds'], 'extra_seconds': other_check['seconds'],
                        'same_gate_decision': adaptive_check['accepted'] == primary_check['accepted']})
                rows.append({'slide': key, 'case': name, 'label': label, 'visual_accepted': visual_accepted,
                             'seconds': round(pair_seconds, 4), **inputs,
                             'candidate_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(), **result})
    positives = [row for row in rows if row['label']]
    negatives = [row for row in rows if not row['label']]
    durations = [row['seconds'] for row in rows]
    report = {'scope': 'offline_candidate_validation_not_lens_discovery', 'recovery_version': RECOVERY_VERSION,
              'mask_scope': 'existing_manifest_masks_and_clean_files',
              'ocr_engine_version': getattr(paddle, 'OCR_ENGINE_VERSION', 'PP-OCRv5_mobile_legacy'),
              'source_ocr_adaptive_fallback': settings.ocr_adaptive_fallback,
              'source_ocr_crop_retries': settings.ocr_max_crop_retries,
              'opencv_version': cv2.__version__, 'threshold': settings.match_threshold,
              'source_ocr_used': args.with_source_ocr, 'positive_pairs': len(positives), 'negative_pairs': len(negatives),
              'true_positives': sum(row['accepted'] for row in positives),
              'false_negatives': sum(not row['accepted'] for row in positives),
              'false_positives': sum(row['accepted'] for row in negatives),
              'average_seconds': round(float(np.mean(durations)), 4),
              'p95_seconds': round(float(np.percentile(durations, 95)), 4),
              'total_seconds': round(time.perf_counter()-started, 3), 'rows': rows}
    if source_mode_comparisons:
        report['source_mode_comparisons'] = source_mode_comparisons
        report['source_mode_same_decisions'] = all(row['same_gate_decision'] for row in source_mode_comparisons)
        report['source_mode_comparison_extra_seconds'] = round(sum(row['extra_seconds'] for row in source_mode_comparisons), 3)
    if source_crop_comparisons:
        report['source_crop_comparisons'] = source_crop_comparisons
        report['source_crop_same_decisions'] = all(row['same_gate_decision'] for row in source_crop_comparisons)
        report['source_crop_same_detected_regions'] = all(row['same_detected_regions'] for row in source_crop_comparisons)
        report['source_crop_comparison_extra_seconds'] = round(sum(row['without_crops']['seconds'] for row in source_crop_comparisons), 3)
    try:
        import resource
        # Linux ru_maxrss includes native OpenCV/Paddle allocations, unlike tracemalloc.
        report['peak_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2)
    except ImportError:
        report['peak_rss_mib'] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in {'rows', 'source_mode_comparisons', 'source_crop_comparisons'}}, ensure_ascii=False), flush=True)
    return 0 if report['false_positives'] == 0 and report['false_negatives'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
