"""Independent, reproducible metrics for manually reviewed local slide fixtures."""
import json
import hashlib
import os
from pathlib import Path
import re
import statistics
import unicodedata

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT / 'tests/regression/expected.json'


def load_dataset(path=DEFAULT_DATASET):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def media_root(dataset, override=None):
    return Path(override or os.environ.get(dataset.get('media_root_env', 'TIKTOK_REGRESSION_MEDIA'))
                or PROJECT / dataset['media_root']).resolve()


def fixture_failures(dataset, root):
    failures={}
    for key,case in dataset['slides'].items():
        path=Path(root)/case['image']
        if not path.is_file():
            failures[key]='MISSING_FIXTURE'
        elif case.get('image_sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=case['image_sha256']:
            failures[key]='FIXTURE_HASH_MISMATCH'
    return failures


def normalise(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', str(text)).casefold() if c.isalnum())


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        row = [i]
        for j, right in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = row
    return previous[-1]


def text_score(expected, actual):
    left, right = normalise(expected), normalise(actual)
    edits = distance(left, right)
    return {'exact_normalised': left == right, 'character_errors': edits,
            'expected_characters': len(left), 'cer': edits / max(1, len(left))}


def ocr_score(case, ocr, shape):
    """Geometry measures detection; transcription is scored separately.

    Manual approximate boxes pass at >=50% detector mask coverage. Recognition
    uses overlapping line centres and spatial order; it never chooses text by
    similarity to the ground truth. Split detector regions may jointly cover a
    single visible line. Repeated words in distinct regions remain distinct.
    """
    import cv2
    import numpy as np
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in ocr.get('detected_polygons', []):
        points = np.asarray(polygon, dtype=np.int32)
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 2:
            cv2.fillPoly(mask, [points], 255)
    regions, texts = [], []
    for region in case.get('expected_regions', []):
        x0, y0, x1, y1 = region['box']
        left, top, right, bottom = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        coverage = float(np.count_nonzero(mask[top:bottom, left:right])) / max(1, (bottom-top)*(right-left))
        evidence = []
        for line in ocr.get('recognized_lines', []):
            points = np.asarray(line.get('polygon', []))
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                continue
            cx, cy = points.mean(axis=0)
            if left <= cx <= right and top <= cy <= bottom:
                evidence.append((float(points[:,0].min()), str(line.get('text',''))))
        actual = ' '.join(t for _,t in sorted(evidence))
        texts.append(actual)
        regions.append({'expected': region['text'], 'actual': actual,
                        'coverage': round(coverage,4), 'detected': coverage >= .5,
                        'recognition': text_score(region['text'], actual)})
    recognition = text_score(case['expected_text'], '\n'.join(texts))
    detection_hits = sum(r['detected'] for r in regions)
    return {'detected_regions': detection_hits, 'expected_regions': len(regions),
            'all_regions_detected': detection_hits == len(regions),
            'visible_text_missing': bool(case['expected_text'].strip()) and not any(texts),
            'recognition': recognition, 'regions': regions}


def translation_score(case, text):
    """Clause checks are a limited objective proxy, not semantic certification."""
    value = unicodedata.normalize('NFKC', str(text)).casefold()
    checks = case.get('translation_checks', [])
    passed = [any(term.casefold() in value for term in alternatives) for alternatives in checks]
    return {'clause_checks_passed': sum(passed), 'clause_checks_total': len(passed),
            'all_clause_checks_passed': bool(passed) and all(passed),
            'missing_checks': [c for c,ok in zip(checks,passed) if not ok],
            'semantic_human_review': 'PENDING'}


def time_summary(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {'n':0, 'average':None, 'p95':None, 'fastest':None, 'slowest':None}
    # Nearest-rank quantile avoids inventing a percentile from interpolation.
    import math
    return {'n':len(values), 'average':round(statistics.mean(values),3),
            'p95':round(values[max(0,math.ceil(.95*len(values))-1)],3),
            'fastest':round(values[0],3),'slowest':round(values[-1],3)}
