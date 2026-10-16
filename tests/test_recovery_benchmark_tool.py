"""Reject compromised calibration inputs before importing the image/OCR runtime."""

import builtins
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from benchmark_recovery import main, preflight_cases


@pytest.fixture
def calibration_files(tmp_path):
    post = tmp_path / '1234567890123456789'
    image = post / 'original/01.jpg'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'independently reviewed original photograph')
    source = post / 'recovered/01_SOURCE.jpg'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'visually reviewed source photograph')
    manifest = post / 'metadata/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'slides': [{
        'position': 1, 'original': 'original/01.jpg', 'clean': 'clean/01_CLEAN.jpg',
        'ocr': {'mask': 'metadata/masks/01_MASK.png'},
    }]}), encoding='utf-8')
    dataset = {'media_root': '.', 'slides': {'known_source': {
        'post_id': post.name, 'position': 1, 'image': post.name + '/original/01.jpg',
        'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
        'source_expected': True, 'source_evidence': {'file': post.name + '/recovered/01_SOURCE.jpg'},
    }}}
    dataset_path = tmp_path / 'expected.json'
    dataset_path.write_text(json.dumps(dataset), encoding='utf-8')
    return dataset, dataset_path, image, source, manifest


def forbid_runtime_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {'cv2', 'numpy', 'src.config', 'src.recovery'} or name.startswith(('src.ocr', 'src.image')):
            pytest.fail('Invalid fixtures reached a runtime import: ' + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)


def run_failure(tmp_path, dataset_path, monkeypatch):
    forbid_runtime_imports(monkeypatch)
    output = tmp_path / 'report.json'
    result = main(['--dataset', str(dataset_path), '--media-root', str(tmp_path), '--output', str(output)])
    report = json.loads(output.read_text(encoding='utf-8'))
    assert result == 1
    assert report['status'] == 'FAIL'
    assert report['error'] == 'FIXTURE_PREFLIGHT_FAILED'
    assert report['expected_original_cases'] == 1
    return report


def test_changed_original_is_rejected_before_runtime_imports(tmp_path, calibration_files, monkeypatch):
    dataset, dataset_path, image, source, manifest = calibration_files
    image.write_bytes(b'a different photograph under the same filename')
    report = run_failure(tmp_path, dataset_path, monkeypatch)
    assert report['fixture_failures'] == {'known_source': ['FIXTURE_HASH_MISMATCH']}


def test_missing_source_is_rejected_before_runtime_imports(tmp_path, calibration_files, monkeypatch):
    dataset, dataset_path, image, source, manifest = calibration_files
    source.unlink()
    report = run_failure(tmp_path, dataset_path, monkeypatch)
    assert report['fixture_failures'] == {'known_source': ['MISSING_SOURCE_FIXTURE']}


def test_manifest_cannot_select_an_unchecked_original(tmp_path, calibration_files, monkeypatch):
    dataset, dataset_path, image, source, manifest = calibration_files
    other = image.with_name('02.jpg')
    other.write_bytes(image.read_bytes())
    value = json.loads(manifest.read_text(encoding='utf-8'))
    value['slides'][0]['original'] = 'original/02.jpg'
    manifest.write_text(json.dumps(value), encoding='utf-8')
    report = run_failure(tmp_path, dataset_path, monkeypatch)
    assert report['fixture_failures'] == {'known_source': ['MANIFEST_ORIGINAL_MISMATCH']}


def test_missing_manifest_is_rejected_before_runtime_imports(tmp_path, calibration_files, monkeypatch):
    dataset, dataset_path, image, source, manifest = calibration_files
    manifest.unlink()
    report = run_failure(tmp_path, dataset_path, monkeypatch)
    assert report['fixture_failures'] == {'known_source': ['MISSING_RECOVERY_MANIFEST']}


def test_valid_preflight_preserves_measured_inputs_and_labels(tmp_path, calibration_files):
    dataset, dataset_path, image, source, manifest = calibration_files
    prepared, failures = preflight_cases(dataset, tmp_path)
    assert failures == {}
    key, expected, post, original, mask, clean, actual_source = prepared[0]
    assert key == 'known_source'
    assert expected is dataset['slides'][key]
    assert expected['source_expected'] is True
    assert original == image and actual_source == source
    assert post == image.parent.parent
    assert mask == post / 'metadata/masks/01_MASK.png'
    assert clean == post / 'clean/01_CLEAN.jpg'
