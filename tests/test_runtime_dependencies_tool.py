"""Dependency proof detects conflicts that pip check cannot identify."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from verify_runtime_dependencies import assess


def runtime():
    return {'opencv_distributions': {'opencv-contrib-python': '4.10.0.84'},
            'opencv_imported_version': '4.10.0', 'expected_versions': {'paddlepaddle': '3.2.0'},
            'installed_versions': {'paddlepaddle': '3.2.0'}, 'pip_check': {'exit_code': 0}}


def test_two_wheels_fail_even_when_pip_check_and_cv2_version_pass():
    snapshot = runtime()
    snapshot['opencv_distributions']['opencv-python-headless'] = '5.0.0.93'
    assert assess(snapshot) == ['EXPECTED_EXACTLY_ONE_OPENCV_CONTRIB_4_10_WHEEL']


def test_clean_matching_cpu_runtime_passes():
    assert assess(runtime()) == []


def test_imported_module_must_match_distribution_metadata():
    snapshot = runtime()
    snapshot['opencv_imported_version'] = '5.0.0'
    assert assess(snapshot) == ['UNEXPECTED_IMPORTED_CV2_VERSION']


def test_missing_cpu_paddle_is_not_accepted_as_a_verified_recipe():
    snapshot = runtime()
    snapshot['installed_versions']['paddlepaddle'] = None
    assert assess(snapshot) == ['PACKAGE_VERSION_MISMATCH:paddlepaddle']
