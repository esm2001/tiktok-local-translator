"""Read-only proof of the installed CPU recipe and the unique cv2 namespace.

pip check alone cannot detect two OpenCV wheels writing the same cv2 files.
Run after a clean backend build: --container tiktok-translator.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
OPENCV_WHEELS = {'opencv-python', 'opencv-python-headless', 'opencv-contrib-python', 'opencv-contrib-python-headless'}


def assess(snapshot):
    errors = []
    installed = snapshot['opencv_distributions']
    if installed != {'opencv-contrib-python': '4.10.0.84'}:
        errors.append('EXPECTED_EXACTLY_ONE_OPENCV_CONTRIB_4_10_WHEEL')
    if snapshot.get('opencv_imported_version') != '4.10.0':
        errors.append('UNEXPECTED_IMPORTED_CV2_VERSION')
    for package, expected in snapshot['expected_versions'].items():
        if snapshot['installed_versions'].get(package) != expected:
            errors.append('PACKAGE_VERSION_MISMATCH:' + package)
    if snapshot['pip_check']['exit_code']:
        errors.append('PIP_CHECK_FAILED')
    if snapshot.get('paddlepaddle_gpu_version'):
        errors.append('EXPECTED_CPU_PADDLE_PACKAGE')
    return errors


def collect(requirements):
    expected = {'paddlepaddle': '3.2.0'}
    for line in requirements.read_text(encoding='utf-8-sig').splitlines():
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s;]+)', line.strip())
        if match:
            expected[match.group(1).lower()] = match.group(2)

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    wheels = {distribution.metadata['Name'].lower(): distribution.version
              for distribution in importlib.metadata.distributions()
              if distribution.metadata['Name'].lower() in OPENCV_WHEELS}
    checked = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, timeout=30)
    try:
        import cv2
        imported = cv2.__version__
    except ImportError:
        imported = None
    snapshot = {'python': sys.version.split()[0], 'requirements': str(requirements),
                'expected_versions': expected, 'installed_versions': {name: version(name) for name in expected},
                'opencv_distributions': wheels, 'opencv_imported_version': imported,
                'paddlepaddle_gpu_version': version('paddlepaddle-gpu'),
                'paddlex_opencv_dependency': [value for value in importlib.metadata.requires('paddlex') or []
                                             if 'opencv' in value.lower()],
                'pip_check': {'exit_code': checked.returncode, 'output': (checked.stdout + checked.stderr).strip()}}
    snapshot['errors'] = assess(snapshot)
    snapshot['status'] = 'PASS' if not snapshot['errors'] else 'FAIL'
    return snapshot


def save(output, snapshot):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container')
    parser.add_argument('--requirements', type=Path, default=PROJECT / 'translator/requirements.txt')
    parser.add_argument('--output', type=Path, default=PROJECT / '.cache/regression/runtime-dependencies.json')
    args = parser.parse_args(argv)
    try:
        if args.container:
            remote = '/tmp/runtime-dependencies-' + uuid.uuid4().hex

            def command(parts):
                result = subprocess.run(parts, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
                if result.returncode:
                    raise RuntimeError(result.stderr[-1500:] or result.stdout[-1500:])
                return result

            command(['docker', 'cp', str(Path(__file__)), args.container + ':' + remote + '.py'])
            # A failed proof still writes its report; preserve that concrete failure.
            executed = subprocess.run(['docker', 'exec', args.container, 'python', remote + '.py',
                                      '--requirements', '/app/requirements.txt', '--output', remote + '.json'],
                                     capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            command(['docker', 'cp', args.container + ':' + remote + '.json', str(args.output)])
            snapshot = json.loads(args.output.read_text(encoding='utf-8'))
            snapshot['container'] = args.container
            snapshot['worker_exit_code'] = executed.returncode
        else:
            snapshot = collect(args.requirements)
        save(args.output, snapshot)
        print(json.dumps({key: snapshot.get(key) for key in ('status', 'errors', 'opencv_distributions', 'opencv_imported_version')}, ensure_ascii=False))
        return 0 if snapshot['status'] == 'PASS' else 1
    except Exception as exc:
        save(args.output, {'status': 'FAIL', 'error': 'RUNTIME_PROOF_FAILED', 'detail': str(exc)})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
