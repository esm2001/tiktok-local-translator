"""Benchmark installed text models against visually transcribed local fixtures.

Only text reaches Ollama. Reports go to .cache/regression; user outputs are never
rewritten. Clause checks are a limited objective proxy, not semantic proof.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time

import requests

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'translator'))
from src.config import Settings
from src.image.cleaning import read_image
from src.ocr.editorial import editorial_lines
from src.translation.ollama import OllamaTranslator, _ocr_evidence
from regression_metrics import DEFAULT_DATASET, load_dataset, media_root, text_score, translation_score


class MemorySampler:
    """Sample actual Ollama process RSS when psutil is installed and accessible."""
    def __init__(self):
        self.peak = None
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        try:
            import psutil
        except ImportError:
            return self

        def sample():
            while not self.stop.is_set():
                total = 0
                try:
                    for process in psutil.process_iter(['name', 'memory_info']):
                        if any(name in (process.info['name'] or '').casefold() for name in ('ollama', 'llama-server')):
                            total += process.info['memory_info'].rss
                    self.peak = max(self.peak or 0, total)
                except (psutil.Error, OSError):
                    pass
                self.stop.wait(.5)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)


def write_report(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_inputs(dataset, args):
    cases = dataset['slides']
    if args.limit:
        cases = dict(list(cases.items())[:args.limit])
    if args.case:
        cases = {key: value for key, value in cases.items() if key in args.case}
    if args.gold_input:
        return cases, {key: case['expected_lines'] for key, case in cases.items()}, {}, 'GOLD_TEXT_UPPER_BOUND'
    if args.ocr_report:
        report = json.loads(args.ocr_report.read_text(encoding='utf-8'))
        measured = {entry['case_id']: entry for entry in report['cases']}
        root = media_root(dataset, args.media_root)
        inputs, ignored = {}, {}
        for key, case in cases.items():
            ocr = measured[key]['variants'][args.ocr_variant]['ocr']
            shape = read_image(root / case['image']).shape
            lines, ignored[key] = editorial_lines(ocr['recognized_lines'], shape)
            inputs[key] = lines
        return cases, inputs, ignored, 'MEASURED_OCR_WITH_PRODUCTION_EDITORIAL_FILTER'
    raise ValueError('Specify --ocr-report for measured OCR, or --gold-input for an explicitly labelled upper-bound translation test.')


def translator_class(baseline):
    if baseline is None:
        return OllamaTranslator
    specification = importlib.util.spec_from_file_location('regression_baseline_ollama', baseline)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.OllamaTranslator


def score_cases(cases, inputs, outputs, raw):
    reports = {}
    for key, case in cases.items():
        result = outputs.get(key, {})
        reconstructed = result.get('original', raw['reconstruction'].get(key, ''))
        spanish = result.get('spanish', raw['translation'].get(key, ''))
        reconstruction = text_score(case['expected_text'], reconstructed)
        words = lambda text: re.findall(r'[^\W_]+', str(text).casefold(), flags=re.UNICODE)
        reconstruction['word_sequence_equal'] = words(case['expected_text']) == words(reconstructed)
        alternatives = case.get('reconstruction_variants', [])
        if alternatives:
            reconstruction['matches_recorded_variant'] = any(text_score(value, reconstructed)['exact_normalised'] for value in alternatives)
            reconstruction['cer_to_best_recorded_variant'] = min(
                [reconstruction['cer']] + [text_score(value, reconstructed)['cer'] for value in alternatives])
        reports[key] = {'expected_text': case['expected_text'], 'input_ocr': inputs[key],
                        'actual_reconstruction': reconstructed, 'actual_translation': spanish,
                        'result_returned_by_pipeline': key in outputs,
                        'reconstruction_score': reconstruction,
                        'translation_score': translation_score(case, spanish),
                        'fallback_to_english': result.get('fallback_to_english', False),
                        'uncertain_fragments': result.get('uncertain_fragments', []),
                        'warnings': result.get('warnings', []),
                        'pipeline_result': result}
    values = list(reports.values())
    total_characters = sum(value['reconstruction_score']['expected_characters'] for value in values)
    summary = {'slides': len(values), 'returned_slides': len(outputs),
               'exact_reconstruction_slides': sum(value['reconstruction_score']['exact_normalised'] for value in values),
               'accepted_reconstruction_slides': sum(value['reconstruction_score']['exact_normalised']
                   or value['reconstruction_score'].get('matches_recorded_variant', False) for value in values),
               'word_sequence_equal_slides': sum(value['reconstruction_score']['word_sequence_equal'] for value in values),
               'reconstruction_cer': sum(value['reconstruction_score']['character_errors'] for value in values) / max(1, total_characters),
               'translation_clause_checks_passed': sum(value['translation_score']['clause_checks_passed'] for value in values),
               'translation_clause_checks_total': sum(value['translation_score']['clause_checks_total'] for value in values),
               'all_translation_clause_checks_slides': sum(value['translation_score']['all_clause_checks_passed'] for value in values),
               'delivered_translation_clause_checks_passed': sum(value['translation_score']['clause_checks_passed']
                   for value in values if value['result_returned_by_pipeline']),
               'delivered_translation_clause_checks_total': sum(value['translation_score']['clause_checks_total'] for value in values),
               'translations_delivered_without_english_fallback': sum(value['result_returned_by_pipeline']
                   and not value['fallback_to_english'] and bool(value['actual_translation']) for value in values),
               'english_fallback_slides': sum(value['fallback_to_english'] for value in values),
               'semantic_human_review': 'PENDING'}
    return reports, summary


def run_general(model, settings, inputs, cases, baseline=None):
    raw = {'reconstruction': {}, 'translation': {}}
    generations, outputs, errors = [], {}, []
    metrics = {'calls': [], 'jobs': [], 'request_count': 0, 'retry_count': 0}
    posts = dict.fromkeys(case['post_id'] for case in cases.values())
    for post_id in posts:
        translator = translator_class(baseline)(settings)
        mapping = {str(case['position']): key for key, case in cases.items() if case['post_id'] == post_id}
        chat = translator._chat

        def capture(session, stage, values, retry_error=''):
            started = time.perf_counter()
            print(json.dumps({'model': model, 'post_id': post_id, 'stage': stage, 'slides': len(values), 'retry': bool(retry_error), 'status': 'STARTING_CALL'}), flush=True)
            generated = chat(session, stage, values, retry_error)
            mapped = {mapping[key]: value for key, value in generated.items()}
            generations.append({'stage': stage, 'post_id': post_id, 'ids': list(values), 'retry': bool(retry_error), 'output': mapped})
            raw[stage].update(mapped)
            print(json.dumps({'model': model, 'stage': stage, 'seconds': round(time.perf_counter() - started, 3), 'status': 'CALL_COMPLETED'}), flush=True)
            return generated
        translator._chat = capture
        # Use the actual production numeric slide IDs and one carousel per job.
        # Baseline receives its old string contract; recognized characters match.
        source = {position: ([item['text'] if isinstance(item, dict) else item for item in inputs[key]]
                             if baseline else inputs[key]) for position, key in mapping.items()}
        started = time.perf_counter()
        try:
            translated = translator.translate(source)
            outputs.update({mapping[key]: value for key, value in translated.items()})
            error = None
        except Exception as exc:
            error = str(exc)
            errors.append(f'{post_id}: {error}')
        for call in translator.metrics.get('calls', []):
            metrics['calls'].append({**call, 'model': model, 'post_id': post_id,
                                     'slides': [mapping[key] for key in call['slides']]})
        metrics['jobs'].append({'post_id': post_id, 'seconds': round(time.perf_counter() - started, 3),
                                'error': error, 'metrics': translator.metrics})
        metrics['retry_count'] += translator.metrics.get('retry_count', 0)
    metrics['request_count'] = len(metrics['calls'])
    return outputs, raw, generations, metrics, '\n'.join(errors) if errors else None


def run_specialized(model, settings, inputs, cases, originals=None):
    """Use the model's published single-user translation template, no JSON batch."""
    outputs, raw, generations, calls = {}, {'reconstruction': {}, 'translation': {}}, [], []
    # Keep translation input constant and explicitly measured: this is a
    # translation-only upper bound, not an OCR reconstruction score.
    with requests.Session() as session:
        for key, case in cases.items():
            original = originals[key] if originals is not None else case['expected_text']
            prefix = ('You are a professional English (en) to Spanish (es-ES) translator. '
                      'Your goal is to accurately convey the meaning and nuances of the original English text '
                      'while adhering to Spanish grammar, vocabulary, and cultural sensitivities.\n'
                      'Produce only the Spanish translation, without any additional explanations or commentary. '
                      'Please translate the following English text into Spanish:\n\n\n')
            started = time.perf_counter()
            try:
                response = session.post(settings.ollama_url + '/api/chat', json={
                    'model': model, 'messages': [{'role': 'user', 'content': prefix + original}],
                    'stream': False, 'keep_alive': '30m',
                    'options': {'temperature': 0, 'repeat_penalty': 1.0, 'num_gpu': 0, 'num_ctx': 2048, 'num_predict': 700}},
                    timeout=(10, 180))
                response.raise_for_status()
                data = response.json()
                spanish = data.get('message', {}).get('content', '').strip()
                if not spanish or data.get('done_reason') == 'length':
                    raise RuntimeError('Missing or truncated specialized translation.')
                outputs[key] = {'original': original, 'spanish': spanish}
                raw['translation'][key] = spanish
                generations.append({'stage': 'specialized_translation', 'ids': [key], 'output': spanish})
                calls.append({'stage': 'translation', 'model': model, 'slides': [key],
                              'http_seconds': round(time.perf_counter() - started, 3),
                              **{field: data.get(field) for field in ('total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration', 'prompt_eval_count', 'eval_count')}})
            except Exception as exc:
                calls.append({'stage': 'translation', 'model': model, 'slides': [key], 'error': str(exc),
                              'http_seconds': round(time.perf_counter() - started, 3)})
    errors = [f"{call['slides'][0]}: {call['error']}" for call in calls if call.get('error')]
    return outputs, raw, generations, {'calls': calls, 'request_count': len(calls)}, '\n'.join(errors) or None


def label_specialized(scored, summary):
    """Distinguish raw template responses from tested production delivery."""
    for key in ('exact_reconstruction_slides', 'accepted_reconstruction_slides', 'word_sequence_equal_slides', 'reconstruction_cer'):
        summary[key] = None
    summary['raw_responses'] = summary.pop('returned_slides')
    summary['production_delivered_slides'] = 'NOT_EVALUATED'
    for key in ('delivered_translation_clause_checks_passed', 'delivered_translation_clause_checks_total',
                'translations_delivered_without_english_fallback', 'english_fallback_slides'):
        summary[key] = None
    for value in scored.values():
        value['raw_response_received'] = value.pop('result_returned_by_pipeline')
        value.pop('fallback_to_english')
        value['production_guard_validation'] = 'NOT_EVALUATED'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--media-root', type=Path)
    parser.add_argument('--ocr-report', type=Path)
    parser.add_argument('--ocr-variant', default='native_retry')
    parser.add_argument('--gold-input', action='store_true')
    parser.add_argument('--models', nargs='+', default=['gemma3:4b', 'qwen2.5:3b'])
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--specialized', action='store_true')
    parser.add_argument('--translation-input-report', type=Path,
                        help='For a specialized translation-only comparison, use the exact reconstruction from this measured report.')
    parser.add_argument('--translation-input-model', default='gemma3:4b')
    parser.add_argument('--case', action='append')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, default=PROJECT / '.cache/regression/text-models.json')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset)
    cases, inputs, ignored, input_kind = read_inputs(dataset, args)
    if not cases:
        parser.error('No cases selected.')
    tags = requests.get(args.ollama_url + '/api/tags', timeout=10).json()
    installed = {item['name']: item for item in tags['models']}
    missing = set(args.models) - set(installed)
    if missing:
        parser.error('Models are not installed: ' + ', '.join(sorted(missing)))
    specialized_originals = None
    if args.translation_input_report:
        source_report = json.loads(args.translation_input_report.read_text(encoding='utf-8'))
        source_model = next(item for item in source_report['models'] if item['model'] == args.translation_input_model)
        specialized_originals = {key: source_model['cases'][key]['actual_reconstruction'] for key in cases}
    if args.specialized and specialized_originals is None and not args.gold_input:
        parser.error('Specialized translation requires --gold-input or --translation-input-report so its input basis is explicit.')
    implementation_path = args.baseline or PROJECT / 'translator/src/translation/ollama.py'
    report = {'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              'selected_cases_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              'implementation_sha256': hashlib.sha256(implementation_path.read_bytes()).hexdigest(),
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'ocr_report_sha256': hashlib.sha256(args.ocr_report.read_bytes()).hexdigest() if args.ocr_report else None,
              'ocr_variant': args.ocr_variant if args.ocr_report else None,
              'selected_case_ids': list(cases),
              'input_kind': ('SAME_RECONSTRUCTED_TEXT_SPECIALIZED_TRANSLATION_ONLY' if specialized_originals is not None
                             else 'GOLD_TEXT_SPECIALIZED_TRANSLATION_ONLY') if args.specialized else input_kind,
              'translation_source_report': str(args.translation_input_report) if args.translation_input_report else None,
              'implementation': str(args.baseline) if args.baseline else 'production',
              'text_only': True, 'cpu_only': True,
              'metric_limitations': ['Normalised exact match/CER measure character fidelity, not semantic correctness.',
                                     'Translation clause presence is a limited proxy; human semantic review remains pending.',
                                     'Raw generations are scored even if the baseline aborts; returned_slides records actual pipeline success.',
                                     'One measured run per model; loading time and cache state are included.'],
              'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count()},
              'inputs': inputs, 'background_evidence_ignored': ignored, 'models': []}
    if args.specialized:
        report['metric_limitations'].append('Specialized responses use the published template only; production guards and end-to-end delivery have not been evaluated.')
    for model in args.models:
        # Release all resident models so each model receives the same cold-load
        # treatment, and the 100% CPU laptop never holds two LLMs concurrently.
        resident = requests.get(args.ollama_url + '/api/ps', timeout=10).json().get('models', [])
        for item in resident:
            response = requests.post(args.ollama_url + '/api/generate', json={'model': item['name'], 'keep_alive': 0}, timeout=120)
            response.raise_for_status()
        settings = Settings(ollama_url=args.ollama_url, ollama_model=model, ollama_translation_model='')
        started = time.perf_counter()
        print(json.dumps({'starting_model': model, 'slides': len(cases), 'baseline': bool(args.baseline), 'specialized': args.specialized}), flush=True)
        with MemorySampler() as sampler:
            if args.specialized:
                outputs, raw, generations, metrics, error = run_specialized(model, settings, inputs, cases, specialized_originals)
            else:
                outputs, raw, generations, metrics, error = run_general(model, settings, inputs, cases, args.baseline)
        elapsed = round(time.perf_counter() - started, 3)
        scored, summary = score_cases(cases, inputs, outputs, raw)
        if args.specialized:
            label_specialized(scored, summary)
        entry = {'model': model, 'model_info': installed[model], 'seconds': elapsed,
                 'peak_ollama_rss_bytes': sampler.peak,
                 'memory_basis': 'Sampled sum RSS of Ollama frontend/UI and llama-server; shared mapped pages may be counted more than once.',
                 'error': error, 'summary': summary,
                 'reconstruction_seconds': sum(call.get('http_seconds', 0) for call in metrics.get('calls', []) if call['stage'] == 'reconstruction'),
                 'translation_seconds': sum(call.get('http_seconds', 0) for call in metrics.get('calls', []) if call['stage'] == 'translation'),
                 'metrics': metrics, 'cases': scored, 'raw_generations': generations}
        report['models'].append(entry)
        write_report(args.output, report)
        print(json.dumps({key: entry[key] for key in ('model', 'seconds', 'peak_ollama_rss_bytes', 'error', 'summary')}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
