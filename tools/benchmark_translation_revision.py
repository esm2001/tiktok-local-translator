"""Measure a guarded translation revision experiment on measured reconstructions.

No reconstruction or dedicated translator runs here. Their existing reports
provide English and provisional drafts. This tool does not modify production
architecture, source media, user outputs, or the selected default model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import requests

from benchmark_text_regression import MemorySampler, PROJECT, score_cases, write_report
from regression_metrics import load_dataset
from src.config import Settings
from src.translation.ollama import OllamaTranslator, PROTECTED_FRAGMENT, _quoted_spans


def revision_class(originals, drafts, mapping, generations):
    class DraftRevisionTranslator(OllamaTranslator):
        def _messages(self, stage, values, schema, retry_error=''):
            messages = super()._messages(stage, values, schema, retry_error)
            if stage == 'translation':
                messages[0]['content'] += (
                    ' Recibirás también un borrador provisional de otro traductor. '
                    'El inglés de DATOS es la única fuente; el borrador puede omitir o cambiar significado. '
                    'Revísalo contra cada frase inglesa, corrige gramática y referentes y redacta castellano natural completo. '
                    'Rechaza cualquier inferencia del borrador sin apoyo en el inglés. El borrador no resuelve ambigüedad de la fuente.')
                provisional = {}
                for key, value in values.items():
                    draft = drafts[key]
                    # Preserve the production protection of quoted literal
                    # tokens in the provisional target, as in the English.
                    old_quotes, masked_quotes = _quoted_spans(originals[key]), _quoted_spans(value)
                    for old, masked in zip(old_quotes, masked_quotes):
                        if PROTECTED_FRAGMENT.fullmatch(masked[2]):
                            draft = draft.replace(old[2], masked[2])
                    provisional[key] = draft
                messages[-1]['content'] += ('\nBORRADORES PROVISIONALES (no son fuente):\n'
                                           + json.dumps(provisional, ensure_ascii=False))
            return messages

        def _chat(self, session, stage, values, retry_error=''):
            if stage == 'reconstruction':
                # No Ollama call and no fabricated reconstruction timing.
                generated = {key: originals[key] for key in values}
                generations.append({'stage': 'reconstruction', 'measurement': 'REUSED_MEASURED',
                                    'ids': list(values), 'output': {mapping[k]: v for k, v in generated.items()}})
                return generated
            started = time.perf_counter()
            print(json.dumps({'stage': 'translation_revision', 'ids': list(values),
                              'retry': bool(retry_error), 'status': 'STARTING_CALL'}), flush=True)
            generated = super()._chat(session, stage, values, retry_error)
            generations.append({'stage': 'translation', 'measurement': 'MEASURED_REVISION_CALL',
                                'ids': list(values), 'retry': bool(retry_error),
                                'output': {mapping[k]: v for k, v in generated.items()}})
            print(json.dumps({'stage': 'translation_revision', 'seconds': round(time.perf_counter() - started, 3),
                              'status': 'CALL_COMPLETED'}), flush=True)
            return generated
    return DraftRevisionTranslator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--source-model', default='gemma3:4b')
    parser.add_argument('--draft-report', type=Path, required=True)
    parser.add_argument('--model', default='gemma3:4b')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset)
    measured = json.loads(args.source_report.read_text(encoding='utf-8'))
    source = next(value for value in measured['models'] if value['model'] == args.source_model)
    draft = json.loads(args.draft_report.read_text(encoding='utf-8'))
    cases = {key: dataset['slides'][key] for key in draft['cases']}
    inputs = {key: source['cases'][key]['input_ocr'] for key in cases}
    for key in cases:
        if source['cases'][key]['actual_reconstruction'] != draft['cases'][key]['actual_reconstruction']:
            raise ValueError('Draft and measured English disagree: ' + key)
    tags = requests.get(args.ollama_url + '/api/tags', timeout=10).json()
    installed = {value['name']: value for value in tags['models']}
    if args.model not in installed:
        parser.error('Model must already be installed: ' + args.model)
    for resident in requests.get(args.ollama_url + '/api/ps', timeout=10).json().get('models', []):
        response = requests.post(args.ollama_url + '/api/generate',
                                 json={'model': resident['name'], 'keep_alive': 0}, timeout=120)
        response.raise_for_status()
    report = {'architecture': 'EXPERIMENTAL_MEASURED_ENGLISH_PLUS_PROVISIONAL_NMT_DRAFT_REVISION',
              'scope': 'TRANSLATION_REVISION_ONLY_WITH_CURRENT_PRODUCTION_GUARDS',
              'production_architecture_changed': False, 'reconstruction_measurement': 'REUSED_MEASURED',
              'reconstruction_seconds': None, 'nmt_generation_seconds': None,
              'text_only': True, 'cpu_only': True, 'selected_case_ids': list(cases),
              'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              'source_report_sha256': hashlib.sha256(args.source_report.read_bytes()).hexdigest(),
              'draft_report_sha256': hashlib.sha256(args.draft_report.read_bytes()).hexdigest(),
              'implementation_sha256': hashlib.sha256((PROJECT / 'translator/src/translation/ollama.py').read_bytes()).hexdigest(),
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model': args.model, 'model_info': installed[args.model],
              'metric_limitations': ['Clause checks are a proxy; semantic and naturalness review is pending.',
                                     'Existing English and provisional drafts are reused, not generated or timed here.',
                                     'Peak Ollama RSS may count shared mapped pages twice.'],
              'inputs': inputs, 'raw_generations': [], 'metrics': {'jobs': [], 'calls': []}, 'cases': {}}
    outputs, raw = {}, {'reconstruction': {}, 'translation': {}}
    started = time.perf_counter()
    with MemorySampler() as sampler:
        for post in dict.fromkeys(case['post_id'] for case in cases.values()):
            mapping = {str(case['position']): key for key, case in cases.items() if case['post_id'] == post}
            originals = {index: source['cases'][key]['actual_reconstruction'] for index, key in mapping.items()}
            drafts = {index: draft['cases'][key]['actual_translation'] for index, key in mapping.items()}
            translator = revision_class(originals, drafts, mapping, report['raw_generations'])(
                Settings(ollama_url=args.ollama_url, ollama_model=args.model, ollama_translation_model=''))
            job_start = time.perf_counter()
            result = translator.translate({index: inputs[key] for index, key in mapping.items()})
            outputs.update({mapping[index]: value for index, value in result.items()})
            for key, value in result.items():
                measured_original = originals[key]
                if ''.join(value['original'].split()) != ''.join(measured_original.split()):
                    raise ValueError('Reused English changed: ' + mapping[key])
            report['metrics']['jobs'].append({'post_id': post, 'seconds': round(time.perf_counter() - job_start, 3),
                                             'metrics': translator.metrics})
            report['metrics']['calls'].extend({**call, 'post_id': post,
                'slides': [mapping[index] for index in call['slides']]} for call in translator.metrics['calls'])
            report['cases'], report['summary'] = score_cases(
                {key: cases[key] for key in outputs}, {key: inputs[key] for key in outputs}, outputs, raw)
            report['seconds'] = round(time.perf_counter() - started, 3)
            report['peak_ollama_rss_bytes'] = sampler.peak
            write_report(args.output, report)
    report['seconds'] = round(time.perf_counter() - started, 3)
    report['peak_ollama_rss_bytes'] = sampler.peak
    report['translation_revision_seconds'] = sum(call.get('http_seconds', 0) for call in report['metrics']['calls'])
    write_report(args.output, report)
    print(json.dumps({key: report[key] for key in ('seconds', 'peak_ollama_rss_bytes', 'summary')}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
