"""Offline safeguards for measured source reuse, without an Ollama call."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from benchmark_translation_schema_order import (
    CaptureSession, main, prepare, rehydrate_source, sha,
)


def measured_slide(text, fragments=()):
    lines = [{'text': line, 'confidence': .96} for line in text.split('\n')]
    reconstruction = {'reconstructed_english': text, 'audit': {'accepted': True}}
    slide = {'position': 1, 'translation_ocr_lines': [line['text'] for line in lines],
             'ocr': {'editorial_lines': lines, 'reconstructed': text,
                     'reconstruction': reconstruction},
             'translation': {'original_reconstructed': text}}
    stored = {'original': text, 'reconstruction': reconstruction,
              'uncertain_fragments': list(fragments)}
    return slide, stored


def fixture(root):
    text = 'Keep looking.\nThe road ends here.'
    slide, stored = measured_slide(text)
    signature = {'translation': 'evidence-slide-fallback-v14',
                 'model': 'gemma3:4b', 'translation_model': 'gemma3:4b'}
    manifest = {'slides': [slide], 'translations': {'1': stored}, 'signature': signature,
                'translation_input_sha256': sha(str({1: slide['ocr']['editorial_lines']})),
                'ollama_metrics': {'calls': [{'stage': 'translation', 'slides': ['1'],
                                             'retry': False, 'source_characters': len(text)}]}}
    location = root / 'run/posts/123/metadata/manifest.json'
    location.parent.mkdir(parents=True)
    location.write_text(json.dumps(manifest), encoding='utf-8')
    dataset = {'slides': {'case1': {'post_id': '123', 'position': 1,
                                  'expected_text': 'GOLD_SENTINEL_NEVER_PROMPT',
                                  'expected_translation': 'SECRET_EXPECTED_TARGET'},
                          'logo': {'post_id': '123', 'position': 2,
                                   'expected_editorial_empty': True,
                                   'expected_text': 'ANOTHER_GOLD_SENTINEL'}}}
    report = {'pipeline_output': '/data/tiktok/run/posts', 'posts': [{'post_id': '123'}],
              'expected_cohorts': {'editorial_text': 1}}
    return dataset, report, manifest, location


class SourceReuseTests(unittest.TestCase):
    def test_unicode_quoted_token_roundtrips_and_failure_fragment_is_not_protected(self):
        text = 'They label it "zürq". Wait here.'
        protected = {'ocr': 'zürq', 'ocr_confidence': .96,
                     'reason': 'QUOTED_TOKEN_PRESERVED'}
        failed = {'ocr': text, 'reason': 'TRANSLATION_VALIDATION_FAILED'}
        slide, stored = measured_slide(text, [protected, failed])
        value = rehydrate_source(slide, stored)
        self.assertEqual(value['model_source'], 'They label it "__KEEP_EN_0__". Wait here.')
        self.assertEqual(value['protected']['__KEEP_EN_0__']['text'], 'zürq')
        self.assertEqual(value['model_source_sha256'], sha(value['model_source']))

    def test_repeated_literal_cannot_be_guessed_from_rendered_text(self):
        text = 'They say "zürq". The label is zürq.'
        slide, stored = measured_slide(text, [{'ocr': 'zürq', 'ocr_confidence': .96,
                                              'reason': 'QUOTED_TOKEN_PRESERVED'}])
        with self.assertRaisesRegex(ValueError, 'unique exact location'):
            rehydrate_source(slide, stored)

    def test_disagreeing_recorded_source_or_protection_aborts(self):
        slide, stored = measured_slide('Keep looking.')
        broken = copy.deepcopy(slide)
        broken['translation']['original_reconstructed'] = 'Changed content.'
        with self.assertRaisesRegex(ValueError, 'fields disagree'):
            rehydrate_source(broken, stored)
        stored['uncertain_fragments'] = [{'ocr': 'Keep looking.', 'ocr_confidence': .96,
                                         'reason': 'LOW_OCR_CONFIDENCE'}]
        with self.assertRaisesRegex(ValueError, 'protections differ'):
            rehydrate_source(slide, stored)

    def test_ground_truth_annotations_select_cases_but_never_supply_prompt_text(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, report, _, _ = fixture(Path(directory))
            result = prepare(dataset, report, Path(directory), 'http://localhost:11434')
        self.assertEqual(result['selected_case_ids'], ['case1'])
        self.assertEqual(result['excluded_cases'][0]['case_id'], 'logo')
        for plan in result['groups'][0]['plans'].values():
            encoded = json.dumps(plan)
            self.assertNotIn('GOLD_SENTINEL', encoded)
            self.assertNotIn('SECRET_EXPECTED_TARGET', encoded)
            self.assertEqual(plan['data'], {'1': 'Keep looking.\nThe road ends here.'})

    def test_changed_historical_character_count_or_input_hash_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, report, manifest, path = fixture(root)
            manifest['ollama_metrics']['calls'][0]['source_characters'] += 1
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source size differs'):
                prepare(dataset, report, root, 'http://localhost:11434')
            manifest['translation_input_sha256'] = 'changed'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'OCR input hash'):
                prepare(dataset, report, root, 'http://localhost:11434')

    def test_default_preparation_performs_no_http_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, report, _, _ = fixture(root)
            dataset_file, source = root / 'dataset.json', root / 'report.json'
            dataset_file.write_text(json.dumps(dataset), encoding='utf-8')
            source.write_text(json.dumps(report), encoding='utf-8')
            with patch('benchmark_translation_schema_order.load_dataset', return_value=dataset), \
                    patch('requests.get', side_effect=AssertionError('Unexpected HTTP')), \
                    patch('requests.Session.post', side_effect=AssertionError('Unexpected model call')):
                self.assertEqual(main(['--dataset', str(dataset_file), '--source-report', str(source),
                                       '--host-output-root', str(root), '--output', str(root / 'prepared.json')]), 0)

    def test_wire_capture_keeps_content_and_order_but_omits_thinking(self):
        underlying = Mock()
        response = underlying.post.return_value
        response.status_code = 200
        response.json.return_value = {'message': {'content': '{"1":{"spanish":"Mira.","uncertain_spans":[]}}',
                                                  'thinking': 'PRIVATE_THINKING_NOT_FOR_REPORT'}, 'done': True}
        calls = []
        capture = CaptureSession(underlying, {'case': {'model_source': 'Look.'}}, {'1': 'case'}, calls)
        capture.post('http://localhost/api/chat', json={'messages': [{'content': '\nDATA / DATOS:\n{"1": "Look."}'}]})
        self.assertEqual(calls[0]['response_key_order_by_id']['1'], ['spanish', 'uncertain_spans'])
        self.assertNotIn('PRIVATE_THINKING', json.dumps(calls))
        underlying.reset_mock()
        with self.assertRaisesRegex(ValueError, 'Wire source changed'):
            capture.post('http://localhost/api/chat', json={'messages': [{'content': '\nDATA / DATOS:\n{"1": "Altered."}'}]})
        underlying.post.assert_not_called()


if __name__ == '__main__':
    unittest.main()
