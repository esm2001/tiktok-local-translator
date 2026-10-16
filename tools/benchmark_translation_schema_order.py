"""Compare translation output order on measured English, without reconstruction.

Preparation is offline by default. --execute makes real, serial Ollama calls.
Only an in-memory subclass changes output property order and matching order
instructions. Production validators, weights, options and bounded retries stay
unchanged. Clause checks are a proxy; adequacy/naturalness need human review.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
from pathlib import Path
import time

import requests

from benchmark_text_regression import MemorySampler, PROJECT, write_report
from regression_metrics import load_dataset, translation_score
from src.config import Settings
from src.translation.ollama import (
    AMBIGUOUS_COMMUNICATION_HEADER, LOW_OCR_CONFIDENCE, METRIC_FIELDS,
    OllamaTranslator, PROTECTED_FRAGMENT, _ocr_evidence, _quoted_spans,
    _validate_original, clean_ocr_lines,
)

DATA_SEPARATOR = '\nDATA / DATOS:\n'
PROTECTION_REASONS = {'LOW_OCR_CONFIDENCE', 'AMBIGUOUS_COMPARATIVE_ATTACHMENT',
                      'QUOTED_TOKEN_PRESERVED'}
ARMS = ('metadata-first', 'spanish-first')


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf-8')).hexdigest()


def data_text(values):
    # Exactly the serialization used by production _messages, including spaces.
    return json.dumps(values, ensure_ascii=False)


def data_from_messages(messages):
    content = messages[-1]['content']
    if content.count(DATA_SEPARATOR) != 1:
        raise ValueError('A request must have one unambiguous source DATA boundary.')
    return content.split(DATA_SEPARATOR, 1)[1]


def prepare_protections(evidence):
    """Recreate deterministic v14 protections, never model-declared spans."""
    protected, useful = {}, []
    for index, item in enumerate(evidence):
        reason = None
        if item['confidence'] is not None and item['confidence'] < LOW_OCR_CONFIDENCE:
            reason = 'LOW_OCR_CONFIDENCE'
        elif index == 0 and AMBIGUOUS_COMMUNICATION_HEADER.fullmatch(item['text']):
            reason = 'AMBIGUOUS_COMPARATIVE_ATTACHMENT'
        if reason:
            marker = f'__KEEP_EN_{len(protected)}__'
            protected[marker] = {**item, 'reason': reason}
            useful.append(marker)
            continue
        text, replacements = item['text'], []
        for start, end, fragment in _quoted_spans(text):
            if len(fragment.split()) == 1 and len(fragment) <= 40:
                marker = f'__KEEP_EN_{len(protected)}__'
                protected[marker] = {**item, 'text': fragment, 'reason': 'QUOTED_TOKEN_PRESERVED'}
                replacements.append((start, end, marker))
        for start, end, marker in reversed(replacements):
            text = text[:start] + marker + text[end:]
        useful.append(text)
    return useful, protected


def rehydrate_source(slide, stored):
    """Require exact rendered English and unique literal protection locations.

    An unavailable original wire hash cannot be reconstructed from a manifest.
    Render equality, OCR input hash, and logged character counts are independent
    checks; none is represented as a captured historical request hash.
    """
    editorial = slide['ocr']['editorial_lines']
    if clean_ocr_lines(editorial) != slide['translation_ocr_lines']:
        raise ValueError('Editorial OCR differs from recorded translation input.')
    evidence = _ocr_evidence(editorial)
    useful, protected = prepare_protections(evidence)
    recorded = [value for value in stored.get('uncertain_fragments', [])
                if value.get('reason') in PROTECTION_REASONS]
    actual = [{'ocr': item['text'], 'ocr_confidence': item['confidence'], 'reason': item['reason']}
              for item in protected.values()]
    if [{key: value.get(key) for key in ('ocr', 'ocr_confidence', 'reason')}
            for value in recorded] != actual:
        raise ValueError('Current deterministic protections differ from measured protections.')
    rendered = stored['original']
    if (rendered != stored['reconstruction']['reconstructed_english']
            or rendered != slide['ocr']['reconstructed']
            or rendered != slide['ocr']['reconstruction']['reconstructed_english']
            or rendered != slide['translation']['original_reconstructed']):
        raise ValueError('Recorded English reconstruction fields disagree.')
    if PROTECTED_FRAGMENT.search(rendered):
        raise ValueError('Rendered source already contains a marker; its origin is ambiguous.')
    source = rendered
    for marker, item in protected.items():
        if source.count(item['text']) != 1:
            raise ValueError('A protected literal has no unique exact location in English.')
        source = source.replace(item['text'], marker, 1)
    if PROTECTED_FRAGMENT.findall(source) != list(protected):
        raise ValueError('Rehydrated markers are missing, duplicated or out of source order.')
    roundtrip = source
    for marker, item in protected.items():
        roundtrip = roundtrip.replace(marker, item['text'])
    if roundtrip != rendered:
        raise ValueError('Protected source does not round-trip byte-for-byte.')
    _validate_original(str(slide['position']), useful, source)
    return {'original_rendered': rendered, 'model_source': source,
            'model_source_sha256': sha(source), 'rendered_sha256': sha(rendered),
            'source_characters': len(source), 'evidence': evidence,
            'evidence_sha256': sha(data_text(evidence)), 'useful_lines': useful,
            'protected': protected, 'protected_ids_valid': True,
            'previous_reconstruction': copy.deepcopy(stored['reconstruction'])}


class SpanishFirstTranslator(OllamaTranslator):
    @staticmethod
    def _schema(values, stage=''):
        schema = copy.deepcopy(OllamaTranslator._schema(values, stage))
        if stage == 'translation':
            for value in schema['properties'].values():
                properties = value['properties']
                value['properties'] = {key: properties[key] for key in ('spanish', 'uncertain_spans')}
                value['required'] = ['spanish', 'uncertain_spans']
        return schema

    def _messages(self, stage, values, schema, retry_error=''):
        messages = super()._messages(stage, values, schema, retry_error)
        if stage != 'translation':
            return messages
        declaration = ('Declara sólo ambigüedad sin resolver por el contexto en uncertain_spans usando source EXACTO de DATOS '
                       'y reason AMBIGUOUS_SOURCE, INCOMPLETE_SOURCE o UNKNOWN_TERM. ')
        translation = ('Después escribe spanish: COPIA cada source declarado literalmente en inglés, en su posición; '
                       'no lo traduzcas. Traduce el resto sin adivinar ni ampliar su significado. ')
        previous = declaration + translation
        if messages[0]['content'].count(previous) != 1:
            raise ValueError('Production ordering instructions changed; review this experiment before running.')
        reordered = (translation.replace('Después escribe spanish', 'Escribe primero spanish', 1)
                     .replace('cada source declarado', 'cada source que declares', 1)
                     + declaration.replace('Declara sólo', 'Declara después sólo', 1))
        messages[0]['content'] = messages[0]['content'].replace(previous, reordered, 1)
        task = 'Cada valor contiene uncertain_spans primero y spanish después.'
        if messages[-1]['content'].count(task) != 1:
            raise ValueError('Production task ordering changed; review before running.')
        messages[-1]['content'] = messages[-1]['content'].replace(
            task, 'Cada valor contiene spanish primero y uncertain_spans después.', 1)
        return messages


def settings_for(signature, url):
    return Settings(ollama_url=url, ollama_model=signature['model'],
                    ollama_translation_model=signature['translation_model'],
                    ollama_keep_alive='30m', translation_timeout_seconds=180,
                    translation_budget_seconds=360)


def request_plan(translator, values):
    schema = translator._schema(values, 'translation')
    messages = translator._messages('translation', values, schema)
    source_json = data_from_messages(messages)
    if source_json != data_text(values):
        raise ValueError('Prompt DATA changed while constructing an arm.')
    cap, characters = translator._output_token_budget('translation', values)
    return {'data': values, 'data_json': source_json, 'data_sha256': sha(source_json),
            'data_bytes': len(source_json.encode('utf-8')), 'source_characters': characters,
            'messages': messages, 'format': schema,
            'options': {'temperature': 0, 'repeat_penalty': 1.0, 'num_ctx': 4096,
                        'num_gpu': 0, 'num_predict': cap}}


def prepare(dataset, source_report, host_output_root, url):
    pipeline_path = source_report['pipeline_output']
    prefix = '/data/tiktok/'
    if not pipeline_path.startswith(prefix) or '..' in Path(pipeline_path).parts:
        raise ValueError('Source report must resolve inside the explicit host output root.')
    posts_root = host_output_root / pipeline_path[len(prefix):]
    selected = {key: case for key, case in dataset['slides'].items()
                if not case.get('expected_editorial_empty', False)}
    excluded = [{'case_id': key, 'reason': 'ANNOTATED_NO_EDITORIAL_TEXT'}
                for key, case in dataset['slides'].items() if key not in selected]
    expected = source_report['expected_cohorts']['editorial_text']
    if len(selected) != expected:
        raise ValueError('Dataset editorial selection does not match measured cohort count.')
    groups, records, signatures = [], {}, []
    for post_id in dict.fromkeys(str(case['post_id']) for case in selected.values()):
        if not post_id.isdecimal():
            raise ValueError('Post IDs must be numeric path components.')
        path = posts_root / post_id / 'metadata/manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        signature = manifest['signature']
        if signature['translation'] != 'evidence-slide-fallback-v14':
            raise ValueError('This isolated experiment requires measured v14 inputs.')
        signatures.append(signature)
        # This is the original production hash, including the excluded logo slide.
        complete_ocr = {slide['position']: slide['ocr']['editorial_lines'] for slide in manifest['slides']}
        original_input_hash = sha(str(complete_ocr))
        if original_input_hash != manifest['translation_input_sha256']:
            raise ValueError('Recorded production OCR input hash no longer matches the manifest.')
        by_position = {str(slide['position']): slide for slide in manifest['slides']}
        mapping = {str(case['position']): key for key, case in selected.items() if str(case['post_id']) == post_id}
        if len(mapping) != sum(str(case['post_id']) == post_id for case in selected.values()):
            raise ValueError('Duplicate production slide IDs in cohort.')
        for index, key in mapping.items():
            records[key] = {'case_id': key, 'post_id': post_id, 'numeric_id': index,
                            **rehydrate_source(by_position[index], manifest['translations'][index])}
        measured_calls = [call for call in manifest['ollama_metrics']['calls'] if call['stage'] == 'translation']
        validations = []
        covered = set()
        for call in measured_calls:
            ids = call['slides']
            # A mixed batch including an excluded slide cannot prove its subset
            # length; use the selected slide's measured individual retry instead.
            if not set(ids).issubset(mapping):
                continue
            values = {index: records[mapping[index]]['model_source'] for index in ids}
            characters = sum(map(len, values.values()))
            if characters != call['source_characters']:
                raise ValueError('Rehydrated source size differs from a measured translation call.')
            historical_hash = call.get('data_sha256')
            if historical_hash is not None and historical_hash != sha(data_text(values)):
                raise ValueError('Rehydrated DATA hash differs from measured wire evidence.')
            validations.append({'ids': ids, 'retry': call['retry'], 'source_characters': characters,
                                'historical_data_sha256': historical_hash,
                                'check': 'MATCHED_CHARACTERS_AND_ANY_AVAILABLE_WIRE_HASH'})
            covered.update(ids)
        if covered != set(mapping):
            raise ValueError('Every selected source needs a measured translation character check.')
        plans = {}
        values = {index: records[key]['model_source'] for index, key in mapping.items()}
        for arm, cls in zip(ARMS, (OllamaTranslator, SpanishFirstTranslator)):
            translator = cls(settings_for(signature, url))
            translator._evidence = {index: records[key]['evidence'] for index, key in mapping.items()}
            plans[arm] = request_plan(translator, values)
        if (plans[ARMS[0]]['data_json'] != plans[ARMS[1]]['data_json']
                or plans[ARMS[0]]['options'] != plans[ARMS[1]]['options']):
            raise ValueError('Output-order arms do not have identical DATA/options.')
        # JSON object order has no schema-validation meaning: assert all rules
        # and values equal when serialized with canonical key order.
        def canonical_schema(value):
            if isinstance(value, dict):
                return {key: (sorted(item) if key == 'required' else canonical_schema(item))
                        for key, item in value.items()}
            if isinstance(value, list):
                return [canonical_schema(item) for item in value]
            return value
        if (json.dumps(canonical_schema(plans[ARMS[0]]['format']), sort_keys=True)
                != json.dumps(canonical_schema(plans[ARMS[1]]['format']), sort_keys=True)):
            raise ValueError('Schema changed beyond property/required ordering.')
        prompt_diff = '\n'.join(difflib.unified_diff(
            [item['content'] for item in plans[ARMS[0]]['messages']],
            [item['content'] for item in plans[ARMS[1]]['messages']],
            fromfile=ARMS[0], tofile=ARMS[1], lineterm=''))
        groups.append({'post_id': post_id, 'mapping': mapping, 'signature': signature,
                       'manifest_sha256': sha(path.read_bytes()), 'ocr_input_sha256': original_input_hash,
                       'logged_source_checks': validations, 'plans': plans, 'prompt_diff': prompt_diff})
    if len(groups) != len(source_report['posts']):
        raise ValueError('Selection must retain the measured production post grouping.')
    if len({(sig['model'], sig['translation_model']) for sig in signatures}) != 1:
        raise ValueError('A paired order comparison requires one unchanged model.')
    return {'status': 'PREPARED_OFFLINE', 'scope': 'TRANSLATION_ONLY_REUSED_V14_ENGLISH',
            'selected_case_ids': list(selected), 'excluded_cases': excluded,
            'records': records, 'groups': groups, 'production_architecture_changed': False,
            'ocr_measurement': 'REUSED_MEASURED', 'reconstruction_measurement': 'REUSED_MEASURED',
            'ocr_seconds': None, 'reconstruction_seconds': None,
            'historical_translation_source_wire_data_hash': 'NOT_CAPTURED_WHERE_ABSENT',
            'post_translation_budget_seconds': 360, 'call_timeout_seconds': 180,
            'budget_scope': 'Fresh translation-only clock per arm/post; unlike residual E2E budget after reconstruction.',
            'grouping_scope': 'Measured post IDs retained. The annotated logo slide is excluded, so holdout is 1 slide here versus 2 in E2E15. Compare new arms only.',
            'semantic_human_review': 'PENDING',
            'limitations': ['No latency claim about E2E; OCR and reconstruction are reused without fabricated times.',
                            'Historical source wire JSON/hash was not persisted when logged_source_checks has null hashes.',
                            'Temperature zero does not guarantee reproducible output; one paired run is exploratory.',
                            'Property order requested by a schema must be checked against actual wire key order.',
                            'Clause checks and production guards do not certify semantic equivalence or naturalness.']}


class CaptureSession:
    """Capture final message.content only, including invalid/truncated JSON."""
    def __init__(self, session, records, mapping, calls):
        self.session, self.records, self.mapping, self.calls = session, records, mapping, calls
        self.context = {}

    def post(self, url, **kwargs):
        if not url.endswith('/api/chat'):
            raise ValueError('This harness permits translation chat requests only.')
        payload = kwargs['json']
        source_json = data_from_messages(payload['messages'])
        values = json.loads(source_json)
        expected = {index: self.records[self.mapping[index]]['model_source'] for index in values}
        if source_json != data_text(expected):
            raise ValueError('Wire source changed from prepared DATA; refusing the model call.')
        call = {**self.context, 'ids': list(values), 'data': values, 'data_sha256': sha(source_json),
                'data_bytes': len(source_json.encode('utf-8')), 'protected_ids_valid': True,
                'request': payload, 'request_sha256': sha(data_text(payload)),
                'timeout': kwargs.get('timeout'), 'response_content': None,
                'response_key_order_by_id': {}, 'parsed_final_content': None}
        self.calls.append(call)
        started = time.perf_counter()
        try:
            response = self.session.post(url, **kwargs)
            call['http_status'] = response.status_code
            try:
                body = response.json()
                # Do not retain message.thinking, even if the backend returns it.
                content = body.get('message', {}).get('content')
                call['response_content'] = content
                call['telemetry'] = {key: body[key] for key in (*METRIC_FIELDS, 'done_reason', 'done') if key in body}
                parsed = json.loads(content) if isinstance(content, str) else None
                call['parsed_final_content'] = parsed
                if isinstance(parsed, dict):
                    # Pairs preserve wire order, including any duplicate keys;
                    # normal parsed JSON remains separate for production parity.
                    pairs = json.loads(content, object_pairs_hook=list)
                    call['response_id_order'] = [key for key, _ in pairs]
                    call['response_key_order_by_id'] = {
                        key: [name for name, _ in value] if isinstance(parsed.get(key), dict) else None
                        for key, value in pairs}
            except (ValueError, TypeError, AttributeError) as exc:
                call['content_parse_error'] = str(exc)
            return response
        except requests.RequestException as exc:
            call['http_error'] = str(exc)
            raise
        finally:
            call['wire_seconds'] = round(time.perf_counter() - started, 3)


def run_job(arm, group, records, url):
    cls = OllamaTranslator if arm == ARMS[0] else SpanishFirstTranslator
    translator = cls(settings_for(group['signature'], url))
    mapping = group['mapping']
    translator._evidence = {index: records[key]['evidence'] for index, key in mapping.items()}
    translator.metrics = {'calls': [], 'request_count': 0, 'retry_count': 0, 'http_seconds': 0.0,
                          'model_switches': [], 'model_switch_count': 0, 'model_switch_seconds': 0.0,
                          'input_tokens': 0, 'output_tokens': 0, **{key: 0 for key in METRIC_FIELDS}}
    originals = {index: records[key]['model_source'] for index, key in mapping.items()}
    calls = []
    chat = translator._chat
    with requests.Session() as underlying:
        session = CaptureSession(underlying, records, mapping, calls)

        def capture(session, stage, values, retry_error=''):
            if stage != 'translation':
                raise ValueError('No reconstruction is permitted in a translation-only experiment.')
            session.context = {'stage': stage, 'retry': bool(retry_error), 'retry_error': retry_error}
            print(json.dumps({'arm': arm, 'post_id': group['post_id'], 'ids': list(values),
                              'retry': bool(retry_error), 'status': 'STARTING_CALL'}), flush=True)
            return chat(session, stage, values, retry_error)
        translator._chat = capture
        translator._started = time.perf_counter()
        spanish = translator._stage(session, 'translation', originals,
            lambda key, value: translator._translation_content(
                key, records[mapping[key]]['useful_lines'], originals[key], value),
            lambda key: originals[key])
        outputs = {mapping[index]: translator._result(index, records[key]['useful_lines'],
                       originals[index], spanish[index], records[key]['protected'])
                   for index, key in mapping.items()}
    for key, value in outputs.items():
        if value['original'] != records[key]['original_rendered']:
            raise ValueError('Delivered reconstructed English changed from reused measurement.')
        value['reconstruction']['measurement'] = 'REUSED_MEASURED'
    return {'arm': arm, 'post_id': group['post_id'],
            'seconds': round(time.perf_counter() - translator._started, 3),
            'wire_calls': calls, 'metrics': translator.metrics, 'outputs': outputs,
            'errors': translator._failure_details, 'semantic_human_review': 'PENDING'}


def execute(report, dataset, url, output):
    # All offline source, protection and hash checks have already passed.
    model = report['groups'][0]['signature']['translation_model']
    version = requests.get(url + '/api/version', timeout=10)
    version.raise_for_status()
    tags = requests.get(url + '/api/tags', timeout=10)
    tags.raise_for_status()
    installed = {value['name']: value for value in tags.json()['models']}
    if model not in installed:
        raise ValueError('Only the already installed measured model may be used: ' + model)
    report.update(status='RUNNING', model_info=installed[model], ollama_version=version.json(), jobs=[])
    report['execution_order'] = [{'post_id': group['post_id'], 'arm': arm}
        for number, group in enumerate(report['groups'])
        for arm in (ARMS if number % 2 == 0 else tuple(reversed(ARMS)))]
    report['order_scope'] = 'Serial, paired posts with alternating arm order; no additional repetitions or forced reloads.'
    started = time.perf_counter()
    with MemorySampler() as memory:
        for planned in report['execution_order']:
            group = next(value for value in report['groups'] if value['post_id'] == planned['post_id'])
            job = run_job(planned['arm'], group, report['records'], url)
            for key, value in job['outputs'].items():
                # Expected data is accessed only after the request and delivery.
                value['delivered_clause_proxy'] = translation_score(dataset['slides'][key], value['spanish'])
            for call in job['wire_calls']:
                parsed = call['parsed_final_content']
                call['raw_clause_proxy'] = {}
                if isinstance(parsed, dict):
                    for index, value in parsed.items():
                        if index in group['mapping'] and isinstance(value, dict) and isinstance(value.get('spanish'), str):
                            key = group['mapping'][index]
                            raw_target = value['spanish']
                            for marker, item in report['records'][key]['protected'].items():
                                raw_target = raw_target.replace(marker, item['text'])
                            call['raw_clause_proxy'][index] = translation_score(dataset['slides'][key], raw_target)
                call['raw_scores_are_not_delivered_scores'] = True
            report['jobs'].append(job)
            report['seconds'] = round(time.perf_counter() - started, 3)
            report['peak_ollama_rss_bytes'] = memory.peak
            write_report(output, report)
    report.update(status='MEASURED_HUMAN_REVIEW_PENDING', seconds=round(time.perf_counter() - started, 3),
                  peak_ollama_rss_bytes=memory.peak)
    write_report(output, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', type=Path, required=True, help='Measured v14 pipeline revalidation JSON.')
    parser.add_argument('--dataset', type=Path, default=PROJECT / '.cache/regression/e2e-expected.json')
    parser.add_argument('--host-output-root', type=Path, default=PROJECT.parent / 'TikTokTraducidos')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--output', type=Path, default=PROJECT / '.cache/regression/text-schema-order.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Prepare and validate offline; no HTTP requests (default).')
    mode.add_argument('--execute', action='store_true', help='Perform real translation calls after preparation.')
    args = parser.parse_args(argv)
    report = prepare(load_dataset(args.dataset), json.loads(args.source_report.read_text(encoding='utf-8')),
                     args.host_output_root, args.ollama_url.rstrip('/'))
    report.update(dataset_sha256=sha(args.dataset.read_bytes()), source_report_sha256=sha(args.source_report.read_bytes()),
                  implementation_sha256=sha((PROJECT / 'translator/src/translation/ollama.py').read_bytes()),
                  benchmark_sha256=sha(Path(__file__).read_bytes()))
    write_report(args.output, report)
    if args.execute:
        execute(report, load_dataset(args.dataset), args.ollama_url.rstrip('/'), args.output)
    print(json.dumps({'status': report['status'], 'cases': len(report['records']),
                      'groups': len(report['groups']), 'excluded': report['excluded_cases'],
                      'payloads': [{'post_id': group['post_id'], 'ids': list(group['mapping']),
                                   'data_sha256': group['plans'][ARMS[0]]['data_sha256']}
                                  for group in report['groups']]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
