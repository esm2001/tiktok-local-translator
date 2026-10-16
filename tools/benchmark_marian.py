"""Measure an optional CTranslate2 EN-to-ES model on existing measured reconstructions.

No weights or runtime libraries are downloaded automatically. This measures raw
dedicated translation; it does not pass through production delivery validators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time

from regression_metrics import load_dataset, time_summary, translation_score

PROJECT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--translation-input-report', type=Path, required=True)
    parser.add_argument('--translation-input-model', default='gemma3:4b')
    parser.add_argument('--dataset', type=Path, default=PROJECT / 'tests/regression/expected.json')
    parser.add_argument('--holdout', type=Path)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--runtime-path', type=Path,
                        help='Optional isolated pip --target directory containing ctranslate2/sentencepiece.')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--sentence-chunks', action='store_true',
                        help='Translate complete reconstructed sentences separately, never raw OCR lines.')
    parser.add_argument('--output', type=Path, default=PROJECT / '.cache/regression/text-marian.json')
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 16:
        parser.error('--threads must be between 1 and 16.')
    dataset = load_dataset(args.dataset)
    cases = dict(dataset['slides'])
    if args.holdout:
        additions = load_dataset(args.holdout)['slides']
        if set(cases).intersection(additions):
            parser.error('Holdout and regression case IDs overlap.')
        cases.update(additions)
    source = json.loads(args.translation_input_report.read_text(encoding='utf-8'))
    measured = next((item for item in source['models'] if item['model'] == args.translation_input_model), None)
    if measured is None:
        parser.error('The requested model is absent from the measured input report.')
    selected = measured['cases']
    if not selected or set(selected).difference(cases):
        parser.error('Measured cases must be nonempty and present in the dataset/holdout.')
    inputs = {key: item.get('actual_reconstruction') for key, item in selected.items()}
    if any(not isinstance(text, str) or not text.strip() for text in inputs.values()):
        parser.error('Every measured reconstruction must be a nonempty string; no ground-truth substitution.')
    required = ('config.json', 'model.bin', 'shared_vocabulary.json', 'source.spm', 'target.spm')
    if any(not (args.model_dir / name).is_file() for name in required):
        parser.error('Local CTranslate2 model is incomplete; prepare it explicitly first.')
    if args.runtime_path:
        sys.path.insert(0, str(args.runtime_path.resolve()))
    try:
        import ctranslate2
        import sentencepiece
        import psutil
    except ImportError as exc:
        parser.error('Optional benchmark dependencies are missing: ' + str(exc))
    peak = [0]
    stopped = threading.Event()
    process = psutil.Process()

    def sample_memory():
        while not stopped.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            stopped.wait(.02)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    started = time.perf_counter()
    results = {}
    try:
        encoder = sentencepiece.SentencePieceProcessor(str(args.model_dir / 'source.spm'))
        decoder = sentencepiece.SentencePieceProcessor(str(args.model_dir / 'target.spm'))
        model = ctranslate2.Translator(str(args.model_dir), device='cpu', compute_type='int8',
                                     inter_threads=1, intra_threads=args.threads)
        config = json.loads((args.model_dir / 'config.json').read_text(encoding='utf-8'))
        initialization = time.perf_counter() - started
        for key, text in inputs.items():
            chunks = [text]
            if args.sentence_chunks:
                chunks, position = [], 0
                for boundary in re.finditer(r'[.!?]\s+', text):
                    # Numbered headings are attached to their sentence.
                    if re.search(r'\b\d+\.$', text[:boundary.start()+1]):
                        continue
                    chunks.append(text[position:boundary.start()+1].strip())
                    position = boundary.end()
                if text[position:].strip():
                    chunks.append(text[position:].strip())
            batches = []
            for chunk in chunks:
                tokens = encoder.encode(chunk, out_type=str)
                if not config.get('add_source_eos', False):
                    tokens.append(config.get('eos_token', '</s>'))
                if len(tokens) > 512:
                    raise ValueError('Source exceeds 512 tokens; refusing truncation: ' + key)
                batches.append(tokens)
            tick = time.perf_counter()
            predictions = model.translate_batch(batches, beam_size=4, max_decoding_length=512)
            hypotheses = [item.hypotheses[0] for item in predictions]
            if any(len(hypothesis) >= 512 for hypothesis in hypotheses):
                raise ValueError('Translation reached decoding limit; not a complete response: ' + key)
            spanish = '\n'.join(decoder.decode(hypothesis) for hypothesis in hypotheses)
            results[key] = {'actual_reconstruction': text, 'actual_translation': spanish,
                            'source_tokens': sum(map(len,batches)), 'source_chunks': chunks,
                            'seconds': round(time.perf_counter() - tick, 6),
                            'translation_score': translation_score(cases[key], spanish),
                            'production_guard_validation': 'NOT_EVALUATED',
                            'raw_response_received': True}
    finally:
        peak[0] = max(peak[0], process.memory_info().rss)
        stopped.set()
        sampler.join(timeout=1)
    report = {
        'status': 'RAW_TRANSLATION_MEASURED', 'text_only': True, 'cpu_only': True,
        'architecture': 'Marian EN-to-ES via CTranslate2 int8; no reconstruction included',
        'sentence_chunks': args.sentence_chunks,
        'dataset_sha256': file_hash(args.dataset),
        'holdout_sha256': file_hash(args.holdout) if args.holdout else None,
        'source_report_sha256': file_hash(args.translation_input_report),
        'source_model': args.translation_input_model,
        'model_inventory': {name: file_hash(args.model_dir / name) for name in required},
        'runtime': {'ctranslate2': ctranslate2.__version__, 'sentencepiece': sentencepiece.__version__},
        'threads': args.threads, 'beam_size': 4, 'initialization_seconds': round(initialization, 3),
        'seconds': round(time.perf_counter() - started, 3), 'peak_process_rss_bytes': peak[0],
        'translation_seconds': round(sum(item['seconds'] for item in results.values()), 3),
        'memory_basis': 'Sampled RSS during initialization/translation; excludes remote Ollama, downloads and later provenance hashing.',
        'slide_times': time_summary(item['seconds'] for item in results.values()),
        'raw_responses': len(results), 'production_delivered_slides': 'NOT_EVALUATED',
        'translation_clause_checks_passed': sum(item['translation_score']['clause_checks_passed'] for item in results.values()),
        'translation_clause_checks_total': sum(item['translation_score']['clause_checks_total'] for item in results.values()),
        'semantic_human_review': 'PENDING', 'cases': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'seconds', 'peak_process_rss_bytes',
                                                 'raw_responses', 'translation_clause_checks_passed',
                                                 'translation_clause_checks_total')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
