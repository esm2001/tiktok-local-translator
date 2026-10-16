"""Translate the existing OCR text, without a generative reconstruction gate.

The local EN/ES model is reused between posts. A failed or incomplete translation
goes to the next engine; the final result always retains the source text.
"""

from __future__ import annotations

from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import re
import time

import requests

from src.config import Settings
from src.translation.ollama import (
    ENGLISH_FUNCTION_WORDS, _format_english_layout, _names, _ocr_evidence, _quoted_spans, _tokens, _validate_spanish, clean_ocr_lines,
)

LOGGER = logging.getLogger('translator')
MODEL_NAME = 'Helsinki-NLP/opus-mt-en-es (CTranslate2 int8)'
TRANSLATION_VERSION = 'dedicated-fallback-v4'
MODEL_FILES = ('config.json', 'model.bin', 'shared_vocabulary.json', 'source.spm', 'target.spm')


@lru_cache(maxsize=2)
def _load_local_model(directory: str, threads: int):
    """Keep model weights and tokenizers loaded while the server stays alive."""
    import ctranslate2
    import sentencepiece

    root = Path(directory)
    if any(not (root / name).is_file() for name in MODEL_FILES):
        raise RuntimeError(f'Modelo EN→ES local incompleto: {root}')
    encoder = sentencepiece.SentencePieceProcessor(str(root / 'source.spm'))
    decoder = sentencepiece.SentencePieceProcessor(str(root / 'target.spm'))
    model = ctranslate2.Translator(str(root), device='cpu', compute_type='int8',
                                  inter_threads=1, intra_threads=threads)
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    return model, encoder, decoder, config


def _model_directory() -> Path:
    override = os.getenv('TRANSLATION_MODEL_DIR', '').strip()
    if override:
        return Path(override)
    project = Path(__file__).resolve().parents[2]
    candidates = [project / 'models/opus-mt-en-es-ct2',
                  project.parent / '.cache/models/opus-mt-en-es-ct2/opus-mt-en-es-ct2',
                  Path('/root/.cache/translation/opus-mt-en-es-ct2')]
    return next((root for root in candidates if (root / 'model.bin').is_file()), candidates[0])


def _source_text(lines: list) -> tuple[str, str, list[str]]:
    # Keep the existing OCR geometry ordering and readable text. Confidence
    # scores remain evidence, and do not prohibit translating recognized words.
    ordered = _ocr_evidence(lines)
    texts = clean_ocr_lines(ordered)
    detected = '\n'.join(texts)
    source = _format_english_layout(detected, texts) if texts else ''
    return detected, source, texts


def _validate_output(key: str, lines: list[str], source: str, translated, allow_push_calque=False) -> str:
    # An initial capitalized label/acronym can be a proper name: untranslated
    # ANAAN on an artwork was rejected as if it were an English verb. Preserve
    # such labels only when the candidate contains the exact same spelling;
    # the remainder still passes the existing content and language checks.
    if not isinstance(translated, str):
        return _validate_spanish(key, lines, source, translated)
    literal_free_source = list(source)
    for start, end, _ in _quoted_spans(source):
        literal_free_source[start:end] = ' ' * (end - start)
    relationship = re.search(
        r'\bpush(?:es|ed|ing)?\s+(?:everyone|everybody|people|someone|somebody|anyone|anybody)\s+away\b',
        ''.join(literal_free_source), re.I)
    target_words = ' '.join(_tokens(translated))
    if (not allow_push_calque and relationship and re.search(r'\bempuj\w*\b', target_words)
            and not re.search(r'\b(?:alej|apart|rechaz|distanci)\w*\b', target_words)):
        raise RuntimeError('PUSH_PEOPLE_AWAY: rechazar o distanciar emocionalmente a otras personas '
                           'no significa empujarlas físicamente; conserva la relación interpersonal.')
    for name in _names(lines):
        source_label = re.search(r'(?m)^' + re.escape(name) + r'[.:]?\s*$', source)
        target_label = re.search(r'(?m)^' + re.escape(name) + r'[.:]?\s*$', translated)
        if not source_label or not target_label:
            continue
        source_caption = source[source_label.end():]
        if set(_tokens(source_caption)).intersection({'i', 'we', 'me', 'us', 'my', 'our', 'you'}):
            continue
        caption = list(translated[target_label.end():])
        for start, end, _ in _quoted_spans(''.join(caption)):
            caption[start:end] = ' ' * (end - start)
        for match in re.finditer(r'(?:^|[.!?;\n])\s*([a-záéíóúüñ]+[éí])\b', ''.join(caption), re.I):
            word = match.group(1).casefold()
            if word not in {'aquí', 'allí', 'ahí', 'así', 'sí', 'café', 'té', 'bebé', 'rubí', 'esquí', 'cliché', 'comité', 'puré', 'paté'}:
                raise RuntimeError(f'NAMED_CAPTION_PERSON: {word} está en primera persona; el pie se refiere a {name} en tercera persona.')
    validation_source, validation_target = source, translated
    validation_lines = list(lines)
    ordinary = ENGLISH_FUNCTION_WORDS | {'no', 'not', 'never', 'nothing', 'none',
                                       'nobody', 'without', 'cannot'}
    labels = []
    for line in lines:
        match = re.match(r'^([A-Z]{3,})(?=\s|[.:]|$)', line.strip())
        if match and match.group(1).casefold() not in ordinary and match.group(1) in translated:
            labels.append(match.group(1))
    for position, label in enumerate(dict.fromkeys(labels)):
        marker = f'__KEEP_EN_{position}__'
        pattern = r'(?<!\w)' + re.escape(label) + r'(?!\w)'
        validation_source = re.sub(pattern, marker, validation_source)
        validation_target = re.sub(pattern, marker, validation_target)
        validation_lines = [re.sub(pattern, marker, line) for line in validation_lines]
    _validate_spanish(key, validation_lines, validation_source, validation_target)
    return translated.strip()


def _caption_context(source: str, lines: list[str]) -> str:
    """Supply the OCR name as subject of an elliptical past-tense caption.

    This is translation context, not an OCR correction. For example a name
    label followed by 'Wrote about music.' refers to that named person, not to
    the translator. Keep the unaltered detected text in the result/manifest.
    """
    names = set(_names(lines))
    paragraphs = source.splitlines()
    irregular_past = {'wrote', 'went', 'made', 'knew', 'thought', 'said', 'saw',
                      'felt', 'found', 'gave', 'took', 'died', 'became', 'was'}
    index = 0
    while index < len(paragraphs) - 1:
        paragraph = paragraphs[index]
        name = paragraph.strip().rstrip('.:')
        if name not in names:
            index += 1
            continue
        caption = paragraphs[index + 1].lstrip()
        first = re.match(r'^[A-Za-z]+\b', caption)
        if not first:
            index += 1
            continue
        word = first.group(0)
        if word.casefold().endswith('ed') or word.casefold() in irregular_past:
            end = index + 2
            while end < len(paragraphs) and paragraphs[end].strip().rstrip('.:') not in names:
                caption += ' ' + paragraphs[end].strip()
                end += 1
            paragraphs[index + 1] = name + ' ' + word.lower() + caption[first.end():]
            del paragraphs[index + 2:end]
        index += 1
    return '\n'.join(paragraphs)


def _remove_repeated_labels(translated, names: list[str]):
    if not isinstance(translated, str):
        return translated
    for name in names:
        # Restore the separate name/caption layout when contextual translation
        # repeats the same label as a grammatical subject immediately below it.
        pattern = (r'(?m)^(' + re.escape(name) + r'[.:]?\s*\n)'
                   + re.escape(name) + r'[,.:]?\s+')
        translated = re.sub(pattern, lambda match: match.group(1), translated)
    translated = '\n'.join(part[:1].upper() + part[1:] if part and part[0].islower() else part
                           for part in translated.splitlines())
    return translated


def _push_paragraph_repair(key: str, lines: list[str], source: str, request: str, primary: str):
    """Reuse validated Marian paragraphs only for the specific push-away calque."""
    try:
        _validate_output(key, lines, source, primary, allow_push_calque=True)
    except RuntimeError:
        return None
    requested, translated = request.splitlines(), primary.splitlines()
    if len(requested) < 2 or len(requested) != len(translated):
        return None
    failed = []
    for index, (original_part, spanish_part) in enumerate(zip(requested, translated)):
        try:
            _validate_output(key, [original_part], original_part, spanish_part)
        except RuntimeError as exc:
            if str(exc).startswith('PUSH_PEOPLE_AWAY:'):
                failed.append(index)
    if len(failed) != 1:
        return None
    index = failed[0]
    return {'index': index, 'request': requested[index], 'paragraphs': translated}


class TextTranslator:
    """Dedicated local translation → Google best effort → Ollama → source."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.metrics = {}
        self._deadline = None

    def _remaining_budget(self) -> float:
        if self._deadline is None:
            return float(self.settings.translation_budget_seconds)
        return self._deadline - time.perf_counter()

    def _request_timeout(self, connect: float, read: float) -> tuple[float, float]:
        remaining = self._remaining_budget()
        if remaining <= 0:
            self.metrics['budget_exhausted'] = True
            raise RuntimeError('Se agotó el presupuesto global de traducción; se conserva el texto original.')
        return min(connect, remaining), min(read, remaining)

    def _local(self, source: str, session) -> str:
        model, encoder, decoder, config = _load_local_model(
            str(_model_directory()), self.settings.cpu_threads)
        # Wrapped OCR clauses are already joined. Preserve paragraph/name
        # boundaries rather than asking the translation model to invent layout.
        chunks = [part.strip() for part in source.splitlines() if part.strip()]
        token_batches = []
        for chunk in chunks:
            tokens = encoder.encode(chunk, out_type=str)
            if not config.get('add_source_eos', False):
                tokens.append(config.get('eos_token', '</s>'))
            if len(tokens) > 512:
                raise RuntimeError('El texto supera el contexto local; se probará el siguiente traductor sin truncarlo.')
            token_batches.append(tokens)
        predictions = model.translate_batch(token_batches, beam_size=4, max_decoding_length=512)
        if len(predictions) != len(token_batches):
            raise RuntimeError('El modelo local no devolvió todos los párrafos.')
        translated = []
        for result in predictions:
            if not result.hypotheses or not result.hypotheses[0] or len(result.hypotheses[0]) >= 512:
                raise RuntimeError('El modelo local dejó un párrafo vacío o truncado.')
            translated.append(decoder.decode(result.hypotheses[0]))
        return '\n'.join(translated)

    def _google(self, source: str, session) -> str:
        if os.getenv('GOOGLE_TRANSLATE_FALLBACK', 'true').lower() not in {'true', '1', 'yes'}:
            raise RuntimeError('Fallback Google desactivado.')
        # Public Translate web endpoint, no API key. This is an opportunistic
        # fallback, never a dependency for the primary offline translator.
        # Send a complete slide whenever possible, so named captions retain
        # their subject and tense. Only oversized sources need several calls.
        chunks = [source] if len(source) <= 3500 else []
        if not chunks:
            current = ''
            for sentence in re.split(r'(?<=[.!?])\s+', source):
                if current and len(current) + len(sentence) > 3500:
                    chunks.append(current)
                    current = ''
                if len(sentence) > 3500:
                    chunks.extend(sentence[index:index + 3500] for index in range(0, len(sentence), 3500))
                else:
                    current = (current + ' ' + sentence).strip()
            if current:
                chunks.append(current)
        translated = []
        for chunk in chunks:
            response = session.get('https://translate.googleapis.com/translate_a/single',
                                   params={'client': 'gtx', 'sl': 'en', 'tl': 'es', 'dt': 't', 'q': chunk},
                                   headers={'User-Agent': 'Mozilla/5.0'}, timeout=self._request_timeout(5, 12))
            response.raise_for_status()
            data = response.json()
            LOGGER.info('[TRANSLATE] google response=%s', json.dumps(data, ensure_ascii=False))
            if not isinstance(data, list) or not data or not isinstance(data[0], list):
                raise RuntimeError('Google devolvió una estructura de traducción no válida.')
            pieces = [item[0] for item in data[0] if isinstance(item, list) and item
                      and isinstance(item[0], str)]
            value = ''.join(pieces).strip()
            if not value:
                raise RuntimeError('Google dejó una parte de la traducción vacía.')
            translated.append(value)
        return '\n'.join(translated)

    def _ollama(self, source: str, session, retry_error: str = '') -> str:
        instruction = ('Translate this English text to Spanish from Spain. '
                       'Correct only obvious OCR spelling errors. Do not add information. '
                       'Preserve names, subjects, negations and every caption. '
                       'A caption below a person\'s name describes that person in the third person. '
                       'Write about a topic means producing text on that topic, not creating the object. '
                       'Return only the translated text.')
        if retry_error:
            instruction += '\nRepair the previous meaning/completeness error: ' + retry_error[:450]
        response = session.post(f'{self.settings.ollama_url}/api/chat', json={
            'model': self.settings.translation_model,
            'messages': [
                {'role': 'system', 'content': instruction},
                {'role': 'user', 'content': source}],
            'stream': False, 'keep_alive': self.settings.ollama_keep_alive,
            'options': {'temperature': 0, 'num_ctx': 4096,
                        'num_predict': min(2800, max(400, len(source))), 'num_gpu': 0}},
            timeout=self._request_timeout(10, min(90, self.settings.translation_timeout_seconds)))
        response.raise_for_status()
        data = response.json()
        LOGGER.info('[TRANSLATE] ollama response=%s', json.dumps(data, ensure_ascii=False))
        if not isinstance(data, dict) or data.get('error') or data.get('done') is False or data.get('done_reason') == 'length':
            raise RuntimeError('Ollama no completó la traducción.')
        message = data.get('message', {})
        if not isinstance(message, dict) or not isinstance(message.get('content'), str):
            raise RuntimeError('Ollama devolvió contenido no válido.')
        return message['content'].strip()

    def translate(self, slides: dict[int, list]) -> dict[str, dict]:
        started = time.perf_counter()
        self._deadline = started + self.settings.translation_budget_seconds
        self.metrics = {'calls': [], 'request_count': 0, 'retry_count': 0, 'http_seconds': 0.0,
                        'fallback_slides': [], 'translation_engines': {},
                        'architecture': 'existing OCR → dedicated EN/ES → Google → Ollama → English'}
        output = {}
        engines = [(MODEL_NAME, self._local), ('google_translate_web', self._google),
                   (f'ollama:{self.settings.translation_model}', self._ollama)]
        unavailable = set()
        with requests.Session() as session:
            for index, lines in slides.items():
                key = str(index)
                detected, source, texts = _source_text(lines)
                request_text = _caption_context(source, texts)
                LOGGER.info('[TRANSLATE] slide=%s OCR=%r request_text=%r', key, detected, request_text)
                errors, spanish, selected = [], '', 'original_english'
                fallback_used = False
                paragraph_repair, paragraph_engines = None, {}
                if source:
                    for position, (name, translate) in enumerate(engines):
                        if self._remaining_budget() <= 0:
                            self.metrics['budget_exhausted'] = True
                            errors.append({'engine': name, 'error': 'Se agotó el presupuesto global de traducción.'})
                            break
                        if name in unavailable:
                            errors.append({'engine': name, 'error': 'Motor no disponible en este proceso.'})
                            continue
                        retry_error = ''
                        engine_source = paragraph_repair['request'] if paragraph_repair and position else request_text
                        for attempt in range(2 if name.startswith('ollama:') else 1):
                            if self._remaining_budget() <= 0:
                                self.metrics['budget_exhausted'] = True
                                errors.append({'engine': name, 'error': 'Se agotó el presupuesto global de traducción.'})
                                break
                            call = {'slide': key, 'stage': 'translation', 'engine': name,
                                    'source': engine_source, 'retry': bool(attempt)}
                            if paragraph_repair and position:
                                call['paragraph'] = paragraph_repair['index'] + 1
                            self.metrics['calls'].append(call)
                            tick, raw = time.perf_counter(), None
                            try:
                                raw = self._ollama(engine_source, session, retry_error) if attempt else translate(engine_source, session)
                                call['raw_response'] = raw
                                LOGGER.info('[TRANSLATE] slide=%s engine=%s response=%r', key, name, raw)
                                normalized = _remove_repeated_labels(raw, _names(texts))
                                if paragraph_repair and position and isinstance(normalized, str):
                                    repaired = list(paragraph_repair['paragraphs'])
                                    repaired[paragraph_repair['index']] = normalized
                                    normalized = '\n'.join(repaired)
                                if re.search(r'\b(?:write|writes|wrote|written|writing)\s+about\b', source, re.I) and not re.search(r'\b(?:escrib|escrit|redact)\w*\b', normalized, re.I):
                                    raise RuntimeError('El traductor omitió la frase write about presente en el OCR.')
                                spanish = _validate_output(key, texts, source, normalized)
                                call['parsed_text'] = spanish
                                selected, fallback_used = name, position > 0
                                if paragraph_repair and position:
                                    selected = MODEL_NAME + ' + ' + name
                                    paragraph_engines = {str(i + 1): name if i == paragraph_repair['index'] else MODEL_NAME
                                                         for i in range(len(paragraph_repair['paragraphs']))}
                                LOGGER.info('[TRANSLATE] slide=%s parsed=%r', key, spanish)
                                break
                            except Exception as exc:
                                spanish = ''
                                message = str(exc) or type(exc).__name__
                                call['error'] = message
                                errors.append({'engine': name, 'error': message})
                                LOGGER.warning('[TRANSLATE] slide=%s engine=%s failed=%s', key, name, message)
                                if name == MODEL_NAME and message.startswith('PUSH_PEOPLE_AWAY:') and isinstance(raw, str):
                                    paragraph_repair = _push_paragraph_repair(key, texts, source, request_text, normalized)
                                if isinstance(exc, (ImportError, ModuleNotFoundError, FileNotFoundError,
                                                    requests.ConnectionError, requests.Timeout)):
                                    unavailable.add(name)
                                elif isinstance(exc, requests.HTTPError) and getattr(exc.response, 'status_code', None) in {403, 429}:
                                    unavailable.add(name)
                                if not isinstance(raw, str) or not raw.strip() or not isinstance(exc, RuntimeError):
                                    break
                                retry_error = message
                            finally:
                                call['http_seconds'] = round(time.perf_counter() - tick, 4)
                                self.metrics['http_seconds'] += call['http_seconds']
                        if spanish:
                            break
                if not spanish:
                    fallback_used = bool(source)
                    spanish = ('[No se pudo traducir automáticamente]\n' + source
                               if source else '[Sin texto detectado en esta diapositiva]')
                english_fallback = selected == 'original_english' and bool(source)
                status = 'ENGLISH_FALLBACK' if english_fallback else 'TRANSLATED' if source else 'NO_TEXT'
                translation = {'text': spanish, 'spanish': spanish, 'engine': selected,
                               'fallback_used': fallback_used, 'fallback_to_english': english_fallback,
                               'status': status, 'failure': errors[-1] if english_fallback and errors else None,
                               'integrity_validated': bool(source) and not english_fallback,
                               'semantic_equivalence_verified': False, 'errors': errors}
                if paragraph_engines:
                    translation['paragraph_engines'] = paragraph_engines
                output[key] = {'original': source, 'ocr_text': detected, 'spanish': spanish,
                               'engine': selected, 'fallback_used': fallback_used,
                               'fallback_to_english': english_fallback, 'retry_recommended': english_fallback,
                               'warnings': [item['error'] for item in errors],
                               'uncertain_fragments': [], 'translation': translation,
                               'reconstruction': {'reconstructed_english': source, 'status': 'OCR_MINIMAL_REFLOW',
                                                  'uncertain_fragments': [], 'audit': {'accepted': True, 'character_edits': 0}}}
                self.metrics['translation_engines'][key] = selected
                if fallback_used:
                    self.metrics['fallback_slides'].append(key)
                LOGGER.info('[TRANSLATE] slide=%s result=%s', key, json.dumps(output[key], ensure_ascii=False))
        self.metrics.update(request_count=len(self.metrics['calls']),
                            retry_count=sum(call['retry'] for call in self.metrics['calls']),
                            seconds=round(time.perf_counter() - started, 3))
        return output
