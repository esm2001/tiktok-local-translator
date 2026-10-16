import copy
import json
import logging
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import cv2

from src.config import Settings
from src.image.cleaning import read_image

_ocr = None
_lock = threading.RLock()
logger = logging.getLogger('translator')
OCR_REFINEMENT_VERSION = 'adaptive-gray-native-v4'
OCR_ENGINE_VERSION = 'PP-OCRv5_mobile_det+en_PP-OCRv5_mobile_rec'
OCR_REFINEMENT_THRESHOLD = .93


def get_ocr(settings: Settings):
    global _ocr
    with _lock:
        if _ocr is None:
            from paddleocr import PaddleOCR
            _ocr = PaddleOCR(lang='en', ocr_version='PP-OCRv5',
                text_detection_model_name='PP-OCRv5_mobile_det',
                text_recognition_model_name='en_PP-OCRv5_mobile_rec',
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, device='cpu',
                enable_mkldnn=True, cpu_threads=settings.cpu_threads)
        return _ocr


def _payload(result) -> dict:
    value = getattr(result, 'json', result)
    if callable(value):
        value = value()
    if isinstance(value, str):
        value = json.loads(value)
    return value.get('res', value) if isinstance(value, dict) else {}


def _list(value):
    return [] if value is None else list(value)


def reading_order(lines: list[dict]) -> list[dict]:
    positioned, unpositioned = [], []
    for line in lines:
        points = np.asarray(line.get('polygon', []))
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 2:
            positioned.append((float(points[:, 1].mean()), float(points[:, 0].min()),
                               float(points[:, 1].max()-points[:, 1].min()), line))
        else:
            unpositioned.append(line)
    rows=[]
    for item in sorted(positioned, key=lambda p:(p[0],p[1])):
        if rows and abs(item[0]-rows[-1][0][0]) <= min(item[2],rows[-1][0][2])*.4:
            rows[-1].append(item)
        else:
            rows.append([item])
    return [item[3] for row in rows for item in sorted(row,key=lambda p:p[1])] + unpositioned


def parse_results(results) -> dict:
    polygons, lines, seen_polys, seen_lines = [], [], set(), set()
    for result in results:
        data = _payload(result)
        det = _list(data.get('dt_polys')) or _list(data.get('rec_polys'))
        rec_polys = _list(data.get('rec_polys'))
        texts = _list(data.get('rec_texts'))
        scores = _list(data.get('rec_scores'))
        for poly in det:
            points = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in np.asarray(poly).tolist() if len(p) >= 2]
            key = tuple(tuple(p) for p in points)
            if len(points) >= 4 and key not in seen_polys:
                seen_polys.add(key)
                polygons.append(points)
        for index, text in enumerate(texts):
            text = str(text or '').strip()
            score = float(scores[index]) if index < len(scores) else 1.0
            polygon = np.asarray(rec_polys[index]).astype(int).tolist() if index < len(rec_polys) else []
            # Repeated text in two locations is evidence for both regions.
            key = (text, tuple(map(tuple, polygon)))
            if text and score >= .40 and key not in seen_lines:
                seen_lines.add(key)
                line = {'text': text, 'confidence': round(score, 3)}
                if index < len(rec_polys):
                    line['polygon'] = polygon
                lines.append(line)
    return {'detected_polygons': polygons, 'recognized_lines': reading_order(lines)}


def _normalised_text(text: str) -> str:
    return ''.join(character for character in text.casefold() if character.isalnum())


def refine_recognized_lines(path: Path, lines: list[dict], settings: Settings) -> list[dict]:
    """Retry a low-confidence line once, preserving full-image detector geometry.

    Native crops correct the supplied slide 07 question and its final full stop.
    Six retries below .93 took roughly .93 seconds on CPU; upscaling split the
    question and is deliberately avoided. Only one reliable recognition can win:
    unrecognized detector fragments do not become additional readable lines.
    Cached lines can use this helper without repeating full-image OCR. Originals
    are never mutated, and an optional retry failure retains the initial reading.
    """
    output = copy.deepcopy(lines)
    image = None
    retries = 0
    for line in output:
        try:
            confidence = float(line.get('confidence', 1.0))
            points = np.asarray(line.get('polygon', []), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        text = str(line.get('text', ''))
        normalised = _normalised_text(text)
        if not np.isfinite(confidence) or confidence >= OCR_REFINEMENT_THRESHOLD or not normalised:
            continue
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2 or not np.isfinite(points).all():
            continue
        if retries >= getattr(settings, 'ocr_max_crop_retries', 6):
            line['refinement_skipped'] = 'CROP_BUDGET_EXHAUSTED'
            continue
        if image is None:
            try:
                image = read_image(Path(path))
            except RuntimeError as exc:
                logger.warning('[ocr] No se pudo repetir el recorte de %s: %s', Path(path).name, exc)
                return output
        height, width = image.shape[:2]
        x0 = max(0, int(np.floor(points[:, 0].min())) - 10)
        y0 = max(0, int(np.floor(points[:, 1].min())) - 10)
        x1 = min(width, int(np.ceil(points[:, 0].max())) + 10)
        y1 = min(height, int(np.ceil(points[:, 1].max())) + 10)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = image[y0:y1, x0:x1].copy()
        started = time.perf_counter()
        retries += 1
        try:
            with _lock:
                retry = parse_results(list(get_ocr(settings).predict(crop)))
        except Exception as exc:
            logger.warning('[ocr] Falló el reintento de %s; se conserva el OCR inicial: %s', Path(path).name, exc)
            continue
        # Extra fragments cannot silently replace a complete initial reading.
        if len(retry['recognized_lines']) != 1:
            continue
        candidate = retry['recognized_lines'][0]
        new_confidence = float(candidate['confidence'])
        candidate_text = str(candidate['text']).strip()
        candidate_normalised = _normalised_text(candidate_text)
        similarity = SequenceMatcher(None, normalised, candidate_normalised, autojunk=False).ratio()
        length_ratio = len(candidate_normalised) / len(normalised)
        if new_confidence - confidence < .02 - 1e-9 or similarity < .80 or length_ratio < .85:
            continue
        line['refinement'] = {
            'version': OCR_REFINEMENT_VERSION, 'previous_text': text,
            'previous_confidence': confidence, 'text_similarity': round(similarity, 3),
            'length_ratio': round(length_ratio, 3),
            'seconds': round(time.perf_counter() - started, 3),
        }
        line['text'] = candidate_text
        line['confidence'] = new_confidence
        logger.info('[ocr] Recorte %s: confianza %.3f -> %.3f', Path(path).name, confidence, new_confidence)
    return output


def coverage_signals(parsed: dict, shape) -> list[str]:
    from src.ocr.editorial import editorial_lines, bounds
    lines,_=editorial_lines(parsed['recognized_lines'],shape)
    if not lines:
        return ['NO_READABLE_TEXT']
    confidences=[float(line.get('confidence',0)) for line in lines]
    signals=[]
    if min(confidences)<.80 or sum(confidences)/len(confidences)<.90:
        signals.append('LOW_RECOGNITION_CONFIDENCE')
    if len(lines)==1 and min(confidences)<.90 and len(_normalised_text(lines[0].get('text','')))<8:
        signals.append('SPARSE_RECOGNITION')
    known={tuple(map(tuple,line['polygon'])) for line in parsed['recognized_lines'] if line.get('polygon')}
    for polygon in parsed['detected_polygons'] if known else ():
        if tuple(map(tuple,polygon)) in known: continue
        box=bounds(polygon)
        if not box: continue
        dx=polygon[1][0]-polygon[0][0];dy=polygon[1][1]-polygon[0][1]
        angle=abs(float(np.degrees(np.arctan2(dy,dx))));angle=min(angle,abs(180-angle))
        if angle<=20 and box[2]-box[0]>=2.5*max(1,box[3]-box[1]) and box[3]-box[1]>=shape[0]*.025:
            signals.append('UNRECOGNISED_HORIZONTAL_REGION');break
    return signals


def _merge_fallback(primary: dict, fallback: dict) -> dict:
    from src.ocr.editorial import bounds
    result=copy.deepcopy(primary)
    added=[]
    for line in fallback['recognized_lines']:
        box=bounds(line.get('polygon',[]))
        if not box or float(line.get('confidence',0))<.70: continue
        closest=None;overlap=0
        for old in result['recognized_lines']:
            other=bounds(old.get('polygon',[]))
            if not other: continue
            intersection=max(0,min(box[2],other[2])-max(box[0],other[0]))*max(0,min(box[3],other[3])-max(box[1],other[1]))
            area=min((box[2]-box[0])*(box[3]-box[1]),(other[2]-other[0])*(other[3]-other[1]))
            ratio=intersection/max(1,area)
            if ratio>overlap: closest,overlap=old,ratio
        if overlap>=.5 and closest is not None:
            oldtext,newtext=_normalised_text(closest['text']),_normalised_text(line['text'])
            similarity=SequenceMatcher(None,oldtext,newtext,autojunk=False).ratio()
            if float(line['confidence'])>=float(closest['confidence'])+.02 and similarity>=.8 and len(newtext)>=len(oldtext)*.85:
                closest['fallback_reading']={'previous_text':closest['text'],'engine':OCR_ENGINE_VERSION+'+grayscale','similarity':round(similarity,3)}
                closest.update(text=line['text'],confidence=line['confidence'])
        elif overlap<.25:
            added.append({**line,'fallback_reading':{'engine':OCR_ENGINE_VERSION+'+grayscale','new_region':True}})
            result['detected_polygons'].append(line['polygon'])
    result['recognized_lines']=reading_order(result['recognized_lines']+added)
    return result


def ocr_image(path, settings: Settings) -> dict:
    started=time.perf_counter()
    with _lock:
        if Path(path).suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}:
            source = str(path)
        else:
            # Paddle's image sampler silently ignores unknown suffixes such as
            # downloaded .img/.bin files; an ndarray avoids that false empty OCR.
            source = read_image(Path(path))
        results = list(get_ocr(settings).predict(source))
        if not results:
            raise RuntimeError(f'PaddleOCR no ha procesado la imagen {Path(path).name}.')
    parsed = parse_results(results)
    primary_region_count = len(parsed['detected_polygons'])
    primary_seconds=time.perf_counter()-started
    primary_lines=copy.deepcopy(parsed['recognized_lines'])
    image=read_image(Path(path))
    from src.ocr.editorial import editorial_lines
    editorial,_=editorial_lines(parsed['recognized_lines'],image.shape)
    t=time.perf_counter()
    refined=refine_recognized_lines(Path(path),editorial,settings)
    replacements={tuple(map(tuple,line.get('polygon',[]))):line for line in refined if line.get('polygon')}
    if len(editorial)==len(parsed['recognized_lines']):
        parsed['recognized_lines']=refined
    else:
        parsed['recognized_lines']=[replacements.get(tuple(map(tuple,line.get('polygon',[]))),line) for line in parsed['recognized_lines']]
    crop_seconds=time.perf_counter()-t
    signals=coverage_signals(parsed,image.shape)
    fallback_seconds=0;error=None
    if signals and getattr(settings,'ocr_adaptive_fallback',True):
        t=time.perf_counter()
        try:
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            with _lock:
                fallback=parse_results(list(get_ocr(settings).predict(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR))))
            parsed=_merge_fallback(parsed,fallback)
        except Exception as exc:
            error=str(exc);logger.warning('[ocr] Fallback falló en %s: %s',Path(path).name,exc)
        fallback_seconds=time.perf_counter()-t
    parsed['primary_lines']=primary_lines
    parsed['diagnostics']={'primary_engine':OCR_ENGINE_VERSION,'fallback_engine':OCR_ENGINE_VERSION+'+grayscale' if fallback_seconds else None,
         'fallback_used':bool(fallback_seconds),'fallback_signals':signals,'fallback_error':error,
         'primary_regions':primary_region_count,
         'detected_regions':len(parsed['detected_polygons']),'recognized_regions':len(parsed['recognized_lines']),
         'unresolved_signals':coverage_signals(parsed,image.shape),
         'timings':{'ocr_primary':round(primary_seconds,3),'ocr_fallback':round(fallback_seconds+crop_seconds,3),'ocr_crops':round(crop_seconds,3)}}
    return parsed
