import copy
import hashlib
import logging
import math
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

from PIL import Image

from src.config import SCHEMA_VERSION, Settings
from src.image.cleaning import make_clean_image, read_image
from src.ocr.paddle import OCR_ENGINE_VERSION, OCR_REFINEMENT_VERSION, ocr_image
from src.ocr.editorial import overlay_polygons
from src.output.manifest import read_json, write_json, write_translations_txt
from src.recovery import RECOVERY_VERSION, QUERY_STRATEGY_VERSION, REVERSE_QUERY_KINDS, prepare_reverse_query, recover_slide, revalidate_saved_source
from src.tiktok.downloader import download_images, extract_post_id, resolve_tiktok_url, validate_tiktok_url
from src.translation.ollama import clean_ocr_lines
from src.translation.text import TRANSLATION_VERSION, TextTranslator as OllamaTranslator

logger = logging.getLogger('translator')
_pipeline_lock = threading.Lock()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pipeline_signature(settings: Settings) -> dict:
    return {'schema': SCHEMA_VERSION, 'ocr': OCR_ENGINE_VERSION, 'ocr_refinement': OCR_REFINEMENT_VERSION, 'model': settings.ollama_model,
        'cleaning': 'TELEA-editorial-mask-v4', 'translation': TRANSLATION_VERSION, 'translation_model': settings.translation_model, 'threshold': settings.match_threshold,
        'minimum_resolution': settings.candidate_min_dimension,'matcher':RECOVERY_VERSION,'query_strategy':QUERY_STRATEGY_VERSION,
        'max_queries':settings.reverse_max_queries,'ocr_adaptive':settings.ocr_adaptive_fallback,'ocr_crop_budget':settings.ocr_max_crop_retries,
        'text_timeout':settings.translation_timeout_seconds,'text_budget':settings.translation_budget_seconds,
        'reverse_budget':settings.reverse_total_budget}


def reusable_source(base: Path, prior: dict, current: dict, previous_signature: dict,
                    current_signature: dict, settings: Settings) -> dict | None:
    """Reuse an intact web download without modifying it or imposing a score gate."""
    recovery = prior.get('reverse_search', {})
    if (recovery.get('status') != 'MATCH' or recovery.get('raw_download') is not True
            or recovery.get('modified') is not False or recovery.get('source') != 'web_download'
            or not str(recovery.get('source_url') or recovery.get('image_url') or '').startswith(('https://', 'http://'))):
        return None
    for version in ('ocr', 'ocr_refinement', 'cleaning', 'matcher', 'query_strategy'):
        if not previous_signature.get(version) or previous_signature[version] != current_signature.get(version):
            return None
    for key in ('original_sha256', 'clean_sha256'):
        if not prior.get(key) or prior[key] != current.get(key):
            return None
    mask_hash = prior.get('ocr', {}).get('mask_sha256')
    if not mask_hash or mask_hash != current.get('ocr', {}).get('mask_sha256'):
        return None
    try:
        managed = str(recovery.get('downloaded', ''))
        if (Path(managed).parent.as_posix() != 'recovered'
                or Path(managed).stem != f"{int(current['position']):02d}_SOURCE"
                or Path(managed).suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.avif'}):
            return None
        source = base / managed
        # A symlink must not turn this managed cache into an external file.
        if source.resolve() != base.resolve() / managed or not source.is_file():
            return None
        if not recovery.get('sha256') or sha256(source) != recovery['sha256']:
            return None
        score = float(recovery.get('score', 0))
        if not math.isfinite(score):
            return None
        with Image.open(source) as image:
            image.load()
            width, height = image.size
        if recovery.get('resolution') != {'width': width, 'height': height}:
            return None
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return None
    cached = copy.deepcopy(recovery)
    cached['cache_hit'] = True
    return cached


def reverse_search_complete(slides: list[dict]) -> bool:
    return bool(slides) and all(
        slide.get('reverse_search', {}).get('status') == 'MATCH'
        and bool(slide['reverse_search'].get('downloaded'))
        and slide['reverse_search'].get('raw_download') is True
        and slide['reverse_search'].get('modified') is False
        for slide in slides)


def validate_outputs(base: Path, slides: list[dict]) -> dict:
    """Check exported files and each nonempty translation against the slide count."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.avif'}
    counts = {name + '_count': sum(p.is_file() and p.suffix.lower() in image_extensions
                                 for p in (base / name).glob('*'))
              for name in ('original', 'clean', 'recovered')}
    text_path = base / 'traducciones.txt'
    text = text_path.read_text(encoding='utf-8') if text_path.is_file() else ''
    import re
    sections = re.split(r'^DIAPOSITIVA (\d+)\s*$', text, flags=re.MULTILINE)
    translated = {int(sections[i]): sections[i + 1].split('TRADUCCIÓN:', 1)[-1].split('\nESTADO:', 1)[0].strip()
                  for i in range(1, len(sections), 2) if 'TRADUCCIÓN:' in sections[i + 1]}
    missing = []
    for slide in slides:
        index = slide['position']
        for label in ('original', 'clean'):
            if not slide.get(label) or not (base / slide[label]).is_file():
                missing.append(f'Diapositiva {index}: falta {label}')
        recovered = slide.get('reverse_search', {})
        if not recovered.get('downloaded') or not (base / recovered['downloaded']).is_file():
            missing.append(f"Diapositiva {index}: falta recovered ({recovered.get('error') or recovered.get('status', 'sin descarga')})")
        elif recovered.get('raw_download') is not True or recovered.get('modified') is not False:
            missing.append(f'Diapositiva {index}: recovered carece de procedencia RAW verificable')
        if not translated.get(index):
            missing.append(f'Diapositiva {index}: falta traducción en TXT')
    expected = len(slides)
    for name, count in counts.items():
        if count != expected:
            missing.append(f'{name}: {count}/{expected}')
    if len(translated) != expected:
        missing.append(f'translation_sections: {len(translated)}/{expected}')
    return {'expected': expected, **counts, 'translation_sections': len(translated),
            'missing': missing, 'complete': not missing}


def translation_retry_due(manifest: dict, settings: Settings) -> bool:
    if not any(item.get('retry_recommended') for item in manifest.get('translations', {}).values()):
        return False
    try:
        generated_at = float(manifest.get('translation_generated_at', 0))
        return not math.isfinite(generated_at) or time.time()-generated_at >= settings.translation_retry_ttl
    except (TypeError, ValueError):
        return True


def completed_cache(base: Path, manifest: dict | None, settings: Settings, reverse_search: bool) -> bool:
    if not manifest or manifest.get('status') != 'OK' or manifest.get('signature') != pipeline_signature(settings):
        return False
    if translation_retry_due(manifest, settings):
        return False
    if reverse_search and (not manifest.get('reverse_search_completed') or not reverse_search_complete(manifest.get('slides', []))):
        return False
    slides = manifest.get('slides', [])
    if not slides or not (base / 'traducciones.txt').is_file():
        return False
    if not validate_outputs(base, slides)['complete']:
        return False
    if manifest.get('translations_sha256') != sha256(base / 'traducciones.txt'):
        return False
    for slide in slides:
        recovery = slide.get('reverse_search', {})
        if reverse_search and recovery.get('status') != 'MATCH':
            try:
                searched_at = float(recovery.get('searched_at', 0))
                if not math.isfinite(searched_at) or time.time()-searched_at > settings.reverse_negative_ttl:
                    return False
            except (ValueError, TypeError):
                return False
        for path in [slide.get('original'), slide.get('clean'), slide.get('ocr', {}).get('mask')]:
            if not path or not (base / path).is_file():
                return False
        if slide.get('original_sha256') != sha256(base / slide['original']):
            return False
        if slide.get('clean_sha256') != sha256(base / slide['clean']):
            return False
        if slide['ocr'].get('mask_sha256') != sha256(base / slide['ocr']['mask']):
            return False
        recovered = slide.get('reverse_search', {}).get('downloaded')
        if recovered:
            if not (base / recovered).is_file() or slide['reverse_search'].get('sha256') != sha256(base / recovered):
                return False
        if slide.get('reverse_search', {}).get('status') == 'MATCH' and reusable_source(
                base, slide, slide, manifest['signature'], pipeline_signature(settings), settings) is None:
            return False
    return True


def result_for(base: Path, manifest: dict, cached=False) -> dict:
    return {'status': 'OK_CACHED' if cached else manifest.get('status', 'OK'), 'post_id': manifest['post_id'],
        'source_url': manifest['source_url'], 'resolved_url': manifest['resolved_url'],
        'media_type': manifest['media_type'], 'images_found': len(manifest.get('slides', [])),
        'clean_images_created': manifest.get('validation', {}).get('clean_count', len(manifest.get('slides', []))),
        'recovered_images': manifest.get('validation', {}).get('recovered_count', sum(s.get('reverse_search', {}).get('status') == 'MATCH' for s in manifest.get('slides', []))),
        'translation_sections': manifest.get('validation', {}).get('translation_sections', 0),
        'validation': manifest.get('validation', {}),
        'review_required': bool(manifest.get('review_required')),
        'translation_fallbacks': sum(s.get('translation', {}).get('fallback_to_english', False) for s in manifest.get('slides', [])),
        'reverse_pending': sum(s.get('reverse_search', {}).get('status') in {'CAPTCHA', 'ERROR', 'UNAVAILABLE', 'TIMEOUT', 'NEEDS_REVIEW'} for s in manifest.get('slides', [])),
        'output_dir': str(base), 'translations_file': str(base / 'traducciones.txt'),
        'manifest': str(base / 'metadata' / 'manifest.json'), 'timings': manifest.get('timings', {}),
        'cached': cached, 'error': '; '.join(manifest.get('validation', {}).get('missing', []))}


def recover_with_strategy(base, slides, settings, reverse_callback, report, source_cache=None, persist=None):
    """Validate each round before spending another query on unresolved slides."""
    source_cache=source_cache or {}; persist=persist or (lambda:None)
    pending=[]; evidence={}; seen={}
    for slide in slides:
        index=slide['position']
        if index in source_cache:
            slide['reverse_search']=source_cache[index];slide['best_image']=source_cache[index]['downloaded']
            cached_times = source_cache[index].get('timings', {}) if source_cache[index].get('cache_revalidated') else {}
            slide['timings']['reverse_search']=cached_times.get('cache_revalidation', 0)
            for name in ('candidate_matching', 'source_ocr', 'cache_revalidation'):
                slide['timings'][name]=cached_times.get(name, 0)
        else:
            pending.append(slide); evidence[index]=[];seen[index]=set()
    kinds=REVERSE_QUERY_KINDS[:settings.reverse_max_queries] if reverse_callback else ('clean_full',)
    for attempt_index,kind in enumerate(kinds,1):
        if not pending: break
        # Each search round gets its own bounded time. A slow first query must
        # not prevent ORIGINAL/crop/text fallbacks for the remaining slides.
        remaining=settings.reverse_total_budget
        batch=[]
        for slide in pending:
            index=slide['position']
            query=prepare_reverse_query(base,slide,kind,attempt_index)
            if query.get('skipped'):
                evidence[index].append({'query':query,'status':'SKIPPED','candidates':[], 'elapsed_seconds':0})
            else:
                query['budget_seconds_remaining']=round(remaining * len(pending),3)
                slide['reverse_query']=query;batch.append(slide)
        if not batch: continue
        if reverse_callback:
            report('lens',f'Búsqueda {attempt_index}/{len(kinds)}: {kind}, {len(batch)} diapositivas pendientes.')
            try:
                searches=reverse_callback(base,batch)
            except Exception as exc:
                logger.exception('El navegador no entregó candidatos; se probará la siguiente búsqueda.')
                searches={s['position']:{'provider':'google_lens','status':'ERROR','error':str(exc),'candidates':[]} for s in batch}
        else:
            searches={s['position']:{'provider':'google_lens','status':'DISABLED','candidates':[]} for s in batch}
        terminal=[]
        for slide in batch:
            index=slide['position']; tick=time.perf_counter()
            search=copy.deepcopy(searches.get(index,{'provider':'google_lens','status':'UNAVAILABLE','candidates':[],'error':'Missing launcher result.'}))
            search.setdefault('query',slide['reverse_query'])
            search['_validation_deadline']=time.monotonic()+settings.reverse_total_budget
            submitted=len(search.get('candidates',[]))
            search['candidates']=[c for c in search.get('candidates',[]) if c.get('image_url') not in seen[index]]
            seen[index].update(c.get('image_url') for c in search['candidates'])
            try:
                recovery=recover_slide(base,slide,search,settings)
            except Exception as exc:
                logger.exception('Falló recuperación de slide %s',index)
                recovery={'provider':'google_lens','status':'ERROR','error':str(exc),'downloaded':None,'fallback':slide['clean'],'score':0,'candidates':[]}
            elapsed=time.perf_counter()-tick+float(search.get('elapsed_seconds',0))
            attempt={'query':search['query'],'status':recovery['status'],'elapsed_seconds':round(elapsed,3),
                     'search_seconds':float(search.get('elapsed_seconds',0)),'candidates_submitted':submitted,
                     'candidates':copy.deepcopy(recovery.get('candidates',[])), 'error':recovery.get('error',''),
                     'timings':recovery.get('timings',{}),'cache_hit':bool(search.get('cache_hit',False)),
                     'diagnostics': search.get('diagnostics', {}),
                     'discovery': {key: search[key] for key in ('fresh_search_status', 'fresh_search_error',
                         'cache_fallback_from', 'discovery_query_file', 'discovery_query_sha256')
                         if search.get(key) is not None}}
            evidence[index].append(attempt)
            aggregate=[c for entry in evidence[index] for c in entry.get('candidates',[])]
            best=max(aggregate,key=lambda c:float(c.get('score',0)),default=None)
            recovery.update(attempts=copy.deepcopy(evidence[index]), candidates=aggregate,
                queries_attempted=sum(a['status']!='SKIPPED' for a in evidence[index]), candidates_checked=len(aggregate),
                best_candidate=recovery.get('best_candidate') if recovery['status']=='MATCH' else best,
                best_score=float(recovery.get('score',0) if recovery['status']=='MATCH' else (best or {}).get('score',0)),searched_at=time.time(),
                strategy_version=QUERY_STRATEGY_VERSION,matcher_version=RECOVERY_VERSION)
            if recovery['status'] in {'NOT_FOUND','CANDIDATES'} and aggregate:
                recovery['status']='NEEDS_REVIEW';recovery['score']=recovery['best_score']
            recovery['strategy_exhausted']=attempt_index==len(kinds) and all(a['status'] in {'NOT_FOUND','NEEDS_REVIEW','MATCH','SKIPPED'} and a.get('query',{}).get('reason')!='TOTAL_TIME_BUDGET_EXHAUSTED' for a in evidence[index])
            slide['reverse_search']=recovery;slide['best_image']=recovery.get('downloaded')
            slide['timings']['reverse_search']=round(sum(a.get('elapsed_seconds',0) for a in evidence[index]),3)
            slide['timings'][f'lens_query_{attempt_index}']=round(float(search.get('elapsed_seconds',0)),3)
            for name in ('candidate_download','candidate_matching','source_ocr'):
                slide['timings'][name]=round(sum(a.get('timings',{}).get(name,0) for a in evidence[index]),3)
            if recovery['status'] == 'MATCH':
                found=sum(bool(s.get('reverse_search',{}).get('downloaded')) for s in slides)
                report('lens', f'{found}/{len(slides)} encontrada; diapositiva {index}', found, len(slides))
            if recovery['status'] in {'MATCH','DISABLED'}:
                terminal.append(index)
            persist()
        pending=[slide for slide in pending if slide['position'] not in terminal]
    for slide in slides:
        slide.pop('reverse_query',None)
        recovery=slide.get('reverse_search',{})
        # Skipped crop preparations are evidence too, including a skipped last query.
        if slide['position'] in evidence:
            recovery['attempts']=evidence[slide['position']]
            recovery['strategy_exhausted']=not recovery.get('budget_exhausted',False) and len(evidence[slide['position']])==len(kinds) and all(a['status'] in {'NOT_FOUND','NEEDS_REVIEW','MATCH','SKIPPED'} and a.get('query',{}).get('reason')!='TOTAL_TIME_BUDGET_EXHAUSTED' for a in evidence[slide['position']])
        if recovery.get('status')=='NEEDS_REVIEW': slide['warnings'].append('SOURCE_CANDIDATES_NEED_REVIEW')
        elif recovery.get('status') in {'ERROR','UNAVAILABLE','TIMEOUT','CAPTCHA'}:
            slide['warnings'].append('REVERSE_SEARCH_'+recovery['status'])
    persist()


def run_pipeline(source_url: str, force=False, settings: Settings | None = None, progress=None, reverse_callback=None) -> dict:
    settings = settings or Settings()
    progress = progress or (lambda stage, message, current=0, total=0: None)
    started = time.perf_counter()
    timings = {'resolve': 0, 'download': 0, 'ocr': 0, 'cleaning': 0, 'reverse_search': 0, 'translation': 0}
    manifest, manifest_path, backup_dir, base = None, None, None, None

    def report(stage, message, current=0, total=0):
        logger.info('[%s] %s', stage, message)
        progress(stage, message, current, total)

    # One job at a time prevents concurrent copies, races and CPU/RAM contention.
    with _pipeline_lock:
        try:
            source_url = validate_tiktok_url(source_url)
            url_cache_path = settings.output_base / '.cache' / 'url-index.json'
            url_index = read_json(url_cache_path) or {}
            if not url_index and settings.output_base.exists():
                for old_path in settings.output_base.glob('*/metadata/manifest.json'):
                    old = read_json(old_path) or {}
                    if old.get('source_url') and old.get('resolved_url'):
                        url_index[old['source_url']] = old['resolved_url']
            resolved = url_index.get(source_url) if not force else None
            if not resolved:
                report('download', 'Resolviendo enlace TikTok...')
                t = time.perf_counter()
                resolved = resolve_tiktok_url(source_url)
                timings['resolve'] = round(time.perf_counter() - t, 3)
            post_id = extract_post_id(resolved)
            if not post_id:
                raise ValueError('No se puede identificar el post; pega el enlace completo de una publicación.')
            url_index[source_url] = resolved
            write_json(url_cache_path, url_index)
            base = settings.output_base / post_id
            base.mkdir(parents=True, exist_ok=True)
            manifest_path = base / 'metadata' / 'manifest.json'
            previous = read_json(manifest_path)
            if not force and completed_cache(base, previous, settings, reverse_callback is not None):
                report('done', 'Post completo en caché; no se repiten fases.')
                result = result_for(base, previous, cached=True)
                result['timings'] = {'total': round(time.perf_counter() - started, 3)}
                return result
            if force and previous:
                # A forced network/OCR/model failure must leave the prior result usable.
                cache_dir = settings.output_base / '.cache'
                cache_dir.mkdir(parents=True, exist_ok=True)
                pending_backup = Path(tempfile.mkdtemp(prefix=f'force-{post_id}-', dir=cache_dir))
                try:
                    shutil.copytree(base, pending_backup / 'post', dirs_exist_ok=True)
                except Exception:
                    shutil.rmtree(pending_backup)
                    raise
                backup_dir = pending_backup
            for name in ['original', 'clean', 'recovered', 'metadata/masks']:
                (base / name).mkdir(parents=True, exist_ok=True)
            # Reuse complete old downloads during schema migration; force downloads to a
            # staging directory so a failed network attempt does not destroy originals.
            originals = sorted((base / 'original').glob('*'))
            originals = [p for p in originals if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
            if force or not originals or not previous:
                report('download', 'Descargando fotografías originales...')
                t = time.perf_counter()
                with tempfile.TemporaryDirectory(prefix='download-', dir=base / 'metadata') as folder:
                    downloaded, detail = download_images(resolved, Path(folder), settings.cookies_file)
                    if not downloaded:
                        skipped = {'status': 'SKIPPED_NO_IMAGES', 'post_id': post_id, 'source_url': source_url,
                            'resolved_url': resolved, 'media_type': 'video_or_no_photo_media', 'images_found': 0,
                            'error': 'Este post no contiene fotografías descargables. Los vídeos no están soportados.',
                            'output_dir': str(base), 'timings': {'download': round(time.perf_counter() - t, 3), 'total': round(time.perf_counter() - started, 3)}}
                        if backup_dir is None:
                            write_json(manifest_path, {**skipped, 'schema_version': SCHEMA_VERSION})
                        else:
                            # No working files changed yet. Preserve the valid post.
                            shutil.rmtree(backup_dir)
                            backup_dir = None
                        report('done', skipped['error'])
                        return skipped
                    for p in (base / 'original').iterdir():
                        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                            p.unlink()
                    originals = []
                    for index, src in enumerate(downloaded, 1):
                        dest = base / 'original' / f'{index:02d}{src.suffix.lower()}'
                        shutil.move(str(src), str(dest))
                        originals.append(dest)
                timings['download'] = round(time.perf_counter() - t, 3)
            else:
                detail = 'Descarga previa conservada.'
            report('download', f'{len(originals)} imágenes')
            manifest = {'schema_version': SCHEMA_VERSION, 'signature': pipeline_signature(settings), 'status': 'PROCESSING',
                'post_id': post_id, 'source_url': source_url, 'resolved_url': resolved, 'media_type': 'photo_carousel',
                'downloader_info': detail, 'ocr_engine': OCR_ENGINE_VERSION, 'cleaning_engine': 'OpenCV TELEA',
                'ollama_model': settings.ollama_model, 'ollama_translation_model': settings.translation_model,
                'ollama_receives_images': False, 'image_text_rendering': False,
                'reverse_search_completed': False, 'slides': [], 'timings': timings, 'translations': {}}
            prior_slides = {s['position']: s for s in (previous or {}).get('slides', [])}
            ocr_jobs = []
            for index, src in enumerate(originals, 1):
                source_hash = sha256(src)
                prior = prior_slides.get(index, {})
                clean = base / 'clean' / f'{index:02d}_CLEAN.jpg'
                mask = base / 'metadata' / 'masks' / f'{index:02d}_MASK.png'
                reusable = (not force and prior.get('original_sha256') == source_hash and prior.get('ocr', {}).get('polygons') is not None
                    and all((previous or {}).get('signature',{}).get(k)==manifest['signature'].get(k) for k in ('ocr','ocr_refinement','ocr_adaptive','ocr_crop_budget')))
                report('ocr', f'Diapositiva {index}/{len(originals)}', index, len(originals))
                t = time.perf_counter()
                if reusable:
                    ocr = {'detected_polygons': prior['ocr']['polygons'],
                           'recognized_lines': copy.deepcopy(prior['ocr']['lines']),
                           'primary_lines':prior['ocr'].get('primary_lines',prior['ocr']['lines']),
                           'diagnostics':{**prior['ocr'].get('diagnostics',{}),'cache_hit':True}}
                else:
                    ocr = ocr_image(src, settings)
                ocr_seconds = round(time.perf_counter() - t, 3)
                timings['ocr'] += ocr_seconds
                ocr_jobs.append((index, src, source_hash, prior, clean, mask, reusable, ocr, ocr_seconds))
            for index, src, source_hash, prior, clean, mask, reusable, ocr, ocr_seconds in ocr_jobs:
                report('clean', f'Diapositiva {index}/{len(originals)}', index, len(originals))
                t = time.perf_counter()
                cleaning_polygons,editorial,background = overlay_polygons(ocr, read_image(src).shape)
                mask_policy_same = (previous or {}).get('signature',{}).get('cleaning')==manifest['signature']['cleaning']
                clean_intact = mask_policy_same and clean.is_file() and prior.get('clean_sha256') == sha256(clean)
                mask_intact = mask_policy_same and mask.is_file() and prior.get('ocr', {}).get('mask_sha256') == sha256(mask)
                if not reusable or not clean_intact or not mask_intact:
                    clean_info = make_clean_image(src, clean, cleaning_polygons, mask)
                else:
                    clean_info = prior.get('cleaning', {})
                clean_seconds = round(time.perf_counter() - t, 3)
                timings['cleaning'] += clean_seconds
                slide = {'position': index, 'slide': index, 'original': f'original/{src.name}', 'original_sha256': source_hash,
                    'clean': f'clean/{clean.name}', 'clean_sha256': sha256(clean), 'cleaning': clean_info,
                    'ocr': {'lines': ocr['recognized_lines'], 'polygons': ocr['detected_polygons'],
                        'raw_lines':ocr['recognized_lines'],'primary_lines':ocr.get('primary_lines',ocr['recognized_lines']),
                        'diagnostics':ocr.get('diagnostics',{}),'primary_engine':OCR_ENGINE_VERSION,
                        'fallback_used':ocr.get('diagnostics',{}).get('fallback_used',False),'editorial_lines':editorial,
                        'background_lines':background,'cleaning_polygons':cleaning_polygons,
                        'regions': len(ocr['detected_polygons']), 'detected_regions':len(ocr['detected_polygons']),
                        'mask': f'metadata/masks/{mask.name}', 'mask_sha256': sha256(mask)},
                    'timings': {'ocr': ocr_seconds, 'clean': clean_seconds,**(ocr.get('diagnostics',{}).get('timings',{}) if not reusable else {'ocr_primary':0,'ocr_fallback':0})}, 'warnings': []}
                slide['ocr_text'] = '\n'.join(str(line.get('text', '')).strip() for line in editorial if str(line.get('text', '')).strip())
                logger.info('[TRANSLATION][OCR] slide %s text=%r', index, slide['ocr_text'])
                if not ocr['recognized_lines']:
                    slide['warnings'].extend(['OCR_NO_TEXT','OCR_NO_TEXT_NEEDS_REVIEW'])
                if background:
                    slide['warnings'].append('BACKGROUND_TEXT_EXCLUDED')
                manifest['slides'].append(slide)
                write_json(manifest_path, manifest)
            t = time.perf_counter()
            timings['ocr_primary']=round(sum(s['timings'].get('ocr_primary',0) for s in manifest['slides']),3)
            timings['ocr_fallback']=round(sum(s['timings'].get('ocr_fallback',0) for s in manifest['slides']),3)
            source_cache = {}
            if not force:
                for slide in manifest['slides']:
                    cached = reusable_source(base, prior_slides.get(slide['position'], {}), slide,
                        (previous or {}).get('signature', {}), manifest['signature'], settings)
                    if cached is None:
                        audit = {}
                        cached = revalidate_saved_source(base, prior_slides.get(slide['position'], {}), slide, settings, audit=audit)
                        if audit:
                            slide['reverse_search_revalidation'] = audit
                    if cached:
                        source_cache[slide['position']] = cached
            recover_with_strategy(base, manifest['slides'], settings, reverse_callback, report, source_cache,
                                  lambda: write_json(manifest_path, manifest))
            manifest['reverse_search_completed'] = reverse_callback is not None and reverse_search_complete(manifest['slides'])
            timings['reverse_search'] = round(time.perf_counter() - t, 3)
            for slide in manifest['slides']:
                recovered = slide.get('reverse_search', {})
                slide['recovered'] = {
                    'file': recovered.get('downloaded'), 'source': 'web_download' if recovered.get('downloaded') else None,
                    'modified': False, 'raw_download': bool(recovered.get('raw_download')),
                    'url': recovered.get('image_url') or recovered.get('source_url'),
                    'domain': recovered.get('source_domain'), 'score': recovered.get('score', 0),
                    'visual_score': recovered.get('visual_score', recovered.get('score', 0)),
                    'ranking': recovered.get('ranking'), 'confidence': recovered.get('confidence'),
                    'contains_text': recovered.get('contains_text', False), 'sha256': recovered.get('sha256'),
                    'http': {'status': recovered.get('http_status'), 'content_type': recovered.get('content_type'),
                             'bytes': recovered.get('bytes'), 'transport': recovered.get('download_transport'),
                             'cache_hit': recovered.get('download_cache_hit', False)},
                    'error': recovered.get('error', '')}
            for name in ('candidate_download', 'candidate_matching', 'source_ocr'):
                timings[name] = round(sum(slide['timings'].get(name, 0) for slide in manifest['slides']), 3)
            for query_index in range(1, settings.reverse_max_queries+1):
                name = f'lens_query_{query_index}'
                timings[name] = round(sum(slide['timings'].get(name, 0) for slide in manifest['slides']), 3)
            report('translate', f'Traduciendo {len(originals)} diapositivas al castellano...')
            slides_text = {s['position']: s['ocr']['editorial_lines'] for s in manifest['slides']}
            for slide in manifest['slides']:
                slide['translation_ocr_lines'] = clean_ocr_lines(slides_text[slide['position']])
            translation_hash = hashlib.sha256(str(slides_text).encode()).hexdigest()
            t = time.perf_counter()
            if not force and previous and previous.get('signature') == manifest['signature'] and previous.get('translation_input_sha256') == translation_hash and previous.get('translations') and not translation_retry_due(previous, settings):
                translations = previous['translations']
                translation_metrics = previous.get('ollama_metrics', {})
                manifest['translation_cache_hit'] = True
                manifest['translation_generated_at'] = previous.get('translation_generated_at', 0)
            else:
                translator = OllamaTranslator(settings)
                try:
                    translations = translator.translate(slides_text)
                finally:
                    # Model/service failures still retain measured diagnostic calls.
                    manifest['ollama_metrics'] = copy.deepcopy(translator.metrics)
                    write_json(manifest_path, manifest)
                translation_metrics = translator.metrics
                manifest['translation_cache_hit'] = False
                manifest['translation_generated_at'] = time.time()
            timings['translation'] = round(time.perf_counter() - t, 3)
            calls = translation_metrics.get('calls',[]) if isinstance(translation_metrics,dict) else []
            timings['reconstruction']=0 if manifest['translation_cache_hit'] else round(sum(c.get('http_seconds',0) for c in calls if c.get('stage')=='reconstruction'),3)
            timings['translation_model']=0 if manifest['translation_cache_hit'] else round(sum(c.get('http_seconds',0) for c in calls if c.get('stage')=='translation'),3)
            manifest.update(translations=translations, translation_input_sha256=translation_hash, ollama_metrics=translation_metrics)
            for slide in manifest['slides']:
                data = translations.get(str(slide['position']), {})
                data['ocr_text'] = slide['ocr_text']
                if not str(data.get('spanish') or '').strip():
                    data['spanish'] = ('[No se pudo traducir automáticamente]\n' + slide['ocr_text']
                                       if slide['ocr_text'] else '[Sin texto detectado en esta diapositiva]')
                    data['fallback_to_english'] = bool(slide['ocr_text'])
                translations[str(slide['position'])] = data
                reconstruction=data.get('reconstruction',{})
                slide['ocr'].update(reconstructed=data.get('original',''),confidence=reconstruction.get('confidence'),
                     reconstruction=reconstruction,uncertain_fragments=reconstruction.get('uncertain_fragments',[]))
                slide['translation'] = {**data.get('translation',{}),'original_reconstructed':data.get('original',''),
                    'text':data['spanish'], 'spanish':data['spanish'],
                    'engine':data.get('translation',{}).get('engine', 'unknown'),
                    'fallback_used':bool(data.get('translation',{}).get('fallback_used', data.get('fallback_to_english',False))),
                    'fallback_to_english':bool(data.get('fallback_to_english',False))}
                slide['warnings'].extend(data.get('warnings',[]))
                if data.get('fallback_to_english'):
                    slide['warnings'].append('LOW_CONFIDENCE_ORIGINAL_ENGLISH_PRESERVED')
                # A batch call's wall time is shared; do not pretend it was measured separately per slide.
                slide['timings']['model_calls_shared_with_batch']=[{'stage':c.get('stage'),'seconds':c.get('http_seconds'),
                    'slides':c.get('slides')} for c in calls if str(slide['position']) in c.get('slides',[])]
            report('save', 'Guardando traducciones.txt y manifest.json...')
            write_translations_txt(base / 'traducciones.txt', len(originals), translations)
            timings['total'] = round(time.perf_counter() - started, 3)
            manifest.update(status='OK', review_required=any(s['warnings'] for s in manifest['slides']),
                translations_file='traducciones.txt', translations_sha256=sha256(base / 'traducciones.txt'), timings={k: round(v, 3) for k, v in timings.items()})
            # Prune only managed artefacts after success, keeping historical photos
            # elsewhere and never exposing an obsolete SOURCE as current output.
            expected_clean = {s['clean'] for s in manifest['slides']}
            expected_masks = {s['ocr']['mask'] for s in manifest['slides']}
            expected_sources = {s['reverse_search'].get('downloaded') for s in manifest['slides']}
            for folder, pattern, expected in [('clean', '*_CLEAN.jpg', expected_clean), ('metadata/masks', '*_MASK.png', expected_masks), ('recovered', '*_SOURCE.*', expected_sources)]:
                for artifact in (base / folder).glob(pattern):
                    if artifact.relative_to(base).as_posix() not in expected:
                        artifact.unlink()
            legacy = base / 'translated'
            if legacy.is_dir():
                import uuid
                archive = settings.output_base / '.cache' / 'legacy' / post_id / uuid.uuid4().hex
                archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy), str(archive / 'translated'))
                manifest['legacy_translated_archived'] = archive.relative_to(settings.output_base).as_posix()
            manifest['validation'] = validate_outputs(base, manifest['slides'])
            if not manifest['validation']['complete']:
                manifest['status'] = 'INCOMPLETE'
                logger.warning('[VALIDATION] INCOMPLETE: %s', '; '.join(manifest['validation']['missing']))
            else:
                logger.info('[VALIDATION] %s originales, CLEAN, RAW y secciones TXT', len(originals))
            write_json(manifest_path, manifest)
            if backup_dir is not None:
                shutil.rmtree(backup_dir)
                backup_dir = None
            report('done', f"Terminado en {timings['total']:.1f} s")
            return result_for(base, manifest)
        except Exception as exc:
            logger.exception('Error en el pipeline')
            timings['total'] = round(time.perf_counter() - started, 3)
            if manifest is not None and manifest_path is not None:
                manifest.update(status='ERROR', error=str(exc), timings=timings)
                write_json(manifest_path, manifest)
            if backup_dir is not None and base is not None:
                # Both paths derive solely from the validated numeric post ID and
                # tempfile under output_base; external directories are never touched.
                if base.resolve().parent != settings.output_base.resolve():
                    raise RuntimeError('La ruta de restauración sale de OUTPUT_BASE.')
                shutil.rmtree(base)
                shutil.copytree(backup_dir / 'post', base)
                shutil.rmtree(backup_dir)
            return {'status': 'ERROR', 'source_url': source_url, 'error': str(exc), 'timings': timings,
                'post_id': (manifest or {}).get('post_id', ''), 'output_dir': str(manifest_path.parent.parent) if manifest_path else '',
                'trace': traceback.format_exc(limit=8)}
