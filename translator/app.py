"""Local API and serialized jobs. Google Lens runs in the Windows launcher."""
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import Settings
from src.pipeline import run_pipeline
from src.tiktok.downloader import validate_tiktok_url

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger('translator')
app = FastAPI(title='TikTok Translator local', version='2.0.0')
settings = Settings()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='carousel')
jobs = {}
jobs_lock = threading.RLock()


class TranslateRequest(BaseModel):
    url: str
    force: bool = False


class JobRequest(TranslateRequest):
    reverse_search: bool = True


class Candidate(BaseModel):
    image_url: str = Field(max_length=4096)
    source_url: str = Field(default='', max_length=4096)
    source_domain: str = Field(default='', max_length=300)
    alternate_urls: list[str] = Field(default_factory=list, max_length=30)
    width: int | None = None
    height: int | None = None
    thumbnail: bool = False
    browser_download: dict = Field(default_factory=dict)


class ReverseSlide(BaseModel):
    position: int = Field(ge=1, le=1000)
    provider: str = 'google_lens'
    status: str = Field(pattern='^(CANDIDATES|NOT_FOUND|CAPTCHA|UNAVAILABLE|ERROR|DISABLED|TIMEOUT)$')
    candidates: list[Candidate] = Field(default_factory=list, max_length=30)
    error: str = Field(default='', max_length=2000)
    elapsed_seconds: float = Field(default=0, ge=0, le=3600)
    query: dict = Field(default_factory=dict)
    cache_hit: bool = False
    timings: dict = Field(default_factory=dict)
    diagnostics: dict = Field(default_factory=dict)
    fresh_search_status: str | None = None
    fresh_search_error: str | None = None
    cache_fallback_from: str | None = None
    discovery_query_file: str | None = None
    discovery_query_sha256: str | None = None


class ReverseRequest(BaseModel):
    slides: list[ReverseSlide] = Field(max_length=1000)
    round_id: str | None = Field(default=None, max_length=64)


def get_job(job_id):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, 'Trabajo no encontrado. Puede haberse reiniciado el servicio.')
        return jobs[job_id]


def execute_job(job_id, req):
    job = get_job(job_id)

    def progress(stage, message, current=0, total=0):
        with jobs_lock:
            job['state'] = 'RUNNING'
            job['progress'] = {'stage': stage, 'message': message, 'current': current, 'total': total}

    def reverse_callback(base, slides):
        with jobs_lock:
            job['reverse_event'].clear()
            job['reverse_data'] = {}
            job['reverse_round'] = job.get('reverse_round', 0) + 1
            round_id = uuid.uuid4().hex
            job['state'] = 'WAITING_REVERSE_SEARCH'
            entries = []
            for s in slides:
                entry = {'position': s['position'], 'clean': s['clean']}
                for key in ('original', 'reverse_query'):
                    if key in s:
                        entry[key] = s[key]
                if s.get('ocr'):
                    entry['ocr'] = {'mask': s['ocr'].get('mask'),
                                    'raw_lines': s['ocr'].get('raw_lines', s['ocr'].get('lines', [])),
                                    'editorial_lines': s['ocr'].get('editorial_lines', s['ocr'].get('lines', []))}
                entries.append(entry)
            job['reverse_search_request'] = {'output_dir': str(base), 'round_id': round_id,
                                            'round_index': job['reverse_round'], 'slides': entries}
        remaining = min((float(s.get('reverse_query', {}).get('budget_seconds_remaining',
                         settings.reverse_wait_seconds)) for s in slides), default=settings.reverse_wait_seconds)
        ready = job['reverse_event'].wait(max(0, min(settings.reverse_wait_seconds, remaining)))
        with jobs_lock:
            if not ready:
                logger.warning('[lens] Launcher no respondió; se intentarán los métodos restantes.')
                job['reverse_data'] = {s['position']: {'provider': 'google_lens', 'status': 'TIMEOUT',
                    'error': 'El launcher no entregó candidatos a tiempo.', 'candidates': []} for s in slides}
            job['state'] = 'RUNNING'
            data = job['reverse_data']
            job.pop('reverse_search_request', None)
            return data

    try:
        result = run_pipeline(req.url, req.force, settings, progress, reverse_callback if req.reverse_search else None)
    except Exception as exc:
        logger.exception('Falló el worker')
        result = {'status': 'ERROR', 'error': str(exc)}
    with jobs_lock:
        job['result'] = result
        job['state'] = 'ERROR' if result['status'] == 'ERROR' else 'COMPLETED'
        job['finished_at'] = time.time()


@app.get('/health')
def health():
    return {'status': 'ok', 'api_version': 2, 'ocr_loaded': __import__('src.ocr.paddle', fromlist=['_ocr'])._ocr is not None,
        'ollama_model': settings.ollama_model, 'ollama_translation_model': settings.translation_model}


@app.post('/translate')
def translate(req: TranslateRequest):
    """Compatibility endpoint: OCR/CLEAN/text; use launcher jobs for Lens."""
    with jobs_lock:
        if any(j['state'] not in {'COMPLETED', 'ERROR'} for j in jobs.values()):
            raise HTTPException(409, 'Hay un carrusel en proceso. Espera a que termine.')
    return run_pipeline(req.url, force=req.force, settings=settings)


@app.post('/jobs', status_code=202)
def create_job(req: JobRequest):
    try:
        validate_tiktok_url(req.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    with jobs_lock:
        for key, job in list(jobs.items()):
            if job.get('finished_at', time.time()) < time.time() - 86400:
                jobs.pop(key)
        if any(j['state'] not in {'COMPLETED', 'ERROR'} for j in jobs.values()):
            raise HTTPException(409, 'Ya hay un carrusel en proceso. Espera a que termine.')
        job_id = uuid.uuid4().hex
        jobs[job_id] = {'job_id': job_id, 'state': 'QUEUED', 'created_at': time.time(),
            'progress': {'stage': 'download', 'message': 'Preparando trabajo...', 'current': 0, 'total': 0},
            'result': None, 'reverse_event': threading.Event(), 'reverse_data': {}}
    executor.submit(execute_job, job_id, req)
    return {'job_id': job_id, 'state': 'QUEUED'}


@app.get('/jobs/{job_id}')
def job_status(job_id: str):
    with jobs_lock:
        return {k: v for k, v in get_job(job_id).items() if k not in {'reverse_event', 'reverse_data'}}


@app.post('/jobs/{job_id}/reverse-search')
def submit_reverse(job_id: str, req: ReverseRequest):
    with jobs_lock:
        job = get_job(job_id)
        if job['state'] != 'WAITING_REVERSE_SEARCH' or job['reverse_event'].is_set():
            raise HTTPException(409, 'El trabajo no está esperando candidatos.')
        current_round = job['reverse_search_request']['round_id']
        if req.round_id != current_round and not (req.round_id is None and job.get('reverse_round') == 1):
            raise HTTPException(409, 'La respuesta pertenece a otra ronda de búsqueda visual.')
        expected = {s['position'] for s in job['reverse_search_request']['slides']}
        positions = [s.position for s in req.slides]
        if len(set(positions)) != len(positions) or set(positions) != expected:
            raise HTTPException(400, 'Envía exactamente un resultado para cada diapositiva pendiente.')
        job['reverse_data'] = {s.position: s.model_dump() for s in req.slides}
        job['reverse_event'].set()
    return {'accepted': True}
