import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    output_base: Path = field(default_factory=lambda: Path(os.getenv('OUTPUT_BASE', '/data/tiktok')))
    cookies_file: Path = field(default_factory=lambda: Path(os.getenv('TIKTOK_COOKIES', '/data/secrets/cookies.txt')))
    ollama_url: str = field(default_factory=lambda: os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434').rstrip('/'))
    ollama_model: str = field(default_factory=lambda: os.getenv('OLLAMA_MODEL', 'gemma3:4b'))
    ollama_translation_model: str = field(default_factory=lambda: os.getenv('OLLAMA_TRANSLATION_MODEL', '').strip())
    ollama_keep_alive: str = field(default_factory=lambda: os.getenv('OLLAMA_KEEP_ALIVE', '30m'))
    cpu_threads: int = field(default_factory=lambda: int(os.getenv('OCR_CPU_THREADS', str(max(2, min(8, os.cpu_count() or 4))))))
    reverse_wait_seconds: int = field(default_factory=lambda: int(os.getenv('REVERSE_SEARCH_WAIT_SECONDS', '1800')))
    match_threshold: float = field(default_factory=lambda: float(os.getenv('VISUAL_MATCH_THRESHOLD', '0.84')))
    candidate_min_dimension: int = field(default_factory=lambda: int(os.getenv('CANDIDATE_MIN_DIMENSION', '400')))
    max_candidates: int = field(default_factory=lambda: int(os.getenv('LENS_MAX_CANDIDATES', '12')))
    candidate_workers: int = 3
    ocr_adaptive_fallback: bool = field(default_factory=lambda: os.getenv('OCR_ADAPTIVE_FALLBACK', 'true').lower() in {'true','1','yes'})
    ocr_max_crop_retries: int = field(default_factory=lambda: int(os.getenv('OCR_MAX_CROP_RETRIES', '6')))
    reverse_max_queries: int = field(default_factory=lambda: int(os.getenv('REVERSE_MAX_QUERIES', '6')))
    reverse_negative_ttl: int = field(default_factory=lambda: int(os.getenv('REVERSE_NEGATIVE_TTL_SECONDS', '3600')))
    reverse_total_budget: int = field(default_factory=lambda: int(os.getenv('REVERSE_TOTAL_BUDGET_SECONDS', '300')))
    translation_timeout_seconds: int = field(default_factory=lambda: int(os.getenv('TRANSLATION_TIMEOUT_SECONDS','180')))
    translation_budget_seconds: int = field(default_factory=lambda: int(os.getenv('TRANSLATION_BUDGET_SECONDS','360')))
    translation_retry_ttl: int = field(default_factory=lambda: int(os.getenv('TRANSLATION_RETRY_TTL_SECONDS','60')))

    @property
    def translation_model(self) -> str:
        return self.ollama_translation_model or self.ollama_model

    def __post_init__(self):
        if not 1 <= self.cpu_threads <= 32:
            raise ValueError('OCR_CPU_THREADS debe estar entre 1 y 32.')
        if not 0.7 <= self.match_threshold <= 1:
            raise ValueError('VISUAL_MATCH_THRESHOLD debe estar entre 0.7 y 1.')
        if self.candidate_min_dimension < 200 or not 1 <= self.max_candidates <= 30:
            raise ValueError('Configuración de candidatos no válida.')
        if not 0 <= self.ocr_max_crop_retries <= 12 or not 1 <= self.reverse_max_queries <= 6:
            raise ValueError('Los presupuestos OCR/queries no son válidos.')
        if not 60 <= self.reverse_negative_ttl <= 86400 or not 30 <= self.translation_timeout_seconds <= 600:
            raise ValueError('Los plazos de caché/traducción no son válidos.')
        if not 30 <= self.reverse_total_budget <= 900:
            raise ValueError('REVERSE_TOTAL_BUDGET_SECONDS debe estar entre 30 y 900.')
        if not 30 <= self.translation_budget_seconds <= 900:
            raise ValueError('TRANSLATION_BUDGET_SECONDS debe estar entre 30 y 900.')
        if not 10 <= self.translation_retry_ttl <= 86400:
            raise ValueError('TRANSLATION_RETRY_TTL_SECONDS debe estar entre 10 y 86400.')


SCHEMA_VERSION = 4
