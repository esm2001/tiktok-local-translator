import json
import os
import logging
from pathlib import Path

logger = logging.getLogger('translator')


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def write_translations_txt(path: Path, count: int, translations: dict):
    parts = []
    for index in range(1, count + 1):
        data = translations.get(str(index), {})
        state=data.get('translation',{}).get('status')
        diagnostic=f"\nESTADO: {state}\n" if state else ''
        original = str(data.get('ocr_text') or data.get('original') or '').strip()
        spanish = str(data.get('spanish') or data.get('translation', {}).get('text') or '').strip()
        if not spanish:
            spanish = ('[No se pudo traducir automáticamente]\n' + original
                       if original else '[Sin texto detectado en esta diapositiva]')
        logger.info('[TRANSLATION][TXT] slide %s engine=%s text=%r', index,
                    data.get('translation', {}).get('engine', 'unknown'), spanish)
        parts.append(f"DIAPOSITIVA {index}\n{'=' * 48}\n\nTEXTO DETECTADO:\n"
            f"{original or '[Sin texto detectado en esta diapositiva]'}\n\nTRADUCCIÓN:\n"
            f"{spanish}\n{diagnostic}")
    temporary = path.with_suffix('.txt.tmp')
    temporary.write_text('\n\n'.join(parts) + '\n', encoding='utf-8')
    os.replace(temporary, path)
