"""Compare installed text models on real OCR, without changing configuration."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'translator'))
from src.config import Settings
from src.translation.ollama import OllamaTranslator

manifest_path = ROOT.parent / 'TikTokTraducidos' / '7659483315203804449' / 'metadata' / 'manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
if 'slides' in manifest:
    slides = {s['position']: s.get('translation_ocr_lines', [line['text'] for line in s['ocr']['lines']]) for s in manifest['slides']}
else:
    slides = {s['position']: s['ocr_lines'] for s in manifest['images']}
reports = []
for model in sys.argv[1:] or ['qwen2.5:3b', 'qwen2.5:1.5b', 'gemma3:4b']:
    translator = OllamaTranslator(Settings(ollama_url='http://127.0.0.1:11434', ollama_model=model,
                                          ollama_translation_model=''))
    t = time.perf_counter()
    try:
        translations = translator.translate(slides)
        report = {'model': model, 'seconds': round(time.perf_counter() - t, 3), 'translations': translations, 'metrics': translator.metrics}
    except Exception as exc:
        report = {'model': model, 'seconds': round(time.perf_counter() - t, 3), 'error': str(exc), 'metrics': translator.metrics}
    reports.append(report)
    folder = ROOT / '.cache' / 'model-benchmarks'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (model.replace(':', '-') + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
