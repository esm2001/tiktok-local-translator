"""Explain an existing slide without invoking OCR, a model or Google.

    python tools/debug_slide.py <POST_ID> 3
"""
import argparse
import json
from pathlib import Path
import re
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from launcher.main import Config


def describe(manifest,position):
    slide=next((s for s in manifest.get('slides',[]) if s.get('position')==position),None)
    if slide is None: raise ValueError(f'No existe la diapositiva {position}.')
    ocr=slide.get('ocr',{}); translation=slide.get('translation') or {}
    data=(manifest.get('translations') or {}).get(str(position),{})
    reconstruction=data.get('reconstruction') or ocr.get('reconstruction') or {}
    reverse=slide.get('reverse_search') or {}
    out=[]
    def section(title,items): out.extend([title,'-'*len(title),*items,''])
    section('OCR',[f"{line.get('confidence','?')}: {line.get('text','')}" for line in ocr.get('raw_lines',ocr.get('lines',[]))]
            + ['Detection diagnostics: '+json.dumps(ocr.get('diagnostics',{}),ensure_ascii=False),
               'Ignored background: '+json.dumps(ocr.get('background_lines',[]),ensure_ascii=False)])
    section('RECONSTRUCTED',[translation.get('original_reconstructed') or data.get('original') or '[No reconstruction saved]',
            json.dumps(reconstruction,ensure_ascii=False,indent=2)])
    section('TRANSLATION',[translation.get('spanish') or data.get('spanish') or '[No translation saved]',
            'Confidence/fallback: '+json.dumps(translation,ensure_ascii=False,indent=2)])
    section('REVERSE SEARCH',[f"Status: {reverse.get('status','NOT_RUN')}",
            'Best candidate: '+str(reverse.get('best_candidate') or reverse.get('image_url') or ''),
            'Best score: '+str(reverse.get('best_score',reverse.get('score',0))),
            'Attempts: '+json.dumps(reverse.get('attempts',[]),ensure_ascii=False,indent=2),
            'Error: '+str(reverse.get('error',''))])
    section('CANDIDATES',[json.dumps(c,ensure_ascii=False,indent=2) for c in reverse.get('candidates',[])])
    section('TIMINGS',[json.dumps(slide.get('timings',{}),ensure_ascii=False,indent=2)])
    section('WARNINGS',list(slide.get('warnings',[]))+([manifest['error']] if manifest.get('error') else []))
    return '\n'.join(out)


def main():
    # Redirected Windows streams can default to a legacy code page, corrupting
    # the Spanish diagnostics and quoted OCR that this tool is meant to expose.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('post_id');parser.add_argument('position',type=int)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'\d{15,25}',args.post_id) or args.position<1: parser.error('POST_ID numérico y posición positiva requeridos.')
    root=args.output_root or Config.from_project(PROJECT).output_dir
    path=root/args.post_id/'metadata/manifest.json'
    try:
        manifest=json.loads(path.read_text(encoding='utf-8'))
        output=json.dumps(manifest,ensure_ascii=False,indent=2) if args.json else describe(manifest,args.position)
    except (OSError,ValueError) as exc:
        parser.exit(1,f'No se puede diagnosticar la diapositiva: {exc}\n')
    print(output)
    return 0


if __name__=='__main__': raise SystemExit(main())
