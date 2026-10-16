"""Run real OCR in the production container against independent local truth.

    .\tools\run_ocr_benchmark.ps1 -Adaptive -Strict

No output post is modified. Model initialisation is reported separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'translator'))
sys.path.insert(0, str(PROJECT / 'tools'))
from regression_metrics import fixture_failures, load_dataset, media_root, ocr_score, time_summary
from src.config import Settings
from src.image.cleaning import read_image
from src.ocr.paddle import OCR_ENGINE_VERSION, OCR_REFINEMENT_VERSION, get_ocr, ocr_image, parse_results, refine_recognized_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=PROJECT/'tests/regression/expected.json')
    parser.add_argument('--media-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', action='store_true')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--case', action='append')
    parser.add_argument('--strict', action='store_true', help='Fail if any visible region is missed; missing fixtures always fail.')
    args = parser.parse_args()
    import cv2
    dataset = load_dataset(args.dataset); root = media_root(dataset,args.media_root)
    unknown=set(args.case or [])-set(dataset['slides'])
    if unknown: parser.error('Unknown dataset case(s): '+', '.join(sorted(unknown)))
    missing=fixture_failures(dataset,root)
    if args.case: missing={k:v for k,v in missing.items() if k in args.case}
    if missing:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({'status':'FAIL','expected_cases':len(args.case or dataset['slides']),
                'missing_fixtures':missing},indent=2),encoding='utf-8')
        print(json.dumps({'missing_fixtures':missing}),flush=True)
        return 1
    try:
        import resource
    except ImportError:
        parser.error('PaddleOCR benchmark runs in the production Linux container; use tools/run_ocr_benchmark.ps1 on Windows.')
    settings = Settings(); started=time.perf_counter(); engine=get_ocr(settings)
    report={'dataset_version':dataset['schema_version'], 'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            'ocr_configuration':OCR_ENGINE_VERSION, 'ocr_refinement':OCR_REFINEMENT_VERSION, 'cpu_threads':settings.cpu_threads,
            'runtime':{'python':sys.version.split()[0], 'opencv':cv2.__version__},
            'initialisation_seconds':round(time.perf_counter()-started,3), 'cases':[],
            'metrics_definition':'Detection: >=50% coverage of independent manual boxes; recognition: normalised editorial CER. Process peak RSS includes shared model and all variants.'}
    for key,case in dataset['slides'].items():
        if args.case and key not in args.case: continue
        path=root/case['image']; item={'case_id':key,'variants':{}}
        if not path.is_file():
            item['error']='MISSING_FIXTURE'; report['cases'].append(item); continue
        image=read_image(path)
        variants=['adaptive'] if args.adaptive else ['primary','native_retry'] + (['clahe','gray'] if args.variants else [])
        primary=None
        for variant in variants:
            tick=time.perf_counter()
            if variant=='adaptive': result=ocr_image(path,settings)
            elif variant=='native_retry':
                result={**primary,'recognized_lines':refine_recognized_lines(path,primary['recognized_lines'],settings)}
            else:
                pixels=image
                if variant in {'clahe','gray'}:
                    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    if variant=='clahe': gray=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(gray)
                    pixels=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
                result=parse_results(list(engine.predict(pixels)))
                if variant=='primary': primary=result
            elapsed=round(time.perf_counter()-tick,3)
            item['variants'][variant]={'seconds':elapsed,'ocr':result,'metrics':ocr_score(case,result,image.shape)}
            print(key,variant,elapsed,'detection',item['variants'][variant]['metrics']['all_regions_detected'],
                  'CER',round(item['variants'][variant]['metrics']['recognition']['cer'],4),flush=True)
        report['cases'].append(item)
        report['process_peak_rss_mb']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={}
    for variant in sorted({v for c in report['cases'] for v in c.get('variants',{})}):
        rows=[c['variants'][variant] for c in report['cases'] if variant in c.get('variants',{})]
        summary[variant]={'slides':len(rows),'detection_slides':sum(r['metrics']['all_regions_detected'] for r in rows),
           'detection_regions':sum(r['metrics']['detected_regions'] for r in rows),
           'expected_regions':sum(r['metrics']['expected_regions'] for r in rows),
           'exact_recognition_slides':sum(r['metrics']['recognition']['exact_normalised'] for r in rows),
           'empty_visible_failures':sum(r['metrics']['visible_text_missing'] for r in rows),
           'cer':sum(r['metrics']['recognition']['character_errors'] for r in rows)/max(1,sum(r['metrics']['recognition']['expected_characters'] for r in rows)),
           'time':time_summary(r['seconds'] for r in rows)}
    report['summary']=summary
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 1 if any('error' in c for c in report['cases']) or (args.strict and any(
        not v['metrics']['all_regions_detected'] or v['metrics']['visible_text_missing']
        for c in report['cases'] for v in c['variants'].values())) else 0


if __name__=='__main__': raise SystemExit(main())
