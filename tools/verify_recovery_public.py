r"""Verify real HTTP -> CPU match -> Paddle OCR -> saved SOURCE in isolation.

Run from Windows with the existing translator container and its cached models:
    .venv\Scripts\python.exe tools/verify_recovery_public.py

The defaults use one public Project Gutenberg image found by title search.
Another public candidate can be checked with --slide-position, --image-url and
--source-url. This does not exercise Lens or contact an LLM. The post is read-only; all
recovery writes use a TemporaryDirectory under OUTPUT_BASE/.cache. The report
and a byte-identical copy of its saved SOURCE remain in the project's .cache.
"""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path


PUBLIC_IMAGE = 'https://www.gutenberg.org/files/60859/60859-h/images/p3.jpg'
PUBLIC_SOURCE = 'https://www.gutenberg.org/files/60859/60859-h/60859-h.htm'
RESULT_PREFIX = 'PUBLIC_RECOVERY_JSON='


REMOTE = r'''
import base64,copy,hashlib,json,shutil,sys,tempfile,time
import cv2
from pathlib import Path
from src.config import Settings
from src.image.cleaning import read_image
from src.recovery import recover_slide

payload=json.load(sys.stdin)
settings=Settings()
post=settings.output_base/payload['post_id']
slide=payload['slide']
position=int(slide['position'])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot():
    return {str(p.relative_to(post)):sha(p) for p in post.rglob('*') if p.is_file()}

before=snapshot()
for relative,expected in payload['input_sha256'].items():
    if before.get(relative)!=expected:
        raise RuntimeError('Host/container input mismatch: '+relative)

cache=settings.output_base/'.cache'
cache.mkdir(parents=True,exist_ok=True)
started=time.perf_counter()
with tempfile.TemporaryDirectory(prefix=f'public-recovery-{position:02d}-',dir=cache) as temporary:
    isolated=Path(temporary)
    isolated_slide=copy.deepcopy(slide)
    for relative in [slide['original'],slide['clean'],slide['ocr']['mask']]:
        destination=isolated/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(post/relative,destination)
    (isolated/'metadata'/f'slide-{position:02d}-input.json').write_text(
        json.dumps(isolated_slide,ensure_ascii=False,indent=2),encoding='utf-8')
    search={'provider':'text_search_public_source','status':'CANDIDATES',
        'candidates':[{'image_url':payload['image_url'],'source_url':payload['source_url']}]}
    result=recover_slide(isolated,isolated_slide,search,settings)
    downloaded=result.get('downloaded')
    artifact=None
    saved=None
    if downloaded:
        saved_path=isolated/downloaded
        if saved_path.resolve().parent!=(isolated/'recovered').resolve():
            raise RuntimeError('Recovered path escaped its isolated directory.')
        raw=saved_path.read_bytes()
        image=read_image(saved_path)
        saved={'filename':saved_path.name,'relative_path':downloaded,
            'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
            'width':int(image.shape[1]),'height':int(image.shape[0]),
            'jpeg_magic':raw.startswith(b'\xff\xd8\xff')}
        artifact=base64.b64encode(raw).decode('ascii')
    after=snapshot()
    checks={
        'status_match':result.get('status')=='MATCH',
        'one_candidate':len(result.get('candidates',[]))==1,
        'source_url':result.get('source_url')==payload['source_url'],
        'image_url':result.get('image_url')==payload['image_url'],
        'source_filename':bool(saved and saved['filename']==f'{position:02d}_SOURCE.jpg'),
        'sha256_metadata':bool(saved and saved['sha256']==result.get('sha256')),
        'byte_count_positive':bool(saved and saved['bytes']>0),
        'saved_resolution':bool(saved and result.get('resolution')==
            {'width':saved['width'],'height':saved['height']}),
        'real_source_ocr_accepted':bool(result.get('validation',{}).get('source_ocr',{}).get('accepted')),
        'original_post_unchanged':before==after,
    }
    report={'provider':'text_search_public_source','lens_used':False,
        'external_uploads':False,'llm_calls':0,'post_id':payload['post_id'],
        'slide_position':slide['position'],'recovery':result,'saved_artifact':saved,
        'checks':checks,'passed':all(checks.values()),
        'seconds':round(time.perf_counter()-started,4),
        'temporary_location':'OUTPUT_BASE/.cache/public-recovery-*',
        'input_sha256':payload['input_sha256'],
        'opencv_version':cv2.__version__,'ocr_cpu_threads':settings.cpu_threads,
        'source_post_file_count':len(before),
        'source_post_snapshot_sha256':hashlib.sha256(
            json.dumps(before,sort_keys=True).encode('utf-8')).hexdigest()}
    output={'report':report,'source_base64':artifact}
report['checks']['temporary_directory_removed']=not isolated.exists()
report['passed']=all(report['checks'].values())
print('PUBLIC_RECOVERY_JSON='+json.dumps(output,ensure_ascii=False),flush=True)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--post-id', default='7659483315203804449')
    parser.add_argument('--container', default='tiktok-translator')
    parser.add_argument('--slide-position', type=int, default=7)
    parser.add_argument('--image-url', default=PUBLIC_IMAGE)
    parser.add_argument('--source-url', default=PUBLIC_SOURCE)
    args = parser.parse_args()
    if not args.post_id.isascii() or not args.post_id.isdecimal():
        parser.error('--post-id must contain ASCII digits only')
    if args.slide_position < 1:
        parser.error('--slide-position must be positive')
    root = Path(__file__).resolve().parents[1]
    post = root.parent / 'TikTokTraducidos' / args.post_id
    manifest = json.loads((post / 'metadata' / 'manifest.json').read_text(encoding='utf-8'))
    slide = next((row for row in manifest['slides'] if row['position'] == args.slide_position),None)
    if slide is None:
        parser.error(f'No slide at position {args.slide_position}')
    inputs = [slide['original'], slide['clean'], slide['ocr']['mask']]
    payload = {'post_id':args.post_id,'slide':slide,'image_url':args.image_url,
        'source_url':args.source_url,'input_sha256':{
            relative:hashlib.sha256((post / relative).read_bytes()).hexdigest()
            for relative in inputs}}
    cache = root / '.cache' / 'recovery-public-probe'
    cache.mkdir(parents=True, exist_ok=True)
    report_stem = 'recover-slide' if args.slide_position == 7 else f'recover-slide-{args.slide_position:02d}'
    process = subprocess.run(
        ['docker','exec','-i',args.container,'python','-c',REMOTE],
        input=json.dumps(payload,ensure_ascii=False),text=True,encoding='utf-8',
        capture_output=True,timeout=180)
    result_line = next((line for line in process.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)),None)
    # Avoid keeping the base64 source bytes in an unhelpful text log.
    ordinary_output = '\n'.join(line for line in process.stdout.splitlines()
        if not line.startswith(RESULT_PREFIX))
    (cache / f'{report_stem}.log').write_text(
        ordinary_output+'\n'+process.stderr,encoding='utf-8')
    if process.returncode or not result_line:
        print(process.stderr[-4000:])
        print('Recovery verification failed; see',cache / f'{report_stem}.log')
        return process.returncode or 1
    output = json.loads(result_line[len(RESULT_PREFIX):])
    report = output['report']
    if output.get('source_base64'):
        raw = base64.b64decode(output['source_base64'],validate=True)
        expected = report['saved_artifact']
        cache_target = cache / expected['filename']
        cache_target.write_bytes(raw)
        cached_raw = cache_target.read_bytes()
        report['cached_source'] = str(cache_target.relative_to(root))
        report['checks']['cached_sha256'] = hashlib.sha256(cached_raw).hexdigest()==expected['sha256']
        report['checks']['cached_bytes'] = len(cached_raw)==expected['bytes']
        previous = cache / 'gutenberg-p3.jpg'
        if args.image_url == PUBLIC_IMAGE and args.slide_position == 7 and previous.exists():
            report['checks']['same_as_previous_public_download'] = previous.read_bytes()==cached_raw
    report['passed'] = all(report['checks'].values())
    report_file = cache / f'{report_stem}-report.json'
    report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
