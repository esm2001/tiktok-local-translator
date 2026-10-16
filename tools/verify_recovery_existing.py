"""Exercise the final candidate OCR gate against pre-existing captioned images."""
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
post = root.parent / 'TikTokTraducidos' / '7659483315203804449'
manifest = json.loads((post / 'metadata' / 'manifest.json').read_text(encoding='utf-8'))
baseline = {'slides': [{'position': s['position'], 'polygons': s['ocr']['polygons']} for s in manifest['slides']]}
remote = '''import sys,json,tempfile
from pathlib import Path
from src.config import Settings
from src.image.cleaning import read_image,write_image,build_text_mask
from src.image.similarity import compare_candidate
from src.recovery import candidate_overlay_check
post=Path('/data/tiktok/7659483315203804449')
folder=Path(tempfile.mkdtemp(prefix='source-ocr-test-'))
rows=[]
for row in json.load(sys.stdin)['slides']:
 i=row['position'];original=post/'original'/f'{i:02d}.jpg';candidate=post/'translated'/f'{i:02d}_ES.jpg'
 if not candidate.exists():
  candidate=next((post.parent/'.cache'/'legacy'/post.name).glob(f'*/translated/{i:02d}_ES.jpg'))
 mask=folder/f'{i:02d}.png';write_image(mask,build_text_mask(read_image(original).shape,row['polygons']))
 comparison=compare_candidate(original,candidate,mask)
 check=candidate_overlay_check(original,candidate,mask,comparison,Settings()) if comparison['accepted'] else {'accepted':False,'reason':comparison['reason']}
 rows.append({'position':i,'visual_accepted':comparison['accepted'],'visual_score':comparison['score'],'ocr_check':check})
print('RECOVERY_JSON='+json.dumps(rows),flush=True)
'''
proc = subprocess.run(['docker', 'exec', '-i', 'tiktok-translator', 'python', '-c', remote], input=json.dumps(baseline), text=True, encoding='utf-8', capture_output=True, timeout=300)
(root / '.cache' / 'recovery-verification.log').write_text(proc.stdout + '\n' + proc.stderr, encoding='utf-8')
for line in proc.stdout.splitlines():
    if line.startswith('RECOVERY_JSON='):
        report = json.loads(line.split('=', 1)[1])
        (root / '.cache' / 'recovery-verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
if proc.returncode:
    print(proc.stderr[-4000:])
raise SystemExit(proc.returncode)
