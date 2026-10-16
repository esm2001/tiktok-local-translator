"""Check real TikTok resolution/download in temporary container storage."""
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
remote = '''import json,tempfile,time,hashlib
from pathlib import Path
from src.tiktok.downloader import resolve_tiktok_url,download_images
from src.config import Settings
t=time.perf_counter();resolved=resolve_tiktok_url('https://vm.tiktok.com/ZGdQG5uEr/');resolve=time.perf_counter()-t
t=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='gallery-live-test-') as folder:
 images,detail=download_images(resolved,Path(folder),Settings().cookies_file)
 report={'resolved_url':resolved,'resolve_seconds':round(resolve,3),'download_seconds':round(time.perf_counter()-t,3),'images':len(images),'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in images],'downloader_info':detail[-500:]}
print('DOWNLOAD_JSON='+json.dumps(report),flush=True)
'''
proc = subprocess.run(['docker', 'exec', 'tiktok-translator', 'python', '-c', remote], text=True, encoding='utf-8', capture_output=True, timeout=350)
(root / '.cache' / 'download-verification.log').write_text(proc.stdout + '\n' + proc.stderr, encoding='utf-8')
for line in proc.stdout.splitlines():
    if line.startswith('DOWNLOAD_JSON='):
        report = json.loads(line.split('=', 1)[1])
        (root / '.cache' / 'download-verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='files'}, indent=2))
if proc.returncode:
    print(proc.stderr[-3000:])
raise SystemExit(proc.returncode)
