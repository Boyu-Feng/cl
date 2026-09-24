"""Resumable byte-range download of the official public game archive."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import requests
import zipfile
from .core import save, digest, workspace_root

ROOT = workspace_root() / 'ttcl/data/alfworld_delta'
OFFICIAL = 'https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip'
SIZE, CHUNK = 36507267, 262144


def part(i):
    lo, hi = i * CHUNK, min((i+1) * CHUNK, SIZE)-1
    path = ROOT / 'parts' / f'{i:04d}'
    if path.exists() and path.stat().st_size == hi-lo+1:
        return i
    for attempt in range(5):
        try:
            r = requests.get('https://ghproxy.net/' + OFFICIAL, headers={'Range': f'bytes={lo}-{hi}'}, timeout=(20, 45))
            r.raise_for_status()
            assert r.headers.get('Content-Range') == f'bytes {lo}-{hi}/{SIZE}', r.headers
            assert len(r.content) == hi-lo+1
            path.write_bytes(r.content)
            return i
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1)


if __name__ == '__main__':
    (ROOT / 'parts').mkdir(parents=True, exist_ok=True)
    count = (SIZE + CHUNK-1)//CHUNK
    with ThreadPoolExecutor(max_workers=24) as pool:
        for k, f in enumerate(as_completed([pool.submit(part, i) for i in range(count)])):
            f.result()
            print(f'{k+1}/{count}', flush=True)
    archive = ROOT / 'games_complete.zip'
    with archive.open('wb') as out:
        for i in range(count):
            out.write((ROOT / 'parts' / f'{i:04d}').read_bytes())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert not name.startswith('/') and '..' not in Path(name).parts
        z.extractall(ROOT)
        print('Extracted', len(z.namelist()), z.namelist()[:5], flush=True)
    save(ROOT / 'download_provenance.json', {'official_url': OFFICIAL, 'transport_proxy': 'https://ghproxy.net/',
         'sha256': digest(archive.read_bytes()), 'bytes': archive.stat().st_size,
         'zip_crc_verified': True, 'timestamp': time.time()})
