"""Pinned HTTP download; work around the installed Hub/httpx2 decoding mismatch."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import requests

REPO='cyankiwi/Qwen3.5-27B-AWQ-4bit'
REV='9d8d63632afafe90bdfcb70164d3e2a0955619f7'
DEST=Path('/root/autodl-tmp/qwen35-27b-awq')/REV
HERE=Path(__file__).resolve().parent


def main():
    info=requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true',timeout=60)
    info.raise_for_status();meta=info.json()
    assert meta['sha']==REV
    (HERE/'download_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    DEST.mkdir(parents=True,exist_ok=True)
    def fetch(s):
        fn=s['rfilename']
        if fn.startswith('.'):return
        link=DEST/fn
        target=Path('/root/qwen35-27b-shards')/fn if fn=='model-00003-of-00004.safetensors' else link
        target.parent.mkdir(parents=True,exist_ok=True)
        size=s['size'];digest=s.get('lfs',{}).get('sha256')
        if not target.exists():
            part=target.with_name(target.name+'.partial')
            for attempt in range(3):
                start=part.stat().st_size if part.exists() else 0
                headers={'Accept-Encoding':'identity'}
                if start:headers['Range']=f'bytes={start}-'
                try:
                    with requests.get(f'https://huggingface.co/{REPO}/resolve/{REV}/{fn}',
                                      headers=headers,stream=True,timeout=(30,120)) as r:
                        r.raise_for_status()
                        mode='ab' if start and r.status_code==206 else 'wb'
                        with part.open(mode) as f:
                            for chunk in r.iter_content(8*1024*1024):f.write(chunk)
                    assert part.stat().st_size==size,(fn,part.stat().st_size,size)
                    part.replace(target);break
                except Exception as e:
                    print('DOWNLOAD_RETRY',fn,attempt,type(e).__name__,flush=True)
                    if attempt==2:raise RuntimeError('Download failed: '+fn) from None
        assert target.stat().st_size==size
        if digest:
            h=hashlib.sha256()
            with target.open('rb') as f:
                for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==digest,'Weight integrity mismatch: '+fn
        if target!=link and not link.exists():link.symlink_to(target)
        print('VERIFIED',fn,size,flush=True)
    files=[s for s in meta['siblings'] if not s['rfilename'].endswith('.safetensors')]
    weights=[s for s in meta['siblings'] if s['rfilename'].endswith('.safetensors')]
    for group in [files,weights]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:list(e.map(fetch,group))
    (HERE/'download_complete.json').write_text(json.dumps(dict(repo=REPO,revision=REV,
        snapshot=str(DEST),verified_weight_sha256=True,completed_unix=time.time()),indent=2)+'\n')
    print('SNAPSHOT_READY',DEST,flush=True)


if __name__=='__main__':main()
