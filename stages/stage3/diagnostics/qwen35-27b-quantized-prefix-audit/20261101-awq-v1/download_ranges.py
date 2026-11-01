"""Resume the same pinned weights with parallel HTTP ranges and SHA256 verification."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import requests
from download_model import REPO,REV,DEST,HERE


def main():
    meta=json.loads((HERE/'download_metadata.json').read_text());assert meta['sha']==REV
    states={};jobs=[];lock=threading.Lock();chunk=64*1024*1024
    for s in meta['siblings']:
        fn=s['rfilename']
        if not fn.endswith('.safetensors'):continue
        link=DEST/fn
        target=Path('/root/qwen35-27b-shards')/fn if fn=='model-00003-of-00004.safetensors' else link
        part=target.with_name(target.name+'.partial');statepath=target.with_name(target.name+'.ranges.json')
        if target.exists():continue
        if statepath.exists():state=json.loads(statepath.read_text())
        else:
            prefix=part.stat().st_size if part.exists() else 0
            state=dict(size=s['size'],done=list(range(prefix//chunk)))
            statepath.write_text(json.dumps(state))
        fd=os.open(part,os.O_RDWR|os.O_CREAT,0o644);os.ftruncate(fd,s['size'])
        states[fn]=(s,link,target,part,statepath,state,fd)
        for start in range(0,s['size'],chunk):
            if start//chunk not in state['done']:jobs.append((fn,start,min(start+chunk,s['size'])-1))
    def fetch(job):
        fn,start,end=job;s,link,target,part,statepath,state,fd=states[fn]
        for attempt in range(4):
            try:
                # Distinct query prevents intermediary caches from reusing a different byte range.
                url=f'https://modelscope.cn/models/{REPO}/resolve/master/{fn}?range_start={start}'
                client=requests.Session();client.trust_env=False
                with client.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},
                                  stream=True,timeout=(30,120)) as r:
                    assert r.status_code==206,(fn,r.status_code)
                    assert r.headers.get('Content-Range')==f'bytes {start}-{end}/{s["size"]}'
                    offset=start
                    for data in r.iter_content(4*1024*1024):
                        assert offset+len(data)<=end+1
                        assert os.pwrite(fd,data,offset)==len(data);offset+=len(data)
                    assert offset==end+1
                with lock:
                    state['done'].append(start//chunk)
                    temp=statepath.with_suffix('.tmp');temp.write_text(json.dumps(state));temp.replace(statepath)
                return
            except Exception as e:
                print('RANGE_RETRY',fn,start,attempt,type(e).__name__,flush=True)
                if attempt==3:raise RuntimeError(f'Range failed: {fn} {start}') from None
    print('RANGE_JOBS',len(jobs),'WORKERS',32,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for n,_ in enumerate(pool.map(fetch,jobs),1):
            if n%16==0:print('RANGES_COMPLETE',n,'OF',len(jobs),flush=True)
    for fn,(s,link,target,part,statepath,state,fd) in states.items():
        os.fsync(fd);os.close(fd)
        h=hashlib.sha256()
        with part.open('rb') as f:
            for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
        assert h.hexdigest()==s['lfs']['sha256'],fn+' SHA256 mismatch'
        part.replace(target)
        if target!=link and not link.exists():link.symlink_to(target)
        print('VERIFIED',fn,s['size'],flush=True)
    (HERE/'download_complete.json').write_text(json.dumps(dict(repo=REPO,revision=REV,
        snapshot=str(DEST),verified_weight_sha256=True,completed_unix=time.time(),
        transport='32 parallel direct ModelScope mirror HTTP ranges; every weight checked against pinned HF revision SHA256'),indent=2)+'\n')
    print('SNAPSHOT_READY',DEST,flush=True)


if __name__=='__main__':main()
