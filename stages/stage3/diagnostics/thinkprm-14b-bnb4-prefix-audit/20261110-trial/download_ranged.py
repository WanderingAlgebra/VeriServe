"""Download immutable official weights by HTTP Range into one data-disk copy."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, build_opener, ProxyHandler

from huggingface_hub import HfApi
from download_model import REPO, REVISION, RUN, TARGET, save

CHUNK = 8 * 1024 * 1024
WORKERS = 64
DIRECT = build_opener(ProxyHandler({}))


def transfer(fd, url, size, start, end):
    # Earlier interrupted attempts wrote contiguous 1 MiB blocks with pwrite.
    # Resume those extents; the official whole-file SHA256 validates the result.
    range_start = start
    for attempt in range(4):
        try:
            written_end = min(end + 1, os.lseek(fd, range_start, os.SEEK_HOLE))
            if written_end == end + 1:
                return dict(content_range="already written", received_bytes=0)
            start = range_start + ((written_end - range_start) // (1024 * 1024)) * (1024 * 1024)
            expected = f"bytes {start}-{end}/{size}"
            request = Request(url, headers={"Range": f"bytes={start}-{end}"})
            attempt_started = time.monotonic()
            with DIRECT.open(request, timeout=30) as response:
                assert response.status == 206, response.status
                assert response.headers["Content-Range"] == expected, response.headers
                offset = start
                while data := response.read(1024 * 1024):
                    assert offset + len(data) <= end + 1
                    assert os.pwrite(fd, data, offset) == len(data)
                    offset += len(data)
                    if time.monotonic() - attempt_started > 45 and offset <= end:
                        raise TimeoutError("Range transfer exceeded 45 seconds; resume received data")
                assert offset == end + 1, (offset, end)
            return dict(content_range=expected, received_bytes=end + 1 - start)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def main():
    started = time.monotonic()
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    weights = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    # Chunk sizes have separate journals; file extents decide what to resume.
    journal = RUN / f"range_progress_{CHUNK // (1024 * 1024)}m.json"
    done = json.loads(journal.read_text()) if journal.exists() else {}
    opened, jobs = [], []
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for item in weights:
                path = TARGET / item.rfilename
                if path.exists():
                    assert path.stat().st_size == item.size
                    continue
                partial = path.with_suffix(path.suffix + ".partial-http")
                fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
                os.ftruncate(fd, item.size)
                opened.append(fd)
                source = f"https://hf-mirror.com/{REPO}/resolve/{REVISION}/{item.rfilename}"
                with DIRECT.open(Request(source, method="HEAD"), timeout=30) as response:
                    url = response.geturl()
                    assert int(response.headers["Content-Length"]) == item.size
                for start in range(0, item.size, CHUNK):
                    end = min(item.size - 1, start + CHUNK - 1)
                    key = f"{item.rfilename}:{start}"
                    if key not in done:
                        future = pool.submit(transfer, fd, url, item.size, start, end)
                        jobs.append((future, key, end - start + 1))
            index = {future: (key, size) for future, key, size in jobs}
            downloaded = 0
            for future in as_completed(index):
                key, size = index[future]
                done[key] = future.result()
                downloaded += done[key]["received_bytes"]
                save(journal.name, done)
                if len(done) % 16 == 0:
                    print("RANGE_PROGRESS", len(done), downloaded,
                          "bytes/sec", downloaded / (time.monotonic() - started), flush=True)
        for item in weights:
            path = TARGET / item.rfilename
            partial = path.with_suffix(path.suffix + ".partial-http")
            source_path = partial if partial.exists() else path
            with source_path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            assert digest == item.lfs.sha256, (item.rfilename, digest, item.lfs.sha256)
            if partial.exists():
                partial.replace(path)
            print("VERIFIED", item.rfilename, item.size, digest, flush=True)
        save("range_download_complete.json", dict(
            model=REPO, revision=REVISION, status="completed", workers=WORKERS,
            chunk_bytes=CHUNK, elapsed_seconds=time.monotonic() - started,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            files=[dict(name=s.rfilename, size=s.size, sha256=s.lfs.sha256) for s in weights],
            provenance="Immutable official repo/revision via hf-mirror.com without proxy; every response checked 206 and exact Content-Range; full SHA256 equals official huggingface.co LFS metadata"))
    finally:
        for fd in opened:
            os.close(fd)


if __name__ == "__main__":
    main()
