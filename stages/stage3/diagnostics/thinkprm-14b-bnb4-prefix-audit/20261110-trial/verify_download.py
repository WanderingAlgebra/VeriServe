"""Finalize ranged files only after official LFS SHA256 and BF16 inspection."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import shutil
import struct
import time

from huggingface_hub import HfApi
from download_model import REPO, REVISION, RUN, TARGET, save


def inspect(item):
    target = TARGET / item.rfilename
    source = target if target.exists() else target.with_suffix(target.suffix + ".partial-http")
    assert source.stat().st_size == item.size
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        handle.seek(0)
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    assert digest == item.lfs.sha256, (item.rfilename, digest, item.lfs.sha256)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    result = dict(file=item.rfilename, size=item.size, sha256=digest,
                  official_lfs_sha256=item.lfs.sha256,
                  tensor_count=len(tensors),
                  dtype_counts=dict(Counter(v["dtype"] for v in tensors.values())))
    print("VERIFIED", json.dumps(result), flush=True)
    return source, target, result


def main():
    start = time.monotonic()
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    weights = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    assert len(weights) == 6
    with ThreadPoolExecutor(max_workers=3) as pool:
        checked = list(pool.map(inspect, weights))
    for source, target, result in checked:
        if source != target:
            source.replace(target)
    evidence = dict(model=REPO, revision=REVISION, target=str(TARGET), status="completed",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        verification_seconds=time.monotonic() - start,
        total_weight_bytes=sum(item.size for item in weights),
        files=[entry[2] for entry in checked],
        free_bytes_after=shutil.disk_usage(TARGET).free,
        transport="Immutable official repo/revision via hf-mirror.com direct HTTP Range; last 8MiB through official huggingface.co resolve with proxy. All six SHA256 values equal current official immutable LFS metadata.")
    save("download_complete.json", evidence)
    manifest = json.loads((RUN / "download_manifest.json").read_text())
    manifest.update(status="completed", transport=evidence["transport"],
                    completed_utc=evidence["completed_utc"])
    save("download_manifest.json", manifest)
    print("MODEL_READY", TARGET, flush=True)


if __name__ == "__main__":
    main()
