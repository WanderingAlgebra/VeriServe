"""Download the official immutable BF16 snapshot once onto the data disk."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import struct
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO = "launch/ThinkPRM-14B"
REVISION = "1d1b415037cccf826fab6cc1b02c1dac199c0c0f"
TARGET = Path("/root/autodl-tmp/thinkprm-14b") / REVISION
RUN = Path(__file__).resolve().parent


def save(name, data):
    (RUN / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    started = time.time()
    TARGET.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    files = {s.rfilename: s.size for s in info.siblings}
    missing_bytes = sum(size for name, size in files.items()
                        if not (TARGET / name).exists())
    free = shutil.disk_usage(TARGET).free
    assert free > missing_bytes, (free, missing_bytes)
    manifest = dict(
        model=REPO, revision=REVISION, target=str(TARGET),
        started_utc=datetime.now(timezone.utc).isoformat(), files=files,
        total_bytes=sum(files.values()), free_bytes_before=free,
        transport="huggingface_hub local_dir with HTTP (Xet disabled); one copy on data disk",
        status="downloading")
    save("download_manifest.json", manifest)
    if not all((TARGET / name).is_file() and (TARGET / name).stat().st_size == size
               for name, size in files.items()):
        for name in files:
            if not name.endswith(".safetensors"):
                hf_hub_download(REPO, name, revision=REVISION, local_dir=TARGET)
        save("tokenizer_ready.json", dict(model=REPO, revision=REVISION,
                                          target=str(TARGET), status="ready"))
        print("TOKENIZER_READY", TARGET, flush=True)
        snapshot_download(REPO, revision=REVISION, local_dir=TARGET, max_workers=3)
    evidence = []
    for name, expected in files.items():
        path = TARGET / name
        assert path.stat().st_size == expected, (name, path.stat().st_size, expected)
        entry = dict(file=name, size=expected)
        if name.endswith(".safetensors"):
            with path.open("rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(length))
            tensors = {k: v for k, v in header.items() if k != "__metadata__"}
            entry.update(tensor_count=len(tensors),
                         dtype_counts=dict(Counter(v["dtype"] for v in tensors.values())),
                         metadata=header.get("__metadata__"),
                         first_tensor=next(iter(tensors.items())))
        evidence.append(entry)
    ranged = RUN / "range_download_complete.json"
    if ranged.exists():
        manifest["transport"] = json.loads(ranged.read_text())["provenance"]
    manifest["status"] = "completed"
    save("download_manifest.json", manifest)
    save("download_complete.json", dict(
        model=REPO, revision=REVISION, target=str(TARGET), status="completed",
        transport=manifest["transport"],
        elapsed_seconds=time.time() - started, files=evidence,
        total_bytes=sum(files.values()), free_bytes_after=shutil.disk_usage(TARGET).free))
    print("DOWNLOAD_COMPLETE", time.time() - started, flush=True)


if __name__ == "__main__":
    main()
