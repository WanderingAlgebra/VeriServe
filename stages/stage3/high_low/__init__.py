"""Offline within-question position diagnostic; legacy stage3 stays unchanged."""
from pathlib import Path

from stages.stage2 import run_probe as old
from stages.stage3 import storage


def source_hashes():
    # Keep new files below this package so legacy stage3's *.py hash set is stable.
    return {**storage.source_hashes(), **{
        str(p.relative_to(old.ROOT)): old.digest(p)
        for p in sorted(Path(__file__).parent.glob('*.py'))}}
