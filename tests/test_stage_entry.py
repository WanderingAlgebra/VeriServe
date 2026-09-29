"""Verify the compatibility entry can import and start stage one from any cwd."""
from pathlib import Path
import subprocess
import sys
import tempfile


def test_root_entry_help_from_another_directory():
    entry = Path(__file__).resolve().parents[1] / "run_experiments.py"
    with tempfile.TemporaryDirectory() as cwd:
        result = subprocess.run([sys.executable, str(entry), "--help"],
                                cwd=cwd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--through {exp1,exp2,exp3,exp4}" in result.stdout
