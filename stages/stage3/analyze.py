"""Compatibility CPU self-check entry."""
import json
from veriserve_research.analysis.fixed_step import self_check


if __name__ == "__main__":
    print(json.dumps(self_check(), ensure_ascii=False))
