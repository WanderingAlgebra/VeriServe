"""Compatibility entry; new execution uses v2 runs, historical runs use their source commit."""
from veriserve_research.intervention.fixed_step import main


if __name__ == "__main__":
    main()
