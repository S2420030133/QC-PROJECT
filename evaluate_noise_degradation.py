"""Compatibility entry point for the reproducible experiment runner."""
from research_experiment import get_args, run

if __name__ == "__main__":
    print("This legacy entry point now runs the unified, split-aware experiment suite.")
    run(get_args())
