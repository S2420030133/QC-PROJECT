"""Compatibility entry point; this synthetic channel model is not hardware-realistic."""
from research_experiment import get_args, run

if __name__ == "__main__":
    print("Evaluating explicit synthetic noise channels with the unified experiment runner.")
    run(get_args())
