"""Compatibility entry point for matched-budget calibration versus ZNE results."""
from research_experiment import get_args, run

if __name__ == "__main__":
    print("Running calibration, affine-readout, linear-ZNE, and quadratic-ZNE comparisons.")
    run(get_args())
