"""Rebuild summaries and figures from saved experimental observations."""
import argparse
from pathlib import Path
import research_experiment as experiment
from research_experiment import summarize_and_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Select results/runs/<run-id>; defaults to legacy results/")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "results"
    if args.run_id:
        root = root / "runs" / args.run_id
    if not (root / "exact_performance.csv").exists():
        raise SystemExit("No results found. Run research_experiment.py first.")
    experiment.OUT = root
    experiment.CHECKPOINTS = root / "checkpoints"
    experiment.SPLITS = root / "splits"
    summarize_and_plot()
    print(f"Updated derived summaries and figures in {root}")
