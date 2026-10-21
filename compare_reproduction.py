"""Compare two completed runs' predictions, metrics, training histories, and weights."""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", default="final-study-20260929")
parser.add_argument("--reproduction", required=True,
                    help="Run ID for the fresh run under results/runs/")
parser.add_argument("--output-dir", help="Output directory; defaults to the reproduction run")
parser.add_argument("--tolerance", type=float, default=1e-10)
args = parser.parse_args()
base = ROOT / "results" / "runs" / args.reference
rep = ROOT / "results" / "runs" / args.reproduction
outdir = Path(args.output_dir) if args.output_dir else rep
if not outdir.is_absolute():
    outdir = ROOT / outdir
outdir.mkdir(parents=True, exist_ok=True)

result_rows = []
failed = False
for filename, keys, metrics in [
    ("exact_performance.csv", ["model_id", "noise", "gamma", "method"],
     ["balanced_accuracy", "roc_auc", "threshold", "score_rmse_vs_clean"]),
    ("finite_shot_performance.csv", ["model_id", "gamma", "shots_per_test_case", "replicate", "method"],
     ["balanced_accuracy", "roc_auc", "selected_threshold", "measurement_shots_total"]),
]:
    x, y = pd.read_csv(base / filename), pd.read_csv(rep / filename)
    joined = x.merge(y, on=keys, suffixes=("_reference", "_reproduction"), how="outer", indicator=True)
    for metric in metrics:
        left, right = f"{metric}_reference", f"{metric}_reproduction"
        if left not in joined or right not in joined:
            continue
        both = joined._merge.eq("both") & joined[left].notna() & joined[right].notna()
        delta = (joined.loc[both, left].astype(float) - joined.loc[both, right].astype(float)).abs()
        tolerance = 0 if metric == "measurement_shots_total" else args.tolerance
        mismatch = int((delta > tolerance).sum())
        rows_missing = int((joined._merge != "both").sum())
        result_rows.append({"file": filename, "metric": metric, "reference_rows": len(x),
            "reproduction_rows": len(y), "matched_rows": int(both.sum()),
            "missing_or_extra_rows": rows_missing,
            "max_absolute_difference": float(delta.max()) if len(delta) else None,
            "mismatches_over_tolerance": mismatch, "tolerance": tolerance})
        failed |= rows_missing != 0 or mismatch != 0
pd.DataFrame(result_rows).to_csv(outdir / "reproduction_comparison.csv", index=False)
print("PERFORMANCE COMPARISON")
print(pd.DataFrame(result_rows).to_string(index=False))

# A clean full retraining should reproduce model parameters and optimization curves
# under the same fixed seeds and software environment.
weight_rows = []
ref_models = pd.read_csv(base / "models.csv")
rep_models = pd.read_csv(rep / "models.csv")
common = sorted(set(ref_models.model_id) & set(rep_models.model_id))
for model_id in common:
    a = np.load(base / "checkpoints" / f"{model_id}.npy")
    b = np.load(rep / "checkpoints" / f"{model_id}.npy")
    if a.shape != b.shape:
        delta, equal = np.inf, False
    else:
        delta, equal = float(np.max(np.abs(a - b))), bool(np.array_equal(a, b))
    weight_rows.append({"model_id": model_id, "reference_shape": str(a.shape),
        "reproduction_shape": str(b.shape), "max_absolute_weight_difference": delta,
        "bitwise_identical": equal, "within_tolerance": delta <= args.tolerance})
weight_df = pd.DataFrame(weight_rows)
weight_df.to_csv(outdir / "model_weight_reproduction_comparison.csv", index=False)
missing_models = sorted(set(ref_models.model_id) ^ set(rep_models.model_id))
weight_failed = bool(missing_models or weight_df.empty or (~weight_df.within_tolerance).any())
failed |= weight_failed
print("\nMODEL WEIGHTS", "missing_or_extra=", missing_models,
      "max_abs_delta=", weight_df.max_absolute_weight_difference.max() if len(weight_df) else None,
      "bitwise_identical=", int(weight_df.bitwise_identical.sum()), "/", len(weight_df))

curve_file = "training_curves.csv"
curves_a, curves_b = pd.read_csv(base / curve_file), pd.read_csv(rep / curve_file)
curve_metrics = [c for c in ["training_binary_cross_entropy", "development_binary_cross_entropy",
                             "development_balanced_accuracy_at_zero"] if c in curves_a and c in curves_b]
curve_keys = ["model_id", "epoch"]
curves = curves_a.merge(curves_b, on=curve_keys, how="outer", suffixes=("_reference", "_reproduction"),
                        indicator=True)
curve_rows=[]
for metric in curve_metrics:
    left, right = f"{metric}_reference", f"{metric}_reproduction"
    both = curves._merge.eq("both") & curves[left].notna() & curves[right].notna()
    delta=(curves.loc[both,left].astype(float)-curves.loc[both,right].astype(float)).abs()
    mismatch=int((delta>args.tolerance).sum())
    missing_rows=int((curves._merge!="both").sum())
    curve_rows.append({"metric":metric,"reference_rows":len(curves_a),"reproduction_rows":len(curves_b),
        "matched_rows":int(both.sum()),"missing_or_extra_rows":missing_rows,
        "max_absolute_difference":float(delta.max()) if len(delta) else None,
        "mismatches_over_tolerance":mismatch,"tolerance":args.tolerance})
    failed |= missing_rows != 0 or mismatch != 0
pd.DataFrame(curve_rows).to_csv(outdir / "training_curve_reproduction_comparison.csv", index=False)
print("\nTRAINING CURVES")
print(pd.DataFrame(curve_rows).to_string(index=False))
if failed:
    raise SystemExit("Runs differ; see comparison CSVs and report differences rather than treating this as a pass.")
print("\nAll checked outputs match within tolerance; shot totals match exactly.")
