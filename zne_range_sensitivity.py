"""Post-process saved finite-shot ZNE estimates with a physical-range constraint."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import research_experiment as exp

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-id", required=True, help="Run ID under results/runs")
args = parser.parse_args()
run = ROOT / "results" / "runs" / args.run_id
observations = pd.read_csv(run / "calibration_and_scale_observations.csv", low_memory=False)
observations = observations[(observations.pipeline.isin(["linear_zne_calibrated", "quadratic_zne_calibrated"])) &
                            (observations.partition.isin(["calibration_scale", "test_scale"]))].copy()
observations["scale_numeric"] = pd.to_numeric(observations.scale, errors="coerce")
models = pd.read_csv(run / "models.csv").set_index("model_id")
scale_coefficients = {
    "linear_zne_calibrated": {1.0: 2.0, 2.0: -1.0},
    "quadratic_zne_calibrated": {1.0: 6.0, 1.5: -8.0, 2.0: 3.0},
}
keys = ["model_id", "gamma", "budget", "replicate", "pipeline"]
rows = []

for key, group in observations.groupby(keys, sort=False):
    model_id, gamma, budget, replicate, pipeline = key
    coefficients = scale_coefficients[pipeline]
    cal = group[group.partition == "calibration_scale"]
    test = group[group.partition == "test_scale"]
    scores = {}
    labels = {}
    costs = {}
    for part, frame in [("cal", cal), ("test", test)]:
        pivot = frame.pivot_table(index="example_index", columns="scale_numeric", values="score", aggfunc="first")
        if any(scale not in pivot.columns for scale in coefficients):
            raise ValueError(f"Missing ZNE scale for {key} / {part}")
        values = np.round(sum(coef * pivot[scale].to_numpy(dtype=float)
                              for scale, coef in coefficients.items()), 12)
        scores[part] = values
        labels[part] = (frame.drop_duplicates("example_index").set_index("example_index")
                        .loc[pivot.index, "y_true"].to_numpy(dtype=int))
        costs[part] = int(frame.drop_duplicates(["example_index", "scale_numeric"])
                          .shots_per_example.sum())
    raw_threshold = exp.best_balanced_threshold(labels["cal"], scores["cal"])
    saved_thresholds = cal.selected_threshold.dropna().astype(float).unique()
    if len(saved_thresholds) != 1 or abs(float(saved_thresholds[0]) - raw_threshold) > 1e-10:
        raise ValueError(f"Reconstructed raw threshold disagrees with saved calibration rule for {key}")
    bounded_cal = np.clip(scores["cal"], -1.0, 1.0)
    bounded_test = np.clip(scores["test"], -1.0, 1.0)
    bounded_threshold = exp.best_balanced_threshold(labels["cal"], bounded_cal)
    raw_metrics = exp.metrics(labels["test"], scores["test"], raw_threshold)
    bounded_metrics = exp.metrics(labels["test"], bounded_test, bounded_threshold)
    model = models.loc[model_id]
    rows.append({"model_id": model_id, "task": model.task, "n_qubits": int(model.n_qubits),
        "architecture": model.architecture, "split": int(model.split), "train_seed": int(model.train_seed),
        "gamma": float(gamma), "shots_per_test_case": int(budget), "replicate": int(replicate),
        "method": pipeline, "raw_threshold": raw_threshold, "bounded_threshold": bounded_threshold,
        "raw_fraction_outside_pauli_range_test": float(np.mean((scores["test"] < -1) | (scores["test"] > 1))),
        "raw_max_abs_test_score": float(np.max(np.abs(scores["test"]))),
        "measurement_shots_calibration": costs["cal"], "measurement_shots_test": costs["test"],
        "measurement_shots_total": costs["cal"] + costs["test"],
        **{f"raw_{k}": v for k, v in raw_metrics.items()},
        **{f"bounded_{k}": v for k, v in bounded_metrics.items()}})

out = pd.DataFrame(rows)
out.to_csv(run / "zne_range_sensitivity.csv", index=False)
summary = (out.groupby(["task", "n_qubits", "architecture", "gamma", "shots_per_test_case", "method"],
                       as_index=False)
           .agg(raw_balanced_accuracy=("raw_balanced_accuracy", "mean"),
                bounded_balanced_accuracy=("bounded_balanced_accuracy", "mean"),
                raw_roc_auc=("raw_roc_auc", "mean"), bounded_roc_auc=("bounded_roc_auc", "mean"),
                mean_fraction_outside_range=("raw_fraction_outside_pauli_range_test", "mean"),
                max_abs_score=("raw_max_abs_test_score", "max"),
                total_shots_mean=("measurement_shots_total", "mean"),
                model_rows=("model_id", "nunique"), outer_splits=("split", "nunique")))
summary.to_csv(run / "zne_range_sensitivity_summary.csv", index=False)
print(summary.to_string(index=False))
