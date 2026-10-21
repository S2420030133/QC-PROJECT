"""Fit-size-matched two-feature logistic baselines for a completed run."""
from __future__ import annotations
import argparse
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import research_experiment as exp

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-id", required=True, help="Completed run ID under results/runs")
args = parser.parse_args()
run = ROOT / "results" / "runs" / args.run_id
manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
base_seed = int(manifest["config"]["seed"])
models = pd.read_csv(run / "models.csv")
core = models[(models.n_qubits == 2) & (models.n_layers == 2) &
              (models.architecture == "cnot_chain") & (~models.pca_standardize) &
              (~models.reverse_labels)].sort_values(["task", "split", "train_seed"])
rows, predictions = [], []

for model in core.itertuples():
    X, y, _ = exp.tasks()[model.task]
    prep_path = run / "checkpoints" / f"{model.model_id}_preprocessing.joblib"
    prep = joblib.load(prep_path)
    train_ids = np.asarray(prep["train_ids"], dtype=int)
    cal_ids = np.asarray(prep["calibration_ids"], dtype=int)
    test_ids = np.asarray(prep["test_ids"], dtype=int)
    xtrain, _, xtest, _ = exp.prepare_features(X, train_ids, cal_ids, test_ids, 2, False)

    # Reproduce train_model's stratified 85/15 fit/development partition exactly.
    labels = y[train_ids]
    model_seed = base_seed + int(model.split) * 100 + int(model.train_seed)
    fit_ids, _ = train_test_split(np.arange(len(labels)), test_size=0.15,
                                  random_state=model_seed + 31415, stratify=labels)
    classifier = LogisticRegression(max_iter=3000, random_state=41)
    classifier.fit(xtrain[fit_ids], labels[fit_ids])
    scores = classifier.predict_proba(xtest)[:, 1]
    ytest = y[test_ids]
    rows.append({"model_id": model.model_id, "task": model.task, "split": int(model.split),
                 "train_seed": int(model.train_seed), "n_fit_examples": int(len(fit_ids)),
                 "n_features": 2, "balanced_accuracy": float(balanced_accuracy_score(ytest, scores >= 0.5)),
                 "roc_auc": float(roc_auc_score(ytest, scores))})
    predictions.extend({"model_id": model.model_id, "task": model.task, "split": int(model.split),
                        "train_seed": int(model.train_seed), "example_index": int(ix),
                        "y_true": int(yt), "score": float(s), "prediction": int(s >= 0.5)}
                       for ix, yt, s in zip(test_ids, ytest, scores))

result = pd.DataFrame(rows)
result.to_csv(run / "classical_matched_fit_size.csv", index=False)
pd.DataFrame(predictions).to_csv(run / "classical_matched_fit_size_per_example.csv", index=False)
summary = (result.groupby("task", as_index=False)
           .agg(balanced_accuracy_mean=("balanced_accuracy", "mean"),
                balanced_accuracy_sd_across_model_rows=("balanced_accuracy", "std"),
                roc_auc_mean=("roc_auc", "mean"), models=("model_id", "nunique"),
                outer_splits=("split", "nunique"), fit_examples_mean=("n_fit_examples", "mean")))
summary.to_csv(run / "classical_matched_fit_size_summary.csv", index=False)
quantum = models[models.model_id.isin(core.model_id)][["model_id", "task", "split", "clean_balanced_accuracy"]]
paired = quantum.merge(result, on=["model_id", "task", "split"], suffixes=("_quantum", "_classical"))
paired["quantum_minus_classical_ba"] = paired.clean_balanced_accuracy - paired.balanced_accuracy
paired.to_csv(run / "classical_matched_fit_size_paired_rows.csv", index=False)
paired_summary=[]
for task, frame in paired.groupby("task"):
    fold_means=frame.groupby("split").quantum_minus_classical_ba.mean().sort_index().to_numpy()
    rng=np.random.default_rng(7301+sum(map(ord,task)))
    boot=np.asarray([rng.choice(fold_means,len(fold_means),replace=True).mean() for _ in range(10000)])
    paired_summary.append({"task":task,"quantum_minus_classical_ba_mean":float(fold_means.mean()),
        "descriptive_bootstrap_low":float(np.quantile(boot,.025)),
        "descriptive_bootstrap_high":float(np.quantile(boot,.975)),
        "fold_differences": ";".join(f"{v:.8f}" for v in fold_means),"outer_splits":len(fold_means)})
paired_summary=pd.DataFrame(paired_summary)
paired_summary.to_csv(run / "classical_matched_fit_size_paired_summary.csv", index=False)
print("\nMatched-fit-size 2D logistic baseline")
print(summary.to_string(index=False))
print("\nQuantum minus classical paired differences (descriptive fold bootstrap)")
print(paired_summary.to_string(index=False))
