"""Integrity and metric reconstruction audit for the fresh WDBC confirmation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results" / "runs" / "independent-confirmation-wdbc-20261016"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    mismatch = []
    for rel, expected in manifest["artifact_sha256"].items():
        path = RUN / Path(rel)
        if not path.is_file() or sha256(path) != expected:
            mismatch.append(rel)
    source_expected = manifest["source_sha256"]["independent_confirmation_wdbc.py"]
    source_actual = sha256(ROOT / "independent_confirmation_wdbc.py")
    if source_actual != source_expected:
        mismatch.append("source:independent_confirmation_wdbc.py")
    analysis_expected = manifest["analysis_source_sha256"]["summarize_independent_confirmation.py"]
    analysis_actual = sha256(ROOT / "summarize_independent_confirmation.py")
    if analysis_actual != analysis_expected:
        mismatch.append("analysis_source:summarize_independent_confirmation.py")
    protocol_ok = sha256(RUN / "preregistered_protocol.json") == manifest["protocol_sha256"]

    membership = pd.read_csv(RUN / "split_membership.csv")
    split_ok = True
    test_count = np.zeros(569, dtype=int)
    for fold in range(5):
        parts = {k: set(v.example_index.astype(int)) for k, v in
                 membership[membership.fold == fold].groupby("partition")}
        split_ok &= set(parts) == {"train", "calibration", "test"}
        split_ok &= not (parts["train"] & parts["calibration"] or
                         parts["train"] & parts["test"] or parts["calibration"] & parts["test"])
        for i in parts["test"]:
            test_count[i] += 1
    test_once = bool(np.all(test_count == 1))

    saved = pd.read_csv(RUN / "finite_shot_results.csv")
    obs = pd.read_csv(RUN / "score_observations.csv", low_memory=False)
    calibrated = saved[saved.measurement_mode == "finite_calibrated"]
    finite_rows_checked = 0
    finite_metric_mismatch = []
    finite_obs = obs[obs.measurement_mode == "finite_calibrated"]
    keys = ["model_id", "method", "draw"]
    for key, sub in finite_obs[finite_obs.partition == "test"].groupby(keys, sort=False):
        # One record per scale/example; all scales carry the same combined score.
        unique = sub.drop_duplicates("example_index")
        predicted = unique.combined_score.to_numpy() >= float(unique.threshold.iloc[0])
        ba = balanced_accuracy_score(unique.y_true.to_numpy(), predicted)
        auc = roc_auc_score(unique.y_true.to_numpy(), unique.combined_score.to_numpy())
        row = calibrated[(calibrated.model_id == key[0]) & (calibrated.method == key[1]) &
                         (calibrated.draw == key[2])]
        finite_rows_checked += 1
        if len(row) != 1 or not np.isclose(ba, row.balanced_accuracy.iloc[0], atol=1e-12) or \
                not np.isclose(auc, row.roc_auc.iloc[0], atol=1e-12):
            finite_metric_mismatch.append(list(key))

    exact = pd.read_csv(RUN / "exact_results.csv")
    exact_obs = obs[obs.measurement_mode == "exact_expectation"]
    exact_rows_checked = 0
    exact_mismatch = []
    for key, sub in exact_obs[exact_obs.partition == "test"].groupby(["model_id", "method"], sort=False):
        unique = sub.drop_duplicates("example_index")
        threshold = float(unique.threshold.iloc[0])
        score = unique.score.to_numpy()
        ba = balanced_accuracy_score(unique.y_true.to_numpy(), score >= threshold)
        auc = roc_auc_score(unique.y_true.to_numpy(), score)
        row = exact[(exact.model_id == key[0]) & (exact.method == key[1])]
        exact_rows_checked += 1
        if len(row) != 1 or not np.isclose(ba, row.balanced_accuracy.iloc[0], atol=1e-12) or \
                not np.isclose(auc, row.roc_auc.iloc[0], atol=1e-12):
            exact_mismatch.append(list(key))

    budget_deviation = float(calibrated.budget_deviation_fraction.max())
    summary = pd.read_csv(RUN / "checkpoint_level_summary.csv")
    checkpoint_summary_mismatches = []
    finite_cp = saved.groupby(["measurement_mode", "model_id", "method"], as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"), roc_auc=("roc_auc", "mean"),
        total_shots=("total_shots", "mean"),
        budget_deviation=("budget_deviation_fraction", "max"), draws=("draw", "nunique"))
    for (mode, method), group in finite_cp.groupby(["measurement_mode", "method"]):
        row = summary[(summary.measurement_mode == mode) & (summary.method == method)]
        checks = {
            "balanced_accuracy_mean": group.balanced_accuracy.mean(),
            "balanced_accuracy_sd_across_checkpoint_means": group.balanced_accuracy.std(ddof=1),
            "roc_auc_mean": group.roc_auc.mean(),
            "roc_auc_sd_across_checkpoint_means": group.roc_auc.std(ddof=1),
        }
        if len(row) != 1 or any(not np.isclose(float(row[column].iloc[0]), value, atol=1e-12)
                                for column, value in checks.items()):
            checkpoint_summary_mismatches.append([mode, method])
    for method, group in exact.groupby("method"):
        cp_group = group.groupby("model_id").agg(balanced_accuracy=("balanced_accuracy", "mean"),
                                                  roc_auc=("roc_auc", "mean"))
        row = summary[(summary.measurement_mode == "exact_expectation") & (summary.method == method)]
        if len(row) != 1 or not np.isclose(float(row.balanced_accuracy_mean.iloc[0]),
                                             cp_group.balanced_accuracy.mean(), atol=1e-12) or \
                not np.isclose(float(row.balanced_accuracy_sd_across_checkpoint_means.iloc[0]),
                               cp_group.balanced_accuracy.std(ddof=1), atol=1e-12):
            checkpoint_summary_mismatches.append(["exact_expectation", method])
    exact_fixed_obs = exact_obs[(exact_obs.method == "noisy_threshold") &
                                (exact_obs.partition == "test")]
    fixed_rows = []
    for model_id, block in exact_fixed_obs.groupby("model_id"):
        unique = block.drop_duplicates("example_index")
        scores = unique.score.to_numpy()
        labels = unique.y_true.to_numpy()
        fixed_rows.append({"model_id": model_id,
                           "balanced_accuracy": balanced_accuracy_score(labels, scores >= 0.0),
                           "roc_auc": roc_auc_score(labels, scores)})
    fixed_cp = pd.DataFrame(fixed_rows)
    fixed_row = summary[(summary.measurement_mode == "exact_expectation") &
                        (summary.method == "fixed_zero_threshold")]
    if len(fixed_row) != 1 or not np.isclose(float(fixed_row.balanced_accuracy_mean.iloc[0]),
                                               fixed_cp.balanced_accuracy.mean(), atol=1e-12) or \
            not np.isclose(float(fixed_row.roc_auc_mean.iloc[0]), fixed_cp.roc_auc.mean(), atol=1e-12):
        checkpoint_summary_mismatches.append(["exact_expectation", "fixed_zero_threshold"])
    checks = {
        "run_id": manifest["run_id"],
        "artifact_hashes": {"expected": len(manifest["artifact_sha256"]), "mismatches": mismatch,
                            "pass": not mismatch},
        "preregistered_protocol_hash_pass": protocol_ok,
        "five_folds_train_cal_test_disjoint": bool(split_ok),
        "every_WDBC_example_tested_exactly_once": test_once,
        "models": int(pd.read_csv(RUN / "models.csv").shape[0]),
        "finite_metric_reconstruction": {"rows_checked": finite_rows_checked,
                                          "mismatches": finite_metric_mismatch,
                                          "pass": not finite_metric_mismatch},
        "exact_metric_reconstruction": {"rows_checked": exact_rows_checked,
                                        "mismatches": exact_mismatch,
                                        "pass": not exact_mismatch},
        "checkpoint_level_summary_reconstruction": {"mismatches": checkpoint_summary_mismatches,
                                                      "pass": not checkpoint_summary_mismatches},
        "maximum_calibrated_pipeline_budget_deviation_fraction": budget_deviation,
        "overall_pass": bool(not mismatch and protocol_ok and split_ok and test_once and
                             not finite_metric_mismatch and not exact_mismatch and
                             not checkpoint_summary_mismatches),
        "limits": "Internal integrity and metric reconstruction only; not a full independent rerun, external validation, or population-level inference."
    }
    (RUN / "independent_confirmation_audit.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    if not checks["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
