"""Derive checkpoint-level summaries from the frozen WDBC confirmation rows."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results" / "runs" / "independent-confirmation-wdbc-20261016"


def main() -> None:
    finite = pd.read_csv(RUN / "finite_shot_results.csv")
    exact = pd.read_csv(RUN / "exact_results.csv")
    observations = pd.read_csv(RUN / "score_observations.csv", low_memory=False)
    rows = []
    for mode, block in finite.groupby("measurement_mode", sort=True):
        checkpoint = block.groupby(["model_id", "method"], as_index=False).agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
            mean_total_shots=("total_shots", "mean"),
            max_budget_deviation_fraction=("budget_deviation_fraction", "max"),
            draws=("draw", "nunique"))
        for method, group in checkpoint.groupby("method", sort=True):
            rows.append({"measurement_mode": mode, "method": method,
                         "balanced_accuracy_mean": group.balanced_accuracy.mean(),
                         "balanced_accuracy_sd_across_checkpoint_means": group.balanced_accuracy.std(ddof=1),
                         "roc_auc_mean": group.roc_auc.mean(),
                         "roc_auc_sd_across_checkpoint_means": group.roc_auc.std(ddof=1),
                         "mean_realized_total_shots_per_test_case": group.mean_total_shots.mean(),
                         "max_budget_deviation_fraction": group.max_budget_deviation_fraction.max(),
                         "n_checkpoints": int(group.model_id.nunique()),
                         "draws_per_checkpoint": int(group.draws.max())})
    for method, block in exact.groupby("method", sort=True):
        checkpoint = block.groupby("model_id", as_index=False).agg(
            balanced_accuracy=("balanced_accuracy", "mean"), roc_auc=("roc_auc", "mean"))
        rows.append({"measurement_mode": "exact_expectation", "method": method,
                     "balanced_accuracy_mean": checkpoint.balanced_accuracy.mean(),
                     "balanced_accuracy_sd_across_checkpoint_means": checkpoint.balanced_accuracy.std(ddof=1),
                     "roc_auc_mean": checkpoint.roc_auc.mean(),
                     "roc_auc_sd_across_checkpoint_means": checkpoint.roc_auc.std(ddof=1),
                     "mean_realized_total_shots_per_test_case": 0.0,
                     "max_budget_deviation_fraction": 0.0,
                     "n_checkpoints": int(checkpoint.model_id.nunique()),
                     "draws_per_checkpoint": 0})
    exact_fixed = observations[(observations.measurement_mode == "exact_expectation") &
                               (observations.method == "noisy_threshold") &
                               (observations.partition == "test")]
    fixed_checkpoint_rows = []
    for model_id, block in exact_fixed.groupby("model_id"):
        unique = block.drop_duplicates("example_index")
        scores = unique.score.to_numpy()
        labels = unique.y_true.to_numpy()
        fixed_checkpoint_rows.append({"model_id": model_id,
                                      "balanced_accuracy": balanced_accuracy_score(labels, scores >= 0.0),
                                      "roc_auc": roc_auc_score(labels, scores)})
    fixed_checkpoint = pd.DataFrame(fixed_checkpoint_rows)
    rows.append({"measurement_mode": "exact_expectation", "method": "fixed_zero_threshold",
                 "balanced_accuracy_mean": fixed_checkpoint.balanced_accuracy.mean(),
                 "balanced_accuracy_sd_across_checkpoint_means": fixed_checkpoint.balanced_accuracy.std(ddof=1),
                 "roc_auc_mean": fixed_checkpoint.roc_auc.mean(),
                 "roc_auc_sd_across_checkpoint_means": fixed_checkpoint.roc_auc.std(ddof=1),
                 "mean_realized_total_shots_per_test_case": 0.0,
                 "max_budget_deviation_fraction": 0.0,
                 "n_checkpoints": int(fixed_checkpoint.model_id.nunique()),
                 "draws_per_checkpoint": 0})
    out = pd.DataFrame(rows).sort_values(["measurement_mode", "method"])
    out.to_csv(RUN / "checkpoint_level_summary.csv", index=False, float_format="%.15g")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
