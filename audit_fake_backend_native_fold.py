"""Recompute metrics from the separate native-gate-folded FakeManila replay."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/runs/final-study-independent-retrain-20261003"
S = RUN / "sensitivity"
COEFS = {1: 15 / 8, 3: -5 / 4, 5: 3 / 8}


def choose_threshold(y, scores):
    scores = np.round(np.asarray(scores, dtype=float), 12)
    candidates = np.unique(np.r_[scores, 0.0])
    values = np.asarray([balanced_accuracy_score(y, scores >= t) for t in candidates])
    best = np.flatnonzero(values == values.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best]))]])


def main():
    observations = pd.read_csv(S / "wdbc_fake_manila_gatefolded_observations.csv")
    metrics = pd.read_csv(S / "wdbc_fake_manila_gatefolded_metrics.csv")
    maximum = {"threshold": 0.0, "balanced_accuracy": 0.0, "roc_auc": 0.0}
    for row in metrics.itertuples():
        model = observations[observations.model_id == row.model_id]
        col = "score" if row.method.endswith("_raw") else "readout_corrected_score"
        frames = {}
        for partition in ("calibration", "test"):
            subset = model[model.partition == partition]
            frames[partition] = subset.pivot(index=["example_index", "label"],
                                             columns="scale", values=col).sort_index()
        if row.method.startswith("noise_threshold_"):
            calibration = frames["calibration"][1].to_numpy()
            test = frames["test"][1].to_numpy()
        else:
            calibration = sum(COEFS[s] * frames["calibration"][s].to_numpy() for s in COEFS)
            test = sum(COEFS[s] * frames["test"][s].to_numpy() for s in COEFS)
        ycal = frames["calibration"].index.get_level_values("label").to_numpy(dtype=int)
        ytest = frames["test"].index.get_level_values("label").to_numpy(dtype=int)
        threshold = choose_threshold(ycal, calibration)
        ba = balanced_accuracy_score(ytest, test >= threshold)
        auc = roc_auc_score(ytest, test)
        maximum["threshold"] = max(maximum["threshold"], abs(threshold - row.threshold))
        maximum["balanced_accuracy"] = max(maximum["balanced_accuracy"], abs(ba - row.balanced_accuracy))
        maximum["roc_auc"] = max(maximum["roc_auc"], abs(auc - row.roc_auc))
    status = "PASS" if max(maximum.values()) < 1e-12 and len(metrics) == 40 else "FAIL"
    result = {"backend": "FakeManilaV2 snapshot simulated with Aer", "metric_rows": len(metrics),
              "recomputed_max_abs_error": maximum, "status": status}
    (S / "wdbc_fake_manila_gatefolded_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
