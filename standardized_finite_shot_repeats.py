"""Repeated matched-budget shot draws from the saved standardized WDBC checkpoints.

This post-hoc measurement study reuses trained parameters and saved outer folds;
it reduces Monte Carlo noise but does not add independent training runs.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import research_experiment as exp

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/runs/final-study-independent-retrain-20261003"
SENS = RUN / "sensitivity"
N_DRAWS = 20
protected=[SENS/"wdbc_standardized_finite_shot_20draws.csv",SENS/"wdbc_standardized_finite_shot_20draws_observations.csv"]
if any(p.exists() for p in protected) and "--overwrite" not in sys.argv:
    raise SystemExit("Completed repeated-shot outputs already exist; use --overwrite only to intentionally replace them.")
models = pd.read_csv(RUN / "models.csv")
core = models[(models.task == "wdbc") & (models.n_qubits == 2) &
              (models.architecture == "cnot_chain") & (models.n_layers == 2) &
              (~models.reverse_labels) & (~models.pca_standardize)].sort_values(["split", "train_seed"])
X, y, _ = exp.tasks()["wdbc"]
all_rows, all_obs = [], []
for mr in core.itertuples():
    split = np.load(RUN / "splits" / f"wdbc_split{int(mr.split)}.npz")
    ca, te = split["calibration"], split["test"]
    prep = joblib.load(SENS / "checkpoints" / f"{mr.model_id}_standardized_preprocessing.joblib")
    w = np.load(SENS / "checkpoints" / f"{mr.model_id}_standardized.npy")
    xc = prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[ca])))
    xt = prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[te])))
    cal_n = max(1, int(round(500 * len(te) / len(ca))))
    test_n = 500
    base = {"model_id": mr.model_id, "split": int(mr.split), "train_seed": int(mr.train_seed),
            "gamma": .3, "budget": 1000, "measurement_mode": "finite_calibrated"}
    for rep in range(N_DRAWS):
        # Separate seed streams by method and repetition; calibration and test use independent draws.
        seed0 = 1_800_000 + 10_000 * int(mr.split) + 100 * int(mr.train_seed) + 10 * rep
        qcal = exp.make_qnode(2, 2, "cnot_chain", "composite", cal_n, seed0 + 1)
        qtest = exp.make_qnode(2, 2, "cnot_chain", "composite", test_n, seed0 + 2)
        cal_scores = exp.batch_scores(qcal, w, xc, .3)
        test_scores = exp.batch_scores(qtest, w, xt, .3)
        threshold = exp.best_balanced_threshold(y[ca], cal_scores)
        for mode, decision in [("finite_fixed", 0.0), ("finite_calibrated", threshold)]:
            all_rows.append({**base, "replicate": rep, "method": "noisy_threshold", "measurement_mode":mode,
                "threshold": float(decision), "balanced_accuracy": exp.balanced_accuracy_score(y[te], test_scores >= decision),
                "roc_auc": exp.roc_auc_score(y[te], test_scores), "calibration_shots": cal_n*len(ca),
                "test_shots": test_n*len(te), "total_shots": cal_n*len(ca)+test_n*len(te)})
        for part, ids, labels, vals, shots in [("calibration", ca, y[ca], cal_scores, cal_n),
                                               ("test", te, y[te], test_scores, test_n)]:
            for ix, label, value in zip(ids, labels, vals):
                all_obs.append({**base, "method":"noisy_threshold", "replicate":rep, "partition":part,
                    "example_index":int(ix), "y_true":int(label), "scale":1.0, "shots_per_example":shots,
                    "score":float(value), "selected_threshold":float(threshold)})
        for method, scales, coeffs in [("linear_zne", [1.,2.], np.array([2.,-1.])),
                                       ("quadratic_zne", [1.,1.5,2.], np.array([6.,-8.,3.]))]:
            cscore, calloc, cscale = exp.finite_zne_scores(w, xc, .3, cal_n, scales, coeffs, 2, 2,
                                                           "cnot_chain", seed0 + (100 if method == "linear_zne" else 200))
            tscore, talloc, tscale = exp.finite_zne_scores(w, xt, .3, test_n, scales, coeffs, 2, 2,
                                                           "cnot_chain", seed0 + (300 if method == "linear_zne" else 400))
            threshold = exp.best_balanced_threshold(y[ca], cscore)
            for mode, decision in [("finite_fixed", 0.0), ("finite_calibrated", threshold)]:
                all_rows.append({**base, "replicate":rep, "method":method, "measurement_mode":mode,
                    "threshold":float(decision), "balanced_accuracy":exp.balanced_accuracy_score(y[te], tscore >= decision),
                    "roc_auc":exp.roc_auc_score(y[te], tscore), "calibration_shots":int(sum(calloc)*len(ca)),
                    "test_shots":int(sum(talloc)*len(te)), "total_shots":int(sum(calloc)*len(ca)+sum(talloc)*len(te))})
            for part, ids, labels, scale_values, allocations in [("calibration",ca,y[ca],cscale,calloc),
                                                                  ("test",te,y[te],tscale,talloc)]:
                for scale, shots, vals in zip(scales, allocations, scale_values):
                    for ix, label, value in zip(ids, labels, vals):
                        all_obs.append({**base,"method":method,"replicate":rep,"partition":part,
                            "example_index":int(ix),"y_true":int(label),"scale":float(scale),
                            "shots_per_example":int(shots),"score":float(value),"selected_threshold":float(threshold)})
    print(f"completed split={mr.split} seed={mr.train_seed}", flush=True)

new = pd.DataFrame(all_rows)
new.to_csv(SENS / "wdbc_standardized_finite_shot_20draws.csv", index=False)
pd.DataFrame(all_obs).to_csv(SENS / "wdbc_standardized_finite_shot_20draws_observations.csv", index=False)
# Replace the old two-draw finite-shot comparison with these 20 draws; preserve exact rows.
comparison_path = SENS / "wdbc_standardized_zne_comparison.csv"
old = pd.read_csv(comparison_path)
methods = ["noisy_threshold", "linear_zne", "quadratic_zne"]
old = old[~(old.method.isin(methods) & old.measurement_mode.str.startswith("finite"))]
combined = pd.concat([old, new], ignore_index=True, sort=False)
combined.to_csv(comparison_path, index=False)
paired = (new[new.measurement_mode == "finite_calibrated"]
          .groupby(["model_id","split","method"], as_index=False).balanced_accuracy.mean()
          .pivot(index=["model_id","split"], columns="method", values="balanced_accuracy"))
paired_rows = []
for method in ["linear_zne", "quadratic_zne"]:
    fold = (paired.noisy_threshold - paired[method]).groupby(level="split").mean()
    paired_rows.extend({"method":method,"split":int(split),"threshold_minus_zne":float(value)}
                       for split,value in fold.items())
pd.DataFrame(paired_rows).to_csv(SENS / "wdbc_standardized_paired_fold_differences.csv", index=False)
summary = new.groupby("method", as_index=False).agg(ba_mean=("balanced_accuracy","mean"),
    ba_sd_across_draw_rows=("balanced_accuracy","std"), auc_mean=("roc_auc","mean"),
    n_checkpoint_draws=("model_id","size"))
new["budget_deviation"] = np.abs(new.total_shots - 2*new.test_shots)/(2*new.test_shots)
deviation = new.groupby("method").budget_deviation.max().rename("max_budget_deviation")
summary = summary.merge(deviation, on="method")
summary.to_csv(SENS / "wdbc_standardized_finite_shot_20draws_summary.csv", index=False)
print(summary.to_string(index=False))
