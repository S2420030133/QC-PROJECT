"""Fresh split/seed confirmation for the standardized WDBC comparison.

This is a separate confirmation run, not external-population validation. The
protocol is written before training/evaluation and the run refuses to overwrite
an existing output directory. Simulator settings and method definitions match
the primary study; all training seeds and outer splits are new.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pennylane as qml
import sklearn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import research_experiment as exp

ROOT = Path(__file__).resolve().parent
RUN_ID = "independent-confirmation-wdbc-20261016"
OUT = ROOT / "results" / "runs" / RUN_ID
SPLIT_SEED = 20261016
TRAIN_SEED_BASE = 20261016
N_SPLITS = 5
N_INITS = 2
N_LAYERS = 2
N_QUBITS = 2
ARCH = "cnot_chain"
GAMMA = 0.30
TOTAL_BUDGET = 1000
TEST_SHOTS = 500
SHOT_DRAWS = 20
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.1
LINEAR_SCALES = [1.0, 2.0]
LINEAR_COEFS = np.array([2.0, -1.0])
QUAD_SCALES = [1.0, 1.5, 2.0]
QUAD_COEFS = np.array([6.0, -8.0, 3.0])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing confirmation run: {OUT}")
    for d in [OUT, OUT / "splits", OUT / "checkpoints", OUT / "preprocessing"]:
        d.mkdir(parents=True, exist_ok=True)

    X, y, task_description = exp.tasks()["wdbc"]
    source_path = Path(__file__).resolve()
    protocol = {
        "run_id": RUN_ID,
        "status": "preregistered_before_results",
        "created_utc_date": "2026-10-16",
        "purpose": "fresh outer-split and initialization confirmation of the standardized WDBC simulator comparison",
        "scope_limit": "same public WDBC population; not independent clinical or external-population validation",
        "dataset": {"name": "Wisconsin Diagnostic Breast Cancer", "description": task_description,
                    "n": int(len(y)), "n_features": int(X.shape[1]),
                    "class_counts": {str(int(k)): int((y == k).sum()) for k in np.unique(y)},
                    "loader": "scikit-learn load_breast_cancer"},
        "split": {"method": "StratifiedKFold", "n_splits": N_SPLITS, "shuffle": True,
                  "random_state": SPLIT_SEED,
                  "train_calibration_split": "outer training pool split with calibration=25% stratified",
                  "preprocessing_fit_partition": "outer training partition only",
                  "standardized_pca": True, "pca_components": 2,
                  "angle_scaling": "training-fit MinMaxScaler to [-pi, pi]"},
        "model": {"n_qubits": N_QUBITS, "n_layers": N_LAYERS, "architecture": ARCH,
                  "readout": "PauliZ on wire 0", "training_noise": "none; default.qubit",
                  "evaluation_simulator": "PennyLane default.mixed", "noise": "composite",
                  "gamma": GAMMA, "noise_scope": "same channel schedule as research_experiment.py"},
        "training": {"initializations_per_fold": N_INITS, "seed_rule": "20261016 + 100*fold + init_index",
                     "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam",
                     "learning_rate": LEARNING_RATE,
                     "inner_development_split": "15% stratified from outer training partition inside train_model",
                     "checkpoint_rule": "last epoch; development set does not select checkpoint"},
        "comparison": {"budget_per_test_case": TOTAL_BUDGET,
                       "calibration_and_test_allocation": "500 test shots/example; calibration shots/example set by round(500*n_test/n_calibration), then exact realized counts recorded",
                       "methods": ["noisy_threshold", "linear_zne", "quadratic_zne"],
                       "thresholds": "fit on calibration labels only for each method and shot draw",
                       "scales": {"linear": LINEAR_SCALES, "quadratic": QUAD_SCALES},
                       "coefficients": {"linear": LINEAR_COEFS.tolist(), "quadratic": QUAD_COEFS.tolist()},
                       "finite_shot_draws_per_checkpoint": SHOT_DRAWS,
                       "fixed_threshold_reference": "zero threshold on noisy observable, 1000 test shots/example; reported separately because it has no calibration cost",
                       "exact_control": "exact simulator expectation at gamma=0.30; calibration threshold fit on exact noisy calibration scores"},
        "software": {"python": sys.version, "platform": platform.platform(),
                     "pennylane": qml.__version__, "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__},
        "source": {"path": source_path.name, "sha256_before_run": sha256(source_path)},
        "analysis_plan": {"primary_metric": "balanced_accuracy", "primary_contrast": "noisy calibrated threshold minus calibrated linear and quadratic ZNE",
                          "aggregation": "average shot draws within checkpoint, average initializations within outer fold, report all five fold contrasts; no population-level inferential claim"}
    }
    dump_json(OUT / "preregistered_protocol.json", protocol)

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    split_rows, models, curves, exact_rows, metrics_rows = [], [], [], [], []
    obs_path = OUT / "score_observations.csv"
    all_test_counts = np.zeros(len(y), dtype=int)
    fit_specs = list(splitter.split(np.zeros(len(y)), y))
    for fold, (train_dev, test) in enumerate(fit_specs):
        tr, cal = train_test_split(train_dev, test_size=0.25,
                                   random_state=SPLIT_SEED + 1009 * (fold + 1), stratify=y[train_dev])
        for name, ids in [("train", tr), ("calibration", cal), ("test", test)]:
            np.save(OUT / "splits" / f"fold{fold}_{name}_indices.npy", ids)
            for idx in ids:
                split_rows.append({"fold": fold, "partition": name, "example_index": int(idx), "label": int(y[idx])})
        assert not (set(tr) & set(cal) or set(tr) & set(test) or set(cal) & set(test))
        all_test_counts[test] += 1
        xtr, xcal, xtest, prep = exp.prepare_features(X, tr, cal, test, N_QUBITS, True)
        joblib.dump(prep, OUT / "preprocessing" / f"fold{fold}.joblib")
        for init in range(N_INITS):
            seed = TRAIN_SEED_BASE + 100 * fold + init
            model_id = f"fold{fold}_init{init}_seed{seed}"
            observations = []
            start = time.time()
            weights, history = exp.train_model(xtr, y[tr], N_QUBITS, N_LAYERS, ARCH,
                                               seed, EPOCHS, BATCH_SIZE, LEARNING_RATE)
            np.save(OUT / "checkpoints" / f"{model_id}.npy", weights)
            for row in history:
                curves.append({"model_id": model_id, "fold": fold, "init": init, "seed": seed, **row})
            qexact = exp.make_qnode(N_QUBITS, N_LAYERS, ARCH, "composite", None, seed)
            # Exact calibrated-threshold and exact ZNE controls.
            cal_noisy = exp.batch_scores(qexact, weights, xcal, GAMMA)
            test_noisy = exp.batch_scores(qexact, weights, xtest, GAMMA)
            for method, cal_scores, test_scores in [
                ("noisy_threshold", cal_noisy, test_noisy),
                ("linear_zne", np.round(sum(c * exp.batch_scores(qexact, weights, xcal, GAMMA*s)
                                             for s, c in zip(LINEAR_SCALES, LINEAR_COEFS)), 12),
                 np.round(sum(c * exp.batch_scores(qexact, weights, xtest, GAMMA*s)
                              for s, c in zip(LINEAR_SCALES, LINEAR_COEFS)), 12)),
                ("quadratic_zne", np.round(sum(c * exp.batch_scores(qexact, weights, xcal, GAMMA*s)
                                               for s, c in zip(QUAD_SCALES, QUAD_COEFS)), 12),
                 np.round(sum(c * exp.batch_scores(qexact, weights, xtest, GAMMA*s)
                              for s, c in zip(QUAD_SCALES, QUAD_COEFS)), 12))]:
                threshold = exp.best_balanced_threshold(y[cal], cal_scores)
                exact_rows.append({"model_id": model_id, "fold": fold, "init": init, "seed": seed,
                                   "method": method, "threshold": threshold,
                                   "balanced_accuracy": float(balanced_accuracy_score(y[test], test_scores >= threshold)),
                                   "roc_auc": float(roc_auc_score(y[test], test_scores)),
                                   "n_test": int(len(test)), "measurement_mode": "exact_expectation"})
                for part, ids, labels, vals in [("calibration", cal, y[cal], cal_scores),
                                                ("test", test, y[test], test_scores)]:
                    for idx, label, score in zip(ids, labels, vals):
                        observations.append({"model_id": model_id, "fold": fold, "init": init, "seed": seed,
                                             "method": method, "measurement_mode": "exact_expectation",
                                             "draw": -1, "partition": part, "example_index": int(idx),
                                             "y_true": int(label), "scale": "combined", "score": float(score),
                                             "threshold": float(threshold)})
            cal_shots = max(1, int(round(TEST_SHOTS * len(test) / len(cal))))
            # Repeated finite-shot evaluations; calibration is refit independently on each draw.
            for draw in range(SHOT_DRAWS):
                draw_seed = 3_100_000 + 10_000 * fold + 100 * init + draw * 7
                finite_specs = [
                    ("noisy_threshold", [1.0], np.array([1.0]), 0),
                    ("linear_zne", LINEAR_SCALES, LINEAR_COEFS, 100),
                    ("quadratic_zne", QUAD_SCALES, QUAD_COEFS, 200),
                ]
                for method, scales, coefs, offset in finite_specs:
                    alloc_cal = exp.optimal_shot_allocation(cal_shots, coefs)
                    alloc_test = exp.optimal_shot_allocation(TEST_SHOTS, coefs)
                    cal_parts, test_parts = [], []
                    for j, (scale, nshots) in enumerate(zip(scales, alloc_cal)):
                        q = exp.make_qnode(N_QUBITS, N_LAYERS, ARCH, "composite", nshots,
                                           draw_seed + offset + j)
                        cal_parts.append(exp.batch_scores(q, weights, xcal, GAMMA*scale))
                    for j, (scale, nshots) in enumerate(zip(scales, alloc_test)):
                        q = exp.make_qnode(N_QUBITS, N_LAYERS, ARCH, "composite", nshots,
                                           draw_seed + offset + 20 + j)
                        test_parts.append(exp.batch_scores(q, weights, xtest, GAMMA*scale))
                    cscore = np.round(np.sum(np.asarray(coefs)[:, None] * np.vstack(cal_parts), axis=0), 12)
                    tscore = np.round(np.sum(np.asarray(coefs)[:, None] * np.vstack(test_parts), axis=0), 12)
                    threshold = exp.best_balanced_threshold(y[cal], cscore)
                    total_cal = int(sum(alloc_cal) * len(cal))
                    total_test = int(sum(alloc_test) * len(test))
                    metrics_rows.append({"model_id": model_id, "fold": fold, "init": init, "seed": seed,
                                         "draw": draw, "draw_seed": draw_seed, "method": method,
                                         "threshold": float(threshold),
                                         "balanced_accuracy": float(balanced_accuracy_score(y[test], tscore >= threshold)),
                                         "roc_auc": float(roc_auc_score(y[test], tscore)),
                                         "calibration_shots": total_cal, "test_shots": total_test,
                                         "total_shots": total_cal + total_test,
                                         "nominal_budget_per_case": TOTAL_BUDGET,
                                         "budget_deviation_fraction": abs((total_cal+total_test)-TOTAL_BUDGET*len(test))/(TOTAL_BUDGET*len(test)),
                                         "n_test": int(len(test)), "measurement_mode": "finite_calibrated"})
                    for part, ids, labels, score_parts, allocations, combined_values in [
                        ("calibration", cal, y[cal], cal_parts, alloc_cal, cscore),
                        ("test", test, y[test], test_parts, alloc_test, tscore)]:
                        for scale, nshots, vals in zip(scales, allocations, score_parts):
                            for idx, label, value, combined_value in zip(ids, labels, vals, combined_values):
                                observations.append({"model_id": model_id, "fold": fold, "init": init,
                                    "seed": seed, "method": method, "measurement_mode": "finite_calibrated",
                                    "draw": draw, "partition": part, "example_index": int(idx),
                                    "y_true": int(label), "scale": float(scale), "shots_per_example": int(nshots),
                                    "score": float(value), "combined_score": float(combined_value),
                                    "threshold": float(threshold), "draw_seed": draw_seed})
            # Fixed threshold is a no-calibration reference and uses all 1000 test shots.
            for draw in range(SHOT_DRAWS):
                seed_fixed = 4_100_000 + 10_000 * fold + 100 * init + draw
                qfixed = exp.make_qnode(N_QUBITS, N_LAYERS, ARCH, "composite", TOTAL_BUDGET, seed_fixed)
                fixed_scores = exp.batch_scores(qfixed, weights, xtest, GAMMA)
                metrics_rows.append({"model_id": model_id, "fold": fold, "init": init, "seed": seed,
                                     "draw": draw, "draw_seed": seed_fixed, "method": "fixed_zero_threshold",
                                     "threshold": 0.0,
                                     "balanced_accuracy": float(balanced_accuracy_score(y[test], fixed_scores >= 0.0)),
                                     "roc_auc": float(roc_auc_score(y[test], fixed_scores)),
                                     "calibration_shots": 0, "test_shots": int(TOTAL_BUDGET*len(test)),
                                     "total_shots": int(TOTAL_BUDGET*len(test)),
                                     "nominal_budget_per_case": TOTAL_BUDGET,
                                     "budget_deviation_fraction": 0.0, "n_test": int(len(test)),
                                     "measurement_mode": "finite_fixed"})
            models.append({"model_id": model_id, "fold": fold, "init": init, "seed": seed,
                           "n_train_partition": int(len(tr)), "n_calibration": int(len(cal)),
                           "n_test": int(len(test)), "pca_variance_sum": prep["pca_explained_variance_sum"],
                           "training_seconds": time.time()-start,
                           "checkpoint_sha256": sha256(OUT / "checkpoints" / f"{model_id}.npy")})
            pd.DataFrame(observations).to_csv(obs_path, mode="a", index=False, header=not obs_path.exists())
            print(f"completed {model_id} ({time.time()-start:.1f}s)", flush=True)

    assert np.all(all_test_counts == 1), "Each observation must appear in exactly one confirmation test fold."
    pd.DataFrame(split_rows).to_csv(OUT / "split_membership.csv", index=False)
    pd.DataFrame(models).to_csv(OUT / "models.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "training_curves.csv", index=False)
    pd.DataFrame(exact_rows).to_csv(OUT / "exact_results.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(OUT / "finite_shot_results.csv", index=False)

    finite = pd.DataFrame(metrics_rows)
    cal = finite[finite.measurement_mode == "finite_calibrated"]
    summary = cal.groupby("method", as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd_over_checkpoint_draws=("balanced_accuracy", "std"),
        auc_mean=("roc_auc", "mean"),
        n_checkpoint_draws=("balanced_accuracy", "size"),
        max_budget_deviation_fraction=("budget_deviation_fraction", "max"))
    summary.to_csv(OUT / "finite_summary.csv", index=False)
    exact_summary = pd.DataFrame(exact_rows).groupby("method", as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd_across_10_checkpoints=("balanced_accuracy", "std"),
        auc_mean=("roc_auc", "mean"), n_checkpoints=("balanced_accuracy", "size"))
    exact_summary.to_csv(OUT / "exact_summary.csv", index=False)
    # Average the 20 RNG draws per checkpoint first, then initializations per fold.
    cp = cal.groupby(["fold", "init", "method"], as_index=False).balanced_accuracy.mean()
    fold = cp.groupby(["fold", "method"], as_index=False).balanced_accuracy.mean()
    wide = fold.pivot(index="fold", columns="method", values="balanced_accuracy")
    contrasts = []
    for target in ["linear_zne", "quadratic_zne"]:
        vals = wide.noisy_threshold - wide[target]
        for f, value in vals.items():
            contrasts.append({"contrast": f"noisy_threshold_minus_{target}", "fold": int(f),
                              "difference_balanced_accuracy": float(value),
                              "aggregation": "mean 20 shot draws per checkpoint, then mean two inits within fold"})
    pd.DataFrame(contrasts).to_csv(OUT / "paired_fold_contrasts.csv", index=False)
    dump_json(OUT / "run_summary.json", {
        "run_id": RUN_ID, "n_models": len(models), "n_test_each_once": bool(np.all(all_test_counts == 1)),
        "finite_summary": summary.to_dict(orient="records"),
        "exact_summary": exact_summary.to_dict(orient="records"),
        "paired_fold_contrasts": contrasts,
        "interpretation_limit": "Five overlapping repeated-split folds on the same dataset are descriptive confirmation, not five independent populations.",
        "run_seconds": float(sum(m["training_seconds"] for m in models)),
    })
    # Hash every run artifact except the manifest itself; paths are POSIX relative paths.
    artifacts = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "run_manifest.json":
            artifacts[p.relative_to(OUT).as_posix()] = sha256(p)
    dump_json(OUT / "run_manifest.json", {
        "run_id": RUN_ID, "artifact_sha256": artifacts,
        "source_sha256": {source_path.name: sha256(source_path)},
        "protocol_sha256": sha256(OUT / "preregistered_protocol.json"),
        "artifacts_hashed": len(artifacts), "manifest_path_convention": "POSIX relative paths"})
    print("\nFinite-shot calibrated summary:\n", summary.to_string(index=False), flush=True)
    print("\nExact summary:\n", exact_summary.to_string(index=False), flush=True)
    print("\nPaired fold contrasts:\n", pd.DataFrame(contrasts).to_string(index=False), flush=True)
    print(f"\nRun saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
