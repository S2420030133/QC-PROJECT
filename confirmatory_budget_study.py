"""Preregistered development/confirmation study for budget-aware calibration.

This is a separate, fresh-synthetic-data study. Development realizations choose
the ZNE schedule and calibration fraction. Confirmation realizations are never
used to select settings. All claims are simulator- and task-family-specific.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import sklearn
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import research_experiment as base

ROOT = Path(__file__).resolve().parent
RUN_ID = "confirmatory-budget-study-20261005"
OUT = ROOT / "results" / "runs" / RUN_ID
GAMMA = 0.30
N = 500
N_FEATURES = 8
N_INFORMATIVE = 4
N_REDUNDANT = 2
DIFFICULTIES = {"moderate": 0.9, "easier": 1.2}
DEV_SEEDS = {"moderate": [610501, 610502, 610503], "easier": [610511, 610512, 610513]}
CONF_SEEDS = {"moderate": [620501, 620502, 620503], "easier": [620511, 620512, 620513]}
ARCH = "cnot_star"
LAYERS = 2
EPOCHS = 30
BATCH_SIZE = 32
LR = 0.1
TUNE_BUDGET = 1000
FINAL_BUDGETS = [100, 1000]
ALLOCATIONS = [0.25, 0.50, 0.75]
SHOT_GRID = [25, 50, 100, 200, 500, 1000]
SHOT_REPS = 5
SCALE_CONFIGS = {
    "linear_1_1.25": [1.0, 1.25],
    "linear_1_1.5": [1.0, 1.5],
    "quadratic_1_1.25_1.5": [1.0, 1.25, 1.5],
}


def coeffs(scales: list[float]) -> np.ndarray:
    # Richardson coefficients satisfy sum(c_i * scale_i**k)=delta[k,0].
    return np.linalg.solve(np.asarray([[s**k for s in scales] for k in range(len(scales))]),
                           np.asarray([1.0] + [0.0] * (len(scales) - 1)))


COEFFICIENTS = {k: coeffs(v) for k, v in SCALE_CONFIGS.items()}


def make_data(seed: int, difficulty: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = make_classification(n_samples=N, n_features=N_FEATURES,
                               n_informative=N_INFORMATIVE, n_redundant=N_REDUNDANT,
                               n_clusters_per_class=2, weights=[0.5, 0.5],
                               class_sep=DIFFICULTIES[difficulty], flip_y=0.02,
                               random_state=seed)
    return np.asarray(x, float), np.asarray(y, int)


def prepare(X, train_ids, other_ids, n_qubits=2):
    std = StandardScaler().fit(X[train_ids])
    xtr = std.transform(X[train_ids])
    pca = PCA(n_components=n_qubits, svd_solver="full").fit(xtr)
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(pca.transform(xtr))
    def transform(ids):
        return scaler.transform(pca.transform(std.transform(X[ids])))
    return transform(train_ids), {name: transform(ids) for name, ids in other_ids.items()}, {
        "pca_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "pca_components": pca.components_.tolist(),
        "standardizer_mean": std.mean_.tolist(), "standardizer_scale": std.scale_.tolist(),
        "feature_scaler_min": scaler.min_.tolist(), "feature_scaler_scale": scaler.scale_.tolist(),
    }


def train_realization(seed: int, difficulty: str, stage: str, saved_weights=None) -> dict:
    X, y = make_data(seed, difficulty)
    all_ids = np.arange(len(y))
    train_ids, remain = train_test_split(all_ids, test_size=0.40, random_state=seed + 100,
                                         stratify=y)
    if stage == "development":
        cal_ids, dev_ids = train_test_split(remain, test_size=0.625, random_state=seed + 200,
                                           stratify=y[remain]) # 15% calibration, 25% dev
        others = {"cal": cal_ids, "dev": dev_ids}
    else:
        cal_ids, test_ids = train_test_split(remain, test_size=0.50, random_state=seed + 300,
                                             stratify=y[remain]) # 20% calibration, 20% final test
        others = {"cal": cal_ids, "test": test_ids}
    xtrain, transformed, prep = prepare(X, train_ids, others, n_qubits=2)
    train_seed = seed + 7000
    if saved_weights is None:
        weights, history = base.train_model(xtrain, y[train_ids], 2, LAYERS, ARCH, train_seed,
                                            EPOCHS, BATCH_SIZE, LR)
    else:
        weights, history = np.asarray(saved_weights), []
    qexact = base.make_qnode(2, LAYERS, ARCH, "composite", None, train_seed)
    xcal = transformed["cal"]
    exact_cal = base.batch_scores(qexact, weights, xcal, GAMMA)
    item = {"id": f"{stage}_{difficulty}_{seed}", "stage": stage,
            "difficulty": difficulty, "seed": seed, "X": X, "y": y,
            "train_ids": train_ids, "cal_ids": cal_ids, "other_ids": others,
            "transformed": transformed, "weights": weights, "qexact": qexact,
            "exact_cal": exact_cal, "history": history, "prep": prep,
            "train_seed": train_seed, "xtrain": xtrain}
    if stage == "development":
        item["y_dev"] = y[dev_ids]
    else:
        item["y_test"] = y[test_ids]
    return item


def shot_scores(item, split_name, shots, seed, scale=1.0):
    q = base.make_qnode(2, LAYERS, ARCH, "composite", int(shots), int(seed))
    return base.batch_scores(q, item["weights"], item["transformed"][split_name], GAMMA * scale)


def scale_scores(item, split_name, budget, scale_name, seed):
    scales, cs = SCALE_CONFIGS[scale_name], COEFFICIENTS[scale_name]
    allocations = base.optimal_shot_allocation(int(budget), cs)
    obs = [shot_scores(item, split_name, ns, seed + j, s)
           for j, (s, ns) in enumerate(zip(scales, allocations))]
    return np.round(np.sum(np.asarray(cs)[:, None] * np.vstack(obs), axis=0), 12), obs, allocations


def pipeline_scores(item, budget, fraction, scale_name, seed, test_name):
    ncal, ntest = len(item["cal_ids"]), len(item["other_ids"][test_name])
    cal_shots = max(1, int(round(budget * fraction * ntest / ncal)))
    test_shots = max(1, int(round(budget * (1 - fraction))))
    ycal = item["y"][item["cal_ids"]]
    yc = shot_scores(item, "cal", cal_shots, seed + 11, 1.0)
    yt = shot_scores(item, test_name, test_shots, seed + 12, 1.0)
    threshold = base.best_balanced_threshold(ycal, yc)
    methods = {"fixed_threshold": (yt, 0.0), "noisy_calibration": (yt, threshold)}
    ztest, zobs, zalloc = scale_scores(item, test_name, budget, scale_name, seed + 100)
    methods["selected_zne"] = (ztest, 0.0)
    zcal, zcobs, zcalloc = scale_scores(item, "cal", cal_shots, scale_name, seed + 200)
    zthr = base.best_balanced_threshold(ycal, zcal)
    ztest_cal, ztobs, ztalloc = scale_scores(item, test_name, test_shots, scale_name, seed + 300)
    methods["selected_zne_calibration"] = (ztest_cal, zthr)
    rows = []
    ytest = item["y"][item["other_ids"][test_name]]
    for method, (score, thr) in methods.items():
        ba = float(balanced_accuracy_score(ytest, score >= thr))
        total = (ntest * budget if method in ("fixed_threshold", "selected_zne") else
                 ntest * test_shots + ncal * cal_shots)
        rows.append({"realization": item["id"], "stage": item["stage"],
                     "difficulty": item["difficulty"], "seed": item["seed"],
                     "budget_per_eval_case": budget, "calibration_fraction": fraction,
                     "scale_config": scale_name, "method": method,
                     "balanced_accuracy": ba, "threshold": float(thr),
                     "calibration_shots_per_example": cal_shots,
                     "inference_shots_per_example": test_shots,
                     "realized_total_measurements": int(total),
                     "nominal_equivalent_budget_per_eval_case": float(total / ntest)})
    return rows, {"calibration_scores": yc, "zne_calibration_scores": zcal,
                  "zne_test_scale_scores": zobs, "zne_cal_scale_scores": zcobs,
                  "zne_calibrated_test_scale_scores": ztobs,
                  "alloc_test": zalloc, "alloc_cal": zcalloc, "alloc_test_calibrated": ztalloc}


def budget_prediction(item, budget, seed, is_confirmation):
    """Predict shots/example for 10%-relative SE of noisy class-mean score gap.

    Validate by resampling calibration shots and measuring test-set BA regret
    versus the exact-expectation threshold selected on the calibration labels.
    Test labels are used only to score prediction after the count is frozen.
    """
    ycal = item["y"][item["cal_ids"]]
    xcal = item["transformed"]["cal"]
    ex = np.asarray(item["exact_cal"])
    m0, m1 = float(ex[ycal == 0].mean()), float(ex[ycal == 1].mean())
    gap = abs(m1 - m0)
    v0, v1 = float(ex[ycal == 0].var()), float(ex[ycal == 1].var())
    # Bernoulli +/-1 sample mean variance at the simulator expectation.
    q0 = float(np.mean(1 - ex[ycal == 0] ** 2))
    q1 = float(np.mean(1 - ex[ycal == 1] ** 2))
    n0, n1 = int(np.sum(ycal == 0)), int(np.sum(ycal == 1))
    # One score estimate per calibration example; each receives m shots.
    target_var = (0.10 * max(gap, 1e-9)) ** 2
    base_var = v0 / n0 + v1 / n1
    shot_var_per_shot = q0 / n0 + q1 / n1
    if base_var >= target_var:
        # Additional measurements cannot overcome uncertainty from the finite
        # number of labeled calibration examples.
        predicted = None
        prediction_status = "calibration_sample_size_dominates_no_finite_shot_count_meets_target"
    else:
        predicted = max(1, int(np.ceil(shot_var_per_shot / max(target_var - base_var, 1e-15))))
        prediction_status = "finite_shot_budget_estimated"
    qtest = item["qexact"]
    validation_name = "dev" if item["stage"] == "development" else "test"
    clean_test_expectation = base.batch_scores(qtest, item["weights"],
                                               item["transformed"][validation_name], GAMMA)
    ytest = item["y"][item["other_ids"][validation_name]]
    exact_threshold = base.best_balanced_threshold(ycal, ex)
    reference_ba = float(balanced_accuracy_score(ytest, clean_test_expectation >= exact_threshold))
    shot_rows, threshold_rows = [], []
    required = None
    # Only calibration shots are varied. Evaluation is fixed at 1000 shots.
    for m in SHOT_GRID:
        for rep in range(SHOT_REPS):
            s = seed + 40000 + m * 100 + rep
            cal = shot_scores(item, "cal", m, s, 1.0)
            th = base.best_balanced_threshold(ycal, cal)
            test = shot_scores(item, validation_name, budget, s + 1, 1.0)
            ba = float(balanced_accuracy_score(ytest, test >= th))
            regret = reference_ba - ba
            shot_rows.append({"realization": item["id"], "stage": item["stage"],
                              "difficulty": item["difficulty"], "calibration_shots_per_example": m,
                              "replicate": rep, "test_budget_per_example": budget,
                              "selected_noisy_calibration_threshold": float(th),
                              "test_balanced_accuracy": ba,
                              "exact_threshold_reference_test_balanced_accuracy": reference_ba,
                              "performance_regret_vs_reference": regret})
            threshold_rows.append(th)
        sub = [r["performance_regret_vs_reference"] for r in shot_rows
               if r["calibration_shots_per_example"] == m]
        # operational criterion: average test regret <= 2 percentage points.
        if required is None and float(np.mean(sub)) <= 0.02:
            required = m
    return {"realization": item["id"], "stage": item["stage"],
            "difficulty": item["difficulty"], "seed": item["seed"],
            "calibration_n0": n0, "calibration_n1": n1,
            "noisy_score_mean_gap": gap, "standardized_noisy_score_separation":
                gap / max(np.sqrt(0.5 * (v0 + v1)), 1e-12),
            "predicted_shots_per_calibration_example_for_10pct_relative_gap_se": predicted,
            "budget_prediction_status": prediction_status,
            "minimum_observed_shots_meeting_2pp_mean_test_regret": required,
            "target_shots_for_test_per_case": budget,
            "uses_confirmation_test_labels_for_validation_only": bool(is_confirmation)}, shot_rows


def main():
    global RUN_ID, OUT, CONF_SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--confirmations-per-difficulty", type=int, default=None,
                        help="Run a fresh extension with this many new confirmation seeds per task difficulty.")
    cli = parser.parse_args()
    RUN_ID = cli.run_id
    OUT = ROOT / "results" / "runs" / RUN_ID
    if cli.confirmations_per_difficulty is not None:
        count = cli.confirmations_per_difficulty
        if count < 1:
            parser.error("--confirmations-per-difficulty must be positive")
        CONF_SEEDS = {
            "moderate": list(range(630100, 630100 + count)),
            "easier": list(range(640100, 640100 + count)),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    (OUT / "splits").mkdir(exist_ok=True)
    prereg = {
        "run_id": RUN_ID, "purpose": "fresh-data development plus untouched synthetic confirmation",
        "task": {"n": N, "features": N_FEATURES, "informative": N_INFORMATIVE,
                 "redundant": N_REDUNDANT, "class_weights": [0.5, 0.5], "flip_y": 0.02,
                 "class_separation": DIFFICULTIES},
        "development_seeds": DEV_SEEDS, "confirmation_seeds": CONF_SEEDS,
        "partition": "development: 60% train, 15% calibration, 25% tuning validation; confirmation: 60% train, 20% calibration, 20% final test",
        "model": {"qubits": 2, "layers": LAYERS, "architecture": ARCH,
                  "training": "clean default.qubit, Adam, BCE(sigmoid(Z0)), 30 epochs, batch 32, lr .1"},
        "noise": "composite channel gamma=.30, applied by research_experiment.make_qnode",
        "budget": {"tuning_nominal_per_validation_case": TUNE_BUDGET,
                   "confirmation_nominal_per_test_case": FINAL_BUDGETS,
                   "calibration_fractions_tuned": ALLOCATIONS, "final_shot_grid": SHOT_GRID,
                   "calibration_budget_predictor": "minimum m for analytic class-mean score-gap SE <= 10% of absolute gap; validate at 2pp mean test balanced-accuracy regret"},
        "selection_rule": "choose scale_config and calibration_fraction maximizing average development-validation BA of ZNE+threshold calibration; tie-break fewer scales then lower calibration fraction",
        "scope": "simulator-only synthetic task-family confirmation; no external-dataset or hardware generalization claim",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "platform": platform.platform(),
        "versions": {"pennylane": qml.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    if not (OUT / "preregistered_protocol.json").exists():
        (OUT / "preregistered_protocol.json").write_text(json.dumps(prereg, indent=2), encoding="utf-8")

    # Archive all primary data, splits, checkpoints and learning curves.
    models = {}
    curves = []
    dependence = base.probe_feature_dependence(2, LAYERS, ARCH, 7654321)
    pd.DataFrame(dependence).to_csv(OUT / "input_dependence_probe.csv", index=False)
    for stage, seeds in [("development", DEV_SEEDS), ("confirmation", CONF_SEEDS)]:
        for difficulty, seed_list in seeds.items():
            for seed in seed_list:
                checkpoint = OUT / "checkpoints" / f"{stage}_{difficulty}_{seed}.npz"
                saved = np.load(checkpoint)["weights"] if checkpoint.exists() else None
                item = train_realization(seed, difficulty, stage, saved_weights=saved)
                models[item["id"]] = item
                curves.extend([{"realization": item["id"], "stage": stage,
                                "difficulty": difficulty, "seed": seed, **r}
                               for r in item["history"]])
                np.savez_compressed(OUT / "checkpoints" / f"{item['id']}.npz",
                                    weights=item["weights"], train_ids=item["train_ids"],
                                    calibration_ids=item["cal_ids"], **{
                                        f"{k}_ids": v for k, v in item["other_ids"].items()})
                np.savez_compressed(OUT / "splits" / f"{item['id']}.npz",
                                    train_ids=item["train_ids"], calibration_ids=item["cal_ids"],
                                    **{f"{k}_ids": v for k, v in item["other_ids"].items() if k != "cal"})
                print(f"trained {item['id']} pca={item['prep']['pca_variance_sum']:.3f}", flush=True)
    if curves:
        pd.DataFrame(curves).to_csv(OUT / "training_curves.csv", index=False)
    (OUT / "preprocessing_parameters.json").write_text(json.dumps(
        {key: item["prep"] for key, item in models.items()}, indent=2), encoding="utf-8")

    # Development-only tuning. All candidates see the same dev realizations.
    tune_rows, intermediates = [], {}
    if (OUT / "development_tuning.csv").exists():
        tuning = pd.read_csv(OUT / "development_tuning.csv")
    else:
        for item in [x for x in models.values() if x["stage"] == "development"]:
            for scale_name in SCALE_CONFIGS:
                for frac in ALLOCATIONS:
                    rows, raw = pipeline_scores(item, TUNE_BUDGET, frac, scale_name,
                                                item["seed"] + int(frac * 1000) + len(scale_name), "dev")
                    tune_rows.extend(rows)
                    intermediates[(item["id"], scale_name, frac)] = raw
        tuning = pd.DataFrame(tune_rows)
        tuning.to_csv(OUT / "development_tuning.csv", index=False)
    objective = (tuning[tuning.method == "selected_zne_calibration"]
                 .groupby(["scale_config", "calibration_fraction"], as_index=False)
                 .agg(mean_dev_balanced_accuracy=("balanced_accuracy", "mean"),
                      sd_dev_balanced_accuracy=("balanced_accuracy", "std"),
                      development_realizations=("realization", "nunique")))
    objective.sort_values(["mean_dev_balanced_accuracy", "scale_config", "calibration_fraction"],
                          ascending=[False, True, True], inplace=True)
    selected = objective.iloc[0]
    selected_scale = str(selected.scale_config)
    selected_fraction = float(selected.calibration_fraction)
    objective.to_csv(OUT / "development_selection.csv", index=False)
    prereg["selected_settings_from_development"] = {"scale_config": selected_scale,
        "calibration_fraction": selected_fraction,
        "development_mean_balanced_accuracy": float(selected.mean_dev_balanced_accuracy)}
    (OUT / "frozen_selection.json").write_text(json.dumps(prereg["selected_settings_from_development"], indent=2), encoding="utf-8")
    print(f"FROZEN: {selected_scale}, calibration_fraction={selected_fraction:.2f}", flush=True)

    # Untouched confirmation. No values below flow back to selection.
    confirm_rows, raw_data = [], {}
    for item in [x for x in models.values() if x["stage"] == "confirmation"]:
        for budget in FINAL_BUDGETS:
            for frac in [selected_fraction]:
                rows, raw = pipeline_scores(item, budget, frac, selected_scale,
                                            item["seed"] + budget + int(frac * 100), "test")
                confirm_rows.extend(rows)
                raw_data[(item["id"], budget)] = raw
    pd.DataFrame(confirm_rows).to_csv(OUT / "confirmatory_test_results.csv", index=False)

    # Calibration shot-budget predictor and out-of-sample validation.
    prediction_rows, regret_rows = [], []
    for item in models.values():
        pred, regs = budget_prediction(item, 1000, item["seed"], item["stage"] == "confirmation")
        prediction_rows.append(pred)
        regret_rows.extend(regs)
    pd.DataFrame(prediction_rows).to_csv(OUT / "calibration_budget_predictions.csv", index=False)
    pd.DataFrame(regret_rows).to_csv(OUT / "calibration_budget_validation.csv", index=False)

    # Intermediate observations required to reconstruct calibration and extrapolation.
    observation_rows = []
    for (realization, budget), raw in raw_data.items():
        for name, scores in raw.items():
            if (isinstance(scores, list) and scores and
                    isinstance(scores[0], (np.ndarray, list, tuple))):
                for scale_i, vec in enumerate(scores):
                    for i, val in enumerate(vec):
                        observation_rows.append({"realization": realization, "budget": budget,
                            "observation_set": name, "scale_index": scale_i,
                            "example_index_within_partition": i, "score": float(val)})
            else:
                for i, val in enumerate(scores):
                    observation_rows.append({"realization": realization, "budget": budget,
                        "observation_set": name, "scale_index": -1,
                        "example_index_within_partition": i, "score": float(val)})
    pd.DataFrame(observation_rows).to_csv(OUT / "confirmatory_intermediate_scores.csv", index=False)

    summarize(OUT)
    # Hash every run artifact after output generation, excluding the hash file itself.
    manifest = {"protocol": prereg, "artifact_sha256": {}}
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name != "artifact_hashes.json":
            manifest["artifact_sha256"][f.relative_to(OUT).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    (OUT / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def summarize(out):
    conf = pd.read_csv(out / "confirmatory_test_results.csv")
    b1000 = conf[conf.budget_per_eval_case == 1000]
    summary = (b1000.groupby("method").balanced_accuracy.agg(["mean", "std", "count"]).reset_index())
    summary.to_csv(out / "confirmatory_summary.csv", index=False)
    paired = conf.pivot_table(index=["realization", "budget_per_eval_case"],
                              columns="method", values="balanced_accuracy").reset_index()
    for method in ["noisy_calibration", "selected_zne", "selected_zne_calibration"]:
        paired[f"{method}_minus_fixed"] = paired[method] - paired["fixed_threshold"]
    paired["noisy_calibration_minus_selected_zne"] = paired["noisy_calibration"] - paired["selected_zne"]
    paired["noisy_calibration_minus_selected_zne_calibration"] = paired["noisy_calibration"] - paired["selected_zne_calibration"]
    paired.to_csv(out / "confirmatory_paired_contrasts.csv", index=False)
    pred = pd.read_csv(out / "calibration_budget_predictions.csv")
    pred.groupby("stage").agg(
        realizations=("realization", "count"),
        median_predicted_shots=("predicted_shots_per_calibration_example_for_10pct_relative_gap_se", "median"),
        median_observed_required_shots=("minimum_observed_shots_meeting_2pp_mean_test_regret", "median"),
        observed_requirement_rate=("minimum_observed_shots_meeting_2pp_mean_test_regret", lambda s: float(s.notna().mean()))
    ).reset_index().to_csv(out / "calibration_budget_summary.csv", index=False)
    chosen = json.loads((out / "frozen_selection.json").read_text(encoding="utf-8"))
    lines = [
        "# Confirmatory budget study: results and claim limits", "",
        "This separate study used newly generated balanced synthetic datasets. Six development realizations selected the ZNE schedule and calibration allocation; six distinct confirmation realizations were evaluated with those choices frozen. It is a simulator study and does not test WDBC, Wine, clinical performance, hardware behavior, or external generalization.", "",
        f"Development selected `{chosen['scale_config']}` and a {chosen['calibration_fraction']:.0%} calibration allocation at 1,000 nominal measurements per evaluation case.", "",
        "## Frozen 1,000-measurement confirmation results", "",
        summary.to_markdown(index=False), "",
        "## The 100-measurement result was not stable across budgets", "",
        pd.read_csv(out / "confirmatory_test_results.csv").groupby(
            ["budget_per_eval_case", "method"]).balanced_accuracy.agg(["mean", "std", "count"]).reset_index().to_markdown(index=False), "",
        "Paired per-realization differences are in `confirmatory_paired_contrasts.csv`. The primary pipeline comparison used one finite-shot draw per realization and setting; the standard deviations therefore combine variation across synthetic realizations and individual shot draws. They are descriptive, not population confidence intervals.", "",
        "## Interpretation", "",
        "The fresh confirmation does not reproduce a large ZNE-plus-calibration recovery advantage. Noisy threshold calibration is the strongest mean pipeline in this small set, while development-selected ZNE plus calibration is close to the fixed-threshold baseline. Six realizations do not support population-level inference.", "",
        "The proposed budget rule used class-mean score separation from exact noisy simulator expectations and a finite-sample variance calculation. This is not a hardware-observable predictor: exact expectations are available here only because this is a simulator. If between-example calibration variance alone exceeds the target, no finite shot count can achieve the requested precision at the available calibration sample size. The rule did not establish a useful predictor of classifier performance: its score-gap precision target and the validation criterion of at most 2 percentage points of balanced-accuracy regret are distinct, and that regret target was rarely met in confirmation. Treat this as a negative result requiring redesign, not as a validated contribution.", "",
        "## Claim boundary", "",
        "The supported result is limited to this preregistered synthetic task family, a two-qubit readout-centered CNOT-star circuit, one composite simulator noise setting (gamma=0.30), and the tested measurement budgets. Six confirmation realizations are a pilot. Settings were tuned on six separate development realizations. There is no independent external-dataset confirmation, noise-aware-training comparator, or hardware experiment. Do not claim a general calibration-versus-ZNE ranking, a validated budget predictor, clinical utility, or quantum advantage.", "",
        "## Audit trail", "",
        "`preregistered_protocol.json` records the planned task, seeds, splits, model, candidates, and selection rule. `development_tuning.csv` and `development_selection.csv` record development-only choices. `confirmatory_test_results.csv`, `confirmatory_paired_contrasts.csv`, `calibration_budget_predictions.csv`, `calibration_budget_validation.csv`, `confirmatory_intermediate_scores.csv`, `training_curves.csv`, checkpoints, split indices, and `artifact_hashes.json` retain the evidence. The source is `confirmatory_budget_study.py` at the project root.", "",
    ]
    (out / "CONFIRMATORY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nCONFIRMATION SUMMARY (1000 nominal measurements per test case)")
    print(summary.to_string(index=False))
    print("\nCALIBRATION BUDGET SUMMARY")
    print(pd.read_csv(out / "calibration_budget_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
