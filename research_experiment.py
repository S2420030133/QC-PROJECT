"""Reproducible pilot for threshold calibration versus ZNE in small VQCs.

All noise probabilities in this script are per-channel-location probabilities;
they are not interpreted as a circuit-level error rate or as T1/T2 times.
The experiments are simulator studies, not hardware validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import pickle
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import sklearn
import autograd.numpy as anp
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CHECKPOINTS = OUT / "checkpoints"
SPLITS = OUT / "splits"
OUT.mkdir(exist_ok=True)
CHECKPOINTS.mkdir(exist_ok=True)
SPLITS.mkdir(exist_ok=True)

GAMMAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
NOISES = ["amplitude_damping", "phase_damping", "depolarizing", "composite"]
SCALE_LINEAR = [1.0, 2.0]
SCALE_QUADRATIC = [1.0, 1.5, 2.0]
COEF_LINEAR = np.array([2.0, -1.0])
COEF_QUADRATIC = np.array([6.0, -8.0, 3.0])


@dataclass(frozen=True)
class ModelSpec:
    task: str
    split: int
    train_seed: int
    n_qubits: int = 2
    n_layers: int = 2
    architecture: str = "cnot_chain"
    pca_standardize: bool = False
    reverse_labels: bool = False

    @property
    def key(self) -> str:
        return (f"{self.task}_s{self.split}_i{self.train_seed}_q{self.n_qubits}_"
                f"d{self.n_layers}_{self.architecture}_std{int(self.pca_standardize)}_"
                f"rev{int(self.reverse_labels)}")


def tasks() -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    bc = load_breast_cancer()
    wine = load_wine()
    # Reproducible synthetic binary tasks; features are generated independently
    # of the held-out splits and exercise balanced and imbalanced prevalences.
    xb, yb = make_classification(n_samples=600, n_features=12, n_informative=6,
                                 n_redundant=2, n_clusters_per_class=2,
                                 weights=[0.5, 0.5], class_sep=1.0, random_state=1729)
    xi, yi = make_classification(n_samples=600, n_features=12, n_informative=6,
                                 n_redundant=2, n_clusters_per_class=2,
                                 weights=[0.8, 0.2], class_sep=1.0, random_state=1730)
    # Wine target 0 versus all remaining wine classes; stated explicitly as a
    # binary benchmark derived from the standard three-class dataset.
    yw = (wine.target != 0).astype(int)
    return {
        "wdbc": (bc.data.astype(float), bc.target.astype(int), "Wisconsin Diagnostic Breast Cancer; class 1 is benign"),
        "wine_0_vs_rest": (wine.data.astype(float), yw, "Wine class 0 versus classes 1/2 (binary reduction)"),
        "synthetic_balanced": (xb.astype(float), yb.astype(int), "Synthetic balanced binary task"),
        "synthetic_imbalanced": (xi.astype(float), yi.astype(int), "Synthetic 80:20 binary task"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_indices(y: np.ndarray, n_splits: int, seed: int, task: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for k, (train_dev, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
        train, cal = train_test_split(train_dev, test_size=0.25, random_state=seed + 1009 * (k + 1),
                                      stratify=y[train_dev])
        out.append((train, cal, test))
        np.savez_compressed(SPLITS / f"{task}_split{k}.npz", train=train, calibration=cal, test=test)
    return out


def prepare_features(X: np.ndarray, train: np.ndarray, cal: np.ndarray, test: np.ndarray,
                     n_qubits: int, pca_standardize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    standardizer = StandardScaler().fit(X[train]) if pca_standardize else None
    x_train = standardizer.transform(X[train]) if standardizer else X[train]
    x_cal = standardizer.transform(X[cal]) if standardizer else X[cal]
    x_test = standardizer.transform(X[test]) if standardizer else X[test]
    pca = PCA(n_components=n_qubits, svd_solver="full").fit(x_train)
    z_train = pca.transform(x_train)
    z_cal = pca.transform(x_cal)
    z_test = pca.transform(x_test)
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(z_train)
    return scaler.transform(z_train), scaler.transform(z_cal), scaler.transform(z_test), {
        "standardizer": standardizer, "pca": pca, "scaler": scaler,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
    }


def apply_noise(gamma: float, wires: list[int], kind: str) -> None:
    if kind == "amplitude_damping":
        for wire in wires:
            qml.AmplitudeDamping(gamma, wires=wire)
    elif kind == "phase_damping":
        for wire in wires:
            qml.PhaseDamping(gamma, wires=wire)
    elif kind == "depolarizing":
        for wire in wires:
            qml.DepolarizingChannel(gamma, wires=wire)
    elif kind == "composite":
        # Preserve the original synthetic schedule: amplitude damping gamma
        # and depolarizing probability gamma/2 at every declared location.
        for wire in wires:
            qml.DepolarizingChannel(gamma / 2.0, wires=wire)
            qml.AmplitudeDamping(gamma, wires=wire)
    else:
        raise ValueError(f"Unknown noise kind {kind}")


def make_qnode(n_qubits: int, n_layers: int, architecture: str, noise_kind: str,
               shots: int | None, seed: int):
    dev = qml.device("default.mixed", wires=n_qubits, seed=seed)

    @qml.qnode(dev)
    def circuit(weights, x, gamma):
        qml.AngleEmbedding(features=x, wires=range(n_qubits), rotation="X")
        for q in range(n_qubits):
            apply_noise(gamma, [q], noise_kind)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                if noise_kind == "composite":
                    apply_noise(gamma / 2.0, [q], "depolarizing")
                elif noise_kind in ("phase_damping", "depolarizing"):
                    apply_noise(gamma, [q], noise_kind)
                qml.RZ(weights[layer, q, 1], wires=q)
                if noise_kind == "composite":
                    apply_noise(gamma / 2.0, [q], "depolarizing")
                elif noise_kind in ("phase_damping", "depolarizing"):
                    apply_noise(gamma, [q], noise_kind)
            if architecture == "cnot_chain":
                edges = [(q, q + 1) for q in range(n_qubits - 1)]
            elif architecture == "cnot_star":
                edges = [(q, 0) for q in range(1, n_qubits)]
            elif architecture == "cz_ring":
                edges = [(q, q + 1) for q in range(n_qubits - 1)]
                if n_qubits > 2:
                    edges.append((n_qubits - 1, 0))
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            for a, b in edges:
                if architecture in ("cnot_chain", "cnot_star"):
                    qml.CNOT(wires=[a, b])
                else:
                    qml.CZ(wires=[a, b])
                # Two-qubit gate noise is applied independently to both wires.
                if noise_kind == "composite":
                    apply_noise(gamma, [a, b], "composite")
                else:
                    apply_noise(gamma, [a, b], noise_kind)
        return qml.expval(qml.PauliZ(0))
    return qml.set_shots(circuit, shots=shots) if shots is not None else circuit


def make_clean_qnode(n_qubits: int, n_layers: int, architecture: str):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(weights, x):
        qml.AngleEmbedding(features=x, wires=range(n_qubits), rotation="X")
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)
            if architecture == "cnot_star":
                edges = [(q, 0) for q in range(1, n_qubits)]
            else:
                edges = [(q, q + 1) for q in range(n_qubits - 1)]
            if architecture == "cz_ring" and n_qubits > 2:
                edges.append((n_qubits - 1, 0))
            for a, b in edges:
                if architecture in ("cnot_chain", "cnot_star"):
                    qml.CNOT(wires=[a, b])
                else:
                    qml.CZ(wires=[a, b])
        return qml.expval(qml.PauliZ(0))
    return circuit


def probe_feature_dependence(n_qubits: int, n_layers: int, architecture: str,
                             seed: int, weight_trials: int = 4,
                             input_trials: int = 32, perturbation: float = 0.2) -> list[dict]:
    """Probe feature-to-readout influence on random inputs/weights before training."""
    rng = np.random.default_rng(seed)
    qnode = make_clean_qnode(n_qubits, n_layers, architecture)
    rows = []
    for feature in range(n_qubits):
        changes = []
        for trial in range(weight_trials):
            weights = rng.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 2))
            inputs = rng.uniform(-np.pi, np.pi, size=(input_trials, n_qubits))
            for x in inputs:
                baseline = float(qnode(weights, x))
                perturbed = x.copy()
                perturbed[feature] = np.clip(perturbed[feature] + perturbation, -np.pi, np.pi)
                changes.append(abs(float(qnode(weights, perturbed)) - baseline))
        rows.append({"n_qubits": n_qubits, "n_layers": n_layers, "architecture": architecture,
                     "feature_index": feature, "mean_absolute_readout_change": float(np.mean(changes)),
                     "median_absolute_readout_change": float(np.median(changes)),
                     "weight_trials": weight_trials, "inputs_per_trial": input_trials,
                     "perturbation_radians": perturbation,
                     "performed_before_training": True})
    return rows


def train_model(X: np.ndarray, y: np.ndarray, n_qubits: int, n_layers: int,
                architecture: str, seed: int, epochs: int, batch_size: int,
                learning_rate: float) -> tuple[np.ndarray, list[dict]]:
    qml.numpy.random.seed(seed)
    xq = qml.numpy.array(X, requires_grad=False)
    yq = np.asarray(y, dtype=float)
    circuit = make_clean_qnode(n_qubits, n_layers, architecture)
    weights = qml.numpy.array(np.random.default_rng(seed).uniform(-np.pi, np.pi,
                            size=(n_layers, n_qubits, 2)), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=learning_rate)

    def loss(w, xb, yb):
        logits = circuit(w, xb)
        p = 1.0 / (1.0 + anp.exp(-logits))
        p = anp.clip(p, 1e-7, 1.0 - 1e-7)
        return -anp.mean(yb * anp.log(p) + (1.0 - yb) * anp.log(1.0 - p))

    fit_ids, dev_ids = train_test_split(np.arange(len(yq)), test_size=0.15,
                                        random_state=seed + 31415, stratify=yq)
    xfit, yfit = xq[fit_ids], yq[fit_ids]
    xdev, ydev = xq[dev_ids], yq[dev_ids]
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(xfit))
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            weights = opt.step(lambda w: loss(w, xfit[ids], yfit[ids]), weights)
        train_loss = float(loss(weights, xfit, yfit))
        dev_loss = float(loss(weights, xdev, ydev))
        dev_scores = np.asarray([float(circuit(weights, x)) for x in xdev])
        history.append({"epoch": epoch + 1, "optimizer_updates": int(np.ceil(len(xfit) / batch_size) * (epoch + 1)),
                        "training_binary_cross_entropy": train_loss,
                        "development_binary_cross_entropy": dev_loss,
                        "development_balanced_accuracy_at_zero": float(
                            balanced_accuracy_score(ydev, dev_scores >= 0.0)),
                        "training_examples": int(len(xfit)), "development_examples": int(len(xdev))})
    return np.asarray(weights, dtype=float), history


def batch_scores(qnode, weights: np.ndarray, X: np.ndarray, gamma: float) -> np.ndarray:
    try:
        scores = np.asarray(qnode(weights, X, gamma), dtype=float)
        if scores.shape == (len(X),):
            return scores
    except (ValueError, NotImplementedError, qml.exceptions.DeviceError):
        pass
    # Fallback for devices that do not support batched mixed-state execution.
    return np.asarray([float(qnode(weights, x, gamma)) for x in X], dtype=float)


def metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.0) -> dict:
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    try:
        auc = float(roc_auc_score(y, score))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "recall_class_0": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "recall_class_1": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": auc,
        "predicted_class_1_fraction": float(pred.mean()),
        "single_class_collapse": bool(len(np.unique(pred)) == 1),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def best_balanced_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    # Shot estimates and Richardson sums can contain floating-point near-ties.
    # Round well below the finite-shot score resolution so identical estimates
    # remain tied and threshold selection is reproducible from saved CSV values.
    scores = np.round(np.asarray(scores, dtype=float), 12)
    candidates = np.unique(np.r_[scores, 0.0])
    vals = np.array([balanced_accuracy_score(y, scores >= t) for t in candidates])
    best = np.flatnonzero(vals == vals.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best]))]])


def fit_affine_readout(cal_scores: np.ndarray, y_cal: np.ndarray,
                       test_scores: np.ndarray) -> tuple[LogisticRegression, np.ndarray]:
    """Fit an affine score-to-logit map after calibration-only standardization."""
    score_scaler = StandardScaler().fit(np.asarray(cal_scores).reshape(-1, 1))
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(score_scaler.transform(np.asarray(cal_scores).reshape(-1, 1)), y_cal)
    decision = model.decision_function(score_scaler.transform(np.asarray(test_scores).reshape(-1, 1)))
    model.score_scaler_ = score_scaler
    return model, np.asarray(decision, dtype=float)


def optimal_shot_allocation(total: int, coefficients: np.ndarray) -> list[int]:
    """Neyman allocation proportional to |coefficient| under +/-1 outcomes."""
    weights = np.abs(coefficients) / np.abs(coefficients).sum()
    raw = weights * total
    alloc = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - alloc))[:total - int(alloc.sum())]:
        alloc[i] += 1
    return [max(1, int(v)) for v in alloc]


def finite_zne_scores(weights: np.ndarray, X: np.ndarray, gamma: float, total_shots: int,
                      scales: list[float], coefficients: np.ndarray, n_qubits: int,
                      n_layers: int, architecture: str, seed: int) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    allocations = optimal_shot_allocation(total_shots, coefficients)
    scale_scores = []
    for j, (scale, shots) in enumerate(zip(scales, allocations)):
        qnode = make_qnode(n_qubits, n_layers, architecture, "composite", shots, seed + j)
        scale_scores.append(batch_scores(qnode, weights, X, gamma * scale))
    combined = np.round(np.sum(np.asarray(coefficients)[:, None] * np.vstack(scale_scores), axis=0), 12)
    return combined, allocations, scale_scores


def evaluate_matched_shot_pipelines(spec: ModelSpec, weights: np.ndarray,
                                    xcal: np.ndarray, xtest: np.ndarray,
                                    ycal: np.ndarray, ytest: np.ndarray,
                                    clean_cal: np.ndarray, clean_test: np.ndarray,
                                    cal_ids: np.ndarray,
                                    test_ids: np.ndarray, qexact, args
                                    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Finite-shot pipelines with explicit clean/noisy calibration and ZNE calibration."""
    shot_rows, exact_rows, test_rows, audit_rows = [], [], [], []
    gammas = [0.1, 0.2, 0.3, 0.4]
    for gamma in gammas:
        exact_base = batch_scores(qexact, weights, xtest, gamma)
        exact_cal_noisy = batch_scores(qexact, weights, xcal, gamma)
        clean_threshold = best_balanced_threshold(ycal, clean_cal)
        noisy_threshold = best_balanced_threshold(ycal, exact_cal_noisy)
        exact_lin = np.round(sum(c * batch_scores(qexact, weights, xtest, gamma * s)
                                 for s, c in zip(SCALE_LINEAR, COEF_LINEAR)), 12)
        exact_quad = np.round(sum(c * batch_scores(qexact, weights, xtest, gamma * s)
                                  for s, c in zip(SCALE_QUADRATIC, COEF_QUADRATIC)), 12)
        exact_lin_cal = np.round(sum(c * batch_scores(qexact, weights, xcal, gamma * s)
                                     for s, c in zip(SCALE_LINEAR, COEF_LINEAR)), 12)
        exact_quad_cal = np.round(sum(c * batch_scores(qexact, weights, xcal, gamma * s)
                                      for s, c in zip(SCALE_QUADRATIC, COEF_QUADRATIC)), 12)
        exact_lin_threshold = best_balanced_threshold(ycal, exact_lin_cal)
        exact_quad_threshold = best_balanced_threshold(ycal, exact_quad_cal)
        exact_affine_model, exact_affine_scores = fit_affine_readout(exact_cal_noisy, ycal, exact_base)
        exact_methods = {
            "fixed_threshold": (exact_base, 0.0),
            "clean_frozen_threshold": (exact_base, clean_threshold),
            "calibrated_threshold": (exact_base, noisy_threshold),
            "affine_readout": (exact_affine_scores, 0.0),
            "linear_zne": (exact_lin, 0.0),
            "quadratic_zne": (exact_quad, 0.0),
            "linear_zne_calibrated": (exact_lin, exact_lin_threshold),
            "quadratic_zne_calibrated": (exact_quad, exact_quad_threshold),
        }
        for method, (scores, threshold) in exact_methods.items():
            if method in {"clean_frozen_threshold", "linear_zne", "quadratic_zne",
                          "linear_zne_calibrated", "quadratic_zne_calibrated"}:
                exact_rows.append({"model_id": spec.key, "task": spec.task, "split": spec.split,
                                   "train_seed": spec.train_seed, "noise": "composite", "gamma": gamma,
                                   "method": method, "threshold": threshold,
                                   "calibration_threshold": threshold,
                                   "score_mae_vs_clean": float(np.mean(np.abs(scores - clean_test))),
                                   "score_rmse_vs_clean": float(np.sqrt(np.mean((scores - clean_test) ** 2))),
                                   **metrics(ytest, scores, threshold)})
        for budget in args.shot_budgets:
            for rep in range(args.shot_repetitions):
                seed = args.seed + 100000 * spec.split + 1000 * spec.train_seed + rep
                cal_total = max(2, int(budget) // 2)
                test_total = int(budget) - cal_total
                cal_shots = max(1, int(round(cal_total * len(xtest) / len(xcal))))
                test_shots = max(1, test_total)

                qfixed = make_qnode(spec.n_qubits, spec.n_layers, spec.architecture,
                                    "composite", int(budget), seed + 1)
                fixed_scores = batch_scores(qfixed, weights, xtest, gamma)
                qtest = make_qnode(spec.n_qubits, spec.n_layers, spec.architecture,
                                   "composite", test_shots, seed + 2)
                noisy_test = batch_scores(qtest, weights, xtest, gamma)
                qclean_cal = make_qnode(spec.n_qubits, spec.n_layers, spec.architecture,
                                        "composite", cal_shots, seed + 3)
                clean_cal_shot_scores = batch_scores(qclean_cal, weights, xcal, 0.0)
                clean_frozen_threshold = best_balanced_threshold(ycal, clean_cal_shot_scores)
                qcal = make_qnode(spec.n_qubits, spec.n_layers, spec.architecture,
                                  "composite", cal_shots, seed + 4)
                noisy_cal_scores = batch_scores(qcal, weights, xcal, gamma)
                noisy_cal_threshold = best_balanced_threshold(ycal, noisy_cal_scores)
                affine_model, affine_test_scores = fit_affine_readout(noisy_cal_scores, ycal, noisy_test)

                methods = {
                    "fixed_threshold": (fixed_scores, 0.0, fixed_scores, 0, int(budget) * len(xtest)),
                    "clean_frozen_threshold": (noisy_test, clean_frozen_threshold, noisy_test,
                                               cal_shots * len(xcal), test_shots * len(xtest)),
                    "calibrated_threshold": (noisy_test, noisy_cal_threshold, noisy_test,
                                             cal_shots * len(xcal), test_shots * len(xtest)),
                    "affine_readout": (affine_test_scores, 0.0, noisy_test,
                                       cal_shots * len(xcal), test_shots * len(xtest)),
                }
                cal_thresholds = {"clean_frozen_threshold": clean_frozen_threshold,
                                  "calibrated_threshold": noisy_cal_threshold}
                exact_thresholds = {"clean_frozen_threshold": clean_threshold,
                                    "calibrated_threshold": noisy_threshold}
                for name, score, threshold, partition_scores, scale, shots_each in [
                    ("clean_calibration", clean_cal_shot_scores, clean_frozen_threshold, clean_cal_shot_scores, 0.0, cal_shots),
                    ("noisy_calibration", noisy_cal_scores, noisy_cal_threshold, noisy_cal_scores, gamma, cal_shots),
                ]:
                    for ix, yt, value in zip(cal_ids, ycal, score):
                        audit_rows.append({"model_id": spec.key, "task": spec.task, "split": spec.split,
                                           "train_seed": spec.train_seed, "gamma": gamma, "budget": int(budget),
                                           "replicate": rep, "pipeline": name, "partition": "calibration",
                                           "example_index": int(ix), "y_true": int(yt), "score": float(value),
                                           "selected_threshold": float(threshold), "scale": scale,
                                           "shots_per_example": int(shots_each)})
                base_exact = {"fixed_threshold": exact_base, "clean_frozen_threshold": exact_base,
                              "calibrated_threshold": exact_base, "affine_readout": exact_affine_scores}
                for method, (score, threshold, observable, cal_cost, test_cost) in methods.items():
                    exact_target = base_exact[method]
                    affine_params = {}
                    if method == "affine_readout":
                        affine_params = {"affine_intercept": float(affine_model.intercept_[0]),
                                         "affine_coefficient": float(affine_model.coef_[0, 0]),
                                         "calibration_score_mean": float(affine_model.score_scaler_.mean_[0]),
                                         "calibration_score_scale": float(affine_model.score_scaler_.scale_[0])}
                    shot_rows.append({"model_id": spec.key, "task": spec.task, "split": spec.split,
                                      "train_seed": spec.train_seed, "gamma": gamma,
                                      "shots_per_test_case": int(budget), "replicate": rep,
                                      "method": method, "selected_threshold": float(threshold),
                                      "measurement_shots_test": int(test_cost),
                                      "measurement_shots_calibration": int(cal_cost),
                                      "measurement_shots_total": int(test_cost + cal_cost),
                                      **affine_params, **metrics(ytest, score, threshold),
                                      "score_mae_vs_clean": float(np.mean(np.abs(observable - clean_test))),
                                      "score_rmse_vs_exact_method": float(np.sqrt(np.mean((score - exact_target) ** 2))),
                                      "observable_rmse_vs_exact": float(np.sqrt(np.mean((observable - exact_base) ** 2)))})
                    for j, (ix, yt, value) in enumerate(zip(test_ids, ytest, score)):
                        test_rows.append({"model_id": spec.key, "task": spec.task, "split": spec.split,
                                          "train_seed": spec.train_seed, "noise": "composite", "gamma": gamma,
                                          "shots_per_test_case": int(budget), "replicate": rep,
                                          "example_index": int(ix), "y_true_model_label": int(yt),
                                          "method": method, "score": float(value),
                                          "raw_observable_score": float(observable[j]),
                                          "clean_score": float(clean_test[j]), "prediction": int(value >= threshold),
                                          "partition": "test"})
                for method, scales, coeffs, exact_zne_cal, exact_threshold in [
                    ("linear_zne", SCALE_LINEAR, COEF_LINEAR, exact_lin_cal, exact_lin_threshold),
                    ("quadratic_zne", SCALE_QUADRATIC, COEF_QUADRATIC, exact_quad_cal, exact_quad_threshold),
                ]:
                    test_scores, test_alloc, test_scale_scores = finite_zne_scores(
                        weights, xtest, gamma, int(budget), scales, coeffs, spec.n_qubits,
                        spec.n_layers, spec.architecture, seed + 100)
                    exact_score, exact_threshold_zero = exact_methods[method]
                    shot_rows.append({"model_id": spec.key, "task": spec.task, "split": spec.split,
                                      "train_seed": spec.train_seed, "gamma": gamma,
                                      "shots_per_test_case": int(budget), "replicate": rep,
                                      "method": method, "selected_threshold": 0.0,
                                      "measurement_shots_test": int(budget) * len(xtest),
                                      "measurement_shots_calibration": 0,
                                      "measurement_shots_total": int(budget) * len(xtest),
                                      "shot_allocation_per_scale": json.dumps(test_alloc),
                                      "variance_amplification_factor_vs_same_budget": float(np.sum(np.abs(coeffs)) ** 2),
                                      **metrics(ytest, test_scores, 0.0),
                                      "score_mae_vs_clean": float(np.mean(np.abs(test_scores - clean_test))),
                                      "score_rmse_vs_exact_method": float(np.sqrt(np.mean((test_scores - exact_score) ** 2))),
                                      "observable_rmse_vs_exact": float(np.sqrt(np.mean((test_scores - exact_score) ** 2)))})
                    for scale, alloc, values in zip(scales, test_alloc, test_scale_scores):
                        for ix, yt, value in zip(test_ids, ytest, values):
                            audit_rows.append({"model_id": spec.key, "task": spec.task, "split": spec.split,
                                               "train_seed": spec.train_seed, "gamma": gamma, "budget": int(budget),
                                               "replicate": rep, "pipeline": method, "partition": "test_scale",
                                               "example_index": int(ix), "y_true": int(yt), "score": float(value),
                                               "selected_threshold": 0.0, "scale": scale,
                                               "shots_per_example": int(alloc)})
                    cal_zne_scores, cal_alloc, cal_scale_scores = finite_zne_scores(
                        weights, xcal, gamma, cal_shots, scales, coeffs, spec.n_qubits,
                        spec.n_layers, spec.architecture, seed + 200)
                    selected = best_balanced_threshold(ycal, cal_zne_scores)
                    half_test_scores, test_alloc_half, test_scale_half = finite_zne_scores(
                        weights, xtest, gamma, test_shots, scales, coeffs, spec.n_qubits,
                        spec.n_layers, spec.architecture, seed + 300)
                    calibrated_method = method + "_calibrated"
                    exact_calibrated_method = method + "_calibrated"
                    exact_target, exact_cal_threshold = exact_methods[exact_calibrated_method]
                    cal_cost = int(sum(cal_alloc) * len(xcal))
                    test_cost = int(sum(test_alloc_half) * len(xtest))
                    shot_rows.append({"model_id": spec.key, "task": spec.task, "split": spec.split,
                                      "train_seed": spec.train_seed, "gamma": gamma,
                                      "shots_per_test_case": int(budget), "replicate": rep,
                                      "method": calibrated_method, "selected_threshold": float(selected),
                                      "measurement_shots_test": test_cost,
                                      "measurement_shots_calibration": cal_cost,
                                      "measurement_shots_total": test_cost + cal_cost,
                                      "shot_allocation_per_scale_test": json.dumps(test_alloc_half),
                                      "shot_allocation_per_scale_calibration": json.dumps(cal_alloc),
                                      "variance_amplification_factor_vs_same_budget": float(np.sum(np.abs(coeffs)) ** 2),
                                      **metrics(ytest, half_test_scores, selected),
                                      "score_mae_vs_clean": float(np.mean(np.abs(half_test_scores - clean_test))),
                                      "score_rmse_vs_exact_method": float(np.sqrt(np.mean((half_test_scores - exact_target) ** 2))),
                                      "observable_rmse_vs_exact": float(np.sqrt(np.mean((half_test_scores - exact_target) ** 2)))})
                    for scale, alloc, values in zip(scales, cal_alloc, cal_scale_scores):
                        for ix, yt, value in zip(cal_ids, ycal, values):
                            audit_rows.append({"model_id": spec.key, "task": spec.task, "split": spec.split,
                                               "train_seed": spec.train_seed, "gamma": gamma, "budget": int(budget),
                                               "replicate": rep, "pipeline": calibrated_method,
                                               "partition": "calibration_scale", "example_index": int(ix),
                                               "y_true": int(yt), "score": float(value),
                                               "selected_threshold": float(selected), "scale": scale,
                                               "shots_per_example": int(alloc)})
                    for scale, alloc, values in zip(scales, test_alloc_half, test_scale_half):
                        for ix, yt, value in zip(test_ids, ytest, values):
                            audit_rows.append({"model_id": spec.key, "task": spec.task, "split": spec.split,
                                               "train_seed": spec.train_seed, "gamma": gamma, "budget": int(budget),
                                               "replicate": rep, "pipeline": calibrated_method,
                                               "partition": "test_scale", "example_index": int(ix),
                                               "y_true": int(yt), "score": float(value),
                                               "selected_threshold": float(selected), "scale": scale,
                                               "shots_per_example": int(alloc)})
                    for j, (ix, yt, value) in enumerate(zip(test_ids, ytest, half_test_scores)):
                        test_rows.append({"model_id": spec.key, "task": spec.task, "split": spec.split,
                                          "train_seed": spec.train_seed, "noise": "composite", "gamma": gamma,
                                          "shots_per_test_case": int(budget), "replicate": rep,
                                          "example_index": int(ix), "y_true_model_label": int(yt),
                                          "method": calibrated_method, "score": float(value),
                                          "raw_observable_score": float(value), "clean_score": float(clean_test[j]),
                                          "prediction": int(value >= selected), "partition": "test"})
                    for ix, yt, value in zip(cal_ids, ycal, cal_zne_scores):
                        audit_rows.append({"model_id": spec.key, "task": spec.task, "split": spec.split,
                                           "train_seed": spec.train_seed, "gamma": gamma, "budget": int(budget),
                                           "replicate": rep, "pipeline": calibrated_method,
                                           "partition": "calibration_combined", "example_index": int(ix),
                                           "y_true": int(yt), "score": float(value),
                                           "selected_threshold": float(selected), "scale": "combined",
                                           "shots_per_example": int(sum(cal_alloc))})
    for row in shot_rows + exact_rows:
        row.update({"n_qubits": spec.n_qubits, "n_layers": spec.n_layers,
                    "architecture": spec.architecture, "pca_standardize": spec.pca_standardize})
    return shot_rows, exact_rows, test_rows, audit_rows


def classical_baselines(X_train: np.ndarray, y_train: np.ndarray, X_cal: np.ndarray,
                        X_test: np.ndarray, y_cal: np.ndarray, y_test: np.ndarray,
                        task: str, split: int, test_ids: np.ndarray) -> tuple[list[dict], list[dict]]:
    out, per_example = [], []
    # Matched two-feature baseline gets the same 2-D quantum embedding.
    for name, model in [
        ("logistic_regression_2d", LogisticRegression(max_iter=3000, random_state=41)),
        ("rbf_svm_2d", SVC(kernel="rbf", probability=True, random_state=41)),
        ("random_forest_2d", RandomForestClassifier(n_estimators=300, random_state=41)),
    ]:
        model.fit(X_train, y_train)
        score = model.predict_proba(X_test)[:, 1]
        row = {"task": task, "split": split, "model": name, **metrics(y_test, score, 0.5)}
        out.append(row)
        per_example.extend({"task": task, "split": split, "model": name,
                            "example_index": int(ix), "y_true": int(yt),
                            "score": float(s), "prediction": int(s >= 0.5)}
                           for ix, yt, s in zip(test_ids, y_test, score))
    return out, per_example


def fit_one(spec: ModelSpec, X: np.ndarray, y: np.ndarray, split_data, args,
            run_arch_sweep: bool = False, reuse_checkpoint_dir: Path | None = None):
    train_ids, cal_ids, test_ids = split_data
    labels = (1 - y) if spec.reverse_labels else y
    xtr, xcal, xtest, prep = prepare_features(X, train_ids, cal_ids, test_ids,
                                              spec.n_qubits, spec.pca_standardize)
    ytr, ycal, ytest = labels[train_ids], labels[cal_ids], labels[test_ids]
    cached_weights = (reuse_checkpoint_dir / f"{spec.key}.npy") if reuse_checkpoint_dir else None
    if cached_weights is not None and cached_weights.exists():
        weights = np.load(cached_weights)
        train_history = []
    else:
        weights, train_history = train_model(xtr, ytr, spec.n_qubits, spec.n_layers, spec.architecture,
                                             args.seed + spec.split * 100 + spec.train_seed,
                                             args.epochs, args.batch_size, args.learning_rate)
    training_curve_rows = [{"model_id": spec.key, "task": spec.task, "split": spec.split,
                            "train_seed": spec.train_seed, **record} for record in train_history]
    np.save(CHECKPOINTS / f"{spec.key}.npy", weights)
    with open(CHECKPOINTS / f"{spec.key}_preprocessing.json", "w", encoding="utf-8") as f:
        json.dump({"spec": asdict(spec), **{k: v for k, v in prep.items() if k.startswith("pca_")}}, f, indent=2)
    # Pickle fitted transformers and indices so transforms can be reconstructed.
    import joblib
    joblib.dump({"standardizer": prep["standardizer"], "pca": prep["pca"], "scaler": prep["scaler"],
                 "train_ids": train_ids, "calibration_ids": cal_ids, "test_ids": test_ids},
                CHECKPOINTS / f"{spec.key}_preprocessing.joblib")

    clean_q = make_clean_qnode(spec.n_qubits, spec.n_layers, spec.architecture)
    clean_train = np.array([float(clean_q(weights, x)) for x in xtr])
    clean_cal = np.array([float(clean_q(weights, x)) for x in xcal])
    clean_test = np.array([float(clean_q(weights, x)) for x in xtest])
    if spec.reverse_labels:
        # These are model-coordinate labels; provide the original-label mapping explicitly.
        pass
    modelrow = {"model_id": spec.key, "task": spec.task, "split": spec.split,
                "train_seed": spec.train_seed, "n_qubits": spec.n_qubits,
                "n_layers": spec.n_layers, "architecture": spec.architecture,
                "pca_standardize": spec.pca_standardize, "reverse_labels": spec.reverse_labels,
                "optimizer_epochs": args.epochs, "optimizer_batch_size": args.batch_size,
                "optimizer_updates": train_history[-1]["optimizer_updates"] if train_history else
                                     int(np.ceil(len(ytr) / args.batch_size) * args.epochs),
                "weights_reused_from_checkpoint": bool(cached_weights is not None and cached_weights.exists()),
                "pca_explained_variance_sum": prep["pca_explained_variance_sum"],
                **{f"clean_{k}": v for k, v in metrics(ytest, clean_test).items()}}
    exact_rows, per_example_rows = [], []
    qnodes = {noise: make_qnode(spec.n_qubits, spec.n_layers, spec.architecture, noise, None, args.seed)
              for noise in NOISES}
    saved_exact = {}
    main_architecture = (spec.n_qubits == 2 and spec.n_layers == 2 and
                         spec.architecture == "cnot_chain" and not spec.pca_standardize and
                         not spec.reverse_labels)
    exact_noise_names = NOISES if main_architecture else ["composite"]
    exact_gammas = GAMMAS if main_architecture else [0.0, 0.20, 0.30, 0.40]
    for noise in exact_noise_names:
        for gamma in exact_gammas:
            stest = batch_scores(qnodes[noise], weights, xtest, gamma)
            scal = batch_scores(qnodes[noise], weights, xcal, gamma)
            sclean = clean_test
            # Oracle ranking/thresholded behavior at the exact expectation level.
            threshold = 0.0
            direct = metrics(ytest, stest, threshold)
            cal_threshold = best_balanced_threshold(ycal, scal)
            calibrated = metrics(ytest, stest, cal_threshold)
            affine, affine_scores = fit_affine_readout(scal, ycal, stest)
            affine_row = metrics(ytest, affine_scores, 0.0)
            common = {"model_id": spec.key, "task": spec.task, "split": spec.split,
                      "train_seed": spec.train_seed, "noise": noise, "gamma": gamma,
                      "calibration_threshold": cal_threshold,
                      "score_mae_vs_clean": float(np.mean(np.abs(stest - sclean))),
                      "score_rmse_vs_clean": float(np.sqrt(np.mean((stest - sclean) ** 2)))}
            exact_rows.append({**common, "method": "fixed_threshold", **direct})
            exact_rows.append({**common, "method": "calibrated_threshold", **calibrated})
            exact_rows.append({**common, "method": "affine_readout", **affine_row})
            for j, (yt, raw, clean) in enumerate(zip(ytest, stest, sclean)):
                for method, value, pred in [
                    ("fixed_threshold", raw, int(raw >= 0.0)),
                    ("calibrated_threshold", raw, int(raw >= cal_threshold)),
                    ("affine_readout", float(affine_scores[j]), int(affine_scores[j] >= 0.0)),
                ]:
                    per_example_rows.append({**common, "example_index": int(test_ids[j]),
                                             "y_true_model_label": int(yt), "method": method,
                                             "score": value, "clean_score": float(clean),
                                             "prediction": pred})
            saved_exact[(noise, gamma)] = {"test": stest, "cal": scal}
    shot_rows, audit_rows = [], []
    canonical_tasks = {"wdbc", "wine_0_vs_rest", "synthetic_balanced"}
    canonical_2q = (spec.task in canonical_tasks and spec.n_qubits == 2 and spec.n_layers == 2 and
                    spec.architecture == "cnot_chain" and not spec.pca_standardize and not spec.reverse_labels)
    broader_verified = (spec.task == "wdbc" and spec.n_qubits > 2 and spec.n_layers == 2 and
                        spec.architecture == "cnot_star" and
                        f"{spec.n_qubits}:{spec.n_layers}:{spec.architecture}" in
                        getattr(args, "verified_architectures", []))
    if canonical_2q or broader_verified:
        qexact = qnodes["composite"]
        finite, exact_comparison, test_observations, audits = evaluate_matched_shot_pipelines(
            spec, weights, xcal, xtest, ycal, ytest, clean_cal, clean_test,
            cal_ids, test_ids, qexact, args)
        shot_rows.extend(finite)
        exact_rows.extend(exact_comparison)
        per_example_rows.extend(test_observations)
        audit_rows.extend(audits)
    for row in exact_rows:
        row.setdefault("n_qubits", spec.n_qubits)
        row.setdefault("n_layers", spec.n_layers)
        row.setdefault("architecture", spec.architecture)
        row.setdefault("pca_standardize", spec.pca_standardize)
    return [modelrow], exact_rows, shot_rows, per_example_rows, audit_rows, training_curve_rows


def run_classical(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per_example = [], []
    data = tasks()
    for task, (X, y, _) in data.items():
        splits = split_indices(y, args.outer_splits, args.seed, task)
        for split, (train, cal, test) in enumerate(splits):
            # Same train/cal/test partition, full feature baselines; feature
            # standardization is fitted on model-training data only.
            std = StandardScaler().fit(X[train])
            xt, xc, xs = std.transform(X[train]), std.transform(X[cal]), std.transform(X[test])
            for name, model in [
                ("logistic_regression_full", LogisticRegression(max_iter=3000, random_state=41)),
                ("rbf_svm_full", SVC(kernel="rbf", probability=True, random_state=41)),
                ("random_forest_full", RandomForestClassifier(n_estimators=300, random_state=41)),
            ]:
                model.fit(xt, y[train])
                score = model.predict_proba(xs)[:, 1]
                rows.append({"task": task, "split": split, "model": name,
                             **metrics(y[test], score, 0.5)})
                per_example.extend({"task": task, "split": split, "model": name,
                                    "example_index": int(ix), "y_true": int(yt),
                                    "score": float(s), "prediction": int(s >= 0.5)}
                                   for ix, yt, s in zip(test, y[test], score))
            # matched 2-feature classical estimators use the same train-fitted
            # PCA/scaler transform as the two-qubit quantum model.
            x2tr, x2cal, x2te, _ = prepare_features(X, train, cal, test, 2, False)
            matched, matched_individual = classical_baselines(x2tr, y[train], x2cal, x2te,
                                                              y[cal], y[test], task, split, test)
            rows.extend(matched)
            per_example.extend(matched_individual)
    return pd.DataFrame(rows), pd.DataFrame(per_example)


def run(args) -> None:
    global OUT, CHECKPOINTS, SPLITS
    run_id = args.run_id or (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" +
                             hashlib.sha256(f"{time.time_ns()}".encode()).hexdigest()[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ValueError("--run-id must be 1-64 letters, digits, dots, underscores, or hyphens")
    OUT = ROOT / "results" / "runs" / run_id
    if OUT.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists; choose a new --run-id: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS = OUT / "checkpoints"
    SPLITS = OUT / "splits"
    CHECKPOINTS.mkdir(exist_ok=True)
    SPLITS.mkdir(exist_ok=True)
    PARTIALS = OUT / "partial_model_results"
    PARTIALS.mkdir(exist_ok=True)
    reuse_dir = (ROOT / "results" / "runs" / args.reuse_checkpoints_from / "checkpoints"
                 if args.reuse_checkpoints_from else None)
    t0 = time.time()
    data = tasks()
    # Core factorial: multiple datasets, outer splits, and training initializations.
    specs: list[ModelSpec] = []
    for task in data:
        for split in range(args.outer_splits):
            for train_seed in range(args.train_seeds):
                specs.append(ModelSpec(task, split, train_seed))
    # Targeted architecture/depth/feature-scaling and label-mapping diagnostics
    # on WDBC, limited to the configured number of outer folds.
    for split in range(min(args.arch_sweep_splits, args.outer_splits)):
        specs.extend([
            ModelSpec("wdbc", split, 0, 2, 1, "cnot_chain"),
            ModelSpec("wdbc", split, 0, 4, 2, "cnot_chain"),
            ModelSpec("wdbc", split, 0, 6, 2, "cnot_chain"),
            ModelSpec("wdbc", split, 0, 2, 2, "cz_ring"),
            ModelSpec("wdbc", split, 0, 2, 2, "cnot_chain", True),
            ModelSpec("wdbc", split, 0, 2, 2, "cnot_chain", False, True),
            *[ModelSpec("wdbc", split, seed, 4, 2, "cnot_star")
              for seed in range(args.train_seeds)],
        ])
    # Evaluate the feature-to-readout pathway on random circuit weights before
    # training. Include the 4-qubit CNOT star across every outer split if verified.
    for split in range(args.outer_splits):
        for train_seed in range(args.train_seeds):
            specs.append(ModelSpec("wdbc", split, train_seed, 4, 2, "cnot_star"))
    specs = list({spec.key: spec for spec in specs}.values())
    probe_configs = sorted({(s.n_qubits, s.n_layers, s.architecture) for s in specs if s.n_qubits > 2})
    probe_rows = []
    for nq, depth, architecture in probe_configs:
        probe_rows.extend(probe_feature_dependence(nq, depth, architecture, args.seed + 1000 + nq))
    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(OUT / "input_dependence_probe.csv", index=False)
    verified_architectures = []
    for (nq, depth, architecture), group in probe_df.groupby(["n_qubits", "n_layers", "architecture"]):
        if len(group) == nq and (group.mean_absolute_readout_change > 1e-4).all():
            verified_architectures.append(f"{nq}:{depth}:{architecture}")
    args.verified_architectures = verified_architectures
    model_rows, exact_rows, shot_rows, per_example_rows, audit_rows, training_curve_rows = [], [], [], [], [], []
    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] training/evaluating {spec.key}", flush=True)
        X, y, _ = data[spec.task]
        splits = split_indices(y, args.outer_splits, args.seed, spec.task)
        partial_path = PARTIALS / f"{spec.key}.pkl"
        if args.resume and partial_path.exists():
            with partial_path.open("rb") as f:
                mr, er, sr, pr, ar, tr = pickle.load(f)
        else:
            mr, er, sr, pr, ar, tr = fit_one(spec, X, y, splits[spec.split], args,
                                              reuse_checkpoint_dir=reuse_dir)
            # A completed per-model result is durable; interrupted runs can resume
            # without repeating any finished simulator evaluation.
            with partial_path.open("wb") as f:
                pickle.dump((mr, er, sr, pr, ar, tr), f, protocol=pickle.HIGHEST_PROTOCOL)
        model_rows.extend(mr)
        exact_rows.extend(er)
        shot_rows.extend(sr)
        per_example_rows.extend(pr)
        audit_rows.extend(ar)
        training_curve_rows.extend(tr)
    pd.DataFrame(model_rows).to_csv(OUT / "models.csv", index=False)
    exact_df = pd.DataFrame(exact_rows)
    exact_df.to_csv(OUT / "exact_performance.csv", index=False)
    shot_df = pd.DataFrame(shot_rows)
    shot_df.to_csv(OUT / "finite_shot_performance.csv", index=False)
    pd.DataFrame(per_example_rows).to_csv(OUT / "per_example_observations.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "calibration_and_scale_observations.csv", index=False)
    pd.DataFrame(training_curve_rows).to_csv(OUT / "training_curves.csv", index=False)
    classical, classical_per_example = run_classical(args)
    classical.to_csv(OUT / "classical_baselines.csv", index=False)
    classical_per_example.to_csv(OUT / "classical_per_example_observations.csv", index=False)
    manifest = {
        "run_id": run_id,
        "study": "Threshold calibration versus zero-noise extrapolation in variational quantum classifiers",
        "status": "computational pilot; results are not hardware measurements",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version, "platform": platform.platform(),
        "versions": {"pennylane": qml.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__},
        "config": vars(args),
        "datasets": {k: {"description": desc, "n_samples": int(len(X)), "n_features": int(X.shape[1]),
                         "class_counts": np.bincount(y).tolist(),
                         "array_sha256": hashlib.sha256(np.ascontiguousarray(X).tobytes() +
                                                         np.ascontiguousarray(y).tobytes()).hexdigest()}
                     for k, (X, y, desc) in data.items()},
        "noise_model": {
            "probabilities_are_per_channel_location": True,
            "amplitude_damping": "AmplitudeDamping(gamma) after encoding rotations and entanglers; aligned with the amplitude component of the composite schedule",
            "phase_damping": "PhaseDamping(gamma) after encoding rotations, variational rotations, and entanglers",
            "depolarizing": "DepolarizingChannel(gamma) after encoding rotations, variational rotations, and entanglers",
            "composite": "AmplitudeDamping(gamma) after encoding rotations and entanglers plus DepolarizingChannel(gamma/2) after encoding, variational, and entangling gates; synthetic channel model",
            "not_a_claim": "No gate durations, T1/T2 parameters, device calibration, or hardware runs are modeled",
        },
        "methods": {"fixed_threshold": "Pauli-Z score threshold 0; sigmoid omitted since it is monotone",
                    "clean_frozen_threshold": "threshold fit on a finite-shot clean calibration partition and frozen across noisy settings",
                    "calibrated_threshold": "balanced-accuracy threshold selected on finite-shot noisy calibration scores",
                    "affine_readout": "calibration-only standardization of raw score followed by L2 logistic affine readout (C=1); predictions are fit on calibration partition",
                    "linear_zne_calibrated": "linear ZNE score with threshold selected on independent noisy ZNE calibration examples",
                    "quadratic_zne_calibrated": "quadratic ZNE score with threshold selected on independent noisy ZNE calibration examples",
                    "linear_zne": {"scales": SCALE_LINEAR, "coefficients": COEF_LINEAR.tolist()},
                    "quadratic_zne": {"scales": SCALE_QUADRATIC, "coefficients": COEF_QUADRATIC.tolist(),
                                      "same_total_shot_optimal_allocation_variance_factor": int(np.sum(np.abs(COEF_QUADRATIC)) ** 2)}},
        "zne_noise_scaling": "Simulator-only channel-probability scaling: multiply every declared composite-channel probability by the ZNE scale; no gate folding or hardware noise scaling is implemented.",
        "partition": "Stratified K-fold outer validation with disjoint test folds; each fold uses 60% model training, 20% calibration, 20% outer test; all transforms fit on model training only",
        "seeds": {"base": args.seed, "training_initializations": list(range(args.train_seeds)),
                  "shot_repetitions": args.shot_repetitions},
        "architecture_input_dependence_probe": {"before_training": True,
            "verified_architectures": verified_architectures,
            "criterion": "Every input feature's mean absolute readout change exceeds 1e-4 under random weights and 0.2 radian perturbations"},
        "limitations": ["Finite-shot mitigation comparisons cover WDBC, binary Wine, balanced synthetic, and the input-probed 4-qubit CNOT star only.",
                        "The K outer test folds are disjoint, but they share overlapping training data; split-clustered intervals are not population-level independent-sample confidence intervals.",
                        "No independent hardware validation is available.",
                        "WDBC is a benchmark dataset, not evidence of clinical diagnostic validity."],
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "legacy_checkpoint": {"path": "trained_weights.npy", "used_in_this_run": False,
                              "note": "Retained from the earlier prototype; its provenance was not established."},
        "elapsed_seconds": time.time() - t0,
        "reproduction_note": ("Evaluation reused the 54 model checkpoints from an interrupted reproduction attempt; "
                              "remaining checkpoints were trained in this run. Per-model simulator outputs were "
                              "computed anew. Training histories are unavailable for reused checkpoints." if reuse_dir else None),
    }
    summarize_and_plot()
    # Partial pickles are internal resume state, not research artifacts.
    for p in PARTIALS.glob("*.pkl"):
        p.unlink()
    PARTIALS.rmdir()
    manifest["artifact_sha256"] = {p.relative_to(ROOT).as_posix(): sha256_file(p)
                                   for p in sorted(OUT.rglob("*")) if p.is_file() and
                                   p.name != "run_manifest.json" and
                                   p.suffix.lower() != ".zip" and
                                   p.name != "independent_audit.json"}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results saved under {OUT}; elapsed {time.time() - t0:.1f}s", flush=True)


def summarize_and_plot() -> None:
    exact_path = OUT / "exact_performance.csv"
    finite_path = OUT / "finite_shot_performance.csv"
    if not exact_path.exists() or not finite_path.exists():
        return
    exact = pd.read_csv(exact_path)
    exact = exact[(exact.noise == "composite") &
                  exact.method.isin(["fixed_threshold", "clean_frozen_threshold", "calibrated_threshold",
                                     "affine_readout", "linear_zne", "quadratic_zne",
                                     "linear_zne_calibrated", "quadratic_zne_calibrated"])].copy()
    core_model = exact.model_id.str.endswith("_q2_d2_cnot_chain_std0_rev0")
    verified_model = exact.model_id.str.endswith("_q4_d2_cnot_star_std0_rev0")
    exact = exact[core_model | verified_model].copy()
    finite_all = pd.read_csv(finite_path)
    exact_long = finite_all[finite_all.shots_per_test_case.astype(str) == "exact"]
    if not exact_long.empty:
        exact_long = exact_long.copy()
        exact_long["value"] = pd.to_numeric(exact_long["value"], errors="coerce")
        exact_zne = exact_long.pivot_table(index=["model_id", "task", "split", "train_seed", "gamma", "method"],
                                           columns="quantity", values="value").reset_index()
        exact_zne["noise"] = "composite"
        if "score_rmse_vs_clean" not in exact_zne:
            exact_zne["score_rmse_vs_clean"] = np.nan
        exact_zne = exact_zne[exact_zne.method.isin(["linear_zne", "quadratic_zne"])]
        exact = pd.concat([exact, exact_zne], ignore_index=True, sort=False)
    agg = exact.groupby(["task", "n_qubits", "n_layers", "architecture", "gamma", "method"], as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"), roc_auc=("roc_auc", "mean"),
        collapse_rate=("single_class_collapse", "mean"), score_rmse=("score_rmse_vs_clean", "mean"))
    agg.to_csv(OUT / "exact_summary.csv", index=False)
    channel_rows = pd.read_csv(exact_path)
    channel_rows = channel_rows[(channel_rows.task == "wdbc") &
                                (channel_rows.method == "fixed_threshold") &
                                channel_rows.model_id.str.endswith("_q2_d2_cnot_chain_std0_rev0")].copy()
    channel_rows.groupby(["noise", "gamma"], as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"), roc_auc=("roc_auc", "mean"),
        collapse_rate=("single_class_collapse", "mean"), model_runs=("model_id", "nunique")
    ).to_csv(OUT / "channel_ablation_summary.csv", index=False)
    finite = finite_all.copy()
    finite = finite[(finite.method.isin(["fixed_threshold", "clean_frozen_threshold", "calibrated_threshold",
                                         "affine_readout", "linear_zne", "quadratic_zne",
                                         "linear_zne_calibrated", "quadratic_zne_calibrated"])) &
                    (finite.model_id.str.endswith("_q2_d2_cnot_chain_std0_rev0") |
                     finite.model_id.str.endswith("_q4_d2_cnot_star_std0_rev0")) &
                    finite.shots_per_test_case.ne("exact")].copy()
    finite["shots_per_test_case"] = pd.to_numeric(finite.shots_per_test_case)
    if not finite.empty:
        finite.groupby(["task", "n_qubits", "n_layers", "architecture", "gamma",
                        "shots_per_test_case", "method"], as_index=False).agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            auc_mean=("roc_auc", "mean"),
            collapse_rate=("single_class_collapse", "mean"),
            score_mae_mean=("score_mae_vs_clean", "mean"),
            sampling_rmse_mean=("score_rmse_vs_exact_method", "mean"),
            observable_rmse_mean=("observable_rmse_vs_exact", "mean"),
            total_shots_mean=("measurement_shots_total", "mean")).to_csv(OUT / "finite_shot_summary.csv", index=False)
        observable_methods = ["fixed_threshold", "clean_frozen_threshold", "calibrated_threshold",
                              "linear_zne", "quadratic_zne", "linear_zne_calibrated",
                              "quadratic_zne_calibrated"]
        error_rows = finite[(finite.task == "wdbc") & (finite.n_qubits == 2) &
                            (finite.architecture == "cnot_chain") & (finite.gamma == 0.3) &
                            finite.method.isin(observable_methods) &
                            (finite.shots_per_test_case == 1000)].groupby("method", as_index=False).agg(
            finite_sampling_rmse_vs_exact=("observable_rmse_vs_exact", "mean"))
        exact_error = exact[(exact.task == "wdbc") & (exact.n_qubits == 2) &
                            (exact.architecture == "cnot_chain") & (exact.gamma == 0.3) &
                            exact.method.isin(observable_methods)].groupby("method", as_index=False).agg(
            exact_score_rmse_vs_clean=("score_rmse_vs_clean", "mean"))
        error_rows = error_rows.merge(exact_error, on="method", how="left")
        error_rows.to_csv(OUT / "error_decomposition_summary.csv", index=False)
    model_info = pd.read_csv(OUT / "models.csv")
    all_exact = pd.read_csv(exact_path)
    collapse = all_exact[(all_exact.noise == "composite") & (all_exact.method == "fixed_threshold")]
    if not collapse.empty:
        # Exact rows already carry circuit metadata. Merge only the label-map
        # flag, avoiding pandas suffixes that would hide n_qubits/architecture.
        collapse = collapse.merge(model_info[["model_id", "reverse_labels"]], on="model_id", how="left")
        collapse_summary = collapse.groupby(["task", "n_qubits", "n_layers", "architecture", "pca_standardize",
                                             "reverse_labels", "gamma"], as_index=False).agg(
            collapse_rate=("single_class_collapse", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
            roc_auc=("roc_auc", "mean"), predicted_class_1_fraction=("predicted_class_1_fraction", "mean"),
            model_runs=("model_id", "nunique"))
        collapse_summary.to_csv(OUT / "collapse_rate_by_model.csv", index=False)
    # Paired contrasts aggregate all repeats and initializations within each
    # outer split, then bootstrap split clusters.
    paired = []
    paired_fold_rows = []
    exact_methods = ["clean_frozen_threshold", "calibrated_threshold", "affine_readout",
                     "linear_zne", "quadratic_zne", "linear_zne_calibrated",
                     "quadratic_zne_calibrated"]
    for (task, nq, architecture), group in exact.groupby(["task", "n_qubits", "architecture"]):
        wide = group.pivot_table(index=["model_id", "split", "gamma"], columns="method",
                                 values="balanced_accuracy").reset_index()
        for method in exact_methods:
            if method not in wide or "fixed_threshold" not in wide:
                continue
            tmp = wide[["split", "gamma", method, "fixed_threshold"]].dropna().copy()
            tmp["difference"] = tmp[method] - tmp["fixed_threshold"]
            for gamma, g in tmp.groupby("gamma"):
                split_means = g.groupby("split").difference.mean().to_numpy()
                if not len(split_means):
                    continue
                rng = np.random.default_rng(991 + int(round(gamma * 1000)) + nq + sum(map(ord, method)))
                boot = np.array([rng.choice(split_means, len(split_means), replace=True).mean()
                                 for _ in range(10000)])
                paired.append({"analysis": "exact", "task": task, "n_qubits": nq,
                               "architecture": architecture, "gamma": gamma,
                               "shots_per_test_case": "exact", "method": method,
                               "reference": "fixed_threshold", "metric": "balanced_accuracy",
                               "mean_paired_difference": float(split_means.mean()),
                               "ci95_low": float(np.quantile(boot, .025)),
                               "ci95_high": float(np.quantile(boot, .975)),
                               "outer_splits": int(len(split_means))})
    if not finite.empty:
        fw = finite.pivot_table(index=["model_id", "task", "n_qubits", "architecture", "split", "gamma",
                                       "shots_per_test_case", "replicate"],
                                columns="method", values="balanced_accuracy").reset_index()
        finite_methods = ["clean_frozen_threshold", "calibrated_threshold", "affine_readout",
                          "linear_zne", "quadratic_zne", "linear_zne_calibrated",
                          "quadratic_zne_calibrated"]
        for method in finite_methods:
            if method not in fw or "fixed_threshold" not in fw:
                continue
            tmp = fw[["task", "n_qubits", "architecture", "split", "gamma",
                      "shots_per_test_case", "replicate", method, "fixed_threshold"]].dropna().copy()
            tmp["difference"] = tmp[method] - tmp["fixed_threshold"]
            for (task, nq, architecture, gamma, budget), g in tmp.groupby(
                    ["task", "n_qubits", "architecture", "gamma", "shots_per_test_case"]):
                split_means = g.groupby("split").difference.mean().to_numpy()
                if not len(split_means):
                    continue
                rng = np.random.default_rng(1991 + int(round(gamma * 1000)) + int(budget) + nq)
                boot = np.array([rng.choice(split_means, len(split_means), replace=True).mean()
                                 for _ in range(10000)])
                paired.append({"analysis": "finite_shot", "task": task, "n_qubits": nq,
                               "architecture": architecture, "gamma": gamma,
                               "shots_per_test_case": int(budget), "method": method,
                               "reference": "fixed_threshold", "metric": "balanced_accuracy",
                               "mean_paired_difference": float(split_means.mean()),
                               "ci95_low": float(np.quantile(boot, .025)),
                               "ci95_high": float(np.quantile(boot, .975)),
                               "outer_splits": int(len(split_means))})
        references = ["clean_frozen_threshold", "calibrated_threshold", "affine_readout"]
        zne_methods = ["linear_zne", "quadratic_zne", "linear_zne_calibrated", "quadratic_zne_calibrated"]
        for reference in references:
            for method in zne_methods:
                if reference not in fw or method not in fw:
                    continue
                tmp = fw[["task", "n_qubits", "architecture", "split", "gamma",
                          "shots_per_test_case", "replicate", method, reference]].dropna().copy()
                tmp["difference"] = tmp[method] - tmp[reference]
                for (task, nq, architecture, gamma, budget), g in tmp.groupby(
                        ["task", "n_qubits", "architecture", "gamma", "shots_per_test_case"]):
                    split_means = g.groupby("split").difference.mean().to_numpy()
                    if not len(split_means):
                        continue
                    for split, fold in g.groupby("split"):
                        paired_fold_rows.append({"analysis":"finite_shot","task":task,"n_qubits":nq,
                            "architecture":architecture,"gamma":gamma,"shots_per_test_case":int(budget),
                            "method":method,"reference":reference,"metric":"balanced_accuracy",
                            "split":int(split),"mean_paired_difference":float(fold.difference.mean())})
                    rng = np.random.default_rng(2991 + int(round(gamma * 1000)) + int(budget) + nq
                                                + sum(ord(c) for c in reference + method))
                    boot = np.array([rng.choice(split_means, len(split_means), replace=True).mean()
                                     for _ in range(10000)])
                    paired.append({"analysis": "finite_shot", "task": task, "n_qubits": nq,
                                   "architecture": architecture, "gamma": gamma,
                                   "shots_per_test_case": int(budget), "method": method,
                                   "reference": reference, "metric": "balanced_accuracy",
                                   "mean_paired_difference": float(split_means.mean()),
                                   "ci95_low": float(np.quantile(boot, .025)),
                                   "ci95_high": float(np.quantile(boot, .975)),
                                   "outer_splits": int(len(split_means))})
    pd.DataFrame(paired).to_csv(OUT / "paired_differences_bootstrap_ci.csv", index=False)
    pd.DataFrame(paired_fold_rows).to_csv(OUT / "paired_fold_differences.csv", index=False)
    # Main figure: exact score ranking versus thresholded performance for WDBC.
    w = agg[(agg.task == "wdbc") & (agg.n_qubits == 2) & (agg.architecture == "cnot_chain")]
    if not w.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for method, frame in w.groupby("method"):
            axes[0].plot(frame.gamma, frame.balanced_accuracy, marker="o", label=method)
            axes[1].plot(frame.gamma, frame.roc_auc, marker="o", label=method)
        axes[0].set(title="WDBC: balanced accuracy", xlabel="Per-location noise probability γ", ylabel="Balanced accuracy", ylim=(0, 1.02))
        axes[1].set(title="WDBC: score ranking", xlabel="Per-location noise probability γ", ylabel="ROC-AUC", ylim=(0, 1.02))
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        fig.savefig(OUT / "exact_accuracy_auc.png", dpi=180)
        plt.close(fig)
    if not finite.empty:
        sub = finite[(finite.task == "wdbc") & (finite.n_qubits == 2) &
                     (finite.architecture == "cnot_chain") & finite.gamma.isin([0.2, 0.3, 0.4])]
        if not sub.empty:
            f, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
            for ax, gamma in zip(axes, [0.2, 0.3, 0.4]):
                panel = sub[sub.gamma == gamma]
                for method, frame in panel.groupby("method"):
                    g = frame.groupby("shots_per_test_case").balanced_accuracy
                    means = g.mean()
                    sds = g.std().fillna(0)
                    # Only two budgets were tested; markers and error bars avoid
                    # implying a continuous trend between them.
                    ax.errorbar(means.index, means.values, yerr=sds.values, marker="o",
                                linestyle="none", label=method)
                ax.set_xscale("log")
                ax.set_ylim(0, 1.02)
                ax.set(title=f"γ={gamma:.1f}", xlabel="Total shots per test case")
                ax.legend(fontsize=7)
            axes[0].set_ylabel("Balanced accuracy")
            f.suptitle("WDBC finite-shot comparison (two budgets; mean ± SD)")
            f.savefig(OUT / "finite_shots_budget.png", dpi=180)
            plt.close(f)
    # Representative score distributions make threshold movement visible; the
    # figure is explicitly one frozen split/run, while aggregate metrics above
    # include all saved model runs.
    examples_path = OUT / "per_example_observations.csv"
    if examples_path.exists():
        usecols = ["model_id", "task", "noise", "gamma", "shots_per_test_case", "method",
                   "y_true_model_label", "score"]
        samples = pd.read_csv(examples_path, usecols=usecols, low_memory=False)
        primary_id = "wdbc_s0_i0_q2_d2_cnot_chain_std0_rev0"
        panels = []
        for gamma in [0.0, 0.3, 0.4]:
            part = samples[(samples.model_id == primary_id) & (samples.task == "wdbc") &
                           (samples.noise == "composite") & (samples.gamma == gamma) &
                           (samples.method == "fixed_threshold")]
            exact_part = part[part.shots_per_test_case.astype(str) == "exact"]
            if exact_part.empty:
                exact_part = part[part.shots_per_test_case.isna()]
            if not exact_part.empty:
                panels.append((gamma, exact_part))
        if panels:
            fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8), sharey=True,
                                     squeeze=False, constrained_layout=True)
            for ax, (gamma, part) in zip(axes[0], panels):
                for label, color, label_name in [(0, "#2563eb", "class 0"), (1, "#f97316", "class 1")]:
                    vals = part.loc[part.y_true_model_label == label, "score"].dropna()
                    ax.hist(vals, bins=18, alpha=.55, density=True, color=color, label=label_name)
                ax.axvline(0.0, color="black", linestyle="--", linewidth=1, label="fixed threshold")
                ax.set(title=f"γ={gamma:.1f}", xlabel="Exact noisy expectation score")
                ax.legend(fontsize=7)
            axes[0][0].set_ylabel("Density")
            fig.savefig(OUT / "score_distributions_wdbc.png", dpi=180)
            plt.close(fig)
    # Contrast extrapolation/model bias at exact expectations with sampling
    # deviation from the matching exact method-specific value.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    exact_w = agg[(agg.task == "wdbc") & (agg.n_qubits == 2) & (agg.architecture == "cnot_chain") &
                  agg.method.isin(["fixed_threshold", "linear_zne", "quadratic_zne"])]
    for method in ["fixed_threshold", "linear_zne", "quadratic_zne"]:
        part = exact_w[exact_w.method == method].dropna(subset=["score_rmse"])
        if not part.empty:
            axes[0].plot(part.gamma, part.score_rmse, marker="o", label=method)
    axes[0].set(title="Observable error relative to clean", xlabel="Per-location γ", ylabel="RMSE")
    axes[0].legend(fontsize=7)
    shot_summary_path = OUT / "finite_shot_summary.csv"
    shot_summary = pd.read_csv(shot_summary_path) if shot_summary_path.exists() else pd.DataFrame()
    # Affine logistic decision scores use a different scale; plot observable estimators only.
    sub = shot_summary[(shot_summary.task == "wdbc") & (shot_summary.n_qubits == 2) &
                       (shot_summary.architecture == "cnot_chain") & shot_summary.gamma.isin([0.3, 0.4]) &
                       shot_summary.method.isin(["fixed_threshold", "linear_zne", "quadratic_zne"])]
    if not sub.empty and "sampling_rmse_mean" in sub:
        for (gamma, method), part in sub.groupby(["gamma", "method"]):
            axes[1].plot(part.shots_per_test_case, part.sampling_rmse_mean, marker="o",
                         label=f"γ={gamma:.1f} {method}")
    axes[1].set(title="Observable sampling deviation", xlabel="Total shots per test case",
                ylabel="RMSE", xscale="log")
    axes[1].legend(fontsize=6, ncol=2)
    fig.savefig(OUT / "score_error_decomposition.png", dpi=180)
    plt.close(fig)
    if (OUT / "collapse_rate_by_model.csv").exists():
        c = pd.read_csv(OUT / "collapse_rate_by_model.csv")
        c = c[(c.task == "wdbc") & c.gamma.isin([0.3, 0.4])]
        if not c.empty:
            def control_name(r):
                if r.architecture == "cz_ring": return "2q CZ"
                if r.architecture == "cnot_star": return "4q star"
                tag = f"{int(r.n_qubits)}q chain d{int(r.n_layers)}"
                if r.pca_standardize: tag += " standardized"
                if r.reverse_labels: tag += " reversed"
                return tag
            c["model"] = c.apply(control_name, axis=1)
            fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
            labels = list(dict.fromkeys(c.model.tolist()))
            xloc = {label: j for j, label in enumerate(labels)}
            gammas = sorted(c.gamma.unique())
            offsets = {g: (j - (len(gammas)-1)/2) * .16 for j,g in enumerate(gammas)}
            for gamma, part in c.groupby("gamma"):
                xs = np.array([xloc[v] for v in part.model]) + offsets[gamma]
                ax.scatter(xs, part.collapse_rate, label=f"γ={gamma:.1f}", s=24)
            ax.set(ylabel="Fraction of model runs with one-class predictions", ylim=(-.03, 1.12),
                   title="Collapse frequency across tested model controls")
            counts_by_model = c.groupby("model", sort=False).model_runs.first().astype(int).to_dict()
            display_labels = [f"{label}\n(n={counts_by_model[label]})" for label in labels]
            ax.set_xticks(range(len(labels)), display_labels, rotation=25, ha="right", fontsize=8)
            ax.legend()
            fig.savefig(OUT / "collapse_controls.png", dpi=180)
            plt.close(fig)


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--train-seeds", type=int, default=2)
    p.add_argument("--arch-sweep-splits", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", type=str, default=None,
                   help="Unique output folder under results/runs; generated when omitted")
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted run from its per-model partial results")
    p.add_argument("--reuse-checkpoints-from", type=str, default=None,
                   help="Reuse trained weights from another run while recomputing evaluation outputs")
    p.add_argument("--shot-budgets", type=lambda s: [int(x) for x in s.split(",")], default=[100, 1000])
    p.add_argument("--shot-repetitions", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    run(get_args())
