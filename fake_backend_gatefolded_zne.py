"""Matched-budget fake-backend sensitivity for the standardized WDBC VQC.

This is a local simulation from a saved IBM backend snapshot, not a QPU run.
It evaluates ten saved checkpoints on their existing calibration/test folds.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from research_experiment import tasks

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/runs/final-study-independent-retrain-20261003"
OUT = RUN / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
SHOTS_PER_PARTITION = 500
READOUT_CAL_SHOTS_PER_STATE = 2000
SCALES = (1, 3, 5)
COEFFICIENTS = np.asarray([15 / 8, -5 / 4, 3 / 8], dtype=float)
SEED = 20261016


def unitary_template() -> tuple[QuantumCircuit, list[Parameter]]:
    x = [Parameter(f"x{i}") for i in range(2)]
    theta = np.asarray([[Parameter(f"t{l}_{q}_{p}") for p in range(2)]
                        for l in range(2) for q in range(2)], dtype=object)
    qc = QuantumCircuit(2)
    qc.rx(x[0], 0)
    qc.rx(x[1], 1)
    for layer in range(2):
        for q in range(2):
            qc.ry(theta[layer * 2 + q, 0], q)
            qc.rz(theta[layer * 2 + q, 1], q)
        qc.cx(0, 1)
    params = [*x, *(theta.flatten().tolist())]
    return qc, params


def fold_native(circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Apply U,U^-1,U gate folding to each transpiled physical gate."""
    if scale not in (1, 3, 5):
        raise ValueError("Only odd fold scales 1, 3, and 5 are supported")
    folded = QuantumCircuit(circuit.num_qubits)
    repetitions = (scale + 1) // 2
    for item in circuit.data:
        op = item.operation
        qargs = [folded.qubits[circuit.find_bit(q).index] for q in item.qubits]
        for occurrence in range(scale):
            actual = op if occurrence % 2 == 0 else op.inverse()
            folded.append(actual, qargs)
    return folded


def measured(circuit: QuantumCircuit) -> QuantumCircuit:
    result = circuit.copy()
    creg = ClassicalRegister(1, "readout")
    result.add_register(creg)
    result.measure(0, creg[0])
    return result


def prep_readout_calibration(backend, simulator):
    circuits = []
    for bit in (0, 1):
        qc = QuantumCircuit(2, 1)
        if bit:
            qc.x(0)
        qc.measure(0, 0)
        circuits.append(transpile(qc, backend, initial_layout=[0, 1], optimization_level=0))
    result = simulator.run(circuits, shots=READOUT_CAL_SHOTS_PER_STATE,
                           seed_simulator=SEED).result()
    matrix = np.zeros((2, 2), dtype=float)
    for true_bit, circuit in enumerate(circuits):
        counts = result.get_counts(circuit)
        total = sum(counts.values())
        matrix[0, true_bit] = counts.get("0", 0) / total
        matrix[1, true_bit] = counts.get("1", 0) / total
    return matrix


def score_from_counts(counts: dict[str, int], confusion: np.ndarray | None = None) -> float:
    total = sum(counts.values())
    observed = np.asarray([counts.get("0", 0), counts.get("1", 0)], dtype=float) / total
    if confusion is None:
        true_prob = observed
    else:
        true_prob = np.linalg.solve(confusion, observed)
    return float(true_prob[0] - true_prob[1])


def best_threshold(y, scores) -> float:
    scores = np.round(np.asarray(scores, dtype=float), 12)
    candidates = np.unique(np.r_[scores, 0.0])
    values = [balanced_accuracy_score(y, scores >= t) for t in candidates]
    best = np.flatnonzero(np.asarray(values) == max(values))
    return float(candidates[best[np.argmin(np.abs(candidates[best]))]])


def feature_rows(raw_x, indices, prep):
    z = prep["standardizer"].transform(raw_x[indices])
    z = prep["pca"].transform(z)
    return prep["scaler"].transform(z)


def main():
    backend = FakeManilaV2()
    simulator = AerSimulator.from_backend(backend)
    confusion = prep_readout_calibration(backend, simulator)
    raw_x, y, _ = tasks()["wdbc"]
    raw_template, params = unitary_template()
    compiled = {}
    for scale in SCALES:
        physical = transpile(raw_template, backend, initial_layout=[0, 1], optimization_level=0)
        folded = fold_native(physical, scale)
        # Re-translate inverse native gates (for example SXdg) into the
        # backend basis without cancelling the inserted folds.
        folded = transpile(folded, backend,
                           initial_layout=list(range(backend.num_qubits)),
                           optimization_level=0)
        compiled[scale] = measured(folded)

    # Equal total calibration and test measurements for each pipeline; ZNE
    # divides each 500-shot half across scales in |coefficient| proportions.
    allocation = np.floor(SHOTS_PER_PARTITION * np.abs(COEFFICIENTS) /
                          np.abs(COEFFICIENTS).sum()).astype(int)
    allocation[np.argmax(np.abs(COEFFICIENTS))] += SHOTS_PER_PARTITION - allocation.sum()
    records, observations = [], []
    checkpoints = sorted((OUT / "checkpoints").glob(
        "wdbc_s*_i*_q2_d2_cnot_chain_std0_rev0_standardized.npy"))
    if len(checkpoints) != 10:
        raise RuntimeError(f"Expected 10 standardized WDBC checkpoints, found {len(checkpoints)}")

    for model_index, weight_path in enumerate(checkpoints):
        key = weight_path.stem.removesuffix("_standardized")
        split = int(key.split("_s")[1].split("_")[0])
        prep = joblib.load(weight_path.with_name(weight_path.stem + "_preprocessing.joblib"))
        weights = np.load(weight_path)
        split_data = np.load(RUN / "splits" / f"wdbc_split{split}.npz")
        cal_idx, test_idx = split_data["calibration"], split_data["test"]
        xcal, xtest = feature_rows(raw_x, cal_idx, prep), feature_rows(raw_x, test_idx, prep)
        ycal, ytest = y[cal_idx], y[test_idx]
        all_scores = {}

        for partition, xvalues, labels in (("calibration", xcal, ycal), ("test", xtest, ytest)):
            by_scale = {}
            for scale, nshots in zip(SCALES, allocation):
                circuits, metadata = [], []
                template = compiled[scale]
                for example_index, xrow in enumerate(xvalues):
                    values = {parameter: float(value) for parameter, value in
                              zip(params, [*xrow, *weights.flatten()])}
                    circuit = template.assign_parameters(values, inplace=False)
                    circuits.append(circuit)
                    metadata.append((example_index, int(labels[example_index])))
                result = simulator.run(circuits, shots=int(nshots),
                                       seed_simulator=SEED + 10000 * model_index +
                                       1000 * scale + (0 if partition == "calibration" else 500)).result()
                raw, corrected = [], []
                for circuit, (example_index, label) in zip(circuits, metadata):
                    counts = result.get_counts(circuit)
                    raw_score = score_from_counts(counts)
                    corrected_score = score_from_counts(counts, confusion)
                    raw.append(raw_score)
                    corrected.append(corrected_score)
                    observations.append({"model_id": key, "split": split, "partition": partition,
                                         "example_index": int((cal_idx if partition == "calibration" else test_idx)[example_index]),
                                         "label": label, "scale": scale, "shots": int(nshots),
                                         "score": raw_score, "readout_corrected_score": corrected_score})
                by_scale[scale] = (np.asarray(raw), np.asarray(corrected))
            all_scores[partition] = by_scale

        methods = {}
        for correction_label, correction_index in (("raw", 0), ("readout_mitigated", 1)):
            cal_scale1 = all_scores["calibration"][1][correction_index]
            test_scale1 = all_scores["test"][1][correction_index]
            methods[f"noise_threshold_{correction_label}"] = (cal_scale1, test_scale1)
            cal_zne, test_zne = [], []
            for data, target in ((all_scores["calibration"], cal_zne), (all_scores["test"], test_zne)):
                scores = sum(float(coef) * data[scale][correction_index]
                             for coef, scale in zip(COEFFICIENTS, SCALES))
                target.extend(scores.tolist())
            methods[f"gate_folded_zne_{correction_label}"] = (np.asarray(cal_zne), np.asarray(test_zne))

        for method, (cal_scores, test_scores) in methods.items():
            threshold = best_threshold(ycal, cal_scores)
            ba = balanced_accuracy_score(ytest, test_scores >= threshold)
            auc = roc_auc_score(ytest, test_scores)
            records.append({"model_id": key, "split": split, "method": method,
                            "threshold": threshold, "balanced_accuracy": ba,
                            "roc_auc": auc, "calibration_shots_per_example": SHOTS_PER_PARTITION,
                            "inference_shots_per_example": SHOTS_PER_PARTITION,
                            "calibration_matrix": json.dumps(confusion.tolist())})
            print(key, method, f"BA={ba:.3f}", f"AUC={auc:.3f}", flush=True)

    metrics = pd.DataFrame(records)
    obs = pd.DataFrame(observations)
    metrics_path = OUT / "wdbc_fake_manila_gatefolded_metrics.csv"
    obs_path = OUT / "wdbc_fake_manila_gatefolded_observations.csv"
    summary_path = OUT / "wdbc_fake_manila_gatefolded_summary.csv"
    protocol_path = OUT / "wdbc_fake_manila_gatefolded_protocol.json"
    metrics.to_csv(metrics_path, index=False)
    obs.to_csv(obs_path, index=False)
    summary = metrics.groupby("method").agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd_checkpoint=("balanced_accuracy", "std"),
        roc_auc_mean=("roc_auc", "mean"), n_checkpoints=("model_id", "nunique")).reset_index()
    summary.to_csv(summary_path, index=False)
    protocol = {"backend": backend.name, "backend_class": type(backend).__name__,
                "snapshot_based_local_simulation": True, "hardware_execution": False,
                "snapshot_note": "Saved provider snapshot; not current hardware calibration.",
                "qiskit_version": __import__("qiskit").__version__,
                "qiskit_aer_version": __import__("qiskit_aer").__version__,
                "qiskit_ibm_runtime_version": __import__("qiskit_ibm_runtime").__version__,
                "fold_scales": list(SCALES), "richardson_coefficients": COEFFICIENTS.tolist(),
                "folding_location": "after transpilation, on native physical gates",
                "transpilation_optimization_level": 0,
                "initial_physical_layout": [0, 1],
                "simulator": "AerSimulator.from_backend(FakeManilaV2())",
                "shots_per_partition_per_example": SHOTS_PER_PARTITION,
                "scale_shots_per_partition": allocation.tolist(),
                "readout_calibration_shots_per_prepared_state": READOUT_CAL_SHOTS_PER_STATE,
                "readout_calibration_matrix": confusion.tolist(),
                "readout_calibration_overhead_total_shots": 2 * READOUT_CAL_SHOTS_PER_STATE,
                "models": len(checkpoints), "folds": 5,
                "same_saved_weights_splits_and_standardized_pca": True,
                "seed": SEED, "outputs": [metrics_path.name, obs_path.name, summary_path.name]}
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY\n", summary.to_string(index=False))
    print("Protocol:", protocol_path)


if __name__ == "__main__":
    main()
