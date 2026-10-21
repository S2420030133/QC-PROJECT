"""Evaluate primary standardized WDBC checkpoints on IBM FakeManila backend.

Includes:
- Raw device noise with fixed threshold
- Raw device noise with calibrated threshold
- Readout-error mitigation (matrix inversion) with fixed and calibrated threshold
- Gate-folded ZNE (scales 1, 3, 5) with fixed and calibrated threshold
- Combined readout-mitigated + gate-folded ZNE with calibrated threshold
"""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import joblib
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
import research_experiment as exp

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/runs/final-study-independent-retrain-20261003"
S = RUN / "sensitivity"
CK = S / "checkpoints"

# Calibration and test settings
SCALES = [1, 3, 5]
COEFS = np.array([15/8, -5/4, 3/8], dtype=float)
BUDGET = 1000
CAL_SHOTS = 500
TEST_SHOTS = 500

# Proportional allocation for 500 shots:
# 500 * (1.875, 1.25, 0.375) / 3.5 -> 268, 178, 54
RAW_ALLOC = 500 * np.abs(COEFS) / np.abs(COEFS).sum()
SCALE_SHOTS = np.floor(RAW_ALLOC).astype(int)
for ix in np.argsort(-(RAW_ALLOC - SCALE_SHOTS))[:500 - int(SCALE_SHOTS.sum())]:
    SCALE_SHOTS[ix] += 1
assert int(SCALE_SHOTS.sum()) == 500

def build_vqc_circuit(weights, x, scale=1):
    """Build parameterized 2-qubit VQC circuit with unitary gate folding."""
    qc = QuantumCircuit(2, 1)
    
    def apply_folded_rx(wire, theta, s):
        for rep in range(s):
            angle = -theta if (rep % 2 == 1) else theta
            qc.rx(angle, wire)
            
    def apply_folded_ry(wire, theta, s):
        for rep in range(s):
            angle = -theta if (rep % 2 == 1) else theta
            qc.ry(angle, wire)
            
    def apply_folded_rz(wire, theta, s):
        for rep in range(s):
            angle = -theta if (rep % 2 == 1) else theta
            qc.rz(angle, wire)
            
    def apply_folded_cx(c, t, s):
        for _ in range(s):
            qc.cx(c, t)

    # Encoding
    apply_folded_rx(0, x[0], scale)
    apply_folded_rx(1, x[1], scale)
    
    # Layers
    for layer in range(2):
        apply_folded_ry(0, weights[layer, 0, 0], scale)
        apply_folded_rz(0, weights[layer, 0, 1], scale)
        apply_folded_ry(1, weights[layer, 1, 0], scale)
        apply_folded_rz(1, weights[layer, 1, 1], scale)
        apply_folded_cx(0, 1, scale)
        
    qc.measure(0, 0)
    return qc


def calibrate_readout_matrix(backend, seed=42):
    """Calibrate single-qubit measurement error matrix on qubit 0."""
    qc0 = QuantumCircuit(2, 1)
    qc0.measure(0, 0)
    qc1 = QuantumCircuit(2, 1)
    qc1.x(0)
    qc1.measure(0, 0)
    tr = transpile([qc0, qc1], backend=backend, optimization_level=0)
    job = backend.run(tr, shots=20000, seed=seed)
    c0 = job.result().get_counts(0)
    c1 = job.result().get_counts(1)
    p00 = c0.get("0", 0) / 20000.0
    p10 = c0.get("1", 0) / 20000.0
    p01 = c1.get("0", 0) / 20000.0
    p11 = c1.get("1", 0) / 20000.0
    M = np.array([[p00, p01], [p10, p11]])
    return M, np.linalg.inv(M)


def mitigate_counts(c_raw, total_shots, M_inv):
    """Apply inverse readout matrix and return expectation <Z_0>."""
    p_raw = np.array([c_raw.get("0", 0) / total_shots, c_raw.get("1", 0) / total_shots])
    p_mit = M_inv @ p_raw
    p_mit = np.clip(p_mit, 0.0, 1.0)
    p_mit /= p_mit.sum()
    return float(p_mit[0] - p_mit[1])


def main():
    print("Initializing FakeManilaV2 backend...", flush=True)
    backend = FakeManilaV2()
    M, M_inv = calibrate_readout_matrix(backend)
    print(f"Readout matrix M:\n{M}", flush=True)
    
    models = pd.read_csv(S / "wdbc_standardized_pca_control.csv")
    core = models[models.method == "quantum_clean_fixed"][["model_id", "split", "train_seed"]].drop_duplicates().sort_values(["split", "train_seed"])
    X, y, _ = exp.tasks()["wdbc"]
    
    obs_rows = []
    metric_rows = []
    
    t0 = time.time()
    for row in core.itertuples():
        mid = row.model_id
        split_id = int(row.split)
        seed = int(row.train_seed)
        print(f"Evaluating {mid} (split={split_id}, seed={seed})...", flush=True)
        
        split = np.load(RUN / "splits" / f"wdbc_split{split_id}.npz")
        ca, te = split["calibration"], split["test"]
        prep = joblib.load(CK / f"{mid}_standardized_preprocessing.joblib")
        w = np.load(CK / f"{mid}_standardized.npy")
        
        xc = prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[ca])))
        xt = prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[te])))
        
        # Build circuits for each scale and partition
        circuits = []
        metadata = [] # (partition, example_index, scale, shots)
        
        # Scale 1 unmitigated / raw uses 500 shots for calibration, 500 for test
        # Gate folding uses SCALE_SHOTS for scales 1, 3, 5
        # We can run scale 1 at 500 shots, and also extract the folded components
        # To strictly match budget:
        # For standard evaluation (scale 1): run 500 shots.
        # For ZNE evaluation: run scale 1 (268 shots), scale 3 (178 shots), scale 5 (54 shots).
        for ix, x_vec in zip(ca, xc):
            # standard scale 1
            circuits.append(build_vqc_circuit(w, x_vec, scale=1))
            metadata.append(("calibration", int(ix), 1, CAL_SHOTS, "standard"))
            # folded scales 3 and 5, plus dedicated scale 1 for ZNE
            for sc, sh in zip(SCALES, SCALE_SHOTS):
                circuits.append(build_vqc_circuit(w, x_vec, scale=sc))
                metadata.append(("calibration", int(ix), sc, sh, "zne"))
                
        for ix, x_vec in zip(te, xt):
            # standard scale 1
            circuits.append(build_vqc_circuit(w, x_vec, scale=1))
            metadata.append(("test", int(ix), 1, TEST_SHOTS, "standard"))
            # folded scales
            for sc, sh in zip(SCALES, SCALE_SHOTS):
                circuits.append(build_vqc_circuit(w, x_vec, scale=sc))
                metadata.append(("test", int(ix), sc, sh, "zne"))
                
        # Transpile all circuits with optimization_level=0 to protect folding
        tr_circuits = transpile(circuits, backend=backend, optimization_level=0)
        
        # Group runs by shot count
        shot_groups = {}
        for idx, (circ, meta) in enumerate(zip(tr_circuits, metadata)):
            sh = meta[3]
            shot_groups.setdefault(sh, []).append((idx, circ))
            
        results_by_idx = {}
        backend_seed = 5_000_000 + 10000 * split_id + 1000 * seed
        for sh, items in shot_groups.items():
            indices, circ_sub = zip(*items)
            job = backend.run(list(circ_sub), shots=sh, seed=backend_seed + sh)
            res = job.result()
            for local_i, global_i in enumerate(indices):
                counts = res.get_counts(local_i)
                results_by_idx[global_i] = counts
                
        # Parse scores for this model
        cal_raw_scores = {}
        cal_mit_scores = {}
        test_raw_scores = {}
        test_mit_scores = {}
        
        cal_zne_raw = {sc: {} for sc in SCALES}
        cal_zne_mit = {sc: {} for sc in SCALES}
        test_zne_raw = {sc: {} for sc in SCALES}
        test_zne_mit = {sc: {} for sc in SCALES}
        
        for idx, meta in enumerate(metadata):
            part, ix, sc, sh, kind = meta
            counts = results_by_idx[idx]
            raw_z = (counts.get("0", 0) - counts.get("1", 0)) / float(sh)
            mit_z = mitigate_counts(counts, sh, M_inv)
            
            if kind == "standard":
                if part == "calibration":
                    cal_raw_scores[ix] = raw_z
                    cal_mit_scores[ix] = mit_z
                else:
                    test_raw_scores[ix] = raw_z
                    test_mit_scores[ix] = mit_z
            else: # zne
                if part == "calibration":
                    cal_zne_raw[sc][ix] = raw_z
                    cal_zne_mit[sc][ix] = mit_z
                else:
                    test_zne_raw[sc][ix] = raw_z
                    test_zne_mit[sc][ix] = mit_z
                    
        # Construct arrays in order
        y_cal = y[ca]
        y_test = y[te]
        
        c_raw = np.array([cal_raw_scores[ix] for ix in ca])
        c_mit = np.array([cal_mit_scores[ix] for ix in ca])
        t_raw = np.array([test_raw_scores[ix] for ix in te])
        t_mit = np.array([test_mit_scores[ix] for ix in te])
        
        # ZNE linear combinations:
        c_zne_raw = np.sum([c * np.array([cal_zne_raw[sc][ix] for ix in ca]) for c, sc in zip(COEFS, SCALES)], axis=0)
        c_zne_mit = np.sum([c * np.array([cal_zne_mit[sc][ix] for ix in ca]) for c, sc in zip(COEFS, SCALES)], axis=0)
        t_zne_raw = np.sum([c * np.array([test_zne_raw[sc][ix] for ix in te]) for c, sc in zip(COEFS, SCALES)], axis=0)
        t_zne_mit = np.sum([c * np.array([test_zne_mit[sc][ix] for ix in te]) for c, sc in zip(COEFS, SCALES)], axis=0)
        
        # Threshold fitting on calibration partition
        thr_raw = exp.best_balanced_threshold(y_cal, c_raw)
        thr_mit = exp.best_balanced_threshold(y_cal, c_mit)
        thr_zne_raw = exp.best_balanced_threshold(y_cal, c_zne_raw)
        thr_zne_mit = exp.best_balanced_threshold(y_cal, c_zne_mit)
        
        pipelines = [
            ("fake_manila_fixed_threshold", t_raw, 0.0, "raw", "fixed"),
            ("fake_manila_calibrated_threshold", t_raw, thr_raw, "raw", "calibrated"),
            ("fake_manila_readout_mitigated_fixed", t_mit, 0.0, "readout_mitigated", "fixed"),
            ("fake_manila_readout_mitigated_calibrated", t_mit, thr_mit, "readout_mitigated", "calibrated"),
            ("fake_manila_gate_folded_zne_fixed", t_zne_raw, 0.0, "gate_folded_zne", "fixed"),
            ("fake_manila_gate_folded_zne_calibrated", t_zne_raw, thr_zne_raw, "gate_folded_zne", "calibrated"),
            ("fake_manila_readout_mitigated_gate_folded_zne_calibrated", t_zne_mit, thr_zne_mit, "readout_mitigated_gate_folded_zne", "calibrated"),
        ]
        
        for name, scores, thr, noise_mit_kind, thr_mode in pipelines:
            preds = (scores >= thr).astype(int)
            ba = exp.balanced_accuracy_score(y_test, preds)
            auc = exp.roc_auc_score(y_test, scores)
            metric_rows.append({
                "model_id": mid,
                "split": split_id,
                "train_seed": seed,
                "method": name,
                "backend": "FakeManilaV2",
                "mitigation": noise_mit_kind,
                "threshold_mode": thr_mode,
                "threshold": float(thr),
                "balanced_accuracy": float(ba),
                "roc_auc": float(auc),
                "nominal_budget": BUDGET
            })
            
        for ix, true_y, raw_val, mit_val, zne_r, zne_m in zip(te, y_test, t_raw, t_mit, t_zne_raw, t_zne_mit):
            obs_rows.append({
                "model_id": mid,
                "split": split_id,
                "train_seed": seed,
                "partition": "test",
                "example_index": int(ix),
                "y_true": int(true_y),
                "score_raw": float(raw_val),
                "score_readout_mitigated": float(mit_val),
                "score_gate_folded_zne": float(zne_r),
                "score_readout_mitigated_zne": float(zne_m)
            })
            
    print(f"All 10 checkpoints executed in {time.time() - t0:.1f} s", flush=True)
    
    df_metrics = pd.DataFrame(metric_rows)
    df_obs = pd.DataFrame(obs_rows)
    
    df_metrics.to_csv(S / "wdbc_fake_backend_metrics.csv", index=False)
    df_obs.to_csv(S / "wdbc_fake_backend_observations.csv", index=False)
    
    summary = df_metrics.groupby("method", as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std"),
        checkpoints=("model_id", "nunique")
    )
    summary.to_csv(S / "wdbc_fake_backend_summary.csv", index=False)
    print("\nSummary Results on FakeManilaV2:")
    print(summary.to_string(index=False))
    
    protocol = {
        "backend": "FakeManilaV2",
        "description": "IBM Quantum Falcon r5.11 processor realistic device simulation with calibrated T1, T2, gate errors, and readout error",
        "checkpoints": 10,
        "nominal_budget_per_test_case": 1000,
        "calibration_shots": 500,
        "test_shots": 500,
        "gate_folding_scales": SCALES,
        "gate_folding_shots": SCALE_SHOTS.tolist(),
        "readout_error_mitigation": "1-qubit matrix inversion on qubit 0 with 20,000 calibration shots",
        "readout_matrix_qubit0": M.tolist(),
        "completion_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    (S / "wdbc_fake_backend_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"\nSaved outputs to {S}")

if __name__ == "__main__":
    main()
