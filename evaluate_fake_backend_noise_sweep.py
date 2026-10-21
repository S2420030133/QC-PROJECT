"""Evaluate primary standardized WDBC checkpoints on IBM FakeManila backend with noise-scale scaling (x1, x3, x10).

Builds a physical device regime map:
- Noise multipliers: 1x, 3x, 10x
- Pipelines: Fixed threshold, Calibrated threshold, Gate-folded ZNE (calibrated)
- Budget: 1,000 nominal measurements per test case (500 cal, 500 test)
"""
from pathlib import Path
import json, time, copy, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import research_experiment as exp

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/runs/final-study-independent-retrain-20261003"
S = RUN / "sensitivity"
CK = S / "checkpoints"

NOISE_FACTORS = [1.0, 3.0, 10.0]
SCALES = [1, 3, 5]
COEFS = np.array([15/8, -5/4, 3/8], dtype=float)
BUDGET = 1000
CAL_SHOTS = 500
TEST_SHOTS = 500

# Proportional allocation for 500 shots
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


def scale_noise_model(base_dict, factor):
    """Scale error probabilities in FakeManila NoiseModel dictionary."""
    if factor == 1.0:
        return NoiseModel.from_dict(base_dict)
    d_new = copy.deepcopy(base_dict)
    for err in d_new.get("errors", []):
        if err["type"] == "qerror":
            probs = np.array(err["probabilities"], dtype=float)
            perr = probs[1:] * factor
            tot = perr.sum()
            if tot >= 1.0:
                perr = perr / tot * 0.99
                p0 = 0.01
            else:
                p0 = 1.0 - tot
            err["probabilities"] = [float(p0)] + [float(x) for x in perr]
        elif err["type"] == "roerror":
            probs = np.array(err["probabilities"], dtype=float)
            e0 = min(probs[0, 1] * factor, 0.49)
            e1 = min(probs[1, 0] * factor, 0.49)
            probs[0, 0] = 1.0 - e0
            probs[0, 1] = e0
            probs[1, 0] = e1
            probs[1, 1] = 1.0 - e1
            err["probabilities"] = probs.tolist()
    return NoiseModel.from_dict(d_new)


def main():
    print("Extracting base NoiseModel from FakeManilaV2...", flush=True)
    base_backend = FakeManilaV2()
    base_nm = NoiseModel.from_backend(base_backend)
    base_dict = base_nm.to_dict()
    
    models = pd.read_csv(S / "wdbc_standardized_pca_control.csv")
    core = models[models.method == "quantum_clean_fixed"][["model_id", "split", "train_seed"]].drop_duplicates().sort_values(["split", "train_seed"])
    X, y, _ = exp.tasks()["wdbc"]
    
    metric_rows = []
    
    t0 = time.time()
    for factor in NOISE_FACTORS:
        print(f"\n--- Running FakeManila noise scale x{factor:.1f} ---", flush=True)
        nm_scaled = scale_noise_model(base_dict, factor)
        sim = AerSimulator(noise_model=nm_scaled)
        
        for row in core.itertuples():
            mid = row.model_id
            split_id = int(row.split)
            seed = int(row.train_seed)
            
            split = np.load(RUN / "splits" / f"wdbc_split{split_id}.npz")
            ca, te = split["calibration"], split["test"]
            prep = joblib.load(CK / f"{mid}_standardized_preprocessing.joblib")
            w = np.load(CK / f"{mid}_standardized.npy")
            
            xc = prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[ca])))
            xt = prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[te])))
            
            circuits = []
            metadata = []
            
            for ix, x_vec in zip(ca, xc):
                circuits.append(build_vqc_circuit(w, x_vec, scale=1))
                metadata.append(("calibration", int(ix), 1, CAL_SHOTS, "standard"))
                for sc, sh in zip(SCALES, SCALE_SHOTS):
                    circuits.append(build_vqc_circuit(w, x_vec, scale=sc))
                    metadata.append(("calibration", int(ix), sc, sh, "zne"))
                    
            for ix, x_vec in zip(te, xt):
                circuits.append(build_vqc_circuit(w, x_vec, scale=1))
                metadata.append(("test", int(ix), 1, TEST_SHOTS, "standard"))
                for sc, sh in zip(SCALES, SCALE_SHOTS):
                    circuits.append(build_vqc_circuit(w, x_vec, scale=sc))
                    metadata.append(("test", int(ix), sc, sh, "zne"))
                    
            tr_circuits = transpile(circuits, backend=sim, optimization_level=0)
            
            shot_groups = {}
            for idx, (circ, meta) in enumerate(zip(tr_circuits, metadata)):
                sh = meta[3]
                shot_groups.setdefault(sh, []).append((idx, circ))
                
            results_by_idx = {}
            backend_seed = 6_000_000 + int(factor * 100000) + 10000 * split_id + 1000 * seed
            for sh, items in shot_groups.items():
                indices, circ_sub = zip(*items)
                job = sim.run(list(circ_sub), shots=sh, seed=backend_seed + sh)
                res = job.result()
                for local_i, global_i in enumerate(indices):
                    counts = res.get_counts(local_i)
                    results_by_idx[global_i] = counts
                    
            cal_raw_scores = {}
            test_raw_scores = {}
            cal_zne_raw = {sc: {} for sc in SCALES}
            test_zne_raw = {sc: {} for sc in SCALES}
            
            for idx, meta in enumerate(metadata):
                part, ix, sc, sh, kind = meta
                counts = results_by_idx[idx]
                raw_z = (counts.get("0", 0) - counts.get("1", 0)) / float(sh)
                if kind == "standard":
                    if part == "calibration":
                        cal_raw_scores[ix] = raw_z
                    else:
                        test_raw_scores[ix] = raw_z
                else:
                    if part == "calibration":
                        cal_zne_raw[sc][ix] = raw_z
                    else:
                        test_zne_raw[sc][ix] = raw_z
                        
            y_cal = y[ca]
            y_test = y[te]
            c_raw = np.array([cal_raw_scores[ix] for ix in ca])
            t_raw = np.array([test_raw_scores[ix] for ix in te])
            
            c_zne = np.sum([c * np.array([cal_zne_raw[sc][ix] for ix in ca]) for c, sc in zip(COEFS, SCALES)], axis=0)
            t_zne = np.sum([c * np.array([test_zne_raw[sc][ix] for ix in te]) for c, sc in zip(COEFS, SCALES)], axis=0)
            
            thr_raw = exp.best_balanced_threshold(y_cal, c_raw)
            thr_zne = exp.best_balanced_threshold(y_cal, c_zne)
            
            pipelines = [
                ("fixed_threshold", t_raw, 0.0, "fixed"),
                ("calibrated_threshold", t_raw, thr_raw, "calibrated"),
                ("gate_folded_zne_calibrated", t_zne, thr_zne, "calibrated"),
            ]
            
            for name, scores, thr, thr_mode in pipelines:
                preds = (scores >= thr).astype(int)
                ba = exp.balanced_accuracy_score(y_test, preds)
                auc = exp.roc_auc_score(y_test, scores)
                metric_rows.append({
                    "noise_scale": factor,
                    "model_id": mid,
                    "split": split_id,
                    "train_seed": seed,
                    "method": name,
                    "threshold_mode": thr_mode,
                    "threshold": float(thr),
                    "balanced_accuracy": float(ba),
                    "roc_auc": float(auc),
                    "nominal_budget": BUDGET
                })
                
    print(f"\nAll noise sweeps completed in {time.time() - t0:.1f} s", flush=True)
    df_metrics = pd.DataFrame(metric_rows)
    df_metrics.to_csv(S / "wdbc_fake_backend_noise_sweep_metrics.csv", index=False)
    
    summary = df_metrics.groupby(["noise_scale", "method"], as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std"),
        threshold_mean=("threshold", "mean"),
        checkpoints=("model_id", "nunique")
    )
    summary.to_csv(S / "wdbc_fake_backend_noise_sweep_summary.csv", index=False)
    print("\nFakeManila Noise Scale Summary:")
    print(summary.to_string(index=False))
    
    # Plot Regime Map Figure
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    methods_plot = [
        ("fixed_threshold", "Fixed threshold (z >= 0)", "#ca0020", "s--"),
        ("calibrated_threshold", "Calibrated threshold", "#2166ac", "o-"),
        ("gate_folded_zne_calibrated", "Gate-folded ZNE + cal.", "#4dac26", "^-.")
    ]
    
    for m_key, m_label, color, fmt_str in methods_plot:
        sub = summary[summary.method == m_key].sort_values("noise_scale")
        ax.errorbar(
            sub["noise_scale"],
            sub["balanced_accuracy_mean"],
            yerr=sub["balanced_accuracy_sd"],
            fmt=fmt_str,
            color=color,
            label=m_label,
            capsize=4,
            linewidth=1.8,
            markersize=6
        )
        
    ax.set_xscale("log")
    ax.set_xticks(NOISE_FACTORS)
    ax.set_xticklabels(["1x (Device snapshot)", "3x", "10x"])
    ax.set_xlabel("Device Noise Multiplier (FakeManilaV2)", fontsize=10)
    ax.set_ylabel("Balanced Accuracy", fontsize=10)
    ax.set_ylim(0.45, 0.95)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=True, fontsize=9, loc="lower left")
    ax.set_title("Device Noise Regime Map: Threshold Calibration vs Fixed & ZNE", fontsize=10)
    fig.tight_layout()
    plot_path = RUN / "fake_manila_regime_map.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    fig.savefig(S / "fake_manila_regime_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved regime map figure to {plot_path}")
    
    protocol = {
        "backend": "FakeManilaV2",
        "description": "Noise-scale multiplier sweep (1x, 3x, 10x) on FakeManila device noise model to chart calibration regime boundary",
        "noise_multipliers": NOISE_FACTORS,
        "checkpoints": 10,
        "budget_per_test_case": BUDGET,
        "completion_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    (S / "wdbc_fake_backend_noise_sweep_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
