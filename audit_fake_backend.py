"""Independent audit of FakeManilaV2 backend results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import research_experiment as exp

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/runs/final-study-independent-retrain-20261003"
S = RUN / "sensitivity"

def main():
    obs = pd.read_csv(S / "wdbc_fake_backend_observations.csv")
    metrics = pd.read_csv(S / "wdbc_fake_backend_metrics.csv")
    protocol = json.loads((S / "wdbc_fake_backend_protocol.json").read_text())
    
    score_cols = {
        "fake_manila_fixed_threshold": "score_raw",
        "fake_manila_calibrated_threshold": "score_raw",
        "fake_manila_readout_mitigated_fixed": "score_readout_mitigated",
        "fake_manila_readout_mitigated_calibrated": "score_readout_mitigated",
        "fake_manila_gate_folded_zne_fixed": "score_gate_folded_zne",
        "fake_manila_gate_folded_zne_calibrated": "score_gate_folded_zne",
        "fake_manila_readout_mitigated_gate_folded_zne_calibrated": "score_readout_mitigated_zne",
    }
    
    max_ba_diff = 0.0
    max_auc_diff = 0.0
    
    for row in metrics.itertuples():
        sub = obs[obs.model_id == row.model_id].sort_values("example_index")
        y_true = sub.y_true.to_numpy(dtype=int)
        col = score_cols[row.method]
        scores = sub[col].to_numpy(dtype=float)
        thr = float(row.threshold)
        
        preds = (scores >= thr).astype(int)
        recomp_ba = exp.balanced_accuracy_score(y_true, preds)
        recomp_auc = exp.roc_auc_score(y_true, scores)
        
        diff_ba = abs(recomp_ba - row.balanced_accuracy)
        diff_auc = abs(recomp_auc - row.roc_auc)
        
        if diff_ba > max_ba_diff:
            max_ba_diff = diff_ba
        if diff_auc > max_auc_diff:
            max_auc_diff = diff_auc
            
    assert max_ba_diff < 1e-12, f"BA diff too high: {max_ba_diff}"
    assert max_auc_diff < 1e-12, f"AUC diff too high: {max_auc_diff}"
    assert metrics.model_id.nunique() == 10
    assert len(metrics) == 70 # 7 methods x 10 checkpoints
    
    audit_record = {
        "backend": protocol["backend"],
        "checkpoints": int(metrics.model_id.nunique()),
        "total_metric_rows": len(metrics),
        "recomputed_ba_max_abs_error": float(max_ba_diff),
        "recomputed_auc_max_abs_error": float(max_auc_diff),
        "status": "PASS"
    }
    
    (S / "wdbc_fake_backend_audit.json").write_text(json.dumps(audit_record, indent=2), encoding="utf-8")
    print(f"FakeBackend Audit Passed successfully!\n{json.dumps(audit_record, indent=2)}")

if __name__ == "__main__":
    main()
