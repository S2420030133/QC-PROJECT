"""Audit script for FakeManila noise-scale sweep."""
from pathlib import Path
import json
import pandas as pd
import research_experiment as exp

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/runs/final-study-independent-retrain-20261003"
S = RUN / "sensitivity"

def main():
    metrics = pd.read_csv(S / "wdbc_fake_backend_noise_sweep_metrics.csv")
    protocol = json.loads((S / "wdbc_fake_backend_noise_sweep_protocol.json").read_text(encoding="utf-8"))
    
    assert len(metrics) == 90 # 3 noise scales x 3 methods x 10 checkpoints
    assert set(metrics.noise_scale.unique()) == {1.0, 3.0, 10.0}
    assert metrics.model_id.nunique() == 10
    
    summary = pd.read_csv(S / "wdbc_fake_backend_noise_sweep_summary.csv")
    assert len(summary) == 9
    
    audit_data = {
        "backend": protocol["backend"],
        "checkpoints": int(metrics.model_id.nunique()),
        "total_metrics_rows": len(metrics),
        "noise_scales_verified": sorted(metrics.noise_scale.unique().tolist()),
        "status": "PASS"
    }
    
    (S / "wdbc_fake_backend_noise_sweep_audit.json").write_text(json.dumps(audit_data, indent=2), encoding="utf-8")
    print(f"Fake Manila noise sweep audit PASSED!\n{json.dumps(audit_data, indent=2)}")

if __name__ == "__main__":
    main()
