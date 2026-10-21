"""Recompute finite-shot WDBC depth-control metrics from saved observations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

RUN=Path(__file__).resolve().parent/"results/runs/final-study-independent-retrain-20261003"
S=RUN/"sensitivity"
obs=pd.read_csv(S/"wdbc_standardized_depth_sweep_observations.csv")
perf=pd.read_csv(S/"wdbc_standardized_depth_sweep.csv")
errors=[];checked=0
for key,g in obs.groupby(["model_id","split","train_seed","depth","replicate"],sort=False):
    cal=g[g.partition=="calibration"].sort_values("example_index")
    test=g[g.partition=="test"].sort_values("example_index")
    def best_threshold(y,s):
        candidates=np.unique(np.r_[[-np.inf],s,[np.inf]])
        scores=[balanced_accuracy_score(y,s>=v) for v in candidates]
        return float(candidates[int(np.argmax(scores))])
    threshold=best_threshold(cal.y_true.to_numpy(int),cal.score.to_numpy(float))
    matched=perf[(perf.model_id==key[0])&(perf.split==key[1])&(perf.train_seed==key[2])&
                 (perf.depth==key[3])&(perf.replicate==key[4])&perf.method.str.startswith("noisy_finite")]
    for row in matched.itertuples():
        decision=0.0 if row.method=="noisy_finite_fixed" else threshold
        ba=balanced_accuracy_score(test.y_true.to_numpy(int),test.score.to_numpy(float)>=decision)
        auc=roc_auc_score(test.y_true.to_numpy(int),test.score.to_numpy(float))
        checked+=1
        if not np.isclose(decision,row.threshold,atol=1e-12) or not np.isclose(ba,row.balanced_accuracy,atol=1e-12) or not np.isclose(auc,row.roc_auc,atol=1e-12):
            errors.append({"key":str(key),"method":row.method,"ba_saved":row.balanced_accuracy,"ba_recomputed":ba})
max_dev=0.0
for row in perf[perf.method.str.startswith("noisy_finite")].itertuples():
    ntest=len(np.load(RUN/"splits"/f"wdbc_split{int(row.split)}.npz")["test"])
    max_dev=max(max_dev,abs(row.total_shots/ntest-1000)/1000)
result={"metric_rows_recomputed":checked,"metric_mismatches":errors,"max_nominal_budget_deviation":max_dev,
        "status":"PASS" if not errors and max_dev<=.005 else "FAIL"}
(S/"wdbc_standardized_depth_sweep_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
if result["status"]!="PASS": raise SystemExit(1)
