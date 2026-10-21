"""Recompute exact standardized WDBC noise-grid metrics from saved score rows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
import research_experiment as exp

RUN=Path(__file__).resolve().parent/"results/runs/final-study-independent-retrain-20261003"
S=RUN/"sensitivity"
obs=pd.read_csv(S/"wdbc_standardized_exact_noise_sweep_observations.csv")
perf=pd.read_csv(S/"wdbc_standardized_exact_noise_sweep.csv")
errors=[];checked=0;groups=0
coeffs={"linear_zne":np.array([2.,-1.]),"quadratic_zne":np.array([6.,-8.,3.])}
for key,g in obs.groupby(["model_id","split","train_seed","gamma","method"],sort=False):
    model,split,seed,gamma,method=key; groups+=1
    cal=g[g.partition=="calibration"].pivot(index=["example_index","y_true"],columns="scale",values="score")
    test=g[g.partition=="test"].pivot(index=["example_index","y_true"],columns="scale",values="score")
    if method=="noisy_threshold":
        scales=[1.0]; coeff=np.array([1.0]); output_methods=[("fixed_threshold",0.0),("noisy_threshold",None)]
    else:
        scales=[1.0,2.0] if method=="linear_zne" else [1.0,1.5,2.0]
        coeff=coeffs[method]; output_methods=[(method+"_calibrated",None)]
    if list(map(float,cal.columns))!=scales or list(map(float,test.columns))!=scales:
        errors.append({"key":str(key),"problem":"missing scale scores"});continue
    cscore=np.round(cal.to_numpy(float)@coeff,12);tscore=np.round(test.to_numpy(float)@coeff,12)
    cy=cal.index.get_level_values("y_true").to_numpy(int);ty=test.index.get_level_values("y_true").to_numpy(int)
    threshold=exp.best_balanced_threshold(cy,cscore)
    saved_thresholds=np.unique(g.threshold.to_numpy(float))
    if len(saved_thresholds)!=1 or not np.isclose(saved_thresholds[0],threshold,atol=1e-12):
        errors.append({"key":str(key),"problem":"saved observation threshold differs from calibration recomputation",
                       "threshold_recomputed":threshold,"threshold_saved":saved_thresholds.tolist()})
    for outmethod,decision in output_methods:
        decision=threshold if decision is None else decision
        match=perf[(perf.model_id==model)&(perf.split==split)&(perf.train_seed==seed)&(perf.gamma==gamma)&(perf.method==outmethod)]
        if len(match)!=1:
            errors.append({"key":str(key),"problem":"missing or duplicate metric row","method":outmethod});continue
        row=match.iloc[0];ba=balanced_accuracy_score(ty,tscore>=decision);auc=roc_auc_score(ty,tscore);checked+=1
        if not np.isclose(decision,row.threshold,atol=1e-12) or not np.isclose(ba,row.balanced_accuracy,atol=1e-12) or not np.isclose(auc,row.roc_auc,atol=1e-12):
            errors.append({"key":str(key),"method":outmethod,"ba_recomputed":float(ba),"ba_saved":float(row.balanced_accuracy),"threshold_recomputed":decision,"threshold_saved":float(row.threshold)})
result={"groups_recomputed":groups,"metric_rows_recomputed":checked,"mismatches":errors,
        "status":"PASS" if not errors else "FAIL"}
(S/"wdbc_standardized_exact_noise_sweep_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
if errors: raise SystemExit(1)
