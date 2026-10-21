"""Recompute standardized transfer finite-shot rows from saved observations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import research_experiment as exp
from sklearn.metrics import roc_auc_score

S=Path(__file__).resolve().parent/"results/runs/final-study-independent-retrain-20261003/sensitivity"
obs=pd.read_csv(S/"standardized_transfer_observations.csv")
perf=pd.read_csv(S/"standardized_transfer_comparison.csv")
coeff={"linear_zne":np.array([2.,-1.]),"quadratic_zne":np.array([6.,-8.,3.])}
errors=[]; checked=0; groups=0; max_dev=0.
for key,g in obs.groupby(["model_id","task","split","train_seed","replicate","method"],sort=False):
    model,task,split,seed,rep,method=key; groups+=1
    cal=g[g.partition=="calibration"].pivot(index=["example_index","y_true"],columns="scale",values="score")
    test=g[g.partition=="test"].pivot(index=["example_index","y_true"],columns="scale",values="score")
    scales=[1.] if method=="noisy_threshold" else ([1.,2.] if method=="linear_zne" else [1.,1.5,2.])
    if list(map(float,cal.columns))!=scales or list(map(float,test.columns))!=scales:
        errors.append({"key":str(key),"problem":"scale observations incomplete"}); continue
    c=cal.to_numpy(float); t=test.to_numpy(float)
    if method=="noisy_threshold": coeffs=np.array([1.])
    else: coeffs=coeff[method]
    cs=np.round(c@coeffs,12); ts=np.round(t@coeffs,12)
    cy=cal.index.get_level_values("y_true").to_numpy(int); ty=test.index.get_level_values("y_true").to_numpy(int)
    th=exp.best_balanced_threshold(cy,cs)
    stored=np.unique(g.selected_threshold.to_numpy(float))
    if len(stored)!=1 or not np.isclose(stored[0],th,atol=1e-12):
        errors.append({"key":str(key),"problem":"selected threshold mismatch","selected":stored.tolist(),"recomputed":th})
    pairs=[("noisy_threshold",th),("fixed_threshold",0.)] if method=="noisy_threshold" else [(method+"_calibrated",th),(method+"_fixed",0.)]
    for name,decision in pairs:
        r=perf[(perf.model_id==model)&(perf.split==split)&(perf.train_seed==seed)&(perf.replicate==rep)&(perf.method==name)]
        if len(r)!=1:
            errors.append({"key":str(key),"problem":"metric row missing/duplicate","method":name});continue
        row=r.iloc[0]; ba=exp.balanced_accuracy_score(ty,ts>=decision); auc=roc_auc_score(ty,ts); checked+=1
        if not np.isclose(row.threshold,decision,atol=1e-12) or not np.isclose(row.balanced_accuracy,ba,atol=1e-12) or not np.isclose(row.roc_auc,auc,atol=1e-12):
            errors.append({"key":str(key),"method":name,"problem":"metric mismatch","ba_saved":float(row.balanced_accuracy),"ba_recomputed":float(ba)})
        deviation=abs(float(row.total_shots)-2*float(row.test_shots))/(2*float(row.test_shots))
        max_dev=max(max_dev,deviation)
result={"observation_groups":groups,"finite_metric_rows_recomputed":checked,"max_relative_budget_deviation":max_dev,
        "mismatches":errors,"status":"PASS" if not errors else "FAIL"}
(S/"standardized_transfer_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
if errors: raise SystemExit(1)
