"""Reconcile earlier exact WDBC control rows to saved per-example noise-grid scores."""
from pathlib import Path
import json
import pandas as pd

RUN=Path(__file__).resolve().parent/"results/runs/final-study-independent-retrain-20261003"
S=RUN/"sensitivity"
controls_path=S/"wdbc_standardized_pca_control.csv"
controls=pd.read_csv(controls_path)
grid=pd.read_csv(S/"wdbc_standardized_exact_noise_sweep.csv")
old=controls.copy()
mapping={"quantum_noisy_exact_fixed":"fixed_threshold","quantum_noisy_exact_calibrated":"noisy_threshold"}
for control_method,grid_method in mapping.items():
    src=grid[(grid.gamma==.3)&(grid.method==grid_method)].set_index("model_id")
    mask=controls.method.eq(control_method)
    for ix,row in controls[mask].iterrows():
        match=src.loc[row.model_id]
        controls.loc[ix,"balanced_accuracy"]=match.balanced_accuracy
        controls.loc[ix,"roc_auc"]=match.roc_auc
        controls.loc[ix,"threshold"]=match.threshold
controls.to_csv(controls_path,index=False)
summary=controls.groupby("method",as_index=False).agg(balanced_accuracy_mean=("balanced_accuracy","mean"),
    balanced_accuracy_sd=("balanced_accuracy","std"),roc_auc_mean=("roc_auc","mean"),runs=("model_id","nunique"))
summary.to_csv(S/"wdbc_standardized_pca_control_summary.csv",index=False)
changes=[]
for method in mapping:
    a=old[old.method==method].set_index("model_id"); b=controls[controls.method==method].set_index("model_id")
    for mid in a.index:
        if a.loc[mid,"balanced_accuracy"]!=b.loc[mid,"balanced_accuracy"] or a.loc[mid,"threshold"]!=b.loc[mid,"threshold"]:
            changes.append({"model_id":mid,"method":method,"previous_balanced_accuracy":float(a.loc[mid,"balanced_accuracy"]),
                "reconciled_balanced_accuracy":float(b.loc[mid,"balanced_accuracy"]),"previous_threshold":float(a.loc[mid,"threshold"]),
                "reconciled_threshold":float(b.loc[mid,"threshold"])})
record={"basis":"Saved exact standardized-PCA per-example scores at gamma 0.30, recomputed from the same saved checkpoint, fold indices, preprocessing, and current PennyLane source.",
    "number_of_corrected_model_rows":len(changes),"changes":changes,
    "reconciled_means":summary[summary.method.isin(mapping)].to_dict(orient="records")}
(S/"wdbc_standardized_exact_metric_reconciliation.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps(record,indent=2))
