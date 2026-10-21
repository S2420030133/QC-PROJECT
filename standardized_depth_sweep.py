"""Matched 1- and 4-layer controls on the standardized WDBC folds/seeds.

The existing 2-layer primary model supplies the middle depth. All three depths
use standardized PCA, the same splits/seeds, and matched nominal shot budgets.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import joblib
import research_experiment as exp

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"results/runs/final-study-independent-retrain-20261003"
S=RUN/"sensitivity"; CK=S/"checkpoints"; CK.mkdir(exist_ok=True)
if (S/"wdbc_standardized_depth_sweep.csv").exists() and "--overwrite" not in sys.argv:
    raise SystemExit("Completed depth-sweep outputs already exist; use --overwrite only to intentionally replace them.")
manifest=json.loads((RUN/"run_manifest.json").read_text());cfg=manifest["config"]
base=pd.read_csv(S/"wdbc_standardized_pca_control.csv")
core=base[base.method=="quantum_clean_fixed"][["model_id","split","train_seed"]].drop_duplicates().sort_values(["split","train_seed"])
X,y,_=exp.tasks()["wdbc"]
rows=[]; obs=[]; histories=[]
for mr in core.itertuples():
    z=np.load(RUN/"splits"/f"wdbc_split{int(mr.split)}.npz")
    tr,ca,te=z["train"],z["calibration"],z["test"]
    xtr,xc,xt,prep=exp.prepare_features(X,tr,ca,te,2,True)
    for depth in [1,4]:
        seed=int(mr.train_seed); mid=f"depth{depth}_split{int(mr.split)}_seed{seed}"
        w,hist=exp.train_model(xtr,y[tr],2,depth,"cnot_chain",seed,int(cfg["epochs"]),
            int(cfg["batch_size"]),float(cfg["learning_rate"]))
        np.save(CK/f"{mid}_standardized.npy",w)
        joblib.dump(prep,CK/f"{mid}_standardized_preprocessing.joblib")
        for h in hist: histories.append({"model_id":mid,"depth":depth,**h})
        clean=exp.make_clean_qnode(2,depth,"cnot_chain")
        clean_cal=np.array([float(clean(w,x)) for x in xc]); clean_test=np.array([float(clean(w,x)) for x in xt])
        clean_thr=exp.best_balanced_threshold(y[ca],clean_cal)
        qexact=exp.make_qnode(2,depth,"cnot_chain","composite",None,seed+72000+depth)
        noisy_cal=exp.batch_scores(qexact,w,xc,.3); noisy_test=exp.batch_scores(qexact,w,xt,.3)
        exact_thr=exp.best_balanced_threshold(y[ca],noisy_cal)
        for method,score,threshold in [("clean_fixed",clean_test,0.),("clean_calibrated",clean_test,clean_thr),
                                       ("noisy_exact_fixed",noisy_test,0.),("noisy_exact_calibrated",noisy_test,exact_thr)]:
            rows.append({"model_id":mid,"split":int(mr.split),"train_seed":seed,"depth":depth,
                "method":method,"balanced_accuracy":exp.balanced_accuracy_score(y[te],score>=threshold),
                "roc_auc":exp.roc_auc_score(y[te],score),"threshold":float(threshold)})
        cal_n=max(1,int(round(500*len(te)/len(ca))))
        for rep in range(20):
            rseed=3_100_000+10000*int(mr.split)+1000*seed+100*depth+rep
            qc=exp.make_qnode(2,depth,"cnot_chain","composite",cal_n,rseed+1)
            qt=exp.make_qnode(2,depth,"cnot_chain","composite",500,rseed+2)
            cs=exp.batch_scores(qc,w,xc,.3); ts=exp.batch_scores(qt,w,xt,.3)
            threshold=exp.best_balanced_threshold(y[ca],cs)
            for method,decision in [("noisy_finite_fixed",0.),("noisy_finite_calibrated",threshold)]:
                rows.append({"model_id":mid,"split":int(mr.split),"train_seed":seed,"depth":depth,
                    "replicate":rep,"method":method,"balanced_accuracy":exp.balanced_accuracy_score(y[te],ts>=decision),
                    "roc_auc":exp.roc_auc_score(y[te],ts),"threshold":float(decision),
                    "calibration_shots":cal_n*len(ca),"test_shots":500*len(te),
                    "total_shots":cal_n*len(ca)+500*len(te)})
            for part,ids,labels,values,shots,thr in [("calibration",ca,y[ca],cs,cal_n,threshold),
                                                      ("test",te,y[te],ts,500,threshold)]:
                for ix,label,value in zip(ids,labels,values):
                    obs.append({"model_id":mid,"split":int(mr.split),"train_seed":seed,"depth":depth,
                        "replicate":rep,"partition":part,"example_index":int(ix),"y_true":int(label),
                        "shots_per_example":shots,"score":float(value),"selected_threshold":float(thr)})
    print(f"completed split={mr.split} seed={mr.train_seed}",flush=True)
out=pd.DataFrame(rows)
out.to_csv(S/"wdbc_standardized_depth_sweep.csv",index=False)
pd.DataFrame(obs).to_csv(S/"wdbc_standardized_depth_sweep_observations.csv",index=False)
pd.DataFrame(histories).to_csv(S/"wdbc_standardized_depth_sweep_training_curves.csv",index=False)
summary=out.groupby(["depth","method"],as_index=False).agg(ba_mean=("balanced_accuracy","mean"),
    ba_sd=("balanced_accuracy","std"),auc_mean=("roc_auc","mean"),rows=("model_id","size"))
summary.to_csv(S/"wdbc_standardized_depth_sweep_summary.csv",index=False)
print(summary.to_string(index=False))
