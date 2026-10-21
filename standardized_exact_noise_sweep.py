"""Exact-expectation noise grid for the primary standardized WDBC checkpoints."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import research_experiment as exp

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"results/runs/final-study-independent-retrain-20261003"
S=RUN/"sensitivity"; CK=S/"checkpoints"
if (S/"wdbc_standardized_exact_noise_sweep.csv").exists() and "--overwrite" not in sys.argv:
    raise SystemExit("Completed exact noise-grid outputs already exist; use --overwrite only to intentionally replace them.")
models=pd.read_csv(RUN/"models.csv")
core=models[(models.task=="wdbc")&(models.n_qubits==2)&(models.architecture=="cnot_chain")&
    (models.n_layers==2)&(~models.reverse_labels)&(~models.pca_standardize)].sort_values(["split","train_seed"])
X,y,_=exp.tasks()["wdbc"]
gammas=[0.0,.1,.2,.3,.4]
rows=[]; obs=[]
for mr in core.itertuples():
    split=np.load(RUN/"splits"/f"wdbc_split{int(mr.split)}.npz")
    ca,te=split["calibration"],split["test"]
    prep=joblib.load(CK/f"{mr.model_id}_standardized_preprocessing.joblib")
    w=np.load(CK/f"{mr.model_id}_standardized.npy")
    xc=prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[ca])))
    xt=prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[te])))
    for gamma in gammas:
        q=exp.make_qnode(2,2,"cnot_chain","composite",None,1_900_000+1000*int(mr.split)+int(mr.train_seed))
        cal_base=exp.batch_scores(q,w,xc,gamma); test_base=exp.batch_scores(q,w,xt,gamma)
        threshold=exp.best_balanced_threshold(y[ca],cal_base)
        pipelines=[("fixed_threshold",test_base,0.0)]
        pipelines.append(("noisy_threshold",test_base,threshold))
        for method,scales,coeffs in [("linear_zne",[1.,2.],np.array([2.,-1.])),
                                     ("quadratic_zne",[1.,1.5,2.],np.array([6.,-8.,3.]))]:
            cscore=np.round(sum(c*exp.batch_scores(q,w,xc,gamma*scale) for scale,c in zip(scales,coeffs)),12)
            tscore=np.round(sum(c*exp.batch_scores(q,w,xt,gamma*scale) for scale,c in zip(scales,coeffs)),12)
            thr=exp.best_balanced_threshold(y[ca],cscore)
            pipelines.append((method+"_calibrated",tscore,thr))
            for part,ids,labels,scale_values in [("calibration",ca,y[ca],[exp.batch_scores(q,w,xc,gamma*s) for s in scales]),
                                                 ("test",te,y[te],[exp.batch_scores(q,w,xt,gamma*s) for s in scales])]:
                for scale,values in zip(scales,scale_values):
                    for ix,label,value in zip(ids,labels,values):
                        obs.append({"model_id":mr.model_id,"split":int(mr.split),"train_seed":int(mr.train_seed),
                            "gamma":gamma,"method":method,"partition":part,"example_index":int(ix),
                            "y_true":int(label),"scale":float(scale),"score":float(value),"threshold":float(thr)})
        for method,score,decision in pipelines:
            rows.append({"model_id":mr.model_id,"split":int(mr.split),"train_seed":int(mr.train_seed),
                "gamma":gamma,"method":method,"threshold":float(decision),
                "balanced_accuracy":exp.balanced_accuracy_score(y[te],score>=decision),
                "roc_auc":exp.roc_auc_score(y[te],score)})
        for part,ids,labels,values in [("calibration",ca,y[ca],cal_base),("test",te,y[te],test_base)]:
            for ix,label,value in zip(ids,labels,values):
                obs.append({"model_id":mr.model_id,"split":int(mr.split),"train_seed":int(mr.train_seed),
                    "gamma":gamma,"method":"noisy_threshold","partition":part,"example_index":int(ix),
                    "y_true":int(label),"scale":1.0,"score":float(value),"threshold":float(threshold)})
    print(f"completed split={mr.split} seed={mr.train_seed}",flush=True)
out=pd.DataFrame(rows); out.to_csv(S/"wdbc_standardized_exact_noise_sweep.csv",index=False)
pd.DataFrame(obs).to_csv(S/"wdbc_standardized_exact_noise_sweep_observations.csv",index=False)
summary=out.groupby(["gamma","method"],as_index=False).agg(balanced_accuracy_mean=("balanced_accuracy","mean"),
    balanced_accuracy_sd=("balanced_accuracy","std"),roc_auc_mean=("roc_auc","mean"),n_models=("model_id","nunique"))
summary.to_csv(S/"wdbc_standardized_exact_noise_sweep_summary.csv",index=False)
print(summary.to_string(index=False))
