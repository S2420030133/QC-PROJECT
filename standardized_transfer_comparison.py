"""Standardized-PCA finite-shot comparison on Wine and balanced synthetic data.

Freshly trains matched 2-qubit models on the saved outer splits, then compares
threshold selection with calibrated linear/quadratic probability-scaled ZNE.
This is an exploratory extension, not independent population validation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import research_experiment as exp

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"results/runs/final-study-independent-retrain-20261003"
OUT=RUN/"sensitivity"
CK=OUT/"checkpoints"
OUT.mkdir(exist_ok=True); CK.mkdir(exist_ok=True)
metrics=[]; observations=[]; training=[]; clean=[]; classical=[]
draws=20; gamma=.30; budget=1000
for task in ["wine_0_vs_rest","synthetic_balanced"]:
    X,y,_=exp.tasks()[task]
    for split_id in range(5):
        split=np.load(RUN/"splits"/f"{task}_split{split_id}.npz")
        train_ids,cal_ids,test_ids=split["train"],split["calibration"],split["test"]
        xtr,xcal,xtest,prep=exp.prepare_features(X,train_ids,cal_ids,test_ids,2,True)
        for train_seed in range(2):
            model_id=f"{task}_s{split_id}_i{train_seed}_q2_d2_cnot_chain_std1"
            w,hist=exp.train_model(xtr,y[train_ids],2,2,"cnot_chain",42+split_id*100+train_seed,30,32,.1)
            np.save(CK/f"{model_id}_standardized.npy",w)
            joblib.dump({"standardizer":prep["standardizer"],"pca":prep["pca"],"scaler":prep["scaler"],
                         "train_ids":train_ids,"calibration_ids":cal_ids,"test_ids":test_ids},
                        CK/f"{model_id}_standardized_preprocessing.joblib")
            for r in hist:
                training.append({"model_id":model_id,"task":task,"split":split_id,"train_seed":train_seed,**r})
            cleanq=exp.make_clean_qnode(2,2,"cnot_chain")
            clean_score=np.asarray([float(cleanq(w,x)) for x in xtest])
            clean.append({"model_id":model_id,"task":task,"split":split_id,"train_seed":train_seed,
                          **exp.metrics(y[test_ids],clean_score,0.0),"pca_explained_variance_sum":prep["pca_explained_variance_sum"]})
            lr=exp.LogisticRegression(max_iter=3000,random_state=41).fit(xtr,y[train_ids])
            lr_score=lr.predict_proba(xtest)[:,1]
            classical.append({"model_id":model_id,"task":task,"split":split_id,"train_seed":train_seed,
                "model":"logistic_regression_standardized_2d",**exp.metrics(y[test_ids],lr_score,.5),
                "pca_explained_variance_sum":prep["pca_explained_variance_sum"]})
            qe=exp.make_qnode(2,2,"cnot_chain","composite",None,3_400_000+1000*split_id+train_seed)
            cexact=exp.batch_scores(qe,w,xcal,gamma); texact=exp.batch_scores(qe,w,xtest,gamma)
            exactthr=exp.best_balanced_threshold(y[cal_ids],cexact)
            metrics.append({"model_id":model_id,"task":task,"split":split_id,"train_seed":train_seed,
                "gamma":gamma,"replicate":-1,"method":"noisy_threshold_exact","measurement_mode":"exact",
                "threshold":exactthr,"balanced_accuracy":exp.balanced_accuracy_score(y[test_ids],texact>=exactthr),
                "roc_auc":exp.roc_auc_score(y[test_ids],texact)})
            cal_n=max(1,int(round((budget/2)*len(test_ids)/len(cal_ids))))
            test_n=budget//2
            for rep in range(draws):
                base=3_500_000+10_000*split_id+1000*train_seed+10*rep
                qc=exp.make_qnode(2,2,"cnot_chain","composite",cal_n,base+1)
                qt=exp.make_qnode(2,2,"cnot_chain","composite",test_n,base+2)
                cs=np.asarray(exp.batch_scores(qc,w,xcal,gamma)); ts=np.asarray(exp.batch_scores(qt,w,xtest,gamma))
                thr=exp.best_balanced_threshold(y[cal_ids],cs)
                for name,score,decision in [("noisy_threshold",ts,thr),("fixed_threshold",ts,0.)]:
                    metrics.append({"model_id":model_id,"task":task,"split":split_id,"train_seed":train_seed,
                        "gamma":gamma,"replicate":rep,"method":name,"measurement_mode":"finite_shot",
                        "threshold":float(decision),"balanced_accuracy":exp.balanced_accuracy_score(y[test_ids],score>=decision),
                        "roc_auc":exp.roc_auc_score(y[test_ids],score),"calibration_shots":cal_n*len(cal_ids),
                        "test_shots":test_n*len(test_ids),"total_shots":cal_n*len(cal_ids)+test_n*len(test_ids)})
                for part,ids,labels,vals,shots in [("calibration",cal_ids,y[cal_ids],cs,cal_n),("test",test_ids,y[test_ids],ts,test_n)]:
                    for ix,label,val in zip(ids,labels,vals):
                        observations.append({"model_id":model_id,"task":task,"split":split_id,"train_seed":train_seed,
                            "gamma":gamma,"replicate":rep,"method":"noisy_threshold","partition":part,
                            "example_index":int(ix),"y_true":int(label),"scale":1.,"score":float(val),
                            "selected_threshold":float(thr),"shots_per_example":shots})
                for method,scales,coeffs,offset in [("linear_zne",[1.,2.],np.array([2.,-1.]),100),
                                                    ("quadratic_zne",[1.,1.5,2.],np.array([6.,-8.,3.]),200)]:
                    cscore,calloc,cscale=exp.finite_zne_scores(w,xcal,gamma,cal_n,scales,coeffs,2,2,"cnot_chain",base+offset)
                    tscore,talloc,tscale=exp.finite_zne_scores(w,xtest,gamma,test_n,scales,coeffs,2,2,"cnot_chain",base+offset+200)
                    zthr=exp.best_balanced_threshold(y[cal_ids],cscore)
                    for name,decision in [(method+"_calibrated",zthr),(method+"_fixed",0.)]:
                        metrics.append({"model_id":model_id,"task":task,"split":split_id,"train_seed":train_seed,
                            "gamma":gamma,"replicate":rep,"method":name,"measurement_mode":"finite_shot",
                            "threshold":float(decision),"balanced_accuracy":exp.balanced_accuracy_score(y[test_ids],tscore>=decision),
                            "roc_auc":exp.roc_auc_score(y[test_ids],tscore),"calibration_shots":int(sum(calloc)*len(cal_ids)),
                            "test_shots":int(sum(talloc)*len(test_ids)),"total_shots":int(sum(calloc)*len(cal_ids)+sum(talloc)*len(test_ids))})
                    for part,ids,labels,scale_vals,alloc in [("calibration",cal_ids,y[cal_ids],cscale,calloc),("test",test_ids,y[test_ids],tscale,talloc)]:
                        for scale,shots,vals in zip(scales,alloc,scale_vals):
                            for ix,label,val in zip(ids,labels,vals):
                                observations.append({"model_id":model_id,"task":task,"split":split_id,"train_seed":train_seed,
                                    "gamma":gamma,"replicate":rep,"method":method,"partition":part,
                                    "example_index":int(ix),"y_true":int(label),"scale":float(scale),"score":float(val),
                                    "selected_threshold":float(zthr),"shots_per_example":int(shots)})
            print(f"trained/evaluated {model_id}",flush=True)
pd.DataFrame(metrics).to_csv(OUT/"standardized_transfer_comparison.csv",index=False)
pd.DataFrame(observations).to_csv(OUT/"standardized_transfer_observations.csv",index=False)
pd.DataFrame(training).to_csv(OUT/"standardized_transfer_training_curves.csv",index=False)
pd.DataFrame(clean).to_csv(OUT/"standardized_transfer_clean.csv",index=False)
pd.DataFrame(classical).to_csv(OUT/"standardized_transfer_classical.csv",index=False)
summary=(pd.DataFrame(metrics).query("measurement_mode == 'finite_shot'")
    .groupby(["task","method"],as_index=False).agg(ba_mean=("balanced_accuracy","mean"),ba_sd=("balanced_accuracy","std"),
    auc_mean=("roc_auc","mean"),n_rows=("model_id","size")))
summary.to_csv(OUT/"standardized_transfer_summary.csv",index=False)
print(summary.to_string(index=False))
