"""Post-hoc calibration-label-count sensitivity using saved standardized WDBC checkpoints."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import research_experiment as exp

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"results/runs/final-study-independent-retrain-20261003"
S=RUN/"sensitivity"
if (S/"wdbc_standardized_calibration_size_sweep.csv").exists() and "--overwrite" not in sys.argv:
    raise SystemExit("Completed calibration-size outputs already exist; use --overwrite only to intentionally replace them.")
sizes=[10,20,50,114]
models=pd.read_csv(RUN/"models.csv")
core=models[(models.task=="wdbc")&(models.n_qubits==2)&(models.architecture=="cnot_chain")&
    (models.n_layers==2)&(~models.reverse_labels)&(~models.pca_standardize)].sort_values(["split","train_seed"])
X,y,_=exp.tasks()["wdbc"]
observations=pd.read_csv(S/"wdbc_standardized_finite_shot_20draws_observations.csv")
rows=[]
for mr in core.itertuples():
    split=np.load(RUN/"splits"/f"wdbc_split{int(mr.split)}.npz")
    ca,te=split["calibration"],split["test"]
    prep=joblib.load(S/"checkpoints"/f"{mr.model_id}_standardized_preprocessing.joblib")
    w=np.load(S/"checkpoints"/f"{mr.model_id}_standardized.npy")
    xtestobs=observations[(observations.model_id==mr.model_id)&(observations.partition=="test")]
    for rep in range(20):
        for method in ["noisy_threshold","linear_zne","quadratic_zne"]:
            testrows=xtestobs[(xtestobs.replicate==rep)&(xtestobs.method==method)]
            if method=="noisy_threshold":
                test_scores=testrows.sort_values("example_index").score.to_numpy()
            else:
                coeff=np.array([2.,-1.]) if method=="linear_zne" else np.array([6.,-8.,3.])
                scales=[1.,2.] if method=="linear_zne" else [1.,1.5,2.]
                vals=[testrows[testrows.scale==scale].sort_values("example_index").score.to_numpy() for scale in scales]
                test_scores=np.round(np.sum(coeff[:,None]*np.vstack(vals),axis=0),12)
            for ncal in sizes:
                seed=2_500_000+10000*int(mr.split)+1000*int(mr.train_seed)+100*rep+ncal
                chosen = (ca.copy() if ncal >= len(ca) else
                          train_test_split(ca,train_size=ncal,random_state=seed,stratify=y[ca])[0])
                xsub=prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[chosen])))
                calshots=max(1,int(round(500*len(te)/ncal)))
                if method=="noisy_threshold":
                    q=exp.make_qnode(2,2,"cnot_chain","composite",calshots,seed+1)
                    calscores=exp.batch_scores(q,w,xsub,.3)
                    actualcal=calshots*ncal
                else:
                    scales=[1.,2.] if method=="linear_zne" else [1.,1.5,2.]
                    coeff=np.array([2.,-1.]) if method=="linear_zne" else np.array([6.,-8.,3.])
                    calscores,alloc,_=exp.finite_zne_scores(w,xsub,.3,calshots,scales,coeff,2,2,
                        "cnot_chain",seed+(10 if method=="linear_zne" else 20))
                    actualcal=sum(alloc)*ncal
                thr=exp.best_balanced_threshold(y[chosen],calscores)
                rows.append({"model_id":mr.model_id,"split":int(mr.split),"train_seed":int(mr.train_seed),
                    "replicate":rep,"calibration_examples":ncal,"method":method,"threshold":float(thr),
                    "balanced_accuracy":exp.balanced_accuracy_score(y[te],test_scores>=thr),
                    "roc_auc":exp.roc_auc_score(y[te],test_scores),"calibration_shots":actualcal,
                    "test_shots":500*len(te),"total_shots":actualcal+500*len(te)})
    print(f"completed split={mr.split} seed={mr.train_seed}",flush=True)
out=pd.DataFrame(rows)
out.to_csv(S/"wdbc_standardized_calibration_size_sweep.csv",index=False)
summary=out.groupby(["method","calibration_examples"],as_index=False).agg(
    balanced_accuracy_mean=("balanced_accuracy","mean"),balanced_accuracy_sd=("balanced_accuracy","std"),
    roc_auc_mean=("roc_auc","mean"),n_checkpoint_draw_rows=("model_id","size"),
    mean_total_shots=("total_shots","mean"))
summary.to_csv(S/"wdbc_standardized_calibration_size_sweep_summary.csv",index=False)
print(summary.to_string(index=False))
