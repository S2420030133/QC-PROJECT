"""Exact calibration-score spread by depth for standardized WDBC checkpoints."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import research_experiment as exp

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"results/runs/final-study-independent-retrain-20261003"
S=RUN/"sensitivity"; CK=S/"checkpoints"
models=pd.read_csv(S/"wdbc_standardized_pca_control.csv")
core=models[models.method=="quantum_clean_fixed"][["model_id","split","train_seed"]].drop_duplicates().sort_values(["split","train_seed"])
X,y,_=exp.tasks()["wdbc"]
rows=[]
for row in core.itertuples():
    split=np.load(RUN/"splits"/f"wdbc_split{int(row.split)}.npz"); ca=split["calibration"]
    for depth in [1,2,4]:
        if depth==2:
            mid=row.model_id; wp=CK/f"{mid}_standardized.npy"; pp=CK/f"{mid}_standardized_preprocessing.joblib"
        else:
            mid=f"depth{depth}_split{int(row.split)}_seed{int(row.train_seed)}"
            wp=CK/f"{mid}_standardized.npy"; pp=CK/f"{mid}_standardized_preprocessing.joblib"
        prep=joblib.load(pp); w=np.load(wp)
        xc=prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[ca])))
        for gamma in [0.0,.30]:
            q=exp.make_qnode(2,depth,"cnot_chain","composite",None,4_100_000+10000*int(row.split)+1000*int(row.train_seed)+depth)
            score=exp.batch_scores(q,w,xc,gamma)
            sd=float(np.std(score,ddof=1)); shot_var=float(np.mean(1-score**2)); gap=float(abs(np.mean(score[y[ca]==1])-np.mean(score[y[ca]==0])))
            rows.append({"model_id":mid,"split":int(row.split),"train_seed":int(row.train_seed),"depth":depth,
                "gamma":gamma,"calibration_score_sd":sd,"class_mean_gap":gap,
                "pooled_class_sd":float(np.sqrt((np.var(score[y[ca]==0],ddof=1)+np.var(score[y[ca]==1],ddof=1))/2)),
                "standardized_class_gap":gap/float(np.sqrt((np.var(score[y[ca]==0],ddof=1)+np.var(score[y[ca]==1],ddof=1))/2)),
                "mean_pauli_sample_variance":shot_var,
                "shot_se_at_500":float(np.sqrt(shot_var/500)),
                "shot_se_over_score_sd_at_500":float(np.sqrt(shot_var/500)/sd) if sd else np.nan,
                "shots_for_shot_se_equal_score_sd":float(shot_var/(sd**2)) if sd else np.nan})
    print(f"scored split={row.split} seed={row.train_seed}",flush=True)
out=pd.DataFrame(rows); out.to_csv(S/"wdbc_standardized_depth_score_spread.csv",index=False)
summary=out.groupby(["depth","gamma"],as_index=False).agg(
    score_sd_mean=("calibration_score_sd","mean"),score_sd_sd=("calibration_score_sd","std"),
    class_gap_mean=("class_mean_gap","mean"),class_gap_sd=("class_mean_gap","std"),
    dprime_mean=("standardized_class_gap","mean"),shot_se_500_mean=("shot_se_at_500","mean"),
    relative_se_500_mean=("shot_se_over_score_sd_at_500","mean"),
    equivalent_shots_mean=("shots_for_shot_se_equal_score_sd","mean"),checkpoints=("model_id","nunique"))
summary.to_csv(S/"wdbc_standardized_depth_score_spread_summary.csv",index=False)
print(summary.to_string(index=False))
