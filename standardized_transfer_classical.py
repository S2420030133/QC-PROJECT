"""Matched 2D logistic baselines for the standardized transfer experiment."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import research_experiment as exp

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"results/runs/final-study-independent-retrain-20261003"
S=RUN/"sensitivity"; CK=S/"checkpoints"
models=pd.read_csv(S/"standardized_transfer_clean.csv").drop_duplicates(["task","split"])
out=[]
for row in models.itertuples():
    X,y,_=exp.tasks()[row.task]
    prep=joblib.load(CK/f"{row.model_id}_standardized_preprocessing.joblib")
    tr=prep["train_ids"]; te=prep["test_ids"]
    xtr=prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[tr])))
    xt=prep["scaler"].transform(prep["pca"].transform(prep["standardizer"].transform(X[te])))
    model=exp.LogisticRegression(max_iter=3000,random_state=41).fit(xtr,y[tr])
    score=model.predict_proba(xt)[:,1]
    out.append({"task":row.task,"split":row.split,"model":"matched_logistic_2d",
        **exp.metrics(y[te],score,.5),"pca_explained_variance_sum":prep["pca"].explained_variance_ratio_.sum(),
        "fit_examples":len(tr)})
pd.DataFrame(out).to_csv(S/"standardized_transfer_classical.csv",index=False)
print(pd.DataFrame(out).groupby("task")[["balanced_accuracy","roc_auc"]].mean())
