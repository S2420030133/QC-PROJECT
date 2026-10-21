"""Record the exact calibration-example subset behind each calibration-size result."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import research_experiment as exp

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"results/runs/final-study-independent-retrain-20261003"
path=RUN/"sensitivity/wdbc_standardized_calibration_size_sweep.csv"
data=pd.read_csv(path)
_,y,_=exp.tasks()["wdbc"]
selections={}
for split in range(5):
    z=np.load(RUN/"splits"/f"wdbc_split{split}.npz")
    selections[split]=z["calibration"]
indices=[]; counts=[]
for r in data.itertuples():
    ca=selections[int(r.split)]
    seed=2_500_000+10000*int(r.split)+1000*int(r.train_seed)+100*int(r.replicate)+int(r.calibration_examples)
    chosen=ca.copy() if int(r.calibration_examples)>=len(ca) else train_test_split(
        ca,train_size=int(r.calibration_examples),random_state=seed,stratify=y[ca])[0]
    indices.append(json.dumps([int(x) for x in chosen]))
    labels=y[chosen]
    counts.append(json.dumps({str(int(k)):int((labels==k).sum()) for k in np.unique(y)}))
data["calibration_example_indices"]=indices
data["calibration_class_counts"]=counts
data.to_csv(path,index=False)
print(f"Recorded {len(data)} exact calibration subsets.")
