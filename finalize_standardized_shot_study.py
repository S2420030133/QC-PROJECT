"""Install 20-draw standardized finite-shot data into the manuscript source tables."""
from pathlib import Path
import pandas as pd

S = Path(__file__).resolve().parent / "results/runs/final-study-independent-retrain-20261003/sensitivity"
new = pd.read_csv(S / "wdbc_standardized_finite_shot_20draws.csv")
old_path = S / "wdbc_standardized_zne_comparison.csv"
old = pd.read_csv(old_path)
methods = ["noisy_threshold", "linear_zne", "quadratic_zne"]
old = old[~(old.method.isin(methods) & old.measurement_mode.str.startswith("finite"))]
pd.concat([old, new], ignore_index=True, sort=False).to_csv(old_path, index=False)
obs_path=S/"wdbc_standardized_zne_intermediate_observations.csv"
oldobs=pd.read_csv(obs_path)
newobs=pd.read_csv(S/"wdbc_standardized_finite_shot_20draws_observations.csv")
oldobs=oldobs[~oldobs.method.isin(methods)]
pd.concat([oldobs,newobs],ignore_index=True,sort=False).to_csv(obs_path,index=False)
paired = (new[new.measurement_mode == "finite_calibrated"]
          .groupby(["model_id", "split", "method"], as_index=False).balanced_accuracy.mean()
          .pivot(index=["model_id", "split"], columns="method", values="balanced_accuracy"))
rows = []
for method in ["linear_zne", "quadratic_zne"]:
    fold = (paired.noisy_threshold - paired[method]).groupby(level="split").mean()
    rows.extend({"method": method, "split": int(split), "threshold_minus_zne": float(value)}
                for split, value in fold.items())
pd.DataFrame(rows).to_csv(S / "wdbc_standardized_paired_fold_differences.csv", index=False)
print(pd.read_csv(old_path).groupby(["method", "measurement_mode"], dropna=False).agg(
    mean_ba=("balanced_accuracy", "mean"), sd=("balanced_accuracy", "std"), rows=("model_id", "size")).to_string())
print("Paired fold means:\n", pd.DataFrame(rows).to_string(index=False))
