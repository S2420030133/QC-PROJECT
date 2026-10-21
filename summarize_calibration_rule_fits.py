"""Summarize repeated finite-shot variation in fitted calibration rules."""
from pathlib import Path
import argparse
import pandas as pd
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
run=ROOT/'results'/'runs'/a.run_id
f=pd.read_csv(run/'finite_shot_performance.csv')
f=f[(f.task=='wdbc')&(f.n_qubits==2)&(f.n_layers==2)&(f.architecture=='cnot_chain')&
    (f.model_id.str.endswith('_q2_d2_cnot_chain_std0_rev0'))&(f.gamma==.3)&(f.shots_per_test_case==1000)&
    f.method.isin(['calibrated_threshold','linear_zne_calibrated','quadratic_zne_calibrated','affine_readout'])]
rows=[]
for method,g in f.groupby('method'):
 for col,label in [('selected_threshold','threshold'),('affine_intercept','affine_intercept'),('affine_coefficient','affine_coefficient')]:
  if col not in g or g[col].notna().sum()==0: continue
  within=g.groupby('model_id')[col].std(ddof=1).dropna()
  rows.append({'method':method,'parameter':label,'mean_within_model_sd_across_two_repeats':within.mean(),
               'models_with_two_repeats':len(within),'noise_gamma':.3,'nominal_budget_per_test_case':1000})
out=pd.DataFrame(rows);out.to_csv(run/'calibration_rule_fit_variability.csv',index=False);print(out.to_string(index=False))
