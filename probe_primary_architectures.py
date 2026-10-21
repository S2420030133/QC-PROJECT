"""Run post-hoc random-weight input-to-readout probes for the primary circuits."""
from pathlib import Path
import argparse
import pandas as pd
import research_experiment as exp
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run-ids',nargs='+',default=['final-study-20260929','final-study-reproduction']);a=p.parse_args()
new=[]
for arch,seed in [('cnot_chain',4202),('cz_ring',4203)]:
 rows=exp.probe_feature_dependence(2,2,arch,seed)
 for row in rows:
  row['performed_before_training']=False
  row['probe_timing']='post hoc on random parameters; not used for model selection'
 new.extend(rows)
for run_id in a.run_ids:
 out=ROOT/'results'/'runs'/run_id/'input_dependence_probe.csv'
 if not out.exists():
  print(f'skip missing run: {run_id}');continue
 old=pd.read_csv(out)
 old=old[~((old.n_qubits==2)&(old.n_layers==2)&(old.architecture.isin(['cnot_chain','cz_ring'])))]
 pd.concat([old,pd.DataFrame(new)],ignore_index=True).to_csv(out,index=False)
 print(f'updated {out}')
