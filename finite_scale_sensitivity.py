"""Finite-shot Richardson scale sensitivity using frozen WDBC checkpoints."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
import research_experiment as exp
p=argparse.ArgumentParser(); p.add_argument('--run-id',required=True); p.add_argument('--repeats',type=int,default=2); a=p.parse_args()
root=Path(__file__).resolve().parent/'results'/'runs'/a.run_id; out=root/'sensitivity'; out.mkdir(exist_ok=True)
models=pd.read_csv(root/'models.csv')
core=models[(models.task=='wdbc')&(models.n_qubits==2)&(models.architecture=='cnot_chain')&(models.n_layers==2)&(~models.pca_standardize)&(~models.reverse_labels)].sort_values(['split','train_seed'])
X,y,_=exp.tasks()['wdbc']; scalesets={'linear_1_2':[1.,2.], 'quadratic_1_1p5_2':[1.,1.5,2.], 'mild_1_1p25':[1.,1.25], 'mild_1_1p5':[1.,1.5], 'mild_1_1p25_1p5':[1.,1.25,1.5]}
rows=[]
for mr in core.itertuples():
 prep=joblib.load(root/'checkpoints'/f'{mr.model_id}_preprocessing.joblib'); tr,ca,te=prep['train_ids'],prep['calibration_ids'],prep['test_ids']; _,xc,xt,_=exp.prepare_features(X,tr,ca,te,2,False); yc,yt=y[ca],y[te]; w=np.load(root/'checkpoints'/f'{mr.model_id}.npy')
 for budget in [100,1000]:
  cal_n=max(2,budget//2); test_n=budget-cal_n
  cal_per=max(1,int(round(cal_n*len(xt)/len(xc)))); test_per=max(1,test_n)
  for rep in range(a.repeats):
   seed=770000+1000*int(mr.split)+100*int(mr.train_seed)+rep
   for name,ss in scalesets.items():
    V=np.array(ss)[None,:]**np.arange(len(ss))[:,None]; co=np.linalg.solve(V,np.eye(len(ss))[0])
    scores,alloc,_=exp.finite_zne_scores(w,xt,.3,test_per,ss,co,2,2,'cnot_chain',seed+100)
    # Use same held-out calibration labels and an equal nominal calibration/test split.
    cs,ca_alloc,_=exp.finite_zne_scores(w,xc,.3,cal_per,ss,co,2,2,'cnot_chain',seed+200)
    threshold=exp.best_balanced_threshold(yc,cs)
    for method,t in [('zne_fixed',0.),('zne_calibrated',threshold)]:
     rows.append({'model_id':mr.model_id,'split':int(mr.split),'train_seed':int(mr.train_seed),'replicate':rep,'gamma':.3,'nominal_budget_per_test_case':budget,'scale_set':name,'scales':';'.join(map(str,ss)),'coefficients':';'.join(f'{v:.8g}' for v in co),'method':method,'threshold':t,'balanced_accuracy':exp.balanced_accuracy_score(yt,scores>=t),'roc_auc':exp.roc_auc_score(yt,scores),'calibration_total_shots':int(sum(ca_alloc)*len(xc)) if method=='zne_calibrated' else 0,'test_total_shots':int(sum(alloc)*len(xt))})
df=pd.DataFrame(rows); df.to_csv(out/'zne_scale_sensitivity_finite.csv',index=False)
s=(df.groupby(['nominal_budget_per_test_case','scale_set','method'],as_index=False).agg(balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),auc_mean=('roc_auc','mean'),n_models=('model_id','nunique'),n_outer_splits=('split','nunique')))
s.to_csv(out/'zne_scale_sensitivity_finite_summary.csv',index=False); print(s.to_string(index=False))
