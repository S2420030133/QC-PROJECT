"""Four-layer WDBC capacity control using the saved folds/seeds and standardized PCA."""
from pathlib import Path
import json,numpy as np,pandas as pd
import research_experiment as exp
root=Path(__file__).resolve().parent; run=root/'results/runs/final-study-independent-retrain-20261003'; out=run/'sensitivity'; X,y,_=exp.tasks()['wdbc']; base=pd.read_csv(out/'wdbc_standardized_pca_control.csv'); core=base[base.method=='quantum_clean_fixed'][['model_id','split','train_seed']].drop_duplicates().sort_values(['split','train_seed']); manifest=json.loads((run/'run_manifest.json').read_text());cfg=manifest['config']; rows=[]
for mr in core.itertuples():
 z=np.load(run/'splits'/f'wdbc_split{int(mr.split)}.npz');tr,ca,te=z['train'],z['calibration'],z['test'];xtr,xc,xt,prep=exp.prepare_features(X,tr,ca,te,2,True);seed=int(mr.train_seed);w,h=exp.train_model(xtr,y[tr],2,4,'cnot_chain',seed,int(cfg['epochs']),int(cfg['batch_size']),float(cfg['learning_rate']))
 clean=exp.make_clean_qnode(2,4,'cnot_chain'); cs=np.array([float(clean(w,x)) for x in xc]); ts=np.array([float(clean(w,x)) for x in xt]);th0=exp.best_balanced_threshold(y[ca],cs)
 q=exp.make_qnode(2,4,'cnot_chain','composite',None,seed+81000); cnoise=exp.batch_scores(q,w,xc,.3);tnoise=exp.batch_scores(q,w,xt,.3);th=exp.best_balanced_threshold(y[ca],cnoise)
 calshots=max(1,int(round(500*len(xt)/len(xc)))); qe_c=exp.make_qnode(2,4,'cnot_chain','composite',calshots,seed+81100);qe_t=exp.make_qnode(2,4,'cnot_chain','composite',500,seed+81200); fcal=exp.batch_scores(qe_c,w,xc,.3);ftest=exp.batch_scores(qe_t,w,xt,.3);fth=exp.best_balanced_threshold(y[ca],fcal)
 mid=f'4layer_split{int(mr.split)}_seed{seed}'
 for method,score,threshold in [('quantum_4layer_clean_fixed',ts,0.),('quantum_4layer_clean_calibrated',ts,th0),('quantum_4layer_noisy_exact_calibrated',tnoise,th),('quantum_4layer_noisy_1000shot_calibrated',ftest,fth)]:
  rows.append({'model_id':mid,'split':int(mr.split),'train_seed':seed,'method':method,'pca_explained_variance_sum':prep['pca_explained_variance_sum'],'balanced_accuracy':exp.balanced_accuracy_score(y[te],score>=threshold),'roc_auc':exp.roc_auc_score(y[te],score)})
 print('finished 4-layer',mr.split,seed,flush=True)
d=pd.DataFrame(rows); p=out/'wdbc_four_layer_control.csv';d.to_csv(p,index=False); sm=d.groupby('method').agg(balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),roc_auc_mean=('roc_auc','mean'),runs=('model_id','nunique')).reset_index();sm.to_csv(out/'wdbc_four_layer_control_summary.csv',index=False);print(sm.to_string(index=False))
