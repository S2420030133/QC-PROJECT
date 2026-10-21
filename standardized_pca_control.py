"""Standardized-before-PCA WDBC control on the saved outer folds and seeds."""
from pathlib import Path
import json, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import research_experiment as exp
ROOT=Path(__file__).resolve().parent; run=ROOT/'results/runs/final-study-independent-retrain-20261003'; out=run/'sensitivity'; out.mkdir(exist_ok=True)
manifest=json.loads((run/'run_manifest.json').read_text()); cfg=manifest['config']; X,y,_=exp.tasks()['wdbc']; models=pd.read_csv(run/'models.csv'); core=models[(models.task=='wdbc')&(models.n_qubits==2)&(models.architecture=='cnot_chain')&(models.n_layers==2)&(~models.reverse_labels)&(~models.pca_standardize)].sort_values(['split','train_seed'])
rows=[]; zne_rows=[]; zne_observations=[]
if len(core)!=10 or core.model_id.duplicated().any(): raise RuntimeError(f'Expected 10 unique unstandardized source checkpoints, got {len(core)}')
for mr in core.itertuples():
 z=np.load(run/'splits'/f'wdbc_split{int(mr.split)}.npz'); tr,ca,te=z['train'],z['calibration'],z['test']; xtr,xc,xt,prep=exp.prepare_features(X,tr,ca,te,2,True); seed=int(mr.train_seed); w,hist=exp.train_model(xtr,y[tr],2,2,'cnot_chain',seed,int(cfg['epochs']),int(cfg['batch_size']),float(cfg['learning_rate']))
 ckpt=out/'checkpoints'; ckpt.mkdir(exist_ok=True); np.save(ckpt/f'{mr.model_id}_standardized.npy',w); joblib.dump(prep,ckpt/f'{mr.model_id}_standardized_preprocessing.joblib')
 clean=exp.make_clean_qnode(2,2,'cnot_chain'); cclean=np.array([float(clean(w,x)) for x in xc]); tclean=np.array([float(clean(w,x)) for x in xt]); clean_threshold=exp.best_balanced_threshold(y[ca],cclean)
 qexact=exp.make_qnode(2,2,'cnot_chain','composite',None,seed+30000); noisecal=exp.batch_scores(qexact,w,xc,.3); noisetest=exp.batch_scores(qexact,w,xt,.3); threshold=exp.best_balanced_threshold(y[ca],noisecal)
 fit_ids,_=train_test_split(np.arange(len(y[tr])),test_size=.15,random_state=seed+31415,stratify=y[tr]); lr=LogisticRegression(max_iter=2000,random_state=seed).fit(xtr[fit_ids],y[tr][fit_ids]); lr_test=lr.predict_proba(xt)[:,1]
 # Same nominal total 1000 budget: half for calibration and half for test inference.
 cal_n=max(1,int(round(500*len(xt)/len(xc)))); test_n=500
 qcal=exp.make_qnode(2,2,'cnot_chain','composite',cal_n,seed+30100); qtest=exp.make_qnode(2,2,'cnot_chain','composite',test_n,seed+30200)
 cshot=exp.batch_scores(qcal,w,xc,.3); tshot=exp.batch_scores(qtest,w,xt,.3); shotthr=exp.best_balanced_threshold(y[ca],cshot)
 base={"model_id":mr.model_id,"split":int(mr.split),"train_seed":seed,"pca_explained_variance_sum":prep['pca_explained_variance_sum']}
 # Two repeated matched-budget draws for noisy threshold calibration, included alongside ZNE.
 for rep in range(2):
  repseed=800000+1000*int(mr.split)+100*int(mr.train_seed)+rep
  qcal_rep=exp.make_qnode(2,2,'cnot_chain','composite',cal_n,repseed+10); qtest_rep=exp.make_qnode(2,2,'cnot_chain','composite',test_n,repseed+20)
  c_rep=exp.batch_scores(qcal_rep,w,xc,.3); t_rep=exp.batch_scores(qtest_rep,w,xt,.3); thr_rep=exp.best_balanced_threshold(y[ca],c_rep)
  for mode,decision in [('finite_fixed',0.),('finite_calibrated',thr_rep)]:
   zne_rows.append({**base,'gamma':.3,'budget':1000,'replicate':rep,'method':'noisy_threshold','measurement_mode':mode,'threshold':float(decision),'balanced_accuracy':exp.balanced_accuracy_score(y[te],t_rep>=decision),'roc_auc':exp.roc_auc_score(y[te],t_rep),'calibration_shots':cal_n*len(xc),'test_shots':test_n*len(xt),'total_shots':cal_n*len(xc)+test_n*len(xt)})
  for part,ids,ys,vals,threshold in [('calibration',ca,y[ca],c_rep,thr_rep),('test',te,y[te],t_rep,thr_rep)]:
   for ix,label,value in zip(ids,ys,vals): zne_observations.append({**base,'method':'noisy_threshold','replicate':rep,'partition':part,'example_index':int(ix),'y_true':int(label),'scale':1.0,'shots_per_example':cal_n if part=='calibration' else test_n,'score':float(value),'selected_threshold':float(threshold)})
 # Matched exact-expectation and finite-shot ZNE comparison on identical standardized folds.
 schedules=[('linear_zne',[1.,2.],np.array([2.,-1.])),('quadratic_zne',[1.,1.5,2.],np.array([6.,-8.,3.]))]
 for method,scales,coeffs in schedules:
  exact_cal=np.round(sum(c*exp.batch_scores(qexact,w,xc,.3*scale) for scale,c in zip(scales,coeffs)),12)
  exact_test=np.round(sum(c*exp.batch_scores(qexact,w,xt,.3*scale) for scale,c in zip(scales,coeffs)),12)
  exact_thr=exp.best_balanced_threshold(y[ca],exact_cal)
  for kind,score,thr in [('exact_fixed',exact_test,0.),('exact_calibrated',exact_test,exact_thr)]:
   zne_rows.append({**base,'gamma':.3,'budget':0,'replicate':0,'method':method,'measurement_mode':kind,'threshold':float(thr),'balanced_accuracy':exp.balanced_accuracy_score(y[te],score>=thr),'roc_auc':exp.roc_auc_score(y[te],score)})
  for rep in range(2):
   repseed=900000+1000*int(mr.split)+100*int(mr.train_seed)+rep
   cal_scores,cal_alloc,cal_scales=exp.finite_zne_scores(w,xc,.3,cal_n,scales,coeffs,2,2,'cnot_chain',repseed+100)
   test_scores,test_alloc,test_scales=exp.finite_zne_scores(w,xt,.3,test_n,scales,coeffs,2,2,'cnot_chain',repseed+200)
   thr=exp.best_balanced_threshold(y[ca],cal_scores)
   for kind,decision in [('finite_fixed',0.),('finite_calibrated',thr)]:
    zne_rows.append({**base,'gamma':.3,'budget':1000,'replicate':rep,'method':method,'measurement_mode':kind,'threshold':float(decision),'balanced_accuracy':exp.balanced_accuracy_score(y[te],test_scores>=decision),'roc_auc':exp.roc_auc_score(y[te],test_scores),'calibration_shots':int(sum(cal_alloc)*len(xc)),'test_shots':int(sum(test_alloc)*len(xt)),'total_shots':int(sum(cal_alloc)*len(xc)+sum(test_alloc)*len(xt))})
   for part,ids,ys,vals,allocs in [('calibration',ca,y[ca],cal_scales,cal_alloc),('test',te,y[te],test_scales,test_alloc)]:
    for scale,shots,values in zip(scales,allocs,vals):
     for ix,label,value in zip(ids,ys,values):
      zne_observations.append({**base,'method':method,'replicate':rep,'partition':part,'example_index':int(ix),'y_true':int(label),'scale':scale,'shots_per_example':int(shots),'score':float(value),'selected_threshold':float(thr) if part=='calibration' else float(thr)})
 for name,score,thr in [('quantum_clean_fixed',tclean,0.),('quantum_clean_calibrated',tclean,clean_threshold),('quantum_noisy_exact_fixed',noisetest,0.),('quantum_noisy_exact_calibrated',noisetest,threshold),('quantum_noisy_1000shot_fixed',tshot,0.),('quantum_noisy_1000shot_calibrated',tshot,shotthr),('logistic_regression_standardized_2d',lr_test,.5)]:
  rows.append({**base,'method':name,'threshold':float(thr),'balanced_accuracy':exp.balanced_accuracy_score(y[te],score>=thr),'roc_auc':exp.roc_auc_score(y[te],score)})
 print('done fold',mr.split,'seed',seed,'PCA variance',round(prep['pca_explained_variance_sum'],4),flush=True)
df=pd.DataFrame(rows); df.to_csv(out/'wdbc_standardized_pca_control.csv',index=False)
zdf=pd.DataFrame(zne_rows); zdf.to_csv(out/'wdbc_standardized_zne_comparison.csv',index=False)
pd.DataFrame(zne_observations).to_csv(out/'wdbc_standardized_zne_intermediate_observations.csv',index=False)
summary=zdf.groupby(['method','measurement_mode','budget'],as_index=False).agg(balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),roc_auc_mean=('roc_auc','mean'),n_runs=('model_id','nunique'),mean_total_shots=('total_shots','mean'))
summary.to_csv(out/'wdbc_standardized_zne_comparison_summary.csv',index=False); print(summary.to_string(index=False))
s=df.groupby('method').agg(balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),roc_auc_mean=('roc_auc','mean'),runs=('model_id','nunique')).reset_index(); s.to_csv(out/'wdbc_standardized_pca_control_summary.csv',index=False); print(s.to_string(index=False))

