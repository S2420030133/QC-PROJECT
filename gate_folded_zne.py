"""Gate-folded ZNE pilot for standardized WDBC checkpoints; simulator-only."""
from pathlib import Path
import json, hashlib, time
import warnings
import numpy as np
import pandas as pd
import joblib
import pennylane as qml
import research_experiment as exp
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pennylane")

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'results/runs/final-study-independent-retrain-20261003'
S=RUN/'sensitivity'; CK=S/'checkpoints'
SCALES=(1,3,5); COEFS=np.array([15/8,-5/4,3/8],dtype=float)
GAMMA=.30; BUDGET=1000; DRAWS=20
# Integer shot allocation proportional to the absolute Richardson coefficients.
def allocation(total):
    raw=total*np.abs(COEFS)/np.abs(COEFS).sum(); counts=np.floor(raw).astype(int)
    for ix in np.argsort(-(raw-counts))[:total-int(counts.sum())]: counts[ix]+=1
    return counts

def apply_composite(gamma,wires):
    for wire in wires:
        qml.DepolarizingChannel(gamma/2,wires=wire)
        qml.AmplitudeDamping(gamma,wires=wire)

def folded_gate(gate, inv, scale, noise):
    # U (U† U)^k implements U while inserting the physical noise after each occurrence.
    for rep in range(scale):
        inverse=(rep%2==1)
        if gate in ('RX','RY','RZ'):
            angle=-inv if inverse else inv
            {'RX':qml.RX,'RY':qml.RY,'RZ':qml.RZ}[gate](angle,wires=noise[0])
        else:
            qml.CNOT(wires=noise[0])
        noise[1](noise[2],noise[3])

def make_folded(scale,shots,seed):
    dev=qml.device('default.mixed',wires=2,shots=shots,seed=seed)
    @qml.qnode(dev)
    def circuit(weights,x,gamma):
        for q in range(2): folded_gate('RX',x[...,q],scale, ( [q], apply_composite, gamma, [q]))
        for layer in range(2):
            for q in range(2):
                folded_gate('RY',weights[layer,q,0],scale,([q],lambda g,ws: [qml.DepolarizingChannel(g/2,wires=w) for w in ws],gamma,[q]))
                folded_gate('RZ',weights[layer,q,1],scale,([q],lambda g,ws: [qml.DepolarizingChannel(g/2,wires=w) for w in ws],gamma,[q]))
            folded_gate('CNOT',0,scale,([0,1],apply_composite,gamma,[0,1]))
        return qml.expval(qml.PauliZ(0))
    return circuit

def score_batch(weights,X,scale,shots,seed,gamma=GAMMA):
    node=make_folded(scale,shots,seed)
    return np.asarray(node(weights,X,gamma),dtype=float).reshape(-1)

# Check one exact scale-one circuit against the recorded composite model before collecting results.
probe_weights=np.load(next(CK.glob('wdbc_s0_i0_q2_d2_cnot_chain_std0_rev0_standardized.npy')))
prep=joblib.load(next(CK.glob('wdbc_s0_i0_q2_d2_cnot_chain_std0_rev0_standardized_preprocessing.joblib')))
X,y,_=exp.tasks()['wdbc']; split=np.load(RUN/'splits/wdbc_split0.npz')
xc=prep['scaler'].transform(prep['pca'].transform(prep['standardizer'].transform(X[split['test'][:3]])))
# Use exact mode for an apples-to-apples scale-1 check.
def make_exact_folded(scale):
    dev=qml.device('default.mixed',wires=2)
    @qml.qnode(dev)
    def circuit(weights,x,gamma):
        for q in range(2):
            for rep in range(scale):
                qml.RX(x[q] if rep%2==0 else -x[q],wires=q); apply_composite(gamma,[q])
        for layer in range(2):
            for q in range(2):
                for rep in range(scale):
                    qml.RY(weights[layer,q,0] if rep%2==0 else -weights[layer,q,0],wires=q)
                    qml.DepolarizingChannel(gamma/2,wires=q)
                for rep in range(scale):
                    qml.RZ(weights[layer,q,1] if rep%2==0 else -weights[layer,q,1],wires=q)
                    qml.DepolarizingChannel(gamma/2,wires=q)
            for rep in range(scale):
                qml.CNOT(wires=[0,1]); apply_composite(gamma,[0,1])
        return qml.expval(qml.PauliZ(0))
    return circuit
folded1=make_exact_folded(1); regular=exp.make_qnode(2,2,'cnot_chain','composite',None,123)
ref=np.array([float(regular(probe_weights,x,GAMMA)) for x in xc]); test=np.array([float(folded1(probe_weights,x,GAMMA)) for x in xc])
max_scale1_delta=float(np.max(np.abs(ref-test)))
if max_scale1_delta>1e-10: raise RuntimeError(f'scale-one folded circuit differs from existing channel model: {max_scale1_delta}')

models=pd.read_csv(S/'wdbc_standardized_pca_control.csv')
models=models[models.method=='quantum_clean_fixed'][['model_id','split','train_seed']].drop_duplicates().sort_values(['split','train_seed'])
raw=[]; metrics=[]; start=time.time()
for m in models.itertuples():
    archive=np.load(RUN/'splits'/f'wdbc_split{int(m.split)}.npz'); prep=joblib.load(CK/f'{m.model_id}_standardized_preprocessing.joblib'); weights=np.load(CK/f'{m.model_id}_standardized.npy')
    cal=archive['calibration']; testidx=archive['test']; ycal=y[cal]; ytest=y[testidx]
    transform=lambda ix: prep['scaler'].transform(prep['pca'].transform(prep['standardizer'].transform(X[ix])))
    Xcal=transform(cal); Xtest=transform(testidx)
    for draw in range(DRAWS):
        seedbase=202610060000+10000*int(m.split)+100*int(m.train_seed)+draw
        cal_direct=score_batch(weights,Xcal,1,500,seedbase+1)
        test_direct=score_batch(weights,Xtest,1,500,seedbase+2)
        threshold_direct=exp.best_balanced_threshold(ycal,cal_direct)
        metric=exp.metrics(ytest,test_direct,threshold_direct)
        metrics.append({'model_id':m.model_id,'split':int(m.split),'train_seed':int(m.train_seed),'draw':draw,'method':'noisy_threshold_calibrated','balanced_accuracy':metric['balanced_accuracy'],'roc_auc':metric['roc_auc'],'threshold':threshold_direct,'calibration_shots_per_case':500,'test_shots_per_case':500,'total_shots_per_test_case':1000})
        scalescores=[]
        for si,scale in enumerate(SCALES):
            alloc=allocation(500)
            c=score_batch(weights,Xcal,scale,int(alloc[si]),seedbase+100+si)
            t=score_batch(weights,Xtest,scale,int(alloc[si]),seedbase+200+si)
            scalescores.append((c,t))
        zcal=sum(coef*pair[0] for coef,pair in zip(COEFS,scalescores)); ztest=sum(coef*pair[1] for coef,pair in zip(COEFS,scalescores))
        threshold=exp.best_balanced_threshold(ycal,zcal); metric=exp.metrics(ytest,ztest,threshold)
        metrics.append({'model_id':m.model_id,'split':int(m.split),'train_seed':int(m.train_seed),'draw':draw,'method':'gate_folded_zne_calibrated','balanced_accuracy':metric['balanced_accuracy'],'roc_auc':metric['roc_auc'],'threshold':threshold,'calibration_shots_per_case':500,'test_shots_per_case':500,'total_shots_per_test_case':1000})
        for ix,idx in enumerate(cal): raw.append({'model_id':m.model_id,'split':int(m.split),'train_seed':int(m.train_seed),'draw':draw,'partition':'calibration','example_index':int(idx),'label':int(y[idx]),'direct_score':float(cal_direct[ix]),'gate_folded_scale1':float(scalescores[0][0][ix]),'scale3':float(scalescores[1][0][ix]),'scale5':float(scalescores[2][0][ix]),'zne_score':float(zcal[ix])})
        for ix,idx in enumerate(testidx): raw.append({'model_id':m.model_id,'split':int(m.split),'train_seed':int(m.train_seed),'draw':draw,'partition':'test','example_index':int(idx),'label':int(y[idx]),'direct_score':float(test_direct[ix]),'gate_folded_scale1':float(scalescores[0][1][ix]),'scale3':float(scalescores[1][1][ix]),'scale5':float(scalescores[2][1][ix]),'zne_score':float(ztest[ix])})
    print(f'completed {m.model_id}',flush=True)

pd.DataFrame(raw).to_csv(S/'wdbc_gate_folded_zne_observations.csv',index=False)
df=pd.DataFrame(metrics); df.to_csv(S/'wdbc_gate_folded_zne_metrics.csv',index=False)
summary=df.groupby('method').balanced_accuracy.agg(['mean','std','count']).reset_index(); summary.to_csv(S/'wdbc_gate_folded_zne_summary.csv',index=False)
record={'scope':'simulator-only; same saved standardized WDBC checkpoints/splits','gamma':GAMMA,'scales':SCALES,'coefficients':COEFS.tolist(),'coefficient_l1_norm':float(np.abs(COEFS).sum()),'variance_factor_equal_variance':float(np.abs(COEFS).sum()**2),'budget_per_test_case':BUDGET,'calibration_fraction':0.5,'shot_draws_per_checkpoint':DRAWS,'checkpoints':len(models),'scale1_exact_max_abs_difference_vs_existing_composite_model':max_scale1_delta,'runtime_seconds':time.time()-start,'pennylane':qml.__version__}
(S/'wdbc_gate_folded_zne_protocol.json').write_text(json.dumps(record,indent=2))
print(summary.to_string(index=False));print(json.dumps(record,indent=2))
