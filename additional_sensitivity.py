"""Targeted sensitivity analyses over frozen final-study checkpoints."""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_experiment as exp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-id", default="final-study-20260929",
                    help="Run ID under results/runs whose frozen checkpoints are analyzed")
args = parser.parse_args()
RUN = Path(__file__).resolve().parent / "results" / "runs" / args.run_id
if not (RUN / "models.csv").is_file():
    raise SystemExit(f"No completed model table found for run {args.run_id!r}: {RUN}")
OUT = RUN / "sensitivity"
OUT.mkdir(exist_ok=True)
models = pd.read_csv(RUN / "models.csv")
core = models[(models.task == "wdbc") & (models.n_qubits == 2) &
              (models.architecture == "cnot_chain") & (models.n_layers == 2) & (~models.pca_standardize) &
              (~models.reverse_labels)].sort_values(["split", "train_seed"])
data = exp.tasks()["wdbc"]
X, y, _ = data
exact_rows=[]
# Exact-expectation sensitivity to smaller probability-scaling schedules.
scale_sets = {"1_1p25": [1.0,1.25], "1_1p5": [1.0,1.5],
              "1_1p25_1p5": [1.0,1.25,1.5]}
for mr in core.itertuples():
    cp=RUN/"checkpoints"/f"{mr.model_id}_preprocessing.joblib"
    prep=joblib.load(cp)
    tr,ca,te=prep["train_ids"],prep["calibration_ids"],prep["test_ids"]
    # Refit identically from saved split indices; transforms are deterministic.
    xt,xc,xv,_=exp.prepare_features(X,tr,ca,te,2,False)
    weights=np.load(RUN/"checkpoints"/f"{mr.model_id}.npy")
    qexact=exp.make_qnode(2,2,"cnot_chain","composite",None,42)
    clean=exp.make_clean_qnode(2,2,"cnot_chain")
    clean_test=np.array([float(clean(weights,x)) for x in xv])
    ytest=y[te]
    for gamma in [.1,.2,.3,.4]:
        for name,scales in scale_sets.items():
            coeff=np.array([np.prod([-s2/(s-s2) for s2 in scales if s2!=s]) for s in scales],dtype=float)
            # Solve the interpolation constraints directly to avoid formula ambiguity.
            # Richardson weights satisfy sum_i c_i * scale_i**k = 1 for k=0
            # and 0 for all higher retained powers.
            vandermonde=np.array(scales,dtype=float)[None,:]**np.arange(len(scales))[:,None]
            coeff=np.linalg.solve(vandermonde,np.eye(len(scales))[0])
            scores=np.round(sum(c*exp.batch_scores(qexact,weights,xv,gamma*scale)
                                for scale,c in zip(scales,coeff)),12)
            exact_rows.append({"model_id":mr.model_id,"split":mr.split,"train_seed":mr.train_seed,
                               "gamma":gamma,"scale_set":name,"scales":";".join(map(str,scales)),
                               "coefficients":";".join(f"{v:.8g}" for v in coeff),
                               "balanced_accuracy":exp.balanced_accuracy_score(ytest,scores>=0),
                               "roc_auc":exp.roc_auc_score(ytest,scores),
                               "observable_rmse_vs_clean":float(np.sqrt(np.mean((scores-clean_test)**2)))})
pd.DataFrame(exact_rows).to_csv(OUT/"zne_scale_sensitivity_exact.csv",index=False)

# Finite-shot allocation sensitivity. Fractions partition a fixed nominal budget
# per test case between labeled calibration and held-out inference.
allocation_rows=[]
for mr in core.itertuples():
    prep=joblib.load(RUN/"checkpoints"/f"{mr.model_id}_preprocessing.joblib")
    tr,ca,te=prep["train_ids"],prep["calibration_ids"],prep["test_ids"]
    _,xcal,xtest,_=exp.prepare_features(X,tr,ca,te,2,False)
    ycal,ytest=y[ca],y[te]
    weights=np.load(RUN/"checkpoints"/f"{mr.model_id}.npy")
    for fraction in [.25,.5,.75]:
        nominal=1000
        cal_nominal=int(round(nominal*fraction)); test_nominal=nominal-cal_nominal
        cal_per=max(1,int(round(cal_nominal*len(xtest)/len(xcal))))
        test_per=max(1,test_nominal)
        for rep in range(2):
            seed=42+100000*int(mr.split)+1000*int(mr.train_seed)+rep+50000
            for method,scales,coeffs in [
                ("calibrated_threshold",None,None),
                ("linear_zne_calibrated",exp.SCALE_LINEAR,exp.COEF_LINEAR),
                ("quadratic_zne_calibrated",exp.SCALE_QUADRATIC,exp.COEF_QUADRATIC)]:
                if method=="calibrated_threshold":
                    qc=exp.make_qnode(2,2,"cnot_chain","composite",cal_per,seed+10)
                    qt=exp.make_qnode(2,2,"cnot_chain","composite",test_per,seed+20)
                    cscore=exp.batch_scores(qc,weights,xcal,.3)
                    tscore=exp.batch_scores(qt,weights,xtest,.3)
                    cal_total=cal_per*len(xcal); test_total=test_per*len(xtest)
                else:
                    cscore,ca_alloc,_=exp.finite_zne_scores(weights,xcal,.3,cal_per,scales,coeffs,2,2,"cnot_chain",seed+10)
                    tscore,te_alloc,_=exp.finite_zne_scores(weights,xtest,.3,test_per,scales,coeffs,2,2,"cnot_chain",seed+20)
                    cal_total=int(sum(ca_alloc)*len(xcal)); test_total=int(sum(te_alloc)*len(xtest))
                threshold=exp.best_balanced_threshold(ycal,cscore)
                allocation_rows.append({"model_id":mr.model_id,"split":mr.split,"train_seed":mr.train_seed,
                    "replicate":rep,"gamma":.3,"nominal_shots_per_test_case":nominal,
                    "calibration_fraction":fraction,"calibration_shots_per_example":cal_per,
                    "test_shots_per_example":test_per,"method":method,"selected_threshold":threshold,
                    "calibration_measurement_shots":cal_total,"test_measurement_shots":test_total,
                    "total_measurement_shots":cal_total+test_total,
                    **exp.metrics(ytest,tscore,threshold)})
pd.DataFrame(allocation_rows).to_csv(OUT/"calibration_allocation_sensitivity.csv",index=False)
summary=(pd.DataFrame(allocation_rows).groupby(["calibration_fraction","method"],as_index=False)
         .agg(balanced_accuracy_mean=("balanced_accuracy","mean"),balanced_accuracy_sd=("balanced_accuracy","std"),
              auc_mean=("roc_auc","mean"),total_shots_mean=("total_measurement_shots","mean"),
              models=("model_id","nunique"),outer_splits=("split","nunique")))
summary.to_csv(OUT/"calibration_allocation_sensitivity_summary.csv",index=False)
print("Wrote sensitivity outputs",len(exact_rows),len(allocation_rows),flush=True)
print(summary.to_string(index=False),flush=True)
