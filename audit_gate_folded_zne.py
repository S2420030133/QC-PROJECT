"""Recompute gate-folded ZNE metrics and protocol consistency from saved scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import research_experiment as exp
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'results/runs/final-study-independent-retrain-20261003'; S=RUN/'sensitivity'
obs=pd.read_csv(S/'wdbc_gate_folded_zne_observations.csv')
metrics=pd.read_csv(S/'wdbc_gate_folded_zne_metrics.csv')
protocol=json.loads((S/'wdbc_gate_folded_zne_protocol.json').read_text())
y=exp.tasks()['wdbc'][1]
errors=[]
for row in metrics.itertuples():
    group=obs[(obs.model_id==row.model_id)&(obs.draw==row.draw)&(obs.partition=='test')]
    group=group.sort_values('example_index')
    truth=group.label.to_numpy(dtype=int)
    score=(group.direct_score if row.method=='noisy_threshold_calibrated' else group.zne_score).to_numpy(dtype=float)
    rec=exp.metrics(truth,score,float(row.threshold))
    errors.append(abs(rec['balanced_accuracy']-row.balanced_accuracy))
expected_models=10; expected_draws=int(protocol['shot_draws_per_checkpoint'])
assert metrics.model_id.nunique()==expected_models
assert len(metrics)==expected_models*expected_draws*2
assert np.max(errors)<1e-12
assert metrics.total_shots_per_test_case.eq(1000).all()
assert protocol['scale1_exact_max_abs_difference_vs_existing_composite_model']<1e-10
result={'models':int(metrics.model_id.nunique()),'draws_per_checkpoint':expected_draws,'metric_rows':int(len(metrics)),'recomputed_balanced_accuracy_max_abs_error':float(max(errors)),'budget_mismatches':int((metrics.total_shots_per_test_case!=1000).sum()),'scale1_equivalence_max_abs_error':protocol['scale1_exact_max_abs_difference_vs_existing_composite_model'],'status':'PASS'}
(S/'wdbc_gate_folded_zne_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
