"""Record experiment protocols and source hashes for post-hoc sensitivity runs."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"results/runs/final-study-independent-retrain-20261003"
manifest_path=RUN/"run_manifest.json"
m=json.loads(manifest_path.read_text(encoding="utf-8"))
def sha(name): return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
m["standardized_wdbc_shot_repeats_extension"]={
 "script":"standardized_finite_shot_repeats.py","script_sha256":sha("standardized_finite_shot_repeats.py"),
 "trained_models":10,"outer_folds":5,"initializations_per_fold":2,"shot_draws_per_checkpoint":20,
 "gamma":0.30,"nominal_total_shots_per_test_case":1000,"calibration_fraction":0.5,
 "seed_rule":"base=1800000+10000*split+1000*train_seed+10*replicate; noisy calibration/test=base+1/+2; linear ZNE calibration/test=base+100/+300; quadratic ZNE calibration/test=base+200/+400; per-scale simulator seeds increment within each call.",
 "outputs":["sensitivity/wdbc_standardized_finite_shot_20draws.csv","sensitivity/wdbc_standardized_finite_shot_20draws_observations.csv","sensitivity/wdbc_standardized_zne_intermediate_observations.csv","sensitivity/wdbc_standardized_zne_comparison.csv","sensitivity/wdbc_standardized_paired_fold_differences.csv"]}
m["standardized_wdbc_calibration_size_extension"]={
 "script":"standardized_calibration_size_sweep.py","script_sha256":sha("standardized_calibration_size_sweep.py"),
 "subset_index_script":"record_calibration_subset_ids.py","subset_index_script_sha256":sha("record_calibration_subset_ids.py"),
 "calibration_example_counts":[10,20,50,114],"draws_per_checkpoint":20,"nominal_total_shots_per_test_case":1000,
 "test_scores_reused_across_calibration_sizes":True,"subset_indices_saved":True,
 "seed_rule":"subset=2500000+10000*split+1000*train_seed+100*replicate+calibration_examples; calibration simulator seed is subset seed plus 1 (noisy threshold), plus 10 (linear ZNE), or plus 20 (quadratic ZNE), with per-scale increments.",
 "outputs":["sensitivity/wdbc_standardized_calibration_size_sweep.csv","sensitivity/wdbc_standardized_calibration_size_sweep_summary.csv"]}
m["standardized_wdbc_depth_extension"]={
 "script":"standardized_depth_sweep.py","script_sha256":sha("standardized_depth_sweep.py"),
 "audit_script":"audit_standardized_depth_sweep.py","audit_script_sha256":sha("audit_standardized_depth_sweep.py"),
 "layers_retrained":[1,4],"comparison_layer_reused":2,"folds":5,"initializations_per_fold":2,"shot_draws_per_checkpoint":20,
 "gamma":0.30,"nominal_total_shots_per_test_case":1000,"calibration_fraction":0.5,
 "seed_rule":"training uses the original 2 initialization seeds per fold; shot base=3100000+10000*split+1000*train_seed+100*depth+replicate; calibration/test simulator seed is base plus 1/2.",
 "audit":"800 finite-shot rows recomputed from saved per-example scores; zero metric mismatches; maximum nominal budget deviation 0.0003894.",
 "outputs":["sensitivity/wdbc_standardized_depth_sweep.csv","sensitivity/wdbc_standardized_depth_sweep_observations.csv","sensitivity/wdbc_standardized_depth_sweep_training_curves.csv","sensitivity/wdbc_standardized_depth_sweep_summary.csv","sensitivity/wdbc_standardized_depth_sweep_audit.json","sensitivity/checkpoints/ depth1 and depth4 weights and preprocessing"]}
m["standardized_wdbc_exact_noise_grid"]={
 "script":"standardized_exact_noise_sweep.py","script_sha256":sha("standardized_exact_noise_sweep.py"),
 "audit_script":"audit_standardized_exact_noise_sweep.py","audit_script_sha256":sha("audit_standardized_exact_noise_sweep.py"),
 "audit":"150 score-observation groups and 200 exact metric rows recomputed; zero mismatches.",
 "folds":5,"initializations_per_fold":2,"gamma_grid":[0.0,0.1,0.2,0.3,0.4],
 "methods":["fixed_threshold","noisy_threshold","linear_zne_calibrated","quadratic_zne_calibrated"],
 "outputs":["sensitivity/wdbc_standardized_exact_noise_sweep.csv","sensitivity/wdbc_standardized_exact_noise_sweep_observations.csv","sensitivity/wdbc_standardized_exact_noise_sweep_summary.csv","sensitivity/wdbc_standardized_exact_noise_sweep_audit.json"]}
m["standardized_primary_exact_metric_reconciliation"]={
 "script":"reconcile_standardized_exact_metrics.py","script_sha256":sha("reconcile_standardized_exact_metrics.py"),
 "corrected_model_rows":10,"method":"quantum_noisy_exact_calibrated","gamma":0.30,
 "basis":"Recomputed from saved per-example exact scores generated using the same checkpoint, preprocessing, split indices, and source implementation; before/after thresholds and metrics are logged.",
 "prior_mean_balanced_accuracy":0.891283164101,"reconciled_mean_balanced_accuracy":0.878386761135,
 "reconciliation_log":"sensitivity/wdbc_standardized_exact_metric_reconciliation.json",
 "note":"A previously reported exact standardized-PCA table did not reproduce from its saved checkpoints and splits. The manuscript and summary now use the reconciled values; the cause of the discrepancy is not established."}
m["standardized_transfer_extension"]={
 "script":"standardized_transfer_comparison.py","script_sha256":sha("standardized_transfer_comparison.py"),
 "classical_script":"standardized_transfer_classical.py","classical_script_sha256":sha("standardized_transfer_classical.py"),
 "audit_script":"audit_standardized_transfer.py","audit_script_sha256":sha("audit_standardized_transfer.py"),
 "tasks":["wine_0_vs_rest","synthetic_balanced"],"outer_folds":5,"initializations_per_fold":2,
 "freshly_retrained_models":20,"shot_draws_per_checkpoint":20,"gamma":0.30,
 "nominal_total_measurements_per_test_case":1000,"calibration_fraction":0.5,
 "audit":"1,200 calibration/test method-draw groups and 2,400 finite-shot metric rows recomputed from saved observations with zero mismatches; maximum budget deviation 0.0115%.",
 "outputs":["sensitivity/standardized_transfer_comparison.csv","sensitivity/standardized_transfer_observations.csv","sensitivity/standardized_transfer_clean.csv","sensitivity/standardized_transfer_classical.csv","sensitivity/standardized_transfer_training_curves.csv","sensitivity/standardized_transfer_summary.csv","sensitivity/standardized_transfer_audit.json","sensitivity/checkpoints/*_standardized.npy and preprocessing.joblib"]}
m["standardized_depth_score_spread_diagnostic"]={
 "script":"depth_score_spread.py","script_sha256":sha("depth_score_spread.py"),
 "depths":[1,2,4],"noise_gamma":0.30,"score_partition":"held-out calibration data",
 "shots_for_single-score_SE_diagnostic":500,
 "outputs":["sensitivity/wdbc_standardized_depth_score_spread.csv","sensitivity/wdbc_standardized_depth_score_spread_summary.csv"]}
gatefold_protocol=json.loads((RUN/"sensitivity"/"wdbc_gate_folded_zne_protocol.json").read_text())
m["gate_folded_zne_extension"]={
 "script":"gate_folded_zne.py","script_sha256":sha("gate_folded_zne.py"),
 "audit_script":"audit_gate_folded_zne.py","audit_script_sha256":sha("audit_gate_folded_zne.py"),
 "protocol":gatefold_protocol,
 "audit":json.loads((RUN/"sensitivity"/"wdbc_gate_folded_zne_audit.json").read_text()),
 "outputs":["sensitivity/wdbc_gate_folded_zne_observations.csv","sensitivity/wdbc_gate_folded_zne_metrics.csv","sensitivity/wdbc_gate_folded_zne_summary.csv","sensitivity/wdbc_gate_folded_zne_protocol.json","sensitivity/wdbc_gate_folded_zne_audit.json"]}
fake_protocol=json.loads((RUN/"sensitivity"/"wdbc_fake_backend_protocol.json").read_text())
m["fake_backend_extension"]={
 "script":"evaluate_fake_backend.py","script_sha256":sha("evaluate_fake_backend.py"),
 "audit_script":"audit_fake_backend.py","audit_script_sha256":sha("audit_fake_backend.py"),
 "protocol":fake_protocol,
 "audit":json.loads((RUN/"sensitivity"/"wdbc_fake_backend_audit.json").read_text()),
 "outputs":["sensitivity/wdbc_fake_backend_observations.csv","sensitivity/wdbc_fake_backend_metrics.csv","sensitivity/wdbc_fake_backend_summary.csv","sensitivity/wdbc_fake_backend_protocol.json","sensitivity/wdbc_fake_backend_audit.json"]}
fake_sweep_protocol=json.loads((RUN/"sensitivity"/"wdbc_fake_backend_noise_sweep_protocol.json").read_text())
m["fake_backend_noise_sweep_extension"]={
 "script":"evaluate_fake_backend_noise_sweep.py","script_sha256":sha("evaluate_fake_backend_noise_sweep.py"),
 "audit_script":"audit_fake_backend_noise_sweep.py","audit_script_sha256":sha("audit_fake_backend_noise_sweep.py"),
 "protocol":fake_sweep_protocol,
 "audit":json.loads((RUN/"sensitivity"/"wdbc_fake_backend_noise_sweep_audit.json").read_text()),
 "outputs":["sensitivity/wdbc_fake_backend_noise_sweep_metrics.csv","sensitivity/wdbc_fake_backend_noise_sweep_summary.csv","sensitivity/wdbc_fake_backend_noise_sweep_protocol.json","sensitivity/wdbc_fake_backend_noise_sweep_audit.json","fake_manila_regime_map.png"]}
m["fake_backend_native_fold_replay"]={
 "script":"fake_backend_gatefolded_zne.py","script_sha256":sha("fake_backend_gatefolded_zne.py"),
 "audit_script":"audit_fake_backend_native_fold.py","audit_script_sha256":sha("audit_fake_backend_native_fold.py"),
 "description":"Separate fixed-seed replay using native-gate folding, 2,000 calibration shots per readout basis state, and the same saved five folds and ten checkpoints; exploratory sensitivity, not the primary table.",
 "protocol":json.loads((RUN/"sensitivity"/"wdbc_fake_manila_gatefolded_protocol.json").read_text()),
 "audit":json.loads((RUN/"sensitivity"/"wdbc_fake_manila_gatefolded_audit.json").read_text()),
 "outputs":["sensitivity/wdbc_fake_manila_gatefolded_observations.csv","sensitivity/wdbc_fake_manila_gatefolded_metrics.csv","sensitivity/wdbc_fake_manila_gatefolded_summary.csv","sensitivity/wdbc_fake_manila_gatefolded_protocol.json","sensitivity/wdbc_fake_manila_gatefolded_audit.json"]}
manifest_path.write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8")
print("Registered repeated-shot, calibration-size, depth, exact-noise-grid, reconciliation, standardized-transfer, score-spread, gate-folded ZNE, fake-backend, and fake-backend noise sweep protocols.")


