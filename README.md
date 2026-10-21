# Noise, decision thresholds, and mitigation in small VQCs

The active manuscript and generator were revised on 2026-10-17 to repair figure placement and table layout, clarify the synthetic-versus-FakeManila interpretation, and identify the existing Platt-style logistic score comparator. See `MANUSCRIPT_REVIEW_20261017.md` for changes, verification, and remaining work. Saved experiment runs and older submission archives retain their original contents.

This repository is a **computational research pilot**, not a publication-ready or clinically validated study. Its question is: when a noisy variational quantum classifier predicts only one class, how much can held-out calibration recover, and when does limited measurement precision erase that recovery? It compares calibration with linear and quadratic zero-noise extrapolation (ZNE) at a nominally matched shot budget. Integer allocation across folds with unequal calibration and test sizes makes realized totals differ by at most 0.443%.

The pilot uses the Wisconsin Diagnostic Breast Cancer (WDBC) benchmark, a binary reduction of the Wine dataset, and balanced and 80:20 imbalanced synthetic binary tasks. Quantum models use training-only feature preprocessing, with separate calibration and outer test partitions. The original quantum sweep includes 2, 4, and 6 qubits, multiple circuit entanglers, two depths, unstandardized PCA, and a reversed WDBC label mapping. The primary WDBC matched-budget comparison uses standardized-before-PCA inputs and compares noisy threshold fitting against linear/quadratic ZNE with calibration cost included. It averages 20 independent shot draws per saved checkpoint (10 trained checkpoints); these are repeated measurement draws, not extra training runs or independent folds. A standardized transfer extension separately trains matched two-qubit models on Wine and balanced synthetic data while reusing the same saved outer split assignments and initialization seeds; it uses the same 20-draw matched-budget comparison. Threshold selection leads the quantum pipelines on those tasks but remains below matched clean logistic-regression baselines. See `headline_primary_wdbc.png`, the manuscript, and supplementary tables in the final-study run directory. A separate gate-folded ZNE simulator study uses scales 1, 3, and 5 with the same checkpoints, split assignments, and total-shot accounting. It yields 0.682 balanced accuracy (AUC 0.757) versus 0.776 for noisy-threshold calibration; the 400 finite-shot metric rows recompute from saved observations with zero mismatches. A local Aer simulation using the saved IBM `FakeManilaV2` 5-qubit Falcon backend snapshot reports 0.873 balanced accuracy for raw-noise threshold calibration, 0.878 after single-qubit readout correction, and 0.885 for calibrated gate-folded ZNE. The 70 metric rows recompute within floating-point precision; differences are within checkpoint variability and do not establish a consistent mitigation benefit. This is not a QPU run or current device calibration. The calibration-size sweep gives 0.760 with 10 labels and 0.783 with 114; its independent calibration-shot RNG stream explains the small difference from the main 0.778 result. The older unstandardized-PCA results remain labeled secondary; the four-qubit CNOT-star inputs passed a pre-training readout-dependence probe. The remaining architecture controls are diagnostic; the weak imbalanced synthetic task remains a reported negative result. The exact WDBC primary result was reconciled from saved per-example scores and is 0.878 balanced accuracy, replacing a prior non-reproducible 0.891 summary value.

Fake-backend runs are local simulations using the saved FakeManilaV2 device snapshot, not current device calibration or hardware execution. To reproduce them, first install the standard project environment and then the optional snapshot-simulation dependencies with `python -m pip install -r requirements-fake-backend.txt`. `evaluate_fake_backend.py` and `evaluate_fake_backend_noise_sweep.py` run the primary matched-budget experiments; their matching `audit_*.py` scripts recompute the saved metrics. An additional fixed-seed native-gate-folding replay is saved as an exploratory sensitivity and is not used in the primary manuscript table.


## View the saved results

Install the requirements, then run `python -m streamlit run app.py`. The dashboard defaults to `final-study-independent-retrain-20261003`; use its saved-run selector to open other available runs.

## Reproduce

Python 3.12 is recommended. Create an isolated environment and install the pinned direct dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The CLI defaults now match the recorded final study: 5 stratified outer splits, 2 training initializations, 1 architecture-sweep split, 30 epochs, 100/1,000 shot budgets, and 2 finite-shot repetitions. Each run gets a new folder under `results/runs/`; specify a unique `--run-id`. Existing run IDs are rejected so a development run cannot overwrite a completed run.

For a quick development run, use fewer folds, one initialization, and one shot repetition:

```powershell
.\.venv\Scripts\python.exe research_experiment.py --run-id dev-check --outer-splits 2 --train-seeds 1 --arch-sweep-splits 1 --epochs 2 --batch-size 32 --shot-budgets 100,1000 --shot-repetitions 1
```

On Linux or macOS, install and run with Python 3.12 as follows:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python research_experiment.py --run-id final-study-reproduction --outer-splits 5 --train-seeds 2 --arch-sweep-splits 1 --epochs 30 --batch-size 32 --shot-budgets 100,1000 --shot-repetitions 2
```

The Windows command below reproduces the final-study configuration in a fresh output folder:

```powershell
.\.venv\Scripts\python.exe research_experiment.py --run-id final-study-reproduction --outer-splits 5 --train-seeds 2 --arch-sweep-splits 1 --epochs 30 --batch-size 32 --shot-budgets 100,1000 --shot-repetitions 2
```


A separate targeted sensitivity script evaluates smaller ZNE probability-scale schedules using saved checkpoints and tests calibration shares of 25%, 50%, and 75% at the primary WDBC setting. It does not retrain models:

```text
python additional_sensitivity.py --run-id <run-id>
```

The range-sensitivity analysis reuses saved finite-shot ZNE scale measurements, clips extrapolated Pauli-Z scores to the physical interval `[-1, 1]`, and refits the decision threshold on calibration observations only. It draws no new quantum samples and does not change the measurement budget:

```text
python zne_range_sensitivity.py --run-id <run-id>
```

The noise parameters are **per-channel-location probabilities** in an explicitly synthetic circuit model, not an overall circuit error rate and not physical (T_1/T_2) times. The composite condition applies amplitude damping with probability \(\gamma\) after encoding and entangling gates, and depolarization with probability \(\gamma/2\) after encoding, variational, and entangling gates. The amplitude-only control uses the same amplitude-damping locations; phase damping and depolarization are separate channels after each declared gate location. No device calibration or hardware experiment is represented.

## Outputs

The run writes an auditable `results/runs/<run-id>/` directory:

- `run_manifest.json`: package versions, seeds, data description, circuit and noise configuration, shot costs, and limitations.
- `models.csv`, `exact_performance.csv`, `finite_shot_performance.csv`, `per_example_observations.csv`, and `classical_baselines.csv`: checkpoints’ metrics and saved per-example model outputs.
- `training_curves.csv`: per-epoch training and development loss and development balanced accuracy. `calibration_and_scale_observations.csv` saves calibration scores, selected thresholds, and individual ZNE scale estimates; finite-shot affine-fit parameters are in `finite_shot_performance.csv`. `input_dependence_probe.csv` records the pre-training feature-to-readout probe.
- `classical_matched_fit_size.csv` and its per-example file: a logistic regression control using the same two PCA features and the exact weight-fit examples used by each corresponding quantum model.
- `checkpoints/`: each frozen parameter array, fitted preprocessing objects, and split indices.
- `splits/`: train, calibration, and test indices for every task and outer split.
- `exact_summary.csv`, `finite_shot_summary.csv`, `channel_ablation_summary.csv`, `error_decomposition_summary.csv`, and `paired_differences_bootstrap_ci.csv`: summaries, channel controls, observable-level error decomposition, and paired split-clustered intervals. The paired table includes direct calibration-versus-ZNE contrasts as well as contrasts against the fixed-threshold reference.
- `exact_accuracy_auc.png`, `score_distributions_wdbc.png`, `finite_shots_budget.png`, `score_error_decomposition.png`, `collapse_controls.png`, and `training_convergence.png`: figures generated from those saved observations.
- `sensitivity/`: targeted ZNE scale and calibration-budget sensitivity outputs, when run.

The calibration threshold maximizes balanced accuracy on the calibration partition. The affine readout standardizes the one-dimensional score using calibration data only, then fits a logistic affine readout on that same independent partition. Fixed-threshold and uncalibrated ZNE methods use the full nominal budget for test inference; calibration methods divide the nominal budget between calibration and test inference. Integer shot allocation can make realized totals differ by at most 0.443% from the comparator. Linear and quadratic ZNE use coefficient-proportional shot allocation, optimal under equal per-shot variance, and report the associated variance amplification. Exact-expectation comparisons isolate extrapolation error from measurement sampling.

Regenerate derived summaries and figures from one saved run with:

```powershell
.\.venv\Scripts\python.exe benchmark_classical_and_plots.py --run-id <run-id>
```

Regenerate the manuscript for that run:

```powershell
.\.venv\Scripts\python.exe generate_paper.py --run-id <run-id>
```

Compare a completed independent rerun with the saved final outputs using `python compare_reproduction.py --reference final-study-20260929 --reproduction final-study-reproduction`; see `REPRODUCTION.md`.

Create an archive for one selected run with:

```powershell
.\.venv\Scripts\python.exe build_and_package.py --run-id <run-id>
```

## Separate fresh-data synthetic diagnostic

`confirmatory_budget_study.py` runs a separate development/held-out diagnostic on new balanced synthetic data. It trains models from scratch, tunes the ZNE scale schedule and calibration share on development seeds only, then evaluates the frozen choice on distinct confirmation seeds. Run it with:

```powershell
python confirmatory_budget_study.py
```

The outputs are in `results/runs/confirmatory-budget-study-20261005/`; `CONFIRMATORY_RESULTS.md` summarizes the settings, both shot budgets, the unsuccessful budget-prediction check, and limitations. This is a small simulator diagnostic with six development realizations, 30 distinct held-out realizations, one finite-shot draw per method and setting, one two-qubit circuit family, and no external dataset or hardware validation. Its fixed-baseline results are near chance and do not confirm the WDBC result or support a general calibration-versus-ZNE ranking or validated budget rule.

## Interpretation limits

Repeated shot draws are repeated measurements of the same frozen examples; they are not independent test sets. Paired confidence intervals resample outer splits as clusters. The final run remains a computational pilot and is not enough to establish a general decision-boundary criterion. The binary Wine task is a documented reduction of the 3-class data. Synthetic tasks are controlled diagnostics. WDBC is a benchmark dataset and the outputs are not patient-confidence estimates or clinical advice. The simulator cannot establish quantum advantage or hardware performance. Author and venue details must be supplied before submission; no institution, email, or author identity is invented here.

## Prior work to address

The generated manuscript engages with noise-robust quantum-classifier encodings by LaRose and Coyle (2020), QuantumNAT's noise-aware training and post-measurement normalization (Wang et al., DAC 2022), digital ZNE and its practice recommendations (Giurgica-Tiron et al., 2020; Majumdar et al., 2023), artefactual ZNE improvements (Köster and Mauerer, 2026), finite-shot threshold instability in quantum-classifier studies, and a September 2026 workshop paper on equal-online-budget mitigation and ZNE failure under coherent miscalibration. These precedents rule out a broad novelty claim based only on damping-induced score shift, budget-matched mitigation, or applying ZNE to another classifier. The cited studies address observable correction or different classifier settings; the present pilot's narrow comparison concerns held-out, label-dependent threshold selection under synthetic noise.
# Fresh split-and-seed WDBC confirmation

The separate preregistered run `results/runs/independent-confirmation-wdbc-20261016/` trains ten new checkpoints on five fresh stratified WDBC folds and uses 20 finite-shot draws per checkpoint. It compares calibrated noisy-threshold selection with calibrated linear and quadratic probability-scaled ZNE under the recorded nominal 1,000-measurement accounting. To reproduce it from a fresh checkout after installing the pinned dependencies, run:

```powershell
.\.venv\Scripts\python.exe independent_confirmation_wdbc.py
```

This command refuses to overwrite the saved completed run. Check its integrity and reconstruct saved metrics with:

```powershell
.\.venv\Scripts\python.exe independent_confirmation_audit.py
```

The saved finite-shot balanced accuracies are 0.790 for threshold calibration, 0.721 for linear ZNE, and 0.596 for quadratic ZNE. This is same-dataset simulator confirmation, not external-population or hardware validation. Full protocol and interpretation limits are in `results/runs/independent-confirmation-wdbc-20261016/INDEPENDENT_CONFIRMATION.md`.
