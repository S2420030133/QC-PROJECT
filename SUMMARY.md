# Research summary

## The question

Can a separate set of labeled examples help fix a quantum circuit classifier's decision boundary after simulated noise shifts its scores, and how does that compare with zero-noise extrapolation (ZNE) when measurements are limited?

## Simple example

Imagine a scale whose readings all shift upward. It might still rank lighter objects below heavier ones, but a fixed “above 20 means heavy” rule can fail. A threshold selected using separate calibration examples may restore useful classifications. In this project the scale reading is a quantum circuit's measured Pauli-Z value, and the “shift” is produced by simulated quantum noise.

## What the saved study shows

The primary WDBC matched-budget comparison now standardizes features before two-component PCA (0.634 variance retained), fits preprocessing on each outer training pool, and compares 10 model runs across five stratified folds and two initializations. Test folds partition the data; training sets share observations. The 1,000-shot pipelines allocate half the nominal budget to calibration and half to test inference; finite-shot results average 20 independent measurement draws per saved checkpoint.

| WDBC pipeline at composite noise $\gamma=0.30$ | Exact balanced accuracy | At 1,000 nominal shots per test case |
|---|---:|---:|
| Noisy threshold selection | 0.878 | 0.778 |
| Linear ZNE with calibration | 0.883 | 0.716 |
| Quadratic ZNE with calibration | 0.884 | 0.582 |
| Fixed threshold | 0.500 | 0.503 |
| Logistic regression on the same standardized two inputs, clean | 0.928 | -- |

Noisy threshold selection exceeded calibrated linear and quadratic ZNE in all five fold-level comparisons. Test folds partition the dataset, while training observations are shared, so this is descriptive split-protocol variability rather than independent-population evidence. The quantum model also trails the matched-input clean logistic baseline. At exact expectations, fixed-threshold balanced accuracy is 0.500 while AUC is 0.958, showing useful score ordering alongside a failed zero threshold.

The older full noise-grid and cross-task analyses use unstandardized PCA and are supplementary secondary analyses. A new standardized-PCA transfer comparison freshly trained models on Wine and balanced synthetic data. At the same nominal 1,000-measurement budget, noisy threshold selection reached 0.701 balanced accuracy on Wine and 0.612 on balanced synthetic; calibrated linear ZNE reached 0.650 and 0.584, and calibrated quadratic ZNE reached 0.557 and 0.534. Matched-input clean logistic regression reached 0.950 and 0.747, respectively, exceeding the quantum models. A fresh 30-dataset synthetic diagnostic has weak discrimination and is not confirmatory evidence for WDBC.

One exact-result discrepancy was found while auditing: the prior standardized-PCA WDBC exact calibrated-threshold mean (0.891) did not reproduce from its saved model rows. A fresh recomputation from saved per-example scores gives 0.878 (AUC 0.958); the paper uses the corrected value and the before/after row values are logged.

The primary WDBC finite-shot comparison now averages 20 independent shot draws per saved checkpoint (10 checkpoints). A post-hoc calibration-size sensitivity compares 10, 20, 50, and 114 labeled calibration examples under the same nominal total measurement budget; noisy-threshold balanced accuracy ranges from 0.760 at 10 examples to 0.783 at 114. The 0.783 differs from the main 0.778 because the calibration-shot RNG stream differs; folds, labels, checkpoints, test draws, and threshold rule are held fixed. It is not an independent replication. See Supplementary Table S14 and the saved calibration-size CSVs.

A matched standardized-PCA 1/2/4-layer diagnostic reports clean fixed balanced accuracies 0.873, 0.887, 0.885; exact noisy-calibrated results 0.849, 0.878, 0.873; and finite-shot results 0.854, 0.778, 0.525. The noisy class-mean score gap shrinks from 0.210 to 0.008 while the 500-shot single-score SE is about 0.043. This is an exploratory, circuit-specific depth mechanism, not a general law; see the main-text depth table and saved score-spread audit.

## What this does not show

- The project is a small classical simulation, not a quantum hardware experiment.
- It does not establish quantum advantage, clinical validity, or general superiority of calibration over ZNE.
- A classical logistic model on the same two PCA features and the same number of weight-fitting examples exceeded the clean two-qubit result on WDBC, Wine, and balanced synthetic data. On the weak imbalanced synthetic task, both were near chance. This is a predictive baseline, not evidence for quantum advantage.
- The imbalanced synthetic task is a negative result: clean balanced accuracy is 52.6% and ROC-AUC is 0.462.
- Five outer folds share training observations. Their intervals do not represent five independent populations.
- The four-qubit CNOT-star has a verified input-to-readout path, but its noisy accuracy is near chance; this does not show useful scaling.
- Threshold and ZNE settings were not tuned on a separate development study. The allocation and scale sensitivities are limited follow-up checks, not an optimized method search.

## Status

The clean independent retraining completed all 56 original model configurations from fresh initializations; standardized Wine and synthetic transfer added 20 more freshly trained models. The complete audit covers 384 run artifacts; the exact-grid audit recomputes 200 rows, and the standardized transfer audit recomputes 2,400 finite-shot metric rows from saved observations without mismatches. The calibration-size sweep compares 10, 20, 50, and 114 labeled examples but reuses checkpoints and test draws, so it is a sensitivity analysis, not independent validation. A separate gate-folded simulator comparison at scales 1, 3, and 5 reaches 0.682 balanced accuracy (SD 0.016 across 10 checkpoint means; AUC 0.757), versus 0.776 for noisy-threshold calibration; its 400 metric rows recompute exactly from saved scores. A local Aer simulation using the saved IBM FakeManilaV2 backend snapshot evaluates all 10 standardized checkpoints with modeled gate and readout errors: raw calibrated thresholding reaches 0.873 balanced accuracy, readout-corrected calibration 0.878, and unitary gate-folded ZNE 0.885. All 70 metric rows recompute within floating-point precision; these differences are within checkpoint variability, do not show a consistent mitigation benefit, and are not hardware measurements. Matched classical baselines remain stronger on clean performance. The author placeholders remain as requested. The regenerated headline and calibration-efficiency figures are in the run and Overleaf directories. The Overleaf package `qml-vqc-overleaf-project.zip` contains the complete manuscript source, supplementary tables, and figures.


See the [full report](RESEARCH_REPORT.md), [manuscript](results/runs/final-study-independent-retrain-20261003/paper.tex), [clean-run manifest](results/runs/final-study-independent-retrain-20261003/run_manifest.json), and [portable instructions](PORTABLE_README.md).
# Fresh confirmation update (2026-10-16)

A separate preregistered WDBC run used five new stratified test folds, ten newly trained checkpoints, standardized PCA, and 20 finite-shot draws per checkpoint. At $\gamma=0.30$ and a nominal 1,000-measurement budget including calibration, calibrated threshold selection reached 0.790 balanced accuracy, compared with 0.721 for linear and 0.596 for quadratic ZNE. It beat both in each of the five fold contrasts. All saved exact and finite metric rows reconstruct from retained observations, and the separate audit passed all 41 run-artifact hashes. This is same-dataset simulator confirmation, not hardware or external-population validation. Full results and limits are in `results/runs/independent-confirmation-wdbc-20261016/INDEPENDENT_CONFIRMATION.md`.
