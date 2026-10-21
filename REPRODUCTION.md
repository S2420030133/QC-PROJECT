# Reproduction record

## Clean full retraining

The reference study is `final-study-20260929`. A second complete run, `final-study-independent-retrain-20261003`, trained and evaluated all 56 configurations from fresh initializations; no saved model checkpoints were reused. The command was:

```powershell
.venv\Scripts\python.exe research_experiment.py --run-id final-study-independent-retrain-20261003 --outer-splits 5 --train-seeds 2 --arch-sweep-splits 1 --epochs 30 --batch-size 32 --shot-budgets 100,1000 --shot-repetitions 2
```

The clean run reproduced every saved model parameter array bitwise (56/56) and all three recorded training-curve measures for all 1,680 model/epoch rows. The per-run comparison matched all 5,312 exact-result rows and 5,120 finite-shot rows. Balanced accuracy, ROC-AUC, selected thresholds, and score-error metrics had no differences at tolerance `1e-10`; measurement totals matched exactly. These checks establish deterministic same-environment reproducibility for the recorded code and fixed seeds. The two runs use the same software environment and data-generation protocol, so this is not an external replication, a cross-platform validation, or evidence of generalization to new populations.

Comparison outputs are saved with the clean run:

- `reproduction_comparison.csv`: exact and finite-shot metric comparisons.
- `model_weight_reproduction_comparison.csv`: per-model weight-array equality.
- `training_curve_reproduction_comparison.csv`: per-epoch loss and development-metric comparison.

To rerun the study in a fresh copy, use a new run ID because the recorded run folders are preserved:

```powershell
.venv\Scripts\python.exe research_experiment.py --run-id my-independent-run --outer-splits 5 --train-seeds 2 --arch-sweep-splits 1 --epochs 30 --batch-size 32 --shot-budgets 100,1000 --shot-repetitions 2
```

Then compare the rerun with the included reference, if present:

```powershell
.venv\Scripts\python.exe compare_reproduction.py --reference final-study-20260929 --reproduction my-independent-run
```

The output comparison is descriptive verification of reproducibility, not a statistical test of the scientific claims.

## Historical recovery attempt

An earlier run (`final-study-reproduction-20261003`) was a recovery evaluation after an interrupted execution. It recomputed evaluations from 54 saved checkpoints and trained two missing configurations. That run was not a full retraining. It is retained for provenance; the fresh 56/56 run above supersedes it as the primary retraining check.

## Final checks

Both the reference and clean-run audit scripts check artifact hashes, partitions, held-out-fold coverage, feature-dependence probes, per-example metrics, selected calibration thresholds, reconstructed ZNE values, and measurement accounting. Their JSON reports are stored beside each run. The audits validate internal consistency; they do not establish that the simulation represents a physical device or that the model generalizes.
