# Portable project quick start

This ZIP is a self-contained project snapshot with experiment source, the saved study, audit script and output, plots, manuscript source, summary, report, and pinned dependency lists. Extract it to any writable folder and run commands from the extracted project root. The archive uses relative paths; it does not depend on the original computer's workspace location.

## Start here

1. Read [SUMMARY.md](SUMMARY.md) for the main result in plain language.
2. Read [RESEARCH_REPORT.md](RESEARCH_REPORT.md) for methods, results, audit evidence, and limitations.
3. Open [paper.tex](paper.tex) for the manuscript. Its figures are included beside it in the archive. `generate_paper.py` is the manuscript source generator; edit it before regenerating the run-specific `.tex` file.
4. Inspect `results/runs/final-study-independent-retrain-20261003/` for the clean retraining, source records, checkpoints, partitions, per-example observations, and run manifest. The ZIP also includes `final-study-20260929/` as the reference run used for the reproduction comparison.

## Recheck the saved run

Requires Python 3.12 with the included dependencies installed. From the extracted project root:

```text
python -m venv .venv
```

On Windows, install and run with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe results\runs\final-study-independent-retrain-20261003\independent_audit.py --run-dir results\runs\final-study-independent-retrain-20261003
```

On Linux or macOS, use:

```sh
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python results/runs/final-study-independent-retrain-20261003/independent_audit.py --run-dir results/runs/final-study-independent-retrain-20261003
```

The audit normalizes manifest paths and is intended to work on Windows, Linux, and macOS.

The audit rechecks hashes, splits, reported finite-shot metrics, ZNE reconstructions, calibration thresholds, architecture probing, and measurement accounting. A passing audit verifies consistency of saved results; it does not retrain all models.

## Reproduce the full experiment

The complete simulation takes about 1 hour 45 minutes on the computer recorded in the manifest; runtime varies by processor. It creates a new results folder and does not overwrite the saved final run.

```powershell
.\.venv\Scripts\python.exe research_experiment.py --run-id final-study-reproduction --outer-splits 5 --train-seeds 2 --arch-sweep-splits 1 --epochs 30 --batch-size 32 --shot-budgets 100,1000 --shot-repetitions 2
```

On Linux and macOS, run the same configuration with `.venv/bin/python research_experiment.py --run-id final-study-reproduction --outer-splits 5 --train-seeds 2 --arch-sweep-splits 1 --epochs 30 --batch-size 32 --shot-budgets 100,1000 --shot-repetitions 2`. The data are loaded from scikit-learn's bundled datasets and generated locally with recorded seeds.

After the rerun finishes, compare it with the reference files using:

```text
python compare_reproduction.py --reference final-study-20260929 --reproduction final-study-reproduction
```

This writes `reproduction_comparison.csv`; see `REPRODUCTION.md` for how to interpret the check.

## Rebuild the manuscript package

```text
python benchmark_classical_and_plots.py --run-id final-study-independent-retrain-20261003
python matched_fit_classical.py --run-id final-study-independent-retrain-20261003
python zne_range_sensitivity.py --run-id final-study-independent-retrain-20261003
python plot_training_curves.py --run-dir results/runs/final-study-independent-retrain-20261003
python plot_score_errors.py --run-dir results/runs/final-study-independent-retrain-20261003
python generate_paper.py --run-id final-study-independent-retrain-20261003
python build_and_package.py --run-id final-study-independent-retrain-20261003 --include-run final-study-20260929
```

The last command creates `results/runs/final-study-independent-retrain-20261003/qml_paper_submission.zip` and a convenient project-root copy named `qml_vqc_research_package.zip`. A saved clean retraining comparison is included. The paper source still requires a successful PDF compilation and visual page-layout check; do not describe it as compiled until that succeeds.

## What the study does not claim

The results are from a small-qubit simulator using synthetic channel probabilities. They do not establish quantum hardware performance, clinical validity, general quantum advantage, or a universal method ranking. Same-environment exact reproducibility does not establish external replication. Author, affiliation, email, and venue details remain placeholders; choose a venue and apply its template before submission.

## Open the results dashboard

Install the pinned dependencies as above, then start the local Streamlit dashboard from the project root:

```text
python -m streamlit run app.py
```

The page defaults to `final-study-independent-retrain-20261003`; use the **Saved experiment run** selector to inspect another saved run. Streamlit prints the local browser address when the server starts.
# Fresh WDBC confirmation

The ZIP includes the preregistered fresh split-and-seed confirmation at `results/runs/independent-confirmation-wdbc-20261016/`. Its separately trained WDBC models reached 0.790 finite-shot balanced accuracy with threshold calibration, versus 0.721 for linear ZNE and 0.596 for quadratic ZNE at the stated budget. To verify the saved files, run `python independent_confirmation_audit.py`. To repeat the full run without overwriting the included evidence, first change `RUN_ID` in `independent_confirmation_wdbc.py` to a new directory name, then run `python independent_confirmation_wdbc.py`. This remains single-dataset simulator evidence.
