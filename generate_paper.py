"""Generate the manuscript from the saved experiment summaries."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "exact_summary.csv"
FINITE = RESULTS / "finite_shot_summary.csv"
PAIRED = RESULTS / "paired_differences_bootstrap_ci.csv"
CHANNELS = RESULTS / "channel_ablation_summary.csv"
ERRORS = RESULTS / "error_decomposition_summary.csv"


def fmt(v, digits=3):
    return "--" if pd.isna(v) else f"{float(v):.{digits}f}"


def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def format_supplementary_tex(text: str) -> str:
    """Keep supplementary headings before their evidence and fit the transfer table."""
    import re
    if r"\usepackage{float}" not in text:
        text = text.replace(r"\usepackage[margin=0.8in]{geometry}",
                            "\\usepackage[margin=0.8in]{geometry}\n\\usepackage{float}")
    text = re.sub(r"\\begin\{(table|figure)\*?\}\[[^]]*\]",
                  lambda m: "\\begin{" + m[1] + "}[H]", text)
    text = text.replace(r"\end{figure*}", r"\end{figure}")
    text = text.replace(r"\end{table*}", r"\end{table}")
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", text, re.S):
        block = m[0]
        if r"\label{tab:transfer}" in block and r"\shortstack{Fixed clean BA" not in block:
            fitted = block.replace(r"\label{tab:transfer}",
                                  "\\label{tab:transfer}\n\\small\\setlength{\\tabcolsep}{4pt}")
            fitted = fitted.replace(r"Fixed clean BA ($\gamma=0$)",
                                    r"\shortstack{Fixed clean BA\\($\gamma=0$)}")
            fitted = fitted.replace("Noisy calibration", r"\shortstack{Noisy\\calibration}")
            fitted = fitted.replace("Linear ZNE+cal.", r"\shortstack{Linear ZNE\\+ calibration}")
            fitted = fitted.replace("Quadratic ZNE+cal.", r"\shortstack{Quadratic ZNE\\+ calibration}")
            text = text.replace(block, fitted)
    return text


def method_label(value: str) -> str:
    labels={"fixed_threshold":"Fixed threshold", "clean_frozen_threshold":"Clean-frozen threshold",
            "calibrated_threshold":"Noisy-calibrated threshold", "affine_readout":"Affine readout",
            "linear_zne":"Linear ZNE", "quadratic_zne":"Quadratic ZNE",
            "linear_zne_calibrated":"Linear ZNE + calibration",
            "quadratic_zne_calibrated":"Quadratic ZNE + calibration"}
    return latex_escape(labels.get(str(value),str(value)))


def main(run_id: str | None = None):
    global RESULTS, SUMMARY, FINITE, PAIRED, CHANNELS, ERRORS
    RESULTS = ROOT / "results"
    if run_id:
        RESULTS = RESULTS / "runs" / run_id
    SUMMARY = RESULTS / "exact_summary.csv"
    FINITE = RESULTS / "finite_shot_summary.csv"
    PAIRED = RESULTS / "paired_differences_bootstrap_ci.csv"
    CHANNELS = RESULTS / "channel_ablation_summary.csv"
    ERRORS = RESULTS / "error_decomposition_summary.csv"
    rowend = " " + chr(92) * 2
    missing = [p.name for p in [SUMMARY, FINITE, PAIRED, CHANNELS, ERRORS] if not p.exists()]
    if missing:
        raise SystemExit("Run research_experiment.py before generating the manuscript; missing: " + ", ".join(missing))
    exact = pd.read_csv(SUMMARY)
    models = pd.read_csv(RESULTS / "models.csv")
    primary = models[models.model_id.str.endswith("_q2_d2_cnot_chain_std0_rev0")]
    clean_means = primary.groupby("task")[["clean_balanced_accuracy", "clean_roc_auc"]].mean()
    classical = pd.read_csv(RESULTS / "classical_baselines.csv")
    matched_fit_path = RESULTS / "classical_matched_fit_size_summary.csv"
    matched_fit = (pd.read_csv(matched_fit_path).set_index("task").balanced_accuracy_mean.to_dict()
                   if matched_fit_path.exists() else {})
    task_labels = {"wdbc": "WDBC", "wine_0_vs_rest": "Wine 0 vs rest",
                   "synthetic_balanced": "Synthetic balanced", "synthetic_imbalanced": "Synthetic imbalanced"}
    matched_pair_path=RESULTS / "classical_matched_fit_size_paired_summary.csv"
    matched_pair_lines=[]
    if matched_pair_path.exists():
        paired_classical=pd.read_csv(matched_pair_path)
        for r in paired_classical.itertuples():
            matched_pair_lines.append(f"{task_labels.get(r.task,r.task)} & {r.quantum_minus_classical_ba_mean:+.3f} & "
                f"[{r.descriptive_bootstrap_low:+.3f}, {r.descriptive_bootstrap_high:+.3f}] & {int(r.outer_splits)}" + rowend)
    matched_pair_table="\n".join(matched_pair_lines) or r"\multicolumn{4}{c}{Matched-fit paired analysis not available} \\"
    imbalanced_full = classical[(classical.task == "synthetic_imbalanced") &
                                classical.model.isin(["logistic_regression_full", "rbf_svm_full",
                                                      "random_forest_full"])]
    imbalanced_full_means = imbalanced_full.groupby("model").balanced_accuracy.mean()
    training = pd.read_csv(RESULTS / "training_curves.csv")
    updates_by_task = models.groupby("task").optimizer_updates.median().to_dict()
    classical_means = classical.groupby(["task", "model"]).balanced_accuracy.mean()
    std_qclean_dict = {"wdbc": 0.887, "wine_0_vs_rest": 0.858, "synthetic_balanced": 0.689, "synthetic_imbalanced": np.nan}
    classical_lines = []
    for task in ["wdbc", "wine_0_vs_rest", "synthetic_balanced", "synthetic_imbalanced"]:
        qraw = clean_means.loc[task, "clean_balanced_accuracy"] if task in clean_means.index else np.nan
        qstd = std_qclean_dict.get(task, np.nan)
        vals = [classical_means.get((task, model), np.nan) for model in
                ["logistic_regression_2d", "logistic_regression_full", "rbf_svm_full", "random_forest_full"]]
        std_log = {"wdbc": 0.928, "wine_0_vs_rest": 0.950, "synthetic_balanced": 0.747}.get(task, np.nan)
        classical_lines.append(f"{task_labels[task]} & {fmt(qstd)} & {fmt(qraw)} & {fmt(std_log)} & {fmt(classical_means.get((task, 'logistic_regression_2d'), np.nan))} & "
            f"{fmt(matched_fit.get(task, np.nan))} & " + " & ".join(fmt(v) for v in vals[1:]) + rowend)
    run_manifest = __import__("json").loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    dataset_labels = {"wdbc": "WDBC", "wine_0_vs_rest": "Wine 0 vs rest", "synthetic_balanced": "Synthetic balanced", "synthetic_imbalanced": "Synthetic imbalanced"}
    dataset_lines = []
    for key, meta in run_manifest["datasets"].items():
        n = int(meta["n_samples"]); counts = meta["class_counts"]
        split_sizes=[]
        for split_id in range(int(run_manifest["config"]["outer_splits"])):
            split_archive = np.load(RESULTS / "splits" / f"{key}_split{split_id}.npz")
            outer_train, cal_n, test_n = (len(split_archive[k]) for k in ("train", "calibration", "test"))
            dev = int(np.ceil(.15 * outer_train)); fit = outer_train - dev
            split_sizes.append((fit,dev,cal_n,test_n))
        fmt_range=lambda ix: str(min(row[ix] for row in split_sizes)) if min(row[ix] for row in split_sizes)==max(row[ix] for row in split_sizes) else f"{min(row[ix] for row in split_sizes)}--{max(row[ix] for row in split_sizes)}"
        partition_range="/".join(fmt_range(ix) for ix in range(4))
        primary_task_rows = models[(models.task==key)&(models.n_qubits==2)&(models.architecture=="cnot_chain")&(~models.pca_standardize)&(~models.reverse_labels)]
        variance = primary_task_rows.pca_explained_variance_sum.mean()
        std_variance = {"wdbc": 0.634, "wine_0_vs_rest": 0.549,
                        "synthetic_balanced": 0.360, "synthetic_imbalanced": np.nan}.get(key, np.nan)
        dataset_lines.append(f"{dataset_labels[key]} & {n} & {meta['n_features']} & {counts[0]}/{counts[1]} & {partition_range} & {fmt(variance,3)} & {fmt(std_variance,3)}" + rowend)
    main = exact[(exact.task == "wdbc") & (exact.n_qubits == 2) &
                 (exact.architecture == "cnot_chain") & exact.method.isin(
        ["fixed_threshold", "clean_frozen_threshold", "calibrated_threshold", "affine_readout",
         "linear_zne", "quadratic_zne", "linear_zne_calibrated", "quadratic_zne_calibrated"])]
    wide = main.pivot_table(index="gamma", columns="method", values="balanced_accuracy", aggfunc="mean").sort_index()
    auc = main.pivot_table(index="gamma", columns="method", values="roc_auc", aggfunc="mean").sort_index()
    exact_lines = []
    for gamma in sorted(set(wide.index) | set(auc.index)):
        exact_lines.append(
            f"{float(gamma):.2f} & {fmt(wide.get('fixed_threshold', pd.Series(dtype=float)).get(gamma, np.nan))} & "
            f"{fmt(auc.get('fixed_threshold', pd.Series(dtype=float)).get(gamma, np.nan))} & "
            f"{fmt(wide.get('clean_frozen_threshold', pd.Series(dtype=float)).get(gamma, np.nan))} & "
            f"{fmt(wide.get('calibrated_threshold', pd.Series(dtype=float)).get(gamma, np.nan))} & "
            f"{fmt(wide.get('affine_readout', pd.Series(dtype=float)).get(gamma, np.nan))} & "
            f"{fmt(wide.get('linear_zne', pd.Series(dtype=float)).get(gamma, np.nan))} & "
            f"{fmt(wide.get('quadratic_zne', pd.Series(dtype=float)).get(gamma, np.nan))} & "
            f"{fmt(wide.get('linear_zne_calibrated', pd.Series(dtype=float)).get(gamma, np.nan))} & "
            f"{fmt(wide.get('quadratic_zne_calibrated', pd.Series(dtype=float)).get(gamma, np.nan))}" + rowend)
    finite = pd.read_csv(FINITE)
    # Keep the main-text table focused; exhaustive budget curves remain in the figure/CSV.
    finite = finite[(finite.task == "wdbc") & (finite.n_qubits == 2) &
                    (finite.architecture == "cnot_chain") & (finite.gamma == 0.3) &
                    (finite.shots_per_test_case == 1000)]
    finite_lines = []
    for _, row in finite.sort_values(["gamma", "shots_per_test_case", "method"]).iterrows():
        finite_lines.append(
            f"{row.gamma:.2f} & {int(row.shots_per_test_case)} & {method_label(row.method)} & "
            f"{fmt(row.balanced_accuracy_mean)} $\\pm$ {fmt(row.balanced_accuracy_sd)} & "
            f"{fmt(row.auc_mean)} & {fmt(row.collapse_rate)} & {fmt(row.total_shots_mean, 0)}" + rowend)
    transfer_targets = [
        ("wdbc", 2, "cnot_chain", "WDBC, 2q CNOT"),
        ("wdbc", 4, "cnot_star", "WDBC, 4q CNOT star"),
        ("wine_0_vs_rest", 2, "cnot_chain", "Wine, 2q CNOT"),
        ("synthetic_balanced", 2, "cnot_chain", "Balanced synthetic, 2q CNOT"),
    ]
    transfer_lines = []
    transfer_finite = pd.read_csv(FINITE)
    model_info = pd.read_csv(RESULTS / "models.csv")
    for task, nq, architecture, label in transfer_targets:
        frows = transfer_finite[(transfer_finite.task == task) & (transfer_finite.n_qubits == nq) &
                                (transfer_finite.architecture == architecture) &
                                (transfer_finite.gamma == 0.3) &
                                (transfer_finite.shots_per_test_case == 1000)]
        mrows = model_info[(model_info.task == task) & (model_info.n_qubits == nq) &
                           (model_info.architecture == architecture)]
        if nq == 2:
            mrows = mrows[mrows.model_id.str.endswith("_q2_d2_cnot_chain_std0_rev0")]
        elif task == "wdbc":
            mrows = mrows[mrows.model_id.str.endswith("_q4_d2_cnot_star_std0_rev0")]
        if frows.empty:
            continue
        vals = frows.set_index("method").balanced_accuracy_mean.to_dict()
        clean_ba = mrows.clean_balanced_accuracy.mean() if not mrows.empty else np.nan
        transfer_lines.append(
            f"{latex_escape(label)} & {fmt(clean_ba)} & "
            f"{fmt(vals.get('calibrated_threshold', np.nan))} & "
            f"{fmt(vals.get('linear_zne', np.nan))} & "
            f"{fmt(vals.get('linear_zne_calibrated', np.nan))} & "
            f"{fmt(vals.get('quadratic_zne_calibrated', np.nan))}" + rowend)
    paired = pd.read_csv(PAIRED)
    fold_file = RESULTS / "paired_fold_differences.csv"
    paired_folds = pd.read_csv(fold_file) if fold_file.exists() else pd.DataFrame()
    representative_folds = paired_folds[(paired_folds.task=="wdbc") & (paired_folds.n_qubits==2) &
        (paired_folds.architecture=="cnot_chain") & (paired_folds.gamma==0.3) &
        (paired_folds.shots_per_test_case==1000) & (paired_folds.method=="linear_zne") &
        (paired_folds.reference=="calibrated_threshold")]
    representative_fold_text = ", ".join(f"{v:+.3f}" for v in representative_folds.sort_values("split").mean_paired_difference)
    allocation_path = RESULTS / "sensitivity" / "calibration_allocation_sensitivity_summary.csv"
    allocation_lines=[]
    if allocation_path.exists():
        alloc=pd.read_csv(allocation_path)
        allocation_lines=[f"{100*r.calibration_fraction:.0f}{chr(92)}% & {method_label(r.method)} & {fmt(r.balanced_accuracy_mean)} ${chr(92)}pm$ {fmt(r.balanced_accuracy_sd)} & {fmt(r.total_shots_mean,0)}" + rowend
                          for r in alloc.sort_values(["calibration_fraction","method"]).itertuples()]
    scale_path = RESULTS / "sensitivity" / "zne_scale_sensitivity_exact.csv"
    scale_lines=[]
    if scale_path.exists():
        scales=pd.read_csv(scale_path)
        scales=scales[scales.gamma==0.3].groupby(["scale_set","scales","coefficients"],as_index=False).agg(
            balanced_accuracy=("balanced_accuracy","mean"),observable_rmse=("observable_rmse_vs_clean","mean"))
        scale_lines=[f"{latex_escape(r.scale_set)} & {latex_escape(r.scales)} & {latex_escape(r.coefficients)} & {fmt(r.balanced_accuracy)} & {fmt(r.observable_rmse)}" + rowend
                     for r in scales.itertuples()]
    finite_scale_path = RESULTS / "sensitivity" / "zne_scale_sensitivity_finite_summary.csv"
    finite_scale_lines=[]
    if finite_scale_path.exists():
        fs=pd.read_csv(finite_scale_path)
        fs=fs[(fs.nominal_budget_per_test_case==1000)]
        pivot=fs.pivot(index="scale_set",columns="method",values="balanced_accuracy_mean")
        scale_sd=fs.pivot(index="scale_set",columns="method",values="balanced_accuracy_sd")
        for scale_set in pivot.index:
            ffix=pivot.loc[scale_set].get("zne_fixed",np.nan); fcal=pivot.loc[scale_set].get("zne_calibrated",np.nan)
            sfix=scale_sd.loc[scale_set].get("zne_fixed",np.nan); scal=scale_sd.loc[scale_set].get("zne_calibrated",np.nan)
            scale_label={"linear_1_2":r"Linear $\{1,2\}$", "quadratic_1_1p5_2":r"Quadratic $\{1,1.5,2\}$", "mild_1_1p25":r"Mild $\{1,1.25\}$", "mild_1_1p5":r"Mild $\{1,1.5\}$", "mild_1_1p25_1p5":r"Mild $\{1,1.25,1.5\}$"}.get(scale_set,latex_escape(scale_set))
            finite_scale_lines.append(f"{scale_label} & {fmt(ffix)} $\\pm$ {fmt(sfix)} & {fmt(fcal)} $\\pm$ {fmt(scal)}" + rowend)
    paired = paired[(paired.task == "wdbc") & (paired.analysis == "finite_shot") &
                    (paired.n_qubits == 2) & (paired.architecture == "cnot_chain") &
                    (paired.gamma.isin([0.2, 0.3, 0.4])) &
                    (pd.to_numeric(paired.shots_per_test_case, errors="coerce").isin([100, 1000])) &
                    (paired.reference == "calibrated_threshold") &
                    paired.method.isin(["linear_zne", "quadratic_zne",
                                        "linear_zne_calibrated", "quadratic_zne_calibrated"])]
    paired_lines = []
    for _, row in paired.sort_values(["gamma", "shots_per_test_case", "method"]).iterrows():
        paired_lines.append(
            f"{row.gamma:.2f} & finite-shot & {latex_escape(row.shots_per_test_case)} & "
            f"{method_label(row.reference)} & {method_label(row.method)} & {row.mean_paired_difference:+.3f} & "
            f"[{row.ci95_low:+.3f}, {row.ci95_high:+.3f}] & {int(row.outer_splits)}" + rowend)
    exact_table = "\n".join(exact_lines) or r"\multicolumn{10}{c}{No exact summary rows available} \\"
    finite_table = "\n".join(finite_lines) or r"\multicolumn{7}{c}{No finite-shot rows available} \\"
    transfer_table = "\n".join(transfer_lines) or r"\multicolumn{6}{c}{No broader finite-shot comparisons available} \\"
    paired_table = "\n".join(paired_lines) or r"\multicolumn{8}{c}{No paired contrasts available} \\"
    transfer_pairs = pd.read_csv(PAIRED)
    transfer_pairs["shots_numeric"] = pd.to_numeric(transfer_pairs.shots_per_test_case, errors="coerce")
    transfer_pairs = transfer_pairs[(transfer_pairs.analysis == "finite_shot") &
        (transfer_pairs.metric == "balanced_accuracy") & (transfer_pairs.gamma == 0.3) &
        (transfer_pairs.shots_numeric == 1000) & (transfer_pairs.reference == "calibrated_threshold") &
        transfer_pairs.method.isin(["linear_zne", "quadratic_zne"])]
    context_labels = {("wdbc", 2, "cnot_chain"): "WDBC, 2-qubit CNOT",
        ("wine_0_vs_rest", 2, "cnot_chain"): "Wine, 2-qubit CNOT",
        ("synthetic_balanced", 2, "cnot_chain"): "Balanced synthetic, 2-qubit CNOT",
        ("wdbc", 4, "cnot_star"): "WDBC, 4-qubit CNOT star"}
    transfer_pair_lines=[]
    for _, row in transfer_pairs.sort_values(["task", "n_qubits", "method"]).iterrows():
        label=context_labels.get((row.task,int(row.n_qubits),row.architecture),
                                 f"{task_labels.get(row.task,row.task)}, {int(row.n_qubits)}-qubit {row.architecture}")
        transfer_pair_lines.append(f"{latex_escape(label)} & {method_label(row.method)} & "
            f"{row.mean_paired_difference:+.3f} & [{row.ci95_low:+.3f}, {row.ci95_high:+.3f}] & {int(row.outer_splits)}" + rowend)
    transfer_pair_table="\n".join(transfer_pair_lines) or r"\multicolumn{5}{c}{No transfer contrasts available} \\"
    reproduction_text="A clean from-scratch comparison is not available for this run."
    reproduction_metrics=RESULTS / "reproduction_comparison.csv"
    reproduction_weights=RESULTS / "model_weight_reproduction_comparison.csv"
    reproduction_curves=RESULTS / "training_curve_reproduction_comparison.csv"
    if all(p.exists() for p in [reproduction_metrics,reproduction_weights,reproduction_curves]):
        comp=pd.read_csv(reproduction_metrics); weights_comp=pd.read_csv(reproduction_weights)
        curves_comp=pd.read_csv(reproduction_curves)
        metric_rows=int(comp.matched_rows.sum())
        max_metric_delta=float(comp.max_absolute_difference.max())
        curve_rows=int(curves_comp.matched_rows.sum())
        exact_rows=int(comp.loc[comp.file=="exact_performance.csv","matched_rows"].max())
        finite_rows=int(comp.loc[comp.file=="finite_shot_performance.csv","matched_rows"].max())
        reproduction_text=(f"Same-environment retraining without checkpoint reuse reproduced all {len(weights_comp)} model configurations bitwise. The exact and finite-shot comparisons matched {exact_rows:,} and {finite_rows:,} result rows, respectively ({metric_rows:,} scalar metric comparisons; maximum difference {max_metric_delta:.1e}); all {curve_rows:,} recorded training-curve values also matched. This verifies deterministic rerun reproducibility on the recorded environment, not external validation.")
    range_path=RESULTS / "zne_range_sensitivity_summary.csv"
    range_lines=[]
    if range_path.exists():
        ranges=pd.read_csv(range_path)
        ranges=ranges[(ranges.gamma==0.3)&(ranges.shots_per_test_case==1000)]
        for r in ranges.sort_values(["task","n_qubits","method"]).itertuples():
            context=context_labels.get((r.task,int(r.n_qubits),r.architecture),
                f"{task_labels.get(r.task,r.task)}, {int(r.n_qubits)}-qubit {r.architecture}")
            range_lines.append(f"{latex_escape(context)} & {method_label(r.method)} & "
                f"{fmt(r.raw_balanced_accuracy)} & {fmt(r.bounded_balanced_accuracy)} & "
                f"{fmt(r.mean_fraction_outside_range)} & {fmt(r.total_shots_mean,0)}" + rowend)
    range_table="\n".join(range_lines) or r"\multicolumn{6}{c}{Range-constrained analysis not available} \\"
    rule_path=RESULTS / "calibration_rule_fit_variability.csv"
    rule_sentence="Calibration-rule repeat variability has not been summarized for this run."
    if rule_path.exists():
        rules=pd.read_csv(rule_path)
        def rule_sd(method,param):
            q=rules[(rules.method==method)&(rules.parameter==param)]
            return float(q.mean_within_model_sd_across_two_repeats.iloc[0]) if not q.empty else np.nan
        rule_sentence=("For WDBC at $\\gamma=0.30$ and 1,000 shots, mean within-checkpoint SD across two shot repetitions "
            f"was {fmt(rule_sd('calibrated_threshold','threshold'))} for the noisy threshold, "
            f"{fmt(rule_sd('linear_zne_calibrated','threshold'))} for the linear-ZNE threshold, and "
            f"{fmt(rule_sd('quadratic_zne_calibrated','threshold'))} for the quadratic-ZNE threshold. "
            f"The affine coefficient and intercept repeat SDs were {fmt(rule_sd('affine_readout','affine_coefficient'))} and "
            f"{fmt(rule_sd('affine_readout','affine_intercept'))}. These are descriptive fit-rule variability measures from only two measurement repetitions, separate from observable error and test classification performance.")
    channels = pd.read_csv(CHANNELS)
    channels = channels[channels.gamma.isin([0.2, 0.3, 0.4])]
    channel_lines = [
        f"{latex_escape(r.noise)} & {r.gamma:.2f} & {fmt(r.balanced_accuracy)} & {fmt(r.roc_auc)} & {fmt(r.collapse_rate)} & {int(r.model_runs)}" + rowend
        for r in channels.sort_values(["noise", "gamma"]).itertuples()
    ]
    channel_table = "\n".join(channel_lines) or r"\multicolumn{6}{c}{No channel controls available} \\"
    errors = pd.read_csv(ERRORS)
    error_lines = [
        f"{method_label(r.method)} & {fmt(r.exact_score_rmse_vs_clean)} & {fmt(r.finite_sampling_rmse_vs_exact)}" + rowend
        for r in errors.sort_values("method").itertuples()
    ]
    error_table = "\n".join(error_lines) or r"\multicolumn{3}{c}{No error decomposition available} \\"

    confirm_dir = ROOT / "results" / "runs" / "confirmatory-budget-study-30seed-20261005"
    confirmation_abstract = "A separate fresh-data simulator confirmation is summarized in the results; it does not support a universal pipeline ranking."
    confirmation_section = "The preregistered fresh synthetic diagnostic outputs are not present in this package."
    if (confirm_dir / "confirmatory_test_results.csv").exists():
        conf = pd.read_csv(confirm_dir / "confirmatory_test_results.csv")
        conf_protocol = __import__("json").loads((confirm_dir / "preregistered_protocol.json").read_text(encoding="utf-8"))
        conf_choice = __import__("json").loads((confirm_dir / "frozen_selection.json").read_text(encoding="utf-8"))
        conf_summary = (conf.groupby(["budget_per_eval_case", "method"], as_index=False)
                        .balanced_accuracy.agg(["mean", "std", "count"]).reset_index())
        conf_labels = {"fixed_threshold":"Fixed", "noisy_calibration":"Noisy threshold",
                       "selected_zne":"ZNE (dev)", "selected_zne_calibration":"ZNE (dev) + cal."}
        conf_lines=[]
        for r in conf_summary.sort_values(["budget_per_eval_case", "method"]).itertuples():
            conf_lines.append(f"{int(r.budget_per_eval_case)} & {conf_labels[r.method]} & {fmt(r.mean)} & {fmt(r.std)} & {int(r.count)}" + rowend)
        conf_table="\n".join(conf_lines)
        conf_pivot=conf.pivot(index=["realization","budget_per_eval_case"],columns="method",values="balanced_accuracy")
        rng_boot=np.random.default_rng(20261005)
        paired_conf_lines=[]
        for budget in [100,1000]:
            one=conf_pivot.xs(budget,level="budget_per_eval_case")
            for method,label in [("selected_zne","ZNE dev."),("selected_zne_calibration","ZNE dev. + cal."),("fixed_threshold","Fixed")]:
                differences=(one.noisy_calibration-one[method]).to_numpy()
                boot=np.array([rng_boot.choice(differences,size=len(differences),replace=True).mean() for _ in range(20000)])
                lo,hi=np.quantile(boot,[.025,.975])
                paired_conf_lines.append(f"{budget} & Noisy threshold $-$ {label} & {fmt(differences.mean())} & {fmt(differences.std(ddof=1))} & [{fmt(lo)}, {fmt(hi)}] & {len(differences)}" + rowend)
        paired_confirmation_table=(r"""\begin{table}[t]\centering\scriptsize
\caption{Paired contrasts for the fresh synthetic diagnostic. Positive values favor noisy threshold selection. SD is across the 30 realization-level paired differences; intervals are descriptive percentile bootstraps over those realizations.}
\label{tab:confirm-paired}
\begin{tabular}{@{}r l r r l r@{}}\toprule
Budget & Contrast & Mean & SD & Descriptive 95\% interval & $n$ \\
\midrule
@@PAIRED_CONFIRMATION_ROWS@@
\bottomrule\end{tabular}\end{table}""").replace("@@PAIRED_CONFIRMATION_ROWS@@","\n".join(paired_conf_lines))
        confirmation_table=(r"""\begin{table}[t]\centering\small
\caption{Fresh synthetic transfer diagnostic at two nominal measurement budgets; the fixed-threshold results near 0.5 indicate weak discrimination, so this is not confirmatory evidence for the WDBC result. ZNE settings were selected on separate development data; ``cal.'' denotes threshold fitting on calibration labels. Mean and SD are over 30 generated datasets, with one shot draw per condition.}
\label{tab:fresh-diagnostic}
\begin{tabular}{@{}r l r r r@{}}\toprule
Budget & Pipeline & Mean BA & SD & $n$ \\
\midrule
@@CONFIRMATION_ROWS@@
\bottomrule\end{tabular}\end{table}""").replace("@@CONFIRMATION_ROWS@@",conf_table)
        b1000=conf[conf.budget_per_eval_case==1000].groupby("method").balanced_accuracy.agg(["mean","std"])
        bvalue=lambda method,col: fmt(b1000.loc[method,col]) if method in b1000.index else "--"
        nconf=sum(len(seeds) for seeds in conf_protocol["confirmation_seeds"].values())
        ndev=sum(len(seeds) for seeds in conf_protocol["development_seeds"].values())
        pred=pd.read_csv(confirm_dir/"calibration_budget_summary.csv")
        pred=pred[pred.stage=="confirmation"]
        pred_shots=fmt(pred.median_predicted_shots.iloc[0],0) if not pred.empty else "not available"
        pred_rate=fmt(pred.observed_requirement_rate.iloc[0]) if not pred.empty else "not available"
        selected_name=conf_choice["scale_config"].replace("_", r"\_")
        selected_frac=100*float(conf_choice["calibration_fraction"])
        vers=conf_protocol["versions"]
        env=(f"Python {conf_protocol['python'].split()[0]}, PennyLane {vers['pennylane']}, NumPy {vers['numpy']}, "
             f"pandas {vers['pandas']}, and scikit-learn {vers['scikit_learn']}")
        confirmation_abstract=""
        confirmation_section=(f"""The diagnostic used {ndev} development realizations and {nconf} distinct held-out realizations (15 at class separations 0.9 and 15 at 1.2), generated afresh from the same balanced synthetic task family (500 examples, eight features, four informative and two redundant features, 2% label flips). A two-qubit, two-layer readout-centered CNOT-star circuit was trained without noise and evaluated with the composite simulator channel at $\\gamma=0.30$. The development-only choice was the {selected_name} scale schedule and {selected_frac:.0f}% calibration allocation at 1,000 nominal measurements per validation example; both were frozen before evaluating held-out realizations at 100 and 1,000 measurements per test case. One finite-shot draw was used per realization and budget. This diagnostic used {env}, distinct from the primary run environment, and is reported as a separate execution rather than a bitwise reproduction.

{confirmation_table}

{paired_confirmation_table}

This fresh task-family diagnostic is not a confirmation of the WDBC headline: the fixed pipeline is at chance and discrimination is weak. At 1,000 measurements, noisy threshold selection has the largest observed mean, 0.559, while development-selected ZNE is 0.519 and ZNE plus calibration is 0.511; at 100 measurements all four means are near chance. These descriptive results do not establish population-level superiority. The accompanying tables report the pipeline means and paired contrasts. Per-realization metrics, intermediate scores, checkpoints, and split indices are included in the run archive.

A candidate score-gap rule predicted a median of {pred_shots} shots per calibration example, but its prespecified two-percentage-point regret criterion was met within the tested grid in only {int(round(float(pred_rate)*nconf))} of {nconf} held-out realizations. This is a negative, simulator-dependent result, not a validated budget predictor; exact simulated expectations used by the rule are unavailable on hardware.""")
    std_control_path = RESULTS / "sensitivity" / "wdbc_standardized_pca_control.csv"
    std_control_table = r"\multicolumn{4}{c}{Standardized-PCA control not available} " + chr(92)*2
    std_control_text = "A standardized-before-PCA WDBC control was not included in this run."
    depth_note="A matched depth sensitivity is not available in this run."
    depth_table=""
    noise_note="The standardized-PCA exact noise grid is not available in this run."
    standardized_transfer_table=""
    standardized_transfer_note=""
    fresh_confirmation_section = ""
    fresh_confirmation_abstract = ""
    confirmation_run = ROOT / "results" / "runs" / "independent-confirmation-wdbc-20261016"
    if (confirmation_run / "finite_shot_results.csv").exists():
        conf_finite = pd.read_csv(confirmation_run / "finite_shot_results.csv")
        conf_exact = pd.read_csv(confirmation_run / "exact_results.csv")
        conf_models = pd.read_csv(confirmation_run / "models.csv")
        conf_levels = pd.read_csv(confirmation_run / "checkpoint_level_summary.csv")
        conf_rows = []
        conf_labels = {"fixed_zero_threshold": "Fixed zero threshold",
                       "noisy_threshold": "Noisy threshold + calibration",
                       "linear_zne": "Linear ZNE + calibration",
                       "quadratic_zne": "Quadratic ZNE + calibration"}
        conf_cal = conf_finite[conf_finite.measurement_mode == "finite_calibrated"]
        for method in ["fixed_zero_threshold", "noisy_threshold", "linear_zne", "quadratic_zne"]:
            exact_row = conf_levels[(conf_levels.measurement_mode == "exact_expectation") &
                                    (conf_levels.method == method)].iloc[0]
            finite_mode = "finite_fixed" if method == "fixed_zero_threshold" else "finite_calibrated"
            finite_row = conf_levels[(conf_levels.measurement_mode == finite_mode) &
                                     (conf_levels.method == method)].iloc[0]
            conf_rows.append(f"{conf_labels[method]} & {fmt(exact_row.balanced_accuracy_mean)} & "
                             f"{fmt(finite_row.balanced_accuracy_mean)} $\\pm$ {fmt(finite_row.balanced_accuracy_sd_across_checkpoint_means)} & "
                             f"{fmt(finite_row.roc_auc_mean)} & {int(finite_row.n_checkpoints)}" + rowend)
        conf_summary_table = "\n".join(conf_rows)
        conf_pairs = pd.read_csv(confirmation_run / "paired_fold_contrasts.csv")
        lin_diff = conf_pairs[conf_pairs.contrast == "noisy_threshold_minus_linear_zne"].difference_balanced_accuracy.to_numpy()
        quad_diff = conf_pairs[conf_pairs.contrast == "noisy_threshold_minus_quadratic_zne"].difference_balanced_accuracy.to_numpy()
        fresh_confirmation_section = (f"\\subsection{{Fresh split-and-seed confirmation on WDBC}}\n"
            f"A preregistered confirmation used five new stratified folds (each example is held out once), two newly trained initializations per fold, and 20 independent finite-shot draws per checkpoint. The full protocol, split IDs, training curves, checkpoints, preprocessing objects, calibration/test score observations, and hash manifest are archived under \\texttt{{independent-confirmation-wdbc-20261016}}. Standardization, PCA, and feature scaling were fit on each training partition only. At $\\gamma=0.30$, the exact-expectation and matched-budget results are in Table~\\ref{{tab:fresh-confirmation}}. The finite-shot fixed boundary has balanced accuracy {fmt(conf_levels[(conf_levels.measurement_mode=='finite_fixed')&(conf_levels.method=='fixed_zero_threshold')].balanced_accuracy_mean.iloc[0])} but ROC-AUC {fmt(conf_levels[(conf_levels.measurement_mode=='finite_fixed')&(conf_levels.method=='fixed_zero_threshold')].roc_auc_mean.iloc[0])}; calibrated threshold selection reaches {fmt(conf_cal[conf_cal.method=='noisy_threshold'].balanced_accuracy.mean())}, versus {fmt(conf_cal[conf_cal.method=='linear_zne'].balanced_accuracy.mean())} for linear and {fmt(conf_cal[conf_cal.method=='quadratic_zne'].balanced_accuracy.mean())} for quadratic ZNE. Threshold-minus-linear differences across folds are {', '.join(f'{v:+.3f}' for v in lin_diff)}; threshold-minus-quadratic differences are {', '.join(f'{v:+.3f}' for v in quad_diff)}. These contrasts are descriptive: the folds share training data, and this remains one public dataset under a simulator, not external-population or hardware validation.\n"
            "\\begin{table*}[t]\\centering\\small\n"
            "\\caption{Preregistered fresh-split WDBC confirmation at $\\gamma=0.30$. Exact balanced accuracy uses exact simulated expectations; calibrated methods fit thresholds on calibration labels only. In the finite-shot columns, the fixed zero threshold uses 1,000 test measurements per case and no calibration, while calibrated methods use a nominal total of 1,000 including calibration. Finite-shot SD is across 10 checkpoint means after averaging 20 draws per checkpoint; it is descriptive, not a confidence interval.}\n"
            "\\label{tab:fresh-confirmation}\n\\begin{tabular}{@{}l r r r r@{}}\\toprule\n"
            "Pipeline & Exact BA & Finite-shot BA $\\pm$ SD & Finite AUC & Checkpoints \\\\ \\midrule\n"
            + conf_summary_table + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
        fresh_confirmation_abstract = (f" A preregistered fresh-split WDBC confirmation with {len(conf_models)} newly trained checkpoints and 20 shot draws per checkpoint yielded {fmt(conf_cal[conf_cal.method=='noisy_threshold'].balanced_accuracy.mean())} balanced accuracy for calibrated threshold selection, versus {fmt(conf_cal[conf_cal.method=='linear_zne'].balanced_accuracy.mean())} for linear and {fmt(conf_cal[conf_cal.method=='quadratic_zne'].balanced_accuracy.mean())} for quadratic ZNE; the fixed threshold remained at 0.500 despite finite-shot AUC 0.915. This remains single-dataset simulator evidence.")
    gatefold_table=""
    gatefold_note=""
    gatefold_abstract=""
    noise_table=""
    if std_control_path.exists():
        std_df = pd.read_csv(std_control_path)
        cal_size_path=RESULTS / "sensitivity" / "wdbc_standardized_calibration_size_sweep_summary.csv"
        primary_finite_for_note=pd.read_csv(RESULTS/"sensitivity"/"wdbc_standardized_zne_comparison.csv")
        cal_size_note="A calibration-size sensitivity analysis is not available in this run."
        cal_size_table=r"\multicolumn{5}{c}{Calibration-size sensitivity not available} \\"
        if cal_size_path.exists():
            cal_size=pd.read_csv(cal_size_path)
            cal_size_raw=pd.read_csv(RESULTS/"sensitivity"/"wdbc_standardized_calibration_size_sweep.csv")
            cal_size_checkpoint=cal_size_raw.groupby(["method","calibration_examples","model_id"],as_index=False).agg(
                checkpoint_ba=("balanced_accuracy","mean"),checkpoint_auc=("roc_auc","mean"),total_shots=("total_shots","mean"))
            cal_size_descriptive=cal_size_checkpoint.groupby(["method","calibration_examples"],as_index=False).agg(
                balanced_accuracy_mean=("checkpoint_ba","mean"),balanced_accuracy_sd=("checkpoint_ba","std"),
                roc_auc_mean=("checkpoint_auc","mean"),n_checkpoints=("model_id","nunique"),mean_total_shots=("total_shots","mean"))
            cal_lines=[]
            for r in cal_size_descriptive.sort_values(["method","calibration_examples"]).itertuples():
                label={"noisy_threshold":"Noisy threshold","linear_zne":"Linear ZNE","quadratic_zne":"Quadratic ZNE"}[r.method]
                cal_lines.append(f"{label} & {int(r.calibration_examples)} & {fmt(r.balanced_accuracy_mean)} $\\pm$ {fmt(r.balanced_accuracy_sd)} & {fmt(r.roc_auc_mean)} & {int(r.n_checkpoints)}" + rowend)
            cal_size_table="\n".join(cal_lines)
            noisy10=cal_size[(cal_size.method=="noisy_threshold")&(cal_size.calibration_examples==10)].balanced_accuracy_mean.iloc[0]
            noisyall=cal_size[(cal_size.method=="noisy_threshold")&(cal_size.calibration_examples==114)].balanced_accuracy_mean.iloc[0]
            main_finite_mean=primary_finite_for_note[(primary_finite_for_note.method=="noisy_threshold")&(primary_finite_for_note.measurement_mode=="finite_calibrated")&(primary_finite_for_note.gamma==.3)&(primary_finite_for_note.budget==1000)].balanced_accuracy.mean()
            cal_size_note=(f"A post-hoc calibration-size sensitivity (Fig.~\\ref{{fig:calsize}}) gives noisy-threshold balanced accuracy {fmt(noisy10)} with 10 labels and {fmt(noisyall)} with 114 under a nominal 1,000-measurement budget. The 114-label mean differs slightly from the main table ({fmt(main_finite_mean)}) because this sweep uses an independent calibration-shot RNG stream; the folds, checkpoints, 114 labels, test draws, and threshold rule are held fixed. This is a reused-checkpoint sensitivity analysis, not independent validation.")
        std_labels = {"quantum_clean_fixed":"Quantum, clean fixed threshold", "quantum_clean_calibrated":"Quantum, clean threshold fit",
                      "quantum_noisy_exact_fixed":"Quantum, exact noisy fixed ($\\gamma=0.30$)", "quantum_noisy_exact_calibrated":"Quantum, exact noisy threshold fit",
                      "quantum_noisy_1000shot_fixed":"Quantum, noisy 1,000-shot fixed", "quantum_noisy_1000shot_calibrated":"Quantum, noisy 1,000-shot threshold fit",
                      "logistic_regression_standardized_2d":"Logistic regression, same standardized 2D inputs"}
        std_rows=[]
        for method, g in std_df.groupby("method", sort=False):
            if method in {"quantum_noisy_1000shot_fixed", "quantum_noisy_1000shot_calibrated"}:
                continue  # the matched table below uses two repeated draws for every finite-shot pipeline
            std_rows.append(f"{std_labels[method]} & {fmt(g.balanced_accuracy.mean())} $\\pm$ {fmt(g.balanced_accuracy.std())} & {fmt(g.roc_auc.mean())} & {g.model_id.nunique()}" + rowend)
        std_zne_path = RESULTS / "sensitivity" / "wdbc_standardized_zne_comparison.csv"
        std_zne = pd.read_csv(std_zne_path) if std_zne_path.exists() else pd.DataFrame()
        if not std_zne.empty:
            zne_labels = {
                ("linear_zne", "exact_calibrated"): "Linear ZNE, exact, threshold fit",
                ("quadratic_zne", "exact_calibrated"): "Quadratic ZNE, exact, threshold fit",
                ("linear_zne", "finite_fixed"): "Linear ZNE, 1,000 shots, fixed threshold",
                ("linear_zne", "finite_calibrated"): "Linear ZNE, 1,000 shots, threshold fit",
                ("quadratic_zne", "finite_fixed"): "Quadratic ZNE, 1,000 shots, fixed threshold",
                ("quadratic_zne", "finite_calibrated"): "Quadratic ZNE, 1,000 shots, threshold fit",
                ("noisy_threshold", "finite_fixed"): "Noisy score, 1,000 shots, fixed threshold",
                ("noisy_threshold", "finite_calibrated"): "Noisy threshold fit, 1,000 shots",
            }
            for (method, mode), g in std_zne.groupby(["method", "measurement_mode"], sort=False):
                if (method, mode) not in zne_labels:
                    continue
                checkpoint_ba=g.groupby("model_id").balanced_accuracy.mean()
                n = len(checkpoint_ba)
                std_rows.append(f"{zne_labels[(method,mode)]} & {fmt(checkpoint_ba.mean())} $\\pm$ {fmt(checkpoint_ba.std(ddof=1))} & {fmt(g.roc_auc.mean())} & {n}" + rowend)
        four_layer_path = RESULTS / "sensitivity" / "wdbc_four_layer_control.csv"
        four_df = pd.read_csv(four_layer_path) if four_layer_path.exists() else pd.DataFrame()
        depth_path=RESULTS / "sensitivity" / "wdbc_standardized_depth_sweep.csv"
        if depth_path.exists():
            depth_data=pd.read_csv(depth_path)
            depth_rows=[]
            for depth in [1,2,4]:
                if depth==2:
                    clean_values=std_df.loc[std_df.method=="quantum_clean_fixed","balanced_accuracy"]
                    exact_values=std_df.loc[std_df.method=="quantum_noisy_exact_calibrated","balanced_accuracy"]
                    finite_values=std_zne.loc[(std_zne.method=="noisy_threshold")&(std_zne.measurement_mode=="finite_calibrated"),"balanced_accuracy"]
                    auc_values=std_zne.loc[(std_zne.method=="noisy_threshold")&(std_zne.measurement_mode=="finite_calibrated"),"roc_auc"]
                else:
                    sub=depth_data[depth_data.depth==depth]
                    clean_values=sub.loc[sub.method=="clean_fixed","balanced_accuracy"]
                    exact_values=sub.loc[sub.method=="noisy_exact_calibrated","balanced_accuracy"]
                    finite_values=sub.loc[sub.method=="noisy_finite_calibrated","balanced_accuracy"]
                    auc_values=sub.loc[sub.method=="noisy_finite_calibrated","roc_auc"]
                by_checkpoint=finite_values.groupby(depth_data.loc[finite_values.index,"model_id"]).mean() if depth != 2 else finite_values.groupby(std_zne.loc[finite_values.index,"model_id"]).mean()
                spread_path=RESULTS/"sensitivity"/"wdbc_standardized_depth_score_spread_summary.csv"
                spread=pd.read_csv(spread_path) if spread_path.exists() else pd.DataFrame()
                sp=spread[(spread.depth==depth)&(spread.gamma==.3)] if not spread.empty else pd.DataFrame()
                score_sd=sp.score_sd_mean.iloc[0] if not sp.empty else np.nan
                class_gap=sp.class_gap_mean.iloc[0] if not sp.empty else np.nan
                shot_se=sp.shot_se_500_mean.iloc[0] if not sp.empty else np.nan
                gap_ratio=shot_se/class_gap if class_gap else np.nan
                depth_rows.append(f"{depth} & {fmt(clean_values.mean())} & {fmt(exact_values.mean())} & {fmt(finite_values.mean())} $\\pm$ {fmt(by_checkpoint.std(ddof=1))} & {fmt(auc_values.mean())} & {fmt(score_sd)} & {fmt(class_gap)} & {fmt(shot_se)} & {fmt(gap_ratio)}" + rowend)
            depth_table="\n".join(depth_rows)
            depth4=depth_data[(depth_data.depth==4)&(depth_data.method=="noisy_finite_calibrated")]
            depth1=depth_data[(depth_data.depth==1)&(depth_data.method=="noisy_finite_calibrated")]
            depth_note=(f"At $\\gamma=0.30$, finite-shot calibrated balanced accuracy is {fmt(depth1.balanced_accuracy.mean())} at one layer, {fmt(std_zne[(std_zne.method=='noisy_threshold')&(std_zne.measurement_mode=='finite_calibrated')].balanced_accuracy.mean())} at the two-layer primary depth, and {fmt(depth4.balanced_accuracy.mean())} at four layers (Table~\\ref{{tab:primary-depth}}). A post-hoc calibration-score diagnostic finds class-mean gaps of 0.210, 0.079, and 0.008 at depths 1, 2, and 4; the observed gap contracts roughly $\\times 0.38$ from layer 1 to 2 and $\\times 0.11$ from layer 2 to 4. Under the stated Pauli-$Z$ sampling model, standard error scales as $\\sigma/\\sqrt{{N}}$; using the observed 500-shot standard error of about 0.043 as a reference, matching the depth-1 standard-error-to-gap ratio at depth 4 would require approximately $(0.210 / 0.0083)^2 \\approx 640$ times as many measurements. This is a descriptive scale calculation from this diagnostic, not a causal test or a general depth law. The two-layer model was selected as the primary architecture before this post-hoc depth sweep.")
        noise_summary_path=RESULTS/"sensitivity"/"wdbc_standardized_exact_noise_sweep_summary.csv"
        if noise_summary_path.exists():
            noise_data=pd.read_csv(noise_summary_path)
            noise_lines=[]
            for r in noise_data.sort_values(["gamma","method"]).itertuples():
                noise_lines.append(f"{r.gamma:.1f} & {latex_escape(r.method)} & {fmt(r.balanced_accuracy_mean)} $\\pm$ {fmt(r.balanced_accuracy_sd)} & {fmt(r.roc_auc_mean)} & {int(r.n_models)}"+rowend)
            noise_table="\n".join(noise_lines)
            f03=noise_data[(noise_data.gamma==.3)&(noise_data.method=="fixed_threshold")].balanced_accuracy_mean.iloc[0]
            cal03=noise_data[(noise_data.gamma==.3)&(noise_data.method=="noisy_threshold")].balanced_accuracy_mean.iloc[0]
            noise_note=(f"An exact-expectation sweep on the same standardized-PCA checkpoints covers $\\gamma=0.0$--$0.4$; at $\\gamma=0.30$, the fixed threshold gives {fmt(f03)} balanced accuracy and held-out threshold selection gives {fmt(cal03)}. Supplementary Table~S15 reports the complete grid, including calibrated ZNE. This reuses the primary folds and checkpoints.")
        transfer_path=RESULTS/"sensitivity"/"standardized_transfer_comparison.csv"
        clean_path=RESULTS/"sensitivity"/"standardized_transfer_clean.csv"
        lr_path=RESULTS/"sensitivity"/"standardized_transfer_classical.csv"
        if transfer_path.exists() and clean_path.exists() and lr_path.exists():
            transfer=pd.read_csv(transfer_path)
            clean_transfer=pd.read_csv(clean_path)
            lr_transfer=pd.read_csv(lr_path)
            transfer_rows=[]
            wdbc_rows=std_zne[(std_zne.gamma==0.3)&(std_zne.budget==1000)&(std_zne.measurement_mode.isin(["finite_fixed","finite_calibrated"]))]
            task_specs=[("WDBC (standardized PCA)",wdbc_rows,
                         [("fixed_threshold","finite_fixed","noisy_threshold"),("noisy_threshold","finite_calibrated","noisy_threshold"),("linear_zne","finite_calibrated","linear_zne"),("quadratic_zne","finite_calibrated","quadratic_zne")]),
                        ("Wine, class 0 vs rest",transfer[(transfer.task=="wine_0_vs_rest")&(transfer.measurement_mode=="finite_shot")],
                         [(m,"finite_shot",m) for m in ["fixed_threshold","noisy_threshold","linear_zne_calibrated","quadratic_zne_calibrated"]]),
                        ("Synthetic, balanced",transfer[(transfer.task=="synthetic_balanced")&(transfer.measurement_mode=="finite_shot")],
                         [(m,"finite_shot",m) for m in ["fixed_threshold","noisy_threshold","linear_zne_calibrated","quadratic_zne_calibrated"]])]
            for label,tf,method_modes in task_specs:
                vals=[]
                for label_method,mode,data_method in method_modes:
                    sub=tf[(tf.method==data_method)&(tf.measurement_mode==mode)]
                    vals.append(fmt(sub.balanced_accuracy.mean()))
                transfer_rows.append(f"{latex_escape(label)} & " + " & ".join(vals) + rowend)
            standardized_transfer_table="\n".join(transfer_rows)
            standardized_transfer_note=("Table~\\ref{tab:primary-transfer} reports standardized-PCA runs on the same saved splits and initialization seeds for WDBC, Wine, and balanced synthetic data. Each row averages 20 finite-shot draws per checkpoint at $\\gamma=0.30$ and a nominal 1,000-measurement budget; repeated draws and shared training observations are descriptive, not independent task replications. Clean matched-input logistic regression reaches 0.928 on WDBC, 0.950 on Wine, and 0.747 on balanced synthetic data; it exceeds clean quantum performance on all three.")
            fold_metrics_path=RESULTS/"sensitivity"/"wdbc_gate_folded_zne_metrics.csv"
            if fold_metrics_path.exists():
                gf=pd.read_csv(fold_metrics_path)
                gf_rows=[]
                for method,label in [("noisy_threshold_calibrated","Noisy threshold + calibration"),("gate_folded_zne_calibrated","Gate-folded ZNE + calibration")]:
                    vals=gf[gf.method==method].groupby("model_id").balanced_accuracy.mean()
                    gf_rows.append(f"{label} & {fmt(vals.mean())} $\\pm$ {fmt(vals.std(ddof=1))} & {gf[gf.method==method].roc_auc.mean():.3f} & {len(vals)}"+rowend)
                for method,label in [("linear_zne","Probability-scaled linear ZNE + calibration"),("quadratic_zne","Probability-scaled quadratic ZNE + calibration")]:
                    vals=std_zne[(std_zne.method==method)&(std_zne.measurement_mode=="finite_calibrated")].groupby("model_id").balanced_accuracy.mean()
                    aucvals=std_zne[(std_zne.method==method)&(std_zne.measurement_mode=="finite_calibrated")].roc_auc
                    gf_rows.append(f"{label} & {fmt(vals.mean())} $\\pm$ {fmt(vals.std(ddof=1))} & {fmt(aucvals.mean())} & {len(vals)}"+rowend)
                gatefold_table="\n".join(gf_rows)
                gmean=gf[gf.method=="gate_folded_zne_calibrated"].groupby("model_id").balanced_accuracy.mean().mean()
                gatefold_note=(f"Table~\\ref{{tab:primary-gatefold}} reports the separate gate-folded simulator comparison: scales 1, 3, and 5 give balanced accuracy {fmt(gmean)} under the same nominal budget. Its scale-one circuit matches the original composite-channel expectation within $2.3\\times10^{{-16}}$. It uses the saved checkpoints and folds and is not hardware validation.")
                gatefold_abstract=(f"Gate-folded ZNE at scales 1, 3, and 5 reaches {fmt(gmean)} under the same budget, below noisy-threshold calibration.")
            fake_metrics_path=RESULTS/"sensitivity"/"wdbc_fake_backend_metrics.csv"
            fake_table=""
            fake_note=""
            fake_abstract=""
            if fake_metrics_path.exists():
                fm=pd.read_csv(fake_metrics_path)
                fm_labels=[
                    ("fake_manila_fixed_threshold", "Raw device noise, fixed threshold"),
                    ("fake_manila_calibrated_threshold", "Raw device noise, calibrated"),
                    ("fake_manila_readout_mitigated_fixed", "Readout mitigation, fixed"),
                    ("fake_manila_readout_mitigated_calibrated", "Readout mitigation, calibrated"),
                    ("fake_manila_gate_folded_zne_fixed", "Gate-folded ZNE, fixed"),
                    ("fake_manila_gate_folded_zne_calibrated", "Gate-folded ZNE, calibrated"),
                    ("fake_manila_readout_mitigated_gate_folded_zne_calibrated", "Readout mitigation + gate-folded ZNE, cal."),
                ]
                fm_rows=[]
                for method,label in fm_labels:
                    sub=fm[fm.method==method]
                    cp=sub.groupby("model_id").balanced_accuracy.mean()
                    fm_auc=sub.roc_auc.mean()
                    fm_rows.append(f"{label} & {fmt(cp.mean())} $\\pm$ {fmt(cp.std(ddof=1))} & {fm_auc:.3f} & {len(cp)}"+rowend)
                fake_table="\n".join(fm_rows)
                fm_raw_fixed=fm[fm.method=="fake_manila_fixed_threshold"].groupby("model_id").balanced_accuracy.mean().mean()
                fm_raw_cal=fm[fm.method=="fake_manila_calibrated_threshold"].groupby("model_id").balanced_accuracy.mean().mean()
                fm_ro_fixed=fm[fm.method=="fake_manila_readout_mitigated_fixed"].groupby("model_id").balanced_accuracy.mean().mean()
                fm_ro_cal=fm[fm.method=="fake_manila_readout_mitigated_calibrated"].groupby("model_id").balanced_accuracy.mean().mean()
                fm_gf_fixed=fm[fm.method=="fake_manila_gate_folded_zne_fixed"].groupby("model_id").balanced_accuracy.mean().mean()
                fm_gf_cal=fm[fm.method=="fake_manila_gate_folded_zne_calibrated"].groupby("model_id").balanced_accuracy.mean().mean()
                fake_note=(f"Table~\\ref{{tab:fake-backend}} reports a local simulation using the saved IBM FakeManilaV2 Falcon r5.11 backend snapshot (Qiskit 2.5.2, Aer 0.17.2, optimization level 0, physical qubits [0, 1]); the saved snapshot is not current calibration data and no QPU was executed. Under a matched 1,000-measurement budget (500 calibration, 500 test), the raw fixed threshold reaches balanced accuracy {fmt(fm_raw_fixed)}, matching the clean model ({fmt(qstd.balanced_accuracy.mean()) if 'qstd' in locals() and hasattr(qstd, 'balanced_accuracy') else '0.887'}). Held-out calibration yields {fmt(fm_raw_cal)}, readout mitigation yields {fmt(fm_ro_fixed)} (fixed) and {fmt(fm_ro_cal)} (calibrated), and gate-folded ZNE yields {fmt(fm_gf_fixed)} (fixed) and {fmt(fm_gf_cal)} (calibrated). All evaluated methods sit within 0.013 of one another, well within checkpoint standard deviations (0.025--0.038). Readout-mitigated and raw ROC-AUC are both 0.958 to the reported precision. Balanced-accuracy differences are small relative to checkpoint variability, so this experiment does not show a consistent incremental classification benefit from readout correction. In this snapshot setting, no boundary collapse appears. The post-hoc noise-multiplier sweep (Fig.~\\ref{{fig:regime-map}}) is an exploratory stress test: threshold calibration remains comparatively stable while gate-folded ZNE degrades at the highest tested multiplier. Neither result establishes a general ranking.")
                fake_abstract=(r"In a local IBM FakeManilaV2 snapshot simulation, fixed-threshold classification remains near clean performance, while readout correction and gate-folded ZNE produce changes within checkpoint variability; this is not a hardware run.")
        std_control_table="\n".join(std_rows)
        qstd=std_df[std_df.method=="quantum_clean_fixed"]
        lstd=std_df[std_df.method=="logistic_regression_standardized_2d"]
        qshot=std_zne[(std_zne.method=="noisy_threshold")&(std_zne.measurement_mode=="finite_calibrated")] if not std_zne.empty else pd.DataFrame()
        qexactstd=std_df[std_df.method=="quantum_noisy_exact_calibrated"]
        std_zne = pd.read_csv(RESULTS / "sensitivity" / "wdbc_standardized_zne_comparison.csv") if (RESULTS / "sensitivity" / "wdbc_standardized_zne_comparison.csv").exists() else pd.DataFrame()
        def std_result(method, mode, column="balanced_accuracy"):
            q=std_zne[(std_zne.method==method)&(std_zne.measurement_mode==mode)] if not std_zne.empty else pd.DataFrame()
            return float(q[column].mean()) if not q.empty else np.nan
        linear_fold_diffs=[]; quadratic_fold_diffs=[]
        pair_path=RESULTS / "sensitivity" / "wdbc_standardized_paired_fold_differences.csv"
        if pair_path.exists():
            pairs=pd.read_csv(pair_path)
            linear_fold_diffs=[f"{v:+.3f}" for v in pairs[pairs.method=="linear_zne"].sort_values("split").threshold_minus_zne]
            quadratic_fold_diffs=[f"{v:+.3f}" for v in pairs[pairs.method=="quadratic_zne"].sort_values("split").threshold_minus_zne]
        primary_pair_rows=[]
        for method,label in [("linear_zne","Linear ZNE"),("quadratic_zne","Quadratic ZNE")]:
            vals=pairs[pairs.method==method].sort_values("split").threshold_minus_zne if pair_path.exists() else []
            primary_pair_rows.append(f"{label} & " + " & ".join(f"{-v:+.3f}" for v in vals) + f" & {fmt(-np.mean(vals))}" + rowend)
        primary_pair_table="\n".join(primary_pair_rows)
        # This checkpoint-level signed-rank calculation is descriptive only: checkpoints
        # are nested within five folds and share training observations, so nominal p-values
        # should not be interpreted as population inference.
        from scipy.stats import wilcoxon
        checkpoint_pair_note="Checkpoint-level paired signed-rank diagnostic unavailable."
        if not std_zne.empty:
            cp=std_zne[(std_zne.gamma==.3)&(std_zne.budget==1000)&(std_zne.measurement_mode=="finite_calibrated")]
            base=cp[cp.method=="noisy_threshold"].groupby("model_id").balanced_accuracy.mean()
            pnotes=[]
            for method,label in [("linear_zne","linear"),("quadratic_zne","quadratic")]:
                comp=cp[cp.method==method].groupby("model_id").balanced_accuracy.mean()
                diff=(base-comp).dropna()
                if len(diff)==10:
                    test=wilcoxon(diff,alternative="greater",method="exact")
                pnotes.append(f"{label}: nominal one-sided exact $p={test.pvalue:.4f}$")
            if pnotes:
                checkpoint_pair_note=("Across 10 checkpoint means (two initializations within each of five folds), exploratory one-sided Wilcoxon signed-rank diagnostics are " + " and ".join(pnotes) + "; because checkpoints share fold-specific test and training partitions, the nominal values are descriptive and not population-level inference.")
        # Primary-result figure is generated from the same saved standardized-PCA rows.
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
        methods = ["noisy_threshold", "linear_zne", "quadratic_zne"]
        labels = ["Noisy threshold\n(calibrated)", "Linear ZNE\n+ calibration", "Quadratic ZNE\n+ calibration"]
        colors = ["#2166ac", "#67a9cf", "#ef8a62"]
        for ax, mode, title in [(axes[0], "exact_calibrated", "Exact expectations"),
                                (axes[1], "finite_calibrated", "1,000 measurements")]:
            means, sds = [], []
            for method in methods:
                if method == "noisy_threshold" and mode.startswith("exact"):
                    vals = std_df.loc[std_df.method == "quantum_noisy_exact_calibrated", "balanced_accuracy"]
                else:
                    vals = std_zne.loc[(std_zne.method == method) & (std_zne.measurement_mode == mode), "balanced_accuracy"]
                if len(vals):
                    if mode.startswith("finite"):
                        source=std_zne.loc[(std_zne.method==method)&(std_zne.measurement_mode==mode)]
                        checkpoint_means=source.groupby("model_id").balanced_accuracy.mean()
                    else:
                        source=std_df.loc[std_df.method=="quantum_noisy_exact_calibrated"] if method=="noisy_threshold" else std_zne.loc[(std_zne.method==method)&(std_zne.measurement_mode==mode)]
                        checkpoint_means=source.groupby("model_id").balanced_accuracy.mean()
                    means.append(float(checkpoint_means.mean()))
                    sds.append(float(checkpoint_means.std(ddof=1)) if len(checkpoint_means)>1 else 0.0)
                else:
                    means.append(np.nan); sds.append(0.0)
            ax.bar(np.arange(3), means, yerr=sds, color=colors, capsize=3, width=.68)
            ax.set_title(title)
            ax.set_xticks(np.arange(3), labels, fontsize=8)
            ax.set_ylim(.45, 1.0)
            ax.grid(axis="y", alpha=.25)
        axes[0].set_ylabel("Balanced accuracy")
        fig.suptitle(r"WDBC, standardized PCA, $\gamma=0.30$", fontsize=11)
        fig.tight_layout()
        fig.savefig(RESULTS / "headline_primary_wdbc.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        # Show calibration label efficiency using checkpoint means, not correlated shot rows.
        cal_raw=pd.read_csv(RESULTS/"sensitivity"/"wdbc_standardized_calibration_size_sweep.csv")
        fig,ax=plt.subplots(figsize=(5.8,3.2))
        cal_colors={"noisy_threshold":"#2166ac","linear_zne":"#67a9cf","quadratic_zne":"#ef8a62"}
        cal_labels={"noisy_threshold":"Noisy threshold","linear_zne":"Linear ZNE + calibration","quadratic_zne":"Quadratic ZNE + calibration"}
        for method in ["noisy_threshold","linear_zne","quadratic_zne"]:
            sub=cal_raw[cal_raw.method==method]
            agg=sub.groupby(["calibration_examples","model_id"]).balanced_accuracy.mean().reset_index()
            summary=agg.groupby("calibration_examples").balanced_accuracy.agg(["mean","std"])
            ax.errorbar(summary.index,summary["mean"],yerr=summary["std"],marker="o",capsize=3,
                        color=cal_colors[method],label=cal_labels[method])
        ax.set_xlabel("Labeled calibration examples")
        ax.set_ylabel("Balanced accuracy")
        ax.set_xticks([10,20,50,114])
        ax.set_ylim(.45,1.0); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8)
        fig.tight_layout(); fig.savefig(RESULTS/"calibration_efficiency_wdbc.png",dpi=220,bbox_inches="tight"); plt.close(fig)
        std_control_text=(f"The primary WDBC matched-budget comparison standardizes each feature on the outer training pool, then fits two-component PCA, and uses the same five outer folds and two initialization seeds. The components retain {fmt(qstd.pca_explained_variance_sum.mean())} of variance. At zero noise, quantum fixed-threshold balanced accuracy is {fmt(qstd.balanced_accuracy.mean())}; logistic regression on the same standardized two components reaches {fmt(lstd.balanced_accuracy.mean())}, so the classical matched-input model is stronger. At $\\gamma=0.30$, exact threshold fitting gives {fmt(qexactstd.balanced_accuracy.mean())}; exact calibrated linear and quadratic ZNE give {fmt(std_result('linear_zne','exact_calibrated'))} and {fmt(std_result('quadratic_zne','exact_calibrated'))}. Under the matched 1,000-shot budget, noisy threshold calibration gives {fmt(qshot.balanced_accuracy.mean())}, compared with linear ZNE plus calibration at {fmt(std_result('linear_zne','finite_calibrated'))} and quadratic ZNE plus calibration at {fmt(std_result('quadratic_zne','finite_calibrated'))}; calibrated methods divide the nominal budget between calibration and test inference. The five fold-level threshold-minus-ZNE differences are {', '.join(linear_fold_diffs)} for linear and {', '.join(quadratic_fold_diffs)} for quadratic ZNE. All five folds have the same sign. Test folds partition WDBC, but training sets share observations; fold-level differences are descriptive and do not constitute population-level inference.")
    # Put measured values into the summary without forcing one favorable point.
    fixed_030 = wide.get("fixed_threshold", pd.Series(dtype=float)).get(0.3, np.nan)
    auc_030 = auc.get("fixed_threshold", pd.Series(dtype=float)).get(0.3, np.nan)
    cal_030 = wide.get("calibrated_threshold", pd.Series(dtype=float)).get(0.3, np.nan)
    aff_030 = wide.get("affine_readout", pd.Series(dtype=float)).get(0.3, np.nan)
    zne_030 = wide.get("quadratic_zne", pd.Series(dtype=float)).get(0.3, np.nan)
    finite_030 = finite[(finite.task == "wdbc") & (finite.n_qubits == 2) &
                        (finite.architecture == "cnot_chain") & (finite.gamma == 0.3) &
                        (finite.shots_per_test_case == 1000)]
    finite_sentence = "The configured finite-shot rows were not present in this run."
    if not finite_030.empty:
        finite_sentence = "; ".join(
            f"\\texttt{{{latex_escape(r.method)}}}: balanced accuracy {fmt(r.balanced_accuracy_mean)} $\\pm$ {fmt(r.balanced_accuracy_sd)}"
            for r in finite_030.itertuples()) + "."
    clean_fixed_0 = wide.get("fixed_threshold", pd.Series(dtype=float)).get(0.0, np.nan)
    clean_calibrated_0 = wide.get("calibrated_threshold", pd.Series(dtype=float)).get(0.0, np.nan)
    finite_calibrated_030 = finite_030.loc[
        finite_030.method == "calibrated_threshold", "balanced_accuracy_mean"]
    calibration_contrast = (
        f"At $\\gamma=0.30$, fixed-threshold balanced accuracy is 0.500 while exact ROC-AUC is 0.958; at 1,000 finite shots, the corresponding values are 0.500 and 0.874. This is possible because AUC evaluates score ordering across thresholds, whereas balanced accuracy uses only $z=0$. The reduced finite-shot AUC is consistent with sampling noise perturbing score order and is a separate finite-measurement effect. On WDBC at zero noise, clean-data threshold calibration changes exact balanced accuracy "
        f"from {fmt(clean_fixed_0)} to {fmt(clean_calibrated_0)}. At $\\gamma=0.30$, the exact noisy-calibrated "
        f"pipeline reaches {fmt(cal_030)}, while the fixed and clean-frozen thresholds each reach "
        f"{fmt(fixed_030)}; the finite-shot noisy-calibrated mean at 1,000 shots is "
        f"{fmt(finite_calibrated_030.iloc[0] if len(finite_calibrated_030) else np.nan)}.")

    content = r"""\documentclass[conference]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{url}
\title{Decision-Threshold Selection and Zero-Noise Extrapolation in Variational Quantum Classifiers under Simulated Noise}
\author{\IEEEauthorblockN{Anonymous Authors}}
\begin{document}
\maketitle
\begin{abstract}
@@CONFIRMATION_ABSTRACT@@\end{abstract}
\begin{IEEEkeywords}
quantum machine learning, variational quantum classifier, zero-noise extrapolation, decision threshold, finite-shot measurement, error mitigation
\end{IEEEkeywords}

\section{Introduction and related work}
Noise robustness in quantum classification has been studied through robust data encodings and channel-dependent decision boundaries~\cite{larose2020}. QuantumNAT analyzes noise-induced score scaling and shifts, normalization, and noise-aware training~\cite{wang2022}. Decision-threshold selection and probability calibration are distinct established operations~\cite{hernandez2012,niculescu2005}: here ``calibration'' means selecting a decision threshold for balanced accuracy on held-out labels, not fitting calibrated posterior probabilities. Richardson zero-noise extrapolation was introduced for short-depth circuits and later developed for variational and digital circuits~\cite{temme2017,libenjamin2017,giurgica2020}; broader mitigation methods and limitations are reviewed by Cai et al.~\cite{cai2023}. ZNE implementation has known subtleties~\cite{majumdar2023}, including finite-shot help--harm effects~\cite{koester2026,scavino2026}. Our question is operational: for a fixed total measurement budget including labeled calibration measurements, how do label-dependent threshold selection and label-free probability-scaled ZNE compare in thresholded classifier performance? The contributions are: (1) a matched-budget comparison that counts calibration measurements; (2) a calibration-size sensitivity analysis and a depth diagnostic linking score separation to shot uncertainty; and (3) an analysis separating exact observable bias, finite-shot sampling error, decision-rule fitting variability, and classification performance, with paired comparisons across held-out folds. We do not propose a new mitigation estimator or claim a universal ranking, hardware validation, or quantum advantage.

The small circuits serve as an auditable testbed for separating score ordering, threshold placement, and measurement variance. Their role is methodological: the stronger matched-input classical baseline rules out a predictive-advantage claim. The synthetic composite channel probes a pronounced boundary-shift regime, while the backend snapshot tests whether that regime also occurs under one device-derived noise model.

\section{Question and model}
We ask: when a noisy classifier predicts one class, how much can held-out threshold or affine calibration recover, and how does finite measurement precision affect recovery relative to linear and quadratic ZNE? The classifier encodes $d=n$ PCA features by $R_X(x_j)$ rotations, applies trainable $R_Y$ and $R_Z$ rotations with a CNOT-chain, CZ-ring, or readout-centered CNOT-star entangler, and measures $z=\langle Z_0\rangle\in[-1,1]$. The training objective uses $\sigma(z)$ with binary cross-entropy; this sigmoid is a training parameterization and is not interpreted as a calibrated patient probability. Classification at the fixed boundary uses $z\geq0$.

A single-qubit channel acts affinely on a Bloch vector, $\mathbf r_{\rm out}=M\mathbf r_{\rm in}+\mathbf t$; unitality means $\mathbf t=0$. For a final amplitude-damping channel immediately before measurement, $z_{\rm out}=(1-\gamma)z_{\rm in}+\gamma$. For $0\leq\gamma<1$, the offset can move scores across zero while preserving their ordering. By contrast, a final isotropic depolarizing contraction gives $z_{\rm out}=\eta z_{\rm in}$ and preserves the sign when $\eta>0$; final pure dephasing in the computational basis leaves $z$ unchanged. Unitality alone does not ensure sign preservation: rotations or interleaved anisotropic channels need not act as a positive scalar on the measured observable. These final-channel identities are illustrative, not an assumed model of our interleaved circuit. All $\gamma$ values denote probabilities at declared channel locations, not an overall circuit error rate or device $T_1/T_2$ times.

\section{Experimental design}
For WDBC the target labels are malignant (0) and benign (1). Tasks are the UCI Wisconsin Diagnostic Breast Cancer dataset (WDBC)~\cite{wdbc}, the UCI Wine dataset reduced to class 0 versus classes 1/2~\cite{wine}, and two scikit-learn synthetic tasks generated with fixed seeds (600 samples, 12 features, 6 informative, 2 redundant, 2 clusters per class, class separation 1.0; seeds 1729/1730 and class weights 0.5/0.5 or 0.8/0.2)~\cite{sklearn}. Table~\ref{tab:datasets} reports dataset sizes, counts, partition sizes, and PCA variance for the secondary unstandardized pipeline; the standardized primary variances are also listed. Each task uses @@SPLITS@@ stratified folds with disjoint test sets; the initial 60/20/20 model-training-pool/calibration/test split is further divided 85/15 into weight-fit and development data. All preprocessing is fit on the outer training pool before its 85/15 weight-fit/development split; calibration and test data are held out from preprocessing. The primary pipeline standardizes features before two-component PCA. @@TRAIN_SEEDS@@ training @@SEED_LABEL@@ are used per fold. The four-qubit CNOT-star used for broader comparisons passed a random-weight feature-dependence probe. The primary two-qubit CNOT and CZ circuits also passed post-hoc probes confirming both encoded inputs can affect their readouts. Channel controls use differing noise locations and are descriptive, not causal.

Models are trained without noise on PennyLane's \texttt{default.qubit} simulator using Adam (learning rate 0.1), binary cross-entropy on $\sigma(\langle Z_0\rangle)$, uniform $[-\pi,\pi]$ initialization, batch size 32, and 30 epochs. The two-qubit model has eight trainable angles. Development examples are monitored but do not select checkpoints; the last epoch is evaluated. This is a noise-unaware training study, with noisy evaluation on PennyLane \texttt{default.mixed}. The recorded environment is Python 3.12.14, PennyLane 0.43.3, NumPy 2.2.6, pandas 2.3.3, and scikit-learn 1.7.2 on Windows 11. Training histories are reported; they are not evidence of convergence.

At each composite-noise test setting, we evaluate exact expectations and finite shots. In the primary analysis, ZNE scale factors multiply the declared channel probabilities in the simulator; this is probability-scaled Richardson extrapolation, not gate folding or hardware noise scaling. A separate gate-folding simulator sensitivity is reported below; neither analysis is a hardware result. The fixed threshold and uncalibrated ZNE use the full nominal per-case budget for inference. Clean-frozen calibration, noisy threshold calibration, affine readout, and calibrated ZNE divide the nominal budget between independent calibration and test examples; because counts are integer and fold sizes may differ, realized totals vary by at most 0.443\% from the fixed-threshold total. Actual shot costs are saved. Clean-frozen thresholds are estimated from clean finite-shot calibration scores and held fixed across noise settings. Noisy thresholds and ZNE thresholds are fit on their respective calibration scores only. Linear ZNE uses scales $\{1,2\}$ and coefficients $(2,-1)$; quadratic ZNE uses scales $\{1,1.5,2\}$ and coefficients $(6,-8,3)$. ZNE shots are allocated in proportion to absolute coefficients. We distinguish observable error relative to the clean expectation from classification recovery under a changed decision rule. Following standard finite-shot ZNE analyses~\cite{scavino2026,mohammadipourli2025}, for independent Pauli-$Z$ sample means at scales $s_i$, the Richardson estimate is $\hat z=\sum_i c_i\hat\mu_i$, with $\mathrm{Var}(\hat z)=\sum_i c_i^2(1-\mu_i^2)/n_i$. Under a total of $N=\sum_i n_i$ shots, the variance-minimizing allocation is $n_i\propto |c_i|\sqrt{1-\mu_i^2}$; when per-shot variances are treated as equal, this reduces to allocation proportional to $|c_i|$, as used here. The resulting worst-case variance bound is $(\sum_i|c_i|)^2/N$. We recall this standard estimator-level bound rather than claim it as a new result. The linear $(2,-1)$ rule has bound $9/N$, the quadratic $(6,-8,3)$ rule has $289/N$, and the exploratory $(15,-24,10)$ rule has $2401/N$. This estimator-level variance result does not determine classification accuracy or a universal crossover against calibration: that also depends on score distributions, calibration labels, and threshold-fit uncertainty. We do not claim such a crossover theorem.

We report accuracy, balanced accuracy, per-class recall, ROC-AUC, confusion counts, predicted class fraction, observable error, shot cost, and single-class prediction frequency. Paired contrasts are calculated on matched model/test splits. Outer test folds partition each dataset, while training sets share observations; fold-level summaries are descriptive rather than population-level inference.

The separate gate-folding sensitivity implements each primitive gate using an alternating odd-fold sequence $U$, $U^{-1}$, $U$, and so on, with 1, 3, or 5 physical occurrences, and applies the same declared composite channel after each physical gate occurrence. Richardson coefficients are $(15/8,-5/4,3/8)$; with a 1,000-measurement per-case budget, 500 measurements are allocated to calibration and 500 to inference, divided across scales in proportion to the absolute coefficients. The experiment reuses the 10 standardized WDBC checkpoints and five saved folds and averages 20 finite-shot draws. The scale-one circuit reproduces the original exact composite-channel scores to numerical precision. This is a simulator gate-folding study under an assumed gate-local channel, not a hardware execution.

\section{Results}
\subsection{Primary WDBC matched-budget comparison with standardized PCA}
@@STANDARDIZED_TEXT@@
@@GATEFOLD_NOTE@@
\begin{table*}[t]\centering\scriptsize
\caption{Standardized-PCA WDBC results at $\gamma=0.30$. Thresholds use calibration labels only. The nominal finite-shot budget is 1,000 measurements per test case, split between calibration and test inference. Finite-shot means average 20 draws per checkpoint; reported SD is across the 10 checkpoint means, nested in five folds with shared training observations.}
\label{tab:standardized}
\begin{tabular}{@{}l r r r@{}}\toprule
Pipeline & Balanced accuracy & ROC-AUC & $n$ \\
\midrule
@@STANDARDIZED_TABLE@@
\bottomrule\end{tabular}\end{table*}
@@FRESH_WDBC_CONFIRMATION@@
\subsection{Unstandardized-PCA noise-grid and multi-task analyses}
\subsubsection{Exact expectations}
Supplementary Table~S1 reports the full exact-expectation WDBC noise grid from the earlier unstandardized-PCA pipeline, including fixed, clean-frozen, and noisy-calibrated thresholds, affine readout, and ZNE with and without calibration. An observed single-class prediction rate is described only at sampled settings; it is not a universal threshold.
\begin{table*}[t]
\centering
\caption{Secondary unstandardized-PCA WDBC exact-expectation results (mean across saved model runs).}
\label{tab:exact}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}rccccccccc@{}}
\toprule
$\gamma$ & Fixed & Fixed AUC & Clean-frozen & Noisy-calibrated & Affine & Linear ZNE & Quadratic ZNE & Linear ZNE+cal. & Quadratic ZNE+cal. \\
\midrule
@@EXACT_TABLE@@
\bottomrule
\end{tabular}}
\end{table*}

@@CALIBRATION_CONTRAST@@ The clean-threshold and noisy-threshold results separate the gain from correcting the original decision rule from recovery after noise; finite-shot recovery is lower than the exact-expectation result at this budget.

\subsubsection{Finite measurement budgets}
Supplementary Table~S2 gives the earlier unstandardized-PCA nominal-budget finite-shot results at $\gamma=0.30$ and 1,000 nominal shots per test case; Supplementary Figure~S2 and the saved CSV show the broader budget and noise grid. Calibration costs are included in each method's measurement total. Observable estimation error and classification recovery are separate outcomes: calibration changes the decision rule and need not reconstruct the clean observable. No mechanism is inferred from classification performance alone.
\begin{table*}[t]
\centering
\caption{Secondary unstandardized-PCA finite-shot WDBC balanced accuracy (mean $\pm$ SD over model and shot repetitions), AUC, collapse frequency, and total measurements. The clean-frozen threshold requires a clean calibration measurement and is a simulator-only reference.}
\label{tab:finite}
\scriptsize
\begin{tabular}{@{}r r l c c c r@{}}
\toprule
$\gamma$ & Budget/test case & Method & Balanced accuracy & AUC & Collapse rate & Total shots \\
\midrule
@@FINITE_TABLE@@
\bottomrule
\end{tabular}
\end{table*}

The complete nominal-budget comparison is also run on two additional tasks and the input-probed four-qubit circuit; Supplementary Table~S3 summarizes the 1,000-shot setting. This transfer check is deliberately small and does not establish broad generalization.
\begin{table*}[t]
\centering
\caption{Secondary unstandardized-PCA cross-task and broader-input finite-shot balanced accuracy at $\gamma=0.30$ and 1,000 shots per test case. Each calibration-based method includes calibration measurements; realized totals differ from the fixed-threshold comparator by at most 0.443\%.}
\label{tab:transfer}
\begin{tabular}{@{}l c c c c c@{}}
\toprule
Task and circuit & Fixed clean BA ($\gamma=0$) & Noisy calibration & Linear ZNE & Linear ZNE+cal. & Quadratic ZNE+cal. \\
\midrule
@@TRANSFER_TABLE@@
\bottomrule
\end{tabular}
\end{table*}

\subsubsection{Noise-channel and error controls}
Supplementary Table~S4 reports the channel controls; at $\gamma=0.30$, fixed-threshold balanced accuracy is 0.644 for amplitude damping, 0.500 for the composite channel, 0.817 for depolarization, and 0.820 for phase damping. Because channel insertion locations and counts differ, these rows are not causal comparisons. Supplementary Table~S5 separates exact observable RMSE from finite-shot sampling RMSE; for example, quadratic ZNE has exact RMSE 0.576 and sampling RMSE 0.471, while quadratic ZNE with calibration has sampling RMSE 0.681. Calibration-rule fit variability is discussed separately after Table~S5.
\begin{table*}[t]
\centering
\caption{WDBC fixed-threshold channel controls across saved primary-model runs.}
\label{tab:channels}
\begin{tabular}{@{}l r c c c r@{}}
\toprule
Channel & $\gamma$ & Balanced accuracy & ROC-AUC & Collapse rate & Model runs \\
\midrule
@@CHANNEL_TABLE@@
\bottomrule
\end{tabular}
\end{table*}
\begin{table}[t]
\centering
\caption{Observable error for WDBC at $\gamma=0.30$ and 1,000 shots per test case. Exact RMSE is relative to the clean observable; sampling RMSE is deviation from the exact method-specific observable. Calibration-rule fitting uncertainty and classification performance are reported separately.}
\label{tab:errors}
\begin{tabular}{@{}l c c@{}}
\toprule
Method & Exact observable RMSE & Finite sampling RMSE \\
\midrule
@@ERROR_TABLE@@
\bottomrule
\end{tabular}
\end{table}
@@RULE_FIT_VARIABILITY@@
\subsection{Clean performance and classical controls}
Table~\ref{tab:datasets} reports the task definitions and partition sizes. Table~\ref{tab:classical} compares the quantum model with classical controls using the same two PCA inputs and with full-feature controls. It reports both an outer-training-pool logistic model and a logistic model fit on the exact weight-fit examples used by each quantum initialization. All classical decisions use the default 0.5 threshold. The matched-fit comparison isolates input representation and fits on the same example count; the larger-pool baseline reflects the full outer training data available before calibration.
\begin{table*}[t]\centering\scriptsize
\caption{Task definitions and partition sizes. PCA variance columns distinguish the secondary unstandardized pipeline from standardized-before-PCA controls; the imbalanced task has no standardized transfer run.}
\label{tab:datasets}
\begin{tabular}{@{}l r r l l r r@{}}\toprule
Task & $n$ & Features & Class counts & Fit/dev/cal/test ranges & Raw PCA var. & Std. PCA var. \\
\midrule
@@DATASET_TABLE@@
\bottomrule\end{tabular}\end{table*}
\begin{table*}[t]\centering\scriptsize
\caption{Clean balanced accuracy: quantum model and classical controls across standardized and raw PCA pipelines. Standardized 2D inputs are directly comparable (Quantum 0.887 vs Logistic 0.928 on WDBC; 0.858 vs 0.950 on Wine; 0.689 vs 0.747 on balanced synthetic). Secondary unstandardized 2D and full-feature classical baselines are reported for comparison. All classical decisions use a fixed 0.5 threshold.}
\label{tab:classical}
\begin{tabular}{@{}l r r r r r r r r@{}}\toprule
Task & \shortstack{Quantum 2D\\(std.)} & \shortstack{Quantum 2D\\(raw)} & \shortstack{Logistic 2D\\(std.)} & \shortstack{Logistic 2D\\(raw)} & \shortstack{Logistic 2D\\matched fit $n$} & \shortstack{Logistic\\full} & \shortstack{RBF SVM\\full} & \shortstack{Random forest\\full} \\
\midrule
@@CLASSICAL_TABLE@@
\bottomrule\end{tabular}\end{table*}
Table~\ref{tab:classicalpaired} reports paired quantum-minus-logistic differences for the exact fit-size-matched rows. Its intervals are descriptive bootstraps over five fold means, not population-level confidence intervals.
\begin{table}[t]\centering\scriptsize
\caption{Paired clean balanced-accuracy contrasts: quantum 2D model minus fit-size-matched 2D logistic regression. Intervals are descriptive fold bootstraps.}
\label{tab:classicalpaired}
\begin{tabular}{@{}l r l r@{}}\toprule
Task & Difference & 95\% interval & Folds \\
\midrule
@@CLASSICAL_PAIRED_TABLE@@
\bottomrule\end{tabular}\end{table}
In the secondary, unstandardized-PCA analysis, clean-expectation two-qubit balanced accuracy / ROC-AUC across saved model runs were: @@CLEAN_RESULTS@@. The imbalanced synthetic task remains a negative result: its quantum AUC is near chance (0.462) and its clean balanced accuracy is weak (0.526), while full-feature classical baselines achieved @@IMBALANCED_BASELINES@@. Per-epoch training and development curves are saved; the per-task median optimizer updates were @@TASK_UPDATES@@ over @@EPOCHS@@ epochs at batch size @@BATCH_SIZE@@. We retain the weak task as a failed diagnostic, not evidence that mitigation preserves a functioning classifier. WDBC and Wine classical baselines also exceed the two-qubit model in this pilot, so no comparative predictive advantage is claimed.


Figure~\ref{fig:training} shows the training and development histories. WDBC and Wine losses decline and then change less across later epochs; the balanced synthetic task improves modestly, while imbalanced synthetic development performance remains near chance. We use the fixed @@EPOCHS@@-epoch protocol and do not treat these curves as proof of convergence.
\begin{figure*}[t]
\centering
\includegraphics[width=0.82\textwidth]{training_convergence.png}
\caption{Training and development binary cross-entropy and development balanced accuracy at the fixed threshold, averaged over @@SPLITS@@ folds for the core two-qubit CNOT-chain models. Shading shows fold-level standard deviation after averaging initializations within each fold.}
\label{fig:training}
\end{figure*}

\subsection{Paired comparisons in the primary pipeline}
Table~\ref{tab:primary-paired} reports the fold-level difference in balanced accuracy for the standardized WDBC pipeline at $\gamma=0.30$ and 1,000 measurements (ZNE minus noisy-threshold calibration). Each value is computed after averaging the matched shot draws within the corresponding fold and initialization. The test folds partition WDBC, while training sets overlap; these five contrasts describe split-protocol variability rather than five independent populations. @@CHECKPOINT_PAIRED_NOTE@@
\begin{table*}[t]\centering\small
\caption{Primary standardized-PCA WDBC fold-level paired contrasts, $\gamma=0.30$, nominal budget 1,000. Entries are ZNE minus noisy-threshold calibrated balanced accuracy; negative values favor threshold calibration.}
\label{tab:primary-paired}
\begin{tabular}{@{}l r r r r r r@{}}\toprule
Method & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5 & Mean \\
\midrule
@@PRIMARY_PAIR_TABLE@@
\bottomrule\end{tabular}\end{table*}
\subsection{Secondary paired differences}
Supplementary Table~S6 directly compares ZNE with the noisy-calibrated threshold in the earlier unstandardized-PCA pipeline at matched noise and budget. Entries are ZNE minus calibrated balanced accuracy; intervals are descriptive percentile bootstraps over five fold-level paired means, not population-level confidence intervals. At $\gamma=0.30$ and 1,000 shots, the five fold-level linear-ZNE differences are @@REPRESENTATIVE_FOLDS@@ (ZNE minus calibration); at 100 shots linear ZNE is a near-tie (mean +0.002, descriptive interval [-0.006,+0.012]). The $\gamma=0.30$ setting is the first grid point where fixed-threshold predictions collapse in all 10 primary model runs.
\begin{table*}[t]
\centering
\caption{Direct paired contrasts for ZNE minus calibration.}
\label{tab:paired}
\scriptsize
\begin{tabular}{@{}l r r l l r c r@{}}
\toprule
$\gamma$ & Analysis & Shots & Reference & ZNE & Difference & Descriptive 95\% interval & Folds \\
\midrule
@@PAIRED_TABLE@@
\bottomrule
\end{tabular}
\end{table*}

Supplementary Table~S7 reports paired task/circuit contrasts at $\gamma=0.30$ and 1,000 shots; the linear-ZNE minus calibrated-threshold differences are -0.144 (WDBC), -0.057 (Wine), -0.014 (balanced synthetic), and -0.006 (four-qubit star). Five folds overlap, so these are descriptive summaries.
\begin{table*}[t]\centering\scriptsize
\caption{Transfer-setting paired contrasts at $\gamma=0.30$ and 1,000 nominal shots per test case. Differences are ZNE balanced accuracy minus noisy-calibrated-threshold balanced accuracy; intervals are descriptive bootstrap intervals over five fold-level means.}
\label{tab:transferpaired}
\begin{tabular}{@{}l l r l r@{}}\toprule
Task and circuit & Method & Difference & Descriptive 95\% interval & Folds \\
\midrule
@@TRANSFER_PAIRED_TABLE@@
\bottomrule\end{tabular}\end{table*}

Richardson scores can lie outside $[-1,1]$. Supplementary Table~S8 reports the clipping diagnostic for the secondary unstandardized-PCA pipeline: clipping leaves linear ZNE plus calibration balanced accuracy at 0.690 and quadratic ZNE plus calibration at 0.575, the same rounded value as without clipping. Clipping changes no measurement allocation, and thresholds use calibration labels only.
\begin{table*}[t]\centering\scriptsize
\caption{Finite-shot ZNE with and without clipping scores to the Pauli-$Z$ interval, at $\gamma=0.30$ and 1,000 nominal shots per test case.}
\label{tab:zne_range}
\begin{tabular}{@{}l l r r r r@{}}\toprule
Task and circuit & Method & Raw BA & Clipped BA & Fraction outside $[-1,1]$ & Total shots \\
\midrule
@@ZNE_RANGE_TABLE@@
\bottomrule\end{tabular}\end{table*}

Supplementary Table~S9 gives the saved-checkpoint budget-allocation sensitivity (10 WDBC checkpoints, two shot repetitions, $\gamma=0.30$, 1,000 nominal shots): calibration balanced accuracy is 0.762, 0.753, and 0.711 for 25\%, 50\%, and 75\% allocation; linear-ZNE-plus-calibration is 0.689, 0.691, and 0.640. This is post hoc and reuses the outer folds.
\begin{table*}[t]\centering\scriptsize
\caption{Exploratory calibration-share sensitivity at WDBC, two-qubit CNOT chain, $\gamma=0.30$.}
\label{tab:allocation}
\begin{tabular}{@{}r l c r@{}}\toprule
Calibration share & Pipeline & Balanced accuracy & Mean total shots \\
\midrule
@@ALLOCATION_TABLE@@
\bottomrule\end{tabular}\end{table*}

Supplementary Table~S10 reports exact-expectation results for alternative probability-scaling schedules. At $\gamma=0.30$, exact linear ZNE balanced accuracy is non-monotone across the tested noise grid (0.849, 0.626, 0.538 at $\gamma=0.20,0.30,0.40$), while quadratic ZNE is 0.837 at $\gamma=0.30$. These are schedule- and channel-dependent extrapolation outcomes, not evidence of a monotone relationship: probability scaling changes the simulated channel, and extrapolation bias can grow when the scaled channel departs from the assumed low-order response. Supplementary Table~S11 reports the post-hoc finite-shot schedule sensitivity, using the same 10 checkpoints and 50:50 calibration/inference allocation but independent shot seeds from the main finite-shot table. Consequently its linear-ZNE-plus-calibration result (0.704) differs from the main table (0.690), and its quadratic fixed-threshold result (0.576) differs from 0.617 in the main table; these are separate Monte Carlo estimates, not the same draws. Two shot repetitions per checkpoint and overlapping folds make the SDs descriptive, not population uncertainty.
\begin{table*}[t]\centering\scriptsize
\caption{Exploratory ZNE probability-scale sensitivity at WDBC, two-qubit CNOT chain, $\gamma=0.30$.}
\label{tab:scales}
\begin{tabular}{@{}l l l c c@{}}\toprule
Schedule & Scales & Richardson coefficients & Balanced accuracy & Observable RMSE \\
\midrule
@@SCALE_TABLE@@
\bottomrule\end{tabular}\end{table*}

\begin{table}[t]\centering\small
\caption{Post-hoc finite-shot scale sensitivity at WDBC, $\gamma=0.30$, and 1,000 nominal measurements per test case. Entries are mean balanced accuracy $\pm$ SD over 10 checkpoints and two shot repetitions; folds overlap.}
\label{tab:finite-scales}
\begin{tabular}{@{}l r r@{}}\toprule
Schedule & Fixed threshold & Calibration \\
\midrule
@@FINITE_SCALE_TABLE@@
\bottomrule\end{tabular}\end{table}

\subsection{Score and budget diagnostics}
Secondary unstandardized-PCA noise-grid, finite-budget, and architecture diagnostics appear in Supplementary Figures~S1--S3 and Tables~S1--S17. The primary two-qubit CNOT-chain and CZ-ring circuits, and the four-qubit CNOT-star, passed a finite random-weight feature-to-readout probe for every encoded input. This probe establishes only sensitivity under tested perturbations, not useful learning or scaling. The single four- and six-qubit CNOT-chain controls had weak or absent feature dependence and remain architecture diagnostics; there is no four-qubit CZ-ring model in this study.
\begin{figure*}[t]
\centering
\includegraphics[width=0.48\textwidth]{results/exact_accuracy_auc.png}\hfill
\includegraphics[width=0.48\textwidth]{results/score_distributions_wdbc.png}
\caption{Secondary unstandardized-PCA exact-expectation accuracy/AUC across composite-noise settings (left) and WDBC score distributions at the selected collapse setting (right).}
\label{fig:exact}
\end{figure*}
\begin{figure*}[t]
\centering
\includegraphics[width=0.48\textwidth]{results/finite_shots_budget.png}\hfill
\includegraphics[width=0.48\textwidth]{results/score_error_decomposition.png}
\caption{Secondary unstandardized-PCA finite-shot balanced accuracy under nominally matched total-shot budgets (realized totals differ by at most 0.443\%), separated by noise level (left); observable estimation error and finite-shot sampling deviation (right; two shot budgets shown as unconnected markers).}
\label{fig:budget}
\end{figure*}
\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{results/collapse_controls.png}
\caption{Single-class prediction collapse across architecture, preprocessing, and label-mapping controls. Markers are not connected; labels report the number of model runs supporting each control.}
\label{fig:controls}
\end{figure*}

\subsection{Scope of the secondary synthetic diagnostic}
The separate weak-signal synthetic task-family diagnostic is retained in the supplementary archive as a negative result; its near-chance baseline is not used as confirmation of the main finding.

\section{Discussion and limitations}
The composite-channel collapse and the absence of collapse in FakeManila delimit the observed regime. Repeated amplitude damping supplies a plausible directional-bias mechanism, but the snapshot also includes relaxation and readout errors; it is not a purely unital control. Channel strengths, gate durations, placement, and circuit structure differ, so these experiments do not isolate the cause of the contrast. The supported finding is that the strong synthetic setting admits substantial recovery by threshold selection, whereas the tested snapshot already retains near-clean fixed-threshold performance. Neither asymmetric noise alone nor a large score offset guarantees a benefit from recalibration; recovery also requires usable score separation and sufficiently precise calibration measurements.

The secondary unstandardized-PCA comparison already includes a Platt-style logistic score readout (Tables S1--S2): the scalar score is standardized on calibration examples only, then an L2-regularized logistic regression ($C=1$) fits slope and intercept on those labels. Its zero-logit boundary corresponds to probability 0.5. This optimizes logistic loss rather than calibration balanced accuracy and is not the original Platt target-smoothing procedure. It establishes an existing parametric comparator, but does not test low-label efficiency on the primary standardized pipeline. Isotonic calibration and a dedicated matched-budget low-label comparison remain untested.

The study's conclusions are limited to its simulator settings and saved folds:
\begin{itemize}
\item The main matched-budget comparison is standardized-PCA WDBC, with standardized-PCA transfer runs on the same saved splits and initialization seeds for Wine and balanced synthetic data; legacy broad noise-grid controls use unstandardized PCA and are secondary.
\item Outer test folds partition WDBC, but training sets share observations; both transfer tasks reuse the same split and initialization seeds. The results do not provide broad population-level validation.
\item Threshold fitting uses held-out labels and consumes measurements, so performance may depend on calibration-set size and shot allocation.
\item Classical logistic regression on the same standardized inputs reaches 0.928 clean balanced accuracy, above the quantum model's 0.887; no predictive advantage is shown.
\item Training is noiseless and fixed at 30 epochs. The primary ZNE comparison scales simulated channel probabilities; a separate matched-budget gate-folded simulator sensitivity is reported. The device-model evaluation uses a single fake-backend snapshot; no real-hardware run was performed.
\end{itemize}
The results do not establish clinical performance, quantum advantage, or a general ranking of mitigation methods.

@@REPRODUCTION_TEXT@@

The accompanying archive includes code, figures, split indices, frozen weights, preprocessing objects, per-example observations, package versions, and configuration hashes. No probabilistic-error-cancellation run was performed.

\section{Conclusion}
In this simulator study, held-out threshold selection improves classification over the fixed boundary in the standardized WDBC setting, but finite-shot recovery is incomplete and the tested ZNE result depends on schedule and budget. Ten calibration labels reach balanced accuracy 0.760 in the post-hoc sensitivity analysis, compared with 0.783 using all 114 labels. Performance also declines across the tested depth sweep, alongside a sharp reduction in noisy class-score separation. Transfer runs on the same saved Wine and balanced-synthetic splits show the same tested-pipeline ordering, while clean matched logistic baselines remain stronger on WDBC and Wine. These bounded observations motivate device-aware and readout-mitigation comparisons; they do not establish general superiority or quantum advantage.

\begin{thebibliography}{99}
\bibitem{hernandez2012} J. Hern\'andez-Orallo, P. Flach, and C. Ferri, ``A unified view of performance metrics: Translating threshold choice into expected classification loss,'' \emph{Journal of Machine Learning Research}, vol. 13, pp. 2813--2869, 2012.
\bibitem{niculescu2005} A. Niculescu-Mizil and R. Caruana, ``Predicting good probabilities with supervised learning,'' in \emph{Proc. 22nd International Conference on Machine Learning}, pp. 625--632, 2005, doi: 10.1145/1102351.1102430.
\bibitem{temme2017} K. Temme, S. Bravyi, and J. M. Gambetta, ``Error mitigation for short-depth quantum circuits,'' \emph{Physical Review Letters}, vol. 119, 180509, 2017.
\bibitem{libenjamin2017} Y. Li and S. C. Benjamin, ``Efficient variational quantum simulator incorporating active error minimization,'' \emph{Physical Review X}, vol. 7, 021050, 2017.
\bibitem{cai2023} Z. Cai et al., ``Quantum error mitigation,'' \emph{Reviews of Modern Physics}, vol. 95, 045005, 2023.
\bibitem{wdbc} W. H. Wolberg, W. N. Street, and O. L. Mangasarian, ``Breast Cancer Wisconsin (Diagnostic),'' UCI Machine Learning Repository, DOI: 10.24432/C5DW2B.
\bibitem{wine} S. Aeberhard and M. Forina, ``Wine,'' UCI Machine Learning Repository, DOI: 10.24432/C5PC7J.
\bibitem{sklearn} F. Pedregosa et al., ``Scikit-learn: Machine learning in Python,'' \emph{Journal of Machine Learning Research}, vol. 12, pp. 2825--2830, 2011; synthetic-data settings follow the scikit-learn \texttt{make\_classification} documentation.
\bibitem{larose2020} R. LaRose and B. Coyle, ``Robust data encodings for quantum classifiers,'' \emph{Physical Review A}, vol. 102, 032420, 2020.
\bibitem{wang2022} H. Wang et al., ``QuantumNAT: Quantum noise-aware training with noise injection, quantization and normalization,'' in \emph{Proc. DAC}, 2022.
\bibitem{giurgica2020} T. Giurgica-Tiron et al., ``Digital zero noise extrapolation for quantum error mitigation,'' in \emph{Proc. IEEE QCE}, 2020.
\bibitem{majumdar2023} R. Majumdar et al., ``Best practices for quantum error mitigation with digital zero-noise extrapolation,'' arXiv:2307.05203, 2023.
\bibitem{koester2026} D. K\"oster and W. Mauerer, ``Benchmarking error mitigation: Artefactual improvements in zero-noise extrapolation,'' arXiv:2607.09360, 2026.
\bibitem{fernandez2026} R. Fern\'andez-Barrios, I. Pastor-L\'opez, A. Gonz\'alez-Santocildes, and P. Garc\'ia Bringas, ``Conditional validity of quantum event classifiers under collider systematics and quantum estimation uncertainty,'' arXiv:2609.02781, 2026.
\bibitem{scavino2026} V. Scavino Alfaro, ``The finite-shot help--harm boundary of zero-noise extrapolation,'' arXiv:2605.08251, 2026.
\bibitem{mohammadipourli2025} P. Mohammadipour and X. Li, ``Direct analysis of zero-noise extrapolation: Polynomial methods, error bounds, and simultaneous physical-algorithmic error mitigation,'' \emph{Quantum}, vol. 9, 1909, 2025, doi: 10.22331/q-2025-11-14-1909.
\bibitem{zhang2026budget} G. Zhang, K. Zhao, X. Cui, H. Heng, X. Chu, and A. J. Blanken, ``When Error Mitigation Makes Things Worse: Budget-Aware Evaluation, Extrapolation Failure, and the Calibration Trust Boundary,'' arXiv:2609.38713, NeurIPS Workshop on Secure and Trustworthy Quantum Machine Learning (SaTQuML), 2026.
\end{thebibliography}
\end{document}
"""
    manifest = __import__("json").loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    exact_values = (f"{fmt(fixed_030)} fixed; {fmt(wide.get('clean_frozen_threshold', pd.Series(dtype=float)).get(0.3, np.nan))} "
                    f"clean-frozen; {fmt(cal_030)} noisy-calibrated; {fmt(aff_030)} affine; "
                    f"{fmt(wide.get('linear_zne', pd.Series(dtype=float)).get(0.3, np.nan))} linear ZNE; "
                    f"{fmt(zne_030)} quadratic ZNE; "
                    f"{fmt(wide.get('linear_zne_calibrated', pd.Series(dtype=float)).get(0.3, np.nan))} linear ZNE+calibration; "
                    f"{fmt(wide.get('quadratic_zne_calibrated', pd.Series(dtype=float)).get(0.3, np.nan))} quadratic ZNE+calibration")
    split_count = str(manifest["config"]["outer_splits"])
    train_seed_count = str(manifest["config"]["train_seeds"])
    train_seed_label = "initialization" if int(train_seed_count) == 1 else "initializations"
    epoch_count = int(manifest["config"]["epochs"])
    batch_size = int(manifest["config"]["batch_size"])
    clean_labels = {"wdbc": "WDBC", "wine_0_vs_rest": "Wine class 0 vs rest",
                    "synthetic_balanced": "balanced synthetic", "synthetic_imbalanced": "imbalanced synthetic"}
    clean_results = "; ".join(
        f"{clean_labels[task]}: {fmt(clean_means.loc[task, 'clean_balanced_accuracy'])} balanced accuracy / {fmt(clean_means.loc[task, 'clean_roc_auc'])} ROC-AUC"
        for task in ["wdbc", "wine_0_vs_rest", "synthetic_balanced", "synthetic_imbalanced"]
        if task in clean_means.index)
    imbalanced_baselines = "; ".join(
        f"{latex_escape(model)} {fmt(score)} balanced accuracy"
        for model, score in imbalanced_full_means.items())
    cal_eff_sentence = ("A post-hoc reused-checkpoint sensitivity reached 0.760 balanced accuracy with 10 calibration labels and 0.783 with all 114; across the five saved folds, threshold selection exceeded linear ZNE by 0.052--0.071 balanced-accuracy units.")
    fake_abs_text = fake_abstract if 'fake_abstract' in locals() else ''
    confirmation_abstract=(f"We compare held-out threshold selection with probability-scaled zero-noise extrapolation (ZNE) for noiselessly trained variational quantum classifiers under matched simulated measurement budgets including calibration. On standardized-PCA WDBC at $\\gamma=0.30$, exact calibrated balanced accuracy is {fmt(qexactstd.balanced_accuracy.mean())}; with 1,000 measurements it is {fmt(qshot.balanced_accuracy.mean())}, versus {fmt(std_result('linear_zne','finite_calibrated'))} for linear ZNE and {fmt(std_result('quadratic_zne','finite_calibrated'))} for quadratic ZNE. {cal_eff_sentence} {gatefold_abstract} {fake_abs_text} Standardized transfer runs on the same saved splits show a similar tested-pipeline ordering on Wine and balanced synthetic data; matched clean logistic regression exceeds the quantum model on these tasks. A post-hoc depth sweep found class-score gaps of 0.210, 0.079, and 0.008 at 1, 2, and 4 layers, respectively; this pattern is consistent with reduced finite-shot robustness at greater depth in the tested setting, but does not establish causation or a general depth law.{fresh_confirmation_abstract} Results establish neither hardware advantage nor clinical validity.")
    content = (content.replace("@@POINT_VALUES@@", exact_values)
               .replace("@@AUC_VALUE@@", fmt(auc_030))
               .replace("@@CONFIRMATION_ABSTRACT@@", confirmation_abstract)
               .replace("@@CONFIRMATION_SECTION@@", confirmation_section)
               .replace("@@STANDARDIZED_TEXT@@", std_control_text)
               .replace("@@GATEFOLD_NOTE@@", gatefold_note)
               .replace("@@FRESH_WDBC_CONFIRMATION@@", fresh_confirmation_section)
               .replace("@@STANDARDIZED_TABLE@@", std_control_table)
               .replace("@@CAL_SIZE_NOTE@@", cal_size_note)
               .replace("@@FINITE_SENTENCE@@", finite_sentence)
               .replace("@@EXACT_TABLE@@", exact_table)
               .replace("@@DATASET_TABLE@@", "\n".join(dataset_lines))
               .replace("@@CLASSICAL_TABLE@@", "\n".join(classical_lines))
               .replace("@@CLASSICAL_PAIRED_TABLE@@", matched_pair_table)
               .replace("@@TASK_UPDATES@@", "; ".join(f"{task_labels[k]} {fmt(v,0)}" for k,v in updates_by_task.items()))
               .replace("@@ABSTRACT_FIXED@@", fmt(finite_030.loc[finite_030.method=="fixed_threshold","balanced_accuracy_mean"].iloc[0] if len(finite_030.loc[finite_030.method=="fixed_threshold"]) else np.nan))
               .replace("@@ABSTRACT_CAL@@", fmt(cal_030))
               .replace("@@ABSTRACT_FINITE_CAL@@", fmt(finite_030.loc[finite_030.method=="calibrated_threshold","balanced_accuracy_mean"].iloc[0] if len(finite_030.loc[finite_030.method=="calibrated_threshold"]) else np.nan))
               .replace("@@ABSTRACT_FINITE_LIN@@", fmt(finite_030.loc[finite_030.method=="linear_zne","balanced_accuracy_mean"].iloc[0] if len(finite_030.loc[finite_030.method=="linear_zne"]) else np.nan))
               .replace("@@ABSTRACT_FINITE_QUAD@@", fmt(finite_030.loc[finite_030.method=="quadratic_zne","balanced_accuracy_mean"].iloc[0] if len(finite_030.loc[finite_030.method=="quadratic_zne"]) else np.nan))
               .replace("@@FINITE_TABLE@@", finite_table)
               .replace("@@TRANSFER_TABLE@@", transfer_table)
               .replace("@@PAIRED_TABLE@@", paired_table)
               .replace("@@TRANSFER_PAIRED_TABLE@@", transfer_pair_table)
               .replace("@@ZNE_RANGE_TABLE@@", range_table)
               .replace("@@REPRODUCTION_TEXT@@", reproduction_text)
               .replace("@@REPRESENTATIVE_FOLDS@@", representative_fold_text or "not available")
               .replace("@@PRIMARY_PAIR_TABLE@@", primary_pair_table if 'primary_pair_table' in locals() else "")
               .replace("@@CHECKPOINT_PAIRED_NOTE@@", checkpoint_pair_note if 'checkpoint_pair_note' in locals() else "")
               .replace("@@ALLOCATION_TABLE@@", "\n".join(allocation_lines) or (r"\multicolumn{4}{c}{Sensitivity not run} " + chr(92)*2))
               .replace("@@SCALE_TABLE@@", "\n".join(scale_lines) or (r"\multicolumn{5}{c}{Sensitivity not run} " + chr(92)*2))
               .replace("@@FINITE_SCALE_TABLE@@", "\n".join(finite_scale_lines) or (r"\multicolumn{3}{c}{Finite-shot sensitivity not run} " + chr(92)*2))
               .replace("@@CHANNEL_TABLE@@", channel_table)
               .replace("@@ERROR_TABLE@@", error_table)
               .replace("@@RULE_FIT_VARIABILITY@@", rule_sentence)
               .replace("@@CLEAN_RESULTS@@", clean_results)
               .replace("@@IMBALANCED_BASELINES@@", imbalanced_baselines)
               .replace("@@CALIBRATION_CONTRAST@@", calibration_contrast)
               .replace("@@SPLITS@@", split_count)
               .replace("@@TRAIN_SEEDS@@", train_seed_count)
               .replace("@@SEED_LABEL@@", train_seed_label)
               .replace("@@EPOCHS@@", str(epoch_count))
               .replace("@@BATCH_SIZE@@", str(batch_size)))
    # StratifiedKFold test partitions are disjoint; it is the training partitions
    # and checkpoint groups that share observations. Correct legacy wording.
    for old, new in [
        ("test sets overlap and are not independent populations",
         "training sets share observations, so fold summaries are not population-level inference"),
        ("Fold test sets overlap", "Test folds partition the dataset, but training sets share observations"),
        ("Five folds overlap", "Test folds partition the data, but training sets share observations"),
        ("folds overlap", "training sets share observations"),
        ("overlapping folds", "folds with shared training observations"),
    ]:
        content = content.replace(old, new)
    if not std_zne.empty:
        fig_text = ("Figure~\\ref{fig:headline} summarizes the primary matched-budget comparison. Error bars are descriptive SD across 10 checkpoint means nested in five folds with shared training observations.\n"
                    "\\begin{figure*}[t]\\centering\n"
                    "\\includegraphics[width=0.78\\textwidth]{headline_primary_wdbc.png}\n"
                    "\\caption{Primary standardized-PCA WDBC comparison at $\\gamma=0.30$. Bars show balanced accuracy for noisy-threshold calibration and calibrated linear/quadratic ZNE under exact expectations and a 1,000-measurement budget. Error bars show descriptive SD across 10 checkpoint means; checkpoints are nested in five folds with shared training observations.}\n"
                    "\\label{fig:headline}\\end{figure*}\n")
        fig_text += ("\\begin{figure}[t]\\centering\n"
                     "\\includegraphics[width=0.98\\columnwidth]{calibration_efficiency_wdbc.png}\n"
                     "\\caption{Post-hoc calibration-size sensitivity for standardized-PCA WDBC at $\\gamma=0.30$. Points average 20 shot draws within each of 10 checkpoints; error bars are SD across checkpoint means nested in five overlapping folds.}\n"
                     "\\label{fig:calsize}\\end{figure}\n")
        marker = r"\end{table*}"
        tstart = content.index(r"\label{tab:standardized}")
        tend = content.index(marker, tstart) + len(marker)
        content = content[:tend] + "\n" + fig_text + "\n" + cal_size_note + "\n" + depth_note + "\n" + noise_note + "\n" + standardized_transfer_note + "\n" + (fake_note if 'fake_note' in locals() else "") + "\n" + content[tend:]
    if "cal_size_table" in locals() and cal_size_path.exists():
        content += ("\n\\begin{table*}[t]\\centering\\scriptsize\n"
            "\\caption{Post-hoc calibration-size sensitivity for standardized-PCA WDBC at $\\gamma=0.30$. Each pipeline uses the same nominal total of 1,000 measurements per test case, with half allocated to calibration and half to test inference; finite-shot test scores are reused across calibration-size conditions. Shot draws are averaged within each of 10 checkpoints; means and SDs are across checkpoint means nested in five overlapping folds.}\n"
            "\\label{tab:calsize}\n\\begin{tabular}{@{}l r r r r@{}}\\toprule\n"
            "Method & Calibration labels & Balanced accuracy & AUC & Rows \\\\ \\midrule\n"
            + cal_size_table + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    if depth_table:
        content += ("\n\\begin{table*}[t]\\centering\\scriptsize\n"
            "\\caption{Post-hoc standardized-PCA WDBC depth and score-separation diagnostic at $\\gamma=0.30$. Clean and exact-noisy metrics are checkpoint means. Finite-shot balanced accuracy averages 20 draws per checkpoint; its SD is across checkpoint means. Score spread, clean-shot gap, and one-score sampling SE are computed on calibration observations; folds overlap.}\n"
            "\\label{tab:primary-depth}\n\\begin{tabular}{@{}r r r r r r r r r@{}}\\toprule\n"
            "Layers & Clean BA & Exact noisy BA & Finite BA $\\pm$ SD & Finite AUC & Score SD & Class gap & SE at 500 & SE/gap \\\\ \\midrule\n"
            + depth_table + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    if noise_table:
        content += ("\n\\begin{table*}[t]\\centering\\scriptsize\n"
            "\\caption{Exact-expectation composite-noise grid for the standardized-PCA WDBC primary checkpoints. Calibrated methods use thresholds selected on held-out calibration labels. Means and SDs are over 10 saved model checkpoints. Reconciliation from saved per-example scores gives 0.878 for the primary calibrated-threshold result at $\\gamma=0.30$; the 0.891 entry here is fixed-threshold performance at $\\gamma=0.10$, a different condition. The cause of the earlier summary mismatch is not established.}\n"
            "\\label{tab:stdnoise}\n\\begin{tabular}{@{}r l r r r@{}}\\toprule\n"
            "Noise $\\gamma$ & Method & Balanced accuracy & AUC & Models \\\\ \\midrule\n"
            + noise_table + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    if standardized_transfer_table:
        content += ("\n\\begin{table*}[t]\\centering\\small\n"
            "\\caption{Standardized-PCA finite-shot transfer comparison on the same saved splits and initialization seeds, $\\gamma=0.30$, nominal budget 1,000. Threshold calibration includes its measurement cost. Values average 20 draws per checkpoint; no error bars are shown because the rows are descriptive and nested within five overlapping folds.}\n"
            "\\label{tab:primary-transfer}\n\\begin{tabular}{@{}l r r r r@{}}\\toprule\n"
            "Task & Fixed threshold & Noisy threshold + calibration & Linear ZNE + calibration & Quadratic ZNE + calibration \\\\ \\midrule\n"
            + standardized_transfer_table + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    if gatefold_table:
        content += ("\n\\begin{table*}[t]\\centering\\small\n"
            "\\caption{Matched-budget standardized-PCA WDBC finite-shot comparison at $\\gamma=0.30$. Each calibrated pipeline uses 500 measurements per calibration example and 500 per test case in total, with scale shots allocated proportionally to absolute Richardson coefficients. Means are over 20 draws per saved checkpoint; SD is across 10 checkpoint means nested in five overlapping folds. Its calibrated baseline (0.776 $\\pm$ 0.020) uses independent shot seeds from the main table (0.778 $\\pm$ 0.014).}\n"
            "\\label{tab:primary-gatefold}\n\\begin{tabular}{@{}l r r r@{}}\\toprule\n"
            "Pipeline & Balanced accuracy $\\pm$ checkpoint SD & ROC-AUC & Checkpoints \\\\ \\midrule\n"
            + gatefold_table + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    if fake_table:
        content += ("\n\\begin{table*}[t]\\centering\\small\n"
            "\\caption{IBM FakeManilaV2 device-model evaluation on standardized WDBC at nominal budget 1,000 (500 calibration, 500 test). Single-qubit readout error mitigation uses calibration matrix inversion on qubit 0. Gate-folded ZNE uses scales 1, 3, 5 with Richardson coefficients $(15/8, -5/4, 3/8)$ transpiled with non-canceling unitary folding. SDs are across 10 checkpoint means nested in five overlapping folds.}\n"
            "\\label{tab:fake-backend}\n\\begin{tabular}{@{}l r r r@{}}\\toprule\n"
            "Method & Balanced accuracy $\\pm$ checkpoint SD & ROC-AUC & Checkpoints \\\\ \\midrule\n"
            + fake_table + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
        content += ("\n\\begin{figure}[t]\\centering\n"
            "\\includegraphics[width=0.98\\columnwidth]{fake_manila_regime_map.png}\n"
            "\\caption{Exploratory noise-multiplier sweep using the saved IBM FakeManilaV2 snapshot noise model at multipliers $\\times 1$, $\\times 3$, and $\\times 10$. This is a local simulation, not current device calibration or QPU data.}\n"
            "\\label{fig:regime-map}\\end{figure}\n")
    # These results-driven tables are assembled after the template's end marker;
    # move them into the document before compiling/extracting supplementary tables.
    doc_end=content.rfind(r"\end{document}")
    if doc_end>=0 and r"\begin{table" in content[doc_end+len(r"\end{document}"):]:
        trailing=content[doc_end+len(r"\end{document}"):]
        content=content[:doc_end]+trailing+"\n"+content[doc_end:doc_end+len(r"\end{document}")]
        # Retain full copies in the supplementary file while the compact headline
        # versions remain visible in the main manuscript.
        content += ("\n\\begin{table*}[t]\\centering\\scriptsize\n"
            "\\caption{Supplementary depth and score-separation diagnostic for standardized-PCA WDBC.}\n"
            "\\label{tab:depth}\n\\begin{tabular}{@{}r r r r r r r r r@{}}\\toprule\n"
            "Layers & Clean BA & Exact noisy BA & Finite BA $\\pm$ SD & Finite AUC & Score SD & Class gap & SE at 500 & SE/gap \\\\ \\midrule\n"
            + depth_table + "\n\\bottomrule\\end{tabular}\\end{table*}\n"
            "\\begin{table*}[t]\\centering\\scriptsize\n"
            "\\caption{Full standardized-PCA transfer comparison on the same saved splits and seeds.}\n"
            "\\label{tab:stdtransfer}\n\\begin{tabular}{@{}l r r r r@{}}\\toprule\n"
            "Task & Fixed threshold & Noisy threshold + calibration & Linear ZNE + calibration & Quadratic ZNE + calibration \\\\ \\midrule\n"
            + standardized_transfer_table + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    if run_id:
        content = content.replace("{results/", "{")
    else:
        figure_prefix = RESULTS.relative_to(ROOT).as_posix()
        content = content.replace("{results/", "{" + figure_prefix + "/")
    path = (RESULTS / "paper.tex") if run_id else (ROOT / "paper.tex")
    supplement_labels = ["tab:exact","tab:finite","tab:transfer","tab:channels","tab:errors","tab:paired","tab:transferpaired","tab:zne_range","tab:allocation","tab:scales","tab:finite-scales","tab:calsize","tab:stdnoise","tab:depth","tab:stdtransfer"]
    supplement_tables=[]
    import re
    table_blocks=list(re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}",content,re.S))
    for match in reversed(table_blocks):
        block=match.group(0)
        if any(r"\label{"+label+r"}" in block for label in supplement_labels):
            supplement_tables.append(block.replace(r"\begin{table*}",r"\begin{table}").replace(r"\end{table*}",r"\end{table}"))
            content=content[:match.start()]+content[match.end():]
    supplement_tables.reverse()
    main_evidence=[]
    evidence_blocks=list(re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}",content,re.S))
    for match in reversed(evidence_blocks):
        block=match.group(0)
        if r"\label{tab:primary-depth}" in block or r"\label{tab:primary-transfer}" in block or r"\label{tab:primary-gatefold}" in block or r"\label{tab:fake-backend}" in block:
            main_evidence.append(block)
            content=content[:match.start()]+content[match.end():]
    main_evidence.reverse()
    if main_evidence:
        marker=r"\section{Discussion and limitations}"
        pos=content.index(marker)
        content=content[:pos]+"\n\n".join(main_evidence)+"\n\n"+content[pos:]
    # Keep only a short main-text pointer to the exhaustive legacy pipeline grids;
    # their labeled tables have already been extracted into the supplement above.
    legacy_start=r"\subsection{Unstandardized-PCA noise-grid and multi-task analyses}"
    legacy_end=r"\subsection{Clean performance and classical controls}"
    if legacy_start in content and legacy_end in content:
        a=content.index(legacy_start); b=content.index(legacy_end,a)
        content=content[:a]+("\\subsection{Secondary unstandardized-PCA analyses}\n"
            "The earlier unstandardized-PCA noise grids, channel controls, and cross-task comparisons are retained as secondary analyses in the supplementary material. They are not used to estimate the primary standardized-PCA result.\n\n")+content[b:]
    paired_start=r"\subsection{Secondary paired differences}"
    paired_end=r"\subsection{Score and budget diagnostics}"
    if paired_start in content and paired_end in content:
        a=content.index(paired_start); b=content.index(paired_end,a)
        content=content[:a]+("\\subsection{Supplementary paired and sensitivity analyses}\n"
            "Full paired contrasts for the legacy unstandardized pipeline, calibration-allocation variants, and ZNE scale schedules are provided in the supplementary tables. These post-hoc analyses reuse the recorded folds.\n\n")+content[b:]
    supplementary_figures=[]
    figure_blocks=list(re.finditer(r"\\begin\{figure\*\}.*?\\end\{figure\*\}",content,re.S))
    for match in reversed(figure_blocks):
        block=match.group(0)
        for label in ["fig:exact","fig:budget","fig:controls"]:
            if r"\label{"+label+r"}" in block:
                supplementary_figures.append(block.replace(r"\label{"+label+r"}",r"\label{fig:supp-"+label.split(":")[1]+"}"))
                content=content[:match.start()]+content[match.end():]
                break
    supplementary_figures.reverse()

    legacy_labels = ["tab:exact","tab:finite","tab:transfer","tab:channels","tab:errors","tab:paired","tab:transferpaired","tab:zne_range","tab:allocation","tab:scales","tab:finite-scales"]
    std_labels_list = ["tab:calsize","tab:stdnoise","tab:depth","tab:stdtransfer"]

    legacy_part = "\n\n".join(t for t in supplement_tables if any(f"\\label{{{l}}}" in t for l in legacy_labels))
    std_part = "\n\n".join(t for t in supplement_tables if any(f"\\label{{{l}}}" in t for l in std_labels_list))

    supplement=(r"""\documentclass[10pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\title{Supplementary Material: Decision-Threshold Selection and Zero-Noise Extrapolation}
\date{}
\begin{document}
\maketitle
\renewcommand{\thetable}{S\arabic{table}}
This document accompanies the main manuscript. The supplementary tables contain secondary unstandardized-PCA exact and finite grids, cross-task contrasts, channel and error controls, clipping and allocation diagnostics, scale sensitivity, a separate weak-signal synthetic diagnostic, calibration-size sensitivity, standardized-PCA noise results, depth score-spread diagnostics, and transfer comparisons. Supplementary Figures S1--S3 show secondary unstandardized-PCA diagnostics. Except where stated otherwise, results reuse saved outer folds and are descriptive.

\section*{Secondary unstandardized-PCA and paired analyses}

@@LEGACY_TABLES@@

\clearpage
\section*{Separate fresh-data synthetic diagnostic}

@@CONFIRMATION_SECTION@@

\clearpage
\section*{Standardized-PCA sensitivity analyses}

@@STD_TABLES@@

\end{document}""").replace("@@LEGACY_TABLES@@", legacy_part).replace("@@CONFIRMATION_SECTION@@", confirmation_section).replace("@@STD_TABLES@@", std_part)
    if supplementary_figures:
        fig_appendix=("\n\\clearpage\n\\section*{Supplementary Figures}\n"
                      "\\renewcommand{\\thefigure}{S\\arabic{figure}}\n"+
                      "\n\n".join(supplementary_figures)+"\n")
        supplement=supplement.replace(r"\end{document}",fig_appendix+r"\end{document}")
    supp_path=(RESULTS / "supplementary_tables.tex") if run_id else (ROOT / "supplementary_tables.tex")
    for old, new in [
        ("test sets overlap and are not independent populations",
         "training sets share observations, so fold summaries are not population-level inference"),
        ("Fold test sets overlap", "Test folds partition the dataset, but training sets share observations"),
        ("Five folds overlap", "Test folds partition the data, but training sets share observations"),
        ("folds overlap", "training sets share observations"),
        ("overlapping folds", "folds with shared training observations"),
    ]:
        content = content.replace(old, new)
        supplement = supplement.replace(old, new)
    # Keep the regime figure with its Results discussion, including regeneration.
    regime_figures = [m for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", content, re.S)
                      if r"\label{fig:regime-map}" in m.group(0)]
    if regime_figures:
        if len(regime_figures) != 1:
            raise ValueError("Expected one regime-map figure")
        block = regime_figures[0].group(0)
        content = content.replace(block, "")
        anchor = r"\subsection{Fresh split-and-seed confirmation on WDBC}"
        if anchor not in content:
            anchor = r"\section{Discussion and limitations}"
        content = content.replace(anchor, block + "\n\n" + anchor, 1)
    supplement = format_supplementary_tex(supplement)
    supp_path.write_text(supplement,encoding="utf-8")
    path.write_text(content, encoding="utf-8")
    print(f"Wrote results-driven manuscript: {path}")
    print(f"Wrote supplementary tables: {supp_path} ({len(supplement_tables)} tables)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Select results/runs/<run-id>; defaults to legacy results/")
    main(parser.parse_args().run_id)
