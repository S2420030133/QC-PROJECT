"""Read-only dashboard for generated experiment artifacts; no baked-in metrics."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"


@st.cache_data(show_spinner="Loading saved experiment results...")
def read_csv_cached(path: str, modified_ns: int, low_memory: bool = False):
    # Include mtime in the cache key so regenerated outputs are picked up.
    return pd.read_csv(path, low_memory=low_memory)


st.set_page_config(page_title="VQC Noise and Calibration Pilot", layout="wide")
st.title("VQC decision-threshold and mitigation pilot")
st.caption("Simulator results loaded from the saved experiment artifacts. This is a research benchmark explorer, not a clinical tool.")

run_dirs = sorted(
    (p for p in (RESULTS_ROOT / "runs").glob("*")
     if (p / "run_manifest.json").is_file()
     and (p / "exact_performance.csv").is_file()
     and (p / "finite_shot_performance.csv").is_file()),
    key=lambda p: p.name,
)
run_choices = {p.name: p for p in run_dirs}
if (RESULTS_ROOT / "run_manifest.json").is_file():
    run_choices["Legacy results (results folder)"] = RESULTS_ROOT
if run_choices:
    preferred = ("final-study-independent-retrain-20261003"
                 if "final-study-independent-retrain-20261003" in run_choices
                 else "final-study-20260929" if "final-study-20260929" in run_choices
                 else next(reversed(run_choices)))
    labels = list(run_choices)
    selected = st.selectbox("Saved experiment run", labels, index=labels.index(preferred))
    RESULTS = run_choices[selected]
    st.caption(f"Showing saved results from `{RESULTS.relative_to(ROOT)}`.")
else:
    RESULTS = RESULTS_ROOT

manifest_path = RESULTS / "run_manifest.json"
if not manifest_path.exists():
    st.info("No experiment output is available yet. Run `research_experiment.py` to create the manifest, per-example outputs, and summaries.")
    st.stop()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
st.caption("Synthetic-channel and saved device-snapshot simulations; no physical QPU run or clinical validation.")
with st.expander("Run configuration and provenance"):
    st.json(manifest)

view_name = st.radio("Results view", ["Study overview", "Secondary unstandardized explorer"], horizontal=True)
if view_name == "Study overview":
    from dashboard_overview import render_overview
    render_overview(RESULTS)
    st.stop()

st.subheader("Secondary unstandardized-PCA results")
st.caption("These older noise-grid controls use raw PCA. Their values differ from the standardized primary manuscript comparison.")

exact_path = RESULTS / "exact_performance.csv"
finite_path = RESULTS / "finite_shot_performance.csv"
obs_path = RESULTS / "per_example_observations.csv"
if not all(p.exists() for p in [exact_path, finite_path, obs_path]):
    st.error("The run is incomplete: one or more observation files are missing.")
    st.stop()
exact = read_csv_cached(str(exact_path), exact_path.stat().st_mtime_ns)
finite = read_csv_cached(str(finite_path), finite_path.stat().st_mtime_ns)
finite = finite[finite.shots_per_test_case.astype(str) != "exact"].copy()
if not finite.empty:
    finite["shots_per_test_case"] = pd.to_numeric(finite.shots_per_test_case)

tasks = sorted(exact.task.dropna().unique().tolist())
default_task = "wdbc" if "wdbc" in tasks else tasks[0]
task = st.selectbox("Benchmark task", tasks, index=tasks.index(default_task))
task_finite = finite[finite.task == task]
circuit_options = {"2-qubit CNOT chain": (2, "cnot_chain")}
if ((task_finite.n_qubits == 4) & (task_finite.architecture == "cnot_star")).any():
    circuit_options["Broader-input 4-qubit CNOT star"] = (4, "cnot_star")
circuit_label = st.selectbox("Circuit", list(circuit_options))
selected_qubits, selected_architecture = circuit_options[circuit_label]
method_labels = {
    "affine_readout": "Affine readout",
    "calibrated_threshold": "Noisy calibrated threshold",
    "clean_frozen_threshold": "Clean-frozen threshold (simulator reference)",
    "fixed_threshold": "Fixed threshold",
    "linear_zne": "Linear ZNE",
    "linear_zne_calibrated": "Linear ZNE + calibration",
    "quadratic_zne": "Quadratic ZNE",
    "quadratic_zne_calibrated": "Quadratic ZNE + calibration",
}
tab_exact, tab_finite, tab_examples = st.tabs(["Exact expectation values", "Finite-shot matched budget", "Per-example observations"])

with tab_exact:
    summary_path = RESULTS / "exact_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[(summary.task == task) &
                          (summary.n_qubits == selected_qubits) &
                          (summary.architecture == selected_architecture)]
    else:
        frame = exact[(exact.task == task) & (exact.noise == "composite") &
                      (exact.n_qubits == selected_qubits) &
                      (exact.architecture == selected_architecture)]
        summary = frame.groupby(["task", "gamma", "method"], as_index=False).agg(
            balanced_accuracy=("balanced_accuracy", "mean"), roc_auc=("roc_auc", "mean"),
            collapse_rate=("single_class_collapse", "mean"),
            score_rmse=("score_rmse_vs_clean", "mean")) if not frame.empty else pd.DataFrame()
    if summary.empty:
        st.info("No exact composite-noise results for this task.")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Balanced accuracy and ROC-AUC")
            fig, ax = plt.subplots(figsize=(8, 4))
            for method, group in summary.groupby("method"):
                ax.plot(group.gamma, group.balanced_accuracy, marker="o",
                        label=method_labels.get(method, method))
            ax.set(xlabel="Per-location probability γ", ylabel="Balanced accuracy", ylim=(0, 1.02))
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)
        with right:
            fig, ax = plt.subplots(figsize=(8, 4))
            for method, group in summary.groupby("method"):
                ax.plot(group.gamma, group.roc_auc, marker="o",
                        label=method_labels.get(method, method))
            ax.set(xlabel="Per-location probability γ", ylabel="ROC-AUC", ylim=(0, 1.02))
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)
        summary = summary.copy()
        summary["method"] = summary.method.map(method_labels).fillna(summary.method)
        st.dataframe(summary, width="stretch")
        st.caption("Means are across saved model runs. Bootstrap intervals are in `paired_differences_bootstrap_ci.csv`; repeated shot samples are not independent test examples.")

with tab_finite:
    frame = finite[(finite.task == task) & (finite.n_qubits == selected_qubits) &
                   (finite.architecture == selected_architecture)]
    if frame.empty:
        st.info("No finite-shot matched-budget results exist for this task and circuit.")
    else:
        gammas = sorted(frame.gamma.unique().tolist())
        budgets = sorted(frame.shots_per_test_case.unique().tolist())
        gamma_index = gammas.index(0.3) if 0.3 in gammas else 0
        budget_index = budgets.index(1000) if 1000 in budgets else len(budgets) - 1
        control_key = f"{task}_{selected_qubits}_{selected_architecture}"
        gamma = st.selectbox("Composite noise setting γ", gammas, index=gamma_index,
                             key=f"finite_gamma_{control_key}")
        budget = st.selectbox("Total shots per test case", budgets, index=budget_index,
                              key=f"finite_budget_{control_key}")
        view = frame[(frame.gamma == gamma) & (frame.shots_per_test_case == budget)]
        agg = view.groupby("method", as_index=False).agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_sd=("balanced_accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"), collapse_rate=("single_class_collapse", "mean"),
            score_mae_vs_clean=("score_mae_vs_clean", "mean"),
            score_rmse_vs_exact=("score_rmse_vs_exact_method", "mean"),
            test_shots_mean=("measurement_shots_test", "mean"),
            calibration_shots_mean=("measurement_shots_calibration", "mean"),
            total_shots_mean=("measurement_shots_total", "mean"))
        agg["method"] = agg.method.map(method_labels).fillna(agg.method)
        st.dataframe(agg, width="stretch")
        st.caption("Balanced-accuracy SD spans model and shot repetitions. For inference, calibration, and ZNE shot allocations, inspect the saved per-run rows for exact shot totals.")

with tab_examples:
    # Streamlit executes every tab body, so an explicit control gates the large CSV.
    if not st.checkbox("Load per-example observations", value=False):
        st.caption("Enable this control to load the selected run's larger observation file.")
        st.stop()
    observations = read_csv_cached(str(obs_path), obs_path.stat().st_mtime_ns, low_memory=False)
    task_obs = observations[(observations.task == task) & (observations.noise == "composite")]
    model_ids = exact.loc[(exact.task == task) & (exact.n_qubits == selected_qubits) &
                         (exact.architecture == selected_architecture), "model_id"].unique()
    task_obs = task_obs[task_obs.model_id.isin(model_ids)]
    if task_obs.empty:
        st.info("No per-example rows are available for the selected task.")
    else:
        gamma = st.selectbox("Noise setting γ", sorted(task_obs.gamma.unique().tolist()), key="obs_gamma")
        task_obs = task_obs[task_obs.gamma == gamma]
        method = st.selectbox("Method", sorted(task_obs.method.unique().tolist()), key="obs_method")
        task_obs = task_obs[task_obs.method == method]
        shot_choices = task_obs.shots_per_test_case.astype(str).unique().tolist()
        shot = st.selectbox("Shots per test case", shot_choices, key="obs_shots")
        reps = sorted(task_obs[task_obs.shots_per_test_case.astype(str) == shot].replicate.unique().tolist())
        rep = st.selectbox("Measurement repetition", reps, key="obs_rep")
        selected = task_obs[(task_obs.gamma == gamma) & (task_obs.method == method) &
                            (task_obs.shots_per_test_case.astype(str) == shot) & (task_obs.replicate == rep)]
        if selected.empty:
            st.info("No observations match those controls.")
        else:
            # Display the first saved model run; full run/model identifiers remain visible in the table.
            model_id = st.selectbox("Frozen model", sorted(selected.model_id.unique().tolist()))
            selected = selected[selected.model_id == model_id]
            cm = confusion_matrix(selected.y_true_model_label, selected.prediction, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4.3, 3.4))
            ConfusionMatrixDisplay(cm, display_labels=["class 0", "class 1"]).plot(ax=ax, colorbar=False)
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(selected, width="stretch", height=360)
