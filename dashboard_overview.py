"""Presentation of primary evidence from saved rows, with explicit dataset scope."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PRIMARY_RUN = "final-study-independent-retrain-20261003"
LABELS = {
    "noisy_threshold": "Noisy threshold + calibration",
    "linear_zne": "Linear ZNE + calibration",
    "quadratic_zne": "Quadratic ZNE + calibration",
    "fixed_zero_threshold": "Fixed zero threshold",
    "quantum_clean_fixed": "Quantum: clean fixed threshold",
    "quantum_clean_calibrated": "Quantum: clean calibrated threshold",
    "quantum_noisy_exact_fixed": "Quantum: exact noisy fixed threshold",
    "quantum_noisy_exact_calibrated": "Quantum: exact noisy calibrated threshold",
    "logistic_regression_standardized_2d": "Logistic regression: same standardized inputs",
}


def checkpoint_summary(frame, groups=("method",)):
    """Average shot draws first; report descriptive SD across checkpoints."""
    groups = list(groups)
    checkpoints = frame.groupby(groups + ["model_id"], as_index=False)[
        ["balanced_accuracy", "roc_auc"]].mean()
    return checkpoints.groupby(groups, as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"),
        checkpoint_sd=("balanced_accuracy", "std"),
        roc_auc=("roc_auc", "mean"), checkpoints=("model_id", "nunique"))


def table(frame):
    st.dataframe(frame, hide_index=True, width="stretch")


def read(path):
    return pd.read_csv(path)


def chart(summary):
    summary = summary.set_index("method").reindex(
        [m for m in ["noisy_threshold", "linear_zne", "quadratic_zne"]
         if m in summary.method.values])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar([LABELS.get(m, m) for m in summary.index], summary.balanced_accuracy,
           yerr=summary.checkpoint_sd, capsize=5, color=["#2166ac", "#67a9cf", "#ef8a62"])
    ax.set(ylabel="Balanced accuracy", ylim=(0.45, 1.0))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_overview(run):
    sensitivity = run / "sensitivity"
    primary_path = sensitivity / "wdbc_standardized_zne_comparison.csv"
    if not primary_path.exists():
        st.info("This saved run has no standardized primary comparison. Choose the final independent retrain run, or open the secondary explorer.")
        return
    st.subheader("Primary study · standardized-PCA WDBC")
    st.caption("Two-qubit CNOT model · composite noise γ = 0.30 · nominal 1,000 measurements per test case, including calibration. Saved results only.")
    primary = read(primary_path)
    primary = primary[(primary.gamma == .3) & (primary.budget == 1000) &
                      (primary.measurement_mode == "finite_calibrated")]
    summary = checkpoint_summary(primary)
    values = summary.set_index("method").balanced_accuracy
    for col, method in zip(st.columns(3), ["noisy_threshold", "linear_zne", "quadratic_zne"]):
        col.metric(LABELS[method], f"{values[method]:.3f}")
    st.caption("Means average shot draws within each checkpoint. Error bars and SD below are across 10 checkpoint means in five folds with shared training observations; they are descriptive.")
    evidence, confirmation, device, sensitivity_tab, files = st.tabs([
        "Primary comparison", "Fresh confirmation", "Device snapshot", "Sensitivity & transfer", "Documents & provenance"])
    with evidence:
        chart(summary)
        table(summary.assign(method=summary.method.map(LABELS).fillna(summary.method)))
        st.subheader("Clean and exact-expectation controls")
        raw_controls = read(sensitivity / "wdbc_standardized_pca_control.csv")
        controls = checkpoint_summary(raw_controls[~raw_controls.method.str.contains("1000shot")])
        table(controls.assign(method=controls.method.map(LABELS).fillna(controls.method)))
        st.info("Matched-input classical logistic regression is stronger than the clean quantum model. These results establish neither predictive quantum advantage nor clinical validity.")
        gate = sensitivity / "wdbc_gate_folded_zne_metrics.csv"
        if gate.exists():
            st.subheader("Separate gate-folded ZNE simulation")
            table(checkpoint_summary(read(gate)))
            st.caption("Independent shot draws on the saved primary checkpoints; the direct-threshold mean can differ slightly from the primary probability-scaled comparison.")
    with confirmation:
        fresh = ROOT / "results/runs/independent-confirmation-wdbc-20261016"
        if run.name == PRIMARY_RUN and (fresh / "finite_shot_results.csv").exists():
            st.subheader("Fresh split-and-seed WDBC confirmation")
            st.caption(f"Separate source: results/runs/{fresh.name}. Five new folds, ten newly trained checkpoints, twenty shot draws per checkpoint.")
            raw = read(fresh / "finite_shot_results.csv")
            confirmed = checkpoint_summary(raw[raw.measurement_mode == "finite_calibrated"])
            chart(confirmed)
            table(confirmed.assign(method=confirmed.method.map(LABELS).fillna(confirmed.method)))
            st.caption("Preregistered same-dataset simulator confirmation. This is not external-population or hardware validation.")
        else:
            st.info("The separate fresh confirmation is shown with the final independent retrain study.")
    with device:
        path = sensitivity / "wdbc_fake_backend_metrics.csv"
        if path.exists():
            st.subheader("IBM FakeManilaV2 snapshot · local simulation")
            table(checkpoint_summary(read(path)))
            st.info("This saved snapshot has near-clean fixed-threshold performance. It is not a physical QPU execution or current backend calibration; the small differences do not establish a consistent mitigation benefit.")
            picture = sensitivity / "fake_manila_regime_map.png"
            if not picture.exists():
                picture = run / picture.name
            if picture.exists():
                st.image(str(picture), caption="Exploratory device-noise multiplier sweep", width=750)
        else:
            st.info("No device snapshot results in this run.")
    with sensitivity_tab:
        panels = [
            ("Calibration label count", "wdbc_standardized_calibration_size_sweep.csv", ("calibration_examples", "method")),
            ("Circuit depth", "wdbc_standardized_depth_sweep.csv", ("depth", "method")),
            ("Standardized transfer", "standardized_transfer_comparison.csv", ("task", "method")),
        ]
        for title, filename, groups in panels:
            path = sensitivity / filename
            if path.exists():
                st.subheader(title)
                frame = read(path)
                if filename == "standardized_transfer_comparison.csv":
                    frame = frame[frame.measurement_mode == "finite_shot"]
                table(checkpoint_summary(frame, groups))
        st.caption("Post-hoc sensitivities reuse saved folds/checkpoints. Depth, task and shot-draw differences are not independent population replications.")
    with files:
        st.subheader("Current documents")
        if run.name == PRIMARY_RUN:
            for label, name in [("Download manuscript PDF", "paper.pdf"),
                                ("Download supplementary PDF", "supplementary_tables.pdf")]:
                path = ROOT / "output/release" / name
                if path.exists():
                    st.download_button(label, path.read_bytes(), file_name=name, mime="application/pdf")
            st.caption("Current release: output/release. Historical papers inside saved run directories retain their original provenance.")
        else:
            st.caption("Current release documents describe the final independent retrain study, not this selected run.")
        audit = ROOT / "output/verification/audit_summary.json"
        if run.name == PRIMARY_RUN and audit.exists():
            import json
            st.subheader("Latest local verification")
            table(pd.DataFrame(json.loads(audit.read_text(encoding="utf-8"))))
        st.caption(f"Loaded primary evidence from {run.relative_to(ROOT)}. Repeated measurement draws are not extra training runs. Hardware execution and a new low-label logistic/isotonic comparison remain outside this completed simulator study.")
