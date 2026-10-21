"""Plot exact observable error and finite-shot sampling deviation from a saved run."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main(run_dir: str) -> None:
    run = Path(run_dir)
    exact = pd.read_csv(run / "exact_summary.csv")
    exact = exact[(exact.task == "wdbc") & (exact.n_qubits == 2) &
                  (exact.architecture == "cnot_chain") &
                  exact.method.isin(["fixed_threshold", "linear_zne", "quadratic_zne"])]
    finite = pd.read_csv(run / "finite_shot_summary.csv")
    finite = finite[(finite.task == "wdbc") & (finite.n_qubits == 2) &
                    (finite.architecture == "cnot_chain") & finite.gamma.isin([0.3, 0.4]) &
                    finite.method.isin(["fixed_threshold", "linear_zne", "quadratic_zne"])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for method in ["fixed_threshold", "linear_zne", "quadratic_zne"]:
        part = exact[exact.method == method].sort_values("gamma")
        if not part.empty:
            axes[0].plot(part.gamma, part.score_rmse, marker="o", label=method)
    axes[0].set(title="Observable error relative to clean", xlabel="Per-location γ", ylabel="RMSE")
    axes[0].legend(fontsize=7)

    colors = {"fixed_threshold": "#2563eb", "linear_zne": "#f97316", "quadratic_zne": "#16a34a"}
    markers = {0.3: "o", 0.4: "s"}
    for (gamma, method), part in finite.groupby(["gamma", "method"]):
        part = part.sort_values("shots_per_test_case")
        axes[1].scatter(part.shots_per_test_case, part.sampling_rmse_mean,
                        color=colors[method], marker=markers[float(gamma)], s=42,
                        label=f"γ={gamma:.1f} {method}")
    axes[1].set(title="Observable sampling deviation (two budgets)",
                xlabel="Total shots per test case", ylabel="RMSE", xscale="log")
    axes[1].legend(fontsize=6, ncol=2)
    fig.savefig(run / "score_error_decomposition.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    main(parser.parse_args().run_dir)
