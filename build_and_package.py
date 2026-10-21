"""Package the manuscript, source, pinned runtime, and generated observations."""
from pathlib import Path
import argparse
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "qml_paper_submission.zip"
SOURCE_FILES = ["research_experiment.py", "evaluate_noise_degradation.py",
                "independent_confirmation_wdbc.py", "independent_confirmation_audit.py", "summarize_independent_confirmation.py",
                "evaluate_realistic_noise.py", "evaluate_zne_mitigation.py",
                "benchmark_classical_and_plots.py", "generate_paper.py", "build_and_package.py",
                "additional_sensitivity.py", "finite_scale_sensitivity.py", "standardized_pca_control.py", "four_layer_control.py", "compare_reproduction.py", "REPRODUCTION.md",
                "probe_primary_architectures.py", "confirmatory_budget_study.py", "plot_training_curves.py", "plot_score_errors.py", "summarize_calibration_rule_fits.py", "finite_scale_sensitivity.py",
                "finalize_run_metadata.py", "register_sensitivity_metadata.py", "matched_fit_classical.py", "zne_range_sensitivity.py",
                "standardized_finite_shot_repeats.py", "finalize_standardized_shot_study.py",
                "standardized_calibration_size_sweep.py", "record_calibration_subset_ids.py",
                "standardized_depth_sweep.py", "audit_standardized_depth_sweep.py",
                "standardized_exact_noise_sweep.py", "audit_standardized_exact_noise_sweep.py",
                "reconcile_standardized_exact_metrics.py", "standardized_transfer_comparison.py",
                "standardized_transfer_classical.py", "audit_standardized_transfer.py",
                "depth_score_spread.py", "gate_folded_zne.py", "audit_gate_folded_zne.py",
                "evaluate_fake_backend.py", "audit_fake_backend.py",
                "evaluate_fake_backend_noise_sweep.py", "audit_fake_backend_noise_sweep.py",
                "fake_backend_gatefolded_zne.py", "requirements-fake-backend.txt",
                "audit_fake_backend_native_fold.py",
                "app.py", "run_app.py", "README.md", "requirements.txt", "requirements-lock.txt",
                ".gitignore", "paper.tex", "supplementary_tables.tex", "SUMMARY.md", "RESEARCH_REPORT.md", "PORTABLE_README.md", "MANUSCRIPT_STATUS.md"]


def main(run_id: str | None = None, include_run_ids: list[str] | None = None):
    results = ROOT / "results"
    if run_id:
        results = results / "runs" / run_id
    output = results / "qml_paper_submission.zip" if run_id else OUTPUT
    paper_source = results / "paper.tex" if run_id else ROOT / "paper.tex"
    required = [paper_source, ROOT / "README.md", ROOT / "requirements.txt",
                ROOT / "requirements-lock.txt", results / "run_manifest.json",
                results / "per_example_observations.csv"]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Cannot package an incomplete research run; missing: " + ", ".join(missing))
    files = [ROOT / name for name in SOURCE_FILES
             if (ROOT / name).is_file() and name != "paper.tex"
             and not (run_id and name == "supplementary_tables.tex")]
    files.append(paper_source)
    files.extend(p for p in results.rglob("*") if p.is_file())
    for extra_id in include_run_ids or []:
        extra = ROOT / "results" / "runs" / extra_id
        if not extra.is_dir():
            raise SystemExit(f"Included run not found: {extra}")
        files.extend(p for p in extra.rglob("*") if p.is_file() and p.suffix.lower() != ".zip")
    written_names = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as zf:
        for path in sorted(set(files)):
            if path == output or ".venv" in path.parts or "__pycache__" in path.parts:
                continue
            archive_paths = [path.relative_to(ROOT).as_posix()]
            if path == paper_source:
                archive_paths.append("paper.tex")
            if run_id and path == results / "supplementary_tables.tex":
                archive_paths.append("supplementary_tables.tex")
            if run_id and path.parent == results and path.suffix.lower() in {".png", ".pdf", ".md", ".tex"}:
                archive_paths.append(path.name)
            for archive_path in dict.fromkeys(archive_paths):
                if archive_path in written_names:
                    continue
                zf.write(path, archive_path)
                written_names.add(archive_path)
    if run_id:
        portable_copy = ROOT / "qml_vqc_research_package.zip"
        shutil.copy2(output, portable_copy)
    print(f"Packaged {len(files)} source and experiment files: {output} ({output.stat().st_size / 1024:.1f} KiB)")
    if run_id:
        print(f"Portable project copy: {portable_copy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Package one results/runs/<run-id> run")
    parser.add_argument("--include-run", action="append", default=[],
                        help="Also package another saved run, including its original relative paths")
    args = parser.parse_args()
    main(args.run_id, args.include_run)
