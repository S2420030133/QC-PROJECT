# Manuscript revision — 2026-10-17

The review supplied in chat prompted a source-checked revision of the active manuscript and its generator. Experiment outputs, saved runs, and their hash manifests were not modified. Older Overleaf folders and submission archives remain historical snapshots.

## Changes

- Moved the FakeManila regime-map figure into Results, immediately after its noise-multiplier discussion. The generator now preserves that placement.
- Explained the small circuits as a controlled benchmark for score ordering, threshold placement, and measurement variance, with no predictive-advantage claim.
- Distinguished final amplitude damping (affine Z offset), final isotropic depolarization with positive contraction (sign preservation), and final computational-basis dephasing (unchanged Z expectation). Unitality alone does not guarantee sign preservation. These identities do not model the complete interleaved circuit.
- Explicitly stated that the FakeManila snapshot also contains relaxation and readout errors. Differences in strengths, timing, placement, and circuits prevent a causal attribution of the synthetic-versus-snapshot contrast to unitality.
- Identified the existing secondary Platt-style comparator: calibration-only score standardization and L2 logistic regression with C=1, followed by the zero-logit decision boundary. It is not original Platt target smoothing. Tables S1–S2 already report it; the low-label standardized-PCA comparison remains untested.
- Replaced placeholder author names with Anonymous Authors as a provisional review version. This is not a claim of full venue-specific anonymization of all supplementary artifacts.
- Added T1 font encoding, restored the missing root-level training-curve image from the saved primary run, and shortened table headers to address compiler overflow.
- Corrected a stale gate-folding caption: the primary calibrated result has checkpoint SD 0.014, not 0.022. Recomputed from saved checkpoint means: mean 0.778074, sample SD 0.014494.

## Verification

- Desktop LaTeX compilation: success after source repairs.
- Exported local Tectonic PDF: **8 pages including references**. The regime-map figure renders on page 5; references are on page 8. All pages were visually inspected, with detailed checks of the repaired tables and final pages. No overfull-box or unresolved-reference warnings remain in the final log. Non-fatal underfull-box and class-initialization font warnings remain; the document body uses T1 Times fonts. The final page starts with the Conclusion, with no isolated carry-over word.
- All numerical table rows match the pre-edit manuscript, and all four main-document figure paths resolve. The caption SD correction noted above is separate from those table rows.
- Isolated regeneration: passed; output redirected to tmp/latex-review/generated so archived run files were not overwritten. Main-source differences were whitespace only.
- Numerical checks against PennyLane Kraus matrices: amplitude-damping identity, Z-dephasing invariance, and depolarizing contraction passed. A unital Pauli-X channel provides the checked counterexample to universal sign preservation.
- No new hardware execution, calibration experiment, model training, or full reproducibility audit was performed during this revision. Previously recorded audit results are historical evidence, not fresh checks from this turn.

## Review claims that should not become manuscript claims

The venue-specific acceptance percentages and “top 5%” ranking have no demonstrated empirical basis. A small QPU experiment may broaden evidence, but it cannot justify a promised acceptance probability. Venue deadlines, current access plans, backend availability, and page limits require current official verification after selecting a venue.

The checked PDF is `output/pdf/qml-vqc-manuscript-revised-20261017.pdf`. The current manuscript-only Overleaf bundle is `overleaf_ready_20261017/qml-vqc-overleaf-project.zip`; its supplementary source and images are copied from the existing project, without a new supplementary-layout audit. Older result-run papers and archives are not updated by this revision.

The existing affine-readout baseline means that “no logistic calibration comparator” is too broad. A useful new experiment would pair threshold fitting, logistic score calibration, and isotonic fitting on identical calibration/test score draws and label subsets (10, 20, 50, full), freeze fitting settings, count the same measurement budget, and report calibration-probability metrics separately from balanced accuracy. Temperature-only scaling of a zero-centered binary logit does not shift its zero decision boundary. Such an extension would be post-hoc and should be labeled accordingly.
