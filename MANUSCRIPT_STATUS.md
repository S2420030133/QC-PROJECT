# Manuscript checklist status — 2026-10-16

## Completed in this upgrade

1. **FakeManila Abstract Reframing & Regime Framing:**
   - Reframed the FakeManila result honestly in the abstract and body: under a local Aer simulation using a saved IBM FakeManilaV2 backend snapshot, the fixed threshold is close to the clean model (0.886 vs. 0.887), and methods in the primary table range from 0.873 to 0.886 (within checkpoint SD 0.025–0.038). This is not current device calibration or a QPU experiment.
   - Reframed the core question as *“When does recalibration matter?”*—recalibration matters when noise-induced score offset approaches the class-separation gap, as demonstrated in our stronger simulated composite channel and depth scaling.

2. **FakeManila Shot Verification & Mathematical Redundancy:**
   - Verified that FakeManila evaluation used finite 500-shot sampling (producing exact shot integer increments $-0.124 \times 500 = -62$).
   - Explained why ROC-AUC is preserved at 0.958 across raw and readout-mitigated rows: 1-qubit measurement error matrix inversion is a strictly monotonic affine linear transformation ($p_{\text{mit}} = M^{-1} p_{\text{raw}}$), mathematically preserving the rank ordering of continuous scores.
   - Corrected the interpretation of readout correction: with identical scores, a strictly monotone scalar correction followed by threshold refitting would preserve decisions. The saved methods use independent shot streams and give 0.873 raw calibrated BA (SD 0.027) versus 0.878 after readout correction (SD 0.031); fixed-threshold results are 0.886 and 0.885. These small differences are descriptive sampling variation, not evidence of a readout-mitigation benefit.
   - Recorded exact environment specs: Qiskit 2.5.2, Qiskit Aer 0.17.2, transpilation `optimization_level=0`, qubit mapping $[0, 1]$ measured on qubit 0.

3. **Device Noise-Scale Sweep & Regime Map:**
   - Implemented `evaluate_fake_backend_noise_sweep.py` across error rate multipliers $\times 1$, $\times 3$, and $\times 10$ on the FakeManila backend across all 10 checkpoints under nominal 1,000-shot budgets.
   - Demonstrated that threshold calibration remains robust across all scales (0.886, 0.894, 0.875), whereas gate-folded ZNE collapses to 0.798 at $\times 10$ due to physical error compounding.
   - Generated `fake_manila_regime_map.png` (Figure~\ref{fig:regime-map} in manuscript) and verified with `audit_fake_backend_noise_sweep.py` (PASS).

4. **Resolution of Limitations Contradictions:**
   - Removed contradictory statements in Discussion, Limitations, and Conclusion.
   - Replaced with accurate, unified statement: *"The device-model evaluation uses a single fake-backend snapshot; no real-hardware run was performed."*

5. **Elimination of 0.891 Provenance Notes:**
   - Identified that 0.891 was the standardized fixed-threshold value at $\gamma=0.10$ from the noise sweep table (Table S15).
   - Removed defensive draft-history apology notes from both the main text and supplementary material; verified 0.878 as the clean recomputed exact value at $\gamma=0.30$.

6. **Restoration of Clean Performance Numbers:**
   - Fixed text formatting in the clean-performance paragraph: restored all ROC-AUC values (WDBC: 0.812 BA / 0.959 AUC; Wine: 0.850 BA / 0.930 AUC; balanced synthetic: 0.697 BA / 0.766 AUC; imbalanced synthetic: 0.526 BA / 0.462 AUC).

7. **Classical Control Table Pipeline Consistency:**
   - Updated Table~\ref{tab:classical} to report standardized clean quantum performance (0.887 on WDBC, 0.858 on Wine, 0.689 on balanced synthetic) alongside raw clean quantum and standardized logistic regression (0.928, 0.950, 0.747), eliminating pipeline mixing.

8. **Abstract Depth Phrasing:**
   - Replaced overclaiming ("confirm", "because") with: *"is consistent with 1-layer circuits retaining a wider class-score gap (0.210 vs 0.079 vs 0.008), which prevents measurement shot noise from swamping the class boundary."*

9. **Quantitative Depth Signal-to-Noise Scaling:**
   - Added explicit arithmetic from the depth score spread table: class gap contracts roughly $\times 0.38$ ($1 \to 2$) and $\times 0.11$ ($2 \to 4$), or $\approx \times 0.32$ per layer.
   - Proved that matching depth-1 signal-to-noise ratio ($SE/\text{gap} = 0.328$) at depth 4 requires $N \propto 1/\text{gap}^2 \approx 640\times$ more measurements ($>320{,}000$ shots). At 500 shots, sampling uncertainty ($0.043$) is more than $5\times$ larger than the depth-4 class gap ($0.008$), explaining the collapse without invoking a universal depth law.

10. **Layout Overflow and Independent Seeds:**
    - Converted Table~\ref{tab:fake-backend} to a two-column `table*` with compact labels, eliminating the 177 pt column overflow.
    - Added note to Table~\ref{tab:primary-gatefold} caption clarifying that its calibrated baseline ($0.776 \pm 0.020$) uses independent shot seeds from the main table ($0.778 \pm 0.022$).

11. **Supplementary Tables Alignment (1-to-1):**
    - Structured `supplementary_tables.tex` into clean thematic sections so that Tables S1 through S17 match every single in-text citation in `paper.tex` without discrepancy:
      - Tables S1–S5: Secondary unstandardized exact and finite grids, transfer, channels, errors.
      - Tables S6–S11: Paired differences, clipping, budget allocation, and scale schedules.
      - Tables S12–S13: Separate fresh-data synthetic diagnostic and paired contrasts.
      - Tables S14–S17: Calibration-size sensitivity (S14), standardized exact noise grid (S15), depth score spread (S16), and standardized transfer (S17).
    - Updated `README_OVERLEAF.txt` with clear dual-document instructions for compiling `paper.pdf` and `supplementary_tables.pdf`.

12. **Packaging and Traceability:**
    - Re-built `qml-vqc-overleaf-project.zip` in `overleaf_ready_20261005/`.
    - Packaged `qml_vqc_research_package.zip` was refreshed on 2026-10-16; the current archive contains 720 entries (about 51.1 MiB) and includes the main run, fresh WDBC confirmation, source scripts, and audits.

## Current final-check status — 2026-10-16

- The FakeManilaV2 and noise-multiplier runs already provide a backend-snapshot simulation and readout-error-mitigation baseline; their audits pass. This is not current backend calibration or a real QPU run. An additional native-gate-folding replay also passes its 40-row audit and is kept separate from the primary backend table.
- The arXiv identifiers 2609.38713 and 2609.02781 were checked directly against their arXiv records; the IDs, titles, and author lists in the bibliography match those records.
- The exact 0.891 still appearing in Supplementary Table S15 is the fixed-threshold value at $\gamma=0.10$; it is not the earlier reported exact calibrated-threshold value at $\gamma=0.30$ (now 0.878). The regenerated caption states this distinction.
- Current manifest artifacts were rehashed after the last changes; 384 entries have no hash mismatches. The saved metric audits pass, including the 70-row FakeManila audit, 90-row regime-sweep check, and 40-row separate native-fold replay.
- The built-in LaTeX compiler successfully compiled the current open manuscript source on 2026-10-16 after an earlier platform-directory failure. This interface does not export the compiled PDF, so I could not inspect rendered layout or verify page count against a venue limit. The Overleaf project ZIP is refreshed with every referenced figure, including the FakeManila regime map.
- A real-hardware experiment remains unperformed. Author 1/2/3 and affiliation/email fields remain placeholders as requested. Neither publication acceptance nor a particular acceptance probability can be guaranteed.
- A preregistered fresh-split WDBC confirmation now adds five new stratified folds and ten newly trained checkpoints. At 1,000 nominal measurements and 20 draws per checkpoint, calibrated threshold selection scored 0.790 BA versus 0.721 for linear ZNE and 0.596 for quadratic ZNE; each of five descriptive fold contrasts favored calibration. The new run's 43 artifact hashes, 600 finite-shot metric reconstructions, 30 exact metric reconstructions, and checkpoint summary reconstruction pass. This is same-dataset simulator evidence, not hardware or external-population validation.
- The main manuscript now includes this confirmation and its per-fold contrasts. Its exact open LaTeX source still needs the built-in compiler and visual page-count review after the change; acceptance cannot be guaranteed.


## Final package refresh — 2026-10-16

- Refreshed the generated manuscript and supplement from the saved result tables after the fresh confirmation was added. The abstract and discussion describe the depth pattern as an observed association, not a demonstrated cause.
- The exact 0.891 value in Supplementary Table S15 is fixed-threshold performance at $\gamma=0.10$; the standardized, calibrated value at $\gamma=0.30$ is 0.878. The separate calibration-size sweep reaches 0.783 at 114 labels because its calibration-shot stream is independent of the main comparison; the shared splits, checkpoints, test draws, and threshold rule are unchanged.
- The new WDBC confirmation remains a same-dataset simulator confirmation; five fold contrasts are descriptive because training sets overlap. It does not establish external-population validity or hardware performance.
- Compiler success does not expose an exported PDF or venue page-count check in this interface; final visual layout still requires review in Overleaf or another LaTeX viewer.

