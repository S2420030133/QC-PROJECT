# NICEDT 2027 submission verification

Checked 17 October 2026. Target identified from the user's NICEDT 2027 poster. This is a verification report; manuscript sources, experimental artifacts, and submission archives were not changed during this check.

**Verdict: the current package is not yet verified as ready for NICEDT 2027.** The main IEEE PDF passes its own layout check, but the venue's linked template differs, some submission rules remain unpublished or inaccessible in the public sources checked, the supplement has a clipped table, and the full research archives are stale.

## Official venue information

| Item | Verified result | Source |
|---|---|---|
| Venue | Fifth NIELIT International Conference on Communication, Electronics and Digital Technologies, NICEDT 2027 | [Conference](https://nicedt.org/) |
| Conference dates | 8–9 January 2027 | [Dates](https://nicedt.org/index.php/important-dates/) |
| Submission deadline | Extended deadline: 21 October 2026. No cutoff time or timezone is stated on the dates page. | [Dates](https://nicedt.org/index.php/important-dates/) |
| Decision / camera-ready | 15 November / 30 November 2026 | [Dates](https://nicedt.org/index.php/important-dates/) |
| Topical fit | Track 3 explicitly includes Quantum Computing; Track 1 includes Machine Learning. Track 3 is my suggested fit for this quantum-noise benchmark, not a decision by the organizers. | [Tracks](https://nicedt.org/index.php/track-details/) |
| Published template guidance | The Submission Guidelines menu links a Springer Nature Word-template manual and a Word template. The manual uses a single-column chapter layout; the current paper uses IEEEtran conference layout. The manual is generic publisher guidance, not a complete conference submission policy. | [Linked manual](https://nicedt.org/wp-content/uploads/2025/11/Instructions_SpringerNature_Word_Template.pdf) |
| Publication | The website describes Springer publication as conditional on publisher standards and quality checks, with remaining papers directed to IJDT. Conference acceptance alone is not a guarantee of Springer publication. | [Conference](https://nicedt.org/) |
| Submission system | Official link leads to an EasyChair login for NICEDT-2027. No authenticated submission form was inspected. | [Submission portal](https://easychair.org/conferences/?conf=nicedt2027) |

The publicly accessible pages and linked manual checked do **not establish** a conference page limit, whether references count, the review anonymity model, whether a separate supplementary PDF/ZIP or code link is permitted, or whether an equivalent Springer LaTeX template is accepted. Do not infer these from other conferences, earlier NICEDT editions, a generic publisher template, or the absence of a stated restriction. The generic manual's named-author example does not establish single-blind initial review.

The prior IEEE QCE example does not govern this submission. Its 2026 [CFP](https://qce.quantum.ieee.org/2026/call-for-technical-papers/) specified 8–10 full-paper pages plus up to two reference pages; its [technical-paper deadline](https://qce.quantum.ieee.org/2026/submission-deadlines/) was 27 April 2026. Neither supports treating eight IEEE pages as NICEDT compliance.

## Document checks

| Check | Result |
|---|---|
| Current main PDF | Eight pages including references, as verified in the preceding revision. Current PDF metadata was checked again here. |
| Supplement independent compilation | PASS: compiled the dated Overleaf supplementary source with local Tectonic into a nine-page PDF. |
| Supplement contents | All 17 tables and three figures appear; all nine rendered pages were inspected at overview scale, with a detailed inspection of the defective page. |
| Supplement visual layout | FAIL: Table S3 on PDF page 2 extends beyond the right page edge. The compiler reports an overfull horizontal box of 87.87361 pt at source lines 62–72. The rightmost header is visibly clipped. |
| Supplement organization | Additional layout issues: page 1 is largely empty, and floating tables/figures precede their section headings in later sections. These should be repaired when preparing the venue version. |
| Current Overleaf ZIP | Its paper.tex and supplementary_tables.tex are byte-identical to the current root sources. It contains 12 files and passes ZIP CRC checks. It contains manuscript assets, not the research code/data release. |
| Full research ZIPs | NOT CURRENT: qml_vqc_research_package.zip and qml_paper_submission.zip are byte-identical 720-entry archives. Their top-level paper.tex and generate_paper.py differ from the current source, and the paper still has Author 1/2/3 placeholders. Both pass ZIP CRC checks, which establishes archive readability, not scientific correctness or currency. |

The nine-page supplement is a separate document. There is no verified permission to append it to the main paper or upload it separately at NICEDT. No venue conversion, source repair, archive rebuild, or scientific rerun was performed during this verification.

## Anonymity scan

Scope: both 720-entry research ZIPs, their nested Overleaf ZIPs and NPZ members, the current 12-entry Overleaf ZIP, current top-level Python sources, main/supplementary TeX, the main PDF, and the freshly compiled supplementary PDF. Checked filenames, textual contents, binary strings, safely decompressed zlib joblib content, PDF text/metadata, and PNG metadata. Serialized model objects were not executed or unpickled.

- No matches for the known local username, personal Windows/macOS/Linux home paths, or repository-profile URLs were found in that scope.
- No substantive author-email hits remained after distinguishing accidental email-like bytes in compressed media from actual text or metadata.
- The main and supplementary PDFs have Creator/Producer/build-date metadata, with no Author field. Current packaged figure metadata lists Matplotlib and resolution, with no personal author field.
- The old full archives contain author placeholders and older manuscript copies. Placeholders are a release-quality issue, not evidence of a disclosed real identity.
- This is a check for obvious direct identifiers. It does not prove full anonymity against inference from public preprints, distinctive titles, acknowledgments, or an eventual repository's ownership/history. No public repository link has been verified. An anonymous author block is provisional until the actual review rules are known.

No submission-integrity or open-science badge was inferred from this scan. No complete scientific reproducibility audit was rerun.

## Remaining actions before submitting

1. Confirm NICEDT's initial-submission template, maximum length and reference counting, anonymous-versus-named review, supplementary/code-link policy, and whether Springer LaTeX is accepted.
2. Prepare the manuscript in the confirmed format and recount pages; the IEEE count is not transferable.
3. Repair and re-render the supplement if the venue permits it.
4. Build a new, clearly dated research release from the reviewed sources while preserving archived experiment provenance. Verify its contents and identifiers again.

The [official contact page](https://nicedt.org/index.php/contact/) lists nicedt@nielit.ac.in. The following draft has **not** been sent:

> Subject: NICEDT 2027 initial-submission format and supplementary-material questions
>
> Dear NICEDT 2027 organizers,
>
> I am preparing a quantum machine-learning benchmark paper for Track 3 ahead of the 21 October deadline. Could you please confirm (1) the required initial-submission template and whether Springer LaTeX is accepted; (2) the maximum page count and whether references/appendices count; (3) whether the review PDF should include author names and affiliations; and (4) whether a separate supplementary PDF and an accompanying code/data repository link are allowed? Please also confirm the deadline cutoff time and timezone.
>
> Thank you.

## Local evidence

- `tmp/pdfs/submission-verification/anonymity_audit.json`: file-level scan and metadata evidence.
- `tmp/pdfs/submission-verification/supplementary_tables.log`: compilation log.
- `tmp/pdfs/submission-verification/supp-2.png`: visible S3 clipping.
- `tmp/pdfs/submission-verification/supp-contact.png`: all nine supplementary pages.
- `tmp/audit_submission_files.py`: repeatable scanner.

Archive SHA-256 values at verification:

```text
qml_vqc_research_package.zip / qml_paper_submission.zip
085b816036911b8fb8d0e1282ca4aae4727a8f8a15bced29253b5b714fde7203

overleaf_ready_20261017/qml-vqc-overleaf-project.zip
ae570fe461d28132a038eba256592213e36b6a09dd79e2c2478b90d0192cc2b7
```
