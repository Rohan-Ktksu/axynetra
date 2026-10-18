# Pre-publication review — 18 October 2026

Reviewed the classifier, dashboard, report generator, tests, configuration and local publication artifacts.

## Changes

- Removed the requested handoff document and removed its tool-specific local path from the old console log.
- Fixed stratified subsampling: exact row limits and deterministic per-class allocation replace random trimming that could remove rare-class examples.
- Reject missing/unsupported targets, overlapping train/test predictors, duplicate training predictors, single-class training and empty validation sets with actionable errors.
- Use a fixed ten-class set for macro selection metrics.
- Invalidate previous completion metadata before fitting so failed reruns are not displayed as completed experiments.
- Replace automatic assertions of official dataset provenance with an explicit unverified-input description in runtime metadata and the dashboard.
- Neutralize spreadsheet formulas in dashboard CSV exports and validate report filenames/SRN text.
- Include dashboard/report dependencies, add development requirements, and configure local-only Streamlit access with telemetry disabled.
- Exclude datasets, generated results/reports, local configuration and common secret formats from publication.

## Verification

17 tests passed in the rebuilt isolated environment, including the end-to-end synthetic training/evaluation/dashboard-loader test and eleven regression tests. Python sources compile. Source scanning found no obvious hard-coded private keys or common API-token patterns; a pattern scan is not proof that every possible secret is absent.

No real UNSW CSV files were present, so real-data performance, duplicate frequency and full-dataset memory usage were not tested. Existing synthetic outputs and the academic PDF are retained locally and excluded from Git; they must be regenerated before use as evidence for the revised code.

The directory was not a Git repository at review start, so there was no prior commit history to audit. After review, local Git was initialized on `main` with `https://github.com/Rohan-Ktksu/axynetra.git` as origin. The remote was empty when checked. The initial commit includes source, tests, documentation and reference diagrams; local datasets, generated reports/results and credentials are excluded. No GitHub upload was made; the owner will push. Ignore patterns do not protect files deliberately force-added later.

The copied virtual environment was rebuilt in place without inherited system packages; activation and pip launchers now reference this project. Direct dependencies are pinned, and `requirements-lock.txt` records the full reviewed development environment (including pytest and pip-audit). It was tested on Windows with Python 3.14.6. `pip check` passed. `pip-audit` found no known vulnerabilities in the installed environment on the review date. This does not rule out undisclosed vulnerabilities or replace future audits.

The user removed the obsolete compiled test cache, and its absence was verified. The requested handoff document is removed. Follow-up scanning found no remaining references in project files or environment launchers; downloaded dependency libraries and package/audit/temporary caches were excluded from that scan. The existing PDF text was also checked earlier with no matching references found.

No claim of production readiness is made. Real data and any public multi-user deployment need separate validation.
