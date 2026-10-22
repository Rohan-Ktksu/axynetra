# Axynetra — CSMA Option 4: Multi-Class Attack Classification

**Tagline:** *Explainable Network Threat Classification & Intelligent Alert Triage*

**Axynetra** is an offline multi-class network-flow security analytics research prototype built for **CSMA Option 4**, designed for the **UNSW-NB15** dataset. It classifies stored network flows into **Normal** traffic and **nine named attack categories** (`Analysis`, `Backdoor`, `DoS`, `Exploits`, `Fuzzers`, `Generic`, `Reconnaissance`, `Shellcode`, `Worms`), separates validation-based model selection from held-out evaluation, provides exploratory feature signals, and generates an illustrative SOC analyst triage queue. CSV origin is not independently authenticated by the software.

> **Important Safeguard:** Axynetra is an **OFFLINE** research prototype operating on stored CSV records. It is **not** a live intrusion detection system (IDS), does not capture live packets, and does not perform automated blocking. Existing artifacts generated with `--smoke-test` use synthetic data solely for code verification and must not be reported as real UNSW-NB15 benchmark results.

---

## Key Methodological Features & Leakage Guardrails

1. **10-Class Normalization & Predictor Hygiene:**
   - Normalizes dataset label variants (`Backdoors` $\rightarrow$ `Backdoor`, whitespace/casing cleanup).
   - Strictly excludes `attack_cat`, binary `label`, and row `id` from predictors to prevent target/index leakage.
2. **Leak-Free Stratified Validation Selection (Evaluation Flaw Fixed):**
   - Splits the training data into an **80% fit-training subset** and a **20% stratified validation subset** (`--val-fraction 0.20`).
   - **Rare-class preservation:** Any attack class with $\ge 2$ training instances is guaranteed at least 1 sample in `fit-train` and at least 1 sample in `validation`. If an ultra-rare class has only 1 instance in a tiny custom subsample, it falls back to `fit-train` and logs the fallback in `run_metadata.json`.
   - All numeric (`SimpleImputer(strategy="median")`, `StandardScaler`) and categorical (`SimpleImputer(strategy="most_frequent")`, `OneHotEncoder(handle_unknown="ignore")`) transformers are fitted **exclusively on training data**.
   - All four classifiers (**Decision Tree**, **Random Forest**, **Extra Trees**, **Logistic Regression**) are ranked **solely by Validation Macro F1** (`validation_leaderboard.csv`).
   - Only the single validation winner is refitted on the full training set and evaluated **exactly once** on the untouched official held-out test set (`heldout_test_metrics.csv`).
3. **Explainability & SOC Alert Triage:**
   - Exports global impurity-based feature importances (`feature_importance.csv`, `feature_importance.png`) for compatible tree models, explicitly caveated as associative and non-causal.
   - Exports top 100 prioritized flows (`example_soc_alerts.csv`) combining uncalibrated model confidence, illustrative severity weights (`0–5`), human-review flags, and per-flow standardized feature deviations (`top_feature_deviations_illustrative`).

---

## Windows PowerShell Quickstart (From Project Root)

### 1. Environment Setup & Smoke Test Verification
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe multiclass_intrusion.py --smoke-test --output outputs
.\.venv\Scripts\python.exe -m pytest -v
```
Use an isolated virtual environment without `--system-site-packages` for a reproducible installation.
The reviewed environment uses Python 3.14.6. `requirements-lock.txt` records its complete development dependency versions, including the audit tool. To reproduce it, install with `python -m pip install -r requirements-lock.txt`. Run `python -m pip_audit` to recheck known vulnerabilities as advisories change.

### 2. Launch Offline Streamlit Analyst Dashboard
```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

### 3. Official UNSW-NB15 Dataset Execution
Download the official pre-split files from [UNSW Canberra — UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and save them in `data/`:
- `data/UNSW_NB15_training-set.csv` (175,341 records)
- `data/UNSW_NB15_testing-set.csv` (82,332 records)

**Short run on stratified subsamples (recommended first pass for modest RAM / integrated graphics):**
```powershell
.\.venv\Scripts\python.exe multiclass_intrusion.py --train data/UNSW_NB15_training-set.csv --test data/UNSW_NB15_testing-set.csv --max-train 50000 --max-test 20000 --output outputs
```

**Full official split run:**
```powershell
.\.venv\Scripts\python.exe multiclass_intrusion.py --train data/UNSW_NB15_training-set.csv --test data/UNSW_NB15_testing-set.csv --max-train 0 --max-test 0 --output outputs
```

### 4. Regenerate Academic PDF Report
```powershell
.\.venv\Scripts\python.exe create_report.py
```
To replace the `SRNNO` placeholder once ready for submission:
```powershell
.\.venv\Scripts\python.exe create_report.py --srn YOUR_SRN
```

---

## Output Artifacts (`outputs/`)
- `validation_leaderboard.csv` & `model_comparison.csv` — 4-model comparison on the stratified validation split (`accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1`, `selected_for_test`).
- `heldout_test_metrics.csv` — One-time evaluation of the validation winner on the untouched held-out test set.
- `classification_report.txt` — Per-class precision, recall, F1, and support on the held-out test set.
- `confusion_matrix.png` — Row-normalized confusion matrix on the held-out test set.
- `feature_importance.csv` & `feature_importance.png` — Top global tree feature importances (non-causal).
- `example_soc_alerts.csv` — Top 100 illustrative prioritized alerts with exploratory local feature deviations.
- `run_metadata.json` — Complete experiment provenance, split sizes, rare-class counts/fallbacks, and caveats.

## Academic Citation
Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)*. Military Communications and Information Systems Conference (MilCIS), IEEE.

## Multiclass ROC-AUC

Validation results now include one-vs-rest ROC-AUC for every candidate; held-out results include it for the validation winner. Scores use `predict_proba`, aligned to the classifier's class order. Model selection remains validation Macro F1.

- `roc_auc_ovr_macro`: the mean across all ten classes, available only when all ten class AUCs are defined.
- `roc_auc_ovr_macro_supported`: the mean over defined classes, accompanied by `roc_auc_defined_classes` and `roc_auc_total_classes`. This subset average must not be presented as full ten-class AUC.
- `validation_roc_auc.csv` and `heldout_test_roc_auc.csv`: per-class scores, support and reasons for undefined values (no positives, no negatives or a class absent from training).
- `roc_curves.png`: held-out one-vs-rest ROC curves. The dashboard and regenerated report expose the new results.

Rerun the pipeline to create these artifacts. Existing local outputs do not gain ROC-AUC until regenerated. Smoke-test AUC remains synthetic and must not be reported as UNSW-NB15 performance. This change adds ROC-AUC; autoencoders remain a proposed extension.

## Input checks and publication hygiene

- Missing or unknown attack categories are rejected, never silently labelled Normal.
- Identical predictor rows crossing train/test boundaries are rejected before subsampling. Duplicate training predictors must be resolved before the validation split. Investigate duplicates rather than silently dropping them: any changed split must be disclosed in reported results.
- Tiny sample limits that cannot represent every class are rejected. Sampling respects its exact row cap and preserves two examples per class when the cap permits.
- Macro metrics use the same ten class labels for all candidates, including absent classes with zero contribution. They may differ from older generated results.
- Metadata is written only after a completed run. A failed rerun cannot leave old completion metadata alongside partial new artifacts. Do not run multiple experiments into the same output directory concurrently.
- Streamlit defaults to localhost and disables usage telemetry. Do not expose the dashboard publicly with private flow data; it is not an authenticated multi-user service. CSV downloads neutralize spreadsheet formula prefixes.
- Generated results, raw CSVs, PDFs, logs, virtual environments, editor settings and common credential files are ignored by Git. Existing local reports predate these fixes: regenerate them before citing results.
- The report SRN accepts only letters, digits, underscores and hyphens, preventing path traversal and report markup injection.

See [REVIEW.md](REVIEW.md) for verification and remaining limits. No remote repository has been configured or pushed by this review.
