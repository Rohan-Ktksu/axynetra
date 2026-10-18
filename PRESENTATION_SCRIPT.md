# Axynetra — 2–3 Minute Academic Presentation Script
**Topic:** CSMA Option 4: Multi-Class Attack Classification (UNSW-NB15)
**Tagline:** *Explainable Network Threat Classification & Intelligent Alert Triage*

---

## [0:00 – 0:30] 1. The Challenge & Project Identity
"Good morning/afternoon everyone. Today I am presenting **Axynetra — Explainable Network Threat Classification & Intelligent Alert Triage**, an offline security-analytics research prototype for **CSMA Option 4: Multi-Class Attack Classification**.

Traditional intrusion detection benchmarks often simplify network defense to a binary question: benign versus malicious. In a real Security Operations Center (SOC), however, analysts need to know *which attack family* is occurring—whether it is `Exploits`, `DoS`, `Backdoor`, `Worms`, or `Reconnaissance`—to prioritize investigation. At the same time, network traffic is heavily imbalanced: a model can achieve high overall accuracy simply by predicting majority classes while completely missing rare, high-severity threats."

## [0:30 – 1:05] 2. Dataset & Data Leakage Prevention
"We use the official **UNSW-NB15** pre-split dataset from UNSW Canberra, consisting of 175,341 training flows and 82,332 testing flows across **10 categories**: `Normal` traffic and nine attack families.

A core contribution of our implementation is strict **data leakage prevention** across three levels:
1. **Schema & Label Hygiene:** We normalize spelling inconsistencies—mapping `Backdoors` to `Backdoor`—and strictly exclude `attack_cat`, the binary `label`, and the row `id` from the predictor matrix.
2. **Train-Only Preprocessing:** Median numerical imputation, `StandardScaler`, categorical mode imputation, and `OneHotEncoder(handle_unknown='ignore')` are fitted *exclusively* on training data.
3. **Fixing Test-Set Selection Leakage:** Instead of ranking models on the official test set, we create an **80/20 stratified train/validation split** from the training partition. Our splitter guarantees that every rare class with at least two training rows is represented in both `fit-train` and `validation` (with a documented fallback if a custom subsample has only a single row). All candidate models are ranked solely by **Validation Macro F1**, and the winning model is evaluated **exactly once** on the untouched held-out test set."

## [1:05 – 1:45] 3. Models & Rare-Class Evaluation Metrics
"We compare four class-weighted classifiers:
- **Decision Tree**
- **Random Forest**
- **Extra Trees**
- **Multinomial Logistic Regression**

Because rare classes such as `Worms` (only 130 training and 44 test records in the full split) and `Shellcode` represent a tiny fraction of traffic, we use **Validation Macro F1** as our primary selection criterion so all 10 classes receive equal weight. We pair Macro F1 with Weighted F1, Macro Precision, Macro Recall, per-class classification reports, and a **row-normalized confusion matrix** that highlights inter-class confusion rates regardless of class size."

## [1:45 – 2:25] 4. Demo Workflow, Explainability & SOC Triage Queue
"Our reproducible workflow runs in three steps:
1. First, `multiclass_intrusion.py` executes the validation leaderboard, evaluates the selected winner once on the held-out test split, and writes all CSV, PNG, and JSON provenance artifacts to `outputs/`.
2. Second, for compatible tree ensembles, Axynetra exports **Top 12 Global Feature Importances** alongside per-alert standardized $z$-score feature deviations from training medians—explicitly caveated as associative and non-causal indicators.
3. Third, `streamlit run dashboard.py` opens our offline analyst dashboard, displaying the validation leaderboard versus held-out test metrics, the normalized confusion matrix, and an **illustrative prioritized SOC investigation queue** that combines predicted class, uncalibrated model confidence, and human-assigned severity weights (`0–5`) to flag suspicious flows for human review."

## [2:25 – 2:55] 5. Limitations & Future Work
"To conclude with responsible engineering boundaries:
- **Offline Prototype:** Axynetra analyzes stored CSV flow records offline; it does not perform live packet capture or automated blocking.
- **Uncalibrated Confidence & Synthetic Guardrails:** Raw classifier probabilities are uncalibrated, and synthetic `--smoke-test` runs are explicitly badged across the CLI, dashboard, and PDF report so they are never mistaken for real UNSW-NB15 results.
- **Future Work:** Next steps include post-hoc probability calibration (isotonic/Platt scaling), SHAP-based local explanations, and evaluating robustness under temporal concept drift across newer network captures. Thank you!"
