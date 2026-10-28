### What the model predicts

Each row describes a network connection (a flow). This is **supervised, multiclass classification**: the model learns from flow measurements paired with a known `attack_cat` label. It predicts one of ten categories: **Normal, Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, or Worms**. Autoencoders and live packet capture are not implemented.

### From data to prediction

1. **Load and verify.** Read the two bundled cleaned CSVs and verify their SHA-256 hashes against the provenance manifest. Use reproducible stratified samples of 10,000 training and 5,000 test rows (seed 42).
2. **Check inputs.** Normalize category spelling; reject missing or unknown targets, infinite predictors, duplicate training predictors, ambiguous headers and exact train/test predictor overlap. Remove `id`, `label` and `attack_cat` from model inputs.
3. **Reserve validation.** Split training rows approximately 80/20 within each category. Categories with at least two examples retain at least one fit and one validation example. Singleton classes remain in fit-training and are logged. Test records stay outside fitting and model selection.
4. **Preprocess inside each pipeline.** Numeric columns use training-median imputation and StandardScaler. The three categorical columns (`proto`, `service`, `state`) use most-frequent imputation and one-hot encoding. Unknown categories are ignored by the encoder. Each candidate fits its own preprocessing on fit-training rows only.
5. **Compare four classifiers.** Train a Decision Tree, Random Forest, Extra Trees and Logistic Regression with class weights to address imbalance. Compare on validation **Macro F1**; weighted F1 and accuracy break ties. No hyperparameter search is performed.
6. **Refit and evaluate.** Build a fresh pipeline for the winning algorithm, fit it on all 10,000 training rows, then evaluate on the reserved 5,000 test rows once in the final evaluation stage of each experiment.
7. **Run interactive inference.** Train Model, Test Model and Predict are separate stages. Training ends after refitting; testing computes held-out results. Predict sends only selected predictors through the same fitted pipeline. Actual labels are compared afterward. Changing a test record or uploading a CSV does not retrain or change evaluation scores. The GitHub Pages website executes Python in a background browser worker; refreshing discards the model. Package versions are recorded, and estimators use one browser thread.

### Why these four algorithms?

| Classifier | How it learns | Why compare it? |
| --- | --- | --- |
| Decision Tree | Splits records using feature thresholds | A single-tree baseline that can capture nonlinear rules |
| Random Forest | Combines many trees trained with bootstrap sampling and random feature subsets | Reduces single-tree variance and models interactions |
| Extra Trees | Combines trees with more randomized split thresholds | A different tree ensemble that may generalize differently |
| Logistic Regression | Learns weighted linear scores for each category | A class-weighted linear baseline after scaling and encoding |

The website shows the actual winner after a run. It does not assume a particular algorithm always wins. Expand the classifier settings below to inspect the exact parameters used by the core code.

### How to read the outputs

**Accuracy** is the fraction of correctly classified test records. **Precision** asks how often a predicted category is correct; **recall** asks how many records of that category were found. **F1** balances precision and recall. **Macro F1** averages all ten category F1 values equally, so rare attacks matter. **Weighted F1** weights categories by their observed support. A high overall score can coexist with poor rare-attack recall.

The **confusion matrix** shows where true and predicted categories disagree; each row is normalized by its true category. **One-vs-rest ROC-AUC** measures how model scores rank one category against the others. It is undefined without both positive and negative examples, or if the model did not learn that category; the output reports defined-class coverage.

**Feature importance** measures global impurity reduction for a compatible tree. If the winner is Logistic Regression, the chart comes from the best compatible validation tree and identifies that source. One-hot encoding expands categorical columns, so transformed importance names differ from the 42 original inputs. Importance is neither causal nor a per-record explanation, and does not remove low-ranked features.

The **alert queue** multiplies the largest uncalibrated model score by a manually assumed severity (0–5). Predictions other than Normal, and scores below 0.60, are flagged for review. Standardized numeric deviations from training medians are exploratory signals, not proof of an attack. These illustrative priorities do not authorize automated blocking.

### Data provenance and limits

The bundled files contain 99,268 cleaned training and 73,791 test rows derived from a community mirror of [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Training records with conflicting labels for identical predictors and remaining training duplicates were removed. Test rows overlapping the original training predictors were removed; remaining within-test duplicates were retained. The website uses smaller stratified samples of these files. The provenance manifest records the source revision and hashes.

These results describe a **cleaned, sampled, modified split**, not the unchanged official benchmark. The root synthetic notebook is a separate demonstration and its scores are not used here. Repeatedly inspecting test predictions must not be used to tune the model. Unseen networks can have different traffic, class frequencies and errors; model scores have not been calibrated into operational attack probabilities.

The original dataset and its research contributions belong to the UNSW-NB15 authors. Dataset attribution and requested academic references are retained in the repository's reference README.
