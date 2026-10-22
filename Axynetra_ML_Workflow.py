# %% [markdown]
# # Axynetra: a complete supervised ML experiment
# **Multi-Class Attack Classification — CSMA Option 4**
#
# This notebook explains each stage rather than hiding training inside one function.
# It reuses tested data/metric helpers from `multiclass_intrusion.py`, while showing
# preprocessing, model fitting, selection, final evaluation and exports explicitly.
#
# **Saved outputs are synthetic demonstrations, not UNSW-NB15 benchmark results.**
# The implemented models are supervised classifiers; no autoencoder is used.
# Restart the kernel and Run All after changing configuration. Do not repeatedly
# inspect test scores to tune models. Only the validation set selects the winner.

# %% [markdown]
# ## 1. Imports and reproducibility
# Output: package versions, random seed and the ten supported categories.

# %%
from pathlib import Path
import json
import platform
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import multiclass_intrusion as mi

ROOT = Path.cwd()
if not (ROOT / 'multiclass_intrusion.py').is_file():
    raise RuntimeError('Open this notebook with the Axynetra project root as the working directory.')
pd.set_option('display.max_columns', 16)
SEED = mi.SEED
versions = {name: importlib.metadata.version(name)
            for name in ['numpy', 'pandas', 'scikit-learn', 'matplotlib']}
print('Python:', platform.python_version(), '| Seed:', SEED)
display(pd.Series(versions, name='version').to_frame())
print('Classes:', ', '.join(mi.CLASSES))

# %% [markdown]
# ## 2. Configure and load data
# `SMOKE_TEST=True` explicitly generates a reproducible synthetic demonstration.
# Set it to `False` only after placing the two real CSVs in `data/`.
# Missing real files cause an error; there is no automatic synthetic fallback.
# Set row limits to zero for the full supplied split. Record subsampling in results.
# Output: run type, source fingerprints and dataset sizes.

# %%
SMOKE_TEST = True
MAX_TRAIN = 50000
MAX_TEST = 20000
VAL_FRACTION = 0.20
TRAIN_PATH = ROOT / 'data' / 'UNSW_NB15_training-set.csv'
TEST_PATH = ROOT / 'data' / 'UNSW_NB15_testing-set.csv'
OUT = ROOT / 'outputs' / 'notebook'
OUT.mkdir(parents=True, exist_ok=True)
# Invalidate completion marker before beginning a new notebook experiment.
(OUT / 'run_metadata.json').unlink(missing_ok=True)

def fingerprint(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

if SMOKE_TEST:
    train, test = mi.make_smoke_data()
    provenance = {'source': 'Synthetic sklearn.make_classification; not UNSW-NB15'}
    RUN_LABEL = 'SYNTHETIC DEMONSTRATION — NOT UNSW-NB15 PERFORMANCE'
else:
    if not TRAIN_PATH.is_file() or not TEST_PATH.is_file():
        raise FileNotFoundError('Download the real train/test CSVs into data/ before disabling SMOKE_TEST.')
    train, test = mi.load_original(TRAIN_PATH, TEST_PATH, MAX_TRAIN, MAX_TEST)
    provenance = {'source': 'Supplied CSV files; origin not independently verified',
                  'train_sha256': fingerprint(TRAIN_PATH), 'test_sha256': fingerprint(TEST_PATH)}
    RUN_LABEL = 'SUPPLIED CSV EXPERIMENT — VERIFY DATASET ORIGIN'
print(RUN_LABEL)
display(pd.DataFrame({'split': ['training', 'held-out test'], 'rows': [len(train), len(test)],
                      'columns': [train.shape[1], test.shape[1]]}))
display(pd.Series(provenance, name='provenance').to_frame())

# %% [markdown]
# ## 3. Inspect training data (EDA)
# Explore training records only. These views explain the available feature types,
# missing values and class imbalance. They do not fit a transformer or use test
# performance for design decisions. Synthetic `feature_XX` columns are not actual
# UNSW-NB15 feature names. Real benchmark features include flow duration/protocol.
# Output: training preview, schema, numeric summary, missingness and class chart.

# %%
display(train.head())
schema = pd.DataFrame({'dtype': train.dtypes.astype(str), 'missing': train.isna().sum(),
                       'unique': train.nunique(dropna=True)})
display(schema)
display(train.describe(include='number').T)
class_counts = train.attack_cat.value_counts().reindex(mi.CLASSES, fill_value=0)
display(class_counts.rename('training_rows').to_frame())
fig, ax = plt.subplots(figsize=(10, 4))
class_counts.plot.bar(ax=ax, color='#247b83')
ax.set(title=f'Training class distribution\n{RUN_LABEL}', ylabel='Records', xlabel='Class')
fig.tight_layout()
fig.savefig(OUT / 'training_class_distribution.png', dpi=150)
display(fig)
plt.close(fig)

# %% [markdown]
# ## 4. Validate inputs and prevent identifiable leakage
# Exclude the target `attack_cat`, binary `label`, and row `id` from predictors.
# Reject missing labels, infinite measurements, duplicate training predictors and
# exact train/test overlap. Investigate duplicate records before changing the split;
# never silently deduplicate an official benchmark and claim its original protocol.
# These checks do not establish the absence of every possible form of leakage.
# Output: accepted predictor list and successful input checks, or an actionable error.

# %%
features = mi.get_predictor_columns(train, test)
for split_name, frame in [('training', train), ('test', test)]:
    frame['attack_cat'] = frame.attack_cat.map(mi.normalized_label)
    mi.validate_labels(frame, split_name)
    numeric = frame[features].select_dtypes(include='number')
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError(f'{split_name} has infinite predictors; investigate them first.')
mi.check_overlap(train, test, features)
if train.duplicated(subset=features).any():
    raise ValueError('Resolve duplicate training predictors before validation splitting.')
if train.attack_cat.nunique() < 2:
    raise ValueError('Training needs at least two classes.')
assert not set(c.strip().lower() for c in features) & mi.LEAKAGE_COLUMNS
print('Input checks passed. Predictor count:', len(features))
print('Excluded:', ', '.join(sorted(mi.LEAKAGE_COLUMNS)))
print('Predictors:', ', '.join(features))

# %% [markdown]
# ## 5. Create fit-training and validation subsets
# Stratify within the training data, approximately 80/20. Classes with at least two
# examples retain one in each subset; singleton classes stay in fit-training with a
# documented fallback. Integer rounding means the ratio is approximate.
# Output: fit/validation sizes and class coverage. The held-out test stays separate.

# %%
fit, validation, split_info = mi.stratified_train_val_split(train, VAL_FRACTION)
if validation.empty:
    raise ValueError('No validation rows; supply more training examples.')
X_fit, y_fit = fit[features], fit.attack_cat
X_val, y_val = validation[features], validation.attack_cat
X_train, y_train = train[features], train.attack_cat
X_test, y_test = test[features], test.attack_cat
display(pd.DataFrame({'fit_training': y_fit.value_counts(), 'validation': y_val.value_counts()})
        .reindex(mi.CLASSES).fillna(0).astype(int))
print(f'Fit-training: {len(fit)} | Validation: {len(validation)} | Held-out test: {len(test)}')
print('Rare-class fallbacks:', split_info['rare_class_fallbacks'] or 'None')

# %% [markdown]
# ## 6. Define preprocessing and candidate classifiers
# Numeric features: median imputation then standardization.
# Categorical features: most-frequent imputation then one-hot encoding, ignoring
# categories unseen during fitting. The target is never included in this transform.
# Each model receives its own preprocessing pipeline, fitted only on fit-training.
# Class weighting helps address imbalance. There is no learned autoencoder stage.
# Output: preprocessing structure and the four model configurations.

# %%
display(mi.build_preprocessor(X_fit))
candidates = mi.model_choices()
display(pd.DataFrame([{'model': name, 'estimator': repr(estimator)}
                      for name, estimator in candidates.items()]))

# %% [markdown]
# ## 7. Train models and compare validation results
# Every candidate fits on fit-training and is measured on validation. Macro F1 gives
# equal weight to the ten defined classes. ROC-AUC uses probability scores in a
# one-vs-rest comparison, aligned to `classifier.classes_`, not predicted labels.
# Full macro AUC is undefined unless all ten class AUCs can be computed; the separately
# named supported-class mean includes a coverage count. It is not a substitute for
# full ten-class AUC. Output: four-model validation leaderboard and comparison chart.

# %%
fitted = {}
validation_rows, validation_auc_rows = [], []
for name, estimator in candidates.items():
    pipeline = Pipeline([('features', mi.build_preprocessor(X_fit)),
                         ('classifier', clone(estimator))])
    pipeline.fit(X_fit, y_fit)
    predictions = pipeline.predict(X_val)
    metrics = mi.compute_metrics(y_val, predictions, name, 'validation')
    auc_summary, auc_rows = mi.compute_roc_metrics(
        y_val, pipeline.predict_proba(X_val), pipeline.named_steps['classifier'].classes_)
    metrics.update(auc_summary)
    validation_rows.append(metrics)
    validation_auc_rows.extend(dict(row, model=name, split='validation') for row in auc_rows)
    fitted[name] = pipeline
leaderboard = pd.DataFrame(validation_rows).sort_values(
    ['macro_f1', 'weighted_f1', 'accuracy'], ascending=False).reset_index(drop=True)
best_name = leaderboard.iloc[0]['model']
leaderboard['selected_for_test'] = leaderboard.model == best_name
display(leaderboard)
fig, ax = plt.subplots(figsize=(10, 4))
leaderboard.set_index('model')[['accuracy', 'macro_f1', 'roc_auc_ovr_macro_supported']].rename(
    columns={'accuracy': 'Accuracy', 'macro_f1': 'Macro F1',
             'roc_auc_ovr_macro_supported': 'ROC-AUC (supported)'}).plot.bar(ax=ax, rot=15)
ax.set(title=f'Validation comparison\n{RUN_LABEL}', ylabel='Metric value', ylim=(0, 1))
fig.tight_layout()
fig.savefig(OUT / 'validation_comparison.png', dpi=150)
display(fig)
plt.close(fig)

# %% [markdown]
# ## 8. Freeze selection and refit the winner
# Choose by validation Macro F1; validation weighted F1 and accuracy break ties.
# Clone the selected estimator and fit a NEW preprocessor on the full training set.
# Other models do not receive held-out test scores. Output: winner and refit pipeline.

# %%
print('Winner selected using validation only:', best_name)
winner = Pipeline([('features', mi.build_preprocessor(X_train)),
                   ('classifier', clone(candidates[best_name]))])
winner.fit(X_train, y_train)
display(winner)

# %% [markdown]
# ## 9. Final held-out evaluation
# This is the single test-evaluation stage of this run. Do not tune the model after
# viewing these results and continue treating this same test split as untouched.
# Accuracy measures overall correctness; macro metrics emphasize all classes equally;
# weighted F1 emphasizes frequent classes. ROC-AUC assesses probability ranking,
# while F1 assesses the resulting predicted categories.
# Output: final metrics, class report and per-class ROC-AUC with undefined reasons.

# %%
test_predictions = winner.predict(X_test)
test_probabilities = winner.predict_proba(X_test)
model_classes = winner.named_steps['classifier'].classes_
test_metrics = mi.compute_metrics(y_test, test_predictions, best_name, 'heldout_test')
auc_summary, test_auc_rows = mi.compute_roc_metrics(y_test, test_probabilities, model_classes)
test_metrics.update(auc_summary)
per_class = classification_report(y_test, test_predictions, labels=mi.CLASSES,
                                  zero_division=0, output_dict=True)
print(RUN_LABEL)
display(pd.DataFrame([test_metrics]))
display(pd.DataFrame(per_class).T)
display(pd.DataFrame(test_auc_rows))

# %% [markdown]
# ## 10. Confusion matrix and ROC curves
# The confusion matrix is normalized by true class: its rows show how each actual
# category was predicted. Each ROC curve contrasts one class against all others.
# These figures describe this run's data source, not operational security effectiveness.
# Output: inline figures and two saved PNG files.

# %%
title = f'{best_name} — {RUN_LABEL}'
mi.save_plot_confusion(y_test, test_predictions, OUT / 'confusion_matrix.png', title)
mi.save_plot_roc(y_test, test_probabilities, model_classes, test_auc_rows, OUT / 'roc_curves.png', title)
display(Image(filename=str(OUT / 'confusion_matrix.png')))
display(Image(filename=str(OUT / 'roc_curves.png')))

# %% [markdown]
# ## 11. Explore global feature importance
# Tree impurity importance is associative and can favour high-cardinality features;
# it does not explain a particular alert or prove causation. If Logistic Regression
# wins, show the highest-ranked validation tree model and identify it explicitly.
# Output: top transformed features, source model and global importance plot.

# %%
importance_model = best_name
importance_pipeline = winner
if not hasattr(winner.named_steps['classifier'], 'feature_importances_'):
    importance_model = next(name for name in leaderboard.model
                            if hasattr(fitted[name].named_steps['classifier'], 'feature_importances_'))
    importance_pipeline = fitted[importance_model]
importance = pd.DataFrame({
    'feature': importance_pipeline.named_steps['features'].get_feature_names_out(),
    'importance': importance_pipeline.named_steps['classifier'].feature_importances_,
    'model': importance_model,
}).sort_values('importance', ascending=False)
print('Importance source model:', importance_model)
display(importance.head(15))
fig, ax = plt.subplots(figsize=(10, 5))
importance.head(12).iloc[::-1].plot.barh(x='feature', y='importance', ax=ax, legend=False, color='#247b83')
ax.set_title(f'Global impurity importance (not causal) — {importance_model}\n{RUN_LABEL}')
fig.tight_layout()
fig.savefig(OUT / 'feature_importance.png', dpi=150)
display(fig)
plt.close(fig)

# %% [markdown]
# ## 12. Build an illustrative analyst review queue
# Predicted attack categories and uncalibrated maximum class scores are combined
# with example severity weights. These hand-assigned priorities are not measured
# business risk. Numeric deviations use training statistics only. No blocking occurs.
# Output: up to 100 prioritized flows; preview the first ten for human review.

# %%
alerts = pd.DataFrame({'flow_index': np.arange(len(X_test)),
                       'predicted_attack': test_predictions,
                       'model_confidence': np.round(test_probabilities.max(axis=1), 4)})
alerts['severity_assumption_0_to_5'] = alerts.predicted_attack.map(mi.SEVERITY)
alerts['priority_score_illustrative'] = (
    alerts.model_confidence * alerts.severity_assumption_0_to_5).round(4)
alerts['needs_human_review'] = (alerts.model_confidence < .60) | (alerts.predicted_attack != 'Normal')
alerts = alerts.sort_values('priority_score_illustrative', ascending=False).head(100).copy()
alerts['top_feature_deviations_illustrative'] = mi.compute_local_feature_signals(
    X_train, X_test.iloc[alerts.flow_index.to_numpy()], top_k=3)
display(alerts.head(10))

# %% [markdown]
# ## 13. Export results and record provenance
# Notebook results live in `outputs/notebook/`, separate from CLI results in `outputs/`.
# Metadata is written last as a completion marker. Raw records are not exported here.
# The notebook includes synthetic training previews and inline images for learning.
# Before publishing a real/private run, clear notebook outputs and inspect metadata.
# Output: artifact manifest and a compact run summary.

# %%
leaderboard.to_csv(OUT / 'validation_leaderboard.csv', index=False)
leaderboard.to_csv(OUT / 'model_comparison.csv', index=False)
pd.DataFrame(validation_auc_rows).to_csv(OUT / 'validation_roc_auc.csv', index=False)
pd.DataFrame([test_metrics]).to_csv(OUT / 'heldout_test_metrics.csv', index=False)
pd.DataFrame(test_auc_rows).to_csv(OUT / 'heldout_test_roc_auc.csv', index=False)
importance.to_csv(OUT / 'feature_importance.csv', index=False)
from dashboard import safe_csv
(OUT / 'example_soc_alerts.csv').write_text(safe_csv(alerts), encoding='utf-8')
(OUT / 'classification_report.txt').write_text(
    RUN_LABEL + '\n' + classification_report(y_test, test_predictions, labels=mi.CLASSES, zero_division=0),
    encoding='utf-8')
metadata = {
    'project': 'Axynetra', 'dataset': RUN_LABEL, 'is_synthetic_smoke_test': SMOKE_TEST,
    'seed': SEED, 'versions': versions, 'provenance': provenance,
    'max_train': MAX_TRAIN, 'max_test': MAX_TEST,
    'train_rows': len(train), 'fit_train_rows': len(fit), 'val_rows': len(validation), 'test_rows': len(test),
    'predictor_count': len(features), 'excluded_leakage_columns': sorted(mi.LEAKAGE_COLUMNS),
    'best_model': best_name, 'feature_importance_model': importance_model,
    'validation_split_info': split_info, 'validation_leaderboard': leaderboard.to_dict('records'),
    'heldout_test_evaluation': test_metrics, 'heldout_test_per_class': per_class,
    'heldout_test_roc_auc': test_auc_rows,
    'selection_protocol': 'Validation Macro F1; refit winner on full train; one held-out evaluation per run.',
}
(OUT / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
display(pd.DataFrame([{'artifact': path.name, 'bytes': path.stat().st_size}
                      for path in sorted(OUT.iterdir()) if path.is_file()]))
print(f'Completed: {RUN_LABEL}')
print(f'Winner: {best_name}; test Macro F1: {test_metrics["macro_f1"]:.4f}; '
      f'test accuracy: {test_metrics["accuracy"]:.4f}')

# %% [markdown]
# ## 14. Interpretation, use cases and next experiments
# - **Teaching:** compare supervised classifiers, preprocessing and validation protocols.
# - **Offline benchmark research:** evaluate labelled UNSW-NB15 flows with documented splits.
# - **Analyst triage prototype:** illustrate attack categories, review priorities and feature signals.
# - **Rare-class study:** inspect Worms/Shellcode recall and macro metrics instead of accuracy alone.
#
# Synthetic results verify software behaviour only. Dataset shift, near-duplicate
# flows, class imbalance and uncalibrated probabilities limit operational conclusions.
# Real data and independent validation are required before security use. An autoencoder
# is a possible future extension, not part of this notebook's implemented pipeline.
#
# References: UNSW-NB15 dataset, Moustafa & Slay (MilCIS, 2015);
# scikit-learn Pipeline, ColumnTransformer and roc_auc_score documentation.
