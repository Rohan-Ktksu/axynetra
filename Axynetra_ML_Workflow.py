# %% [markdown]
# # Axynetra: a complete supervised ML experiment
# **Multi-Class Attack Classification — CSMA Option 4**
#
# This notebook explains each stage rather than hiding training inside one function.
# All ML and plotting code is visible here, using pandas, NumPy, scikit-learn and
# Matplotlib directly. No application helper modules are imported.
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
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve)

ROOT = Path.cwd()
pd.set_option('display.max_columns', 16)
SEED = 42
CLASSES = ['Normal', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers',
           'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
LEAKAGE_COLUMNS = {'attack_cat', 'label', 'id'}
SEVERITY = {'Normal': 0, 'Analysis': 2, 'Backdoor': 5, 'DoS': 4, 'Exploits': 5,
            'Fuzzers': 2, 'Generic': 3, 'Reconnaissance': 3, 'Shellcode': 5, 'Worms': 5}
versions = {name: importlib.metadata.version(name)
            for name in ['numpy', 'pandas', 'scikit-learn', 'matplotlib']}
print('Python:', platform.python_version(), '| Seed:', SEED)
display(pd.Series(versions, name='version').to_frame())
print('Classes:', ', '.join(CLASSES))

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
    X, y = make_classification(n_samples=2200, n_features=22, n_informative=17,
                               n_redundant=3, n_classes=10, n_clusters_per_class=1,
                               flip_y=0.08, random_state=SEED)
    data = pd.DataFrame(X, columns=[f'feature_{i:02d}' for i in range(22)])
    data['proto'] = np.where(data.feature_01 > 0, 'tcp', 'udp')
    data['service'] = np.where(data.feature_05 > 0, 'dns', 'http')
    data['attack_cat'] = np.array(CLASSES)[y]
    train = data.iloc[:1700].copy().reset_index(drop=True)
    test = data.iloc[1700:].copy().reset_index(drop=True)
    provenance = {'source': 'Synthetic sklearn.make_classification; not UNSW-NB15'}
    RUN_LABEL = 'SYNTHETIC DEMONSTRATION — NOT UNSW-NB15 PERFORMANCE'
else:
    if not TRAIN_PATH.is_file() or not TEST_PATH.is_file():
        raise FileNotFoundError('Download the real train/test CSVs into data/ before disabling SMOKE_TEST.')
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
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
if 'attack_cat' not in train:
    raise ValueError('Training CSV must contain the attack_cat target column.')
class_counts = train.attack_cat.value_counts(dropna=False)
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
label_map = {label.lower(): label for label in CLASSES}
label_map['backdoors'] = 'Backdoor'
for frame in [train, test]:
    names = [str(column).strip().lower() for column in frame.columns]
    if len(names) != len(set(names)) or any(
            re.fullmatch(r'(attack_cat|label|id)\.\d+', name) for name in names):
        raise ValueError('Duplicate or ambiguous CSV columns may cause target leakage.')
features = [column for column in train if column.strip().lower() not in LEAKAGE_COLUMNS]
if not features or any(column not in test for column in features):
    raise ValueError('Training/test CSVs need matching predictor columns.')
for split_name, frame in [('training', train), ('test', test)]:
    if frame.empty or 'attack_cat' not in frame:
        raise ValueError(f'{split_name} must be nonempty and contain attack_cat.')
    cleaned = frame.attack_cat.astype('string').str.strip().str.lower()
    frame['attack_cat'] = cleaned.map(label_map)
    if not frame.attack_cat.isin(CLASSES).all():
        raise ValueError(f'{split_name} contains missing or unknown attack categories.')
    numeric = frame[features].select_dtypes(include='number')
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError(f'{split_name} has infinite predictors; investigate them first.')
train_hashes = pd.util.hash_pandas_object(train[features], index=False)
test_hashes = pd.util.hash_pandas_object(test[features], index=False)
if train_hashes.isin(test_hashes).any():
    raise ValueError('Identical predictors occur in both training and test data.')
if train.duplicated(subset=features).any():
    raise ValueError('Resolve duplicate training predictors before validation splitting.')
if train.attack_cat.nunique() < 2:
    raise ValueError('Training needs at least two classes.')
# Optional sampling happens after full-input overlap checks. Explicit per-class
# allocations enforce the requested size without accidentally dropping rare classes.
for split_name, limit in [('train', MAX_TRAIN), ('test', MAX_TEST)]:
    frame = train if split_name == 'train' else test
    if limit > 0 and len(frame) > limit:
        groups = list(frame.groupby('attack_cat', sort=False))
        if limit < len(groups):
            raise ValueError('Row limit is too small to represent all classes.')
        sizes = np.array([len(group) for _, group in groups])
        minimum = 2 if limit >= 2 * len(groups) else 1
        counts = np.minimum(sizes, minimum)
        ideal = sizes * limit / len(frame)
        while counts.sum() < limit:
            deficits = np.where(counts < sizes, ideal - counts, -np.inf)
            counts[np.argmax(deficits)] += 1
        sampled = pd.concat([group.sample(n=int(count), random_state=SEED)
                             for (_, group), count in zip(groups, counts)], ignore_index=True)
        frame = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)
        if split_name == 'train':
            train = frame
        else:
            test = frame
assert not set(c.strip().lower() for c in features) & LEAKAGE_COLUMNS
print('Input checks passed. Predictor count:', len(features))
print('Excluded:', ', '.join(sorted(LEAKAGE_COLUMNS)))
print('Predictors:', ', '.join(features))

# %% [markdown]
# ## 5. Create fit-training and validation subsets
# Stratify within the training data, approximately 80/20. Classes with at least two
# examples retain one in each subset; singleton classes stay in fit-training with a
# documented fallback. Integer rounding means the ratio is approximate.
# Output: fit/validation sizes and class coverage. The held-out test stays separate.

# %%
if not .05 <= VAL_FRACTION <= .5:
    raise ValueError('VAL_FRACTION must be between 0.05 and 0.50.')
train = train.reset_index(drop=True)
fit_parts, validation_parts, fallbacks = [], [], []
for label, group in train.groupby('attack_cat', sort=False):
    if len(group) == 1:
        fit_parts.append(group)
        fallbacks.append(f'{label}: singleton kept in fit-training only')
    else:
        n_validation = max(1, min(len(group)-1, round(len(group) * VAL_FRACTION)))
        held_for_validation = group.sample(n=n_validation, random_state=SEED)
        validation_parts.append(held_for_validation)
        fit_parts.append(group.drop(index=held_for_validation.index))
fit = pd.concat(fit_parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
validation = (pd.concat(validation_parts, ignore_index=True).sample(frac=1, random_state=SEED)
              .reset_index(drop=True)) if validation_parts else train.iloc[:0].copy()
split_info = {'val_fraction': VAL_FRACTION, 'fit_train_rows': len(fit), 'val_rows': len(validation),
              'fit_train_class_counts': fit.attack_cat.value_counts().to_dict(),
              'val_class_counts': validation.attack_cat.value_counts().to_dict(),
              'rare_class_fallbacks': fallbacks}
if validation.empty:
    raise ValueError('No validation rows; supply more training examples.')
X_fit, y_fit = fit[features], fit.attack_cat
X_val, y_val = validation[features], validation.attack_cat
X_train, y_train = train[features], train.attack_cat
X_test, y_test = test[features], test.attack_cat
display(pd.DataFrame({'fit_training': y_fit.value_counts(), 'validation': y_val.value_counts()})
        .reindex(CLASSES).fillna(0).astype(int))
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
categorical_columns = X_fit.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
numeric_columns = [column for column in features if column not in categorical_columns]
numeric_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])
categorical_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore')),
])
preprocessor = ColumnTransformer([
    ('numeric', numeric_transformer, numeric_columns),
    ('categorical', categorical_transformer, categorical_columns),
], sparse_threshold=.3)
display(preprocessor)
candidates = {
    'Decision Tree': DecisionTreeClassifier(max_depth=22, min_samples_leaf=4,
                                           class_weight='balanced', random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=90, max_depth=25, min_samples_leaf=2,
                                            class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
    'Extra Trees': ExtraTreesClassifier(n_estimators=90, min_samples_leaf=2,
                                       class_weight='balanced', n_jobs=-1, random_state=SEED),
    'Logistic Regression': LogisticRegression(max_iter=800, class_weight='balanced',
                                               solver='lbfgs', random_state=SEED),
}
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
    pipeline = Pipeline([('features', clone(preprocessor)),
                         ('classifier', clone(estimator))])
    pipeline.fit(X_fit, y_fit)
    predictions = pipeline.predict(X_val)
    metrics = {'model': name, 'split': 'validation',
               'accuracy': float(accuracy_score(y_val, predictions)),
               'macro_precision': float(precision_score(y_val, predictions, labels=CLASSES, average='macro', zero_division=0)),
               'macro_recall': float(recall_score(y_val, predictions, labels=CLASSES, average='macro', zero_division=0)),
               'macro_f1': float(f1_score(y_val, predictions, labels=CLASSES, average='macro', zero_division=0)),
               'weighted_f1': float(f1_score(y_val, predictions, average='weighted', zero_division=0))}
    probabilities = pipeline.predict_proba(X_val)
    learned_classes = pipeline.named_steps['classifier'].classes_.tolist()
    auc_rows = []
    for label in CLASSES:
        binary = y_val.to_numpy() == label
        positives = int(binary.sum())
        reason = ('no_positive_examples' if positives == 0 else
                  'no_negative_examples' if positives == len(y_val) else
                  'class_not_learned' if label not in learned_classes else '')
        auc = None if reason else float(roc_auc_score(binary, probabilities[:, learned_classes.index(label)]))
        auc_rows.append({'class': label, 'support': positives, 'roc_auc_ovr': auc, 'status': reason or 'defined'})
    defined = [row['roc_auc_ovr'] for row in auc_rows if row['roc_auc_ovr'] is not None]
    metrics.update(roc_auc_ovr_macro=float(np.mean(defined)) if len(defined) == len(CLASSES) else None,
                   roc_auc_ovr_macro_supported=float(np.mean(defined)) if defined else None,
                   roc_auc_defined_classes=len(defined), roc_auc_total_classes=len(CLASSES))
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
winner = Pipeline([('features', clone(preprocessor)),
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
test_metrics = {'model': best_name, 'split': 'heldout_test',
                'accuracy': float(accuracy_score(y_test, test_predictions)),
                'macro_precision': float(precision_score(y_test, test_predictions, labels=CLASSES, average='macro', zero_division=0)),
                'macro_recall': float(recall_score(y_test, test_predictions, labels=CLASSES, average='macro', zero_division=0)),
                'macro_f1': float(f1_score(y_test, test_predictions, labels=CLASSES, average='macro', zero_division=0)),
                'weighted_f1': float(f1_score(y_test, test_predictions, average='weighted', zero_division=0))}
test_auc_rows = []
for label in CLASSES:
    binary = y_test.to_numpy() == label
    positives = int(binary.sum())
    reason = ('no_positive_examples' if positives == 0 else
              'no_negative_examples' if positives == len(y_test) else
              'class_not_learned' if label not in model_classes else '')
    auc = None if reason else float(roc_auc_score(binary, test_probabilities[:, list(model_classes).index(label)]))
    test_auc_rows.append({'class': label, 'support': positives, 'roc_auc_ovr': auc, 'status': reason or 'defined'})
defined = [row['roc_auc_ovr'] for row in test_auc_rows if row['roc_auc_ovr'] is not None]
test_metrics.update(roc_auc_ovr_macro=float(np.mean(defined)) if len(defined) == len(CLASSES) else None,
                    roc_auc_ovr_macro_supported=float(np.mean(defined)) if defined else None,
                    roc_auc_defined_classes=len(defined), roc_auc_total_classes=len(CLASSES))
per_class = classification_report(y_test, test_predictions, labels=CLASSES,
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
fig, ax = plt.subplots(figsize=(10, 8))
matrix = confusion_matrix(y_test, test_predictions, labels=CLASSES, normalize='true')
ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(
    ax=ax, cmap='Blues', xticks_rotation=45, values_format='.2f')
ax.set_title(f'{best_name}: row-normalized confusion matrix\n{RUN_LABEL}')
fig.tight_layout()
fig.savefig(OUT / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
display(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 7))
for row in test_auc_rows:
    if row['roc_auc_ovr'] is None:
        continue
    label = row['class']
    binary = y_test.to_numpy() == label
    column = list(model_classes).index(label)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(binary, test_probabilities[:, column])
    ax.plot(false_positive_rate, true_positive_rate, label=f"{label}: AUC={row['roc_auc_ovr']:.3f}")
ax.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
ax.set(xlabel='False positive rate', ylabel='True positive rate',
       title=f'{best_name}: one-vs-rest ROC curves\n{RUN_LABEL}')
ax.legend(loc='lower right', fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'roc_curves.png', dpi=150, bbox_inches='tight')
display(fig)
plt.close(fig)

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
alerts['severity_assumption_0_to_5'] = alerts.predicted_attack.map(SEVERITY)
alerts['priority_score_illustrative'] = (
    alerts.model_confidence * alerts.severity_assumption_0_to_5).round(4)
alerts['needs_human_review'] = (alerts.model_confidence < .60) | (alerts.predicted_attack != 'Normal')
alerts = alerts.sort_values('priority_score_illustrative', ascending=False).head(100).copy()
if numeric_columns:
    training_numeric = X_train[numeric_columns].apply(pd.to_numeric, errors='coerce')
    alert_numeric = X_test.iloc[alerts.flow_index.to_numpy()][numeric_columns].apply(pd.to_numeric, errors='coerce')
    medians = training_numeric.median().fillna(0)
    standard_deviations = training_numeric.std().replace(0, 1).fillna(1)
    deviations = (alert_numeric.fillna(medians) - medians) / standard_deviations
    signals = []
    for _, row in deviations.iterrows():
        largest = row.abs().nlargest(min(3, len(numeric_columns))).index
        signals.append(', '.join(f'{column}({row[column]:+.1f}z)' for column in largest))
    alerts['top_feature_deviations_illustrative'] = signals
else:
    alerts['top_feature_deviations_illustrative'] = 'n/a'
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
# Neutralize spreadsheet formulas in any string data, without changing numeric values.
export_alerts = alerts.copy()
for column in export_alerts.select_dtypes(include=['object', 'string']).columns:
    export_alerts[column] = export_alerts[column].map(
        lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')) else value)
export_alerts.to_csv(OUT / 'example_soc_alerts.csv', index=False)
(OUT / 'classification_report.txt').write_text(
    RUN_LABEL + '\n' + classification_report(y_test, test_predictions, labels=CLASSES, zero_division=0),
    encoding='utf-8')
metadata = {
    'project': 'Axynetra', 'dataset': RUN_LABEL, 'is_synthetic_smoke_test': SMOKE_TEST,
    'seed': SEED, 'versions': versions, 'provenance': provenance,
    'max_train': MAX_TRAIN, 'max_test': MAX_TEST,
    'train_rows': len(train), 'fit_train_rows': len(fit), 'val_rows': len(validation), 'test_rows': len(test),
    'predictor_count': len(features), 'excluded_leakage_columns': sorted(LEAKAGE_COLUMNS),
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
