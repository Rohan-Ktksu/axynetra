#!/usr/bin/env python3
"""Axynetra — CSMA Option 4: Explainable Multi-Class Attack Classification & Intelligent Alert Triage.

Data: UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv.
No network traffic is captured or injected; this is an OFFLINE network-flow classification
research prototype using stored CSV data.
Run: python multiclass_intrusion.py --train data/UNSW_NB15_training-set.csv --test data/UNSW_NB15_testing-set.csv
Test installation: python multiclass_intrusion.py --smoke-test
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
CLASSES = ["Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
           "Reconnaissance", "Shellcode", "Worms"]
SEVERITY = {"Normal": 0, "Analysis": 2, "Backdoor": 5, "DoS": 4,
            "Exploits": 5, "Fuzzers": 2, "Generic": 3, "Reconnaissance": 3,
            "Shellcode": 5, "Worms": 5}
LEAKAGE_COLUMNS = {"attack_cat", "label", "id"}

_CANONICAL_MAP = {c.lower(): c for c in CLASSES}
_CANONICAL_MAP["backdoors"] = "Backdoor"


def normalized_label(s):
    """Normalize UNSW-NB15 attack category strings (e.g. 'Backdoors' -> 'Backdoor')."""
    if pd.isna(s):
        return s
    cleaned = str(s).strip()
    lower = cleaned.lower()
    if lower in _CANONICAL_MAP:
        return _CANONICAL_MAP[lower]
    return cleaned


def get_predictor_columns(train_df, test_df):
    """Return common predictor columns while strictly excluding target/ID leakage columns."""
    for frame in (train_df, test_df):
        normalized = [str(c).strip().lower() for c in frame.columns]
        if len(normalized) != len(set(normalized)):
            raise ValueError('Duplicate or ambiguous column names are not supported.')
        if any(re.fullmatch(r'(attack_cat|label|id)\.\d+', c) for c in normalized):
            raise ValueError('Duplicated target/ID CSV headers could leak labels into predictors.')
    required = [c for c in train_df.columns if str(c).strip().lower() not in LEAKAGE_COLUMNS]
    if any(c not in test_df.columns for c in required):
        raise ValueError('Test data is missing training predictor columns.')
    feature_cols = [
        c for c in train_df.columns
        if c in test_df.columns and str(c).strip().lower() not in LEAKAGE_COLUMNS
    ]
    if not feature_cols:
        raise ValueError("Train and test have no common predictor columns.")
    return feature_cols


def stratified_subset(df, limit, key="attack_cat"):
    """Approximate stratified sample without fitting anything on test data.

    Preserves rare classes: any class with >= 2 rows in df retains at least 2 rows
    whenever limit >= 2 * n_classes, ensuring downstream validation splitting can
    place at least 1 row in train and 1 row in validation.
    """
    if limit <= 0 or len(df) <= limit:
        return df.copy().reset_index(drop=True)
    groups = list(df.groupby(key, sort=False))
    if limit < len(groups):
        raise ValueError('Sample limit must be at least the number of classes.')
    sizes = np.array([len(group) for _, group in groups])
    minimum = 2 if limit >= 2 * len(groups) else 1
    counts = np.minimum(sizes, minimum)
    ideal = sizes * (limit / len(df))
    while counts.sum() < limit:
        deficits = np.where(counts < sizes, ideal - counts, -np.inf)
        counts[int(np.argmax(deficits))] += 1
    sampled = pd.concat([group.sample(n=int(n), random_state=SEED)
                         for (_, group), n in zip(groups, counts)], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def stratified_train_val_split(train_df, val_fraction=0.20, key="attack_cat", seed=SEED):
    """Create a leak-free stratified validation split from training data ONLY.

    Rare-class guarantee & fallback behavior:
    - For every class with >= 2 samples in train_df, at least 1 sample is placed in
      validation and at least 1 sample remains in train_sub:
      n_val = max(1, min(n - 1, int(round(n * val_fraction)))).
    - Fallback: If an ultra-rare class has only 1 sample in train_df, it cannot be
      placed in both splits without duplication; it is kept in train_sub so models
      can still learn the class, and the fallback event is logged in split_meta.
    """
    if not (0.05 <= val_fraction <= 0.50):
        raise ValueError("val_fraction must be between 0.05 and 0.50.")
    train_df = train_df.reset_index(drop=True)
    train_parts = []
    val_parts = []
    fallbacks = []
    for cls_name, group in train_df.groupby(key, sort=False):
        n = len(group)
        if n < 2:
            train_parts.append(group)
            fallbacks.append(
                f"Class '{cls_name}' had only {n} sample(s) in training data; "
                "kept in train_sub (0 in val_sub) to avoid synthetic duplication."
            )
        else:
            n_val = max(1, min(n - 1, int(round(n * val_fraction))))
            val_sample = group.sample(n=n_val, random_state=seed)
            train_sample = group.drop(index=val_sample.index)
            val_parts.append(val_sample)
            train_parts.append(train_sample)

    train_sub = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if val_parts:
        val_sub = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        val_sub = train_df.iloc[0:0].copy()

    split_meta = {
        "val_fraction": val_fraction,
        "fit_train_rows": int(len(train_sub)),
        "val_rows": int(len(val_sub)),
        "fit_train_class_counts": train_sub[key].value_counts().to_dict(),
        "val_class_counts": val_sub[key].value_counts().to_dict(),
        "rare_class_fallbacks": fallbacks,
        "rare_class_policy": (
            "Classes with >=2 training rows are guaranteed >=1 row in train_sub and >=1 row in val_sub. "
            "Classes with 1 row fall back to train_sub only and are logged in rare_class_fallbacks."
        ),
    }
    return train_sub, val_sub, split_meta


def load_original(train_path, test_path, max_train, max_test):
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    for name, d in [('train', train), ('test', test)]:
        if 'attack_cat' not in d:
            raise ValueError(f'{name} must contain attack_cat (UNSW-NB15 split).')
        d['attack_cat'] = d['attack_cat'].map(normalized_label)
        validate_labels(d, name)
    check_overlap(train, test, get_predictor_columns(train, test))
    train = stratified_subset(train, max_train)
    test = stratified_subset(test, max_test)
    return train, test


def validate_labels(frame, name):
    if frame.empty or 'attack_cat' not in frame:
        raise ValueError(f'{name} must be nonempty and contain attack_cat.')
    if not frame['attack_cat'].isin(CLASSES).all():
        raise ValueError(f'{name} has missing or unsupported attack categories; labels must not be guessed.')


def check_overlap(train, test, features):
    """Reject identical predictor records crossing evaluation boundaries."""
    left = pd.util.hash_pandas_object(train[features], index=False)
    right = pd.util.hash_pandas_object(test[features], index=False)
    if left.isin(right).any():
        raise ValueError('Identical predictor rows occur in both train and test; resolve overlap before evaluation.')


def make_smoke_data():
    """Synthetic data solely for a pipeline smoke test, NEVER UNSW evaluation."""
    X, y = make_classification(n_samples=2200, n_features=22, n_informative=17,
                               n_redundant=3, n_classes=10, n_clusters_per_class=1,
                               flip_y=0.08, random_state=SEED)
    df = pd.DataFrame(X, columns=[f'feature_{i:02d}' for i in range(22)])
    df['proto'] = np.where(df.feature_01 > 0, 'tcp', 'udp')
    df['service'] = np.where(df.feature_05 > 0, 'dns', 'http')
    df['attack_cat'] = [CLASSES[int(k)] for k in y]
    return df.iloc[:1700].copy().reset_index(drop=True), df.iloc[1700:].copy().reset_index(drop=True)


def build_preprocessor(X):
    cat = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype) in ('category', 'string', 'str')]
    num = [c for c in X.columns if c not in cat]
    transformers = []
    if num:
        transformers.append(('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())]), num))
    if cat:
        transformers.append(('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))]), cat))
    preprocessing = ColumnTransformer(transformers, sparse_threshold=0.3)
    return preprocessing


def model_choices():
    return {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=22, min_samples_leaf=4, class_weight='balanced', random_state=SEED),
        'Random Forest': RandomForestClassifier(
            n_estimators=90, max_depth=25, min_samples_leaf=2,
            class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=90, min_samples_leaf=2,
            class_weight='balanced', n_jobs=-1, random_state=SEED),
        'Logistic Regression': LogisticRegression(
            max_iter=800, class_weight='balanced', solver='lbfgs', random_state=SEED)
    }


def compute_metrics(y_true, y_pred, model_name, split_name):
    return {
        'model': model_name,
        'split': split_name,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, labels=CLASSES, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, labels=CLASSES, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=CLASSES, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }


def save_plot_confusion(y_true, y_pred, out, title):
    present = set(y_true) | set(y_pred)
    labels = [c for c in CLASSES if c in present] + sorted(present - set(CLASSES))
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=True, values_format='.2f')
    ax.set_title(title + ' — row-normalized confusion matrix')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_local_feature_signals(X_train_df, X_eval_df, top_k=3):
    """Compute illustrative per-row numeric z-score deviations from training medians.

    Fitted strictly on X_train_df (no test leakage). Provides an analyst-friendly
    exploratory local indicator of which numeric features deviated most on a given flow.
    """
    num_cols = [
        c for c in X_train_df.columns
        if X_train_df[c].dtype != 'object' and str(X_train_df[c].dtype) not in ('category', 'string', 'str')
    ]
    if not num_cols or len(X_eval_df) == 0:
        return ["n/a"] * len(X_eval_df)
    tr_num = X_train_df[num_cols].apply(pd.to_numeric, errors='coerce')
    ev_num = X_eval_df[num_cols].apply(pd.to_numeric, errors='coerce')
    medians = tr_num.median(axis=0).fillna(0.0)
    stds = tr_num.std(axis=0).replace(0, 1.0).fillna(1.0)
    z_mat = ((ev_num.fillna(medians) - medians) / stds).to_numpy(dtype=float)
    abs_z = np.abs(z_mat)
    col_names = np.array(num_cols)
    k = min(top_k, len(num_cols))
    signals = []
    for row_idx in range(len(X_eval_df)):
        top_indices = np.argsort(abs_z[row_idx])[::-1][:k]
        parts = [f"{col_names[i]}({z_mat[row_idx, i]:+.1f}z)" for i in top_indices]
        signals.append(", ".join(parts))
    return signals


def run(train, test, output, smoke, val_fraction=0.20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Leak-free predictor selection: strictly exclude attack_cat, label, and id.
    feature_cols = get_predictor_columns(train, test)
    train = train.copy()
    test = test.copy()
    train['attack_cat'] = train['attack_cat'].map(normalized_label)
    test['attack_cat'] = test['attack_cat'].map(normalized_label)
    validate_labels(train, 'train')
    validate_labels(test, 'test')
    for name, frame in [('train', train), ('test', test)]:
        numeric = frame[feature_cols].select_dtypes(include='number')
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f'{name} contains infinite predictors; replace or investigate invalid measurements.')
    check_overlap(train, test, feature_cols)
    # Repeated records must not enter both fit-training and validation.
    if train.duplicated(subset=feature_cols).any():
        raise ValueError('Duplicate training predictors must be resolved before validation splitting.')
    if train['attack_cat'].nunique() < 2:
        raise ValueError('Training requires at least two classes.')

    # 2. Stratified validation split created strictly from training data.
    train_sub, val_sub, split_meta = stratified_train_val_split(
        train, val_fraction=val_fraction, key='attack_cat', seed=SEED
    )
    if val_sub.empty:
        raise ValueError('No validation rows: provide at least two records in a class.')
    # Metadata is the completion marker. Never display old results after a failed rerun.
    (output / 'run_metadata.json').unlink(missing_ok=True)
    for filename in ('feature_importance.csv', 'feature_importance.png'):
        (output / filename).unlink(missing_ok=True)

    X_tr_sub = train_sub[feature_cols].copy()
    y_tr_sub = train_sub['attack_cat'].astype(str)
    X_val = val_sub[feature_cols].copy()
    y_val = val_sub['attack_cat'].astype(str)

    X_tr_full = train[feature_cols].copy()
    y_tr_full = train['attack_cat'].astype(str)
    X_te = test[feature_cols].copy()
    y_te = test['attack_cat'].astype(str)

    tag = 'SYNTHETIC SMOKE TEST - NOT UNSW PERFORMANCE' if smoke else 'USER-SUPPLIED CSV SPLIT (SOURCE NOT VERIFIED)'
    print('RUN TYPE:', tag)
    print(f"Total Train rows: {len(train)} (Fit Train: {len(train_sub)} | Validation: {len(val_sub)}) "
          f"| Untouched Test rows: {len(test)} | Predictors: {len(feature_cols)}")
    print('Fit-Train categories:', y_tr_sub.value_counts().to_dict())
    print('Validation categories:', y_val.value_counts().to_dict())
    print('Held-Out Test categories:', y_te.value_counts().to_dict())
    if split_meta['rare_class_fallbacks']:
        for note in split_meta['rare_class_fallbacks']:
            print('RARE-CLASS FALLBACK:', note)

    # 3. Model comparison & selection on VALIDATION split ONLY (held-out test set is untouched).
    print('\n--- Phase 1: Stratified Validation Leaderboard (Model Selection) ---')
    val_results = []
    val_fitted = {}
    val_predictions = {}
    choices = model_choices()
    for name, estimator in choices.items():
        pipeline = Pipeline([
            ('features', build_preprocessor(X_tr_sub)),
            ('classifier', clone(estimator))
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            pipeline.fit(X_tr_sub, y_tr_sub)
        val_pred = pipeline.predict(X_val)
        metrics_row = compute_metrics(y_val, val_pred, name, split_name='validation')
        val_results.append(metrics_row)
        val_fitted[name] = pipeline
        val_predictions[name] = val_pred
        print(f"[VAL] {name:19s} | Accuracy={metrics_row['accuracy']:.4f} | "
              f"Macro F1={metrics_row['macro_f1']:.4f} | Weighted F1={metrics_row['weighted_f1']:.4f}")

    df_val = pd.DataFrame(val_results).sort_values(
        by=['macro_f1', 'weighted_f1', 'accuracy'], ascending=False
    ).reset_index(drop=True)
    best = str(df_val.iloc[0]['model'])
    df_val['selected_for_test'] = df_val['model'] == best

    # Save validation leaderboard (and model_comparison.csv for backward compatibility)
    df_val.to_csv(output / 'validation_leaderboard.csv', index=False)
    df_val.to_csv(output / 'model_comparison.csv', index=False)
    print('SELECTED BEST MODEL BY VALIDATION MACRO F1:', best)

    # 4. Refit ONLY the selected winning model on full training data and evaluate ONCE on held-out test set.
    print('\n--- Phase 2: One-Time Held-Out Test Set Evaluation (Selected Winner Only) ---')
    winner = Pipeline([
        ('features', build_preprocessor(X_tr_full)),
        ('classifier', clone(choices[best]))
    ])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        winner.fit(X_tr_full, y_tr_full)

    test_pred = winner.predict(X_te)
    test_metrics = compute_metrics(y_te, test_pred, best, split_name='heldout_test')
    pd.DataFrame([test_metrics]).to_csv(output / 'heldout_test_metrics.csv', index=False)
    print(f"[TEST] {best:18s} | Accuracy={test_metrics['accuracy']:.4f} | "
          f"Macro F1={test_metrics['macro_f1']:.4f} | Weighted F1={test_metrics['weighted_f1']:.4f}")

    report_body = classification_report(y_te, test_pred, labels=CLASSES, zero_division=0)
    report_header = (
        f"Axynetra Evaluation Report ({tag})\n"
        f"Selected Model (by Validation Macro F1): {best}\n"
        f"Evaluation Split: Supplied held-out test set (one evaluation per run; not used for selection)\n"
        f"Train Rows: {len(train)} (Fit: {len(train_sub)}, Val: {len(val_sub)}) | Held-Out Test Rows: {len(test)}\n\n"
    )
    full_report = report_header + report_body
    print('\nHeld-out test classification report:\n' + report_body)
    (output / 'classification_report.txt').write_text(full_report, encoding='utf-8')

    title_prefix = f"{best} ({'Synthetic Smoke Test' if smoke else 'UNSW-NB15 Held-Out Test'})"
    save_plot_confusion(y_te, test_pred, output / 'confusion_matrix.png', title_prefix)

    # 5. Global feature importance for compatible tree models (fitted on training data only).
    tree_source_model = best
    tree_pipeline = winner
    if not hasattr(winner.named_steps['classifier'], 'feature_importances_'):
        tree_candidates = [m for m in df_val['model'] if hasattr(val_fitted[m].named_steps['classifier'], 'feature_importances_')]
        if tree_candidates:
            tree_source_model = tree_candidates[0]
            tree_pipeline = val_fitted[tree_source_model]

    if hasattr(tree_pipeline.named_steps['classifier'], 'feature_importances_'):
        fn = tree_pipeline.named_steps['features'].get_feature_names_out()
        importance = tree_pipeline.named_steps['classifier'].feature_importances_
        fi = pd.DataFrame({
            'feature': fn,
            'importance': importance,
            'model': tree_source_model
        }).sort_values('importance', ascending=False)
        fi.to_csv(output / 'feature_importance.csv', index=False)
        top = fi.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9.2, 5.5))
        ax.barh(top['feature'], top['importance'], color='#1F4B7E')
        run_label = 'Synthetic Smoke' if smoke else 'UNSW-NB15'
        ax.set_title(f"Top 12 global feature importances — {tree_source_model} ({run_label})")
        ax.set_xlabel('Impurity-based global importance (associative, NOT causal or per-alert explanation)')
        fig.tight_layout()
        fig.savefig(output / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 6. Illustrative SOC triage queue; confidence is an uncalibrated model score, not operational risk.
    prob = winner.predict_proba(X_te)
    classes = winner.named_steps['classifier'].classes_
    confidence = np.max(prob, axis=1)
    predicted = classes[np.argmax(prob, axis=1)]
    alert = pd.DataFrame({
        'flow_index': np.arange(len(X_te)),
        'predicted_attack': predicted,
        'model_confidence': np.round(confidence, 4),
        'severity_assumption_0_to_5': [SEVERITY.get(c, 3) for c in predicted],
    })
    alert['priority_score_illustrative'] = np.round(
        alert['model_confidence'] * alert['severity_assumption_0_to_5'], 4
    )
    alert['needs_human_review'] = (alert['model_confidence'] < 0.60) | (alert['predicted_attack'] != 'Normal')
    top_alerts = alert.sort_values('priority_score_illustrative', ascending=False).head(100).copy()
    top_X = X_te.iloc[top_alerts['flow_index'].values]
    top_alerts['top_feature_deviations_illustrative'] = compute_local_feature_signals(X_tr_full, top_X, top_k=3)
    top_alerts.to_csv(output / 'example_soc_alerts.csv', index=False)

    per_class_report_dict = classification_report(y_te, test_pred, labels=CLASSES, zero_division=0, output_dict=True)
    metadata = {
        'project': 'Axynetra — Explainable Network Threat Classification & Intelligent Alert Triage',
        'dataset': 'SYNTHETIC SMOKE TEST ONLY' if smoke else 'User-supplied CSV split (source not verified)',
        'is_synthetic_smoke_test': bool(smoke),
        'train_rows': int(len(train)),
        'fit_train_rows': int(len(train_sub)),
        'val_rows': int(len(val_sub)),
        'test_rows': int(len(test)),
        'predictor_count': int(len(feature_cols)),
        'excluded_leakage_columns': sorted(LEAKAGE_COLUMNS),
        'selection_protocol': (
            'Stratified validation split created from training data only. '
            'Candidate models are ranked strictly by validation macro_f1. '
            'The winning model is evaluated once per run on the supplied held-out test set.'
        ),
        'best_model': best,
        'feature_importance_model': tree_source_model,
        'validation_split_info': split_meta,
        'validation_leaderboard': df_val.to_dict(orient='records'),
        'model_comparison': df_val.to_dict(orient='records'),
        'heldout_test_evaluation': test_metrics,
        'heldout_test_per_class': per_class_report_dict,
        'caveat': (
            'Offline CSV analysis prototype only (not a live IDS). '
            'Do not treat model confidence as calibrated probability or deployment risk. '
            'Impurity-based feature importance is global and non-causal.'
        ),
    }
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Output artifacts saved in:', output.resolve())
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Axynetra offline multi-class network intrusion classification (UNSW-NB15).'
    )
    parser.add_argument('--train', type=Path, help='UNSW_NB15_training-set.csv')
    parser.add_argument('--test', type=Path, help='UNSW_NB15_testing-set.csv')
    parser.add_argument('--max-train', type=int, default=50000, help='Set to 0 for full training set')
    parser.add_argument('--max-test', type=int, default=20000, help='Set to 0 for full testing set')
    parser.add_argument('--val-fraction', type=float, default=0.20, help='Validation split fraction from training set (default: 0.20)')
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--smoke-test', action='store_true', help='Run on generated synthetic data for installation validation')
    args = parser.parse_args()
    if args.smoke_test:
        train, test = make_smoke_data()
    elif args.train and args.test:
        train, test = load_original(args.train, args.test, args.max_train, args.max_test)
    else:
        parser.error('Supply --train and --test for actual experiment, or --smoke-test to verify code.')
    run(train, test, args.output, args.smoke_test, val_fraction=args.val_fraction)


if __name__ == '__main__':
    main()
