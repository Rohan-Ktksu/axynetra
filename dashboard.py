"""Axynetra interactive prediction website and offline results viewer.

Tagline: Explainable Network Threat Classification & Intelligent Alert Triage.
Run: python -m streamlit run dashboard.py
Install: python -m pip install streamlit
The default website prepares its model from the bundled reference CSVs.
Pass -- --results outputs to inspect existing command-line artifacts.
"""
from pathlib import Path
import json
import pandas as pd


def load_dashboard_artifacts(root=Path('outputs')):
    """Load and validate offline evaluation artifacts for the Axynetra dashboard.

    Separated from Streamlit UI rendering so artifact loading can be unit-tested directly.
    """
    root = Path(root)
    meta_file = root / 'run_metadata.json'
    if not meta_file.exists():
        raise FileNotFoundError(f'No results found at {meta_file}. Run multiclass_intrusion.py with --output {root} first.')

    meta = json.loads(meta_file.read_text(encoding='utf-8'))
    smoke = 'SYNTHETIC' in str(meta.get('dataset', '')).upper() or bool(meta.get('is_synthetic_smoke_test', False))

    val_csv = root / 'validation_leaderboard.csv'
    comp_csv = root / 'model_comparison.csv'
    leaderboard_path = val_csv if val_csv.exists() else comp_csv
    leaderboard_df = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame(meta.get('model_comparison', []))

    test_csv = root / 'heldout_test_metrics.csv'
    if test_csv.exists():
        test_metrics_df = pd.read_csv(test_csv)
    elif 'heldout_test_evaluation' in meta and isinstance(meta['heldout_test_evaluation'], dict):
        test_metrics_df = pd.DataFrame([meta['heldout_test_evaluation']])
    else:
        test_metrics_df = pd.DataFrame()

    report_path = root / 'classification_report.txt'
    report_text = report_path.read_text(encoding='utf-8') if report_path.exists() else ''

    fi_path = root / 'feature_importance.csv'
    fi_df = pd.read_csv(fi_path) if fi_path.exists() else None

    alerts_path = root / 'example_soc_alerts.csv'
    alerts_df = pd.read_csv(alerts_path) if alerts_path.exists() else pd.DataFrame()

    return {
        'root': root,
        'meta': meta,
        'smoke': smoke,
        'leaderboard_df': leaderboard_df,
        'test_metrics_df': test_metrics_df,
        'report_text': report_text,
        'confusion_matrix_path': (root / 'confusion_matrix.png') if (root / 'confusion_matrix.png').exists() else None,
        'feature_importance_df': fi_df,
        'feature_importance_png': (root / 'feature_importance.png') if (root / 'feature_importance.png').exists() else None,
        'alerts_df': alerts_df,
    }


def render_dashboard(root=Path('outputs')):
    import streamlit as st

    st.set_page_config(page_title='Axynetra | Threat Analysis', page_icon='🛡️', layout='wide')
    st.title('🛡️ Axynetra — Explainable Network Threat Classification & Intelligent Alert Triage')
    st.caption(
        'Offline multi-class network-flow evaluation and illustrative analyst triage prototype (UNSW-NB15). '
        'Not a live intrusion detection system (IDS), does not perform real-time packet capture or automatic blocking, '
        'and is not claimed to be production-ready.'
    )

    with st.expander('Notebook and CSV reference'):
        st.markdown(
            '[Open in Colab](https://colab.research.google.com/github/Rohan-Ktksu/axynetra/blob/main/reference/Axynetra_UNSW_NB15.ipynb) '
            '· [Jupyter output viewer](https://nbviewer.org/github/Rohan-Ktksu/axynetra/blob/main/reference/Axynetra_UNSW_NB15.ipynb) '
            '· [Notebook with saved outputs](https://github.com/Rohan-Ktksu/axynetra/blob/main/reference/Axynetra_UNSW_NB15.ipynb) '
            '· [Reference folder](https://github.com/Rohan-Ktksu/axynetra/tree/main/reference)'
        )
        st.markdown(
            '[Training CSV](https://raw.githubusercontent.com/Rohan-Ktksu/axynetra/main/reference/data/UNSW_NB15_training_clean.csv) '
            '· [Test CSV](https://raw.githubusercontent.com/Rohan-Ktksu/axynetra/main/reference/data/UNSW_NB15_testing_no_overlap.csv)'
        )
        st.caption('Saved notebook outputs use a cleaned, sampled UNSW-NB15 experiment; the root notebook uses separate synthetic demonstration data.')

    try:
        bundle = load_dashboard_artifacts(root)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        st.stop()
        return

    meta = bundle['meta']
    smoke = bundle['smoke']
    if smoke:
        st.warning(
            '⚠️ SYNTHETIC SMOKE TEST DATA: These artifacts were generated on synthetic data solely to validate the code pipeline. '
            'They are NOT real UNSW-NB15 cybersecurity performance metrics.'
        )
    else:
        st.info(
            'Stored CSV evaluation loaded. Dataset origin is not independently verified. '
            'The pipeline selects models on validation data and evaluates the winner once per run.'
        )

    fit_rows = meta.get('fit_train_rows', meta.get('train_rows', 0))
    val_rows = meta.get('val_rows', 0)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric('Total Training Rows', f"{meta.get('train_rows', 0):,}",
                help=f"Fit subset: {fit_rows:,} rows | Validation subset: {val_rows:,} rows")
    col2.metric('Validation Rows', f"{val_rows:,}")
    col3.metric('Held-Out Test Rows', f"{meta.get('test_rows', 0):,}")
    col4.metric('Predictor Features', meta.get('predictor_count', 0))
    col5.metric('Validation Macro-F1 Winner', meta.get('best_model', 'N/A'))

    st.subheader('1. Stratified Validation Split — Model Selection Leaderboard')
    st.caption(
        'All four candidate classifiers are trained on the fit-training subset and ranked solely by Validation Macro F1. '
        'The held-out test set is never used for model ranking or hyperparameter tuning.'
    )
    results = bundle['leaderboard_df']
    st.dataframe(results, hide_index=True, width='stretch')
    chart_cols = [c for c in ['accuracy', 'macro_f1', 'weighted_f1'] if c in results.columns]
    if 'model' in results.columns and chart_cols:
        st.bar_chart(results.set_index('model')[chart_cols])

    st.subheader('2. Untouched Held-Out Test Set — One-Time Final Evaluation')
    st.caption('Evaluated exactly once using the single winning model selected on the validation split.')
    if not bundle['test_metrics_df'].empty:
        st.dataframe(bundle['test_metrics_df'], hide_index=True, width='stretch')
    st.caption('ROC-AUC uses one-vs-rest probability scores. Full macro requires all ten classes; supported macro averages only defined classes. Missing positives, negatives or learned classes are reported as undefined. Model selection still uses validation Macro F1.')
    roc_path = bundle['root'] / 'heldout_test_roc_auc.csv'
    if roc_path.exists():
        st.dataframe(pd.read_csv(roc_path), hide_index=True, width='stretch')
    roc_plot = bundle['root'] / 'roc_curves.png'
    if roc_plot.exists():
        st.image(str(roc_plot), caption='Held-out one-vs-rest ROC curves', width='stretch')

    col_cm, col_rep = st.columns([1.15, 1.0])
    with col_cm:
        st.markdown('**Row-Normalized Held-Out Test Confusion Matrix**')
        if bundle['confusion_matrix_path'] is not None:
            st.image(str(bundle['confusion_matrix_path']),
                     caption=f"Held-out test confusion matrix ({meta.get('best_model', '')})",
                     width='stretch')
    with col_rep:
        st.markdown('**Class-Wise Classification Report (Held-Out Test Set)**')
        if bundle['report_text']:
            st.code(bundle['report_text'], language='text')

    if bundle['feature_importance_df'] is not None:
        st.subheader('3. Global Feature Importance (Tree-Based Impurity Reduction)')
        st.caption(
            'Caveat: Impurity-based feature importance measures global split reduction across the training set. '
            'It is associative, NOT causal, and is not a reliable explanation for individual flow predictions.'
        )
        imp = bundle['feature_importance_df'].head(15)
        st.bar_chart(imp.set_index('feature')['importance'])

    st.subheader('4. Illustrative Prioritized SOC Investigation Queue')
    st.caption(
        'Severity weights (0–5) are human-assigned examples; model confidence is an uncalibrated score, not true probability. '
        'Top feature deviations show standardized z-score differences from training medians as an exploratory aid. '
        'Alerts are strictly for human analyst review — never automated blocking.'
    )
    q = bundle['alerts_df']
    if not q.empty and 'predicted_attack' in q.columns:
        chosen = st.multiselect('Filter by predicted attack category', sorted(q['predicted_attack'].unique()))
        if chosen:
            q = q[q['predicted_attack'].isin(chosen)]
        st.dataframe(q, hide_index=True, width='stretch')
        st.download_button('Export filtered alerts (CSV)', safe_csv(q), 'axynetra_alerts.csv', 'text/csv')
    st.caption('Source dataset: UNSW-NB15 (UNSW Canberra). Axynetra offline academic research prototype.')


def safe_csv(frame):
    """Neutralize spreadsheet formulas in string cells and column headings."""
    def safe(value):
        if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
            return "'" + value
        return value
    result = frame.copy()
    for column in result.columns:
        result[column] = result[column].map(safe)
    result.columns = [safe(str(column)) for column in result.columns]
    return result.to_csv(index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='View saved Axynetra experiment artifacts.')
    parser.add_argument('--results', type=Path, help='View an existing artifact directory instead of the interactive website')
    args = parser.parse_args()
    if args.results is not None:
        render_dashboard(args.results)
    else:
        from website import render_website
        render_website()
