"""Axynetra's multipage Streamlit website, owned by Rohan-Ktksu.

Run with: python -m streamlit run dashboard.py
"""
import io
import json
import zipfile

import pandas as pd
import streamlit as st

import multiclass_intrusion as mi
from dashboard import load_dashboard_artifacts, safe_csv
from website_core import ROOT, load_reference_data, predict_flows, read_prediction_csv

OUTPUT = ROOT / 'outputs' / 'website'
REPOSITORY = 'https://github.com/Rohan-Ktksu/axynetra'
BRANCH = 'codex/organized-prediction-website'
SOURCE = f'{REPOSITORY}/blob/{BRANCH}'

# These are descriptive groups, not learned rankings or an additional selector.
FEATURE_GROUPS = {
    'Connection context': ('dur proto service state', 'Duration, protocol, application service and connection state describe the type of flow.'),
    'Traffic volume': ('spkts dpkts sbytes dbytes rate sload dload smean dmean', 'Packet counts, bytes, transfer rates and mean packet sizes describe traffic volume in each direction.'),
    'Packet behavior': ('sttl dttl sloss dloss', 'Time-to-live and retransmission/loss measurements describe packet behavior.'),
    'Timing': ('sinpkt dinpkt sjit djit tcprtt synack ackdat', 'Packet spacing, jitter and TCP handshake timing describe how traffic evolves.'),
    'TCP state': ('swin dwin stcpb dtcpb', 'TCP window sizes and sequence-number bases describe transport state.'),
    'Application activity': ('trans_depth response_body_len is_ftp_login ct_ftp_cmd ct_flw_http_mthd', 'HTTP transaction depth, response size and HTTP/FTP activity describe application interactions.'),
    'Connection history': ('ct_srv_src ct_state_ttl ct_dst_ltm ct_src_dport_ltm ct_dst_sport_ltm ct_dst_src_ltm ct_src_ltm ct_srv_dst', 'Dataset-supplied connection counts summarize recent repeated services, endpoints, ports and state/TTL combinations.'),
    'Endpoint indicator': ('is_sm_ips_ports', 'Indicates matching source/destination IPs and ports in the supplied flow record.'),
}


@st.cache_data(show_spinner=False)
def reference_data():
    return load_reference_data()


@st.cache_resource(show_spinner=False)
def experiment():
    train, test = reference_data()
    meta, model = mi.run_experiment(train, test, OUTPUT, smoke=False)
    return meta, model


def prepare_experiment():
    try:
        with st.spinner('Training four candidates, selecting on validation, and evaluating the winner. The first run can take a few minutes…'):
            result = experiment()
        st.session_state['experiment_ready'] = True
        return result
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f'Model preparation failed: {exc}')
        return None


def page_intro(title, description):
    st.title(title)
    st.write(description)


def render_overview():
    page_intro('Network threats, made understandable.',
               'Axynetra turns stored network-flow records into attack classifications and a clear investigation queue.')
    st.caption('A machine-learning project by Rohan Ktksu · GitHub: Rohan-Ktksu')
    a, b, c = st.columns(3)
    a.metric('Attack categories + Normal', '10')
    b.metric('Input features', '42')
    c.metric('Candidate models', '4')
    st.markdown('### Explore the project')
    for title, body in [
        ('Predict', 'Choose bundled held-out test records or upload a flow CSV. Press Predict to run the selected model and inspect its output.'),
        ('Results', 'Compare validation models, inspect held-out metrics and error charts, review alerts, and download experiment files.'),
        ('Model & Features', 'Follow the full training workflow, understand the four algorithms, and inspect every included and excluded feature.'),
    ]:
        with st.container(border=True):
            st.subheader(title)
            st.write(body)
    st.info('This website uses the bundled cleaned, sampled UNSW-NB15 experiment: 10,000 training and 5,000 held-out test records. It does not use the synthetic notebook results.')
    if st.button('Prepare model & results', type='primary'):
        if prepare_experiment() is not None:
            st.success('Model ready. Open Predict or Results from the navigation.')
    if st.session_state.get('experiment_ready'):
        meta, _ = experiment()
        st.success(f"Ready: {meta['best_model']} · selected by validation Macro F1")
    with st.expander('Dataset and project sources'):
        st.markdown(f'[Project repository]({REPOSITORY}) · [CSV notebook]({SOURCE}/reference/Axynetra_UNSW_NB15.ipynb) · '
                    '[UNSW-NB15 dataset authors](https://research.unsw.edu.au/projects/unsw-nb15-dataset)')
        provenance = json.loads((ROOT / 'reference/data/DATA_PROVENANCE.json').read_text(encoding='utf-8'))
        st.write(f"Bundled files: {provenance['clean_train_rows']:,} cleaned training rows and {provenance['clean_test_rows']:,} test rows. Both file hashes are checked before training.")
        st.caption('Community-mirror data; duplicates/conflicting training labels and cross-split overlap were removed. This is a modified split, not the unchanged official benchmark.')
        st.dataframe(pd.read_csv(ROOT / 'reference/data/cleaning_summary.csv'), hide_index=True, width='stretch')


def render_predict():
    page_intro('Predict a network flow', 'Run real inference through the same preprocessing and winning classifier used in the experiment.')
    try:
        train, test = reference_data()
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return
    mode = st.radio('Test data source', ['Bundled test data', 'Upload CSV'], horizontal=True)
    frame = None
    identity = None
    if mode == 'Bundled test data':
        category = st.selectbox('Test category', ['All categories'] + mi.CLASSES,
                                help='This label only filters examples. It is excluded from model inputs.')
        pool = test if category == 'All categories' else test[test.attack_cat == category]
        count = st.slider('Number of test records', 1, min(100, len(pool)), min(10, len(pool)))
        frame = pool.sample(n=count, random_state=mi.SEED).reset_index(drop=True)
        identity = (mode, category, count)
        st.caption('A fixed sample from the reserved 5,000-row test set. Actual labels are used only for comparison after prediction.')
    else:
        st.write('Upload the 42 input columns from the template. `id`, `label` and `attack_cat` are optional and never enter the model. Up to 1,000 rows / 5 MB.')
        st.download_button('Download input template', safe_csv(test[mi.get_predictor_columns(train, test)].head(5)),
                           'axynetra_input_template.csv', 'text/csv')
        upload = st.file_uploader('Network-flow CSV', type=['csv'], max_upload_size=5)
        if upload is not None:
            content = upload.getvalue()
            identity = (mode, content)
            try:
                frame = read_prediction_csv(content)
            except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(f'Unable to read CSV: {exc}')
    if st.session_state.get('prediction_identity') != identity:
        st.session_state.pop('prediction_output', None)
    if frame is None:
        return
    with st.expander(f'Inspect test data · {len(frame):,} records', expanded=True):
        st.dataframe(frame, hide_index=True, width='stretch')
    if st.button('Predict', type='primary'):
        # Validate before expensive training so malformed uploads get immediate feedback.
        from website_core import prediction_inputs
        try:
            prediction_inputs(frame, train)
            prepared = prepare_experiment()
            if prepared is not None:
                meta, model = prepared
                results, scores = predict_flows(model, train, frame)
                st.session_state['prediction_output'] = (results, scores, meta['best_model'])
                st.session_state['prediction_identity'] = identity
        except (ValueError, TypeError) as exc:
            st.session_state.pop('prediction_output', None)
            st.error(f'Prediction failed: {exc}')
    if 'prediction_output' not in st.session_state:
        return
    results, scores, name = st.session_state['prediction_output']
    st.success(f'Prediction complete · {name} · {len(results):,} records')
    a, b, c = st.columns(3)
    a.metric('Predicted attacks', int((results.predicted_attack != 'Normal').sum()))
    b.metric('Predicted Normal', int((results.predicted_attack == 'Normal').sum()))
    c.metric('Flagged for review', int(results.needs_human_review.sum()))
    st.subheader('Prediction output')
    st.dataframe(results, hide_index=True, width='stretch')
    if len(results) == 1:
        st.bar_chart(scores.drop(columns='record').T.rename(columns={0: 'Model score'}))
    if 'correct' in results and results.correct.notna().any():
        st.caption(f"Agreement on this selected sample: {results.correct.mean():.1%}. Use Results for the complete held-out evaluation.")
    st.caption('Model scores are uncalibrated. Severity (0–5) is a human-assigned assumption; priority = score × severity. Review flags mark attacks or scores below 0.60. No automatic blocking occurs.')
    with st.expander('Scores for every learned category'):
        st.dataframe(scores, hide_index=True, width='stretch')
    st.download_button('Download predictions', safe_csv(results), 'axynetra_predictions.csv', 'text/csv')


def render_results():
    page_intro('Experiment results', 'Every output below comes from one validation-selected experiment on the bundled reference CSVs.')
    if not st.session_state.get('experiment_ready'):
        st.info('Generate the experiment once to view all outputs. Predict also prepares these results automatically.')
        if st.button('Generate model & results', type='primary'):
            prepare_experiment()
        if not st.session_state.get('experiment_ready'):
            return
    meta, _ = experiment()
    bundle = load_dashboard_artifacts(OUTPUT)
    a, b, c, d = st.columns(4)
    a.metric('Training rows', f"{meta['train_rows']:,}")
    b.metric('Validation rows', f"{meta['val_rows']:,}")
    c.metric('Held-out rows', f"{meta['test_rows']:,}")
    d.metric('Selected model', meta['best_model'])
    selection, evaluation, features, alerts, downloads = st.tabs([
        'Model comparison', 'Held-out evaluation', 'Feature importance', 'Alert queue', 'Downloads'])
    with selection:
        st.subheader('Choose on validation, then refit')
        st.write('Four candidates use the same stratified fit/validation split. Highest Macro F1 wins; weighted F1 and accuracy break ties. The winner is refitted on all training rows.')
        st.dataframe(bundle['leaderboard_df'], hide_index=True, width='stretch')
        st.bar_chart(bundle['leaderboard_df'].set_index('model')[['accuracy', 'macro_f1', 'weighted_f1']], stack=False)
    with evaluation:
        metrics = meta['heldout_test_evaluation']
        a, b, c = st.columns(3)
        a.metric('Held-out accuracy', f"{metrics['accuracy']:.1%}")
        b.metric('Held-out Macro F1', f"{metrics['macro_f1']:.4f}")
        c.metric('Held-out weighted F1', f"{metrics['weighted_f1']:.4f}")
        st.caption('One final evaluation stage per run. The test set never selects the winner. Macro F1 gives equal weight to all ten categories; accuracy can hide rare-class errors.')
        st.dataframe(bundle['test_metrics_df'], hide_index=True, width='stretch')
        st.image(str(OUTPUT / 'confusion_matrix.png'), caption='Row-normalized confusion matrix: true categories by predicted categories', width='stretch')
        st.code(bundle['report_text'], language='text')
        st.image(str(OUTPUT / 'roc_curves.png'), caption='One-vs-rest ROC curves on the held-out test data', width='stretch')
        st.dataframe(pd.read_csv(OUTPUT / 'heldout_test_roc_auc.csv'), hide_index=True, width='stretch')
        st.caption('Full macro ROC-AUC requires all ten categories to have defined AUCs. Supported macro excludes undefined classes; coverage is reported in the table.')
    with features:
        st.subheader('Global feature importance')
        st.write(f"Source model: **{meta['feature_importance_model']}**. If the winner is not a tree, the highest-ranked compatible validation tree supplies this chart.")
        importance = bundle['feature_importance_df']
        if importance is not None:
            st.bar_chart(importance.head(20).set_index('feature')['importance'])
            st.dataframe(importance, hide_index=True, width='stretch')
        st.caption('Impurity reduction is a global association measure. One-hot categories appear separately. These rankings do not select features and do not explain why an individual record received its prediction.')
    with alerts:
        st.subheader('Illustrative investigation queue')
        st.caption('Top 100 held-out flows ordered by score × assumed severity. Training-median z-score deviations are exploratory signals, not causal explanations.')
        queue = bundle['alerts_df']
        chosen = st.multiselect('Predicted categories', mi.CLASSES)
        if chosen:
            queue = queue[queue.predicted_attack.isin(chosen)]
        st.dataframe(queue, hide_index=True, width='stretch')
        st.download_button('Export filtered alerts', safe_csv(queue), 'axynetra_alerts.csv', 'text/csv')
    with downloads:
        st.write('All experiment files are grouped under `outputs/website/`. Uploaded CSVs and interactive predictions are kept in the current session; they do not replace the evaluation outputs.')
        files = sorted(p for p in OUTPUT.iterdir() if p.is_file())
        st.dataframe(pd.DataFrame([{'artifact': p.name, 'bytes': p.stat().st_size} for p in files]), hide_index=True, width='stretch')
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in files:
                zipped.write(path, arcname=path.name)
        st.download_button('Download all experiment outputs (ZIP)', archive.getvalue(), 'axynetra_experiment_outputs.zip', 'application/zip')
        st.download_button('Download run metadata', (OUTPUT / 'run_metadata.json').read_bytes(), 'run_metadata.json', 'application/json')


def render_model():
    page_intro('The model & its features', 'Understand what Axynetra learns, which inputs it uses, and how its results should be read.')
    st.markdown((ROOT / 'docs/model_and_features.md').read_text(encoding='utf-8'))
    with st.expander('View the complete ML pipeline diagram'):
        st.image(str(ROOT / 'docs/pipeline/axynetra_ml_pipeline.png'), width='stretch')
    try:
        train, test = reference_data()
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return
    selected = mi.get_predictor_columns(train, test)
    st.subheader(f'All {len(selected)} selected predictor features')
    st.write('Every eligible flow measurement is retained. Feature selection here means excluding identifiers and target leakage; no RFE, PCA, mutual-information filter or top-k importance cutoff is applied.')
    group_by_feature = {name: (group, explanation) for group, (names, explanation) in FEATURE_GROUPS.items() for name in names.split()}
    schema = pd.DataFrame([{'feature': name, 'type': str(train[name].dtype),
                            'group': group_by_feature.get(name, ('Other', 'Flow measurement'))[0],
                            'role': 'Categorical → one-hot' if name in ['proto', 'service', 'state'] else 'Numeric → impute + scale'}
                           for name in selected])
    st.dataframe(schema, hide_index=True, width='stretch')
    st.download_button('Download feature list', safe_csv(schema), 'axynetra_features.csv', 'text/csv')
    for group, (names, explanation) in FEATURE_GROUPS.items():
        with st.expander(group):
            st.write(explanation)
            st.code(', '.join(names.split()), language='text')
    st.subheader('Excluded columns')
    st.table(pd.DataFrame([
        {'column': 'attack_cat', 'reason': 'The ten-class target. Including it would reveal the answer.'},
        {'column': 'label', 'reason': 'Binary normal/attack annotation. It leaks target information.'},
        {'column': 'id', 'reason': 'Row identifier, not a network behavior measurement.'},
    ]))
    st.subheader('Classifiers and exact settings')
    for name, estimator in mi.model_choices().items():
        with st.expander(name):
            st.code(str(estimator), language='python')
    if st.session_state.get('experiment_ready'):
        meta, _ = experiment()
        st.success(f"Current run: {meta['best_model']} · {meta['predictor_count']} raw predictors → {meta['transformed_feature_count']} transformed inputs")
    st.markdown(f'[Training and evaluation code]({SOURCE}/multiclass_intrusion.py) · [Inference code]({SOURCE}/website_core.py) · '
                f'[Data provenance]({SOURCE}/reference/data/DATA_PROVENANCE.json) · [CSV notebook]({SOURCE}/reference/Axynetra_UNSW_NB15.ipynb)')


def render_website():
    st.set_page_config(page_title='Axynetra | Rohan Ktksu', page_icon='🛡️', layout='wide')
    with st.sidebar:
        st.title('🛡️ Axynetra')
        st.caption('Network threat classification')
        st.markdown('**Rohan Ktksu**  \n[Rohan-Ktksu on GitHub](https://github.com/Rohan-Ktksu)')
    page = st.navigation([
        st.Page(render_overview, title='Overview', icon='🏠', default=True),
        st.Page(render_predict, title='Predict', icon='🔎', url_path='predict'),
        st.Page(render_results, title='Results', icon='📊', url_path='results'),
        st.Page(render_model, title='Model & Features', icon='🧠', url_path='model'),
    ])
    page.run()
    st.divider()
    st.caption('Axynetra · Rohan Ktksu · Offline academic prototype using stored CSVs. Model scores are uncalibrated; analyst review is required. No packet capture or automatic blocking.')
