"""Browser bridge; calls the shared training and inference implementations."""
import base64
import io
import json
import platform
import zipfile
from pathlib import Path

import pandas as pd
import sklearn
from js import report_progress

import multiclass_intrusion as mi
import website_core as core

# Web workers cannot spawn OS processes; use the same estimators in one thread.
original_model_choices = mi.model_choices


def browser_model_choices():
    choices = original_model_choices()
    for model in choices.values():
        if 'n_jobs' in model.get_params():
            model.set_params(n_jobs=1)
    return choices


mi.model_choices = browser_model_choices


def records(frame):
    return json.loads(frame.to_json(orient='records'))


def browser_train():
    global train, test, fitted
    report_progress(0.05, 'Verifying CSV hashes and preparing the 10,000 / 5,000 split')
    train, test = core.load_reference_data()
    fitted = mi.fit_experiment(train, test, Path('/axynetra/outputs'), False,
                               progress=lambda n, text: report_progress(0.15 + n * 0.85, text))
    return json.dumps({'best_model': fitted['best'], 'leaderboard': records(fitted['df_val']),
                       'python': platform.python_version(), 'sklearn': sklearn.__version__})


def browser_test():
    global metadata
    metadata, _ = mi.evaluate_experiment(fitted, progress=lambda n, text: report_progress(n, text))
    metadata['dataset'] = 'UNSW-NB15 community mirror; cleaned, sampled, modified split'
    metadata['execution'] = {'runtime': 'Pyodide browser worker', 'python': platform.python_version(),
                             'scikit-learn': sklearn.__version__, 'estimator_jobs': 1}
    root = Path('/axynetra/outputs')
    (root / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(root.iterdir()):
            if path.is_file():
                zipped.write(path, arcname=path.name)
    return json.dumps({
        'meta': metadata,
        'images': {name: base64.b64encode((root / (name + '.png')).read_bytes()).decode()
                   for name in ['confusion_matrix', 'roc_curves', 'feature_importance']
                   if (root / (name + '.png')).exists()},
        'report': (root / 'classification_report.txt').read_text(),
        'auc': records(pd.read_csv(root / 'heldout_test_roc_auc.csv')),
        'alerts': records(pd.read_csv(root / 'example_soc_alerts.csv')),
        'zip': base64.b64encode(archive.getvalue()).decode(),
    })


def browser_predict(request_json):
    request = json.loads(request_json)
    if request.get('csv') is not None:
        frame = core.read_prediction_csv(request['csv'].encode('utf-8'))
    else:
        category = request.get('category', 'All categories')
        if category != 'All categories' and category not in mi.CLASSES:
            raise ValueError('Choose a supported test category.')
        count = request.get('count', 10)
        if not isinstance(count, int) or not 1 <= count <= 100:
            raise ValueError('Choose between 1 and 100 records.')
        pool = test if category == 'All categories' else test[test.attack_cat == category]
        frame = pool.sample(n=min(count, len(pool)), random_state=mi.SEED)
    result, scores = core.predict_flows(fitted['winner'], train, frame)
    return json.dumps({'results': records(result), 'scores': records(scores), 'inputs': records(frame)})


def browser_template():
    return test[mi.get_predictor_columns(train, test)].head(5).to_csv(index=False)
