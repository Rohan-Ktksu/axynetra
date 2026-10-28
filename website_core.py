"""Reference-data loading and validated inference shared by the website and tests."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import multiclass_intrusion as mi

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'reference' / 'data'
MAX_UPLOAD_ROWS = 1000

FEATURE_GROUPS = {
    'Connection context': ('dur proto service state', 'Flow duration, protocol, service and connection state.'),
    'Traffic volume': ('spkts dpkts sbytes dbytes rate sload dload smean dmean', 'Directional packets, bytes, transfer rates and mean packet sizes.'),
    'Packet behavior': ('sttl dttl sloss dloss', 'Directional time-to-live and retransmission/loss measurements.'),
    'Timing': ('sinpkt dinpkt sjit djit tcprtt synack ackdat', 'Packet spacing, jitter and TCP handshake timings.'),
    'TCP state': ('swin dwin stcpb dtcpb', 'Directional TCP window sizes and sequence-number bases.'),
    'Application activity': ('trans_depth response_body_len is_ftp_login ct_ftp_cmd ct_flw_http_mthd', 'HTTP transaction depth, response size and HTTP/FTP activity.'),
    'Connection history': ('ct_srv_src ct_state_ttl ct_dst_ltm ct_src_dport_ltm ct_dst_sport_ltm ct_dst_src_ltm ct_src_ltm ct_srv_dst', 'Dataset-supplied counts of recent repeated services, endpoints, ports and state/TTL combinations.'),
    'Endpoint indicator': ('is_sm_ips_ports', 'Whether the supplied flow has matching source/destination IPs and ports.'),
}


def load_reference_data(max_train=10000, max_test=5000):
    """Verify the bundled files, then use the CLI's overlap checks and sampling."""
    provenance = json.loads((DATA / 'DATA_PROVENANCE.json').read_text(encoding='utf-8'))
    paths = [DATA / 'UNSW_NB15_training_clean.csv', DATA / 'UNSW_NB15_testing_no_overlap.csv']
    for path in paths:
        expected = next(v['sha256'] for k, v in provenance['files'].items()
                        if k.replace('\\', '/').split('/')[-1] == path.name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'{path.name} failed its SHA-256 check. Restore the bundled reference CSV.')
    return mi.load_original(*paths, max_train, max_test)


def read_prediction_csv(content):
    """Validate raw headers before pandas can rename duplicates or infer types."""
    if len(content) > 5 * 1024 * 1024:
        raise ValueError('Upload a CSV of at most 5 MB.')
    text = content.decode('utf-8-sig')
    headers = next(csv.reader(io.StringIO(text)), [])
    normalized = [h.strip().lower() for h in headers]
    if not headers or len(normalized) != len(set(normalized)):
        raise ValueError('CSV headers must be nonempty and unique (ignoring case and spaces).')
    frame = pd.read_csv(io.StringIO(text), nrows=MAX_UPLOAD_ROWS + 1)
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(f'Upload at most {MAX_UPLOAD_ROWS:,} records per prediction.')
    return frame


def prediction_inputs(frame, training):
    """Align only training predictors; reject missing columns and invalid numbers."""
    features = mi.get_predictor_columns(training, frame)
    if frame.empty:
        raise ValueError('Provide at least one network-flow record.')
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(f'Predict at most {MAX_UPLOAD_ROWS:,} records at a time.')
    extra = [c for c in frame if c not in features and str(c).strip().lower() not in mi.LEAKAGE_COLUMNS]
    if extra:
        raise ValueError('Unexpected columns: ' + ', '.join(map(str, extra)))
    inputs = frame[features].copy()
    numeric = training[features].select_dtypes(include='number').columns
    for name in numeric:
        original = inputs[name]
        converted = pd.to_numeric(original, errors='coerce')
        if (original.notna() & converted.isna()).any() or np.isinf(converted.to_numpy(dtype=float)).any():
            raise ValueError(f'{name} must contain finite numbers or blank values.')
        inputs[name] = converted
    for name in set(features) - set(numeric):
        inputs[name] = inputs[name].map(lambda value: str(value) if pd.notna(value) else np.nan).astype(object)
    return inputs


def predict_flows(model, training, frame):
    """Run the selected pipeline on predictors only, with optional truth comparison."""
    inputs = prediction_inputs(frame, training)
    probabilities = model.predict_proba(inputs)
    classes = model.named_steps['classifier'].classes_
    predicted = classes[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    severity = np.array([mi.SEVERITY[c] for c in predicted])
    results = pd.DataFrame({
        'record': np.arange(1, len(inputs) + 1),
        'predicted_attack': predicted,
        'model_score_uncalibrated': confidence,
        'severity_assumption_0_to_5': severity,
        'priority_score_illustrative': confidence * severity,
        'needs_human_review': (confidence < 0.60) | (predicted != 'Normal'),
    })
    if 'attack_cat' in frame:
        truth = frame['attack_cat'].map(mi.normalized_label).reset_index(drop=True)
        results['actual_attack'] = truth
        results['correct'] = pd.array(
            [bool(a == b) if pd.notna(a) and a in mi.CLASSES else pd.NA for a, b in zip(truth, predicted)], dtype='boolean')
    scores = pd.DataFrame(probabilities, columns=classes)
    scores.insert(0, 'record', results['record'])
    return results, scores
