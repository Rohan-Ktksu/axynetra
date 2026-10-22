"""Prepare a documented duplicate-controlled experiment; keep raw CSVs unchanged."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ORIGINAL = DATA / 'original'
SOURCE = 'https://raw.githubusercontent.com/Nir-J/ML-Projects/44bbd02b9e70f74dd915218d5eb702f125341ffc/UNSW-Network_Packet_Classification/'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    train_path = ORIGINAL / 'UNSW_NB15_training-set.csv'
    test_path = ORIGINAL / 'UNSW_NB15_testing-set.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    features = [c for c in train if c not in ['attack_cat', 'label', 'id']]
    train_hashes = pd.util.hash_pandas_object(train[features], index=False)
    test_hashes = pd.util.hash_pandas_object(test[features], index=False)
    # Training labels alone determine ambiguous training groups.
    targets_per_group = train.groupby(train_hashes).attack_cat.nunique()
    conflicting_hashes = targets_per_group[targets_per_group > 1].index
    conflicting_rows = train_hashes.isin(conflicting_hashes)
    clean_train = train.loc[~conflicting_rows].drop_duplicates(subset=features).copy()
    # Test filtering depends only on predictors, not test labels. Use ALL original
    # training hashes, including those removed above, as a conservative exclusion.
    overlap = test_hashes.isin(train_hashes)
    clean_test = test.loc[~overlap].copy()
    out_train = DATA / 'UNSW_NB15_training_clean.csv'
    out_test = DATA / 'UNSW_NB15_testing_no_overlap.csv'
    clean_train.to_csv(out_train, index=False)
    clean_test.to_csv(out_test, index=False)
    metadata = {
        'dataset': 'UNSW-NB15 community mirror; duplicate-controlled experiment',
        'official_reference': 'https://research.unsw.edu.au/projects/unsw-nb15-dataset',
        'mirror_revision': '44bbd02b9e70f74dd915218d5eb702f125341ffc',
        'raw_train_rows': len(train), 'raw_test_rows': len(test), 'predictors': len(features),
        'training_duplicate_predictor_rows': int(train.duplicated(features).sum()),
        'conflicting_training_groups': len(conflicting_hashes),
        'conflicting_training_rows_removed': int(conflicting_rows.sum()),
        'remaining_same_label_duplicates_removed': int((~conflicting_rows).sum()-len(clean_train)),
        'overlapping_test_rows_removed': int(overlap.sum()),
        'clean_train_rows': len(clean_train), 'clean_test_rows': len(clean_test),
        'clean_train_class_counts': clean_train.attack_cat.value_counts().to_dict(),
        'clean_test_class_counts': clean_test.attack_cat.value_counts().to_dict(),
        'test_duplicate_rows_retained': int(clean_test.duplicated(features).sum()),
        'test_labels_used_to_clean_training': False,
        'warning': 'Modified split, not the unchanged official benchmark. Notebook defaults to stratified row caps. Original test duplicates are retained after cross-split overlap removal.',
        'files': {},
    }
    for path in [train_path, test_path, out_train, out_test]:
        metadata['files'][str(path.relative_to(ROOT))] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
        if path.parent == ORIGINAL:
            metadata['files'][str(path.relative_to(ROOT))]['download_url'] = SOURCE + path.name
    (DATA / 'DATA_PROVENANCE.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    summary = pd.DataFrame([
        {'stage': 'Original training', 'rows': len(train)},
        {'stage': 'Original test', 'rows': len(test)},
        {'stage': 'Conflicting training rows removed', 'rows': int(conflicting_rows.sum())},
        {'stage': 'Same-label training duplicates removed', 'rows': int((~conflicting_rows).sum()-len(clean_train))},
        {'stage': 'Test overlap removed', 'rows': int(overlap.sum())},
        {'stage': 'Clean training', 'rows': len(clean_train)},
        {'stage': 'Non-overlapping test', 'rows': len(clean_test)},
    ])
    summary.to_csv(DATA / 'cleaning_summary.csv', index=False)
    print(summary.to_string(index=False))
    print('All ten training classes retained:', clean_train.attack_cat.nunique()==10)


if __name__ == '__main__':
    main()
