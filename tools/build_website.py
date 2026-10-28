"""Refresh the GitHub Pages data preview and the exact browser Python sources."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import multiclass_intrusion as mi
from website_core import FEATURE_GROUPS, load_reference_data


def build():
    train, test = load_reference_data()
    selected = mi.get_predictor_columns(train, test)
    groups = {name: group for group, (names, _) in FEATURE_GROUPS.items() for name in names.split()}
    assert set(groups) == set(selected)
    data = {
        'owner': 'Rohan Ktksu', 'account': 'Rohan-Ktksu', 'train_rows': len(train), 'test_rows': len(test),
        'fit_rows': mi.stratified_train_val_split(train)[2]['fit_train_rows'],
        'classes': mi.CLASSES, 'preview': json.loads(train.head(6).to_json(orient='records')),
        'features': [{'feature': c, 'type': str(train[c].dtype), 'group': groups[c],
                      'processing': 'One-hot encoding' if c in ['proto', 'service', 'state'] else 'Median + StandardScaler'} for c in selected],
        'groups': [{'group': g, 'features': names.split(), 'meaning': meaning} for g, (names, meaning) in FEATURE_GROUPS.items()],
        'models': {name: str(model) for name, model in mi.model_choices().items()},
        'class_counts': train.attack_cat.value_counts().to_dict(),
    }
    out = ROOT / 'docs'
    (out / 'runtime').mkdir(exist_ok=True)
    (out / 'site-data.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    for name in ['multiclass_intrusion.py', 'website_core.py']:
        shutil.copyfile(ROOT / name, out / 'runtime' / name)
    (out / '.nojekyll').touch()
    print('Pages assets refreshed; 42 features, verified data previews, two shared Python modules.')


if __name__ == '__main__':
    build()
