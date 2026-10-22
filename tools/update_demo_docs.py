"""Refresh README examples only from a completed synthetic notebook run."""
import json
from pathlib import Path
import re
import shutil
import csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'notebook'


def main():
    metadata = json.loads((OUT / 'run_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('is_synthetic_smoke_test') is not True:
        raise ValueError('Public demo documentation can only be generated from synthetic data.')
    metrics = metadata['heldout_test_evaluation']
    lines = ['<!-- BEGIN SYNTHETIC SUMMARY -->',
             '| Demonstration output | Observed value |', '| --- | --- |']
    for label, value in [('Training rows', metadata['train_rows']),
                         ('Fit-training rows', metadata['fit_train_rows']),
                         ('Validation rows', metadata['val_rows']), ('Held-out test rows', metadata['test_rows']),
                         ('Predictors after exclusions', metadata['predictor_count']),
                         ('Validation-selected winner', metadata['best_model']),
                         ('Held-out accuracy', f"{metrics['accuracy']:.4f}"),
                         ('Held-out Macro F1', f"{metrics['macro_f1']:.4f}"),
                         ('Held-out one-vs-rest macro ROC-AUC', f"{metrics['roc_auc_ovr_macro']:.4f}"),
                         ('Defined AUC classes', f"{metrics['roc_auc_defined_classes']}/10")]:
        lines.append(f'| {label} | {value} |')
    lines += ['', '**Validation results — synthetic demonstration**', '',
              '| Model | Accuracy | Macro F1 | Macro ROC-AUC |', '| --- | ---: | ---: | ---: |']
    for row in metadata['validation_leaderboard']:
        lines.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['roc_auc_ovr_macro']:.4f} |")
    with (OUT / 'example_soc_alerts.csv').open(encoding='utf-8', newline='') as source:
        alerts = list(csv.DictReader(source))[:3]
    lines += ['', '**Example review queue — synthetic demonstration**', '',
              '| Flow index | Predicted category | Uncalibrated score | Illustrative priority |',
              '| --- | --- | ---: | ---: |']
    for row in alerts:
        lines.append(f"| {row['flow_index']} | {row['predicted_attack']} | {row['model_confidence']} | {row['priority_score_illustrative']} |")
    lines += ['<!-- END SYNTHETIC SUMMARY -->']
    summary = '\n'.join(lines)
    path = ROOT / 'README.md'
    readme = path.read_text(encoding='utf-8')
    if '{{DEMO_RESULTS}}' in readme:
        readme = readme.replace('{{DEMO_RESULTS}}', summary)
    else:
        readme, count = re.subn(r'<!-- BEGIN SYNTHETIC SUMMARY -->.*?<!-- END SYNTHETIC SUMMARY -->',
                                lambda _: summary, readme, flags=re.S)
        if count != 1:
            raise ValueError('Expected exactly one demonstration-summary block in README.')
    path.write_text(readme, encoding='utf-8')
    images = ROOT / 'docs' / 'demo'
    images.mkdir(parents=True, exist_ok=True)
    for name in ['training_class_distribution.png', 'validation_comparison.png',
                 'confusion_matrix.png', 'roc_curves.png']:
        shutil.copyfile(OUT / name, images / name)
    print('Updated synthetic README tables and four demonstration charts.')


if __name__ == '__main__':
    main()
