import argparse
import re
from xml.sax.saxutils import escape
from pathlib import Path
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, Preformatted, HRFlowable)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import textwrap
import json


def build_figures(root: Path):
    # 1. Architecture diagram showing leak-free validation selection + single held-out test evaluation
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    blocks = [
        (0.15, 3.2, 2.25, 1.05, 'UNSW-NB15 CSV', 'Train & held-out test'),
        (2.95, 3.2, 2.45, 1.05, 'Stratified Train/Val', 'Fit preproc on train only'),
        (5.98, 3.2, 2.60, 1.05, '4 ML Classifiers', 'Ranked by Val Macro-F1'),
        (9.10, 3.2, 2.70, 1.05, 'Held-Out Test (1x)', 'Untouched final metrics'),
        (5.98, 1.0, 2.60, 1.05, 'Explainability', 'Global tree importances'),
        (9.10, 1.0, 2.70, 1.05, 'Analyst Queue', 'Illustrative SOC triage'),
    ]
    for x, y, w, h, heading, sub in blocks:
        p = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.13',
            facecolor='#EAF4F7', edgecolor='#3C8594', lw=1.8
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + 0.64, heading, color='#14344A', fontsize=11.5,
                ha='center', va='center', fontweight='bold')
        ax.text(x + w / 2, y + 0.30, sub, color='#4F6578', fontsize=8.8,
                ha='center', va='center')
    for sx, ex in [(2.43, 2.91), (5.44, 5.94), (8.61, 9.06)]:
        ax.annotate('', xy=(ex, 3.72), xytext=(sx, 3.72),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#3C8594'))
    ax.annotate('', xy=(7.25, 2.10), xytext=(7.25, 3.15),
                arrowprops=dict(arrowstyle='->', lw=2, color='#3C8594'))
    ax.annotate('', xy=(9.04, 1.53), xytext=(8.63, 1.53),
                arrowprops=dict(arrowstyle='->', lw=2, color='#3C8594'))
    fig.suptitle('Axynetra offline defensive-analytics workflow (leak-free validation selection)',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(root / 'architecture_flow.png', dpi=155, bbox_inches='tight')
    plt.close(fig)

    # 2. Published UNSW split distribution (NOT the model prediction output)
    labels = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Recon.', 'Analysis', 'Backdoor', 'Shellcode', 'Worms']
    train_counts = [56000, 40000, 33393, 18184, 12264, 10491, 2000, 1746, 1133, 130]
    test_counts = [37000, 18871, 11132, 6062, 4089, 3496, 677, 583, 378, 44]
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, train_counts, 0.35, label='Official train records', color='#2B738D')
    ax.bar(x + 0.18, test_counts, 0.35, label='Official test records', color='#C49A57')
    ax.set_xticks(x, labels, rotation=38, ha='right')
    ax.set_ylabel('Number of records')
    ax.set_title('UNSW-NB15 published class distribution (not prediction output)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'published_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_pdf(srn: str = 'SRNNO'):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', srn):
        raise ValueError('SRN must contain 1–40 letters, digits, underscores or hyphens.')
    root = Path(__file__).resolve().parent
    outputs = root / 'outputs'
    build_figures(root)

    font_dir = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
    pdfmetrics.registerFont(TTFont('DejaVu', str(font_dir / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuBold', str(font_dir / 'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuMono', str(font_dir / 'DejaVuSansMono.ttf')))
    NAVY = HexColor('#15253B')
    MUTE = HexColor('#56667B')
    TEAL = HexColor('#137D8D')
    PALE = HexColor('#EAF4F7')
    BLUE = HexColor('#1F4B7E')

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Cover', fontName='DejaVuBold', fontSize=24, leading=31, textColor=NAVY, spaceAfter=22, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='Subcover', fontName='DejaVu', fontSize=12, leading=19, textColor=MUTE, alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name='H1x', fontName='DejaVuBold', fontSize=15, leading=20, textColor=BLUE, spaceBefore=15, spaceAfter=9, keepWithNext=True))
    styles.add(ParagraphStyle(name='H2x', fontName='DejaVuBold', fontSize=11, leading=16, textColor=TEAL, spaceBefore=11, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle(name='Px', fontName='DejaVu', fontSize=9.25, leading=15, spaceAfter=8, textColor=NAVY))
    styles.add(ParagraphStyle(name='Smallx', fontName='DejaVu', fontSize=8, leading=12, spaceAfter=5, textColor=MUTE))
    styles.add(ParagraphStyle(name='Cellx', fontName='DejaVu', fontSize=8, leading=12, textColor=NAVY))
    styles.add(ParagraphStyle(name='CellBoldx', fontName='DejaVuBold', fontSize=8, leading=12, textColor=NAVY))
    styles.add(ParagraphStyle(name='CodeCaption', fontName='DejaVuMono', fontSize=6.6, leading=10, textColor=NAVY))

    filename = root / f'{srn}_axynetra_multiclass_attack_classification_csit5a.pdf'
    doc = SimpleDocTemplate(
        str(filename), pagesize=(595.3, 841.9),
        leftMargin=48, rightMargin=48, topMargin=57, bottomMargin=55,
        title='Axynetra - Multiclass Cyberattack Classification - CSIT5A'
    )
    story = []
    P = lambda x: Paragraph(x, styles['Px'])
    S = lambda x: Paragraph(x, styles['Smallx'])
    H1 = lambda x: Paragraph(x, styles['H1x'])
    H2 = lambda x: Paragraph(x, styles['H2x'])

    def table(headers, rows, widths=None):
        data = [[Paragraph(escape(str(v)), styles['CellBoldx']) for v in headers]]
        for row in rows:
            data.append([Paragraph(escape(str(v)), styles['Cellx']) for v in row])
        tb = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        tb.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PALE),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.3, HexColor('#D2DCE5')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F6F9FB')]),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ]))
        return tb

    def figblock(path, w=470):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            ow, oh = im.size
        return Image(str(path), width=w, height=w * oh / ow)

    meta = json.loads((outputs / 'run_metadata.json').read_text(encoding='utf-8'))
    is_smoke = 'SYNTHETIC' in str(meta.get('dataset', '')).upper() or bool(meta.get('is_synthetic_smoke_test', False))
    eval_status = (
        'Synthetic pipeline smoke-test completed; official dataset experiment pending'
        if is_smoke else
        f"Supplied CSV split evaluated; source not verified ({meta['train_rows']:,} train / {meta['test_rows']:,} test)"
    )

    story += [
        Spacer(1, 80),
        Paragraph('AXYNETRA', styles['Cover']),
        Paragraph('Explainable Network Threat Classification &amp; Intelligent Alert Triage', styles['Subcover']),
        HRFlowable(width='100%', thickness=2, color=TEAL, spaceAfter=24),
        P('<b>CSMA Assignment | Option 4 - Multi-Class Attack Classification</b>'),
        table(
            ['Submission field', 'Value'],
            [
                ['Course / section', 'CSMA / CSIT 5A'],
                ['SRN', f'{srn} (replace SRNNO before submission if placeholder)'],
                ['Implementation', 'Python 3, scikit-learn, pandas, Matplotlib, Streamlit'],
                ['Dataset', 'UNSW-NB15 (official predefined train/test splits)'],
                ['Evaluation protocol', 'Stratified validation split for model selection; 1x untouched held-out test'],
                ['Evaluation status', eval_status],
            ],
            [165, 310]
        ),
        Spacer(1, 18),
        P('<b>Project value:</b> Unlike binary malicious/benign models, Axynetra distinguishes nine attack families and benign traffic. It enforces strict train/validation/test separation, minority-class-aware macro-F1 model selection, global feature importance with non-causal guardrails, and an illustrative analyst triage queue.'),
        S('Important integrity note: Synthetic smoke-test measurements are clearly separated from genuine UNSW-NB15 dataset evaluations and must never be misrepresented as real benchmark accuracy.'),
        PageBreak()
    ]

    story += [
        H1('1. Problem statement'),
        P('Develop a supervised machine-learning model to classify different types of cyberattacks using the UNSW-NB15 dataset. Perform preprocessing and feature engineering, train multiple classification algorithms, and compare their performance using confusion matrices, precision, recall, F1-score and macro/weighted averages.'),
        H2('Extended innovation objective'),
        P('Develop an offline security-analyst research workflow that prevents test-set selection leakage via a stratified validation split, highlights global feature importance with appropriate caveats, recognizes extreme class imbalance, and ranks predicted attack flows for human investigation. This is an offline CSV classification prototype, <b>not</b> a live intrusion-detection or automated blocking system.'),
        H1('2. Why this topic is meaningful'),
        P('Cybersecurity datasets contain attacks with substantially different frequencies. A classifier can appear strong on overall accuracy while failing rare categories such as Worms or Shellcode. Evaluating each of the ten categories and selecting models by validation macro-averaged F1 gives a rigorous assessment.'),
        H1('3. Dataset description'),
        P('UNSW-NB15 was developed at UNSW Canberra using IXIA PerfectStorm to generate a mixture of normal activities and simulated modern attack behaviors. The published pre-split subset includes 175,341 training records and 82,332 testing records across 10 target categories.'),
        table(
            ['Data property', 'Description'],
            [
                ['Target column', 'attack_cat; Normal plus nine attack types (Backdoors normalized to Backdoor)'],
                ['Examples of numerical features', 'dur, sbytes, dbytes, sttl, dttl, rate, ct_state_ttl, ct_srv_src'],
                ['Examples of categorical features', 'proto, service, state'],
                ['Excluded from predictors', 'attack_cat, binary label, id (strictly excluded to prevent target/ID leakage)'],
                ['Training records (official full)', '175,341 (split internally into 80% fit-train and 20% stratified validation)'],
                ['Testing records (official full)', '82,332 (untouched held-out test set evaluated once on validation winner)'],
                ['Data source', 'https://research.unsw.edu.au/projects/unsw-nb15-dataset'],
            ],
            [162, 313]
        ),
        Spacer(1, 7),
        figblock(root / 'published_class_distribution.png', 476),
        S('Figure 1. Published class counts (UNSW-NB15 official predefined split). Source: UNSW dataset documentation. This is not a model prediction chart.'),
        Spacer(1, 16)
    ]

    story += [
        H1('4. Proposed system architecture'),
        figblock(root / 'architecture_flow.png', 490),
        S('Figure 2. Axynetra offline workflow. Candidate models are compared on a stratified validation split; the winning model is evaluated once on the held-out test set.'),
        H1('5. Machine-learning methodology & leakage prevention'),
        table(
            ['Stage', 'Method and rationale'],
            [
                ['Data inspection & normalization', 'Validate attack_cat presence; normalize spelling variants such as Backdoors -> Backdoor; exclude label and id.'],
                ['Stratified validation split', 'Split training data (80% fit-train, 20% validation) before preprocessing. Guarantee >=1 validation and >=1 train row for every class with >=2 rows; log fallback if a class has only 1 row.'],
                ['Preprocessing (Train-only fit)', 'Fit median numerical imputation, StandardScaler, mode categorical imputation, and OneHotEncoder(handle_unknown="ignore") strictly on the training split.'],
                ['Algorithm comparison (4 models)', 'Decision Tree, Random Forest, Extra Trees, and multinomial Logistic Regression trained on fit-train and ranked on validation macro F1.'],
                ['One-time held-out test evaluation', 'Refit only the validation macro-F1 winner on full training data and evaluate exactly once on the official held-out test set.'],
                ['Global & local explanations', 'Export impurity-based global feature importances for compatible tree models (non-causal caveat) plus illustrative standardized z-score feature deviations per alert.'],
                ['Analyst triage queue', 'Combine predicted class, uncalibrated model confidence, and illustrative severity weights (0–5); flag all non-normal or low-confidence flows for human review.'],
            ],
            [140, 335]
        ),
        H2('Design decisions'),
        P('Model ranking and selection are performed <b>strictly on the stratified validation split using macro F1</b> so that the official held-out test set is never used for model selection or tuning. Categorical values unseen at evaluation time are ignored safely. Model confidence scores are explicitly documented as uncalibrated.'),
        H1('6. Evaluation criteria'),
        P('<b>Precision</b> = TP / (TP + FP); <b>Recall</b> = TP / (TP + FN); <b>F1</b> = 2PR / (P + R). <b>Macro F1</b> averages class-wise F1 equally across all 10 classes so rare classes (e.g., Worms) carry equal weight to frequent classes, while <b>Weighted F1</b> weights by class support.'),
        Spacer(1, 16)
    ]

    val_list = meta.get('validation_leaderboard', meta.get('model_comparison', []))
    val_rows = [
        [
            r['model'] + (' *' if r.get('selected_for_test') else ''),
            f"{r['accuracy']:.3f}",
            f"{r.get('macro_precision', 0.0):.3f}",
            f"{r.get('macro_recall', 0.0):.3f}",
            f"{r['macro_f1']:.3f}",
            f"{r['weighted_f1']:.3f}",
        ]
        for r in val_list
    ]
    test_eval = meta.get('heldout_test_evaluation', {})
    test_row = [[
        test_eval.get('model', meta.get('best_model', 'N/A')),
        f"{test_eval.get('accuracy', 0.0):.3f}",
        f"{test_eval.get('macro_precision', 0.0):.3f}",
        f"{test_eval.get('macro_recall', 0.0):.3f}",
        f"{test_eval.get('macro_f1', 0.0):.3f}",
        f"{test_eval.get('weighted_f1', 0.0):.3f}",
    ]] if test_eval else []

    fit_r = meta.get('fit_train_rows', meta.get('train_rows', 0))
    val_r = meta.get('val_rows', 0)
    test_r = meta.get('test_rows', 0)

    story += [
        H1('7. Executed pipeline verification (Synthetic Smoke Test)' if is_smoke else '7. Executed UNSW-NB15 validation & held-out test evaluation'),
        P(
            f"<b>Experiment status:</b> Executed on {meta['train_rows']:,} training records "
            f"(split into {fit_r:,} fit-train and {val_r:,} stratified validation records) and "
            f"{test_r:,} untouched held-out test records with {meta['predictor_count']} predictors. "
            + ('This synthetic smoke run validates the end-to-end leak-free code pipeline and is NOT a scientific measurement of UNSW-NB15 performance.'
               if is_smoke else
               'This run reflects measured performance on the official UNSW-NB15 dataset split.')
        ),
        H2('7.1 Stratified validation leaderboard (used for model selection)'),
        table(['Algorithm (* = Val Winner)', 'Val Acc', 'Val Macro P', 'Val Macro R', 'Val Macro F1', 'Val Weighted F1'],
              val_rows, [145, 62, 68, 68, 66, 66]),
        H2('7.2 One-time untouched held-out test evaluation (selected winner only)'),
        table(['Selected Winner', 'Test Acc', 'Test Macro P', 'Test Macro R', 'Test Macro F1', 'Test Weighted F1'],
              test_row, [145, 62, 68, 68, 66, 66]),
        Spacer(1, 7),
        figblock(outputs / 'confusion_matrix.png', 430),
        S('Figure 3. Row-normalized held-out test confusion matrix ('
          + ('Synthetic smoke test only — not UNSW-NB15 data' if is_smoke else 'Supplied held-out CSV data; origin not verified') + ').'),
        Spacer(1, 16)
    ]

    story += [
        H1('8. How to run on the official data (Windows PowerShell)'),
        P('1. Download the official academic-use <b>UNSW_NB15_training-set.csv</b> and <b>UNSW_NB15_testing-set.csv</b> from UNSW Canberra and place them in <b>data/</b>.'),
        P('2. From the project root in Windows PowerShell, set up the virtual environment and run the smoke test:'),
        Preformatted(
            'py -m venv .venv\n'
            '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n'
            '.\\.venv\\Scripts\\python.exe -m pip install streamlit reportlab pytest\n'
            '.\\.venv\\Scripts\\python.exe multiclass_intrusion.py --smoke-test --output outputs',
            styles['CodeCaption']
        ),
        P('3. Run a stratified subsample on the real dataset (recommended first pass for modest RAM / integrated graphics):'),
        Preformatted(
            '.\\.venv\\Scripts\\python.exe multiclass_intrusion.py --train data/UNSW_NB15_training-set.csv --test data/UNSW_NB15_testing-set.csv --max-train 50000 --max-test 20000 --output outputs',
            styles['CodeCaption']
        ),
        P('4. Run on the full official dataset split (--max-train 0 --max-test 0) and regenerate the PDF report:'),
        Preformatted(
            '.\\.venv\\Scripts\\python.exe multiclass_intrusion.py --train data/UNSW_NB15_training-set.csv --test data/UNSW_NB15_testing-set.csv --max-train 0 --max-test 0 --output outputs\n'
            '.\\.venv\\Scripts\\python.exe create_report.py',
            styles['CodeCaption']
        ),
        H1('9. Original dataset results'),
    ]

    if is_smoke:
        story += [
            P('The genuine UNSW-NB15 model comparison will populate here automatically when <b>create_report.py</b> is re-executed after running <b>multiclass_intrusion.py</b> on the official CSVs. In accordance with academic integrity safeguards, no UNSW-NB15 metrics are fabricated while the official CSV files are not present.'),
            table(
                ['Measure', 'Original UNSW-NB15 Status'],
                [
                    ['Training / test records', '175,341 / 82,332 (official full split; 80/20 train/val internal split)'],
                    ['Validation Macro-F1 winner', 'Pending execution with official UNSW-NB15 CSV files'],
                    ['Held-out test accuracy', 'Pending execution with official UNSW-NB15 CSV files'],
                    ['Held-out test Macro F1', 'Pending execution with official UNSW-NB15 CSV files'],
                    ['Rare-class recall (Worms / Shellcode)', 'Pending execution with official UNSW-NB15 CSV files'],
                ],
                [220, 255]
            ),
            Spacer(1, 16)
        ]
    else:
        per_cls = meta.get('heldout_test_per_class', {})
        worms_rec = per_cls.get('Worms', {}).get('recall', 0.0)
        shell_rec = per_cls.get('Shellcode', {}).get('recall', 0.0)
        story += [
            P('Measured performance on the supplied CSV split using validation selection and one held-out evaluation per run (dataset origin is not verified):'),
            table(
                ['Measure', 'Measured UNSW-NB15 Value'],
                [
                    ['Training / Val / Held-Out Test rows', f"{meta['train_rows']:,} ({fit_r:,} fit / {val_r:,} val) / {test_r:,} test"],
                    ['Validation Macro-F1 winner', str(meta.get('best_model'))],
                    ['Held-out test accuracy', f"{test_eval.get('accuracy', 0.0):.4f}"],
                    ['Held-out test Macro F1', f"{test_eval.get('macro_f1', 0.0):.4f}"],
                    ['Held-out test Weighted F1', f"{test_eval.get('weighted_f1', 0.0):.4f}"],
                    ['Rare-class recall (Worms / Shellcode)', f"Worms: {worms_rec:.4f} | Shellcode: {shell_rec:.4f}"],
                ],
                [220, 255]
            ),
            Spacer(1, 16)
        ]

    story += [
        H1('10. Innovative additions and impact'),
        H2('10.1 Leak-free validation selection & rare-class guarantee'),
        P('Candidate models are ranked strictly on a stratified validation split derived from training data, keeping the official test split untouched until final evaluation. Every rare class with at least 2 training samples is guaranteed representation in both fit-train and validation splits.'),
        H2('10.2 Attack-type taxonomy & imbalance awareness'),
        P('Rather than a binary alert flag, Axynetra classifies flows into Normal, Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, or Worms using class-weighted estimators and macro-F1 optimization.'),
        H2('10.3 Analyst-oriented global & local feature signals'),
        P('Global impurity-based feature importances are paired with per-alert standardized z-score feature deviations from training medians, explicitly caveated as associative and non-causal indicators.'),
        H2('10.4 Priority-based alert triage'),
        P('The exported SOC queue combines predicted attack category, uncalibrated model confidence, and illustrative severity weights (0–5), flagging all non-normal or low-confidence flows for human review.'),
        H1('11. Limitations and responsible interpretation'),
        P('Axynetra is an offline research prototype trained on stored historical CSV records. It does not perform live packet capture, real-time intrusion prevention, or automated blocking. Model probabilities are uncalibrated, impurity importances are not causal explanations, and performance can shift under concept drift or adversarial evasion.'),
        H1('12. Conclusion'),
        P('Axynetra demonstrates a complete, reproducible multi-class cybersecurity analytics pipeline for CSMA Option 4: leak-free train/validation/test separation, four-model comparison under extreme class imbalance, explainable feature reporting, an interactive offline Streamlit dashboard, and automated unit tests.'),
        H1('13. References'),
        S('[1] UNSW Canberra. UNSW-NB15 dataset. https://research.unsw.edu.au/projects/unsw-nb15-dataset'),
        S('[2] Moustafa, N. and Slay, J. (2015), UNSW-NB15: a comprehensive data set for network intrusion detection systems, MilCIS 2015.'),
        S('[3] Moualla et al. (2021), Improving the Performance of Machine Learning-Based Network Intrusion Detection Systems on the UNSW-NB15 Dataset. https://pmc.ncbi.nlm.nih.gov/articles/PMC8221886/'),
        PageBreak(),
        H1('Appendix A. Python source code (multiclass_intrusion.py)'),
        S('Complete offline training, validation selection, held-out evaluation, explainability, and SOC triage source code.')
    ]

    code = (root / 'multiclass_intrusion.py').read_text(encoding='utf-8').splitlines()
    for p in range(0, len(code), 47):
        chunk = code[p:p + 47]
        wrapped = []
        for line in chunk:
            wrapped.extend(
                textwrap.wrap(line, width=96, replace_whitespace=False, drop_whitespace=False, subsequent_indent='   ')
                or ['']
            )
        story.append(Preformatted('\n'.join(wrapped), styles['CodeCaption'], maxLineLength=97))
        if p + 47 < len(code):
            story.append(PageBreak())

    def footer(c, d):
        c.saveState()
        w, h = d.pagesize
        c.setStrokeColor(HexColor('#CBD9E4'))
        c.line(48, 41, w - 48, 41)
        c.setFont('DejaVu', 7)
        c.setFillColor(MUTE)
        c.drawString(48, 28, 'CSMA | CSIT 5A | Axynetra | Offline Academic Research Prototype')
        c.drawRightString(w - 48, 28, f'Page {d.page}')
        c.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(filename, filename.stat().st_size)
    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Axynetra PDF report.')
    parser.add_argument('--srn', type=str, default='SRNNO', help='Student SRN for filename and cover table (default: SRNNO)')
    args = parser.parse_args()
    generate_pdf(args.srn)
