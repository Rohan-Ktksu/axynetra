"""Render a fixed-layout README diagram without Mermaid HTML-label clipping."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]


def main():
    fig, ax = plt.subplots(figsize=(12, 17.8), facecolor='#f8fafc')
    ax.set(xlim=(0, 12), ylim=(0, 17.8))
    ax.axis('off')
    ax.text(6, 17.35, 'AXYNETRA · SUPERVISED ML PIPELINE', ha='center',
            fontsize=18, weight='bold', color='#15344b')
    ax.text(6, 16.97, 'Train and select using training data; evaluate on held-out test data',
            ha='center', fontsize=11, color='#526575')

    def box(x, y, text, width=6.1, height=.93, face='#e7f3f5', edge='#267b83'):
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                     boxstyle='round,pad=0.06,rounding_size=0.1',
                     linewidth=1.4, edgecolor=edge, facecolor=face))
        ax.text(x, y, text, ha='center', va='center', fontsize=12.2,
                color='#15344b', linespacing=1.45)

    def arrow(start, end, **kwargs):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='-|>', color='#526575', lw=1.6,
                                    mutation_scale=15, **kwargs))

    box(6, 16.1, '1. Load data\nSupplied train/test CSVs or explicit synthetic mode', width=9)
    box(6, 14.65, '2. Validate inputs and prevent identifiable leakage\nCheck labels, duplicates and overlap; exclude target / label / ID', width=10.4)
    arrow((6,15.60),(6,15.18))
    box(4, 13.15, '3. Training records\nInspect features, missingness and class counts')
    box(10, 13.15, 'Held-out test records\nReserved for\nfinal evaluation', width=2.7, height=1.3,
        face='#fff3df', edge='#b7812e')
    arrow((4,14.13),(4,13.69))
    arrow((10,14.13),(10,13.86))
    stages = [
        (11.65, '4. Stratified training split\nApproximately 80% fit-training / 20% validation'),
        (10.15, '5. Fit preprocessing and four classifiers\nUse fit-training records only'),
        (8.65, '6. Evaluate candidates on validation\nApply already-fitted transforms; compare metrics'),
        (7.15, '7. Select the winner\nValidation Macro F1; validation-only tie-breakers'),
        (5.65, '8. Refit the selected pipeline\nNew preprocessor + selected classifier\nFit on the full training set'),
        (4.15, '9. Final held-out test evaluation\nEvaluate the winner once per experiment run'),
        (2.65, '10. Metrics and exploratory signals\nClass report / confusion matrix / ROC-AUC\nGlobal feature importance'),
        (1.15, '11. Export results and display dashboard\nIllustrative analyst queue · human review'),
    ]
    previous = 13.15
    for y, text in stages:
        box(4, y, text)
        arrow((4, previous-.53),(4,y+.53))
        previous = y
    ax.plot([10,10],[12.44,4.15],color='#b7812e',lw=1.7)
    arrow((10,4.15),(7.15,4.15))
    ax.text(10.25,8.35,'No fitting or model selection',rotation=90,
            ha='center',va='center',fontsize=11,color='#946416')
    ax.text(6,.26,'Numeric: impute + scale  |  Categorical: impute + one-hot encode  |  No autoencoder',
            ha='center',fontsize=10.5,color='#526575')
    target = ROOT/'docs'/'pipeline'
    target.mkdir(parents=True,exist_ok=True)
    fig.savefig(target/'axynetra_ml_pipeline.png',dpi=160,bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(target/'axynetra_ml_pipeline.svg',bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Created PNG and SVG pipeline diagrams.')


if __name__=='__main__':
    main()
