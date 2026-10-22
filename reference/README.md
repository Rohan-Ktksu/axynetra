# Axynetra notebook and CSV reference

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rohan-Ktksu/axynetra/blob/main/reference/Axynetra_UNSW_NB15.ipynb)

[Read the notebook and saved outputs](Axynetra_UNSW_NB15.ipynb) · [Matching Python cells](Axynetra_UNSW_NB15.py) · [CSV data folder](data/)

[View saved outputs in Jupyter nbviewer](https://nbviewer.org/github/Rohan-Ktksu/axynetra/blob/main/reference/Axynetra_UNSW_NB15.ipynb) opens a read-only notebook view; use Colab to execute cells.

This is the CSV-based experiment previously kept in `D:\Axynetra_Jupyter`. It now lives with the GitHub project so the notebook, data and outputs can be opened together. It uses pandas, NumPy, scikit-learn and Matplotlib directly. The root project's synthetic demonstration remains separate.

## Open in a notebook interface

**In your browser:** use the Colab button. Saved outputs are included; connect a Python runtime and select **Runtime → Run all** to recompute them. The first cell fetches the repository's CSVs and checks their SHA-256 values against the included provenance. Colab uses its installed library versions, which the notebook prints; results can differ from the pinned local environment.

**In local Jupyter:** from the main project folder:

```powershell
cd "D:\Axynetra"
.\.venv\Scripts\python.exe -m pip install -r reference\requirements.txt
.\.venv\Scripts\python.exe -m notebook
```

Open `reference/Axynetra_UNSW_NB15.ipynb`, select the project's Python kernel and **Run → Run All Cells**. You can also open it in VS Code. Both the project root and this folder are supported working directories. Results are exported to `reference/outputs/notebook/`, which is ignored by Git; the notebook retains its displayed outputs.

## Included files

| File | Contents |
| --- | --- |
| [Axynetra_UNSW_NB15.ipynb](Axynetra_UNSW_NB15.ipynb) | Complete executed notebook: code, tables, metrics and figures |
| [Axynetra_UNSW_NB15.py](Axynetra_UNSW_NB15.py) | Matching, editable percent-format Python cells |
| [Training CSV](data/UNSW_NB15_training_clean.csv) | 99,268 cleaned training records |
| [Test CSV](data/UNSW_NB15_testing_no_overlap.csv) | 73,791 records after removing overlap with original training predictors |
| [Cleaning summary](data/cleaning_summary.csv) | Record counts at each cleaning stage |
| [Data provenance](data/DATA_PROVENANCE.json) | Original source URLs, pinned mirror revision, hashes and class counts |
| [requirements.txt](requirements.txt) | Pinned local notebook dependencies |
| [prepare_data.py](prepare_data.py) | Reproduce cleaning after supplying the original CSVs in `data/original/` |
| [tools/build_notebook.py](tools/build_notebook.py) | Regenerate the notebook from the matching Python source |

Default execution uses stratified caps of **10,000 training / 5,000 held-out test records**. Set `MAX_TRAIN=0` and `MAX_TEST=0` to use all cleaned rows. Four supervised classifiers compete on validation Macro F1; the winner is refitted on training data before held-out evaluation. Autoencoders are not implemented.

## Dataset origin and modified split

The included files derive from the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset), obtained from a community mirror pinned in the provenance file. The mirror is not an authenticated download directly from UNSW. Untouched originals are not bundled here; their download URLs and hashes are recorded in the manifest.

Cleaning removed 30,528 training rows with conflicting labels for identical predictors, then 45,545 remaining same-label training duplicates. It removed 8,541 test rows whose predictors overlap original training. Test labels did not determine training cleaning; remaining within-test duplicates were retained. All ten categories remain, with 42 predictor features after excluding `id`, `label` and `attack_cat`.

Report these outputs as a **cleaned, sampled UNSW-NB15 experiment**, not performance on the unchanged official benchmark. The root README's synthetic charts are a different experiment.

To regenerate the notebook and its outputs from the project root:

```powershell
.\.venv\Scripts\python.exe reference\tools\build_notebook.py --execute
```

## Attribution

The dataset remains the work of its authors; this project does not claim ownership. Follow the academic-use and citation conditions on the [official dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset), which requests these references:

1. Moustafa, N., & Slay, J. (2015). [UNSW-NB15: a comprehensive data set for network intrusion detection systems](https://ieeexplore.ieee.org/document/7348942). MilCIS.
2. Moustafa, N., & Slay, J. (2016). The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset. Information Security Journal.
3. Moustafa, N., et al. (2017). Novel geometric area analysis technique for anomaly detection using trapezoidal area estimation on large-scale networks. IEEE Transactions on Big Data.
4. Moustafa, N., et al. (2017). Big data analytics for intrusion detection system: statistical decision-making using finite dirichlet mixture models. Data Analytics and Decision Support for Cybersecurity.
5. Sarhan, M., Layeghy, S., Moustafa, N., & Portmann, M. (2020). NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems. Big Data Technologies and Applications.
