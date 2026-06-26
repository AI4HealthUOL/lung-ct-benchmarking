# Foundation Models vs. Radiomics for Lung Computed Tomography: A Benchmark of Feature Extractors, Classification Heads, and Segmentation Choices

![](./assets/highlight.png "FeatInv overview")

## Setup

```bash
pip install -r requirements.txt
export ROOT=/path/to/data
export HF_TOKEN=...
export TABPFN_TOKEN=...
# Optional experiment tracking:
export MLFLOW_TRACKING_URI=...
export MLFLOW_TRACKING_USERNAME=...
export MLFLOW_TRACKING_PASSWORD=...
```

Expected directory structure under `ROOT` (see `config.py` for the full
list): `NSCLC-Radiomics-NIFTI/`, `NSCLC-Radiogenomics-NIFTI/`, the two clinical
CSVs under `lung1/` and `lung2/`, and `radiomics_benchmark/cv_folds.json`.

## 🚀 Pipeline

```bash
# 1. Extract radiomics features for both datasets
python extract_radiomics.py --dataset lung1
python extract_radiomics.py --dataset lung2

# 2. Attach clinical labels
python prepare_labels.py --dataset lung1
python prepare_labels.py --dataset lung2    # reuses the lung1 volume threshold

# 3a. Cross-validation benchmark
python benchmark.py --task cv --radiomics
python benchmark.py --task cv --curia
python benchmark.py --task cv --curia2
python benchmark.py --task cv --dinov3

# 3b. External validation
python benchmark.py --task external --all

# 4. Bootstrap confidence intervals over the prediction CSVs
python bootstrap.py --dir results/cv/curia/predictions
```
