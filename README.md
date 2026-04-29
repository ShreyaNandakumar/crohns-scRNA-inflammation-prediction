# Crohn’s scRNA Inflammation Prediction

Machine learning pipeline for classifying **inflamed vs non-inflamed Crohn’s disease biopsies** using **single-cell RNA-seq**, **PCA**, **kNN**, and **XGBoost**.

## Team
- Mrunmayee Wankhede
- Samridhi Makkar
- Shreya Balamurugan
- Shreya Nandakumar

---

## Project summary

This project uses processed single-cell RNA-seq data from **GEO: GSE282122** to predict whether a Crohn’s disease biopsy is **inflamed** or **non-inflamed**.

The original data are cell-level, but the final machine learning task is performed at the **sample level**. After preprocessing, cells from each biopsy are aggregated into one biopsy-level expression profile, and classification is then performed on these biopsy-level vectors.

We compare:
- **PCA** for dimensionality reduction
- **kNN** as a baseline classifier in PCA space
- **XGBoost** as a nonlinear classifier using PCA features and raw gene features

---

## Dataset

- **Source:** GEO GSE282122  
- **Paper:** Thomas et al., *A longitudinal single-cell atlas of anti-tumour necrosis factor treatment in inflammatory bowel disease*, Nature Immunology, 2024  
- **GEO link:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE282122

### Subset used in this project
- **49** pre-treatment Crohn’s disease biopsies
- **16** patients
- **28 inflamed** / **21 non-inflamed**
- **4 gut sites**
- **4 sequencing batches**

### Final ML dataset
- **49 samples × 2000 features**
- Features are aggregated sample-level gene expression values from the top 2,000 highly variable genes

---

## Repository structure

```text
crohns-scRNA-inflammation-prediction/
├── Data/
│   └── GSE282122_metadata.txt
├── Results/
│   ├── PCA/
│   └── XGBoost/
├── Scripts/
│   ├── metadataSamples.py
│   ├── copy_selected_samples.py
│   ├── merge_all_49.py
│   ├── final_preprocessing.py
│   ├── build_sample_level_dataset.py
│   ├── pca_visualization.py
│   ├── PCA.py
│   ├── kNN.py
│   ├── XGBoost.py
│   ├── XGBoost_plots.py
│   └── CD_PreTreatment_Metadata.csv
└── README.md
```
## Scripts

- `metadataSamples.py`: Filters GEO metadata to the selected pre-treatment Crohn’s samples.
- `copy_selected_samples.py`: Copies the selected 49 sample folders from the larger local dataset.
- `merge_all_49.py`: Loads the selected `.h5` files and merges them into one AnnData object.
- `final_preprocessing.py`: Attaches metadata, performs QC, normalization, HVG selection, scaling, and label creation.
- `build_sample_level_dataset.py`: Aggregates cells into one sample-level profile per biopsy.
- `pca_visualization.py`: Creates exploratory PCA and variance plots.
- `PCA.py`: Implements PCA from scratch inside grouped cross-validation folds and saves train/test PCA splits.
- `kNN.py`: Runs a manual kNN classifier in PCA space.
- `XGBoost.py`: Runs XGBoost across grouped CV folds and saves metrics and feature importances.
- `XGBoost_plots.py`: Plots averaged XGBoost feature importances.

## Setup

Create and activate a Python environment:

```bash
python3 -m venv crohns_scanpy_venv
source crohns_scanpy_venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy matplotlib scanpy xgboost scikit-misc
```

> **Important:** Some scripts contain absolute local paths such as `/Users/...`. Before running the pipeline, update these paths to match your machine.

## Reproducing the Pipeline

### 1. Prepare GEO metadata

Run:

```bash
python Scripts/metadataSamples.py
```

Output:

```text
Scripts/CD_PreTreatment_Metadata.csv
```

### 2. Download and select the 49 sample folders

Download the processed 10x sample folders for the selected 49 biopsies from GSE282122 and place them locally.

Update paths in:

```text
Scripts/copy_selected_samples.py
```

Then run:

```bash
python Scripts/copy_selected_samples.py
```

### 3. Merge the selected samples

Update paths in:

```text
Scripts/merge_all_49.py
```

Run:

```bash
python Scripts/merge_all_49.py
```

Output:

```text
merged_49_samples.h5ad
```

### 4. Preprocess the merged data

Update paths in:

```text
Scripts/final_preprocessing.py
```

Run:

```bash
python Scripts/final_preprocessing.py
```

This step performs metadata attachment, QC filtering, normalization, log transform, HVG selection, scaling, and label creation.

Output:

```text
preprocessed_49_samples.h5ad
```

### 5. Build the sample-level dataset

Update paths in:

```text
Scripts/build_sample_level_dataset.py
```

Run:

```bash
python Scripts/build_sample_level_dataset.py
```

Outputs:

```text
sample_level_X.npy
sample_level_y.npy
sample_level_groups.npy
sample_level_metadata.csv
sample_level_X.csv
```

### 6. Generate exploratory PCA plots

Update paths in:

```text
Scripts/pca_visualization.py
```

Run:

```bash
python Scripts/pca_visualization.py
```

Outputs are saved in:

```text
Results/PCA/
```

### 7. Create PCA features inside grouped CV folds

Update paths in:

```text
Scripts/PCA.py
```

Run:

```bash
python Scripts/PCA.py
```

This step performs grouped cross-validation by patient, fits PCA on the training fold only, projects train and test samples into PCA space, and saves PCA splits for downstream models.

Output:

```text
Results/PCA/pca_splits.npy
```

### 8. Run kNN

```bash
python Scripts/kNN.py
```

### 9. Run XGBoost

```bash
python Scripts/XGBoost.py
```

Outputs:

```text
Results/XGBoost/xgboost_results.csv
Results/XGBoost/feature_importance_*.csv
```

### 10. Plot XGBoost feature importances

```bash
python Scripts/XGBoost_plots.py
```

Plots are saved in:

```text
Results/XGBoost/plots/
```

## Main Outputs

### PCA

```text
Results/PCA/pca_sample_level_by_batch.png
Results/PCA/pca_sample_level_by_inflammation.png
Results/PCA/pca_sample_level_by_site.png
Results/PCA/pca_sample_level_cumulative_variance.png
```
### kNN
- kNN results are generated by `Scripts/kNN.py`

### XGBoost

```text
Results/XGBoost/xgboost_results.csv
Results/XGBoost/plots/feature_importance_best_auroc.png
Results/XGBoost/plots/feature_importance_best_balanced.png
```

## Main Findings

- PCA showed partial separation by inflammation and some structure by gut site.
- Batch did not dominate the first two principal components.
- XGBoost outperformed kNN overall.
- The best raw-feature XGBoost model gave the strongest classification performance.
- PCA-based XGBoost achieved the highest AUROC in one setting.

## Limitations

- Small sample size of 49 biopsies.
- Possible gut site effect.
- Sample-level aggregation may still reflect cell-type composition differences.

## Future Work

- Perform within-cell-type modeling.
- Apply stronger batch correction if needed.
- Expand to larger sample sets.

## Citation

Wankhede, M., Balamurugan, S., Makkar, S., & Nandakumar, S. (2026). Predicting inflamed vs non-inflamed Crohn's tissue from single-cell RNA-seq. GitHub. https://github.com/ShreyaNandakumar/crohns-scRNA-inflammation-prediction
