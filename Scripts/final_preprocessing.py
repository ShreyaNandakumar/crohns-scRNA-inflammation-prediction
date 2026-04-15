import pandas as pd
import scanpy as sc

# Paths
MERGED_PATH = "/Users/shreyanandakumar/Downloads/merged_49_samples.h5ad"
METADATA_CSV = "/Users/shreyanandakumar/crohns-scRNA-inflammation-prediction/Scripts/CD_PreTreatment_Metadata.csv"
OUTPUT_PATH = "/Users/shreyanandakumar/Downloads/preprocessed_49_samples.h5ad"

# Parameters
N_HVG = 2000
MIN_GENES = 200
MAX_PCT_MITO = 20
MIN_CELLS = 10

# Step 1: load merged data
combined = sc.read_h5ad(MERGED_PATH)
print(f"Loaded: {combined.shape[0]:,} cells × {combined.shape[1]:,} genes")

# Step 2: load metadata and map to samples
meta = pd.read_csv(METADATA_CSV)
meta["folder_name"] = meta["title"].str.replace("-reup", "", regex=False)

# Keep only one row per folder_name in case of accidental duplicates
meta = meta.drop_duplicates(subset="folder_name")

meta_dict = meta.set_index("folder_name").to_dict(orient="index")

combined.obs["patient"] = combined.obs["sample_id"].map(lambda x: meta_dict.get(x, {}).get("patient"))
combined.obs["site"] = combined.obs["sample_id"].map(lambda x: meta_dict.get(x, {}).get("site"))
combined.obs["batch"] = combined.obs["sample_id"].map(lambda x: meta_dict.get(x, {}).get("batch"))
combined.obs["inflammation"] = combined.obs["sample_id"].map(lambda x: meta_dict.get(x, {}).get("inflammation"))
combined.obs["inflammation_score"] = combined.obs["sample_id"].map(lambda x: meta_dict.get(x, {}).get("inflammation score"))

print("\nMissing metadata counts:")
print(combined.obs[["patient", "site", "batch", "inflammation", "inflammation_score"]].isna().sum())

# Optional safety filter: keep only cells with mapped metadata
combined = combined[combined.obs["patient"].notna()].copy()

# Step 3: QC metrics
combined.var["mt"] = combined.var_names.str.upper().str.startswith("MT-")

sc.pp.calculate_qc_metrics(
    combined,
    qc_vars=["mt"],
    percent_top=None,
    log1p=False,
    inplace=True
)

before_cells = combined.n_obs
before_genes = combined.n_vars

combined = combined[combined.obs["n_genes_by_counts"] >= MIN_GENES].copy()
combined = combined[combined.obs["pct_counts_mt"] <= MAX_PCT_MITO].copy()
sc.pp.filter_genes(combined, min_cells=MIN_CELLS)

print(f"\nAfter QC: {combined.n_obs:,} cells ({before_cells - combined.n_obs:,} removed)")
print(f"After QC: {combined.n_vars:,} genes ({before_genes - combined.n_vars:,} removed)")

# Step 4: save raw counts before normalization
combined.layers["counts"] = combined.X.copy()

# Step 5: highly variable genes on counts for seurat_v3
sc.pp.highly_variable_genes(
    combined,
    n_top_genes=N_HVG,
    batch_key="batch",
    flavor="seurat_v3",
    layer="counts",
    subset=False
)

print(f"\nHVGs selected: {combined.var['highly_variable'].sum()}")

# Step 6: normalize and log-transform
sc.pp.normalize_total(combined, target_sum=1e4)
sc.pp.log1p(combined)

# Step 7: keep only HVGs and scale
combined = combined[:, combined.var["highly_variable"]].copy()
sc.pp.scale(combined, max_value=10)

print(f"After HVG subset: {combined.shape}")

# Step 8: create labels
print("\nInflammation categories before filtering:")
print(combined.obs["inflammation"].value_counts(dropna=False))

combined = combined[combined.obs["inflammation"].isin(["Inflamed", "Non_Inflamed"])].copy()
combined.obs["label"] = (combined.obs["inflammation"] == "Inflamed").astype(int)

print("\nClass balance:")
print(combined.obs["label"].value_counts())

print("\nCells per patient:")
print(combined.obs.groupby(["patient", "inflammation"]).size().unstack(fill_value=0))

# Step 9: save processed file
combined.write_h5ad(OUTPUT_PATH)
print(f"\nSaved to: {OUTPUT_PATH}")
print(f"Final X shape for ML: {combined.X.shape}")