import numpy as np
import pandas as pd
import scanpy as sc

# ── paths ──────────────────────────────────────────────────────────────────────
MERGED_PATH  = "/Users/shreyabalamurugan/merged_49_samples.h5ad"
METADATA_CSV = "/Users/shreyabalamurugan/crohns-scRNA-inflammation-prediction/Scripts/CD_PreTreatment_Metadata.csv"
OUTPUT_PATH  = "/Users/shreyabalamurugan/preprocessed.h5ad"

N_HVG        = 2000
MIN_GENES    = 200
MAX_PCT_MITO = 20
MIN_CELLS    = 10

# ── step 1: load merged file ───────────────────────────────────────────────────
combined = sc.read_h5ad(MERGED_PATH)
print(f"Loaded: {combined.shape[0]:,} cells × {combined.shape[1]:,} genes")

# ── step 2: attach metadata ────────────────────────────────────────────────────
meta = pd.read_csv(METADATA_CSV)
meta["folder_name"] = meta["title"].str.replace("-reup", "", regex=False)

# build a mapping from sample_id to metadata
meta_dict = meta.set_index("folder_name").to_dict(orient="index")

combined.obs["patient"]            = combined.obs["sample_id"].map(lambda x: meta_dict.get(x, {}).get("patient"))
combined.obs["site"]               = combined.obs["sample_id"].map(lambda x: meta_dict.get(x, {}).get("site"))
combined.obs["batch"]              = combined.obs["sample_id"].map(lambda x: meta_dict.get(x, {}).get("batch"))
combined.obs["inflammation"]       = combined.obs["sample_id"].map(lambda x: meta_dict.get(x, {}).get("inflammation"))
combined.obs["inflammation_score"] = combined.obs["sample_id"].map(lambda x: meta_dict.get(x, {}).get("inflammation score"))

print("Metadata attached. Sample check:")
print(combined.obs[["sample_id", "patient", "inflammation"]].drop_duplicates().head(10))

# ── step 3: QC filtering ───────────────────────────────────────────────────────
combined.var["mt"] = combined.var_names.str.startswith("MT-")

sc.pp.calculate_qc_metrics(
    combined,
    qc_vars=["mt"],
    percent_top=None,
    log1p=False,
    inplace=True
)

before = combined.n_obs
combined = combined[combined.obs["n_genes_by_counts"] >= MIN_GENES, :]
combined = combined[combined.obs["pct_counts_mt"]     <= MAX_PCT_MITO, :]
sc.pp.filter_genes(combined, min_cells=MIN_CELLS)

print(f"After QC: {combined.n_obs:,} cells ({before - combined.n_obs:,} removed) × {combined.n_vars:,} genes")

# ── step 4: normalize + log1p ─────────────────────────────────────────────────
combined.layers["counts"] = combined.X.copy()
sc.pp.normalize_total(combined, target_sum=1e4)
sc.pp.log1p(combined)

# ── step 5: highly variable gene selection ────────────────────────────────────
sc.pp.highly_variable_genes(
    combined,
    n_top_genes=N_HVG,
    batch_key="batch",
    flavor="seurat_v3",
    subset=False
)

print(f"HVGs selected: {combined.var['highly_variable'].sum()}")

# ── step 6: subset to HVGs + scale ───────────────────────────────────────────
combined = combined[:, combined.var["highly_variable"]].copy()
sc.pp.scale(combined, max_value=10)

print(f"After HVG subset: {combined.shape}")

# ── step 7: encode labels ─────────────────────────────────────────────────────
combined.obs["label"] = (combined.obs["inflammation"] == "Inflamed").astype(int)

print("\nClass balance:")
print(combined.obs["label"].value_counts())

print("\nCells per patient:")
print(combined.obs.groupby(["patient", "inflammation"]).size().unstack(fill_value=0))

# ── step 8: save ──────────────────────────────────────────────────────────────
combined.write_h5ad(OUTPUT_PATH)
print(f"\nSaved to {OUTPUT_PATH}")
print(f"X matrix shape for ML: {combined.X.shape}")