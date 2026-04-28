import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# Paths
INPUT_PATH = "/Users/shreyanandakumar/Downloads/preprocessed_49_samples.h5ad"
OUTPUT_PREFIX = "/Users/shreyanandakumar/Downloads/sample_level"

# Load preprocessed AnnData
adata = sc.read_h5ad(INPUT_PATH)
print(f"Loaded AnnData: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# Check required metadata columns
required_cols = ["sample_id", "patient", "label", "inflammation"]
missing_cols = [col for col in required_cols if col not in adata.obs.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in adata.obs: {missing_cols}")

# Get unique samples
sample_ids = adata.obs["sample_id"].unique()
print(f"Number of unique samples: {len(sample_ids)}")

# Aggregate mean expression per sample
X_sample_list = []
metadata_rows = []

for sample_id in sample_ids:
    sample_mask = adata.obs["sample_id"] == sample_id
    adata_sample = adata[sample_mask]

    X = adata_sample.X

    # mean across cells
    if sparse.issparse(X):
        mean_expr = np.asarray(X.mean(axis=0)).ravel()
    else:
        mean_expr = X.mean(axis=0)

    X_sample_list.append(mean_expr)

    # metadata for this sample
    sample_obs = adata_sample.obs.iloc[0]
    metadata_rows.append({
        "sample_id": sample_id,
        "patient": sample_obs["patient"],
        "label": int(sample_obs["label"]),
        "inflammation": sample_obs["inflammation"],
        "batch": sample_obs["batch"],
        "site": sample_obs["site"]
    })

# Create final arrays/dataframes
X_sample = np.vstack(X_sample_list)
sample_metadata = pd.DataFrame(metadata_rows)

print("Sample-level feature matrix shape:", X_sample.shape)
print("Sample-level metadata shape:", sample_metadata.shape)

# Save outputs
np.save(f"{OUTPUT_PREFIX}_X.npy", X_sample)
np.save(f"{OUTPUT_PREFIX}_y.npy", sample_metadata["label"].values)
np.save(f"{OUTPUT_PREFIX}_groups.npy", sample_metadata["patient"].values)

sample_metadata.to_csv(f"{OUTPUT_PREFIX}_metadata.csv", index=False)
pd.DataFrame(X_sample, index=sample_metadata["sample_id"], columns=adata.var_names).to_csv(
    f"{OUTPUT_PREFIX}_X.csv"
)

print("\nSaved files:")
print(f"{OUTPUT_PREFIX}_X.npy")
print(f"{OUTPUT_PREFIX}_y.npy")
print(f"{OUTPUT_PREFIX}_groups.npy")
print(f"{OUTPUT_PREFIX}_metadata.csv")
print(f"{OUTPUT_PREFIX}_X.csv")

# Sanity checks
print("\nClass balance at sample level:")
print(sample_metadata["label"].value_counts())

print("\nSamples per patient:")
print(sample_metadata["patient"].value_counts().sort_index())

print("\nInflammation labels:")
print(sample_metadata["inflammation"].value_counts())
