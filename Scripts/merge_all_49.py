import os
import scanpy as sc

TOP_FOLDER = "/Users/shreyanandakumar/Downloads/selected_49_samples"
OUTPUT_PATH = "/Users/shreyanandakumar/Downloads/merged_49_samples.h5ad"

adatas = []
sample_ids = []

for root, dirs, files in os.walk(TOP_FOLDER):
    dirs.sort()
    files.sort()

    for f in files:
        if f.endswith(".h5"):
            full_path = os.path.join(root, f)
            sample_id = os.path.basename(root)

            print("Loading:", full_path)
            adata = sc.read_10x_h5(full_path)
            adata.var_names_make_unique()
            adata.obs["sample_id"] = sample_id

            adatas.append(adata)
            sample_ids.append(sample_id)

print(f"Total files loaded: {len(adatas)}")

if len(adatas) == 0:
    raise ValueError("No .h5 files found. Check folder path.")

if len(sample_ids) != len(set(sample_ids)):
    print("Warning: duplicate sample IDs detected.")

combined = sc.concat(
    adatas,
    join="inner",
    index_unique="-"
)

combined.obs_names_make_unique()

print("Combined shape:", combined.shape)
print("Unique samples:", combined.obs["sample_id"].nunique())

combined.write_h5ad(OUTPUT_PATH)
print(f"Saved merged file to: {OUTPUT_PATH}")