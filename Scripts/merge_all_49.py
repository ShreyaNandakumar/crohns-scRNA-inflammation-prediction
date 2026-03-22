import scanpy as sc
import os

top_folder = "path_name"

adatas = []

for root, dirs, files in os.walk(top_folder):
    for f in files:
        if f.endswith(".h5"):
            full_path = os.path.join(root, f)
            print("Loading:", full_path)
            adata = sc.read_10x_h5(full_path)
            adata.var_names_make_unique()       # fix per-sample first
            sample_id = os.path.basename(root)
            adata.obs['sample_id'] = sample_id
            adatas.append(adata)

print(f"Total files loaded: {len(adatas)}")

if len(adatas) == 0:
    raise ValueError("No .h5 files found! Check folder path.")

combined = sc.concat(
    adatas,
    join='outer',
    label='sample_id',
    keys=[a.obs['sample_id'][0] for a in adatas],
    index_unique='-'   # avoids cell barcode collisions across samples
)

print("Combined shape:", combined.shape)

combined.write("/Users/mrunmayeewankhede/Desktop/merged_49_samples.h5ad")
print("Saved merged_49_samples.h5ad")