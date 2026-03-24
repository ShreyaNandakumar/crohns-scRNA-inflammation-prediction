import scanpy as sc
import anndata as ad
import os

top_folder = "/Users/mrunmayeewankhede/Desktop/cmu/Spring 26/Machine Learning for Scientists/Project/selected_49_samples"

adatas = []

#load all files
for root, dirs, files in os.walk(top_folder):
    for f in files:
        if f.endswith(".h5"):
            full_path = os.path.join(root, f)
            sample_id = os.path.basename(root)

            print("Loading:", full_path)
            adata = sc.read_10x_h5(full_path)

            print(f"{sample_id} shape:", adata.shape)

            #fix duplicate gene names
            adata.var_names_make_unique()

            #add sample ID
            adata.obs['sample_id'] = sample_id

            adatas.append(adata)

#check how many files were loaded
print("Total files loaded:", len(adatas))


#merge all loaded files
if len(adatas) > 0:
    combined = ad.concat(adatas, join='outer')

    print("Combined shape:", combined.shape)
else:
    raise ValueError("No files loaded!")


#save the merged file
combined.write("/Users/mrunmayeewankhede/Desktop/updated_merged_49_samples.h5ad")
print("Saved merged file!")