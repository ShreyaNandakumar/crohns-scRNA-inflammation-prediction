#code to visualize the merged file and check that it loaded correctly

import anndata
#load the file into an AnnData object
adata = anndata.read_h5ad("location of all merged_49_samples.h5ad file")
#access different parts of the data (like the main matrix)
print(adata.shape)
print(adata.X)
print(adata.obs) #cell metadata
print(adata.var) #gene metadata
