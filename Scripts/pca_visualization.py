import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
X_PATH = "/Users/shreyanandakumar/Downloads/sample_level_X.npy"
META_PATH = "/Users/shreyanandakumar/Downloads/sample_level_metadata.csv"
OUTPUT_PREFIX = "/Users/shreyanandakumar/Downloads/pca_sample_level"

# Load data
X = np.load(X_PATH)
meta = pd.read_csv(META_PATH)

print("X shape:", X.shape)
print("Metadata shape:", meta.shape)

# Center data
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# Manual PCA using SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Principal components / scores
X_pca = X_centered @ Vt.T

# Explained variance
n_samples = X.shape[0]
explained_variance = (S ** 2) / (n_samples - 1)
explained_variance_ratio = explained_variance / explained_variance.sum()
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Print top variance info
print("\nExplained variance ratio for first 10 PCs:")
for i in range(min(10, len(explained_variance_ratio))):
    print(f"PC{i+1}: {explained_variance_ratio[i]:.4f}")

print("\nCumulative explained variance ratio for first 10 PCs:")
for i in range(min(10, len(cumulative_variance_ratio))):
    print(f"PC1-PC{i+1}: {cumulative_variance_ratio[i]:.4f}")

# Save PCA scores
pc_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
pca_df = pd.DataFrame(X_pca, columns=pc_columns)
pca_df = pd.concat([meta.reset_index(drop=True), pca_df], axis=1)

pca_df.to_csv(f"{OUTPUT_PREFIX}_scores.csv", index=False)
print(f"\nSaved PCA scores to {OUTPUT_PREFIX}_scores.csv")

# Scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_scree.png", dpi=300)
plt.show()

# Cumulative variance plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("Cumulative Explained Variance")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_cumulative_variance.png", dpi=300)
plt.show()

# Helper function for PCA scatter plots
def plot_pca_scatter(df, color_col, title, output_file):
    plt.figure(figsize=(8, 6))

    categories = df[color_col].astype(str).unique()

    for cat in categories:
        idx = df[color_col].astype(str) == cat
        plt.scatter(
            df.loc[idx, "PC1"],
            df.loc[idx, "PC2"],
            label=cat,
            alpha=0.8,
            s=60
        )

    plt.xlabel(f"PC1 ({explained_variance_ratio[0]*100:.2f}% var)")
    plt.ylabel(f"PC2 ({explained_variance_ratio[1]*100:.2f}% var)")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

# Step 8: plots
plot_pca_scatter(
    pca_df,
    color_col="patient",
    title="PCA of Sample-Level Data Colored by Patient",
    output_file=f"{OUTPUT_PREFIX}_by_patient.png"
)

plot_pca_scatter(
    pca_df,
    color_col="inflammation",
    title="PCA of Sample-Level Data Colored by Inflammation",
    output_file=f"{OUTPUT_PREFIX}_by_inflammation.png"
)

plot_pca_scatter(
    pca_df,
    color_col="batch",
    title="PCA of Sample-Level Data Colored by Batch",
    output_file=f"{OUTPUT_PREFIX}_by_batch.png"
)


plot_pca_scatter(
    pca_df,
    color_col="site",
    title="PCA of Sample-Level Data Colored by Site",
    output_file=f"{OUTPUT_PREFIX}_by_site.png"
)

print("\nDone.")
