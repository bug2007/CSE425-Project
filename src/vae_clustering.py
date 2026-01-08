import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
VIS_DIR = os.path.join(RESULTS_DIR, "latent_visualisation")
os.makedirs(VIS_DIR, exist_ok=True)

NUM_CLUSTERS = 10
PCA_DIM = 32

# LOAD MFCC FEATURES
print("Loading MFCC features")
X = np.load(os.path.join(DATA_DIR, "X_mfcc_truncated.npy"))  # (3996, 1172, 20)

# TEST Data sanity
assert X.ndim == 3, f"Expected 3D MFCCs, got {X.ndim}D"
print("MFCC shape =", X.shape)

# FLATTEN MFCCs
X_flat = X.reshape(X.shape[0], -1)

# TEST Flattening
assert X_flat.shape[0] == X.shape[0]
print("Flattened shape =", X_flat.shape)

# PCA FEATURE EXTRACTION
print("Running PCA")
pca = PCA(n_components=PCA_DIM, random_state=42)
Z_pca = pca.fit_transform(X_flat)

# TEST PCA output
assert Z_pca.shape == (X.shape[0], PCA_DIM)
print("PCA features shape =", Z_pca.shape)

# K-MEANS CLUSTERING
print("Running K-Means on PCA features")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
labels_pca = kmeans.fit_predict(Z_pca)

# TEST Clustering output
unique_clusters = np.unique(labels_pca)
assert len(unique_clusters) > 1, "K-Means collapsed to 1 cluster"
print("Number of clusters =", len(unique_clusters))

# CLUSTERING METRICS
sil_score = silhouette_score(Z_pca, labels_pca)
ch_score = calinski_harabasz_score(Z_pca, labels_pca)

print(f"PCA + KMeans Silhouette Score: {sil_score:.4f}")
print(f"PCA + KMeans Calinski-Harabasz Index: {ch_score:.2f}")

# SAVE METRICS
csv_path = os.path.join(RESULTS_DIR, "vae_clustering_metrics.csv")
file_exists = os.path.isfile(csv_path)

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Method", "Silhouette", "Calinski-Harabasz"])
    writer.writerow(["PCA+KMeans", sil_score, ch_score])

print("Metrics saved to:", csv_path)

# VAE LATENT FEATURES + K-MEANS

print("\nLoading VAE latent vectors")
Z_vae = np.load(os.path.join(DATA_DIR, "Z_vae.npy"))

# TEST Latent sanity
assert Z_vae.shape[0] == X.shape[0]
assert Z_vae.shape[1] == 32
print("VAE latent shape =", Z_vae.shape)

# K-MEANS ON VAE LATENTS
print("Running K-Means on VAE latents")
kmeans_vae = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
labels_vae = kmeans_vae.fit_predict(Z_vae)

unique_vae = np.unique(labels_vae)
print("Clusters found (VAE):", len(unique_vae))

# CLUSTERING METRICS
if len(unique_vae) < 2:
    sil_vae = -1.0
    ch_vae = -1.0
    print("WARNING: VAE clustering collapsed to 1 cluster")
else:
    sil_vae = silhouette_score(Z_vae, labels_vae)
    ch_vae = calinski_harabasz_score(Z_vae, labels_vae)

print(f"VAE + KMeans Silhouette Score: {sil_vae:.4f}")
print(f"VAE + KMeans Calinski-Harabasz Index: {ch_vae:.2f}")

# SAVE METRICS
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["VAE+KMeans", sil_vae, ch_vae])

print("VAE metrics saved to:", csv_path)

# t-SNE VISUALIZATION

print("\nRunning t-SNE visualizations")

TSNE_DIM = 2
tsne = TSNE(
    n_components=TSNE_DIM,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,   
    random_state=42
)

# t-SNE on PCA features
print("Running t-SNE on PCA features")
Z_pca_2d = tsne.fit_transform(Z_pca)

assert Z_pca_2d.shape == (Z_pca.shape[0], 2)
print("PCA t-SNE shape =", Z_pca_2d.shape)

plt.figure(figsize=(6, 5))
plt.scatter(Z_pca_2d[:, 0], Z_pca_2d[:, 1], c=labels_pca, s=5, cmap="tab10")
plt.title("t-SNE of PCA + KMeans")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "tsne_pca_audio.png"), dpi=300)
plt.close()

# t-SNE on VAE latents
print("Running t-SNE on VAE latents")
Z_vae_2d = tsne.fit_transform(Z_vae)

assert Z_vae_2d.shape == (Z_vae.shape[0], 2)
print("VAE t-SNE shape =", Z_vae_2d.shape)

plt.figure(figsize=(6, 5))
plt.scatter(Z_vae_2d[:, 0], Z_vae_2d[:, 1], c=labels_vae, s=5, cmap="tab10")
plt.title("t-SNE of VAE Latent + KMeans")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "tsne_vae_audio.png"), dpi=300)
plt.close()

print("t-SNE plots saved in:", VIS_DIR)
