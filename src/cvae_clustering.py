import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score
)


Z_PATH = "data/Z_cvae.npy"
GENRE_PATH = "data/genre_onehot.npy"
RESULTS_DIR = "results/latent_visualisation"
CSV_PATH = os.path.join(RESULTS_DIR, "cvae_clustering_metrics.csv")

N_CLUSTERS = 10
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# METRIC: CLUSTER PURITY
def cluster_purity(y_true, y_pred):
    contingency = pd.crosstab(y_pred, y_true)
    return np.sum(np.max(contingency.values, axis=1)) / np.sum(contingency.values)

# LOAD DATA
print("Loading CVAE latent vectors")
Z = np.load(Z_PATH)

print("Loading genre labels")
genre_onehot = np.load(GENRE_PATH)
y_true = np.argmax(genre_onehot, axis=1)


assert Z.ndim == 2, "Z_cvae must be 2D"
assert Z.shape[0] == genre_onehot.shape[0], "Mismatch in number of samples"
assert genre_onehot.shape[1] == N_CLUSTERS, "Expected 10 genre classes"

print(f"Z shape = {Z.shape}")
print(f"Genre labels shape = {genre_onehot.shape}")

# KMEANS CLUSTERING (CVAE)
print("Running KMeans on CVAE latent space")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
y_pred = kmeans.fit_predict(Z)

# METRICS
sil = silhouette_score(Z, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
ari = adjusted_rand_score(y_true, y_pred)
purity = cluster_purity(y_true, y_pred)

print("CVAE + KMeans metrics:")
print(f"Silhouette: {sil:.4f}")
print(f"NMI: {nmi:.4f}")
print(f"ARI: {ari:.4f}")
print(f"Purity: {purity:.4f}")

# SAVE METRICS 
df = pd.DataFrame([{
    "Model": "CVAE",
    "Clustering": "KMeans",
    "Silhouette": sil,
    "NMI": nmi,
    "ARI": ari,
    "Purity": purity
}])

df.to_csv(CSV_PATH, index=False)
print(f"Saved metrics to: {CSV_PATH}")

# t-SNE VISUALIZATION
print("Running t-SNE")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    random_state=RANDOM_STATE,
    init="pca"
)

Z_tsne = tsne.fit_transform(Z)

# t-SNE by TRUE GENRE 
plt.figure(figsize=(7, 6))
plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=y_true, cmap="tab10", s=8)
plt.title("t-SNE (CVAE Latent) — Colored by True Genre")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "tsne_cvae_true_genre.png"), dpi=300)
plt.close()

#  t-SNE by CLUSTER 
plt.figure(figsize=(7, 6))
plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=y_pred, cmap="tab10", s=8)
plt.title("t-SNE (CVAE Latent) — Colored by KMeans Cluster")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "tsne_cvae_clusters.png"), dpi=300)
plt.close()

print("Saved t-SNE visualizations")




# LOAD ORIGINAL FEATURES (for PCA/AE/raw clustering)
print("Loading hybrid audio + lyrics features")
X_audio_lyric = np.load("data/X_audio_lyric.npy")
print("Loading genre condition vectors...")
genre_onehot = np.load("data/genre_onehot.npy")

# get final feature matrix
X_full = np.hstack([X_audio_lyric, genre_onehot])  # shape: (N, F_audio+lyrics + 10 genres)
print(f"Combined feature shape: {X_full.shape}")

# Labels for evaluation
y_true = np.argmax(genre_onehot, axis=1)

assert X_full.shape[0] == y_true.shape[0], "Mismatch in number of samples"
assert genre_onehot.shape[1] == 10, "Expected 10 genre classes"


from sklearn.decomposition import PCA

# # PCA + KMeans Clustering
print("Running PCA on full features")
pca = PCA(n_components=32, random_state=42)  # reduce to same dim as CVAE latent
Z_pca = pca.fit_transform(X_full)
print(f"PCA features shape = {Z_pca.shape}")
assert Z_pca.shape[0] == X_full.shape[0], "PCA: Number of samples mismatch"
assert Z_pca.shape[1] == 32, "PCA: Latent dim mismatch"

print("Running KMeans on PCA features")
kmeans_pca = KMeans(n_clusters=10, random_state=42, n_init=20)
y_pred_pca = kmeans_pca.fit_predict(Z_pca)

#METRICS
sil_pca = silhouette_score(Z_pca, y_pred_pca)
nmi_pca = normalized_mutual_info_score(y_true, y_pred_pca)
ari_pca = adjusted_rand_score(y_true, y_pred_pca)
purity_pca = cluster_purity(y_true, y_pred_pca)

print("PCA + KMeans metrics:")
print(f"Silhouette: {sil_pca:.4f}")
print(f"NMI: {nmi_pca:.4f}")
print(f"ARI: {ari_pca:.4f}")
print(f"Purity: {purity_pca:.4f}")

# SAVE 
df_pca = pd.DataFrame([{
    "Model": "PCA",
    "Clustering": "KMeans",
    "Silhouette": sil_pca,
    "NMI": nmi_pca,
    "ARI": ari_pca,
    "Purity": purity_pca
}])

if os.path.exists(CSV_PATH):
    df_prev = pd.read_csv(CSV_PATH)
    df_combined = pd.concat([df_prev, df_pca], ignore_index=True)
else:
    df_combined = df_pca

df_combined.to_csv(CSV_PATH, index=False)
print(f"Saved PCA metrics to: {CSV_PATH}")


import torch
import torch.nn as nn
import torch.optim as optim

# AUTOENCODER DEFINITION
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# PREPARE DATA
print("Preparing data for Autoencoder")
X_tensor = torch.from_numpy(X_full).float()
dataset = torch.utils.data.TensorDataset(X_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# TRAIN AUTOENCODER
input_dim = X_full.shape[1]
latent_dim = 32
ae = Autoencoder(input_dim, latent_dim)
optimizer = optim.Adam(ae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("Training Autoencoder for 30 epochs")
ae.train()
for epoch in range(30):
    total_loss = 0
    for batch in loader:
        x_batch = batch[0]
        optimizer.zero_grad()
        recon, _ = ae(x_batch)
        loss = criterion(recon, x_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    avg_loss = total_loss / X_full.shape[0]
    if (epoch+1) % 5 == 0 or epoch==0:
        print(f"Epoch [{epoch+1}/30], Loss: {avg_loss:.6f}")

# EXTRACT LATENT FEATURES
ae.eval()
with torch.no_grad():
    _, Z_ae = ae(X_tensor)

print(f"Autoencoder latent shape = {Z_ae.shape}")
assert Z_ae.shape == (X_full.shape[0], latent_dim), "Latent dim mismatch"

# KMEANS ON AE LATENTS
print("Running KMeans on Autoencoder latent space")
y_pred_ae = KMeans(n_clusters=10, random_state=42, n_init=20).fit_predict(Z_ae.numpy())

#  METRICS 
sil_ae = silhouette_score(Z_ae.numpy(), y_pred_ae)
nmi_ae = normalized_mutual_info_score(y_true, y_pred_ae)
ari_ae = adjusted_rand_score(y_true, y_pred_ae)
purity_ae = cluster_purity(y_true, y_pred_ae)

print("Autoencoder + KMeans metrics:")
print(f"Silhouette: {sil_ae:.4f}")
print(f"NMI: {nmi_ae:.4f}")
print(f"ARI: {ari_ae:.4f}")
print(f"Purity: {purity_ae:.4f}")

# SAVE 
df_ae = pd.DataFrame([{
    "Model": "Autoencoder",
    "Clustering": "KMeans",
    "Silhouette": sil_ae,
    "NMI": nmi_ae,
    "ARI": ari_ae,
    "Purity": purity_ae
}])

df_combined = pd.concat([df_combined, df_ae], ignore_index=True)
df_combined.to_csv(CSV_PATH, index=False)
print(f"Saved Autoencoder metrics to: {CSV_PATH}")





# DIRECT AUDIO+LYRICS+GENRE CLUSTERING
print("Direct KMeans on audio + lyrics + genre features")


genre = np.load("data/genre_onehot.npy")         # shape (3996, genre_dim)

# Concatenate features
print(f"Combined audio+lyrics+genre shape = {X_full.shape}")
assert X_full.shape[0] == genre.shape[0], "Number of samples mismatch"

#  KMeans 
direct_kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
y_direct = direct_kmeans.fit_predict(X_full)

#  Metrics 
sil = silhouette_score(X_full, y_direct)
nmi = normalized_mutual_info_score(y_true, y_direct)
ari = adjusted_rand_score(y_true, y_direct)
purity = cluster_purity(y_true, y_direct)

print("Direct audio+lyrics+genre KMeans metrics:")
print(f"Silhouette: {sil:.4f}")
print(f"NMI: {nmi:.4f}")
print(f"ARI: {ari:.4f}")
print(f"Purity: {purity:.4f}")

#save to csv
df_direct = pd.DataFrame([{
    "Model": "Direct Features",
    "Clustering": "KMeans",
    "Silhouette": sil,
    "NMI": nmi,
    "ARI": ari,
    "Purity": purity
}])

if os.path.exists(CSV_PATH):
    df_existing = pd.read_csv(CSV_PATH)
    df_final = pd.concat([df_existing, df_direct], ignore_index=True)
else:
    df_final = df_direct

df_final.to_csv(CSV_PATH, index=False)
print(f"Saved metrics saved to: {CSV_PATH}")




