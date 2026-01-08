import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


Z_PATH = "data/Z_audio_lyric.npy"
LABELS_PATH = "data/y_labels_ready.npy"  
OUT_DIR = Path("results/latent_visualisation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "conv_vae_clustering_metrics.csv"

MODEL_NAME = "Conv-VAE"

# LOAD DATA
Z = np.load(Z_PATH)
Z = StandardScaler().fit_transform(Z)  

print("Loaded Z_audio_lyric:", Z.shape)

y_true = None
if Path(LABELS_PATH).exists():
    y_true = np.load(LABELS_PATH)
    print("Loaded genre labels:", y_true.shape)
else:
    print("No labels found, ARI will be NaN")

results = []


def compute_metrics(name, labels):
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})

    sil = np.nan
    db = np.nan
    ari = np.nan

    valid_mask = labels != -1

    if n_clusters >= 2:
        try:
            sil = silhouette_score(Z[valid_mask], labels[valid_mask])
            db = davies_bouldin_score(Z[valid_mask], labels[valid_mask])
        except Exception:
            pass

    if y_true is not None and np.any(valid_mask):
        try:
            ari = adjusted_rand_score(y_true[valid_mask], labels[valid_mask])
        except Exception:
            pass

    results.append({
        "model": MODEL_NAME,
        "clustering": name,
        "silhouette": sil,
        "davies_bouldin": db,
        "ari": ari
    })

    print(f"{name}: clusters={n_clusters}, silhouette={sil}, DB={db}, ARI={ari}")

# K-MEANS
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(Z)
compute_metrics("KMeans", labels_km)

# AGGLOMERATIVE
agg = AgglomerativeClustering(n_clusters=10)
labels_agg = agg.fit_predict(Z)
compute_metrics("Agglomerative", labels_agg)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=10)
labels_db = dbscan.fit_predict(Z)
compute_metrics("DBSCAN", labels_db)

# SAVE RESULTS
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)

print("\nSaved metrics to:", OUT_CSV)
print(df)
