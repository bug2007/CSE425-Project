import numpy as np
import os
from sentence_transformers import SentenceTransformer


PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
LYRICS_PATH = os.path.join(DATA_DIR, "X_lyrics.npy")
OUTPUT_PATH = os.path.join(DATA_DIR, "X_lyrics_emb.npy")

# LOAD LYRICS
X_lyrics = np.load(LYRICS_PATH, allow_pickle=True)
print("Loaded lyrics, first 3:", X_lyrics[:3])

# INITIALIZE EMBEDDING MODEL
print("Loading sentence embedding model")
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings

# EMBEDDING
print("Creating embeddings for lyrics")
X_emb = model.encode(X_lyrics, batch_size=32, show_progress_bar=True)

# TEST shapes
assert X_emb.shape[0] == X_lyrics.shape[0], "Number of embeddings must match number of tracks"
assert X_emb.ndim == 2, "Embeddings must be 2D (N, embedding_dim)"
print("Embeddings shape =", X_emb.shape)

# SAVE
np.save(OUTPUT_PATH, X_emb)
print("Saved lyrics embeddings to:", OUTPUT_PATH)
