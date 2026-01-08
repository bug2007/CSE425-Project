import numpy as np
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "X_audio_lyric.npy")

LYRICS_PATH = os.path.join(DATA_DIR, "X_lyrics_emb.npy")
AUDIO_PATH = os.path.join(DATA_DIR, "X_mfcc_conv.npy")

# LOAD AUDIO MFCC (conv)
print("Loading audio MFCC (conv)")
X_audio = np.load(AUDIO_PATH)  # shape: (N, 1, H, W)

# test Audio shape
assert X_audio.ndim == 4, "Audio MFCC must be 4D (N, C, H, W)"
print("Audio MFCC shape =", X_audio.shape)

# Flatten audio for hybrid concatenation
N, C, H, W = X_audio.shape
X_audio_flat = X_audio.reshape(N, C * H * W)
print("Audio MFCC flattened shape =", X_audio_flat.shape)

# LOAD LYRICS EMBEDDINGS
print("Loading lyrics embeddings...")
X_lyrics = np.load(LYRICS_PATH)

# test lyrics embeddings
print("Lyrics embeddings shape before alignment:", X_lyrics.shape)
if X_lyrics.ndim == 1:
    X_lyrics = X_lyrics.reshape(-1, 1)
print("Lyrics embeddings reshaped to 2D:", X_lyrics.shape)
assert X_lyrics.dtype in [np.float32, np.float64], "Lyrics embeddings must be numeric floats"
assert X_lyrics.ndim == 2, "Lyrics embeddings must be 2D (N, embedding_dim)"
assert X_lyrics.shape[0] == X_audio_flat.shape[0], "Number of samples mismatch between audio and lyrics"
print("Shapes already aligned. N =", N)

# Concatenate audio and lyrics
print("Creating hybrid features (audio + lyrics)...")
X_audio_lyric = np.concatenate([X_audio_flat, X_lyrics], axis=1)

# Test shape
expected_dim = X_audio_flat.shape[1] + X_lyrics.shape[1]
assert X_audio_lyric.shape == (N, expected_dim), f"Hybrid shape mismatch: {X_audio_lyric.shape} vs ({N}, {expected_dim})"
print("Hybrid features shape =", X_audio_lyric.shape)

# SAVE
np.save(OUTPUT_PATH, X_audio_lyric)
print("Hybrid features saved to:", OUTPUT_PATH)
