import numpy as np
from sklearn.preprocessing import OneHotEncoder

LABELS_PATH = "data/y_labels_ready.npy"
OUTPUT_PATH = "data/genre_onehot.npy"

print("Loading genre labels")
y = np.load(LABELS_PATH)

assert y.ndim == 1, "Genre labels must be 1D"
N = len(y)

print("Number of samples:", N)
print("Unique genres:", np.unique(y))

# One-hot encode
encoder = OneHotEncoder(sparse_output=False)
y_oh = encoder.fit_transform(y.reshape(-1, 1))

# Test shapes
assert y_oh.shape[0] == N
assert y_oh.shape[1] == len(np.unique(y))

print("One-hot genre shape =", y_oh.shape)

np.save(OUTPUT_PATH, y_oh)
print("Saved genre condition vectors to:", OUTPUT_PATH)
