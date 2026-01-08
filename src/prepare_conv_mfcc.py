import numpy as np
import os

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT, "data")

MFCC_PATH = os.path.join(DATA_DIR, "X_mfcc_truncated.npy")
OUTPUT_PATH = os.path.join(DATA_DIR, "X_mfcc_conv.npy")

# LOAD MFCC
print("Loading MFCC features")
mfcc = np.load(MFCC_PATH)   # (N, time, mfcc)

# TEST shape
assert mfcc.ndim == 3, "MFCC must be 3D (N, time, mfcc)"
print("Original MFCC shape =", mfcc.shape)

# TRANSPOSE FOR CONV
# (N, time, mfcc) -> (N, 1, mfcc, time)
mfcc_conv = mfcc.transpose(0, 2, 1)
mfcc_conv = np.expand_dims(mfcc_conv, axis=1)

# TEST Conv shape
N, C, H, W = mfcc_conv.shape
assert C == 1, "Channel dimension must be 1"
assert H == 20, "MFCC coefficient count must be 20"
print("Conv MFCC shape =", mfcc_conv.shape)

# SAVE
np.save(OUTPUT_PATH, mfcc_conv)
print("Conv MFCC saved to:", OUTPUT_PATH)
