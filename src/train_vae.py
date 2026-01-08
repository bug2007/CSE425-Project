import os
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_ROOT)
from src.vae import VAE  

BATCH_SIZE = 64
LATENT_DIM = 32
LEARNING_RATE = 1e-3
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KL_WARMUP_EPOCHS = 10


# LOAD DATA
DATA_DIR = os.path.join(PROJ_ROOT, "data")
print("Loading data from:", DATA_DIR)

X = np.load(os.path.join(DATA_DIR, "X_mfcc_truncated.npy"))  # (3996, 1172, 20)
y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))          # (3996,)

# Check data shapes
print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

# Flatten MFCCs for fully connected VAE
num_samples = X.shape[0]
X_flat = X.reshape(num_samples, -1).astype(np.float32)

print("Flattened X shape:", X_flat.shape)
print("Mean/std check - mean:", X_flat.mean(), "std:", X_flat.std())
print("First 5 labels:", y[:5])

# create pytorch dataset and dataloader
X_tensor = torch.from_numpy(X_flat)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Test Sample batch
for batch in dataloader:
    print("TEST Batch shape:", batch[0].shape)  
    break

# Initialize VAE
input_dim = X_flat.shape[1]
vae = VAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
mse_loss = nn.MSELoss()

print("VAE initialized.Input dim:", input_dim, "Latent dim:", LATENT_DIM)

# TRAIN
vae.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0
    for batch in dataloader:
        batch_X = batch[0].to(DEVICE)

        optimizer.zero_grad()
        recon, mu, logvar = vae(batch_X)

        # VAE loss = reconstruction + KL divergence
        recon_loss = mse_loss(recon, batch_X)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        beta = min(1.0, epoch / KL_WARMUP_EPOCHS)

        loss = recon_loss + beta * kl_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss /= len(dataset)
    print(f"Epoch {epoch}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# SAVE LATENT REPRESENTATIONS
vae.eval()
with torch.no_grad():
    X_tensor = X_tensor.to(DEVICE)
    _, mu, _ = vae(X_tensor)
    Z = mu.cpu().numpy()  # latent vectors

OUTPUT_LATENT_PATH = os.path.join(PROJ_ROOT, "data")
np.save(os.path.join(OUTPUT_LATENT_PATH, "Z_vae.npy"), Z)
print("Latent vectors saved to:", os.path.join(OUTPUT_LATENT_PATH, "Z_vae.npy"))
print("Z shape:", Z.shape)

# Check latent vectors
assert Z.shape == (num_samples, LATENT_DIM), f"Latent shape mismatch: {Z.shape} vs {(num_samples, LATENT_DIM)}"
print("Latent vectors shape is correct")