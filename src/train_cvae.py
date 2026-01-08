import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from cvae import CVAEHard

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 30
LATENT_DIM = 32
LEARNING_RATE = 1e-3

HYBRID_PATH = "data/X_audio_lyric.npy"
GENRE_PATH = "data/genre_onehot.npy"
OUTPUT_LATENT_PATH = "data/Z_cvae.npy"

# LOAD DATA
print("Loading audio lyric features")
X_audio_lyric = np.load(HYBRID_PATH)  # shape (N, audio+lyrics)
X_audio_lyric = X_audio_lyric.astype(np.float32)

print("Loading genre condition")
genre_onehot = np.load(GENRE_PATH)  # shape (N, num_genres)
genre_onehot = genre_onehot.astype(np.float32)

N = X_audio_lyric.shape[0]
assert genre_onehot.shape[0] == N, "Number of samples mismatch between hybrid features and genre labels"
print(f"Number of samples = {N}")

# DATA LOADER
dataset = TensorDataset(torch.from_numpy(X_audio_lyric), torch.from_numpy(genre_onehot))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODEL
input_dim = X_audio_lyric.shape[1]
condition_dim = genre_onehot.shape[1]

model = CVAEHard(input_dim=input_dim, condition_dim=condition_dim, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# LOSS FUNCTION
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss()(recon_x, x)
    # KL divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# TRAIN 
print("Training")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, batch_c in dataloader:
        batch_x = batch_x.to(DEVICE)
        batch_c = batch_c.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar = model(batch_x, batch_c)
        loss = loss_function(recon, batch_x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    avg_loss = total_loss / N
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

# Extract latents and save
print("Extracting latent vectors")
model.eval()
with torch.no_grad():
    X_tensor = torch.from_numpy(X_audio_lyric).to(DEVICE)
    C_tensor = torch.from_numpy(genre_onehot).to(DEVICE)
    mu, logvar = model.encode(X_tensor, C_tensor)
    Z_cvae = model.reparameterize(mu, logvar).cpu().numpy()

print(f"Z_cvae shape = {Z_cvae.shape}")
np.save(OUTPUT_LATENT_PATH, Z_cvae)
print(f"Saved latent vectors to: {OUTPUT_LATENT_PATH}")
