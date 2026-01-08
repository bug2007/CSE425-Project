
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from conv_vae import ConvVAEHybrid

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
LATENT_DIM = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_PATH = "data/X_mfcc_conv.npy"
LYRICS_PATH = "data/X_lyrics_emb.npy"
OUTPUT_LATENT_PATH = "data/Z_audio_lyric.npy"

# LOAD DATA
print("Loading audio MFCC (conv)")
X_audio = np.load(AUDIO_PATH)    # (N, 1, 20, 1172)

print("Loading lyrics embedding")
X_lyrics = np.load(LYRICS_PATH)  # (N, 384)


assert X_audio.ndim == 4, "Audio must be (N,1,20,T)"
assert X_lyrics.ndim == 2, "Lyrics must be (N,embedding_dim)"
assert X_audio.shape[0] == X_lyrics.shape[0], "Audio/Lyrics count mismatch"

N = X_audio.shape[0]
lyrics_dim = X_lyrics.shape[1]

print("Audio shape =", X_audio.shape)
print("Lyrics shape =", X_lyrics.shape)


X_audio = torch.tensor(X_audio, dtype=torch.float32)
X_lyrics = torch.tensor(X_lyrics, dtype=torch.float32)

dataset = TensorDataset(X_audio, X_lyrics)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODEL
model = ConvVAEHybrid(
    latent_dim=LATENT_DIM,
    lyrics_dim=lyrics_dim
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# TRAIN
print("\nTraining Hybrid Conv-VAE")

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0

    for audio, lyrics in loader:
        audio = audio.to(DEVICE)
        lyrics = lyrics.to(DEVICE)

        recon, mu, logvar = model(audio, lyrics)

        # Reconstruction loss 
        recon_loss = F.mse_loss(recon, audio, reduction="mean")

        # KL divergence
        kl = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        loss = recon_loss + kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f}")

# EXTRACT LATENTS
print("\nExtracting latent vectors")

model.eval()
Z = []

with torch.no_grad():
    for audio, lyrics in loader:
        audio = audio.to(DEVICE)
        lyrics = lyrics.to(DEVICE)

        mu, _ = model.encode(audio, lyrics)
        Z.append(mu.cpu().numpy())

Z = np.concatenate(Z, axis=0)


assert Z.shape == (N, LATENT_DIM), "Latent shape incorrect"
print("Z_audio_lyric shape =", Z.shape)

# SAVE
np.save(OUTPUT_LATENT_PATH, Z)
print("Saved hybrid latents to:", OUTPUT_LATENT_PATH)
