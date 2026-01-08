import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAEHybrid(nn.Module):
    def __init__(self, latent_dim=32, lyrics_dim=384):
        super().__init__()

        self.lyrics_dim = lyrics_dim

        # ENCODER (audio conv)
        # Input: (B, 1, 20, W) 
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # (B,16,H/2,W/2)
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B,32,H/4,W/4)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B,64,H/8,W/8)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # compute flatten_dim dynamically in forward
        self.fc_mu = nn.Linear(1, latent_dim)  # dummy init
        self.fc_logvar = nn.Linear(1, latent_dim)

        # DECODER
        self.fc_dec = nn.Linear(latent_dim, 1)  # dummy init
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def encode(self, x, X_lyrics=None):
        B, C, H, W = x.shape
        out = self.enc(x)
        self.last_conv_shape = out.shape[1:]  # save for decoder

        # flatten conv output
        out_flat = out.view(B, -1)

        # combine lyrics 
        if X_lyrics is not None:
            z_input = torch.cat([out_flat, X_lyrics], dim=1)
        else:
            z_input = out_flat

        # initialize fc layers on first pass
        if not hasattr(self, "fc_mu_initialized"):
            self.fc_mu = nn.Linear(z_input.shape[1], 32)
            self.fc_logvar = nn.Linear(z_input.shape[1], 32)
            self.fc_dec = nn.Linear(32, z_input.shape[1])
            self.fc_mu_initialized = True

        mu = self.fc_mu(z_input)
        logvar = self.fc_logvar(z_input)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_shape, X_lyrics=None):
        B = z.size(0)
        x = self.fc_dec(z)

        # only keep conv features lyrics included, 
        conv_flat_dim = self.last_conv_shape[0]*self.last_conv_shape[1]*self.last_conv_shape[2]
        if X_lyrics is not None:
            x = x[:, :conv_flat_dim]

        x = x.view(B, *self.last_conv_shape)
        x = self.dec(x)

        # Crop to match input exactly
        _, _, H, W = target_shape
        x = x[:, :, :H, :W]
        return x

    def forward(self, x, X_lyrics=None):
        mu, logvar = self.encode(x, X_lyrics)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, target_shape=x.shape, X_lyrics=X_lyrics)
        return recon, mu, logvar


def test_conv_vae_hybrid():
    print("Running ConvVAEtests")

    batch_size = 4
    # variable width for hybrid
    x = torch.randn(batch_size, 1, 20, 1556)
    lyrics = torch.randn(batch_size, 384)

    model = ConvVAEHybrid(latent_dim=32, lyrics_dim=lyrics.shape[1])
    recon, mu, logvar = model(x, lyrics)

    assert recon.shape == x.shape, f"Recon shape mismatch: {recon.shape} vs {x.shape}"
    assert mu.shape == (batch_size, 32)
    assert logvar.shape == (batch_size, 32)

    print("ConvVAEforward pass OK")

if __name__ == "__main__":
    test_conv_vae_hybrid()
