import torch
import torch.nn as nn
import torch.nn.functional as F

# CVAE for audio + lyrics + genre
class CVAEHard(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        # Encoder: input + condition
        self.fc1 = nn.Linear(input_dim + condition_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder: latent + condition
        self.fc_dec1 = nn.Linear(latent_dim + condition_dim, 256)
        self.fc_dec2 = nn.Linear(256, 512)
        self.fc_out = nn.Linear(512, input_dim)

    def encode(self, x, c):
        x_cond = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1(x_cond))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z_cond = torch.cat([z, c], dim=1)
        h = F.relu(self.fc_dec1(z_cond))
        h = F.relu(self.fc_dec2(h))
        x_recon = self.fc_out(h)
        return x_recon

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar


def test_cvae_hard():
    print("Running CVAEHard tests...")

    batch_size = 4
    input_dim = 23824    # audio + lyrics flattened
    condition_dim = 10   # number of genres
    latent_dim = 32

    x = torch.randn(batch_size, input_dim)
    c = torch.randn(batch_size, condition_dim)

    model = CVAEHard(input_dim=input_dim, condition_dim=condition_dim, latent_dim=latent_dim)
    recon, mu, logvar = model(x, c)

    assert recon.shape == x.shape, f"Recon shape mismatch: {recon.shape} vs {x.shape}"
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)

    print("CVAE forward pass OK")

if __name__ == "__main__":
    test_cvae_hard()


