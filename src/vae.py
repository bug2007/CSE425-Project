import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_dim=23440, latent_dim=32):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ENCODER
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # DECODER
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.fc3(z))
        h = self.relu(self.fc4(h))
        return self.fc5(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def test_vae():
    batch_size = 4
    input_dim = 23440
    latent_dim = 32
    x = torch.randn(batch_size, input_dim)
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape, f"Reconstruction shape mismatch: {recon.shape} vs {x.shape}"
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    print("VAE test passed")

if __name__ == "__main__":
    test_vae()
