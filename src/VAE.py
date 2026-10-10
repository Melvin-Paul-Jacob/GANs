import os
import cv2
import torch
import torchvision
import numpy as np

from tqdm import tqdm
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


# ============================================================
# Configuration
# ============================================================

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

num_epochs = 100
batch_size = 64
learning_rate = 0.0002

image_size = 28
image_channels = 1
latent_dim = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./models", exist_ok=True)


# ============================================================
# Dataset
# ============================================================

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------
        self.encoder = nn.Sequential(
            # 1 × 28 × 28
            nn.Conv2d(image_channels, 128, kernel_size=4, stride=2, padding=1),
            # 128 × 14 × 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # 256 × 7 × 7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))
        # 256 × 7 × 7 = 12544
        self.flatten_dim = 256 * 7 * 7
        # Latent mean
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)

        # Latent log variance
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            # 256 × 7 × 7
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # 128 × 14 × 14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, image_channels, kernel_size=4, stride=2, padding=1),
            # 1 × 28 × 28
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Standard deviation
        std = torch.exp(0.5 * logvar)
        # Random noise
        eps = torch.randn_like(std)
        # z = μ + σϵ
        z = mu + eps * std
        return z

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 7, 7)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


# ============================================================
# Loss
# ============================================================

def vae_loss(reconstruction, real_images, mu, logvar):
    # --------------------------------------------------------
    # Reconstruction loss
    # --------------------------------------------------------
    reconstruction_loss = F.mse_loss(reconstruction, real_images, reduction="sum")
    # --------------------------------------------------------
    # KL divergence
    # --------------------------------------------------------
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # --------------------------------------------------------
    # Total VAE loss
    # --------------------------------------------------------
    total_loss = (reconstruction_loss + kl_loss)
    return (total_loss, reconstruction_loss, kl_loss)


# ============================================================
# Model
# ============================================================
vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# ============================================================
# Fixed latent vectors for visualization
# ============================================================

fixed_noise = torch.randn(64, latent_dim, device=device)
frames = []

# ============================================================
# Training
# ============================================================

for epoch in range(num_epochs):

    progress_bar = tqdm(data_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=True)
    for real_images, _ in progress_bar:
        real_images = real_images.to(device)
        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------
        reconstruction, mu, logvar = vae(real_images)
        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------
        loss, reconstruction_loss, kl_loss = vae_loss(reconstruction, real_images, mu, logvar)
        # Normalize by batch size
        loss = loss / real_images.size(0)
        # ----------------------------------------------------
        # Backprop
        # ----------------------------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ----------------------------------------------------
        # tqdm
        # ----------------------------------------------------
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.2f}",
            "Recon": f"{reconstruction_loss.item() / real_images.size(0):.2f}",
            "KL": f"{kl_loss.item() / real_images.size(0):.2f}"
        })

    # ========================================================
    # Generate samples from latent space
    # ========================================================

    vae.eval()
    with torch.no_grad():
        fake_samples = vae.decode(fixed_noise).cpu()
        fake_grid = torchvision.utils.make_grid(fake_samples, padding=2, normalize=True)
        # CHW → HWC
        image = fake_grid.permute(1, 2, 0).numpy()
        # MNIST is grayscale
        image = (image[:, :, 0] * 255).astype(np.uint8)
        # Store frame for GIF
        frames.append(Image.fromarray(image))
        cv2.imshow("Generated Images", image)
        cv2.waitKey(20)
    vae.train()

# ============================================================
# Save GIF
# ============================================================

frames[0].save("./models/vae_training.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)

# ============================================================
# Save model
# ============================================================
torch.save(vae.state_dict(), "./models/vae.pth")

print("VAE training complete.")
print("Model saved to ./models/vae.pth")
print("GIF saved to ./models/vae_training.gif")