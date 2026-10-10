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

num_epochs = 100
batch_size = 64
learning_rate = 0.0001

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(image_channels, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 7, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

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

criterion = nn.BCELoss()

# ============================================================
# Model
# ============================================================
discriminator = Discriminator()
discriminator = discriminator.to(device)

vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

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
        batch_size = real_images.size(0)
        label_real = torch.ones(batch_size, 1, device=device)
        label_fake = torch.zeros(batch_size, 1, device=device)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------
        # Train discriminator with real images
        optimizer_d.zero_grad()
        output_real = discriminator(real_images).view(-1, 1)
        loss_real = criterion(output_real, label_real)
        
        reconstruction, mu, logvar = vae(real_images)

        # Train discriminator with fake images
        output_fake = discriminator(reconstruction.detach()).view(-1, 1)
        loss_fake = criterion(output_fake, label_fake)
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()
        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------
        loss, reconstruction_loss, kl_loss = vae_loss(reconstruction, real_images, mu, logvar)
        # Normalize by batch size
        loss = loss / batch_size
        loss_d = loss_real.item() + loss_fake.item()
        # ----------------------------------------------------
        # Backprop
        # ----------------------------------------------------
        # ----------------------------------------------------
        # 3. Train VAE
        # ----------------------------------------------------
        # Freeze discriminator weights, but allow gradients
        # to flow through it to the reconstructed images.
        for p in discriminator.parameters():
            p.requires_grad_(False)
        optimizer.zero_grad()
        output = discriminator(reconstruction).view(-1, 1)
        c_loss = criterion(output, label_real)
        loss += c_loss
        loss.backward()
        optimizer.step()

        # Re-enable discriminator training.
        for p in discriminator.parameters():
            p.requires_grad_(True)
        # ----------------------------------------------------
        # tqdm
        # ----------------------------------------------------
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.2f}",
            "Recon": f"{reconstruction_loss.item() / real_images.size(0):.2f}",
            "KL": f"{kl_loss.item() / real_images.size(0):.2f}",
            "D_loss": f"{loss_d:.4f}",
            "C_loss": f"{c_loss.item():.4f}",
            "D_real": f"{output_real.mean().item():.4f}",
            "D_fake": f"{output_fake.mean().item():.4f}"
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

frames[0].save("./models/gan-vae_training2.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)

# ============================================================
# Save model
# ============================================================
torch.save(vae.state_dict(), "./models/gan-vae.pth")

print("GAN-VAE training complete.")
print("Model saved to ./models/gan-vae.pth")
print("GIF saved to ./models/gan-vae_training.gif")