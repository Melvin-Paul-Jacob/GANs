import os
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# models
from SRGAN import SRGAN


torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

implemented_gans = {"SRGAN":SRGAN}
gan_type = "SRGAN"

if gan_type not in implemented_gans:
    raise ValueError(f"Wrong gan_type:'{gan_type}' provided. Use one of {list(implemented_gans.keys())}")

print(f"Using {gan_type}")
# ============================================================
# Hyperparameters
# ============================================================
num_epochs = 100
batch_size = 64
learning_rate = 0.0002
image_size = 28
image_channels = 1
latent_dim = 100
# Number of feature maps (used in dcgan)
generator_features = 64
discriminator_features = 64

# ============================================================
# Dataset and DataLoader
# ============================================================
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)
num_classes = len(mnist_dataset.classes)

# ============================================================
# Weight Initialization
# ============================================================
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_( module.bias.data, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# Create Models
# ============================================================
gan = implemented_gans[gan_type](latent_dim=latent_dim, 
                                 image_channels=image_channels,
                                 generator_features=generator_features, 
                                 discriminator_features=discriminator_features,
                                 num_classes=num_classes,
                                 image_size=image_size)
generator = gan.generator.to(device)
discriminator = gan.discriminator.to(device)
print(generator)
print(discriminator)

# Initialize weights
initialize_weights(generator)
initialize_weights(discriminator)

# ============================================================
# Loss Function
# ============================================================
criterion = nn.BCELoss()

# ============================================================
# Optimizers
# ============================================================
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# ============================================================
# Fixed Test
# ============================================================
samples_per_class = 10
total = num_classes * samples_per_class
fixed_noise = torch.randn(100, latent_dim, 1, 1, device=device)
# Generate each digit repeatedly (used in cgan)
fixed_labels = torch.tensor(
    [i for i in range(num_classes) for _ in range(samples_per_class)],
    dtype=torch.long, device=device)

# ============================================================
# Create Output Directory
# ============================================================
os.makedirs("./models", exist_ok=True)
os.makedirs("./results", exist_ok=True)
frames = []

# ============================================================
# Training
# ============================================================

for epoch in range(num_epochs):
    progress_bar = tqdm(data_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=True)
    for i, (real_images, labels) in enumerate(progress_bar):
        # Move real images to device
        real_images = real_images.to(device)
        labels = labels.to(device)
        current_batch_size = real_images.size(0)

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------
        label_real = torch.ones(current_batch_size, 1, device=device)
        label_fake = torch.zeros(current_batch_size, 1, device=device)
        fake_labels = torch.randint(0, num_classes, (current_batch_size,), device=device)

        # ====================================================
        # Train Discriminator
        # ====================================================
        optimizer_d.zero_grad()

        # ----------------------------------------------------
        # Real Images
        # ----------------------------------------------------
        output_real = discriminator(x=real_images, labels=labels).view(-1, 1)
        loss_real = criterion(output_real, label_real)
        loss_real.backward()

        # ----------------------------------------------------
        # Fake Images
        # ----------------------------------------------------
        noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(z=noise, labels=fake_labels)
        output_fake = discriminator(x=fake_images.detach(), labels=fake_labels).view(-1, 1)
        loss_fake = criterion(output_fake, label_fake)
        loss_fake.backward()
        optimizer_d.step()

        # ====================================================
        # Train Generator
        # ====================================================
        optimizer_g.zero_grad()
        output = discriminator(x=fake_images, labels=fake_labels).view(-1, 1)
        loss_g = criterion(output,label_real)
        loss_g.backward()
        optimizer_g.step()

        # ====================================================
        # Calculate Discriminator Loss
        # ====================================================
        loss_d = loss_real.item() + loss_fake.item()

        # ====================================================
        # Update Progress Bar
        # ====================================================

        progress_bar.set_postfix({
            "D_loss": f"{loss_d:.4f}",
            "G_loss": f"{loss_g.item():.4f}",
            "D_real": f"{output_real.mean().item():.4f}",
            "D_fake": f"{output_fake.mean().item():.4f}"
        })

    # ========================================================
    # Generate Images After Each Epoch
    # ========================================================
    generator.eval()
    with torch.no_grad():
        fake_samples = generator(z=fixed_noise, labels=fixed_labels).cpu()
        # Create image grid
        fake_grid = torchvision.utils.make_grid(fake_samples, nrow=num_classes, padding=2, normalize=True)
        # Convert tensor to NumPy
        image = fake_grid.permute(1, 2, 0).numpy()
        # Convert grayscale image
        image = (image[:, :, 0] * 255).astype(np.uint8)
        # Add frame to GIF
        frames.append(Image.fromarray(image))
        # Display generated images
        cv2.imshow("DCGAN - Generated Images", image)
        cv2.waitKey(20)
    generator.train()


gif_pth = f"./results/{gan_type}_training.gif"
gan_pth = f"./models/{gan_type}_generator.pth"
dis_path = f"./models/{gan_type}_discriminator.pth"
# ============================================================
# Save Training GIF
# ============================================================
frames[0].save(gif_pth, save_all=True, append_images=frames[1:], duration=100, loop=0)
cv2.destroyAllWindows()

# ============================================================
# Save Models
# ============================================================
torch.save(generator.state_dict(), gan_pth)
torch.save(discriminator.state_dict(), dis_path)

# ============================================================
# Final Messages
# ============================================================
print("\nDCGAN training complete.")
print(f"GIF saved to {gif_pth}")
print(f"Generator saved to {gan_pth}")
print(f"Discriminator saved to {dis_path}")