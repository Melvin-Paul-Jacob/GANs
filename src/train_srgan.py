import os
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
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
# 28 -> 7 -> 28
upscale_factor = 4
generator_features = 64
discriminator_features = 64
num_residual_blocks = 16
# Loss weights
lambda_content = 1.0
lambda_adversarial = 0.001

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
gan = SRGAN(image_channels=image_channels,
            generator_features=generator_features,
            discriminator_features=discriminator_features,
            num_residual_blocks=num_residual_blocks,
            upscale_factor=upscale_factor
        )
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
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

# ============================================================
# Optimizers
# ============================================================
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

def create_low_resolution_images(images):
    """
    Convert HR images into LR images.

    Example:
        28x28 -> 7x7

    The SRGAN generator then converts:
        7x7 -> 28x28
    """
    scale = 1.0 / upscale_factor
    low_resolution = F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)
    return low_resolution

def tensor_to_uint8(image):
    # [-1, 1] -> [0, 1]
    image = (image + 1.0) / 2.0
    image = image.clamp(0, 1)
    image = image.squeeze(0)
    image = image.cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image

# ============================================================
# Fixed Test
# ============================================================
fixed_hr = torch.stack([mnist_dataset[i][0] for i in range(10)]).to(device)
fixed_lr = create_low_resolution_images(fixed_hr)
lr_image = F.interpolate(fixed_lr, size=(image_size, image_size), mode="nearest")

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
    for i, (real_images, _) in enumerate(progress_bar):
        # Move real images to device
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)
        low_res = create_low_resolution_images(real_images)

        # ====================================================
        # Train Discriminator
        # ====================================================
        optimizer_d.zero_grad()

        # ----------------------------------------------------
        # Real Images
        # ----------------------------------------------------
        output_real = discriminator(real_images)
        label_real = torch.ones_like(output_real, device=device)
        loss_real = adversarial_criterion(output_real, label_real)

        # ----------------------------------------------------
        # Fake Images
        # ----------------------------------------------------
        fake_hr = generator(low_res)
        output_fake = discriminator(fake_hr.detach())
        label_fake = torch.zeros_like(output_fake, device=device)
        loss_fake = adversarial_criterion(output_fake, label_fake)
        loss_d = (loss_real + loss_fake)
        loss_d.backward()
        optimizer_d.step()

        # ====================================================
        # Train Generator
        # ====================================================
        optimizer_g.zero_grad()
        fake_hr = generator(low_res)
        output_fake_for_generator = discriminator(fake_hr)
        label_real_for_generator = torch.ones_like(output_fake_for_generator, device=device)
        loss_g_adversarial = adversarial_criterion(output_fake_for_generator, label_real_for_generator)
        loss_g_content = content_criterion(fake_hr, real_images)
        loss_g = (lambda_content * loss_g_content + lambda_adversarial * loss_g_adversarial)

        loss_g.backward()
        optimizer_g.step()


        # ====================================================
        # Update Progress Bar
        # ====================================================

        progress_bar.set_postfix({
            "D_loss": f"{loss_d:.4f}",
            "G_loss": f"{loss_g.item():.4f}",
            "D_real": f"{output_real.mean().item():.4f}",
            "D_fake": f"{output_fake.mean().item():.4f}",
            "Content": f"{loss_g_content.item():.4f}"
        })

    # ========================================================
    # Generate Images After Each Epoch
    # ========================================================
    generator.eval()
    with torch.no_grad():
        fake_samples = generator(fixed_lr)
        comparison = torch.cat([fixed_hr, lr_image, fake_samples], dim=0)
        # Create image grid
        comparison_grid = torchvision.utils.make_grid(comparison.cpu(), nrow=10, padding=2, normalize=True)
        image = comparison_grid.permute(1, 2, 0).numpy()
        # Convert grayscale image
        image = (image[:, :, 0] * 255).astype(np.uint8)
        # Add frame to GIF
        frames.append(Image.fromarray(image))
        # Display generated images
        cv2.imshow("SRGAN - HR | LR | SR", image)
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