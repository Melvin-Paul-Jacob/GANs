import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, features=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d( features, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.PReLU(),
            nn.Conv2d( features, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features)
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_features, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features * (scale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, image_channels=1, generator_features=64, num_residual_blocks=16, upscale_factor=4):
        super(Generator, self).__init__()
        self.upscale_factor = upscale_factor
        # ----------------------------------------------------
        # Initial feature extraction
        # ----------------------------------------------------
        self.initial = nn.Sequential(
            nn.Conv2d(image_channels, generator_features, kernel_size=9, stride=1, padding=4),
            nn.PReLU())

        # ----------------------------------------------------
        # Residual blocks
        # ----------------------------------------------------

        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_blocks.append(
                ResidualBlock(generator_features)
            )
        self.residual_blocks = nn.Sequential(
            *residual_blocks
        )

        # ----------------------------------------------------
        # Residual block output
        # ----------------------------------------------------
        self.residual_output = nn.Sequential(
            nn.Conv2d(generator_features, generator_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(generator_features)
        )

        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------
        upsample_blocks = []

        # Example:
        # upscale_factor = 4
        #
        # 2x -> 2x
        # 4x -> 2x + 2x

        num_upsample_blocks = int(torch.log2(torch.tensor(upscale_factor)).item())
        for _ in range(num_upsample_blocks):
            upsample_blocks.append(UpsampleBlock(generator_features, scale_factor=2))

        self.upsampling = nn.Sequential(*upsample_blocks)

        # ----------------------------------------------------
        # Final output
        # ----------------------------------------------------
        self.output = nn.Sequential(
            nn.Conv2d(generator_features, image_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        # Initial features
        initial_features = self.initial(x)
        # Residual learning
        residual = self.residual_blocks(initial_features)
        residual = self.residual_output(residual)
        # Skip connection
        x = initial_features + residual
        # Upsampling
        x = self.upsampling(x)
        # Final image
        x = self.output(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, image_channels=1, discriminator_features=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_features, out_features, stride=1, first_block=False):

            layers = [nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1)]
            if not first_block:
                layers.append(nn.BatchNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.features = nn.Sequential(
            *discriminator_block(image_channels, discriminator_features, stride=1, first_block=True),
            *discriminator_block(discriminator_features, discriminator_features, stride=2),
            *discriminator_block(discriminator_features, discriminator_features * 2, stride=1),
            *discriminator_block(discriminator_features * 2, discriminator_features * 2, stride=2),
            *discriminator_block(discriminator_features * 2, discriminator_features * 4, stride=1),
            *discriminator_block(discriminator_features * 4, discriminator_features * 4, stride=2),
            *discriminator_block(discriminator_features * 4, discriminator_features * 8, stride=1),
            *discriminator_block(discriminator_features * 8, discriminator_features * 8, stride=2)
        )

        # ----------------------------------------------------
        # Classification layers
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(discriminator_features * 8, discriminator_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(discriminator_features * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class SRGAN:
    def __init__(self, image_channels=1, generator_features=64, discriminator_features=64, num_residual_blocks=16, upscale_factor=4, **kwargs):
        self.generator = Generator(
            image_channels=image_channels,
            generator_features=generator_features,
            num_residual_blocks=num_residual_blocks,
            upscale_factor=upscale_factor
        )
        self.discriminator = Discriminator(
            image_channels=image_channels,
            discriminator_features=discriminator_features
        )