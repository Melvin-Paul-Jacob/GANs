import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, image_channels, generator_features):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: (latent_dim) x 1 x 1
            # Output: 256 x 7 x 7
            nn.ConvTranspose2d(latent_dim, generator_features * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(generator_features * 4),
            nn.ReLU(True),
            # 256 x 7 x 7 -> 128 x 14 x 14
            nn.ConvTranspose2d(generator_features * 4, generator_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(generator_features * 2),
            nn.ReLU(True),

            # 128 x 14 x 14 -> 64 x 28 x 28
            nn.ConvTranspose2d(generator_features * 2, generator_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(generator_features),
            nn.ReLU(True),
            # 64 x 28 x 28 -> 1 x 28 x 28
            nn.Conv2d(generator_features, image_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, **kwargs):
        return self.model(z)
    
class Discriminator(nn.Module):
    def __init__(self, image_channels, discriminator_features):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input:  1 x 28 x 28  
            # Output:  64 x 14 x 14

            nn.Conv2d(image_channels, discriminator_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 14 x 14  ->  128 x 7 x 7
            nn.Conv2d(discriminator_features, discriminator_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 7 x 7 -> 256 x 3 x 3
            nn.Conv2d(discriminator_features * 2, discriminator_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 3 x 3 -> 1 x 1 x 1

            nn.Conv2d(discriminator_features * 4, 1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, **kwargs):
        return self.model(x)

class DCGAN:
    def __init__(self, latent_dim, image_channels, generator_features, discriminator_features, **kwargs):
        self.generator = Generator(latent_dim, image_channels, generator_features)
        self.discriminator = Discriminator(image_channels, discriminator_features)