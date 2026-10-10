import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, image_channels, num_classes):
        super(Generator, self).__init__()
        # Label embedding converts digit labels into vectors
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 256, 7, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, image_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels, **kwargs):
        # Convert labels into embeddings
        label_embedding = self.label_embedding(labels)
        # Reshape labels to [batch_size, latent_dim, 1, 1]
        label_embedding = label_embedding.unsqueeze(2).unsqueeze(3)
        # Concatenate noise and label embedding
        z = torch.cat((z, label_embedding), dim=1)
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, image_channels, num_classes, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        # Convert digit labels into image-like feature maps
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)

        self.model = nn.Sequential(
            nn.Conv2d(image_channels + 1, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 7, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x, labels, **kwargs):
        # Convert labels into a 28x28 feature map
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(labels.size(0), 1, self.image_size, self.image_size)
        # Concatenate image and label information
        x = torch.cat((x, label_embedding), dim=1)
        return self.model(x)

class cGAN:
    def __init__(self, latent_dim, image_channels, num_classes, image_size, **kwargs):
        self.generator = Generator(latent_dim, image_channels, num_classes)
        self.discriminator = Discriminator(image_channels, num_classes, image_size)