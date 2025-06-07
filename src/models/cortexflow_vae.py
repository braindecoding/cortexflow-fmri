"""
CortexFlow VAE Components
Advanced Variational Autoencoder for CortexFlow framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CortexFlowVAE(nn.Module):
    """Advanced VAE for CortexFlow with improved architecture."""
    
    def __init__(self, latent_dim=512, image_size=28, channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels
        
        # Encoder with residual connections
        self.encoder = nn.Sequential(
            # First block: 28x28 -> 14x14
            nn.Conv2d(channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # Second block: 14x14 -> 7x7
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # Third block: 7x7 -> 3x3
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # Fourth block: 3x3 -> 1x1
            nn.Conv2d(256, 512, 3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Flatten()
        )
        
        # Latent space projections
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 1, 1)),
            
            # First block: 1x1 -> 3x3
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # Second block: 3x3 -> 7x7
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # Third block: 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # Fourth block: 14x14 -> 28x28
            nn.ConvTranspose2d(64, channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent to image."""
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        """Full VAE forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, num_samples, device):
        """Sample from prior."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

class CortexFlowBetaVAE(CortexFlowVAE):
    """Beta-VAE variant with controllable disentanglement."""
    
    def __init__(self, latent_dim=512, image_size=28, channels=1, beta=1.0):
        super().__init__(latent_dim, image_size, channels)
        self.beta = beta

    def compute_loss(self, x, recon, mu, logvar):
        """Compute Beta-VAE loss."""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

class CortexFlowConditionalVAE(nn.Module):
    """Conditional VAE for class-conditioned generation."""
    
    def __init__(self, latent_dim=512, image_size=28, channels=1, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, 64)
        
        # Encoder (with class conditioning)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 4, stride=2, padding=1),  # +1 for class channel
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        # Latent projections
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder (with class conditioning)
        self.decoder_input = nn.Linear(latent_dim + 64, 512)  # +64 for class embedding
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, class_labels):
        """Encode with class conditioning."""
        batch_size = x.size(0)
        
        # Create class channel
        class_emb = self.class_embedding(class_labels)  # [B, 64]
        class_channel = class_emb.view(batch_size, 1, 1, 1).expand(-1, 1, self.image_size, self.image_size)
        
        # Concatenate image and class channel
        x_cond = torch.cat([x, class_channel], dim=1)
        
        # Encode
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

    def decode(self, z, class_labels):
        """Decode with class conditioning."""
        class_emb = self.class_embedding(class_labels)
        z_cond = torch.cat([z, class_emb], dim=1)
        h = self.decoder_input(z_cond)
        return self.decoder(h)

    def forward(self, x, class_labels):
        """Forward pass with class conditioning."""
        mu, logvar = self.encode(x, class_labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, class_labels)
        return recon, mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
