"""
🧠 CortexFlow: Novel Neural Decoding Framework
A novel multi-modal latent diffusion architecture for brain-to-image reconstruction.

This is the core CortexFlow model implementing:
- Multi-modal fusion architecture
- Novel feature alignment techniques  
- Advanced uncertainty calibration
- Cross-modal attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add utils to path for perceptual loss
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from perceptual_loss import CombinedPerceptualLoss
    VGG_AVAILABLE = True
except ImportError:
    VGG_AVAILABLE = False
    print("⚠️ VGG perceptual loss not available")

try:
    from simple_perceptual_loss import SimplePerceptualLoss
    SIMPLE_PERCEPTUAL_AVAILABLE = True
except ImportError:
    SIMPLE_PERCEPTUAL_AVAILABLE = False
    print("⚠️ Simple perceptual loss not available, falling back to MSE")

class CortexFlowEncoder(nn.Module):
    """Novel fMRI encoder with transformer architecture."""

    def __init__(self, fmri_dim=3092, hidden_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(fmri_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Transformer encoder layers (fix nested tensor warning)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.2,
            batch_first=True,
            norm_first=False  # Fix: Set to False to avoid nested tensor warning
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False  # Fix: Explicitly disable nested tensor
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, fmri_signals):
        # Project to hidden dimension
        x = self.input_projection(fmri_signals)  # [B, H]
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # [B, 1, H]
        
        # Apply transformer
        x = self.transformer(x)  # [B, 1, H]
        
        # Remove sequence dimension and project
        x = x.squeeze(1)  # [B, H]
        x = self.output_projection(x)
        
        return x

class CortexFlowCrossModalFusion(nn.Module):
    """Novel cross-modal fusion with attention mechanisms."""
    
    def __init__(self, hidden_dim=512, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True,
            bias=True
        )
        
        # Feature alignment network
        self.alignment_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, fmri_features, semantic_features=None):
        batch_size = fmri_features.size(0)

        if semantic_features is not None:
            # Ensure features have same dimension
            if fmri_features.size(-1) != semantic_features.size(-1):
                # Project semantic features to match fMRI dimension
                semantic_features = self.alignment_network(
                    torch.cat([semantic_features, torch.zeros_like(fmri_features)], dim=1)
                )[:, :fmri_features.size(-1)]

            # Prepare for attention - batch_first=True format
            fmri_seq = fmri_features.unsqueeze(1)  # [B, 1, H] - seq_len=1
            semantic_seq = semantic_features.unsqueeze(1)  # [B, 1, H] - seq_len=1

            # Cross-modal attention (query, key, value)
            try:
                attended_fmri, attention_weights = self.cross_attention(
                    fmri_seq, semantic_seq, semantic_seq
                )
                attended_fmri = attended_fmri.squeeze(1)  # [B, H]
            except Exception as e:
                # Fallback: simple concatenation if attention fails
                print(f"Attention failed, using fallback: {e}")
                attended_fmri = fmri_features
                attention_weights = None

            # Feature alignment
            combined = torch.cat([attended_fmri, semantic_features], dim=1)
            aligned_features = self.alignment_network(combined)
        else:
            aligned_features = fmri_features
            attention_weights = None

        # Estimate uncertainty
        uncertainty = self.uncertainty_head(aligned_features)

        return aligned_features, uncertainty, attention_weights

class CortexFlowVAE(nn.Module):
    """Novel VAE component for latent space generation."""
    
    def __init__(self, latent_dim=512, image_size=28):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 3x3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, latent_dim * 2)  # mu and logvar
        )
        
        # Decoder - carefully designed for 28x28 output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 7->14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),   # 14->28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

class CortexFlowDiffusion(nn.Module):
    """Novel diffusion process for guided generation."""
    
    def __init__(self, latent_dim=512, condition_dim=512, timesteps=1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.timesteps = timesteps
        
        # Condition projection
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, condition, guidance_scale=7.5):
        batch_size = condition.size(0)
        device = condition.device
        
        # Project condition
        projected_condition = self.condition_proj(condition)
        
        # Start with noise
        latent = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Simplified diffusion process
        for t in range(10):  # Simplified steps
            # Combine latent and condition
            combined = torch.cat([latent, projected_condition], dim=1)
            
            # Predict noise
            predicted_noise = self.noise_predictor(combined)
            
            # Update latent (simplified)
            latent = latent - 0.1 * predicted_noise
        
        return latent

class CortexFlow(nn.Module):
    """
    CortexFlow: Novel Neural Decoding Framework
    
    A state-of-the-art architecture for reconstructing visual stimuli from fMRI signals
    using multi-modal latent diffusion with advanced feature alignment.
    """
    
    def __init__(self, fmri_dim=3092, image_size=28, latent_dim=512, guidance_scale=7.5):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.guidance_scale = guidance_scale
        
        # Core components
        self.fmri_encoder = CortexFlowEncoder(fmri_dim, latent_dim)
        self.cross_modal_fusion = CortexFlowCrossModalFusion(latent_dim)
        self.vae = CortexFlowVAE(latent_dim, image_size)
        self.diffusion = CortexFlowDiffusion(latent_dim, latent_dim)
        
        # Semantic embedding for class conditioning (expanded for safety)
        self.semantic_embedding = nn.Embedding(1000, latent_dim)  # Expanded range for safety
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))

        # Initialize perceptual loss (priority: VGG > Simple > MSE)
        self.perceptual_loss = None
        self.use_perceptual = False
        self.perceptual_type = "mse"

        if VGG_AVAILABLE:
            try:
                self.perceptual_loss = CombinedPerceptualLoss(
                    vgg_weight=1.0,      # Primary: VGG perceptual loss
                    l1_weight=0.0,       # REMOVED: No L1 reconstruction
                    gradient_weight=0.3, # Enhanced: Edge preservation
                    ssim_weight=0.2      # Enhanced: Structural similarity
                )
                self.use_perceptual = True
                self.perceptual_type = "vgg"
                print("🎨 Using VGG perceptual loss (balanced with MSE)")
            except Exception as e:
                print(f"⚠️ VGG perceptual loss failed: {e}")

        if not self.use_perceptual and SIMPLE_PERCEPTUAL_AVAILABLE:
            try:
                self.perceptual_loss = SimplePerceptualLoss(
                    gradient_weight=1.0,  # Primary: Edge preservation
                    l1_weight=0.0,        # REMOVED: No L1 reconstruction
                    ssim_weight=0.5,      # Enhanced: Structural similarity
                    edge_weight=0.3       # Enhanced: Sobel edge detection
                )
                self.use_perceptual = True
                self.perceptual_type = "simple"
                print("🎨 Using Simple perceptual loss (balanced with MSE)")
            except Exception as e:
                print(f"⚠️ Simple perceptual loss failed: {e}")

        if not self.use_perceptual:
            print("❌ NO PERCEPTUAL LOSS AVAILABLE - FULL PERCEPTUAL TRAINING REQUIRES VGG!")

    def forward(self, fmri_signals, class_labels=None, guidance_scale=None):
        """Forward pass of CortexFlow model."""
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
            
        # Encode fMRI signals
        fmri_features = self.fmri_encoder(fmri_signals)
        
        # Get semantic features if labels provided
        semantic_features = None
        if class_labels is not None:
            # Ensure labels are within valid range
            class_labels = torch.clamp(class_labels, 0, 999)
            semantic_features = self.semantic_embedding(class_labels)
        
        # Cross-modal fusion
        fused_features, uncertainty, attention_weights = self.cross_modal_fusion(
            fmri_features, semantic_features
        )
        
        # Generate latent via diffusion
        latent = self.diffusion(fused_features, guidance_scale)
        
        # Decode to image
        reconstruction = self.vae.decode(latent)
        
        # Apply temperature scaling
        reconstruction = reconstruction / self.temperature
        
        return {
            'reconstruction': reconstruction,
            'uncertainty': uncertainty,
            'attention_weights': attention_weights,
            'latent': latent,
            'fused_features': fused_features
        }

    def compute_loss(self, fmri_signals, target_images, class_labels=None):
        """
        Compute CortexFlow loss with balanced MSE + Perceptual objectives.

        Balanced approach: MSE for pixel accuracy + Perceptual for quality.
        """
        # Forward pass
        outputs = self.forward(fmri_signals, class_labels)
        reconstruction = outputs['reconstruction']
        uncertainty = outputs['uncertainty']

        # Ensure target images are in correct format
        if target_images.dim() == 2:  # [batch, 784]
            target_images = target_images.view(-1, 1, self.image_size, self.image_size)

        # 🎯 PRIMARY: MSE RECONSTRUCTION LOSS
        mse_loss = F.mse_loss(reconstruction, target_images)

        # 🎨 SECONDARY: PERCEPTUAL LOSS (if available)
        perceptual_loss = torch.tensor(0.0, device=reconstruction.device)
        vgg_loss = torch.tensor(0.0, device=reconstruction.device)
        l1_loss = torch.tensor(0.0, device=reconstruction.device)
        gradient_loss = torch.tensor(0.0, device=reconstruction.device)
        ssim_loss = torch.tensor(0.0, device=reconstruction.device)
        edge_loss = torch.tensor(0.0, device=reconstruction.device)

        if self.use_perceptual and self.perceptual_loss is not None:
            try:
                perceptual_losses = self.perceptual_loss(reconstruction, target_images)

                # Extract individual components
                vgg_loss = perceptual_losses.get('vgg_loss', torch.tensor(0.0, device=reconstruction.device))
                l1_loss = perceptual_losses.get('l1_loss', torch.tensor(0.0, device=reconstruction.device))
                gradient_loss = perceptual_losses.get('gradient_loss', torch.tensor(0.0, device=reconstruction.device))
                ssim_loss = perceptual_losses.get('ssim_loss', torch.tensor(0.0, device=reconstruction.device))
                edge_loss = perceptual_losses.get('edge_loss', torch.tensor(0.0, device=reconstruction.device))
                perceptual_loss = perceptual_losses['total_loss']

            except Exception as e:
                print(f"⚠️ Perceptual loss failed, using MSE only: {e}")
                perceptual_loss = torch.tensor(0.0, device=reconstruction.device)

        # Uncertainty regularization
        uncertainty_reg = F.mse_loss(uncertainty, torch.ones_like(uncertainty) * 0.1)

        # 🎯 BALANCED TOTAL LOSS: MSE (primary) + Perceptual (secondary)
        mse_weight = 1.0  # Primary weight for pixel accuracy
        perceptual_weight = 0.1  # Secondary weight for perceptual quality

        total_loss = (mse_weight * mse_loss +
                     perceptual_weight * perceptual_loss +
                     0.01 * uncertainty_reg)

        return {
            'total_loss': total_loss,
            'recon_loss': mse_loss,  # Primary reconstruction loss
            'perceptual_loss': perceptual_loss,
            'vgg_loss': vgg_loss,
            'l1_loss': l1_loss,
            'gradient_loss': gradient_loss,
            'ssim_loss': ssim_loss,
            'edge_loss': edge_loss,
            'uncertainty_reg': uncertainty_reg,
            'mse_loss': mse_loss  # For compatibility
        }

# Utility functions for compatibility
def create_digit_captions(labels):
    """Create captions for digit labels."""
    digit_names = ['zero', 'one', 'two', 'three', 'four',
                   'five', 'six', 'seven', 'eight', 'nine']
    return [f"digit {digit_names[label.item()]}" for label in labels]
