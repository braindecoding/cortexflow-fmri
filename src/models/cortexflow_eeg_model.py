"""
🧠⚡ CortexFlow EEG Model for Brain-to-Image Reconstruction

Specialized architecture for EEG data with temporal and spatial processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class TemporalTransformer(nn.Module):
    """Transformer for temporal EEG processing."""
    
    def __init__(self, channels: int, seq_length: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Project each channel to d_model
        self.channel_projection = nn.Linear(seq_length, d_model)
        
        # Positional encoding for channels
        self.pos_encoding = nn.Parameter(torch.randn(channels, d_model))
        
        # Transformer encoder (fix nested tensor warning)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=False  # Fix: Set to False to avoid nested tensor warning
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False  # Fix: Explicitly disable nested tensor
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: EEG data [batch_size, channels, time_points]
            
        Returns:
            Temporal features [batch_size, channels, d_model]
        """
        batch_size, channels, time_points = x.shape
        
        # Project each channel's time series to d_model
        x_proj = self.channel_projection(x)  # [batch_size, channels, d_model]
        
        # Add positional encoding
        x_proj = x_proj + self.pos_encoding.unsqueeze(0)
        
        # Apply transformer
        temporal_features = self.transformer(x_proj)  # [batch_size, channels, d_model]
        
        # Output projection
        output = self.output_proj(temporal_features)
        
        return output


class SpatialTransformer(nn.Module):
    """Transformer for spatial EEG channel relationships."""
    
    def __init__(self, channels: int, seq_length: int, d_model: int = 128, nhead: int = 8, num_layers: int = 3):
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Project time series to d_model
        self.time_projection = nn.Linear(seq_length, d_model)
        
        # Spatial attention for channel relationships (fix nested tensor warning)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=False  # Fix: Set to False to avoid nested tensor warning
        )
        self.spatial_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False  # Fix: Explicitly disable nested tensor
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: EEG data [batch_size, channels, time_points]
            
        Returns:
            Spatial features [batch_size, channels, d_model]
        """
        batch_size, channels, time_points = x.shape
        
        # Project time dimension
        x_proj = self.time_projection(x)  # [batch_size, channels, d_model]
        
        # Apply spatial transformer
        spatial_features = self.spatial_transformer(x_proj)
        
        # Output projection
        output = self.output_proj(spatial_features)
        
        return output


class EEGEncoder(nn.Module):
    """Combined temporal and spatial EEG encoder."""
    
    def __init__(self, channels: int, seq_length: int, d_model: int = 128):
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Temporal and spatial processing
        self.temporal_transformer = TemporalTransformer(channels, seq_length, d_model)
        self.spatial_transformer = SpatialTransformer(channels, seq_length, d_model)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_proj = nn.Sequential(
            nn.Linear(channels * d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: EEG data [batch_size, channels, time_points]
            
        Returns:
            EEG features [batch_size, 256]
        """
        # Process temporal and spatial features
        temporal_features = self.temporal_transformer(x)  # [batch_size, channels, d_model]
        spatial_features = self.spatial_transformer(x)    # [batch_size, channels, d_model]
        
        # Fuse features
        combined = torch.cat([temporal_features, spatial_features], dim=-1)  # [batch_size, channels, d_model*2]
        fused_features = self.fusion(combined)  # [batch_size, channels, d_model]
        
        # Global pooling and final projection
        flattened = fused_features.view(fused_features.size(0), -1)  # [batch_size, channels*d_model]
        output = self.final_proj(flattened)  # [batch_size, 256]
        
        return output


class ImageDecoder(nn.Module):
    """Decoder to generate 28x28 images from EEG features."""
    
    def __init__(self, feature_dim: int = 256, image_size: int = 28):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        
        # Progressive upsampling
        self.decoder = nn.Sequential(
            # Start from feature vector
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Reshape to spatial
            nn.Linear(512, 7 * 7 * 64),
            nn.ReLU(),
        )
        
        # Convolutional upsampling
        self.conv_decoder = nn.Sequential(
            # 7x7x64 -> 14x14x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 14x14x32 -> 28x28x16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 28x28x16 -> 28x28x1
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: EEG features [batch_size, feature_dim]
            
        Returns:
            Generated images [batch_size, 1, 28, 28]
        """
        # Decode to spatial features
        x = self.decoder(features)  # [batch_size, 7*7*64]
        
        # Reshape to spatial
        x = x.view(x.size(0), 64, 7, 7)  # [batch_size, 64, 7, 7]
        
        # Convolutional upsampling
        output = self.conv_decoder(x)  # [batch_size, 1, 28, 28]
        
        return output


class CortexFlowEEG(nn.Module):
    """CortexFlow model for EEG-to-image reconstruction."""
    
    def __init__(self, 
                 channels: int = 14,
                 seq_length: int = 256,
                 image_size: int = 28,
                 d_model: int = 128):
        """
        Initialize CortexFlow EEG model.
        
        Args:
            channels: Number of EEG channels
            seq_length: Length of EEG time series
            image_size: Output image size (28 for 28x28)
            d_model: Model dimension for transformers
        """
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        self.image_size = image_size
        self.d_model = d_model
        
        # EEG encoder
        self.eeg_encoder = EEGEncoder(channels, seq_length, d_model)
        
        # Image decoder
        self.image_decoder = ImageDecoder(feature_dim=256, image_size=image_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            eeg_data: EEG data [batch_size, channels, time_points]
            
        Returns:
            Reconstructed images [batch_size, 1, 28, 28]
        """
        # Encode EEG to features
        eeg_features = self.eeg_encoder(eeg_data)  # [batch_size, 256]
        
        # Decode to images
        reconstructed_images = self.image_decoder(eeg_features)  # [batch_size, 1, 28, 28]
        
        return reconstructed_images
    
    def get_feature_dim(self) -> int:
        """Get the dimension of EEG features."""
        return 256


# Loss functions
class EEGReconstructionLoss(nn.Module):
    """Balanced MSE + Perceptual loss for EEG-to-image reconstruction."""

    def __init__(self, mse_weight: float = 1.0, gradient_weight: float = 0.1, edge_weight: float = 0.05, ssim_weight: float = 0.03):
        super().__init__()
        self.mse_weight = mse_weight
        self.gradient_weight = gradient_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute balanced MSE + Perceptual reconstruction loss.

        Args:
            predictions: Predicted images [batch_size, 1, 28, 28]
            targets: Target images [batch_size, 784] or [batch_size, 1, 28, 28]

        Returns:
            Total loss and loss components
        """
        # Ensure targets are in correct format
        if targets.dim() == 2:  # [batch_size, 784]
            targets = targets.view(targets.size(0), 1, 28, 28)

        # 🎯 PRIMARY: MSE RECONSTRUCTION LOSS
        mse_loss = torch.nn.functional.mse_loss(predictions, targets)

        # 🎨 SECONDARY: PERCEPTUAL COMPONENTS
        # Gradient-based perceptual loss (edge preservation)
        pred_grad_x = torch.abs(predictions[:, :, :, 1:] - predictions[:, :, :, :-1])
        pred_grad_y = torch.abs(predictions[:, :, 1:, :] - predictions[:, :, :-1, :])

        target_grad_x = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])
        target_grad_y = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])

        gradient_loss = (
            torch.nn.functional.mse_loss(pred_grad_x, target_grad_x) +
            torch.nn.functional.mse_loss(pred_grad_y, target_grad_y)
        )

        # Sobel edge detection loss
        def sobel_edges(x):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

            edges_x = torch.nn.functional.conv2d(x, sobel_x, padding=1)
            edges_y = torch.nn.functional.conv2d(x, sobel_y, padding=1)

            return torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)

        pred_edges = sobel_edges(predictions)
        target_edges = sobel_edges(targets)
        edge_loss = torch.nn.functional.mse_loss(pred_edges, target_edges)

        # SSIM-based structural loss
        def ssim_loss(x, y, window_size=11):
            mu_x = torch.nn.functional.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
            mu_y = torch.nn.functional.avg_pool2d(y, window_size, stride=1, padding=window_size//2)

            mu_x_sq = mu_x ** 2
            mu_y_sq = mu_y ** 2
            mu_xy = mu_x * mu_y

            sigma_x_sq = torch.nn.functional.avg_pool2d(x**2, window_size, stride=1, padding=window_size//2) - mu_x_sq
            sigma_y_sq = torch.nn.functional.avg_pool2d(y**2, window_size, stride=1, padding=window_size//2) - mu_y_sq
            sigma_xy = torch.nn.functional.avg_pool2d(x*y, window_size, stride=1, padding=window_size//2) - mu_xy

            c1, c2 = 0.01**2, 0.03**2
            ssim_map = ((2*mu_xy + c1) * (2*sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))

            return 1 - ssim_map.mean()

        structural_loss = ssim_loss(predictions, targets)

        # 🎯 BALANCED TOTAL LOSS: MSE (primary) + Perceptual (secondary)
        total_loss = (
            self.mse_weight * mse_loss +
            self.gradient_weight * gradient_loss +
            self.edge_weight * edge_loss +
            self.ssim_weight * structural_loss
        )

        loss_dict = {
            'total': total_loss,
            'mse': mse_loss,
            'gradient': gradient_loss,
            'edge': edge_loss,
            'ssim': structural_loss
        }

        return total_loss, loss_dict


# Demo
if __name__ == "__main__":
    print("🧠⚡ Demo: CortexFlow EEG Model")
    print("=" * 40)
    
    # Test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model for MindBigData (14 channels, 256 time points)
    model = CortexFlowEEG(channels=14, seq_length=256).to(device)
    
    # Test input
    batch_size = 4
    eeg_data = torch.randn(batch_size, 14, 256).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(eeg_data)
    
    print(f"📊 Model test:")
    print(f"  Input shape: {eeg_data.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n✅ CortexFlow EEG model ready!")
