"""
Monte Carlo Simple CortexFlow: Simple Architecture with Uncertainty Estimation

This module extends the Simple CortexFlow with Monte Carlo Dropout for uncertainty
quantification while maintaining the simplicity and efficiency of the base model.

Key Features:
- Simple encoder-decoder architecture (like Simple CortexFlow)
- Monte Carlo Dropout for uncertainty estimation
- Epistemic and aleatoric uncertainty quantification
- Efficient inference with configurable MC samples
- Compatible with existing Simple CortexFlow training pipelines

Author: CortexFlow Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class MonteCarloDropout(nn.Module):
    """
    Monte Carlo Dropout layer that applies dropout during both training and inference.
    This enables uncertainty estimation through multiple forward passes.
    """
    
    def __init__(self, dropout_rate: float = 0.15):
        super().__init__()
        self.dropout_rate = dropout_rate
        
    def forward(self, x: torch.Tensor, force_dropout: bool = False) -> torch.Tensor:
        """
        Apply dropout with option to force during inference.
        
        Args:
            x: Input tensor
            force_dropout: Force dropout even during eval mode (for MC sampling)
        """
        if self.training or force_dropout:
            return F.dropout(x, p=self.dropout_rate, training=True)
        else:
            return x


class MCSimpleCortexFlow(nn.Module):
    """
    Monte Carlo Simple CortexFlow: Simple architecture with uncertainty estimation.
    
    Combines the efficiency of Simple CortexFlow with Monte Carlo Dropout for
    uncertainty quantification. Maintains the same simple encoder-decoder structure
    but adds uncertainty estimation capabilities.
    
    Architecture:
    - Encoder: input_dim -> hidden_dim (with MC dropout)
    - Decoder: hidden_dim -> 784 (28x28 image) (with MC dropout)
    - Uncertainty Head: hidden_dim -> 1 (uncertainty estimation)
    - Output: Sigmoid activation + uncertainty estimates
    
    Args:
        input_dim (int): Dimension of input brain signals
        image_size (int): Size of output images (default: 28 for 28x28)
        hidden_dim (int): Hidden layer dimension (default: 512)
        dropout_rate (float): Dropout rate for MC sampling (default: 0.15)
        mc_samples (int): Number of Monte Carlo samples for uncertainty (default: 10)
        uncertainty_weight (float): Weight for uncertainty loss (default: 0.1)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        image_size: int = 28, 
        hidden_dim: int = 512,
        dropout_rate: float = 0.15,
        mc_samples: int = 10,
        uncertainty_weight: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.mc_samples = mc_samples
        self.uncertainty_weight = uncertainty_weight
        self.output_dim = image_size * image_size
        
        # Encoder: Brain signals -> Hidden representation (with MC Dropout)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            MonteCarloDropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            MonteCarloDropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder: Hidden representation -> Image (with MC Dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            MonteCarloDropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(inplace=True),
            MonteCarloDropout(dropout_rate),
            nn.Linear(hidden_dim * 4, self.output_dim),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            MonteCarloDropout(dropout_rate * 0.5),  # Lower dropout for uncertainty
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty values
        )
        
        # Track uncertainty statistics
        self.uncertainty_history = []
        
    def forward(self, brain_signals: torch.Tensor, return_uncertainty: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass: Brain signals -> Reconstructed images + uncertainty
        
        Args:
            brain_signals: Input brain signals [batch_size, input_dim]
            return_uncertainty: Whether to compute uncertainty estimates
            
        Returns:
            Dictionary containing:
            - reconstruction: Reconstructed images [batch_size, 1, image_size, image_size]
            - uncertainty: Uncertainty estimates [batch_size, 1] (if return_uncertainty=True)
            - features: Hidden features [batch_size, hidden_dim]
        """
        # Encode brain signals to hidden representation
        hidden_features = self.encoder(brain_signals)
        
        # Decode hidden representation to image
        flat_image = self.decoder(hidden_features)
        
        # Reshape to image format [batch_size, 1, height, width]
        reconstruction = flat_image.view(-1, 1, self.image_size, self.image_size)
        
        result = {
            'reconstruction': reconstruction,
            'features': hidden_features
        }
        
        # Compute uncertainty if requested
        if return_uncertainty:
            uncertainty = self.uncertainty_head(hidden_features)
            result['uncertainty'] = uncertainty
            
            # Track uncertainty statistics
            if self.training:
                self.uncertainty_history.append(uncertainty.mean().item())
        
        return result
    
    def forward_with_mc_sampling(self, brain_signals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Monte Carlo sampling for uncertainty estimation.
        
        This method performs multiple forward passes with dropout enabled
        to estimate both epistemic (model) and aleatoric (data) uncertainty.
        
        Args:
            brain_signals: Input brain signals [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
            - reconstruction_mean: Mean reconstruction across MC samples
            - reconstruction_std: Standard deviation across MC samples
            - epistemic_uncertainty: Model uncertainty
            - aleatoric_uncertainty: Data uncertainty
            - total_uncertainty: Combined uncertainty
        """
        if self.training:
            # During training, use single forward pass
            return self.forward(brain_signals, return_uncertainty=True)
        
        # Set to eval mode but force dropout for MC sampling
        self.eval()
        
        mc_reconstructions = []
        mc_uncertainties = []
        
        with torch.no_grad():
            for _ in range(self.mc_samples):
                # Force dropout during inference for MC sampling
                result = self._forward_with_forced_dropout(brain_signals)
                mc_reconstructions.append(result['reconstruction'])
                mc_uncertainties.append(result['uncertainty'])
        
        # Stack MC samples
        mc_reconstructions = torch.stack(mc_reconstructions, dim=0)  # [mc_samples, batch, 1, H, W]
        mc_uncertainties = torch.stack(mc_uncertainties, dim=0)      # [mc_samples, batch, 1]
        
        # Calculate statistics
        reconstruction_mean = mc_reconstructions.mean(dim=0)
        reconstruction_std = mc_reconstructions.std(dim=0)
        
        # Epistemic uncertainty (model uncertainty) - variance across MC samples
        epistemic_uncertainty = mc_reconstructions.var(dim=0).mean(dim=[1, 2, 3], keepdim=True)
        
        # Aleatoric uncertainty (data uncertainty) - mean of predicted uncertainties
        aleatoric_uncertainty = mc_uncertainties.mean(dim=0)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'reconstruction': reconstruction_mean,
            'reconstruction_mean': reconstruction_mean,
            'reconstruction_std': reconstruction_std,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'mc_samples': self.mc_samples
        }
    
    def _forward_with_forced_dropout(self, brain_signals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Helper method to force dropout during inference for MC sampling."""
        # Manually apply forward pass with forced dropout
        x = brain_signals
        
        # Encoder with forced dropout
        for layer in self.encoder:
            if isinstance(layer, MonteCarloDropout):
                x = layer(x, force_dropout=True)
            else:
                x = layer(x)
        
        hidden_features = x
        
        # Decoder with forced dropout
        for layer in self.decoder:
            if isinstance(layer, MonteCarloDropout):
                x = layer(x, force_dropout=True)
            else:
                x = layer(x)
        
        flat_image = x
        reconstruction = flat_image.view(-1, 1, self.image_size, self.image_size)
        
        # Uncertainty head with forced dropout
        unc_x = hidden_features
        for layer in self.uncertainty_head:
            if isinstance(layer, MonteCarloDropout):
                unc_x = layer(unc_x, force_dropout=True)
            else:
                unc_x = layer(unc_x)
        
        uncertainty = unc_x
        
        return {
            'reconstruction': reconstruction,
            'uncertainty': uncertainty,
            'features': hidden_features
        }

    def compute_loss(self, brain_signals: torch.Tensor, target_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss with uncertainty-aware components.

        Args:
            brain_signals: Input brain signals [batch_size, input_dim]
            target_images: Target images [batch_size, 784] or [batch_size, 1, 28, 28]

        Returns:
            Dictionary containing loss components
        """
        # Forward pass
        outputs = self.forward(brain_signals, return_uncertainty=True)
        reconstruction = outputs['reconstruction']
        uncertainty = outputs['uncertainty']

        # Ensure target images are in correct format [batch_size, 1, height, width]
        if target_images.dim() == 2:  # [batch_size, 784]
            target_images = target_images.view(-1, 1, self.image_size, self.image_size)
        elif target_images.dim() == 3:  # [batch_size, height, width]
            target_images = target_images.unsqueeze(1)  # Add channel dimension

        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstruction, target_images)

        # Uncertainty loss - encourage reasonable uncertainty estimates
        # Lower uncertainty for better reconstructions
        pixel_errors = (reconstruction - target_images).pow(2).mean(dim=[1, 2, 3], keepdim=True)
        # Ensure uncertainty and pixel_errors have same shape
        if uncertainty.shape != pixel_errors.shape:
            pixel_errors = pixel_errors.squeeze()  # Remove extra dimensions
            if pixel_errors.dim() == 1:
                pixel_errors = pixel_errors.unsqueeze(1)  # Add back if needed
        uncertainty_loss = F.mse_loss(uncertainty, pixel_errors.detach())

        # Total loss
        total_loss = reconstruction_loss + self.uncertainty_weight * uncertainty_loss

        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'uncertainty_loss': uncertainty_loss,
            'reconstruction': reconstruction,
            'uncertainty': uncertainty
        }

    def get_model_info(self) -> Dict[str, any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'Monte Carlo Simple CortexFlow',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'mc_samples': self.mc_samples,
            'uncertainty_weight': self.uncertainty_weight,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'uncertainty_history_length': len(self.uncertainty_history)
        }

    def get_uncertainty_stats(self) -> Dict[str, float]:
        """Get uncertainty statistics from training history."""
        if not self.uncertainty_history:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

        history = np.array(self.uncertainty_history)
        return {
            'mean': float(history.mean()),
            'std': float(history.std()),
            'min': float(history.min()),
            'max': float(history.max()),
            'samples': len(history)
        }


def create_mc_simple_cortexflow(
    input_dim: int,
    image_size: int = 28,
    hidden_dim: int = 512,
    dropout_rate: float = 0.15,
    mc_samples: int = 10,
    uncertainty_weight: float = 0.1
) -> MCSimpleCortexFlow:
    """
    Factory function to create Monte Carlo Simple CortexFlow model.

    Args:
        input_dim: Dimension of input brain signals
        image_size: Size of output images (default: 28)
        hidden_dim: Hidden layer dimension (default: 512)
        dropout_rate: Dropout rate for MC sampling (default: 0.15)
        mc_samples: Number of MC samples for uncertainty (default: 10)
        uncertainty_weight: Weight for uncertainty loss (default: 0.1)

    Returns:
        MCSimpleCortexFlow model instance
    """
    model = MCSimpleCortexFlow(
        input_dim=input_dim,
        image_size=image_size,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        mc_samples=mc_samples,
        uncertainty_weight=uncertainty_weight
    )

    print(f"🎲 Monte Carlo Simple CortexFlow Created:")
    print(f"   📊 Input dimension: {input_dim}")
    print(f"   🖼️  Output dimension: {image_size}×{image_size} ({image_size*image_size} pixels)")
    print(f"   🧠 Hidden dimension: {hidden_dim}")
    print(f"   🎯 Dropout rate: {dropout_rate}")
    print(f"   🎲 MC samples: {mc_samples}")
    print(f"   ⚖️  Uncertainty weight: {uncertainty_weight}")
    print(f"   🔧 Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


if __name__ == "__main__":
    # Example usage and testing
    print("🎲 Monte Carlo Simple CortexFlow Model")
    print("=" * 60)

    # Test model creation
    model = create_mc_simple_cortexflow(input_dim=967)  # Miyawaki config
    print(f"✅ Model created: {model.get_model_info()}")

    # Test forward pass
    batch_size = 8
    brain_signals = torch.randn(batch_size, 967)
    target_images = torch.randn(batch_size, 784)

    # Standard forward pass
    print(f"\n🧪 Testing standard forward pass:")
    outputs = model(brain_signals)
    print(f"✅ Forward pass: {brain_signals.shape} -> {outputs['reconstruction'].shape}")
    print(f"✅ Uncertainty shape: {outputs['uncertainty'].shape}")

    # Monte Carlo forward pass
    print(f"\n🎲 Testing Monte Carlo forward pass:")
    model.eval()
    mc_outputs = model.forward_with_mc_sampling(brain_signals)
    print(f"✅ MC reconstruction: {mc_outputs['reconstruction'].shape}")
    print(f"✅ Epistemic uncertainty: {mc_outputs['epistemic_uncertainty'].shape}")
    print(f"✅ Aleatoric uncertainty: {mc_outputs['aleatoric_uncertainty'].shape}")
    print(f"✅ Total uncertainty: {mc_outputs['total_uncertainty'].shape}")

    # Loss computation
    print(f"\n💰 Testing loss computation:")
    model.train()
    loss_dict = model.compute_loss(brain_signals, target_images)
    print(f"✅ Total loss: {loss_dict['total_loss'].item():.6f}")
    print(f"✅ Reconstruction loss: {loss_dict['reconstruction_loss'].item():.6f}")
    print(f"✅ Uncertainty loss: {loss_dict['uncertainty_loss'].item():.6f}")

    print(f"\n🎉 All tests passed! Monte Carlo Simple CortexFlow is ready!")
