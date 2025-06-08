#!/usr/bin/env python3
"""
🔬 UNIFIED CORTEXFLOW - REFACTORED VERSION
================================================================================
Fixed version of Unified CortexFlow with proper dimension handling
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class AdaptiveComplexityModule(nn.Module):
    """
    FIXED: Adaptive complexity module with consistent tensor dimensions
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Complexity predictor
        self.complexity_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Simple path (like Simple CortexFlow)
        self.simple_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Complex path - FIXED: Consistent output dimensions
        self.complex_encoder = self._build_hierarchical_encoder()
        
        # Feature fusion - FIXED: Proper dimension calculation
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 2, hidden_dim // 2),  # 256 + 256 = 512 → 256
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _build_hierarchical_encoder(self):
        """FIXED: Build hierarchical encoder with consistent dimensions."""
        # Use a single encoder that outputs hidden_dim // 2 to match simple path
        complex_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),  # Match simple path output
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2)  # Ensure consistent output
        )
        return complex_encoder
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        batch_size = x.size(0)
        
        # Predict complexity need
        complexity_score = self.complexity_predictor(x).mean().item()
        
        # Simple path
        simple_features = self.simple_encoder(x)  # [batch, hidden_dim // 2]
        
        # Complex path - FIXED: Single encoder output
        complex_features = self.complex_encoder(x)  # [batch, hidden_dim // 2]
        
        # Adaptive fusion based on complexity score
        if complexity_score > 0.5:  # Use complex path
            # FIXED: Both features now have same dimensions
            combined = torch.cat([simple_features, complex_features], dim=-1)  # [batch, hidden_dim]
            output = self.fusion(combined)  # [batch, hidden_dim // 2]
        else:  # Use simple path
            output = simple_features  # [batch, hidden_dim // 2]
        
        return output, complexity_score

class UncertaintyEstimator(nn.Module):
    """
    Uncertainty estimation module using Monte Carlo Dropout
    """
    
    def __init__(self, hidden_dim: int, mc_samples: int = 10):
        super().__init__()
        self.mc_samples = mc_samples
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, features: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if training or self.mc_samples == 1:
            uncertainty = self.uncertainty_head(features)
            return features, uncertainty
        
        # Monte Carlo sampling during inference
        uncertainties = []
        for _ in range(self.mc_samples):
            self.train()  # Enable dropout
            uncertainty = self.uncertainty_head(features)
            uncertainties.append(uncertainty)
        
        self.eval()  # Restore eval mode
        
        # Calculate mean and variance
        uncertainties = torch.stack(uncertainties, dim=0)
        mean_uncertainty = uncertainties.mean(dim=0)
        uncertainty_variance = uncertainties.var(dim=0)
        
        # Combine epistemic and aleatoric uncertainty
        total_uncertainty = mean_uncertainty + uncertainty_variance
        
        return features, total_uncertainty

class UnifiedCortexFlow(nn.Module):
    """
    FIXED: Unified CortexFlow with proper dimension handling
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 784,  # 28x28 images
        hidden_dim: int = 512,
        use_uncertainty: bool = True,
        use_alignment: bool = True,
        mc_samples: int = 10,
        alignment_weight: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_uncertainty = use_uncertainty
        self.use_alignment = use_alignment
        self.alignment_weight = alignment_weight
        
        # Core components
        self.adaptive_encoder = AdaptiveComplexityModule(input_dim, hidden_dim)
        
        if use_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(hidden_dim // 2, mc_samples)
        
        # Decoder (optimized from Simple CortexFlow)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Sigmoid()
        )
        
        # Feature alignment head
        if use_alignment:
            self.alignment_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim // 8)
            )
        
        # Performance monitoring
        self.complexity_history = []
        self.uncertainty_history = []
        
    def forward(
        self, 
        x: torch.Tensor, 
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        # Adaptive encoding
        encoded_features, complexity_score = self.adaptive_encoder(x)
        self.complexity_history.append(complexity_score)
        
        # Uncertainty estimation
        uncertainty = None
        if self.use_uncertainty:
            encoded_features, uncertainty = self.uncertainty_estimator(
                encoded_features, self.training
            )
            if uncertainty is not None:
                self.uncertainty_history.append(uncertainty.mean().item())
        
        # Decode to output
        reconstruction = self.decoder(encoded_features)
        
        # Feature alignment
        alignment_loss = torch.tensor(0.0, device=x.device)
        if self.use_alignment and target is not None:
            # Simple alignment: compare reconstruction with target directly
            alignment_loss = F.mse_loss(reconstruction, target.view(target.size(0), -1))
        
        return {
            'reconstruction': reconstruction,
            'uncertainty': uncertainty,
            'alignment_loss': alignment_loss,
            'complexity_score': complexity_score,
            'encoded_features': encoded_features
        }
    
    def get_complexity_stats(self) -> Dict[str, float]:
        """Get statistics about model complexity usage."""
        if not self.complexity_history:
            return {'mean_complexity': 0.0, 'complexity_trend': 0.0}
        
        history = np.array(self.complexity_history[-100:])  # Last 100 samples
        return {
            'mean_complexity': float(history.mean()),
            'complexity_std': float(history.std()),
            'complexity_trend': float(np.polyfit(range(len(history)), history, 1)[0])
        }
    
    def get_uncertainty_stats(self) -> Dict[str, float]:
        """Get statistics about uncertainty estimates."""
        if not self.uncertainty_history:
            return {'mean_uncertainty': 0.0, 'uncertainty_trend': 0.0}
        
        history = np.array(self.uncertainty_history[-100:])  # Last 100 samples
        return {
            'mean_uncertainty': float(history.mean()),
            'uncertainty_std': float(history.std()),
            'uncertainty_trend': float(np.polyfit(range(len(history)), history, 1)[0])
        }
    
    def set_complexity_mode(self, mode: str):
        """
        Set complexity mode:
        - 'adaptive': Use adaptive complexity (default)
        - 'simple': Force simple path
        - 'complex': Force complex path
        """
        if mode == 'simple':
            # Modify complexity predictor to always return low scores
            for param in self.adaptive_encoder.complexity_predictor.parameters():
                param.requires_grad = False
            self.adaptive_encoder.complexity_predictor[2].bias.data.fill_(-2.0)
        elif mode == 'complex':
            # Modify complexity predictor to always return high scores
            for param in self.adaptive_encoder.complexity_predictor.parameters():
                param.requires_grad = False
            self.adaptive_encoder.complexity_predictor[2].bias.data.fill_(2.0)
        else:  # adaptive
            # Restore adaptive behavior
            for param in self.adaptive_encoder.complexity_predictor.parameters():
                param.requires_grad = True

class UnifiedLoss(nn.Module):
    """
    Unified loss function that combines:
    1. Reconstruction loss
    2. Uncertainty loss
    3. Alignment loss
    4. Complexity regularization
    """

    def __init__(
        self,
        uncertainty_weight: float = 0.1,
        alignment_weight: float = 0.1,
        complexity_weight: float = 0.01
    ):
        super().__init__()
        self.uncertainty_weight = uncertainty_weight
        self.alignment_weight = alignment_weight
        self.complexity_weight = complexity_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        # Reconstruction loss
        reconstruction_loss = F.mse_loss(outputs['reconstruction'], targets)

        # Uncertainty loss (negative log-likelihood)
        uncertainty_loss = torch.tensor(0.0, device=targets.device)
        if outputs['uncertainty'] is not None:
            # Uncertainty-weighted reconstruction loss
            precision = 1.0 / (outputs['uncertainty'] + 1e-8)
            uncertainty_loss = (
                precision * F.mse_loss(outputs['reconstruction'], targets, reduction='none').mean(dim=1, keepdim=True) +
                torch.log(outputs['uncertainty'] + 1e-8)
            ).mean()

        # Alignment loss
        alignment_loss = outputs['alignment_loss']

        # Complexity regularization (encourage efficiency)
        complexity_reg = outputs['complexity_score'] * self.complexity_weight

        # Total loss
        total_loss = (
            reconstruction_loss +
            self.uncertainty_weight * uncertainty_loss +
            self.alignment_weight * alignment_loss +
            complexity_reg
        )

        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'uncertainty_loss': uncertainty_loss,
            'alignment_loss': alignment_loss,
            'complexity_reg': complexity_reg
        }

def create_unified_model(
    input_dim: int,
    output_dim: int = 784,
    hidden_dim: int = 512,
    config: str = 'balanced'
) -> Tuple[UnifiedCortexFlow, UnifiedLoss]:
    """
    FIXED: Create unified model with predefined configurations.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden dimension
        config: Configuration preset ('simple', 'balanced', 'advanced')

    Returns:
        Tuple of (model, loss_function)
    """

    if config == 'simple':
        # Simple configuration (like Simple CortexFlow)
        model = UnifiedCortexFlow(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            use_uncertainty=False,
            use_alignment=False
        )
        model.set_complexity_mode('simple')
        loss_fn = UnifiedLoss(uncertainty_weight=0.0, alignment_weight=0.0)

    elif config == 'advanced':
        # Advanced configuration (all features)
        model = UnifiedCortexFlow(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            use_uncertainty=True,
            use_alignment=True,
            mc_samples=15,
            alignment_weight=0.15
        )
        loss_fn = UnifiedLoss(uncertainty_weight=0.15, alignment_weight=0.15)

    else:  # balanced
        # Balanced configuration (best of all worlds)
        model = UnifiedCortexFlow(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            use_uncertainty=True,
            use_alignment=True,
            mc_samples=10,
            alignment_weight=0.1
        )
        loss_fn = UnifiedLoss(uncertainty_weight=0.1, alignment_weight=0.1)

    return model, loss_fn

if __name__ == "__main__":
    # Test the FIXED unified model
    print("🔬 Testing FIXED Unified CortexFlow")

    # Test all configurations
    configs = ['simple', 'balanced', 'advanced']

    for config in configs:
        print(f"\n🧪 Testing {config} configuration:")
        try:
            # Create model
            model, loss_fn = create_unified_model(input_dim=967, config=config)

            # Test forward pass
            x = torch.randn(32, 967)
            target = torch.randn(32, 784)

            outputs = model(x, target)
            losses = loss_fn(outputs, target)

            print(f"  ✅ {config.upper()} config: SUCCESS")
            print(f"  📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  🎯 Total loss: {losses['total_loss'].item():.6f}")
            print(f"  🔧 Complexity: {outputs['complexity_score']:.3f}")

            if outputs['uncertainty'] is not None:
                print(f"  🎲 Uncertainty: {outputs['uncertainty'].mean().item():.6f}")

        except Exception as e:
            print(f"  ❌ {config.upper()} config: FAILED - {e}")

    print(f"\n🎉 FIXED Unified CortexFlow testing completed!")
