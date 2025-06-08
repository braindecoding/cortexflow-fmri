#!/usr/bin/env python3
"""
HierarchicalCortexFlow: Multi-Scale Temporal Processing for fMRI-to-Image Reconstruction
Advanced architecture with temporal pyramids and progressive reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class HierarchicalConfig:
    """Enhanced configuration for HierarchicalCortexFlow."""
    # Base config (compatible with your existing Config)
    SEED: int = 42
    NUM_EPOCHS: int = 200
    PATIENCE: int = 20
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    BATCH_SIZE: int = 32
    IMAGE_SIZE: int = 28
    DROPOUT_RATE: float = 0.2
    GRAD_CLIP_NORM: float = 1.0
    
    # New hierarchical parameters
    TEMPORAL_SCALES: List[int] = None  # Will be [1, 2, 4, 8]
    HIDDEN_DIM: int = 512
    NUM_PYRAMID_LEVELS: int = 4
    PROGRESSIVE_WEIGHTS: List[float] = None  # Will be [0.1, 0.2, 0.3, 0.4]
    USE_SKIP_CONNECTIONS: bool = True
    FEATURE_FUSION_TYPE: str = "attention"  # "concat", "attention", "weighted"
    
    def __post_init__(self):
        if self.TEMPORAL_SCALES is None:
            self.TEMPORAL_SCALES = [1, 2, 4, 8]
        if self.PROGRESSIVE_WEIGHTS is None:
            self.PROGRESSIVE_WEIGHTS = [0.1, 0.2, 0.3, 0.4]


class TemporalEncoder(nn.Module):
    """Multi-scale temporal encoder for processing fMRI at different temporal resolutions."""
    
    def __init__(self, input_dim: int, scale: int, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.scale = scale
        self.input_dim = input_dim
        
        # Temporal convolution for this scale
        self.temporal_conv = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=min(scale * 2 + 1, input_dim // 4),
            stride=1,
            padding=scale
        )
        
        # Scale-specific encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Attention mechanism for temporal features
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Apply temporal convolution
        x_temp = x.unsqueeze(1)  # Add channel dimension: [batch, 1, input_dim]
        temp_features = self.temporal_conv(x_temp)  # [batch, 32, output_length]

        # Average across channels to get [batch, output_length]
        temp_features = temp_features.mean(dim=1)

        # If temporal features are different size, interpolate to match input
        if temp_features.shape[1] != x.shape[1]:
            # Reshape for 1D interpolation: [batch, 1, length]
            temp_features = F.interpolate(
                temp_features.unsqueeze(1),
                size=x.shape[1],
                mode='linear',
                align_corners=False
            ).squeeze(1)  # [batch, length]

        # Combine with original signal
        enhanced_signal = x + 0.1 * temp_features

        # Pass through encoder
        encoded = self.encoder(enhanced_signal)

        # Apply temporal attention
        attended, _ = self.temporal_attention(
            encoded.unsqueeze(1),
            encoded.unsqueeze(1),
            encoded.unsqueeze(1)
        )
        attended = attended.squeeze(1)

        return attended


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for fusing multi-scale temporal features."""
    
    def __init__(self, feature_dim: int, num_scales: int, fusion_type: str = "attention"):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        
        if fusion_type == "attention":
            # Cross-scale attention mechanism
            self.scale_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
            # Scale-wise projection layers
            self.scale_projections = nn.ModuleList([
                nn.Linear(feature_dim, feature_dim) for _ in range(num_scales)
            ])
            
            # Final fusion layer
            self.fusion_layer = nn.Sequential(
                nn.Linear(feature_dim * num_scales, feature_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim * 2, feature_dim)
            )
            
        elif fusion_type == "weighted":
            # Learnable weights for each scale
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            
        elif fusion_type == "concat":
            # Simple concatenation with projection
            self.projection = nn.Sequential(
                nn.Linear(feature_dim * num_scales, feature_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim * 2, feature_dim)
            )
    
    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        if self.fusion_type == "attention":
            # Project each scale
            projected_features = []
            for i, features in enumerate(scale_features):
                projected = self.scale_projections[i](features)
                projected_features.append(projected)
            
            # Stack for attention
            stacked = torch.stack(projected_features, dim=1)  # [batch, num_scales, feature_dim]
            
            # Apply cross-scale attention
            attended, attention_weights = self.scale_attention(stacked, stacked, stacked)
            
            # Concatenate and fuse
            concatenated = attended.flatten(start_dim=1)
            fused = self.fusion_layer(concatenated)
            
        elif self.fusion_type == "weighted":
            # Weighted combination
            weights = F.softmax(self.scale_weights, dim=0)
            weighted_features = []
            for i, features in enumerate(scale_features):
                weighted_features.append(features * weights[i])
            fused = torch.stack(weighted_features, dim=0).sum(dim=0)
            
        elif self.fusion_type == "concat":
            # Simple concatenation
            concatenated = torch.cat(scale_features, dim=1)
            fused = self.projection(concatenated)
        
        return fused


class ProgressiveDecoder(nn.Module):
    """Progressive decoder with skip connections and multi-scale outputs."""
    
    def __init__(self, feature_dim: int, image_size: int, num_levels: int = 4, use_skip: bool = True):
        super().__init__()
        self.num_levels = num_levels
        self.use_skip = use_skip
        self.image_size = image_size
        
        # Progressive decoder levels
        self.decoder_levels = nn.ModuleList()
        current_dim = feature_dim
        
        for level in range(num_levels):
            # Calculate output size for this level
            level_size = (image_size // (2 ** (num_levels - level - 1))) ** 2
            
            decoder_block = nn.Sequential(
                nn.Linear(current_dim, current_dim * 2),
                nn.LayerNorm(current_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                nn.Linear(current_dim * 2, current_dim * 4),
                nn.LayerNorm(current_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                nn.Linear(current_dim * 4, level_size),
                nn.Sigmoid()
            )
            
            self.decoder_levels.append(decoder_block)
            
            if self.use_skip and level < num_levels - 1:
                # Skip connection processing
                current_dim = current_dim + feature_dim // 2
        
        # Skip connection projections
        if self.use_skip:
            self.skip_projections = nn.ModuleList([
                nn.Linear(feature_dim, feature_dim // 2) 
                for _ in range(num_levels - 1)
            ])
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        current_features = features
        
        for level in range(self.num_levels):
            # Decode at current level
            level_output = self.decoder_levels[level](current_features)
            
            # Reshape to image format
            level_image_size = self.image_size // (2 ** (self.num_levels - level - 1))
            level_output = level_output.view(-1, 1, level_image_size, level_image_size)
            
            # Upsample to final image size if needed
            if level_image_size != self.image_size:
                level_output = F.interpolate(
                    level_output, 
                    size=(self.image_size, self.image_size), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            outputs[f'level_{level}'] = level_output
            
            # Add skip connection for next level
            if self.use_skip and level < self.num_levels - 1:
                skip_features = self.skip_projections[level](features)
                current_features = torch.cat([current_features, skip_features], dim=1)
        
        return outputs


class HierarchicalCortexFlow(nn.Module):
    """
    HierarchicalCortexFlow: Advanced multi-scale temporal processing architecture
    for fMRI-to-image reconstruction with progressive decoding.
    """
    
    def __init__(self, input_dim: int, config: HierarchicalConfig = HierarchicalConfig()):
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        
        print(f"🏗️  Building HierarchicalCortexFlow with {len(config.TEMPORAL_SCALES)} temporal scales")
        
        # Multi-scale temporal encoders
        self.temporal_encoders = nn.ModuleList([
            TemporalEncoder(
                input_dim=input_dim,
                scale=scale,
                hidden_dim=config.HIDDEN_DIM,
                dropout_rate=config.DROPOUT_RATE
            ) for scale in config.TEMPORAL_SCALES
        ])
        
        # Feature pyramid network for multi-scale fusion
        self.feature_pyramid = FeaturePyramidNetwork(
            feature_dim=config.HIDDEN_DIM // 2,
            num_scales=len(config.TEMPORAL_SCALES),
            fusion_type=config.FEATURE_FUSION_TYPE
        )
        
        # Progressive decoder
        self.progressive_decoder = ProgressiveDecoder(
            feature_dim=config.HIDDEN_DIM // 2,
            image_size=config.IMAGE_SIZE,
            num_levels=config.NUM_PYRAMID_LEVELS,
            use_skip=config.USE_SKIP_CONNECTIONS
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HierarchicalCortexFlow.
        
        Args:
            input_data: fMRI data [batch_size, input_dim]
            
        Returns:
            Dictionary containing multi-scale reconstructions
        """
        # Extract features at multiple temporal scales
        scale_features = []
        for i, encoder in enumerate(self.temporal_encoders):
            scale_feature = encoder(input_data)
            scale_features.append(scale_feature)
        
        # Fuse multi-scale features
        fused_features = self.feature_pyramid(scale_features)
        
        # Progressive decoding
        progressive_outputs = self.progressive_decoder(fused_features)
        
        # Main output is the highest resolution
        main_output = progressive_outputs[f'level_{self.config.NUM_PYRAMID_LEVELS - 1}']
        
        return {
            'reconstruction': main_output,
            'progressive_outputs': progressive_outputs,
            'scale_features': scale_features,
            'fused_features': fused_features
        }
    
    def compute_loss(self, input_data: torch.Tensor, target_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale hierarchical loss.
        
        Args:
            input_data: fMRI data [batch_size, input_dim]
            target_images: Target images [batch_size, channels, height, width]
            
        Returns:
            Dictionary containing loss components
        """
        outputs = self.forward(input_data)
        
        # Ensure target images have correct shape
        if target_images.dim() == 2:
            target_images = target_images.view(-1, 1, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        
        # Multi-scale reconstruction loss
        progressive_losses = []
        progressive_outputs = outputs['progressive_outputs']
        
        for level in range(self.config.NUM_PYRAMID_LEVELS):
            level_output = progressive_outputs[f'level_{level}']
            level_loss = F.mse_loss(level_output, target_images)
            progressive_losses.append(level_loss)
        
        # Weighted combination of progressive losses
        total_progressive_loss = sum(
            weight * loss for weight, loss in 
            zip(self.config.PROGRESSIVE_WEIGHTS, progressive_losses)
        )
        
        # Main reconstruction loss
        main_reconstruction = outputs['reconstruction']
        main_loss = F.mse_loss(main_reconstruction, target_images)
        
        # Perceptual loss (gradient-based edge preservation)
        edge_loss = self._compute_edge_loss(main_reconstruction, target_images)
        
        # Feature diversity loss (encourage different scales to learn different features)
        diversity_loss = self._compute_diversity_loss(outputs['scale_features'])
        
        # Total loss
        total_loss = (
            0.6 * main_loss + 
            0.3 * total_progressive_loss + 
            0.08 * edge_loss + 
            0.02 * diversity_loss
        )
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'progressive_loss': total_progressive_loss,
            'edge_loss': edge_loss,
            'diversity_loss': diversity_loss,
            'reconstruction': main_reconstruction,
            'progressive_outputs': progressive_outputs
        }
    
    def _compute_edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge preservation loss using gradient information."""
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        # Edge magnitude
        pred_edges = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        return F.mse_loss(pred_edges, target_edges)
    
    def _compute_diversity_loss(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """Encourage different temporal scales to learn diverse features."""
        if len(scale_features) < 2:
            return torch.tensor(0.0, device=scale_features[0].device)
        
        diversity_loss = 0.0
        num_pairs = 0
        
        for i in range(len(scale_features)):
            for j in range(i + 1, len(scale_features)):
                # Compute cosine similarity between scale features
                feat_i = F.normalize(scale_features[i], dim=1)
                feat_j = F.normalize(scale_features[j], dim=1)
                similarity = (feat_i * feat_j).sum(dim=1).mean()
                
                # Encourage low similarity (diversity)
                diversity_loss += similarity**2
                num_pairs += 1
        
        return diversity_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=scale_features[0].device)


# Enhanced trainer class for HierarchicalCortexFlow
class HierarchicalModelTrainer:
    """Enhanced trainer for HierarchicalCortexFlow with multi-scale loss handling."""
    
    def __init__(self, model: HierarchicalCortexFlow, device: torch.device, config: HierarchicalConfig):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer with different learning rates for different components
        param_groups = [
            {'params': model.temporal_encoders.parameters(), 'lr': config.LEARNING_RATE},
            {'params': model.feature_pyramid.parameters(), 'lr': config.LEARNING_RATE * 0.8},
            {'params': model.progressive_decoder.parameters(), 'lr': config.LEARNING_RATE * 1.2}
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, verbose=True
        )
        
        # Loss tracking
        self.loss_history = {
            'total_loss': [],
            'main_loss': [],
            'progressive_loss': [],
            'edge_loss': [],
            'diversity_loss': []
        }
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch with detailed loss tracking."""
        self.model.train()
        epoch_losses = {key: 0.0 for key in self.loss_history.keys()}
        
        for batch_idx, (input_data, images, _) in enumerate(train_loader):
            input_data, images = input_data.to(self.device), images.to(self.device)
            
            self.optimizer.zero_grad()
            loss_dict = self.model.compute_loss(input_data, images)
            
            total_loss = loss_dict['total_loss']
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.GRAD_CLIP_NORM)
            
            self.optimizer.step()
            
            # Track losses
            for key in epoch_losses.keys():
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
        
        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= len(train_loader)
            self.loss_history[key].append(epoch_losses[key])
        
        return epoch_losses
    
    def validate_epoch(self, test_loader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Validate for one epoch with sample outputs."""
        self.model.eval()
        epoch_losses = {key: 0.0 for key in self.loss_history.keys()}
        sample_outputs = None
        
        with torch.no_grad():
            for batch_idx, (input_data, images, _) in enumerate(test_loader):
                input_data, images = input_data.to(self.device), images.to(self.device)
                
                loss_dict = self.model.compute_loss(input_data, images)
                
                # Track losses
                for key in epoch_losses.keys():
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item()
                
                # Save sample outputs from first batch
                if batch_idx == 0:
                    sample_outputs = {
                        'original': images[:8].cpu(),
                        'reconstruction': loss_dict['reconstruction'][:8].cpu(),
                        'progressive_outputs': {
                            level: output[:8].cpu() 
                            for level, output in loss_dict['progressive_outputs'].items()
                        }
                    }
        
        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= len(test_loader)
        
        return epoch_losses, sample_outputs


def create_hierarchical_model(input_dim: int, config: HierarchicalConfig = None) -> HierarchicalCortexFlow:
    """Factory function to create HierarchicalCortexFlow model."""
    if config is None:
        config = HierarchicalConfig()
    
    print(f"🚀 Creating HierarchicalCortexFlow:")
    print(f"   📊 Input dimension: {input_dim}")
    print(f"   ⏱️  Temporal scales: {config.TEMPORAL_SCALES}")
    print(f"   🏗️  Hidden dimension: {config.HIDDEN_DIM}")
    print(f"   📈 Pyramid levels: {config.NUM_PYRAMID_LEVELS}")
    print(f"   🔗 Skip connections: {config.USE_SKIP_CONNECTIONS}")
    print(f"   🤝 Feature fusion: {config.FEATURE_FUSION_TYPE}")
    
    model = HierarchicalCortexFlow(input_dim, config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   🎯 Total parameters: {total_params:,}")
    print(f"   🎯 Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Example usage and testing
    print("🧠 Testing HierarchicalCortexFlow Architecture")
    print("=" * 60)
    
    # Test configuration
    config = HierarchicalConfig(
        TEMPORAL_SCALES=[1, 2, 4, 8],
        HIDDEN_DIM=512,
        NUM_PYRAMID_LEVELS=4,
        IMAGE_SIZE=28
    )
    
    # Create model
    input_dim = 1000  # Example fMRI dimension
    model = create_hierarchical_model(input_dim, config)
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy data
    batch_size = 4
    dummy_fmri = torch.randn(batch_size, input_dim, device=device)
    dummy_images = torch.randn(batch_size, 1, 28, 28, device=device)
    
    print(f"\n🔄 Testing forward pass with batch size {batch_size}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_fmri)
        print(f"✅ Forward pass successful!")
        print(f"   📊 Main reconstruction shape: {outputs['reconstruction'].shape}")
        print(f"   📈 Progressive outputs: {len(outputs['progressive_outputs'])} levels")
    
    # Test loss computation
    loss_dict = model.compute_loss(dummy_fmri, dummy_images)
    print(f"\n📉 Loss computation:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"   {key}: {value.item():.6f}")
    
    print(f"\n🎉 HierarchicalCortexFlow testing completed successfully!")