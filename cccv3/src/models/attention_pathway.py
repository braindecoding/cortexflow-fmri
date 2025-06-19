"""
CCCV3 Attention Pathway
======================

Recreates the proven CCCV2 attention mechanisms that showed good performance
on large datasets (MindBigData, Crell). This pathway focuses on feature
relationships and selective attention.

Key Features:
- Multi-head self-attention for feature relationships
- Residual connections for gradient flow
- Adaptive attention based on dataset characteristics
- Strong performance on large datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionPathway(nn.Module):
    """
    Attention pathway based on proven CCCV2 attention mechanisms
    
    Architecture Philosophy:
    - Self-attention for feature relationships
    - Multi-head attention for diverse perspectives
    - Residual connections for training stability
    - Adaptive to dataset characteristics
    """
    
    def __init__(self, input_dim, output_dim=512, num_heads=4, dropout=0.1):
        super(AttentionPathway, self).__init__()
        self.pathway_name = "Attention"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Input projection to attention space
        self.input_projection = nn.Linear(input_dim, output_dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            output_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Forward pass through Attention pathway
        
        Args:
            x: [batch_size, input_dim] - fMRI features
        Returns:
            features: [batch_size, output_dim] - Attention pathway features
            pathway_info: Dict with pathway-specific information
        """
        # Project to attention space
        projected = self.input_projection(x)  # [batch_size, output_dim]
        
        # Add sequence dimension for attention
        projected = projected.unsqueeze(1)  # [batch_size, 1, output_dim]
        
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(projected, projected, projected)
        attended = self.layer_norm1(projected + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(attended)
        features = self.layer_norm2(attended + ffn_output)
        
        # Remove sequence dimension
        features = features.squeeze(1)  # [batch_size, output_dim]
        
        # Final normalization
        features = self.output_norm(features)
        
        # Pathway information for analysis
        pathway_info = {
            'pathway_name': self.pathway_name,
            'num_heads': self.num_heads,
            'attention_weights': attn_weights.squeeze(1),  # [batch_size, 1, 1] -> [batch_size]
            'feature_norm': torch.norm(features, dim=1).mean().item(),
            'attention_entropy': self._compute_attention_entropy(attn_weights)
        }
        
        return features, pathway_info
    
    def _compute_attention_entropy(self, attn_weights):
        """Compute entropy of attention weights for analysis"""
        # attn_weights: [batch_size, num_heads, 1, 1]
        weights = attn_weights.squeeze(-1).squeeze(-1)  # [batch_size, num_heads]
        weights = F.softmax(weights, dim=-1)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        return entropy.item()


class MultiScaleAttentionPathway(nn.Module):
    """
    Multi-scale attention pathway with hierarchical processing
    
    Improvements:
    - Multiple attention scales
    - Hierarchical feature processing
    - Adaptive scale weighting
    """
    
    def __init__(self, input_dim, output_dim=512, scales=[4, 8, 16], dropout=0.1):
        super(MultiScaleAttentionPathway, self).__init__()
        self.pathway_name = "Multi-Scale-Attention"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scales = scales
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, output_dim)
        
        # Multi-scale attention modules
        self.attention_modules = nn.ModuleList([
            nn.MultiheadAttention(output_dim, num_heads, dropout=dropout, batch_first=True)
            for num_heads in scales
        ])
        
        # Layer normalization for each scale
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(output_dim) for _ in scales
        ])
        
        # Scale fusion mechanism
        self.scale_fusion = nn.Sequential(
            nn.Linear(output_dim * len(scales), output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Adaptive scale weighting
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        
    def forward(self, x):
        """
        Forward pass with multi-scale attention
        """
        # Project to attention space
        projected = self.input_projection(x).unsqueeze(1)  # [batch_size, 1, output_dim]
        
        # Multi-scale attention processing
        scale_features = []
        scale_attentions = []
        
        for i, (attention_module, layer_norm) in enumerate(zip(self.attention_modules, self.layer_norms)):
            # Apply attention at this scale
            attn_output, attn_weights = attention_module(projected, projected, projected)
            
            # Residual connection and normalization
            scale_feature = layer_norm(projected + attn_output)
            scale_features.append(scale_feature.squeeze(1))  # Remove sequence dim
            scale_attentions.append(attn_weights)
        
        # Adaptive scale weighting
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        # Weighted combination of scales
        weighted_features = []
        for i, features in enumerate(scale_features):
            weighted = scale_weights[i] * features
            weighted_features.append(weighted)
        
        # Concatenate and fuse
        concatenated = torch.cat(weighted_features, dim=-1)
        fused_features = self.scale_fusion(concatenated)
        
        # Pathway information
        pathway_info = {
            'pathway_name': self.pathway_name,
            'scales': self.scales,
            'scale_weights': scale_weights.detach().cpu().numpy(),
            'scale_attentions': scale_attentions,
            'feature_norm': torch.norm(fused_features, dim=1).mean().item()
        }
        
        return fused_features, pathway_info


class AttentionPathwayDecoder(nn.Module):
    """
    Decoder specifically designed for Attention pathway features
    
    Maintains attention-based processing while decoding to visual output
    """
    
    def __init__(self, input_dim=512, output_dim=784):
        super(AttentionPathwayDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Attention-aware decoder
        self.decoder = nn.Sequential(
            # Feature refinement
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Dropout(0.02),
            
            # Visual feature generation
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()  # Visual output in [0, 1] range
        )
        
    def forward(self, features):
        """
        Decode Attention pathway features to visual output
        
        Args:
            features: [batch_size, input_dim] - Attention pathway features
        Returns:
            visual_output: [batch_size, 1, 28, 28] - Reconstructed images
        """
        # Decode features
        output = self.decoder(features)
        
        # Reshape to image format
        visual_output = output.view(-1, 1, 28, 28)
        
        return visual_output


class CompleteAttentionModel(nn.Module):
    """
    Complete Attention model for standalone testing
    
    Combines Attention pathway with appropriate decoder for end-to-end testing
    """
    
    def __init__(self, input_dim, dataset_size=100, use_multiscale=True, device='cuda'):
        super(CompleteAttentionModel, self).__init__()
        self.model_name = "Attention-Complete"
        self.device = device

        # Choose pathway version based on dataset size
        if use_multiscale and dataset_size > 300:
            # Multi-scale for large datasets
            self.pathway = MultiScaleAttentionPathway(
                input_dim, output_dim=512, scales=[4, 8, 16]
            ).to(device)
        else:
            # Simple attention for smaller datasets
            num_heads = min(8, max(2, dataset_size // 50))  # Adaptive heads
            self.pathway = AttentionPathway(
                input_dim, output_dim=512, num_heads=num_heads
            ).to(device)

        # Decoder
        self.decoder = AttentionPathwayDecoder(input_dim=512, output_dim=784).to(device)
        
    def forward(self, x):
        """
        Complete forward pass
        
        Args:
            x: [batch_size, input_dim] - fMRI features
        Returns:
            visual_output: [batch_size, 1, 28, 28] - Reconstructed images
            pathway_features: [batch_size, 512] - Attention pathway features
            pathway_info: Dict with pathway information
        """
        # Encode through Attention pathway
        pathway_features, pathway_info = self.pathway(x)
        
        # Decode to visual output
        visual_output = self.decoder(pathway_features)
        
        return visual_output, pathway_features, pathway_info


# Factory functions for easy model creation
def create_attention_pathway(input_dim, output_dim=512, dataset_size=100, multiscale=True):
    """Factory function to create Attention pathway"""
    if multiscale and dataset_size > 300:
        return MultiScaleAttentionPathway(input_dim, output_dim)
    else:
        num_heads = min(8, max(2, dataset_size // 50))
        return AttentionPathway(input_dim, output_dim, num_heads)


def create_complete_attention_model(input_dim, dataset_size=100, multiscale=True, device='cuda'):
    """Factory function to create complete Attention model"""
    return CompleteAttentionModel(input_dim, dataset_size, multiscale, device)


# Pathway configuration for different datasets
def get_attention_pathway_config(dataset_name):
    """Get optimized configuration for Attention pathway based on dataset"""
    
    configs = {
        'miyawaki': {
            'output_dim': 512,
            'num_heads': 2,  # Small dataset, fewer heads
            'use_multiscale': False,
            'dropout': 0.15,
            'expected_performance': 'Moderate - may overfit on small data'
        },
        'vangerven': {
            'output_dim': 512,
            'num_heads': 2,  # Small dataset, fewer heads
            'use_multiscale': False,
            'dropout': 0.1,
            'expected_performance': 'Moderate - attention may be overkill'
        },
        'mindbigdata': {
            'output_dim': 512,
            'num_heads': 16,  # Large dataset, more heads
            'use_multiscale': True,
            'dropout': 0.05,
            'expected_performance': 'Excellent - proven good on large datasets'
        },
        'crell': {
            'output_dim': 512,
            'num_heads': 8,  # Medium dataset, balanced heads
            'use_multiscale': True,
            'dropout': 0.08,
            'expected_performance': 'Good - attention helps with complexity'
        }
    }
    
    return configs.get(dataset_name, configs['mindbigdata'])  # Default to large dataset config


# Export main classes and functions
__all__ = [
    'AttentionPathway',
    'MultiScaleAttentionPathway',
    'AttentionPathwayDecoder',
    'CompleteAttentionModel',
    'create_attention_pathway',
    'create_complete_attention_model',
    'get_attention_pathway_config'
]
