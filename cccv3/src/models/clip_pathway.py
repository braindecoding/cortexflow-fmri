"""
CCCV3 CLIP-Inspired Pathway
===========================

Recreates the proven CCCV1 CLIP-inspired architecture that showed strong performance
on Miyawaki and other datasets. This pathway focuses on semantic understanding
and CLIP-like feature alignment.

Key Features:
- CLIP-inspired semantic encoding
- LayerNorm + SiLU activations
- Semantic feature alignment
- Strong performance on small datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CLIPInspiredPathway(nn.Module):
    """
    CLIP-inspired pathway based on proven CCCV1 architecture
    
    Architecture Philosophy:
    - Semantic understanding through CLIP-like encoding
    - LayerNorm for stable training
    - SiLU activations for smooth gradients
    - Feature normalization for alignment
    """
    
    def __init__(self, input_dim, output_dim=512, clip_residual_weight=0.08):
        super(CLIPInspiredPathway, self).__init__()
        self.pathway_name = "CLIP-Inspired"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.clip_residual_weight = clip_residual_weight
        
        # CLIP-inspired encoder (based on CCCV1 success)
        self.encoder = nn.Sequential(
            # First layer: Input expansion
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.06),
            
            # Second layer: Feature extraction
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.04),
            
            # Third layer: Semantic encoding
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.03),
            
            # Final layer: CLIP-like features
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()  # CLIP-like normalization
        )
        
        # CLIP alignment module (semantic enhancement)
        self.clip_aligner = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.SiLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        """
        Forward pass through CLIP pathway
        
        Args:
            x: [batch_size, input_dim] - fMRI features
        Returns:
            features: [batch_size, output_dim] - CLIP-aligned features
            pathway_info: Dict with pathway-specific information
        """
        # Encode to CLIP-like space
        encoded_features = self.encoder(x)
        
        # CLIP alignment with residual connection
        aligned_features = encoded_features + self.clip_residual_weight * self.clip_aligner(encoded_features)
        
        # Normalize to unit sphere (CLIP-like)
        features = F.normalize(aligned_features, p=2, dim=1)
        
        # Pathway information for analysis
        pathway_info = {
            'pathway_name': self.pathway_name,
            'clip_residual_weight': self.clip_residual_weight,
            'feature_norm': torch.norm(features, dim=1).mean().item(),
            'alignment_strength': torch.norm(self.clip_aligner(encoded_features), dim=1).mean().item()
        }
        
        return features, pathway_info


class AdaptiveCLIPPathway(nn.Module):
    """
    Adaptive CLIP pathway with dataset-specific optimizations
    
    Improvements:
    - Dataset-aware dropout rates
    - Adaptive CLIP residual weighting
    - Enhanced semantic alignment
    """
    
    def __init__(self, input_dim, output_dim=512, dataset_size=100, dataset_complexity=0.5):
        super(AdaptiveCLIPPathway, self).__init__()
        self.pathway_name = "CLIP-Inspired-Adaptive"
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Adaptive parameters based on dataset characteristics
        if dataset_size < 100:
            # Small datasets: higher regularization
            self.dropout_rates = [0.08, 0.06, 0.04]
            self.clip_residual_weight = 0.1
        elif dataset_size > 500:
            # Large datasets: lower regularization
            self.dropout_rates = [0.04, 0.03, 0.02]
            self.clip_residual_weight = 0.06
        else:
            # Medium datasets: balanced
            self.dropout_rates = [0.06, 0.04, 0.03]
            self.clip_residual_weight = 0.08
        
        # Adaptive complexity based on dataset
        hidden_dim = int(1024 * (1 + dataset_complexity))
        
        # Enhanced encoder
        self.encoder = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rates[0]),
            
            # Hidden layer 1
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rates[1]),
            
            # Hidden layer 2
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(self.dropout_rates[2]),
            
            # Output layer
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
        # Enhanced CLIP alignment
        self.clip_aligner = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim // 2, output_dim),
            nn.Tanh()
        )
        
        # Semantic consistency module
        self.semantic_consistency = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass with adaptive CLIP processing
        """
        # Encode to CLIP-like space
        encoded_features = self.encoder(x)
        
        # CLIP alignment with adaptive weighting
        clip_enhancement = self.clip_aligner(encoded_features)
        aligned_features = encoded_features + self.clip_residual_weight * clip_enhancement
        
        # Semantic consistency enhancement
        consistency_features = self.semantic_consistency(aligned_features)
        final_features = aligned_features + 0.1 * consistency_features
        
        # Normalize to unit sphere
        features = F.normalize(final_features, p=2, dim=1)
        
        # Pathway information
        pathway_info = {
            'pathway_name': self.pathway_name,
            'dropout_rates': self.dropout_rates,
            'clip_residual_weight': self.clip_residual_weight,
            'feature_norm': torch.norm(features, dim=1).mean().item(),
            'alignment_strength': torch.norm(clip_enhancement, dim=1).mean().item(),
            'consistency_strength': torch.norm(consistency_features, dim=1).mean().item()
        }
        
        return features, pathway_info


class CLIPPathwayDecoder(nn.Module):
    """
    Decoder specifically designed for CLIP pathway features
    
    Maintains semantic understanding while decoding to visual output
    """
    
    def __init__(self, input_dim=512, output_dim=784):
        super(CLIPPathwayDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # CLIP-aware decoder
        self.decoder = nn.Sequential(
            # Semantic to visual transition
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Dropout(0.02),
            
            # Visual feature expansion
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()  # Visual output in [0, 1] range
        )
        
    def forward(self, features):
        """
        Decode CLIP pathway features to visual output
        
        Args:
            features: [batch_size, input_dim] - CLIP pathway features
        Returns:
            visual_output: [batch_size, 1, 28, 28] - Reconstructed images
        """
        # Decode features
        output = self.decoder(features)
        
        # Reshape to image format
        visual_output = output.view(-1, 1, 28, 28)
        
        return visual_output


class CompleteCLIPModel(nn.Module):
    """
    Complete CLIP-inspired model for standalone testing
    
    Combines CLIP pathway with appropriate decoder for end-to-end testing
    """
    
    def __init__(self, input_dim, dataset_size=100, dataset_complexity=0.5, use_adaptive=True, device='cuda'):
        super(CompleteCLIPModel, self).__init__()
        self.model_name = "CLIP-Inspired-Complete"
        self.device = device

        # Choose pathway version
        if use_adaptive:
            self.pathway = AdaptiveCLIPPathway(
                input_dim, output_dim=512,
                dataset_size=dataset_size,
                dataset_complexity=dataset_complexity
            ).to(device)
        else:
            self.pathway = CLIPInspiredPathway(input_dim, output_dim=512).to(device)

        # Decoder
        self.decoder = CLIPPathwayDecoder(input_dim=512, output_dim=784).to(device)
        
    def forward(self, x):
        """
        Complete forward pass
        
        Args:
            x: [batch_size, input_dim] - fMRI features
        Returns:
            visual_output: [batch_size, 1, 28, 28] - Reconstructed images
            pathway_features: [batch_size, 512] - CLIP pathway features
            pathway_info: Dict with pathway information
        """
        # Encode through CLIP pathway
        pathway_features, pathway_info = self.pathway(x)
        
        # Decode to visual output
        visual_output = self.decoder(pathway_features)
        
        return visual_output, pathway_features, pathway_info


# Factory functions for easy model creation
def create_clip_pathway(input_dim, output_dim=512, dataset_size=100, adaptive=True):
    """Factory function to create CLIP pathway"""
    if adaptive:
        return AdaptiveCLIPPathway(input_dim, output_dim, dataset_size)
    else:
        return CLIPInspiredPathway(input_dim, output_dim)


def create_complete_clip_model(input_dim, dataset_size=100, dataset_complexity=0.5, adaptive=True, device='cuda'):
    """Factory function to create complete CLIP model"""
    return CompleteCLIPModel(input_dim, dataset_size, dataset_complexity, adaptive, device)


# Pathway configuration for different datasets
def get_clip_pathway_config(dataset_name):
    """Get optimized configuration for CLIP pathway based on dataset"""
    
    configs = {
        'miyawaki': {
            'output_dim': 512,
            'clip_residual_weight': 0.1,
            'dataset_complexity': 0.8,  # High complexity for small dataset
            'use_adaptive': True,
            'expected_performance': 'Excellent - proven strong on Miyawaki'
        },
        'vangerven': {
            'output_dim': 512,
            'clip_residual_weight': 0.08,
            'dataset_complexity': 0.6,  # Medium complexity
            'use_adaptive': True,
            'expected_performance': 'Good - semantic understanding helps'
        },
        'mindbigdata': {
            'output_dim': 512,
            'clip_residual_weight': 0.06,
            'dataset_complexity': 0.4,  # Lower complexity for large dataset
            'use_adaptive': True,
            'expected_performance': 'Good - semantic features scale well'
        },
        'crell': {
            'output_dim': 512,
            'clip_residual_weight': 0.08,
            'dataset_complexity': 0.5,  # Balanced complexity
            'use_adaptive': True,
            'expected_performance': 'Good - balanced semantic approach'
        }
    }
    
    return configs.get(dataset_name, configs['miyawaki'])  # Default to Miyawaki config


# Export main classes and functions
__all__ = [
    'CLIPInspiredPathway',
    'AdaptiveCLIPPathway',
    'CLIPPathwayDecoder', 
    'CompleteCLIPModel',
    'create_clip_pathway',
    'create_complete_clip_model',
    'get_clip_pathway_config'
]
