"""
CCCV3 CortexFlow-Lite Pathway
============================

Recreates the proven CortexFlow-Lite architecture that won on Vangerven dataset.
This pathway focuses on simplicity and robustness for small-medium datasets.

Key Features:
- Simple ReLU-based architecture
- Moderate dropout for regularization
- Proven effectiveness on Vangerven dataset
- Fast training and inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CortexFlowLitePathway(nn.Module):
    """
    CortexFlow-Lite pathway - proven winner on Vangerven dataset
    
    Architecture Philosophy:
    - Simplicity over complexity
    - ReLU activations for stability
    - Moderate dropout for regularization
    - Direct feature mapping without fancy mechanisms
    """
    
    def __init__(self, input_dim, output_dim=128, dropout_rate=0.3):
        super(CortexFlowLitePathway, self).__init__()
        self.pathway_name = "CortexFlow-Lite"
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple but effective encoder
        self.encoder = nn.Sequential(
            # First layer: Input to hidden
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second layer: Feature extraction
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),  # Reduced dropout in deeper layers
            
            # Third layer: Feature refinement
            nn.Linear(256, output_dim),
            nn.ReLU()
        )
        
        # Pathway-specific normalization
        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Forward pass through Lite pathway
        
        Args:
            x: [batch_size, input_dim] - fMRI features
        Returns:
            features: [batch_size, output_dim] - Lite pathway features
            pathway_info: Dict with pathway-specific information
        """
        # Simple encoding
        features = self.encoder(x)
        
        # Normalize output
        features = self.output_norm(features)
        
        # Pathway information for analysis
        pathway_info = {
            'pathway_name': self.pathway_name,
            'feature_norm': torch.norm(features, dim=1).mean().item(),
            'activation_sparsity': (features > 0).float().mean().item()
        }
        
        return features, pathway_info


class OptimizedLitePathway(nn.Module):
    """
    Optimized version of CortexFlow-Lite with additional improvements
    
    Improvements:
    - Batch normalization for better training
    - Residual connections for gradient flow
    - Adaptive dropout based on dataset size
    """
    
    def __init__(self, input_dim, output_dim=128, dataset_size=100):
        super(OptimizedLitePathway, self).__init__()
        self.pathway_name = "CortexFlow-Lite-Optimized"
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Adaptive dropout based on dataset size
        base_dropout = 0.3
        if dataset_size < 100:
            self.dropout_rate = base_dropout * 1.5  # Higher dropout for small datasets
        elif dataset_size > 500:
            self.dropout_rate = base_dropout * 0.7  # Lower dropout for large datasets
        else:
            self.dropout_rate = base_dropout
        
        # Encoder with residual connections
        self.input_layer = nn.Linear(input_dim, 512)
        self.input_bn = nn.BatchNorm1d(512)
        
        self.hidden_layer1 = nn.Linear(512, 512)
        self.hidden_bn1 = nn.BatchNorm1d(512)
        
        self.hidden_layer2 = nn.Linear(512, 256)
        self.hidden_bn2 = nn.BatchNorm1d(256)
        
        self.output_layer = nn.Linear(256, output_dim)
        self.output_bn = nn.BatchNorm1d(output_dim)
        
        # Residual projection layers
        self.residual_proj1 = nn.Linear(512, 512)
        self.residual_proj2 = nn.Linear(512, 256)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        """
        Forward pass with residual connections and batch normalization
        """
        # Input layer
        x1 = F.relu(self.input_bn(self.input_layer(x)))
        x1 = self.dropout(x1)
        
        # First hidden layer with residual
        x2 = self.hidden_layer1(x1)
        x2 = self.hidden_bn1(x2)
        x2 = F.relu(x2 + self.residual_proj1(x1))  # Residual connection
        x2 = self.dropout(x2)
        
        # Second hidden layer with residual
        x3 = self.hidden_layer2(x2)
        x3 = self.hidden_bn2(x3)
        x3 = F.relu(x3 + self.residual_proj2(x2))  # Residual connection
        x3 = self.dropout(x3)
        
        # Output layer
        features = self.output_layer(x3)
        features = self.output_bn(features)
        features = F.relu(features)
        
        # Pathway information
        pathway_info = {
            'pathway_name': self.pathway_name,
            'dropout_rate': self.dropout_rate,
            'feature_norm': torch.norm(features, dim=1).mean().item(),
            'activation_sparsity': (features > 0).float().mean().item()
        }
        
        return features, pathway_info


class LitePathwayDecoder(nn.Module):
    """
    Decoder specifically designed for Lite pathway features
    
    Matches the simplicity of the Lite pathway while maintaining effectiveness
    """
    
    def __init__(self, input_dim=128, output_dim=784):
        super(LitePathwayDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple decoder matching Lite philosophy
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
    def forward(self, features):
        """
        Decode Lite pathway features to visual output
        
        Args:
            features: [batch_size, input_dim] - Lite pathway features
        Returns:
            visual_output: [batch_size, 1, 28, 28] - Reconstructed images
        """
        # Decode features
        output = self.decoder(features)
        
        # Reshape to image format
        visual_output = output.view(-1, 1, 28, 28)
        
        return visual_output


class CompleteLiteModel(nn.Module):
    """
    Complete CortexFlow-Lite model for standalone testing
    
    Combines Lite pathway with appropriate decoder for end-to-end testing
    """
    
    def __init__(self, input_dim, dataset_size=100, use_optimized=True, device='cuda'):
        super(CompleteLiteModel, self).__init__()
        self.model_name = "CortexFlow-Lite-Complete"
        self.device = device

        # Choose pathway version
        if use_optimized:
            self.pathway = OptimizedLitePathway(input_dim, output_dim=128, dataset_size=dataset_size).to(device)
        else:
            self.pathway = CortexFlowLitePathway(input_dim, output_dim=128).to(device)

        # Decoder
        self.decoder = LitePathwayDecoder(input_dim=128, output_dim=784).to(device)
        
    def forward(self, x):
        """
        Complete forward pass
        
        Args:
            x: [batch_size, input_dim] - fMRI features
        Returns:
            visual_output: [batch_size, 1, 28, 28] - Reconstructed images
            pathway_features: [batch_size, 128] - Lite pathway features
            pathway_info: Dict with pathway information
        """
        # Encode through Lite pathway
        pathway_features, pathway_info = self.pathway(x)
        
        # Decode to visual output
        visual_output = self.decoder(pathway_features)
        
        return visual_output, pathway_features, pathway_info


# Factory functions for easy model creation
def create_lite_pathway(input_dim, output_dim=128, dataset_size=100, optimized=True):
    """Factory function to create Lite pathway"""
    if optimized:
        return OptimizedLitePathway(input_dim, output_dim, dataset_size)
    else:
        return CortexFlowLitePathway(input_dim, output_dim)


def create_complete_lite_model(input_dim, dataset_size=100, optimized=True, device='cuda'):
    """Factory function to create complete Lite model"""
    return CompleteLiteModel(input_dim, dataset_size, optimized, device)


# Pathway configuration for different datasets
def get_lite_pathway_config(dataset_name):
    """Get optimized configuration for Lite pathway based on dataset"""
    
    configs = {
        'miyawaki': {
            'output_dim': 128,
            'dropout_rate': 0.4,  # Higher dropout for small dataset
            'use_optimized': True,
            'expected_performance': 'Good - simple architecture works well'
        },
        'vangerven': {
            'output_dim': 128,
            'dropout_rate': 0.3,  # Proven optimal for Vangerven
            'use_optimized': True,
            'expected_performance': 'Excellent - proven winner'
        },
        'mindbigdata': {
            'output_dim': 128,
            'dropout_rate': 0.2,  # Lower dropout for large dataset
            'use_optimized': True,
            'expected_performance': 'Moderate - may need more complexity'
        },
        'crell': {
            'output_dim': 128,
            'dropout_rate': 0.25, # Balanced for medium dataset
            'use_optimized': True,
            'expected_performance': 'Good - balanced approach'
        }
    }
    
    return configs.get(dataset_name, configs['vangerven'])  # Default to Vangergen config


# Export main classes and functions
__all__ = [
    'CortexFlowLitePathway',
    'OptimizedLitePathway', 
    'LitePathwayDecoder',
    'CompleteLiteModel',
    'create_lite_pathway',
    'create_complete_lite_model',
    'get_lite_pathway_config'
]
