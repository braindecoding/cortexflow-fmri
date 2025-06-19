"""
CCCV3 Adaptive Fusion Module
============================

Smart fusion of multiple pathways based on dataset characteristics and
pathway strengths. This module learns to optimally combine:
- CortexFlow-Lite pathway (simple & robust)
- CLIP-inspired pathway (semantic understanding)
- Attention pathway (feature relationships)

Key Features:
- Dataset-aware pathway weighting
- Performance-guided adaptation
- Dynamic fusion strategies
- Pathway contribution analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveMultiPathwayFusion(nn.Module):
    """
    Adaptive fusion module for combining multiple pathways
    
    Strategy:
    - Learn optimal pathway weights for each dataset
    - Adapt based on dataset characteristics (size, complexity)
    - Dynamic weighting during inference
    - Performance feedback integration
    """
    
    def __init__(self, pathway_dims=[128, 512, 512], output_dim=512, num_datasets=4):
        super(AdaptiveMultiPathwayFusion, self).__init__()
        self.pathway_dims = pathway_dims  # [Lite, CLIP, Attention]
        self.output_dim = output_dim
        self.num_datasets = num_datasets
        self.num_pathways = len(pathway_dims)
        
        # Project all pathways to common dimension
        self.pathway_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.SiLU()
            ) for dim in pathway_dims
        ])
        
        # Dataset embedding for adaptive weights
        self.dataset_embedding = nn.Embedding(num_datasets, output_dim)
        
        # Pathway weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(output_dim + 3, output_dim),  # +3 for dataset stats
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(output_dim, self.num_pathways),
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
        
        # Pathway interaction module
        self.interaction_module = nn.Sequential(
            nn.Linear(output_dim * self.num_pathways, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
        # Performance memory for adaptive learning
        self.register_buffer('performance_memory', torch.zeros(num_datasets, self.num_pathways))
        self.register_buffer('update_count', torch.zeros(num_datasets))
        
    def forward(self, pathway_features, dataset_id=None, dataset_stats=None):
        """
        Adaptive fusion of pathway features
        
        Args:
            pathway_features: List of [batch_size, pathway_dim] tensors
                             [lite_features, clip_features, attention_features]
            dataset_id: Dataset identifier for adaptive weighting
            dataset_stats: [batch_size, 3] dataset characteristics
                          [size_norm, complexity, performance_history]
        Returns:
            fused_features: [batch_size, output_dim] fused features
            fusion_info: Dict with fusion analysis information
        """
        batch_size = pathway_features[0].shape[0]
        
        # Project all pathways to common dimension
        projected_features = []
        for features, projection in zip(pathway_features, self.pathway_projections):
            projected = projection(features)
            projected_features.append(projected)
        
        # Adaptive pathway weighting
        if dataset_id is not None and dataset_stats is not None:
            # Get dataset embedding
            dataset_emb = self.dataset_embedding(dataset_id)
            if dataset_emb.dim() == 3:
                dataset_emb = dataset_emb.squeeze(0)
            dataset_emb = dataset_emb.expand(batch_size, -1)
            
            # Combine with dataset statistics
            fusion_input = torch.cat([dataset_emb, dataset_stats], dim=-1)
            pathway_weights = self.weight_predictor(fusion_input)
        else:
            # Default uniform weights
            pathway_weights = torch.ones(batch_size, self.num_pathways, 
                                       device=pathway_features[0].device)
            pathway_weights = pathway_weights / self.num_pathways
        
        # Weighted fusion
        weighted_features = []
        for i, features in enumerate(projected_features):
            weight = pathway_weights[:, i:i+1]  # [batch_size, 1]
            weighted = weight * features
            weighted_features.append(weighted)
        
        # Concatenate for interaction
        concatenated = torch.cat(weighted_features, dim=-1)
        
        # Pathway interaction
        fused_features = self.interaction_module(concatenated)
        
        # Fusion information for analysis
        fusion_info = {
            'pathway_weights': pathway_weights.detach().cpu(),
            'pathway_contributions': [torch.norm(wf, dim=1).mean().item() for wf in weighted_features],
            'fusion_strength': torch.norm(fused_features, dim=1).mean().item(),
            'weight_entropy': self._compute_weight_entropy(pathway_weights)
        }
        
        return fused_features, fusion_info
    
    def _compute_weight_entropy(self, weights):
        """Compute entropy of pathway weights for analysis"""
        # weights: [batch_size, num_pathways]
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        return entropy.item()
    
    def update_performance_memory(self, dataset_id, pathway_performances):
        """
        Update performance memory for adaptive learning
        
        Args:
            dataset_id: Dataset identifier (0-3)
            pathway_performances: [num_pathways] performance scores
        """
        with torch.no_grad():
            alpha = 0.1  # Learning rate for memory update
            
            # Convert to tensor if needed
            if not isinstance(pathway_performances, torch.Tensor):
                pathway_performances = torch.tensor(pathway_performances, 
                                                  device=self.performance_memory.device)
            
            # Exponential moving average update
            self.performance_memory[dataset_id] = (
                alpha * pathway_performances + 
                (1 - alpha) * self.performance_memory[dataset_id]
            )
            
            self.update_count[dataset_id] += 1
    
    def get_pathway_preferences(self, dataset_id):
        """Get learned pathway preferences for a dataset"""
        if dataset_id < self.num_datasets:
            preferences = self.performance_memory[dataset_id]
            return F.softmax(preferences, dim=0).cpu().numpy()
        else:
            return np.ones(self.num_pathways) / self.num_pathways


class StrategicFusionModule(nn.Module):
    """
    Strategic fusion module with predefined strategies for different scenarios
    
    Strategies:
    - Small datasets: Favor Lite + CLIP pathways
    - Large datasets: Favor Attention + CLIP pathways
    - Balanced datasets: Equal weighting with learned adjustments
    """
    
    def __init__(self, pathway_dims=[128, 512, 512], output_dim=512):
        super(StrategicFusionModule, self).__init__()
        self.pathway_dims = pathway_dims
        self.output_dim = output_dim
        self.num_pathways = len(pathway_dims)
        
        # Project all pathways to common dimension
        self.pathway_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.SiLU()
            ) for dim in pathway_dims
        ])
        
        # Strategy-specific fusion networks
        self.small_dataset_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),  # Lite + CLIP
            nn.LayerNorm(output_dim),
            nn.SiLU()
        )
        
        self.large_dataset_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),  # Attention + CLIP
            nn.LayerNorm(output_dim),
            nn.SiLU()
        )
        
        self.balanced_fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),  # All pathways
            nn.LayerNorm(output_dim),
            nn.SiLU()
        )
        
        # Strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(3, 16),  # Dataset stats -> hidden
            nn.SiLU(),
            nn.Linear(16, 3),  # Hidden -> strategy weights
            nn.Softmax(dim=-1)
        )
        
    def forward(self, pathway_features, dataset_stats=None):
        """
        Strategic fusion based on dataset characteristics
        
        Args:
            pathway_features: [lite_features, clip_features, attention_features]
            dataset_stats: [batch_size, 3] dataset characteristics
        Returns:
            fused_features: [batch_size, output_dim]
            fusion_info: Dict with strategy information
        """
        batch_size = pathway_features[0].shape[0]
        
        # Project all pathways
        lite_proj, clip_proj, attn_proj = [
            proj(feat) for proj, feat in zip(self.pathway_projections, pathway_features)
        ]
        
        if dataset_stats is not None:
            # Determine fusion strategy
            strategy_weights = self.strategy_selector(dataset_stats)  # [batch_size, 3]
            
            # Apply different fusion strategies
            small_fused = self.small_dataset_fusion(torch.cat([lite_proj, clip_proj], dim=-1))
            large_fused = self.large_dataset_fusion(torch.cat([attn_proj, clip_proj], dim=-1))
            balanced_fused = self.balanced_fusion(torch.cat([lite_proj, clip_proj, attn_proj], dim=-1))
            
            # Weighted combination of strategies
            fused_features = (strategy_weights[:, 0:1] * small_fused +
                            strategy_weights[:, 1:2] * large_fused +
                            strategy_weights[:, 2:3] * balanced_fused)
            
            strategy_info = strategy_weights.detach().cpu().numpy()
        else:
            # Default balanced fusion
            fused_features = self.balanced_fusion(torch.cat([lite_proj, clip_proj, attn_proj], dim=-1))
            strategy_info = np.array([[1/3, 1/3, 1/3]] * batch_size)
        
        fusion_info = {
            'strategy_weights': strategy_info,
            'fusion_strategy': 'strategic',
            'fusion_strength': torch.norm(fused_features, dim=1).mean().item()
        }
        
        return fused_features, fusion_info


class UnifiedFusionDecoder(nn.Module):
    """
    Unified decoder for fused pathway features
    
    Designed to work optimally with features from multiple pathways
    """
    
    def __init__(self, input_dim=512, output_dim=784):
        super(UnifiedFusionDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Multi-pathway aware decoder
        self.decoder = nn.Sequential(
            # Feature integration
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Dropout(0.02),
            
            # Visual feature generation
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.01),
            
            # Final output
            nn.Linear(input_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, fused_features):
        """
        Decode fused pathway features to visual output
        
        Args:
            fused_features: [batch_size, input_dim] - Fused pathway features
        Returns:
            visual_output: [batch_size, 1, 28, 28] - Reconstructed images
        """
        # Decode features
        output = self.decoder(fused_features)
        
        # Reshape to image format
        visual_output = output.view(-1, 1, 28, 28)
        
        return visual_output


# Factory functions
def create_adaptive_fusion(pathway_dims=[128, 512, 512], output_dim=512, num_datasets=4):
    """Factory function to create adaptive fusion module"""
    return AdaptiveMultiPathwayFusion(pathway_dims, output_dim, num_datasets)


def create_strategic_fusion(pathway_dims=[128, 512, 512], output_dim=512):
    """Factory function to create strategic fusion module"""
    return StrategicFusionModule(pathway_dims, output_dim)


def create_unified_decoder(input_dim=512, output_dim=784):
    """Factory function to create unified decoder"""
    return UnifiedFusionDecoder(input_dim, output_dim)


# Fusion strategy configuration
def get_fusion_strategy_config(dataset_name):
    """Get optimal fusion strategy for each dataset"""
    
    configs = {
        'miyawaki': {
            'strategy': 'small_dataset',
            'primary_pathways': ['lite', 'clip'],
            'expected_weights': [0.4, 0.6, 0.0],  # Lite, CLIP, Attention
            'rationale': 'Small dataset benefits from simple + semantic'
        },
        'vangerven': {
            'strategy': 'small_dataset',
            'primary_pathways': ['lite', 'clip'],
            'expected_weights': [0.7, 0.3, 0.0],  # Favor Lite (proven winner)
            'rationale': 'Lite pathway proven winner, CLIP for enhancement'
        },
        'mindbigdata': {
            'strategy': 'large_dataset',
            'primary_pathways': ['attention', 'clip'],
            'expected_weights': [0.1, 0.4, 0.5],  # Favor Attention + CLIP
            'rationale': 'Large dataset benefits from attention + semantics'
        },
        'crell': {
            'strategy': 'balanced',
            'primary_pathways': ['lite', 'clip', 'attention'],
            'expected_weights': [0.3, 0.4, 0.3],  # Balanced approach
            'rationale': 'Medium dataset benefits from all pathways'
        }
    }
    
    return configs.get(dataset_name, configs['crell'])  # Default to balanced


# Export main classes and functions
__all__ = [
    'AdaptiveMultiPathwayFusion',
    'StrategicFusionModule',
    'UnifiedFusionDecoder',
    'create_adaptive_fusion',
    'create_strategic_fusion',
    'create_unified_decoder',
    'get_fusion_strategy_config'
]
