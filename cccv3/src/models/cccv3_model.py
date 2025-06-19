"""
CCCV3 Complete Multi-Pathway Model
==================================

Complete CortexFlow-CLIP-CNN V3 model that combines all three pathways:
1. CortexFlow-Lite pathway (simple & robust)
2. CLIP-inspired pathway (semantic understanding)
3. Attention pathway (feature relationships)

With adaptive fusion for optimal performance across all datasets.

Goal: 100% success rate with 5-15% improvements over previous versions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import pathway modules
try:
    from .lite_pathway import create_lite_pathway, get_lite_pathway_config
    from .clip_pathway import create_clip_pathway, get_clip_pathway_config
    from .attention_pathway import create_attention_pathway, get_attention_pathway_config
    from .fusion_module import create_adaptive_fusion, create_unified_decoder, get_fusion_strategy_config
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from lite_pathway import create_lite_pathway, get_lite_pathway_config
    from clip_pathway import create_clip_pathway, get_clip_pathway_config
    from attention_pathway import create_attention_pathway, get_attention_pathway_config
    from fusion_module import create_adaptive_fusion, create_unified_decoder, get_fusion_strategy_config


class CCCV3MultiPathwayModel(nn.Module):
    """
    Complete CCCV3 Multi-Pathway Neural Decoding Model
    
    Architecture:
    Input fMRI → [Lite Pathway] → Features 1 (128D)
               → [CLIP Pathway] → Features 2 (512D) → Adaptive → Fused → Visual
               → [Attention Pathway] → Features 3 (512D)    Fusion   Features  Output
    
    Innovation:
    - Combines strengths of all previous architectures
    - Adaptive fusion based on dataset characteristics
    - Optimal pathway weighting for each scenario
    """
    
    def __init__(self, input_dim, dataset_name='unknown', dataset_size=100, device='cuda'):
        super(CCCV3MultiPathwayModel, self).__init__()
        self.model_name = "CortexFlow-CLIP-CNN-V3-MultiPathway"
        self.input_dim = input_dim
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
        self.device = device
        
        # Get pathway configurations
        lite_config = get_lite_pathway_config(dataset_name)
        clip_config = get_clip_pathway_config(dataset_name)
        attention_config = get_attention_pathway_config(dataset_name)
        fusion_config = get_fusion_strategy_config(dataset_name)
        
        print(f"   🏗️ CCCV3 Architecture for {dataset_name.upper()}:")
        print(f"      Lite pathway: {lite_config['expected_performance']}")
        print(f"      CLIP pathway: {clip_config['expected_performance']}")
        print(f"      Attention pathway: {attention_config['expected_performance']}")
        print(f"      Fusion strategy: {fusion_config['strategy']}")
        
        # Create individual pathways
        self.lite_pathway = create_lite_pathway(
            input_dim, 
            output_dim=128, 
            dataset_size=dataset_size,
            optimized=True
        ).to(device)
        
        self.clip_pathway = create_clip_pathway(
            input_dim,
            output_dim=512,
            dataset_size=dataset_size,
            adaptive=True
        ).to(device)
        
        self.attention_pathway = create_attention_pathway(
            input_dim,
            output_dim=512,
            dataset_size=dataset_size,
            multiscale=(dataset_size > 300)
        ).to(device)
        
        # Adaptive fusion module
        self.fusion_module = create_adaptive_fusion(
            pathway_dims=[128, 512, 512],  # [Lite, CLIP, Attention]
            output_dim=512,
            num_datasets=4
        ).to(device)
        
        # Unified decoder
        self.decoder = create_unified_decoder(
            input_dim=512,
            output_dim=784
        ).to(device)
        
        # Dataset mapping for fusion
        self.dataset_mapping = {
            'miyawaki': 0,
            'vangerven': 1, 
            'mindbigdata': 2,
            'crell': 3
        }
        
        # Store expected fusion weights for analysis
        self.expected_fusion_weights = fusion_config['expected_weights']
        
    def forward(self, x, return_pathway_info=False):
        """
        Complete multi-pathway forward pass
        
        Args:
            x: [batch_size, input_dim] - fMRI features
            return_pathway_info: Whether to return detailed pathway information
        Returns:
            visual_output: [batch_size, 1, 28, 28] - Reconstructed images
            fused_features: [batch_size, 512] - Fused pathway features
            pathway_info: Dict with detailed pathway and fusion information (if requested)
        """
        batch_size = x.shape[0]
        
        # Process through all pathways
        lite_features, lite_info = self.lite_pathway(x)
        clip_features, clip_info = self.clip_pathway(x)
        attention_features, attention_info = self.attention_pathway(x)
        
        # Prepare dataset information for fusion
        dataset_id = torch.tensor([self.dataset_mapping.get(self.dataset_name, 0)], 
                                device=x.device)
        
        # Dataset statistics: [size_normalized, complexity, performance_history]
        size_norm = min(1.0, self.dataset_size / 1000.0)  # Normalize to [0, 1]
        complexity = 0.5  # Default complexity, can be learned
        performance_history = 0.5  # Default, updated during training
        
        dataset_stats = torch.tensor([[size_norm, complexity, performance_history]], 
                                   device=x.device).repeat(batch_size, 1)
        
        # Adaptive fusion
        fused_features, fusion_info = self.fusion_module(
            [lite_features, clip_features, attention_features],
            dataset_id=dataset_id,
            dataset_stats=dataset_stats
        )
        
        # Decode to visual output
        visual_output = self.decoder(fused_features)
        
        if return_pathway_info:
            pathway_info = {
                'lite_info': lite_info,
                'clip_info': clip_info,
                'attention_info': attention_info,
                'fusion_info': fusion_info,
                'expected_weights': self.expected_fusion_weights,
                'actual_weights': fusion_info['pathway_weights'].mean(dim=0).numpy(),
                'pathway_contributions': fusion_info['pathway_contributions']
            }
            return visual_output, fused_features, pathway_info
        else:
            return visual_output, fused_features
    
    def get_pathway_analysis(self, x):
        """
        Detailed analysis of pathway contributions
        
        Args:
            x: [batch_size, input_dim] - fMRI features
        Returns:
            analysis: Dict with comprehensive pathway analysis
        """
        with torch.no_grad():
            visual_output, fused_features, pathway_info = self.forward(x, return_pathway_info=True)
            
            analysis = {
                'model_name': self.model_name,
                'dataset_name': self.dataset_name,
                'dataset_size': self.dataset_size,
                'pathway_weights': {
                    'expected': self.expected_fusion_weights,
                    'actual': pathway_info['actual_weights'],
                    'difference': pathway_info['actual_weights'] - np.array(self.expected_fusion_weights)
                },
                'pathway_contributions': {
                    'lite': pathway_info['pathway_contributions'][0],
                    'clip': pathway_info['pathway_contributions'][1],
                    'attention': pathway_info['pathway_contributions'][2]
                },
                'fusion_analysis': {
                    'weight_entropy': pathway_info['fusion_info']['weight_entropy'],
                    'fusion_strength': pathway_info['fusion_info']['fusion_strength']
                },
                'individual_pathway_info': {
                    'lite': pathway_info['lite_info'],
                    'clip': pathway_info['clip_info'],
                    'attention': pathway_info['attention_info']
                }
            }
            
            return analysis
    
    def update_performance_feedback(self, performance_scores):
        """
        Update fusion module with performance feedback
        
        Args:
            performance_scores: [3] - Performance scores for [Lite, CLIP, Attention]
        """
        dataset_id = self.dataset_mapping.get(self.dataset_name, 0)
        self.fusion_module.update_performance_memory(dataset_id, performance_scores)


class CCCV3Trainer:
    """
    Specialized trainer for CCCV3 multi-pathway model
    
    Features:
    - Individual pathway loss monitoring
    - Adaptive fusion weight tracking
    - Performance feedback integration
    """
    
    def __init__(self, model, device, pathway_loss_weights=[0.2, 0.3, 0.5]):
        self.model = model
        self.device = device
        self.pathway_loss_weights = pathway_loss_weights  # [Lite, CLIP, Attention]
        
    def compute_multi_pathway_loss(self, visual_output, targets, pathway_info=None):
        """
        Compute loss with pathway-specific components
        
        Args:
            visual_output: [batch_size, 1, 28, 28] - Model output
            targets: [batch_size, 1, 28, 28] - Target images
            pathway_info: Dict with pathway information
        Returns:
            total_loss: Combined loss
            loss_components: Dict with individual loss components
        """
        # Main reconstruction loss
        reconstruction_loss = F.mse_loss(visual_output, targets)
        
        # Pathway-specific losses (if pathway info available)
        pathway_losses = [0.0, 0.0, 0.0]
        
        if pathway_info is not None:
            # Fusion weight regularization (encourage expected weights)
            expected_weights = torch.tensor(self.model.expected_fusion_weights,
                                          device=self.device)
            actual_weights = pathway_info['fusion_info']['pathway_weights'].mean(dim=0).to(self.device)

            weight_regularization = F.mse_loss(actual_weights, expected_weights)
            
            # Weight entropy regularization (prevent over-concentration)
            weight_entropy = pathway_info['fusion_info']['weight_entropy']
            entropy_regularization = -0.1 * weight_entropy  # Encourage diversity
            
            total_loss = (reconstruction_loss + 
                         0.01 * weight_regularization + 
                         0.001 * entropy_regularization)
        else:
            total_loss = reconstruction_loss
            weight_regularization = 0.0
            entropy_regularization = 0.0
        
        loss_components = {
            'reconstruction_loss': reconstruction_loss.item(),
            'weight_regularization': weight_regularization.item() if isinstance(weight_regularization, torch.Tensor) else weight_regularization,
            'entropy_regularization': entropy_regularization.item() if isinstance(entropy_regularization, torch.Tensor) else entropy_regularization,
            'total_loss': total_loss.item(),
            'pathway_losses': pathway_losses
        }
        
        return total_loss, loss_components


# Factory function for easy model creation
def create_cccv3_model(input_dim, dataset_name='unknown', dataset_size=100, device='cuda'):
    """Factory function to create CCCV3 multi-pathway model"""
    return CCCV3MultiPathwayModel(input_dim, dataset_name, dataset_size, device)


def create_cccv3_trainer(model, device, pathway_weights=[0.2, 0.3, 0.5]):
    """Factory function to create CCCV3 trainer"""
    return CCCV3Trainer(model, device, pathway_weights)


# Model configuration for different datasets
def get_cccv3_config(dataset_name):
    """Get complete CCCV3 configuration for a dataset"""
    
    configs = {
        'miyawaki': {
            'expected_improvement': '15-20%',
            'primary_strategy': 'Lite + CLIP dominance',
            'training_epochs': 120,
            'learning_rate': 0.0003,
            'pathway_weights': [0.4, 0.6, 0.0]
        },
        'vangerven': {
            'expected_improvement': '10-15%', 
            'primary_strategy': 'Lite pathway dominance',
            'training_epochs': 100,
            'learning_rate': 0.0005,
            'pathway_weights': [0.7, 0.3, 0.0]
        },
        'mindbigdata': {
            'expected_improvement': '5-10%',
            'primary_strategy': 'Attention + CLIP dominance',
            'training_epochs': 80,
            'learning_rate': 0.001,
            'pathway_weights': [0.1, 0.4, 0.5]
        },
        'crell': {
            'expected_improvement': '8-12%',
            'primary_strategy': 'Balanced multi-pathway',
            'training_epochs': 100,
            'learning_rate': 0.0007,
            'pathway_weights': [0.3, 0.4, 0.3]
        }
    }
    
    return configs.get(dataset_name, configs['crell'])


# Export main classes and functions
__all__ = [
    'CCCV3MultiPathwayModel',
    'CCCV3Trainer',
    'create_cccv3_model',
    'create_cccv3_trainer',
    'get_cccv3_config'
]
