"""
CortexFlow-CLIP-CNN V2: Hierarchical Feature Fusion
===================================================

Revolutionary hierarchical processing for neural decoding:
1. Multi-Resolution Feature Extraction
2. Feature Pyramid Networks (FPN) for neural signals
3. Adaptive Fusion based on dataset characteristics
4. Coarse-to-Fine Processing Pipeline

Innovation: First hierarchical multi-scale neural decoding
Goal: Capture both global semantics and local details
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiResolutionEncoder(nn.Module):
    """
    Multi-resolution feature extraction at different scales
    
    Innovation:
    - Parallel processing at multiple resolutions
    - Captures both global and local patterns
    - Adaptive pooling for different input sizes
    """
    
    def __init__(self, input_dim, embed_dim=512, num_scales=4):
        super(MultiResolutionEncoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        
        # Multi-scale encoders
        self.scale_encoders = nn.ModuleList()
        
        for i in range(num_scales):
            # Different compression ratios for different scales
            scale_dim = embed_dim // (2 ** i)  # 512, 256, 128, 64
            
            encoder = nn.Sequential(
                nn.Linear(input_dim, scale_dim * 2),
                nn.LayerNorm(scale_dim * 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                
                nn.Linear(scale_dim * 2, scale_dim),
                nn.LayerNorm(scale_dim),
                nn.SiLU(),
                nn.Dropout(0.05)
            )
            
            self.scale_encoders.append(encoder)
        
        # Scale-specific attention
        self.scale_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim // (2 ** i), num_heads=max(1, 8 // (2 ** i)), 
                                batch_first=True, dropout=0.1)
            for i in range(num_scales)
        ])
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            multi_scale_features: List of [batch_size, scale_dim] tensors
        """
        multi_scale_features = []
        
        for i, (encoder, attention) in enumerate(zip(self.scale_encoders, self.scale_attention)):
            # Extract features at this scale
            scale_features = encoder(x)  # [batch_size, scale_dim]
            
            # Apply self-attention at this scale
            scale_features_seq = scale_features.unsqueeze(1)  # [batch_size, 1, scale_dim]
            attended_features, _ = attention(scale_features_seq, scale_features_seq, scale_features_seq)
            attended_features = attended_features.squeeze(1)  # [batch_size, scale_dim]
            
            multi_scale_features.append(attended_features)
        
        return multi_scale_features


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network adapted for neural decoding
    
    Innovation:
    - Bottom-up pathway: detailed to semantic features
    - Top-down pathway: semantic to detailed features
    - Lateral connections: feature fusion at each level
    """
    
    def __init__(self, input_dim, embed_dim=512, num_levels=4):
        super(FeaturePyramidNetwork, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        
        # Bottom-up pathway (detailed -> semantic)
        self.bottom_up = nn.ModuleList()
        current_dim = input_dim
        
        for i in range(num_levels):
            level_dim = embed_dim // (2 ** (num_levels - 1 - i))  # 64, 128, 256, 512
            
            bottom_up_layer = nn.Sequential(
                nn.Linear(current_dim, level_dim),
                nn.LayerNorm(level_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            )
            
            self.bottom_up.append(bottom_up_layer)
            current_dim = level_dim
        
        # Top-down pathway (semantic -> detailed)
        self.top_down = nn.ModuleList()
        
        for i in range(num_levels - 1):
            higher_dim = embed_dim // (2 ** (i))      # 512, 256, 128
            lower_dim = embed_dim // (2 ** (i + 1))   # 256, 128, 64
            
            top_down_layer = nn.Sequential(
                nn.Linear(higher_dim, lower_dim),
                nn.LayerNorm(lower_dim),
                nn.SiLU()
            )
            
            self.top_down.append(top_down_layer)
        
        # Lateral connections for fusion
        self.lateral_connections = nn.ModuleList()

        for i in range(num_levels - 1):  # Only need num_levels - 1 lateral connections
            level_dim = embed_dim // (2 ** (num_levels - 2 - i))  # Adjust indexing

            lateral_layer = nn.Sequential(
                nn.Linear(level_dim, level_dim),
                nn.LayerNorm(level_dim),
                nn.SiLU()
            )

            self.lateral_connections.append(lateral_layer)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            pyramid_features: List of [batch_size, level_dim] tensors
            fused_features: [batch_size, embed_dim] final fused features
        """
        # Bottom-up pathway
        bottom_up_features = []
        current_features = x
        
        for bottom_up_layer in self.bottom_up:
            current_features = bottom_up_layer(current_features)
            bottom_up_features.append(current_features)
        
        # Top-down pathway with lateral connections
        pyramid_features = [bottom_up_features[-1]]  # Start with highest level

        for i, (top_down_layer, lateral_layer) in enumerate(zip(self.top_down, self.lateral_connections)):
            # Top-down processing
            top_down_features = top_down_layer(pyramid_features[-1])

            # Lateral connection - use correct indexing
            lateral_features = lateral_layer(bottom_up_features[-(i+2)])

            # Fusion
            fused_level_features = top_down_features + lateral_features
            pyramid_features.append(fused_level_features)
        
        # Reverse to match bottom-up order
        pyramid_features = pyramid_features[::-1]
        
        # Final fusion
        fused_features = pyramid_features[-1]  # Use highest level as base
        
        return pyramid_features, fused_features


class AdaptiveFusionModule(nn.Module):
    """
    Adaptive fusion of multi-scale features based on dataset characteristics
    
    Innovation:
    - Dataset-aware fusion weights
    - Performance-guided adaptation
    - Dynamic feature selection
    """
    
    def __init__(self, feature_dims, output_dim=512, num_datasets=4):
        super(AdaptiveFusionModule, self).__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.num_datasets = num_datasets
        
        # Project all features to common dimension
        self.feature_projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        
        # Dataset embedding for adaptive weights
        self.dataset_embedding = nn.Embedding(num_datasets, output_dim)
        
        # Fusion weight predictor
        self.fusion_predictor = nn.Sequential(
            nn.Linear(output_dim + 3, output_dim),  # +3 for dataset stats
            nn.SiLU(),
            nn.Linear(output_dim, len(feature_dims)),
            nn.Softmax(dim=-1)
        )
        
        # Feature interaction module
        self.feature_interaction = nn.Sequential(
            nn.Linear(output_dim * len(feature_dims), output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
    def forward(self, multi_scale_features, dataset_id=None, dataset_stats=None):
        """
        Args:
            multi_scale_features: List of feature tensors at different scales
            dataset_id: Dataset identifier for adaptive fusion
            dataset_stats: [batch_size, 3] dataset characteristics
        Returns:
            fused_features: [batch_size, output_dim]
            fusion_weights: [batch_size, num_scales] fusion weights
        """
        batch_size = multi_scale_features[0].shape[0]
        
        # Project all features to common dimension
        projected_features = []
        for features, projection in zip(multi_scale_features, self.feature_projections):
            projected = projection(features)
            projected_features.append(projected)
        
        # Adaptive fusion weights
        if dataset_id is not None and dataset_stats is not None:
            # Get dataset embedding
            dataset_emb = self.dataset_embedding(dataset_id)
            if dataset_emb.dim() == 3:
                dataset_emb = dataset_emb.squeeze(0)
            dataset_emb = dataset_emb.expand(batch_size, -1)
            
            # Combine with dataset statistics
            fusion_input = torch.cat([dataset_emb, dataset_stats], dim=-1)
            fusion_weights = self.fusion_predictor(fusion_input)
        else:
            # Default uniform weights
            fusion_weights = torch.ones(batch_size, len(projected_features), 
                                      device=multi_scale_features[0].device)
            fusion_weights = fusion_weights / len(projected_features)
        
        # Weighted fusion
        weighted_features = []
        for i, features in enumerate(projected_features):
            weight = fusion_weights[:, i:i+1]  # [batch_size, 1]
            weighted = weight * features
            weighted_features.append(weighted)
        
        # Concatenate for interaction
        concatenated = torch.cat(weighted_features, dim=-1)
        
        # Feature interaction
        fused_features = self.feature_interaction(concatenated)
        
        return fused_features, fusion_weights


class HierarchicalCortexFlowV2(nn.Module):
    """
    Complete hierarchical feature fusion model for CCCV2
    
    Integrates:
    - Multi-resolution encoding
    - Feature pyramid networks
    - Adaptive fusion
    """
    
    def __init__(self, input_dim, device='cuda'):
        super(HierarchicalCortexFlowV2, self).__init__()
        self.name = "CortexFlow-CLIP-CNN-V2-Hierarchical"
        self.device = device
        self.input_dim = input_dim
        
        # Multi-resolution encoder
        self.multi_res_encoder = MultiResolutionEncoder(input_dim, embed_dim=512, num_scales=4).to(device)
        
        # Feature pyramid network
        self.fpn = FeaturePyramidNetwork(input_dim, embed_dim=512, num_levels=4).to(device)
        
        # Adaptive fusion
        # Multi-res features: [512, 256, 128, 64]
        # FPN features: [64, 128, 256, 512]
        all_feature_dims = [512, 256, 128, 64, 64, 128, 256, 512]
        self.adaptive_fusion = AdaptiveFusionModule(all_feature_dims, output_dim=512).to(device)
        
        # Final encoding to CLIP space
        self.clip_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.03),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Tanh()
        ).to(device)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.02),
            
            nn.Linear(512, 784),
            nn.Sigmoid()
        ).to(device)
        
    def forward(self, x, dataset_id=None, dataset_stats=None):
        """
        Complete hierarchical processing
        
        Args:
            x: [batch_size, input_dim]
            dataset_id: Dataset identifier for adaptive fusion
            dataset_stats: Dataset characteristics
        Returns:
            visual_output: [batch_size, 1, 28, 28]
            encoded_features: [batch_size, 512]
            hierarchy_info: Dict with hierarchical processing info
        """
        # Multi-resolution encoding
        multi_res_features = self.multi_res_encoder(x)
        
        # Feature pyramid processing
        pyramid_features, fpn_fused = self.fpn(x)
        
        # Combine all hierarchical features
        all_features = multi_res_features + pyramid_features
        
        # Adaptive fusion
        fused_features, fusion_weights = self.adaptive_fusion(
            all_features, dataset_id=dataset_id, dataset_stats=dataset_stats
        )
        
        # Encode to CLIP space
        encoded_features = self.clip_encoder(fused_features)
        encoded_features = F.normalize(encoded_features, p=2, dim=1)
        
        # Decode to visual output
        visual_output = self.decoder(encoded_features)
        visual_output = visual_output.view(-1, 1, 28, 28)
        
        # Collect hierarchy information
        hierarchy_info = {
            'multi_res_features': multi_res_features,
            'pyramid_features': pyramid_features,
            'fusion_weights': fusion_weights,
            'fpn_fused': fpn_fused
        }
        
        return visual_output, encoded_features, hierarchy_info


# Factory function for easy model creation
def create_hierarchical_model(input_dim, device='cuda'):
    """Factory function to create hierarchical fusion model"""
    return HierarchicalCortexFlowV2(input_dim, device=device)


# Export main classes
__all__ = [
    'MultiResolutionEncoder',
    'FeaturePyramidNetwork',
    'AdaptiveFusionModule',
    'HierarchicalCortexFlowV2',
    'create_hierarchical_model'
]
