"""
CortexFlow-CLIP-CNN V2: Multi-Scale Attention Mechanisms
=======================================================

Revolutionary attention mechanisms for neural decoding:
1. Self-Attention for fMRI feature relationships
2. Cross-Modal Attention for brain-visual alignment
3. Multi-Scale Attention Pyramid for hierarchical processing
4. Adaptive Attention Weighting for dataset-specific focus

Innovation: First attention-based neural decoding framework
Goal: Selective feature focus and semantic understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for fMRI spatial relationships"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class fMRISelfAttention(nn.Module):
    """
    Self-attention mechanism for fMRI feature relationships
    
    Innovation:
    - Identifies important brain regions automatically
    - Models spatial dependencies in neural signals
    - Adaptive feature weighting based on semantic content
    """
    
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(fMRISelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization and residual connections
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for spatial relationships
        self.pos_encoding = PositionalEncoding(embed_dim)
        
    def forward(self, fmri_features, mask=None):
        """
        Args:
            fmri_features: [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
        Returns:
            attended_features: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # Add positional encoding
        fmri_features = self.pos_encoding(fmri_features)
        
        # Self-attention with residual connection
        attn_output, attn_weights = self.multihead_attn(
            fmri_features, fmri_features, fmri_features, 
            attn_mask=mask, need_weights=True
        )
        
        # First residual connection and layer norm
        x = self.layer_norm1(fmri_features + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        attended_features = self.layer_norm2(x + ffn_output)
        
        return attended_features, attn_weights


class CrossModalAttention(nn.Module):
    """
    Cross-attention between fMRI and visual features
    
    Innovation:
    - Aligns brain signals with visual concepts
    - Semantic-guided feature selection
    - Multi-modal understanding for better reconstruction
    """
    
    def __init__(self, fmri_dim=512, visual_dim=512, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.fmri_dim = fmri_dim
        self.visual_dim = visual_dim
        self.num_heads = num_heads
        
        # Project to common dimension if needed
        if fmri_dim != visual_dim:
            self.fmri_proj = nn.Linear(fmri_dim, visual_dim)
        else:
            self.fmri_proj = nn.Identity()
        
        # Cross-attention mechanism
        self.cross_attn = nn.MultiheadAttention(
            visual_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(visual_dim)
        
        # Semantic alignment module
        self.semantic_aligner = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.SiLU(),
            nn.Linear(visual_dim, visual_dim),
            nn.Tanh()
        )
        
    def forward(self, fmri_features, visual_features):
        """
        Args:
            fmri_features: [batch_size, fmri_seq_len, fmri_dim]
            visual_features: [batch_size, visual_seq_len, visual_dim]
        Returns:
            aligned_features: [batch_size, fmri_seq_len, visual_dim]
            attention_weights: [batch_size, num_heads, fmri_seq_len, visual_seq_len]
        """
        # Project fMRI features to visual dimension
        fmri_projected = self.fmri_proj(fmri_features)
        
        # Cross-attention: Q=fmri, K=V=visual
        attn_output, attn_weights = self.cross_attn(
            fmri_projected, visual_features, visual_features,
            need_weights=True
        )
        
        # Residual connection and layer norm
        aligned_features = self.layer_norm(fmri_projected + attn_output)
        
        # Semantic alignment enhancement
        semantic_enhancement = self.semantic_aligner(aligned_features)
        final_features = aligned_features + 0.1 * semantic_enhancement
        
        return final_features, attn_weights


class MultiScaleAttentionPyramid(nn.Module):
    """
    Hierarchical attention at multiple scales
    
    Innovation:
    - Local attention for fine details
    - Regional attention for mid-level features
    - Global attention for overall semantics
    - Adaptive scale fusion based on dataset characteristics
    """
    
    def __init__(self, input_dim, scales=[16, 8, 4], dropout=0.1):
        super(MultiScaleAttentionPyramid, self).__init__()
        self.scales = scales
        self.input_dim = input_dim
        
        # Multi-scale attention modules
        self.local_attn = fMRISelfAttention(
            input_dim, num_heads=scales[0], dropout=dropout
        )  # Fine-grained (16 heads)
        
        self.regional_attn = fMRISelfAttention(
            input_dim, num_heads=scales[1], dropout=dropout
        )  # Mid-level (8 heads)
        
        self.global_attn = fMRISelfAttention(
            input_dim, num_heads=scales[2], dropout=dropout
        )  # Coarse (4 heads)
        
        # Scale fusion mechanism
        self.scale_fusion = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        # Adaptive weighting for scales
        self.scale_weights = nn.Parameter(torch.ones(3))
        
        # Dataset-adaptive scaling
        self.dataset_adapter = nn.Linear(1, 3)  # Dataset size -> scale weights
        
    def forward(self, x, dataset_size=None):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            dataset_size: Optional dataset size for adaptive weighting
        Returns:
            fused_features: [batch_size, seq_len, input_dim]
            attention_maps: Dict of attention weights for each scale
        """
        # Multi-scale attention processing
        local_features, local_attn = self.local_attn(x)
        regional_features, regional_attn = self.regional_attn(local_features)
        global_features, global_attn = self.global_attn(regional_features)
        
        # Adaptive scale weighting
        if dataset_size is not None:
            adaptive_weights = self.dataset_adapter(dataset_size.float().unsqueeze(-1))
            scale_weights = torch.softmax(self.scale_weights + adaptive_weights.squeeze(), dim=0)
        else:
            scale_weights = torch.softmax(self.scale_weights, dim=0)
        
        # Weighted combination of scales
        weighted_local = scale_weights[0] * local_features
        weighted_regional = scale_weights[1] * regional_features
        weighted_global = scale_weights[2] * global_features
        
        # Concatenate and fuse
        concatenated = torch.cat([weighted_local, weighted_regional, weighted_global], dim=-1)
        fused_features = self.scale_fusion(concatenated)
        
        # Collect attention maps
        attention_maps = {
            'local': local_attn,
            'regional': regional_attn,
            'global': global_attn,
            'scale_weights': scale_weights
        }
        
        return fused_features, attention_maps


class AdaptiveAttentionWeighting(nn.Module):
    """
    Adaptive attention weighting based on dataset characteristics
    
    Innovation:
    - Dataset-specific attention patterns
    - Automatic adaptation to data properties
    - Performance-guided weight adjustment
    """
    
    def __init__(self, embed_dim=512, num_datasets=4):
        super(AdaptiveAttentionWeighting, self).__init__()
        self.embed_dim = embed_dim
        self.num_datasets = num_datasets
        
        # Dataset embedding
        self.dataset_embedding = nn.Embedding(num_datasets, embed_dim)
        
        # Attention weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(embed_dim + 3, embed_dim),  # +3 for dataset stats
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Performance feedback mechanism
        self.performance_memory = nn.Parameter(torch.zeros(num_datasets, embed_dim))
        
    def forward(self, features, dataset_id, dataset_stats):
        """
        Args:
            features: [batch_size, seq_len, embed_dim]
            dataset_id: Dataset identifier (0-3)
            dataset_stats: [batch_size, 3] (size, complexity, performance)
        Returns:
            weighted_features: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = features.shape
        
        # Get dataset embedding
        dataset_emb = self.dataset_embedding(dataset_id)  # [embed_dim]
        if dataset_emb.dim() == 3:  # Handle extra dimension
            dataset_emb = dataset_emb.squeeze(0)
        dataset_emb = dataset_emb.expand(batch_size, -1)  # [batch_size, embed_dim]
        
        # Combine with dataset statistics
        combined_info = torch.cat([dataset_emb, dataset_stats], dim=-1)  # [batch_size, embed_dim+3]
        
        # Predict attention weights
        attention_weights = self.weight_predictor(combined_info)  # [batch_size, embed_dim]
        attention_weights = attention_weights.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, embed_dim]
        
        # Apply adaptive weighting
        weighted_features = features * attention_weights
        
        return weighted_features, attention_weights
    
    def update_performance_memory(self, dataset_id, performance_score):
        """Update performance memory for adaptive learning"""
        with torch.no_grad():
            # Exponential moving average update
            alpha = 0.1
            self.performance_memory[dataset_id] = (
                alpha * performance_score + 
                (1 - alpha) * self.performance_memory[dataset_id]
            )


class CortexFlowAttentionEncoder(nn.Module):
    """
    Complete attention-based encoder for CCCV2
    
    Integrates all attention mechanisms:
    - Multi-scale attention pyramid
    - Cross-modal attention
    - Adaptive weighting
    """
    
    def __init__(self, input_dim, embed_dim=512, num_heads=8, num_layers=3, dropout=0.1, device='cuda'):
        super(CortexFlowAttentionEncoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.device = device

        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim).to(device)

        # Multi-scale attention pyramid
        self.attention_pyramid = MultiScaleAttentionPyramid(embed_dim, dropout=dropout).to(device)

        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            fMRISelfAttention(embed_dim, num_heads, dropout).to(device)
            for _ in range(num_layers)
        ])

        # Cross-modal attention (for future visual guidance)
        self.cross_modal_attn = CrossModalAttention(embed_dim, embed_dim, num_heads, dropout).to(device)

        # Adaptive weighting
        self.adaptive_weighting = AdaptiveAttentionWeighting(embed_dim).to(device)

        # Output projection to CLIP space
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 512),  # CLIP embedding dimension
            nn.Tanh()
        ).to(device)
        
    def forward(self, x, dataset_id=None, dataset_stats=None, visual_features=None):
        """
        Complete attention-based encoding
        
        Args:
            x: [batch_size, input_dim] - fMRI features
            dataset_id: Dataset identifier for adaptive weighting
            dataset_stats: Dataset statistics for adaptation
            visual_features: Optional visual features for cross-modal attention
        
        Returns:
            encoded_features: [batch_size, 512] - CLIP-aligned features
            attention_info: Dict with attention weights and maps
        """
        batch_size = x.shape[0]
        
        # Project to embedding dimension
        x = self.input_projection(x)  # [batch_size, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, embed_dim] for attention
        
        # Multi-scale attention pyramid
        dataset_size = torch.tensor([dataset_stats[0, 0]] if dataset_stats is not None else [100.0]).to(x.device)
        pyramid_features, attention_maps = self.attention_pyramid(x, dataset_size)
        
        # Stack of attention layers
        current_features = pyramid_features
        layer_attentions = []
        
        for layer in self.attention_layers:
            current_features, layer_attn = layer(current_features)
            layer_attentions.append(layer_attn)
        
        # Cross-modal attention if visual features provided
        if visual_features is not None:
            visual_features = visual_features.unsqueeze(1)  # Add sequence dimension
            current_features, cross_attn = self.cross_modal_attn(current_features, visual_features)
        else:
            cross_attn = None
        
        # Adaptive weighting
        if dataset_id is not None and dataset_stats is not None:
            current_features, adaptive_weights = self.adaptive_weighting(
                current_features, dataset_id, dataset_stats
            )
        else:
            adaptive_weights = None
        
        # Remove sequence dimension and project to output
        current_features = current_features.squeeze(1)  # [batch_size, embed_dim]
        encoded_features = self.output_projection(current_features)  # [batch_size, 512]
        
        # Normalize to unit sphere (CLIP-like)
        encoded_features = F.normalize(encoded_features, p=2, dim=1)
        
        # Collect attention information
        attention_info = {
            'pyramid_maps': attention_maps,
            'layer_attentions': layer_attentions,
            'cross_modal_attention': cross_attn,
            'adaptive_weights': adaptive_weights
        }
        
        return encoded_features, attention_info


# Factory function for easy model creation
def create_attention_encoder(input_dim, config=None):
    """Factory function to create attention-based encoder"""
    
    if config is None:
        config = {
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1
        }
    
    return CortexFlowAttentionEncoder(
        input_dim=input_dim,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )


# Export main classes
__all__ = [
    'fMRISelfAttention',
    'CrossModalAttention', 
    'MultiScaleAttentionPyramid',
    'AdaptiveAttentionWeighting',
    'CortexFlowAttentionEncoder',
    'create_attention_encoder'
]
