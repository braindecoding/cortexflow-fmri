#!/usr/bin/env python3
"""
CortexFlow Model Implementations for Actual Experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CortexFlowSimple(nn.Module):
    """CortexFlow-Simple: Foundation Architecture"""
    
    def __init__(self, input_dim, hidden_dim=512, output_dim=784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

class CortexFlowMC(nn.Module):
    """CortexFlow-MC: Uncertainty-Aware Architecture"""
    
    def __init__(self, input_dim, hidden_dim=512, output_dim=784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()
        )
        
    def forward(self, x, mc_samples=1):
        if mc_samples == 1:
            z = self.encoder(x)
            y = self.decoder(z)
            uncertainty = self.uncertainty_head(z)
            return y, uncertainty
        else:
            # Monte Carlo sampling
            predictions = []
            uncertainties = []
            
            for _ in range(mc_samples):
                z = self.encoder(x)
                y = self.decoder(z)
                uncertainty = self.uncertainty_head(z)
                predictions.append(y)
                uncertainties.append(uncertainty)
            
            predictions = torch.stack(predictions)
            uncertainties = torch.stack(uncertainties)
            
            # Calculate epistemic and aleatoric uncertainty
            mean_pred = predictions.mean(dim=0)
            epistemic = predictions.var(dim=0)
            aleatoric = uncertainties.mean(dim=0)
            
            return mean_pred, epistemic, aleatoric

class CortexFlowHierarchical(nn.Module):
    """CortexFlow-Hierarchical: Multi-Scale Processing Architecture"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=784):
        super().__init__()
        self.scales = [1, 2, 4, 8]
        
        # Temporal encoders for different scales
        self.temporal_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=scale, padding=scale//2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(hidden_dim)
            ) for scale in self.scales
        ])
        
        # Cross-scale attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Progressive decoder
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim // (2**i)),
                nn.Sigmoid()
            ) for i in range(len(self.scales))
        ])
        
        # Final fusion
        self.fusion = nn.Linear(sum(output_dim // (2**i) for i in range(len(self.scales))), output_dim)
        
    def forward(self, x):
        # Reshape for temporal processing
        x_temp = x.unsqueeze(1)  # Add channel dimension
        
        # Multi-scale encoding
        scale_features = []
        for encoder in self.temporal_encoders:
            feat = encoder(x_temp).squeeze(-1)  # Remove temporal dimension
            scale_features.append(feat)
        
        # Cross-scale attention
        scale_features = torch.stack(scale_features, dim=0)  # [scales, batch, hidden]
        attended_features, _ = self.attention(scale_features, scale_features, scale_features)
        
        # Progressive decoding
        scale_outputs = []
        for i, decoder in enumerate(self.decoders):
            output = decoder(attended_features[i])
            scale_outputs.append(output)
        
        # Fusion
        fused = torch.cat(scale_outputs, dim=-1)
        final_output = torch.sigmoid(self.fusion(fused))
        
        return final_output

class CortexFlowEnhanced(nn.Module):
    """CortexFlow-Enhanced: Advanced Integration Architecture"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=784):
        super().__init__()
        # Combine hierarchical processing with uncertainty
        self.hierarchical = CortexFlowHierarchical(input_dim, hidden_dim, output_dim)
        
        # Additional uncertainty estimation
        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()
        )
        
        # Feature alignment
        self.alignment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, mc_samples=1):
        # Hierarchical processing
        hierarchical_output = self.hierarchical(x)
        
        if mc_samples == 1:
            # Single forward pass
            z = self.uncertainty_encoder(x)
            uncertainty = self.uncertainty_head(z)
            alignment = self.alignment_head(z)
            
            return hierarchical_output, uncertainty, alignment
        else:
            # Monte Carlo sampling
            predictions = []
            uncertainties = []
            alignments = []
            
            for _ in range(mc_samples):
                z = self.uncertainty_encoder(x)
                uncertainty = self.uncertainty_head(z)
                alignment = self.alignment_head(z)
                
                predictions.append(hierarchical_output)
                uncertainties.append(uncertainty)
                alignments.append(alignment)
            
            predictions = torch.stack(predictions)
            uncertainties = torch.stack(uncertainties)
            alignments = torch.stack(alignments)
            
            mean_pred = predictions.mean(dim=0)
            epistemic = predictions.var(dim=0)
            aleatoric = uncertainties.mean(dim=0)
            mean_alignment = alignments.mean(dim=0)
            
            return mean_pred, epistemic, aleatoric, mean_alignment

class CortexFlowUnified(nn.Module):
    """CortexFlow-Unified: Adaptive Intelligence Architecture"""
    
    def __init__(self, input_dim, hidden_dim=512, output_dim=784):
        super().__init__()
        # Complexity predictor
        self.complexity_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Simple pathway
        self.simple_path = CortexFlowSimple(input_dim, hidden_dim, output_dim)
        
        # Complex pathway (hierarchical)
        self.complex_path = CortexFlowHierarchical(input_dim, hidden_dim//2, output_dim)
        
        # Feature fusion attention
        self.fusion_attention = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Uncertainty estimation
        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()
        )
        
    def forward(self, x, mc_samples=1, config='balanced'):
        # Predict complexity
        complexity = self.complexity_predictor(x)
        
        # Dual pathway processing
        simple_output = self.simple_path(x)
        complex_output = self.complex_path(x)
        
        # Adaptive fusion based on complexity
        if complexity.mean() > 0.5:
            # High complexity: use complex path more
            fused_features = torch.cat([simple_output, complex_output], dim=-1)
            attention_weights = self.fusion_attention(fused_features)
            output = attention_weights[:, 0:1] * simple_output + attention_weights[:, 1:2] * complex_output
        else:
            # Low complexity: use simple path more
            output = 0.8 * simple_output + 0.2 * complex_output
        
        if config == 'simple':
            return output
        
        # Uncertainty estimation for balanced/advanced configs
        if mc_samples == 1:
            z = self.uncertainty_encoder(x)
            uncertainty = self.uncertainty_head(z)
            return output, uncertainty, complexity
        else:
            # Monte Carlo sampling
            predictions = []
            uncertainties = []
            complexities = []
            
            for _ in range(mc_samples):
                z = self.uncertainty_encoder(x)
                uncertainty = self.uncertainty_head(z)
                
                predictions.append(output)
                uncertainties.append(uncertainty)
                complexities.append(complexity)
            
            predictions = torch.stack(predictions)
            uncertainties = torch.stack(uncertainties)
            complexities = torch.stack(complexities)
            
            mean_pred = predictions.mean(dim=0)
            epistemic = predictions.var(dim=0)
            aleatoric = uncertainties.mean(dim=0)
            mean_complexity = complexities.mean(dim=0)
            
            return mean_pred, epistemic, aleatoric, mean_complexity
