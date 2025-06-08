#!/usr/bin/env python3
"""
Complete CortexFlow Model Implementations
All 5 variants for comprehensive training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CortexFlowSimple(nn.Module):
    """CortexFlow-Simple: Foundation Architecture"""
    
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

class CortexFlowMC(nn.Module):
    """CortexFlow-MC: Monte Carlo Uncertainty Architecture"""
    
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.15),  # MC Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # MC Dropout in decoder
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus()
        )
        
    def forward(self, x, mc_samples=1, training=None):
        if training is None:
            training = self.training
            
        if mc_samples == 1:
            z = self.encoder(x)
            y = self.decoder(z)
            uncertainty = self.uncertainty_head(z)
            return y, uncertainty
        else:
            # Monte Carlo sampling
            self.train()  # Enable dropout for MC sampling
            predictions = []
            uncertainties = []
            
            for _ in range(mc_samples):
                z = self.encoder(x)
                y = self.decoder(z)
                uncertainty = self.uncertainty_head(z)
                predictions.append(y)
                uncertainties.append(uncertainty)
            
            if not training:
                self.eval()  # Restore original mode
            
            predictions = torch.stack(predictions)
            uncertainties = torch.stack(uncertainties)
            
            # Calculate epistemic and aleatoric uncertainty
            mean_pred = predictions.mean(dim=0)
            epistemic = predictions.var(dim=0)
            aleatoric = uncertainties.mean(dim=0)
            
            return mean_pred, epistemic, aleatoric

class CortexFlowHierarchical(nn.Module):
    """CortexFlow-Hierarchical: Multi-Scale Processing Architecture"""
    
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        self.scales = [1, 2, 4]
        hidden_dim = 256
        
        # Multi-scale encoders
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // scale),
                nn.ReLU()
            ) for scale in self.scales
        ])
        
        # Cross-scale attention
        total_hidden = sum(hidden_dim // scale for scale in self.scales)
        self.attention = nn.MultiheadAttention(total_hidden, num_heads=4, batch_first=True)
        
        # Hierarchical decoder
        self.decoder = nn.Sequential(
            nn.Linear(total_hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Multi-scale encoding
        scale_features = []
        for encoder in self.scale_encoders:
            feat = encoder(x)
            scale_features.append(feat)
        
        # Concatenate scale features
        combined_features = torch.cat(scale_features, dim=-1)
        
        # Self-attention across scales
        combined_features = combined_features.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.attention(combined_features, combined_features, combined_features)
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Decode
        output = self.decoder(attended_features)
        return output

class CortexFlowEnhanced(nn.Module):
    """CortexFlow-Enhanced: Hierarchical + MC + Alignment"""
    
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        # Hierarchical component
        self.hierarchical = CortexFlowHierarchical(input_dim, output_dim)
        
        # Uncertainty estimation
        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus()
        )
        
        # Feature alignment
        self.alignment_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()
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
    
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        # Complexity predictor
        self.complexity_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Simple pathway
        self.simple_path = CortexFlowSimple(input_dim, output_dim)
        
        # Complex pathway (hierarchical)
        self.complex_path = CortexFlowHierarchical(input_dim, output_dim)
        
        # Adaptive fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 2 + 1, 256),  # +1 for complexity score
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        
        # Uncertainty estimation
        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus()
        )
        
    def forward(self, x, mc_samples=1, return_complexity=False):
        # Predict complexity
        complexity = self.complexity_predictor(x)
        
        # Dual pathway processing
        simple_output = self.simple_path(x)
        complex_output = self.complex_path(x)
        
        # Adaptive fusion
        fusion_input = torch.cat([simple_output, complex_output, complexity], dim=-1)
        fusion_weights = self.fusion_gate(fusion_input)
        
        output = (fusion_weights[:, 0:1] * simple_output + 
                 fusion_weights[:, 1:2] * complex_output)
        
        if mc_samples == 1:
            z = self.uncertainty_encoder(x)
            uncertainty = self.uncertainty_head(z)
            
            if return_complexity:
                return output, uncertainty, complexity
            else:
                return output, uncertainty
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
            
            if return_complexity:
                return mean_pred, epistemic, aleatoric, mean_complexity
            else:
                return mean_pred, epistemic, aleatoric

def create_model(variant, input_dim, output_dim=784):
    """Factory function to create CortexFlow models"""
    models = {
        'simple': CortexFlowSimple,
        'mc': CortexFlowMC,
        'hierarchical': CortexFlowHierarchical,
        'enhanced': CortexFlowEnhanced,
        'unified': CortexFlowUnified
    }
    
    if variant not in models:
        raise ValueError(f"Unknown variant: {variant}")
    
    return models[variant](input_dim, output_dim)

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test all models
    input_dim = 967
    batch_size = 16
    x = torch.randn(batch_size, input_dim)
    
    for variant in ['simple', 'mc', 'hierarchical', 'enhanced', 'unified']:
        print(f"\n{variant.upper()} Model:")
        model = create_model(variant, input_dim)
        params = count_parameters(model)
        print(f"Parameters: {params:,}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            if variant == 'simple':
                output = model(x)
                print(f"Output shape: {output.shape}")
            elif variant == 'mc':
                output, uncertainty = model(x)
                print(f"Output shape: {output.shape}, Uncertainty shape: {uncertainty.shape}")
            elif variant == 'hierarchical':
                output = model(x)
                print(f"Output shape: {output.shape}")
            elif variant == 'enhanced':
                output, uncertainty, alignment = model(x)
                print(f"Output: {output.shape}, Uncertainty: {uncertainty.shape}, Alignment: {alignment.shape}")
            elif variant == 'unified':
                output, uncertainty, complexity = model(x, return_complexity=True)
                print(f"Output: {output.shape}, Uncertainty: {uncertainty.shape}, Complexity: {complexity.shape}")
        
        print("✅ Model test passed")
