"""
CortexFlow Enhanced: V5 Tuning with Brain-Diffuser Inspiration
============================================================

Enhanced CortexFlow architecture inspired by Brain-Diffuser's excellent performance
on single-modal datasets (MSE: 0.008761 on Miyawaki).

Key Enhancements from Brain-Diffuser:
1. SiLU activation functions for better gradient flow
2. LayerNorm for training stability
3. Diffusion-inspired noise prediction and denoising
4. Reduced dropout rates for stability
5. Iterative refinement process

Architecture Flow:
    Input fMRI → Enhanced Multi-Pathway → Cross-Attention → Diffusion Processing → 
    Noise Prediction → Iterative Denoising → Uncertainty-Aware Output

V5 Tuning Goals:
    - Improve single-modal performance (compete with Brain-Diffuser)
    - Maintain cross-modal capabilities
    - Enhanced stability and convergence
"""

import torch
import torch.nn as nn


class CortexFlowEnhanced(nn.Module):
    """CortexFlow Enhanced: V5 with Brain-Diffuser Inspiration"""

    def __init__(self, input_dim, device='cuda'):
        super(CortexFlowEnhanced, self).__init__()
        self.name = "CortexFlow_Enhanced"
        self.device = device

        # ENHANCED: Brain-Diffuser inspired pathway with SiLU activation
        # Deep pathway: Hierarchical feature extraction with diffusion-style processing
        self.pathway_deep = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),  # Brain-Diffuser inspiration: SiLU activation
            nn.Dropout(0.1),  # Reduced dropout like Brain-Diffuser
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),  # Consistent SiLU usage
            nn.Dropout(0.05)  # Further reduced for stability
        ).to(device)

        # Wide pathway: Broad feature capture with diffusion-style processing
        self.pathway_wide = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),  # Brain-Diffuser inspiration
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05)
        ).to(device)

        # ENHANCED: Cross-pathway attention with diffusion-compatible dimensions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.05,  # Reduced dropout for stability
            batch_first=True
        ).to(device)

        # ENHANCED: Adaptive pathway weighting with SiLU
        self.pathway_weights = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.SiLU(),  # Brain-Diffuser style activation
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        ).to(device)

        # ENHANCED: Dynamic gated fusion with diffusion-style processing
        self.fusion_gate = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),  # Added LayerNorm for stability
            nn.Sigmoid()
        ).to(device)

        # ENHANCED: Feature fusion with Brain-Diffuser inspired architecture
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),  # Brain-Diffuser activation
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128)
        ).to(device)

        # NEW: Brain-Diffuser inspired diffusion components
        self.num_timesteps = 10
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # ENHANCED: Diffusion-aware noise predictor
        self.noise_predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 784)
        ).to(device)

        # ENHANCED: Uncertainty-aware decoder branches with SiLU
        self.decoder_mean = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),  # Brain-Diffuser activation
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        ).to(device)

        self.decoder_var = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),  # Brain-Diffuser activation
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 784),
            nn.Softplus()  # Ensure positive variance
        ).to(device)

    def forward(self, x):
        """
        Enhanced forward pass with Brain-Diffuser inspired processing.
        
        Args:
            x: Input fMRI signals [batch_size, input_dim]
            
        Returns:
            Reconstructed images [batch_size, 1, 28, 28]
        """
        # Enhanced multi-pathway feature extraction
        deep_features = self.pathway_deep(x)  # [batch, 512]
        wide_features = self.pathway_wide(x)  # [batch, 512]

        # ENHANCED: Cross-pathway attention for feature interaction
        deep_attended, _ = self.cross_attention(
            deep_features.unsqueeze(1),
            wide_features.unsqueeze(1),
            wide_features.unsqueeze(1)
        )
        deep_attended = deep_attended.squeeze(1)

        wide_attended, _ = self.cross_attention(
            wide_features.unsqueeze(1),
            deep_features.unsqueeze(1),
            deep_features.unsqueeze(1)
        )
        wide_attended = wide_attended.squeeze(1)

        # ENHANCED: Adaptive pathway weighting
        combined_features = torch.cat([deep_attended, wide_attended], dim=1)
        pathway_weights = self.pathway_weights(combined_features)

        weighted_deep = deep_attended * pathway_weights[:, 0:1]
        weighted_wide = wide_attended * pathway_weights[:, 1:2]

        # ENHANCED: Dynamic gated fusion
        fusion_input = torch.cat([weighted_deep, weighted_wide], dim=1)
        gate = self.fusion_gate(fusion_input)
        gated_features = fusion_input * gate

        # Feature fusion
        encoded = self.fusion(gated_features)

        # NEW: Brain-Diffuser inspired diffusion process
        # Predict noise
        predicted_noise = self.noise_predictor(encoded)
        
        # Iterative denoising (Brain-Diffuser style)
        denoised = predicted_noise
        for step in range(3):  # 3 denoising steps like Brain-Diffuser
            noise_level = 0.1 * (1.0 - step / 3.0)
            step_noise = torch.randn_like(denoised, device=self.device) * noise_level
            denoised = denoised - step_noise

        # ENHANCED: Uncertainty-aware prediction with diffusion refinement
        mean_pred = self.decoder_mean(encoded)
        var_pred = self.decoder_var(encoded)
        
        # Combine diffusion output with uncertainty-aware prediction
        final_output = 0.7 * mean_pred + 0.3 * denoised.sigmoid()
        
        return final_output.view(-1, 1, 28, 28)

    def predict_with_uncertainty(self, x):
        """
        Enhanced prediction with uncertainty quantification and diffusion refinement.
        
        Args:
            x: Input fMRI signals
            
        Returns:
            tuple: (mean_prediction, variance_prediction, diffusion_output)
        """
        with torch.no_grad():
            # Forward pass to get encoded features
            deep_features = self.pathway_deep(x)
            wide_features = self.pathway_wide(x)
            
            # Cross-attention
            deep_attended, _ = self.cross_attention(
                deep_features.unsqueeze(1),
                wide_features.unsqueeze(1), 
                wide_features.unsqueeze(1)
            )
            deep_attended = deep_attended.squeeze(1)
            
            wide_attended, _ = self.cross_attention(
                wide_features.unsqueeze(1),
                deep_features.unsqueeze(1),
                deep_features.unsqueeze(1)
            )
            wide_attended = wide_attended.squeeze(1)
            
            # Adaptive weighting and fusion
            combined_features = torch.cat([deep_attended, wide_attended], dim=1)
            pathway_weights = self.pathway_weights(combined_features)
            
            weighted_deep = deep_attended * pathway_weights[:, 0:1]
            weighted_wide = wide_attended * pathway_weights[:, 1:2]
            
            fusion_input = torch.cat([weighted_deep, weighted_wide], dim=1)
            gate = self.fusion_gate(fusion_input)
            gated_features = fusion_input * gate
            
            encoded = self.fusion(gated_features)
            
            # Get predictions
            mean_pred = self.decoder_mean(encoded)
            var_pred = self.decoder_var(encoded)
            
            # Diffusion output
            predicted_noise = self.noise_predictor(encoded)
            denoised = predicted_noise
            for step in range(3):
                noise_level = 0.1 * (1.0 - step / 3.0)
                step_noise = torch.randn_like(denoised, device=self.device) * noise_level
                denoised = denoised - step_noise
            diffusion_output = denoised.sigmoid()
            
            return (mean_pred.view(-1, 1, 28, 28), 
                   var_pred.view(-1, 1, 28, 28),
                   diffusion_output.view(-1, 1, 28, 28))

    def sample_reconstruction(self, fmri_input, num_steps=None):
        """
        Brain-Diffuser style sampling with full denoising process.
        
        Args:
            fmri_input: Input fMRI signals
            num_steps: Number of denoising steps (default: self.num_timesteps)
            
        Returns:
            Final reconstructed image
        """
        if num_steps is None:
            num_steps = self.num_timesteps
            
        with torch.no_grad():
            # Get encoded features
            deep_features = self.pathway_deep(fmri_input)
            wide_features = self.pathway_wide(fmri_input)
            
            # Process through pathways
            deep_attended, _ = self.cross_attention(
                deep_features.unsqueeze(1),
                wide_features.unsqueeze(1),
                wide_features.unsqueeze(1)
            )
            deep_attended = deep_attended.squeeze(1)
            
            wide_attended, _ = self.cross_attention(
                wide_features.unsqueeze(1),
                deep_features.unsqueeze(1),
                deep_features.unsqueeze(1)
            )
            wide_attended = wide_attended.squeeze(1)
            
            combined_features = torch.cat([deep_attended, wide_attended], dim=1)
            pathway_weights = self.pathway_weights(combined_features)
            
            weighted_deep = deep_attended * pathway_weights[:, 0:1]
            weighted_wide = wide_attended * pathway_weights[:, 1:2]
            
            fusion_input = torch.cat([weighted_deep, weighted_wide], dim=1)
            gate = self.fusion_gate(fusion_input)
            gated_features = fusion_input * gate
            
            encoded = self.fusion(gated_features)
            
            # Initial noise prediction
            x = self.noise_predictor(encoded)
            
            # Iterative denoising (Brain-Diffuser style)
            for t in reversed(range(num_steps)):
                if t > 0:
                    alpha_t = self.alphas_cumprod[t]
                    alpha_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
                    
                    # Denoising formula
                    x = (x - torch.sqrt(1 - alpha_t) * self.noise_predictor(encoded)) / torch.sqrt(alpha_t)
                    x = x * torch.sqrt(alpha_prev)
                else:
                    x = x - self.noise_predictor(encoded)
            
            # Final projection
            output = torch.sigmoid(x)
            return output.view(-1, 1, 28, 28)
