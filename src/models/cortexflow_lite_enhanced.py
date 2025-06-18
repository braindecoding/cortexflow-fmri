"""
CortexFlow Lite Enhanced: Optimized to Beat Brain-Diffuser
=========================================================

Enhanced version of CortexFlow Lite specifically optimized to compete with Brain-Diffuser
on Miyawaki dataset. Incorporates best practices from diffusion models while maintaining
lightweight architecture.

Key Enhancements:
1. SiLU activations (like Brain-Diffuser)
2. LayerNorm for stability (like Brain-Diffuser)
3. Reduced dropout rates (like Brain-Diffuser)
4. Diffusion-inspired processing
5. Advanced residual connections
6. Attention mechanisms
7. Progressive refinement

Target: Beat Brain-Diffuser MSE 0.008330 on Miyawaki
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CortexFlowLiteEnhanced(nn.Module):
    """CortexFlow Lite Enhanced: Optimized to Beat Brain-Diffuser"""

    def __init__(self, input_dim, device='cuda'):
        super(CortexFlowLiteEnhanced, self).__init__()
        self.name = "CortexFlow-Lite-Enhanced"
        self.device = device

        # ENHANCED: Brain-Diffuser inspired projection with SiLU and LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),  # LayerNorm like Brain-Diffuser
            nn.SiLU(),           # SiLU like Brain-Diffuser
            nn.Dropout(0.1),     # Reduced dropout like Brain-Diffuser
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05),    # Even lower dropout
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        ).to(device)

        # ENHANCED: Diffusion-inspired noise predictor
        self.noise_predictor = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 784)
        ).to(device)

        # ENHANCED: Attention mechanism for feature refinement
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.05,
            batch_first=True
        ).to(device)

        # ENHANCED: Progressive refinement network
        self.refinement_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(784, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Linear(512, 784),
                nn.Tanh()  # Residual refinement
            ).to(device) for _ in range(3)  # 3 refinement steps
        ])

        # ENHANCED: Advanced CNN with residual connections
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.LayerNorm([64, 28, 28]),  # LayerNorm for CNN
            nn.SiLU(),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LayerNorm([128, 28, 28]),
            nn.SiLU(),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LayerNorm([256, 28, 28]),
            nn.SiLU(),
        ).to(device)

        self.cnn_decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LayerNorm([128, 28, 28]),
            nn.SiLU(),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LayerNorm([64, 28, 28]),
            nn.SiLU(),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LayerNorm([32, 28, 28]),
            nn.SiLU(),
            
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        ).to(device)

        # ENHANCED: Uncertainty estimation (like advanced models)
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)

        # ENHANCED: Diffusion parameters (like Brain-Diffuser)
        self.num_timesteps = 5  # Lightweight diffusion
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(device)

    def forward(self, x):
        """
        Enhanced forward pass with Brain-Diffuser inspired processing.
        
        Args:
            x: Input fMRI signals [batch_size, input_dim]
            
        Returns:
            Reconstructed images [batch_size, 1, 28, 28]
        """
        batch_size = x.size(0)
        
        # ENHANCED: Feature extraction with attention
        features = self.projection(x)  # [B, 256]
        
        # ENHANCED: Self-attention for feature refinement
        attended_features, _ = self.attention(
            features.unsqueeze(1),  # [B, 1, 256]
            features.unsqueeze(1),
            features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)  # [B, 256]
        
        # ENHANCED: Combine original and attended features
        enhanced_features = features + attended_features
        
        # ENHANCED: Diffusion-inspired noise prediction
        predicted_noise = self.noise_predictor(enhanced_features)  # [B, 784]
        
        # ENHANCED: Progressive refinement (like iterative denoising)
        refined_output = predicted_noise
        for refinement_layer in self.refinement_layers:
            residual = refinement_layer(refined_output)
            refined_output = refined_output + 0.1 * residual  # Small residual updates
        
        # ENHANCED: Uncertainty-aware processing
        uncertainty = self.uncertainty_estimator(enhanced_features)  # [B, 1]
        
        # ENHANCED: Uncertainty-weighted output
        base_output = torch.sigmoid(refined_output)
        uncertainty_adjusted = base_output * (1.0 - uncertainty) + 0.5 * uncertainty
        
        # ENHANCED: CNN processing with residual connections
        reshaped = uncertainty_adjusted.view(-1, 1, 28, 28)
        
        # Encoder path
        encoded = self.cnn_encoder(reshaped)
        
        # Skip connection
        skip_connection = reshaped.expand(-1, 256, -1, -1)  # Match channels
        enhanced_encoded = encoded + 0.1 * skip_connection
        
        # Decoder path
        final_output = self.cnn_decoder(enhanced_encoded)
        
        return final_output

    def predict_with_uncertainty(self, x):
        """
        Enhanced prediction with uncertainty quantification.
        
        Args:
            x: Input fMRI signals
            
        Returns:
            tuple: (prediction, uncertainty_score)
        """
        with torch.no_grad():
            features = self.projection(x)
            
            attended_features, _ = self.attention(
                features.unsqueeze(1),
                features.unsqueeze(1),
                features.unsqueeze(1)
            )
            attended_features = attended_features.squeeze(1)
            
            enhanced_features = features + attended_features
            uncertainty = self.uncertainty_estimator(enhanced_features)
            
            prediction = self.forward(x)
            
            return prediction, uncertainty

    def diffusion_sample(self, x, num_steps=None):
        """
        Brain-Diffuser style sampling for enhanced quality.
        
        Args:
            x: Input fMRI signals
            num_steps: Number of diffusion steps
            
        Returns:
            Enhanced reconstruction
        """
        if num_steps is None:
            num_steps = self.num_timesteps
            
        with torch.no_grad():
            # Get base features
            features = self.projection(x)
            attended_features, _ = self.attention(
                features.unsqueeze(1),
                features.unsqueeze(1),
                features.unsqueeze(1)
            )
            attended_features = attended_features.squeeze(1)
            enhanced_features = features + attended_features
            
            # Initial prediction
            output = self.noise_predictor(enhanced_features)
            
            # Diffusion-style refinement
            for step in range(num_steps):
                noise_level = 0.1 * (1.0 - step / num_steps)
                step_noise = torch.randn_like(output, device=self.device) * noise_level
                
                # Denoise
                output = output - step_noise * 0.5
                
                # Refine
                for refinement_layer in self.refinement_layers:
                    residual = refinement_layer(output)
                    output = output + 0.05 * residual
            
            # Final processing
            uncertainty = self.uncertainty_estimator(enhanced_features)
            base_output = torch.sigmoid(output)
            uncertainty_adjusted = base_output * (1.0 - uncertainty) + 0.5 * uncertainty
            
            reshaped = uncertainty_adjusted.view(-1, 1, 28, 28)
            encoded = self.cnn_encoder(reshaped)
            skip_connection = reshaped.expand(-1, 256, -1, -1)
            enhanced_encoded = encoded + 0.1 * skip_connection
            final_output = self.cnn_decoder(enhanced_encoded)
            
            return final_output


class CortexFlowLiteUltra(nn.Module):
    """CortexFlow Lite Ultra: Maximum optimization for Miyawaki"""

    def __init__(self, input_dim, device='cuda'):
        super(CortexFlowLiteUltra, self).__init__()
        self.name = "CortexFlow-Lite-Ultra"
        self.device = device

        # ULTRA: Even more aggressive optimization
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.05),
            
            nn.Linear(1024, 1024),  # Deeper network
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.03),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.03),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        ).to(device)

        # ULTRA: Multiple attention heads
        self.multi_attention = nn.ModuleList([
            nn.MultiheadAttention(256, 8, dropout=0.03, batch_first=True).to(device)
            for _ in range(3)  # 3 attention layers
        ])

        # ULTRA: Advanced decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.03),
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.03),
            
            nn.Linear(1024, 784),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        """Ultra-optimized forward pass"""
        # Feature extraction
        features = self.feature_extractor(x)  # [B, 256]
        
        # Multiple attention layers
        attended = features
        for attention_layer in self.multi_attention:
            attended_out, _ = attention_layer(
                attended.unsqueeze(1),
                attended.unsqueeze(1),
                attended.unsqueeze(1)
            )
            attended = attended + attended_out.squeeze(1)  # Residual connection
        
        # Decode to image
        output = self.decoder(attended)
        return output.view(-1, 1, 28, 28)
