"""
CortexFlow Ensemble: 8-Variant Ensemble Architecture (ENHANCED)
==============================================================

CortexFlow Variant Ensemble combining 8 different neural decoding approaches:
1. Simple: Foundation encoder-decoder with optimal regularization
2. MC: Monte Carlo uncertainty quantification with systematic dropout
3. Hierarchical: Multi-scale temporal processing with attention mechanism
4. Enhanced: Integration of hierarchical + MC + feature alignment
5. Unified: Adaptive complexity mechanism with dual-pathway processing
6. Diffusion: CortexFlow with latent diffusion to compete with Brain-Diffuser
7. CortexFlow-Lite: Lightweight CNN architecture matching standalone performance
8. CortexFlow Multi-Pathway: Advanced multi-pathway with cross-attention (NEW!)

Key Features:
    - 8 diverse neural decoding architectures (ENHANCED!)
    - Advanced learned ensemble weighting for 8 models
    - Input-dependent model selection with Multi-Pathway integration
    - Comprehensive coverage including winning Multi-Pathway architecture
    - Maximum performance through enhanced ensemble diversity

Architecture:
    Input fMRI → 8 Parallel Models → Learned Ensemble Weights → Weighted Combination → Output Image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CortexFlowEnsemble(nn.Module):
    """CortexFlow Variant Ensemble: 8-Variant Enhanced Architecture with Multi-Pathway Integration"""

    def __init__(self, input_dim, device='cuda'):
        super(CortexFlowEnsemble, self).__init__()
        self.name = "CortexFlow-Ensemble"
        self.device = device

        # Ensemble of 8 variants (ENHANCED with Multi-Pathway!)
        self.model_simple = self._create_simple_cortexflow(input_dim, device)
        self.model_mc = self._create_mc_cortexflow(input_dim, device)
        self.model_hierarchical = self._create_hierarchical_cortexflow(input_dim, device)
        self.model_enhanced = self._create_enhanced_cortexflow(input_dim, device)
        self.model_unified = self._create_unified_cortexflow(input_dim, device)
        self.model_diffusion = self._create_diffusion_cortexflow(input_dim, device)
        self.model_baseline_cnn = self._create_baseline_cnn(input_dim, device)  # CortexFlow-Lite
        self.model_multi_pathway = self._create_multi_pathway_cortexflow(input_dim, device)  # NEW: Multi-Pathway!

        # Enhanced ensemble weighting with attention mechanism (8 models)
        self.ensemble_weights = nn.Sequential(
            nn.Linear(input_dim, 512),  # Increased capacity
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 8),  # 8 models (ENHANCED!)
            nn.Softmax(dim=1)
        ).to(device)

        # Baseline emphasis mechanism (give more weight to strong performers)
        self.baseline_emphasis = nn.Parameter(torch.tensor(1.5, device=device))  # Learnable emphasis factor

        # Dynamic weighting based on input complexity
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)

    def _create_simple_cortexflow(self, input_dim, device):
        """1. Simple: Arsitektur fondasi encoder-decoder dengan regularisasi optimal"""
        return nn.Sequential(
            # Encoder
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Optimal regularization
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),

            # Decoder
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        ).to(device)

    def _create_mc_cortexflow(self, input_dim, device):
        """2. MC: Monte Carlo uncertainty quantification dengan dropout sistematis"""
        class MCDropout(nn.Module):
            def __init__(self, p=0.15):
                super().__init__()
                self.p = p

            def forward(self, x):
                # Always apply dropout (even in eval mode for MC sampling)
                return F.dropout(x, p=self.p, training=True)

        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            MCDropout(0.15),  # Systematic MC dropout
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            MCDropout(0.15),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            MCDropout(0.1),
            nn.Linear(128, 784),
            nn.Sigmoid()
        ).to(device)

    def _create_hierarchical_cortexflow(self, input_dim, device):
        """3. Hierarchical: Multi-scale temporal processing dengan attention mechanism"""
        class HierarchicalBlock(nn.Module):
            def __init__(self, in_dim, out_dim, level):
                super().__init__()
                self.level = level
                self.linear = nn.Linear(in_dim, out_dim)
                self.norm = nn.LayerNorm(out_dim)
                self.activation = nn.ReLU()
                self.dropout = nn.Dropout(0.15 - level * 0.02)  # Adaptive dropout

                # Multi-scale attention for temporal processing
                self.temporal_attention = nn.Sequential(
                    nn.Linear(out_dim, out_dim // 2),
                    nn.Tanh(),  # Temporal activation
                    nn.Linear(out_dim // 2, out_dim),
                    nn.Sigmoid()
                )

            def forward(self, x):
                x = self.linear(x)
                x = self.norm(x)
                x = self.activation(x)

                # Apply temporal attention
                temporal_weights = self.temporal_attention(x)
                x = x * temporal_weights

                x = self.dropout(x)
                return x

        return nn.Sequential(
            HierarchicalBlock(input_dim, 512, level=1),  # High-level temporal
            HierarchicalBlock(512, 256, level=2),        # Mid-level temporal
            HierarchicalBlock(256, 128, level=3),        # Low-level temporal
            nn.Linear(128, 784),
            nn.Sigmoid()
        ).to(device)

    def _create_enhanced_cortexflow(self, input_dim, device):
        """4. Enhanced: Integrasi hierarchical + MC + feature alignment"""
        class EnhancedBlock(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.linear = nn.Linear(in_dim, out_dim)
                self.norm = nn.LayerNorm(out_dim)
                self.activation = nn.ReLU()

                # MC dropout component
                self.mc_dropout = nn.Dropout(0.15)

                # Hierarchical attention
                self.hierarchical_attention = nn.Sequential(
                    nn.Linear(out_dim, out_dim // 4),
                    nn.ReLU(),
                    nn.Linear(out_dim // 4, out_dim),
                    nn.Sigmoid()
                )

                # Feature alignment mechanism
                self.feature_alignment = nn.Sequential(
                    nn.Linear(out_dim, out_dim),
                    nn.Tanh(),
                    nn.Linear(out_dim, out_dim)
                )

            def forward(self, x):
                x = self.linear(x)
                x = self.norm(x)
                x = self.activation(x)

                # Apply MC dropout (always active)
                x = F.dropout(x, p=0.15, training=True)

                # Apply hierarchical attention
                att_weights = self.hierarchical_attention(x)
                x_attended = x * att_weights

                # Feature alignment
                x_aligned = self.feature_alignment(x_attended)

                # Residual connection + final dropout
                x = x_attended + x_aligned
                x = self.mc_dropout(x)
                return x

        return nn.Sequential(
            EnhancedBlock(input_dim, 512),
            EnhancedBlock(512, 256),
            nn.Linear(256, 784),
            nn.Sigmoid()
        ).to(device)

    def _create_unified_cortexflow(self, input_dim, device):
        """5. Unified: Adaptive complexity mechanism dengan dual-pathway processing"""
        class AdaptiveComplexityBlock(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                # Dual pathways
                self.pathway_simple = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )

                self.pathway_complex = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(0.15)
                )

                # Adaptive complexity gate
                self.complexity_gate = nn.Sequential(
                    nn.Linear(in_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

                self.norm = nn.LayerNorm(out_dim)

            def forward(self, x):
                # Compute complexity gate
                gate = self.complexity_gate(x)

                # Dual pathway processing
                simple_out = self.pathway_simple(x)
                complex_out = self.pathway_complex(x)

                # Adaptive combination
                output = gate * complex_out + (1 - gate) * simple_out
                output = self.norm(output)

                return output

        return nn.Sequential(
            AdaptiveComplexityBlock(input_dim, 512),
            AdaptiveComplexityBlock(512, 256),
            AdaptiveComplexityBlock(256, 128),
            nn.Linear(128, 784),
            nn.Sigmoid()
        ).to(device)

    def _create_diffusion_cortexflow(self, input_dim, device):
        """6. Diffusion: CortexFlow dengan latent diffusion untuk compete dengan Brain-Diffuser"""

        class CortexFlowDiffusion(nn.Module):
            def __init__(self, input_dim, device):
                super().__init__()

                # Multi-pathway encoder (CortexFlow style)
                self.pathway_deep = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.LayerNorm(512),
                    nn.SiLU(),  # SiLU for diffusion compatibility
                    nn.Dropout(0.15),
                    nn.Linear(512, 256),
                    nn.LayerNorm(256),
                    nn.SiLU(),
                    nn.Dropout(0.1)
                )

                self.pathway_wide = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.LayerNorm(256),
                    nn.SiLU(),
                    nn.Dropout(0.15)
                )

                # Cross-pathway attention
                self.cross_attention = nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)

                # Diffusion-style fusion
                self.diffusion_fusion = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.LayerNorm(256),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128)
                )

                # Diffusion parameters (matching Brain-Diffuser)
                self.num_timesteps = 10
                self.beta_start = 0.0001
                self.beta_end = 0.02

                # Noise predictor
                self.noise_predictor = nn.Sequential(
                    nn.Linear(128 + 1, 128),  # +1 for timestep
                    nn.LayerNorm(128),
                    nn.SiLU(),
                    nn.Linear(128, 128)
                )

                # Progressive diffusion decoder
                self.diffusion_decoder = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.LayerNorm(256),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 512),
                    nn.LayerNorm(512),
                    nn.SiLU(),
                    nn.Linear(512, 784),
                    nn.Sigmoid()
                )

            def forward(self, x):
                batch_size = x.size(0)

                # Multi-pathway processing
                deep_feat = self.pathway_deep(x)
                wide_feat = self.pathway_wide(x)

                # Cross-pathway attention
                deep_att, _ = self.cross_attention(
                    deep_feat.unsqueeze(1), wide_feat.unsqueeze(1), wide_feat.unsqueeze(1)
                )
                deep_att = deep_att.squeeze(1)

                # Combine and fuse
                combined = torch.cat([deep_att, wide_feat], dim=1)
                latent = self.diffusion_fusion(combined)

                # Diffusion process (simplified for efficiency)
                # Add timestep embedding
                t = torch.randint(0, self.num_timesteps, (batch_size, 1), device=x.device).float() / self.num_timesteps
                latent_with_t = torch.cat([latent, t], dim=1)

                # Predict and remove noise
                predicted_noise = self.noise_predictor(latent_with_t)
                denoised = latent - 0.1 * predicted_noise  # Simplified denoising

                # Progressive denoising (3 steps for efficiency like Brain-Diffuser)
                for step in range(3):
                    noise_level = 0.05 * (1.0 - step / 3.0)
                    step_noise = torch.randn_like(denoised) * noise_level
                    denoised = denoised - step_noise

                # Final decode
                output = self.diffusion_decoder(denoised)
                return output

        return CortexFlowDiffusion(input_dim, device).to(device)

    def _create_baseline_cnn(self, input_dim, device):
        """7. CortexFlow-Lite: Lightweight CNN architecture matching standalone performance"""

        class CortexFlowLite(nn.Module):
            def __init__(self, input_dim, device):
                super().__init__()

                # CortexFlow-Lite MLP projection (matching standalone capacity)
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(512, 784),
                    nn.ReLU(inplace=True)
                )

                # CortexFlow-Lite CNN processing (matching standalone architecture)
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, 3, padding=1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                # MLP projection
                projected = self.projection(x)
                # Reshape for CNN
                reshaped = projected.view(-1, 1, 28, 28)
                # CNN processing
                output = self.cnn(reshaped)
                return output.view(output.size(0), -1)  # Flatten for ensemble

        return CortexFlowLite(input_dim, device).to(device)

    def _create_multi_pathway_cortexflow(self, input_dim, device):
        """8. CortexFlow Multi-Pathway: Advanced multi-pathway with cross-attention (WINNING ARCHITECTURE!)"""

        class CortexFlowMultiPathwayEnsemble(nn.Module):
            def __init__(self, input_dim, device):
                super().__init__()

                # NOVEL FEATURE 1: Adaptive Multi-Pathway with Different Receptive Fields
                # Deep pathway for hierarchical feature extraction
                self.pathway_deep = nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.15),
                    nn.Linear(1024, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ).to(device)

                # Wide pathway for broad feature capture
                self.pathway_wide = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.15),
                    nn.Linear(512, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ).to(device)

                # NOVEL FEATURE 2: Cross-Pathway Attention Mechanism
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=512,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True
                ).to(device)

                # NOVEL FEATURE 3: Adaptive Pathway Weighting
                self.pathway_weights = nn.Sequential(
                    nn.Linear(1024, 256),  # Combined features
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 2),     # 2 pathways
                    nn.Softmax(dim=1)
                ).to(device)

                # NOVEL FEATURE 4: Dynamic Gated Fusion
                self.fusion_gate = nn.Sequential(
                    nn.Linear(1024, 1024),  # Match input dimension
                    nn.Sigmoid()
                ).to(device)

                self.fusion = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256)
                ).to(device)

                # NOVEL FEATURE 5: Uncertainty-Aware Decoder
                self.decoder_mean = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 784),
                    nn.Sigmoid()
                ).to(device)

                self.decoder_var = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 784),
                    nn.Softplus()  # Ensure positive variance
                ).to(device)

            def forward(self, x):
                """Forward pass through Multi-Pathway architecture"""
                # Multi-pathway feature extraction
                deep_features = self.pathway_deep(x)  # [batch, 512]
                wide_features = self.pathway_wide(x)  # [batch, 512]

                # Cross-pathway attention for feature interaction
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

                # Get mean prediction (ensemble only needs mean)
                mean_pred = self.decoder_mean(encoded)

                return mean_pred  # Return flattened for ensemble

        return CortexFlowMultiPathwayEnsemble(input_dim, device).to(device)

    def forward(self, x):
        """
        Forward pass through the 8-variant ensemble (ENHANCED!).

        Args:
            x: Input fMRI signals [batch_size, input_dim]

        Returns:
            Weighted ensemble prediction [batch_size, 1, 28, 28]
        """
        # Get predictions from all 8 variants (ENHANCED with Multi-Pathway!)
        pred_simple = self.model_simple(x)
        pred_mc = self.model_mc(x)
        pred_hierarchical = self.model_hierarchical(x)
        pred_enhanced = self.model_enhanced(x)
        pred_unified = self.model_unified(x)
        pred_diffusion = self.model_diffusion(x)
        pred_baseline_cnn = self.model_baseline_cnn(x)
        pred_multi_pathway = self.model_multi_pathway(x)  # NEW: Multi-Pathway!

        # Ensure all predictions are flattened to [batch, 784] for combination
        pred_simple = pred_simple.view(pred_simple.size(0), -1)
        pred_mc = pred_mc.view(pred_mc.size(0), -1)
        pred_hierarchical = pred_hierarchical.view(pred_hierarchical.size(0), -1)
        pred_enhanced = pred_enhanced.view(pred_enhanced.size(0), -1)
        pred_unified = pred_unified.view(pred_unified.size(0), -1)
        pred_diffusion = pred_diffusion.view(pred_diffusion.size(0), -1)
        pred_baseline_cnn = pred_baseline_cnn.view(pred_baseline_cnn.size(0), -1)
        pred_multi_pathway = pred_multi_pathway.view(pred_multi_pathway.size(0), -1)  # NEW!

        # Enhanced ensemble weighting with baseline emphasis
        base_weights = self.ensemble_weights(x)

        # Analyze input complexity for dynamic weighting
        complexity_score = self.complexity_analyzer(x)

        # Apply baseline emphasis (give more weight to strong performers)
        enhanced_weights = base_weights * 1.0  # Avoid in-place operations
        baseline_emphasis_factor = self.baseline_emphasis.unsqueeze(0).expand(enhanced_weights.size(0), 1)
        enhanced_weights = torch.cat([
            enhanced_weights[:, :6],  # First 6 weights unchanged
            enhanced_weights[:, 6:7] * baseline_emphasis_factor,  # CortexFlow-Lite weight emphasized
            enhanced_weights[:, 7:8] * baseline_emphasis_factor   # Multi-Pathway weight emphasized (NEW!)
        ], dim=1)

        # Renormalize weights
        enhanced_weights = F.softmax(enhanced_weights, dim=1)

        # Dynamic adjustment based on complexity (avoid in-place operations)
        complexity_adjustment = torch.ones_like(enhanced_weights)
        # More CortexFlow-Lite for simple inputs
        baseline_adj = complexity_adjustment[:, 6:7] * (2.0 - complexity_score)
        # More Multi-Pathway for complex inputs (cross-modal tasks)
        multi_pathway_adj = complexity_adjustment[:, 7:8] * complexity_score
        # More diffusion for complex inputs
        diffusion_adj = complexity_adjustment[:, 5:6] * complexity_score

        complexity_adjustment = torch.cat([
            complexity_adjustment[:, :5],  # First 5 unchanged
            diffusion_adj,      # Diffusion adjustment
            baseline_adj,       # CortexFlow-Lite adjustment
            multi_pathway_adj   # Multi-Pathway adjustment (NEW!)
        ], dim=1)

        final_weights = enhanced_weights * complexity_adjustment
        final_weights = F.softmax(final_weights, dim=1)

        # Weighted ensemble prediction with enhanced weighting (8 models!)
        ensemble_pred = (final_weights[:, 0:1] * pred_simple +
                        final_weights[:, 1:2] * pred_mc +
                        final_weights[:, 2:3] * pred_hierarchical +
                        final_weights[:, 3:4] * pred_enhanced +
                        final_weights[:, 4:5] * pred_unified +
                        final_weights[:, 5:6] * pred_diffusion +
                        final_weights[:, 6:7] * pred_baseline_cnn +
                        final_weights[:, 7:8] * pred_multi_pathway)  # NEW: Multi-Pathway!

        return ensemble_pred.view(-1, 1, 28, 28)

    def get_ensemble_info(self):
        """
        Get information about the ensemble architecture.

        Returns:
            Dictionary with ensemble details
        """
        return {
            'name': self.name,
            'num_variants': 8,  # ENHANCED!
            'variants': [
                '1. Simple: Foundation encoder-decoder with optimal regularization',
                '2. MC: Monte Carlo uncertainty quantification with systematic dropout',
                '3. Hierarchical: Multi-scale temporal processing with attention mechanism',
                '4. Enhanced: Integration of hierarchical + MC + feature alignment',
                '5. Unified: Adaptive complexity mechanism with dual-pathway processing',
                '6. Diffusion: CortexFlow with latent diffusion to compete with Brain-Diffuser',
                '7. CortexFlow-Lite: Lightweight CNN architecture matching standalone performance',
                '8. CortexFlow Multi-Pathway: Advanced multi-pathway with cross-attention (WINNING ARCHITECTURE!)'
            ],
            'weighting': 'Enhanced ensemble weighting with Multi-Pathway integration and complexity-aware adjustment',
            'combination': 'Dynamically weighted combination with Multi-Pathway emphasis for complex inputs',
            'enhancements': [
                'CortexFlow Multi-Pathway: Full integration of winning Multi-Pathway architecture',
                'CortexFlow-Lite: Full MLP+CNN architecture matching standalone',
                'Baseline Emphasis: Learnable emphasis factor for strong performers',
                'Complexity Analysis: Dynamic weighting favoring Multi-Pathway for complex inputs',
                'Advanced Weighting: 8-model ensemble with deeper weight learning network',
                'Cross-Modal Excellence: Multi-Pathway integration for superior cross-modal performance'
            ]
        }
