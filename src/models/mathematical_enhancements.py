#!/usr/bin/env python3
"""
Mathematical Enhancements for CortexFlow Framework
Novel mathematical formulations to increase novelty and theoretical rigor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class AdaptiveComplexityLoss(nn.Module):
    """
    Novel Adaptive Complexity Loss Function
    Dynamically adjusts loss based on input complexity characteristics
    """
    
    def __init__(self, lambda_complexity: float = 0.1, lambda_consistency: float = 0.05, 
                 alpha: float = 0.2):
        super().__init__()
        self.lambda_complexity = lambda_complexity
        self.lambda_consistency = lambda_consistency
        self.alpha = alpha
        
    def forward(self, predictions_simple: torch.Tensor, predictions_complex: torch.Tensor,
                targets: torch.Tensor, complexity_score: torch.Tensor,
                model_params_simple: dict, model_params_complex: dict) -> torch.Tensor:
        """
        Compute adaptive complexity loss
        
        Args:
            predictions_simple: Simple model predictions
            predictions_complex: Complex model predictions  
            targets: Ground truth targets
            complexity_score: Input complexity score [0,1]
            model_params_simple: Simple model parameters
            model_params_complex: Complex model parameters
        """
        # Base reconstruction loss
        L_base = F.mse_loss(predictions_simple, targets)
        
        # Complexity-aware regularization
        psi_term = self._complexity_regularization(
            complexity_score, model_params_simple, model_params_complex
        )
        
        # Consistency penalty
        phi_term = self._consistency_penalty(
            predictions_simple, predictions_complex, complexity_score
        )
        
        # Total adaptive loss
        L_adaptive = L_base + self.lambda_complexity * psi_term + self.lambda_consistency * phi_term
        
        return L_adaptive
    
    def _complexity_regularization(self, complexity_score: torch.Tensor,
                                 params_simple: dict, params_complex: dict) -> torch.Tensor:
        """Complexity-aware regularization term Ψ(c, θ, x)"""
        
        # Entropy regularization H(c) = -c log(c) - (1-c) log(1-c)
        eps = 1e-8
        c = torch.clamp(complexity_score, eps, 1-eps)
        entropy_reg = -(c * torch.log(c) + (1-c) * torch.log(1-c))
        
        # Gradient norm terms (simplified for implementation)
        grad_norm_simple = sum(p.norm() for p in params_simple.values() if p.requires_grad)
        grad_norm_complex = sum(p.norm() for p in params_complex.values() if p.requires_grad)
        
        psi = (c * grad_norm_complex + (1-c) * grad_norm_simple + 
               self.alpha * entropy_reg.mean())
        
        return psi
    
    def _consistency_penalty(self, pred_simple: torch.Tensor, pred_complex: torch.Tensor,
                           complexity_score: torch.Tensor) -> torch.Tensor:
        """Consistency penalty Φ(c, x)"""
        
        # Maximum penalty when c ≈ 0.5 (uncertain complexity)
        uncertainty_weight = 1 - torch.abs(2 * complexity_score - 1)
        consistency_diff = F.mse_loss(pred_simple, pred_complex, reduction='none')
        
        phi = (consistency_diff * uncertainty_weight.unsqueeze(-1)).mean()
        
        return phi

class CalibratedUncertaintyLoss(nn.Module):
    """
    Novel Calibrated Uncertainty Loss
    Enhances uncertainty quantification with calibration guarantees
    """
    
    def __init__(self, lambda_cal: float = 0.1, lambda_sharp: float = 0.05):
        super().__init__()
        self.lambda_cal = lambda_cal
        self.lambda_sharp = lambda_sharp
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                epistemic_var: torch.Tensor, aleatoric_var: torch.Tensor,
                confidence_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute calibrated uncertainty loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            epistemic_var: Epistemic uncertainty variance
            aleatoric_var: Aleatoric uncertainty variance  
            confidence_scores: Confidence scores for calibration
        """
        
        # Enhanced uncertainty decomposition with interaction term
        interaction_var = 2 * self._covariance(predictions, aleatoric_var)
        total_var = epistemic_var + aleatoric_var + interaction_var
        
        # Negative log-likelihood
        L_nll = torch.log(total_var + 1e-8) + (targets - predictions)**2 / (total_var + 1e-8)
        L_nll = L_nll.mean()
        
        # Calibration loss
        L_calibration = self._calibration_loss(predictions, targets, confidence_scores)
        
        # Sharpness loss (encourages confident predictions)
        L_sharpness = -torch.log(total_var + 1e-8).mean()
        
        total_loss = L_nll + self.lambda_cal * L_calibration + self.lambda_sharp * L_sharpness
        
        return total_loss
    
    def _covariance(self, predictions: torch.Tensor, aleatoric_var: torch.Tensor) -> torch.Tensor:
        """Compute covariance between predictions and aleatoric variance"""
        pred_mean = predictions.mean(dim=0, keepdim=True)
        var_mean = aleatoric_var.mean(dim=0, keepdim=True)
        
        cov = ((predictions - pred_mean) * (aleatoric_var - var_mean)).mean(dim=0)
        return cov.mean()
    
    def _calibration_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                         confidence_scores: torch.Tensor) -> torch.Tensor:
        """Calibration loss: |P(correct|confidence) - confidence|"""
        
        # Discretize confidence scores into bins
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        calibration_error = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this confidence bin
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            
            if in_bin.sum() > 0:
                # Accuracy in this bin
                bin_predictions = predictions[in_bin]
                bin_targets = targets[in_bin]
                bin_accuracy = (F.mse_loss(bin_predictions, bin_targets, reduction='none') < 0.1).float().mean()
                
                # Average confidence in this bin
                bin_confidence = confidence_scores[in_bin].mean()
                
                # Calibration error for this bin
                calibration_error += torch.abs(bin_accuracy - bin_confidence) * in_bin.sum().float()
        
        return calibration_error / confidence_scores.size(0)

class WassersteinCrossModalLoss(nn.Module):
    """
    Novel Wasserstein Cross-Modal Loss
    Uses optimal transport for principled cross-modal alignment
    """
    
    def __init__(self, lambda_cycle: float = 0.1):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        
    def forward(self, fmri_features: torch.Tensor, eeg_features: torch.Tensor,
                transform_eeg_to_fmri: nn.Module, transform_fmri_to_eeg: nn.Module) -> torch.Tensor:
        """
        Compute Wasserstein cross-modal loss
        
        Args:
            fmri_features: fMRI feature representations
            eeg_features: EEG feature representations
            transform_eeg_to_fmri: EEG to fMRI transformation
            transform_fmri_to_eeg: fMRI to EEG transformation
        """
        
        # Approximate 2-Wasserstein distance using Sinkhorn algorithm
        W2_distance = self._sinkhorn_distance(fmri_features, eeg_features)
        
        # Cycle consistency loss
        fmri_reconstructed = transform_eeg_to_fmri(transform_fmri_to_eeg(fmri_features))
        eeg_reconstructed = transform_fmri_to_eeg(transform_eeg_to_fmri(eeg_features))
        
        L_cycle = (F.mse_loss(fmri_features, fmri_reconstructed) + 
                  F.mse_loss(eeg_features, eeg_reconstructed)) / 2
        
        total_loss = W2_distance + self.lambda_cycle * L_cycle
        
        return total_loss
    
    def _sinkhorn_distance(self, x: torch.Tensor, y: torch.Tensor, 
                          eps: float = 0.1, max_iter: int = 100) -> torch.Tensor:
        """
        Approximate 2-Wasserstein distance using Sinkhorn algorithm
        """
        # Compute cost matrix (squared Euclidean distance)
        C = torch.cdist(x, y, p=2) ** 2
        
        # Sinkhorn iterations
        n, m = C.shape
        mu = torch.ones(n, device=x.device) / n
        nu = torch.ones(m, device=y.device) / m
        
        K = torch.exp(-C / eps)
        
        u = torch.ones_like(mu)
        for _ in range(max_iter):
            v = nu / (K.T @ u + 1e-8)
            u = mu / (K @ v + 1e-8)
        
        # Compute optimal transport cost
        transport_plan = u.unsqueeze(1) * K * v.unsqueeze(0)
        wasserstein_distance = (transport_plan * C).sum()
        
        return wasserstein_distance

class InformationTheoreticComplexity(nn.Module):
    """
    Novel Information-Theoretic Complexity Measures
    Uses mutual information and differential entropy for complexity assessment
    """
    
    def __init__(self):
        super().__init__()
        
    def mutual_information_complexity(self, x: torch.Tensor, z_simple: torch.Tensor,
                                    z_complex: torch.Tensor) -> torch.Tensor:
        """
        Compute MI-based complexity: C_MI(x) = I(X; Z_simple) / I(X; Z_complex)
        """
        
        # Approximate mutual information using MINE (Mutual Information Neural Estimation)
        mi_simple = self._mine_estimate(x, z_simple)
        mi_complex = self._mine_estimate(x, z_complex)
        
        complexity_ratio = mi_simple / (mi_complex + 1e-8)
        
        return complexity_ratio
    
    def differential_entropy_complexity(self, z_simple: torch.Tensor,
                                      z_complex: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy-based complexity: C_H(x) = [H(z_complex) - H(z_simple)] / [H(z_complex) + ε]
        """
        
        # Estimate differential entropy using k-nearest neighbors
        h_simple = self._knn_entropy(z_simple)
        h_complex = self._knn_entropy(z_complex)
        
        complexity_score = (h_complex - h_simple) / (h_complex + 1e-8)
        
        return complexity_score
    
    def _mine_estimate(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Simplified MINE estimation for mutual information"""
        
        # Create joint and marginal samples
        joint_samples = torch.cat([x, z], dim=-1)
        
        # Shuffle z to create marginal samples
        z_shuffled = z[torch.randperm(z.size(0))]
        marginal_samples = torch.cat([x, z_shuffled], dim=-1)
        
        # Simple neural network for T_θ
        hidden_dim = joint_samples.size(-1)
        T_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ).to(x.device)
        
        # MINE objective (simplified)
        joint_scores = T_network(joint_samples)
        marginal_scores = T_network(marginal_samples)
        
        mi_estimate = joint_scores.mean() - torch.log(torch.exp(marginal_scores).mean() + 1e-8)
        
        return mi_estimate
    
    def _knn_entropy(self, z: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Estimate differential entropy using k-nearest neighbors"""
        
        n, d = z.shape
        
        # Compute pairwise distances
        distances = torch.cdist(z, z, p=2)
        
        # Find k-th nearest neighbor distance for each point
        knn_distances, _ = torch.topk(distances, k + 1, largest=False, dim=-1)
        rho_k = knn_distances[:, k]  # k-th nearest neighbor distance
        
        # Entropy estimate: H ≈ ψ(k) - ψ(n) + log(V_d) + (d/n) * Σ log(ρ_k)
        # Simplified version
        log_volume_unit_ball = d * np.log(np.pi) / 2 - torch.lgamma(torch.tensor(d/2 + 1))
        entropy = (torch.log(rho_k + 1e-8).mean() * d + 
                  log_volume_unit_ball + 
                  np.log(n-1) - np.log(k))
        
        return entropy

# Example usage and integration
class EnhancedCortexFlowLoss(nn.Module):
    """
    Integrated enhanced loss combining all mathematical contributions
    """
    
    def __init__(self):
        super().__init__()
        self.adaptive_loss = AdaptiveComplexityLoss()
        self.uncertainty_loss = CalibratedUncertaintyLoss()
        self.cross_modal_loss = WassersteinCrossModalLoss()
        self.complexity_measure = InformationTheoreticComplexity()
        
    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        """
        Compute enhanced CortexFlow loss with all mathematical contributions
        """
        
        losses = {}
        
        # Adaptive complexity loss
        if 'simple_pred' in outputs and 'complex_pred' in outputs:
            losses['adaptive'] = self.adaptive_loss(
                outputs['simple_pred'], outputs['complex_pred'], targets,
                outputs['complexity_score'], outputs['simple_params'], outputs['complex_params']
            )
        
        # Calibrated uncertainty loss
        if 'epistemic_var' in outputs and 'aleatoric_var' in outputs:
            losses['uncertainty'] = self.uncertainty_loss(
                outputs['predictions'], targets, outputs['epistemic_var'],
                outputs['aleatoric_var'], outputs['confidence_scores']
            )
        
        # Cross-modal loss
        if 'fmri_features' in outputs and 'eeg_features' in outputs:
            losses['cross_modal'] = self.cross_modal_loss(
                outputs['fmri_features'], outputs['eeg_features'],
                outputs['eeg_to_fmri_transform'], outputs['fmri_to_eeg_transform']
            )
        
        # Information-theoretic complexity
        if 'z_simple' in outputs and 'z_complex' in outputs:
            losses['complexity_mi'] = self.complexity_measure.mutual_information_complexity(
                outputs['input'], outputs['z_simple'], outputs['z_complex']
            )
        
        return losses
