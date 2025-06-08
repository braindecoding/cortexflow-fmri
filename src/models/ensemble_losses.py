#!/usr/bin/env python3
"""
Novel Ensemble Loss Functions for CortexFlow-Ensemble
Mathematical formulations for breakthrough ensemble training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class AdaptiveEnsembleLoss(nn.Module):
    """
    Novel Adaptive Ensemble Loss Function
    Combines reconstruction, diversity, and agreement objectives
    """
    
    def __init__(self, lambda_diversity: float = 0.1, lambda_agreement: float = 0.05,
                 lambda_complexity: float = 0.02, lambda_uncertainty: float = 0.1):
        super().__init__()
        self.lambda_diversity = lambda_diversity
        self.lambda_agreement = lambda_agreement
        self.lambda_complexity = lambda_complexity
        self.lambda_uncertainty = lambda_uncertainty
        
    def forward(self, ensemble_outputs: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive ensemble loss
        
        Args:
            ensemble_outputs: Output dictionary from CortexFlowEnsemble
            targets: Ground truth targets [B, output_dim]
        
        Returns:
            loss_dict: Comprehensive loss breakdown
        """
        
        # Primary reconstruction loss
        L_reconstruction = F.mse_loss(ensemble_outputs['ensemble_prediction'], targets)
        
        # Diversity loss (encourage model diversity)
        L_diversity = self._diversity_loss(
            ensemble_outputs['individual_predictions'],
            ensemble_outputs['ensemble_weights']
        )
        
        # Agreement loss (penalize disagreement when confident)
        L_agreement = self._agreement_loss(
            ensemble_outputs['individual_predictions'],
            ensemble_outputs['ensemble_weights'],
            ensemble_outputs['agreement_matrix']
        )
        
        # Complexity regularization
        L_complexity = self._complexity_regularization(
            ensemble_outputs['complexity_score'],
            ensemble_outputs['ensemble_weights']
        )
        
        # Uncertainty calibration loss
        L_uncertainty = self._uncertainty_calibration_loss(
            ensemble_outputs['ensemble_prediction'],
            targets,
            ensemble_outputs['total_uncertainty']
        )
        
        # Total adaptive ensemble loss
        L_total = (L_reconstruction + 
                  self.lambda_diversity * L_diversity +
                  self.lambda_agreement * L_agreement +
                  self.lambda_complexity * L_complexity +
                  self.lambda_uncertainty * L_uncertainty)
        
        return {
            'total_loss': L_total,
            'reconstruction_loss': L_reconstruction,
            'diversity_loss': L_diversity,
            'agreement_loss': L_agreement,
            'complexity_loss': L_complexity,
            'uncertainty_loss': L_uncertainty
        }
    
    def _diversity_loss(self, predictions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Encourage diversity among ensemble models
        L_diversity = -Σ_{i≠j} w_i * w_j * correlation(pred_i, pred_j)
        """
        batch_size, num_models, output_dim = predictions.shape
        
        diversity_loss = 0.0
        for i in range(num_models):
            for j in range(i + 1, num_models):
                # Compute correlation between model i and j
                pred_i = predictions[:, i, :]  # [B, output_dim]
                pred_j = predictions[:, j, :]  # [B, output_dim]
                
                # Pearson correlation coefficient
                correlation = self._pearson_correlation(pred_i, pred_j)
                
                # Weight by ensemble weights
                weight_product = weights[:, i] * weights[:, j]
                
                # Negative correlation encourages diversity
                diversity_loss += (weight_product * correlation).mean()
        
        return diversity_loss
    
    def _agreement_loss(self, predictions: torch.Tensor, weights: torch.Tensor,
                       agreement_matrix: torch.Tensor) -> torch.Tensor:
        """
        Penalize disagreement when models are confident
        L_agreement = Σ_{i,j} w_i * w_j * ||pred_i - pred_j||² * confidence_factor
        """
        batch_size, num_models, output_dim = predictions.shape
        
        agreement_loss = 0.0
        for i in range(num_models):
            for j in range(i + 1, num_models):
                pred_i = predictions[:, i, :]
                pred_j = predictions[:, j, :]
                
                # L2 distance between predictions
                pred_distance = F.mse_loss(pred_i, pred_j, reduction='none').mean(dim=1)
                
                # Weight by ensemble weights and agreement
                weight_product = weights[:, i] * weights[:, j]
                agreement_factor = agreement_matrix[:, i, j]
                
                # Higher penalty when models should agree but don't
                agreement_loss += (weight_product * agreement_factor * pred_distance).mean()
        
        return agreement_loss
    
    def _complexity_regularization(self, complexity_score: torch.Tensor,
                                 ensemble_weights: torch.Tensor) -> torch.Tensor:
        """
        Regularize ensemble complexity based on input complexity
        Encourage simpler ensembles for simple inputs
        """
        # Entropy of ensemble weights (higher = more complex ensemble)
        weight_entropy = -torch.sum(ensemble_weights * torch.log(ensemble_weights + 1e-8), dim=1)
        
        # Penalize high entropy when input is simple
        complexity_penalty = (1 - complexity_score.squeeze()) * weight_entropy
        
        return complexity_penalty.mean()
    
    def _uncertainty_calibration_loss(self, predictions: torch.Tensor,
                                    targets: torch.Tensor,
                                    uncertainties: torch.Tensor) -> torch.Tensor:
        """
        Ensure uncertainty estimates are well-calibrated
        """
        # Prediction errors
        errors = F.mse_loss(predictions, targets, reduction='none').mean(dim=1)
        
        # Uncertainty should correlate with errors
        # Use negative log-likelihood for calibration
        nll = torch.log(uncertainties + 1e-8) + errors / (uncertainties + 1e-8)
        
        return nll.mean()
    
    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation coefficient"""
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)
        
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        numerator = (x_centered * y_centered).sum(dim=1)
        denominator = torch.sqrt((x_centered ** 2).sum(dim=1) * (y_centered ** 2).sum(dim=1))
        
        correlation = numerator / (denominator + 1e-8)
        return correlation

class InformationTheoreticEnsembleLoss(nn.Module):
    """
    Novel Information-Theoretic Loss for Ensemble Optimization
    Based on mutual information and entropy maximization
    """
    
    def __init__(self, lambda_info: float = 0.1):
        super().__init__()
        self.lambda_info = lambda_info
        
    def forward(self, ensemble_outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute information-theoretic ensemble loss
        """
        predictions = ensemble_outputs['individual_predictions']
        weights = ensemble_outputs['ensemble_weights']
        
        # Base reconstruction loss
        ensemble_pred = ensemble_outputs['ensemble_prediction']
        L_base = F.mse_loss(ensemble_pred, targets)
        
        # Information gain loss
        L_info = self._information_gain_loss(predictions, weights, targets)
        
        return L_base + self.lambda_info * L_info
    
    def _information_gain_loss(self, predictions: torch.Tensor,
                             weights: torch.Tensor,
                             targets: torch.Tensor) -> torch.Tensor:
        """
        Maximize information gain from ensemble
        I_gain = H(Y|X) - H(Y|X, ensemble)
        """
        batch_size, num_models, output_dim = predictions.shape
        
        # Estimate entropy of individual models
        individual_entropies = []
        for i in range(num_models):
            pred_i = predictions[:, i, :]
            entropy_i = self._estimate_entropy(pred_i, targets)
            individual_entropies.append(entropy_i)
        
        # Weighted average of individual entropies
        avg_individual_entropy = sum(w * h for w, h in zip(weights.T, individual_entropies))
        
        # Ensemble entropy
        ensemble_pred = torch.sum(predictions * weights.unsqueeze(-1), dim=1)
        ensemble_entropy = self._estimate_entropy(ensemble_pred, targets)
        
        # Information gain (negative because we want to maximize)
        info_gain = avg_individual_entropy.mean() - ensemble_entropy
        
        return -info_gain  # Negative to maximize
    
    def _estimate_entropy(self, predictions: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
        """Estimate entropy using prediction errors"""
        errors = F.mse_loss(predictions, targets, reduction='none').mean(dim=1)
        
        # Use error variance as entropy proxy
        entropy = torch.var(errors) + 1e-8
        
        return torch.log(entropy)

class BayesianEnsembleLoss(nn.Module):
    """
    Novel Bayesian Ensemble Loss
    Incorporates prior knowledge about ensemble weights
    """
    
    def __init__(self, lambda_prior: float = 0.05, prior_concentration: float = 1.0):
        super().__init__()
        self.lambda_prior = lambda_prior
        self.prior_concentration = prior_concentration
        
    def forward(self, ensemble_outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Bayesian ensemble loss with Dirichlet prior
        """
        weights = ensemble_outputs['ensemble_weights']
        ensemble_pred = ensemble_outputs['ensemble_prediction']
        
        # Likelihood term
        L_likelihood = F.mse_loss(ensemble_pred, targets)
        
        # Prior term (Dirichlet prior on weights)
        L_prior = self._dirichlet_prior_loss(weights)
        
        return L_likelihood + self.lambda_prior * L_prior
    
    def _dirichlet_prior_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Dirichlet prior loss: -log p(w|α)
        where α is concentration parameter
        """
        # Dirichlet log probability (simplified)
        alpha = torch.full_like(weights, self.prior_concentration)
        
        # Log probability of Dirichlet distribution
        log_prob = torch.sum((alpha - 1) * torch.log(weights + 1e-8), dim=1)
        
        # Negative log probability (to minimize)
        return -log_prob.mean()

class ComprehensiveEnsembleLoss(nn.Module):
    """
    Comprehensive Ensemble Loss combining all novel formulations
    """
    
    def __init__(self):
        super().__init__()
        self.adaptive_loss = AdaptiveEnsembleLoss()
        self.info_loss = InformationTheoreticEnsembleLoss()
        self.bayesian_loss = BayesianEnsembleLoss()
        
    def forward(self, ensemble_outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive ensemble loss
        """
        # Adaptive ensemble loss (primary)
        adaptive_losses = self.adaptive_loss(ensemble_outputs, targets)
        
        # Information-theoretic loss
        info_loss = self.info_loss(ensemble_outputs, targets)
        
        # Bayesian loss
        bayesian_loss = self.bayesian_loss(ensemble_outputs, targets)
        
        # Combined loss
        total_loss = (adaptive_losses['total_loss'] + 
                     0.1 * info_loss + 
                     0.05 * bayesian_loss)
        
        return {
            'total_loss': total_loss,
            'adaptive_loss': adaptive_losses['total_loss'],
            'info_loss': info_loss,
            'bayesian_loss': bayesian_loss,
            'reconstruction_loss': adaptive_losses['reconstruction_loss'],
            'diversity_loss': adaptive_losses['diversity_loss'],
            'agreement_loss': adaptive_losses['agreement_loss'],
            'complexity_loss': adaptive_losses['complexity_loss'],
            'uncertainty_loss': adaptive_losses['uncertainty_loss']
        }
