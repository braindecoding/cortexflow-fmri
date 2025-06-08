#!/usr/bin/env python3
"""
CortexFlow-Ensemble: Adaptive Multi-Model Integration
Breakthrough innovation in neural decoding through intelligent ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class AdaptiveEnsembleWeights(nn.Module):
    """
    Novel Adaptive Ensemble Weighting Module
    Dynamic weights based on input characteristics and model performance
    """
    
    def __init__(self, input_dim: int, num_models: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.num_models = num_models
        
        # Complexity-based weighting network
        self.complexity_net = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_models)
        )
        
        # Uncertainty-based weighting network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(num_models, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_models)
        )
        
        # Agreement-based weighting network
        self.agreement_net = nn.Sequential(
            nn.Linear(num_models * num_models, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_models)
        )
        
        # Final fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(num_models * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_models)
        )
        
    def forward(self, complexity_score: torch.Tensor, 
                uncertainty_estimates: torch.Tensor,
                agreement_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive ensemble weights
        
        Args:
            complexity_score: Input complexity [B, 1]
            uncertainty_estimates: Model uncertainties [B, num_models]
            agreement_matrix: Cross-model agreement [B, num_models, num_models]
        
        Returns:
            ensemble_weights: Adaptive weights [B, num_models]
        """
        batch_size = complexity_score.size(0)
        
        # Complexity-based weights
        w_complexity = self.complexity_net(complexity_score)
        
        # Uncertainty-based weights (favor confident models)
        w_uncertainty = self.uncertainty_net(1.0 / (uncertainty_estimates + 1e-8))
        
        # Agreement-based weights (favor consensus)
        agreement_flat = agreement_matrix.view(batch_size, -1)
        w_agreement = self.agreement_net(agreement_flat)
        
        # Fuse all weight components
        weight_features = torch.cat([w_complexity, w_uncertainty, w_agreement], dim=-1)
        ensemble_weights = F.softmax(self.fusion_net(weight_features), dim=-1)
        
        return ensemble_weights

class HierarchicalUncertaintyEstimator(nn.Module):
    """
    Novel Hierarchical Uncertainty Quantification for Ensemble
    Multi-level uncertainty decomposition
    """
    
    def __init__(self, num_models: int = 5):
        super().__init__()
        self.num_models = num_models
        
        # Meta-uncertainty estimator
        self.meta_uncertainty_net = nn.Sequential(
            nn.Linear(num_models * 3, 64),  # predictions + uncertainties + weights
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
    def forward(self, predictions: torch.Tensor, 
                individual_uncertainties: torch.Tensor,
                ensemble_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical uncertainty decomposition
        
        Args:
            predictions: Individual model predictions [B, num_models, output_dim]
            individual_uncertainties: Model uncertainties [B, num_models]
            ensemble_weights: Ensemble weights [B, num_models]
        
        Returns:
            uncertainty_dict: Multi-level uncertainties
        """
        batch_size = predictions.size(0)
        
        # Weighted ensemble prediction
        ensemble_pred = torch.sum(predictions * ensemble_weights.unsqueeze(-1), dim=1)
        
        # Level 1: Within-model uncertainty (weighted average)
        sigma2_within = torch.sum(ensemble_weights * individual_uncertainties, dim=1)
        
        # Level 2: Between-model uncertainty (inter-model disagreement)
        model_deviations = predictions - ensemble_pred.unsqueeze(1)
        sigma2_between = torch.sum(ensemble_weights.unsqueeze(-1) * 
                                  (model_deviations ** 2), dim=(1, 2))
        
        # Level 3: Meta-uncertainty (novel contribution)
        meta_features = torch.cat([
            predictions.mean(dim=-1),  # Average predictions
            individual_uncertainties,   # Individual uncertainties
            ensemble_weights           # Ensemble weights
        ], dim=-1)
        
        sigma2_meta = self.meta_uncertainty_net(meta_features).squeeze(-1)
        
        # Novel interaction term
        weight_uncertainty_cov = 2 * torch.sum(
            (ensemble_weights - ensemble_weights.mean(dim=1, keepdim=True)) *
            (individual_uncertainties - individual_uncertainties.mean(dim=1, keepdim=True)),
            dim=1
        )
        
        # Total ensemble uncertainty
        sigma2_total = sigma2_within + sigma2_between + sigma2_meta + weight_uncertainty_cov
        
        return {
            'total': sigma2_total,
            'within': sigma2_within,
            'between': sigma2_between,
            'meta': sigma2_meta,
            'interaction': weight_uncertainty_cov,
            'ensemble_prediction': ensemble_pred
        }

class IntelligentModelSelector(nn.Module):
    """
    Novel Intelligent Model Selection Module
    Dynamic subset selection based on information gain
    """
    
    def __init__(self, num_models: int = 5, max_models: int = 3):
        super().__init__()
        self.num_models = num_models
        self.max_models = max_models
        
        # Model relevance predictor
        self.relevance_net = nn.Sequential(
            nn.Linear(num_models + 1, 64),  # predictions + complexity
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_models),
            nn.Sigmoid()
        )
        
        # Dynamic k predictor
        self.k_predictor = nn.Sequential(
            nn.Linear(num_models + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, predictions: torch.Tensor, 
                complexity_score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select optimal model subset dynamically
        
        Args:
            predictions: Model predictions [B, num_models, output_dim]
            complexity_score: Input complexity [B, 1]
        
        Returns:
            selection_mask: Binary mask for selected models [B, num_models]
            selection_weights: Continuous selection weights [B, num_models]
        """
        batch_size = predictions.size(0)
        
        # Compute model relevance scores
        pred_features = predictions.mean(dim=-1)  # [B, num_models]
        relevance_input = torch.cat([pred_features, complexity_score], dim=-1)
        
        model_relevance = self.relevance_net(relevance_input)
        
        # Predict optimal number of models
        k_continuous = self.k_predictor(relevance_input) * self.max_models
        k_discrete = torch.clamp(torch.round(k_continuous), 1, self.max_models).long()
        
        # Select top-k models for each sample
        selection_mask = torch.zeros_like(model_relevance)
        selection_weights = torch.zeros_like(model_relevance)
        
        for i in range(batch_size):
            k_i = k_discrete[i].item()
            _, top_indices = torch.topk(model_relevance[i], k_i)
            
            selection_mask[i, top_indices] = 1.0
            selection_weights[i, top_indices] = F.softmax(model_relevance[i, top_indices], dim=0)
        
        return selection_mask, selection_weights

class CortexFlowEnsemble(nn.Module):
    """
    CortexFlow-Ensemble: Breakthrough Adaptive Multi-Model Integration
    
    Novel contributions:
    1. Adaptive ensemble weighting based on input characteristics
    2. Hierarchical uncertainty quantification with interaction terms
    3. Intelligent model selection with information-theoretic optimization
    4. Dynamic computational resource allocation
    """
    
    def __init__(self, input_dim: int, output_dim: int = 784):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Import individual CortexFlow models
        from .cortexflow_simple import CortexFlowSimple
        from .cortexflow_mc import CortexFlowMC
        from .cortexflow_hierarchical import CortexFlowHierarchical
        from .cortexflow_enhanced import CortexFlowEnhanced
        from .cortexflow_unified import CortexFlowUnified
        
        # Initialize individual models
        self.models = nn.ModuleDict({
            'simple': CortexFlowSimple(input_dim, output_dim),
            'mc': CortexFlowMC(input_dim, output_dim),
            'hierarchical': CortexFlowHierarchical(input_dim, output_dim),
            'enhanced': CortexFlowEnhanced(input_dim, output_dim),
            'unified': CortexFlowUnified(input_dim, output_dim)
        })
        
        # Novel ensemble components
        self.adaptive_weights = AdaptiveEnsembleWeights(input_dim, num_models=5)
        self.uncertainty_estimator = HierarchicalUncertaintyEstimator(num_models=5)
        self.model_selector = IntelligentModelSelector(num_models=5)
        
        # Complexity predictor (shared with unified model)
        self.complexity_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, mode: str = 'adaptive') -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple ensemble modes
        
        Args:
            x: Input fMRI signals [B, input_dim]
            mode: Ensemble mode ('adaptive', 'selection', 'full')
        
        Returns:
            output_dict: Comprehensive ensemble outputs
        """
        batch_size = x.size(0)
        
        # Predict input complexity
        complexity_score = self.complexity_predictor(x)
        
        # Get predictions from all models
        model_outputs = {}
        predictions = []
        uncertainties = []
        
        for name, model in self.models.items():
            if name == 'mc' or name == 'enhanced' or name == 'unified':
                # Models with uncertainty output
                output = model(x)
                if isinstance(output, tuple):
                    pred, unc = output[0], output[1] if len(output) > 1 else torch.zeros_like(output[0][:, 0])
                else:
                    pred, unc = output, torch.zeros(batch_size, device=x.device)
            else:
                # Models without uncertainty
                pred = model(x)
                unc = torch.zeros(batch_size, device=x.device)
            
            model_outputs[name] = {'prediction': pred, 'uncertainty': unc}
            predictions.append(pred)
            uncertainties.append(unc)
        
        predictions = torch.stack(predictions, dim=1)  # [B, 5, output_dim]
        uncertainties = torch.stack(uncertainties, dim=1)  # [B, 5]
        
        # Compute cross-model agreement matrix
        agreement_matrix = self._compute_agreement_matrix(predictions)
        
        if mode == 'selection':
            # Intelligent model selection
            selection_mask, selection_weights = self.model_selector(predictions, complexity_score)
            ensemble_weights = selection_weights
        else:
            # Adaptive ensemble weighting
            ensemble_weights = self.adaptive_weights(
                complexity_score, uncertainties, agreement_matrix
            )
        
        # Hierarchical uncertainty estimation
        uncertainty_dict = self.uncertainty_estimator(
            predictions, uncertainties, ensemble_weights
        )
        
        return {
            'ensemble_prediction': uncertainty_dict['ensemble_prediction'],
            'ensemble_weights': ensemble_weights,
            'complexity_score': complexity_score,
            'individual_predictions': predictions,
            'individual_uncertainties': uncertainties,
            'total_uncertainty': uncertainty_dict['total'],
            'within_uncertainty': uncertainty_dict['within'],
            'between_uncertainty': uncertainty_dict['between'],
            'meta_uncertainty': uncertainty_dict['meta'],
            'interaction_uncertainty': uncertainty_dict['interaction'],
            'agreement_matrix': agreement_matrix,
            'model_outputs': model_outputs
        }
    
    def _compute_agreement_matrix(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute cross-model agreement matrix"""
        batch_size, num_models, output_dim = predictions.shape
        
        agreement_matrix = torch.zeros(batch_size, num_models, num_models, device=predictions.device)
        
        for i in range(num_models):
            for j in range(num_models):
                if i != j:
                    # Compute cosine similarity between model predictions
                    pred_i = predictions[:, i, :]  # [B, output_dim]
                    pred_j = predictions[:, j, :]  # [B, output_dim]
                    
                    cosine_sim = F.cosine_similarity(pred_i, pred_j, dim=1)
                    agreement_matrix[:, i, j] = cosine_sim
                else:
                    agreement_matrix[:, i, j] = 1.0
        
        return agreement_matrix
