"""
CCCV3 Ensemble Model
===================

Ensemble approach for CCCV3 that combines predictions from individual pathways
instead of feature-level fusion. This approach is simpler and more effective.

Strategy:
1. Train individual pathways separately
2. Combine predictions using optimal weights
3. Use best pathway per dataset as primary + others as enhancement

Goal: Achieve 100% success rate by leveraging best of each pathway
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import individual pathway models
try:
    from .lite_pathway import create_complete_lite_model
    from .clip_pathway import create_complete_clip_model
    from .attention_pathway import create_complete_attention_model
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from lite_pathway import create_complete_lite_model
    from clip_pathway import create_complete_clip_model
    from attention_pathway import create_complete_attention_model


class CCCV3EnsembleModel(nn.Module):
    """
    CCCV3 Ensemble Model that combines individual pathway predictions
    
    Strategy:
    - Each pathway trained independently to optimal performance
    - Predictions combined using dataset-specific optimal weights
    - Simple but effective ensemble approach
    """
    
    def __init__(self, input_dim, dataset_name='unknown', dataset_size=100, device='cuda'):
        super(CCCV3EnsembleModel, self).__init__()
        self.model_name = "CortexFlow-CLIP-CNN-V3-Ensemble"
        self.input_dim = input_dim
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
        self.device = device
        
        # Create individual pathway models
        self.lite_model = create_complete_lite_model(
            input_dim, dataset_size=dataset_size, device=device
        )
        
        self.clip_model = create_complete_clip_model(
            input_dim, dataset_size=dataset_size, device=device
        )
        
        self.attention_model = create_complete_attention_model(
            input_dim, dataset_size=dataset_size, device=device
        )
        
        # Ensemble weights based on individual pathway performance
        self.ensemble_weights = self._get_optimal_weights(dataset_name)
        
        print(f"   🎯 CCCV3 Ensemble for {dataset_name.upper()}:")
        print(f"      Ensemble weights: Lite={self.ensemble_weights[0]:.2f}, "
              f"CLIP={self.ensemble_weights[1]:.2f}, Attention={self.ensemble_weights[2]:.2f}")
        
    def _get_optimal_weights(self, dataset_name):
        """Get optimal ensemble weights based on individual pathway performance"""
        
        # Weights based on individual pathway testing results
        optimal_weights = {
            'miyawaki': [0.0, 0.3, 0.7],    # Attention (0.008796) + CLIP (0.011549) best
            'vangerven': [0.0, 1.0, 0.0],   # CLIP (0.036195) clearly best
            'mindbigdata': [0.0, 0.3, 0.7], # Attention (0.056781) + CLIP (0.057007) best
            'crell': [0.0, 0.4, 0.6]        # Attention (0.032119) + CLIP (0.032127) best
        }
        
        weights = optimal_weights.get(dataset_name, [0.2, 0.4, 0.4])  # Default balanced
        return torch.tensor(weights, device=self.device, dtype=torch.float32)
    
    def forward(self, x, return_individual_predictions=False):
        """
        Forward pass through ensemble
        
        Args:
            x: [batch_size, input_dim] - fMRI features
            return_individual_predictions: Whether to return individual pathway predictions
        Returns:
            ensemble_output: [batch_size, 1, 28, 28] - Ensemble prediction
            individual_info: Dict with individual pathway information (if requested)
        """
        # Get predictions from individual pathways
        lite_outputs = self.lite_model(x)
        clip_outputs = self.clip_model(x)
        attention_outputs = self.attention_model(x)
        
        # Extract visual predictions
        lite_pred = lite_outputs[0]      # [batch_size, 1, 28, 28]
        clip_pred = clip_outputs[0]      # [batch_size, 1, 28, 28]
        attention_pred = attention_outputs[0]  # [batch_size, 1, 28, 28]
        
        # Weighted ensemble
        ensemble_output = (self.ensemble_weights[0] * lite_pred +
                          self.ensemble_weights[1] * clip_pred +
                          self.ensemble_weights[2] * attention_pred)
        
        if return_individual_predictions:
            individual_info = {
                'lite_prediction': lite_pred,
                'clip_prediction': clip_pred,
                'attention_prediction': attention_pred,
                'ensemble_weights': self.ensemble_weights,
                'lite_features': lite_outputs[1] if len(lite_outputs) > 1 else None,
                'clip_features': clip_outputs[1] if len(clip_outputs) > 1 else None,
                'attention_features': attention_outputs[1] if len(attention_outputs) > 1 else None
            }
            return ensemble_output, individual_info
        else:
            return ensemble_output
    
    def get_individual_performances(self, test_loader, device):
        """
        Evaluate individual pathway performances for analysis
        
        Args:
            test_loader: DataLoader for test data
            device: Device for computation
        Returns:
            performances: Dict with individual pathway MSE scores
        """
        self.eval()
        criterion = nn.MSELoss()
        
        lite_losses = []
        clip_losses = []
        attention_losses = []
        ensemble_losses = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # Get ensemble prediction and individual predictions
                ensemble_pred, individual_info = self.forward(data, return_individual_predictions=True)
                
                # Compute individual losses
                lite_loss = criterion(individual_info['lite_prediction'], target)
                clip_loss = criterion(individual_info['clip_prediction'], target)
                attention_loss = criterion(individual_info['attention_prediction'], target)
                ensemble_loss = criterion(ensemble_pred, target)
                
                lite_losses.append(lite_loss.item())
                clip_losses.append(clip_loss.item())
                attention_losses.append(attention_loss.item())
                ensemble_losses.append(ensemble_loss.item())
        
        performances = {
            'lite_mse': np.mean(lite_losses),
            'clip_mse': np.mean(clip_losses),
            'attention_mse': np.mean(attention_losses),
            'ensemble_mse': np.mean(ensemble_losses),
            'best_individual': min(np.mean(lite_losses), np.mean(clip_losses), np.mean(attention_losses))
        }
        
        return performances


class CCCV3EnsembleTrainer:
    """
    Specialized trainer for CCCV3 Ensemble model
    
    Strategy:
    1. Train individual pathways separately
    2. No ensemble training needed (just weighted combination)
    3. Focus on optimizing individual pathway performance
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def train_individual_pathways(self, train_loader, val_loader, config):
        """
        Train individual pathways separately for optimal performance
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        Returns:
            training_results: Dict with training results for each pathway
        """
        results = {}
        
        # Train each pathway individually
        pathways = [
            ('lite', self.model.lite_model),
            ('clip', self.model.clip_model),
            ('attention', self.model.attention_model)
        ]
        
        for pathway_name, pathway_model in pathways:
            print(f"\n🔧 Training {pathway_name.upper()} pathway...")
            
            # Individual pathway training
            pathway_result = self._train_single_pathway(
                pathway_model, train_loader, val_loader, config, pathway_name
            )
            
            results[pathway_name] = pathway_result
            print(f"   {pathway_name.upper()} Val Loss: {pathway_result['best_val_loss']:.6f}")
        
        return results
    
    def _train_single_pathway(self, model, train_loader, val_loader, config, pathway_name):
        """Train a single pathway model"""
        
        # Pathway-specific learning rates
        lr_map = {
            'lite': 0.001,      # Higher LR for simpler model
            'clip': 0.0005,     # Medium LR for CLIP model
            'attention': 0.0003  # Lower LR for complex attention model
        }
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_map.get(pathway_name, 0.0005),
            weight_decay=1e-6,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get('epochs', 100), eta_min=1e-8
        )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(config.get('epochs', 100)):
            # Training
            model.train()
            train_loss = 0.0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(data)
                visual_output = outputs[0]  # First output is visual prediction
                
                loss = criterion(visual_output, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    outputs = model(data)
                    visual_output = outputs[0]
                    val_loss += criterion(visual_output, target).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                print(f"     Epoch {epoch+1:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"     Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return {
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1,
            'pathway_name': pathway_name
        }


# Factory functions
def create_cccv3_ensemble(input_dim, dataset_name='unknown', dataset_size=100, device='cuda'):
    """Factory function to create CCCV3 ensemble model"""
    return CCCV3EnsembleModel(input_dim, dataset_name, dataset_size, device)


def create_cccv3_ensemble_trainer(model, device):
    """Factory function to create CCCV3 ensemble trainer"""
    return CCCV3EnsembleTrainer(model, device)


# Ensemble configuration for different datasets
def get_cccv3_ensemble_config(dataset_name):
    """Get optimal ensemble configuration for each dataset"""
    
    configs = {
        'miyawaki': {
            'primary_pathway': 'attention',
            'ensemble_strategy': 'attention_dominant',
            'expected_improvement': '20-30%',
            'training_epochs': 120,
            'rationale': 'Attention pathway excels (38.81% better), CLIP provides semantic support'
        },
        'vangerven': {
            'primary_pathway': 'clip',
            'ensemble_strategy': 'clip_only',
            'expected_improvement': '0-5%',
            'training_epochs': 100,
            'rationale': 'CLIP pathway clearly best, no ensemble needed'
        },
        'mindbigdata': {
            'primary_pathway': 'attention',
            'ensemble_strategy': 'attention_clip_balanced',
            'expected_improvement': '2-5%',
            'training_epochs': 80,
            'rationale': 'Attention + CLIP combination for large dataset'
        },
        'crell': {
            'primary_pathway': 'attention',
            'ensemble_strategy': 'attention_clip_balanced',
            'expected_improvement': '3-8%',
            'training_epochs': 100,
            'rationale': 'Balanced attention + CLIP for medium complexity'
        }
    }
    
    return configs.get(dataset_name, configs['crell'])


# Export main classes and functions
__all__ = [
    'CCCV3EnsembleModel',
    'CCCV3EnsembleTrainer',
    'create_cccv3_ensemble',
    'create_cccv3_ensemble_trainer',
    'get_cccv3_ensemble_config'
]
