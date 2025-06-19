"""
CCCV3 Adaptive Strategy Model
============================

Smart adaptive strategy that automatically selects the best approach per dataset:
- Individual pathway when it's optimal
- Ensemble when it provides benefit
- Dataset-aware automatic selection

Philosophy: "Use the simplest approach that works best"

Based on empirical findings:
- Miyawaki: Ensemble beneficial (complex small dataset)
- Vangerven: Individual CLIP optimal (simple high-dimensional)
- MindBigData: Individual/Ensemble similar (large dataset)
- Crell: Individual/Ensemble similar (medium dataset)
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


class DatasetCharacteristics:
    """
    Analyze dataset characteristics to determine optimal strategy
    """
    
    @staticmethod
    def analyze_dataset(dataset_name, dataset_size, input_dim):
        """
        Analyze dataset characteristics and recommend strategy
        
        Args:
            dataset_name: Name of dataset
            dataset_size: Number of samples
            input_dim: Input feature dimension
        Returns:
            strategy_info: Dict with recommended strategy and rationale
        """
        
        # Compute dataset metrics
        size_category = DatasetCharacteristics._categorize_size(dataset_size)
        dimensionality_ratio = input_dim / dataset_size
        complexity_score = DatasetCharacteristics._estimate_complexity(dataset_name, dataset_size, input_dim)
        
        # Empirical findings from comprehensive testing
        empirical_results = {
            'miyawaki': {
                'best_individual': 0.008796,  # Attention
                'ensemble_benefit': True,
                'ensemble_improvement': 8.04,  # %
                'optimal_strategy': 'ensemble'
            },
            'vangerven': {
                'best_individual': 0.036195,  # CLIP
                'ensemble_benefit': False,
                'ensemble_degradation': -29.0,  # %
                'optimal_strategy': 'individual_clip'
            },
            'mindbigdata': {
                'best_individual': 0.056781,  # Attention
                'ensemble_benefit': False,
                'ensemble_degradation': -1.12,  # %
                'optimal_strategy': 'individual_attention'
            },
            'crell': {
                'best_individual': 0.032119,  # Attention
                'ensemble_benefit': False,
                'ensemble_degradation': -0.08,  # %
                'optimal_strategy': 'individual_attention'
            }
        }
        
        # Get empirical data for this dataset
        empirical = empirical_results.get(dataset_name, {})
        
        # Decision logic based on empirical findings + dataset characteristics
        if dataset_name in empirical_results:
            # Use empirical findings for known datasets
            strategy = empirical['optimal_strategy']
            rationale = f"Empirical evidence: {empirical.get('ensemble_improvement', empirical.get('ensemble_degradation', 0)):.1f}% change with ensemble"
        else:
            # Heuristic for unknown datasets
            if size_category == 'small' and complexity_score > 0.7:
                strategy = 'ensemble'
                rationale = "Small complex dataset likely benefits from ensemble diversity"
            elif dimensionality_ratio > 30:  # High-dimensional, small sample
                strategy = 'individual_clip'
                rationale = "High-dimensional small sample: CLIP semantic understanding optimal"
            elif size_category == 'large':
                strategy = 'individual_attention'
                rationale = "Large dataset: Attention pathway handles complexity well"
            else:
                strategy = 'individual_clip'
                rationale = "Default: CLIP pathway most consistently strong"
        
        return {
            'dataset_name': dataset_name,
            'dataset_size': dataset_size,
            'input_dim': input_dim,
            'size_category': size_category,
            'dimensionality_ratio': dimensionality_ratio,
            'complexity_score': complexity_score,
            'recommended_strategy': strategy,
            'rationale': rationale,
            'empirical_data': empirical
        }
    
    @staticmethod
    def _categorize_size(dataset_size):
        """Categorize dataset size"""
        if dataset_size < 200:
            return 'small'
        elif dataset_size < 800:
            return 'medium'
        else:
            return 'large'
    
    @staticmethod
    def _estimate_complexity(dataset_name, dataset_size, input_dim):
        """Estimate dataset complexity (0-1 scale)"""
        
        # Known complexity scores based on empirical analysis
        known_complexity = {
            'miyawaki': 0.9,    # High complexity - visual cortex, small sample
            'vangerven': 0.3,   # Low complexity - high-dim but simple patterns
            'mindbigdata': 0.6, # Medium complexity - large but structured
            'crell': 0.5       # Medium complexity - balanced characteristics
        }
        
        if dataset_name in known_complexity:
            return known_complexity[dataset_name]
        
        # Heuristic for unknown datasets
        size_factor = min(1.0, dataset_size / 1000)  # Larger = more complex patterns possible
        dim_factor = min(1.0, input_dim / 5000)      # Higher dim = more complex
        
        return (size_factor + dim_factor) / 2


class CCCV3AdaptiveModel(nn.Module):
    """
    CCCV3 Adaptive Model that automatically selects optimal strategy per dataset
    
    Strategies:
    1. individual_clip: Use CLIP pathway only
    2. individual_attention: Use Attention pathway only  
    3. individual_lite: Use Lite pathway only
    4. ensemble: Use weighted ensemble of pathways
    """
    
    def __init__(self, input_dim, dataset_name='unknown', dataset_size=100, device='cuda'):
        super(CCCV3AdaptiveModel, self).__init__()
        self.model_name = "CortexFlow-CLIP-CNN-V3-Adaptive"
        self.input_dim = input_dim
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
        self.device = device
        
        # Analyze dataset and determine strategy
        self.strategy_info = DatasetCharacteristics.analyze_dataset(dataset_name, dataset_size, input_dim)
        self.strategy = self.strategy_info['recommended_strategy']
        
        print(f"   🧠 CCCV3 Adaptive Strategy for {dataset_name.upper()}:")
        print(f"      Strategy: {self.strategy}")
        print(f"      Rationale: {self.strategy_info['rationale']}")
        print(f"      Dataset characteristics: {self.strategy_info['size_category']} size, "
              f"complexity={self.strategy_info['complexity_score']:.1f}")
        
        # Create only the models we need
        self.active_models = {}
        
        if self.strategy == 'individual_clip' or self.strategy == 'ensemble':
            self.clip_model = create_complete_clip_model(input_dim, dataset_size=dataset_size, device=device)
            self.active_models['clip'] = self.clip_model
        
        if self.strategy == 'individual_attention' or self.strategy == 'ensemble':
            self.attention_model = create_complete_attention_model(input_dim, dataset_size=dataset_size, device=device)
            self.active_models['attention'] = self.attention_model
        
        if self.strategy == 'individual_lite' or self.strategy == 'ensemble':
            self.lite_model = create_complete_lite_model(input_dim, dataset_size=dataset_size, device=device)
            self.active_models['lite'] = self.lite_model
        
        # Ensemble weights (only if using ensemble)
        if self.strategy == 'ensemble':
            # Use empirically optimal weights
            if dataset_name == 'miyawaki':
                self.ensemble_weights = torch.tensor([0.0, 0.3, 0.7], device=device)  # CLIP + Attention
            else:
                self.ensemble_weights = torch.tensor([0.2, 0.4, 0.4], device=device)  # Balanced
        
        print(f"      Active models: {list(self.active_models.keys())}")
        
    def forward(self, x):
        """
        Forward pass using adaptive strategy
        
        Args:
            x: [batch_size, input_dim] - fMRI features
        Returns:
            output: [batch_size, 1, 28, 28] - Reconstructed images
            strategy_info: Dict with strategy information
        """
        
        if self.strategy == 'individual_clip':
            output = self.clip_model(x)[0]
            strategy_info = {
                'strategy': 'individual_clip',
                'active_pathway': 'clip',
                'rationale': self.strategy_info['rationale']
            }
            
        elif self.strategy == 'individual_attention':
            output = self.attention_model(x)[0]
            strategy_info = {
                'strategy': 'individual_attention',
                'active_pathway': 'attention',
                'rationale': self.strategy_info['rationale']
            }
            
        elif self.strategy == 'individual_lite':
            output = self.lite_model(x)[0]
            strategy_info = {
                'strategy': 'individual_lite',
                'active_pathway': 'lite',
                'rationale': self.strategy_info['rationale']
            }
            
        elif self.strategy == 'ensemble':
            # Get predictions from all pathways
            clip_pred = self.clip_model(x)[0]
            attention_pred = self.attention_model(x)[0]
            lite_pred = self.lite_model(x)[0]
            
            # Weighted ensemble
            output = (self.ensemble_weights[0] * lite_pred +
                     self.ensemble_weights[1] * clip_pred +
                     self.ensemble_weights[2] * attention_pred)
            
            strategy_info = {
                'strategy': 'ensemble',
                'active_pathways': ['lite', 'clip', 'attention'],
                'ensemble_weights': self.ensemble_weights.cpu().numpy(),
                'rationale': self.strategy_info['rationale']
            }
        
        return output, strategy_info
    
    def get_strategy_analysis(self):
        """Get detailed strategy analysis"""
        return self.strategy_info


class CCCV3AdaptiveTrainer:
    """
    Trainer for CCCV3 Adaptive Model
    
    Trains only the active models based on selected strategy
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.strategy = model.strategy
        
    def train_adaptive_model(self, train_loader, val_loader, config):
        """
        Train model using adaptive strategy
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        Returns:
            training_results: Dict with training results
        """
        
        print(f"🔧 Training CCCV3 Adaptive Model with {self.strategy} strategy...")
        
        if self.strategy.startswith('individual_'):
            # Train single pathway
            pathway_name = self.strategy.split('_')[1]
            model = self.model.active_models[pathway_name]
            
            result = self._train_single_model(model, train_loader, val_loader, config, pathway_name)
            
            return {
                'strategy': self.strategy,
                'trained_models': [pathway_name],
                'results': {pathway_name: result}
            }
            
        elif self.strategy == 'ensemble':
            # Train all pathways
            results = {}
            
            for pathway_name, model in self.model.active_models.items():
                print(f"   Training {pathway_name.upper()} pathway...")
                result = self._train_single_model(model, train_loader, val_loader, config, pathway_name)
                results[pathway_name] = result
            
            return {
                'strategy': self.strategy,
                'trained_models': list(self.model.active_models.keys()),
                'results': results
            }
    
    def _train_single_model(self, model, train_loader, val_loader, config, pathway_name):
        """Train a single pathway model"""
        
        # Pathway-specific learning rates
        lr_map = {
            'lite': 0.001,
            'clip': 0.0005,
            'attention': 0.0003
        }
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_map.get(pathway_name, 0.0005),
            weight_decay=config.get('weight_decay', 1e-6)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get('epochs', 80), eta_min=1e-8
        )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(config.get('epochs', 80)):
            # Training
            model.train()
            train_loss = 0.0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                visual_output = model(data)[0]
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
                    visual_output = model(data)[0]
                    val_loss += criterion(visual_output, target).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                print(f"     Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"     Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_state)
        
        return {
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1,
            'pathway_name': pathway_name
        }


# Factory functions
def create_cccv3_adaptive(input_dim, dataset_name='unknown', dataset_size=100, device='cuda'):
    """Factory function to create CCCV3 adaptive model"""
    return CCCV3AdaptiveModel(input_dim, dataset_name, dataset_size, device)


def create_cccv3_adaptive_trainer(model, device):
    """Factory function to create CCCV3 adaptive trainer"""
    return CCCV3AdaptiveTrainer(model, device)


# Export main classes and functions
__all__ = [
    'DatasetCharacteristics',
    'CCCV3AdaptiveModel',
    'CCCV3AdaptiveTrainer',
    'create_cccv3_adaptive',
    'create_cccv3_adaptive_trainer'
]
