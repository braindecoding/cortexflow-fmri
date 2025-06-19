"""
CCCV3 Ultimate Adaptive Model
============================

Ultimate adaptive strategy with transfer learning from best individual models.

Strategy:
1. Pre-train individual pathways to optimal performance
2. Store best model templates per dataset type
3. Transfer optimal weights when using adaptive strategy
4. Fine-tune in adaptive context for perfect performance

Goal: Achieve best individual performance + adaptive efficiency

Philosophy: "Transfer excellence, adapt intelligently"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle

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


class ModelTemplateManager:
    """
    Manages pre-trained model templates for transfer learning
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.templates = {}
        self.template_dir = "cccv3/model_templates"
        
        # Ensure template directory exists
        try:
            os.makedirs(self.template_dir, exist_ok=True)
        except Exception:
            self.template_dir = "/tmp/cccv3_templates"
            os.makedirs(self.template_dir, exist_ok=True)
        
        # Best individual performance targets (from comprehensive testing)
        self.performance_targets = {
            'miyawaki': {
                'clip': 0.011549,
                'attention': 0.008796,  # Best
                'lite': 0.039827
            },
            'vangerven': {
                'clip': 0.036195,  # Best
                'attention': 0.039399,
                'lite': 0.042930
            },
            'mindbigdata': {
                'clip': 0.057007,
                'attention': 0.056781,  # Best
                'lite': 0.058857
            },
            'crell': {
                'clip': 0.032127,
                'attention': 0.032119,  # Best
                'lite': 0.034691
            }
        }
    
    def create_optimal_template(self, pathway_name, input_dim, dataset_name, dataset_size):
        """
        Create optimally configured model template
        
        Args:
            pathway_name: 'clip', 'attention', or 'lite'
            input_dim: Input feature dimension
            dataset_name: Target dataset name
            dataset_size: Target dataset size
        Returns:
            model: Optimally configured model
        """
        
        if pathway_name == 'clip':
            model = create_complete_clip_model(input_dim, dataset_size=dataset_size, device=self.device)
        elif pathway_name == 'attention':
            model = create_complete_attention_model(input_dim, dataset_size=dataset_size, device=self.device)
        elif pathway_name == 'lite':
            model = create_complete_lite_model(input_dim, dataset_size=dataset_size, device=self.device)
        else:
            raise ValueError(f"Unknown pathway: {pathway_name}")
        
        return model
    
    def get_optimal_hyperparameters(self, pathway_name, dataset_name):
        """Get optimal hyperparameters for pathway-dataset combination"""
        
        # Empirically optimized hyperparameters
        hyperparams = {
            'clip': {
                'miyawaki': {'lr': 0.0003, 'weight_decay': 1e-7, 'epochs': 120},
                'vangerven': {'lr': 0.0002, 'weight_decay': 1e-8, 'epochs': 100},
                'mindbigdata': {'lr': 0.0005, 'weight_decay': 1e-6, 'epochs': 80},
                'crell': {'lr': 0.0004, 'weight_decay': 1e-6, 'epochs': 100}
            },
            'attention': {
                'miyawaki': {'lr': 0.0002, 'weight_decay': 1e-8, 'epochs': 120},
                'vangerven': {'lr': 0.0003, 'weight_decay': 1e-7, 'epochs': 100},
                'mindbigdata': {'lr': 0.0003, 'weight_decay': 1e-6, 'epochs': 80},
                'crell': {'lr': 0.0003, 'weight_decay': 1e-6, 'epochs': 100}
            },
            'lite': {
                'miyawaki': {'lr': 0.001, 'weight_decay': 1e-6, 'epochs': 100},
                'vangerven': {'lr': 0.0008, 'weight_decay': 1e-6, 'epochs': 100},
                'mindbigdata': {'lr': 0.001, 'weight_decay': 1e-6, 'epochs': 60},
                'crell': {'lr': 0.001, 'weight_decay': 1e-6, 'epochs': 80}
            }
        }
        
        return hyperparams.get(pathway_name, {}).get(dataset_name, 
                                                    {'lr': 0.0005, 'weight_decay': 1e-6, 'epochs': 80})
    
    def save_template(self, model, pathway_name, dataset_name, performance):
        """Save model template with performance metadata"""
        
        template_path = os.path.join(self.template_dir, f"{pathway_name}_{dataset_name}_template.pth")
        
        template_data = {
            'model_state_dict': model.state_dict(),
            'pathway_name': pathway_name,
            'dataset_name': dataset_name,
            'performance': performance,
            'model_config': {
                'input_dim': getattr(model, 'input_dim', None),
                'model_name': getattr(model, 'model_name', f"{pathway_name}_model")
            }
        }
        
        torch.save(template_data, template_path)
        print(f"   💾 Saved template: {pathway_name}_{dataset_name} (MSE: {performance:.6f})")
    
    def load_template(self, pathway_name, dataset_name):
        """Load model template if available"""
        
        template_path = os.path.join(self.template_dir, f"{pathway_name}_{dataset_name}_template.pth")
        
        if os.path.exists(template_path):
            template_data = torch.load(template_path, map_location=self.device)
            return template_data
        
        return None


class CCCV3UltimateModel(nn.Module):
    """
    CCCV3 Ultimate Adaptive Model with transfer learning
    
    Features:
    1. Adaptive strategy selection based on empirical evidence
    2. Transfer learning from optimal individual models
    3. Fine-tuning in adaptive context
    4. Maximum performance + efficiency
    """
    
    def __init__(self, input_dim, dataset_name='unknown', dataset_size=100, device='cuda'):
        super(CCCV3UltimateModel, self).__init__()
        self.model_name = "CortexFlow-CLIP-CNN-V3-Ultimate"
        self.input_dim = input_dim
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
        self.device = device
        
        # Initialize template manager
        self.template_manager = ModelTemplateManager(device)
        
        # Determine optimal strategy
        self.strategy = self._determine_optimal_strategy(dataset_name)
        
        print(f"   🧠 CCCV3 Ultimate Strategy for {dataset_name.upper()}:")
        print(f"      Strategy: {self.strategy}")
        print(f"      Transfer learning: Enabled")
        
        # Create models based on strategy
        self.active_models = {}
        self._initialize_models()
        
        print(f"      Active models: {list(self.active_models.keys())}")
        
    def _determine_optimal_strategy(self, dataset_name):
        """Determine optimal strategy based on empirical evidence"""
        
        # Empirical optimal strategies
        optimal_strategies = {
            'miyawaki': 'ensemble',  # 8.04% ensemble benefit
            'vangerven': 'individual_clip',  # CLIP clearly best
            'mindbigdata': 'individual_attention',  # Attention slightly best
            'crell': 'individual_attention'  # Attention slightly best
        }
        
        return optimal_strategies.get(dataset_name, 'individual_clip')
    
    def _initialize_models(self):
        """Initialize models based on strategy with transfer learning"""
        
        if self.strategy == 'individual_clip':
            self.clip_model = self._create_model_with_transfer('clip')
            self.active_models['clip'] = self.clip_model
            
        elif self.strategy == 'individual_attention':
            self.attention_model = self._create_model_with_transfer('attention')
            self.active_models['attention'] = self.attention_model
            
        elif self.strategy == 'individual_lite':
            self.lite_model = self._create_model_with_transfer('lite')
            self.active_models['lite'] = self.lite_model
            
        elif self.strategy == 'ensemble':
            self.clip_model = self._create_model_with_transfer('clip')
            self.attention_model = self._create_model_with_transfer('attention')
            self.lite_model = self._create_model_with_transfer('lite')
            
            self.active_models['clip'] = self.clip_model
            self.active_models['attention'] = self.attention_model
            self.active_models['lite'] = self.lite_model
            
            # Optimal ensemble weights for Miyawaki
            self.ensemble_weights = torch.tensor([0.0, 0.3, 0.7], device=self.device)
    
    def _create_model_with_transfer(self, pathway_name):
        """Create model with transfer learning from template"""
        
        # Create base model
        model = self.template_manager.create_optimal_template(
            pathway_name, self.input_dim, self.dataset_name, self.dataset_size
        )
        
        # Try to load template
        template_data = self.template_manager.load_template(pathway_name, self.dataset_name)
        
        if template_data:
            try:
                model.load_state_dict(template_data['model_state_dict'])
                print(f"      ✅ Loaded template: {pathway_name} (MSE: {template_data['performance']:.6f})")
            except Exception as e:
                print(f"      ⚠️ Template load failed for {pathway_name}: {e}")
        else:
            print(f"      📝 No template found for {pathway_name}, will train from scratch")
        
        return model
    
    def forward(self, x):
        """Forward pass using ultimate adaptive strategy"""
        
        if self.strategy == 'individual_clip':
            output = self.clip_model(x)[0]
            strategy_info = {
                'strategy': 'individual_clip',
                'active_pathway': 'clip',
                'transfer_learning': True
            }
            
        elif self.strategy == 'individual_attention':
            output = self.attention_model(x)[0]
            strategy_info = {
                'strategy': 'individual_attention',
                'active_pathway': 'attention',
                'transfer_learning': True
            }
            
        elif self.strategy == 'individual_lite':
            output = self.lite_model(x)[0]
            strategy_info = {
                'strategy': 'individual_lite',
                'active_pathway': 'lite',
                'transfer_learning': True
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
                'transfer_learning': True
            }
        
        return output, strategy_info


class CCCV3UltimateTrainer:
    """
    Ultimate trainer with transfer learning and optimal hyperparameters
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.strategy = model.strategy
        self.template_manager = model.template_manager
        
    def train_ultimate_model(self, train_loader, val_loader, config):
        """
        Train model using ultimate strategy with transfer learning
        """
        
        print(f"🔧 Training CCCV3 Ultimate Model with {self.strategy} strategy...")
        
        if self.strategy.startswith('individual_'):
            # Train single pathway with optimal hyperparameters
            pathway_name = self.strategy.split('_')[1]
            model = self.model.active_models[pathway_name]
            
            result = self._train_single_model_ultimate(
                model, train_loader, val_loader, config, pathway_name
            )
            
            # Save as template if performance is good
            if result['best_val_loss'] <= self.template_manager.performance_targets.get(
                self.model.dataset_name, {}
            ).get(pathway_name, float('inf')) * 1.1:  # Within 10% of target
                self.template_manager.save_template(
                    model, pathway_name, self.model.dataset_name, result['best_val_loss']
                )
            
            return {
                'strategy': self.strategy,
                'trained_models': [pathway_name],
                'results': {pathway_name: result}
            }
            
        elif self.strategy == 'ensemble':
            # Train all pathways with optimal hyperparameters
            results = {}
            
            for pathway_name, model in self.model.active_models.items():
                print(f"   Training {pathway_name.upper()} pathway...")
                result = self._train_single_model_ultimate(
                    model, train_loader, val_loader, config, pathway_name
                )
                results[pathway_name] = result
                
                # Save as template if performance is good
                if result['best_val_loss'] <= self.template_manager.performance_targets.get(
                    self.model.dataset_name, {}
                ).get(pathway_name, float('inf')) * 1.1:
                    self.template_manager.save_template(
                        model, pathway_name, self.model.dataset_name, result['best_val_loss']
                    )
            
            return {
                'strategy': self.strategy,
                'trained_models': list(self.model.active_models.keys()),
                'results': results
            }
    
    def _train_single_model_ultimate(self, model, train_loader, val_loader, config, pathway_name):
        """Train single model with ultimate optimization"""
        
        # Get optimal hyperparameters
        optimal_params = self.template_manager.get_optimal_hyperparameters(
            pathway_name, self.model.dataset_name
        )
        
        # Create optimizer with optimal parameters
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimal_params['lr'],
            weight_decay=optimal_params['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optimal_params['epochs'], eta_min=1e-8
        )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 25  # Increased patience for ultimate training
        
        print(f"     Optimal params: lr={optimal_params['lr']}, wd={optimal_params['weight_decay']}")
        
        for epoch in range(optimal_params['epochs']):
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
            'pathway_name': pathway_name,
            'optimal_params': optimal_params
        }


# Factory functions
def create_cccv3_ultimate(input_dim, dataset_name='unknown', dataset_size=100, device='cuda'):
    """Factory function to create CCCV3 ultimate model"""
    return CCCV3UltimateModel(input_dim, dataset_name, dataset_size, device)


def create_cccv3_ultimate_trainer(model, device):
    """Factory function to create CCCV3 ultimate trainer"""
    return CCCV3UltimateTrainer(model, device)


# Export main classes and functions
__all__ = [
    'ModelTemplateManager',
    'CCCV3UltimateModel', 
    'CCCV3UltimateTrainer',
    'create_cccv3_ultimate',
    'create_cccv3_ultimate_trainer'
]
