"""
Vangerven Ensemble Refinement
============================

Specialized approach to fix Vangerven ensemble performance issue.

Problem: CCCV3 ensemble (0.047117) worse than individual CLIP (0.036195)
Goal: Achieve ensemble performance better than best individual

Strategies:
1. Transfer learning from best individual models
2. Dynamic ensemble weight optimization
3. Vangerven-specific ensemble architecture
4. Advanced fusion strategies
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)
sys.path.append(parent_dir)

# Import models
try:
    from cccv3.src.models.clip_pathway import create_complete_clip_model
    from cccv3.src.models.lite_pathway import create_complete_lite_model
    from cccv3.src.models.attention_pathway import create_complete_attention_model
except ImportError:
    try:
        sys.path.append(os.path.join(parent_dir, 'src', 'models'))
        from clip_pathway import create_complete_clip_model
        from lite_pathway import create_complete_lite_model
        from attention_pathway import create_complete_attention_model
    except ImportError:
        print("❌ Could not import pathway models")
        sys.exit(1)

# Import utilities
try:
    from src.data import load_dataset_gpu_optimized
except ImportError:
    print("⚠️ Parent directory imports not available")

def setup_device():
    """Setup CUDA device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
        return device
    else:
        return torch.device('cpu')

class VangervenOptimizedEnsemble(nn.Module):
    """
    Vangerven-optimized ensemble with advanced strategies
    """
    
    def __init__(self, input_dim, device='cuda'):
        super(VangervenOptimizedEnsemble, self).__init__()
        self.model_name = "CCCV3-Vangerven-Optimized-Ensemble"
        self.device = device
        
        # Create individual models
        self.clip_model = create_complete_clip_model(input_dim, dataset_size=100, device=device)
        self.lite_model = create_complete_lite_model(input_dim, dataset_size=100, device=device)
        self.attention_model = create_complete_attention_model(input_dim, dataset_size=100, device=device)
        
        # Learnable ensemble weights (instead of fixed)
        self.ensemble_weights = nn.Parameter(torch.tensor([0.1, 0.8, 0.1], device=device))
        
        # Ensemble fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(3, 16),  # 3 predictions -> hidden
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),  # hidden -> single weight
            nn.Sigmoid()
        ).to(device)
        
        # Prediction refinement
        self.refinement_network = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        ).to(device)
        
    def forward(self, x, use_refinement=True):
        """
        Advanced ensemble forward pass
        """
        # Get individual predictions
        clip_pred = self.clip_model(x)[0]      # [batch, 1, 28, 28]
        lite_pred = self.lite_model(x)[0]      # [batch, 1, 28, 28]
        attention_pred = self.attention_model(x)[0]  # [batch, 1, 28, 28]
        
        # Normalize ensemble weights
        weights = torch.softmax(self.ensemble_weights, dim=0)
        
        # Basic weighted ensemble
        basic_ensemble = (weights[0] * lite_pred + 
                         weights[1] * clip_pred + 
                         weights[2] * attention_pred)
        
        if use_refinement:
            # Advanced fusion using prediction quality
            pred_quality = torch.stack([
                torch.mean(lite_pred.view(lite_pred.shape[0], -1), dim=1),
                torch.mean(clip_pred.view(clip_pred.shape[0], -1), dim=1),
                torch.mean(attention_pred.view(attention_pred.shape[0], -1), dim=1)
            ], dim=1)  # [batch, 3]
            
            # Dynamic weighting based on prediction quality
            dynamic_weights = self.fusion_network(pred_quality)  # [batch, 1]
            dynamic_weights = dynamic_weights.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1, 1]

            # Adaptive ensemble
            adaptive_ensemble = (dynamic_weights * basic_ensemble +
                               (1 - dynamic_weights) * clip_pred)  # Favor CLIP when uncertain
            
            # Refinement
            refined_ensemble = self.refinement_network(adaptive_ensemble)
            
            return refined_ensemble
        else:
            return basic_ensemble

class VangervenEnsembleTrainer:
    """
    Specialized trainer for Vangerven ensemble
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def train_with_transfer_learning(self, train_loader, val_loader, config):
        """
        Train with transfer learning from best individual models
        """
        print("🔧 Phase 1: Training individual pathways...")
        
        # Train individual pathways first
        individual_results = self._train_individual_pathways(train_loader, val_loader, config)
        
        print("🔧 Phase 2: Training ensemble fusion...")
        
        # Freeze individual pathways, train only ensemble components
        self._freeze_individual_pathways()
        
        # Train ensemble fusion
        ensemble_results = self._train_ensemble_fusion(train_loader, val_loader, config)
        
        return {
            'individual_results': individual_results,
            'ensemble_results': ensemble_results
        }
    
    def _train_individual_pathways(self, train_loader, val_loader, config):
        """Train individual pathways separately"""
        
        pathways = [
            ('clip', self.model.clip_model),
            ('lite', self.model.lite_model),
            ('attention', self.model.attention_model)
        ]
        
        results = {}
        
        for pathway_name, pathway_model in pathways:
            print(f"   Training {pathway_name.upper()} pathway...")
            
            # Pathway-specific optimizer
            optimizer = optim.AdamW(
                pathway_model.parameters(),
                lr=0.0005 if pathway_name == 'clip' else 0.001,
                weight_decay=1e-6
            )
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.get('epochs', 80), eta_min=1e-8
            )
            
            criterion = nn.MSELoss()
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(config.get('epochs', 80)):
                # Training
                pathway_model.train()
                train_loss = 0.0
                
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    visual_output = pathway_model(data)[0]
                    loss = criterion(visual_output, target)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(pathway_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                pathway_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        visual_output = pathway_model(data)[0]
                        val_loss += criterion(visual_output, target).item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                scheduler.step()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = pathway_model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if epoch % 20 == 0:
                    print(f"     Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")
                
                if patience_counter >= 15:
                    print(f"     Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model
            pathway_model.load_state_dict(best_state)
            results[pathway_name] = best_val_loss
            print(f"   {pathway_name.upper()} best val loss: {best_val_loss:.6f}")
        
        return results
    
    def _freeze_individual_pathways(self):
        """Freeze individual pathway parameters"""
        for param in self.model.clip_model.parameters():
            param.requires_grad = False
        for param in self.model.lite_model.parameters():
            param.requires_grad = False
        for param in self.model.attention_model.parameters():
            param.requires_grad = False
    
    def _train_ensemble_fusion(self, train_loader, val_loader, config):
        """Train only ensemble fusion components"""
        
        # Optimizer for ensemble components only
        ensemble_params = list(self.model.fusion_network.parameters()) + \
                         list(self.model.refinement_network.parameters()) + \
                         [self.model.ensemble_weights]
        
        optimizer = optim.AdamW(ensemble_params, lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-8)
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(50):  # Shorter training for fusion
            # Training
            self.model.train()
            train_loss = 0.0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                ensemble_output = self.model(data, use_refinement=True)
                loss = criterion(ensemble_output, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(ensemble_params, max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    ensemble_output = self.model(data, use_refinement=True)
                    val_loss += criterion(ensemble_output, target).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    'fusion_network': self.model.fusion_network.state_dict(),
                    'refinement_network': self.model.refinement_network.state_dict(),
                    'ensemble_weights': self.model.ensemble_weights.clone()
                }
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                weights = torch.softmax(self.model.ensemble_weights, dim=0)
                print(f"     Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")
                print(f"       Weights: Lite={weights[0]:.3f}, CLIP={weights[1]:.3f}, Attn={weights[2]:.3f}")
            
            if patience_counter >= 10:
                print(f"     Early stopping at epoch {epoch+1}")
                break
        
        # Load best ensemble state
        self.model.fusion_network.load_state_dict(best_state['fusion_network'])
        self.model.refinement_network.load_state_dict(best_state['refinement_network'])
        self.model.ensemble_weights.data = best_state['ensemble_weights']
        
        return best_val_loss

def test_vangerven_refinement():
    """Test refined Vangerven ensemble approach"""
    
    print("🎯 Vangerven Ensemble Refinement Testing")
    print("=" * 50)
    
    device = setup_device()
    
    # Load Vangerven dataset
    try:
        if 'load_dataset_gpu_optimized' in globals():
            X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized('vangerven', device)
        else:
            print("❌ Dataset loading function not available")
            return
        
        print(f"✅ Vangerven loaded: Train={len(X_train)}, Test={len(X_test)}, Input_dim={input_dim}")
        
    except Exception as e:
        print(f"❌ Error loading Vangerven: {e}")
        return
    
    # Combine for cross-validation
    X_all = torch.cat([X_train, X_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)
    
    print(f"📊 Total samples: {len(X_all)}")
    
    # 5-fold CV for efficiency
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    config = {
        'epochs': 80,
        'weight_decay': 1e-6
    }
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all)):
        print(f"\n🔄 Fold {fold + 1}/5")
        
        # Split data
        train_data = X_all[train_idx]
        train_targets = y_all[train_idx]
        val_data = X_all[val_idx]
        val_targets = y_all[val_idx]
        
        # Create data loaders
        batch_size = 8  # Small batch for Vangerven
        train_dataset = TensorDataset(train_data, train_targets)
        val_dataset = TensorDataset(val_data, val_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model and trainer
        model = VangervenOptimizedEnsemble(input_dim, device)
        trainer = VangervenEnsembleTrainer(model, device)
        
        # Train with transfer learning
        training_results = trainer.train_with_transfer_learning(train_loader, val_loader, config)
        
        # Evaluate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                ensemble_output = model(data, use_refinement=True)
                val_loss += nn.MSELoss()(ensemble_output, target).item()
        
        val_loss /= len(val_loader)
        fold_results.append(val_loss)
        
        print(f"   Fold {fold + 1} MSE: {val_loss:.6f}")
        
        # Show final ensemble weights
        final_weights = torch.softmax(model.ensemble_weights, dim=0)
        print(f"   Final weights: Lite={final_weights[0]:.3f}, CLIP={final_weights[1]:.3f}, Attn={final_weights[2]:.3f}")
        
        # Clear memory
        del model, trainer
        torch.cuda.empty_cache()
    
    # Final results
    mean_mse = np.mean(fold_results)
    std_mse = np.std(fold_results)
    
    print(f"\n🎯 VANGERVEN REFINEMENT RESULTS:")
    print(f"   Refined Ensemble MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"   Original Ensemble MSE: 0.047117 ± 0.005465")
    print(f"   Best Individual CLIP: ~0.036195")
    
    if mean_mse < 0.047117:
        improvement = ((0.047117 - mean_mse) / 0.047117) * 100
        print(f"   🏆 IMPROVEMENT: {improvement:.2f}% better than original ensemble!")
        
        if mean_mse < 0.036195:
            individual_improvement = ((0.036195 - mean_mse) / 0.036195) * 100
            print(f"   🎉 BEATS INDIVIDUAL: {individual_improvement:.2f}% better than best individual!")
        else:
            gap = ((mean_mse - 0.036195) / 0.036195) * 100
            print(f"   📈 Still {gap:.2f}% behind best individual")
    else:
        print(f"   📉 No improvement over original ensemble")
    
    print(f"\n🚀 Vangerven Ensemble Refinement Complete!")

if __name__ == "__main__":
    test_vangerven_refinement()
