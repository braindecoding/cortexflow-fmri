#!/usr/bin/env python3
"""
FULL CortexFlow-Ensemble Training
Complete implementation with all novel components
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
import json
import time
from typing import Dict, List, Tuple

# Import individual CortexFlow models
class CortexFlowSimple(nn.Module):
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, output_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class CortexFlowMC(nn.Module):
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, output_dim), nn.Sigmoid()
        )
        self.uncertainty = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim), nn.Softplus()
        )
    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        unc = self.uncertainty(z).mean(dim=-1)
        return output, unc

class CortexFlowHierarchical(nn.Module):
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        self.scale1 = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.scale2 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.scale3 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(448, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, output_dim), nn.Sigmoid()
        )
    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        combined = torch.cat([s1, s2, s3], dim=-1)
        return self.decoder(combined)

class CortexFlowEnhanced(nn.Module):
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        self.hierarchical = CortexFlowHierarchical(input_dim, output_dim)
        self.uncertainty = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.Softplus()
        )
        self.alignment = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.Tanh()
        )
    def forward(self, x):
        output = self.hierarchical(x)
        unc = self.uncertainty(x).mean(dim=-1)
        align = self.alignment(x)
        return output, unc

class CortexFlowUnified(nn.Module):
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        self.complexity = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        self.simple = CortexFlowSimple(input_dim, output_dim)
        self.complex = CortexFlowHierarchical(input_dim, output_dim)
        self.uncertainty = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, output_dim), nn.Softplus()
        )
    def forward(self, x):
        comp = self.complexity(x)
        simple_out = self.simple(x)
        complex_out = self.complex(x)
        output = comp * complex_out + (1 - comp) * simple_out
        unc = self.uncertainty(x).mean(dim=-1)
        return output, unc, comp

# Full CortexFlow Ensemble Implementation
class FullCortexFlowEnsemble(nn.Module):
    """
    FULL CortexFlow-Ensemble Implementation
    Complete with all novel components
    """
    
    def __init__(self, input_dim: int, output_dim: int = 784):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize individual models
        self.models = nn.ModuleDict({
            'simple': CortexFlowSimple(input_dim, output_dim),
            'mc': CortexFlowMC(input_dim, output_dim),
            'hierarchical': CortexFlowHierarchical(input_dim, output_dim),
            'enhanced': CortexFlowEnhanced(input_dim, output_dim),
            'unified': CortexFlowUnified(input_dim, output_dim)
        })
        
        # Adaptive ensemble weighting
        self.adaptive_weights = nn.Sequential(
            nn.Linear(input_dim + 5 + 5, 128),  # input + predictions + uncertainties
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Softmax(dim=-1)
        )
        
        # Hierarchical uncertainty estimator
        self.meta_uncertainty = nn.Sequential(
            nn.Linear(5 * 3, 64),  # predictions + uncertainties + weights
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # Intelligent model selector
        self.model_selector = nn.Sequential(
            nn.Linear(input_dim + 5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Sigmoid()
        )
        
        # Complexity predictor
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
        Full ensemble forward pass with all novel components
        """
        batch_size = x.size(0)
        
        # Predict input complexity
        complexity_score = self.complexity_predictor(x)
        
        # Get predictions from all models
        model_outputs = {}
        predictions = []
        uncertainties = []
        
        for name, model in self.models.items():
            if name in ['mc', 'enhanced', 'unified']:
                # Models with uncertainty output
                if name == 'unified':
                    output, unc, comp = model(x)
                else:
                    output, unc = model(x)
                pred, uncertainty = output, unc
            else:
                # Models without uncertainty
                pred = model(x)
                uncertainty = torch.zeros(batch_size, device=x.device)
            
            model_outputs[name] = {'prediction': pred, 'uncertainty': uncertainty}
            predictions.append(pred)
            uncertainties.append(uncertainty)
        
        predictions = torch.stack(predictions, dim=1)  # [B, 5, output_dim]
        uncertainties = torch.stack(uncertainties, dim=1)  # [B, 5]
        
        # Compute cross-model agreement
        agreement_matrix = self._compute_agreement_matrix(predictions)
        
        if mode == 'selection':
            # Intelligent model selection
            pred_features = predictions.mean(dim=-1)  # [B, 5]
            selector_input = torch.cat([x, pred_features], dim=-1)
            selection_weights = self.model_selector(selector_input)
            ensemble_weights = selection_weights
        else:
            # Adaptive ensemble weighting
            pred_features = predictions.mean(dim=-1)  # [B, 5]
            unc_features = uncertainties  # [B, 5]
            weight_input = torch.cat([x, pred_features, unc_features], dim=-1)
            ensemble_weights = self.adaptive_weights(weight_input)
        
        # Weighted ensemble prediction
        ensemble_pred = torch.sum(predictions * ensemble_weights.unsqueeze(-1), dim=1)
        
        # Hierarchical uncertainty estimation
        uncertainty_dict = self._hierarchical_uncertainty(
            predictions, uncertainties, ensemble_weights
        )
        
        return {
            'ensemble_prediction': ensemble_pred,
            'ensemble_weights': ensemble_weights,
            'complexity_score': complexity_score.squeeze(-1),
            'individual_predictions': predictions,
            'individual_uncertainties': uncertainties,
            'total_uncertainty': uncertainty_dict['total'],
            'within_uncertainty': uncertainty_dict['within'],
            'between_uncertainty': uncertainty_dict['between'],
            'meta_uncertainty': uncertainty_dict['meta'],
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
                    pred_i = predictions[:, i, :]
                    pred_j = predictions[:, j, :]
                    cosine_sim = nn.functional.cosine_similarity(pred_i, pred_j, dim=1)
                    agreement_matrix[:, i, j] = cosine_sim
                else:
                    agreement_matrix[:, i, j] = 1.0
        
        return agreement_matrix
    
    def _hierarchical_uncertainty(self, predictions: torch.Tensor, 
                                individual_uncertainties: torch.Tensor,
                                ensemble_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute hierarchical uncertainty decomposition"""
        
        # Weighted ensemble prediction
        ensemble_pred = torch.sum(predictions * ensemble_weights.unsqueeze(-1), dim=1)
        
        # Level 1: Within-model uncertainty
        sigma2_within = torch.sum(ensemble_weights * individual_uncertainties, dim=1)
        
        # Level 2: Between-model uncertainty
        model_deviations = predictions - ensemble_pred.unsqueeze(1)
        sigma2_between = torch.sum(ensemble_weights.unsqueeze(-1) * 
                                  (model_deviations ** 2), dim=(1, 2))
        
        # Level 3: Meta-uncertainty
        meta_features = torch.cat([
            predictions.mean(dim=-1),  # [B, 5]
            individual_uncertainties,   # [B, 5]
            ensemble_weights           # [B, 5]
        ], dim=-1)
        
        sigma2_meta = self.meta_uncertainty(meta_features).squeeze(-1)
        
        # Total uncertainty
        sigma2_total = sigma2_within + sigma2_between + sigma2_meta
        
        return {
            'total': sigma2_total,
            'within': sigma2_within,
            'between': sigma2_between,
            'meta': sigma2_meta
        }

class FullEnsembleLoss(nn.Module):
    """Full ensemble loss with all novel components"""
    
    def __init__(self, lambda_diversity: float = 0.1, lambda_agreement: float = 0.05,
                 lambda_complexity: float = 0.02, lambda_uncertainty: float = 0.1):
        super().__init__()
        self.lambda_diversity = lambda_diversity
        self.lambda_agreement = lambda_agreement
        self.lambda_complexity = lambda_complexity
        self.lambda_uncertainty = lambda_uncertainty
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        # Main reconstruction loss
        recon_loss = nn.MSELoss()(outputs['ensemble_prediction'], targets)
        
        # Diversity loss
        predictions = outputs['individual_predictions']
        diversity_loss = 0.0
        for i in range(5):
            for j in range(i+1, 5):
                similarity = nn.functional.cosine_similarity(
                    predictions[:, i], predictions[:, j], dim=-1
                ).mean()
                diversity_loss += similarity
        diversity_loss = diversity_loss / 10
        
        # Agreement loss
        weights = outputs['ensemble_weights']
        agreement_matrix = outputs['agreement_matrix']
        agreement_loss = 0.0
        for i in range(5):
            for j in range(i+1, 5):
                weight_product = weights[:, i] * weights[:, j]
                agreement_factor = agreement_matrix[:, i, j]
                pred_distance = nn.functional.mse_loss(
                    predictions[:, i], predictions[:, j], reduction='none'
                ).mean(dim=-1)
                agreement_loss += (weight_product * agreement_factor * pred_distance).mean()
        agreement_loss = agreement_loss / 10
        
        # Complexity regularization
        complexity_score = outputs['complexity_score']
        weight_entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
        complexity_reg = (1 - complexity_score).mean() * weight_entropy
        
        # Uncertainty calibration
        errors = nn.functional.mse_loss(
            outputs['ensemble_prediction'], targets, reduction='none'
        ).mean(dim=-1)
        uncertainties = outputs['total_uncertainty']
        uncertainty_loss = (torch.log(uncertainties + 1e-8) + 
                          errors / (uncertainties + 1e-8)).mean()
        
        total_loss = (recon_loss - 
                     self.lambda_diversity * diversity_loss +
                     self.lambda_agreement * agreement_loss +
                     self.lambda_complexity * complexity_reg +
                     self.lambda_uncertainty * uncertainty_loss)
        
        return total_loss

def load_dataset(dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load dataset for training"""
    files = {
        'miyawaki': 'data/processed/miyawaki_structured_28x28.mat',
        'vangerven': 'data/processed/digit69_28x28.mat',
        'mindbigdata': 'data/processed/mindbigdata.mat',
        'crell': 'data/processed/crell.mat'
    }
    
    try:
        data = loadmat(files[dataset_name])
        if dataset_name == 'miyawaki':
            X, y = data['fmriTrn'], data['stimTrn']
        else:
            arrays = [(k, v) for k, v in data.items() 
                     if isinstance(v, np.ndarray) and v.ndim >= 2 and not k.startswith('__')]
            arrays.sort(key=lambda x: x[1].size, reverse=True)
            X, y = arrays[0][1], arrays[1][1]
        
        if X.ndim > 2: X = X.reshape(X.shape[0], -1)
        if y.ndim > 2: y = y.reshape(y.shape[0], -1)
        if y.shape[1] != 784:
            if y.shape[1] > 784: y = y[:, :784]
            else: y = np.pad(y, ((0, 0), (0, 784 - y.shape[1])))
        
        X = (X - X.mean()) / (X.std() + 1e-8)
        y = y / (y.max() + 1e-8) if y.max() > 1.0 else (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, None

def main():
    """Main full ensemble training"""
    print("🚀 FULL CORTEXFLOW-ENSEMBLE TRAINING")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Create results directory
    results_dir = Path("results/full_ensemble_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n📊 FULL Training on {dataset_name}...")
        
        # Load dataset
        X, y = load_dataset(dataset_name)
        if X is None:
            continue
        
        print(f"✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Create model
        model = FullCortexFlowEnsemble(X.shape[1])
        model = model.to(device)
        
        print(f"✅ FULL Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create data loaders
        dataset = torch.utils.data.TensorDataset(X, y)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Training setup
        criterion = FullEnsembleLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training
        print(f"🚀 FULL Training...")
        best_val_loss = float('inf')
        
        start_time = time.time()
        
        for epoch in range(20):
            # Training
            model.train()
            epoch_train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                outputs = model(data, mode='adaptive')
                loss = criterion(outputs, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Validation
            model.eval()
            epoch_val_loss = 0.0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data, mode='adaptive')
                    loss = criterion(outputs, target)
                    epoch_val_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1:2d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'full_ensemble_{dataset_name}_best.pth')
                print(f"    ✅ New best: {best_val_loss:.6f}")
            
            scheduler.step(avg_val_loss)
        
        training_time = time.time() - start_time
        
        # Evaluation
        model.eval()
        test_losses = []
        ensemble_weights_all = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data, mode='adaptive')
                loss = nn.MSELoss()(outputs['ensemble_prediction'], target)
                test_losses.append(loss.item())
                ensemble_weights_all.append(outputs['ensemble_weights'].cpu())
        
        avg_weights = torch.cat(ensemble_weights_all, dim=0).mean(dim=0)
        
        results = {
            'training_time': training_time,
            'best_val_loss': best_val_loss,
            'test_mse': np.mean(test_losses),
            'average_weights': avg_weights.tolist()
        }
        
        all_results[dataset_name] = results
        
        print(f"✅ FULL {dataset_name} completed:")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Test MSE: {results['test_mse']:.6f}")
        print(f"   Avg weights: {[f'{w:.2f}' for w in results['average_weights']]}")
    
    # Save results
    with open(results_dir / "full_ensemble_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n🎉 FULL ENSEMBLE TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Results: {results_dir}")
    
    print("\n📊 FULL ENSEMBLE RESULTS:")
    for dataset, results in all_results.items():
        print(f"{dataset:12}: MSE={results['test_mse']:.6f}, Time={results['training_time']:.1f}s")

if __name__ == "__main__":
    main()
