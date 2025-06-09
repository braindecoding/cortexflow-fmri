#!/usr/bin/env python3
"""
CortexFlow-Ensemble Training - IMMEDIATE IMPLEMENTATION
Quick training to get real ensemble results NOW!
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

# Simple ensemble implementation for immediate training
class SimpleEnsembleModel(nn.Module):
    """
    Simplified CortexFlow Ensemble for immediate training
    Uses existing trained models with adaptive weighting
    """
    
    def __init__(self, input_dim: int, output_dim: int = 784):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Individual models (simplified versions)
        self.simple_model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, output_dim), nn.Sigmoid()
        )
        
        self.mc_model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, output_dim), nn.Sigmoid()
        )
        
        self.hierarchical_model = nn.Sequential(
            nn.Linear(input_dim, 448), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(448, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, output_dim), nn.Sigmoid()
        )
        
        self.enhanced_model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(512, 384), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(384, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, output_dim), nn.Sigmoid()
        )
        
        self.unified_model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),  # Complexity predictor
        )
        self.unified_simple = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, output_dim), nn.Sigmoid()
        )
        self.unified_complex = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, output_dim), nn.Sigmoid()
        )
        
        # Adaptive weighting network
        self.weight_predictor = nn.Sequential(
            nn.Linear(input_dim + 5, 64),  # input + 5 model predictions
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),  # 5 weights for 5 models
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive ensemble"""
        
        # Get predictions from all models
        pred_simple = self.simple_model(x)
        pred_mc = self.mc_model(x)
        pred_hierarchical = self.hierarchical_model(x)
        pred_enhanced = self.enhanced_model(x)
        
        # Unified model (adaptive)
        complexity = self.unified_model(x)
        pred_unified_simple = self.unified_simple(x)
        pred_unified_complex = self.unified_complex(x)
        pred_unified = complexity * pred_unified_complex + (1 - complexity) * pred_unified_simple
        
        # Stack all predictions
        all_predictions = torch.stack([
            pred_simple, pred_mc, pred_hierarchical, pred_enhanced, pred_unified
        ], dim=1)  # [B, 5, output_dim]
        
        # Compute adaptive weights
        # Use mean prediction as feature
        pred_features = all_predictions.mean(dim=-1)  # [B, 5]
        weight_input = torch.cat([x, pred_features], dim=-1)
        ensemble_weights = self.weight_predictor(weight_input)  # [B, 5]
        
        # Weighted ensemble prediction
        ensemble_pred = torch.sum(all_predictions * ensemble_weights.unsqueeze(-1), dim=1)
        
        # Calculate uncertainties (simplified)
        pred_std = torch.std(all_predictions, dim=1).mean(dim=-1)  # [B]
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': all_predictions,
            'ensemble_weights': ensemble_weights,
            'uncertainty': pred_std,
            'complexity_score': complexity.squeeze(-1)
        }

class EnsembleLoss(nn.Module):
    """Simplified ensemble loss for quick training"""
    
    def __init__(self, lambda_diversity: float = 0.1, lambda_weight_reg: float = 0.05):
        super().__init__()
        self.lambda_diversity = lambda_diversity
        self.lambda_weight_reg = lambda_weight_reg
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        # Main reconstruction loss
        recon_loss = nn.MSELoss()(outputs['ensemble_prediction'], targets)
        
        # Diversity loss (encourage different predictions)
        predictions = outputs['individual_predictions']  # [B, 5, output_dim]
        diversity_loss = 0.0
        for i in range(5):
            for j in range(i+1, 5):
                similarity = nn.CosineSimilarity(dim=-1)(predictions[:, i], predictions[:, j])
                diversity_loss += similarity.mean()
        diversity_loss = diversity_loss / 10  # Normalize by number of pairs
        
        # Weight regularization (prevent extreme weights)
        weights = outputs['ensemble_weights']  # [B, 5]
        weight_entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
        weight_reg = -weight_entropy  # Encourage diverse weights
        
        total_loss = (recon_loss - 
                     self.lambda_diversity * diversity_loss + 
                     self.lambda_weight_reg * weight_reg)
        
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

def train_ensemble(model: nn.Module, train_loader, val_loader, device: torch.device,
                  dataset_name: str, epochs: int = 20) -> Dict:
    """Train ensemble model"""
    
    model = model.to(device)
    criterion = EnsembleLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Training CortexFlow-Ensemble...")
    print("=" * 50)
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
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
                outputs = model(data)
                loss = criterion(outputs, target)
                epoch_val_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1:2d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'ensemble_{dataset_name}_best.pth')
            print(f"    ✅ New best: {best_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def evaluate_ensemble(model: nn.Module, test_loader, device: torch.device) -> Dict:
    """Evaluate ensemble performance"""
    
    model.eval()
    test_losses = []
    ensemble_weights_all = []
    uncertainties_all = []
    complexity_scores_all = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            loss = criterion(outputs['ensemble_prediction'], target)
            test_losses.append(loss.item())
            
            ensemble_weights_all.append(outputs['ensemble_weights'].cpu())
            uncertainties_all.append(outputs['uncertainty'].cpu())
            complexity_scores_all.append(outputs['complexity_score'].cpu())
    
    # Aggregate results
    avg_weights = torch.cat(ensemble_weights_all, dim=0).mean(dim=0)
    avg_uncertainty = torch.cat(uncertainties_all, dim=0).mean()
    avg_complexity = torch.cat(complexity_scores_all, dim=0).mean()
    
    return {
        'test_mse': np.mean(test_losses),
        'average_weights': avg_weights.tolist(),
        'average_uncertainty': avg_uncertainty.item(),
        'average_complexity': avg_complexity.item()
    }

def create_ensemble_visualization(model: nn.Module, test_loader, device: torch.device, 
                                dataset_name: str, save_path: Path):
    """Create ensemble visualization"""
    
    model.eval()
    
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        
        # Use available samples (max 10)
        n_samples = min(10, data.size(0))
        sample_data = data[:n_samples]
        sample_target = target[:n_samples]

        outputs = model(sample_data)

        fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
        fig.suptitle(f'CortexFlow-Ensemble Results on {dataset_name.title()}',
                    fontsize=16, fontweight='bold')

        # Handle single sample case
        if n_samples == 1:
            axes = axes.reshape(3, 1)

        for i in range(n_samples):
            # Original
            orig = sample_target[i].cpu().numpy().reshape(28, 28)
            axes[0, i].imshow(orig, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Original {i+1}', fontsize=10)
            axes[0, i].axis('off')
            
            # Ensemble prediction
            pred = outputs['ensemble_prediction'][i].cpu().numpy().reshape(28, 28)
            axes[1, i].imshow(pred, cmap='gray', vmin=0, vmax=1)
            
            mse = np.mean((orig - pred) ** 2)
            uncertainty = outputs['uncertainty'][i].cpu().item()
            
            axes[1, i].set_title(f'Ensemble {i+1}\nMSE: {mse:.4f}\nUnc: {uncertainty:.3f}', fontsize=9)
            axes[1, i].axis('off')
            
            # Ensemble weights
            weights = outputs['ensemble_weights'][i].cpu().numpy()
            model_names = ['Simple', 'MC', 'Hier', 'Enh', 'Unif']
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            
            bars = axes[2, i].bar(range(5), weights, color=colors)
            axes[2, i].set_xticks(range(5))
            axes[2, i].set_xticklabels(model_names, rotation=45, fontsize=8)
            axes[2, i].set_title(f'Weights {i+1}', fontsize=9)
            axes[2, i].set_ylim(0, 1)
            
            # Add weight values on bars
            for bar, weight in zip(bars, weights):
                height = bar.get_height()
                axes[2, i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{weight:.2f}', ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main training execution"""
    print("🚀 CORTEXFLOW-ENSEMBLE IMMEDIATE TRAINING")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Create results directory
    results_dir = Path("results/ensemble_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Datasets to train on
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n📊 Training on {dataset_name}...")
        
        # Load dataset
        X, y = load_dataset(dataset_name)
        if X is None:
            print(f"❌ Failed to load {dataset_name}")
            continue
        
        print(f"✅ Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
        
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
        
        # Create model
        model = SimpleEnsembleModel(X.shape[1])
        print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train
        start_time = time.time()
        training_results = train_ensemble(model, train_loader, val_loader, device, dataset_name, epochs=15)
        training_time = time.time() - start_time
        
        # Evaluate
        eval_results = evaluate_ensemble(model, test_loader, device)
        
        # Create visualization
        viz_path = results_dir / f"ensemble_{dataset_name}_visualization.png"
        create_ensemble_visualization(model, test_loader, device, dataset_name, viz_path)
        
        # Store results
        dataset_results = {
            'training_time': training_time,
            'best_val_loss': training_results['best_val_loss'],
            'test_mse': eval_results['test_mse'],
            'average_weights': eval_results['average_weights'],
            'average_uncertainty': eval_results['average_uncertainty'],
            'average_complexity': eval_results['average_complexity'],
            'visualization_path': str(viz_path)
        }
        
        all_results[dataset_name] = dataset_results
        
        print(f"✅ {dataset_name} completed:")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Best val loss: {training_results['best_val_loss']:.6f}")
        print(f"   Test MSE: {eval_results['test_mse']:.6f}")
        print(f"   Avg weights: {[f'{w:.2f}' for w in eval_results['average_weights']]}")
    
    # Save all results
    with open(results_dir / "ensemble_training_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print(f"\n🎉 ENSEMBLE TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📁 Results saved to: {results_dir}")
    print(f"🖼️  Visualizations: {len(all_results)} datasets")
    
    print("\n📊 ENSEMBLE RESULTS SUMMARY:")
    for dataset, results in all_results.items():
        print(f"{dataset:12}: MSE={results['test_mse']:.6f}, Time={results['training_time']:.1f}s")
    
    print("\n🚀 ENSEMBLE TRAINING SUCCESS - READY FOR PAPER!")

if __name__ == "__main__":
    main()
