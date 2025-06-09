#!/usr/bin/env python3
"""
CortexFlow Ensemble Training with Perceptual Loss
Fix blocky reconstruction issues with advanced loss functions
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
import sys
sys.path.append('src/models')

from train_full_ensemble import FullCortexFlowEnsemble, load_dataset
from perceptual_losses import ComprehensiveReconstructionLoss, create_advanced_loss

def create_improved_ensemble(input_dim: int, output_dim: int = 784):
    """Create ensemble with improved output activation"""
    
    class ImprovedCortexFlowEnsemble(FullCortexFlowEnsemble):
        def __init__(self, input_dim, output_dim=784):
            super().__init__(input_dim, output_dim)
            
            # Replace sigmoid with tanh + scaling for better gradients
            self._fix_output_activations()
        
        def _fix_output_activations(self):
            """Fix output activations to prevent saturation"""
            
            # Fix Simple model
            simple_layers = list(self.models['simple'].net.children())
            simple_layers[-1] = nn.Tanh()  # Replace Sigmoid with Tanh
            self.models['simple'].net = nn.Sequential(*simple_layers)
            
            # Fix MC model decoder
            mc_decoder_layers = list(self.models['mc'].decoder.children())
            mc_decoder_layers[-1] = nn.Tanh()
            self.models['mc'].decoder = nn.Sequential(*mc_decoder_layers)
            
            # Fix Hierarchical model decoder
            hier_decoder_layers = list(self.models['hierarchical'].decoder.children())
            hier_decoder_layers[-1] = nn.Tanh()
            self.models['hierarchical'].decoder = nn.Sequential(*hier_decoder_layers)
            
            # Fix Enhanced model (through hierarchical)
            enhanced_hier_decoder = list(self.models['enhanced'].hierarchical.decoder.children())
            enhanced_hier_decoder[-1] = nn.Tanh()
            self.models['enhanced'].hierarchical.decoder = nn.Sequential(*enhanced_hier_decoder)
            
            # Fix Unified models
            unified_simple_layers = list(self.models['unified'].simple.net.children())
            unified_simple_layers[-1] = nn.Tanh()
            self.models['unified'].simple.net = nn.Sequential(*unified_simple_layers)
            
            unified_complex_decoder = list(self.models['unified'].complex.decoder.children())
            unified_complex_decoder[-1] = nn.Tanh()
            self.models['unified'].complex.decoder = nn.Sequential(*unified_complex_decoder)
        
        def forward(self, x: torch.Tensor, mode: str = 'adaptive'):
            """Forward with output scaling"""
            outputs = super().forward(x, mode)
            
            # Scale tanh output [-1,1] to [0,1] for image reconstruction
            outputs['ensemble_prediction'] = (outputs['ensemble_prediction'] + 1) / 2
            
            # Scale individual predictions too
            scaled_preds = (outputs['individual_predictions'] + 1) / 2
            outputs['individual_predictions'] = scaled_preds
            
            return outputs
    
    return ImprovedCortexFlowEnsemble(input_dim, output_dim)

def train_with_perceptual_loss(model, train_loader, val_loader, device, dataset_name, epochs=25):
    """Train ensemble with comprehensive perceptual loss"""
    
    model = model.to(device)
    
    # Use comprehensive reconstruction loss
    criterion = ComprehensiveReconstructionLoss(
        lambda_mse=1.0,
        lambda_perceptual=0.1,
        lambda_ssim=0.1,
        lambda_gradient=0.05,
        lambda_frequency=0.02
    )
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)  # Lower LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🎨 Training with Perceptual Loss...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        epoch_loss_components = {
            'mse': 0.0, 'perceptual': 0.0, 'ssim': 0.0, 
            'gradient': 0.0, 'frequency': 0.0
        }
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data, mode='adaptive')
            
            # Compute comprehensive loss
            loss_dict = criterion(outputs['ensemble_prediction'], target)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Track loss components
            for key in epoch_loss_components:
                if f'{key}_loss' in loss_dict:
                    epoch_loss_components[key] += loss_dict[f'{key}_loss'].item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                outputs = model(data, mode='adaptive')
                loss_dict = criterion(outputs['ensemble_prediction'], target)
                epoch_val_loss += loss_dict['total_loss'].item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print detailed loss breakdown every 5 epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1:2d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
            for key, value in epoch_loss_components.items():
                avg_component = value / len(train_loader)
                print(f"    {key:10}: {avg_component:.6f}")
        else:
            print(f"Epoch {epoch+1:2d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'perceptual_ensemble_{dataset_name}_best.pth')
            print(f"    ✅ New best: {best_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def create_perceptual_visualization(model, test_loader, device, dataset_name, save_path):
    """Create visualization with perceptual loss training"""
    
    model.eval()
    
    with torch.no_grad():
        # Get samples
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        
        n_samples = min(8, data.size(0))
        sample_data = data[:n_samples]
        sample_target = target[:n_samples]
        
        outputs = model(sample_data, mode='adaptive')
        
        # Create visualization
        fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
        fig.suptitle(f'CortexFlow Ensemble with Perceptual Loss - {dataset_name.title()}', 
                    fontsize=14, fontweight='bold')
        
        if n_samples == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(n_samples):
            # Original
            original = sample_target[i].cpu().numpy().reshape(28, 28)
            axes[0, i].imshow(original, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Original {i+1}', fontsize=10)
            axes[0, i].axis('off')
            
            # Ensemble prediction
            ensemble_pred = outputs['ensemble_prediction'][i].cpu().numpy().reshape(28, 28)
            axes[1, i].imshow(ensemble_pred, cmap='gray', vmin=0, vmax=1)
            
            # Calculate metrics
            mse = np.mean((original - ensemble_pred) ** 2)
            uncertainty = outputs['total_uncertainty'][i].cpu().item()
            
            axes[1, i].set_title(f'Perceptual Ensemble\nMSE: {mse:.4f}\nUnc: {uncertainty:.3f}', 
                               fontsize=9)
            axes[1, i].axis('off')
            
            # Ensemble weights
            weights = outputs['ensemble_weights'][i].cpu().numpy()
            model_names = ['Simple', 'MC', 'Hier', 'Enh', 'Unif']
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            
            bars = axes[2, i].bar(range(5), weights, color=colors, alpha=0.7)
            axes[2, i].set_xticks(range(5))
            axes[2, i].set_xticklabels(model_names, rotation=45, fontsize=8)
            axes[2, i].set_title(f'Weights {i+1}', fontsize=9)
            axes[2, i].set_ylim(0, 1)
            
            # Add weight values
            for bar, weight in zip(bars, weights):
                height = bar.get_height()
                if height > 0.05:
                    axes[2, i].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                   f'{weight:.2f}', ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

def main():
    """Main perceptual training execution"""
    print("🎨 CORTEXFLOW ENSEMBLE - PERCEPTUAL LOSS TRAINING")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Create results directory
    results_dir = Path("results/perceptual_ensemble_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Focus on Miyawaki first (the problematic one)
    dataset_name = 'miyawaki'
    
    print(f"\n📊 Perceptual Training on {dataset_name}...")
    
    # Load dataset
    X, y = load_dataset(dataset_name)
    if X is None:
        print(f"❌ Failed to load {dataset_name}")
        return
    
    print(f"✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create improved model
    model = create_improved_ensemble(X.shape[1])
    model = model.to(device)
    
    print(f"✅ Improved Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
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
    
    # Training with perceptual loss
    start_time = time.time()
    
    training_results = train_with_perceptual_loss(
        model, train_loader, val_loader, device, dataset_name, epochs=25
    )
    
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
    
    # Create visualization
    viz_path = results_dir / f"perceptual_ensemble_{dataset_name}_visualization.png"
    create_perceptual_visualization(model, test_loader, device, dataset_name, viz_path)
    
    # Results
    results = {
        'dataset': dataset_name,
        'training_time': training_time,
        'best_val_loss': training_results['best_val_loss'],
        'test_mse': np.mean(test_losses),
        'average_weights': avg_weights.tolist(),
        'visualization_path': str(viz_path)
    }
    
    # Save results
    with open(results_dir / f"perceptual_ensemble_{dataset_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ PERCEPTUAL TRAINING COMPLETED!")
    print("=" * 70)
    print(f"📁 Results: {results_dir}")
    print(f"⏱️  Training time: {training_time:.2f}s")
    print(f"📉 Best val loss: {training_results['best_val_loss']:.6f}")
    print(f"📊 Test MSE: {results['test_mse']:.6f}")
    print(f"⚖️  Avg weights: {[f'{w:.2f}' for w in results['average_weights']]}")
    print(f"🖼️  Visualization: {viz_path}")
    
    print(f"\n🎯 COMPARISON WITH PREVIOUS:")
    print(f"Previous MSE (blocky): 0.104720")
    print(f"Perceptual MSE:        {results['test_mse']:.6f}")
    improvement = ((0.104720 - results['test_mse']) / 0.104720) * 100
    print(f"Improvement:           {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"\n🎉 SUCCESS! Perceptual loss improved reconstruction quality!")
    else:
        print(f"\n🤔 Need further tuning of loss weights...")

if __name__ == "__main__":
    main()
