#!/usr/bin/env python3
"""
Comprehensive Training Script - All Datasets
Generate REAL training results for all 4 datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
import json
import time
import warnings
warnings.filterwarnings('ignore')

class CortexFlowModel(nn.Module):
    """CortexFlow model for comprehensive training"""
    
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def load_dataset(dataset_name):
    """Load dataset with proper preprocessing"""
    print(f"📊 Loading {dataset_name} dataset...")
    
    file_paths = {
        'miyawaki': 'data/processed/miyawaki_structured_28x28.mat',
        'vangerven': 'data/processed/digit69_28x28.mat',
        'mindbigdata': 'data/processed/mindbigdata.mat',
        'crell': 'data/processed/crell.mat'
    }
    
    try:
        data = loadmat(file_paths[dataset_name])
        
        # Extract training data based on dataset structure
        if dataset_name == 'miyawaki':
            X = data['fmriTrn']  # (107, 967)
            y = data['stimTrn']  # (107, 784)
        else:
            # For other datasets, find the largest arrays
            arrays = [(k, v) for k, v in data.items() 
                     if isinstance(v, np.ndarray) and v.ndim >= 2 and not k.startswith('__')]
            arrays.sort(key=lambda x: x[1].size, reverse=True)
            
            if len(arrays) >= 2:
                X = arrays[0][1]  # Largest array (likely fMRI/EEG)
                y = arrays[1][1]  # Second largest (likely images)
            else:
                print(f"❌ Could not identify data arrays in {dataset_name}")
                return None, None
        
        # Process shapes
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        
        # Ensure y is 784 dimensions (28x28)
        if y.shape[1] != 784:
            if y.shape[1] > 784:
                y = y[:, :784]
            else:
                pad_size = 784 - y.shape[1]
                y = np.pad(y, ((0, 0), (0, pad_size)), mode='constant')
        
        # Normalize data
        X = (X - X.mean()) / (X.std() + 1e-8)
        
        # Normalize y based on data type
        if y.max() > 1.0:  # Likely 0-255 range
            y = y / 255.0
        else:  # Already 0-1 range
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        print(f"✅ {dataset_name} loaded: X{X.shape}, y{y.shape}")
        print(f"   X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"   y range: [{y.min():.3f}, {y.max():.3f}]")
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None, None

def train_on_dataset(dataset_name, X, y, epochs=25):
    """Train model on specific dataset"""
    print(f"\n🚀 Training CortexFlow on {dataset_name}...")
    
    # Create model
    model = CortexFlowModel(X.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Split data
    n = len(X)
    n_train = int(0.75 * n)
    n_val = int(0.15 * n)
    
    # Random shuffle
    indices = torch.randperm(n)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                val_loss = criterion(output, batch_y)
                val_losses.append(val_loss.item())
        
        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['epochs'].append(epoch)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'results/actual_experiments/models/{dataset_name}_best.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, LR={optimizer.param_groups[0]['lr']:.6f}")
    
    training_time = time.time() - start_time
    
    # Test evaluation
    model.eval()
    test_losses = []
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            test_loss = criterion(output, batch_y)
            test_losses.append(test_loss.item())
            
            predictions.append(output.cpu().numpy())
            targets.append(batch_y.cpu().numpy())
    
    test_mse = np.mean(test_losses)
    
    # Calculate SSIM approximation
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    ssim_scores = []
    for i in range(min(len(predictions), 50)):  # Sample for efficiency
        pred_img = predictions[i].reshape(28, 28)
        target_img = targets[i].reshape(28, 28)
        
        # Simplified SSIM
        mu1, mu2 = pred_img.mean(), target_img.mean()
        sigma1, sigma2 = pred_img.std(), target_img.std()
        sigma12 = np.mean((pred_img - mu1) * (target_img - mu2))
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        ssim_scores.append(max(0, min(1, ssim)))
    
    avg_ssim = np.mean(ssim_scores)
    
    results = {
        'dataset': dataset_name,
        'test_mse': test_mse,
        'test_ssim': avg_ssim,
        'best_val_loss': best_val_loss,
        'final_epoch': len(history['epochs']) - 1,
        'training_time': training_time,
        'n_samples': n,
        'n_parameters': sum(p.numel() for p in model.parameters()),
        'history': history
    }
    
    print(f"✅ {dataset_name} complete:")
    print(f"   Test MSE: {test_mse:.6f}")
    print(f"   Test SSIM: {avg_ssim:.3f}")
    print(f"   Training time: {training_time:.1f}s")
    print(f"   Parameters: {results['n_parameters']:,}")
    
    return results, model

def main():
    """Main execution - comprehensive training"""
    print("🚀 COMPREHENSIVE TRAINING ON ALL DATASETS")
    print("=" * 60)
    
    # Create directories
    results_dir = Path("results/actual_experiments")
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "data").mkdir(exist_ok=True)
    
    # Datasets to train
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")
        
        # Load dataset
        X, y = load_dataset(dataset_name)
        if X is None or y is None:
            print(f"❌ Skipping {dataset_name} due to loading error")
            continue
        
        # Train model
        try:
            results, model = train_on_dataset(dataset_name, X, y, epochs=30)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"❌ Training failed for {dataset_name}: {e}")
            continue
    
    # Save comprehensive results
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for dataset, results in all_results.items():
        json_results[dataset] = {
            'dataset': results['dataset'],
            'test_mse': float(results['test_mse']),
            'test_ssim': float(results['test_ssim']),
            'best_val_loss': float(results['best_val_loss']),
            'final_epoch': int(results['final_epoch']),
            'training_time': float(results['training_time']),
            'n_samples': int(results['n_samples']),
            'n_parameters': int(results['n_parameters']),
            'history': {
                'train_loss': [float(x) for x in results['history']['train_loss']],
                'val_loss': [float(x) for x in results['history']['val_loss']],
                'epochs': [int(x) for x in results['history']['epochs']],
                'learning_rate': [float(x) for x in results['history']['learning_rate']]
            }
        }
    
    with open(results_dir / "data" / "comprehensive_training_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print summary
    print(f"\n🎉 COMPREHENSIVE TRAINING COMPLETE!")
    print("=" * 60)
    print("📊 SUMMARY RESULTS:")
    print(f"{'Dataset':<12} {'MSE':<10} {'SSIM':<8} {'Epochs':<8} {'Time(s)':<8} {'Params':<10}")
    print("-" * 70)
    
    for dataset, results in all_results.items():
        print(f"{dataset:<12} {results['test_mse']:<10.6f} {results['test_ssim']:<8.3f} "
              f"{results['final_epoch']:<8} {results['training_time']:<8.1f} {results['n_parameters']:<10,}")
    
    print(f"\n💾 Results saved to: {results_dir}")
    print("✅ All datasets trained with ACTUAL results!")

if __name__ == "__main__":
    main()
