#!/usr/bin/env python3
"""
Actual Training Script with Real Datasets
Generate REAL training curves and results
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

class SimpleCortexFlow(nn.Module):
    """Simple CortexFlow implementation for actual training"""
    
    def __init__(self, input_dim, hidden_dim=512, output_dim=784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

def load_dataset(dataset_name):
    """Load actual dataset"""
    print(f"📊 Loading {dataset_name} dataset...")
    
    file_paths = {
        'miyawaki': 'data/processed/miyawaki_structured_28x28.mat',
        'vangerven': 'data/processed/digit69_28x28.mat'
    }
    
    if dataset_name not in file_paths:
        print(f"❌ Dataset {dataset_name} not supported for quick training")
        return None, None
    
    try:
        data = loadmat(file_paths[dataset_name])
        
        # Find the data arrays
        arrays = [(k, v) for k, v in data.items() 
                 if isinstance(v, np.ndarray) and v.ndim >= 2 and not k.startswith('__')]
        arrays.sort(key=lambda x: x[1].size, reverse=True)
        
        if len(arrays) >= 2:
            X_key, X = arrays[0]
            y_key, y = arrays[1]
            
            # Process shapes
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            if y.ndim > 2:
                y = y.reshape(y.shape[0], -1)
            
            # Ensure y is 784 dimensions
            if y.shape[1] != 784:
                if y.shape[1] > 784:
                    y = y[:, :784]
                else:
                    pad_size = 784 - y.shape[1]
                    y = np.pad(y, ((0, 0), (0, pad_size)), mode='constant')
            
            # Normalize
            X = (X - X.mean()) / (X.std() + 1e-8)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            
            print(f"✅ {dataset_name} loaded: X{X.shape}, y{y.shape}")
            return torch.FloatTensor(X), torch.FloatTensor(y)
        else:
            print(f"❌ Could not identify data arrays in {dataset_name}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None, None

def train_model_actual(model, train_loader, val_loader, dataset_name, max_epochs=50):
    """Train model and record ACTUAL training curves"""
    print(f"🚀 Training CortexFlow-Simple on {dataset_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
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
    
    for epoch in range(max_epochs):
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
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f}s, Best Val Loss: {best_val_loss:.6f}")
    
    return history, best_val_loss

def create_actual_training_curves(all_histories, results_dir):
    """Create training curves from actual training data"""
    print("📈 Creating ACTUAL training curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('CortexFlow: Kurva Training Aktual dengan Dataset Asli', 
                 fontsize=14, fontweight='bold')
    
    colors = ['#4A90E2', '#7ED321']
    
    for i, (dataset, history) in enumerate(all_histories.items()):
        ax = axes[i]
        
        epochs = history['epochs']
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        
        # Plot actual training curves
        ax.plot(epochs, train_loss, color=colors[i], linewidth=2, 
               label='Training Loss', alpha=0.8)
        ax.plot(epochs, val_loss, color=colors[i], linewidth=2, 
               linestyle='--', label='Validation Loss', alpha=0.8)
        
        # Mark best point
        best_epoch = np.argmin(val_loss)
        best_val = val_loss[best_epoch]
        ax.scatter(best_epoch, best_val, color='red', s=100, zorder=5, 
                  marker='*', label=f'Best: {best_val:.4f}')
        
        ax.set_title(f'{dataset.title()} Dataset', fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (MSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(results_dir / "actual_training_curves.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ ACTUAL training curves saved")

def main():
    """Main execution - Run actual training"""
    print("🚀 RUNNING ACTUAL TRAINING EXPERIMENTS...")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("results/actual_experiments/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Datasets to train (start with 2 for speed)
    datasets = ['miyawaki', 'vangerven']
    all_histories = {}
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n📊 Processing {dataset_name}...")
        
        # Load dataset
        X, y = load_dataset(dataset_name)
        if X is None or y is None:
            continue
        
        # Split data
        n_samples = len(X)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        # Random shuffle
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Create and train model
        model = SimpleCortexFlow(X.shape[1])
        history, best_val_loss = train_model_actual(
            model, train_loader, val_loader, dataset_name, max_epochs=30
        )
        
        # Evaluate on test set
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                test_loss = nn.MSELoss()(output, batch_y)
                test_losses.append(test_loss.item())
        
        test_mse = np.mean(test_losses)
        
        all_histories[dataset_name] = history
        all_results[dataset_name] = {
            'test_mse': test_mse,
            'best_val_loss': best_val_loss,
            'final_epoch': len(history['epochs']) - 1,
            'n_samples': n_samples
        }
        
        print(f"✅ {dataset_name} complete: Test MSE = {test_mse:.6f}")
    
    # Create visualizations
    if all_histories:
        create_actual_training_curves(all_histories, results_dir)
    
    # Save results
    with open(results_dir.parent / "data" / "actual_training_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n🎉 ACTUAL TRAINING EXPERIMENTS COMPLETE!")
    print("=" * 60)
    print("📊 Results:")
    for dataset, results in all_results.items():
        print(f"   {dataset}: MSE = {results['test_mse']:.6f}, Epochs = {results['final_epoch']}")
    
    print(f"\n📁 Files saved to: {results_dir}")
    print("✅ REAL training curves generated!")

if __name__ == "__main__":
    main()
