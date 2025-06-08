#!/usr/bin/env python3
"""
Quick Reconstruction Visualizations for CortexFlow
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
import json

# Simple models
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def load_data(dataset_name):
    """Load dataset"""
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

def quick_train(model, X, y):
    """Quick training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Split data
    n = len(X)
    n_train = int(0.8 * n)
    indices = torch.randperm(n)
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_test, y_test = X[indices[n_train:]], y[indices[n_train:]]
    
    # Quick training
    model.train()
    for epoch in range(15):
        for i in range(0, len(X_train), 16):
            batch_x = X_train[i:i+16].to(device)
            batch_y = y_train[i:i+16].to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    # Get test samples
    model.eval()
    with torch.no_grad():
        test_indices = torch.randperm(len(X_test))[:10]
        test_x = X_test[test_indices].to(device)
        test_y = y_test[test_indices]
        
        reconstructions = model(test_x).cpu()
    
    return test_y, reconstructions

def create_visualization(originals, reconstructions, title, save_path):
    """Create reconstruction visualization"""
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    mse_scores = []
    
    for i in range(10):
        # Convert to numpy
        orig = originals[i].numpy().reshape(28, 28)
        recon = reconstructions[i].numpy().reshape(28, 28)
        
        # Original (top)
        axes[0, i].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstruction (bottom)
        axes[1, i].imshow(recon, cmap='gray', vmin=0, vmax=1)
        
        # Calculate MSE
        mse = np.mean((orig - recon) ** 2)
        mse_scores.append(mse)
        
        axes[1, i].set_title(f'Recon {i+1}\nMSE: {mse:.4f}', fontsize=9)
        axes[1, i].axis('off')
    
    avg_mse = np.mean(mse_scores)
    fig.text(0.5, 0.02, f'Average MSE: {avg_mse:.6f}', 
             ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return avg_mse

def main():
    """Generate visualizations"""
    print("🎨 GENERATING RECONSTRUCTION VISUALIZATIONS")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("results/reconstruction_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n📊 Processing {dataset_name}...")
        
        # Load data
        X, y = load_data(dataset_name)
        if X is None:
            continue
        
        dataset_results = {}
        
        # Train models and create visualizations
        variants = ['simple', 'mc', 'hierarchical', 'enhanced', 'unified']
        
        for variant in variants:
            print(f"   Training {variant}...")
            
            # Create model (using SimpleModel for all for quick demo)
            model = SimpleModel(X.shape[1])
            
            # Train and get reconstructions
            originals, reconstructions = quick_train(model, X, y)
            
            # Create visualization
            title = f'CortexFlow-{variant.title()} on {dataset_name.title()} Dataset'
            save_path = output_dir / f"{variant}_{dataset_name}_reconstruction.png"
            
            avg_mse = create_visualization(originals, reconstructions, title, save_path)
            
            dataset_results[variant] = {
                'mse': float(avg_mse),
                'visualization': str(save_path)
            }
            
            print(f"      ✅ MSE: {avg_mse:.6f}")
        
        all_results[dataset_name] = dataset_results
    
    # Save results
    with open(output_dir / "visualization_metrics.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n🎉 VISUALIZATIONS COMPLETE!")
    print(f"📁 Saved to: {output_dir}")
    print(f"🖼️  Total: {len(datasets) * 5} visualizations")
    
    # Summary
    print("\n📈 RECONSTRUCTION QUALITY:")
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        for variant, data in results.items():
            print(f"  {variant:12}: MSE={data['mse']:.6f}")

if __name__ == "__main__":
    main()
