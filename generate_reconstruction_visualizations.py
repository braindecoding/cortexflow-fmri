#!/usr/bin/env python3
"""
Generate Comprehensive Reconstruction Visualizations
For all CortexFlow variants on all datasets
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import json
import warnings
warnings.filterwarnings('ignore')

# Import our models
import sys
import os
sys.path.append('.')

# Simple model implementations for visualization
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

class MCModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 784), nn.Sigmoid()
        )
        self.uncertainty = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 784), nn.Softplus()
        )
    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        unc = self.uncertainty(z)
        return output, unc

class HierarchicalModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scale1 = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.scale2 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.scale3 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(448, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid()
        )
    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        combined = torch.cat([s1, s2, s3], dim=-1)
        return self.decoder(combined)

class EnhancedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hierarchical = HierarchicalModel(input_dim)
        self.uncertainty = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Softplus()
        )
        self.alignment = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.Tanh()
        )
    def forward(self, x):
        output = self.hierarchical(x)
        unc = self.uncertainty(x)
        align = self.alignment(x)
        return output, unc, align

class UnifiedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.complexity = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        self.simple = SimpleModel(input_dim)
        self.complex = HierarchicalModel(input_dim)
        self.uncertainty = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Softplus()
        )
    def forward(self, x):
        comp = self.complexity(x)
        simple_out = self.simple(x)
        complex_out = self.complex(x)
        output = comp * complex_out + (1 - comp) * simple_out
        unc = self.uncertainty(x)
        return output, unc, comp

def load_dataset(dataset_name):
    """Load and preprocess dataset"""
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

def train_and_visualize_model(model, X, y, variant_name, dataset_name, device):
    """Train model and generate reconstructions"""
    model = model.to(device)
    
    # Quick training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Split data
    n = len(X)
    n_train = int(0.8 * n)
    indices = torch.randperm(n)
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_test, y_test = X[indices[n_train:]], y[indices[n_train:]]
    
    # Training
    model.train()
    for epoch in range(20):  # Quick training
        for i in range(0, len(X_train), 16):
            batch_x = X_train[i:i+16].to(device)
            batch_y = y_train[i:i+16].to(device)
            
            optimizer.zero_grad()
            
            if variant_name == 'simple':
                output = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant_name == 'mc':
                output, _ = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant_name == 'hierarchical':
                output = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant_name == 'enhanced':
                output, _, _ = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant_name == 'unified':
                output, _, _ = model(batch_x)
                loss = criterion(output, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        # Select 10 test samples
        test_indices = torch.randperm(len(X_test))[:10]
        test_x = X_test[test_indices].to(device)
        test_y = y_test[test_indices]
        
        if variant_name == 'simple':
            reconstructions = model(test_x)
        elif variant_name == 'mc':
            reconstructions, _ = model(test_x)
        elif variant_name == 'hierarchical':
            reconstructions = model(test_x)
        elif variant_name == 'enhanced':
            reconstructions, _, _ = model(test_x)
        elif variant_name == 'unified':
            reconstructions, _, _ = model(test_x)
        
        reconstructions = reconstructions.cpu()
    
    return test_y, reconstructions

def create_reconstruction_visualization(originals, reconstructions, variant_name, dataset_name, save_path):
    """Create visualization with originals on top, reconstructions on bottom"""
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    fig.suptitle(f'CortexFlow-{variant_name.title()} on {dataset_name.title()} Dataset', fontsize=16, fontweight='bold')
    
    # Calculate metrics
    mse_scores = []
    ssim_scores = []
    
    for i in range(10):
        # Original images (top row)
        orig_img = originals[i].detach().numpy().reshape(28, 28)
        axes[0, i].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {i+1}', fontsize=10)
        axes[0, i].axis('off')

        # Reconstructed images (bottom row)
        recon_img = reconstructions[i].detach().numpy().reshape(28, 28)
        axes[1, i].imshow(recon_img, cmap='gray', vmin=0, vmax=1)

        # Calculate metrics
        mse = np.mean((orig_img - recon_img) ** 2)
        ssim_score = ssim(orig_img, recon_img, data_range=1.0)
        
        mse_scores.append(mse)
        ssim_scores.append(ssim_score)
        
        axes[1, i].set_title(f'Recon {i+1}\nMSE: {mse:.4f}\nSSIM: {ssim_score:.3f}', fontsize=9)
        axes[1, i].axis('off')
    
    # Add overall metrics
    avg_mse = np.mean(mse_scores)
    avg_ssim = np.mean(ssim_scores)
    
    fig.text(0.5, 0.02, f'Average MSE: {avg_mse:.6f} | Average SSIM: {avg_ssim:.3f}', 
             ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return avg_mse, avg_ssim

def main():
    """Generate all reconstruction visualizations"""
    print("🎨 GENERATING COMPREHENSIVE RECONSTRUCTION VISUALIZATIONS")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Create output directory
    output_dir = Path("results/reconstruction_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    variants = {
        'simple': SimpleModel,
        'mc': MCModel,
        'hierarchical': HierarchicalModel,
        'enhanced': EnhancedModel,
        'unified': UnifiedModel
    }
    
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    all_metrics = {}
    total_visualizations = len(variants) * len(datasets)
    current_viz = 0
    
    for dataset_name in datasets:
        print(f"\n📊 Processing {dataset_name} dataset...")
        
        # Load dataset
        X, y = load_dataset(dataset_name)
        if X is None:
            continue
        
        dataset_metrics = {}
        
        for variant_name, model_class in variants.items():
            current_viz += 1
            print(f"[{current_viz}/{total_visualizations}] Training {variant_name} on {dataset_name}...")
            
            # Create and train model
            model = model_class(X.shape[1])
            originals, reconstructions = train_and_visualize_model(
                model, X, y, variant_name, dataset_name, device
            )
            
            # Create visualization
            save_path = output_dir / f"{variant_name}_{dataset_name}_reconstruction.png"
            avg_mse, avg_ssim = create_reconstruction_visualization(
                originals, reconstructions, variant_name, dataset_name, save_path
            )
            
            dataset_metrics[variant_name] = {
                'mse': avg_mse,
                'ssim': avg_ssim,
                'visualization': str(save_path)
            }
            
            print(f"   ✅ {variant_name}: MSE={avg_mse:.6f}, SSIM={avg_ssim:.3f}")
        
        all_metrics[dataset_name] = dataset_metrics
    
    # Save metrics
    with open(output_dir / "reconstruction_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n🎉 ALL VISUALIZATIONS COMPLETE!")
    print("=" * 70)
    print(f"📁 Saved to: {output_dir}")
    print(f"🖼️  Total visualizations: {current_viz}")
    print(f"📊 Metrics saved: reconstruction_metrics.json")
    
    # Print summary
    print("\n📈 RECONSTRUCTION QUALITY SUMMARY:")
    for dataset, metrics in all_metrics.items():
        print(f"\n{dataset.upper()}:")
        for variant, data in metrics.items():
            print(f"  {variant:12}: MSE={data['mse']:.6f}, SSIM={data['ssim']:.3f}")

if __name__ == "__main__":
    main()
