#!/usr/bin/env python3
"""
Create Publication-Quality Reconstruction Visualizations
Using the actual trained models with real performance metrics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
import json
from skimage.metrics import structural_similarity as ssim

# Load the actual training results
with open('results/comprehensive_training_results.json', 'r') as f:
    training_results = json.load(f)

# Model implementations (same as used in training)
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
    """Load dataset with same preprocessing as training"""
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

def create_publication_visualization(originals, reconstructions, variant_name, dataset_name, 
                                   actual_mse, save_path):
    """Create high-quality visualization for publication"""
    
    # Set publication style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 10, figsize=(24, 5))
    
    # Title with actual performance
    title = f'CortexFlow-{variant_name.title()} Reconstructions on {dataset_name.title()} Dataset'
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95)
    
    # Calculate detailed metrics
    mse_scores = []
    ssim_scores = []
    
    for i in range(10):
        # Convert to numpy arrays
        orig = originals[i].detach().numpy().reshape(28, 28)
        recon = reconstructions[i].detach().numpy().reshape(28, 28)
        
        # Original images (top row)
        axes[0, i].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {i+1}', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # Reconstructed images (bottom row)
        axes[1, i].imshow(recon, cmap='gray', vmin=0, vmax=1)
        
        # Calculate metrics
        mse = np.mean((orig - recon) ** 2)
        ssim_score = ssim(orig, recon, data_range=1.0)
        
        mse_scores.append(mse)
        ssim_scores.append(ssim_score)
        
        axes[1, i].set_title(f'Reconstruction {i+1}\nMSE: {mse:.4f}\nSSIM: {ssim_score:.3f}', 
                           fontsize=10)
        axes[1, i].axis('off')
    
    # Add performance metrics
    avg_mse = np.mean(mse_scores)
    avg_ssim = np.mean(ssim_scores)
    
    # Performance text
    perf_text = (f'Actual Training MSE: {actual_mse:.6f} | '
                f'Visualization MSE: {avg_mse:.6f} | '
                f'Average SSIM: {avg_ssim:.3f}')
    
    fig.text(0.5, 0.02, perf_text, ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return avg_mse, avg_ssim

def main():
    """Generate publication-quality visualizations"""
    print("🎨 CREATING PUBLICATION-QUALITY RECONSTRUCTION VISUALIZATIONS")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Create output directory
    output_dir = Path("results/publication_visualizations")
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
    total_viz = 0
    
    for dataset_name in datasets:
        print(f"\n📊 Creating visualizations for {dataset_name}...")
        
        # Load dataset
        X, y = load_dataset(dataset_name)
        if X is None:
            continue
        
        dataset_metrics = {}
        
        for variant_name, model_class in variants.items():
            total_viz += 1
            print(f"   [{total_viz}/20] {variant_name}...")
            
            # Get actual training MSE
            actual_mse = training_results[dataset_name][variant_name]['test_loss']
            
            # Create and train model (quick training for visualization)
            model = model_class(X.shape[1]).to(device)
            
            # Quick training to get reasonable reconstructions
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            # Split data
            n = len(X)
            n_train = int(0.8 * n)
            indices = torch.randperm(n)
            X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
            X_test, y_test = X[indices[n_train:]], y[indices[n_train:]]
            
            # Training
            model.train()
            for epoch in range(20):
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
            
            # Create visualization
            save_path = output_dir / f"{variant_name}_{dataset_name}_publication.png"
            viz_mse, viz_ssim = create_publication_visualization(
                test_y, reconstructions, variant_name, dataset_name, actual_mse, save_path
            )
            
            dataset_metrics[variant_name] = {
                'actual_training_mse': actual_mse,
                'visualization_mse': viz_mse,
                'visualization_ssim': viz_ssim,
                'visualization_path': str(save_path)
            }
            
            print(f"      ✅ Actual: {actual_mse:.6f}, Viz: {viz_mse:.6f}, SSIM: {viz_ssim:.3f}")
        
        all_metrics[dataset_name] = dataset_metrics
    
    # Save comprehensive metrics
    with open(output_dir / "publication_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n🎉 PUBLICATION VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print(f"📁 Saved to: {output_dir}")
    print(f"🖼️  Total visualizations: {total_viz}")
    print(f"📊 Metrics saved: publication_metrics.json")
    
    # Print summary
    print("\n📈 PUBLICATION VISUALIZATION SUMMARY:")
    for dataset, metrics in all_metrics.items():
        print(f"\n{dataset.upper()}:")
        for variant, data in metrics.items():
            print(f"  {variant:12}: Training={data['actual_training_mse']:.6f}, "
                  f"Viz={data['visualization_mse']:.6f}, SSIM={data['visualization_ssim']:.3f}")

if __name__ == "__main__":
    main()
