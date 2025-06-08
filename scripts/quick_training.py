#!/usr/bin/env python3
"""
Quick Training Script - Generate REAL Training Curves
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
import warnings
warnings.filterwarnings('ignore')

class QuickCortexFlow(nn.Module):
    """Simplified CortexFlow for quick training"""
    
    def __init__(self, input_dim, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def load_miyawaki_quick():
    """Quick load Miyawaki dataset"""
    try:
        data = loadmat('data/processed/miyawaki_structured_28x28.mat')
        
        # Get training data
        X = data['fmriTrn']  # (107, 967)
        y = data['stimTrn']  # (107, 784)
        
        # Normalize
        X = (X - X.mean()) / (X.std() + 1e-8)
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        print(f"✅ Miyawaki loaded: X{X.shape}, y{y.shape}")
        return torch.FloatTensor(X), torch.FloatTensor(y)
        
    except Exception as e:
        print(f"❌ Error loading Miyawaki: {e}")
        return None, None

def quick_train(model, X, y, epochs=20):
    """Quick training with real curves"""
    print("🚀 Starting quick training...")
    
    # Split data
    n = len(X)
    n_train = int(0.8 * n)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    
    # Data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training history
    train_losses = []
    val_losses = []
    epochs_list = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        # Record
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        epochs_list.append(epoch)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train={avg_train:.6f}, Val={avg_val:.6f}")
    
    print("✅ Training complete!")
    
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'epochs': epochs_list
    }, model

def create_real_training_curves(history, results_dir):
    """Create REAL training curves"""
    print("📈 Creating REAL training curves...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    # Plot REAL curves
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax.plot(epochs, val_loss, 'r--', linewidth=2, label='Validation Loss', alpha=0.8)
    
    # Mark best point
    best_epoch = np.argmin(val_loss)
    best_val = val_loss[best_epoch]
    ax.scatter(best_epoch, best_val, color='red', s=100, zorder=5, 
              marker='*', label=f'Best Val: {best_val:.4f}')
    
    ax.set_title('CortexFlow: Kurva Training Aktual (Dataset Miyawaki)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Add final values
    final_train = train_loss[-1]
    final_val = val_loss[-1]
    ax.text(0.02, 0.98, f'Final Training: {final_train:.4f}\nFinal Validation: {final_val:.4f}', 
           transform=ax.transAxes, fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
           verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(results_dir / "real_training_curves_actual.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ REAL training curves saved")

def create_real_reconstructions(model, X, y, results_dir):
    """Create REAL reconstruction examples"""
    print("🖼️ Creating REAL reconstruction examples...")
    
    model.eval()
    
    # Get first 4 samples
    with torch.no_grad():
        outputs = model(X[:4])
    
    # Convert to numpy
    originals = y[:4].numpy()
    reconstructions = outputs.numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('CortexFlow: Rekonstruksi Visual Aktual (Dataset Miyawaki)', 
                 fontsize=14, fontweight='bold')
    
    for i in range(4):
        # Original
        orig_img = originals[i].reshape(28, 28)
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        recon_img = reconstructions[i].reshape(28, 28)
        axes[1, i].imshow(recon_img, cmap='gray')
        
        # Calculate MSE for this sample
        mse = np.mean((orig_img - recon_img)**2)
        axes[1, i].set_title(f'Rekonstruksi {i+1}\nMSE: {mse:.4f}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(results_dir / "real_reconstructions_actual.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ REAL reconstructions saved")

def main():
    """Main execution"""
    print("🚀 GENERATING REAL TRAINING DATA...")
    print("=" * 50)
    
    # Create results directory
    results_dir = Path("results/actual_experiments/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y = load_miyawaki_quick()
    if X is None:
        print("❌ Cannot load data")
        return
    
    # Create model
    model = QuickCortexFlow(X.shape[1])
    print(f"📊 Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    history, trained_model = quick_train(model, X, y, epochs=20)
    
    # Create visualizations
    create_real_training_curves(history, results_dir)
    create_real_reconstructions(trained_model, X, y, results_dir)
    
    # Save training data
    with open(results_dir.parent / "data" / "real_training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n🎉 REAL TRAINING DATA GENERATED!")
    print("📁 Files created:")
    print("   - real_training_curves_actual.png")
    print("   - real_reconstructions_actual.png")
    print("   - real_training_history.json")
    print("\n✅ Now using ACTUAL training results!")

if __name__ == "__main__":
    main()
