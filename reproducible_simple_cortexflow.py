#!/usr/bin/env python3
"""
Reproducible Simple CortexFlow Training on All Datasets
Clean, consistent implementation with fixed seeds for reproducibility
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import random
import scipy.io
from PIL import Image

# Add src to path
sys.path.append('src')

from data.data_loader import FMRIDataLoader
from utils.visualization import plot_reconstruction_comparison

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleCortexFlow(nn.Module):
    """Simple CortexFlow model with MSE loss only - Universal for all datasets."""
    
    def __init__(self, input_dim, image_size=28, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        
        # Simple encoder: input -> hidden features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Simple decoder: hidden features -> image
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 4, image_size * image_size),
            nn.Sigmoid()  # Output [0, 1]
        )
        
    def forward(self, input_data):
        """Forward pass."""
        # Encode input to hidden features
        features = self.encoder(input_data)
        
        # Decode to image
        reconstruction = self.decoder(features)
        
        # Reshape to image format
        reconstruction = reconstruction.view(-1, 1, self.image_size, self.image_size)
        
        return reconstruction
    
    def compute_loss(self, input_data, target_images):
        """Compute simple MSE loss."""
        # Forward pass
        reconstruction = self.forward(input_data)
        
        # Ensure target images are in correct format
        if target_images.dim() == 2:  # [batch, 784]
            target_images = target_images.view(-1, 1, self.image_size, self.image_size)
        
        # Simple MSE loss
        mse_loss = nn.functional.mse_loss(reconstruction, target_images)
        
        return {
            'total_loss': mse_loss,
            'mse_loss': mse_loss,
            'reconstruction': reconstruction
        }

def train_dataset(dataset_name, data_loader_func, device, results_summary):
    """Train Simple CortexFlow on a specific dataset."""
    print(f"\n{'='*80}")
    print(f"🧠 REPRODUCIBLE SIMPLE CORTEXFLOW: {dataset_name.upper()} DATASET")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set seeds for this dataset
    set_seeds(42)
    
    # Load data
    print(f"\n📊 Loading {dataset_name} dataset...")
    train_loader, test_loader, input_dim = data_loader_func()
    
    print(f"✅ Dataset loaded: {input_dim} input dimensions")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Test batches: {len(test_loader)}")
    
    # Initialize model
    print(f"\n🧠 Initializing Simple CortexFlow model...")
    model = SimpleCortexFlow(
        input_dim=input_dim,
        image_size=28,
        hidden_dim=512
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Total parameters: {total_params:,}")
    print(f"🔧 Trainable parameters: {trainable_params:,}")
    print(f"🎯 Loss: Simple MSE only")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training parameters
    num_epochs = 200
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Training history
    train_losses = []
    test_losses = []
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (input_data, images, labels) in enumerate(train_loader):
            input_data = input_data.to(device)
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model.compute_loss(input_data, images)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item()
            
            # Progress update (every 10 batches or if few batches)
            if batch_idx % max(1, len(train_loader) // 5) == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Batch {batch_idx:3d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.6f}")
        
        # Average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for input_data, images, labels in test_loader:
                input_data = input_data.to(device)
                images = images.to(device)
                
                loss_dict = model.compute_loss(input_data, images)
                test_loss += loss_dict['total_loss'].item()
        
        avg_test_loss = test_loss / len(test_loader)
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        
        # Record losses
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        # Print epoch summary (every 5 epochs or significant epochs)
        if epoch % 5 == 0 or epoch < 10:
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch+1:3d}/{num_epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Test Loss:  {avg_test_loss:.6f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Time: {elapsed/60:.1f}m")
        
        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            
            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'dataset': dataset_name,
                'input_dim': input_dim,
                'total_params': total_params
            }, f'checkpoints/simple_{dataset_name.lower()}_model.pt')
            
            if epoch % 5 == 0 or epoch < 10:
                print(f"  ✅ New best model saved! (Loss: {best_loss:.6f})")
        else:
            patience_counter += 1
            if epoch % 5 == 0 or epoch < 10:
                print(f"  ⏳ Patience: {patience_counter}/{patience}")
        
        if epoch % 5 == 0 or epoch < 10:
            print("-" * 80)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time/60:.1f} minutes!")
    print(f"🏆 Best test loss: {best_loss:.6f}")
    
    # Generate sample reconstructions
    print("🎨 Generating sample reconstructions...")
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        input_data, images, labels = next(iter(test_loader))
        input_data = input_data[:8].to(device)  # First 8 samples
        images = images[:8].to(device)
        
        # Generate reconstructions
        reconstructions = model(input_data)
        
        # Reshape images for visualization
        if images.dim() == 2:  # [batch, 784]
            images = images.view(-1, 1, 28, 28)
        
        # Plot comparisons
        images_viz = images.cpu().squeeze(1) if images.dim() == 4 else images.cpu()
        reconstructions_viz = reconstructions.cpu().squeeze(1) if reconstructions.dim() == 4 else reconstructions.cpu()
        
        plot_reconstruction_comparison(
            images_viz, 
            reconstructions_viz,
            save_path=f'results/reproducible_{dataset_name.lower()}_reconstructions.png',
            title=f'Reproducible Simple CortexFlow: {dataset_name} Reconstructions'
        )
    
    # Store results
    results_summary[dataset_name] = {
        'best_test_loss': best_loss,
        'training_time_minutes': total_time / 60,
        'total_epochs': epoch + 1,
        'total_params': total_params,
        'input_dim': input_dim,
        'final_train_loss': train_losses[-1],
        'convergence_epoch': len(train_losses) - patience_counter
    }
    
    print(f"\n✅ {dataset_name} training completed successfully!")
    print(f"📁 Results saved to:")
    print(f"  - checkpoints/simple_{dataset_name.lower()}_model.pt")
    print(f"  - results/reproducible_{dataset_name.lower()}_reconstructions.png")
    
    return results_summary

def main():
    """Main reproducible training function."""
    print("🧠 REPRODUCIBLE SIMPLE CORTEXFLOW TRAINING ON ALL DATASETS")
    print("=" * 80)
    print(f"Experiment start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set global seeds
    set_seeds(42)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎮 Using device: {device}")
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Results summary
    results_summary = {}
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Dataset loading functions
    def load_miyawaki():
        data_loader = FMRIDataLoader(data_path='data/miyawaki_structured_28x28.mat')
        train_loader = data_loader.create_dataloader(split='train', batch_size=32, shuffle=True)
        test_loader = data_loader.create_dataloader(split='test', batch_size=32, shuffle=False)
        input_dim = data_loader.get_fmri('train').shape[1]
        return train_loader, test_loader, input_dim
    
    def load_vangerven():
        data_loader = FMRIDataLoader(data_path='data/digit69_28x28.mat')
        train_loader = data_loader.create_dataloader(split='train', batch_size=32, shuffle=True)
        test_loader = data_loader.create_dataloader(split='test', batch_size=32, shuffle=False)
        input_dim = data_loader.get_fmri('train').shape[1]
        return train_loader, test_loader, input_dim
    
    # Train on all datasets
    datasets = [
        ('Miyawaki', load_miyawaki),
        ('Vangerven', load_vangerven),
    ]
    
    total_start_time = time.time()
    
    for dataset_name, loader_func in datasets:
        try:
            results_summary = train_dataset(dataset_name, loader_func, device, results_summary)
        except Exception as e:
            print(f"❌ Error training {dataset_name}: {e}")
            results_summary[dataset_name] = {'error': str(e)}
    
    total_time = time.time() - total_start_time
    
    # Print final summary
    print(f"\n{'='*80}")
    print("🎉 REPRODUCIBLE TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Total experiment time: {total_time/60:.1f} minutes")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print("-" * 80)
    for dataset, results in results_summary.items():
        if 'error' in results:
            print(f"❌ {dataset:12s}: ERROR - {results['error']}")
        else:
            print(f"✅ {dataset:12s}: Loss {results['best_test_loss']:.6f} | "
                  f"Time {results['training_time_minutes']:.1f}m | "
                  f"Epochs {results['total_epochs']:3d} | "
                  f"Params {results['total_params']:,}")
    
    print(f"\n🏆 BEST PERFORMANCE:")
    if results_summary:
        valid_results = {k: v for k, v in results_summary.items() if 'error' not in v}
        if valid_results:
            best_dataset = min(valid_results.keys(), key=lambda k: valid_results[k]['best_test_loss'])
            best_loss = valid_results[best_dataset]['best_test_loss']
            print(f"🥇 {best_dataset}: {best_loss:.6f} test loss")
    
    print(f"\n📁 All results saved to:")
    print(f"  - checkpoints/simple_*_model.pt")
    print(f"  - results/reproducible_*_reconstructions.png")

if __name__ == "__main__":
    main()
