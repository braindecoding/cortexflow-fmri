#!/usr/bin/env python3
"""
Unified CortexFlow Training Script
Trains the unified model that combines best features from all architectures
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.unified_cortexflow_fixed import create_unified_model, UnifiedCortexFlow, UnifiedLoss
from scipy.io import loadmat

def load_data(data_path):
    """Load and preprocess data."""
    print(f"📁 Loading data from: {data_path}")
    
    try:
        data = loadmat(data_path)
        
        # Extract data based on available keys
        available_keys = list(data.keys())
        print(f"  ✅ Available keys: {[k for k in available_keys if not k.startswith('__')]}")
        
        # Try different key combinations
        if 'fmriTrn' in data:
            fmri_train = data['fmriTrn']
            stim_train = data['stimTrn'] 
            fmri_test = data['fmriTest']
            stim_test = data['stimTest']
            label_train = data['labelTrn'] if 'labelTrn' in data else None
            label_test = data['labelTest'] if 'labelTest' in data else None
        else:
            # Alternative key names
            keys = [k for k in available_keys if not k.startswith('__')]
            print(f"  ⚠️  Using alternative keys: {keys}")
            fmri_train = data[keys[0]]
            stim_train = data[keys[1]] if len(keys) > 1 else fmri_train
            fmri_test = fmri_train[:10]  # Use subset for testing
            stim_test = stim_train[:10]
            label_train = None
            label_test = None
        
        print(f"  ✅ fmriTrn: {fmri_train.shape}")
        print(f"  ✅ stimTrn: {stim_train.shape}")
        print(f"  ✅ fmriTest: {fmri_test.shape}")
        print(f"  ✅ stimTest: {stim_test.shape}")
        
        if label_train is not None:
            print(f"  ✅ labelTrn: {label_train.shape}")
            print(f"  ✅ labelTest: {label_test.shape}")
        
        return {
            'fmri_train': fmri_train,
            'stim_train': stim_train,
            'fmri_test': fmri_test,
            'stim_test': stim_test,
            'label_train': label_train,
            'label_test': label_test
        }
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def preprocess_data(data_dict):
    """Preprocess and normalize data."""
    print("\n📊 Data Information:")
    
    # Convert to tensors and normalize
    fmri_train = torch.FloatTensor(data_dict['fmri_train'])
    stim_train = torch.FloatTensor(data_dict['stim_train'])
    fmri_test = torch.FloatTensor(data_dict['fmri_test'])
    stim_test = torch.FloatTensor(data_dict['stim_test'])
    
    # Normalize fMRI data (z-score)
    fmri_mean = fmri_train.mean(dim=0, keepdim=True)
    fmri_std = fmri_train.std(dim=0, keepdim=True) + 1e-8
    fmri_train = (fmri_train - fmri_mean) / fmri_std
    fmri_test = (fmri_test - fmri_mean) / fmri_std
    
    # Normalize stimuli to [0, 1]
    stim_train = (stim_train - stim_train.min()) / (stim_train.max() - stim_train.min() + 1e-8)
    stim_test = (stim_test - stim_test.min()) / (stim_test.max() - stim_test.min() + 1e-8)
    
    # Handle labels if available
    labels_train = None
    labels_test = None
    if data_dict['label_train'] is not None:
        labels_train = torch.LongTensor(data_dict['label_train'].flatten())
        labels_test = torch.LongTensor(data_dict['label_test'].flatten())
    
    print(f"\n🎯 Train Data:")
    print(f"  stimuli: {stim_train.shape}")
    print(f"    Range: [{stim_train.min():.3f}, {stim_train.max():.3f}]")
    print(f"  fmri: {fmri_train.shape}")
    print(f"    Mean: {fmri_train.mean():.3f}, Std: {fmri_train.std():.3f}")
    if labels_train is not None:
        print(f"  labels: {labels_train.shape}")
    
    print(f"\n🎯 Test Data:")
    print(f"  stimuli: {stim_test.shape}")
    print(f"    Range: [{stim_test.min():.3f}, {stim_test.max():.3f}]")
    print(f"  fmri: {fmri_test.shape}")
    print(f"    Mean: {fmri_test.mean():.3f}, Std: {fmri_test.std():.3f}")
    if labels_test is not None:
        print(f"  labels: {labels_test.shape}")
    
    return {
        'fmri_train': fmri_train,
        'stim_train': stim_train,
        'fmri_test': fmri_test,
        'stim_test': stim_test,
        'labels_train': labels_train,
        'labels_test': labels_test
    }

def train_unified_model(model, loss_fn, data, config_name, dataset_name, device, max_epochs=200):
    """Train the unified model."""
    print(f"\n{'='*80}")
    print(f"🔬 UNIFIED CORTEXFLOW: {dataset_name.upper()} DATASET")
    print(f"{'='*80}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Move to device
    model = model.to(device)
    
    # Data
    fmri_train = data['fmri_train'].to(device)
    stim_train = data['stim_train'].to(device)
    fmri_test = data['fmri_test'].to(device)
    stim_test = data['stim_test'].to(device)
    
    print(f"✅ Dataset loaded: {fmri_train.shape[1]} input dimensions")
    print(f"🔧 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"⚙️  Configuration: {config_name}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    train_losses = []
    test_losses = []
    complexity_scores = []
    uncertainty_scores = []
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        epoch_complexity = []
        epoch_uncertainty = []
        
        # Mini-batch training
        batch_size = min(32, fmri_train.size(0))
        num_batches = (fmri_train.size(0) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, fmri_train.size(0))
            
            batch_fmri = fmri_train[start_idx:end_idx]
            batch_stim = stim_train[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_fmri, batch_stim)
            losses = loss_fn(outputs, batch_stim)
            
            # Backward pass
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += losses['total_loss'].item()
            epoch_complexity.append(outputs['complexity_score'])
            
            if outputs['uncertainty'] is not None:
                epoch_uncertainty.append(outputs['uncertainty'].mean().item())
        
        train_loss /= num_batches
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(fmri_test, stim_test)
            test_losses_dict = loss_fn(test_outputs, stim_test)
            test_loss = test_losses_dict['total_loss'].item()
        
        # Record metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        complexity_scores.append(np.mean(epoch_complexity))
        if epoch_uncertainty:
            uncertainty_scores.append(np.mean(epoch_uncertainty))
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'config': config_name
            }
            
            checkpoint_dir = f'../../checkpoints/unified'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, f'{checkpoint_dir}/unified_{config_name}_{dataset_name.lower()}_model.pt')
            
            print(f"\nEpoch {epoch+1:3d} Unified Summary:")
            print(f"  Total Loss:      {test_loss:.6f}")
            print(f"  Reconstruction:  {test_losses_dict['reconstruction_loss'].item():.6f}")
            print(f"  Uncertainty:     {test_losses_dict['uncertainty_loss'].item():.6f}")
            print(f"  Alignment:       {test_losses_dict['alignment_loss'].item():.6f}")
            print(f"  Complexity:      {complexity_scores[-1]:.3f}")
            if uncertainty_scores:
                print(f"  Uncertainty:     {uncertainty_scores[-1]:.6f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Time: {(time.time() - start_time)/60:.1f}m")
            print(f"  ✅ New best unified model saved!")
            print("-" * 80)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
                break
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1:3d} Progress:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Test Loss:  {test_loss:.6f}")
            print(f"  Complexity: {complexity_scores[-1]:.3f}")
            print(f"  Patience: {patience_counter}/{patience}")
    
    training_time = time.time() - start_time
    
    print(f"\n🎉 Unified training completed in {training_time/60:.1f} minutes!")
    print(f"🏆 Best test loss: {best_loss:.6f}")
    print(f"📊 Total epochs: {len(train_losses)}")
    
    # Get final statistics
    complexity_stats = model.get_complexity_stats()
    uncertainty_stats = model.get_uncertainty_stats()
    
    print(f"\n📊 UNIFIED MODEL STATISTICS:")
    print(f"  Mean Complexity: {complexity_stats['mean_complexity']:.3f}")
    print(f"  Complexity Trend: {complexity_stats['complexity_trend']:.6f}")
    if uncertainty_stats['mean_uncertainty'] > 0:
        print(f"  Mean Uncertainty: {uncertainty_stats['mean_uncertainty']:.6f}")
        print(f"  Uncertainty Trend: {uncertainty_stats['uncertainty_trend']:.6f}")
    
    return {
        'best_loss': best_loss,
        'training_time': training_time,
        'epochs': len(train_losses),
        'complexity_stats': complexity_stats,
        'uncertainty_stats': uncertainty_stats,
        'train_losses': train_losses,
        'test_losses': test_losses
    }

def main():
    """Main training function."""
    print("🔬 UNIFIED CORTEXFLOW TRAINING")
    print("=" * 80)
    print(f"Experiment start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Using device: {device}")
    
    # Test different configurations
    configs = ['simple', 'balanced', 'advanced']
    datasets = [
        ('miyawaki_structured_28x28.mat', 'Miyawaki'),
        ('digit69_28x28.mat', 'Vangerven'),
        ('mindbigdata.mat', 'MindBigData'),
        ('crell.mat', 'Crell')
    ]

    results = {}

    for dataset_file, dataset_name in datasets:
        # Load data
        data_path = f'../../data/processed/{dataset_file}'
        data_dict = load_data(data_path)
        
        if data_dict is None:
            print(f"❌ Skipping {dataset_name} due to data loading error")
            continue
        
        data = preprocess_data(data_dict)
        input_dim = data['fmri_train'].shape[1]
        
        for config_name in configs:
            print(f"\n{'='*80}")
            print(f"🧪 TESTING CONFIG: {config_name.upper()} ON {dataset_name.upper()}")
            print(f"{'='*80}")
            
            # Create model
            model, loss_fn = create_unified_model(
                input_dim=input_dim,
                config=config_name
            )
            
            # Train model
            result = train_unified_model(
                model, loss_fn, data, config_name, dataset_name, device
            )
            
            results[f"{config_name}_{dataset_name}"] = result
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 UNIFIED CORTEXFLOW TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    print(f"\n📊 FINAL UNIFIED RESULTS:")
    print("-" * 80)
    
    for key, result in results.items():
        config, dataset = key.split('_', 1)
        print(f"✅ {config.capitalize()} {dataset}: Loss {result['best_loss']:.6f} | "
              f"Time {result['training_time']/60:.1f}m | "
              f"Epochs {result['epochs']} | "
              f"Complexity {result['complexity_stats']['mean_complexity']:.3f}")
    
    print(f"\n🎉 All unified experiments completed successfully!")
    print(f"📁 Check 'checkpoints/unified/' for saved models")

if __name__ == "__main__":
    main()
