#!/usr/bin/env python3
"""
🚀 TRAIN REMAINING DATASETS (MINDBIGDATA & CRELL)
================================================================================
Complete training for MindBigData and Crell datasets with Monte Carlo Simple CortexFlow
================================================================================
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from pathlib import Path

# Import Monte Carlo Simple CortexFlow
from src.models.mc_simple_cortexflow import create_mc_simple_cortexflow

class TrainingConfig:
    """Configuration for training remaining datasets."""
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    PATIENCE = 20
    WEIGHT_DECAY = 1e-5
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Datasets to train
    DATASETS = {
        'mindbigdata': {
            'file': 'data/processed/mindbigdata.mat',
            'input_dim': 3092,
            'description': 'EEG-based Neural Signals'
        },
        'crell': {
            'file': 'data/processed/crell.mat',
            'input_dim': 3092,
            'description': 'Advanced fMRI'
        }
    }

def load_dataset(dataset_name):
    """Load and preprocess dataset."""
    config = TrainingConfig.DATASETS[dataset_name]
    filepath = config['file']
    
    print(f"📁 Loading {dataset_name} from: {filepath}")
    
    try:
        data = scipy.io.loadmat(filepath)
        
        # Extract data
        fmri_train = torch.FloatTensor(data['fmriTrn'])
        stim_train = torch.FloatTensor(data['stimTrn'])
        fmri_test = torch.FloatTensor(data['fmriTest'])
        stim_test = torch.FloatTensor(data['stimTest'])
        
        # Normalize stimuli to [0, 1] if needed
        if stim_train.max() > 1.0:
            stim_train = stim_train / 255.0
            stim_test = stim_test / 255.0
            
        print(f"  ✅ fmriTrn: {fmri_train.shape}")
        print(f"  ✅ stimTrn: {stim_train.shape}")
        print(f"  ✅ fmriTest: {fmri_test.shape}")
        print(f"  ✅ stimTest: {stim_test.shape}")
        
        return {
            'train_fmri': fmri_train,
            'train_stim': stim_train,
            'test_fmri': fmri_test,
            'test_stim': stim_test,
            'input_dim': config['input_dim']
        }
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None

def train_model(model, dataset, dataset_name):
    """Train model on dataset."""
    device = TrainingConfig.DEVICE
    model = model.to(device)
    
    # Prepare data
    train_fmri = dataset['train_fmri'].to(device)
    train_stim = dataset['train_stim'].to(device)
    test_fmri = dataset['test_fmri'].to(device)
    test_stim = dataset['test_stim'].to(device)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_fmri, train_stim)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=TrainingConfig.BATCH_SIZE, 
        shuffle=True
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE, 
                          weight_decay=TrainingConfig.WEIGHT_DECAY)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []
    
    print(f"\n🚀 Training {dataset_name}")
    print(f"🔧 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    start_time = time.time()
    
    for epoch in range(TrainingConfig.NUM_EPOCHS):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        for batch_fmri, batch_stim in train_loader:
            optimizer.zero_grad()
            
            # Forward pass and loss computation
            loss_dict = model.compute_loss(batch_fmri, batch_stim)
            loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_fmri)
            test_loss = nn.MSELoss()(test_outputs['reconstruction'], test_stim.view(-1, 1, 28, 28))
        
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss.item())
        
        # Early stopping
        if test_loss.item() < best_loss:
            best_loss = test_loss.item()
            patience_counter = 0
            
            # Save best model
            results_dir = Path(f"results/mc_simple")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'model_info': model.get_model_info(),
                'uncertainty_stats': model.get_uncertainty_stats()
            }, results_dir / f"mc_simple_{dataset_name}_model.pt")
            
            print(f"Epoch {epoch+1:3d}: Train {avg_train_loss:.6f}, Test {test_loss.item():.6f} ✅")
        else:
            patience_counter += 1
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Train {avg_train_loss:.6f}, Test {test_loss.item():.6f} (patience: {patience_counter})")
        
        if patience_counter >= TrainingConfig.PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    return {
        'best_loss': best_loss,
        'final_epoch': epoch + 1,
        'training_time': training_time,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'parameters': sum(p.numel() for p in model.parameters())
    }

def create_uncertainty_visualization(model, dataset, dataset_name):
    """Create uncertainty visualization for the dataset."""
    device = TrainingConfig.DEVICE
    model = model.to(device)
    model.eval()
    
    test_fmri = dataset['test_fmri'].to(device)
    test_stim = dataset['test_stim'].to(device)
    
    # Get Monte Carlo predictions
    with torch.no_grad():
        mc_outputs = model.forward_with_mc_sampling(test_fmri)
        
        reconstruction = mc_outputs['reconstruction']
        epistemic_uncertainty = mc_outputs['epistemic_uncertainty']
        aleatoric_uncertainty = mc_outputs['aleatoric_uncertainty']
        total_uncertainty = mc_outputs['total_uncertainty']
    
    # Create visualization
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    for i in range(5):
        # Original
        axes[0, i].imshow(test_stim[i].view(28, 28).cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstruction[i].view(28, 28).cpu().numpy(), cmap='gray')
        axes[1, i].set_title(f'Reconstruction {i+1}')
        axes[1, i].axis('off')
        
        # Total Uncertainty
        uncertainty_img = total_uncertainty[i].view(28, 28).cpu().numpy()
        im = axes[2, i].imshow(uncertainty_img, cmap='hot')
        axes[2, i].set_title(f'Uncertainty {i+1}')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Monte Carlo Simple CortexFlow - {dataset_name.upper()}\nReconstructions and Uncertainty', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    results_dir = Path("results/mc_simple")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = results_dir / f"mc_simple_{dataset_name}_uncertainty.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"📊 Uncertainty visualization saved: {viz_path}")
    
    plt.close()

def main():
    """Main training function."""
    print("🚀 TRAINING REMAINING DATASETS WITH MONTE CARLO SIMPLE CORTEXFLOW")
    print("="*80)
    print(f"Training start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎮 Using device: {TrainingConfig.DEVICE}")
    
    # Store all results
    all_results = {}
    experiment_start = time.time()
    
    # Train each remaining dataset
    for dataset_name in TrainingConfig.DATASETS.keys():
        print(f"\n{'='*60}")
        print(f"🧪 TRAINING DATASET: {dataset_name.upper()}")
        print(f"📊 {TrainingConfig.DATASETS[dataset_name]['description']}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name)
            if dataset is None:
                all_results[dataset_name] = {'error': 'Failed to load dataset'}
                continue
            
            # Create model
            model = create_mc_simple_cortexflow(dataset['input_dim'])
            
            # Train model
            result = train_model(model, dataset, dataset_name)
            all_results[dataset_name] = result
            
            # Create uncertainty visualization
            create_uncertainty_visualization(model, dataset, dataset_name)
            
            print(f"✅ {dataset_name}: SUCCESS")
            print(f"   Best loss: {result['best_loss']:.6f}")
            print(f"   Training time: {result['training_time']:.1f}s")
            
        except Exception as e:
            print(f"❌ Error training {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}
    
    # Summary
    total_time = time.time() - experiment_start
    
    print(f"\n🎉 REMAINING DATASET TRAINING COMPLETED!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    print(f"\n📊 RESULTS SUMMARY:")
    print("-" * 80)
    for dataset_name, result in all_results.items():
        if 'error' in result:
            print(f"❌ {dataset_name:12} : FAILED - {result['error']}")
        else:
            print(f"✅ {dataset_name:12} : Loss {result['best_loss']:.6f}, "
                  f"Time {result['training_time']:.1f}s, "
                  f"Params {result['parameters']:,}")
    
    # Success rate
    successful = sum(1 for r in all_results.values() if 'error' not in r)
    total = len(all_results)
    print(f"\n📈 SUCCESS RATE: {successful}/{total} ({100*successful/total:.1f}%)")
    
    if successful == total:
        print("🎉 ALL REMAINING DATASETS TRAINED SUCCESSFULLY!")
    else:
        print("⚠️  Some datasets failed training")

if __name__ == "__main__":
    main()
