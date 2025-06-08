"""
Monte Carlo Simple CortexFlow Training Script

This script trains the Monte Carlo Simple CortexFlow model that combines:
- Simple and efficient encoder-decoder architecture
- Monte Carlo Dropout for uncertainty estimation
- Uncertainty-aware loss function
- Comprehensive evaluation with uncertainty metrics

Features:
- Uncertainty-aware training with reconstruction + uncertainty loss
- Monte Carlo sampling during evaluation
- Uncertainty visualization and analysis
- Compatible with existing CortexFlow datasets
- Efficient training with early stopping

Author: CortexFlow Team
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from typing import Dict, List, Tuple

from src.models.mc_simple_cortexflow import create_mc_simple_cortexflow, MCSimpleCortexFlow


class MCSimpleConfig:
    """Configuration for Monte Carlo Simple CortexFlow training."""
    
    # Model parameters
    HIDDEN_DIM = 512
    DROPOUT_RATE = 0.15
    MC_SAMPLES = 10
    UNCERTAINTY_WEIGHT = 0.1
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP_NORM = 1.0
    
    # Early stopping
    PATIENCE = 20
    MIN_DELTA = 1e-6
    
    # Evaluation
    MC_EVAL_SAMPLES = 20  # More samples for evaluation
    
    # Visualization
    NUM_SAMPLES_TO_PLOT = 8
    SAVE_PLOTS = True


def load_dataset(dataset_path: str, device: torch.device) -> Tuple[DataLoader, DataLoader, Dict]:
    """Load and prepare dataset for training."""
    print(f"📁 Loading data from: {os.path.basename(dataset_path)}")
    
    # Load .mat file
    data = scipy.io.loadmat(dataset_path)
    
    # Extract data
    fmri_train = torch.FloatTensor(data['fmriTrn']).to(device)
    stim_train = torch.FloatTensor(data['stimTrn']).to(device)
    fmri_test = torch.FloatTensor(data['fmriTest']).to(device)
    stim_test = torch.FloatTensor(data['stimTest']).to(device)
    
    # Normalize stimuli to [0, 1] if needed
    if stim_train.max() > 1.0:
        stim_train = stim_train / 255.0
        stim_test = stim_test / 255.0
    
    # Create labels (dummy for compatibility)
    labels_train = torch.zeros(fmri_train.size(0), dtype=torch.long).to(device)
    labels_test = torch.zeros(fmri_test.size(0), dtype=torch.long).to(device)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(fmri_train, stim_train, labels_train)
    test_dataset = TensorDataset(fmri_test, stim_test, labels_test)
    
    train_loader = DataLoader(train_dataset, batch_size=MCSimpleConfig.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=MCSimpleConfig.BATCH_SIZE, shuffle=False)
    
    # Print dataset info
    print(f"  ✅ fmriTrn: {fmri_train.shape}")
    print(f"  ✅ stimTrn: {stim_train.shape}")
    print(f"  ✅ fmriTest: {fmri_test.shape}")
    print(f"  ✅ stimTest: {stim_test.shape}")
    
    dataset_info = {
        'input_dim': fmri_train.size(1),
        'output_dim': stim_train.size(1),
        'train_samples': fmri_train.size(0),
        'test_samples': fmri_test.size(0),
        'fmri_train': fmri_train,
        'stim_train': stim_train,
        'fmri_test': fmri_test,
        'stim_test': stim_test
    }
    
    return train_loader, test_loader, dataset_info


def train_epoch(model: MCSimpleCortexFlow, train_loader: DataLoader, 
                optimizer: optim.Optimizer, device: torch.device) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_uncertainty_loss = 0.0
    num_batches = 0
    
    for fmri_batch, stim_batch, _ in train_loader:
        fmri_batch = fmri_batch.to(device)
        stim_batch = stim_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        loss_dict = model.compute_loss(fmri_batch, stim_batch)
        
        # Backward pass
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MCSimpleConfig.GRAD_CLIP_NORM)
        optimizer.step()
        
        # Track losses
        total_loss += loss_dict['total_loss'].item()
        total_recon_loss += loss_dict['reconstruction_loss'].item()
        total_uncertainty_loss += loss_dict['uncertainty_loss'].item()
        num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches,
        'uncertainty_loss': total_uncertainty_loss / num_batches
    }


def evaluate_model(model: MCSimpleCortexFlow, test_loader: DataLoader, 
                  device: torch.device, use_mc_sampling: bool = True) -> Dict[str, float]:
    """Evaluate model with uncertainty estimation."""
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_uncertainty_loss = 0.0
    epistemic_uncertainties = []
    aleatoric_uncertainties = []
    total_uncertainties = []
    num_batches = 0
    
    with torch.no_grad():
        for fmri_batch, stim_batch, _ in test_loader:
            fmri_batch = fmri_batch.to(device)
            stim_batch = stim_batch.to(device)
            
            if use_mc_sampling:
                # Use Monte Carlo sampling for uncertainty estimation
                # Temporarily set mc_samples for evaluation
                original_mc_samples = model.mc_samples
                model.mc_samples = MCSimpleConfig.MC_EVAL_SAMPLES
                
                mc_outputs = model.forward_with_mc_sampling(fmri_batch)
                
                # Calculate loss using mean reconstruction
                if stim_batch.dim() == 2:
                    stim_batch = stim_batch.view(-1, 1, 28, 28)
                elif stim_batch.dim() == 3:
                    stim_batch = stim_batch.unsqueeze(1)
                
                recon_loss = nn.functional.mse_loss(mc_outputs['reconstruction'], stim_batch)
                
                # Collect uncertainty statistics
                epistemic_uncertainties.append(mc_outputs['epistemic_uncertainty'].mean().item())
                aleatoric_uncertainties.append(mc_outputs['aleatoric_uncertainty'].mean().item())
                total_uncertainties.append(mc_outputs['total_uncertainty'].mean().item())
                
                total_recon_loss += recon_loss.item()
                total_loss += recon_loss.item()  # No uncertainty loss in evaluation
                
                # Restore original mc_samples
                model.mc_samples = original_mc_samples
                
            else:
                # Standard forward pass
                loss_dict = model.compute_loss(fmri_batch, stim_batch)
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['reconstruction_loss'].item()
                total_uncertainty_loss += loss_dict['uncertainty_loss'].item()
            
            num_batches += 1
    
    result = {
        'total_loss': total_loss / num_batches,
        'reconstruction_loss': total_recon_loss / num_batches,
        'uncertainty_loss': total_uncertainty_loss / num_batches
    }
    
    if use_mc_sampling and epistemic_uncertainties:
        result.update({
            'epistemic_uncertainty': np.mean(epistemic_uncertainties),
            'aleatoric_uncertainty': np.mean(aleatoric_uncertainties),
            'total_uncertainty': np.mean(total_uncertainties),
            'epistemic_std': np.std(epistemic_uncertainties),
            'aleatoric_std': np.std(aleatoric_uncertainties),
            'total_std': np.std(total_uncertainties)
        })
    
    return result


def create_uncertainty_visualization(model: MCSimpleCortexFlow, dataset_info: Dict, 
                                   save_path: str, device: torch.device):
    """Create visualization showing reconstructions with uncertainty estimates."""
    model.eval()
    
    # Use test data for visualization
    fmri_test = dataset_info['fmri_test'][:MCSimpleConfig.NUM_SAMPLES_TO_PLOT]
    stim_test = dataset_info['stim_test'][:MCSimpleConfig.NUM_SAMPLES_TO_PLOT]
    
    with torch.no_grad():
        # Get Monte Carlo outputs
        original_mc_samples = model.mc_samples
        model.mc_samples = MCSimpleConfig.MC_EVAL_SAMPLES
        
        mc_outputs = model.forward_with_mc_sampling(fmri_test)
        
        model.mc_samples = original_mc_samples
    
    # Prepare data for plotting
    targets = stim_test.cpu().numpy().reshape(-1, 28, 28)
    reconstructions = mc_outputs['reconstruction'].cpu().numpy().reshape(-1, 28, 28)
    reconstruction_std = mc_outputs['reconstruction_std'].cpu().numpy().reshape(-1, 28, 28)
    epistemic_unc = mc_outputs['epistemic_uncertainty'].cpu().numpy().flatten()
    aleatoric_unc = mc_outputs['aleatoric_uncertainty'].cpu().numpy().flatten()
    total_unc = mc_outputs['total_uncertainty'].cpu().numpy().flatten()
    
    # Create visualization
    fig, axes = plt.subplots(4, MCSimpleConfig.NUM_SAMPLES_TO_PLOT, 
                            figsize=(2*MCSimpleConfig.NUM_SAMPLES_TO_PLOT, 8))
    
    for i in range(MCSimpleConfig.NUM_SAMPLES_TO_PLOT):
        # Original image
        axes[0, i].imshow(targets[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Target {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Reconstruction\nTotal Unc: {total_unc[i]:.3f}')
        axes[1, i].axis('off')
        
        # Reconstruction uncertainty (std)
        axes[2, i].imshow(reconstruction_std[i], cmap='hot', vmin=0, vmax=reconstruction_std[i].max())
        axes[2, i].set_title(f'Pixel Uncertainty\nEpistemic: {epistemic_unc[i]:.3f}')
        axes[2, i].axis('off')
        
        # Error map
        error_map = np.abs(targets[i] - reconstructions[i])
        axes[3, i].imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
        axes[3, i].set_title(f'Error Map\nAleatoric: {aleatoric_unc[i]:.3f}')
        axes[3, i].axis('off')
    
    # Add row labels
    row_labels = ['Target', 'Reconstruction', 'Pixel Uncertainty', 'Error Map']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, rotation=90, size='large')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Uncertainty visualization saved: {save_path}")


def train_mc_simple_cortexflow(dataset_path: str, dataset_name: str, 
                              save_dir: str, device: torch.device) -> Dict:
    """Train Monte Carlo Simple CortexFlow model."""
    
    print(f"\n{'='*80}")
    print(f"🎲 MONTE CARLO SIMPLE CORTEXFLOW: {dataset_name.upper()} DATASET")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load dataset
    train_loader, test_loader, dataset_info = load_dataset(dataset_path, device)
    print(f"✅ Dataset loaded: {dataset_info['input_dim']} input dimensions")
    
    # Create model
    model = create_mc_simple_cortexflow(
        input_dim=dataset_info['input_dim'],
        hidden_dim=MCSimpleConfig.HIDDEN_DIM,
        dropout_rate=MCSimpleConfig.DROPOUT_RATE,
        mc_samples=MCSimpleConfig.MC_SAMPLES,
        uncertainty_weight=MCSimpleConfig.UNCERTAINTY_WEIGHT
    ).to(device)
    
    print(f"🔧 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=MCSimpleConfig.LEARNING_RATE, 
                          weight_decay=MCSimpleConfig.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training tracking
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []
    uncertainty_stats = []
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(MCSimpleConfig.NUM_EPOCHS):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_metrics['total_loss'])
        
        # Evaluate
        test_metrics = evaluate_model(model, test_loader, device, use_mc_sampling=True)
        test_losses.append(test_metrics['total_loss'])
        
        # Learning rate scheduling
        scheduler.step(test_metrics['total_loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track uncertainty statistics
        if 'total_uncertainty' in test_metrics:
            uncertainty_stats.append({
                'epoch': epoch + 1,
                'epistemic': test_metrics['epistemic_uncertainty'],
                'aleatoric': test_metrics['aleatoric_uncertainty'],
                'total': test_metrics['total_uncertainty']
            })
        
        # Early stopping check
        if test_metrics['total_loss'] < best_loss - MCSimpleConfig.MIN_DELTA:
            best_loss = test_metrics['total_loss']
            patience_counter = 0
            
            # Save best model
            checkpoint_path = os.path.join(save_dir, f'mc_simple_{dataset_name}_model.pt')
            # Create config dict without non-serializable objects
            config_dict = {
                'HIDDEN_DIM': MCSimpleConfig.HIDDEN_DIM,
                'DROPOUT_RATE': MCSimpleConfig.DROPOUT_RATE,
                'MC_SAMPLES': MCSimpleConfig.MC_SAMPLES,
                'UNCERTAINTY_WEIGHT': MCSimpleConfig.UNCERTAINTY_WEIGHT,
                'BATCH_SIZE': MCSimpleConfig.BATCH_SIZE,
                'LEARNING_RATE': MCSimpleConfig.LEARNING_RATE,
                'NUM_EPOCHS': MCSimpleConfig.NUM_EPOCHS,
                'WEIGHT_DECAY': MCSimpleConfig.WEIGHT_DECAY,
                'GRAD_CLIP_NORM': MCSimpleConfig.GRAD_CLIP_NORM,
                'PATIENCE': MCSimpleConfig.PATIENCE,
                'MIN_DELTA': MCSimpleConfig.MIN_DELTA,
                'MC_EVAL_SAMPLES': MCSimpleConfig.MC_EVAL_SAMPLES,
                'NUM_SAMPLES_TO_PLOT': MCSimpleConfig.NUM_SAMPLES_TO_PLOT,
                'SAVE_PLOTS': MCSimpleConfig.SAVE_PLOTS
            }

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'config': config_dict,
                'model_info': model.get_model_info(),
                'uncertainty_stats': model.get_uncertainty_stats()
            }, checkpoint_path)
            
            print(f"\nEpoch {epoch + 1:3d} Summary:")
            print(f"  Total Loss:      {test_metrics['total_loss']:.6f}")
            print(f"  Reconstruction:  {test_metrics['reconstruction_loss']:.6f}")
            if 'total_uncertainty' in test_metrics:
                print(f"  Epistemic Unc:   {test_metrics['epistemic_uncertainty']:.6f}")
                print(f"  Aleatoric Unc:   {test_metrics['aleatoric_uncertainty']:.6f}")
                print(f"  Total Unc:       {test_metrics['total_uncertainty']:.6f}")
            print(f"  LR: {current_lr:.2e}")
            print(f"  Time: {(time.time() - start_time)/60:.1f}m")
            print(f"  ✅ New best model saved!")
            print("-" * 80)
            
        else:
            patience_counter += 1
            if epoch % 5 == 0:  # Print every 5 epochs
                print(f"Epoch {epoch + 1:3d} Progress:")
                print(f"  Train Loss: {train_metrics['total_loss']:.6f}")
                print(f"  Test Loss:  {test_metrics['total_loss']:.6f}")
                if 'total_uncertainty' in test_metrics:
                    print(f"  Total Unc:  {test_metrics['total_uncertainty']:.6f}")
                print(f"  Patience: {patience_counter}/{MCSimpleConfig.PATIENCE}")
        
        # Early stopping
        if patience_counter >= MCSimpleConfig.PATIENCE:
            print(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
            break
    
    training_time = (time.time() - start_time) / 60
    print(f"\n🎉 Training completed in {training_time:.1f} minutes!")
    print(f"🏆 Best test loss: {best_loss:.6f}")
    print(f"📊 Total epochs: {epoch + 1}")
    
    # Create uncertainty visualization
    if MCSimpleConfig.SAVE_PLOTS:
        viz_path = os.path.join(save_dir, f'mc_simple_{dataset_name}_uncertainty.png')
        create_uncertainty_visualization(model, dataset_info, viz_path, device)
    
    return {
        'best_loss': best_loss,
        'total_epochs': epoch + 1,
        'training_time': training_time,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'uncertainty_stats': uncertainty_stats,
        'model_info': model.get_model_info(),
        'final_uncertainty_stats': model.get_uncertainty_stats()
    }


def main():
    """Main training function."""
    print("🎲 MONTE CARLO SIMPLE CORTEXFLOW TRAINING")
    print("=" * 80)
    print(f"Experiment start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Using device: {device}")

    # Create save directory
    save_dir = "../../results/mc_simple"
    os.makedirs(save_dir, exist_ok=True)

    # Dataset configurations
    datasets = [
        {
            'path': '../../data/processed/miyawaki_structured_28x28.mat',
            'name': 'miyawaki'
        },
        {
            'path': '../../data/processed/digit69_28x28.mat',
            'name': 'vangerven'
        }
    ]

    results = {}
    experiment_start_time = time.time()

    # Train on each dataset
    for dataset_config in datasets:
        try:
            result = train_mc_simple_cortexflow(
                dataset_path=dataset_config['path'],
                dataset_name=dataset_config['name'],
                save_dir=save_dir,
                device=device
            )
            results[dataset_config['name']] = result

        except Exception as e:
            print(f"❌ Error training {dataset_config['name']}: {e}")
            results[dataset_config['name']] = {'error': str(e)}

    # Print final summary
    print(f"\n{'='*80}")
    print("🎉 TRAINING COMPLETED!")
    print(f"Total experiment time: {(time.time() - experiment_start_time)/60:.1f} minutes")

    print(f"\n📊 FINAL RESULTS:")
    print("-" * 80)
    for dataset_name, result in results.items():
        if 'error' in result:
            print(f"❌ {dataset_name.capitalize():12s}: FAILED")
        else:
            print(f"✅ {dataset_name.capitalize():12s}: {result['best_loss']:.6f} "
                  f"({result['total_epochs']} epochs, {result['training_time']:.1f}m)")

    print(f"\n✅ Monte Carlo Simple CortexFlow experiment completed successfully!")


if __name__ == "__main__":
    main()
