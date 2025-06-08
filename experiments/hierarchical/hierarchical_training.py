#!/usr/bin/env python3
"""
HierarchicalCortexFlow Training: Advanced Multi-Scale Architecture
Integration of hierarchical.py with clean training pipeline
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
import scipy.io
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

# Add src to path
sys.path.append('../../src')
sys.path.append('../../src/models')

# Import hierarchical architecture
from hierarchical import HierarchicalCortexFlow, HierarchicalConfig, create_hierarchical_model

# Import base components from original training
try:
    from data.data_loader import FMRIDataLoader
    from utils.visualization import plot_reconstruction_comparison
except ImportError:
    # Fallback implementations
    class FMRIDataLoader:
        def __init__(self, data_path):
            self.data_path = data_path
            self.data = scipy.io.loadmat(data_path)
            
        def get_fmri(self, split):
            key = 'fmriTrn' if split == 'train' else 'fmriTest'
            return self.data.get(key, np.random.randn(100, 1000))
                
        def create_dataloader(self, split, batch_size, shuffle):
            fmri_data = self.get_fmri(split)
            stim_key = 'stimTrn' if split == 'train' else 'stimTest'
            images = self.data.get(stim_key, np.random.randn(len(fmri_data), 784))
            labels = np.random.randint(0, 10, len(fmri_data))
            
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(fmri_data),
                torch.FloatTensor(images),
                torch.LongTensor(labels)
            )
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def plot_reconstruction_comparison(original, reconstructed, save_path, title):
        fig, axes = plt.subplots(2, min(8, len(original)), figsize=(15, 4))
        for i in range(min(8, len(original))):
            axes[0, i].imshow(original[i], cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(reconstructed[i].detach().numpy(), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


@dataclass
class HierarchicalTrainingResult:
    """Enhanced training result for hierarchical model."""
    best_test_loss: float
    training_time_minutes: float
    total_epochs: int
    total_params: int
    input_dim: int
    final_train_loss: float
    convergence_epoch: int
    # Hierarchical-specific metrics
    best_main_loss: float
    best_progressive_loss: float
    best_edge_loss: float
    best_diversity_loss: float


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HierarchicalExperimentLogger:
    """Enhanced logger for hierarchical training with detailed loss tracking."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.start_time = time.time()
        
    def log_start(self, input_dim: int, total_params: int, config: HierarchicalConfig):
        """Log experiment start with hierarchical details."""
        print(f"\n{'='*80}")
        print(f"🏗️ HIERARCHICAL CORTEXFLOW: {self.dataset_name.upper()} DATASET")
        print(f"{'='*80}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"✅ Dataset loaded: {input_dim} input dimensions")
        print(f"🔧 Total parameters: {total_params:,}")
        print(f"⏱️  Temporal scales: {config.TEMPORAL_SCALES}")
        print(f"🏗️  Hidden dimension: {config.HIDDEN_DIM}")
        print(f"📈 Pyramid levels: {config.NUM_PYRAMID_LEVELS}")
        print(f"🔗 Skip connections: {config.USE_SKIP_CONNECTIONS}")
        print(f"🤝 Feature fusion: {config.FEATURE_FUSION_TYPE}")
        
    def log_epoch(self, epoch: int, loss_dict: Dict[str, float], 
                  lr: float, is_best: bool, patience: int):
        """Log epoch results with detailed loss breakdown."""
        if epoch % 5 == 0 or epoch < 10:
            elapsed = time.time() - self.start_time
            print(f"\nEpoch {epoch+1:3d} Summary:")
            print(f"  Total Loss:      {loss_dict.get('total_loss', 0):.6f}")
            print(f"  Main Loss:       {loss_dict.get('main_loss', 0):.6f}")
            print(f"  Progressive:     {loss_dict.get('progressive_loss', 0):.6f}")
            print(f"  Edge Loss:       {loss_dict.get('edge_loss', 0):.6f}")
            print(f"  Diversity Loss:  {loss_dict.get('diversity_loss', 0):.6f}")
            print(f"  LR: {lr:.2e}")
            print(f"  Time: {elapsed/60:.1f}m")
            
            if is_best:
                print(f"  ✅ New best model saved! (Loss: {loss_dict.get('total_loss', 0):.6f})")
            else:
                print(f"  ⏳ Patience: {patience}/20")
            print("-" * 80)
    
    def log_completion(self, best_loss: float, total_epochs: int):
        """Log training completion."""
        total_time = time.time() - self.start_time
        print(f"\n🎉 Hierarchical training completed in {total_time/60:.1f} minutes!")
        print(f"🏆 Best test loss: {best_loss:.6f}")
        print(f"📊 Total epochs: {total_epochs}")


def create_data_loaders(dataset_path: str, config: HierarchicalConfig) -> Tuple[Any, Any, int]:
    """Create data loaders for training and testing."""
    data_loader = FMRIDataLoader(data_path=dataset_path)
    train_loader = data_loader.create_dataloader(
        split='train', batch_size=config.BATCH_SIZE, shuffle=True
    )
    test_loader = data_loader.create_dataloader(
        split='test', batch_size=config.BATCH_SIZE, shuffle=False
    )
    input_dim = data_loader.get_fmri('train').shape[1]
    return train_loader, test_loader, input_dim


def save_hierarchical_checkpoint(model: HierarchicalCortexFlow, optimizer, epoch: int,
                                best_loss: float, dataset_name: str, **kwargs):
    """Save hierarchical model checkpoint."""
    os.makedirs('../../checkpoints', exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'dataset': dataset_name,
        'model_type': 'hierarchical',
        'config': model.config,
        **kwargs
    }
    torch.save(checkpoint, f'../../checkpoints/hierarchical/hierarchical_{dataset_name.lower()}_model.pt')


def generate_hierarchical_visualizations(model: HierarchicalCortexFlow, test_loader, 
                                        device: torch.device, dataset_name: str):
    """Generate enhanced visualizations for hierarchical model."""
    model.eval()
    with torch.no_grad():
        input_data, images, _ = next(iter(test_loader))
        input_data = input_data[:8].to(device)
        images = images[:8].to(device)

        # Get hierarchical outputs
        outputs = model(input_data)
        main_reconstruction = outputs['reconstruction']
        progressive_outputs = outputs['progressive_outputs']

        if images.dim() == 2:
            images = images.view(-1, 1, 28, 28)

        images_viz = images.cpu().squeeze(1) if images.dim() == 4 else images.cpu()
        main_viz = main_reconstruction.cpu().squeeze(1) if main_reconstruction.dim() == 4 else main_reconstruction.cpu()

        # Main reconstruction comparison
        plot_reconstruction_comparison(
            images_viz,
            main_viz,
            save_path=f'../../results/hierarchical/hierarchical_{dataset_name.lower()}_reconstructions.png',
            title=f'HierarchicalCortexFlow: {dataset_name} Main Reconstructions'
        )
        
        # Progressive reconstruction visualization
        plot_progressive_reconstructions(
            images_viz, progressive_outputs, 
            save_path=f'../../results/hierarchical/hierarchical_{dataset_name.lower()}_progressive.png',
            title=f'HierarchicalCortexFlow: {dataset_name} Progressive Reconstructions'
        )


def plot_progressive_reconstructions(original, progressive_outputs, save_path, title):
    """Plot progressive reconstruction levels."""
    num_levels = len(progressive_outputs)
    num_samples = min(4, len(original))

    fig, axes = plt.subplots(num_samples, num_levels + 1, figsize=(20, 8))

    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(original[i], cmap='gray')
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')

        # Progressive levels
        for level in range(num_levels):
            level_output = progressive_outputs[f'level_{level}'][i].cpu().squeeze()
            axes[i, level + 1].imshow(level_output, cmap='gray')
            if i == 0:
                axes[i, level + 1].set_title(f'Level {level}')
            axes[i, level + 1].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_hierarchical_dataset(dataset_name: str, dataset_path: str, device: torch.device,
                              config: HierarchicalConfig = None) -> HierarchicalTrainingResult:
    """Train hierarchical model on a single dataset."""
    if config is None:
        config = HierarchicalConfig()

    set_seeds(config.SEED)

    # Initialize components
    logger = HierarchicalExperimentLogger(dataset_name)
    train_loader, test_loader, input_dim = create_data_loaders(dataset_path, config)

    # Create hierarchical model
    model = create_hierarchical_model(input_dim, config)
    model = model.to(device)

    # Enhanced optimizer with different learning rates for different components
    param_groups = [
        {'params': model.temporal_encoders.parameters(), 'lr': config.LEARNING_RATE},
        {'params': model.feature_pyramid.parameters(), 'lr': config.LEARNING_RATE * 0.8},
        {'params': model.progressive_decoder.parameters(), 'lr': config.LEARNING_RATE * 1.2}
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.log_start(input_dim, total_params, config)

    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    best_losses = {
        'main_loss': float('inf'),
        'progressive_loss': float('inf'),
        'edge_loss': float('inf'),
        'diversity_loss': float('inf')
    }

    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_losses = {'total_loss': 0.0, 'main_loss': 0.0, 'progressive_loss': 0.0,
                       'edge_loss': 0.0, 'diversity_loss': 0.0}

        for input_data, images, _ in train_loader:
            input_data, images = input_data.to(device), images.to(device)

            optimizer.zero_grad()
            loss_dict = model.compute_loss(input_data, images)

            total_loss = loss_dict['total_loss']
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
            optimizer.step()

            # Track losses
            for key in train_losses.keys():
                if key in loss_dict:
                    train_losses[key] += loss_dict[key].item()

        # Average training losses
        for key in train_losses.keys():
            train_losses[key] /= len(train_loader)

        # Validation phase
        model.eval()
        test_losses = {'total_loss': 0.0, 'main_loss': 0.0, 'progressive_loss': 0.0,
                      'edge_loss': 0.0, 'diversity_loss': 0.0}

        with torch.no_grad():
            for input_data, images, _ in test_loader:
                input_data, images = input_data.to(device), images.to(device)
                loss_dict = model.compute_loss(input_data, images)

                for key in test_losses.keys():
                    if key in loss_dict:
                        test_losses[key] += loss_dict[key].item()

        # Average test losses
        for key in test_losses.keys():
            test_losses[key] /= len(test_loader)

        scheduler.step(test_losses['total_loss'])

        # Check for improvement
        is_best = test_losses['total_loss'] < best_loss
        if is_best:
            best_loss = test_losses['total_loss']
            patience_counter = 0

            # Update best individual losses
            for key in best_losses.keys():
                if key in test_losses:
                    best_losses[key] = test_losses[key]

            save_hierarchical_checkpoint(
                model, optimizer, epoch, best_loss, dataset_name,
                train_losses=train_losses, test_losses=test_losses,
                input_dim=input_dim, total_params=total_params
            )
        else:
            patience_counter += 1

        logger.log_epoch(
            epoch, test_losses,
            optimizer.param_groups[0]['lr'],
            is_best, patience_counter
        )

        if patience_counter >= config.PATIENCE:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            break

    logger.log_completion(best_loss, epoch + 1)
    generate_hierarchical_visualizations(model, test_loader, device, dataset_name)

    return HierarchicalTrainingResult(
        best_test_loss=best_loss,
        training_time_minutes=(time.time() - logger.start_time) / 60,
        total_epochs=epoch + 1,
        total_params=total_params,
        input_dim=input_dim,
        final_train_loss=train_losses['total_loss'],
        convergence_epoch=epoch + 1 - patience_counter,
        best_main_loss=best_losses['main_loss'],
        best_progressive_loss=best_losses['progressive_loss'],
        best_edge_loss=best_losses['edge_loss'],
        best_diversity_loss=best_losses['diversity_loss']
    )


def main():
    """Main hierarchical training function."""
    print("🏗️ HIERARCHICAL CORTEXFLOW TRAINING")
    print("=" * 80)
    print(f"Experiment start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seeds()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎮 Using device: {device}")

    os.makedirs('../../results', exist_ok=True)

    # Enhanced hierarchical configuration
    config = HierarchicalConfig(
        TEMPORAL_SCALES=[1, 2, 4, 8],
        HIDDEN_DIM=512,
        NUM_PYRAMID_LEVELS=4,
        PROGRESSIVE_WEIGHTS=[0.1, 0.2, 0.3, 0.4],
        USE_SKIP_CONNECTIONS=True,
        FEATURE_FUSION_TYPE="attention",
        LEARNING_RATE=1e-3,
        NUM_EPOCHS=200,
        PATIENCE=20
    )

    datasets = [
        ('Miyawaki', '../../data/processed/miyawaki_structured_28x28.mat'),
        ('Vangerven', '../../data/processed/digit69_28x28.mat'),
    ]

    results = {}
    total_start_time = time.time()

    for dataset_name, dataset_path in datasets:
        try:
            result = train_hierarchical_dataset(dataset_name, dataset_path, device, config)
            results[dataset_name] = result
        except Exception as e:
            print(f"❌ Error training {dataset_name}: {e}")
            results[dataset_name] = None

    # Print final summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*80}")
    print("🎉 HIERARCHICAL TRAINING COMPLETED!")
    print(f"Total experiment time: {total_time/60:.1f} minutes")

    print(f"\n📊 FINAL HIERARCHICAL RESULTS:")
    print("-" * 80)
    for dataset, result in results.items():
        if result:
            print(f"✅ {dataset:12s}: Total {result.best_test_loss:.6f} | "
                  f"Main {result.best_main_loss:.6f} | "
                  f"Time {result.training_time_minutes:.1f}m | "
                  f"Epochs {result.total_epochs:3d}")
        else:
            print(f"❌ {dataset:12s}: FAILED")


if __name__ == "__main__":
    main()
