#!/usr/bin/env python3
"""
Enhanced HierarchicalCortexFlow Training: Monte Carlo + Feature Alignment
Advanced training with uncertainty estimation and feature consistency
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
import scipy.io
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

# Add src to path
sys.path.append('../../src')
sys.path.append('../../src/models')

# Import hierarchical architecture
from hierarchical import HierarchicalCortexFlow, HierarchicalConfig, create_hierarchical_model

# Import base components
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
class EnhancedConfig(HierarchicalConfig):
    """Enhanced configuration with Monte Carlo and Alignment settings."""
    # Monte Carlo Dropout settings
    MC_DROPOUT_RATE: float = 0.15
    MC_SAMPLES: int = 10
    ENABLE_MC_DROPOUT: bool = True
    
    # Feature Alignment settings
    ENABLE_FEATURE_ALIGNMENT: bool = True
    ALIGNMENT_WEIGHT: float = 0.1
    ALIGNMENT_TEMPERATURE: float = 0.1
    
    # Enhanced training settings
    UNCERTAINTY_WEIGHT: float = 0.05
    CONSISTENCY_WEIGHT: float = 0.08


class MonteCarloDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(self, dropout_rate: float = 0.15):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply dropout even during inference for MC sampling."""
        if training or self.training:
            return self.dropout(x)
        else:
            # Force dropout during MC sampling
            return F.dropout(x, p=self.dropout_rate, training=True)


class FeatureAlignmentModule(nn.Module):
    """Feature alignment module for consistency across scales."""
    
    def __init__(self, feature_dim: int, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """Compute alignment loss between features."""
        if len(features_list) < 2:
            return torch.tensor(0.0, device=features_list[0].device)
        
        # Project features to common space
        projected_features = [self.projection(feat) for feat in features_list]
        
        alignment_loss = 0.0
        num_pairs = 0
        
        for i in range(len(projected_features)):
            for j in range(i + 1, len(projected_features)):
                # Normalize features
                feat_i = F.normalize(projected_features[i], dim=1)
                feat_j = F.normalize(projected_features[j], dim=1)
                
                # Compute cosine similarity
                similarity = torch.mm(feat_i, feat_j.t()) / self.temperature
                
                # Alignment loss (encourage high similarity for same samples)
                batch_size = feat_i.size(0)
                targets = torch.arange(batch_size, device=feat_i.device)
                
                loss_i = F.cross_entropy(similarity, targets)
                loss_j = F.cross_entropy(similarity.t(), targets)
                
                alignment_loss += (loss_i + loss_j) / 2
                num_pairs += 1
        
        return alignment_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=features_list[0].device)


class EnhancedHierarchicalCortexFlow(HierarchicalCortexFlow):
    """Enhanced HierarchicalCortexFlow with Monte Carlo and Feature Alignment."""
    
    def __init__(self, input_dim: int, config: EnhancedConfig = EnhancedConfig()):
        super().__init__(input_dim, config)
        self.enhanced_config = config
        
        # Monte Carlo Dropout layers
        if config.ENABLE_MC_DROPOUT:
            self.mc_dropout = MonteCarloDropout(config.MC_DROPOUT_RATE)
        
        # Feature Alignment module
        if config.ENABLE_FEATURE_ALIGNMENT:
            self.feature_alignment = FeatureAlignmentModule(
                feature_dim=config.HIDDEN_DIM // 2,
                temperature=config.ALIGNMENT_TEMPERATURE
            )
        
        print(f"🔬 Enhanced features enabled:")
        print(f"   🎲 Monte Carlo Dropout: {config.ENABLE_MC_DROPOUT}")
        print(f"   🎯 Feature Alignment: {config.ENABLE_FEATURE_ALIGNMENT}")
        print(f"   📊 MC Samples: {config.MC_SAMPLES}")
    
    def forward_with_uncertainty(self, input_data: torch.Tensor, 
                               mc_samples: int = None) -> Dict[str, torch.Tensor]:
        """Forward pass with Monte Carlo uncertainty estimation."""
        if not self.enhanced_config.ENABLE_MC_DROPOUT:
            return self.forward(input_data)
        
        if mc_samples is None:
            mc_samples = self.enhanced_config.MC_SAMPLES
        
        # Collect multiple forward passes
        mc_outputs = []
        mc_features = []
        
        for _ in range(mc_samples):
            outputs = self.forward(input_data)
            mc_outputs.append(outputs['reconstruction'])
            mc_features.append(outputs['fused_features'])
        
        # Stack outputs
        mc_reconstructions = torch.stack(mc_outputs, dim=0)  # [mc_samples, batch, ...]
        mc_feature_stack = torch.stack(mc_features, dim=0)   # [mc_samples, batch, features]
        
        # Compute statistics
        mean_reconstruction = mc_reconstructions.mean(dim=0)
        uncertainty = mc_reconstructions.var(dim=0)
        
        # Get single forward pass for other outputs
        single_output = self.forward(input_data)
        
        return {
            'reconstruction': mean_reconstruction,
            'uncertainty': uncertainty,
            'mc_reconstructions': mc_reconstructions,
            'mc_features': mc_feature_stack,
            'progressive_outputs': single_output['progressive_outputs'],
            'scale_features': single_output['scale_features'],
            'fused_features': single_output['fused_features']
        }
    
    def forward(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with MC dropout and feature alignment."""
        # Extract features at multiple temporal scales
        scale_features = []
        for encoder in self.temporal_encoders:
            scale_feature = encoder(input_data)
            
            # Apply Monte Carlo dropout
            if self.enhanced_config.ENABLE_MC_DROPOUT:
                scale_feature = self.mc_dropout(scale_feature)
            
            scale_features.append(scale_feature)
        
        # Fuse multi-scale features
        fused_features = self.feature_pyramid(scale_features)
        
        # Apply MC dropout to fused features
        if self.enhanced_config.ENABLE_MC_DROPOUT:
            fused_features = self.mc_dropout(fused_features)
        
        # Progressive decoding
        progressive_outputs = self.progressive_decoder(fused_features)
        
        # Main output is the highest resolution
        main_output = progressive_outputs[f'level_{self.config.NUM_PYRAMID_LEVELS - 1}']
        
        return {
            'reconstruction': main_output,
            'progressive_outputs': progressive_outputs,
            'scale_features': scale_features,
            'fused_features': fused_features
        }
    
    def compute_enhanced_loss(self, input_data: torch.Tensor, 
                            target_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced loss computation with uncertainty and alignment."""
        # Get outputs with uncertainty
        outputs = self.forward_with_uncertainty(input_data)
        
        # Ensure target images have correct shape
        if target_images.dim() == 2:
            target_images = target_images.view(-1, 1, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        
        # Standard hierarchical losses
        base_losses = super().compute_loss(input_data, target_images)
        
        enhanced_losses = base_losses.copy()
        
        # Uncertainty loss (encourage confident predictions)
        if 'uncertainty' in outputs:
            uncertainty_loss = outputs['uncertainty'].mean()
            enhanced_losses['uncertainty_loss'] = uncertainty_loss
        else:
            enhanced_losses['uncertainty_loss'] = torch.tensor(0.0, device=target_images.device)
        
        # Feature alignment loss
        if self.enhanced_config.ENABLE_FEATURE_ALIGNMENT and hasattr(self, 'feature_alignment'):
            alignment_loss = self.feature_alignment(outputs['scale_features'])
            enhanced_losses['alignment_loss'] = alignment_loss
        else:
            enhanced_losses['alignment_loss'] = torch.tensor(0.0, device=target_images.device)
        
        # Consistency loss across MC samples
        consistency_loss = torch.tensor(0.0, device=target_images.device)
        if 'mc_reconstructions' in outputs:
            mc_reconstructions = outputs['mc_reconstructions']
            # Compute variance across MC samples as consistency measure
            consistency_loss = mc_reconstructions.var(dim=0).mean()
        enhanced_losses['consistency_loss'] = consistency_loss
        
        # Enhanced total loss
        enhanced_total_loss = (
            0.5 * base_losses['total_loss'] +
            self.enhanced_config.UNCERTAINTY_WEIGHT * enhanced_losses['uncertainty_loss'] +
            self.enhanced_config.ALIGNMENT_WEIGHT * enhanced_losses['alignment_loss'] +
            self.enhanced_config.CONSISTENCY_WEIGHT * enhanced_losses['consistency_loss']
        )
        
        enhanced_losses['enhanced_total_loss'] = enhanced_total_loss
        
        return enhanced_losses


def create_enhanced_model(input_dim: int, config: EnhancedConfig = None) -> EnhancedHierarchicalCortexFlow:
    """Factory function to create Enhanced HierarchicalCortexFlow model."""
    if config is None:
        config = EnhancedConfig()
    
    print(f"🚀 Creating Enhanced HierarchicalCortexFlow:")
    print(f"   📊 Input dimension: {input_dim}")
    print(f"   ⏱️  Temporal scales: {config.TEMPORAL_SCALES}")
    print(f"   🏗️  Hidden dimension: {config.HIDDEN_DIM}")
    print(f"   📈 Pyramid levels: {config.NUM_PYRAMID_LEVELS}")
    print(f"   🎲 MC Dropout rate: {config.MC_DROPOUT_RATE}")
    print(f"   🎯 Alignment weight: {config.ALIGNMENT_WEIGHT}")
    
    model = EnhancedHierarchicalCortexFlow(input_dim, config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   🎯 Total parameters: {total_params:,}")
    print(f"   🎯 Trainable parameters: {trainable_params:,}")
    
    return model


@dataclass
class EnhancedTrainingResult:
    """Enhanced training result with uncertainty metrics."""
    best_test_loss: float
    training_time_minutes: float
    total_epochs: int
    total_params: int
    input_dim: int
    final_train_loss: float
    convergence_epoch: int
    # Enhanced metrics
    best_uncertainty_loss: float
    best_alignment_loss: float
    best_consistency_loss: float
    mean_uncertainty: float
    uncertainty_reduction: float


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EnhancedExperimentLogger:
    """Enhanced logger with uncertainty and alignment tracking."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.start_time = time.time()

    def log_start(self, input_dim: int, total_params: int, config: EnhancedConfig):
        """Log experiment start with enhanced details."""
        print(f"\n{'='*80}")
        print(f"🔬 ENHANCED HIERARCHICAL CORTEXFLOW: {self.dataset_name.upper()} DATASET")
        print(f"{'='*80}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"✅ Dataset loaded: {input_dim} input dimensions")
        print(f"🔧 Total parameters: {total_params:,}")
        print(f"⏱️  Temporal scales: {config.TEMPORAL_SCALES}")
        print(f"🏗️  Hidden dimension: {config.HIDDEN_DIM}")
        print(f"📈 Pyramid levels: {config.NUM_PYRAMID_LEVELS}")
        print(f"🎲 Monte Carlo: {config.ENABLE_MC_DROPOUT} (samples: {config.MC_SAMPLES})")
        print(f"🎯 Feature Alignment: {config.ENABLE_FEATURE_ALIGNMENT}")

    def log_epoch(self, epoch: int, loss_dict: Dict[str, float],
                  lr: float, is_best: bool, patience: int):
        """Log epoch results with enhanced metrics."""
        if epoch % 5 == 0 or epoch < 10:
            elapsed = time.time() - self.start_time
            print(f"\nEpoch {epoch+1:3d} Enhanced Summary:")
            print(f"  Enhanced Loss:   {loss_dict.get('enhanced_total_loss', 0):.6f}")
            print(f"  Base Loss:       {loss_dict.get('total_loss', 0):.6f}")
            print(f"  Uncertainty:     {loss_dict.get('uncertainty_loss', 0):.6f}")
            print(f"  Alignment:       {loss_dict.get('alignment_loss', 0):.6f}")
            print(f"  Consistency:     {loss_dict.get('consistency_loss', 0):.6f}")
            print(f"  LR: {lr:.2e}")
            print(f"  Time: {elapsed/60:.1f}m")

            if is_best:
                print(f"  ✅ New best enhanced model saved!")
            else:
                print(f"  ⏳ Patience: {patience}/20")
            print("-" * 80)

    def log_completion(self, best_loss: float, total_epochs: int):
        """Log training completion."""
        total_time = time.time() - self.start_time
        print(f"\n🎉 Enhanced hierarchical training completed in {total_time/60:.1f} minutes!")
        print(f"🏆 Best enhanced test loss: {best_loss:.6f}")
        print(f"📊 Total epochs: {total_epochs}")


def create_data_loaders(dataset_path: str, config: EnhancedConfig) -> Tuple[Any, Any, int]:
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


def save_enhanced_checkpoint(model: EnhancedHierarchicalCortexFlow, optimizer, epoch: int,
                           best_loss: float, dataset_name: str, **kwargs):
    """Save enhanced model checkpoint."""
    os.makedirs('../../checkpoints', exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'dataset': dataset_name,
        'model_type': 'enhanced_hierarchical',
        'config': model.enhanced_config,
        **kwargs
    }
    torch.save(checkpoint, f'checkpoints/enhanced_{dataset_name.lower()}_model.pt')


def generate_enhanced_visualizations(model: EnhancedHierarchicalCortexFlow, test_loader,
                                   device: torch.device, dataset_name: str):
    """Generate enhanced visualizations with uncertainty."""
    model.eval()
    with torch.no_grad():
        input_data, images, _ = next(iter(test_loader))
        input_data = input_data[:8].to(device)
        images = images[:8].to(device)

        # Get enhanced outputs with uncertainty
        outputs = model.forward_with_uncertainty(input_data)
        main_reconstruction = outputs['reconstruction']
        uncertainty = outputs['uncertainty']

        if images.dim() == 2:
            images = images.view(-1, 1, 28, 28)

        images_viz = images.cpu().squeeze(1) if images.dim() == 4 else images.cpu()
        main_viz = main_reconstruction.cpu().squeeze(1) if main_reconstruction.dim() == 4 else main_reconstruction.cpu()
        uncertainty_viz = uncertainty.cpu().squeeze(1) if uncertainty.dim() == 4 else uncertainty.cpu()

        # Main reconstruction comparison
        plot_reconstruction_comparison(
            images_viz,
            main_viz,
            save_path=f'results/enhanced_{dataset_name.lower()}_reconstructions.png',
            title=f'Enhanced HierarchicalCortexFlow: {dataset_name} Reconstructions'
        )

        # Uncertainty visualization
        plot_uncertainty_visualization(
            images_viz, main_viz, uncertainty_viz,
            save_path=f'results/enhanced_{dataset_name.lower()}_uncertainty.png',
            title=f'Enhanced HierarchicalCortexFlow: {dataset_name} Uncertainty'
        )


def plot_uncertainty_visualization(original, reconstructed, uncertainty, save_path, title):
    """Plot reconstruction with uncertainty maps."""
    num_samples = min(4, len(original))

    fig, axes = plt.subplots(3, num_samples, figsize=(16, 8))

    for i in range(num_samples):
        # Original image
        axes[0, i].imshow(original[i], cmap='gray')
        axes[0, i].set_title('Original' if i == 0 else '')
        axes[0, i].axis('off')

        # Reconstructed image
        axes[1, i].imshow(reconstructed[i], cmap='gray')
        axes[1, i].set_title('Reconstructed' if i == 0 else '')
        axes[1, i].axis('off')

        # Uncertainty map
        im = axes[2, i].imshow(uncertainty[i], cmap='hot')
        axes[2, i].set_title('Uncertainty' if i == 0 else '')
        axes[2, i].axis('off')

        if i == num_samples - 1:
            plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_enhanced_dataset(dataset_name: str, dataset_path: str, device: torch.device,
                          config: EnhancedConfig = None) -> EnhancedTrainingResult:
    """Train enhanced model on a single dataset."""
    if config is None:
        config = EnhancedConfig()

    set_seeds(config.SEED)

    # Initialize components
    logger = EnhancedExperimentLogger(dataset_name)
    train_loader, test_loader, input_dim = create_data_loaders(dataset_path, config)

    # Create enhanced model
    model = create_enhanced_model(input_dim, config)
    model = model.to(device)

    # Enhanced optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    total_params = sum(p.numel() for p in model.parameters())
    logger.log_start(input_dim, total_params, config)

    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    best_metrics = {
        'uncertainty_loss': float('inf'),
        'alignment_loss': float('inf'),
        'consistency_loss': float('inf')
    }

    initial_uncertainty = None
    final_uncertainty = None

    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_losses = {
            'enhanced_total_loss': 0.0, 'total_loss': 0.0, 'uncertainty_loss': 0.0,
            'alignment_loss': 0.0, 'consistency_loss': 0.0
        }

        for input_data, images, _ in train_loader:
            input_data, images = input_data.to(device), images.to(device)

            optimizer.zero_grad()
            loss_dict = model.compute_enhanced_loss(input_data, images)

            total_loss = loss_dict['enhanced_total_loss']
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
        test_losses = {
            'enhanced_total_loss': 0.0, 'total_loss': 0.0, 'uncertainty_loss': 0.0,
            'alignment_loss': 0.0, 'consistency_loss': 0.0
        }

        with torch.no_grad():
            for input_data, images, _ in test_loader:
                input_data, images = input_data.to(device), images.to(device)
                loss_dict = model.compute_enhanced_loss(input_data, images)

                for key in test_losses.keys():
                    if key in loss_dict:
                        test_losses[key] += loss_dict[key].item()

                # Track uncertainty evolution
                if epoch == 0 and initial_uncertainty is None:
                    outputs = model.forward_with_uncertainty(input_data)
                    if 'uncertainty' in outputs:
                        initial_uncertainty = outputs['uncertainty'].mean().item()

                if epoch == config.NUM_EPOCHS - 1:
                    outputs = model.forward_with_uncertainty(input_data)
                    if 'uncertainty' in outputs:
                        final_uncertainty = outputs['uncertainty'].mean().item()

        # Average test losses
        for key in test_losses.keys():
            test_losses[key] /= len(test_loader)

        scheduler.step(test_losses['enhanced_total_loss'])

        # Check for improvement
        is_best = test_losses['enhanced_total_loss'] < best_loss
        if is_best:
            best_loss = test_losses['enhanced_total_loss']
            patience_counter = 0

            # Update best metrics
            for key in best_metrics.keys():
                if key in test_losses:
                    best_metrics[key] = test_losses[key]

            save_enhanced_checkpoint(
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
    generate_enhanced_visualizations(model, test_loader, device, dataset_name)

    # Calculate uncertainty reduction
    uncertainty_reduction = 0.0
    if initial_uncertainty is not None and final_uncertainty is not None:
        uncertainty_reduction = (initial_uncertainty - final_uncertainty) / initial_uncertainty

    return EnhancedTrainingResult(
        best_test_loss=best_loss,
        training_time_minutes=(time.time() - logger.start_time) / 60,
        total_epochs=epoch + 1,
        total_params=total_params,
        input_dim=input_dim,
        final_train_loss=train_losses['enhanced_total_loss'],
        convergence_epoch=epoch + 1 - patience_counter,
        best_uncertainty_loss=best_metrics['uncertainty_loss'],
        best_alignment_loss=best_metrics['alignment_loss'],
        best_consistency_loss=best_metrics['consistency_loss'],
        mean_uncertainty=final_uncertainty or 0.0,
        uncertainty_reduction=uncertainty_reduction
    )


def main():
    """Main training function for Enhanced HierarchicalCortexFlow."""
    print("🔬 ENHANCED HIERARCHICAL CORTEXFLOW TRAINING")
    print("=" * 80)
    print(f"Experiment start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎮 Using device: {device}")

    # Enhanced configuration
    config = EnhancedConfig(
        # Base hierarchical settings
        TEMPORAL_SCALES=[1, 2, 4, 8],
        HIDDEN_DIM=512,
        NUM_PYRAMID_LEVELS=4,

        # Enhanced settings
        ENABLE_MC_DROPOUT=True,
        MC_DROPOUT_RATE=0.15,
        MC_SAMPLES=10,

        ENABLE_FEATURE_ALIGNMENT=True,
        ALIGNMENT_WEIGHT=0.1,
        ALIGNMENT_TEMPERATURE=0.1,

        UNCERTAINTY_WEIGHT=0.05,
        CONSISTENCY_WEIGHT=0.08,

        # Training settings
        NUM_EPOCHS=100,
        LEARNING_RATE=1e-3,
        BATCH_SIZE=16,
        PATIENCE=20
    )

    # Dataset configurations
    datasets = [
        {
            'name': 'Miyawaki',
            'path': '../../data/processed/miyawaki_structured_28x28.mat',
            'description': 'Visual cortex fMRI → handwritten digits'
        },
        {
            'name': 'Vangerven',
            'path': '../../data/processed/digit69_28x28.mat',
            'description': 'Visual cortex fMRI → digit recognition'
        }
    ]

    results = {}
    total_start_time = time.time()

    # Train on each dataset
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        dataset_path = dataset_info['path']

        print(f"\n📁 Loading data from: {os.path.basename(dataset_path)}")

        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            print(f"⏭️  Skipping {dataset_name} dataset")
            continue

        try:
            # Load and display data info
            data = scipy.io.loadmat(dataset_path)
            for key, value in data.items():
                if not key.startswith('__') and hasattr(value, 'shape'):
                    print(f"  ✅ {key}: {value.shape}")

            # Display data information
            print(f"\n📊 Data Information:")
            train_fmri = data.get('fmriTrn', np.array([]))
            train_stim = data.get('stimTrn', np.array([]))
            test_fmri = data.get('fmriTest', np.array([]))
            test_stim = data.get('stimTest', np.array([]))

            if len(train_fmri) > 0:
                print(f"\n🎯 Train Data:")
                print(f"  stimuli: torch.Size({list(train_stim.shape)})")
                print(f"    Range: [{train_stim.min():.3f}, {train_stim.max():.3f}]")
                print(f"  fmri: torch.Size({list(train_fmri.shape)})")
                print(f"    Mean: {train_fmri.mean():.3f}, Std: {train_fmri.std():.3f}")

                if 'labelTrn' in data:
                    labels = data['labelTrn'].flatten()
                    print(f"  labels: torch.Size([{len(labels)}])")

            if len(test_fmri) > 0:
                print(f"\n🎯 Test Data:")
                print(f"  stimuli: torch.Size({list(test_stim.shape)})")
                print(f"    Range: [{test_stim.min():.3f}, {test_stim.max():.3f}]")
                print(f"  fmri: torch.Size({list(test_fmri.shape)})")
                print(f"    Mean: {test_fmri.mean():.3f}, Std: {test_fmri.std():.3f}")

                if 'labelTest' in data:
                    labels = data['labelTest'].flatten()
                    print(f"  labels: torch.Size([{len(labels)}])")

            # Train enhanced model
            result = train_enhanced_dataset(dataset_name, dataset_path, device, config)
            results[dataset_name] = result

        except Exception as e:
            print(f"❌ Error training {dataset_name}: {e}")
            results[dataset_name] = None

    # Final summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(f"🎉 ENHANCED HIERARCHICAL TRAINING COMPLETED!")
    print(f"Total experiment time: {total_time/60:.1f} minutes")

    print(f"\n📊 FINAL ENHANCED RESULTS:")
    print("-" * 80)

    for dataset_name, result in results.items():
        if result is not None:
            print(f"✅ {dataset_name:<12}: Enhanced {result.best_test_loss:.6f} | "
                  f"Uncertainty {result.best_uncertainty_loss:.6f} | "
                  f"Alignment {result.best_alignment_loss:.6f} | "
                  f"Time {result.training_time_minutes:.1f}m | "
                  f"Epochs {result.total_epochs:3d}")
        else:
            print(f"❌ {dataset_name:<12}: FAILED")

    # Enhanced metrics summary
    print(f"\n🔬 ENHANCED METRICS SUMMARY:")
    print("-" * 80)

    for dataset_name, result in results.items():
        if result is not None:
            print(f"\n📈 {dataset_name} Enhanced Analysis:")
            print(f"   🎯 Best Enhanced Loss: {result.best_test_loss:.6f}")
            print(f"   🎲 Uncertainty Loss: {result.best_uncertainty_loss:.6f}")
            print(f"   🎯 Alignment Loss: {result.best_alignment_loss:.6f}")
            print(f"   🔄 Consistency Loss: {result.best_consistency_loss:.6f}")
            print(f"   📊 Mean Uncertainty: {result.mean_uncertainty:.6f}")
            print(f"   📉 Uncertainty Reduction: {result.uncertainty_reduction:.2%}")
            print(f"   ⚡ Parameters: {result.total_params:,}")
            print(f"   ⏱️  Training Time: {result.training_time_minutes:.1f} minutes")

    print(f"\n🎉 Enhanced HierarchicalCortexFlow training completed successfully!")
    print(f"📁 Check '../../results/' for enhanced visualizations with uncertainty maps")
    print(f"💾 Check 'checkpoints/' for saved enhanced models")


if __name__ == "__main__":
    main()
