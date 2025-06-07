#!/usr/bin/env python3
"""
Simple CortexFlow Trainer: Unified Training Framework

A clean, efficient training framework for Simple CortexFlow model that works
across all brain signal datasets (fMRI and EEG) with reproducible results.

Features:
- Unified training for all datasets
- Reproducible results with fixed seeds
- Early stopping and learning rate scheduling
- Comprehensive logging and visualization
- Model checkpointing and evaluation
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
from typing import Dict, Tuple, Optional, List
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.simple_cortexflow import SimpleCortexFlow, create_simple_cortexflow
from utils.visualization import plot_reconstruction_comparison


class SimpleCortexFlowTrainer:
    """
    Unified trainer for Simple CortexFlow model.
    
    Handles training, validation, and evaluation for all brain signal datasets
    with consistent hyperparameters and reproducible results.
    """
    
    def __init__(
        self,
        model: SimpleCortexFlow,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        max_epochs: int = 200,
        seed: int = 42
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_epochs = max_epochs
        self.seed = seed
        
        # Set seeds for reproducibility
        self._set_seeds()
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.test_losses = []
        self.training_time = 0
        
    def _set_seeds(self):
        """Set all random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (brain_signals, images, labels) in enumerate(train_loader):
            brain_signals = brain_signals.to(self.device)
            images = images.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass and loss computation
            loss_dict = self.model.compute_loss(brain_signals, images)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Progress logging (for small datasets, log more frequently)
            log_interval = max(1, num_batches // 5)
            if batch_idx % log_interval == 0:
                print(f"  Batch {batch_idx:3d}/{num_batches} | Loss: {loss.item():.6f}")
        
        return total_loss / num_batches
    
    def validate_epoch(self, test_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for brain_signals, images, labels in test_loader:
                brain_signals = brain_signals.to(self.device)
                images = images.to(self.device)
                
                loss_dict = self.model.compute_loss(brain_signals, images)
                total_loss += loss_dict['total_loss'].item()
        
        return total_loss / len(test_loader)
    
    def train(
        self, 
        train_loader: DataLoader, 
        test_loader: DataLoader,
        dataset_name: str = "Unknown",
        save_dir: str = "checkpoints"
    ) -> Dict[str, any]:
        """
        Train the model with early stopping and comprehensive logging.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            dataset_name: Name of dataset for logging
            save_dir: Directory to save checkpoints
            
        Returns:
            Training results dictionary
        """
        print(f"\n{'='*80}")
        print(f"🧠 SIMPLE CORTEXFLOW TRAINING: {dataset_name.upper()}")
        print(f"{'='*80}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Model info
        model_info = self.model.get_model_info()
        print(f"\n🔧 Model Configuration:")
        print(f"  Total parameters: {model_info['total_parameters']:,}")
        print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"  Model size: {model_info['model_size_mb']:.1f} MB")
        print(f"  Input dimension: {model_info['input_dim']}")
        
        print(f"\n📊 Dataset Information:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Batch size: {train_loader.batch_size}")
        
        print(f"\n🚀 Training Configuration:")
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Patience: {self.patience}")
        print(f"  Seed: {self.seed}")
        
        print(f"\n🚀 Starting training...")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            test_loss = self.validate_epoch(test_loader)
            
            # Update learning rate
            self.scheduler.step(test_loss)
            
            # Record losses
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            # Print epoch summary (every 5 epochs or important epochs)
            if epoch % 5 == 0 or epoch < 10 or test_loss < self.best_loss:
                elapsed = time.time() - start_time
                print(f"\nEpoch {epoch+1:3d}/{self.max_epochs} Summary:")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Test Loss:  {test_loss:.6f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                print(f"  Time: {elapsed/60:.1f}m")
            
            # Save best model
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.patience_counter = 0
                
                # Save checkpoint
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f'simple_cortexflow_{dataset_name.lower()}.pt')
                
                self.model.save_model(checkpoint_path, {
                    'epoch': epoch,
                    'train_losses': self.train_losses,
                    'test_losses': self.test_losses,
                    'best_loss': self.best_loss,
                    'dataset_name': dataset_name,
                    'training_config': {
                        'learning_rate': self.learning_rate,
                        'weight_decay': self.weight_decay,
                        'patience': self.patience,
                        'seed': self.seed
                    }
                })
                
                if epoch % 5 == 0 or epoch < 10:
                    print(f"  ✅ New best model saved! (Loss: {self.best_loss:.6f})")
            else:
                self.patience_counter += 1
                if epoch % 5 == 0 or epoch < 10:
                    print(f"  ⏳ Patience: {self.patience_counter}/{self.patience}")
            
            if epoch % 5 == 0 or epoch < 10:
                print("-" * 80)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
                break
        
        self.training_time = time.time() - start_time
        
        print(f"\n🎉 Training completed in {self.training_time/60:.1f} minutes!")
        print(f"🏆 Best test loss: {self.best_loss:.6f}")
        
        # Generate training results
        results = {
            'dataset_name': dataset_name,
            'best_test_loss': self.best_loss,
            'training_time_minutes': self.training_time / 60,
            'total_epochs': len(self.train_losses),
            'convergence_epoch': len(self.train_losses) - self.patience_counter,
            'final_train_loss': self.train_losses[-1],
            'model_info': model_info,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }
        
        return results
    
    def evaluate_and_visualize(
        self, 
        test_loader: DataLoader, 
        dataset_name: str,
        save_dir: str = "results",
        num_samples: int = 8
    ):
        """Evaluate model and generate visualizations."""
        print(f"\n🎨 Generating evaluation and visualizations...")
        
        self.model.eval()
        with torch.no_grad():
            # Get a batch of test data
            brain_signals, images, labels = next(iter(test_loader))
            brain_signals = brain_signals[:num_samples].to(self.device)
            images = images[:num_samples].to(self.device)
            
            # Generate reconstructions
            reconstructions = self.model.reconstruct(brain_signals)
            
            # Prepare images for visualization
            if images.dim() == 2:  # [batch, 784]
                images = images.view(-1, 1, 28, 28)
            
            # Convert to numpy for visualization
            images_viz = images.cpu().squeeze(1) if images.dim() == 4 else images.cpu()
            reconstructions_viz = reconstructions.cpu().squeeze(1) if reconstructions.dim() == 4 else reconstructions.cpu()
            
            # Create visualization
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'simple_cortexflow_{dataset_name.lower()}_results.png')
            
            plot_reconstruction_comparison(
                images_viz, 
                reconstructions_viz,
                save_path=save_path,
                title=f'Simple CortexFlow: {dataset_name} Results'
            )
            
            print(f"  ✅ Visualizations saved to: {save_path}")
    
    def plot_training_curves(self, dataset_name: str, save_dir: str = "results"):
        """Plot and save training curves."""
        plt.figure(figsize=(12, 4))
        
        # Linear scale
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue', alpha=0.7)
        plt.plot(self.test_losses, label='Test Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Simple CortexFlow Training: {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Log scale
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss', color='blue', alpha=0.7)
        plt.plot(self.test_losses, label='Test Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title(f'Training Curves (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'simple_cortexflow_{dataset_name.lower()}_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Training curves saved to: {save_path}")


def create_trainer(
    input_dim: int,
    device: str = 'cuda',
    **kwargs
) -> SimpleCortexFlowTrainer:
    """
    Factory function to create Simple CortexFlow trainer.
    
    Args:
        input_dim: Input dimension for the model
        device: Device to use for training
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured trainer instance
    """
    model = create_simple_cortexflow(input_dim=input_dim, device=device)
    return SimpleCortexFlowTrainer(model=model, device=device, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("🧠 Simple CortexFlow Trainer")
    print("=" * 50)
    
    # Create trainer for Miyawaki dataset
    trainer = create_trainer(input_dim=967)  # Miyawaki config
    print(f"✅ Trainer created for input_dim=967")
    print(f"✅ Model info: {trainer.model.get_model_info()}")
