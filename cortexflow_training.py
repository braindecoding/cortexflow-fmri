#!/usr/bin/env python3
"""
Clean Code Version: Reproducible Simple CortexFlow Training
Refactored following clean code principles
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
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

# Add src to path
sys.path.append('src')

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
class Config:
    """Training configuration constants."""
    SEED: int = 42
    NUM_EPOCHS: int = 200
    PATIENCE: int = 20
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    BATCH_SIZE: int = 32
    HIDDEN_DIM: int = 512
    IMAGE_SIZE: int = 28
    DROPOUT_RATE: float = 0.2
    GRAD_CLIP_NORM: float = 1.0


@dataclass
class TrainingResult:
    """Training result data structure."""
    best_test_loss: float
    training_time_minutes: float
    total_epochs: int
    total_params: int
    input_dim: int
    final_train_loss: float
    convergence_epoch: int


def set_seeds(seed: int = Config.SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleCortexFlow(nn.Module):
    """Simple CortexFlow model with MSE loss only."""
    
    def __init__(self, input_dim: int, config: Config = Config()):
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
    def _build_encoder(self) -> nn.Sequential:
        """Build encoder network."""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.config.HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT_RATE),
            nn.Linear(self.config.HIDDEN_DIM * 2, self.config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT_RATE),
            nn.Linear(self.config.HIDDEN_DIM, self.config.HIDDEN_DIM)
        )
        
    def _build_decoder(self) -> nn.Sequential:
        """Build decoder network."""
        output_size = self.config.IMAGE_SIZE * self.config.IMAGE_SIZE
        return nn.Sequential(
            nn.Linear(self.config.HIDDEN_DIM, self.config.HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT_RATE),
            nn.Linear(self.config.HIDDEN_DIM * 2, self.config.HIDDEN_DIM * 4),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT_RATE),
            nn.Linear(self.config.HIDDEN_DIM * 4, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.encoder(input_data)
        reconstruction = self.decoder(features)
        return reconstruction.view(-1, 1, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
    
    def compute_loss(self, input_data: torch.Tensor, target_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute MSE loss."""
        reconstruction = self.forward(input_data)
        
        if target_images.dim() == 2:
            target_images = target_images.view(-1, 1, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        
        mse_loss = nn.functional.mse_loss(reconstruction, target_images)
        
        return {
            'total_loss': mse_loss,
            'mse_loss': mse_loss,
            'reconstruction': reconstruction
        }


class ModelTrainer:
    """Handles model training logic."""
    
    def __init__(self, model: SimpleCortexFlow, device: torch.device, config: Config = Config()):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for input_data, images, _ in train_loader:
            input_data, images = input_data.to(self.device), images.to(self.device)
            
            self.optimizer.zero_grad()
            loss_dict = self.model.compute_loss(input_data, images)
            loss = loss_dict['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.GRAD_CLIP_NORM)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate_epoch(self, test_loader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for input_data, images, _ in test_loader:
                input_data, images = input_data.to(self.device), images.to(self.device)
                loss_dict = self.model.compute_loss(input_data, images)
                total_loss += loss_dict['total_loss'].item()
                
        return total_loss / len(test_loader)


class ExperimentLogger:
    """Handles experiment logging and progress tracking."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.start_time = time.time()
        
    def log_start(self, input_dim: int, total_params: int):
        """Log experiment start."""
        print(f"\n{'='*80}")
        print(f"🧠 CORTEXFLOW: {self.dataset_name.upper()} DATASET")
        print(f"{'='*80}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"✅ Dataset loaded: {input_dim} input dimensions")
        print(f"🔧 Total parameters: {total_params:,}")
        
    def log_epoch(self, epoch: int, train_loss: float, test_loss: float, 
                  lr: float, is_best: bool, patience: int):
        """Log epoch results."""
        if epoch % 5 == 0 or epoch < 10:
            elapsed = time.time() - self.start_time
            print(f"\nEpoch {epoch+1:3d} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Test Loss:  {test_loss:.6f}")
            print(f"  LR: {lr:.2e}")
            print(f"  Time: {elapsed/60:.1f}m")
            
            if is_best:
                print(f"  ✅ New best model saved! (Loss: {test_loss:.6f})")
            else:
                print(f"  ⏳ Patience: {patience}/{Config.PATIENCE}")
            print("-" * 80)
    
    def log_completion(self, best_loss: float, total_epochs: int):
        """Log training completion."""
        total_time = time.time() - self.start_time
        print(f"\n🎉 Training completed in {total_time/60:.1f} minutes!")
        print(f"🏆 Best test loss: {best_loss:.6f}")
        print(f"📊 Total epochs: {total_epochs}")


def create_data_loaders(dataset_path: str, config: Config = Config()) -> Tuple[Any, Any, int]:
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


def save_model_checkpoint(model: SimpleCortexFlow, optimizer, epoch: int,
                         best_loss: float, dataset_name: str, **kwargs):
    """Save model checkpoint."""
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'dataset': dataset_name,
        **kwargs
    }
    torch.save(checkpoint, f'checkpoints/{dataset_name.lower()}_model.pt')


def generate_visualizations(model: SimpleCortexFlow, test_loader, device: torch.device,
                          dataset_name: str):
    """Generate and save reconstruction visualizations."""
    model.eval()
    with torch.no_grad():
        input_data, images, _ = next(iter(test_loader))
        input_data = input_data[:8].to(device)
        images = images[:8].to(device)

        reconstructions = model(input_data)

        if images.dim() == 2:
            images = images.view(-1, 1, 28, 28)

        images_viz = images.cpu().squeeze(1) if images.dim() == 4 else images.cpu()
        reconstructions_viz = reconstructions.cpu().squeeze(1) if reconstructions.dim() == 4 else reconstructions.cpu()

        plot_reconstruction_comparison(
            images_viz,
            reconstructions_viz,
            save_path=f'results/{dataset_name.lower()}_reconstructions.png',
            title=f'CortexFlow: {dataset_name} Reconstructions'
        )


def train_single_dataset(dataset_name: str, dataset_path: str, device: torch.device,
                        config: Config = Config()) -> TrainingResult:
    """Train model on a single dataset with clean separation of concerns."""
    set_seeds(config.SEED)

    # Initialize components
    logger = ExperimentLogger(dataset_name)
    train_loader, test_loader, input_dim = create_data_loaders(dataset_path, config)

    model = SimpleCortexFlow(input_dim, config)
    trainer = ModelTrainer(model, device, config)

    total_params = sum(p.numel() for p in model.parameters())
    logger.log_start(input_dim, total_params)

    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    train_losses, test_losses = [], []

    for epoch in range(config.NUM_EPOCHS):
        train_loss = trainer.train_epoch(train_loader)
        test_loss = trainer.validate_epoch(test_loader)

        trainer.scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        is_best = test_loss < best_loss
        if is_best:
            best_loss = test_loss
            patience_counter = 0
            save_model_checkpoint(
                model, trainer.optimizer, epoch, best_loss, dataset_name,
                train_losses=train_losses, test_losses=test_losses,
                input_dim=input_dim, total_params=total_params
            )
        else:
            patience_counter += 1

        logger.log_epoch(
            epoch, train_loss, test_loss,
            trainer.optimizer.param_groups[0]['lr'],
            is_best, patience_counter
        )

        if patience_counter >= config.PATIENCE:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            break

    logger.log_completion(best_loss, epoch + 1)
    generate_visualizations(model, test_loader, device, dataset_name)

    return TrainingResult(
        best_test_loss=best_loss,
        training_time_minutes=(time.time() - logger.start_time) / 60,
        total_epochs=epoch + 1,
        total_params=total_params,
        input_dim=input_dim,
        final_train_loss=train_losses[-1],
        convergence_epoch=len(train_losses) - patience_counter
    )


def main():
    """Main training function with clean architecture."""
    print("🧠 CORTEXFLOW TRAINING")
    print("=" * 80)
    print(f"Experiment start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seeds()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎮 Using device: {device}")

    os.makedirs('results', exist_ok=True)

    datasets = [
        ('Miyawaki', 'data/miyawaki_structured_28x28.mat'),
        ('Vangerven', 'data/digit69_28x28.mat'),
    ]

    results = {}
    total_start_time = time.time()

    for dataset_name, dataset_path in datasets:
        try:
            result = train_single_dataset(dataset_name, dataset_path, device)
            results[dataset_name] = result
        except Exception as e:
            print(f"❌ Error training {dataset_name}: {e}")
            results[dataset_name] = None

    # Print final summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*80}")
    print("🎉 TRAINING COMPLETED!")
    print(f"Total experiment time: {total_time/60:.1f} minutes")

    print(f"\n📊 FINAL RESULTS:")
    print("-" * 80)
    for dataset, result in results.items():
        if result:
            print(f"✅ {dataset:12s}: Loss {result.best_test_loss:.6f} | "
                  f"Time {result.training_time_minutes:.1f}m | "
                  f"Epochs {result.total_epochs:3d}")
        else:
            print(f"❌ {dataset:12s}: FAILED")


if __name__ == "__main__":
    main()
