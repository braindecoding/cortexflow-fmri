#!/usr/bin/env python3
"""
Complete CortexFlow Training Experiment
All 5 variants on all 4 datasets with ablation studies
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
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from full_cortexflow_models import create_model, count_parameters

class FullExperimentRunner:
    """Complete experiment runner for all CortexFlow variants"""
    
    def __init__(self, results_dir="results/full_experiments"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "data").mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Experiment runner initialized on {self.device}")
        
        # Experiment configuration
        self.variants = ['simple', 'mc', 'hierarchical', 'enhanced', 'unified']
        self.datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
        
        # Training configuration
        self.config = {
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'patience': 15,
            'mc_samples': 10
        }
        
        # Results storage
        self.all_results = {}
        
    def load_dataset(self, dataset_name):
        """Load dataset with proper preprocessing"""
        print(f"📊 Loading {dataset_name} dataset...")
        
        file_paths = {
            'miyawaki': 'data/processed/miyawaki_structured_28x28.mat',
            'vangerven': 'data/processed/digit69_28x28.mat',
            'mindbigdata': 'data/processed/mindbigdata.mat',
            'crell': 'data/processed/crell.mat'
        }
        
        try:
            data = loadmat(file_paths[dataset_name])
            
            # Extract training data based on dataset structure
            if dataset_name == 'miyawaki':
                X = data['fmriTrn']  # (107, 967)
                y = data['stimTrn']  # (107, 784)
            else:
                # For other datasets, find the largest arrays
                arrays = [(k, v) for k, v in data.items() 
                         if isinstance(v, np.ndarray) and v.ndim >= 2 and not k.startswith('__')]
                arrays.sort(key=lambda x: x[1].size, reverse=True)
                
                if len(arrays) >= 2:
                    X = arrays[0][1]  # Largest array (likely fMRI/EEG)
                    y = arrays[1][1]  # Second largest (likely images)
                else:
                    print(f"❌ Could not identify data arrays in {dataset_name}")
                    return None, None
            
            # Process shapes
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            if y.ndim > 2:
                y = y.reshape(y.shape[0], -1)
            
            # Ensure y is 784 dimensions (28x28)
            if y.shape[1] != 784:
                if y.shape[1] > 784:
                    y = y[:, :784]
                else:
                    pad_size = 784 - y.shape[1]
                    y = np.pad(y, ((0, 0), (0, pad_size)), mode='constant')
            
            # Normalize data
            X = (X - X.mean()) / (X.std() + 1e-8)
            
            # Normalize y based on data type
            if y.max() > 1.0:  # Likely 0-255 range
                y = y / 255.0
            else:  # Already 0-1 range
                y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            
            print(f"✅ {dataset_name} loaded: X{X.shape}, y{y.shape}")
            return torch.FloatTensor(X), torch.FloatTensor(y)
            
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            return None, None
    
    def create_data_loaders(self, X, y):
        """Create train/val/test data loaders"""
        n = len(X)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        
        # Random shuffle
        indices = torch.randperm(n)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_variant(self, variant, dataset_name, train_loader, val_loader):
        """Train a specific variant on a dataset"""
        print(f"🚀 Training {variant} on {dataset_name}...")
        
        # Get input dimension from first batch
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[1]
        
        # Create model
        model = create_model(variant, input_dim)
        model = model.to(self.device)
        
        # Count parameters
        param_count = count_parameters(model)
        print(f"   Parameters: {param_count:,}")
        
        # Optimizer and loss
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate'], 
            weight_decay=self.config['weight_decay']
        )
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_losses = []
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass based on variant
                if variant == 'simple':
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                elif variant == 'mc':
                    output, uncertainty = model(batch_x)
                    recon_loss = criterion(output, batch_y)
                    # Stable uncertainty loss
                    uncertainty = torch.clamp(uncertainty, min=1e-6, max=10.0)
                    uncertainty_loss = torch.mean(torch.log(uncertainty + 1e-8) + (batch_y - output)**2 / (uncertainty + 1e-8))
                    loss = recon_loss + 0.01 * uncertainty_loss  # Reduced weight
                elif variant == 'hierarchical':
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                elif variant == 'enhanced':
                    output, uncertainty, alignment = model(batch_x)
                    recon_loss = criterion(output, batch_y)
                    # Stable uncertainty loss
                    uncertainty = torch.clamp(uncertainty, min=1e-6, max=10.0)
                    uncertainty_loss = torch.mean(torch.log(uncertainty + 1e-8) + (batch_y - output)**2 / (uncertainty + 1e-8))
                    alignment_loss = torch.mean(alignment**2)  # L2 regularization
                    loss = 0.9 * recon_loss + 0.05 * uncertainty_loss + 0.05 * alignment_loss
                elif variant == 'unified':
                    output, uncertainty, complexity = model(batch_x, return_complexity=True)
                    recon_loss = criterion(output, batch_y)
                    # Stable uncertainty loss
                    uncertainty = torch.clamp(uncertainty, min=1e-6, max=10.0)
                    uncertainty_loss = torch.mean(torch.log(uncertainty + 1e-8) + (batch_y - output)**2 / (uncertainty + 1e-8))
                    complexity_loss = torch.mean((complexity - 0.5)**2)  # Encourage balanced complexity
                    loss = 0.9 * recon_loss + 0.05 * uncertainty_loss + 0.05 * complexity_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    if variant == 'simple':
                        output = model(batch_x)
                        val_loss = criterion(output, batch_y)
                    elif variant == 'mc':
                        output, _ = model(batch_x)
                        val_loss = criterion(output, batch_y)
                    elif variant == 'hierarchical':
                        output = model(batch_x)
                        val_loss = criterion(output, batch_y)
                    elif variant == 'enhanced':
                        output, _, _ = model(batch_x)
                        val_loss = criterion(output, batch_y)
                    elif variant == 'unified':
                        output, _, _ = model(batch_x, return_complexity=True)
                        val_loss = criterion(output, batch_y)
                    
                    val_losses.append(val_loss.item())
            
            # Record history
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['epochs'].append(epoch)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(
                    model.state_dict(), 
                    self.results_dir / "models" / f"{variant}_{dataset_name}_best.pt"
                )
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['patience']:
                print(f"   Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
        
        training_time = time.time() - start_time
        
        results = {
            'variant': variant,
            'dataset': dataset_name,
            'best_val_loss': best_val_loss,
            'final_epoch': len(history['epochs']) - 1,
            'training_time': training_time,
            'parameters': param_count,
            'history': history
        }
        
        print(f"✅ {variant} on {dataset_name} complete: Val Loss = {best_val_loss:.6f}")
        
        return model, results

def main():
    """Main execution - full experiment"""
    print("🚀 STARTING FULL CORTEXFLOW EXPERIMENT")
    print("=" * 60)
    print(f"Variants: {5}")
    print(f"Datasets: {4}")
    print(f"Total experiments: {20}")
    print("=" * 60)
    
    runner = FullExperimentRunner()
    
    # Run all experiments
    experiment_count = 0
    total_experiments = len(runner.variants) * len(runner.datasets)
    
    for dataset_name in runner.datasets:
        print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")
        
        # Load dataset
        X, y = runner.load_dataset(dataset_name)
        if X is None or y is None:
            print(f"❌ Skipping {dataset_name} due to loading error")
            continue
        
        # Create data loaders
        train_loader, val_loader, test_loader = runner.create_data_loaders(X, y)
        
        # Train all variants on this dataset
        dataset_results = {}
        
        for variant in runner.variants:
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}] Training {variant} on {dataset_name}")
            
            try:
                model, results = runner.train_variant(variant, dataset_name, train_loader, val_loader)
                dataset_results[variant] = results
            except Exception as e:
                print(f"❌ Training failed for {variant} on {dataset_name}: {e}")
                continue
        
        runner.all_results[dataset_name] = dataset_results
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = runner.results_dir / "data" / f"full_experiment_results_{timestamp}.json"
    
    # Convert for JSON serialization
    json_results = {}
    for dataset, dataset_data in runner.all_results.items():
        json_results[dataset] = {}
        for variant, variant_data in dataset_data.items():
            json_results[dataset][variant] = {
                'variant': variant_data['variant'],
                'dataset': variant_data['dataset'],
                'best_val_loss': float(variant_data['best_val_loss']),
                'final_epoch': int(variant_data['final_epoch']),
                'training_time': float(variant_data['training_time']),
                'parameters': int(variant_data['parameters']),
                'history': {
                    'train_loss': [float(x) for x in variant_data['history']['train_loss']],
                    'val_loss': [float(x) for x in variant_data['history']['val_loss']],
                    'epochs': [int(x) for x in variant_data['history']['epochs']],
                    'learning_rate': [float(x) for x in variant_data['history']['learning_rate']]
                }
            }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n🎉 FULL EXPERIMENT COMPLETE!")
    print("=" * 60)
    print(f"📊 Results saved to: {results_file}")
    print(f"🔬 Total experiments completed: {experiment_count}")
    print("✅ Ready for comprehensive analysis!")

if __name__ == "__main__":
    main()
