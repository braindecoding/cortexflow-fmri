#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE CORTEXFLOW TRAINING
Train all 5 variants on all 4 datasets for journal publication
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our models
from scripts.full_cortexflow_models import create_model, count_parameters

def load_dataset(dataset_name):
    """Load and preprocess dataset"""
    print(f"📊 Loading {dataset_name}...")
    
    file_paths = {
        'miyawaki': 'data/processed/miyawaki_structured_28x28.mat',
        'vangerven': 'data/processed/digit69_28x28.mat', 
        'mindbigdata': 'data/processed/mindbigdata.mat',
        'crell': 'data/processed/crell.mat'
    }
    
    try:
        data = loadmat(file_paths[dataset_name])
        
        if dataset_name == 'miyawaki':
            X = data['fmriTrn']
            y = data['stimTrn']
        else:
            # Find largest arrays
            arrays = [(k, v) for k, v in data.items() 
                     if isinstance(v, np.ndarray) and v.ndim >= 2 and not k.startswith('__')]
            arrays.sort(key=lambda x: x[1].size, reverse=True)
            X, y = arrays[0][1], arrays[1][1]
        
        # Reshape and normalize
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
            
        # Ensure y is 784 dimensions
        if y.shape[1] != 784:
            if y.shape[1] > 784:
                y = y[:, :784]
            else:
                y = np.pad(y, ((0, 0), (0, 784 - y.shape[1])), mode='constant')
        
        # Normalize
        X = (X - X.mean()) / (X.std() + 1e-8)
        y = y / (y.max() + 1e-8) if y.max() > 1.0 else (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        print(f"✅ {dataset_name}: X{X.shape}, y{y.shape}")
        return torch.FloatTensor(X), torch.FloatTensor(y)
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None, None

def train_model(variant, dataset_name, X, y, device):
    """Train a single model variant"""
    print(f"\n🚀 Training {variant} on {dataset_name}")
    
    # Create data loaders
    n = len(X)
    n_train = int(0.8 * n)
    indices = torch.randperm(n)
    
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_test, y_test = X[indices[n_train:]], y[indices[n_train:]]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)
    
    # Create model
    model = create_model(variant, X.shape[1])
    model = model.to(device)
    param_count = count_parameters(model)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_loss = float('inf')
    patience = 0
    max_patience = 10
    start_time = time.time()
    
    for epoch in range(50):
        # Training
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass based on variant
            if variant == 'simple':
                output = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant == 'mc':
                output, uncertainty = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant == 'hierarchical':
                output = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant == 'enhanced':
                output, uncertainty, alignment = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant == 'unified':
                output, uncertainty, complexity = model(batch_x, return_complexity=True)
                loss = criterion(output, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                if variant == 'simple':
                    output = model(batch_x)
                elif variant == 'mc':
                    output, _ = model(batch_x)
                elif variant == 'hierarchical':
                    output = model(batch_x)
                elif variant == 'enhanced':
                    output, _, _ = model(batch_x)
                elif variant == 'unified':
                    output, _, _ = model(batch_x, return_complexity=True)
                
                test_loss = criterion(output, batch_y)
                test_losses.append(test_loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_test_loss = np.mean(test_losses)
        
        scheduler.step(avg_test_loss)
        
        # Early stopping
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience = 0
        else:
            patience += 1
            
        if patience >= max_patience:
            break
            
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Train={avg_train_loss:.6f}, Test={avg_test_loss:.6f}")
    
    training_time = time.time() - start_time
    
    print(f"✅ {variant} on {dataset_name}: Test Loss = {best_loss:.6f}, Time = {training_time:.1f}s")
    
    return {
        'variant': variant,
        'dataset': dataset_name,
        'test_loss': best_loss,
        'training_time': training_time,
        'parameters': param_count,
        'epochs': epoch + 1
    }

def main():
    """Main training execution"""
    print("🚀 COMPREHENSIVE CORTEXFLOW TRAINING")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    variants = ['simple', 'mc', 'hierarchical', 'enhanced', 'unified']
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    print(f"📊 Training {len(variants)} variants on {len(datasets)} datasets")
    print(f"🎯 Total experiments: {len(variants) * len(datasets)}")
    print("=" * 60)
    
    # Results storage
    all_results = {}
    experiment_count = 0
    total_experiments = len(variants) * len(datasets)
    
    # Train all combinations
    for dataset_name in datasets:
        print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")
        
        # Load dataset
        X, y = load_dataset(dataset_name)
        if X is None:
            continue
            
        dataset_results = {}
        
        for variant in variants:
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}]")
            
            try:
                result = train_model(variant, dataset_name, X, y, device)
                dataset_results[variant] = result
            except Exception as e:
                print(f"❌ Training failed: {e}")
                continue
        
        all_results[dataset_name] = dataset_results
    
    # Save results
    results_dir = Path("results/comprehensive_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"all_variants_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Results saved: {results_file}")
    print(f"🔬 Experiments completed: {experiment_count}")
    
    # Print summary
    print("\n📈 RESULTS SUMMARY:")
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        for variant, result in results.items():
            print(f"  {variant:12}: {result['test_loss']:.6f} (params: {result['parameters']:,})")

if __name__ == "__main__":
    main()
