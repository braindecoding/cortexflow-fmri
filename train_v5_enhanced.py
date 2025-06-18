"""
CortexFlow V5 Enhanced Training: Brain-Diffuser Inspired Tuning
==============================================================

V5 Training script for CortexFlow Enhanced with Brain-Diffuser inspiration.
Focus on improving single-modal performance while maintaining cross-modal capabilities.

Key Features:
- CortexFlow Enhanced with Brain-Diffuser components
- Optimized training for single-modal datasets
- Enhanced stability and convergence
- Comprehensive evaluation and comparison

Goals:
- Compete with Brain-Diffuser on Miyawaki (target: < 0.008761)
- Maintain cross-modal performance
- Improved overall consistency
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import scipy.io as sio

# Import models
from src.models import (
    StandardBaselineCNN, OptimizedMinDVis, OptimizedBrainDiffuser,
    CortexFlowMultiPathway, CortexFlowEnhanced, CortexFlowEnsemble
)
from src.evaluation.metrics import calculate_comprehensive_metrics
from src.data import load_dataset_gpu_optimized

def setup_device():
    """Setup CUDA device with optimizations"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        
        # Memory optimization
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        
        return device
    else:
        print("⚠️ CUDA not available, using CPU")
        return torch.device('cpu')

def load_project_config():
    """Load project configuration"""
    try:
        with open('project_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ project_config.json not found, using default config")
        return {
            "datasets": {
                "miyawaki": {"input_dim": 967},
                "vangerven": {"input_dim": 3092},
                "mindbigdata": {"input_dim": 3092},
                "crell": {"input_dim": 3092}
            }
        }

def get_enhanced_training_config(dataset_name):
    """Get enhanced training configuration optimized for V5"""
    base_configs = {
        'miyawaki': {
            'epochs': 200,  # Increased for better convergence
            'lr': 0.0008,   # Slightly reduced for stability
            'batch_size': 48,  # Optimized batch size
            'patience': 25,
            'weight_decay': 1e-5  # Reduced weight decay
        },
        'vangerven': {
            'epochs': 150,
            'lr': 0.001,
            'batch_size': 32,
            'patience': 20,
            'weight_decay': 1e-5
        },
        'mindbigdata': {
            'epochs': 120,
            'lr': 0.0012,
            'batch_size': 40,
            'patience': 15,
            'weight_decay': 1e-5
        },
        'crell': {
            'epochs': 140,
            'lr': 0.001,
            'batch_size': 36,
            'patience': 18,
            'weight_decay': 1e-5
        }
    }
    return base_configs.get(dataset_name, base_configs['miyawaki'])

def train_model_enhanced(model, train_loader, val_loader, config, device, dataset_name):
    """Enhanced training with Brain-Diffuser inspired optimizations"""
    
    # Enhanced optimizer with lower weight decay
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),  # Standard Adam betas
        eps=1e-8
    )
    
    # Enhanced scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,  # Less aggressive reduction
        patience=config['patience']//2,
        min_lr=1e-6
    )
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    print(f"🚀 Training {model.name} on {dataset_name.upper()}")
    print(f"📊 Config: epochs={config['epochs']}, lr={config['lr']}, batch_size={config['batch_size']}")
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Progress reporting
        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{config['epochs']}: "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {current_lr:.2e}, Patience: {patience_counter}/{config['patience']}")
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"🛑 Early stopping at epoch {epoch}")
            break
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    return {
        'best_val_loss': best_val_loss,
        'training_history': training_history,
        'epochs_trained': epoch + 1
    }

def evaluate_model_comprehensive(model, test_loader, device, dataset_name):
    """Comprehensive evaluation with multiple metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(predictions, targets, device)
    
    print(f"📊 {model.name} on {dataset_name.upper()}:")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   PSNR: {metrics['psnr']:.2f} dB")
    print(f"   SSIM: {metrics['ssim']:.4f}")
    print(f"   LPIPS: {metrics['lpips']:.4f}")
    
    return metrics

def main():
    """Main V5 Enhanced training function"""
    print("🧠 CortexFlow V5 Enhanced Training: Brain-Diffuser Inspired Tuning")
    print("=" * 70)
    
    # Setup
    device = setup_device()
    config = load_project_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Models to train (focus on enhanced model)
    models_to_train = [
        ('CortexFlow_Enhanced', CortexFlowEnhanced),
        ('Brain_Diffuser', OptimizedBrainDiffuser),  # For comparison
        ('CortexFlow_Multi-Pathway', CortexFlowMultiPathway)  # Original for comparison
    ]
    
    # Datasets to train on
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    # Results storage
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"🎯 Training {len(models_to_train)} models on {len(datasets)} datasets")
    print(f"📅 Timestamp: {timestamp}")
    print()
    
    for dataset_name in datasets:
        print(f"📁 Loading dataset: {dataset_name.upper()}")
        
        # Load dataset
        try:
            X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)

            if X_train is None:
                print(f"❌ Failed to load {dataset_name}")
                continue

            # Split train data for validation (80/20 split)
            train_size = int(0.8 * len(X_train))
            X_train_split = X_train[:train_size]
            y_train_split = y_train[:train_size]
            X_val = X_train[train_size:]
            y_val = y_train[train_size:]

            # Create data loaders
            train_dataset = TensorDataset(X_train_split, y_train_split)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)
            
            training_config = get_enhanced_training_config(dataset_name)
            
            train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=training_config['batch_size'], shuffle=False)
            
            print(f"✅ Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            continue
        
        dataset_results = {}
        
        for model_name, model_class in models_to_train:
            print(f"\n🔧 Training {model_name} on {dataset_name.upper()}")
            
            try:
                # Initialize model
                model = model_class(input_dim, device)
                
                # Train model
                training_result = train_model_enhanced(
                    model, train_loader, val_loader, training_config, device, dataset_name
                )
                
                # Evaluate model
                test_metrics = evaluate_model_comprehensive(model, test_loader, device, dataset_name)
                
                # Store results
                dataset_results[model_name] = {
                    'training': training_result,
                    'metrics': test_metrics
                }
                
                print(f"✅ {model_name} completed: MSE={test_metrics['mse']:.6f}")
                
                # Memory cleanup
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue
        
        all_results[dataset_name] = dataset_results
        print(f"📊 {dataset_name.upper()} completed\n")
    
    # Save results
    results_dir = f"results/v5_enhanced_training_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/training_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print("🎉 V5 Enhanced Training Completed!")
    print("=" * 50)
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n📊 {dataset_name.upper()} Results:")
        for model_name, results in dataset_results.items():
            mse = results['metrics']['mse']
            print(f"   {model_name}: MSE = {mse:.6f}")
    
    print(f"\n💾 Results saved to: {results_dir}/")
    print("🚀 V5 Enhanced Training Complete!")

if __name__ == "__main__":
    main()
