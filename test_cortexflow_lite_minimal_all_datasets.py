"""
CortexFlow-Lite-Minimal: Test on All Datasets
=============================================

Test the winning CortexFlow-Lite-Minimal architecture on all datasets
to verify if it can beat Brain-Diffuser consistently across datasets.

Hypothesis: If CortexFlow-Lite-Minimal beats Brain-Diffuser on Miyawaki,
it should also perform well on other datasets due to its fundamental
effectiveness.

Test Plan:
1. Train CortexFlow-Lite-Minimal on all 4 datasets
2. Train Brain-Diffuser on all 4 datasets (for comparison)
3. Compare performance across datasets
4. Verify consistency of the winning approach
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime

# Import models
from src.models import OptimizedBrainDiffuser, CortexFlowLiteMinimal
from src.evaluation.metrics import calculate_comprehensive_metrics
from src.data import load_dataset_gpu_optimized

def setup_device():
    """Setup CUDA device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        return device
    else:
        return torch.device('cpu')

def get_optimal_config_for_dataset(dataset_name):
    """Get optimal configuration for each dataset"""
    base_config = {
        'lr': 0.001,
        'weight_decay': 1e-06,
        'epochs': 200,
        'patience': 25,
        'gradient_clip': 0.5,
        'scheduler_factor': 0.5,
        'scheduler_patience': 15
    }
    
    # Dataset-specific optimizations
    dataset_configs = {
        'miyawaki': {
            'batch_size': 64,  # Proven optimal
            'epochs': 250
        },
        'vangerven': {
            'batch_size': 32,
            'epochs': 150
        },
        'mindbigdata': {
            'batch_size': 40,
            'epochs': 120
        },
        'crell': {
            'batch_size': 36,
            'epochs': 140
        }
    }
    
    if dataset_name in dataset_configs:
        base_config.update(dataset_configs[dataset_name])
    
    return base_config

def train_model_optimal(model, train_loader, val_loader, config, device, dataset_name):
    """Optimal training with proven configuration"""
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        min_lr=1e-7
    )
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"🚀 Training {model.name} on {dataset_name.upper()}")
    print(f"📊 Config: lr={config['lr']}, batch_size={config['batch_size']}, epochs={config['epochs']}")
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
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
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Progress reporting
        if epoch % 25 == 0 or epoch == config['epochs'] - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{config['epochs']}: "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {current_lr:.2e}, Patience: {patience_counter}/{config['patience']}")
        
        if patience_counter >= config['patience']:
            print(f"🛑 Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return {
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }

def evaluate_model(model, test_loader, device, dataset_name):
    """Evaluate model performance"""
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
    
    metrics = calculate_comprehensive_metrics(predictions, targets, device)
    
    print(f"📊 {model.name} on {dataset_name.upper()}:")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   PSNR: {metrics['psnr']:.2f} dB")
    print(f"   SSIM: {metrics['ssim']:.4f}")
    print(f"   LPIPS: {metrics['lpips']:.4f}")
    
    return metrics

def main():
    """Main testing function for all datasets"""
    print("🎯 CortexFlow-Lite-Minimal: Test on All Datasets")
    print("=" * 55)
    print("🔬 Hypothesis: CortexFlow-Lite-Minimal should beat Brain-Diffuser")
    print("   consistently across all datasets due to its fundamental effectiveness")
    print()
    
    device = setup_device()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Models to compare
    models_to_test = [
        ('CortexFlow-Lite-Minimal', CortexFlowLiteMinimal),
        ('Brain-Diffuser', OptimizedBrainDiffuser)
    ]
    
    # All datasets
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    # Results storage
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"🎯 Testing {len(models_to_test)} models on {len(datasets)} datasets")
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
            
            # Get optimal config for this dataset
            config = get_optimal_config_for_dataset(dataset_name)
            
            # Split for validation
            train_size = int(0.8 * len(X_train))
            X_train_split = X_train[:train_size]
            y_train_split = y_train[:train_size]
            X_val = X_train[train_size:]
            y_val = y_train[train_size:]
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_split, y_train_split)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
            
            print(f"✅ Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            continue
        
        dataset_results = {}
        
        for model_name, model_class in models_to_test:
            print(f"\n🔧 Training {model_name} on {dataset_name.upper()}")
            
            try:
                # Initialize model
                model = model_class(input_dim, device)
                
                # Train model
                training_result = train_model_optimal(
                    model, train_loader, val_loader, config, device, dataset_name
                )
                
                # Evaluate model
                test_metrics = evaluate_model(model, test_loader, device, dataset_name)
                
                # Store results
                dataset_results[model_name] = {
                    'training': training_result,
                    'metrics': test_metrics,
                    'config': config
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
    results_dir = f"results/cortexflow_lite_minimal_all_datasets_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/all_datasets_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Comprehensive analysis
    print("🎉 All Datasets Testing Complete!")
    print("=" * 50)
    
    print("\n📊 COMPREHENSIVE RESULTS:")
    print("=" * 30)
    
    cortex_wins = 0
    brain_wins = 0
    ties = 0
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        
        if 'CortexFlow-Lite-Minimal' in dataset_results and 'Brain-Diffuser' in dataset_results:
            cortex_mse = dataset_results['CortexFlow-Lite-Minimal']['metrics']['mse']
            brain_mse = dataset_results['Brain-Diffuser']['metrics']['mse']
            
            print(f"   CortexFlow-Lite-Minimal: MSE={cortex_mse:.6f}")
            print(f"   Brain-Diffuser:          MSE={brain_mse:.6f}")
            
            if cortex_mse < brain_mse:
                improvement = ((brain_mse - cortex_mse) / brain_mse) * 100
                print(f"   🏆 CortexFlow-Lite-Minimal WINS by {improvement:.2f}%")
                cortex_wins += 1
            elif brain_mse < cortex_mse:
                gap = ((cortex_mse - brain_mse) / brain_mse) * 100
                print(f"   🥈 Brain-Diffuser wins by {gap:.2f}%")
                brain_wins += 1
            else:
                print(f"   🤝 TIE")
                ties += 1
    
    # Final verdict
    print(f"\n🏆 FINAL VERDICT:")
    print(f"=" * 20)
    print(f"CortexFlow-Lite-Minimal wins: {cortex_wins}/{len(datasets)} datasets")
    print(f"Brain-Diffuser wins: {brain_wins}/{len(datasets)} datasets")
    print(f"Ties: {ties}/{len(datasets)} datasets")
    
    if cortex_wins > brain_wins:
        print(f"\n🎉 HYPOTHESIS CONFIRMED!")
        print(f"CortexFlow-Lite-Minimal is consistently superior!")
        print(f"The fundamental approach beats complex diffusion across datasets!")
    elif brain_wins > cortex_wins:
        print(f"\n🤔 HYPOTHESIS PARTIALLY REJECTED")
        print(f"Brain-Diffuser still dominates on most datasets")
        print(f"Miyawaki win might be dataset-specific")
    else:
        print(f"\n⚖️ MIXED RESULTS")
        print(f"Performance varies by dataset characteristics")
    
    print(f"\n💾 Results saved to: {results_dir}/")
    print("🚀 Comprehensive Testing Complete!")

if __name__ == "__main__":
    main()
