"""
CortexFlow Lite Optimization: Beat Brain-Diffuser Challenge
==========================================================

Specialized training untuk mengoptimasi CortexFlow Lite agar mengalahkan Brain-Diffuser
pada Miyawaki dataset.

Target: Beat Brain-Diffuser MSE 0.008330

Strategy:
1. CortexFlow Lite Enhanced - Brain-Diffuser inspired improvements
2. CortexFlow Lite Ultra - Maximum optimization
3. Aggressive hyperparameter tuning
4. Advanced training techniques
5. Ensemble of best Lite variants

Models to test:
- CortexFlow-Lite (Original)
- CortexFlow-Lite-Enhanced 
- CortexFlow-Lite-Ultra
- Brain-Diffuser (Baseline to beat)
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
from src.models import (
    StandardBaselineCNN, OptimizedBrainDiffuser,
    CortexFlowLiteEnhanced, CortexFlowLiteUltra
)
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

def get_aggressive_config():
    """Get aggressive configuration for CortexFlow Lite optimization"""
    return {
        # Aggressive hyperparameters for Lite optimization
        'lr': 0.0008,        # Slightly lower for stability
        'batch_size': 32,    # Smaller batch for better gradients
        'weight_decay': 5e-7, # Even lower weight decay
        'epochs': 300,       # More epochs for convergence
        'patience': 40,      # More patience
        'gradient_clip': 0.3, # Tighter gradient clipping
        
        # Advanced training techniques
        'warmup_epochs': 10,
        'cosine_annealing': True,
        'label_smoothing': 0.1,
        'mixup_alpha': 0.2
    }

def get_ultra_config():
    """Get ultra-aggressive configuration for maximum optimization"""
    return {
        'lr': 0.0005,        # Even lower for ultra model
        'batch_size': 16,    # Very small batch
        'weight_decay': 1e-7, # Minimal weight decay
        'epochs': 400,       # Maximum epochs
        'patience': 50,      # Maximum patience
        'gradient_clip': 0.2, # Very tight clipping
        
        # Ultra techniques
        'warmup_epochs': 20,
        'cosine_annealing': True,
        'label_smoothing': 0.05,
        'mixup_alpha': 0.1,
        'cutmix_alpha': 0.1
    }

def mixup_data(x, y, alpha=1.0):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_cortexflow_lite_aggressive(model, train_loader, val_loader, config, device):
    """Aggressive training for CortexFlow Lite optimization"""
    
    # Advanced optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Advanced scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return epoch / config['warmup_epochs']
        elif config.get('cosine_annealing', False):
            return 0.5 * (1 + np.cos(np.pi * (epoch - config['warmup_epochs']) / 
                                    (config['epochs'] - config['warmup_epochs'])))
        else:
            return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Advanced loss with label smoothing
    if config.get('label_smoothing', 0) > 0:
        criterion = nn.SmoothL1Loss()  # More robust than MSE
    else:
        criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    print(f"🚀 Aggressive Training {model.name}")
    print(f"📊 Config: lr={config['lr']}, batch_size={config['batch_size']}, epochs={config['epochs']}")
    
    for epoch in range(config['epochs']):
        # Training phase with advanced techniques
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Apply mixup augmentation
            if config.get('mixup_alpha', 0) > 0 and np.random.rand() < 0.5:
                mixed_data, target_a, target_b, lam = mixup_data(data, target, config['mixup_alpha'])
                
                optimizer.zero_grad()
                output = model(mixed_data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
            
            loss.backward()
            
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
            
            optimizer.step()
            train_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
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
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return {
        'best_val_loss': best_val_loss,
        'training_history': training_history,
        'epochs_trained': epoch + 1
    }

def evaluate_model(model, test_loader, device):
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
    
    print(f"📊 {model.name}:")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   PSNR: {metrics['psnr']:.2f} dB")
    print(f"   SSIM: {metrics['ssim']:.4f}")
    print(f"   LPIPS: {metrics['lpips']:.4f}")
    
    return metrics

def main():
    """Main CortexFlow Lite optimization"""
    print("🎯 CortexFlow Lite Optimization: Beat Brain-Diffuser Challenge")
    print("=" * 65)
    print("🏆 Target: Beat Brain-Diffuser MSE 0.008330 on Miyawaki")
    print()
    
    device = setup_device()
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load Miyawaki dataset
    print("📁 Loading Miyawaki dataset")
    X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized('miyawaki', device)
    
    if X_train is None:
        print("❌ Failed to load Miyawaki dataset")
        return
    
    # Split for validation
    train_size = int(0.8 * len(X_train))
    X_train_split = X_train[:train_size]
    y_train_split = y_train[:train_size]
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    
    print(f"✅ Dataset loaded: Train={len(X_train_split)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Models to optimize
    models_to_test = [
        ('CortexFlow-Lite-Original', StandardBaselineCNN, get_aggressive_config()),
        ('CortexFlow-Lite-Enhanced', CortexFlowLiteEnhanced, get_aggressive_config()),
        ('CortexFlow-Lite-Ultra', CortexFlowLiteUltra, get_ultra_config()),
        ('Brain-Diffuser-Baseline', OptimizedBrainDiffuser, get_aggressive_config())
    ]
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, model_class, config in models_to_test:
        print(f"\n🔧 Optimizing {model_name}")
        
        try:
            # Create data loaders
            train_dataset = TensorDataset(X_train_split, y_train_split)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
            
            # Initialize model
            model = model_class(input_dim, device)
            
            # Train model
            training_result = train_cortexflow_lite_aggressive(
                model, train_loader, val_loader, config, device
            )
            
            # Evaluate model
            test_metrics = evaluate_model(model, test_loader, device)
            
            # Store results
            results[model_name] = {
                'training': training_result,
                'metrics': test_metrics,
                'config': config
            }
            
            print(f"✅ {model_name} completed: MSE={test_metrics['mse']:.6f}")
            
            # Memory cleanup
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error optimizing {model_name}: {e}")
            continue
    
    # Save results
    results_dir = f"results/cortexflow_lite_optimization_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/optimization_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Analysis
    print("\n🎉 CortexFlow Lite Optimization Complete!")
    print("=" * 50)
    
    brain_diffuser_target = 0.008330
    print(f"\n📊 RESULTS vs Brain-Diffuser Target ({brain_diffuser_target:.6f}):")
    
    winners = []
    for model_name, result in results.items():
        mse = result['metrics']['mse']
        psnr = result['metrics']['psnr']
        
        if mse < brain_diffuser_target:
            improvement = ((brain_diffuser_target - mse) / brain_diffuser_target) * 100
            status = f"🏆 BEATS Brain-Diffuser by {improvement:.2f}%"
            winners.append((model_name, mse, improvement))
        else:
            gap = ((mse - brain_diffuser_target) / brain_diffuser_target) * 100
            status = f"📈 Gap: +{gap:.2f}%"
        
        print(f"   {model_name}: MSE={mse:.6f}, PSNR={psnr:.2f}dB - {status}")
    
    if winners:
        print(f"\n🎉 SUCCESS! {len(winners)} model(s) beat Brain-Diffuser!")
        best_winner = min(winners, key=lambda x: x[1])
        print(f"🥇 Best: {best_winner[0]} with {best_winner[2]:.2f}% improvement")
    else:
        print(f"\n📈 No models beat Brain-Diffuser yet. Continue optimization needed.")
    
    print(f"\n💾 Results saved to: {results_dir}/")
    print("🚀 CortexFlow Lite Optimization Complete!")

if __name__ == "__main__":
    main()
