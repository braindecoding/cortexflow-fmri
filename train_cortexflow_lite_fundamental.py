"""
CortexFlow Lite Fundamental Optimization: Pure Diffusion Approach
================================================================

Fundamental approach untuk mengalahkan Brain-Diffuser dengan mengadopsi
core principles yang membuat Brain-Diffuser successful.

Strategy:
1. Pure diffusion paradigm
2. Simplified but effective architectures
3. Focus on what works
4. Multiple fundamental variants

Target: Beat Brain-Diffuser MSE 0.008330 with fundamental approach
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
    OptimizedBrainDiffuser,
    CortexFlowLiteDiffusion, CortexFlowLiteMinimal, CortexFlowLiteOptimal,
    CortexFlowLiteDeep, CortexFlowLiteEnsemble
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

def get_fundamental_config():
    """Get fundamental configuration focused on what works"""
    return {
        # Proven optimal hyperparameters
        'lr': 0.001,         # Proven optimal from V5
        'batch_size': 64,    # Proven optimal for Miyawaki
        'weight_decay': 1e-06, # Proven optimal
        'epochs': 250,       # Sufficient for convergence
        'patience': 30,      # Good patience
        'gradient_clip': 0.5, # Standard clipping
        
        # Simple but effective
        'scheduler_factor': 0.5,
        'scheduler_patience': 15
    }

def train_fundamental_model(model, train_loader, val_loader, config, device):
    """Fundamental training focused on proven techniques"""
    
    # Proven optimizer configuration
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Proven scheduler
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
    training_history = []
    
    print(f"🚀 Fundamental Training {model.name}")
    print(f"📊 Proven Config: lr={config['lr']}, batch_size={config['batch_size']}")
    
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
            
            # Proven gradient clipping
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
        
        # Learning rate scheduling
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
    """Main fundamental optimization"""
    print("🎯 CortexFlow Lite Fundamental Optimization: Pure Diffusion Approach")
    print("=" * 70)
    print("🏆 Target: Beat Brain-Diffuser MSE 0.008330 with fundamental approach")
    print("🔬 Strategy: Adopt core principles that make Brain-Diffuser successful")
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
    
    # Get proven configuration
    config = get_fundamental_config()
    
    # Create data loaders with proven batch size
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Fundamental models to test
    models_to_test = [
        ('Brain-Diffuser-Target', OptimizedBrainDiffuser),
        ('CortexFlow-Lite-Diffusion', CortexFlowLiteDiffusion),
        ('CortexFlow-Lite-Minimal', CortexFlowLiteMinimal),
        ('CortexFlow-Lite-Optimal', CortexFlowLiteOptimal),
        ('CortexFlow-Lite-Deep', CortexFlowLiteDeep),
        ('CortexFlow-Lite-Ensemble', CortexFlowLiteEnsemble)
    ]
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, model_class in models_to_test:
        print(f"\n🔧 Training {model_name}")
        
        try:
            # Initialize model
            model = model_class(input_dim, device)
            
            # Train model with fundamental approach
            training_result = train_fundamental_model(
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
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    # Save results
    results_dir = f"results/cortexflow_lite_fundamental_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/fundamental_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Analysis
    print("\n🎉 CortexFlow Lite Fundamental Optimization Complete!")
    print("=" * 60)
    
    brain_diffuser_target = 0.008330
    print(f"\n📊 RESULTS vs Brain-Diffuser Target ({brain_diffuser_target:.6f}):")
    
    winners = []
    best_cortex_lite = None
    brain_diffuser_actual = None
    
    for model_name, result in results.items():
        mse = result['metrics']['mse']
        psnr = result['metrics']['psnr']
        
        if 'Brain-Diffuser' in model_name:
            brain_diffuser_actual = mse
            status = "🎯 Target Reference"
        elif mse < brain_diffuser_target:
            improvement = ((brain_diffuser_target - mse) / brain_diffuser_target) * 100
            status = f"🏆 BEATS Target by {improvement:.2f}%"
            winners.append((model_name, mse, improvement))
        else:
            gap = ((mse - brain_diffuser_target) / brain_diffuser_target) * 100
            status = f"📈 Gap: +{gap:.2f}%"
        
        # Track best CortexFlow Lite variant
        if 'CortexFlow-Lite' in model_name:
            if best_cortex_lite is None or mse < best_cortex_lite[1]:
                best_cortex_lite = (model_name, mse)
        
        print(f"   {model_name}: MSE={mse:.6f}, PSNR={psnr:.2f}dB - {status}")
    
    # Final analysis
    print(f"\n🔍 FUNDAMENTAL ANALYSIS:")
    
    if brain_diffuser_actual:
        print(f"🎯 Brain-Diffuser Actual: {brain_diffuser_actual:.6f}")
        
        if best_cortex_lite:
            if best_cortex_lite[1] < brain_diffuser_actual:
                improvement = ((brain_diffuser_actual - best_cortex_lite[1]) / brain_diffuser_actual) * 100
                print(f"🏆 SUCCESS! {best_cortex_lite[0]} beats Brain-Diffuser by {improvement:.2f}%")
            else:
                gap = ((best_cortex_lite[1] - brain_diffuser_actual) / brain_diffuser_actual) * 100
                print(f"📈 Best CortexFlow Lite ({best_cortex_lite[0]}) gap: +{gap:.2f}%")
    
    if winners:
        print(f"\n🎉 {len(winners)} model(s) beat the original target!")
        best_winner = min(winners, key=lambda x: x[1])
        print(f"🥇 Best Overall: {best_winner[0]} with {best_winner[2]:.2f}% improvement")
    
    print(f"\n💾 Results saved to: {results_dir}/")
    print("🚀 CortexFlow Lite Fundamental Optimization Complete!")

if __name__ == "__main__":
    main()
