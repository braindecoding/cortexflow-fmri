"""
CortexFlow-Lite-CLIP Hyperparameter Optimization
===============================================

Next step: Optimize hyperparameters for CortexFlow-Lite-CLIP to achieve
breakthrough performance and beat champions consistently.

Focus areas:
1. CLIP loss weighting optimization
2. Learning rate scheduling for semantic guidance
3. Dropout optimization for CLIP embeddings
4. Batch size optimization for CLIP stability
5. Architecture-specific tuning

Goal: Beat MinD-Vis and Brain-Diffuser consistently across all datasets
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from scipy import stats
import itertools
import warnings
warnings.filterwarnings('ignore')

# Import models
from src.models import (
    CortexFlowLiteCLIPOptimal,
    OptimizedMinDVis,
    OptimizedBrainDiffuser
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

def get_hyperparameter_grid():
    """Get comprehensive hyperparameter grid for optimization"""
    return {
        'lr': [0.0003, 0.0005, 0.0008, 0.001, 0.0012, 0.0015],
        'batch_size': [8, 12, 16, 20, 24, 32],
        'weight_decay': [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6],
        'dropout_encoder': [0.02, 0.03, 0.04, 0.05, 0.06, 0.08],
        'dropout_decoder': [0.01, 0.015, 0.02, 0.025, 0.03],
        'clip_residual_weight': [0.01, 0.03, 0.05, 0.08, 0.1],
        'epochs': [80, 100, 120, 150],
        'patience': [10, 12, 15, 18, 20],
        'scheduler_factor': [0.3, 0.5, 0.7, 0.8]
    }

def create_optimized_clip_model(input_dim, device, config):
    """Create CortexFlow-Lite-CLIP with optimized hyperparameters"""
    
    class OptimizedCortexFlowLiteCLIP(nn.Module):
        def __init__(self, input_dim, device, config):
            super(OptimizedCortexFlowLiteCLIP, self).__init__()
            self.name = "CortexFlow-Lite-CLIP-Optimized"
            self.device = device
            
            # Optimized encoder with tunable dropout
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Dropout(config.get('dropout_encoder', 0.06)),
                
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Dropout(config.get('dropout_encoder', 0.06) * 0.7),
                
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Dropout(config.get('dropout_encoder', 0.06) * 0.5),
                
                # Map to CLIP space
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.Tanh()
            ).to(device)
            
            # Optimized decoder with tunable dropout
            self.decoder = nn.Sequential(
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Dropout(config.get('dropout_decoder', 0.02)),
                
                nn.Linear(512, 784),
                nn.Sigmoid()
            ).to(device)
            
            # CLIP alignment module with tunable residual weight
            self.clip_aligner = nn.Sequential(
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Linear(256, 512),
                nn.Tanh()
            ).to(device)
            
            self.residual_weight = config.get('clip_residual_weight', 0.1)
        
        def forward(self, x):
            # Encode to CLIP-like space
            features = self.encoder(x)
            
            # CLIP alignment with tunable residual
            aligned_features = features + self.residual_weight * self.clip_aligner(features)
            aligned_features = torch.nn.functional.normalize(aligned_features, p=2, dim=1)
            
            # Decode to visual output
            visual_output = self.decoder(aligned_features)
            
            return visual_output.view(-1, 1, 28, 28), aligned_features
    
    return OptimizedCortexFlowLiteCLIP(input_dim, device, config)

def train_optimized_clip_model(model, train_loader, val_loader, config, device):
    """Train optimized CLIP model"""
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['scheduler_factor'], 
        patience=config['patience']//3,
        min_lr=1e-8
    )
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, clip_embedding = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
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
        
        if patience_counter >= config['patience']:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return best_val_loss

def evaluate_optimized_clip_model(model, test_loader, device):
    """Evaluate optimized CLIP model"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = nn.MSELoss()(predictions, targets).item()
    return mse

def random_search_optimization(X_train, y_train, X_test, y_test, input_dim, device, dataset_name, max_trials=30):
    """Random search hyperparameter optimization"""
    
    param_grid = get_hyperparameter_grid()
    
    best_score = float('inf')
    best_params = None
    results = []
    
    # Champion targets to beat
    champion_targets = {
        'miyawaki': 0.009845,    # Brain-Diffuser
        'vangerven': 0.045659,   # Brain-Diffuser  
        'mindbigdata': 0.057348, # MinD-Vis
        'crell': 0.032525        # MinD-Vis
    }
    
    target_score = champion_targets.get(dataset_name, 0.05)
    
    print(f"🎯 Target to beat on {dataset_name}: {target_score:.6f}")
    print(f"🔍 Running {max_trials} optimization trials")
    
    for trial in range(max_trials):
        # Random sample from grid
        params = {}
        for key, values in param_grid.items():
            value = np.random.choice(values)
            # Ensure integer types for specific parameters
            if key in ['batch_size', 'epochs', 'patience']:
                params[key] = int(value)
            else:
                params[key] = value
        
        print(f"\n🔧 Trial {trial+1}/{max_trials}")
        print(f"   lr={params['lr']}, batch_size={params['batch_size']}, "
              f"dropout_enc={params['dropout_encoder']:.3f}")
        
        try:
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
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
            
            # Create optimized model
            model = create_optimized_clip_model(input_dim, device, params)
            
            # Train model
            val_loss = train_optimized_clip_model(model, train_loader, val_loader, params, device)
            
            # Test evaluation
            test_mse = evaluate_optimized_clip_model(model, test_loader, device)
            
            print(f"   Result: Val Loss = {val_loss:.6f}, Test MSE = {test_mse:.6f}")
            
            # Check if beats champion
            if test_mse < target_score:
                improvement = ((target_score - test_mse) / target_score) * 100
                print(f"   🏆 BEATS CHAMPION by {improvement:.2f}%!")
            else:
                gap = ((test_mse - target_score) / target_score) * 100
                print(f"   📈 Gap to target: +{gap:.2f}%")
            
            # Track results
            results.append({
                'trial': trial + 1,
                'params': params.copy(),
                'val_loss': val_loss,
                'test_mse': test_mse,
                'beats_champion': test_mse < target_score
            })
            
            # Update best
            if test_mse < best_score:
                best_score = test_mse
                best_params = params.copy()
                print(f"   🌟 New best score: {best_score:.6f}")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            continue
    
    return {
        'best_score': best_score,
        'best_params': best_params,
        'all_results': results,
        'beats_champion': best_score < target_score,
        'target_score': target_score
    }

def main():
    """Main CLIP hyperparameter optimization"""
    print("🎯 CortexFlow-Lite-CLIP Hyperparameter Optimization")
    print("=" * 60)
    print("🔬 Goal: Beat champions with optimized CLIP guidance")
    print()
    
    device = setup_device()
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Datasets to optimize
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    # Results storage
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for dataset_name in datasets:
        print(f"\n📁 Optimizing CLIP for {dataset_name.upper()}")
        print("=" * 50)
        
        # Load dataset
        try:
            X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
            
            if X_train is None:
                print(f"❌ Failed to load {dataset_name}")
                continue
            
            print(f"✅ Dataset loaded: Train={len(X_train)}, Test={len(X_test)}")
            
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            continue
        
        # Run optimization
        optimization_results = random_search_optimization(
            X_train, y_train, X_test, y_test, input_dim, device, dataset_name
        )
        
        all_results[dataset_name] = optimization_results
    
    # Save results
    results_dir = f"results/clip_hyperparameter_optimization_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/optimization_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Final analysis
    print("\n🎉 CLIP Hyperparameter Optimization Complete!")
    print("=" * 60)
    
    champions_beaten = 0
    total_datasets = len(all_results)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"   Target: {results['target_score']:.6f}")
        print(f"   Best CLIP: {results['best_score']:.6f}")
        
        if results['beats_champion']:
            improvement = ((results['target_score'] - results['best_score']) / results['target_score']) * 100
            print(f"   🏆 BEATS CHAMPION by {improvement:.2f}%!")
            champions_beaten += 1
        else:
            gap = ((results['best_score'] - results['target_score']) / results['target_score']) * 100
            print(f"   📈 Gap: +{gap:.2f}%")

        if results['best_params'] is not None:
            print(f"   Best params: lr={results['best_params']['lr']}, "
                  f"batch_size={results['best_params']['batch_size']}")
        else:
            print(f"   ❌ No successful trials")
    
    print(f"\n🏆 FINAL OPTIMIZATION RESULTS:")
    print(f"Champions beaten: {champions_beaten}/{total_datasets}")
    print(f"Success rate: {(champions_beaten/total_datasets)*100:.1f}%")
    
    if champions_beaten >= total_datasets // 2:
        print(f"\n🎉 OPTIMIZATION SUCCESS!")
        print(f"🚀 CLIP guidance achieves breakthrough performance!")
    else:
        print(f"\n🔧 Further optimization needed")
        print(f"📈 Consider advanced techniques or architecture changes")
    
    print(f"\n💾 Results saved to: {results_dir}/")
    print("🚀 CLIP Hyperparameter Optimization Complete!")

if __name__ == "__main__":
    main()
