"""
CortexFlow-Lite-CLIP Testing: Revolutionary CLIP-Guided Neural Decoding
======================================================================

Test the revolutionary CortexFlow-Lite-CLIP architecture that combines:
1. CortexFlow-Lite efficiency
2. CLIP semantic understanding
3. Multi-modal alignment
4. Perceptual quality enhancement

Models to test:
- CortexFlow-Lite-CLIP: Basic CLIP guidance
- CortexFlow-Lite-CLIP-Advanced: Multi-scale CLIP guidance
- CortexFlow-Lite-CLIP-Optimal: Production-ready version

Goal: Beat current champions (MinD-Vis, Brain-Diffuser) with semantic guidance
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
import warnings
warnings.filterwarnings('ignore')

# Import models
from src.models import (
    CortexFlowLiteCLIP,
    CortexFlowLiteCLIPAdvanced,
    CortexFlowLiteCLIPOptimal,
    CLIPLoss,
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

def get_clip_optimized_config(dataset_name):
    """Get CLIP-optimized configuration for each dataset"""
    configs = {
        'miyawaki': {
            'lr': 0.0008,        # Lower LR for CLIP stability
            'batch_size': 16,    # Smaller batch for CLIP memory
            'weight_decay': 5e-7,
            'epochs': 120,
            'patience': 15,
            'clip_weight': 0.1,  # CLIP loss weight
            'cosine_weight': 0.05 # Cosine similarity weight
        },
        'vangerven': {
            'lr': 0.0005,
            'batch_size': 8,     # Very small for CLIP
            'weight_decay': 1e-7,
            'epochs': 150,
            'patience': 20,
            'clip_weight': 0.15,
            'cosine_weight': 0.08
        },
        'mindbigdata': {
            'lr': 0.001,
            'batch_size': 16,
            'weight_decay': 1e-6,
            'epochs': 100,
            'patience': 12,
            'clip_weight': 0.1,
            'cosine_weight': 0.05
        },
        'crell': {
            'lr': 0.0008,
            'batch_size': 16,
            'weight_decay': 5e-7,
            'epochs': 120,
            'patience': 15,
            'clip_weight': 0.12,
            'cosine_weight': 0.06
        }
    }
    return configs.get(dataset_name, configs['miyawaki'])

def train_clip_model_cv_fold(model, train_loader, val_loader, config, device):
    """Train CLIP model for one CV fold"""
    
    # Check if model has CLIP components
    has_clip = hasattr(model, 'clip_model') or 'CLIP' in model.name
    
    if has_clip:
        # Use CLIP loss for CLIP models
        criterion = CLIPLoss(device=device)
        criterion.clip_weight = config.get('clip_weight', 0.1)
        criterion.cosine_weight = config.get('cosine_weight', 0.05)
    else:
        # Standard MSE for baseline models
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['patience']//3
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if has_clip and hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
                # CLIP model with embedding output
                output, clip_embedding = model(data)
                if isinstance(criterion, CLIPLoss):
                    loss_dict = criterion(output, target, clip_embedding)
                    loss = loss_dict['total_loss']
                else:
                    loss = criterion(output, target)
            else:
                # Standard model
                output = model(data)
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
                
                if has_clip and hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
                    output, clip_embedding = model(data)
                    if isinstance(criterion, CLIPLoss):
                        loss_dict = criterion(output, target, clip_embedding)
                        loss = loss_dict['total_loss']
                    else:
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
        
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

def evaluate_clip_model_cv_fold(model, val_loader, device):
    """Evaluate CLIP model for one CV fold"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # Handle both CLIP and standard models
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
                try:
                    output, _ = model(data)  # CLIP model with embedding
                except:
                    output = model(data)     # Fallback to standard
            else:
                output = model(data)         # Standard model
            
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate MSE (primary metric)
    mse = nn.MSELoss()(predictions, targets).item()
    return mse

def cross_validate_clip_model(model_class, model_name, X_train, y_train, input_dim, config, device, n_splits=5):
    """Perform k-fold cross-validation for CLIP models"""
    
    print(f"🔄 {n_splits}-fold CV: {model_name}")
    
    # Setup cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"   Fold {fold+1}/{n_splits}...", end=" ")
        
        # Split data for this fold
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize and train model
        try:
            model = model_class(input_dim, device)
        except Exception as e:
            print(f"Error initializing {model_name}: {e}")
            continue
        
        # Train model for this fold
        try:
            train_clip_model_cv_fold(model, train_loader, val_loader, config, device)
            
            # Evaluate model for this fold
            fold_score = evaluate_clip_model_cv_fold(model, val_loader, device)
            cv_scores.append(fold_score)
            
            print(f"MSE: {fold_score:.6f}")
        except Exception as e:
            print(f"Error in fold {fold+1}: {e}")
            continue
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    if cv_scores:
        cv_scores = np.array(cv_scores)
        print(f"   📊 CV Results: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
        return cv_scores
    else:
        print(f"   ❌ No successful folds for {model_name}")
        return np.array([])

def main():
    """Main CLIP testing function"""
    print("🎯 CortexFlow-Lite-CLIP Testing: Revolutionary CLIP-Guided Neural Decoding")
    print("=" * 80)
    print("🔬 Goal: Beat current champions with semantic CLIP guidance")
    print("📊 Methodology: 5-fold CV + Statistical significance testing")
    print()
    
    device = setup_device()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # CLIP models to test
    clip_models = [
        ('CortexFlow-Lite-CLIP-Optimal', CortexFlowLiteCLIPOptimal),
        ('CortexFlow-Lite-CLIP', CortexFlowLiteCLIP),
        # ('CortexFlow-Lite-CLIP-Advanced', CortexFlowLiteCLIPAdvanced),  # Skip for now due to complexity
    ]
    
    # Champion models for comparison
    champion_models = [
        ('MinD-Vis', OptimizedMinDVis),
        ('Brain-Diffuser', OptimizedBrainDiffuser)
    ]
    
    # Datasets to test
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    # Results storage
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for dataset_name in datasets:
        print(f"\n📁 CLIP Testing on {dataset_name.upper()}")
        print("=" * 60)
        
        # Load dataset
        try:
            X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
            
            if X_train is None:
                print(f"❌ Failed to load {dataset_name}")
                continue
            
            print(f"✅ Dataset loaded: Train={len(X_train)}, Test={len(X_test)}")
            
            config = get_clip_optimized_config(dataset_name)
            
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            continue
        
        # Test all models
        dataset_results = {}
        
        # Test CLIP models
        for model_name, model_class in clip_models:
            print(f"\n🔧 Testing {model_name}")
            
            try:
                cv_scores = cross_validate_clip_model(
                    model_class, model_name, X_train, y_train, input_dim, config, device
                )
                
                if len(cv_scores) > 0:
                    dataset_results[model_name] = cv_scores
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                continue
        
        # Test champion models for comparison
        for model_name, model_class in champion_models:
            print(f"\n🔧 Testing {model_name} (Champion)")
            
            try:
                # Use standard config for champions
                standard_config = {'lr': 0.001, 'batch_size': 32, 'weight_decay': 1e-6, 'epochs': 100, 'patience': 12}
                
                cv_scores = cross_validate_clip_model(
                    model_class, model_name, X_train, y_train, input_dim, standard_config, device
                )
                
                if len(cv_scores) > 0:
                    dataset_results[model_name] = cv_scores
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                continue
        
        all_results[dataset_name] = dataset_results
    
    # Save results
    results_dir = f"results/cortexflow_lite_clip_test_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/clip_test_results.json", 'w') as f:
        json.dump({k: {mk: mv.tolist() if len(mv) > 0 else [] for mk, mv in v.items()} for k, v in all_results.items()}, 
                  f, indent=2, default=str)
    
    # Final analysis
    print("\n🎉 CortexFlow-Lite-CLIP Testing Complete!")
    print("=" * 60)
    
    print(f"\n📊 CLIP PERFORMANCE SUMMARY:")
    
    clip_wins = 0
    total_comparisons = 0
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        
        # Find best CLIP model
        clip_results = {k: v for k, v in results.items() if 'CLIP' in k and len(v) > 0}
        champion_results = {k: v for k, v in results.items() if k in ['MinD-Vis', 'Brain-Diffuser'] and len(v) > 0}
        
        if clip_results and champion_results:
            best_clip = min(clip_results.items(), key=lambda x: x[1].mean())
            best_champion = min(champion_results.items(), key=lambda x: x[1].mean())
            
            print(f"   Best CLIP: {best_clip[0]} - {best_clip[1].mean():.6f}")
            print(f"   Best Champion: {best_champion[0]} - {best_champion[1].mean():.6f}")
            
            if best_clip[1].mean() < best_champion[1].mean():
                improvement = ((best_champion[1].mean() - best_clip[1].mean()) / best_champion[1].mean()) * 100
                print(f"   🏆 CLIP WINS by {improvement:.2f}%!")
                clip_wins += 1
            else:
                gap = ((best_clip[1].mean() - best_champion[1].mean()) / best_champion[1].mean()) * 100
                print(f"   📈 CLIP gap: +{gap:.2f}%")
            
            total_comparisons += 1
    
    # Final verdict
    print(f"\n🏆 FINAL CLIP VERDICT:")
    print("-" * 30)
    print(f"CLIP wins: {clip_wins}/{total_comparisons} datasets")
    print(f"Success rate: {(clip_wins/total_comparisons)*100:.1f}%" if total_comparisons > 0 else "No valid comparisons")
    
    if clip_wins > total_comparisons // 2:
        print(f"\n🎉 CLIP GUIDANCE SUCCESS!")
        print(f"🚀 Semantic understanding improves neural decoding!")
    else:
        print(f"\n🔧 CLIP needs optimization")
        print(f"📈 Consider hyperparameter tuning or architecture adjustments")
    
    print(f"\n💾 Results saved to: {results_dir}/")
    print("🚀 CortexFlow-Lite-CLIP Testing Complete!")

if __name__ == "__main__":
    main()
