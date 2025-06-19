"""
CCCV3 Ensemble Testing
=====================

Test the CCCV3 Ensemble model that combines individual pathway predictions
using optimal weights based on individual pathway performance.

Strategy:
1. Train individual pathways separately to optimal performance
2. Combine predictions using dataset-specific optimal weights
3. Compare ensemble vs best individual pathway

Goal: Achieve 100% success rate by leveraging best of each pathway
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)
sys.path.append(parent_dir)

# Import CCCV3 ensemble model
try:
    from cccv3.src.models.cccv3_ensemble import (
        create_cccv3_ensemble, 
        create_cccv3_ensemble_trainer, 
        get_cccv3_ensemble_config
    )
except ImportError:
    try:
        sys.path.append(os.path.join(parent_dir, 'src', 'models'))
        from cccv3_ensemble import (
            create_cccv3_ensemble, 
            create_cccv3_ensemble_trainer, 
            get_cccv3_ensemble_config
        )
    except ImportError:
        print("❌ Could not import CCCV3 ensemble model")
        sys.exit(1)

# Import utilities
try:
    from src.data import load_dataset_gpu_optimized
except ImportError:
    print("⚠️ Parent directory imports not available")

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

def evaluate_ensemble_model(model, test_loader, device):
    """Evaluate ensemble model with individual pathway analysis"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Get ensemble prediction
            ensemble_pred = model(data, return_individual_predictions=False)
            all_predictions.append(ensemble_pred.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    ensemble_mse = nn.MSELoss()(predictions, targets).item()
    
    # Get individual pathway performances
    individual_performances = model.get_individual_performances(test_loader, device)
    
    return {
        'ensemble_mse': ensemble_mse,
        'individual_performances': individual_performances,
        'predictions': predictions,
        'targets': targets
    }

def test_cccv3_ensemble_on_dataset(dataset_name, device):
    """Test CCCV3 Ensemble on a dataset"""
    
    print(f"\n📁 Testing CCCV3 Ensemble on {dataset_name.upper()}")
    print("=" * 60)
    
    # Load dataset
    try:
        if 'load_dataset_gpu_optimized' in globals():
            X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
        else:
            print("❌ Dataset loading function not available")
            return None
        
        if X_train is None:
            print(f"❌ Failed to load {dataset_name}")
            return None
        
        dataset_size = len(X_train)
        print(f"✅ Dataset loaded: Train={dataset_size}, Test={len(X_test)}, Input_dim={input_dim}")
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None
    
    # Get ensemble configuration
    config = get_cccv3_ensemble_config(dataset_name)
    print(f"📋 Ensemble Config: {config['ensemble_strategy']}")
    print(f"🎯 Expected improvement: {config['expected_improvement']}")
    print(f"💡 Strategy: {config['rationale']}")
    
    # Create CCCV3 ensemble model
    model = create_cccv3_ensemble(input_dim, dataset_name, dataset_size, device)
    print(f"🏗️ Model: {model.model_name}")
    print(f"📊 Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = create_cccv3_ensemble_trainer(model, device)
    
    # Prepare data loaders
    train_size = int(0.8 * len(X_train))
    X_train_split = X_train[:train_size]
    y_train_split = y_train[:train_size]
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    
    batch_size = min(16, len(X_train_split) // 4)
    
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Training configuration
    training_config = {
        'epochs': config.get('training_epochs', 100),
        'weight_decay': 1e-6
    }
    
    print(f"\n🔧 Training Individual Pathways...")
    
    # Train individual pathways
    training_results = trainer.train_individual_pathways(train_loader, val_loader, training_config)
    
    print(f"\n📊 Individual Pathway Training Results:")
    for pathway_name, result in training_results.items():
        print(f"   {pathway_name.upper()}: Val Loss = {result['best_val_loss']:.6f}")
    
    # Evaluate ensemble
    print(f"\n🎯 Evaluating CCCV3 Ensemble...")
    test_results = evaluate_ensemble_model(model, test_loader, device)
    
    print(f"\n🎯 CCCV3 Ensemble Results on {dataset_name.upper()}:")
    print(f"   Ensemble MSE: {test_results['ensemble_mse']:.6f}")
    print(f"\n📊 Individual Pathway Test Performance:")
    perf = test_results['individual_performances']
    print(f"   Lite MSE: {perf['lite_mse']:.6f}")
    print(f"   CLIP MSE: {perf['clip_mse']:.6f}")
    print(f"   Attention MSE: {perf['attention_mse']:.6f}")
    print(f"   Best Individual: {perf['best_individual']:.6f}")
    
    # Compare ensemble vs best individual
    ensemble_mse = test_results['ensemble_mse']
    best_individual_mse = perf['best_individual']
    
    if ensemble_mse < best_individual_mse:
        improvement = ((best_individual_mse - ensemble_mse) / best_individual_mse) * 100
        print(f"\n🏆 ENSEMBLE WINS by {improvement:.2f}%!")
        winner = 'ensemble'
    else:
        gap = ((ensemble_mse - best_individual_mse) / best_individual_mse) * 100
        print(f"\n📈 Best Individual wins by {gap:.2f}%")
        winner = 'individual'
    
    return {
        'dataset_name': dataset_name,
        'dataset_size': dataset_size,
        'config': config,
        'training_results': training_results,
        'test_results': test_results,
        'ensemble_mse': ensemble_mse,
        'best_individual_mse': best_individual_mse,
        'winner': winner,
        'model': model
    }

def compare_with_baselines(result):
    """Compare ensemble results with known baselines"""
    
    dataset_name = result['dataset_name']
    ensemble_mse = result['ensemble_mse']
    
    # Known baseline results from previous testing
    baselines = {
        'miyawaki': {
            'CCCV1': 0.012326,
            'CCCV2': 0.014374,
            'Best_Individual': 0.008796  # Attention pathway
        },
        'vangerven': {
            'CCCV1': 0.036487,
            'CCCV2': 0.036926,
            'Best_Individual': 0.036195  # CLIP pathway
        },
        'mindbigdata': {
            'CCCV1': 0.058601,
            'CCCV2': 0.056883,
            'Best_Individual': 0.056781  # Attention pathway
        },
        'crell': {
            'CCCV1': 0.032218,
            'CCCV2': 0.032058,
            'Best_Individual': 0.032119  # Attention pathway
        }
    }
    
    dataset_baselines = baselines.get(dataset_name, {})
    
    print(f"\n📊 Baseline Comparison for {dataset_name.upper()}:")
    
    wins = 0
    total_comparisons = 0
    
    for baseline_name, baseline_mse in dataset_baselines.items():
        total_comparisons += 1
        
        if ensemble_mse < baseline_mse:
            improvement = ((baseline_mse - ensemble_mse) / baseline_mse) * 100
            print(f"   vs {baseline_name}: {ensemble_mse:.6f} vs {baseline_mse:.6f} → 🏆 +{improvement:.2f}%")
            wins += 1
        else:
            gap = ((ensemble_mse - baseline_mse) / baseline_mse) * 100
            print(f"   vs {baseline_name}: {ensemble_mse:.6f} vs {baseline_mse:.6f} → 📈 -{gap:.2f}%")
    
    win_rate = (wins / total_comparisons) * 100 if total_comparisons > 0 else 0
    print(f"   Win Rate: {wins}/{total_comparisons} ({win_rate:.1f}%)")
    
    return wins, total_comparisons

def main():
    """Main CCCV3 ensemble testing function"""
    print("🎯 CCCV3 Ensemble Testing")
    print("=" * 35)
    print("🔬 Testing ensemble approach for CCCV3")
    print("📊 Goal: 100% success rate by combining best pathways")
    print()
    
    device = setup_device()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Test datasets
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    # Results storage
    all_results = {}
    total_wins = 0
    total_comparisons = 0
    
    for dataset_name in datasets:
        result = test_cccv3_ensemble_on_dataset(dataset_name, device)
        if result:
            all_results[dataset_name] = result
            
            # Compare with baselines
            wins, comparisons = compare_with_baselines(result)
            total_wins += wins
            total_comparisons += comparisons
    
    # Final analysis
    print("\n🎉 CCCV3 Ensemble Testing Complete!")
    print("=" * 50)
    
    ensemble_wins = 0
    total_datasets = len(all_results)
    
    for dataset_name, result in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"   Ensemble MSE: {result['ensemble_mse']:.6f}")
        print(f"   Best Individual: {result['best_individual_mse']:.6f}")
        print(f"   Winner: {result['winner']}")
        
        if result['winner'] == 'ensemble':
            ensemble_wins += 1
    
    print(f"\n🏆 FINAL CCCV3 ENSEMBLE RESULTS:")
    print(f"Ensemble vs Individual: {ensemble_wins}/{total_datasets}")
    print(f"Ensemble success rate: {(ensemble_wins/total_datasets)*100:.1f}%")
    
    overall_win_rate = (total_wins / total_comparisons) * 100 if total_comparisons > 0 else 0
    print(f"Overall baseline win rate: {total_wins}/{total_comparisons} ({overall_win_rate:.1f}%)")
    
    if ensemble_wins == total_datasets:
        print(f"\n🎉 PERFECT ENSEMBLE SUCCESS!")
        print(f"🚀 CCCV3 Ensemble beats all individual pathways!")
    elif ensemble_wins >= total_datasets * 0.75:
        print(f"\n🎉 EXCELLENT ENSEMBLE SUCCESS!")
        print(f"🚀 CCCV3 Ensemble shows strong improvements!")
    elif ensemble_wins >= total_datasets * 0.5:
        print(f"\n✅ GOOD ENSEMBLE SUCCESS!")
        print(f"🔧 CCCV3 Ensemble shows promise!")
    else:
        print(f"\n🔧 ENSEMBLE NEEDS REFINEMENT")
        print(f"📈 Consider weight adjustments")
    
    print("\n🚀 CCCV3 Ensemble Development Complete!")

if __name__ == "__main__":
    main()
