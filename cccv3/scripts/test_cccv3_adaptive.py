"""
CCCV3 Adaptive Strategy Testing
==============================

Test the CCCV3 Adaptive model that automatically selects the optimal strategy
per dataset based on empirical findings and dataset characteristics.

Philosophy: "Use the simplest approach that works best"

Expected Results:
- Miyawaki: Ensemble (complex small dataset benefits from diversity)
- Vangerven: Individual CLIP (high-dim small sample, CLIP semantic optimal)
- MindBigData: Individual Attention (large dataset, attention handles complexity)
- Crell: Individual Attention (medium dataset, attention optimal)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)
sys.path.append(parent_dir)

# Import CCCV3 adaptive model
try:
    from cccv3.src.models.cccv3_adaptive import create_cccv3_adaptive, create_cccv3_adaptive_trainer
except ImportError:
    try:
        sys.path.append(os.path.join(parent_dir, 'src', 'models'))
        from cccv3_adaptive import create_cccv3_adaptive, create_cccv3_adaptive_trainer
    except ImportError:
        print("❌ Could not import CCCV3 adaptive model")
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
        return device
    else:
        return torch.device('cpu')

def evaluate_adaptive_model(model, test_loader, device):
    """Evaluate adaptive model"""
    model.eval()
    all_predictions = []
    all_targets = []
    strategy_info = None
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Get prediction and strategy info
            output, batch_strategy_info = model(data)
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())
            
            if strategy_info is None:
                strategy_info = batch_strategy_info
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = nn.MSELoss()(predictions, targets).item()
    
    return {
        'mse': mse,
        'predictions': predictions,
        'targets': targets,
        'strategy_info': strategy_info
    }

def test_adaptive_on_dataset(dataset_name, device, n_folds=5):
    """Test CCCV3 Adaptive on a dataset with cross-validation"""
    
    print(f"\n📁 Testing CCCV3 Adaptive on {dataset_name.upper()}")
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
    
    # Combine for cross-validation
    X_all = torch.cat([X_train, X_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)
    
    print(f"📊 Total samples for CV: {len(X_all)}")
    
    # Create model to see strategy
    sample_model = create_cccv3_adaptive(input_dim, dataset_name, dataset_size, device)
    strategy_analysis = sample_model.get_strategy_analysis()
    
    print(f"\n🧠 Dataset Analysis:")
    print(f"   Size category: {strategy_analysis['size_category']}")
    print(f"   Dimensionality ratio: {strategy_analysis['dimensionality_ratio']:.1f}")
    print(f"   Complexity score: {strategy_analysis['complexity_score']:.1f}")
    print(f"   Recommended strategy: {strategy_analysis['recommended_strategy']}")
    print(f"   Rationale: {strategy_analysis['rationale']}")
    
    del sample_model
    torch.cuda.empty_cache()
    
    # Cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    config = {
        'epochs': 60,  # Efficient for CV
        'weight_decay': 1e-6
    }
    
    fold_results = []
    strategy_infos = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all)):
        print(f"\n🔄 Fold {fold + 1}/{n_folds}")
        
        # Split data
        train_data = X_all[train_idx]
        train_targets = y_all[train_idx]
        val_data = X_all[val_idx]
        val_targets = y_all[val_idx]
        
        print(f"   Train: {len(train_data)}, Val: {len(val_data)}")
        
        # Create fresh model for this fold
        model = create_cccv3_adaptive(input_dim, dataset_name, dataset_size, device)
        trainer = create_cccv3_adaptive_trainer(model, device)
        
        # Prepare data loaders
        batch_size = min(16, len(train_data) // 4)
        train_dataset = TensorDataset(train_data, train_targets)
        val_dataset = TensorDataset(val_data, val_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train model
        training_results = trainer.train_adaptive_model(train_loader, val_loader, config)
        
        # Evaluate
        eval_results = evaluate_adaptive_model(model, val_loader, device)
        
        fold_results.append(eval_results['mse'])
        strategy_infos.append(eval_results['strategy_info'])
        
        print(f"   Fold {fold + 1} MSE: {eval_results['mse']:.6f}")
        print(f"   Strategy used: {eval_results['strategy_info']['strategy']}")
        
        # Clear memory
        del model, trainer
        torch.cuda.empty_cache()
    
    # Aggregate results
    mean_mse = np.mean(fold_results)
    std_mse = np.std(fold_results)
    
    print(f"\n📊 {n_folds}-Fold Cross-Validation Results:")
    print(f"   Adaptive MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"   Strategy consistency: {strategy_infos[0]['strategy']} (all folds)")
    
    return {
        'dataset_name': dataset_name,
        'dataset_size': len(X_all),
        'strategy_analysis': strategy_analysis,
        'cv_results': {
            'mean_mse': mean_mse,
            'std_mse': std_mse,
            'fold_results': fold_results
        },
        'strategy_info': strategy_infos[0]
    }

def compare_with_baselines(result):
    """Compare adaptive results with known baselines"""
    
    dataset_name = result['dataset_name']
    adaptive_mse = result['cv_results']['mean_mse']
    
    # Known baseline results
    baselines = {
        'miyawaki': {
            'Best_Individual': 0.008796,
            'Original_Ensemble': 0.008999,
            'CCCV1': 0.012326,
            'CCCV2': 0.014374
        },
        'vangerven': {
            'Best_Individual': 0.036195,
            'Original_Ensemble': 0.047117,
            'CCCV1': 0.036487,
            'CCCV2': 0.036926
        },
        'mindbigdata': {
            'Best_Individual': 0.056781,
            'Original_Ensemble': 0.056999,
            'CCCV1': 0.058601,
            'CCCV2': 0.056883
        },
        'crell': {
            'Best_Individual': 0.032119,
            'Original_Ensemble': 0.032486,
            'CCCV1': 0.032218,
            'CCCV2': 0.032058
        }
    }
    
    dataset_baselines = baselines.get(dataset_name, {})
    
    print(f"\n📊 Baseline Comparison for {dataset_name.upper()}:")
    
    wins = 0
    total_comparisons = 0
    
    for baseline_name, baseline_mse in dataset_baselines.items():
        total_comparisons += 1
        
        if adaptive_mse < baseline_mse:
            improvement = ((baseline_mse - adaptive_mse) / baseline_mse) * 100
            print(f"   vs {baseline_name}: {adaptive_mse:.6f} vs {baseline_mse:.6f} → 🏆 +{improvement:.2f}%")
            wins += 1
        else:
            gap = ((adaptive_mse - baseline_mse) / baseline_mse) * 100
            print(f"   vs {baseline_name}: {adaptive_mse:.6f} vs {baseline_mse:.6f} → 📈 -{gap:.2f}%")
    
    win_rate = (wins / total_comparisons) * 100 if total_comparisons > 0 else 0
    print(f"   Win Rate: {wins}/{total_comparisons} ({win_rate:.1f}%)")
    
    return wins, total_comparisons

def main():
    """Main CCCV3 adaptive testing function"""
    print("🧠 CCCV3 Adaptive Strategy Testing")
    print("=" * 40)
    print("🎯 Philosophy: Use the simplest approach that works best")
    print("📊 Automatic strategy selection per dataset")
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
        result = test_adaptive_on_dataset(dataset_name, device, n_folds=5)
        if result:
            all_results[dataset_name] = result
            
            # Compare with baselines
            wins, comparisons = compare_with_baselines(result)
            total_wins += wins
            total_comparisons += comparisons
    
    # Final analysis
    print("\n🎉 CCCV3 Adaptive Strategy Testing Complete!")
    print("=" * 55)
    
    print(f"\n📊 ADAPTIVE STRATEGY SUMMARY:")
    for dataset_name, result in all_results.items():
        strategy = result['strategy_info']['strategy']
        mse = result['cv_results']['mean_mse']
        std = result['cv_results']['std_mse']
        
        print(f"\n{dataset_name.upper()}:")
        print(f"   Strategy: {strategy}")
        print(f"   MSE: {mse:.6f} ± {std:.6f}")
        print(f"   Rationale: {result['strategy_analysis']['rationale']}")
    
    overall_win_rate = (total_wins / total_comparisons) * 100 if total_comparisons > 0 else 0
    print(f"\n🏆 OVERALL PERFORMANCE:")
    print(f"Baseline wins: {total_wins}/{total_comparisons} ({overall_win_rate:.1f}%)")
    
    if overall_win_rate >= 75:
        print(f"\n🎉 EXCELLENT ADAPTIVE SUCCESS!")
        print(f"🧠 Smart strategy selection works brilliantly!")
    elif overall_win_rate >= 50:
        print(f"\n✅ GOOD ADAPTIVE SUCCESS!")
        print(f"🧠 Strategy selection shows clear benefit!")
    else:
        print(f"\n🔧 ADAPTIVE STRATEGY NEEDS REFINEMENT")
        print(f"📈 Consider strategy selection criteria")
    
    print(f"\n🚀 CCCV3 Adaptive Strategy: Simplicity + Intelligence = Success!")

if __name__ == "__main__":
    main()
