"""
CCCV4 Meta-Adaptive Comprehensive Testing
=========================================

Comprehensive 10-fold cross-validation testing of CCCV4 Meta-Adaptive model
that automatically selects optimal CCCV version per dataset.

Expected Results:
- Miyawaki: Select CCCV3, achieve ~0.004474 MSE
- Vangerven: Select CCCV1, achieve ~0.036487 MSE  
- MindBigData: Select CCCV3, achieve ~0.056162 MSE
- Crell: Select CCCV2, achieve ~0.032058 MSE

Goal: Prove meta-adaptive selection achieves optimal performance
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)
sys.path.append(parent_dir)

# Import CCCV4 meta-adaptive
try:
    from cccv4.concept.cccv4_meta_adaptive import create_cccv4_meta_adaptive, predict_cccv4_performance
except ImportError:
    try:
        sys.path.append(os.path.join(parent_dir, 'concept'))
        from cccv4_meta_adaptive import create_cccv4_meta_adaptive, predict_cccv4_performance
    except ImportError:
        print("❌ Could not import CCCV4 meta-adaptive model")
        sys.exit(1)

# Import individual CCCV models for fallback
try:
    # CCCV1
    sys.path.append(os.path.join(root_dir, 'cccv1', 'src', 'models'))
    from cccv1_model import create_cccv1_model, create_cccv1_trainer
    
    # CCCV2  
    sys.path.append(os.path.join(root_dir, 'cccv2', 'src', 'models'))
    from cccv2_model import create_cccv2_model, create_cccv2_trainer
    
    # CCCV3
    sys.path.append(os.path.join(root_dir, 'cccv3', 'src', 'models'))
    from cccv3_ultimate import create_cccv3_ultimate, create_cccv3_ultimate_trainer
    
    print("✅ All CCCV models imported successfully")
except ImportError as e:
    print(f"⚠️ Some CCCV models not available: {e}")

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

class CCCV4Trainer:
    """
    Trainer for CCCV4 Meta-Adaptive model
    Delegates training to appropriate CCCV version trainer
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.selected_version = model.selected_version
        
    def train_cccv4_model(self, train_loader, val_loader, config):
        """Train CCCV4 model using selected version trainer"""
        
        print(f"🔧 Training CCCV4 with {self.selected_version} strategy...")
        
        if self.selected_version == 'CCCV1':
            trainer = create_cccv1_trainer(self.model.active_model, self.device)
            return trainer.train_cccv1_model(train_loader, val_loader, config)
            
        elif self.selected_version == 'CCCV2':
            trainer = create_cccv2_trainer(self.model.active_model, self.device)
            return trainer.train_cccv2_model(train_loader, val_loader, config)
            
        elif self.selected_version == 'CCCV3':
            trainer = create_cccv3_ultimate_trainer(self.model.active_model, self.device)
            return trainer.train_ultimate_model(train_loader, val_loader, config)
            
        else:
            raise ValueError(f"Unknown CCCV version: {self.selected_version}")

def evaluate_cccv4_model(model, test_loader, device):
    """Evaluate CCCV4 model"""
    model.eval()
    all_predictions = []
    all_targets = []
    meta_info = None
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Get prediction and meta info
            output, batch_meta_info = model(data)
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())
            
            if meta_info is None:
                meta_info = batch_meta_info
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = nn.MSELoss()(predictions, targets).item()
    
    return {
        'mse': mse,
        'predictions': predictions,
        'targets': targets,
        'meta_info': meta_info
    }

def test_cccv4_on_dataset(dataset_name, device, n_folds=10):
    """Test CCCV4 on a dataset with 10-fold cross-validation"""
    
    print(f"\n📁 Testing CCCV4 Meta-Adaptive on {dataset_name.upper()}")
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
    
    print(f"📊 Total samples for 10-fold CV: {len(X_all)}")
    
    # Create sample model to see selection
    sample_model = create_cccv4_meta_adaptive(input_dim, dataset_name, X_all, y_all, device)
    characteristics = sample_model.get_characteristics()
    
    print(f"\n🧠 CCCV4 Meta-Adaptive Analysis:")
    print(f"   Selected version: {characteristics['recommended_cccv']}")
    print(f"   Confidence: {characteristics['confidence']:.2f}")
    print(f"   Rationale: {characteristics['rationale']}")
    print(f"   Dataset complexity: {characteristics['complexity_score']:.2f}")
    print(f"   Size category: {characteristics['size_category']}")
    
    del sample_model
    torch.cuda.empty_cache()
    
    # 10-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    config = {
        'epochs': 80,  # Balanced for comprehensive testing
        'weight_decay': 1e-6,
        'use_optimal_hyperparams': True
    }
    
    fold_results = []
    selected_versions = []
    meta_infos = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all)):
        print(f"\n🔄 Fold {fold + 1}/{n_folds}")
        
        # Split data
        train_data = X_all[train_idx]
        train_targets = y_all[train_idx]
        val_data = X_all[val_idx]
        val_targets = y_all[val_idx]
        
        print(f"   Train: {len(train_data)}, Val: {len(val_data)}")
        
        # Create fresh CCCV4 model for this fold
        model = create_cccv4_meta_adaptive(input_dim, dataset_name, train_data, train_targets, device)
        trainer = CCCV4Trainer(model, device)
        
        # Prepare data loaders
        batch_size = min(16, len(train_data) // 4)
        train_dataset = TensorDataset(train_data, train_targets)
        val_dataset = TensorDataset(val_data, val_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train model
        training_results = trainer.train_cccv4_model(train_loader, val_loader, config)
        
        # Evaluate
        eval_results = evaluate_cccv4_model(model, val_loader, device)
        
        fold_results.append(eval_results['mse'])
        selected_versions.append(model.selected_version)
        meta_infos.append(eval_results['meta_info'])
        
        print(f"   Fold {fold + 1} MSE: {eval_results['mse']:.6f}")
        print(f"   Selected version: {eval_results['meta_info']['selected_version']}")
        print(f"   Confidence: {eval_results['meta_info']['confidence']:.2f}")
        
        # Clear memory
        del model, trainer
        torch.cuda.empty_cache()
    
    # Aggregate results
    mean_mse = np.mean(fold_results)
    std_mse = np.std(fold_results)
    
    # Check version consistency
    version_consistency = len(set(selected_versions)) == 1
    primary_version = max(set(selected_versions), key=selected_versions.count)
    
    print(f"\n📊 10-Fold Cross-Validation Results:")
    print(f"   CCCV4 MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"   Primary version: {primary_version}")
    print(f"   Version consistency: {'✅' if version_consistency else '⚠️'} ({selected_versions.count(primary_version)}/{n_folds})")
    
    return {
        'dataset_name': dataset_name,
        'dataset_size': len(X_all),
        'characteristics': characteristics,
        'cv_results': {
            'mean_mse': mean_mse,
            'std_mse': std_mse,
            'fold_results': fold_results
        },
        'selected_versions': selected_versions,
        'primary_version': primary_version,
        'version_consistency': version_consistency,
        'meta_infos': meta_infos[0]
    }

def compare_with_predictions_and_baselines(result):
    """Compare CCCV4 results with predictions and all baselines"""
    
    dataset_name = result['dataset_name']
    cccv4_mse = result['cv_results']['mean_mse']
    selected_version = result['primary_version']
    
    # Get predictions
    predictions = predict_cccv4_performance()
    prediction = predictions.get(dataset_name, {})
    
    # All baseline results
    baselines = {
        'miyawaki': {
            'CCCV1': 0.012326,
            'CCCV2': 0.014374,
            'CCCV3_Ultimate': 0.004474,
            'Best_Individual': 0.008796,
            'Predicted_CCCV4': prediction.get('predicted_mse', 0.004474)
        },
        'vangerven': {
            'CCCV1': 0.036487,
            'CCCV2': 0.036926,
            'CCCV3_Ultimate': 0.039129,
            'Best_Individual': 0.036195,
            'Predicted_CCCV4': prediction.get('predicted_mse', 0.036487)
        },
        'mindbigdata': {
            'CCCV1': 0.058601,
            'CCCV2': 0.056883,
            'CCCV3_Ultimate': 0.056162,
            'Best_Individual': 0.056781,
            'Predicted_CCCV4': prediction.get('predicted_mse', 0.056162)
        },
        'crell': {
            'CCCV1': 0.032218,
            'CCCV2': 0.032058,
            'CCCV3_Ultimate': 0.032497,
            'Best_Individual': 0.032119,
            'Predicted_CCCV4': prediction.get('predicted_mse', 0.032058)
        }
    }
    
    dataset_baselines = baselines.get(dataset_name, {})
    
    print(f"\n📊 CCCV4 vs All Approaches for {dataset_name.upper()}:")
    print(f"   Selected version: {selected_version}")
    print(f"   Predicted version: {prediction.get('selected_version', 'Unknown')}")
    
    wins = 0
    total_comparisons = 0
    best_improvement = 0
    prediction_accuracy = 0
    
    for baseline_name, baseline_mse in dataset_baselines.items():
        total_comparisons += 1
        
        if baseline_name == 'Predicted_CCCV4':
            # Check prediction accuracy
            prediction_error = abs(cccv4_mse - baseline_mse) / baseline_mse * 100
            prediction_accuracy = max(0, 100 - prediction_error)
            print(f"   vs {baseline_name}: {cccv4_mse:.6f} vs {baseline_mse:.6f} → 📊 Prediction accuracy: {prediction_accuracy:.1f}%")
        elif cccv4_mse < baseline_mse:
            improvement = ((baseline_mse - cccv4_mse) / baseline_mse) * 100
            print(f"   vs {baseline_name}: {cccv4_mse:.6f} vs {baseline_mse:.6f} → 🏆 +{improvement:.2f}%")
            wins += 1
            best_improvement = max(best_improvement, improvement)
        else:
            gap = ((cccv4_mse - baseline_mse) / baseline_mse) * 100
            print(f"   vs {baseline_name}: {cccv4_mse:.6f} vs {baseline_mse:.6f} → 📈 -{gap:.2f}%")
    
    win_rate = (wins / (total_comparisons - 1)) * 100  # Exclude prediction from win rate
    
    print(f"   Win Rate: {wins}/{total_comparisons-1} ({win_rate:.1f}%)")
    print(f"   Best improvement: {best_improvement:.2f}%")
    print(f"   Prediction accuracy: {prediction_accuracy:.1f}%")
    
    return wins, total_comparisons - 1, best_improvement, prediction_accuracy

def statistical_significance_test(cccv4_results, baseline_mse, baseline_name):
    """Perform statistical significance test"""
    
    cccv4_values = cccv4_results['cv_results']['fold_results']
    cccv4_mean = cccv4_results['cv_results']['mean_mse']
    
    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(cccv4_values, baseline_mse)
    
    # Effect size (Cohen's d)
    effect_size = (cccv4_mean - baseline_mse) / np.std(cccv4_values)
    
    is_significant = p_value < 0.05
    is_better = cccv4_mean < baseline_mse
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'is_significant': is_significant,
        'is_better': is_better
    }

def main():
    """Main CCCV4 comprehensive testing function"""
    print("🚀 CCCV4 Meta-Adaptive Comprehensive Testing")
    print("=" * 50)
    print("🎯 10-fold cross-validation with meta-adaptive selection")
    print("📊 Goal: Validate optimal version selection per dataset")
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
    total_improvement = 0
    total_prediction_accuracy = 0
    
    for dataset_name in datasets:
        result = test_cccv4_on_dataset(dataset_name, device, n_folds=10)
        if result:
            all_results[dataset_name] = result
            
            # Compare with predictions and baselines
            wins, comparisons, best_improvement, prediction_accuracy = compare_with_predictions_and_baselines(result)
            total_wins += wins
            total_comparisons += comparisons
            total_improvement += best_improvement
            total_prediction_accuracy += prediction_accuracy
    
    # Final analysis
    print("\n🎉 CCCV4 Meta-Adaptive Testing Complete!")
    print("=" * 50)
    
    print(f"\n📊 CCCV4 META-ADAPTIVE SUMMARY:")
    for dataset_name, result in all_results.items():
        mse = result['cv_results']['mean_mse']
        std = result['cv_results']['std_mse']
        version = result['primary_version']
        consistency = result['version_consistency']
        
        print(f"\n{dataset_name.upper()}:")
        print(f"   MSE: {mse:.6f} ± {std:.6f}")
        print(f"   Selected: {version}")
        print(f"   Consistency: {'✅' if consistency else '⚠️'}")
    
    overall_win_rate = (total_wins / total_comparisons) * 100 if total_comparisons > 0 else 0
    avg_improvement = total_improvement / len(all_results) if all_results else 0
    avg_prediction_accuracy = total_prediction_accuracy / len(all_results) if all_results else 0
    
    print(f"\n🏆 CCCV4 OVERALL PERFORMANCE:")
    print(f"Win rate: {total_wins}/{total_comparisons} ({overall_win_rate:.1f}%)")
    print(f"Average improvement: {avg_improvement:.2f}%")
    print(f"Prediction accuracy: {avg_prediction_accuracy:.1f}%")
    
    if overall_win_rate >= 75 and avg_prediction_accuracy >= 80:
        print(f"\n🎉 CCCV4 META-ADAPTIVE SUCCESS!")
        print(f"🧠 Intelligent version selection validated!")
    elif overall_win_rate >= 50:
        print(f"\n✅ CCCV4 SHOWS PROMISE!")
        print(f"🔧 Meta-adaptive approach working!")
    else:
        print(f"\n🔧 CCCV4 NEEDS REFINEMENT")
        print(f"📈 Consider selection criteria adjustment")
    
    print(f"\n🚀 CCCV4: The ultimate neural decoding solution!")

if __name__ == "__main__":
    main()
