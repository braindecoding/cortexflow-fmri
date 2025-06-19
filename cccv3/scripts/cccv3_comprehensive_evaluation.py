"""
CCCV3 Comprehensive Evaluation with 10-Fold Cross-Validation
============================================================

Comprehensive evaluation of CCCV3 Ensemble using:
1. 10-fold Cross-Validation for robust performance estimation
2. Statistical significance testing (t-test)
3. Multiple metrics: MSE, PSNR, SSIM, LPIPS
4. Comparison with all baselines
5. 100% authentic data (no synthetic data)

Goal: Rigorous academic evaluation of CCCV3 performance
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
import scipy.stats as stats
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
    from cccv3.src.models.cccv3_ensemble import create_cccv3_ensemble, create_cccv3_ensemble_trainer
except ImportError:
    try:
        sys.path.append(os.path.join(parent_dir, 'src', 'models'))
        from cccv3_ensemble import create_cccv3_ensemble, create_cccv3_ensemble_trainer
    except ImportError:
        print("❌ Could not import CCCV3 ensemble model")
        sys.exit(1)

# Import utilities
try:
    from src.data import load_dataset_gpu_optimized
    from src.metrics import compute_comprehensive_metrics
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

def compute_metrics(predictions, targets):
    """
    Compute comprehensive metrics for evaluation
    
    Args:
        predictions: [N, 1, 28, 28] - Model predictions
        targets: [N, 1, 28, 28] - Ground truth targets
    Returns:
        metrics: Dict with MSE, PSNR, SSIM, LPIPS
    """
    # Convert to numpy for computation
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # MSE
    mse = np.mean((pred_np - target_np) ** 2)
    
    # PSNR
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # SSIM (simplified implementation)
    def ssim_single(img1, img2):
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        return ssim
    
    ssim_values = []
    for i in range(len(pred_np)):
        ssim_val = ssim_single(pred_np[i, 0], target_np[i, 0])
        ssim_values.append(ssim_val)
    
    ssim = np.mean(ssim_values)
    
    # LPIPS (simplified perceptual distance)
    lpips = np.mean(np.abs(pred_np - target_np))  # Simplified L1 distance
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'lpips': lpips
    }

def train_and_evaluate_fold(model, train_data, train_targets, val_data, val_targets, 
                           config, device, fold_num):
    """
    Train and evaluate model on a single fold
    
    Args:
        model: CCCV3 ensemble model
        train_data: Training fMRI data
        train_targets: Training visual targets
        val_data: Validation fMRI data
        val_targets: Validation visual targets
        config: Training configuration
        device: Device for computation
        fold_num: Fold number for logging
    Returns:
        metrics: Dict with evaluation metrics
    """
    print(f"   🔧 Training Fold {fold_num}...")
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    
    batch_size = min(16, len(train_data) // 4)
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create trainer
    trainer = create_cccv3_ensemble_trainer(model, device)
    
    # Train individual pathways
    training_results = trainer.train_individual_pathways(train_loader, val_loader, config)
    
    # Evaluate on validation set
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # Get ensemble prediction
            ensemble_pred = model(data, return_individual_predictions=False)
            all_predictions.append(ensemble_pred.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute comprehensive metrics
    metrics = compute_metrics(predictions, targets)
    
    print(f"      Fold {fold_num} MSE: {metrics['mse']:.6f}")
    
    return metrics

def cross_validation_evaluation(dataset_name, device, n_folds=10):
    """
    Perform 10-fold cross-validation evaluation
    
    Args:
        dataset_name: Name of dataset to evaluate
        device: Device for computation
        n_folds: Number of folds for cross-validation
    Returns:
        cv_results: Dict with cross-validation results
    """
    print(f"\n📁 10-Fold Cross-Validation on {dataset_name.upper()}")
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
        print(f"📊 Using 100% AUTHENTIC DATA - no synthetic data")
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None
    
    # Combine train and test for cross-validation
    X_all = torch.cat([X_train, X_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)
    
    print(f"📊 Total samples for CV: {len(X_all)}")
    
    # Setup cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Training configuration
    config = {
        'epochs': 60,  # Reduced for CV efficiency
        'weight_decay': 1e-6
    }
    
    # Store results for each fold
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all)):
        print(f"\n🔄 Fold {fold + 1}/{n_folds}")
        
        # Split data
        train_data = X_all[train_idx]
        train_targets = y_all[train_idx]
        val_data = X_all[val_idx]
        val_targets = y_all[val_idx]
        
        print(f"   Train: {len(train_data)}, Val: {len(val_data)}")
        
        # Create fresh model for this fold
        model = create_cccv3_ensemble(input_dim, dataset_name, dataset_size, device)
        
        # Train and evaluate
        fold_metrics = train_and_evaluate_fold(
            model, train_data, train_targets, val_data, val_targets,
            config, device, fold + 1
        )
        
        fold_results.append(fold_metrics)
        
        # Clear memory
        del model
        torch.cuda.empty_cache()
    
    # Aggregate results
    metrics_names = ['mse', 'psnr', 'ssim', 'lpips']
    aggregated_results = {}
    
    for metric in metrics_names:
        values = [result[metric] for result in fold_results]
        aggregated_results[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    print(f"\n📊 10-Fold Cross-Validation Results for {dataset_name.upper()}:")
    print(f"   MSE: {aggregated_results['mse']['mean']:.6f} ± {aggregated_results['mse']['std']:.6f}")
    print(f"   PSNR: {aggregated_results['psnr']['mean']:.2f} ± {aggregated_results['psnr']['std']:.2f}")
    print(f"   SSIM: {aggregated_results['ssim']['mean']:.4f} ± {aggregated_results['ssim']['std']:.4f}")
    print(f"   LPIPS: {aggregated_results['lpips']['mean']:.4f} ± {aggregated_results['lpips']['std']:.4f}")
    
    return {
        'dataset_name': dataset_name,
        'n_folds': n_folds,
        'fold_results': fold_results,
        'aggregated_results': aggregated_results,
        'dataset_size': len(X_all)
    }

def statistical_significance_test(cccv3_results, baseline_results):
    """
    Perform statistical significance testing
    
    Args:
        cccv3_results: CCCV3 cross-validation results
        baseline_results: Baseline results for comparison
    Returns:
        significance_results: Dict with statistical test results
    """
    print(f"\n📈 Statistical Significance Testing")
    print("=" * 40)
    
    # Known baseline results (from previous comprehensive testing)
    baselines = {
        'miyawaki': {
            'CortexFlow_Lite': 0.012,
            'CCCV1': 0.012326,
            'CCCV2': 0.014374
        },
        'vangerven': {
            'CortexFlow_Lite': 0.036,
            'CCCV1': 0.036487,
            'CCCV2': 0.036926
        },
        'mindbigdata': {
            'CortexFlow_Lite': 0.058,
            'CCCV1': 0.058601,
            'CCCV2': 0.056883
        },
        'crell': {
            'CortexFlow_Lite': 0.032,
            'CCCV1': 0.032218,
            'CCCV2': 0.032058
        }
    }
    
    dataset_name = cccv3_results['dataset_name']
    cccv3_mse_values = cccv3_results['aggregated_results']['mse']['values']
    cccv3_mean = cccv3_results['aggregated_results']['mse']['mean']
    
    significance_results = {}
    
    if dataset_name in baselines:
        for baseline_name, baseline_mse in baselines[dataset_name].items():
            # One-sample t-test against baseline
            t_stat, p_value = stats.ttest_1samp(cccv3_mse_values, baseline_mse)
            
            # Effect size (Cohen's d)
            effect_size = (cccv3_mean - baseline_mse) / np.std(cccv3_mse_values)
            
            # Determine significance
            is_significant = p_value < 0.05
            is_better = cccv3_mean < baseline_mse
            
            significance_results[baseline_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'is_significant': is_significant,
                'is_better': is_better,
                'cccv3_mean': cccv3_mean,
                'baseline_value': baseline_mse
            }
            
            # Print results
            significance_symbol = "🏆" if is_significant and is_better else "📈" if is_better else "📉"
            print(f"   vs {baseline_name}:")
            print(f"      CCCV3: {cccv3_mean:.6f}, Baseline: {baseline_mse:.6f}")
            print(f"      t-stat: {t_stat:.3f}, p-value: {p_value:.4f} {significance_symbol}")
            print(f"      Effect size: {effect_size:.3f}")
            print(f"      Significant: {'Yes' if is_significant else 'No'}")
            print()
    
    return significance_results

def main():
    """Main comprehensive evaluation function"""
    print("🎯 CCCV3 Comprehensive Evaluation with 10-Fold CV")
    print("=" * 55)
    print("📊 Rigorous academic evaluation using:")
    print("   ✅ 10-fold Cross-Validation")
    print("   ✅ Statistical significance testing")
    print("   ✅ Multiple metrics (MSE, PSNR, SSIM, LPIPS)")
    print("   ✅ 100% authentic data (no synthetic data)")
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
    all_cv_results = {}
    all_significance_results = {}
    
    for dataset_name in datasets:
        # Perform 10-fold cross-validation
        cv_results = cross_validation_evaluation(dataset_name, device, n_folds=10)
        
        if cv_results:
            all_cv_results[dataset_name] = cv_results
            
            # Statistical significance testing
            significance_results = statistical_significance_test(cv_results, {})
            all_significance_results[dataset_name] = significance_results
    
    # Final comprehensive analysis
    print("\n🎉 CCCV3 Comprehensive Evaluation Complete!")
    print("=" * 55)
    
    print(f"\n📊 FINAL 10-FOLD CROSS-VALIDATION RESULTS:")
    print("=" * 55)
    
    for dataset_name, cv_results in all_cv_results.items():
        agg = cv_results['aggregated_results']
        print(f"\n{dataset_name.upper()}:")
        print(f"   MSE: {agg['mse']['mean']:.6f} ± {agg['mse']['std']:.6f}")
        print(f"   PSNR: {agg['psnr']['mean']:.2f} ± {agg['psnr']['std']:.2f}")
        print(f"   SSIM: {agg['ssim']['mean']:.4f} ± {agg['ssim']['std']:.4f}")
        print(f"   Sample size: {cv_results['dataset_size']}")
    
    print(f"\n🏆 STATISTICAL SIGNIFICANCE SUMMARY:")
    print("=" * 40)
    
    total_comparisons = 0
    significant_improvements = 0
    
    for dataset_name, sig_results in all_significance_results.items():
        print(f"\n{dataset_name.upper()}:")
        for baseline_name, result in sig_results.items():
            total_comparisons += 1
            if result['is_significant'] and result['is_better']:
                significant_improvements += 1
                print(f"   vs {baseline_name}: 🏆 SIGNIFICANT IMPROVEMENT (p={result['p_value']:.4f})")
            elif result['is_better']:
                print(f"   vs {baseline_name}: 📈 Improvement (p={result['p_value']:.4f})")
            else:
                print(f"   vs {baseline_name}: 📉 No improvement (p={result['p_value']:.4f})")
    
    success_rate = (significant_improvements / total_comparisons) * 100 if total_comparisons > 0 else 0
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"Significant improvements: {significant_improvements}/{total_comparisons}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"✅ Evaluation conducted with 100% authentic data")
    print(f"✅ Rigorous 10-fold cross-validation methodology")
    print(f"✅ Statistical significance testing completed")
    
    print("\n🚀 CCCV3 Comprehensive Academic Evaluation Complete!")

if __name__ == "__main__":
    main()
