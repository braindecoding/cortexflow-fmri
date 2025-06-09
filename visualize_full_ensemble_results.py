#!/usr/bin/env python3
"""
Visualize Full Ensemble Reconstruction Results
Create comprehensive visualizations showing ensemble vs individual models
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import json

# Import the full ensemble model
from train_full_ensemble import FullCortexFlowEnsemble, load_dataset

def load_trained_ensemble(dataset_name: str, input_dim: int, device: torch.device):
    """Load trained ensemble model"""
    model = FullCortexFlowEnsemble(input_dim)
    
    # Try to load trained weights
    checkpoint_path = f'full_ensemble_{dataset_name}_best.pth'
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded trained ensemble for {dataset_name}")
    else:
        print(f"❌ No trained model found for {dataset_name}, using random weights")
    
    model = model.to(device)
    model.eval()
    return model

def create_comprehensive_visualization(dataset_name: str, device: torch.device):
    """Create comprehensive visualization for a dataset"""
    
    # Load dataset
    X, y = load_dataset(dataset_name)
    if X is None:
        print(f"❌ Failed to load {dataset_name}")
        return
    
    # Load trained ensemble
    model = load_trained_ensemble(dataset_name, X.shape[1], device)
    
    # Get test samples
    n_samples = min(10, X.shape[0])
    indices = torch.randperm(X.shape[0])[:n_samples]
    test_x = X[indices].to(device)
    test_y = y[indices]
    
    # Get ensemble outputs
    with torch.no_grad():
        ensemble_outputs = model(test_x, mode='adaptive')
    
    # Create visualization
    fig, axes = plt.subplots(4, n_samples, figsize=(2*n_samples, 8))
    fig.suptitle(f'CortexFlow Full Ensemble Results - {dataset_name.title()} Dataset', 
                fontsize=16, fontweight='bold')
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(4, 1)
    
    metrics_data = []
    
    for i in range(n_samples):
        # Original image (row 0)
        original = test_y[i].numpy().reshape(28, 28)
        axes[0, i].imshow(original, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {i+1}', fontsize=10, fontweight='bold')
        axes[0, i].axis('off')
        
        # Ensemble prediction (row 1)
        ensemble_pred = ensemble_outputs['ensemble_prediction'][i].cpu().numpy().reshape(28, 28)
        axes[1, i].imshow(ensemble_pred, cmap='gray', vmin=0, vmax=1)
        
        # Calculate metrics
        mse = np.mean((original - ensemble_pred) ** 2)
        ssim_score = ssim(original, ensemble_pred, data_range=1.0)
        uncertainty = ensemble_outputs['total_uncertainty'][i].cpu().item()
        
        axes[1, i].set_title(f'Ensemble {i+1}\nMSE: {mse:.4f}\nSSIM: {ssim_score:.3f}\nUnc: {uncertainty:.3f}', 
                           fontsize=9)
        axes[1, i].axis('off')
        
        # Best individual model prediction (row 2)
        individual_preds = ensemble_outputs['individual_predictions'][i].cpu()  # [5, 784]
        individual_mses = []
        
        for j in range(5):
            ind_pred = individual_preds[j].numpy().reshape(28, 28)
            ind_mse = np.mean((original - ind_pred) ** 2)
            individual_mses.append(ind_mse)
        
        best_idx = np.argmin(individual_mses)
        best_pred = individual_preds[best_idx].numpy().reshape(28, 28)
        best_mse = individual_mses[best_idx]
        best_ssim = ssim(original, best_pred, data_range=1.0)
        
        model_names = ['Simple', 'MC', 'Hierarchical', 'Enhanced', 'Unified']
        
        axes[2, i].imshow(best_pred, cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'Best Individual\n{model_names[best_idx]}\nMSE: {best_mse:.4f}\nSSIM: {best_ssim:.3f}', 
                           fontsize=9)
        axes[2, i].axis('off')
        
        # Ensemble weights (row 3)
        weights = ensemble_outputs['ensemble_weights'][i].cpu().numpy()
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        bars = axes[3, i].bar(range(5), weights, color=colors, alpha=0.7)
        axes[3, i].set_xticks(range(5))
        axes[3, i].set_xticklabels(model_names, rotation=45, fontsize=8)
        axes[3, i].set_title(f'Ensemble Weights {i+1}', fontsize=9)
        axes[3, i].set_ylim(0, 1)
        
        # Add weight values on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            if height > 0.05:  # Only show if weight is significant
                axes[3, i].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{weight:.2f}', ha='center', va='bottom', fontsize=7)
        
        # Store metrics (convert to Python native types)
        metrics_data.append({
            'sample': i+1,
            'ensemble_mse': float(mse),
            'ensemble_ssim': float(ssim_score),
            'ensemble_uncertainty': float(uncertainty),
            'best_individual_mse': float(best_mse),
            'best_individual_ssim': float(best_ssim),
            'best_individual_model': model_names[best_idx],
            'ensemble_weights': [float(w) for w in weights]
        })
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save visualization
    output_dir = Path("results/full_ensemble_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = output_dir / f"full_ensemble_{dataset_name}_comprehensive.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Visualization saved: {viz_path}")
    
    # Save metrics
    metrics_path = output_dir / f"full_ensemble_{dataset_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    return metrics_data, viz_path

def create_comparison_summary(all_metrics: dict):
    """Create summary comparison across all datasets"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CortexFlow Full Ensemble - Performance Summary', fontsize=16, fontweight='bold')
    
    datasets = list(all_metrics.keys())
    
    # Plot 1: MSE Comparison
    ensemble_mses = []
    individual_mses = []
    
    for dataset in datasets:
        metrics = all_metrics[dataset]
        ens_mse = np.mean([m['ensemble_mse'] for m in metrics])
        ind_mse = np.mean([m['best_individual_mse'] for m in metrics])
        ensemble_mses.append(ens_mse)
        individual_mses.append(ind_mse)
    
    x = np.arange(len(datasets))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, ensemble_mses, width, label='Ensemble', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, individual_mses, width, label='Best Individual', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Dataset')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('MSE Comparison: Ensemble vs Best Individual')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: SSIM Comparison
    ensemble_ssims = []
    individual_ssims = []
    
    for dataset in datasets:
        metrics = all_metrics[dataset]
        ens_ssim = np.mean([m['ensemble_ssim'] for m in metrics])
        ind_ssim = np.mean([m['best_individual_ssim'] for m in metrics])
        ensemble_ssims.append(ens_ssim)
        individual_ssims.append(ind_ssim)
    
    axes[0, 1].bar(x - width/2, ensemble_ssims, width, label='Ensemble', color='blue', alpha=0.7)
    axes[0, 1].bar(x + width/2, individual_ssims, width, label='Best Individual', color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Dataset')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].set_title('SSIM Comparison: Ensemble vs Best Individual')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(datasets)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average Ensemble Weights
    model_names = ['Simple', 'MC', 'Hierarchical', 'Enhanced', 'Unified']
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, dataset in enumerate(datasets):
        metrics = all_metrics[dataset]
        avg_weights = np.mean([m['ensemble_weights'] for m in metrics], axis=0)
        
        axes[1, 0].bar([j + i*0.15 for j in range(5)], avg_weights, 
                      width=0.15, label=dataset, alpha=0.7)
    
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Average Weight')
    axes[1, 0].set_title('Average Ensemble Weights by Dataset')
    axes[1, 0].set_xticks(range(5))
    axes[1, 0].set_xticklabels(model_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Uncertainty Distribution
    for dataset in datasets:
        metrics = all_metrics[dataset]
        uncertainties = [m['ensemble_uncertainty'] for m in metrics]
        axes[1, 1].hist(uncertainties, bins=10, alpha=0.6, label=dataset)
    
    axes[1, 1].set_xlabel('Uncertainty')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Uncertainty Distribution by Dataset')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save summary
    output_dir = Path("results/full_ensemble_visualizations")
    summary_path = output_dir / "full_ensemble_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Summary saved: {summary_path}")
    
    return summary_path

def print_detailed_analysis(all_metrics: dict):
    """Print detailed analysis of results"""
    
    print("\n" + "="*80)
    print("🔍 DETAILED FULL ENSEMBLE ANALYSIS")
    print("="*80)
    
    for dataset, metrics in all_metrics.items():
        print(f"\n📊 {dataset.upper()} DATASET:")
        print("-" * 50)
        
        # Calculate averages
        avg_ens_mse = np.mean([m['ensemble_mse'] for m in metrics])
        avg_ens_ssim = np.mean([m['ensemble_ssim'] for m in metrics])
        avg_ens_unc = np.mean([m['ensemble_uncertainty'] for m in metrics])
        avg_ind_mse = np.mean([m['best_individual_mse'] for m in metrics])
        avg_ind_ssim = np.mean([m['best_individual_ssim'] for m in metrics])
        
        print(f"Ensemble Performance:")
        print(f"  MSE:         {avg_ens_mse:.6f}")
        print(f"  SSIM:        {avg_ens_ssim:.3f}")
        print(f"  Uncertainty: {avg_ens_unc:.6f}")
        
        print(f"Best Individual Performance:")
        print(f"  MSE:         {avg_ind_mse:.6f}")
        print(f"  SSIM:        {avg_ind_ssim:.3f}")
        
        # Performance comparison
        mse_diff = ((avg_ens_mse - avg_ind_mse) / avg_ind_mse) * 100
        ssim_diff = ((avg_ens_ssim - avg_ind_ssim) / avg_ind_ssim) * 100
        
        print(f"Comparison (Ensemble vs Best Individual):")
        print(f"  MSE Difference:  {mse_diff:+.1f}%")
        print(f"  SSIM Difference: {ssim_diff:+.1f}%")
        
        # Ensemble weights analysis
        avg_weights = np.mean([m['ensemble_weights'] for m in metrics], axis=0)
        model_names = ['Simple', 'MC', 'Hierarchical', 'Enhanced', 'Unified']
        
        print(f"Average Ensemble Weights:")
        for name, weight in zip(model_names, avg_weights):
            print(f"  {name:12}: {weight:.3f}")
        
        # Dominant model
        dominant_idx = np.argmax(avg_weights)
        print(f"Dominant Model: {model_names[dominant_idx]} ({avg_weights[dominant_idx]:.3f})")

def main():
    """Main visualization execution"""
    print("🎨 FULL ENSEMBLE RECONSTRUCTION VISUALIZATION")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    all_metrics = {}
    
    # Create visualizations for each dataset
    for dataset_name in datasets:
        print(f"\n📊 Creating visualization for {dataset_name}...")
        
        try:
            metrics, viz_path = create_comprehensive_visualization(dataset_name, device)
            all_metrics[dataset_name] = metrics
            print(f"✅ {dataset_name} visualization complete")
        except Exception as e:
            print(f"❌ Error with {dataset_name}: {e}")
            continue
    
    if all_metrics:
        # Create summary comparison
        print(f"\n📈 Creating summary comparison...")
        summary_path = create_comparison_summary(all_metrics)
        
        # Print detailed analysis
        print_detailed_analysis(all_metrics)
        
        # Save comprehensive results
        output_dir = Path("results/full_ensemble_visualizations")
        with open(output_dir / "comprehensive_analysis.json", 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"\n🎉 VISUALIZATION COMPLETE!")
        print("=" * 70)
        print(f"📁 All results saved to: {output_dir}")
        print(f"🖼️  Individual visualizations: {len(all_metrics)} datasets")
        print(f"📊 Summary comparison: {summary_path}")
        print(f"📋 Comprehensive analysis: comprehensive_analysis.json")
        
    else:
        print("❌ No visualizations could be created")

if __name__ == "__main__":
    main()
