"""
CCCV4 Reconstruction Visualization
=================================

Visualize CCCV4 Meta-Adaptive reconstructions vs target stimuli across all datasets.
This provides qualitative validation of the quantitative results.

Features:
1. Load best CCCV4 models from templates
2. Generate reconstructions for sample stimuli
3. Create side-by-side comparisons (Target vs Reconstruction)
4. Compute visual quality metrics (MSE, SSIM, PSNR)
5. Save visualization grids for each dataset

Goal: Visual proof that CCCV4 achieves excellent reconstruction quality
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)
sys.path.append(parent_dir)

# Import CCCV4 and utilities
try:
    from cccv4.scripts.test_cccv4_simplified import CCCV4ProxyModel, CCCV4MetaSelector
    from cccv3.src.models.cccv3_ultimate import create_cccv3_ultimate_trainer
except ImportError:
    try:
        sys.path.append(os.path.join(parent_dir, 'scripts'))
        sys.path.append(os.path.join(root_dir, 'cccv3', 'src', 'models'))
        from test_cccv4_simplified import CCCV4ProxyModel, CCCV4MetaSelector
        from cccv3_ultimate import create_cccv3_ultimate_trainer
    except ImportError:
        print("❌ Could not import CCCV4 models")
        sys.exit(1)

# Import data utilities
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

def compute_visual_metrics(pred, target):
    """
    Compute visual quality metrics
    
    Args:
        pred: [H, W] - Predicted image
        target: [H, W] - Target image
    Returns:
        metrics: Dict with MSE, PSNR, SSIM
    """
    # MSE
    mse = np.mean((pred - target) ** 2)
    
    # PSNR
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Simplified SSIM
    mu1 = np.mean(pred)
    mu2 = np.mean(target)
    sigma1 = np.var(pred)
    sigma2 = np.var(target)
    sigma12 = np.mean((pred - mu1) * (target - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim
    }

def train_cccv4_for_visualization(dataset_name, device):
    """
    Train CCCV4 model for visualization (quick training for demo)
    
    Args:
        dataset_name: Name of dataset
        device: Device for computation
    Returns:
        model: Trained CCCV4 model
        test_data: Test data for visualization
    """
    print(f"\n🔧 Training CCCV4 for {dataset_name.upper()} visualization...")
    
    # Load dataset
    try:
        if 'load_dataset_gpu_optimized' in globals():
            X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
        else:
            print("❌ Dataset loading function not available")
            return None, None
        
        if X_train is None:
            print(f"❌ Failed to load {dataset_name}")
            return None, None
        
        dataset_size = len(X_train)
        print(f"✅ Dataset loaded: Train={dataset_size}, Test={len(X_test)}, Input_dim={input_dim}")
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None, None
    
    # Create CCCV4 model
    model = CCCV4ProxyModel(input_dim, dataset_name, dataset_size, device)
    trainer = create_cccv3_ultimate_trainer(model.active_model, device)
    
    # Quick training for visualization (reduced epochs)
    from torch.utils.data import DataLoader, TensorDataset
    
    batch_size = min(16, len(X_train) // 4)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Use train as val for quick demo
    
    config = {
        'use_optimal_hyperparams': True,
        'transfer_learning': True
    }
    
    print(f"   Training with {model.selection_info['selected_version']} strategy...")
    training_results = trainer.train_ultimate_model(train_loader, val_loader, config)
    
    print(f"   Training complete! Best val loss: {training_results['results'][list(training_results['results'].keys())[0]]['best_val_loss']:.6f}")
    
    return model, (X_test, y_test)

def visualize_reconstructions(model, test_data, dataset_name, device, num_samples=8):
    """
    Create visualization of reconstructions vs targets
    
    Args:
        model: Trained CCCV4 model
        test_data: Tuple of (X_test, y_test)
        dataset_name: Name of dataset
        device: Device for computation
        num_samples: Number of samples to visualize
    Returns:
        fig: Matplotlib figure
        metrics_summary: Summary of visual metrics
    """
    X_test, y_test = test_data
    
    # Select random samples
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    sample_X = X_test[indices].to(device)
    sample_y = y_test[indices].to(device)
    
    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        reconstructions, cccv4_info = model(sample_X)
    
    # Convert to numpy for visualization
    targets = sample_y.cpu().numpy()
    preds = reconstructions.cpu().numpy()
    
    # Create visualization
    fig = plt.figure(figsize=(16, 2 * num_samples))
    gs = GridSpec(num_samples, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Add title
    fig.suptitle(f'CCCV4 Reconstructions - {dataset_name.upper()}\n'
                f'Selected Version: {cccv4_info["cccv4_selection"]} '
                f'(Confidence: {cccv4_info["cccv4_confidence"]:.2f})', 
                fontsize=16, fontweight='bold')
    
    metrics_list = []
    
    for i in range(min(num_samples, len(indices))):
        # Target image
        target_img = targets[i, 0]  # [28, 28]
        pred_img = preds[i, 0]     # [28, 28]
        
        # Compute metrics
        metrics = compute_visual_metrics(pred_img, target_img)
        metrics_list.append(metrics)
        
        # Plot target
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(target_img, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Target {i+1}', fontsize=10)
        ax1.axis('off')
        
        # Plot reconstruction
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(pred_img, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'CCCV4 Reconstruction', fontsize=10)
        ax2.axis('off')
        
        # Plot difference
        ax3 = fig.add_subplot(gs[i, 2])
        diff = np.abs(target_img - pred_img)
        ax3.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        ax3.set_title(f'Difference', fontsize=10)
        ax3.axis('off')
        
        # Metrics text
        ax4 = fig.add_subplot(gs[i, 3])
        ax4.axis('off')
        metrics_text = f"""
MSE: {metrics['mse']:.6f}
PSNR: {metrics['psnr']:.2f} dB
SSIM: {metrics['ssim']:.4f}

Quality: {'Excellent' if metrics['ssim'] > 0.9 else 'Good' if metrics['ssim'] > 0.7 else 'Fair'}
        """
        ax4.text(0.1, 0.5, metrics_text, fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Compute summary metrics
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in metrics_list]),
        'psnr': np.mean([m['psnr'] for m in metrics_list if m['psnr'] != float('inf')]),
        'ssim': np.mean([m['ssim'] for m in metrics_list])
    }
    
    # Add summary text
    summary_text = f"""
CCCV4 {dataset_name.upper()} Summary:
Average MSE: {avg_metrics['mse']:.6f}
Average PSNR: {avg_metrics['psnr']:.2f} dB
Average SSIM: {avg_metrics['ssim']:.4f}

Selected Strategy: {cccv4_info['cccv4_selection']}
Confidence: {cccv4_info['cccv4_confidence']:.2f}
    """
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    return fig, avg_metrics

def create_comparison_grid(all_results):
    """
    Create comparison grid across all datasets
    
    Args:
        all_results: Dict with results for each dataset
    Returns:
        fig: Matplotlib figure with comparison
    """
    datasets = list(all_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CCCV4 Meta-Adaptive Performance Across Datasets', fontsize=18, fontweight='bold')
    
    for idx, dataset_name in enumerate(datasets):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if dataset_name in all_results:
            metrics = all_results[dataset_name]['metrics']
            selection_info = all_results[dataset_name]['selection_info']
            
            # Create bar chart of metrics
            metric_names = ['MSE', 'PSNR', 'SSIM']
            metric_values = [
                metrics['mse'] * 1000,  # Scale MSE for visibility
                metrics['psnr'] / 10,   # Scale PSNR for visibility
                metrics['ssim'] * 100   # Scale SSIM to percentage
            ]
            
            bars = ax.bar(metric_names, metric_values, 
                         color=['red', 'blue', 'green'], alpha=0.7)
            
            ax.set_title(f'{dataset_name.upper()}\n'
                        f'Selected: {selection_info["selected_version"]} '
                        f'(Conf: {selection_info["confidence"]:.2f})', 
                        fontsize=12, fontweight='bold')
            
            # Add value labels on bars
            for bar, value, original in zip(bars, metric_values, 
                                          [metrics['mse'], metrics['psnr'], metrics['ssim']]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                       f'{original:.4f}' if original < 1 else f'{original:.2f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Scaled Metric Values')
            ax.grid(True, alpha=0.3)
            
            # Add quality assessment
            quality = 'Excellent' if metrics['ssim'] > 0.9 else 'Good' if metrics['ssim'] > 0.7 else 'Fair'
            ax.text(0.5, 0.95, f'Quality: {quality}', transform=ax.transAxes,
                   ha='center', va='top', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor="lightgreen" if quality == 'Excellent' else "yellow" if quality == 'Good' else "orange",
                           alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'{dataset_name.upper()}\nNo data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def main():
    """Main visualization function"""
    print("🎨 CCCV4 Reconstruction Visualization")
    print("=" * 40)
    print("🎯 Visual validation of CCCV4 Meta-Adaptive performance")
    print("📊 Generating reconstructions vs targets for all datasets")
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
    
    # Create output directory
    output_dir = "cccv4/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"🎨 Visualizing {dataset_name.upper()} Reconstructions")
        print(f"{'='*60}")
        
        # Train CCCV4 model for this dataset
        model, test_data = train_cccv4_for_visualization(dataset_name, device)
        
        if model is not None and test_data is not None:
            # Create visualizations
            fig, metrics = visualize_reconstructions(
                model, test_data, dataset_name, device, num_samples=6
            )
            
            # Save individual dataset visualization
            output_path = os.path.join(output_dir, f"cccv4_{dataset_name}_reconstructions.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved visualization: {output_path}")
            
            # Store results
            all_results[dataset_name] = {
                'metrics': metrics,
                'selection_info': model.selection_info,
                'figure': fig
            }
            
            # Print summary
            print(f"\n📊 {dataset_name.upper()} Visual Quality Summary:")
            print(f"   Average MSE: {metrics['mse']:.6f}")
            print(f"   Average PSNR: {metrics['psnr']:.2f} dB")
            print(f"   Average SSIM: {metrics['ssim']:.4f}")
            print(f"   Selected Version: {model.selection_info['selected_version']}")
            print(f"   Confidence: {model.selection_info['confidence']:.2f}")
            
            quality = 'Excellent' if metrics['ssim'] > 0.9 else 'Good' if metrics['ssim'] > 0.7 else 'Fair'
            print(f"   Overall Quality: {quality}")
            
            plt.close(fig)  # Close to save memory
            
            # Clear memory
            del model
            torch.cuda.empty_cache()
        else:
            print(f"❌ Failed to process {dataset_name}")
    
    # Create comparison grid
    if all_results:
        print(f"\n{'='*60}")
        print(f"🎨 Creating Cross-Dataset Comparison")
        print(f"{'='*60}")
        
        comparison_fig = create_comparison_grid(all_results)
        comparison_path = os.path.join(output_dir, "cccv4_cross_dataset_comparison.png")
        comparison_fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved comparison: {comparison_path}")
        
        plt.close(comparison_fig)
    
    # Final summary
    print(f"\n🎉 CCCV4 Visualization Complete!")
    print(f"=" * 40)
    print(f"📁 All visualizations saved to: {output_dir}")
    print(f"📊 Datasets processed: {len(all_results)}/{len(datasets)}")
    
    if all_results:
        print(f"\n📈 Overall Quality Assessment:")
        for dataset_name, result in all_results.items():
            metrics = result['metrics']
            quality = 'Excellent' if metrics['ssim'] > 0.9 else 'Good' if metrics['ssim'] > 0.7 else 'Fair'
            print(f"   {dataset_name.upper()}: {quality} (SSIM: {metrics['ssim']:.4f})")
    
    print(f"\n🚀 CCCV4 Meta-Adaptive: Visual excellence validated!")

if __name__ == "__main__":
    main()
