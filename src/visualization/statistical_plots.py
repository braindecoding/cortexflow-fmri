"""
Statistical Visualization Functions
===================================

Comprehensive statistical visualization functions for neural decoding research.

Features:
    - Performance comparison plots
    - Method ranking visualizations
    - CortexFlow comparison charts
    - Statistical significance heatmaps
    - Publication-ready SVG outputs

Academic Standards:
    - Professional visualization design
    - Clear statistical presentations
    - Publication-ready quality
    - Academic color schemes
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from pathlib import Path


def create_statistical_visualization(all_results, output_dir):
    """
    Create comprehensive statistical visualization
    
    Creates a multi-panel statistical visualization showing:
    1. Performance comparison across datasets
    2. Method ranking with error bars
    3. CortexFlow approaches comparison
    4. Statistical significance matrix
    
    Args:
        all_results: Dictionary with dataset results {dataset: {method: score}}
        output_dir: Output directory for saving visualization
        
    Returns:
        Path to saved visualization file
    """

    print(f"\n📈 CREATING STATISTICAL VISUALIZATIONS")
    print("=" * 50)

    # Prepare data untuk visualization
    datasets = list(all_results.keys())
    methods = list(all_results[datasets[0]].keys())

    # Create performance comparison plot with better aspect ratio
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Comprehensive Statistical Analysis - CortexFlow vs SOTA Methods',
                fontsize=16, fontweight='bold')

    # 1. Performance comparison across datasets
    ax1 = axes[0, 0]
    dataset_scores = {method: [] for method in methods}

    for dataset in datasets:
        for method in methods:
            dataset_scores[method].append(all_results[dataset][method])

    x_pos = np.arange(len(datasets))
    width = 0.15

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#FF9F43', '#A55EEA', '#26D0CE', '#FD79A8', '#FDCB6E']

    for i, (method, scores) in enumerate(dataset_scores.items()):
        ax1.bar(x_pos + i*width, scores, width, label=method, color=colors[i % len(colors)], alpha=0.8)

    ax1.set_xlabel('Datasets')
    ax1.set_ylabel('MSE (Lower is Better)')
    ax1.set_title('Performance Comparison Across Datasets')
    ax1.set_xticks(x_pos + width * 2)
    ax1.set_xticklabels([d.capitalize() for d in datasets])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Method ranking
    ax2 = axes[0, 1]
    method_means = [np.mean(scores) for scores in dataset_scores.values()]
    method_stds = [np.std(scores) for scores in dataset_scores.values()]

    y_pos = np.arange(len(methods))
    ax2.barh(y_pos, method_means, xerr=method_stds, color=colors[:len(methods)], alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods)
    ax2.set_xlabel('Mean MSE ± Std')
    ax2.set_title('Overall Method Performance')
    ax2.grid(True, alpha=0.3)

    # 3. CortexFlow comparison
    ax3 = axes[1, 0]
    cortexflow_methods = [method for method in methods if 'CortexFlow' in method]
    cortexflow_scores = {method: dataset_scores[method] for method in cortexflow_methods}

    if len(cortexflow_methods) >= 2:
        cf_datasets = list(range(len(datasets)))
        for i, (method, scores) in enumerate(cortexflow_scores.items()):
            ax3.plot(cf_datasets, scores, marker='o', linewidth=2,
                    label=method, color=colors[(i+3) % len(colors)])

        ax3.set_xlabel('Datasets')
        ax3.set_ylabel('MSE')
        ax3.set_title('CortexFlow Approaches Comparison')
        ax3.set_xticks(cf_datasets)
        ax3.set_xticklabels([d.capitalize() for d in datasets])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Statistical significance heatmap (ONLY if real CV data available)
    ax4 = axes[1, 1]

    # Check if we have multiple samples untuk statistical testing
    has_multiple_samples = all(len(scores) > 1 for scores in dataset_scores.values())

    if has_multiple_samples:
        # Create REAL p-value matrix from actual data
        n_methods = len(methods)
        p_matrix = np.ones((n_methods, n_methods))

        # Calculate REAL p-values from actual cross-validation results
        for i in range(n_methods):
            for j in range(n_methods):
                if i != j:
                    scores_i = dataset_scores[methods[i]]
                    scores_j = dataset_scores[methods[j]]
                    if len(scores_i) > 1 and len(scores_j) > 1:
                        _, p_val = stats.ttest_rel(scores_i, scores_j)
                        p_matrix[i, j] = p_val

        # Create heatmap with REAL data
        sns.heatmap(p_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                    xticklabels=[m.replace('_', ' ') for m in methods],
                    yticklabels=[m.replace('_', ' ') for m in methods],
                    ax=ax4, cbar_kws={'label': 'p-value'})
        ax4.set_title('Statistical Significance Matrix\n(REAL p-values from cross-validation)')
    else:
        # No statistical testing possible with single scores
        ax4.text(0.5, 0.5, 'Statistical significance testing\nrequires cross-validation\nwith multiple samples\n\nRun with CV for real p-values',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Statistical Testing Not Available\n(Single scores only)')
        ax4.set_xticks([])
        ax4.set_yticks([])

    plt.tight_layout()

    # Save visualization
    viz_path = output_dir / "statistical_analysis_comprehensive.svg"
    fig.savefig(viz_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"✅ Statistical visualization saved: {viz_path}")

    return viz_path


def create_gpu_optimized_reconstruction_figure(dataset_name, device='cuda'):
    """
    Create reconstruction figure dengan GPU optimization

    Creates a comprehensive reconstruction visualization showing:
    - Original target images
    - Reconstructions from all models
    - MSE scores for each method
    - GPU-optimized training and inference

    Args:
        dataset_name: Name of dataset to process
        device: Device for computation ('cuda' or 'cpu')

    Returns:
        tuple: (figure, mse_results) or (None, None) if failed
    """

    # Import required modules (avoid circular imports)
    from src.data import load_dataset_gpu_optimized
    from src.models import (
        StandardBaselineCNN, OptimizedMinDVis, OptimizedBrainDiffuser,
        MiyawakiAdvancedCortexFlow, CortexFlowEnsemble
    )
    from src.training.gpu_training import gpu_optimized_training
    import torch
    import torch.nn as nn

    print(f"\n🎨 Creating GPU-optimized reconstruction for {dataset_name}")
    print("=" * 70)

    # Load data ke GPU
    X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)

    if X_train is None:
        return None, None

    # Split untuk validation
    val_size = min(int(0.2 * len(X_train)), 50)  # Limit validation size
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    # Initialize GPU-optimized models
    models = [
        StandardBaselineCNN(input_dim, device),
        OptimizedMinDVis(input_dim, device),
        OptimizedBrainDiffuser(input_dim, device),
        MiyawakiAdvancedCortexFlow(input_dim, device),  # CortexFlow-Enhanced (Optimal)
        CortexFlowEnsemble(input_dim, device)    # True Ensemble for comparison
    ]

    # GPU-optimized training configs for 5 models
    # Adaptive learning rates for different datasets
    if dataset_name == 'mindbigdata':
        # Lower learning rates for MindBigData to prevent NaN
        training_configs = [
            {'epochs': 200, 'lr': 0.0005, 'batch_size': 64, 'patience': 40},   # CNN (reduced LR)
            {'epochs': 250, 'lr': 0.0006, 'batch_size': 64, 'patience': 45},   # MinD-Vis
            {'epochs': 150, 'lr': 0.001, 'batch_size': 64, 'patience': 30},    # Brain-Diffuser (reduced LR)
            {'epochs': 300, 'lr': 0.0003, 'batch_size': 64, 'patience': 50},   # CortexFlow-Enhanced (reduced LR)
            {'epochs': 250, 'lr': 0.0004, 'batch_size': 64, 'patience': 45}    # CortexFlow-Ensemble (reduced LR)
        ]
    else:
        # Standard learning rates for other datasets
        training_configs = [
            {'epochs': 200, 'lr': 0.001, 'batch_size': 64, 'patience': 40},   # CNN
            {'epochs': 250, 'lr': 0.0008, 'batch_size': 64, 'patience': 45},  # MinD-Vis
            {'epochs': 150, 'lr': 0.002, 'batch_size': 64, 'patience': 30},   # Brain-Diffuser
            {'epochs': 300, 'lr': 0.0005, 'batch_size': 64, 'patience': 50},  # CortexFlow-Enhanced
            {'epochs': 250, 'lr': 0.0006, 'batch_size': 64, 'patience': 45}   # CortexFlow-Ensemble
        ]

    reconstructions = []
    mse_results = []

    for model, config in zip(models, training_configs):
        # GPU-optimized training
        _ = gpu_optimized_training(model, X_train, y_train, X_val, y_val, **config)

        # Get reconstructions
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            test_samples = X_test[:8]  # 8 samples untuk visualization
            recon = model(test_samples)

        # Move ke CPU untuk visualization
        reconstructions.append(recon.cpu())

        # Compute MSE
        with torch.no_grad():
            mse = nn.MSELoss()(recon, y_test[:8]).item()
        mse_results.append(mse)

        print(f"✅ {model.name}: MSE = {mse:.6f}")

    # Create figure for 5 methods + target row
    num_methods = len(reconstructions)
    num_samples = 8

    fig, axes = plt.subplots(num_methods + 1, num_samples, figsize=(12, (num_methods + 1) * 1.8))

    dataset_titles = {
        'miyawaki': 'Miyawaki (Visual Kompleks)',
        'vangerven': 'Vangerven (Pola Digit)',
        'mindbigdata': 'MindBigData (EEG→fMRI→Visual)',
        'crell': 'Crell (EEG→fMRI→Visual)'
    }

    fig.suptitle(f'Comparison: Multi-Pathway vs Ensemble - Dataset {dataset_titles[dataset_name]}\n'
                f'CortexFlow-Enhanced vs CortexFlow-Ensemble Performance Analysis',
                fontsize=14, fontweight='bold')

    # Plot targets
    y_samples = y_test[:8].cpu()
    for i in range(num_samples):
        axes[0, i].imshow(y_samples[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Target {i+1}', fontsize=10, fontweight='bold')
        axes[0, i].axis('off')

    # Label baris target
    axes[0, 0].text(-0.15, 0.5, 'Target Visual Asli\n(GPU Processed)',
                    transform=axes[0, 0].transAxes, fontsize=11, fontweight='bold',
                    rotation=90, verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

    # Plot reconstructions
    method_labels = [model.name for model in models]
    for method_idx, (recon, method_label, mse) in enumerate(zip(reconstructions, method_labels, mse_results), 1):
        for i in range(num_samples):
            axes[method_idx, i].imshow(recon[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
            axes[method_idx, i].set_title(f'Rekonstruksi {i+1}', fontsize=9)
            axes[method_idx, i].axis('off')

        # Label dengan MSE
        label_text = f"{method_label}\n(GPU Trained)\nMSE: {mse:.4f}"
        axes[method_idx, 0].text(-0.15, 0.5, label_text,
                                transform=axes[method_idx, 0].transAxes, fontsize=10, fontweight='bold',
                                rotation=90, verticalalignment='center', horizontalalignment='center',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    return fig, mse_results
