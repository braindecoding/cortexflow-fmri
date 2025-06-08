#!/usr/bin/env python3
"""
📊 CREATE COMPLETE VISUALIZATION FOR ALL 4 DATASETS
================================================================================
Generate comprehensive visualizations for all trained models and datasets
================================================================================
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import Monte Carlo Simple CortexFlow
from src.models.mc_simple_cortexflow import create_mc_simple_cortexflow

def load_trained_model(dataset_name, input_dim):
    """Load trained model for dataset."""
    model_path = Path(f"results/mc_simple/mc_simple_{dataset_name}_model.pt")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Create model
    model = create_mc_simple_cortexflow(input_dim)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Loaded trained model: {dataset_name}")
    print(f"   Best loss: {checkpoint['best_loss']:.6f}")
    print(f"   Epoch: {checkpoint['epoch']}")
    
    return model, checkpoint

def create_uncertainty_visualization_fixed(model, dataset_name, test_fmri, test_stim):
    """Create fixed uncertainty visualization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_fmri = test_fmri.to(device)
    test_stim = test_stim.to(device)
    
    # Get predictions with uncertainty
    with torch.no_grad():
        # Standard forward pass
        outputs = model(test_fmri)
        reconstruction = outputs['reconstruction']
        
        # Get uncertainty estimates (if available)
        if hasattr(model, 'forward_with_mc_sampling'):
            try:
                mc_outputs = model.forward_with_mc_sampling(test_fmri)
                total_uncertainty = mc_outputs.get('total_uncertainty', None)
            except:
                total_uncertainty = None
        else:
            total_uncertainty = None
    
    # Create visualization
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    for i in range(min(5, test_stim.shape[0])):
        # Original
        axes[0, i].imshow(test_stim[i].view(28, 28).cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        recon_img = reconstruction[i].view(28, 28).cpu().numpy()
        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].set_title(f'Reconstruction {i+1}')
        axes[1, i].axis('off')
        
        # Uncertainty (if available)
        if total_uncertainty is not None and i < total_uncertainty.shape[0]:
            # Handle different uncertainty shapes
            unc_tensor = total_uncertainty[i]
            if unc_tensor.numel() == 784:  # 28x28
                uncertainty_img = unc_tensor.view(28, 28).cpu().numpy()
            elif unc_tensor.numel() == 1:  # scalar uncertainty
                uncertainty_img = np.full((28, 28), unc_tensor.item())
            else:
                # Use reconstruction error as uncertainty proxy
                error = torch.abs(reconstruction[i] - test_stim[i].view(1, 28, 28))
                uncertainty_img = error.view(28, 28).cpu().numpy()
            
            im = axes[2, i].imshow(uncertainty_img, cmap='hot')
            axes[2, i].set_title(f'Uncertainty {i+1}')
        else:
            # Use reconstruction error as uncertainty proxy
            error = torch.abs(reconstruction[i] - test_stim[i].view(1, 28, 28))
            uncertainty_img = error.view(28, 28).cpu().numpy()
            im = axes[2, i].imshow(uncertainty_img, cmap='hot')
            axes[2, i].set_title(f'Recon Error {i+1}')
        
        axes[2, i].axis('off')
    
    plt.suptitle(f'Monte Carlo Simple CortexFlow - {dataset_name.upper()}\nReconstructions and Uncertainty', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    results_dir = Path("results/mc_simple")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = results_dir / f"mc_simple_{dataset_name}_uncertainty.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"📊 Uncertainty visualization saved: {viz_path}")
    
    plt.close()

def create_complete_summary():
    """Create complete summary of all datasets."""
    datasets = {
        'miyawaki': {
            'file': 'data/processed/miyawaki_structured_28x28.mat',
            'input_dim': 967,
            'description': 'Visual Cortex fMRI'
        },
        'vangerven': {
            'file': 'data/processed/digit69_28x28.mat',
            'input_dim': 3092,
            'description': 'Digit Recognition fMRI'
        },
        'mindbigdata': {
            'file': 'data/processed/mindbigdata.mat',
            'input_dim': 3092,
            'description': 'EEG-based Neural Signals'
        },
        'crell': {
            'file': 'data/processed/crell.mat',
            'input_dim': 3092,
            'description': 'Advanced fMRI'
        }
    }
    
    print("📊 CREATING COMPLETE VISUALIZATION FOR ALL 4 DATASETS")
    print("="*80)
    
    results_summary = {}
    
    for dataset_name, config in datasets.items():
        print(f"\n🧪 Processing {dataset_name.upper()} ({config['description']})")
        print("-"*60)
        
        try:
            # Load dataset
            data = scipy.io.loadmat(config['file'])
            test_fmri = torch.FloatTensor(data['fmriTest'])
            test_stim = torch.FloatTensor(data['stimTest'])
            
            # Normalize stimuli if needed
            if test_stim.max() > 1.0:
                test_stim = test_stim / 255.0
            
            # Load trained model
            model, checkpoint = load_trained_model(dataset_name, config['input_dim'])
            
            if model is not None:
                # Create visualization
                create_uncertainty_visualization_fixed(model, dataset_name, test_fmri, test_stim)
                
                results_summary[dataset_name] = {
                    'status': 'success',
                    'best_loss': checkpoint['best_loss'],
                    'epoch': checkpoint['epoch'],
                    'description': config['description'],
                    'input_dim': config['input_dim'],
                    'test_samples': test_fmri.shape[0]
                }
                
                print(f"✅ {dataset_name}: Visualization completed")
            else:
                results_summary[dataset_name] = {
                    'status': 'failed',
                    'reason': 'Model not found'
                }
                
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            results_summary[dataset_name] = {
                'status': 'failed',
                'reason': str(e)
            }
    
    # Create summary comparison
    create_performance_comparison(results_summary)
    
    return results_summary

def create_performance_comparison(results_summary):
    """Create performance comparison visualization."""
    successful_results = {k: v for k, v in results_summary.items() 
                         if v['status'] == 'success'}
    
    if not successful_results:
        print("❌ No successful results to compare")
        return
    
    # Extract data for comparison
    datasets = list(successful_results.keys())
    losses = [successful_results[d]['best_loss'] for d in datasets]
    input_dims = [successful_results[d]['input_dim'] for d in datasets]
    test_samples = [successful_results[d]['test_samples'] for d in datasets]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Best Loss Comparison
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars1 = axes[0].bar(range(len(datasets)), losses, color=colors[:len(datasets)])
    axes[0].set_xlabel('Datasets')
    axes[0].set_ylabel('Best Test Loss')
    axes[0].set_title('🎯 Best Test Loss Comparison', fontweight='bold')
    axes[0].set_xticks(range(len(datasets)))
    axes[0].set_xticklabels([d.title() for d in datasets], rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, loss in zip(bars1, losses):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Input Dimensions
    bars2 = axes[1].bar(range(len(datasets)), input_dims, color=colors[:len(datasets)])
    axes[1].set_xlabel('Datasets')
    axes[1].set_ylabel('Input Dimensions')
    axes[1].set_title('📊 Input Dimension Comparison', fontweight='bold')
    axes[1].set_xticks(range(len(datasets)))
    axes[1].set_xticklabels([d.title() for d in datasets], rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, dim in zip(bars2, input_dims):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{dim}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Test Samples
    bars3 = axes[2].bar(range(len(datasets)), test_samples, color=colors[:len(datasets)])
    axes[2].set_xlabel('Datasets')
    axes[2].set_ylabel('Test Samples')
    axes[2].set_title('🧪 Test Sample Count', fontweight='bold')
    axes[2].set_xticks(range(len(datasets)))
    axes[2].set_xticklabels([d.title() for d in datasets], rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, samples in zip(bars3, test_samples):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{samples}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Monte Carlo Simple CortexFlow - Complete Dataset Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison
    results_dir = Path("results/mc_simple")
    comparison_path = results_dir / "complete_dataset_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"📊 Performance comparison saved: {comparison_path}")
    
    plt.close()

def main():
    """Main function."""
    print("📊 CREATING COMPLETE VISUALIZATION FOR ALL 4 DATASETS")
    print("="*80)
    
    results = create_complete_summary()
    
    print(f"\n🎉 COMPLETE VISUALIZATION COMPLETED!")
    print("="*80)
    
    print(f"\n📊 FINAL SUMMARY:")
    print("-"*60)
    
    for dataset_name, result in results.items():
        if result['status'] == 'success':
            print(f"✅ {dataset_name.upper():12} : Loss {result['best_loss']:.6f}, "
                  f"Epoch {result['epoch']}, "
                  f"Input {result['input_dim']}")
        else:
            print(f"❌ {dataset_name.upper():12} : FAILED - {result.get('reason', 'Unknown error')}")
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    total = len(results)
    print(f"\n📈 SUCCESS RATE: {successful}/{total} ({100*successful/total:.1f}%)")
    
    if successful == total:
        print("🎉 ALL 4 DATASETS SUCCESSFULLY TRAINED AND VISUALIZED!")
    else:
        print("⚠️  Some datasets need attention")

if __name__ == "__main__":
    main()
