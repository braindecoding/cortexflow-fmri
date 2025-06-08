#!/usr/bin/env python3
"""
Create Uncertainty Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_uncertainty_visualization():
    """Create uncertainty quantification visualization"""
    print("🎯 Creating uncertainty visualization...")
    
    results_dir = Path("results/actual_experiments/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Data from Table 6
    datasets = ['Miyawaki', 'Vangerven', 'MindBigData', 'Crell']
    
    # CortexFlow-MC data
    mc_epistemic = [0.024, 0.031, 0.045, 0.042]
    mc_aleatoric = [0.018, 0.022, 0.035, 0.028]
    
    # CortexFlow-Enhanced data
    enhanced_epistemic = [0.035, 0.038, 0.055, 0.048]
    enhanced_aleatoric = [0.025, 0.028, 0.042, 0.035]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CortexFlow: Kuantifikasi Ketidakpastian Sistematis', 
                 fontsize=16, fontweight='bold')
    
    # Colors
    colors = ['#FF6B6B', '#4ECDC4']
    
    # Plot 1: MC Uncertainty
    ax1 = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, mc_epistemic, width, 
                   label='Epistemic', color=colors[0], alpha=0.8)
    bars2 = ax1.bar(x + width/2, mc_aleatoric, width,
                   label='Aleatoric', color=colors[1], alpha=0.8)
    
    ax1.set_title('CortexFlow-MC', fontweight='bold')
    ax1.set_ylabel('Uncertainty Value')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Enhanced Uncertainty
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - width/2, enhanced_epistemic, width, 
                   label='Epistemic', color=colors[0], alpha=0.8)
    bars4 = ax2.bar(x + width/2, enhanced_aleatoric, width,
                   label='Aleatoric', color=colors[1], alpha=0.8)
    
    ax2.set_title('CortexFlow-Enhanced', fontweight='bold')
    ax2.set_ylabel('Uncertainty Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Total Uncertainty Comparison
    ax3 = axes[1, 0]
    mc_total = [e + a for e, a in zip(mc_epistemic, mc_aleatoric)]
    enhanced_total = [e + a for e, a in zip(enhanced_epistemic, enhanced_aleatoric)]
    
    bars5 = ax3.bar(x - width/2, mc_total, width, 
                   label='CortexFlow-MC', color='#4A90E2', alpha=0.8)
    bars6 = ax3.bar(x + width/2, enhanced_total, width,
                   label='CortexFlow-Enhanced', color='#F5A623', alpha=0.8)
    
    ax3.set_title('Total Uncertainty Comparison', fontweight='bold')
    ax3.set_xlabel('Datasets')
    ax3.set_ylabel('Total Uncertainty')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Cross-Modal Uncertainty Pattern
    ax4 = axes[1, 1]
    fmri_datasets = ['Miyawaki', 'Vangerven']
    eeg_datasets = ['MindBigData', 'Crell']
    
    fmri_mc = np.mean([mc_total[0], mc_total[1]])
    eeg_mc = np.mean([mc_total[2], mc_total[3]])
    fmri_enhanced = np.mean([enhanced_total[0], enhanced_total[1]])
    eeg_enhanced = np.mean([enhanced_total[2], enhanced_total[3]])
    
    categories = ['fMRI Native', 'EEG→fMRI']
    mc_values = [fmri_mc, eeg_mc]
    enhanced_values = [fmri_enhanced, eeg_enhanced]
    
    x_cat = np.arange(len(categories))
    bars7 = ax4.bar(x_cat - width/2, mc_values, width, 
                   label='CortexFlow-MC', color='#4A90E2', alpha=0.8)
    bars8 = ax4.bar(x_cat + width/2, enhanced_values, width,
                   label='CortexFlow-Enhanced', color='#F5A623', alpha=0.8)
    
    ax4.set_title('Cross-Modal Uncertainty Pattern', fontweight='bold')
    ax4.set_xlabel('Data Modality')
    ax4.set_ylabel('Average Total Uncertainty')
    ax4.set_xticks(x_cat)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels and insights
    for bars in [bars7, bars8]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add insight text
    uncertainty_increase = ((eeg_mc - fmri_mc) / fmri_mc) * 100
    ax4.text(0.02, 0.98, f'Cross-modal uncertainty\nincrease: {uncertainty_increase:.1f}%', 
            transform=ax4.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(results_dir / "uncertainty_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Uncertainty visualization saved")

def main():
    """Main execution"""
    print("🚀 Creating Uncertainty Visualization...")
    create_uncertainty_visualization()
    print("🎉 Complete!")

if __name__ == "__main__":
    main()
