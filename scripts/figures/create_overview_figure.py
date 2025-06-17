#!/usr/bin/env python3
"""
CortexFlow Architecture Overview Generator
Creates a comprehensive overview of all CortexFlow architectures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set publication-ready style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def create_mini_architecture(ax, x, y, width, height, title, components, color='lightblue'):
    """Create a mini architecture diagram"""
    # Title box
    title_box = FancyBboxPatch(
        (x, y + height - 0.8), width, 0.6,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(title_box)
    ax.text(x + width/2, y + height - 0.5, title, 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Component boxes
    comp_height = (height - 1) / len(components)
    for i, comp in enumerate(components):
        comp_y = y + height - 1.2 - (i + 1) * comp_height
        comp_box = FancyBboxPatch(
            (x + 0.1, comp_y), width - 0.2, comp_height - 0.1,
            boxstyle="round,pad=0.01",
            facecolor='white',
            edgecolor='gray',
            linewidth=0.8
        )
        ax.add_patch(comp_box)
        ax.text(x + width/2, comp_y + comp_height/2 - 0.05, comp, 
                ha='center', va='center', fontsize=7)

def create_cortexflow_overview():
    """Create comprehensive CortexFlow architecture overview"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    
    # Main title
    ax.text(10, 13.5, 'CortexFlow Neural Decoding Framework: Complete Architecture Overview', 
            ha='center', va='center', fontsize=18, weight='bold')
    
    # Subtitle
    ax.text(10, 12.8, 'Enhanced 5-Fold Cross-Validation with 8-Variant Ensemble Architecture', 
            ha='center', va='center', fontsize=14, style='italic')
    
    # Input section
    input_box = FancyBboxPatch(
        (8.5, 11.5), 3, 0.8,
        boxstyle="round,pad=0.02",
        facecolor='lightgreen',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(10, 11.9, 'fMRI Input (Variable Dimensions)', 
            ha='center', va='center', fontsize=12, weight='bold')
    
    # CortexFlow variants (top row)
    variants_top = [
        ('CortexFlow-Lite', ['Linear 512', 'BatchNorm+ReLU', 'Dropout 0.2', 'Linear 256', 'Linear 512', 'Output 784'], 'lightblue'),
        ('CortexFlow-MC', ['Linear 512', 'LayerNorm+ReLU', 'MC Dropout', 'Linear 256', 'Uncertainty', 'Output 784'], 'red'),
        ('CortexFlow-Hierarchical', ['Level 1: Coarse', 'Level 2: Medium', 'Level 3: Fine', 'Fusion', 'Output 784'], 'lightyellow'),
        ('CortexFlow-Enhanced', ['MC Component', 'Hierarchical', 'Feature Align', 'Integration', 'Output 784'], 'orange')
    ]
    
    x_positions_top = [1, 5.5, 10, 14.5]
    for i, (title, components, color) in enumerate(variants_top):
        create_mini_architecture(ax, x_positions_top[i], 8.5, 3.5, 2.5, title, components, color)
        # Arrow from input
        ax.annotate('', xy=(x_positions_top[i] + 1.75, 10.8), xytext=(10, 11.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # CortexFlow variants (bottom row)
    variants_bottom = [
        ('CortexFlow-Unified', ['Simple Path', 'Complex Path', 'Complexity Gate', 'Adaptive', 'Output 784'], 'lightcoral'),
        ('CortexFlow-Diffusion', ['Multi-Path Enc', 'Cross-Attention', 'Diffusion', 'Progressive', 'Output 784'], 'lightpink'),
        ('CortexFlow-CNN', ['Conv Layer 1', 'Conv Layer 2', 'Conv Layer 3', 'Adaptive Pool', 'Output 784'], 'lightgray'),
        ('CortexFlow Multi-Pathway', ['Deep Pathway', 'Wide Pathway', 'Cross-Attention', 'Fusion', 'Output 784'], 'gold')
    ]
    
    x_positions_bottom = [1, 5.5, 10, 14.5]
    for i, (title, components, color) in enumerate(variants_bottom):
        create_mini_architecture(ax, x_positions_bottom[i], 5.5, 3.5, 2.5, title, components, color)
        # Arrow from input
        ax.annotate('', xy=(x_positions_bottom[i] + 1.75, 7.8), xytext=(10, 11.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Ensemble section
    ensemble_box = FancyBboxPatch(
        (6, 3), 8, 1.5,
        boxstyle="round,pad=0.02",
        facecolor='purple',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(ensemble_box)
    ax.text(10, 4, 'CortexFlow-Ensemble: Learned Weighting Network', 
            ha='center', va='center', fontsize=12, weight='bold', color='white')
    ax.text(10, 3.5, '8-Variant Intelligent Combination: Σ wᵢ × fᵢ(x)', 
            ha='center', va='center', fontsize=10, style='italic', color='white')
    
    # Arrows to ensemble
    for x in x_positions_top + x_positions_bottom:
        start_y = 8.5 if x in x_positions_top else 5.5
        ax.annotate('', xy=(10, 4.5), xytext=(x + 1.75, start_y),
                   arrowprops=dict(arrowstyle='->', lw=1, color='purple'))
    
    # SOTA comparison section
    sota_box = FancyBboxPatch(
        (1, 0.5), 7, 1.5,
        boxstyle="round,pad=0.02",
        facecolor='cyan',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(sota_box)
    ax.text(4.5, 1.5, 'SOTA Baselines', 
            ha='center', va='center', fontsize=12, weight='bold')
    ax.text(2.5, 1, 'Brain-Diffuser\n(Diffusion)', ha='center', va='center', fontsize=9)
    ax.text(6.5, 1, 'MinD-Vis\n(CVPR 2023)', ha='center', va='center', fontsize=9)
    
    # Performance results section
    perf_box = FancyBboxPatch(
        (12, 0.5), 7, 1.5,
        boxstyle="round,pad=0.02",
        facecolor='lightgreen',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(perf_box)
    ax.text(15.5, 1.7, 'Performance Results (5-Fold CV)', 
            ha='center', va='center', fontsize=12, weight='bold')
    ax.text(15.5, 1.2, '🥇 Vangerven: CortexFlow-Lite (0.041823)', 
            ha='center', va='center', fontsize=8)
    ax.text(15.5, 0.9, '🥇 MindBigData: Multi-Pathway (0.054573)', 
            ha='center', va='center', fontsize=8)
    ax.text(15.5, 0.6, '🥇 Crell: Ensemble (0.028666)', 
            ha='center', va='center', fontsize=8)
    
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/cortexflow_complete_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/cortexflow_complete_overview.svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("🎨 Creating CortexFlow Complete Overview...")
    create_cortexflow_overview()
    print("✅ Overview figure created successfully!")
    print("📁 Saved as cortexflow_complete_overview.png and .svg")
