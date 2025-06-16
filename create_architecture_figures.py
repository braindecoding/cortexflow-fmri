#!/usr/bin/env python3
"""
CortexFlow Architecture Visualization Generator
Creates publication-ready architecture diagrams for all CortexFlow models
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os

# Set publication-ready style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def create_layer_box(ax, x, y, width, height, text, color='lightblue', text_color='black'):
    """Create a styled layer box"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=9, 
            color=text_color, weight='bold')

def create_arrow(ax, start, end, color='black', style='->', width=2):
    """Create connection arrow"""
    arrow = ConnectionPatch(start, end, "data", "data",
                          arrowstyle=style, shrinkA=5, shrinkB=5,
                          mutation_scale=20, fc=color, ec=color, lw=width)
    ax.add_patch(arrow)

def create_cortexflow_lite_architecture():
    """Create CortexFlow-Lite architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Title
    ax.text(6, 9.5, 'CortexFlow-Lite Architecture', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Input layer
    create_layer_box(ax, 1, 8, 2, 0.8, 'fMRI Input\n(Variable Dim)', 'lightgreen')
    
    # Hidden layers
    create_layer_box(ax, 1, 6.5, 2, 0.8, 'Linear Layer\n512 neurons', 'lightblue')
    create_layer_box(ax, 1, 5, 2, 0.8, 'BatchNorm1d\n+ ReLU', 'lightyellow')
    create_layer_box(ax, 1, 3.5, 2, 0.8, 'Dropout (0.2)', 'lightcoral')
    
    create_layer_box(ax, 5, 6.5, 2, 0.8, 'Linear Layer\n256 neurons', 'lightblue')
    create_layer_box(ax, 5, 5, 2, 0.8, 'BatchNorm1d\n+ ReLU', 'lightyellow')
    create_layer_box(ax, 5, 3.5, 2, 0.8, 'Dropout (0.15)', 'lightcoral')
    
    create_layer_box(ax, 9, 6.5, 2, 0.8, 'Linear Layer\n512 neurons', 'lightblue')
    create_layer_box(ax, 9, 5, 2, 0.8, 'BatchNorm1d\n+ ReLU', 'lightyellow')
    
    # Output layer
    create_layer_box(ax, 9, 2, 2, 0.8, 'Output Layer\n784 (28×28)', 'lightgreen')
    
    # Arrows
    create_arrow(ax, (2, 8), (2, 7.3))
    create_arrow(ax, (2, 6.5), (2, 5.8))
    create_arrow(ax, (2, 5), (2, 4.3))
    create_arrow(ax, (3, 3.9), (5, 6.9))
    
    create_arrow(ax, (6, 6.5), (6, 5.8))
    create_arrow(ax, (6, 5), (6, 4.3))
    create_arrow(ax, (7, 3.9), (9, 6.9))
    
    create_arrow(ax, (10, 6.5), (10, 5.8))
    create_arrow(ax, (10, 5), (10, 2.8))
    
    # Add mathematical notation
    ax.text(6, 1, r'$f(x) = W_3 \cdot \sigma(BN(W_2 \cdot \sigma(BN(W_1 \cdot x))))$', 
            ha='center', va='center', fontsize=12, style='italic')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/cortexflow_lite_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/cortexflow_lite_architecture.svg', bbox_inches='tight')
    plt.close()

def create_cortexflow_mc_architecture():
    """Create CortexFlow-MC (Monte Carlo) architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Title
    ax.text(6, 9.5, 'CortexFlow-MC (Monte Carlo) Architecture', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Input layer
    create_layer_box(ax, 1, 8, 2, 0.8, 'fMRI Input\n(Variable Dim)', 'lightgreen')
    
    # Hidden layers with MC Dropout
    create_layer_box(ax, 1, 6.5, 2, 0.8, 'Linear Layer\n512 neurons', 'lightblue')
    create_layer_box(ax, 1, 5, 2, 0.8, 'LayerNorm\n+ ReLU', 'lightyellow')
    create_layer_box(ax, 1, 3.5, 2, 0.8, 'MC Dropout\n(Always Active)', 'red', 'white')
    
    create_layer_box(ax, 5, 6.5, 2, 0.8, 'Linear Layer\n256 neurons', 'lightblue')
    create_layer_box(ax, 5, 5, 2, 0.8, 'LayerNorm\n+ ReLU', 'lightyellow')
    create_layer_box(ax, 5, 3.5, 2, 0.8, 'MC Dropout\n(Always Active)', 'red', 'white')
    
    create_layer_box(ax, 9, 6.5, 2, 0.8, 'Linear Layer\n128 neurons', 'lightblue')
    create_layer_box(ax, 9, 5, 2, 0.8, 'LayerNorm\n+ ReLU', 'lightyellow')
    
    # Output layer
    create_layer_box(ax, 9, 2, 2, 0.8, 'Output Layer\n784 (28×28)', 'lightgreen')
    
    # Uncertainty quantification box
    create_layer_box(ax, 4, 0.5, 4, 0.8, 'Uncertainty Quantification\n(Multiple Forward Passes)', 'orange')
    
    # Arrows
    create_arrow(ax, (2, 8), (2, 7.3))
    create_arrow(ax, (2, 6.5), (2, 5.8))
    create_arrow(ax, (2, 5), (2, 4.3))
    create_arrow(ax, (3, 3.9), (5, 6.9))
    
    create_arrow(ax, (6, 6.5), (6, 5.8))
    create_arrow(ax, (6, 5), (6, 4.3))
    create_arrow(ax, (7, 3.9), (9, 6.9))
    
    create_arrow(ax, (10, 6.5), (10, 5.8))
    create_arrow(ax, (10, 5), (10, 2.8))
    create_arrow(ax, (10, 2), (8, 1.3))
    
    # Add mathematical notation
    ax.text(6, 1.8, r'$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} f(x, \epsilon_t)$ where $\epsilon_t \sim Dropout$', 
            ha='center', va='center', fontsize=11, style='italic')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/cortexflow_mc_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/cortexflow_mc_architecture.svg', bbox_inches='tight')
    plt.close()

def create_cortexflow_hierarchical_architecture():
    """Create CortexFlow-Hierarchical architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Title
    ax.text(7, 11.5, 'CortexFlow-Hierarchical Architecture',
            ha='center', va='center', fontsize=16, weight='bold')

    # Input layer
    create_layer_box(ax, 1, 10, 2, 0.8, 'fMRI Input\n(Variable Dim)', 'lightgreen')

    # Level 1 - Coarse
    create_layer_box(ax, 0.5, 8.5, 3, 0.8, 'Level 1: Coarse Processing\n512 → 256', 'lightblue')
    create_layer_box(ax, 0.5, 7.5, 3, 0.8, 'Temporal Attention\nDropout (0.3)', 'lightyellow')

    # Level 2 - Medium
    create_layer_box(ax, 4.5, 8.5, 3, 0.8, 'Level 2: Medium Processing\n256 → 128', 'lightblue')
    create_layer_box(ax, 4.5, 7.5, 3, 0.8, 'Temporal Attention\nDropout (0.2)', 'lightyellow')

    # Level 3 - Fine
    create_layer_box(ax, 8.5, 8.5, 3, 0.8, 'Level 3: Fine Processing\n128 → 64', 'lightblue')
    create_layer_box(ax, 8.5, 7.5, 3, 0.8, 'Temporal Attention\nDropout (0.1)', 'lightyellow')

    # Hierarchical fusion
    create_layer_box(ax, 4.5, 5.5, 3, 0.8, 'Hierarchical Fusion\nMulti-Scale Integration', 'orange')

    # Output layers
    create_layer_box(ax, 4.5, 3.5, 3, 0.8, 'Final Processing\n128 → 784', 'lightblue')
    create_layer_box(ax, 4.5, 2, 3, 0.8, 'Output Layer\n784 (28×28)', 'lightgreen')

    # Arrows - hierarchical connections
    create_arrow(ax, (2, 10), (2, 9.3))
    create_arrow(ax, (2, 8.5), (2, 8.3))
    create_arrow(ax, (3.5, 7.9), (4.5, 8.9))

    create_arrow(ax, (6, 8.5), (6, 8.3))
    create_arrow(ax, (7.5, 7.9), (8.5, 8.9))

    create_arrow(ax, (10, 8.5), (10, 8.3))

    # Fusion arrows
    create_arrow(ax, (2, 7.5), (5, 6.3))
    create_arrow(ax, (6, 7.5), (6, 6.3))
    create_arrow(ax, (10, 7.5), (7, 6.3))

    create_arrow(ax, (6, 5.5), (6, 4.3))
    create_arrow(ax, (6, 3.5), (6, 2.8))

    # Add mathematical notation
    ax.text(7, 0.5, r'$H_i = LayerNorm(Linear(x)) \odot \sigma(MLP_{temporal}(x))$',
            ha='center', va='center', fontsize=11, style='italic')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/cortexflow_hierarchical_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/cortexflow_hierarchical_architecture.svg', bbox_inches='tight')
    plt.close()

def create_cortexflow_multipathway_architecture():
    """Create CortexFlow Multi-Pathway architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Title
    ax.text(8, 11.5, 'CortexFlow Multi-Pathway Architecture',
            ha='center', va='center', fontsize=16, weight='bold')

    # Input layer
    create_layer_box(ax, 7, 10, 2, 0.8, 'fMRI Input\n(Variable Dim)', 'lightgreen')

    # Pathway 1 - Deep Processing
    create_layer_box(ax, 1, 8.5, 2.5, 0.8, 'Pathway 1: Deep\n512 → 256', 'lightblue')
    create_layer_box(ax, 1, 7.5, 2.5, 0.8, 'Deep Processing\n256 → 128', 'lightblue')
    create_layer_box(ax, 1, 6.5, 2.5, 0.8, 'Deep Features\n128 → 64', 'lightblue')

    # Pathway 2 - Wide Processing
    create_layer_box(ax, 12.5, 8.5, 2.5, 0.8, 'Pathway 2: Wide\n512 → 512', 'lightcoral')
    create_layer_box(ax, 12.5, 7.5, 2.5, 0.8, 'Wide Processing\n512 → 256', 'lightcoral')
    create_layer_box(ax, 12.5, 6.5, 2.5, 0.8, 'Wide Features\n256 → 128', 'lightcoral')

    # Cross-pathway attention
    create_layer_box(ax, 6, 5.5, 4, 0.8, 'Cross-Pathway Attention\nQuery-Key-Value Mechanism', 'gold')

    # Adaptive fusion
    create_layer_box(ax, 6, 4, 4, 0.8, 'Adaptive Fusion\nGated Combination', 'orange')

    # Uncertainty-aware decoder
    create_layer_box(ax, 6, 2.5, 4, 0.8, 'Uncertainty-Aware Decoder\n→ 784 (28×28)', 'lightgreen')

    # Input connections
    create_arrow(ax, (7.5, 10), (2.25, 9.3))
    create_arrow(ax, (8.5, 10), (13.75, 9.3))

    # Pathway 1 connections
    create_arrow(ax, (2.25, 8.5), (2.25, 8.3))
    create_arrow(ax, (2.25, 7.5), (2.25, 7.3))

    # Pathway 2 connections
    create_arrow(ax, (13.75, 8.5), (13.75, 8.3))
    create_arrow(ax, (13.75, 7.5), (13.75, 7.3))

    # Cross-attention connections
    create_arrow(ax, (3.5, 6.9), (6, 6.1))
    create_arrow(ax, (12.5, 6.9), (10, 6.1))

    # Fusion connections
    create_arrow(ax, (8, 5.5), (8, 4.8))
    create_arrow(ax, (8, 4), (8, 3.3))

    # Add mathematical notation
    ax.text(8, 1, r'$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$',
            ha='center', va='center', fontsize=11, style='italic')

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/cortexflow_multipathway_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/cortexflow_multipathway_architecture.svg', bbox_inches='tight')
    plt.close()

def create_cortexflow_enhanced_architecture():
    """Create CortexFlow-Enhanced architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Title
    ax.text(7, 11.5, 'CortexFlow-Enhanced Architecture',
            ha='center', va='center', fontsize=16, weight='bold')

    # Input layer
    create_layer_box(ax, 6, 10, 2, 0.8, 'fMRI Input\n(Variable Dim)', 'lightgreen')

    # MC Component
    create_layer_box(ax, 1, 8.5, 3, 0.8, 'MC Component\nUncertainty Quantification', 'red', 'white')

    # Hierarchical Component
    create_layer_box(ax, 5, 8.5, 4, 0.8, 'Hierarchical Component\nMulti-Scale Processing', 'lightblue')

    # Feature Alignment
    create_layer_box(ax, 10, 8.5, 3, 0.8, 'Feature Alignment\nCross-Component Sync', 'lightyellow')

    # Enhanced Block Integration
    create_layer_box(ax, 4, 6.5, 6, 0.8, 'Enhanced Block Integration\nMC + Hierarchical + Alignment', 'orange')

    # Residual connections
    create_layer_box(ax, 4, 5, 6, 0.8, 'Residual Connections\nGradient Flow Enhancement', 'lightcoral')

    # Final processing
    create_layer_box(ax, 5, 3, 4, 0.8, 'Final Processing\n512 → 256 → 784', 'lightblue')
    create_layer_box(ax, 5.5, 1.5, 3, 0.8, 'Output Layer\n784 (28×28)', 'lightgreen')

    # Connections
    create_arrow(ax, (6.5, 10), (2.5, 9.3))
    create_arrow(ax, (7, 10), (7, 9.3))
    create_arrow(ax, (7.5, 10), (11.5, 9.3))

    create_arrow(ax, (2.5, 8.5), (5, 7.3))
    create_arrow(ax, (7, 8.5), (7, 7.3))
    create_arrow(ax, (11.5, 8.5), (9, 7.3))

    create_arrow(ax, (7, 6.5), (7, 5.8))
    create_arrow(ax, (7, 5), (7, 3.8))
    create_arrow(ax, (7, 3), (7, 2.3))

    # Add mathematical notation
    ax.text(7, 0.5, r'$Enhanced = x_{attended} + FeatureAlignment(x_{attended})$',
            ha='center', va='center', fontsize=11, style='italic')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/cortexflow_enhanced_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/cortexflow_enhanced_architecture.svg', bbox_inches='tight')
    plt.close()

def create_cortexflow_ensemble_architecture():
    """Create CortexFlow-Ensemble architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Title
    ax.text(8, 13.5, 'CortexFlow-Ensemble Architecture (8-Variant)',
            ha='center', va='center', fontsize=16, weight='bold')

    # Input layer
    create_layer_box(ax, 7, 12, 2, 0.8, 'fMRI Input\n(Variable Dim)', 'lightgreen')

    # 8 Variants in a grid
    variants = [
        ('CortexFlow-Lite', 'lightblue'),
        ('CortexFlow-MC', 'red'),
        ('CortexFlow-Hierarchical', 'lightyellow'),
        ('CortexFlow-Enhanced', 'orange'),
        ('CortexFlow-Unified', 'lightcoral'),
        ('CortexFlow-Diffusion', 'lightpink'),
        ('CortexFlow-CNN', 'lightgray'),
        ('CortexFlow Multi-Pathway', 'gold')
    ]

    # Create variant boxes in 2x4 grid
    positions = [(1, 10), (4, 10), (7, 10), (10, 10),
                 (1, 8.5), (4, 8.5), (7, 8.5), (10, 8.5)]

    for i, ((name, color), (x, y)) in enumerate(zip(variants, positions)):
        create_layer_box(ax, x, y, 2.5, 0.8, name, color)
        # Input connections
        create_arrow(ax, (8, 12), (x+1.25, y+0.8))

    # Learned weighting network
    create_layer_box(ax, 5, 6.5, 6, 0.8, 'Learned Weighting Network\n512 → 256 → 128 → 8 weights', 'purple', 'white')

    # Softmax normalization
    create_layer_box(ax, 6, 5, 4, 0.8, 'Softmax Normalization\nΣ weights = 1', 'cyan')

    # Weighted combination
    create_layer_box(ax, 6, 3.5, 4, 0.8, 'Weighted Combination\nΣ wᵢ × fᵢ(x)', 'orange')

    # Output
    create_layer_box(ax, 6.5, 2, 3, 0.8, 'Ensemble Output\n784 (28×28)', 'lightgreen')

    # Connections from variants to weighting
    for x, y in positions:
        create_arrow(ax, (x+1.25, y), (7, 7.3))

    # Weighting network connections
    create_arrow(ax, (8, 6.5), (8, 5.8))
    create_arrow(ax, (8, 5), (8, 4.3))
    create_arrow(ax, (8, 3.5), (8, 2.8))

    # Add mathematical notation
    ax.text(8, 0.5, r'$y_{ensemble} = \sum_{i=1}^{8} w_i \cdot f_i(x)$ where $\sum w_i = 1$',
            ha='center', va='center', fontsize=12, style='italic')

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/cortexflow_ensemble_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/cortexflow_ensemble_architecture.svg', bbox_inches='tight')
    plt.close()

def create_brain_diffuser_architecture():
    """Create Brain-Diffuser architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Title
    ax.text(7, 11.5, 'Brain-Diffuser Architecture',
            ha='center', va='center', fontsize=16, weight='bold')

    # Input layer
    create_layer_box(ax, 6, 10, 2, 0.8, 'fMRI Input\n(Variable Dim)', 'lightgreen')

    # Encoder layers
    create_layer_box(ax, 6, 8.5, 2, 0.8, 'Linear Layer\n512 neurons', 'lightblue')
    create_layer_box(ax, 6, 7.5, 2, 0.8, 'SiLU + LayerNorm', 'lightyellow')
    create_layer_box(ax, 6, 6.5, 2, 0.8, 'Linear Layer\n256 neurons', 'lightblue')
    create_layer_box(ax, 6, 5.5, 2, 0.8, 'SiLU + LayerNorm', 'lightyellow')

    # Diffusion process
    create_layer_box(ax, 2, 4, 4, 0.8, 'Noise Addition\nβ Linear Schedule', 'red', 'white')
    create_layer_box(ax, 8, 4, 4, 0.8, 'Denoising Process\n10 Timesteps', 'orange')

    # Iterative refinement
    create_layer_box(ax, 5, 2.5, 4, 0.8, 'Iterative Denoising\nNoise Prediction', 'lightcoral')

    # Output layer
    create_layer_box(ax, 6, 1, 2, 0.8, 'Output Layer\n784 (28×28)', 'lightgreen')

    # Connections
    create_arrow(ax, (7, 10), (7, 9.3))
    create_arrow(ax, (7, 8.5), (7, 8.3))
    create_arrow(ax, (7, 7.5), (7, 7.3))
    create_arrow(ax, (7, 6.5), (7, 6.3))
    create_arrow(ax, (7, 5.5), (4, 4.8))
    create_arrow(ax, (7, 5.5), (10, 4.8))

    create_arrow(ax, (4, 4), (6, 3.3))
    create_arrow(ax, (10, 4), (8, 3.3))
    create_arrow(ax, (7, 2.5), (7, 1.8))

    # Add mathematical notation
    ax.text(7, 0.2, r'$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon$ where $\epsilon \sim \mathcal{N}(0,I)$',
            ha='center', va='center', fontsize=10, style='italic')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/brain_diffuser_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/brain_diffuser_architecture.svg', bbox_inches='tight')
    plt.close()

def create_mindvis_architecture():
    """Create MinD-Vis architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Title
    ax.text(7, 11.5, 'MinD-Vis Architecture (CVPR 2023)',
            ha='center', va='center', fontsize=16, weight='bold')

    # Input layer
    create_layer_box(ax, 6, 10, 2, 0.8, 'fMRI Input\n(Variable Dim)', 'lightgreen')

    # Sparse masked modeling
    create_layer_box(ax, 6, 8.5, 2, 0.8, 'Sparse Masking\n15% Random Mask', 'red', 'white')

    # Encoder layers
    create_layer_box(ax, 6, 7, 2, 0.8, 'Linear Layer\n512 neurons', 'lightblue')
    create_layer_box(ax, 6, 6, 2, 0.8, 'Linear Layer\n256 neurons', 'lightblue')
    create_layer_box(ax, 6, 5, 2, 0.8, 'Linear Layer\n128 neurons', 'lightblue')

    # Conditional diffusion decoder
    create_layer_box(ax, 2, 3.5, 4, 0.8, 'Conditional Diffusion\nDecoder', 'orange')
    create_layer_box(ax, 8, 3.5, 4, 0.8, 'Noise Injection\nDiffusion Simulation', 'lightcoral')

    # Reconstruction
    create_layer_box(ax, 5, 2, 4, 0.8, 'Image Reconstruction\nConditional Generation', 'lightyellow')

    # Output layer
    create_layer_box(ax, 6, 0.5, 2, 0.8, 'Output Layer\n784 (28×28)', 'lightgreen')

    # Connections
    create_arrow(ax, (7, 10), (7, 9.3))
    create_arrow(ax, (7, 8.5), (7, 7.8))
    create_arrow(ax, (7, 7), (7, 6.8))
    create_arrow(ax, (7, 6), (7, 5.8))
    create_arrow(ax, (7, 5), (4, 4.3))
    create_arrow(ax, (7, 5), (10, 4.3))

    create_arrow(ax, (4, 3.5), (6, 2.8))
    create_arrow(ax, (10, 3.5), (8, 2.8))
    create_arrow(ax, (7, 2), (7, 1.3))

    # Add mathematical notation
    ax.text(7, -0.2, r'$p(x|c) = \int p(x|z,c) p(z|c) dz$ where $c$ is fMRI condition',
            ha='center', va='center', fontsize=10, style='italic')

    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/mindvis_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/mindvis_architecture.svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("🎨 Creating CortexFlow Architecture Figures...")

    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Generate architecture diagrams
    print("📊 Creating CortexFlow-Lite architecture...")
    create_cortexflow_lite_architecture()

    print("📊 Creating CortexFlow-MC architecture...")
    create_cortexflow_mc_architecture()

    print("📊 Creating CortexFlow-Hierarchical architecture...")
    create_cortexflow_hierarchical_architecture()

    print("📊 Creating CortexFlow Multi-Pathway architecture...")
    create_cortexflow_multipathway_architecture()

    print("📊 Creating CortexFlow-Enhanced architecture...")
    create_cortexflow_enhanced_architecture()

    print("📊 Creating CortexFlow-Ensemble architecture...")
    create_cortexflow_ensemble_architecture()

    print("📊 Creating Brain-Diffuser architecture...")
    create_brain_diffuser_architecture()

    print("📊 Creating MinD-Vis architecture...")
    create_mindvis_architecture()

    print("✅ All architecture figures created successfully!")
    print("📁 Saved in figures/ directory as PNG and SVG formats")
