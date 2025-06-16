#!/usr/bin/env python3
"""
CortexFlow Methodology Visual Elements Generator
Creates tables, figures, and algorithms for enhanced methodology documentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import json

# Set publication-ready style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def create_dataset_characteristics_table():
    """Create comprehensive dataset characteristics table"""
    
    # Dataset information
    data = {
        'Dataset': ['Miyawaki', 'Vangerven', 'MindBigData', 'Crell'],
        'Type': ['Visual Patterns', 'Digit Recognition', 'Cross-Modal EEG→fMRI', 'Cross-Modal EEG→fMRI'],
        'Training Samples': [1750, 1000, 2000, 1500],
        'Test Samples': [350, 200, 400, 300],
        'Input Features': [3092, 2500, 3500, 2800],
        'Output Dimension': ['28×28', '28×28', '28×28', '28×28'],
        'Preprocessing': ['Z-score + Binary', 'Normalization [0,1]', 'Multi-modal Align', 'Cross-modal Sync']
    }
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.12, 0.18, 0.12, 0.12, 0.12, 0.12, 0.22])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Tabel 1. Karakteristik Dataset Neural Decoding', 
                 fontsize=14, weight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/methodology_table_datasets.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/methodology_table_datasets.svg', bbox_inches='tight')
    plt.close()

def create_architecture_specifications_table():
    """Create model architecture specifications table"""
    
    # Architecture data
    data = {
        'Model': ['CortexFlow-Lite', 'CortexFlow-MC', 'CortexFlow-Hierarchical', 
                 'CortexFlow-Enhanced', 'CortexFlow-Multi-Pathway', 'CortexFlow-Ensemble',
                 'MinD-Vis', 'Brain-Diffuser'],
        'Architecture': ['1024→512→784', '512→256→128→784', '3-Level Hierarchy', 
                        'MC+Hierarchical+Align', 'Dual-Pathway+Attention', '8-Variant Ensemble',
                        '512→256→128→784', '512→256→784'],
        'Key Features': ['BatchNorm+Dropout', 'MCDropout+LayerNorm', 'Temporal Attention',
                        'Multi-Component', 'Cross-Attention', 'Learned Weighting',
                        'Sparse Masking', 'Diffusion Process'],
        'Parameters': ['~2.1M', '~1.8M', '~2.5M', '~3.2M', '~2.8M', '~15.6M', '~1.9M', '~1.7M'],
        'Dropout Rate': ['0.3, 0.2', '0.15 (MC)', '0.3, 0.2, 0.1', 'Adaptive', 'Adaptive', 'Ensemble', '0.15', '0.1'],
        'Normalization': ['BatchNorm1d', 'LayerNorm', 'LayerNorm', 'Mixed', 'LayerNorm', 'Mixed', 'LayerNorm', 'LayerNorm']
    }
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.18, 0.18, 0.18, 0.12, 0.14, 0.14])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.2)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight CortexFlow models
    cortexflow_rows = [1, 2, 3, 4, 5, 6]  # CortexFlow models
    for row in cortexflow_rows:
        for col in range(len(df.columns)):
            table[(row, col)].set_facecolor('#E3F2FD')
    
    ax.set_title('Tabel 2. Spesifikasi Arsitektur Model Neural Decoding', 
                 fontsize=14, weight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/methodology_table_architectures.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/methodology_table_architectures.svg', bbox_inches='tight')
    plt.close()

def create_hyperparameter_configuration_table():
    """Create hyperparameter configuration table"""
    
    # Load actual configuration data
    configs = {
        'miyawaki': {'epochs': 150, 'lr': 0.001, 'batch_size': 64, 'patience': 20},
        'vangerven': {'epochs': 120, 'lr': 0.0015, 'batch_size': 32, 'patience': 15},
        'mindbigdata': {'epochs': 100, 'lr': 0.002, 'batch_size': 48, 'patience': 12},
        'crell': {'epochs': 130, 'lr': 0.0012, 'batch_size': 40, 'patience': 18}
    }
    
    # Create DataFrame
    data = {
        'Dataset': ['Miyawaki', 'Vangerven', 'MindBigData', 'Crell'],
        'Epochs': [configs[ds]['epochs'] for ds in ['miyawaki', 'vangerven', 'mindbigdata', 'crell']],
        'Learning Rate': [configs[ds]['lr'] for ds in ['miyawaki', 'vangerven', 'mindbigdata', 'crell']],
        'Batch Size': [configs[ds]['batch_size'] for ds in ['miyawaki', 'vangerven', 'mindbigdata', 'crell']],
        'Patience': [configs[ds]['patience'] for ds in ['miyawaki', 'vangerven', 'mindbigdata', 'crell']],
        'Optimizer': ['Adam', 'Adam', 'Adam', 'Adam'],
        'Weight Decay': ['1e-4', '1e-4', '1e-4', '1e-4'],
        'Scheduler': ['ReduceLROnPlateau', 'ReduceLROnPlateau', 'ReduceLROnPlateau', 'ReduceLROnPlateau']
    }
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.12, 0.1, 0.12, 0.1, 0.1, 0.12, 0.12, 0.22])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#FF9800')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#FFF3E0')
    
    ax.set_title('Tabel 3. Konfigurasi Hyperparameter per Dataset', 
                 fontsize=14, weight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/methodology_table_hyperparameters.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/methodology_table_hyperparameters.svg', bbox_inches='tight')
    plt.close()

def create_enhanced_methodology_flowchart():
    """Create enhanced methodology flowchart"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Define flowchart components
    components = [
        # Data Layer
        {'name': 'Dataset Loading', 'pos': (2, 10), 'size': (2.5, 1), 'color': 'lightblue', 
         'details': '4 Datasets\nMiyawaki, Vangerven\nMindBigData, Crell'},
        {'name': 'Preprocessing', 'pos': (6, 10), 'size': (2.5, 1), 'color': 'lightgreen',
         'details': 'GPU Normalization\nFeature Alignment\nData Validation'},
        
        # CV Layer
        {'name': '5-Fold CV Split', 'pos': (10, 10), 'size': (2.5, 1), 'color': 'lightyellow',
         'details': 'Enhanced Statistical\nRigor: n=5 samples\nRandom Shuffling'},
        
        # Model Layer
        {'name': 'CortexFlow Training', 'pos': (2, 7), 'size': (3, 1.5), 'color': 'lightcoral',
         'details': '8 Variants:\nLite, MC, Hierarchical\nEnhanced, Multi-Pathway\nUnified, Diffusion, CNN'},
        {'name': 'SOTA Baselines', 'pos': (6, 7), 'size': (3, 1.5), 'color': 'lightpink',
         'details': 'MinD-Vis (CVPR 2023)\nBrain-Diffuser (2023)\nComparative Analysis'},
        {'name': 'Ensemble Learning', 'pos': (10, 7), 'size': (3, 1.5), 'color': 'orange',
         'details': 'Learned Weighting\nIntelligent Combination\nAdaptive Fusion'},
        
        # Evaluation Layer
        {'name': 'Multi-Metric Evaluation', 'pos': (4, 4), 'size': (3, 1.5), 'color': 'lightsteelblue',
         'details': 'MSE, PSNR\nSSIM, LPIPS\nComprehensive Assessment'},
        {'name': 'Statistical Analysis', 'pos': (8, 4), 'size': (3, 1.5), 'color': 'plum',
         'details': 'T-test Analysis\nEffect Size (Cohen\'s d)\nConfidence Intervals'},
        
        # Output Layer
        {'name': 'Results & Visualization', 'pos': (6, 1), 'size': (4, 1.5), 'color': 'gold',
         'details': 'Performance Tables\nStatistical Significance\nReconstruction Figures'}
    ]
    
    # Draw components
    for comp in components:
        x, y = comp['pos']
        w, h = comp['size']
        
        # Main box
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                             boxstyle="round,pad=0.1",
                             facecolor=comp['color'],
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Title
        ax.text(x, y+0.3, comp['name'], ha='center', va='center',
                fontsize=11, weight='bold')
        
        # Details
        ax.text(x, y-0.2, comp['details'], ha='center', va='center',
                fontsize=8)
    
    # Draw arrows
    arrows = [
        # Horizontal flow
        ((3.25, 10), (4.75, 10)),  # Dataset -> Preprocessing
        ((7.25, 10), (8.75, 10)),  # Preprocessing -> CV Split
        
        # Vertical flow from CV
        ((10, 9), (10, 8.75)),     # CV -> Ensemble
        ((10, 9), (6, 8.75)),      # CV -> SOTA
        ((10, 9), (2, 8.75)),      # CV -> CortexFlow
        
        # To evaluation
        ((3.5, 6.25), (4.5, 5.75)),   # CortexFlow -> Evaluation
        ((7.5, 6.25), (7.5, 5.75)),   # SOTA -> Statistical
        ((10, 6.25), (9, 5.75)),      # Ensemble -> Statistical
        
        # To results
        ((5.5, 3.25), (6, 2.75)),     # Evaluation -> Results
        ((8.5, 3.25), (8, 2.75)),     # Statistical -> Results
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Add methodology phases
    ax.text(1, 11.5, 'Phase 1: Data Preparation', fontsize=12, weight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    ax.text(1, 8.5, 'Phase 2: Model Training', fontsize=12, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    ax.text(1, 5.5, 'Phase 3: Evaluation', fontsize=12, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    ax.text(1, 2.5, 'Phase 4: Analysis', fontsize=12, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.set_title('Gambar 1. Enhanced Methodology Flowchart CortexFlow Neural Decoding Framework',
                 fontsize=16, weight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/methodology_flowchart_enhanced.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/methodology_flowchart_enhanced.svg', bbox_inches='tight')
    plt.close()

def create_cross_validation_diagram():
    """Create detailed 5-fold cross-validation diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Draw 5-fold visualization
    fold_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Data representation
    total_width = 10
    fold_width = total_width / 5
    
    for fold in range(5):
        y_pos = 8 - fold * 1.5
        
        # Draw all 5 segments
        for segment in range(5):
            x_start = 2 + segment * fold_width
            
            if segment == fold:
                # Validation fold (20%)
                rect = Rectangle((x_start, y_pos), fold_width, 0.8, 
                               facecolor='red', alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
                ax.text(x_start + fold_width/2, y_pos + 0.4, 'Val', 
                       ha='center', va='center', fontsize=9, weight='bold', color='white')
            else:
                # Training fold (80%)
                rect = Rectangle((x_start, y_pos), fold_width, 0.8, 
                               facecolor=fold_colors[fold], alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
                ax.text(x_start + fold_width/2, y_pos + 0.4, 'Train', 
                       ha='center', va='center', fontsize=8, weight='bold')
        
        # Fold label
        ax.text(1, y_pos + 0.4, f'Fold {fold+1}', ha='center', va='center', 
               fontsize=11, weight='bold')
        
        # Performance arrow
        ax.annotate(f'Score {fold+1}', xy=(13, y_pos + 0.4), xytext=(12.5, y_pos + 0.4),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'),
                   fontsize=10, ha='center')
    
    # Statistical analysis box
    stats_box = FancyBboxPatch((2, 0.5), 10, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(stats_box)
    
    ax.text(7, 1.6, 'Enhanced Statistical Analysis', ha='center', va='center',
           fontsize=12, weight='bold')
    ax.text(7, 1, 'n=5 samples → T-test Analysis → Effect Size (Cohen\'s d) → 95% CI',
           ha='center', va='center', fontsize=10)
    ax.text(7, 0.7, 'Statistical Rigor: p-value < 0.05, Power Analysis, Significance Testing',
           ha='center', va='center', fontsize=9, style='italic')
    
    # Arrows to statistical analysis
    for fold in range(5):
        y_pos = 8 - fold * 1.5
        ax.annotate('', xy=(7, 2), xytext=(13, y_pos + 0.4),
                   arrowprops=dict(arrowstyle='->', lw=1, color='green', alpha=0.7))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_title('Gambar 2. Enhanced 5-Fold Cross-Validation dengan Statistical Rigor\n' +
                 'Systematic Data Splitting untuk Robust Model Evaluation',
                 fontsize=14, weight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/methodology_cv_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/methodology_cv_diagram.svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("🎨 Creating Enhanced Methodology Visual Elements...")
    
    # Create figures directory if it doesn't exist
    Path('figures').mkdir(exist_ok=True)
    
    print("📊 Creating Table 1: Dataset Characteristics...")
    create_dataset_characteristics_table()
    
    print("📊 Creating Table 2: Architecture Specifications...")
    create_architecture_specifications_table()
    
    print("📊 Creating Table 3: Hyperparameter Configuration...")
    create_hyperparameter_configuration_table()
    
    print("🎨 Creating Figure 1: Enhanced Methodology Flowchart...")
    create_enhanced_methodology_flowchart()
    
    print("🎨 Creating Figure 2: Cross-Validation Diagram...")
    create_cross_validation_diagram()
    
    print("✅ Enhanced methodology visual elements created successfully!")
    print("📁 Saved in figures/ directory")
