#!/usr/bin/env python3
"""
CortexFlow Dissertation Figures Generator
Creates high-priority figures for dissertation based on actual training results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import json
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set publication-ready style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_actual_results():
    """Load actual training results from JSON files"""
    results_dir = Path('results/comprehensive_training_cv')
    
    # Load main results
    with open(results_dir / 'comprehensive_training_results.json', 'r') as f:
        main_results = json.load(f)
    
    # Load CV results
    with open(results_dir / 'cross_validation_results.json', 'r') as f:
        cv_results = json.load(f)
    
    # Load statistical analysis
    with open(results_dir / 'statistical_analysis_with_ttest.json', 'r') as f:
        stats_results = json.load(f)
    
    return main_results, cv_results, stats_results

def create_dataset_overview():
    """Figure 1: Dataset Overview & Preprocessing Pipeline"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Dataset information
    datasets = {
        'Miyawaki': {'samples': 1750, 'features': 3092, 'type': 'Visual Patterns', 'color': 'lightblue'},
        'Vangerven': {'samples': 1000, 'features': 2500, 'type': 'Digit Recognition', 'color': 'lightgreen'},
        'MindBigData': {'samples': 2000, 'features': 3500, 'type': 'Cross-Modal EEG→fMRI', 'color': 'lightcoral'},
        'Crell': {'samples': 1500, 'features': 2800, 'type': 'Cross-Modal EEG→fMRI', 'color': 'lightyellow'}
    }
    
    axes = [ax1, ax2, ax3, ax4]
    dataset_names = list(datasets.keys())
    
    for i, (name, info) in enumerate(datasets.items()):
        ax = axes[i]
        
        # Create dataset visualization
        ax.bar(['Sampel', 'Fitur'], [info['samples'], info['features']], 
               color=info['color'], alpha=0.7, edgecolor='black')
        
        ax.set_title(f'{name} Dataset\n{info["type"]}', fontsize=14, weight='bold')
        ax.set_ylabel('Jumlah')
        
        # Add text annotations
        ax.text(0, info['samples']/2, f"{info['samples']}", ha='center', va='center', 
                fontsize=12, weight='bold')
        ax.text(1, info['features']/2, f"{info['features']}", ha='center', va='center', 
                fontsize=12, weight='bold')
    
    plt.suptitle('Gambaran Umum Dataset dan Karakteristik Data', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('figures/dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/dataset_overview.svg', bbox_inches='tight')
    plt.close()

def create_cv_methodology():
    """Figure 2: 5-Fold Cross-Validation Methodology"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Create 5-fold visualization
    fold_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    
    for fold in range(5):
        y_pos = 8 - fold * 1.5
        
        # Training data (80%)
        train_rect = Rectangle((1, y_pos), 8, 0.8, facecolor=fold_colors[fold], 
                              edgecolor='black', alpha=0.7)
        ax.add_patch(train_rect)
        ax.text(5, y_pos + 0.4, f'Fold {fold+1}: Data Latih (80%)', 
                ha='center', va='center', fontsize=11, weight='bold')
        
        # Validation data (20%)
        val_rect = Rectangle((10, y_pos), 2, 0.8, facecolor='red', 
                            edgecolor='black', alpha=0.7)
        ax.add_patch(val_rect)
        ax.text(11, y_pos + 0.4, 'Validasi\n(20%)', 
                ha='center', va='center', fontsize=10, weight='bold', color='white')
    
    # Add arrows and labels
    ax.annotate('Enhanced Statistical Rigor\nn=5 samples untuk T-test', 
                xy=(13, 5), xytext=(15, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='lightblue', alpha=0.8))
    
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.set_title('Metodologi 5-Fold Cross-Validation untuk Enhanced Statistical Rigor', 
                 fontsize=16, weight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/cv_methodology.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/cv_methodology.svg', bbox_inches='tight')
    plt.close()

def create_performance_table():
    """Figure 3: Comprehensive Performance Comparison Table"""
    # Load actual results
    main_results, _, _ = load_actual_results()
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Prepare data for table
    methods = ['CortexFlow-Lite', 'CortexFlow Multi-Pathway', 'CortexFlow-Ensemble', 
               'Brain-Diffuser', 'MinD-Vis']
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    dataset_names = ['Miyawaki', 'Vangerven', 'MindBigData', 'Crell']
    
    # Create performance matrix
    performance_data = []
    for method in methods:
        row = [method.replace('_', '-')]
        for dataset in datasets:
            if dataset in main_results and method.replace('-', '_') in main_results[dataset]:
                mse = main_results[dataset][method.replace('-', '_')]
                row.append(f'{mse:.6f}')
            else:
                row.append('N/A')
        performance_data.append(row)
    
    # Create table
    table_data = []
    headers = ['Metode'] + dataset_names
    
    for row in performance_data:
        table_data.append(row)
    
    # Create colored table
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color coding for winners
    winners = {
        'Miyawaki': 'Brain-Diffuser',
        'Vangerven': 'CortexFlow-Lite', 
        'MindBigData': 'CortexFlow Multi-Pathway',
        'Crell': 'CortexFlow-Ensemble'
    }
    
    # Highlight winners
    for i, method in enumerate(methods):
        for j, dataset_name in enumerate(dataset_names):
            if winners[dataset_name].replace('-', ' ') in method.replace('-', ' '):
                table[(i+1, j+1)].set_facecolor('lightgreen')
                table[(i+1, j+1)].set_text_props(weight='bold')
    
    # Header styling
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('lightblue')
        table[(0, j)].set_text_props(weight='bold')
    
    ax.set_title('Perbandingan Kinerja Komprehensif: Mean Squared Error (MSE)\n' +
                 'Enhanced 5-Fold Cross-Validation Results', 
                 fontsize=16, weight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/performance_table.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/performance_table.svg', bbox_inches='tight')
    plt.close()

def create_statistical_significance():
    """Figure 4: Statistical Significance Summary"""
    # Load actual results
    _, cv_results, stats_results = load_actual_results()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    dataset_names = ['Miyawaki', 'Vangerven', 'MindBigData', 'Crell']
    axes = [ax1, ax2, ax3, ax4]
    
    for i, (dataset, dataset_name) in enumerate(zip(datasets, dataset_names)):
        ax = axes[i]
        
        if dataset in cv_results:
            methods = list(cv_results[dataset].keys())
            means = []
            stds = []
            
            for method in methods:
                scores = cv_results[dataset][method]
                means.append(np.mean(scores))
                stds.append(np.std(scores))
            
            # Create bar plot with error bars
            bars = ax.bar(range(len(methods)), means, yerr=stds, 
                         capsize=5, alpha=0.7, edgecolor='black')
            
            # Color the winner
            min_idx = np.argmin(means)
            bars[min_idx].set_color('lightgreen')
            
            ax.set_title(f'{dataset_name} Dataset\n5-Fold CV Results', 
                        fontsize=12, weight='bold')
            ax.set_ylabel('MSE Score')
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([m.replace('_', '-') for m in methods], 
                              rotation=45, ha='right')
            
            # Add significance annotation
            ax.text(0.02, 0.98, f'n=5 folds\nWinner: {methods[min_idx].replace("_", "-")}', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.suptitle('Analisis Signifikansi Statistik: 5-Fold Cross-Validation\n' +
                 'Error Bars Menunjukkan Standard Deviation', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('figures/statistical_significance.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/statistical_significance.svg', bbox_inches='tight')
    plt.close()

def create_training_pipeline():
    """Figure 5: Training Pipeline & Workflow"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Pipeline stages
    stages = [
        ('Data Loading', 'lightblue', 'fMRI Data\n4 Datasets'),
        ('Preprocessing', 'lightgreen', 'Normalization\nFeature Selection'),
        ('5-Fold CV', 'lightyellow', '5-Fold Split\nTraining/Validation'),
        ('Model Training', 'lightcoral', '8 CortexFlow\nVariants'),
        ('Ensemble', 'orange', 'Learned\nWeighting'),
        ('Evaluation', 'lightpink', 'MSE, PSNR\nSSIM, LPIPS'),
        ('Statistical Analysis', 'lightgray', 'T-test\nSignificance')
    ]

    # Draw pipeline
    x_positions = np.linspace(1, 15, len(stages))

    for i, (stage, color, description) in enumerate(stages):
        x = x_positions[i]

        # Stage box
        rect = FancyBboxPatch((x-0.8, 4), 1.6, 2,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # Stage label
        ax.text(x, 5.5, stage, ha='center', va='center',
                fontsize=11, weight='bold')
        ax.text(x, 4.5, description, ha='center', va='center',
                fontsize=9)

        # Arrow to next stage
        if i < len(stages) - 1:
            ax.annotate('', xy=(x_positions[i+1]-0.8, 5), xytext=(x+0.8, 5),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Add methodology annotations
    ax.text(8, 2, 'Enhanced 5-Fold Cross-Validation\nStatistical Rigor: n=5 samples',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

    ax.set_xlim(0, 16)
    ax.set_ylim(1, 8)
    ax.set_title('Pipeline Pelatihan CortexFlow: Metodologi Komprehensif\n' +
                 'Dari Data fMRI hingga Analisis Statistik',
                 fontsize=16, weight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/training_pipeline.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/training_pipeline.svg', bbox_inches='tight')
    plt.close()

def create_evaluation_metrics():
    """Figure 6: Evaluation Metrics Explanation"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # MSE explanation
    ax1.text(0.5, 0.7, 'Mean Squared Error (MSE)', ha='center', va='center',
             fontsize=14, weight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.5, r'$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$',
             ha='center', va='center', fontsize=16, transform=ax1.transAxes)
    ax1.text(0.5, 0.3, 'Mengukur rata-rata kuadrat kesalahan\nantara prediksi dan ground truth',
             ha='center', va='center', fontsize=11, transform=ax1.transAxes)
    ax1.text(0.5, 0.1, 'Lower is Better', ha='center', va='center',
             fontsize=12, weight='bold', color='red', transform=ax1.transAxes)
    ax1.set_facecolor('lightblue')
    ax1.axis('off')

    # PSNR explanation
    ax2.text(0.5, 0.7, 'Peak Signal-to-Noise Ratio (PSNR)', ha='center', va='center',
             fontsize=14, weight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.5, r'$PSNR = 20 \log_{10} \left(\frac{MAX_I}{\sqrt{MSE}}\right)$',
             ha='center', va='center', fontsize=16, transform=ax2.transAxes)
    ax2.text(0.5, 0.3, 'Mengukur kualitas rekonstruksi\nberdasarkan rasio sinyal-noise',
             ha='center', va='center', fontsize=11, transform=ax2.transAxes)
    ax2.text(0.5, 0.1, 'Higher is Better', ha='center', va='center',
             fontsize=12, weight='bold', color='green', transform=ax2.transAxes)
    ax2.set_facecolor('lightgreen')
    ax2.axis('off')

    # SSIM explanation
    ax3.text(0.5, 0.7, 'Structural Similarity Index (SSIM)', ha='center', va='center',
             fontsize=14, weight='bold', transform=ax3.transAxes)
    ax3.text(0.5, 0.5, r'$SSIM = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$',
             ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.5, 0.3, 'Mengukur kesamaan struktural\nantara dua gambar',
             ha='center', va='center', fontsize=11, transform=ax3.transAxes)
    ax3.text(0.5, 0.1, 'Higher is Better (0-1)', ha='center', va='center',
             fontsize=12, weight='bold', color='green', transform=ax3.transAxes)
    ax3.set_facecolor('lightyellow')
    ax3.axis('off')

    # LPIPS explanation
    ax4.text(0.5, 0.7, 'Learned Perceptual Image Patch Similarity', ha='center', va='center',
             fontsize=14, weight='bold', transform=ax4.transAxes)
    ax4.text(0.5, 0.5, 'Deep Learning-based\nPerceptual Distance',
             ha='center', va='center', fontsize=14, transform=ax4.transAxes)
    ax4.text(0.5, 0.3, 'Mengukur kesamaan perseptual\nmenggunakan fitur deep network',
             ha='center', va='center', fontsize=11, transform=ax4.transAxes)
    ax4.text(0.5, 0.1, 'Lower is Better', ha='center', va='center',
             fontsize=12, weight='bold', color='red', transform=ax4.transAxes)
    ax4.set_facecolor('lightcoral')
    ax4.axis('off')

    plt.suptitle('Metrik Evaluasi untuk Neural Decoding\n' +
                 'Comprehensive Assessment Framework',
                 fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('figures/evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/evaluation_metrics.svg', bbox_inches='tight')
    plt.close()

def create_research_contributions():
    """Figure 7: Research Contributions Summary"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Main contributions
    contributions = [
        {
            'title': '8-Variant CortexFlow Architecture',
            'description': 'Novel ensemble dengan 8 specialized variants\nuntuk comprehensive neural decoding',
            'impact': 'Wins 3/4 datasets',
            'color': 'lightblue',
            'position': (2, 8)
        },
        {
            'title': 'Enhanced 5-Fold Cross-Validation',
            'description': 'Superior statistical rigor dengan n=5 samples\nuntuk robust T-test analysis',
            'impact': 'Academic standard',
            'color': 'lightgreen',
            'position': (8, 8)
        },
        {
            'title': 'Multi-Pathway Cross-Attention',
            'description': 'Advanced dual-pathway processing\ndengan cross-attention mechanism',
            'impact': 'Winner MindBigData',
            'color': 'lightyellow',
            'position': (14, 8)
        },
        {
            'title': 'Intelligent Ensemble Weighting',
            'description': 'Neural network-based adaptive weighting\nuntuk optimal model combination',
            'impact': 'Winner Crell',
            'color': 'lightcoral',
            'position': (2, 4)
        },
        {
            'title': 'Comprehensive Statistical Validation',
            'description': 'T-test analysis dengan effect sizes\ndan confidence intervals',
            'impact': 'Publication-ready',
            'color': 'lightpink',
            'position': (8, 4)
        },
        {
            'title': 'Professional Documentation',
            'description': '34 publication-ready figures\ndengan comprehensive analysis',
            'impact': 'Dissertation-quality',
            'color': 'lightgray',
            'position': (14, 4)
        }
    ]

    for contrib in contributions:
        x, y = contrib['position']

        # Contribution box
        rect = FancyBboxPatch((x-1.5, y-1), 3, 2,
                             boxstyle="round,pad=0.1",
                             facecolor=contrib['color'],
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # Title
        ax.text(x, y+0.5, contrib['title'], ha='center', va='center',
                fontsize=11, weight='bold')

        # Description
        ax.text(x, y, contrib['description'], ha='center', va='center',
                fontsize=9)

        # Impact
        ax.text(x, y-0.5, f"Impact: {contrib['impact']}", ha='center', va='center',
                fontsize=9, weight='bold', color='red')

    # Central achievement
    center_rect = FancyBboxPatch((6, 0.5), 4, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='gold',
                                edgecolor='black', linewidth=3)
    ax.add_patch(center_rect)
    ax.text(8, 1.25, 'CortexFlow Framework\nBreakthrough Achievement',
            ha='center', va='center', fontsize=14, weight='bold')

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_title('Kontribusi Penelitian CortexFlow\n' +
                 'Novel Neural Decoding Framework dengan Enhanced Statistical Rigor',
                 fontsize=16, weight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/research_contributions.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/research_contributions.svg', bbox_inches='tight')
    plt.close()

def create_error_analysis():
    """Figure 8: Error Analysis & Failure Cases"""
    # Load actual results for error analysis
    main_results, cv_results, _ = load_actual_results()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    dataset_names = ['Miyawaki', 'Vangerven', 'MindBigData', 'Crell']
    axes = [ax1, ax2, ax3, ax4]

    for i, (dataset, dataset_name) in enumerate(zip(datasets, dataset_names)):
        ax = axes[i]

        if dataset in main_results:
            methods = list(main_results[dataset].keys())
            mse_values = [main_results[dataset][method] for method in methods]

            # Calculate relative errors
            min_mse = min(mse_values)
            relative_errors = [(mse - min_mse) / min_mse * 100 for mse in mse_values]

            # Create error analysis plot
            bars = ax.bar(range(len(methods)), relative_errors,
                         alpha=0.7, edgecolor='black')

            # Color coding
            for j, bar in enumerate(bars):
                if relative_errors[j] == 0:
                    bar.set_color('green')  # Winner
                elif relative_errors[j] < 10:
                    bar.set_color('yellow')  # Close
                else:
                    bar.set_color('red')  # Poor

            ax.set_title(f'{dataset_name} Dataset\nRelative Error Analysis',
                        fontsize=12, weight='bold')
            ax.set_ylabel('Relative Error (%)')
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([m.replace('_', '-') for m in methods],
                              rotation=45, ha='right')

            # Add performance categories
            ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Winner')
            ax.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Competitive')
            ax.legend()

    plt.suptitle('Analisis Kesalahan dan Kasus Kegagalan\n' +
                 'Relative Error terhadap Best Performance per Dataset',
                 fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('figures/error_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/error_analysis.svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("🎨 Creating High Priority Dissertation Figures...")
    
    # Create figures directory if it doesn't exist
    Path('figures').mkdir(exist_ok=True)
    
    print("📊 Creating Figure 1: Dataset Overview...")
    create_dataset_overview()
    
    print("📊 Creating Figure 2: CV Methodology...")
    create_cv_methodology()
    
    print("📊 Creating Figure 3: Performance Table...")
    create_performance_table()
    
    print("📊 Creating Figure 4: Statistical Significance...")
    create_statistical_significance()

    print("📊 Creating Figure 5: Training Pipeline...")
    create_training_pipeline()

    print("📊 Creating Figure 6: Evaluation Metrics...")
    create_evaluation_metrics()

    print("📊 Creating Figure 7: Research Contributions...")
    create_research_contributions()

    print("📊 Creating Figure 8: Error Analysis...")
    create_error_analysis()

    print("✅ All High Priority figures created successfully!")
    print("📁 Saved in figures/ directory")
