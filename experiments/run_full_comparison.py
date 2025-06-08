#!/usr/bin/env python3
"""
🚀 RUN FULL CORTEXFLOW COMPARISON
================================================================================
Execute comprehensive comparison across all datasets and architectures
================================================================================
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from datetime import datetime

def run_full_comparison():
    """Run the full comparison training."""
    print("🚀 STARTING FULL CORTEXFLOW COMPARISON")
    print("="*80)
    
    try:
        # Run the comparison training
        result = subprocess.run([
            'python3', 'experiments/full_comparison_training.py'
        ], capture_output=True, text=True, cwd='.')
        
        print(result.stdout)
        if result.stderr:
            print("⚠️ Warnings/Errors:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running comparison: {e}")
        return False

def create_comparison_visualizations():
    """Create comprehensive comparison visualizations."""
    results_file = Path("results/full_comparison/full_comparison_results.json")
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    print("📊 Creating comparison visualizations...")
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Prepare data for visualization
    comparison_data = []
    for dataset_name, dataset_results in results.items():
        for arch_name, arch_results in dataset_results.items():
            if 'error' not in arch_results:
                comparison_data.append({
                    'Dataset': dataset_name.title(),
                    'Architecture': arch_name.replace('_', ' ').title(),
                    'Best Loss': arch_results['best_loss'],
                    'Training Time (min)': arch_results['training_time'] / 60,
                    'Parameters': arch_results['parameters'],
                    'Epochs': arch_results['final_epoch']
                })
    
    df = pd.DataFrame(comparison_data)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance Comparison (Best Loss)
    plt.subplot(2, 3, 1)
    pivot_loss = df.pivot(index='Dataset', columns='Architecture', values='Best Loss')
    sns.heatmap(pivot_loss, annot=True, fmt='.4f', cmap='RdYlBu_r', cbar_kws={'label': 'Best Loss'})
    plt.title('🏆 Performance Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    plt.xlabel('Architecture')
    plt.ylabel('Dataset')
    
    # 2. Training Time Comparison
    plt.subplot(2, 3, 2)
    pivot_time = df.pivot(index='Dataset', columns='Architecture', values='Training Time (min)')
    sns.heatmap(pivot_time, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Training Time (min)'})
    plt.title('⏱️ Training Time Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Architecture')
    plt.ylabel('Dataset')
    
    # 3. Model Complexity (Parameters)
    plt.subplot(2, 3, 3)
    pivot_params = df.pivot(index='Dataset', columns='Architecture', values='Parameters')
    sns.heatmap(pivot_params, annot=True, fmt='.0f', cmap='Purples', cbar_kws={'label': 'Parameters'})
    plt.title('🔧 Model Complexity\n(Parameter Count)', fontsize=14, fontweight='bold')
    plt.xlabel('Architecture')
    plt.ylabel('Dataset')
    
    # 4. Performance vs Complexity Scatter
    plt.subplot(2, 3, 4)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    arch_colors = {arch: colors[i] for i, arch in enumerate(df['Architecture'].unique())}
    
    for arch in df['Architecture'].unique():
        arch_data = df[df['Architecture'] == arch]
        plt.scatter(arch_data['Parameters'], arch_data['Best Loss'], 
                   label=arch, alpha=0.7, s=100, color=arch_colors[arch])
    
    plt.xlabel('Parameters')
    plt.ylabel('Best Loss')
    plt.title('📊 Performance vs Complexity', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Training Efficiency (Performance per Time)
    plt.subplot(2, 3, 5)
    df['Efficiency'] = 1 / (df['Best Loss'] * df['Training Time (min)'])
    
    for arch in df['Architecture'].unique():
        arch_data = df[df['Architecture'] == arch]
        plt.scatter(arch_data['Training Time (min)'], arch_data['Best Loss'], 
                   label=arch, alpha=0.7, s=100, color=arch_colors[arch])
    
    plt.xlabel('Training Time (min)')
    plt.ylabel('Best Loss')
    plt.title('⚡ Training Efficiency', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Overall Rankings
    plt.subplot(2, 3, 6)
    
    # Calculate rankings
    rankings = []
    for dataset in df['Dataset'].unique():
        dataset_data = df[df['Dataset'] == dataset].copy()
        dataset_data['Loss Rank'] = dataset_data['Best Loss'].rank()
        dataset_data['Time Rank'] = dataset_data['Training Time (min)'].rank()
        dataset_data['Overall Rank'] = (dataset_data['Loss Rank'] + dataset_data['Time Rank']) / 2
        rankings.append(dataset_data)
    
    ranking_df = pd.concat(rankings)
    pivot_rank = ranking_df.pivot(index='Dataset', columns='Architecture', values='Overall Rank')
    sns.heatmap(pivot_rank, annot=True, fmt='.1f', cmap='RdYlGn_r', cbar_kws={'label': 'Overall Rank'})
    plt.title('🏅 Overall Rankings\n(Lower is Better)', fontsize=14, fontweight='bold')
    plt.xlabel('Architecture')
    plt.ylabel('Dataset')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = Path("results/full_comparison/comparison_visualization.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {viz_path}")
    
    # Create summary table
    create_summary_table(df)
    
    plt.show()

def create_summary_table(df):
    """Create a summary table of results."""
    print("\n📋 FULL COMPARISON SUMMARY TABLE")
    print("="*120)
    
    # Best performance per dataset
    print("\n🏆 BEST PERFORMANCE PER DATASET:")
    print("-" * 60)
    for dataset in df['Dataset'].unique():
        dataset_data = df[df['Dataset'] == dataset]
        best_row = dataset_data.loc[dataset_data['Best Loss'].idxmin()]
        print(f"{dataset:12} | {best_row['Architecture']:25} | Loss: {best_row['Best Loss']:.6f}")
    
    # Fastest training per dataset
    print("\n⚡ FASTEST TRAINING PER DATASET:")
    print("-" * 60)
    for dataset in df['Dataset'].unique():
        dataset_data = df[df['Dataset'] == dataset]
        fastest_row = dataset_data.loc[dataset_data['Training Time (min)'].idxmin()]
        print(f"{dataset:12} | {fastest_row['Architecture']:25} | Time: {fastest_row['Training Time (min)']:.1f}m")
    
    # Overall architecture performance
    print("\n🎯 OVERALL ARCHITECTURE PERFORMANCE:")
    print("-" * 80)
    arch_summary = df.groupby('Architecture').agg({
        'Best Loss': ['mean', 'std', 'min'],
        'Training Time (min)': ['mean', 'std'],
        'Parameters': 'mean'
    }).round(6)
    
    print(arch_summary)
    
    # Save summary to file
    summary_path = Path("results/full_comparison/summary_table.txt")
    with open(summary_path, 'w') as f:
        f.write("FULL CORTEXFLOW COMPARISON SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("BEST PERFORMANCE PER DATASET:\n")
        f.write("-" * 60 + "\n")
        for dataset in df['Dataset'].unique():
            dataset_data = df[df['Dataset'] == dataset]
            best_row = dataset_data.loc[dataset_data['Best Loss'].idxmin()]
            f.write(f"{dataset:12} | {best_row['Architecture']:25} | Loss: {best_row['Best Loss']:.6f}\n")
        
        f.write("\nOVERALL ARCHITECTURE PERFORMANCE:\n")
        f.write("-" * 80 + "\n")
        f.write(str(arch_summary))
    
    print(f"\n📄 Summary saved: {summary_path}")

def main():
    """Main execution function."""
    print("🚀 FULL CORTEXFLOW COMPARISON RUNNER")
    print("="*80)
    
    # Create results directory
    results_dir = Path("results/full_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    success = run_full_comparison()
    
    if success:
        print("\n✅ Training completed successfully!")
        
        # Create visualizations
        create_comparison_visualizations()
        
        print("\n🎉 FULL COMPARISON COMPLETED!")
        print("📁 Check results/full_comparison/ for detailed results")
        
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()
