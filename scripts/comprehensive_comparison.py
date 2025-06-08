#!/usr/bin/env python3
"""
Comprehensive CortexFlow Results Comparison
Detailed analysis and comparison of all three architectures
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from datetime import datetime

def load_results_data():
    """Load and organize results from all experiments."""
    print("📊 LOADING RESULTS DATA")
    print("=" * 60)
    
    # Results from the comprehensive test
    results = {
        'Simple CortexFlow': {
            'miyawaki': {
                'loss': 0.018006,
                'epochs': 58,
                'time_minutes': 0.2,
                'parameters': 6009616
            },
            'vangerven': {
                'loss': 0.037846,
                'epochs': 53,
                'time_minutes': 0.2,
                'parameters': 8185616
            }
        },
        'Hierarchical CortexFlow': {
            'miyawaki': {
                'loss': 0.079622,
                'epochs': 107,
                'time_minutes': 1.1,
                'parameters': 20310350
            },
            'vangerven': {
                'loss': 0.108537,
                'epochs': 32,
                'time_minutes': 0.5,
                'parameters': 29014350
            }
        },
        'Enhanced CortexFlow': {
            'miyawaki': {
                'loss': 0.056370,
                'epochs': 71,
                'time_minutes': 3.6,
                'parameters': 20376782
            },
            'vangerven': {
                'loss': 0.080576,
                'epochs': 50,
                'time_minutes': 2.4,
                'parameters': 29080782
            }
        }
    }
    
    print("✅ Results data loaded successfully")
    return results

def create_performance_comparison(results):
    """Create comprehensive performance comparison charts."""
    print("\n📈 CREATING PERFORMANCE COMPARISON CHARTS")
    print("=" * 60)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🧠 CortexFlow Architecture Comparison', fontsize=20, fontweight='bold')
    
    # Prepare data for plotting
    architectures = list(results.keys())
    datasets = ['miyawaki', 'vangerven']
    
    # 1. Loss Comparison
    ax1 = axes[0, 0]
    miyawaki_losses = [results[arch]['miyawaki']['loss'] for arch in architectures]
    vangerven_losses = [results[arch]['vangerven']['loss'] for arch in architectures]
    
    x = np.arange(len(architectures))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, miyawaki_losses, width, label='Miyawaki', alpha=0.8)
    bars2 = ax1.bar(x + width/2, vangerven_losses, width, label='Vangerven', alpha=0.8)
    
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('Test Loss')
    ax1.set_title('🎯 Test Loss Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([arch.replace(' CortexFlow', '') for arch in architectures], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Training Time Comparison
    ax2 = axes[0, 1]
    miyawaki_times = [results[arch]['miyawaki']['time_minutes'] for arch in architectures]
    vangerven_times = [results[arch]['vangerven']['time_minutes'] for arch in architectures]
    
    bars1 = ax2.bar(x - width/2, miyawaki_times, width, label='Miyawaki', alpha=0.8)
    bars2 = ax2.bar(x + width/2, vangerven_times, width, label='Vangerven', alpha=0.8)
    
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('Training Time (minutes)')
    ax2.set_title('⏱️ Training Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([arch.replace(' CortexFlow', '') for arch in architectures], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Parameters Comparison
    ax3 = axes[0, 2]
    miyawaki_params = [results[arch]['miyawaki']['parameters']/1e6 for arch in architectures]
    vangerven_params = [results[arch]['vangerven']['parameters']/1e6 for arch in architectures]
    
    bars1 = ax3.bar(x - width/2, miyawaki_params, width, label='Miyawaki', alpha=0.8)
    bars2 = ax3.bar(x + width/2, vangerven_params, width, label='Vangerven', alpha=0.8)
    
    ax3.set_xlabel('Architecture')
    ax3.set_ylabel('Parameters (Millions)')
    ax3.set_title('🔧 Model Parameters Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([arch.replace(' CortexFlow', '') for arch in architectures], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Epochs Comparison
    ax4 = axes[1, 0]
    miyawaki_epochs = [results[arch]['miyawaki']['epochs'] for arch in architectures]
    vangerven_epochs = [results[arch]['vangerven']['epochs'] for arch in architectures]
    
    bars1 = ax4.bar(x - width/2, miyawaki_epochs, width, label='Miyawaki', alpha=0.8)
    bars2 = ax4.bar(x + width/2, vangerven_epochs, width, label='Vangerven', alpha=0.8)
    
    ax4.set_xlabel('Architecture')
    ax4.set_ylabel('Training Epochs')
    ax4.set_title('🔄 Training Epochs Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([arch.replace(' CortexFlow', '') for arch in architectures], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Efficiency Analysis (Loss vs Time)
    ax5 = axes[1, 1]
    
    for i, arch in enumerate(architectures):
        miyawaki_loss = results[arch]['miyawaki']['loss']
        miyawaki_time = results[arch]['miyawaki']['time_minutes']
        vangerven_loss = results[arch]['vangerven']['loss']
        vangerven_time = results[arch]['vangerven']['time_minutes']
        
        ax5.scatter(miyawaki_time, miyawaki_loss, s=100, alpha=0.7, 
                   label=f'{arch.replace(" CortexFlow", "")} (Miyawaki)')
        ax5.scatter(vangerven_time, vangerven_loss, s=100, alpha=0.7, marker='s',
                   label=f'{arch.replace(" CortexFlow", "")} (Vangerven)')
    
    ax5.set_xlabel('Training Time (minutes)')
    ax5.set_ylabel('Test Loss')
    ax5.set_title('⚡ Efficiency Analysis (Loss vs Time)')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Radar Chart
    ax6 = axes[1, 2]
    
    # Normalize metrics for radar chart (lower is better, so invert)
    metrics = ['Loss (Miyawaki)', 'Loss (Vangerven)', 'Time Efficiency', 'Parameter Efficiency']
    
    # Calculate normalized scores (0-1, higher is better)
    scores = {}
    for arch in architectures:
        miyawaki_loss_score = 1 - (results[arch]['miyawaki']['loss'] / max([results[a]['miyawaki']['loss'] for a in architectures]))
        vangerven_loss_score = 1 - (results[arch]['vangerven']['loss'] / max([results[a]['vangerven']['loss'] for a in architectures]))
        time_score = 1 - ((results[arch]['miyawaki']['time_minutes'] + results[arch]['vangerven']['time_minutes']) / 
                         max([(results[a]['miyawaki']['time_minutes'] + results[a]['vangerven']['time_minutes']) for a in architectures]))
        param_score = 1 - ((results[arch]['miyawaki']['parameters'] + results[arch]['vangerven']['parameters']) / 
                          max([(results[a]['miyawaki']['parameters'] + results[a]['vangerven']['parameters']) for a in architectures]))
        
        scores[arch] = [miyawaki_loss_score, vangerven_loss_score, time_score, param_score]
    
    # Create simple bar chart instead of radar
    x_pos = np.arange(len(metrics))
    for i, arch in enumerate(architectures):
        ax6.bar(x_pos + i*0.25, scores[arch], 0.25, label=arch.replace(' CortexFlow', ''), alpha=0.8)
    
    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Normalized Score (Higher = Better)')
    ax6.set_title('📊 Overall Performance Score')
    ax6.set_xticks(x_pos + 0.25)
    ax6.set_xticklabels(metrics, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_path = 'results/comparisons/architecture_comparison.png'
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved comprehensive comparison to: {comparison_path}")
    
    plt.show()
    
    return fig

def create_detailed_analysis_table(results):
    """Create detailed analysis table."""
    print("\n📋 CREATING DETAILED ANALYSIS TABLE")
    print("=" * 60)
    
    # Create detailed comparison table
    table_data = []
    
    for arch in results.keys():
        for dataset in ['miyawaki', 'vangerven']:
            data = results[arch][dataset]
            table_data.append({
                'Architecture': arch,
                'Dataset': dataset.capitalize(),
                'Test Loss': f"{data['loss']:.6f}",
                'Training Epochs': data['epochs'],
                'Training Time (min)': f"{data['time_minutes']:.1f}",
                'Parameters (M)': f"{data['parameters']/1e6:.1f}",
                'Loss per Epoch': f"{data['loss']/data['epochs']:.8f}",
                'Loss per Minute': f"{data['loss']/data['time_minutes']:.6f}",
                'Efficiency Score': f"{(1/data['loss']) * (1/data['time_minutes']):.2f}"
            })
    
    df = pd.DataFrame(table_data)
    
    # Save to CSV
    csv_path = 'results/comparisons/detailed_comparison.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"📊 Saved detailed table to: {csv_path}")
    
    # Print formatted table
    print("\n" + "="*120)
    print("📊 DETAILED ARCHITECTURE COMPARISON TABLE")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)
    
    return df

def main():
    """Main comparison function."""
    print("🔍 COMPREHENSIVE CORTEXFLOW RESULTS COMPARISON")
    print("=" * 80)
    print(f"Analysis start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load results
    results = load_results_data()
    
    # Create performance comparison charts
    fig = create_performance_comparison(results)
    
    # Create detailed analysis table
    df = create_detailed_analysis_table(results)
    
    # Summary analysis
    print(f"\n🎯 SUMMARY ANALYSIS")
    print("=" * 60)
    
    print(f"\n🏆 BEST PERFORMANCE BY METRIC:")
    print("-" * 40)
    
    # Find best performers
    best_miyawaki_loss = min([(arch, results[arch]['miyawaki']['loss']) for arch in results.keys()], key=lambda x: x[1])
    best_vangerven_loss = min([(arch, results[arch]['vangerven']['loss']) for arch in results.keys()], key=lambda x: x[1])
    fastest_training = min([(arch, results[arch]['miyawaki']['time_minutes'] + results[arch]['vangerven']['time_minutes']) for arch in results.keys()], key=lambda x: x[1])
    most_efficient = min([(arch, results[arch]['miyawaki']['parameters'] + results[arch]['vangerven']['parameters']) for arch in results.keys()], key=lambda x: x[1])
    
    print(f"🎯 Best Miyawaki Loss: {best_miyawaki_loss[0]} ({best_miyawaki_loss[1]:.6f})")
    print(f"🎯 Best Vangerven Loss: {best_vangerven_loss[0]} ({best_vangerven_loss[1]:.6f})")
    print(f"⚡ Fastest Training: {fastest_training[0]} ({fastest_training[1]:.1f} min)")
    print(f"🔧 Most Parameter Efficient: {most_efficient[0]} ({most_efficient[1]/1e6:.1f}M params)")
    
    print(f"\n📊 ARCHITECTURE RANKINGS:")
    print("-" * 40)
    
    # Calculate overall scores
    overall_scores = {}
    for arch in results.keys():
        # Weighted score: 40% loss performance, 30% time efficiency, 30% parameter efficiency
        miyawaki_loss_rank = sorted(results.keys(), key=lambda x: results[x]['miyawaki']['loss']).index(arch) + 1
        vangerven_loss_rank = sorted(results.keys(), key=lambda x: results[x]['vangerven']['loss']).index(arch) + 1
        time_rank = sorted(results.keys(), key=lambda x: results[x]['miyawaki']['time_minutes'] + results[x]['vangerven']['time_minutes']).index(arch) + 1
        param_rank = sorted(results.keys(), key=lambda x: results[x]['miyawaki']['parameters'] + results[x]['vangerven']['parameters']).index(arch) + 1
        
        overall_score = (miyawaki_loss_rank + vangerven_loss_rank) * 0.4 + time_rank * 0.3 + param_rank * 0.3
        overall_scores[arch] = overall_score
    
    ranked_archs = sorted(overall_scores.items(), key=lambda x: x[1])
    
    for i, (arch, score) in enumerate(ranked_archs, 1):
        print(f"{i}. {arch} (Score: {score:.2f})")
    
    print(f"\n🎉 COMPARISON ANALYSIS COMPLETED!")
    print(f"📁 Check 'results/comparisons/' for detailed charts and data")
    print("=" * 80)

if __name__ == "__main__":
    main()
