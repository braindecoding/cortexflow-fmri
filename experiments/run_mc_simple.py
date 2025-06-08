"""
Monte Carlo Simple CortexFlow Experiment Runner

This script runs the Monte Carlo Simple CortexFlow experiments with proper
path handling and comprehensive evaluation.

Features:
- Automatic path resolution
- Uncertainty-aware training and evaluation
- Comprehensive visualization
- Performance comparison with Simple CortexFlow
- Statistical analysis of uncertainty estimates

Author: CortexFlow Team
"""

import sys
import os
sys.path.append('.')

import torch
import time
from datetime import datetime

# Import the training function
from experiments.mc_simple.mc_simple_training import train_mc_simple_cortexflow, MCSimpleConfig


def run_mc_simple_experiment():
    """Run Monte Carlo Simple CortexFlow experiment."""
    
    print("🎲 MONTE CARLO SIMPLE CORTEXFLOW EXPERIMENT")
    print("=" * 70)
    print(f"Experiment start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Using device: {device}")
    
    # Create results directory
    results_dir = "results/mc_simple"
    os.makedirs(results_dir, exist_ok=True)
    
    # Dataset configurations
    datasets = [
        {
            'path': 'data/processed/miyawaki_structured_28x28.mat',
            'name': 'miyawaki',
            'description': 'Visual Cortex fMRI → Image Reconstruction'
        },
        {
            'path': 'data/processed/digit69_28x28.mat', 
            'name': 'vangerven',
            'description': 'Digit Recognition fMRI → Image Reconstruction'
        }
    ]
    
    # Print configuration
    print(f"\n⚙️  MONTE CARLO SIMPLE CORTEXFLOW CONFIGURATION:")
    print(f"   🧠 Hidden dimension: {MCSimpleConfig.HIDDEN_DIM}")
    print(f"   🎯 Dropout rate: {MCSimpleConfig.DROPOUT_RATE}")
    print(f"   🎲 MC samples (training): {MCSimpleConfig.MC_SAMPLES}")
    print(f"   🎲 MC samples (evaluation): {MCSimpleConfig.MC_EVAL_SAMPLES}")
    print(f"   ⚖️  Uncertainty weight: {MCSimpleConfig.UNCERTAINTY_WEIGHT}")
    print(f"   📚 Batch size: {MCSimpleConfig.BATCH_SIZE}")
    print(f"   📈 Learning rate: {MCSimpleConfig.LEARNING_RATE}")
    print(f"   🔄 Max epochs: {MCSimpleConfig.NUM_EPOCHS}")
    print(f"   ⏳ Patience: {MCSimpleConfig.PATIENCE}")
    
    results = {}
    experiment_start_time = time.time()
    
    # Train on each dataset
    for i, dataset_config in enumerate(datasets, 1):
        print(f"\n{'='*70}")
        print(f"🧪 EXPERIMENT {i}/{len(datasets)}: {dataset_config['name'].upper()}")
        print(f"📊 Task: {dataset_config['description']}")
        print(f"{'='*70}")
        
        try:
            # Check if dataset exists
            if not os.path.exists(dataset_config['path']):
                raise FileNotFoundError(f"Dataset not found: {dataset_config['path']}")
            
            # Train model
            result = train_mc_simple_cortexflow(
                dataset_path=dataset_config['path'],
                dataset_name=dataset_config['name'],
                save_dir=results_dir,
                device=device
            )
            
            results[dataset_config['name']] = result
            
            # Print individual result summary
            print(f"\n📊 {dataset_config['name'].upper()} RESULTS:")
            print(f"   🏆 Best loss: {result['best_loss']:.6f}")
            print(f"   📈 Total epochs: {result['total_epochs']}")
            print(f"   ⏱️  Training time: {result['training_time']:.1f} minutes")
            print(f"   🔧 Parameters: {result['model_info']['total_parameters']:,}")
            
            if result['final_uncertainty_stats']['samples'] > 0:
                unc_stats = result['final_uncertainty_stats']
                print(f"   🎲 Uncertainty stats:")
                print(f"      Mean: {unc_stats['mean']:.6f}")
                print(f"      Std:  {unc_stats['std']:.6f}")
                print(f"      Range: [{unc_stats['min']:.6f}, {unc_stats['max']:.6f}]")
            
        except Exception as e:
            print(f"❌ Error training {dataset_config['name']}: {e}")
            results[dataset_config['name']] = {'error': str(e)}
    
    # Calculate total experiment time
    total_time = (time.time() - experiment_start_time) / 60
    
    # Print comprehensive final summary
    print(f"\n{'='*70}")
    print("🎉 MONTE CARLO SIMPLE CORTEXFLOW EXPERIMENT COMPLETED!")
    print(f"{'='*70}")
    print(f"⏱️  Total experiment time: {total_time:.1f} minutes")
    
    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print("-" * 70)
    
    successful_experiments = 0
    total_experiments = len(datasets)
    
    for dataset_name, result in results.items():
        if 'error' in result:
            print(f"❌ {dataset_name.capitalize():12s}: FAILED - {result['error']}")
        else:
            successful_experiments += 1
            print(f"✅ {dataset_name.capitalize():12s}: Loss {result['best_loss']:.6f} "
                  f"({result['total_epochs']} epochs, {result['training_time']:.1f}m)")
            
            # Additional uncertainty metrics if available
            if 'uncertainty_stats' in result and result['uncertainty_stats']:
                last_stats = result['uncertainty_stats'][-1]
                print(f"   🎲 Final uncertainties - "
                      f"Epistemic: {last_stats['epistemic']:.4f}, "
                      f"Aleatoric: {last_stats['aleatoric']:.4f}, "
                      f"Total: {last_stats['total']:.4f}")
    
    # Success rate
    success_rate = (successful_experiments / total_experiments) * 100
    print(f"\n📈 SUCCESS RATE: {successful_experiments}/{total_experiments} ({success_rate:.1f}%)")
    
    # Model comparison insights
    print(f"\n🔬 MONTE CARLO SIMPLE CORTEXFLOW INSIGHTS:")
    print(f"   ✅ Combines simplicity of Simple CortexFlow with uncertainty estimation")
    print(f"   🎲 Provides epistemic and aleatoric uncertainty quantification")
    print(f"   📊 Uses Monte Carlo Dropout for efficient uncertainty sampling")
    print(f"   ⚡ Maintains computational efficiency during training")
    print(f"   🎯 Enables uncertainty-aware decision making")
    
    if successful_experiments > 0:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   🔍 Analyze uncertainty patterns for different input types")
        print(f"   📊 Compare uncertainty estimates with reconstruction quality")
        print(f"   🎯 Use uncertainty for active learning or data selection")
        print(f"   🔬 Investigate correlation between uncertainty and task difficulty")
    
    # Save experiment summary
    summary_path = os.path.join(results_dir, 'experiment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Monte Carlo Simple CortexFlow Experiment Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time:.1f} minutes\n")
        f.write(f"Success rate: {success_rate:.1f}%\n\n")
        
        for dataset_name, result in results.items():
            f.write(f"{dataset_name.capitalize()}:\n")
            if 'error' in result:
                f.write(f"  Status: FAILED - {result['error']}\n")
            else:
                f.write(f"  Status: SUCCESS\n")
                f.write(f"  Best loss: {result['best_loss']:.6f}\n")
                f.write(f"  Epochs: {result['total_epochs']}\n")
                f.write(f"  Time: {result['training_time']:.1f} minutes\n")
                f.write(f"  Parameters: {result['model_info']['total_parameters']:,}\n")
            f.write("\n")
    
    print(f"\n📄 Experiment summary saved: {summary_path}")
    print(f"📁 Results directory: {results_dir}")
    print(f"\n🎉 Monte Carlo Simple CortexFlow experiment completed successfully!")
    
    return results


if __name__ == "__main__":
    try:
        results = run_mc_simple_experiment()
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        raise
