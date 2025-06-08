#!/usr/bin/env python3
"""
🚀 RUN ALL CORTEXFLOW ARCHITECTURES ON ALL 4 DATASETS
================================================================================
Runs training on all available CortexFlow architectures across all 4 datasets
using the same data loading approach as Monte Carlo Simple
================================================================================
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime

def print_header(title):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"🚀 {title}")
    print(f"{'='*80}")

def print_section(title):
    """Print formatted section."""
    print(f"\n{'─'*60}")
    print(f"🧪 {title}")
    print(f"{'─'*60}")

def run_architecture_training(arch_name, script_path):
    """Run training for a specific architecture."""
    print_section(f"RUNNING {arch_name.upper()}")
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False, 0, "Script not found"
    
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd='.')
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            # Count successful datasets from output
            output_lines = result.stdout.split('\n')
            success_count = 0
            for line in output_lines:
                if '✅' in line and ('Loss' in line or 'SUCCESS' in line):
                    success_count += 1
            
            print(f"✅ {arch_name} completed successfully in {elapsed_time/60:.1f} minutes!")
            print(f"📊 Successful datasets: {success_count}")
            
            # Print key results
            for line in output_lines[-20:]:
                if '✅' in line or '❌' in line:
                    print(f"   {line}")
            
            return True, success_count, f"Success in {elapsed_time/60:.1f}m"
        else:
            print(f"❌ {arch_name} failed with return code {result.returncode}")
            print(f"Error output: {result.stderr[-500:]}")  # Last 500 chars of error
            return False, 0, f"Failed: {result.stderr[-100:]}"
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {arch_name} crashed: {e}")
        return False, 0, f"Crashed: {str(e)[:100]}"

def main():
    """Run all CortexFlow architectures on all 4 datasets."""
    print_header("CORTEXFLOW ALL ARCHITECTURES - 4 DATASETS TRAINING")
    print(f"🎯 Running all available CortexFlow architectures on all 4 datasets")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Target: 4 datasets per architecture (Miyawaki, Vangerven, MindBigData, Crell)")
    
    # Define experiments to run
    experiments = [
        ("Monte Carlo Simple CortexFlow", "experiments/train_remaining_datasets.py"),
        ("Simple CortexFlow", "experiments/run_simple.py"),
        ("Hierarchical CortexFlow", "experiments/run_hierarchical.py"),
        ("Enhanced CortexFlow", "experiments/run_enhanced.py"),
        ("Unified CortexFlow", "experiments/run_unified.py")
    ]
    
    results = {}
    total_start_time = time.time()
    
    print(f"\n📋 PLANNED EXPERIMENTS:")
    for i, (name, script) in enumerate(experiments, 1):
        status = "✅ Available" if os.path.exists(script) else "❌ Missing"
        print(f"   {i}. {name}: {status}")
    
    # Run each experiment
    for exp_name, script_path in experiments:
        if os.path.exists(script_path):
            success, dataset_count, details = run_architecture_training(exp_name, script_path)
            results[exp_name] = {
                'success': success,
                'dataset_count': dataset_count,
                'details': details
            }
        else:
            print_section(f"SKIPPING {exp_name.upper()}")
            print(f"⏭️  Script not found: {script_path}")
            results[exp_name] = {
                'success': False,
                'dataset_count': 0,
                'details': 'Script not found'
            }
    
    # Summary
    total_time = time.time() - total_start_time
    successful_archs = sum(1 for r in results.values() if r['success'])
    total_archs = len(results)
    total_datasets = sum(r['dataset_count'] for r in results.values())
    
    print_header("FINAL COMPREHENSIVE SUMMARY")
    
    print(f"📊 ARCHITECTURE RESULTS:")
    print("-" * 80)
    for arch_name, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        datasets = f"{result['dataset_count']}/4 datasets"
        print(f"   {status}: {arch_name}")
        print(f"      📊 {datasets} | {result['details']}")
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   🎯 Architecture Success Rate: {successful_archs}/{total_archs} ({100*successful_archs/total_archs:.1f}%)")
    print(f"   📊 Total Successful Datasets: {total_datasets}")
    print(f"   🎯 Target Datasets: {total_archs * 4} (4 per architecture)")
    print(f"   📈 Dataset Success Rate: {total_datasets}/{total_archs * 4} ({100*total_datasets/(total_archs * 4):.1f}%)")
    print(f"   ⏱️  Total Time: {total_time/60:.1f} minutes")
    print(f"   🏁 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Performance analysis
    print(f"\n🏆 PERFORMANCE ANALYSIS:")
    print("-" * 80)
    
    # Best performing architecture
    best_arch = max(results.items(), key=lambda x: x[1]['dataset_count'])
    print(f"🥇 Best Architecture: {best_arch[0]} ({best_arch[1]['dataset_count']}/4 datasets)")
    
    # Fully successful architectures
    full_success = [name for name, result in results.items() if result['dataset_count'] == 4]
    if full_success:
        print(f"🎉 Fully Successful (4/4): {', '.join(full_success)}")
    
    # Partially successful architectures
    partial_success = [name for name, result in results.items() if 0 < result['dataset_count'] < 4]
    if partial_success:
        print(f"⚠️  Partially Successful: {', '.join(partial_success)}")
    
    # Failed architectures
    failed = [name for name, result in results.items() if result['dataset_count'] == 0]
    if failed:
        print(f"❌ Failed Architectures: {', '.join(failed)}")
    
    if successful_archs == total_archs and total_datasets == total_archs * 4:
        print(f"\n🎉 PERFECT SUCCESS! ALL ARCHITECTURES ON ALL DATASETS!")
        print(f"🔬 CortexFlow comprehensive evaluation completed flawlessly!")
    elif successful_archs == total_archs:
        print(f"\n✅ ALL ARCHITECTURES WORKING!")
        print(f"📊 {total_datasets}/{total_archs * 4} datasets successful")
    else:
        print(f"\n⚠️  Some architectures need attention")
        print(f"📋 Check individual experiment logs for details")
    
    return successful_archs == total_archs and total_datasets == total_archs * 4

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
