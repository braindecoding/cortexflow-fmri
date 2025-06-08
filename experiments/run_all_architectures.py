#!/usr/bin/env python3
"""
🚀 RUN ALL CORTEXFLOW ARCHITECTURES
================================================================================
Runs training on all available CortexFlow architectures across all 4 datasets
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

def run_experiment(experiment_name, script_path):
    """Run a single experiment."""
    print_section(f"RUNNING {experiment_name.upper()}")
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=False, text=True, cwd='.')
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {experiment_name} completed successfully in {elapsed_time/60:.1f} minutes!")
            return True
        else:
            print(f"❌ {experiment_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {experiment_name} crashed: {e}")
        return False

def main():
    """Run all CortexFlow architectures."""
    print_header("CORTEXFLOW ALL ARCHITECTURES TRAINING")
    print(f"🎯 Running all available CortexFlow architectures on all 4 datasets")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
            results[exp_name] = run_experiment(exp_name, script_path)
        else:
            print_section(f"SKIPPING {exp_name.upper()}")
            print(f"⏭️  Script not found: {script_path}")
            results[exp_name] = False
    
    # Summary
    total_time = time.time() - total_start_time
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print_header("FINAL SUMMARY")
    
    print(f"📊 EXPERIMENT RESULTS:")
    print("-" * 60)
    for exp_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {status}: {exp_name}")
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   🎯 Success Rate: {successful}/{total} ({100*successful/total:.1f}%)")
    print(f"   ⏱️  Total Time: {total_time/60:.1f} minutes")
    print(f"   🏁 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful == total:
        print(f"\n🎉 ALL ARCHITECTURES COMPLETED SUCCESSFULLY!")
        print(f"🔬 CortexFlow comprehensive evaluation completed!")
    else:
        print(f"\n⚠️  Some architectures failed or were unavailable")
        print(f"📋 Check individual experiment logs for details")
    
    return successful == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
