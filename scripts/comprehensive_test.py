#!/usr/bin/env python3
"""
Comprehensive CortexFlow Test
Tests all architectures in organized structure
"""

import os
import sys
import subprocess
from pathlib import Path
import time

def run_experiment(experiment_name, script_path):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING {experiment_name.upper()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              cwd=Path(__file__).parent.parent)
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {experiment_name} completed successfully in {elapsed_time/60:.1f} minutes!")
            return True
        else:
            print(f"\n❌ {experiment_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ {experiment_name} crashed: {e}")
        return False

def main():
    """Run comprehensive test of all architectures."""
    print("🚀 COMPREHENSIVE CORTEXFLOW TEST")
    print("=" * 80)
    print(f"Test start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test experiments
    experiments = [
        ("Simple CortexFlow", "experiments/run_simple.py"),
        ("Hierarchical CortexFlow", "experiments/run_hierarchical.py"),
        ("Enhanced CortexFlow", "experiments/run_enhanced.py")
    ]
    
    results = {}
    total_start_time = time.time()
    
    for exp_name, script_path in experiments:
        if os.path.exists(script_path):
            results[exp_name] = run_experiment(exp_name, script_path)
        else:
            print(f"\n⏭️  Skipping {exp_name}: {script_path} not found")
            results[exp_name] = False
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"🎉 COMPREHENSIVE TEST COMPLETED!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*80}")
    
    print(f"\n📊 FINAL RESULTS:")
    print("-" * 60)
    
    passed = 0
    for exp_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{exp_name:<25}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Summary: {passed}/{len(results)} experiments passed")
    
    if passed == len(results):
        print("🎉 ALL EXPERIMENTS SUCCESSFUL! CortexFlow is fully functional!")
        return 0
    else:
        print("⚠️  Some experiments failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main())
