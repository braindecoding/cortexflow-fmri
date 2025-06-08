#!/usr/bin/env python3
"""
Fix All Paths Script
Comprehensive fix for all path issues in organized structure
"""

import os
import re
from pathlib import Path

def fix_hierarchical_paths():
    """Fix paths in hierarchical experiment."""
    print("🔧 Fixing Hierarchical Experiment Paths")
    print("-" * 60)
    
    file_path = 'experiments/hierarchical/hierarchical_training.py'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix checkpoint paths
        content = re.sub(
            r"torch\.save\(checkpoint, f'checkpoints/hierarchical_\{dataset_name\.lower\(\)\}_model\.pt'\)",
            "torch.save(checkpoint, f'../../checkpoints/hierarchical/hierarchical_{dataset_name.lower()}_model.pt')",
            content
        )
        
        # Fix results paths
        content = re.sub(
            r"save_path=f'results/hierarchical_\{dataset_name\.lower\(\)\}_reconstructions\.png'",
            "save_path=f'../../results/hierarchical/hierarchical_{dataset_name.lower()}_reconstructions.png'",
            content
        )
        content = re.sub(
            r"save_path=f'results/hierarchical_\{dataset_name\.lower\(\)\}_progressive\.png'",
            "save_path=f'../../results/hierarchical/hierarchical_{dataset_name.lower()}_progressive.png'",
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed paths in {file_path}")
    else:
        print(f"⏭️  Skipped {file_path} (not found)")

def fix_enhanced_paths():
    """Fix paths in enhanced experiment."""
    print("\n🔧 Fixing Enhanced Experiment Paths")
    print("-" * 60)
    
    file_path = 'experiments/enhanced/enhanced_hierarchical_training.py'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix checkpoint paths
        content = re.sub(
            r"torch\.save\(checkpoint, f'checkpoints/enhanced_\{dataset_name\.lower\(\)\}_model\.pt'\)",
            "torch.save(checkpoint, f'../../checkpoints/enhanced/enhanced_{dataset_name.lower()}_model.pt')",
            content
        )
        
        # Fix results paths
        content = re.sub(
            r"save_path=f'results/enhanced_\{dataset_name\.lower\(\)\}_reconstructions\.png'",
            "save_path=f'../../results/enhanced/enhanced_{dataset_name.lower()}_reconstructions.png'",
            content
        )
        content = re.sub(
            r"save_path=f'results/enhanced_\{dataset_name\.lower\(\)\}_uncertainty\.png'",
            "save_path=f'../../results/enhanced/enhanced_{dataset_name.lower()}_uncertainty.png'",
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed paths in {file_path}")
    else:
        print(f"⏭️  Skipped {file_path} (not found)")

def fix_test_paths():
    """Fix paths in test files."""
    print("\n🔧 Fixing Test File Paths")
    print("-" * 60)
    
    test_files = [
        'tests/test_hierarchical.py',
        'tests/test_reproducibility.py'
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Fix checkpoint paths
            content = re.sub(
                r"'checkpoints/",
                "'../checkpoints/",
                content
            )
            
            # Fix results paths
            content = re.sub(
                r"'results/",
                "'../results/",
                content
            )
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"✅ Fixed paths in {file_path}")
        else:
            print(f"⏭️  Skipped {file_path} (not found)")

def create_comprehensive_test():
    """Create comprehensive test script."""
    print("\n🧪 Creating Comprehensive Test Script")
    print("-" * 60)
    
    test_script = '''#!/usr/bin/env python3
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
    print(f"\\n{'='*80}")
    print(f"🧪 TESTING {experiment_name.upper()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              cwd=Path(__file__).parent.parent)
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\\n✅ {experiment_name} completed successfully in {elapsed_time/60:.1f} minutes!")
            return True
        else:
            print(f"\\n❌ {experiment_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\\n❌ {experiment_name} crashed: {e}")
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
            print(f"\\n⏭️  Skipping {exp_name}: {script_path} not found")
            results[exp_name] = False
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print(f"\\n{'='*80}")
    print(f"🎉 COMPREHENSIVE TEST COMPLETED!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*80}")
    
    print(f"\\n📊 FINAL RESULTS:")
    print("-" * 60)
    
    passed = 0
    for exp_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{exp_name:<25}: {status}")
        if success:
            passed += 1
    
    print(f"\\n🎯 Summary: {passed}/{len(results)} experiments passed")
    
    if passed == len(results):
        print("🎉 ALL EXPERIMENTS SUCCESSFUL! CortexFlow is fully functional!")
        return 0
    else:
        print("⚠️  Some experiments failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main())
'''
    
    with open('scripts/comprehensive_test.py', 'w') as f:
        f.write(test_script)
    
    print(f"🧪 Created: scripts/comprehensive_test.py")

def main():
    """Main function to fix all paths."""
    print("🔧 COMPREHENSIVE PATH FIXER")
    print("=" * 80)
    
    # Fix all experiment paths
    fix_hierarchical_paths()
    fix_enhanced_paths()
    
    # Fix test paths
    fix_test_paths()
    
    # Create comprehensive test
    create_comprehensive_test()
    
    print(f"\\n{'='*80}")
    print(f"🎉 ALL PATHS FIXED!")
    print(f"{'='*80}")
    
    print(f"\\n📊 SUMMARY:")
    print("-" * 60)
    print(f"✅ Fixed hierarchical experiment paths")
    print(f"✅ Fixed enhanced experiment paths")
    print(f"✅ Fixed test file paths")
    print(f"✅ Created comprehensive test script")
    
    print(f"\\n🎯 NEXT STEPS:")
    print("-" * 60)
    print(f"1. Run comprehensive test: python scripts/comprehensive_test.py")
    print(f"2. Or run individual experiments:")
    print(f"   - python experiments/run_simple.py")
    print(f"   - python experiments/run_hierarchical.py")
    print(f"   - python experiments/run_enhanced.py")

if __name__ == "__main__":
    main()
