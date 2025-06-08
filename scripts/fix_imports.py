#!/usr/bin/env python3
"""
Fix Import Paths Script
Updates import paths in moved files to work with new project structure
"""

import os
import re
from pathlib import Path

def fix_experiment_imports():
    """Fix import paths in experiment files."""
    print("🔧 Fixing Import Paths in Experiments")
    print("=" * 60)
    
    experiment_files = [
        'experiments/simple/cortexflow_training.py',
        'experiments/hierarchical/hierarchical_training.py',
        'experiments/enhanced/enhanced_hierarchical_training.py'
    ]
    
    for file_path in experiment_files:
        if os.path.exists(file_path):
            print(f"📝 Fixing imports in: {file_path}")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Fix sys.path.append to go up two levels to reach src
            content = re.sub(
                r"sys\.path\.append\('src'\)",
                "sys.path.append('../../src')",
                content
            )
            content = re.sub(
                r"sys\.path\.append\('src/models'\)",
                "sys.path.append('../../src/models')",
                content
            )
            
            # Fix data paths
            content = re.sub(
                r"'data/miyawaki_structured_28x28\.mat'",
                "'../../data/processed/miyawaki_structured_28x28.mat'",
                content
            )
            content = re.sub(
                r"'data/digit69_28x28\.mat'",
                "'../../data/processed/digit69_28x28.mat'",
                content
            )
            
            # Fix checkpoint paths
            content = re.sub(
                r"'checkpoints'",
                "'../../checkpoints'",
                content
            )
            content = re.sub(
                r"os\.makedirs\('checkpoints'",
                "os.makedirs('../../checkpoints'",
                content
            )
            
            # Fix results paths
            content = re.sub(
                r"'results/'",
                "'../../results/'",
                content
            )
            content = re.sub(
                r"os\.makedirs\('results'",
                "os.makedirs('../../results'",
                content
            )
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"   ✅ Fixed imports in {file_path}")
        else:
            print(f"   ⏭️  Skipped {file_path} (not found)")

def fix_test_imports():
    """Fix import paths in test files."""
    print("\n🧪 Fixing Import Paths in Tests")
    print("=" * 60)
    
    test_files = [
        'tests/test_hierarchical.py',
        'tests/test_reproducibility.py'
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"📝 Fixing imports in: {file_path}")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Fix sys.path.append
            content = re.sub(
                r"sys\.path\.append\('src'\)",
                "sys.path.append('../src')",
                content
            )
            content = re.sub(
                r"sys\.path\.append\('src/models'\)",
                "sys.path.append('../src/models')",
                content
            )
            
            # Fix data paths
            content = re.sub(
                r"'data/miyawaki_structured_28x28\.mat'",
                "'../data/processed/miyawaki_structured_28x28.mat'",
                content
            )
            content = re.sub(
                r"'data/digit69_28x28\.mat'",
                "'../data/processed/digit69_28x28.mat'",
                content
            )
            
            # Fix results paths
            content = re.sub(
                r"'test_results'",
                "'results'",
                content
            )
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"   ✅ Fixed imports in {file_path}")
        else:
            print(f"   ⏭️  Skipped {file_path} (not found)")

def create_experiment_runners():
    """Create runner scripts for each experiment type."""
    print("\n🚀 Creating Experiment Runner Scripts")
    print("=" * 60)
    
    # Simple experiment runner
    simple_runner = '''#!/usr/bin/env python3
"""
Simple CortexFlow Experiment Runner
Run from project root: python experiments/run_simple.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run simple CortexFlow experiment."""
    print("🚀 Running Simple CortexFlow Experiment")
    print("=" * 60)
    
    # Change to experiment directory
    experiment_dir = Path(__file__).parent / 'simple'
    os.chdir(experiment_dir)
    
    # Run experiment
    result = subprocess.run([sys.executable, 'cortexflow_training.py'], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\\n✅ Simple experiment completed successfully!")
    else:
        print("\\n❌ Simple experiment failed!")
        return result.returncode
    
    return 0

if __name__ == "__main__":
    exit(main())
'''
    
    # Hierarchical experiment runner
    hierarchical_runner = '''#!/usr/bin/env python3
"""
Hierarchical CortexFlow Experiment Runner
Run from project root: python experiments/run_hierarchical.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run hierarchical CortexFlow experiment."""
    print("🏗️ Running Hierarchical CortexFlow Experiment")
    print("=" * 60)
    
    # Change to experiment directory
    experiment_dir = Path(__file__).parent / 'hierarchical'
    os.chdir(experiment_dir)
    
    # Run experiment
    result = subprocess.run([sys.executable, 'hierarchical_training.py'], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\\n✅ Hierarchical experiment completed successfully!")
    else:
        print("\\n❌ Hierarchical experiment failed!")
        return result.returncode
    
    return 0

if __name__ == "__main__":
    exit(main())
'''
    
    # Enhanced experiment runner
    enhanced_runner = '''#!/usr/bin/env python3
"""
Enhanced CortexFlow Experiment Runner
Run from project root: python experiments/run_enhanced.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run enhanced CortexFlow experiment."""
    print("🔬 Running Enhanced CortexFlow Experiment")
    print("=" * 60)
    
    # Change to experiment directory
    experiment_dir = Path(__file__).parent / 'enhanced'
    os.chdir(experiment_dir)
    
    # Run experiment
    result = subprocess.run([sys.executable, 'enhanced_hierarchical_training.py'], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\\n✅ Enhanced experiment completed successfully!")
    else:
        print("\\n❌ Enhanced experiment failed!")
        return result.returncode
    
    return 0

if __name__ == "__main__":
    exit(main())
'''
    
    runners = [
        ('experiments/run_simple.py', simple_runner),
        ('experiments/run_hierarchical.py', hierarchical_runner),
        ('experiments/run_enhanced.py', enhanced_runner),
    ]
    
    for runner_path, content in runners:
        with open(runner_path, 'w') as f:
            f.write(content)
        print(f"🚀 Created: {runner_path}")

def create_quick_test_script():
    """Create a quick test script to verify the organized structure."""
    print("\n🧪 Creating Quick Test Script")
    print("=" * 60)
    
    test_script = '''#!/usr/bin/env python3
"""
Quick Test Script for Organized CortexFlow Structure
Tests if all components can be imported and basic functionality works
"""

import os
import sys
import torch
from pathlib import Path

def test_imports():
    """Test if all imports work correctly."""
    print("🧪 Testing Imports")
    print("-" * 40)
    
    # Add src to path
    sys.path.append('src')
    sys.path.append('src/models')
    
    try:
        from hierarchical import HierarchicalCortexFlow, HierarchicalConfig, create_hierarchical_model
        print("✅ Hierarchical imports successful")
    except ImportError as e:
        print(f"❌ Hierarchical import failed: {e}")
        return False
    
    return True

def test_data_access():
    """Test if data files can be accessed."""
    print("\\n📊 Testing Data Access")
    print("-" * 40)
    
    data_files = [
        'data/processed/miyawaki_structured_28x28.mat',
        'data/processed/digit69_28x28.mat'
    ]
    
    all_accessible = True
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"✅ {data_file} accessible")
        else:
            print(f"❌ {data_file} not found")
            all_accessible = False
    
    return all_accessible

def test_directory_structure():
    """Test if directory structure is correct."""
    print("\\n📁 Testing Directory Structure")
    print("-" * 40)
    
    required_dirs = [
        'experiments/simple',
        'experiments/hierarchical', 
        'experiments/enhanced',
        'experiments/configs',
        'src/models',
        'data/processed',
        'data/raw',
        'data/external',
        'results/simple',
        'results/hierarchical',
        'results/enhanced',
        'checkpoints/hierarchical',
        'checkpoints/enhanced',
        'tests',
        'docs',
        'scripts'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} missing")
            all_exist = False
    
    return all_exist

def test_model_creation():
    """Test if models can be created."""
    print("\\n🧠 Testing Model Creation")
    print("-" * 40)
    
    try:
        sys.path.append('src')
        sys.path.append('src/models')
        from hierarchical import HierarchicalConfig, create_hierarchical_model
        
        config = HierarchicalConfig()
        model = create_hierarchical_model(input_dim=100, config=config)
        
        print(f"✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔬 CORTEXFLOW ORGANIZED STRUCTURE TEST")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Access Test", test_data_access),
        ("Directory Structure Test", test_directory_structure),
        ("Model Creation Test", test_model_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print(f"\\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Organized structure is working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the organized structure.")
        return 1

if __name__ == "__main__":
    exit(main())
'''
    
    with open('scripts/quick_test.py', 'w') as f:
        f.write(test_script)
    print(f"🧪 Created: scripts/quick_test.py")

def main():
    """Main function to fix all import paths."""
    print("🔧 CORTEXFLOW IMPORT PATH FIXER")
    print("=" * 80)
    
    # Fix experiment imports
    fix_experiment_imports()
    
    # Fix test imports
    fix_test_imports()
    
    # Create experiment runners
    create_experiment_runners()
    
    # Create quick test script
    create_quick_test_script()
    
    print(f"\n{'='*80}")
    print(f"🎉 IMPORT PATH FIXING COMPLETED!")
    print(f"{'='*80}")
    
    print(f"\n📊 SUMMARY:")
    print("-" * 60)
    print(f"✅ Fixed import paths in experiment files")
    print(f"✅ Fixed import paths in test files")
    print(f"✅ Created experiment runner scripts")
    print(f"✅ Created quick test script")
    
    print(f"\n🎯 NEXT STEPS:")
    print("-" * 60)
    print(f"1. Run quick test: python scripts/quick_test.py")
    print(f"2. Test experiments:")
    print(f"   - python experiments/run_simple.py")
    print(f"   - python experiments/run_hierarchical.py")
    print(f"   - python experiments/run_enhanced.py")
    print(f"3. Run tests: python tests/test_hierarchical.py")

if __name__ == "__main__":
    main()
