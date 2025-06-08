#!/usr/bin/env python3
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
    print("\n📊 Testing Data Access")
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
    print("\n📁 Testing Directory Structure")
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
    print("\n🧠 Testing Model Creation")
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
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Organized structure is working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the organized structure.")
        return 1

if __name__ == "__main__":
    exit(main())
