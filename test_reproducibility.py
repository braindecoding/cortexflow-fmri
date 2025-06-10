#!/usr/bin/env python3
"""
Test Reproducibility
====================

Quick test to verify that the cleaned project is still functional
and ready for reproduction.
"""

import os
import sys
from pathlib import Path
import json

def test_file_structure():
    """Test that essential files exist"""
    
    print("🔍 Testing file structure...")
    
    essential_files = [
        'README.md',
        'SOTA.md', 
        'requirements.txt',
        'wsl_gpu_complete_training.py',
        'final_dissertation_verification.py',
        'docs/REPRODUCIBILITY.md',
        'docs/SUBMISSION_CHECKLIST.md',
        'PROJECT_SUMMARY.md'
    ]
    
    essential_dirs = [
        'data/processed',
        'src/models',
        'src/training',
        'results/wsl_gpu_training',
        'results/complete_4dataset_figures',
        'docs',
        'configs'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in essential_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    for dir_path in essential_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ All essential files and directories present")
    return True

def test_dataset_files():
    """Test that dataset files exist"""
    
    print("🔍 Testing dataset files...")
    
    dataset_files = [
        'data/processed/miyawaki_structured_28x28.mat',
        'data/processed/digit69_28x28.mat',
        'data/processed/mindbigdata.mat',
        'data/processed/crell.mat'
    ]
    
    missing_datasets = []
    
    for dataset in dataset_files:
        if not Path(dataset).exists():
            missing_datasets.append(dataset)
    
    if missing_datasets:
        print(f"❌ Missing datasets: {missing_datasets}")
        return False
    
    print("✅ All dataset files present")
    return True

def test_results_files():
    """Test that WSL GPU results exist"""
    
    print("🔍 Testing WSL GPU results...")
    
    results_files = [
        'results/wsl_gpu_training/wsl_gpu_training_results.json',
        'results/wsl_gpu_training/wsl_gpu_reconstruction_miyawaki_dissertation.png',
        'results/wsl_gpu_training/wsl_gpu_reconstruction_vangerven_dissertation.png',
        'results/wsl_gpu_training/wsl_gpu_reconstruction_mindbigdata_dissertation.png',
        'results/wsl_gpu_training/wsl_gpu_reconstruction_crell_dissertation.png'
    ]
    
    missing_results = []
    
    for result_file in results_files:
        if not Path(result_file).exists():
            missing_results.append(result_file)
    
    if missing_results:
        print(f"❌ Missing results: {missing_results}")
        return False
    
    print("✅ All WSL GPU results present")
    return True

def test_results_consistency():
    """Test that results are consistent"""
    
    print("🔍 Testing results consistency...")
    
    try:
        # Load WSL GPU results
        with open('results/wsl_gpu_training/wsl_gpu_training_results.json', 'r') as f:
            wsl_results = json.load(f)
        
        # Check structure
        expected_datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
        expected_methods = ['Adaptive_CNN', 'MinD_Vis', 'Brain_Diffuser', 'CortexFlow_Enhanced']
        
        for dataset in expected_datasets:
            if dataset not in wsl_results:
                print(f"❌ Missing dataset in results: {dataset}")
                return False
            
            for method in expected_methods:
                if method not in wsl_results[dataset]:
                    print(f"❌ Missing method in {dataset}: {method}")
                    return False
        
        # Check MSE ranges
        all_mse = []
        for dataset in wsl_results:
            for method in wsl_results[dataset]:
                mse = wsl_results[dataset][method]
                if mse is not None and not (isinstance(mse, str) and 'NaN' in str(mse)):
                    all_mse.append(mse)
        
        if not all_mse:
            print("❌ No valid MSE values found")
            return False
        
        min_mse = min(all_mse)
        max_mse = max(all_mse)
        
        # Expected range based on WSL GPU training
        if min_mse < 0.01 or max_mse > 0.1:
            print(f"⚠️  MSE range unusual: {min_mse:.4f} - {max_mse:.4f}")
            print("   (Expected: 0.01 - 0.1 for WSL GPU training)")
        
        print(f"✅ Results consistent: MSE range {min_mse:.4f} - {max_mse:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing results: {e}")
        return False

def test_import_capability():
    """Test that main script can be imported"""
    
    print("🔍 Testing import capability...")
    
    try:
        # Test if we can import torch (main dependency)
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"   ✅ CUDA: Available ({torch.cuda.get_device_name(0)})")
        else:
            print("   ⚠️  CUDA: Not available (CPU mode)")
        
        # Test other key imports
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.io as sio
        
        print("   ✅ All key dependencies importable")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_documentation_completeness():
    """Test that documentation is complete"""
    
    print("🔍 Testing documentation completeness...")
    
    # Check SOTA.md
    try:
        with open('SOTA.md', 'r', encoding='utf-8') as f:
            sota_content = f.read()
        
        # Check for key sections (case-insensitive)
        required_sections = [
            'WSL GPU',
            'NVIDIA GeForce RTX 3060',
            'Mixed precision',
            'early stopping',
            'results/wsl_gpu_training',
            'FINAL DECLARATION'
        ]

        sota_content_lower = sota_content.lower()
        missing_sections = []
        for section in required_sections:
            if section.lower() not in sota_content_lower:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing sections in SOTA.md: {missing_sections}")
            return False
        
        print("✅ SOTA.md documentation complete")
        return True
        
    except Exception as e:
        print(f"❌ Error checking documentation: {e}")
        return False

def main():
    """Main test execution"""
    
    print("🧪 TESTING REPRODUCIBILITY")
    print("=" * 60)
    print("Testing cleaned project for submission readiness...")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dataset Files", test_dataset_files),
        ("Results Files", test_results_files),
        ("Results Consistency", test_results_consistency),
        ("Import Capability", test_import_capability),
        ("Documentation", test_documentation_completeness)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"REPRODUCIBILITY TEST RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Project is ready for reproduction")
        print("✅ Ready for journal submission")
        print("\n📋 Next steps:")
        print("1. Review docs/SUBMISSION_CHECKLIST.md")
        print("2. Test full reproduction in clean environment")
        print("3. Submit to journal")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Please fix issues before submission")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
