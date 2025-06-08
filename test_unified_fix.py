#!/usr/bin/env python3
"""
Test script to verify Unified CortexFlow fix
"""

import os
import sys
import torch
from datetime import datetime

# Add project root to path
project_root = os.path.abspath('.')
sys.path.append(project_root)

def test_data_loading():
    """Test data loading with corrected paths."""
    print("🔍 Testing data loading...")
    
    data_files = [
        'data/processed/miyawaki_structured_28x28.mat',
        'data/processed/digit69_28x28.mat', 
        'data/processed/mindbigdata.mat',
        'data/processed/crell.mat'
    ]
    
    for file_path in data_files:
        exists = os.path.exists(file_path)
        print(f"  {file_path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    
    return all(os.path.exists(f) for f in data_files)

def test_unified_import():
    """Test importing Unified CortexFlow."""
    print("🔍 Testing Unified CortexFlow import...")
    
    try:
        from src.models.unified_cortexflow_fixed import create_unified_model, UnifiedCortexFlow, UnifiedLoss
        print("  ✅ Import successful")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_model_creation():
    """Test creating Unified CortexFlow model."""
    print("🔍 Testing model creation...")
    
    try:
        from src.models.unified_cortexflow_fixed import create_unified_model
        
        # Test simple config
        model, loss_fn = create_unified_model(input_dim=967, config='simple')
        print(f"  ✅ Simple config: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        x = torch.randn(4, 967)
        target = torch.randn(4, 784)
        
        outputs = model(x, target)
        losses = loss_fn(outputs, target)
        
        print(f"  ✅ Forward pass successful")
        print(f"  📊 Reconstruction shape: {outputs['reconstruction'].shape}")
        print(f"  🎯 Total loss: {losses['total_loss'].item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading_scipy():
    """Test loading data with scipy."""
    print("🔍 Testing scipy data loading...")
    
    try:
        import scipy.io
        data = scipy.io.loadmat('data/processed/miyawaki_structured_28x28.mat')
        keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"  ✅ Data loaded successfully")
        print(f"  📊 Keys: {keys}")
        return True
        
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 UNIFIED CORTEXFLOW FIX VERIFICATION")
    print("=" * 60)
    print(f"Test start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    tests = [
        ("Data Files Existence", test_data_loading),
        ("Scipy Data Loading", test_data_loading_scipy),
        ("Unified Import", test_unified_import),
        ("Model Creation", test_model_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"🧪 {test_name}")
        print(f"{'='*40}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Unified CortexFlow fix is working!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
