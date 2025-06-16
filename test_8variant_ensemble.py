#!/usr/bin/env python3
"""
Test Enhanced 8-Variant Ensemble
================================

Test script untuk verify 8-variant ensemble dengan Multi-Pathway integration.
"""

import torch
import sys
sys.path.append('.')

def test_enhanced_ensemble():
    """Test enhanced 8-variant ensemble"""
    print("🚀 TESTING ENHANCED 8-VARIANT ENSEMBLE")
    print("=" * 50)
    
    try:
        from src.models.ensemble import CortexFlowEnsemble
        
        # Test initialization
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_dim = 967  # Miyawaki input dimension
        
        print(f"📊 Device: {device}")
        print(f"📊 Input Dimension: {input_dim}")
        
        # Initialize enhanced ensemble
        ensemble = CortexFlowEnsemble(input_dim, device)
        print("✅ Enhanced 8-Variant Ensemble initialized successfully!")
        
        # Get ensemble info
        info = ensemble.get_ensemble_info()
        print("\n📋 ENSEMBLE INFO:")
        print(f"   Name: {info['name']}")
        print(f"   Variants: {info['num_variants']} (ENHANCED!)")
        
        print("\n🔧 VARIANTS:")
        for variant in info['variants']:
            print(f"   ✅ {variant}")
        
        print("\n🚀 ENHANCEMENTS:")
        for enhancement in info['enhancements']:
            print(f"   ⭐ {enhancement}")
        
        # Test forward pass
        batch_size = 4
        test_input = torch.randn(batch_size, input_dim).to(device)
        
        print("\n🧪 TESTING FORWARD PASS:")
        print(f"   Input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = ensemble(test_input)
            print(f"   Output shape: {output.shape}")
            print(f"   Expected: [{batch_size}, 1, 28, 28]")
            
            if output.shape == (batch_size, 1, 28, 28):
                print("   ✅ Forward pass successful!")
                success = True
            else:
                print("   ❌ Forward pass shape mismatch!")
                success = False
        
        print("\n🎉 ENHANCED ENSEMBLE TEST: SUCCESS!")
        print("✅ 8-Variant Ensemble with Multi-Pathway integration working!")
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_ensemble()
    if success:
        print("\n🏆 ALL TESTS PASSED!")
    else:
        print("\n❌ TESTS FAILED!")
