#!/usr/bin/env python3
"""
Simple test script for hierarchical architecture debugging
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add src to path
sys.path.append('../src/models')

from hierarchical import HierarchicalCortexFlow, HierarchicalConfig, create_hierarchical_model

def test_hierarchical_forward():
    """Test basic forward pass of hierarchical model."""
    print("🧪 Testing HierarchicalCortexFlow Forward Pass")
    print("=" * 60)
    
    # Simple config for testing
    config = HierarchicalConfig(
        TEMPORAL_SCALES=[1, 2],  # Reduced for testing
        HIDDEN_DIM=256,          # Reduced for testing
        NUM_PYRAMID_LEVELS=2,    # Reduced for testing
        IMAGE_SIZE=28
    )
    
    # Test with small input
    input_dim = 100  # Small for testing
    batch_size = 4
    
    print(f"📊 Test Configuration:")
    print(f"   Input dim: {input_dim}")
    print(f"   Batch size: {batch_size}")
    print(f"   Temporal scales: {config.TEMPORAL_SCALES}")
    print(f"   Hidden dim: {config.HIDDEN_DIM}")
    print(f"   Pyramid levels: {config.NUM_PYRAMID_LEVELS}")
    
    try:
        # Create model
        model = create_hierarchical_model(input_dim, config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create test data
        test_fmri = torch.randn(batch_size, input_dim, device=device)
        test_images = torch.randn(batch_size, 1, 28, 28, device=device)
        
        print(f"\n🔄 Testing forward pass...")
        print(f"   Input fMRI shape: {test_fmri.shape}")
        print(f"   Target images shape: {test_images.shape}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(test_fmri)
            print(f"✅ Forward pass successful!")
            print(f"   Main reconstruction shape: {outputs['reconstruction'].shape}")
            print(f"   Progressive outputs: {len(outputs['progressive_outputs'])} levels")
            
            for level, output in outputs['progressive_outputs'].items():
                print(f"   {level}: {output.shape}")
        
        # Test loss computation
        print(f"\n📉 Testing loss computation...")
        loss_dict = model.compute_loss(test_fmri, test_images)
        print(f"✅ Loss computation successful!")
        
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                print(f"   {key}: {value.item():.6f}")
        
        print(f"\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual components separately."""
    print("\n🔧 Testing Individual Components")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test TemporalEncoder
    print("🧠 Testing TemporalEncoder...")
    try:
        from hierarchical import TemporalEncoder
        encoder = TemporalEncoder(input_dim=100, scale=2, hidden_dim=256).to(device)
        test_input = torch.randn(4, 100, device=device)
        output = encoder(test_input)
        print(f"✅ TemporalEncoder: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"❌ TemporalEncoder failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test FeaturePyramidNetwork
    print("\n🏗️ Testing FeaturePyramidNetwork...")
    try:
        from hierarchical import FeaturePyramidNetwork
        fpn = FeaturePyramidNetwork(feature_dim=128, num_scales=2, fusion_type="attention").to(device)
        test_features = [torch.randn(4, 128, device=device) for _ in range(2)]
        output = fpn(test_features)
        print(f"✅ FeaturePyramidNetwork: {[f.shape for f in test_features]} -> {output.shape}")
    except Exception as e:
        print(f"❌ FeaturePyramidNetwork failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ProgressiveDecoder
    print("\n📈 Testing ProgressiveDecoder...")
    try:
        from hierarchical import ProgressiveDecoder
        decoder = ProgressiveDecoder(feature_dim=128, image_size=28, num_levels=2).to(device)
        test_features = torch.randn(4, 128, device=device)
        outputs = decoder(test_features)
        print(f"✅ ProgressiveDecoder: {test_features.shape} -> {len(outputs)} levels")
        for level, output in outputs.items():
            print(f"   {level}: {output.shape}")
    except Exception as e:
        print(f"❌ ProgressiveDecoder failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 HIERARCHICAL ARCHITECTURE DEBUG TEST")
    print("=" * 80)
    
    # Test individual components first
    test_individual_components()
    
    # Test full model
    success = test_hierarchical_forward()
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED! Hierarchical architecture is working correctly.")
    else:
        print(f"\n❌ TESTS FAILED! Need to debug hierarchical architecture.")
