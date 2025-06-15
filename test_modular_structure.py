"""
Test Modular Structure
=====================

Demonstration of clean modular imports and architecture.
"""

import torch

# Test clean imports from modular structure
try:
    from src.models.baseline import StandardBaselineCNN
    from src.models.cortexflow import CortexFlowMultiPathway
    
    print("🎉 MODULAR STRUCTURE TEST")
    print("=" * 50)
    
    print("✅ CLEAN IMPORTS SUCCESSFUL:")
    print("   from src.models.baseline import StandardBaselineCNN")
    print("   from src.models.cortexflow import CortexFlowMultiPathway")
    
    # Test model initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = 3000
    
    print(f"\n🔧 TESTING MODEL INITIALIZATION:")
    print(f"   Device: {device}")
    print(f"   Input dimension: {input_dim}")
    
    # Test Baseline CNN
    baseline = StandardBaselineCNN(input_dim, device)
    print(f"   ✅ {baseline.name}: Initialized successfully")
    
    # Test CortexFlow Multi-Pathway
    cortexflow = CortexFlowMultiPathway(input_dim, device)
    print(f"   ✅ {cortexflow.name}: Initialized successfully")
    
    # Test forward pass
    print(f"\n🚀 TESTING FORWARD PASS:")
    test_input = torch.randn(2, input_dim).to(device)
    
    with torch.no_grad():
        baseline_output = baseline(test_input)
        cortexflow_output = cortexflow(test_input)
        
        print(f"   ✅ Baseline output: {baseline_output.shape}")
        print(f"   ✅ CortexFlow output: {cortexflow_output.shape}")
    
    # Test uncertainty estimation
    print(f"\n🎯 TESTING UNCERTAINTY ESTIMATION:")
    with torch.no_grad():
        mean_pred, var_pred = cortexflow.get_uncertainty(test_input)
        print(f"   ✅ Mean prediction: {mean_pred.shape}")
        print(f"   ✅ Variance prediction: {var_pred.shape}")
    
    print(f"\n🎉 MODULAR STRUCTURE: FULLY FUNCTIONAL!")
    print("✅ Clean imports working")
    print("✅ Model initialization working") 
    print("✅ Forward pass working")
    print("✅ Advanced features working")
    print("✅ Professional architecture achieved!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Need to complete modular refactoring")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
