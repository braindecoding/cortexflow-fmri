#!/usr/bin/env python3
"""
Reproducibility Test for CortexFlow Architectures
Tests if models produce consistent results across multiple runs
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import random
import scipy.io
from datetime import datetime
from typing import Dict, List, Tuple
import json

# Add src to path
sys.path.append('../src')
sys.path.append('../src/models')

# Import all architectures
try:
    from hierarchical import HierarchicalCortexFlow, HierarchicalConfig, create_hierarchical_model
    from enhanced_hierarchical_training import EnhancedHierarchicalCortexFlow, EnhancedConfig, create_enhanced_model
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure hierarchical.py and enhanced_hierarchical_training.py exist")
    sys.exit(1)

def set_all_seeds(seed: int = 42):
    """Set all possible random seeds for maximum reproducibility."""
    print(f"🎯 Setting all seeds to: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch deterministic operations
    torch.use_deterministic_algorithms(True, warn_only=True)

def create_synthetic_data(input_dim: int, num_samples: int = 20, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic data for reproducibility testing."""
    set_all_seeds(seed)
    
    # Create synthetic fMRI data
    fmri_data = torch.randn(num_samples, input_dim)
    
    # Create synthetic image data (28x28 flattened)
    image_data = torch.randn(num_samples, 784)
    
    return fmri_data, image_data

def test_model_reproducibility(model_class, config_class, model_name: str, 
                             input_dim: int = 100, num_runs: int = 3) -> Dict:
    """Test if a model produces reproducible results."""
    print(f"\n🧪 Testing {model_name} Reproducibility")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    # Create test data once
    test_fmri, test_images = create_synthetic_data(input_dim, num_samples=10, seed=42)
    test_fmri = test_fmri.to(device)
    test_images = test_images.to(device)
    
    for run in range(num_runs):
        print(f"\n🔄 Run {run + 1}/{num_runs}")
        
        # Set seeds before each run
        set_all_seeds(42)  # Same seed for reproducibility
        
        # Create model
        if model_class == HierarchicalCortexFlow:
            config = config_class()
            model = create_hierarchical_model(input_dim, config)
        else:  # Enhanced model
            config = config_class()
            model = create_enhanced_model(input_dim, config)
        
        model = model.to(device)
        model.eval()
        
        # Test forward pass
        with torch.no_grad():
            if hasattr(model, 'forward_with_uncertainty'):
                outputs = model.forward_with_uncertainty(test_fmri, mc_samples=5)
                reconstruction = outputs['reconstruction']
            else:
                outputs = model(test_fmri)
                reconstruction = outputs['reconstruction']
        
        # Store results
        run_result = {
            'run': run + 1,
            'reconstruction_mean': reconstruction.mean().item(),
            'reconstruction_std': reconstruction.std().item(),
            'reconstruction_sum': reconstruction.sum().item(),
            'model_params': sum(p.numel() for p in model.parameters()),
            'output_shape': list(reconstruction.shape)
        }
        
        results.append(run_result)
        
        print(f"   📊 Reconstruction mean: {run_result['reconstruction_mean']:.8f}")
        print(f"   📊 Reconstruction std:  {run_result['reconstruction_std']:.8f}")
        print(f"   📊 Reconstruction sum:  {run_result['reconstruction_sum']:.8f}")
    
    # Check reproducibility
    means = [r['reconstruction_mean'] for r in results]
    stds = [r['reconstruction_std'] for r in results]
    sums = [r['reconstruction_sum'] for r in results]
    
    mean_variance = np.var(means)
    std_variance = np.var(stds)
    sum_variance = np.var(sums)
    
    is_reproducible = (mean_variance < 1e-10 and 
                      std_variance < 1e-10 and 
                      sum_variance < 1e-10)
    
    summary = {
        'model_name': model_name,
        'num_runs': num_runs,
        'is_reproducible': is_reproducible,
        'mean_variance': mean_variance,
        'std_variance': std_variance,
        'sum_variance': sum_variance,
        'results': results
    }
    
    if is_reproducible:
        print(f"\n✅ {model_name} is REPRODUCIBLE!")
        print(f"   📊 Mean variance: {mean_variance:.2e}")
        print(f"   📊 Std variance:  {std_variance:.2e}")
        print(f"   📊 Sum variance:  {sum_variance:.2e}")
    else:
        print(f"\n❌ {model_name} is NOT reproducible!")
        print(f"   📊 Mean variance: {mean_variance:.2e}")
        print(f"   📊 Std variance:  {std_variance:.2e}")
        print(f"   📊 Sum variance:  {sum_variance:.2e}")
    
    return summary

def test_training_reproducibility(model_class, config_class, model_name: str, 
                                input_dim: int = 100, num_epochs: int = 3) -> Dict:
    """Test if training produces reproducible results."""
    print(f"\n🏋️ Testing {model_name} Training Reproducibility")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_results = []
    
    # Create training data
    train_fmri, train_images = create_synthetic_data(input_dim, num_samples=50, seed=42)
    train_fmri = train_fmri.to(device)
    train_images = train_images.to(device)
    
    for run in range(2):  # Test 2 training runs
        print(f"\n🔄 Training Run {run + 1}/2")
        
        # Set seeds before each training run
        set_all_seeds(42)
        
        # Create model
        if model_class == HierarchicalCortexFlow:
            config = config_class()
            model = create_hierarchical_model(input_dim, config)
        else:
            config = config_class()
            model = create_enhanced_model(input_dim, config)
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        epoch_losses = []
        
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            if hasattr(model, 'compute_enhanced_loss'):
                loss_dict = model.compute_enhanced_loss(train_fmri, train_images)
                loss = loss_dict['enhanced_total_loss']
            else:
                loss_dict = model.compute_loss(train_fmri, train_images)
                loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            print(f"   Epoch {epoch + 1}: Loss = {loss.item():.8f}")
        
        training_results.append({
            'run': run + 1,
            'final_loss': epoch_losses[-1],
            'epoch_losses': epoch_losses,
            'loss_progression': [epoch_losses[i] - epoch_losses[i-1] for i in range(1, len(epoch_losses))]
        })
    
    # Check training reproducibility
    final_losses = [r['final_loss'] for r in training_results]
    loss_variance = np.var(final_losses)
    
    is_training_reproducible = loss_variance < 1e-10
    
    summary = {
        'model_name': model_name,
        'num_training_runs': 2,
        'num_epochs': num_epochs,
        'is_training_reproducible': is_training_reproducible,
        'final_loss_variance': loss_variance,
        'training_results': training_results
    }
    
    if is_training_reproducible:
        print(f"\n✅ {model_name} training is REPRODUCIBLE!")
        print(f"   📊 Final loss variance: {loss_variance:.2e}")
    else:
        print(f"\n❌ {model_name} training is NOT reproducible!")
        print(f"   📊 Final loss variance: {loss_variance:.2e}")
    
    return summary

def test_data_loading_reproducibility() -> Dict:
    """Test if data loading is reproducible."""
    print(f"\n📁 Testing Data Loading Reproducibility")
    print("=" * 60)
    
    datasets = [
        '../data/processed/miyawaki_structured_28x28.mat',
        '../data/processed/digit69_28x28.mat'
    ]
    
    data_results = {}
    
    for dataset_path in datasets:
        if not os.path.exists(dataset_path):
            print(f"⏭️  Skipping {dataset_path} (not found)")
            continue
        
        dataset_name = os.path.basename(dataset_path).replace('.mat', '')
        print(f"\n📊 Testing {dataset_name}")
        
        # Load data multiple times
        checksums = []
        shapes = []
        
        for i in range(3):
            data = scipy.io.loadmat(dataset_path)
            
            # Calculate checksums for reproducibility
            if 'fmriTrn' in data:
                fmri_checksum = np.sum(data['fmriTrn'])
                stim_checksum = np.sum(data['stimTrn'])
                checksums.append((fmri_checksum, stim_checksum))
                shapes.append((data['fmriTrn'].shape, data['stimTrn'].shape))
        
        # Check if all checksums are identical
        is_data_reproducible = len(set(checksums)) == 1 and len(set(str(s) for s in shapes)) == 1
        
        data_results[dataset_name] = {
            'is_reproducible': is_data_reproducible,
            'checksums': checksums,
            'shapes': shapes
        }
        
        if is_data_reproducible:
            print(f"   ✅ {dataset_name} data loading is reproducible")
        else:
            print(f"   ❌ {dataset_name} data loading is NOT reproducible")
    
    return data_results

def main():
    """Main reproducibility testing function."""
    print("🔬 CORTEXFLOW REPRODUCIBILITY TEST")
    print("=" * 80)
    print(f"Test start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Using device: {device}")
    
    # Test data loading reproducibility
    data_results = test_data_loading_reproducibility()
    
    # Test model reproducibility
    model_tests = [
        (HierarchicalCortexFlow, HierarchicalConfig, "HierarchicalCortexFlow"),
        (EnhancedHierarchicalCortexFlow, EnhancedConfig, "EnhancedHierarchicalCortexFlow")
    ]
    
    inference_results = []
    training_results = []
    
    for model_class, config_class, model_name in model_tests:
        try:
            # Test inference reproducibility
            inference_result = test_model_reproducibility(
                model_class, config_class, model_name, input_dim=100, num_runs=3
            )
            inference_results.append(inference_result)
            
            # Test training reproducibility
            training_result = test_training_reproducibility(
                model_class, config_class, model_name, input_dim=100, num_epochs=3
            )
            training_results.append(training_result)
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 REPRODUCIBILITY TEST COMPLETED!")
    print(f"{'='*80}")
    
    print(f"\n📊 SUMMARY:")
    print("-" * 60)
    
    # Data loading summary
    print(f"📁 Data Loading:")
    for dataset_name, result in data_results.items():
        status = "✅ REPRODUCIBLE" if result['is_reproducible'] else "❌ NOT REPRODUCIBLE"
        print(f"   {dataset_name}: {status}")
    
    # Inference summary
    print(f"\n🧪 Model Inference:")
    for result in inference_results:
        status = "✅ REPRODUCIBLE" if result['is_reproducible'] else "❌ NOT REPRODUCIBLE"
        print(f"   {result['model_name']}: {status}")
    
    # Training summary
    print(f"\n🏋️ Model Training:")
    for result in training_results:
        status = "✅ REPRODUCIBLE" if result['is_training_reproducible'] else "❌ NOT REPRODUCIBLE"
        print(f"   {result['model_name']}: {status}")
    
    # Save detailed results
    detailed_results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'data_loading': data_results,
        'inference_tests': inference_results,
        'training_tests': training_results
    }
    
    os.makedirs('results', exist_ok=True)
    with open('test_results/reproducibility_test.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: test_results/reproducibility_test.json")
    
    # Overall assessment
    all_data_reproducible = all(r['is_reproducible'] for r in data_results.values())
    all_inference_reproducible = all(r['is_reproducible'] for r in inference_results)
    all_training_reproducible = all(r['is_training_reproducible'] for r in training_results)
    
    if all_data_reproducible and all_inference_reproducible and all_training_reproducible:
        print(f"\n🎉 ALL TESTS PASSED! CortexFlow is fully reproducible! ✅")
    else:
        print(f"\n⚠️  Some reproducibility issues detected. Check details above.")

if __name__ == "__main__":
    main()
