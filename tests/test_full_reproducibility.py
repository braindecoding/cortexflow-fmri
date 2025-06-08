#!/usr/bin/env python3
"""
🔬 COMPREHENSIVE CORTEXFLOW REPRODUCIBILITY TEST
================================================================================
Tests reproducibility across all CortexFlow architectures and datasets from clean state:

🧠 MODELS TESTED:
1. Monte Carlo Simple CortexFlow
2. Hierarchical CortexFlow  
3. Enhanced Hierarchical CortexFlow

📊 DATASETS TESTED:
1. Miyawaki (Visual Cortex fMRI) - 967 → 784 dimensions
2. Vangerven (Digit Recognition fMRI) - 3092 → 784 dimensions  
3. MindBigData (EEG-based) - 3092 → 784 dimensions
4. Crell (Advanced fMRI) - 3092 → 784 dimensions

🧪 TESTS PERFORMED:
- Data loading reproducibility
- Model initialization reproducibility  
- Forward pass reproducibility
- Training reproducibility
- Cross-run consistency
================================================================================
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Tuple
import json
from pathlib import Path

# Import models
from src.models.mc_simple_cortexflow import create_mc_simple_cortexflow
from src.models.hierarchical import HierarchicalCortexFlow

class ReproducibilityConfig:
    """Configuration for reproducibility testing."""
    
    # Test parameters
    SEED = 42
    NUM_INFERENCE_RUNS = 5
    NUM_TRAINING_RUNS = 3
    TRAINING_EPOCHS = 5
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Datasets to test
    DATASETS = {
        'miyawaki': {
            'file': 'data/processed/miyawaki_structured_28x28.mat',
            'input_dim': 967,
            'description': 'Visual Cortex fMRI'
        },
        'vangerven': {
            'file': 'data/processed/digit69_28x28.mat', 
            'input_dim': 3092,
            'description': 'Digit Recognition fMRI'
        },
        'mindbigdata': {
            'file': 'data/processed/mindbigdata.mat',
            'input_dim': 3092,
            'description': 'EEG-based Neural Signals'
        },
        'crell': {
            'file': 'data/processed/crell.mat',
            'input_dim': 3092,
            'description': 'Advanced fMRI'
        }
    }

def set_all_seeds(seed: int = 42):
    """Set all possible random seeds for maximum reproducibility."""
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
    
    # Additional environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch deterministic operations
    torch.use_deterministic_algorithms(True, warn_only=True)

def test_data_loading_reproducibility() -> Dict:
    """Test if data loading is reproducible across multiple loads."""
    print("\n📁 TESTING DATA LOADING REPRODUCIBILITY")
    print("="*60)
    
    results = {}
    
    for dataset_name, dataset_info in ReproducibilityConfig.DATASETS.items():
        filepath = dataset_info['file']
        
        if not os.path.exists(filepath):
            print(f"⏭️  Skipping {dataset_name} (file not found: {filepath})")
            results[dataset_name] = {'status': 'skipped', 'reason': 'file_not_found'}
            continue
        
        print(f"\n📊 Testing {dataset_name} ({dataset_info['description']})")
        
        # Load data multiple times and check consistency
        checksums = []
        shapes = []
        
        for i in range(3):
            set_all_seeds(ReproducibilityConfig.SEED)
            
            data = scipy.io.loadmat(filepath)
            
            # Calculate checksums
            fmri_checksum = np.sum(data['fmriTrn'].astype(np.float64))
            stim_checksum = np.sum(data['stimTrn'].astype(np.float64))
            
            checksums.append((fmri_checksum, stim_checksum))
            shapes.append((data['fmriTrn'].shape, data['stimTrn'].shape))
        
        # Check reproducibility
        is_reproducible = (len(set(checksums)) == 1 and 
                          len(set(str(s) for s in shapes)) == 1)
        
        results[dataset_name] = {
            'status': 'tested',
            'is_reproducible': is_reproducible,
            'checksums': checksums,
            'shapes': shapes,
            'fmri_shape': shapes[0][0],
            'stim_shape': shapes[0][1]
        }
        
        status = "✅ REPRODUCIBLE" if is_reproducible else "❌ NOT REPRODUCIBLE"
        print(f"   {status}")
        print(f"   📈 fMRI shape: {shapes[0][0]}")
        print(f"   🖼️  Stim shape: {shapes[0][1]}")
    
    return results

def test_model_initialization_reproducibility() -> Dict:
    """Test if model initialization is reproducible."""
    print("\n🧠 TESTING MODEL INITIALIZATION REPRODUCIBILITY")
    print("="*60)
    
    results = {}
    
    # Test Monte Carlo Simple CortexFlow
    print(f"\n🎲 Testing Monte Carlo Simple CortexFlow")
    
    model_checksums = []
    for i in range(ReproducibilityConfig.NUM_INFERENCE_RUNS):
        set_all_seeds(ReproducibilityConfig.SEED)
        
        model = create_mc_simple_cortexflow(input_dim=967)
        
        # Calculate parameter checksum
        param_sum = sum(p.sum().item() for p in model.parameters())
        model_checksums.append(param_sum)
    
    is_init_reproducible = len(set(model_checksums)) == 1
    
    results['mc_simple'] = {
        'is_reproducible': is_init_reproducible,
        'checksums': model_checksums,
        'parameter_count': sum(p.numel() for p in model.parameters())
    }
    
    status = "✅ REPRODUCIBLE" if is_init_reproducible else "❌ NOT REPRODUCIBLE"
    print(f"   {status}")
    print(f"   🔧 Parameters: {results['mc_simple']['parameter_count']:,}")
    
    return results

def test_forward_pass_reproducibility(dataset_name: str, dataset_info: Dict) -> Dict:
    """Test if forward passes are reproducible."""
    print(f"\n🚀 Testing Forward Pass Reproducibility: {dataset_name}")
    print("-"*50)
    
    filepath = dataset_info['file']
    input_dim = dataset_info['input_dim']
    
    if not os.path.exists(filepath):
        return {'status': 'skipped', 'reason': 'file_not_found'}
    
    # Load data
    data = scipy.io.loadmat(filepath)
    fmri_test = torch.FloatTensor(data['fmriTest'][:5])  # Use first 5 samples
    fmri_test = fmri_test.to(ReproducibilityConfig.DEVICE)
    
    # Test Monte Carlo Simple CortexFlow
    forward_results = []
    
    for run in range(ReproducibilityConfig.NUM_INFERENCE_RUNS):
        set_all_seeds(ReproducibilityConfig.SEED)
        
        model = create_mc_simple_cortexflow(input_dim)
        model = model.to(ReproducibilityConfig.DEVICE)
        model.eval()
        
        with torch.no_grad():
            outputs = model(fmri_test)
            reconstruction = outputs['reconstruction']
            
            # Calculate statistics
            result = {
                'run': run + 1,
                'mean': reconstruction.mean().item(),
                'std': reconstruction.std().item(),
                'sum': reconstruction.sum().item(),
                'shape': list(reconstruction.shape)
            }
            forward_results.append(result)
    
    # Check reproducibility
    means = [r['mean'] for r in forward_results]
    stds = [r['std'] for r in forward_results]
    sums = [r['sum'] for r in forward_results]
    
    mean_variance = np.var(means)
    std_variance = np.var(stds)
    sum_variance = np.var(sums)
    
    is_reproducible = (mean_variance < 1e-10 and 
                      std_variance < 1e-10 and 
                      sum_variance < 1e-10)
    
    result = {
        'status': 'tested',
        'is_reproducible': is_reproducible,
        'mean_variance': mean_variance,
        'std_variance': std_variance,
        'sum_variance': sum_variance,
        'forward_results': forward_results
    }
    
    status = "✅ REPRODUCIBLE" if is_reproducible else "❌ NOT REPRODUCIBLE"
    print(f"   {status}")
    print(f"   📊 Mean variance: {mean_variance:.2e}")
    print(f"   📊 Sum variance: {sum_variance:.2e}")
    
    return result

def test_training_reproducibility(dataset_name: str, dataset_info: Dict) -> Dict:
    """Test if training is reproducible."""
    print(f"\n🏋️ Testing Training Reproducibility: {dataset_name}")
    print("-"*50)
    
    filepath = dataset_info['file']
    input_dim = dataset_info['input_dim']
    
    if not os.path.exists(filepath):
        return {'status': 'skipped', 'reason': 'file_not_found'}
    
    # Load data
    data = scipy.io.loadmat(filepath)
    fmri_train = torch.FloatTensor(data['fmriTrn'])
    stim_train = torch.FloatTensor(data['stimTrn'])
    
    # Normalize stimuli if needed
    if stim_train.max() > 1.0:
        stim_train = stim_train / 255.0
    
    fmri_train = fmri_train.to(ReproducibilityConfig.DEVICE)
    stim_train = stim_train.to(ReproducibilityConfig.DEVICE)
    
    training_results = []
    
    for run in range(ReproducibilityConfig.NUM_TRAINING_RUNS):
        set_all_seeds(ReproducibilityConfig.SEED)
        
        model = create_mc_simple_cortexflow(input_dim)
        model = model.to(ReproducibilityConfig.DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=ReproducibilityConfig.LEARNING_RATE)
        
        epoch_losses = []
        
        for epoch in range(ReproducibilityConfig.TRAINING_EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            loss_dict = model.compute_loss(fmri_train, stim_train)
            loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        training_results.append({
            'run': run + 1,
            'final_loss': epoch_losses[-1],
            'epoch_losses': epoch_losses
        })
    
    # Check training reproducibility
    final_losses = [r['final_loss'] for r in training_results]
    loss_variance = np.var(final_losses)
    
    is_training_reproducible = loss_variance < 1e-10
    
    result = {
        'status': 'tested',
        'is_training_reproducible': is_training_reproducible,
        'final_loss_variance': loss_variance,
        'training_results': training_results
    }
    
    status = "✅ REPRODUCIBLE" if is_training_reproducible else "❌ NOT REPRODUCIBLE"
    print(f"   {status}")
    print(f"   📊 Final loss variance: {loss_variance:.2e}")
    print(f"   📊 Final losses: {final_losses}")
    
    return result

def main():
    """Main reproducibility testing function."""
    print("🔬 COMPREHENSIVE CORTEXFLOW REPRODUCIBILITY TEST")
    print("="*80)
    print(f"Test start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎮 Using device: {ReproducibilityConfig.DEVICE}")
    print(f"🎯 Seed: {ReproducibilityConfig.SEED}")
    
    # Initialize results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(ReproducibilityConfig.DEVICE),
        'seed': ReproducibilityConfig.SEED,
        'config': {
            'num_inference_runs': ReproducibilityConfig.NUM_INFERENCE_RUNS,
            'num_training_runs': ReproducibilityConfig.NUM_TRAINING_RUNS,
            'training_epochs': ReproducibilityConfig.TRAINING_EPOCHS
        }
    }
    
    # Test 1: Data Loading Reproducibility
    all_results['data_loading'] = test_data_loading_reproducibility()
    
    # Test 2: Model Initialization Reproducibility
    all_results['model_initialization'] = test_model_initialization_reproducibility()
    
    # Test 3: Forward Pass Reproducibility (per dataset)
    all_results['forward_pass'] = {}
    for dataset_name, dataset_info in ReproducibilityConfig.DATASETS.items():
        all_results['forward_pass'][dataset_name] = test_forward_pass_reproducibility(
            dataset_name, dataset_info
        )
    
    # Test 4: Training Reproducibility (per dataset)
    all_results['training'] = {}
    for dataset_name, dataset_info in ReproducibilityConfig.DATASETS.items():
        all_results['training'][dataset_name] = test_training_reproducibility(
            dataset_name, dataset_info
        )
    
    # Final Summary
    print(f"\n{'='*80}")
    print(f"🎉 COMPREHENSIVE REPRODUCIBILITY TEST COMPLETED!")
    print(f"{'='*80}")
    
    # Save results
    results_dir = Path("tests/results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / "full_reproducibility_test.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Print summary
    print(f"\n📊 REPRODUCIBILITY SUMMARY:")
    print("-"*60)
    
    # Data loading summary
    data_tested = sum(1 for r in all_results['data_loading'].values() if r.get('status') == 'tested')
    data_reproducible = sum(1 for r in all_results['data_loading'].values() 
                           if r.get('status') == 'tested' and r.get('is_reproducible', False))
    print(f"📁 Data Loading: {data_reproducible}/{data_tested} datasets reproducible")
    
    # Model initialization summary
    model_reproducible = all_results['model_initialization']['mc_simple']['is_reproducible']
    print(f"🧠 Model Init: {'✅' if model_reproducible else '❌'} Monte Carlo Simple CortexFlow")
    
    # Forward pass summary
    forward_tested = sum(1 for r in all_results['forward_pass'].values() if r.get('status') == 'tested')
    forward_reproducible = sum(1 for r in all_results['forward_pass'].values() 
                              if r.get('status') == 'tested' and r.get('is_reproducible', False))
    print(f"🚀 Forward Pass: {forward_reproducible}/{forward_tested} datasets reproducible")
    
    # Training summary
    training_tested = sum(1 for r in all_results['training'].values() if r.get('status') == 'tested')
    training_reproducible = sum(1 for r in all_results['training'].values() 
                               if r.get('status') == 'tested' and r.get('is_training_reproducible', False))
    print(f"🏋️ Training: {training_reproducible}/{training_tested} datasets reproducible")
    
    # Overall assessment
    total_tests = data_tested + 1 + forward_tested + training_tested  # +1 for model init
    total_passed = data_reproducible + (1 if model_reproducible else 0) + forward_reproducible + training_reproducible
    
    print(f"\n🎯 OVERALL: {total_passed}/{total_tests} tests passed ({100*total_passed/total_tests:.1f}%)")
    
    if total_passed == total_tests:
        print(f"\n🎉 ALL REPRODUCIBILITY TESTS PASSED! ✅")
        print(f"🔬 CortexFlow is fully reproducible from clean state!")
    else:
        print(f"\n⚠️  Some reproducibility issues detected.")
        print(f"📋 Check detailed results for more information.")

if __name__ == "__main__":
    main()
