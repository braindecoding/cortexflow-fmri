#!/usr/bin/env python3
"""
🧪 TEST ALL 4 DATASETS WITH MONTE CARLO SIMPLE CORTEXFLOW
================================================================================
Quick test to verify all 4 datasets work with Monte Carlo Simple CortexFlow:
1. Miyawaki (Visual Cortex fMRI) - 967 → 784 dimensions
2. Vangerven (Digit Recognition fMRI) - 3092 → 784 dimensions  
3. MindBigData (EEG-based) - 3092 → 784 dimensions
4. Crell (Advanced fMRI) - 3092 → 784 dimensions
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
from datetime import datetime
import time

# Import Monte Carlo Simple CortexFlow
from src.models.mc_simple_cortexflow import create_mc_simple_cortexflow

class TestConfig:
    """Configuration for dataset testing."""
    
    # Training parameters (reduced for quick testing)
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10  # Quick test with fewer epochs
    PATIENCE = 5
    WEIGHT_DECAY = 1e-5
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Datasets
    DATASETS = {
        'miyawaki': {
            'file': 'miyawaki_structured_28x28.mat',
            'description': 'Visual Cortex fMRI',
            'input_dim': 967,
            'output_dim': 784
        },
        'vangerven': {
            'file': 'digit69_28x28.mat', 
            'description': 'Digit Recognition fMRI',
            'input_dim': 3092,
            'output_dim': 784
        },
        'mindbigdata': {
            'file': 'mindbigdata.mat',
            'description': 'EEG-based Neural Signals',
            'input_dim': 3092,
            'output_dim': 784
        },
        'crell': {
            'file': 'crell.mat',
            'description': 'Advanced fMRI',
            'input_dim': 3092,
            'output_dim': 784
        }
    }

def load_dataset(dataset_name):
    """Load and preprocess dataset."""
    config = TestConfig.DATASETS[dataset_name]
    filepath = f"data/processed/{config['file']}"
    
    print(f"📁 Loading {dataset_name} from: {config['file']}")
    
    try:
        data = scipy.io.loadmat(filepath)
        
        # Extract data
        fmri_train = torch.FloatTensor(data['fmriTrn'])
        stim_train = torch.FloatTensor(data['stimTrn'])
        fmri_test = torch.FloatTensor(data['fmriTest'])
        stim_test = torch.FloatTensor(data['stimTest'])
        
        # Normalize stimuli to [0, 1] if needed
        if stim_train.max() > 1.0:
            stim_train = stim_train / 255.0
            stim_test = stim_test / 255.0
            
        print(f"  ✅ fmriTrn: {fmri_train.shape}")
        print(f"  ✅ stimTrn: {stim_train.shape}")
        print(f"  ✅ fmriTest: {fmri_test.shape}")
        print(f"  ✅ stimTest: {stim_test.shape}")
        
        return {
            'train_fmri': fmri_train,
            'train_stim': stim_train,
            'test_fmri': fmri_test,
            'test_stim': stim_test,
            'input_dim': config['input_dim'],
            'output_dim': config['output_dim']
        }
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None

def quick_train_test(model, dataset, dataset_name):
    """Quick training test on a dataset."""
    device = TestConfig.DEVICE
    model = model.to(device)
    
    # Prepare data
    train_fmri = dataset['train_fmri'].to(device)
    train_stim = dataset['train_stim'].to(device)
    test_fmri = dataset['test_fmri'].to(device)
    test_stim = dataset['test_stim'].to(device)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_fmri, train_stim)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=TestConfig.BATCH_SIZE, 
        shuffle=True
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=TestConfig.LEARNING_RATE, 
                          weight_decay=TestConfig.WEIGHT_DECAY)
    
    print(f"\n🚀 Quick training test: {dataset_name}")
    print(f"🔧 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(TestConfig.NUM_EPOCHS):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        for batch_fmri, batch_stim in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model.compute_loss(batch_fmri, batch_stim)
            loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_fmri)
            test_loss = nn.MSELoss()(test_outputs['reconstruction'], test_stim.view(-1, 1, 28, 28))
        
        if test_loss.item() < best_loss:
            best_loss = test_loss.item()
            
        if epoch % 2 == 0:
            print(f"  Epoch {epoch+1:2d}: Train {avg_train_loss:.6f}, Test {test_loss.item():.6f}")
    
    training_time = time.time() - start_time
    
    print(f"  🏆 Best test loss: {best_loss:.6f}")
    print(f"  ⏱️  Training time: {training_time:.1f}s")
    
    return {
        'best_loss': best_loss,
        'training_time': training_time,
        'parameters': sum(p.numel() for p in model.parameters())
    }

def main():
    """Run quick test on all datasets."""
    print("🧪 TESTING ALL 4 DATASETS WITH MONTE CARLO SIMPLE CORTEXFLOW")
    print("="*80)
    print(f"Test start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎮 Using device: {TestConfig.DEVICE}")
    
    # Store all results
    all_results = {}
    experiment_start = time.time()
    
    # Test each dataset
    for dataset_name in TestConfig.DATASETS.keys():
        print(f"\n{'='*60}")
        print(f"🧪 TESTING DATASET: {dataset_name.upper()}")
        print(f"📊 {TestConfig.DATASETS[dataset_name]['description']}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name)
            if dataset is None:
                all_results[dataset_name] = {'error': 'Failed to load dataset'}
                continue
            
            # Create model
            model = create_mc_simple_cortexflow(dataset['input_dim'])
            
            # Quick training test
            result = quick_train_test(model, dataset, dataset_name)
            all_results[dataset_name] = result
            
            print(f"✅ {dataset_name}: SUCCESS")
            
        except Exception as e:
            print(f"❌ Error testing {dataset_name}: {e}")
            all_results[dataset_name] = {'error': str(e)}
    
    # Summary
    total_time = time.time() - experiment_start
    
    print(f"\n🎉 ALL DATASET TESTS COMPLETED!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    print(f"\n📊 RESULTS SUMMARY:")
    print("-" * 80)
    for dataset_name, result in all_results.items():
        if 'error' in result:
            print(f"❌ {dataset_name:12} : FAILED - {result['error']}")
        else:
            print(f"✅ {dataset_name:12} : Loss {result['best_loss']:.6f}, "
                  f"Time {result['training_time']:.1f}s, "
                  f"Params {result['parameters']:,}")
    
    # Success rate
    successful = sum(1 for r in all_results.values() if 'error' not in r)
    total = len(all_results)
    print(f"\n📈 SUCCESS RATE: {successful}/{total} ({100*successful/total:.1f}%)")
    
    if successful == total:
        print("🎉 ALL DATASETS ARE COMPATIBLE WITH CORTEXFLOW!")
    else:
        print("⚠️  Some datasets need fixing before full comparison")

if __name__ == "__main__":
    main()
