#!/usr/bin/env python3
"""
🚀 FULL CORTEXFLOW COMPARISON TRAINING
================================================================================
Comprehensive training and comparison across all 4 datasets and all architectures:

📊 DATASETS:
1. Miyawaki (Visual Cortex fMRI) - 967 → 784 dimensions
2. Vangerven (Digit Recognition fMRI) - 3092 → 784 dimensions  
3. MindBigData (EEG-based) - 3092 → 784 dimensions
4. Crell (Advanced fMRI) - 3092 → 784 dimensions

🧠 ARCHITECTURES:
1. Simple CortexFlow
2. Monte Carlo Simple CortexFlow
3. Hierarchical CortexFlow
4. Enhanced Hierarchical CortexFlow

Total: 4 datasets × 4 architectures = 16 experiments
================================================================================
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io

from datetime import datetime
import time
import json
from pathlib import Path

# Import all models
from src.models.mc_simple_cortexflow import create_mc_simple_cortexflow
from src.models.hierarchical import HierarchicalCortexFlow

class FullComparisonConfig:
    """Configuration for full comparison training."""
    
    # Training parameters (reduced for faster comparison)
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    PATIENCE = 15
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
    
    # Model architectures (starting with MC Simple for now)
    ARCHITECTURES = {
        'mc_simple': {
            'name': 'Monte Carlo Simple CortexFlow',
            'create_fn': lambda input_dim: create_mc_simple_cortexflow(input_dim),
            'color': '#4ECDC4'
        }
    }

def create_simple_cortexflow(input_dim):
    """Create Simple CortexFlow model."""
    class SimpleCortexFlow(nn.Module):
        def __init__(self, input_dim, hidden_dim=512, output_dim=784):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim//2)
            )

            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim//2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded.view(-1, 1, 28, 28)

    return SimpleCortexFlow(input_dim)

def create_hierarchical_cortexflow(input_dim):
    """Create Hierarchical CortexFlow model."""
    return HierarchicalCortexFlow(
        input_dim=input_dim,
        temporal_scales=[1, 2, 4, 8],
        hidden_dim=512,
        pyramid_levels=4,
        skip_connections=True,
        feature_fusion='attention'
    )

def create_enhanced_hierarchical_cortexflow(input_dim):
    """Create Enhanced Hierarchical CortexFlow model."""
    # For now, use the same as hierarchical since enhanced model needs to be implemented
    return create_hierarchical_cortexflow(input_dim)

def load_dataset(dataset_name):
    """Load and preprocess dataset."""
    config = FullComparisonConfig.DATASETS[dataset_name]
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

def train_model(model, dataset, dataset_name, architecture_name):
    """Train a model on a dataset."""
    device = FullComparisonConfig.DEVICE
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
        batch_size=FullComparisonConfig.BATCH_SIZE, 
        shuffle=True
    )
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=FullComparisonConfig.LEARNING_RATE, 
                          weight_decay=FullComparisonConfig.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []
    
    print(f"\n🚀 Training {architecture_name} on {dataset_name}")
    print(f"🔧 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    start_time = time.time()
    
    for epoch in range(FullComparisonConfig.NUM_EPOCHS):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        for batch_fmri, batch_stim in train_loader:
            optimizer.zero_grad()
            
            if 'mc_simple' in architecture_name.lower():
                outputs = model(batch_fmri)
                if isinstance(outputs, dict):
                    loss_dict = model.compute_loss(batch_fmri, batch_stim)
                    loss = loss_dict['total_loss']
                else:
                    loss = criterion(outputs, batch_stim.view(-1, 1, 28, 28))
            else:
                outputs = model(batch_fmri)
                if isinstance(outputs, dict):
                    loss = criterion(outputs['reconstruction'], batch_stim.view(-1, 1, 28, 28))
                else:
                    loss = criterion(outputs, batch_stim.view(-1, 1, 28, 28))
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            if 'mc_simple' in architecture_name.lower():
                test_outputs = model(test_fmri)
                if isinstance(test_outputs, dict):
                    test_loss = criterion(test_outputs['reconstruction'], test_stim.view(-1, 1, 28, 28))
                else:
                    test_loss = criterion(test_outputs, test_stim.view(-1, 1, 28, 28))
            else:
                test_outputs = model(test_fmri)
                if isinstance(test_outputs, dict):
                    test_loss = criterion(test_outputs['reconstruction'], test_stim.view(-1, 1, 28, 28))
                else:
                    test_loss = criterion(test_outputs, test_stim.view(-1, 1, 28, 28))
        
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss.item())
        
        # Early stopping
        if test_loss.item() < best_loss:
            best_loss = test_loss.item()
            patience_counter = 0
            print(f"Epoch {epoch+1:3d}: Train {avg_train_loss:.6f}, Test {test_loss.item():.6f} ✅")
        else:
            patience_counter += 1
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Train {avg_train_loss:.6f}, Test {test_loss.item():.6f} (patience: {patience_counter})")
        
        if patience_counter >= FullComparisonConfig.PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    return {
        'best_loss': best_loss,
        'final_epoch': epoch + 1,
        'training_time': training_time,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'parameters': sum(p.numel() for p in model.parameters())
    }

def main():
    """Run full comparison training."""
    print("🚀 FULL CORTEXFLOW COMPARISON TRAINING")
    print("="*80)
    print(f"Experiment start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎮 Using device: {FullComparisonConfig.DEVICE}")
    
    # Create results directory
    results_dir = Path("results/full_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = {}
    experiment_start = time.time()
    
    # Load all datasets
    datasets = {}
    for dataset_name in FullComparisonConfig.DATASETS.keys():
        print(f"\n📊 Loading {dataset_name}...")
        dataset = load_dataset(dataset_name)
        if dataset is not None:
            datasets[dataset_name] = dataset
        else:
            print(f"❌ Skipping {dataset_name} due to loading error")
    
    print(f"\n✅ Successfully loaded {len(datasets)} datasets")
    
    # Run experiments
    total_experiments = len(datasets) * len(FullComparisonConfig.ARCHITECTURES)
    current_experiment = 0
    
    for dataset_name, dataset in datasets.items():
        all_results[dataset_name] = {}
        
        for arch_name, arch_config in FullComparisonConfig.ARCHITECTURES.items():
            current_experiment += 1
            
            print(f"\n{'='*80}")
            print(f"🧪 EXPERIMENT {current_experiment}/{total_experiments}")
            print(f"📊 Dataset: {dataset_name.upper()}")
            print(f"🧠 Architecture: {arch_config['name']}")
            print(f"{'='*80}")
            
            try:
                # Create model
                model = arch_config['create_fn'](dataset['input_dim'])
                
                # Train model
                result = train_model(model, dataset, dataset_name, arch_name)
                all_results[dataset_name][arch_name] = result
                
                print(f"✅ {arch_name} on {dataset_name}: Loss {result['best_loss']:.6f}")
                
            except Exception as e:
                print(f"❌ Error training {arch_name} on {dataset_name}: {e}")
                all_results[dataset_name][arch_name] = {'error': str(e)}
    
    # Save results
    total_time = time.time() - experiment_start
    
    # Create summary
    summary = {
        'experiment_info': {
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_minutes': total_time / 60,
            'total_experiments': total_experiments,
            'device': str(FullComparisonConfig.DEVICE)
        },
        'results': all_results
    }
    
    # Save to JSON
    with open(results_dir / 'full_comparison_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 FULL COMPARISON COMPLETED!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📄 Results saved to: {results_dir / 'full_comparison_results.json'}")

if __name__ == "__main__":
    main()
