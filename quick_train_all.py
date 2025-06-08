#!/usr/bin/env python3
"""
Quick training of all CortexFlow variants
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import json
import time

# Simple model implementations
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class MCModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 784), nn.Sigmoid()
        )
        self.uncertainty = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 784), nn.Softplus()
        )
    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        unc = self.uncertainty(z)
        return output, unc

class HierarchicalModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scale1 = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.scale2 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.scale3 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(448, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid()
        )
    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        combined = torch.cat([s1, s2, s3], dim=-1)
        return self.decoder(combined)

class EnhancedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hierarchical = HierarchicalModel(input_dim)
        self.uncertainty = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Softplus()
        )
        self.alignment = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.Tanh()
        )
    def forward(self, x):
        output = self.hierarchical(x)
        unc = self.uncertainty(x)
        align = self.alignment(x)
        return output, unc, align

class UnifiedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.complexity = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        self.simple = SimpleModel(input_dim)
        self.complex = HierarchicalModel(input_dim)
        self.uncertainty = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Softplus()
        )
    def forward(self, x):
        comp = self.complexity(x)
        simple_out = self.simple(x)
        complex_out = self.complex(x)
        output = comp * complex_out + (1 - comp) * simple_out
        unc = self.uncertainty(x)
        return output, unc, comp

def load_data(dataset_name):
    """Load dataset"""
    files = {
        'miyawaki': 'data/processed/miyawaki_structured_28x28.mat',
        'vangerven': 'data/processed/digit69_28x28.mat',
        'mindbigdata': 'data/processed/mindbigdata.mat',
        'crell': 'data/processed/crell.mat'
    }
    
    try:
        data = loadmat(files[dataset_name])
        if dataset_name == 'miyawaki':
            X, y = data['fmriTrn'], data['stimTrn']
        else:
            arrays = [(k, v) for k, v in data.items() 
                     if isinstance(v, np.ndarray) and v.ndim >= 2 and not k.startswith('__')]
            arrays.sort(key=lambda x: x[1].size, reverse=True)
            X, y = arrays[0][1], arrays[1][1]
        
        if X.ndim > 2: X = X.reshape(X.shape[0], -1)
        if y.ndim > 2: y = y.reshape(y.shape[0], -1)
        if y.shape[1] != 784:
            if y.shape[1] > 784: y = y[:, :784]
            else: y = np.pad(y, ((0, 0), (0, 784 - y.shape[1])))
        
        X = (X - X.mean()) / (X.std() + 1e-8)
        y = y / (y.max() + 1e-8) if y.max() > 1.0 else (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, None

def train_model(model, X, y, variant_name, dataset_name):
    """Train model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Split data
    n = len(X)
    n_train = int(0.8 * n)
    indices = torch.randperm(n)
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_test, y_test = X[indices[n_train:]], y[indices[n_train:]]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(30):
        # Train
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            if variant_name == 'simple':
                output = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant_name == 'mc':
                output, _ = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant_name == 'hierarchical':
                output = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant_name == 'enhanced':
                output, _, _ = model(batch_x)
                loss = criterion(output, batch_y)
            elif variant_name == 'unified':
                output, _, _ = model(batch_x)
                loss = criterion(output, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Test
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                if variant_name == 'simple':
                    output = model(batch_x)
                elif variant_name == 'mc':
                    output, _ = model(batch_x)
                elif variant_name == 'hierarchical':
                    output = model(batch_x)
                elif variant_name == 'enhanced':
                    output, _, _ = model(batch_x)
                elif variant_name == 'unified':
                    output, _, _ = model(batch_x)
                
                test_loss = criterion(output, batch_y)
                test_losses.append(test_loss.item())
        
        avg_test_loss = np.mean(test_losses)
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
    
    training_time = time.time() - start_time
    param_count = sum(p.numel() for p in model.parameters())
    
    print(f"✅ {variant_name} on {dataset_name}: {best_loss:.6f} ({param_count:,} params, {training_time:.1f}s)")
    
    return {
        'variant': variant_name,
        'dataset': dataset_name,
        'test_loss': best_loss,
        'parameters': param_count,
        'training_time': training_time
    }

def main():
    print("🚀 TRAINING ALL CORTEXFLOW VARIANTS")
    print("=" * 50)
    
    variants = {
        'simple': SimpleModel,
        'mc': MCModel,
        'hierarchical': HierarchicalModel,
        'enhanced': EnhancedModel,
        'unified': UnifiedModel
    }
    
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n📊 Dataset: {dataset_name}")
        X, y = load_data(dataset_name)
        if X is None:
            continue
            
        dataset_results = {}
        
        for variant_name, model_class in variants.items():
            model = model_class(X.shape[1])
            result = train_model(model, X, y, variant_name, dataset_name)
            dataset_results[variant_name] = result
        
        all_results[dataset_name] = dataset_results
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/comprehensive_training_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print("📊 Results saved to: results/comprehensive_training_results.json")
    
    # Summary
    print("\n📈 SUMMARY:")
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        for variant, result in results.items():
            print(f"  {variant:12}: {result['test_loss']:.6f}")

if __name__ == "__main__":
    main()
