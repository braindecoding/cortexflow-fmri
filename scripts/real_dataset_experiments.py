#!/usr/bin/env python3
"""
CortexFlow Real Dataset Experiments
Using actual fMRI and EEG datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.io import loadmat
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class RealDatasetLoader:
    """Load and preprocess real datasets"""
    
    def __init__(self, data_dir="data/processed"):
        self.data_dir = Path(data_dir)
        print(f"🔧 Dataset loader initialized: {self.data_dir}")
        
    def load_miyawaki(self):
        """Load Miyawaki dataset"""
        print("📊 Loading Miyawaki dataset...")
        
        file_path = self.data_dir / "miyawaki_structured_28x28.mat"
        if not file_path.exists():
            print(f"❌ Miyawaki dataset not found: {file_path}")
            return None, None
            
        data = loadmat(file_path)
        
        # Extract fMRI and image data
        if 'fmri_data' in data and 'image_data' in data:
            X = data['fmri_data']  # fMRI voxel data
            y = data['image_data']  # 28x28 images flattened
        elif 'X' in data and 'y' in data:
            X = data['X']
            y = data['y']
        else:
            # Try to find the right keys
            keys = [k for k in data.keys() if not k.startswith('__')]
            print(f"Available keys: {keys}")
            # Use the largest arrays
            arrays = [(k, v) for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim >= 2]
            arrays.sort(key=lambda x: x[1].size, reverse=True)
            if len(arrays) >= 2:
                X = arrays[0][1]  # Largest array (likely fMRI)
                y = arrays[1][1]  # Second largest (likely images)
            else:
                print("❌ Could not identify data arrays")
                return None, None
        
        # Ensure correct shapes
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
            
        # Ensure y is 784 dimensions (28x28)
        if y.shape[1] != 784:
            if y.shape[1] > 784:
                y = y[:, :784]
            else:
                # Pad if necessary
                pad_size = 784 - y.shape[1]
                y = np.pad(y, ((0, 0), (0, pad_size)), mode='constant')
        
        print(f"✅ Miyawaki loaded: X{X.shape}, y{y.shape}")
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def load_vangerven(self):
        """Load Vangerven dataset (using digit69 as proxy)"""
        print("📊 Loading Vangerven dataset...")
        
        file_path = self.data_dir / "digit69_28x28.mat"
        if not file_path.exists():
            print(f"❌ Vangerven dataset not found: {file_path}")
            return None, None
            
        data = loadmat(file_path)
        
        # Extract data
        if 'fmri_data' in data and 'image_data' in data:
            X = data['fmri_data']
            y = data['image_data']
        elif 'X' in data and 'y' in data:
            X = data['X']
            y = data['y']
        else:
            # Find the right arrays
            keys = [k for k in data.keys() if not k.startswith('__')]
            print(f"Available keys: {keys}")
            arrays = [(k, v) for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim >= 2]
            arrays.sort(key=lambda x: x[1].size, reverse=True)
            if len(arrays) >= 2:
                X = arrays[0][1]
                y = arrays[1][1]
            else:
                print("❌ Could not identify data arrays")
                return None, None
        
        # Process shapes
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
            
        # Ensure y is 784 dimensions
        if y.shape[1] != 784:
            if y.shape[1] > 784:
                y = y[:, :784]
            else:
                pad_size = 784 - y.shape[1]
                y = np.pad(y, ((0, 0), (0, pad_size)), mode='constant')
        
        print(f"✅ Vangerven loaded: X{X.shape}, y{y.shape}")
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def load_mindbigdata(self):
        """Load MindBigData dataset"""
        print("📊 Loading MindBigData dataset...")
        
        file_path = self.data_dir / "mindbigdata.mat"
        if not file_path.exists():
            print(f"❌ MindBigData dataset not found: {file_path}")
            return None, None
            
        data = loadmat(file_path)
        
        # Extract data
        if 'eeg_fmri_data' in data and 'image_data' in data:
            X = data['eeg_fmri_data']
            y = data['image_data']
        elif 'X' in data and 'y' in data:
            X = data['X']
            y = data['y']
        else:
            keys = [k for k in data.keys() if not k.startswith('__')]
            print(f"Available keys: {keys}")
            arrays = [(k, v) for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim >= 2]
            arrays.sort(key=lambda x: x[1].size, reverse=True)
            if len(arrays) >= 2:
                X = arrays[0][1]
                y = arrays[1][1]
            else:
                print("❌ Could not identify data arrays")
                return None, None
        
        # Process shapes
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
            
        # Ensure y is 784 dimensions
        if y.shape[1] != 784:
            if y.shape[1] > 784:
                y = y[:, :784]
            else:
                pad_size = 784 - y.shape[1]
                y = np.pad(y, ((0, 0), (0, pad_size)), mode='constant')
        
        print(f"✅ MindBigData loaded: X{X.shape}, y{y.shape}")
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def load_crell(self):
        """Load Crell dataset"""
        print("📊 Loading Crell dataset...")
        
        file_path = self.data_dir / "crell.mat"
        if not file_path.exists():
            print(f"❌ Crell dataset not found: {file_path}")
            return None, None
            
        data = loadmat(file_path)
        
        # Extract data
        if 'eeg_fmri_data' in data and 'image_data' in data:
            X = data['eeg_fmri_data']
            y = data['image_data']
        elif 'X' in data and 'y' in data:
            X = data['X']
            y = data['y']
        else:
            keys = [k for k in data.keys() if not k.startswith('__')]
            print(f"Available keys: {keys}")
            arrays = [(k, v) for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim >= 2]
            arrays.sort(key=lambda x: x[1].size, reverse=True)
            if len(arrays) >= 2:
                X = arrays[0][1]
                y = arrays[1][1]
            else:
                print("❌ Could not identify data arrays")
                return None, None
        
        # Process shapes
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
            
        # Ensure y is 784 dimensions
        if y.shape[1] != 784:
            if y.shape[1] > 784:
                y = y[:, :784]
            else:
                pad_size = 784 - y.shape[1]
                y = np.pad(y, ((0, 0), (0, pad_size)), mode='constant')
        
        print(f"✅ Crell loaded: X{X.shape}, y{y.shape}")
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def load_all_datasets(self):
        """Load all available datasets"""
        datasets = {}
        
        # Try to load each dataset
        loaders = {
            'miyawaki': self.load_miyawaki,
            'vangerven': self.load_vangerven,
            'mindbigdata': self.load_mindbigdata,
            'crell': self.load_crell
        }
        
        for name, loader in loaders.items():
            try:
                X, y = loader()
                if X is not None and y is not None:
                    datasets[name] = {'X': X, 'y': y}
                    print(f"✅ {name}: {X.shape[0]} samples")
                else:
                    print(f"❌ Failed to load {name}")
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
        
        return datasets

def main():
    """Main function to load and inspect real datasets"""
    print("🚀 Loading Real CortexFlow Datasets...")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("results/real_experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    loader = RealDatasetLoader()
    datasets = loader.load_all_datasets()
    
    if not datasets:
        print("❌ No datasets loaded successfully!")
        return
    
    print(f"\n✅ Successfully loaded {len(datasets)} datasets:")
    for name, data in datasets.items():
        X, y = data['X'], data['y']
        print(f"   - {name}: X{X.shape}, y{y.shape}")
    
    # Save dataset info
    dataset_info = {}
    for name, data in datasets.items():
        X, y = data['X'], data['y']
        dataset_info[name] = {
            'n_samples': int(X.shape[0]),
            'input_dim': int(X.shape[1]),
            'output_dim': int(y.shape[1]),
            'input_range': [float(X.min()), float(X.max())],
            'output_range': [float(y.min()), float(y.max())],
            'input_mean': float(X.mean()),
            'output_mean': float(y.mean())
        }
    
    with open(results_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save datasets for training
    torch.save(datasets, results_dir / "real_datasets.pt")
    
    print(f"\n💾 Datasets saved to: {results_dir}")
    print("🎯 Ready for training experiments!")
    
    return datasets

if __name__ == "__main__":
    main()
