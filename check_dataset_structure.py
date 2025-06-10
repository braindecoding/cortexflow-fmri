#!/usr/bin/env python3
"""
Check Dataset Structure
=======================

Memeriksa struktur data untuk semua 4 dataset
"""

import scipy.io as sio
from pathlib import Path

def check_dataset_structure():
    """Check structure of all datasets"""
    
    print("MEMERIKSA STRUKTUR DATASET")
    print("=" * 50)
    
    data_path = Path("data/processed")
    datasets = ['miyawaki_structured_28x28.mat', 'digit69_28x28.mat', 'mindbigdata.mat', 'crell.mat']
    
    for dataset_file in datasets:
        print(f"\n{dataset_file}:")
        print("-" * 30)
        
        try:
            mat_file = data_path / dataset_file
            data = sio.loadmat(str(mat_file))
            
            print("Keys dalam file:")
            for key in data.keys():
                if not key.startswith('__'):
                    print(f"  {key}: {data[key].shape if hasattr(data[key], 'shape') else type(data[key])}")
                    
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_dataset_structure()
