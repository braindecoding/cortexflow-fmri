#!/usr/bin/env python3
import scipy.io
import os

datasets = [
    'data/processed/miyawaki_structured_28x28.mat',
    'data/processed/digit69_28x28.mat', 
    'data/processed/mindbigdata.mat',
    'data/processed/crell.mat'
]

for dataset_path in datasets:
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(dataset_path):
        print("❌ File not found!")
        continue
        
    try:
        data = scipy.io.loadmat(dataset_path)
        keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Keys: {keys}")
        
        for key in keys:
            value = data[key]
            print(f"  {key}: {type(value)} - {getattr(value, 'shape', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
