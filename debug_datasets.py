#!/usr/bin/env python3
"""
Debug dataset structure for MindBigData and Crell
"""

import scipy.io
import numpy as np
import os

def debug_dataset(filepath, name):
    """Debug dataset structure."""
    print(f"\n{'='*60}")
    print(f"🔍 DEBUGGING {name.upper()}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
    
    try:
        data = scipy.io.loadmat(filepath)
        print(f"✅ File loaded successfully: {filepath}")
        print(f"📊 Keys in file: {list(data.keys())}")
        
        for key, value in data.items():
            if not key.startswith('__'):
                print(f"\n🔑 Key: '{key}'")
                print(f"   Type: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"   Shape: {value.shape}")
                if hasattr(value, 'dtype'):
                    print(f"   Data type: {value.dtype}")
                if hasattr(value, 'shape') and len(value.shape) > 0 and value.size > 0:
                    print(f"   Min/Max: {value.min():.6f} / {value.max():.6f}")
                    print(f"   Sample values: {value.flat[:5]}")
        
        # Check for standard keys
        standard_keys = ['fmriTrn', 'fmriTest', 'stimTrn', 'stimTest']
        alternative_keys = ['fmri', 'stim']
        
        print(f"\n📋 KEY ANALYSIS:")
        for key in standard_keys:
            status = "✅ Found" if key in data else "❌ Missing"
            print(f"   {key}: {status}")
        
        print(f"\n📋 ALTERNATIVE KEYS:")
        for key in alternative_keys:
            status = "✅ Found" if key in data else "❌ Missing"
            print(f"   {key}: {status}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        return None

def main():
    """Main debug function."""
    print("🔍 DATASET STRUCTURE DEBUGGING")
    print("="*80)
    
    # Debug datasets
    datasets = [
        ("data/processed/miyawaki_structured_28x28.mat", "Miyawaki"),
        ("data/processed/digit69_28x28.mat", "Vangerven"),
        ("data/processed/mindbigdata.mat", "MindBigData"),
        ("data/processed/crell.mat", "Crell")
    ]
    
    results = {}
    
    for filepath, name in datasets:
        results[name] = debug_dataset(filepath, name)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    
    for name, data in results.items():
        if data is not None:
            keys = [k for k in data.keys() if not k.startswith('__')]
            print(f"✅ {name}: {len(keys)} keys - {keys}")
        else:
            print(f"❌ {name}: Failed to load")

if __name__ == "__main__":
    main()
