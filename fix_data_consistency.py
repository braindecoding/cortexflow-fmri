#!/usr/bin/env python3
"""
Fix Data Consistency
====================

Fix the critical data mismatch:
- Use fMRI as input (X)
- Use visual stimuli as targets (y)
- Ensure proper fMRI → Visual reconstruction task
"""

import torch
import numpy as np
import scipy.io as sio
from pathlib import Path

def analyze_dataset_structure(dataset_name):
    """Analyze the actual structure of datasets"""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {dataset_name.upper()} DATASET STRUCTURE")
    print(f"{'='*60}")
    
    try:
        data_path = Path("data/processed")
        
        if dataset_name == 'miyawaki':
            mat_file = data_path / "miyawaki_structured_28x28.mat"
        elif dataset_name == 'vangerven':
            mat_file = data_path / "digit69_28x28.mat"
        elif dataset_name == 'mindbigdata':
            mat_file = data_path / "mindbigdata.mat"
        elif dataset_name == 'crell':
            mat_file = data_path / "crell.mat"
        else:
            return
        
        if not mat_file.exists():
            print(f"❌ Dataset file not found: {mat_file}")
            return
        
        # Load .mat file
        data = sio.loadmat(str(mat_file))
        keys = [k for k in data.keys() if not k.startswith('__')]
        
        print(f"Available keys: {keys}")
        
        for key in keys:
            if isinstance(data[key], np.ndarray):
                shape = data[key].shape
                dtype = data[key].dtype
                print(f"  {key}: {shape} ({dtype})")
                
                # Analyze content
                if 'fmri' in key.lower():
                    print(f"    → fMRI data (neural signals)")
                elif 'stim' in key.lower():
                    print(f"    → Stimulus data (visual patterns)")
                elif 'label' in key.lower():
                    print(f"    → Label data")
                else:
                    print(f"    → Unknown data type")
        
        # Determine correct X and y
        print(f"\n🎯 CORRECT TASK SETUP for {dataset_name}:")
        
        if dataset_name == 'miyawaki':
            print("  X (input): fmriTrn/fmriTest (fMRI signals)")
            print("  y (target): stimTrn/stimTest (visual stimuli)")
            print("  Task: fMRI → Visual Reconstruction")
        elif dataset_name == 'vangerven':
            print("  X (input): fmriTrn/fmriTest (fMRI signals)")
            print("  y (target): stimTrn/stimTest (digit patterns)")
            print("  Task: fMRI → Digit Reconstruction")
        else:
            print("  Need to determine correct X and y mapping")
        
    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")

def create_corrected_dataset(dataset_name):
    """Create correctly structured dataset"""
    
    print(f"\n🔧 CREATING CORRECTED DATASET for {dataset_name}")
    print("-" * 50)
    
    try:
        data_path = Path("data/processed")
        
        if dataset_name == 'miyawaki':
            mat_file = data_path / "miyawaki_structured_28x28.mat"
            data = sio.loadmat(str(mat_file))
            
            # Correct mapping: fMRI → Visual
            X_train = data['fmriTrn']  # fMRI signals as input
            y_train = data['stimTrn']  # Visual stimuli as targets
            X_test = data['fmriTest']  # fMRI signals as input
            y_test = data['stimTest']  # Visual stimuli as targets
            
            print(f"✅ X_train (fMRI): {X_train.shape}")
            print(f"✅ y_train (visual): {y_train.shape}")
            print(f"✅ X_test (fMRI): {X_test.shape}")
            print(f"✅ y_test (visual): {y_test.shape}")
            
            # Convert to tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            
            # Normalize
            X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-8)
            X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min() + 1e-8)
            
            # Reshape y to image format
            if y_train.shape[1] == 784:
                y_train = y_train.view(-1, 1, 28, 28)
            if y_test.shape[1] == 784:
                y_test = y_test.view(-1, 1, 28, 28)
            
            # Normalize visual data
            y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min() + 1e-8)
            y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min() + 1e-8)
            
            print(f"✅ Final X_train: {X_train.shape}")
            print(f"✅ Final y_train: {y_train.shape}")
            print(f"✅ Final X_test: {X_test.shape}")
            print(f"✅ Final y_test: {y_test.shape}")
            
            return X_train, y_train, X_test, y_test
            
        elif dataset_name == 'vangerven':
            mat_file = data_path / "digit69_28x28.mat"
            data = sio.loadmat(str(mat_file))
            
            # Check available keys
            keys = [k for k in data.keys() if not k.startswith('__')]
            print(f"Available keys: {keys}")
            
            # Find fMRI and stimulus data
            fmri_keys = [k for k in keys if 'fmri' in k.lower()]
            stim_keys = [k for k in keys if 'stim' in k.lower()]
            
            print(f"fMRI keys: {fmri_keys}")
            print(f"Stimulus keys: {stim_keys}")
            
            if fmri_keys and stim_keys:
                X_data = data[fmri_keys[0]]  # fMRI as input
                y_data = data[stim_keys[0]]  # Stimuli as targets
                
                print(f"✅ X_data (fMRI): {X_data.shape}")
                print(f"✅ y_data (stimuli): {y_data.shape}")
                
                # Convert and process
                X = torch.tensor(X_data, dtype=torch.float32)
                y = torch.tensor(y_data, dtype=torch.float32)
                
                # Normalize
                X = (X - X.min()) / (X.max() - X.min() + 1e-8)
                
                # Reshape y to image format
                if y.shape[1] == 784:
                    y = y.view(-1, 1, 28, 28)
                
                # Normalize visual data
                if y.max() > 1.0:
                    y = y / 255.0
                
                print(f"✅ Final X: {X.shape}")
                print(f"✅ Final y: {y.shape}")
                
                return X, y, None, None
            else:
                print("❌ Could not find fMRI and stimulus data")
                return None, None, None, None
        
        else:
            print(f"❌ Dataset {dataset_name} not implemented yet")
            return None, None, None, None
            
    except Exception as e:
        print(f"❌ Error creating corrected dataset for {dataset_name}: {e}")
        return None, None, None, None

def main():
    """Main analysis and correction"""
    
    print("DATA CONSISTENCY ANALYSIS AND CORRECTION")
    print("=" * 80)
    
    datasets = ['miyawaki', 'vangerven']
    
    for dataset in datasets:
        # Analyze structure
        analyze_dataset_structure(dataset)
        
        # Create corrected dataset
        X_train, y_train, X_test, y_test = create_corrected_dataset(dataset)
        
        if X_train is not None:
            print(f"✅ {dataset} dataset corrected successfully")
        else:
            print(f"❌ Failed to correct {dataset} dataset")
    
    print(f"\n{'='*80}")
    print("CRITICAL FINDINGS:")
    print("=" * 80)
    print("❌ PREVIOUS TRAINING WAS INCORRECT!")
    print("❌ Models were trained to predict fMRI signals, not visual images!")
    print("❌ All performance metrics are invalid for visual reconstruction!")
    print("❌ Need to RE-TRAIN all models with correct X→y mapping!")
    print("")
    print("✅ CORRECT TASK SETUP:")
    print("✅ X (input): fMRI signals")
    print("✅ y (target): Visual stimuli/images")
    print("✅ Task: Neural signal → Visual reconstruction")
    print("")
    print("🔧 REQUIRED ACTIONS:")
    print("1. Re-train ALL models with corrected data mapping")
    print("2. Re-evaluate performance with correct targets")
    print("3. Update all figures with correct reconstruction results")
    print("4. Ensure scientific integrity in documentation")

if __name__ == "__main__":
    main()
