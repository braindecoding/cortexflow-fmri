#!/usr/bin/env python3
"""
Verify Stimulus Consistency
===========================

Check if the stimulus used in visualization matches the targets used in training.
This is critical for scientific integrity.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
from PIL import Image

def load_training_targets(dataset_name):
    """Load targets used in training"""
    
    print(f"Loading training targets for {dataset_name}...")
    
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
            return None
        
        if not mat_file.exists():
            print(f"Dataset file not found: {mat_file}")
            return None
        
        # Load .mat file
        data = sio.loadmat(str(mat_file))
        keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Available keys in {dataset_name}: {keys}")
        
        # Find target data (usually largest array)
        arrays = []
        for k in keys:
            if isinstance(data[k], np.ndarray) and data[k].dtype.kind in 'fc':
                arrays.append((k, data[k]))
        
        if len(arrays) < 2:
            print(f"Insufficient data arrays in {dataset_name}")
            return None
        
        # Sort by size and take targets (usually largest)
        arrays.sort(key=lambda x: x[1].size)
        y_data = arrays[-1][1]  # Targets (largest array)
        
        print(f"Training targets shape: {y_data.shape}")
        print(f"Training targets key: {arrays[-1][0]}")
        
        # Convert to tensor and normalize
        y = torch.tensor(y_data, dtype=torch.float32)
        
        # Fix shapes
        if y.dim() == 2 and y.shape[1] == 784:
            y = y.view(-1, 1, 28, 28)
        elif y.dim() == 3:
            y = y.unsqueeze(1)
        
        # Normalize
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        return y
        
    except Exception as e:
        print(f"Error loading training targets for {dataset_name}: {e}")
        return None

def load_visualization_stimuli(dataset_name):
    """Load stimuli used in visualization"""
    
    print(f"Loading visualization stimuli for {dataset_name}...")
    
    if dataset_name == 'miyawaki':
        try:
            data_path = Path("data/processed/miyawaki_structured_28x28.mat")
            data = sio.loadmat(str(data_path))
            
            if 'stimTest' in data:
                stimuli = data['stimTest']
                print(f"Found stimTest: {stimuli.shape}")
            elif 'stimTrn' in data:
                stimuli = data['stimTrn']
                print(f"Found stimTrn: {stimuli.shape}")
            else:
                print("No stimulus data found")
                return None
            
            # Reshape and normalize
            if stimuli.shape[1] == 784:
                stimuli = stimuli.reshape(-1, 28, 28)
            
            stimuli = (stimuli - stimuli.min()) / (stimuli.max() - stimuli.min() + 1e-8)
            return stimuli
            
        except Exception as e:
            print(f"Error loading Miyawaki stimuli: {e}")
            return None
    
    elif dataset_name == 'vangerven':
        try:
            data_path = Path("data/processed/digit69_28x28.mat")
            data = sio.loadmat(str(data_path))
            
            if 'stimTest' in data:
                stimuli = data['stimTest']
                print(f"Found stimTest: {stimuli.shape}")
            elif 'stimTrn' in data:
                stimuli = data['stimTrn']
                print(f"Found stimTrn: {stimuli.shape}")
            else:
                print("No stimulus data found")
                return None
            
            # Reshape and normalize
            if stimuli.shape[1] == 784:
                stimuli = stimuli.reshape(-1, 28, 28)
            
            if stimuli.max() > 1.0:
                stimuli = stimuli / 255.0
            
            return stimuli
            
        except Exception as e:
            print(f"Error loading Vangerven stimuli: {e}")
            return None
    
    elif dataset_name == 'mindbigdata':
        try:
            stimuli_dir = Path("data/external/MindbigdataStimuli")
            if not stimuli_dir.exists():
                print(f"MindBigData stimuli directory not found")
                return None
            
            stimuli = []
            for i in range(10):
                img_path = stimuli_dir / f"{i}.jpg"
                if img_path.exists():
                    img = Image.open(img_path).convert('L')
                    img = img.resize((28, 28))
                    img_array = np.array(img) / 255.0
                    stimuli.append(img_array)
            
            if stimuli:
                return np.array(stimuli)
            else:
                return None
                
        except Exception as e:
            print(f"Error loading MindBigData stimuli: {e}")
            return None
    
    elif dataset_name == 'crell':
        try:
            stimuli_dir = Path("data/external/crellStimuli")
            if not stimuli_dir.exists():
                print(f"Crell stimuli directory not found")
                return None
            
            letters = ['a', 'd', 'e', 'f', 'j', 'n', 'o', 's', 't', 'v']
            stimuli = []
            
            for letter in letters:
                img_path = stimuli_dir / f"{letter}.png"
                if img_path.exists():
                    img = Image.open(img_path).convert('L')
                    img = img.resize((28, 28))
                    img_array = np.array(img) / 255.0
                    stimuli.append(img_array)
            
            if stimuli:
                return np.array(stimuli)
            else:
                return None
                
        except Exception as e:
            print(f"Error loading Crell stimuli: {e}")
            return None
    
    return None

def compare_targets_vs_stimuli(dataset_name):
    """Compare training targets vs visualization stimuli"""
    
    print(f"\n{'='*60}")
    print(f"COMPARING TARGETS vs STIMULI for {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load both
    training_targets = load_training_targets(dataset_name)
    viz_stimuli = load_visualization_stimuli(dataset_name)
    
    if training_targets is None:
        print(f"❌ Could not load training targets for {dataset_name}")
        return False
    
    if viz_stimuli is None:
        print(f"❌ Could not load visualization stimuli for {dataset_name}")
        return False
    
    print(f"\nTraining targets shape: {training_targets.shape}")
    print(f"Visualization stimuli shape: {viz_stimuli.shape}")
    
    # Compare shapes
    if training_targets.shape[0] != viz_stimuli.shape[0]:
        print(f"⚠️  Different number of samples: {training_targets.shape[0]} vs {viz_stimuli.shape[0]}")
    
    # Compare first few samples
    num_compare = min(5, training_targets.shape[0], viz_stimuli.shape[0])
    
    print(f"\nComparing first {num_compare} samples...")
    
    # Create comparison figure
    fig, axes = plt.subplots(2, num_compare, figsize=(15, 6))
    fig.suptitle(f'{dataset_name.title()}: Training Targets vs Visualization Stimuli', fontsize=14)
    
    mse_values = []
    
    for i in range(num_compare):
        # Training target
        if training_targets.dim() == 4:
            target_img = training_targets[i, 0].numpy()
        else:
            target_img = training_targets[i].numpy()
        
        # Visualization stimulus
        stim_img = viz_stimuli[i]
        
        # Plot
        axes[0, i].imshow(target_img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Training Target {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(stim_img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Viz Stimulus {i+1}')
        axes[1, i].axis('off')
        
        # Calculate MSE
        mse = np.mean((target_img - stim_img) ** 2)
        mse_values.append(mse)
        print(f"Sample {i+1} MSE: {mse:.6f}")
    
    plt.tight_layout()
    
    # Save comparison
    output_dir = Path("results/stimulus_verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"targets_vs_stimuli_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Overall assessment
    avg_mse = np.mean(mse_values)
    print(f"\nAverage MSE between targets and stimuli: {avg_mse:.6f}")
    
    if avg_mse < 0.01:
        print("✅ CONSISTENT: Training targets match visualization stimuli")
        return True
    elif avg_mse < 0.1:
        print("⚠️  MODERATE DIFFERENCE: Some mismatch between targets and stimuli")
        return False
    else:
        print("❌ MAJOR MISMATCH: Training targets do NOT match visualization stimuli")
        return False

def main():
    """Main verification"""
    
    print("STIMULUS CONSISTENCY VERIFICATION")
    print("=" * 80)
    print("Checking if training targets match visualization stimuli...")
    
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    results = {}
    
    for dataset in datasets:
        consistent = compare_targets_vs_stimuli(dataset)
        results[dataset] = consistent
    
    print(f"\n{'='*80}")
    print("FINAL ASSESSMENT:")
    print(f"{'='*80}")
    
    all_consistent = True
    for dataset, consistent in results.items():
        status = "✅ CONSISTENT" if consistent else "❌ INCONSISTENT"
        print(f"{dataset.title()}: {status}")
        if not consistent:
            all_consistent = False
    
    print(f"\n{'='*80}")
    if all_consistent:
        print("✅ ALL DATASETS CONSISTENT")
        print("✅ No re-training needed")
        print("✅ Current results are valid")
    else:
        print("❌ SOME DATASETS INCONSISTENT")
        print("❌ Re-training may be needed for scientific integrity")
        print("❌ Current visualization may be misleading")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
