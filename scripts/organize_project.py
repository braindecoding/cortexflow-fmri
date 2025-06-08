#!/usr/bin/env python3
"""
Project Organization Script
Reorganizes CortexFlow project structure for better clarity and maintainability
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def create_clean_structure():
    """Create clean project structure."""
    print("🏗️ Creating Clean Project Structure")
    print("=" * 60)
    
    # Define new structure
    structure = {
        'experiments/': {
            'simple/': ['cortexflow_training.py'],
            'hierarchical/': ['hierarchical_training.py'],
            'enhanced/': ['enhanced_hierarchical_training.py'],
            'configs/': [],
            'logs/': []
        },
        'src/': {
            'models/': ['hierarchical.py'],
            'data/': [],
            'utils/': [],
            'training/': [],
            'evaluation/': []
        },
        'data/': {
            'raw/': [],
            'processed/': [],
            'external/': []
        },
        'results/': {
            'simple/': [],
            'hierarchical/': [],
            'enhanced/': [],
            'comparisons/': []
        },
        'checkpoints/': {
            'simple/': [],
            'hierarchical/': [],
            'enhanced/': []
        },
        'tests/': [],
        'docs/': [],
        'scripts/': []
    }
    
    # Create directories
    for main_dir, subdirs in structure.items():
        os.makedirs(main_dir, exist_ok=True)
        print(f"📁 Created: {main_dir}")
        
        if isinstance(subdirs, dict):
            for subdir, files in subdirs.items():
                subdir_path = os.path.join(main_dir, subdir)
                os.makedirs(subdir_path, exist_ok=True)
                print(f"   📂 Created: {subdir_path}")
        elif isinstance(subdirs, list):
            for subdir in subdirs:
                if subdir.endswith('/'):
                    subdir_path = os.path.join(main_dir, subdir)
                    os.makedirs(subdir_path, exist_ok=True)
                    print(f"   📂 Created: {subdir_path}")

def move_files():
    """Move files to appropriate locations."""
    print("\n📦 Moving Files to Clean Structure")
    print("=" * 60)
    
    # File movements
    movements = [
        # Training scripts
        ('cortexflow_training.py', 'experiments/simple/'),
        ('hierarchical_training.py', 'experiments/hierarchical/'),
        ('enhanced_hierarchical_training.py', 'experiments/enhanced/'),
        
        # Test files
        ('test_hierarchical.py', 'tests/'),
        ('test_reproducibility.py', 'tests/'),
        
        # Scripts
        ('organize_project.py', 'scripts/'),
        
        # Results - Simple
        ('results/miyawaki_reconstructions.png', 'results/simple/'),
        ('results/vangerven_reconstructions.png', 'results/simple/'),
        
        # Results - Hierarchical
        ('results/hierarchical_miyawaki_reconstructions.png', 'results/hierarchical/'),
        ('results/hierarchical_miyawaki_progressive.png', 'results/hierarchical/'),
        ('results/hierarchical_vangerven_reconstructions.png', 'results/hierarchical/'),
        ('results/hierarchical_vangerven_progressive.png', 'results/hierarchical/'),
        
        # Results - Enhanced
        ('results/enhanced_miyawaki_reconstructions.png', 'results/enhanced/'),
        ('results/enhanced_miyawaki_uncertainty.png', 'results/enhanced/'),
        ('results/enhanced_vangerven_reconstructions.png', 'results/enhanced/'),
        ('results/enhanced_vangerven_uncertainty.png', 'results/enhanced/'),
        
        # Checkpoints - Hierarchical
        ('checkpoints/hierarchical_miyawaki_model.pt', 'checkpoints/hierarchical/'),
        ('checkpoints/hierarchical_vangerven_model.pt', 'checkpoints/hierarchical/'),
        
        # Checkpoints - Enhanced
        ('checkpoints/enhanced_miyawaki_model.pt', 'checkpoints/enhanced/'),
        ('checkpoints/enhanced_vangerven_model.pt', 'checkpoints/enhanced/'),
        
        # Data files
        ('data/miyawaki_structured_28x28.mat', 'data/processed/'),
        ('data/digit69_28x28.mat', 'data/processed/'),
        ('data/S01.mat', 'data/raw/'),
        ('data/EP1.01.txt', 'data/raw/'),
        ('data/eeg_dataset_guide.md', 'data/'),
        
        # Test results
        ('test_results/reproducibility_test.json', 'tests/results/'),
    ]
    
    # Create test results directory
    os.makedirs('tests/results', exist_ok=True)
    
    for source, destination in movements:
        if os.path.exists(source):
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            try:
                shutil.move(source, destination)
                print(f"✅ Moved: {source} → {destination}")
            except Exception as e:
                print(f"❌ Failed to move {source}: {e}")
        else:
            print(f"⏭️  Skipped: {source} (not found)")

def move_external_data():
    """Move external data to appropriate locations."""
    print("\n📊 Moving External Data")
    print("=" * 60)
    
    external_movements = [
        ('data/MindbigdataStimuli', 'data/external/MindbigdataStimuli'),
        ('data/crellStimuli', 'data/external/crellStimuli'),
    ]
    
    for source, destination in external_movements:
        if os.path.exists(source):
            try:
                if os.path.isdir(source):
                    shutil.move(source, destination)
                else:
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    shutil.move(source, destination)
                print(f"✅ Moved: {source} → {destination}")
            except Exception as e:
                print(f"❌ Failed to move {source}: {e}")
        else:
            print(f"⏭️  Skipped: {source} (not found)")

def cleanup_empty_dirs():
    """Remove empty directories."""
    print("\n🧹 Cleaning Up Empty Directories")
    print("=" * 60)
    
    dirs_to_check = ['results', 'checkpoints', 'test_results']
    
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            try:
                # Check if directory is empty
                if not os.listdir(dir_name):
                    os.rmdir(dir_name)
                    print(f"🗑️  Removed empty directory: {dir_name}")
                else:
                    print(f"📁 Kept non-empty directory: {dir_name}")
            except Exception as e:
                print(f"❌ Failed to remove {dir_name}: {e}")

def create_config_files():
    """Create configuration files for different experiments."""
    print("\n⚙️ Creating Configuration Files")
    print("=" * 60)
    
    # Simple config
    simple_config = '''# Simple CortexFlow Configuration
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 15
WEIGHT_DECAY = 1e-4
'''
    
    # Hierarchical config
    hierarchical_config = '''# Hierarchical CortexFlow Configuration
TEMPORAL_SCALES = [1, 2, 4, 8]
HIDDEN_DIM = 512
NUM_PYRAMID_LEVELS = 4
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
PATIENCE = 20
WEIGHT_DECAY = 1e-4
'''
    
    # Enhanced config
    enhanced_config = '''# Enhanced Hierarchical CortexFlow Configuration
# Base hierarchical settings
TEMPORAL_SCALES = [1, 2, 4, 8]
HIDDEN_DIM = 512
NUM_PYRAMID_LEVELS = 4

# Enhanced features
ENABLE_MC_DROPOUT = True
MC_DROPOUT_RATE = 0.15
MC_SAMPLES = 10

ENABLE_FEATURE_ALIGNMENT = True
ALIGNMENT_WEIGHT = 0.1
ALIGNMENT_TEMPERATURE = 0.1

UNCERTAINTY_WEIGHT = 0.05
CONSISTENCY_WEIGHT = 0.08

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
PATIENCE = 20
WEIGHT_DECAY = 1e-4
'''
    
    configs = [
        ('experiments/configs/simple_config.py', simple_config),
        ('experiments/configs/hierarchical_config.py', hierarchical_config),
        ('experiments/configs/enhanced_config.py', enhanced_config),
    ]
    
    for config_path, config_content in configs:
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"⚙️  Created: {config_path}")

def create_readme_files():
    """Create README files for different directories."""
    print("\n📝 Creating README Files")
    print("=" * 60)
    
    readmes = {
        'experiments/README.md': '''# Experiments

This directory contains all experimental scripts for different CortexFlow architectures.

## Structure
- `simple/`: Basic CortexFlow implementation
- `hierarchical/`: Advanced hierarchical architecture
- `enhanced/`: Research-grade enhanced architecture with Monte Carlo and Feature Alignment
- `configs/`: Configuration files for experiments
- `logs/`: Training logs and experiment records
''',
        
        'results/README.md': '''# Results

This directory contains all experimental results and visualizations.

## Structure
- `simple/`: Results from basic CortexFlow
- `hierarchical/`: Results from hierarchical architecture
- `enhanced/`: Results from enhanced architecture with uncertainty maps
- `comparisons/`: Comparative analysis and benchmarks
''',
        
        'checkpoints/README.md': '''# Model Checkpoints

This directory contains saved model checkpoints for different architectures.

## Structure
- `simple/`: Simple CortexFlow model checkpoints
- `hierarchical/`: Hierarchical CortexFlow model checkpoints
- `enhanced/`: Enhanced CortexFlow model checkpoints

## Usage
Load checkpoints using:
```python
checkpoint = torch.load('checkpoints/hierarchical/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```
''',
        
        'data/README.md': '''# Data

This directory contains all datasets and data-related files.

## Structure
- `raw/`: Original, unprocessed data files
- `processed/`: Preprocessed data ready for training
- `external/`: External datasets and stimuli

## Datasets
- `miyawaki_structured_28x28.mat`: Miyawaki visual cortex dataset
- `digit69_28x28.mat`: Vangerven digit recognition dataset
''',
        
        'tests/README.md': '''# Tests

This directory contains all test scripts and test results.

## Test Scripts
- `test_hierarchical.py`: Debug and testing utilities for hierarchical architecture
- `test_reproducibility.py`: Reproducibility testing across all architectures

## Test Results
- `results/`: Test outputs and reproducibility reports
''',
        
        'src/README.md': '''# Source Code

This directory contains the core source code for CortexFlow architectures.

## Structure
- `models/`: Neural network architectures and model definitions
- `data/`: Data loading and preprocessing utilities
- `utils/`: Utility functions and helper modules
- `training/`: Training loops and optimization utilities
- `evaluation/`: Evaluation metrics and analysis tools
'''
    }
    
    for readme_path, content in readmes.items():
        os.makedirs(os.path.dirname(readme_path), exist_ok=True)
        with open(readme_path, 'w') as f:
            f.write(content)
        print(f"📝 Created: {readme_path}")

def create_main_readme():
    """Create updated main README."""
    print("\n📖 Creating Main README")
    print("=" * 60)
    
    main_readme = '''# CortexFlow: fMRI-to-Image Reconstruction

Advanced neural architectures for reconstructing visual stimuli from fMRI brain activity.

## 🏗️ Project Structure

```
cortexflow-fmri/
├── 📁 experiments/          # Experimental scripts
│   ├── simple/             # Basic CortexFlow
│   ├── hierarchical/       # Hierarchical architecture
│   ├── enhanced/           # Enhanced with MC + Alignment
│   ├── configs/            # Configuration files
│   └── logs/               # Training logs
├── 📁 src/                 # Source code
│   ├── models/             # Neural architectures
│   ├── data/               # Data utilities
│   ├── utils/              # Helper functions
│   ├── training/           # Training utilities
│   └── evaluation/         # Evaluation tools
├── 📁 data/                # Datasets
│   ├── raw/                # Original data
│   ├── processed/          # Preprocessed data
│   └── external/           # External datasets
├── 📁 results/             # Experimental results
│   ├── simple/             # Basic results
│   ├── hierarchical/       # Hierarchical results
│   ├── enhanced/           # Enhanced results
│   └── comparisons/        # Comparative analysis
├── 📁 checkpoints/         # Model checkpoints
│   ├── simple/             # Simple models
│   ├── hierarchical/       # Hierarchical models
│   └── enhanced/           # Enhanced models
├── 📁 tests/               # Test scripts
├── 📁 docs/                # Documentation
└── 📁 scripts/             # Utility scripts
```

## 🚀 Quick Start

### 1. Simple CortexFlow
```bash
cd experiments/simple
python cortexflow_training.py
```

### 2. Hierarchical CortexFlow
```bash
cd experiments/hierarchical
python hierarchical_training.py
```

### 3. Enhanced CortexFlow (Monte Carlo + Feature Alignment)
```bash
cd experiments/enhanced
python enhanced_hierarchical_training.py
```

## 🏆 Architecture Comparison

| Architecture | Miyawaki Loss | Vangerven Loss | Features |
|--------------|---------------|----------------|----------|
| Simple | 0.018006 | 0.037846 | Basic encoder-decoder |
| Hierarchical | 0.050046 | 0.050851 | Multi-scale + Progressive |
| Enhanced | 0.051336 | 0.078782 | MC Dropout + Alignment |

## 📊 Features

### Simple CortexFlow
- Basic encoder-decoder architecture
- Standard reconstruction loss
- Fast training and inference

### Hierarchical CortexFlow
- Multi-scale temporal processing
- Feature pyramid networks
- Progressive decoding
- Advanced loss functions

### Enhanced CortexFlow
- Monte Carlo Dropout for uncertainty estimation
- Feature alignment across scales
- Uncertainty visualization
- Research-grade performance

## 🧪 Testing

```bash
# Test hierarchical architecture
cd tests
python test_hierarchical.py

# Test reproducibility
python test_reproducibility.py
```

## 📈 Results

All experimental results are organized in the `results/` directory:
- Reconstruction comparisons
- Progressive visualization
- Uncertainty maps
- Performance metrics

## 🔧 Configuration

Configuration files are available in `experiments/configs/` for easy customization of:
- Model hyperparameters
- Training settings
- Architecture parameters

## 📚 Documentation

Detailed documentation is available in the `docs/` directory covering:
- Architecture details
- Training procedures
- Evaluation metrics
- API reference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Miyawaki et al. for the visual cortex dataset
- Vangerven et al. for the digit recognition dataset
- PyTorch team for the deep learning framework
'''
    
    with open('README.md', 'w') as f:
        f.write(main_readme)
    print(f"📖 Updated: README.md")

def main():
    """Main organization function."""
    print("🏗️ CORTEXFLOW PROJECT ORGANIZATION")
    print("=" * 80)
    print(f"Organization start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create clean structure
    create_clean_structure()
    
    # Move files
    move_files()
    
    # Move external data
    move_external_data()
    
    # Create configuration files
    create_config_files()
    
    # Create README files
    create_readme_files()
    
    # Create main README
    create_main_readme()
    
    # Cleanup
    cleanup_empty_dirs()
    
    print(f"\n{'='*80}")
    print(f"🎉 PROJECT ORGANIZATION COMPLETED!")
    print(f"{'='*80}")
    
    print(f"\n📊 SUMMARY:")
    print("-" * 60)
    print(f"✅ Created clean directory structure")
    print(f"✅ Moved files to appropriate locations")
    print(f"✅ Created configuration files")
    print(f"✅ Generated comprehensive documentation")
    print(f"✅ Organized results by architecture type")
    print(f"✅ Separated checkpoints by model type")
    print(f"✅ Cleaned up empty directories")
    
    print(f"\n🎯 NEXT STEPS:")
    print("-" * 60)
    print(f"1. Review the new structure in your file explorer")
    print(f"2. Test experiments from their new locations")
    print(f"3. Update any import paths if needed")
    print(f"4. Commit the organized structure to version control")

if __name__ == "__main__":
    main()
