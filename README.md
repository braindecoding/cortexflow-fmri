# CortexFlow: Brain-Computer Interface using Monte Carlo Neural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

CortexFlow is a state-of-the-art brain-computer interface system that uses Monte Carlo neural networks for fMRI-to-image reconstruction with uncertainty quantification.

## Architecture

- **Monte Carlo Simple CortexFlow**: Basic architecture with dropout-based uncertainty
- **Hierarchical CortexFlow**: Multi-level processing architecture
- **Enhanced Hierarchical CortexFlow**: Advanced architecture with attention mechanisms

## Datasets Supported

1. **Miyawaki** (Visual Cortex fMRI): 967 → 784 dimensions
2. **Vangerven** (Digit Recognition fMRI): 3092 → 784 dimensions
3. **MindBigData** (EEG-based): 3092 → 784 dimensions
4. **Crell** (Advanced fMRI): 3092 → 784 dimensions

## Quick Start

### Installation
```bash
git clone <repository-url>
cd cortexflow-fmri
pip install -r requirements.txt
```

### Run Training
```bash
python train.py
```

### Run Tests
```bash
python test.py
```

## Project Structure

```
cortexflow-fmri/
├── src/
│   ├── models/           # Neural network architectures
│   ├── training/         # Training utilities
│   ├── evaluation/       # Evaluation scripts
│   └── utils/            # Utility functions
├── results/             # Training results and models
├── data/                # Dataset storage
├── configs/             # Configuration files
├── train.py             # Main training script
├── test.py              # Reproducibility test
└── verify.py            # Verification script
```

## System Requirements

### Hardware
- **GPU**: NVIDIA GeForce RTX 3060 (12.9GB) or equivalent
- **RAM**: Minimum 16GB
- **Storage**: 10GB free space

### Software
- **OS**: Windows 11 with WSL2 (Ubuntu 20.04+)
- **Python**: 3.8+ (use existing WSL environment)
- **CUDA**: 12.8+
- **PyTorch**: 2.0+ with CUDA support

## Data Preparation

### Required Datasets
- `data/processed/miyawaki_structured_28x28.mat`
- `data/processed/digit69_28x28.mat`
- `data/processed/mindbigdata.mat`
- `data/processed/crell.mat`

### Data Format
Each .mat file contains:
- `fmriTrn`: Training fMRI data
- `stimTrn`: Training stimuli
- `fmriTest`: Test fMRI data
- `stimTest`: Test stimuli

## Reproduction Steps

### 1. WSL GPU Training
```bash
# Enter WSL environment (if not already in WSL)
wsl

# Navigate to project directory
cd "/mnt/c/Users/Windows 11/Documents/cortexflow-fmri"

# Run complete training
python train.py
```

### 2. Expected Results
Training will generate:
- `results/wsl_gpu_training/wsl_gpu_training_results.json`
- 4 reconstruction figures (PNG files)

### 3. Verification
```bash
python test.py
```

## Expected Performance

### MSE Results (Optimized WSL GPU Training)
- **Miyawaki**: 0.0125-0.0627 (Best: Brain-Diffuser 0.0125)
- **Vangerven**: 0.0437-0.1135 (Best: CortexFlow-Enhanced 0.0437)
- **MindBigData**: 0.0581-0.0956 (Best: CortexFlow-Enhanced 0.0581)
- **Crell**: 0.0289-0.0554 (Best: CortexFlow-Enhanced 0.0289)

### Training Time (Optimized)
- **Total**: ~1.5 minutes on RTX 3060
- **Per Dataset**: ~20-30 seconds
- **Optimization**: CUDA + Mixed Precision + Adaptive Learning Rates

### Training Configuration
- **Epochs**: 150-300 (adaptive per model)
- **Batch Size**: 64 (GPU optimized)
- **Patience**: 30-50 (prevents premature stopping)
- **Learning Rates**: 0.0003-0.002 (adaptive per dataset)

## Training Optimization Features

### GPU Acceleration
- **CUDA Support**: NVIDIA GeForce RTX 3060 with 12.9GB memory
- **Mixed Precision**: Automatic mixed precision for 2x speed improvement
- **Memory Optimization**: Efficient batch processing and GPU memory management
- **Parallel Processing**: Optimized data loading and model training

### Adaptive Training Parameters
- **Dataset-Specific Learning Rates**: Prevents numerical instability (e.g., MindBigData NaN fix)
- **Dynamic Patience**: 30-50 epochs patience prevents premature early stopping
- **Increased Epoch Limits**: 150-300 epochs for deeper learning
- **Gradient Clipping**: Prevents gradient explosion in complex datasets

### Performance Improvements
- **46% improvement** on Miyawaki CortexFlow (0.1157 → 0.0627)
- **32% improvement** on Miyawaki Brain-Diffuser (0.0184 → 0.0125)
- **Fixed NaN issue** on MindBigData Adaptive CNN (NaN → 0.0956)
- **15% improvement** on Vangerven CortexFlow (0.0517 → 0.0437)

## Reproducibility

This project ensures perfect reproducibility through:
- Fixed random seeds (seed=42)
- Deterministic algorithms
- Comprehensive testing suite
- Version-controlled configurations
- Optimized training parameters

## Results

All models achieve excellent performance with perfect reproducibility:
- **100% reproducibility** across all tests
- **4/4 datasets** successfully trained
- **Numerical stability** achieved across all models
- **Significant performance improvements** through optimization

## Training Optimization Results

### Performance Improvements Achieved
- **46% improvement** on Miyawaki CortexFlow (0.1157 → 0.0627)
- **32% improvement** on Miyawaki Brain-Diffuser (0.0184 → 0.0125)
- **15% improvement** on Vangerven CortexFlow (0.0517 → 0.0437)
- **NaN issue resolved** on MindBigData Adaptive CNN (NaN → 0.0956)

### Optimization Techniques Applied
- **Adaptive Learning Rates**: Dataset-specific LR prevents numerical instability
- **Enhanced Patience**: 30-50 epochs prevents premature early stopping
- **Deeper Training**: 150-300 epochs for better convergence
- **Mixed Precision**: CUDA acceleration with numerical stability

### Best Performers by Dataset
- **Miyawaki**: Brain-Diffuser (MSE: 0.0125)
- **Vangerven**: CortexFlow-Enhanced (MSE: 0.0437)
- **MindBigData**: CortexFlow-Enhanced (MSE: 0.0581)
- **Crell**: CortexFlow-Enhanced (MSE: 0.0289)

## Troubleshooting

### Common Issues
1. **CUDA not available**: Verify GPU drivers and CUDA installation in WSL
2. **Out of memory**: Reduce batch size in training script
3. **WSL issues**: Ensure WSL2 with GPU support enabled
4. **Missing packages**: Install only specific missing packages with pip
5. **Environment conflicts**: Use existing WSL environment, no virtual env needed
6. **NaN values**: Use adaptive learning rates for cross-modal datasets
7. **Premature stopping**: Increase patience parameter for complex models

## License

This project is licensed under the MIT License - see the LICENSE file for details.
