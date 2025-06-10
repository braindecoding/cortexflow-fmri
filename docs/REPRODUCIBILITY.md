# Reproducibility Guide

## System Requirements

### Hardware
- **GPU**: NVIDIA GeForce RTX 3060 (12.9GB) or equivalent
- **RAM**: Minimum 16GB
- **Storage**: 10GB free space

### Software
- **OS**: Windows 11 with WSL2 (Ubuntu 20.04+)
- **Python**: 3.8+ (use existing WSL environment)
- **CUDA**: 12.8+
- **PyTorch**: 2.0+ with CUDA support (use existing installation)
- **Dependencies**: Use existing WSL environment packages

## Installation

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd cortexflow-fmri
   ```

2. **Use Existing WSL Environment**:
   ```bash
   # No virtual environment setup needed
   # Use existing WSL Python environment with dependencies already installed
   # Verify Python and dependencies are available
   python --version
   pip list | grep torch
   ```

3. **Verify GPU Setup**:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

4. **Install Missing Dependencies (if needed)**:
   ```bash
   # Only install if specific packages are missing
   pip install torch torchvision numpy scipy matplotlib
   ```

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

# Use existing WSL environment (no virtual env activation needed)
# Run complete training directly
python wsl_gpu_complete_training.py
```

### 2. Expected Results
Training will generate:
- `results/wsl_gpu_training/wsl_gpu_training_results.json`
- 4 reconstruction figures (PNG files)

### 3. Verification
```bash
python final_dissertation_verification.py
```

## Expected Performance

### MSE Results (WSL GPU Training)
- **Miyawaki**: 0.0176-0.0809
- **Vangerven**: 0.0438-0.0533
- **MindBigData**: 0.0556-0.0675
- **Crell**: 0.0288-0.0519

### Training Time
- **Total**: ~45-60 minutes on RTX 3060
- **Per Dataset**: ~10-15 minutes

## Troubleshooting

### Common Issues
1. **CUDA not available**: Verify GPU drivers and CUDA installation in WSL
2. **Out of memory**: Reduce batch size in training script
3. **WSL issues**: Ensure WSL2 with GPU support enabled
4. **Missing packages**: Install only specific missing packages with pip
5. **Environment conflicts**: Use existing WSL environment, no virtual env needed

### Contact
For reproduction issues, please refer to SOTA.md for detailed methodology.
