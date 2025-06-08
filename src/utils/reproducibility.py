#!/usr/bin/env python3
"""
🔬 CORTEXFLOW REPRODUCIBILITY SETUP
================================================================================
Ensures perfect reproducibility across all experiments
================================================================================
"""

import os
import random
import numpy as np
import torch

def set_all_seeds(seed=42):
    """Set all possible random seeds for maximum reproducibility."""
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch deterministic operations
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"🔬 Reproducibility setup complete with seed: {seed}")

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎮 Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"🖥️  Using CPU")
    
    return device

if __name__ == "__main__":
    set_all_seeds()
    device = get_device()
