#!/usr/bin/env python3
"""
🔬 CortexFlow Reproducibility Utilities

Comprehensive reproducibility management for CortexFlow framework.
"""

import os
import random
import numpy as np
import torch
import logging
from typing import Optional, Dict, Any
import json
from pathlib import Path
from datetime import datetime


def set_global_seed(seed: int = 42) -> None:
    """
    Set global random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"🔬 Global seed set to: {seed}")


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information for reproducibility."""
    device_info = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
    }
    
    if torch.cuda.is_available():
        device_info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            'gpu_count': torch.cuda.device_count(),
        })
    
    return device_info


def setup_reproducible_environment(seed: int = 42, 
                                  log_level: str = 'INFO',
                                  save_config: bool = True,
                                  config_path: str = 'results/experiment_config.json') -> Dict[str, Any]:
    """
    Setup complete reproducible environment for CortexFlow experiments.
    
    Args:
        seed: Random seed value
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        save_config: Whether to save configuration to file
        config_path: Path to save configuration
    
    Returns:
        Dictionary containing experiment configuration
    """
    # Set global seed
    set_global_seed(seed)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get device information
    device_info = get_device_info()
    
    # Create experiment configuration
    config = {
        'experiment': {
            'timestamp': datetime.now().isoformat(),
            'seed': seed,
            'log_level': log_level,
        },
        'environment': device_info,
        'reproducibility': {
            'deterministic': True,
            'benchmark': False,
            'global_seed_set': True,
        }
    }
    
    # Print configuration
    print("🧠⚡ CortexFlow Reproducible Environment Setup")
    print("=" * 50)
    print(f"🔬 Seed: {seed}")
    print(f"🎮 Device: {device_info['device']}")
    if device_info['cuda_available']:
        print(f"🚀 GPU: {device_info['gpu_name']}")
        print(f"💾 Memory: {device_info['gpu_memory']}")
    print(f"🐍 Python: {device_info['python_version']}")
    print(f"🔥 PyTorch: {device_info['torch_version']}")
    print("=" * 50)
    
    # Save configuration if requested
    if save_config:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📁 Configuration saved: {config_path}")
    
    return config


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """Get comprehensive model information for reproducibility."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info = {
        'model_class': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'device': str(next(model.parameters()).device),
    }
    
    return model_info


def save_experiment_metadata(config: Dict[str, Any],
                           model: Optional[torch.nn.Module] = None,
                           dataset_info: Optional[Dict[str, Any]] = None,
                           training_info: Optional[Dict[str, Any]] = None,
                           save_path: str = 'results/experiment_metadata.json') -> None:
    """
    Save comprehensive experiment metadata for reproducibility.
    
    Args:
        config: Experiment configuration from setup_reproducible_environment
        model: PyTorch model (optional)
        dataset_info: Dataset information (optional)
        training_info: Training information (optional)
        save_path: Path to save metadata
    """
    metadata = {
        'experiment_config': config,
        'timestamp': datetime.now().isoformat(),
    }
    
    if model is not None:
        metadata['model'] = get_model_info(model)
    
    if dataset_info is not None:
        metadata['dataset'] = dataset_info
    
    if training_info is not None:
        metadata['training'] = training_info
    
    # Save metadata
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📊 Experiment metadata saved: {save_path}")


def verify_reproducibility(config_path: str = 'results/experiment_config.json') -> bool:
    """
    Verify that the current environment matches saved configuration.
    
    Args:
        config_path: Path to saved configuration
    
    Returns:
        True if environment matches, False otherwise
    """
    if not Path(config_path).exists():
        print(f"⚠️ Configuration file not found: {config_path}")
        return False
    
    # Load saved configuration
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    
    # Get current environment
    current_device_info = get_device_info()
    
    # Compare key environment variables
    saved_env = saved_config.get('environment', {})
    
    checks = {
        'Python version': saved_env.get('python_version') == current_device_info['python_version'],
        'PyTorch version': saved_env.get('torch_version') == current_device_info['torch_version'],
        'CUDA availability': saved_env.get('cuda_available') == current_device_info['cuda_available'],
    }
    
    if current_device_info['cuda_available']:
        checks['CUDA version'] = saved_env.get('cuda_version') == current_device_info['cuda_version']
        checks['GPU name'] = saved_env.get('gpu_name') == current_device_info['gpu_name']
    
    # Print verification results
    print("🔍 Reproducibility Verification")
    print("=" * 30)
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}: {passed}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("✅ Environment verification passed!")
    else:
        print("⚠️ Environment verification failed - results may not be reproducible")
    
    return all_passed


def create_reproducible_dataloader(dataset, batch_size: int, shuffle: bool = True, 
                                 seed: int = 42, **kwargs) -> torch.utils.data.DataLoader:
    """
    Create a reproducible DataLoader with proper seeding.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        seed: Random seed for worker initialization
        **kwargs: Additional DataLoader arguments
    
    Returns:
        Reproducible DataLoader
    """
    def worker_init_fn(worker_id):
        """Initialize worker with proper seeding."""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Set generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=worker_init_fn,
        generator=generator,
        **kwargs
    )


# Convenience function for quick setup
def quick_setup(seed: int = 42) -> Dict[str, Any]:
    """Quick reproducible environment setup for CortexFlow."""
    return setup_reproducible_environment(seed=seed, save_config=True)


if __name__ == "__main__":
    # Test reproducibility utilities
    print("🧪 Testing CortexFlow Reproducibility Utilities")
    
    # Setup environment
    config = quick_setup(seed=42)
    
    # Verify setup
    verify_reproducibility()
    
    print("✅ Reproducibility utilities test completed!")
