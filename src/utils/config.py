"""
Configuration and Utility Functions
===================================

Utility functions for neural decoding research including:
- Reproducibility settings
- Unified training configurations
- Random seed management
- Configuration management

Features:
    - Consistent reproducibility across experiments
    - Unified training configurations for all datasets
    - Easy configuration management
    - Academic research standards
"""

import torch
import numpy as np
import random


def set_reproducibility_seeds(seed=42):
    """
    Set all random seeds for reproducibility
    
    Sets seeds for:
    - PyTorch (CPU and GPU)
    - NumPy
    - Python random
    - CUDA deterministic operations
    
    Args:
        seed: Random seed value (default: 42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Reproducibility seeds set to {seed}")


# UNIFIED TRAINING CONFIGURATIONS FOR CONSISTENCY
UNIFIED_TRAINING_CONFIGS = {
    'miyawaki': {
        'epochs': 200, 'lr': 0.001, 'batch_size': 64, 'patience': 40,
        'models': {
            'CortexFlow_Lite': {'epochs': 200, 'lr': 0.001, 'batch_size': 64, 'patience': 40},
            'MinD_Vis': {'epochs': 250, 'lr': 0.0008, 'batch_size': 64, 'patience': 45},
            'Brain_Diffuser': {'epochs': 150, 'lr': 0.002, 'batch_size': 64, 'patience': 30},
            'CortexFlow_Enhanced': {'epochs': 300, 'lr': 0.0005, 'batch_size': 64, 'patience': 50},
            'CortexFlow_Ensemble': {'epochs': 250, 'lr': 0.0006, 'batch_size': 64, 'patience': 45}
        }
    },
    'vangerven': {
        'epochs': 200, 'lr': 0.001, 'batch_size': 64, 'patience': 40,
        'models': {
            'CortexFlow_Lite': {'epochs': 200, 'lr': 0.001, 'batch_size': 64, 'patience': 40},
            'MinD_Vis': {'epochs': 250, 'lr': 0.0008, 'batch_size': 64, 'patience': 45},
            'Brain_Diffuser': {'epochs': 150, 'lr': 0.002, 'batch_size': 64, 'patience': 30},
            'CortexFlow_Enhanced': {'epochs': 300, 'lr': 0.0005, 'batch_size': 64, 'patience': 50},
            'CortexFlow_Ensemble': {'epochs': 250, 'lr': 0.0006, 'batch_size': 64, 'patience': 45}
        }
    },
    'mindbigdata': {
        'epochs': 200, 'lr': 0.0005, 'batch_size': 64, 'patience': 40,  # Lower LR for stability
        'models': {
            'CortexFlow_Lite': {'epochs': 200, 'lr': 0.0005, 'batch_size': 64, 'patience': 40},
            'MinD_Vis': {'epochs': 250, 'lr': 0.0006, 'batch_size': 64, 'patience': 45},
            'Brain_Diffuser': {'epochs': 150, 'lr': 0.001, 'batch_size': 64, 'patience': 30},
            'CortexFlow_Enhanced': {'epochs': 300, 'lr': 0.0003, 'batch_size': 64, 'patience': 50},
            'CortexFlow_Ensemble': {'epochs': 250, 'lr': 0.0004, 'batch_size': 64, 'patience': 45}
        }
    },
    'crell': {
        'epochs': 200, 'lr': 0.001, 'batch_size': 64, 'patience': 40,
        'models': {
            'CortexFlow_Lite': {'epochs': 200, 'lr': 0.001, 'batch_size': 64, 'patience': 40},
            'MinD_Vis': {'epochs': 250, 'lr': 0.0008, 'batch_size': 64, 'patience': 45},
            'Brain_Diffuser': {'epochs': 150, 'lr': 0.002, 'batch_size': 64, 'patience': 30},
            'CortexFlow_Enhanced': {'epochs': 300, 'lr': 0.0005, 'batch_size': 64, 'patience': 50},
            'CortexFlow_Ensemble': {'epochs': 250, 'lr': 0.0006, 'batch_size': 64, 'patience': 45}
        }
    }
}


def get_unified_config(dataset_name, model_name):
    """
    Get unified training configuration for consistency
    
    Provides consistent training configurations across all experiments
    for reproducible research results.
    
    Args:
        dataset_name: Name of the dataset ('miyawaki', 'vangerven', etc.)
        model_name: Name of the model ('CortexFlow_Lite', 'CortexFlow_Enhanced', etc.)
        
    Returns:
        Dictionary with training configuration parameters
    """
    dataset_config = UNIFIED_TRAINING_CONFIGS.get(dataset_name, UNIFIED_TRAINING_CONFIGS['miyawaki'])

    # Get model-specific config, fallback to base dataset config (without 'models' key)
    if model_name in dataset_config['models']:
        model_config = dataset_config['models'][model_name]
    else:
        # Create base config without 'models' key
        model_config = {
            'epochs': dataset_config['epochs'],
            'lr': dataset_config['lr'],
            'batch_size': dataset_config['batch_size'],
            'patience': dataset_config['patience']
        }

    return model_config


def get_all_dataset_configs():
    """
    Get all available dataset configurations
    
    Returns:
        Dictionary with all dataset configurations
    """
    return UNIFIED_TRAINING_CONFIGS.copy()


def get_available_datasets():
    """
    Get list of available datasets
    
    Returns:
        List of available dataset names
    """
    return list(UNIFIED_TRAINING_CONFIGS.keys())


def get_available_models():
    """
    Get list of available models
    
    Returns:
        List of available model names
    """
    # Get models from first dataset (all datasets have same models)
    first_dataset = list(UNIFIED_TRAINING_CONFIGS.keys())[0]
    return list(UNIFIED_TRAINING_CONFIGS[first_dataset]['models'].keys())


def validate_config(dataset_name, model_name):
    """
    Validate dataset and model configuration
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if dataset_name not in UNIFIED_TRAINING_CONFIGS:
        print(f"❌ Invalid dataset: {dataset_name}")
        print(f"   Available datasets: {get_available_datasets()}")
        return False
    
    if model_name not in UNIFIED_TRAINING_CONFIGS[dataset_name]['models']:
        print(f"❌ Invalid model: {model_name}")
        print(f"   Available models: {get_available_models()}")
        return False
    
    return True


def print_config_summary():
    """Print summary of all available configurations"""
    print("📋 UNIFIED TRAINING CONFIGURATIONS SUMMARY")
    print("=" * 60)
    
    for dataset_name, dataset_config in UNIFIED_TRAINING_CONFIGS.items():
        print(f"\n📊 Dataset: {dataset_name.upper()}")
        print(f"   Base config: epochs={dataset_config['epochs']}, lr={dataset_config['lr']}, "
              f"batch_size={dataset_config['batch_size']}, patience={dataset_config['patience']}")
        
        print(f"   Models:")
        for model_name, model_config in dataset_config['models'].items():
            print(f"     {model_name}: epochs={model_config['epochs']}, lr={model_config['lr']}, "
                  f"batch_size={model_config['batch_size']}, patience={model_config['patience']}")


# Set global reproducibility by default
set_reproducibility_seeds(42)

# Note: For production, you can disable deterministic mode for speed:
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False


if __name__ == "__main__":
    # Test configuration functions
    print("🧪 TESTING CONFIGURATION FUNCTIONS")
    print("=" * 50)
    
    # Test reproducibility
    set_reproducibility_seeds(123)
    
    # Test configuration retrieval
    config = get_unified_config('miyawaki', 'CortexFlow_Enhanced')
    print(f"✅ Config retrieved: {config}")
    
    # Test validation
    is_valid = validate_config('miyawaki', 'CortexFlow_Enhanced')
    print(f"✅ Config validation: {is_valid}")
    
    # Test invalid config
    is_valid = validate_config('invalid_dataset', 'invalid_model')
    print(f"✅ Invalid config test: {is_valid}")
    
    # Print summary
    print_config_summary()
    
    print("\n✅ Configuration functions working correctly!")
