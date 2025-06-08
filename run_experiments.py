#!/usr/bin/env python3
"""
🚀 CORTEXFLOW MAIN EXPERIMENT RUNNER
================================================================================
Central script to run all experiments with perfect reproducibility
================================================================================
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.reproducibility import set_all_seeds, get_device
import json

def load_config():
    """Load project configuration."""
    config_path = Path('configs/project_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def run_all_experiments():
    """Run all experiments in sequence."""
    print("🚀 CORTEXFLOW EXPERIMENT SUITE")
    print("="*80)
    
    # Setup reproducibility
    config = load_config()
    seed = config['reproducibility']['seed']
    set_all_seeds(seed)
    device = get_device()
    
    print(f"\n📋 Configuration loaded from: configs/project_config.json")
    print(f"🎯 Seed: {seed}")
    print(f"🎮 Device: {device}")
    
    # Import and run experiments
    try:
        from experiments.run_all_experiments import main as run_experiments
        run_experiments()
    except ImportError:
        print("⚠️  Experiment runner not found. Please run individual experiments.")
    
    print("\n🎉 All experiments completed!")

if __name__ == "__main__":
    run_all_experiments()
