#!/usr/bin/env python3
"""
Hierarchical CortexFlow Experiment Runner
Run from project root: python experiments/run_hierarchical.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run hierarchical CortexFlow experiment."""
    print("🏗️ Running Hierarchical CortexFlow Experiment")
    print("=" * 60)
    
    # Change to experiment directory
    experiment_dir = Path(__file__).parent / 'hierarchical'
    os.chdir(experiment_dir)
    
    # Run experiment
    result = subprocess.run([sys.executable, 'hierarchical_training.py'], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ Hierarchical experiment completed successfully!")
    else:
        print("\n❌ Hierarchical experiment failed!")
        return result.returncode
    
    return 0

if __name__ == "__main__":
    exit(main())
