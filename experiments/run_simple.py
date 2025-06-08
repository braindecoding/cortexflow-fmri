#!/usr/bin/env python3
"""
Simple CortexFlow Experiment Runner
Run from project root: python experiments/run_simple.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run simple CortexFlow experiment."""
    print("🚀 Running Simple CortexFlow Experiment")
    print("=" * 60)
    
    # Change to experiment directory
    experiment_dir = Path(__file__).parent / 'simple'
    os.chdir(experiment_dir)
    
    # Run experiment
    result = subprocess.run([sys.executable, 'cortexflow_training.py'], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ Simple experiment completed successfully!")
    else:
        print("\n❌ Simple experiment failed!")
        return result.returncode
    
    return 0

if __name__ == "__main__":
    exit(main())
