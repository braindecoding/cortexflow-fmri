#!/usr/bin/env python3
"""
Enhanced CortexFlow Experiment Runner
Run from project root: python experiments/run_enhanced.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run enhanced CortexFlow experiment."""
    print("🔬 Running Enhanced CortexFlow Experiment")
    print("=" * 60)
    
    # Change to experiment directory
    experiment_dir = Path(__file__).parent / 'enhanced'
    os.chdir(experiment_dir)
    
    # Run experiment
    result = subprocess.run([sys.executable, 'enhanced_hierarchical_training.py'], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ Enhanced experiment completed successfully!")
    else:
        print("\n❌ Enhanced experiment failed!")
        return result.returncode
    
    return 0

if __name__ == "__main__":
    exit(main())
