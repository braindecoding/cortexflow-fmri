#!/usr/bin/env python3
"""
Unified CortexFlow Experiment Runner
Runs the unified model experiments with different configurations
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run unified CortexFlow experiment."""
    print("🔬 Running Unified CortexFlow Experiment")
    print("=" * 60)
    
    # Change to unified experiment directory
    unified_dir = Path(__file__).parent / "unified"
    
    try:
        # Run the unified training script
        result = subprocess.run([
            sys.executable, "unified_training.py"
        ], cwd=unified_dir, check=True)
        
        print("✅ Unified experiment completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Unified experiment failed with return code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Unified experiment crashed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
