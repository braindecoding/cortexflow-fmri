#!/usr/bin/env python3
"""
🧹 CORTEXFLOW CLEAN STATE SCRIPT
================================================================================
Removes all generated files to ensure clean state for reproducibility testing
================================================================================
"""

import os
import shutil
import sys
from pathlib import Path
import glob

def print_header(title):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"🧹 {title}")
    print(f"{'='*80}")

def print_section(title):
    """Print formatted section."""
    print(f"\n{'─'*60}")
    print(f"🗑️  {title}")
    print(f"{'─'*60}")

def remove_trained_models():
    """Remove all trained model files."""
    print_section("REMOVING TRAINED MODELS")
    
    model_patterns = [
        "results/**/*.pt",
        "results/**/*.pth",
        "checkpoints/**/*.pt",
        "checkpoints/**/*.pth"
    ]
    
    removed_count = 0
    for pattern in model_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                os.remove(file_path)
                print(f"✅ Removed model: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {file_path}: {e}")
    
    print(f"🧹 Removed {removed_count} model files")

def remove_visualizations():
    """Remove all visualization files."""
    print_section("REMOVING VISUALIZATIONS")
    
    viz_patterns = [
        "results/**/*.png",
        "results/**/*.jpg", 
        "results/**/*.jpeg",
        "results/**/*.pdf",
        "tests/results/*.png",
        "tests/results/*.jpg",
        "experiments/**/*.png"
    ]
    
    removed_count = 0
    for pattern in viz_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                os.remove(file_path)
                print(f"✅ Removed visualization: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {file_path}: {e}")
    
    print(f"🧹 Removed {removed_count} visualization files")

def remove_logs_and_results():
    """Remove log files and result files."""
    print_section("REMOVING LOGS AND RESULTS")
    
    log_patterns = [
        "results/**/*.json",
        "results/**/*.txt",
        "results/**/*.log",
        "tests/results/*.json",
        "tests/results/*.txt",
        "experiments/logs/*.log",
        "experiments/logs/*.txt"
    ]
    
    removed_count = 0
    for pattern in log_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                os.remove(file_path)
                print(f"✅ Removed result: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {file_path}: {e}")
    
    print(f"🧹 Removed {removed_count} result files")

def remove_cache_files():
    """Remove Python cache files."""
    print_section("REMOVING CACHE FILES")
    
    removed_count = 0
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"✅ Removed cache: {pycache_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {pycache_path}: {e}")
    
    # Remove .pyc files
    for file_path in glob.glob("**/*.pyc", recursive=True):
        try:
            os.remove(file_path)
            print(f"✅ Removed pyc: {file_path}")
            removed_count += 1
        except Exception as e:
            print(f"❌ Failed to remove {file_path}: {e}")
    
    print(f"🧹 Removed {removed_count} cache files")

def remove_temp_files():
    """Remove temporary files."""
    print_section("REMOVING TEMPORARY FILES")
    
    temp_patterns = [
        "*.tmp",
        "*.temp",
        ".DS_Store",
        "Thumbs.db",
        "*.swp",
        "*.swo",
        "*~"
    ]
    
    removed_count = 0
    for pattern in temp_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                os.remove(file_path)
                print(f"✅ Removed temp: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {file_path}: {e}")
    
    print(f"🧹 Removed {removed_count} temporary files")

def verify_clean_state():
    """Verify that clean state is achieved."""
    print_section("VERIFYING CLEAN STATE")
    
    # Check for remaining files
    check_patterns = [
        ("Model files", "results/**/*.pt"),
        ("Visualizations", "results/**/*.png"),
        ("Test results", "tests/results/*.json"),
        ("Cache files", "**/__pycache__", True)
    ]
    
    all_clean = True
    
    for description, pattern, *is_dir in check_patterns:
        if is_dir:
            # Check for directories
            found_dirs = []
            for root, dirs, files in os.walk('.'):
                if '__pycache__' in dirs:
                    found_dirs.append(os.path.join(root, '__pycache__'))
            
            if found_dirs:
                print(f"⚠️  {description}: {len(found_dirs)} remaining")
                all_clean = False
            else:
                print(f"✅ {description}: Clean")
        else:
            # Check for files
            found_files = glob.glob(pattern, recursive=True)
            if found_files:
                print(f"⚠️  {description}: {len(found_files)} remaining")
                all_clean = False
            else:
                print(f"✅ {description}: Clean")
    
    return all_clean

def preserve_essential_files():
    """Ensure essential files are preserved."""
    print_section("PRESERVING ESSENTIAL FILES")
    
    essential_files = [
        "README.md",
        "requirements.txt",
        "configs/project_config.json",
        "src/utils/reproducibility.py",
        "run_experiments.py"
    ]
    
    all_present = True
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"✅ Preserved: {file_path}")
        else:
            print(f"⚠️  Missing: {file_path}")
            all_present = False
    
    return all_present

def main():
    """Main clean state function."""
    print_header("CORTEXFLOW CLEAN STATE OPERATION")
    print("🎯 Removing all generated files for clean state reproducibility test")
    
    # Perform cleanup operations
    remove_trained_models()
    remove_visualizations()
    remove_logs_and_results()
    remove_cache_files()
    remove_temp_files()
    
    # Verify clean state
    print_header("CLEAN STATE VERIFICATION")
    
    is_clean = verify_clean_state()
    essential_preserved = preserve_essential_files()
    
    print_header("CLEAN STATE SUMMARY")
    
    if is_clean and essential_preserved:
        print("✅ CLEAN STATE ACHIEVED SUCCESSFULLY!")
        print("🔬 Ready for reproducibility testing from clean state")
        print("\n📋 Next steps:")
        print("   1. Run: python run_experiments.py")
        print("   2. Verify: All results should be identical to previous runs")
        print("   3. Check: Reproducibility metrics should be perfect")
        return True
    else:
        print("⚠️  CLEAN STATE INCOMPLETE")
        if not is_clean:
            print("   - Some generated files remain")
        if not essential_preserved:
            print("   - Some essential files are missing")
        print("\n📋 Manual cleanup may be required")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
