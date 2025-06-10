#!/usr/bin/env python3
"""
Test WSL Environment
===================

Quick test to verify that the existing WSL environment has all
necessary dependencies for reproduction.
"""

import sys
import subprocess

def test_wsl_environment():
    """Test WSL environment dependencies"""
    
    print("🔍 TESTING WSL ENVIRONMENT READINESS")
    print("=" * 60)
    print("Testing existing WSL environment for CortexFlow reproduction...")
    
    # Test core dependencies
    dependencies = [
        ('torch', 'PyTorch for deep learning'),
        ('numpy', 'NumPy for numerical computing'),
        ('scipy', 'SciPy for scientific computing'),
        ('matplotlib', 'Matplotlib for visualization'),
        ('pathlib', 'Pathlib for file handling')
    ]
    
    missing_deps = []
    available_deps = []
    
    for dep_name, description in dependencies:
        try:
            __import__(dep_name)
            available_deps.append((dep_name, description))
            print(f"✅ {dep_name}: Available")
        except ImportError:
            missing_deps.append((dep_name, description))
            print(f"❌ {dep_name}: Missing")
    
    # Test PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"✅ CUDA: Available ({gpu_name}, CUDA {cuda_version})")
        else:
            print("⚠️  CUDA: Not available (will use CPU)")
    except ImportError:
        print("❌ PyTorch: Not available")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"DEPENDENCY CHECK: {len(available_deps)}/{len(dependencies)} available")
    
    if missing_deps:
        print("\n❌ MISSING DEPENDENCIES:")
        for dep_name, description in missing_deps:
            print(f"   - {dep_name}: {description}")
        
        print("\n📦 INSTALLATION COMMANDS:")
        dep_names = [dep[0] for dep in missing_deps]
        print(f"   pip install {' '.join(dep_names)}")
        
        return False
    else:
        print("\n✅ ALL DEPENDENCIES AVAILABLE!")
        print("✅ WSL environment ready for CortexFlow reproduction")
        print("✅ No additional package installation needed")
        return True

def test_file_access():
    """Test file access in WSL environment"""
    
    print("\n🔍 Testing file access...")
    
    # Test if we can access the project files
    import os
    from pathlib import Path
    
    essential_files = [
        'wsl_gpu_complete_training.py',
        'data/processed',
        'results/wsl_gpu_training'
    ]
    
    accessible_files = []
    inaccessible_files = []
    
    for file_path in essential_files:
        if Path(file_path).exists():
            accessible_files.append(file_path)
            print(f"✅ {file_path}: Accessible")
        else:
            inaccessible_files.append(file_path)
            print(f"❌ {file_path}: Not found")
    
    if inaccessible_files:
        print(f"\n⚠️  Some files not accessible: {inaccessible_files}")
        print("   Make sure you're in the correct directory")
        return False
    else:
        print("\n✅ All essential files accessible")
        return True

def test_python_version():
    """Test Python version compatibility"""
    
    print("\n🔍 Testing Python version...")
    
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major >= 3 and python_version.minor >= 8:
        print("✅ Python version compatible (3.8+)")
        return True
    else:
        print("❌ Python version too old (need 3.8+)")
        return False

def generate_environment_report():
    """Generate environment report for documentation"""
    
    print("\n📊 Generating environment report...")
    
    try:
        import torch
        import numpy as np
        import scipy
        import matplotlib
        
        report = f"""# WSL Environment Report

## Python Environment
- **Python**: {sys.version}
- **Platform**: {sys.platform}

## Core Dependencies
- **PyTorch**: {torch.__version__}
- **NumPy**: {np.__version__}
- **SciPy**: {scipy.__version__}
- **Matplotlib**: {matplotlib.__version__}

## GPU Information
"""
        
        if torch.cuda.is_available():
            report += f"""- **CUDA Available**: Yes
- **GPU**: {torch.cuda.get_device_name(0)}
- **CUDA Version**: {torch.version.cuda}
- **GPU Memory**: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB
"""
        else:
            report += "- **CUDA Available**: No (CPU mode)\n"
        
        report += f"""
## Environment Status
- **Ready for CortexFlow**: Yes
- **Virtual Environment**: Not needed (using existing WSL environment)
- **Additional Setup**: None required

## Reproduction Command
```bash
# In WSL environment
cd "/mnt/c/Users/Windows 11/Documents/cortexflow-fmri"
python wsl_gpu_complete_training.py
```

Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open('WSL_ENVIRONMENT_REPORT.md', 'w') as f:
            f.write(report)
        
        print("✅ Environment report saved: WSL_ENVIRONMENT_REPORT.md")
        return True
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return False

def main():
    """Main test execution"""
    
    tests = [
        ("Python Version", test_python_version),
        ("WSL Dependencies", test_wsl_environment),
        ("File Access", test_file_access),
        ("Environment Report", generate_environment_report)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"WSL ENVIRONMENT TEST: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 WSL ENVIRONMENT READY!")
        print("✅ No virtual environment setup needed")
        print("✅ Can proceed with CortexFlow reproduction")
        print("\n📋 Next steps:")
        print("1. Run: python wsl_gpu_complete_training.py")
        print("2. Verify results with: python final_dissertation_verification.py")
    else:
        print("❌ WSL ENVIRONMENT NEEDS SETUP!")
        print("⚠️  Install missing dependencies before reproduction")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
