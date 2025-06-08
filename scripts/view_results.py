#!/usr/bin/env python3
"""
View CortexFlow Results
Open and display comparison results
"""

import os
import subprocess
import sys
from pathlib import Path

def open_file(file_path):
    """Open file with default application."""
    try:
        if os.name == 'nt':  # Windows
            os.startfile(file_path)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.call(['open', file_path])
        else:
            print(f"Please manually open: {file_path}")
    except Exception as e:
        print(f"Could not open {file_path}: {e}")

def main():
    """Main function to view results."""
    print("🔍 CORTEXFLOW RESULTS VIEWER")
    print("=" * 60)
    
    # Check if results exist
    comparison_dir = Path("results/comparisons")
    if not comparison_dir.exists():
        print("❌ No comparison results found!")
        print("Run the comparison first: python scripts/comprehensive_comparison.py")
        return
    
    # List available files
    files = {
        "1": ("📊 Architecture Comparison Chart", "results/comparisons/architecture_comparison.png"),
        "2": ("📋 Detailed Comparison Table", "results/comparisons/detailed_comparison.csv"),
        "3": ("📝 Summary Report", "results/comparisons/COMPARISON_SUMMARY.md"),
        "4": ("🧠 Simple Results", "results/simple/"),
        "5": ("🏗️ Hierarchical Results", "results/hierarchical/"),
        "6": ("🔬 Enhanced Results", "results/enhanced/"),
    }
    
    print("\n📁 Available Results:")
    print("-" * 40)
    for key, (name, path) in files.items():
        status = "✅" if os.path.exists(path) else "❌"
        print(f"{key}. {status} {name}")
    
    print("\n🎯 Quick Summary:")
    print("-" * 40)
    print("🏆 Winner: Simple CortexFlow")
    print("   • Best loss: 0.018006 (Miyawaki), 0.037846 (Vangerven)")
    print("   • Fastest: 0.4 minutes total training")
    print("   • Most efficient: 7.1M parameters average")
    print("")
    print("🥈 Runner-up: Enhanced CortexFlow")
    print("   • Advanced features: MC Dropout + Alignment")
    print("   • Good performance: 0.056370 (Miyawaki), 0.080576 (Vangerven)")
    print("   • Research-ready with uncertainty estimation")
    print("")
    print("🥉 Third: Hierarchical CortexFlow")
    print("   • Multi-scale processing: [1,2,4,8] temporal scales")
    print("   • Complex architecture: 24.7M parameters average")
    print("   • Good for temporal analysis research")
    
    # Interactive selection
    print(f"\n🔍 Select file to open (1-{len(files)}) or 'q' to quit:")
    
    while True:
        choice = input("Enter choice: ").strip().lower()
        
        if choice == 'q':
            print("👋 Goodbye!")
            break
        elif choice in files:
            name, path = files[choice]
            if os.path.exists(path):
                print(f"📂 Opening: {name}")
                if os.path.isdir(path):
                    print(f"📁 Directory contents of {path}:")
                    for item in os.listdir(path):
                        print(f"   📄 {item}")
                else:
                    open_file(path)
            else:
                print(f"❌ File not found: {path}")
        else:
            print(f"❌ Invalid choice. Please enter 1-{len(files)} or 'q'")

if __name__ == "__main__":
    main()
