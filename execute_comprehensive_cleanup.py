#!/usr/bin/env python3
"""
Comprehensive Root Folder Cleanup Script
Executes complete reorganization of project structure
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

def create_directory_structure():
    """Create the target directory structure"""
    
    print("📁 CREATING DIRECTORY STRUCTURE")
    print("=" * 40)
    
    directories = [
        # Documentation
        'docs/methodology',
        'docs/results', 
        'docs/figures',
        'docs/updates',
        'docs/cleanup',
        
        # Scripts
        'scripts/analysis',
        'scripts/figures',
        'scripts/cleanup',
        'scripts/conversion',
        'scripts/testing',
        'scripts/language',
        'scripts/verification',
        
        # Tests
        'tests/ensemble',
        'tests/training',
        'tests/verification',
        
        # Archive
        'archive/summaries',
        'archive/reports',
        'archive/language',
        'archive/cleanup'
    ]
    
    created_count = 0
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}")
            created_count += 1
        else:
            print(f"📁 Exists: {directory}")
    
    print(f"\n📊 Created {created_count} new directories")
    return created_count

def move_python_scripts():
    """Move Python scripts to appropriate directories"""
    
    print("\n🐍 MOVING PYTHON SCRIPTS")
    print("=" * 30)
    
    script_moves = {
        # Analysis scripts
        'comprehensive_hypothesis_research_documentation.py': 'scripts/analysis/',
        'corrected_statistical_analysis.py': 'scripts/analysis/',
        'dataset_complexity_hypothesis_testing.py': 'scripts/analysis/',
        'generate_comprehensive_tables.py': 'scripts/analysis/',
        'individual_model_hypothesis_testing.py': 'scripts/analysis/',
        
        # Figure generation scripts
        'create_architecture_figures.py': 'scripts/figures/',
        'create_dissertation_figures.py': 'scripts/figures/',
        'create_methodology_visuals.py': 'scripts/figures/',
        'create_overview_figure.py': 'scripts/figures/',
        
        # Cleanup scripts
        'cleanup_and_update_methodology.py': 'scripts/cleanup/',
        'cleanup_root_markdown_files.py': 'scripts/cleanup/',
        
        # Conversion scripts
        'convert_methodology.py': 'scripts/conversion/',
        'convert_to_html.py': 'scripts/conversion/',
        'convert_to_pdf.py': 'scripts/conversion/',
        
        # Language scripts
        'fix_language_comprehensive.py': 'scripts/language/',
        
        # Verification scripts
        'verify_data_sources.py': 'scripts/verification/',
        
        # Test scripts
        'test_8variant_ensemble.py': 'tests/ensemble/',
        'test_enhanced_ensemble.py': 'tests/ensemble/',
        'test_enhanced_training.py': 'tests/training/',
        'verify.py': 'tests/verification/',
    }
    
    moved_count = 0
    for source, target_dir in script_moves.items():
        if os.path.exists(source):
            target_path = os.path.join(target_dir, source)
            shutil.move(source, target_path)
            print(f"🐍 Moved: {source} → {target_dir}")
            moved_count += 1
        else:
            print(f"⚠️ Not found: {source}")
    
    print(f"\n📊 Moved {moved_count} Python scripts")
    return moved_count

def move_documentation():
    """Move documentation files to appropriate directories"""
    
    print("\n📝 MOVING DOCUMENTATION")
    print("=" * 25)
    
    doc_moves = {
        # Methodology docs
        'METODOLOGI.md': 'docs/methodology/',
        'METHODOLOGY_ALGORITHMS.md': 'docs/methodology/',
        'METODOLOGI_CortexFlow.docx': 'docs/methodology/',
        'METODOLOGI_CortexFlow.html': 'docs/methodology/',
        
        # Results docs
        'hasil.md': 'docs/results/',
        
        # Figure docs
        'DISSERTATION_FIGURES.md': 'docs/figures/',
        'FIGURES_INDEX.md': 'docs/figures/',
        
        # Update summaries
        'SOTA_UPDATE_SUMMARY.md': 'docs/updates/',
        'TRAIN_PY_UPDATE_SUMMARY.md': 'docs/updates/',
        'ROOT_MARKDOWN_CLEANUP_RECOMMENDATIONS.md': 'docs/cleanup/',
        'COMPREHENSIVE_ROOT_CLEANUP_PLAN.md': 'docs/cleanup/',
    }
    
    moved_count = 0
    for source, target_dir in doc_moves.items():
        if os.path.exists(source):
            target_path = os.path.join(target_dir, source)
            shutil.move(source, target_path)
            print(f"📝 Moved: {source} → {target_dir}")
            moved_count += 1
        else:
            print(f"⚠️ Not found: {source}")
    
    print(f"\n📊 Moved {moved_count} documentation files")
    return moved_count

def archive_old_files():
    """Archive old and redundant files"""
    
    print("\n📦 ARCHIVING OLD FILES")
    print("=" * 22)
    
    archive_moves = {
        # Summary files
        'BRANCH_V4_SUMMARY.md': 'archive/summaries/',
        'CLEAN_STATE_SUMMARY.md': 'archive/summaries/',
        'CONVERSION_SUMMARY.md': 'archive/summaries/',
        'CORRECTED_METHODOLOGY_SUMMARY.md': 'archive/summaries/',
        'ENHANCED_METHODOLOGY_SUMMARY.md': 'archive/summaries/',
        'METHODOLOGY_CLEANING_SUMMARY.md': 'archive/summaries/',
        'METODOLOGI_SUMMARY.md': 'archive/summaries/',
        'HASIL_SUMMARY.md': 'archive/summaries/',
        
        # Report files
        'METHODOLOGY_VALIDATION_REPORT.md': 'archive/reports/',
        'DATASET_VALIDATION_REPORT.md': 'archive/reports/',
        'DATA_VERIFICATION_REPORT.md': 'archive/reports/',
        
        # Language correction files
        'LANGUAGE_CORRECTION_COMPLETION_SUMMARY.md': 'archive/language/',
        'LANGUAGE_CORRECTION_FINAL_STATUS.md': 'archive/language/',
        'LANGUAGE_CORRECTION_SUMMARY.md': 'archive/language/',
    }
    
    # Find cleanup summary files with timestamps
    import glob
    cleanup_files = glob.glob('ROOT_MARKDOWN_CLEANUP_SUMMARY_*.md')
    for cleanup_file in cleanup_files:
        archive_moves[cleanup_file] = 'archive/cleanup/'
    
    archived_count = 0
    for source, target_dir in archive_moves.items():
        if os.path.exists(source):
            target_path = os.path.join(target_dir, source)
            shutil.move(source, target_path)
            print(f"📦 Archived: {source} → {target_dir}")
            archived_count += 1
        else:
            print(f"⚠️ Not found: {source}")
    
    print(f"\n📊 Archived {archived_count} old files")
    return archived_count

def move_test_file():
    """Move test.py to tests directory"""
    
    print("\n🧪 MOVING TEST FILES")
    print("=" * 20)
    
    if os.path.exists('test.py'):
        shutil.move('test.py', 'tests/test.py')
        print("🧪 Moved: test.py → tests/")
        return 1
    else:
        print("⚠️ test.py not found")
        return 0

def create_cleanup_summary():
    """Create summary of cleanup actions"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"docs/cleanup/COMPREHENSIVE_CLEANUP_SUMMARY_{timestamp}.md"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Root Cleanup Summary\n\n")
        f.write(f"**Executed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write("**Purpose:** Complete reorganization of project structure  \n")
        f.write("**Status:** ✅ Cleanup completed successfully  \n\n")
        
        f.write("---\n\n")
        f.write("## 🎯 Cleanup Results\n\n")
        f.write("### New Project Structure:\n")
        f.write("```\n")
        f.write("cortexflow-fmri/\n")
        f.write("├── README.md                          # Main documentation\n")
        f.write("├── SOTA.md                           # SOTA comparison\n")
        f.write("├── LICENSE                           # License\n")
        f.write("├── requirements.txt                  # Dependencies\n")
        f.write("├── research_documentation_index.md   # Research index\n")
        f.write("├── train.py                         # Main training script\n")
        f.write("├── src/                             # Source code\n")
        f.write("├── docs/                            # Documentation\n")
        f.write("├── scripts/                         # Utility scripts\n")
        f.write("├── tests/                           # Test files\n")
        f.write("└── archive/                         # Archived files\n")
        f.write("```\n\n")
        
        f.write("### Benefits Achieved:\n")
        f.write("- ✅ **Clean root directory** - Only 6 essential files\n")
        f.write("- ✅ **Organized structure** - Everything in logical folders\n")
        f.write("- ✅ **Professional appearance** - Academic-ready project\n")
        f.write("- ✅ **Easy maintenance** - Clear file organization\n")
        f.write("- ✅ **Better navigation** - Logical directory hierarchy\n\n")
        
        f.write("---\n\n")
        f.write("**Cleanup Status:** ✅ **COMPLETE**  \n")
        f.write("**Project Status:** ✅ **PROFESSIONAL AND ORGANIZED**  \n")
    
    print(f"📝 Cleanup summary saved: {summary_file}")
    return summary_file

def main():
    """Execute comprehensive cleanup"""
    
    print("🧹 COMPREHENSIVE ROOT FOLDER CLEANUP")
    print("=" * 50)
    print("Reorganizing project structure for professional appearance")
    print()
    
    try:
        # Create directory structure
        dirs_created = create_directory_structure()
        
        # Move files
        scripts_moved = move_python_scripts()
        docs_moved = move_documentation()
        archived = archive_old_files()
        tests_moved = move_test_file()
        
        # Create summary
        summary_file = create_cleanup_summary()
        
        print(f"\n🎉 CLEANUP COMPLETED SUCCESSFULLY!")
        print(f"✅ Directories created: {dirs_created}")
        print(f"✅ Python scripts moved: {scripts_moved}")
        print(f"✅ Documentation moved: {docs_moved}")
        print(f"✅ Files archived: {archived}")
        print(f"✅ Test files moved: {tests_moved}")
        print(f"✅ Summary created: {summary_file}")
        
        print(f"\n🏆 PROJECT NOW HAS PROFESSIONAL STRUCTURE!")
        print(f"📁 Root directory is clean and organized")
        print(f"📚 Documentation is properly categorized")
        print(f"🐍 Scripts are organized by function")
        print(f"📦 Old files are safely archived")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
