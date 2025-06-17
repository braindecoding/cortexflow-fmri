#!/usr/bin/env python3
"""
Root Markdown Files Cleanup Script
Identifies and organizes markdown files in root folder
Removes outdated files and keeps only essential documentation
"""

import os
import shutil
from datetime import datetime

def analyze_root_markdown_files():
    """Analyze all markdown files in root folder"""
    
    print("🔍 ANALYZING ROOT MARKDOWN FILES")
    print("=" * 50)
    
    # Get all markdown files in root
    root_files = [f for f in os.listdir('.') if f.endswith('.md')]
    
    # Categorize files
    essential_files = [
        'README.md',
        'SOTA.md', 
        'research_documentation_index.md',
        'SOTA_UPDATE_SUMMARY.md',
        'TRAIN_PY_UPDATE_SUMMARY.md'
    ]
    
    # Files that are likely outdated or redundant
    potentially_outdated = [
        'BRANCH_V4_SUMMARY.md',
        'CLEAN_STATE_SUMMARY.md', 
        'CONVERSION_SUMMARY.md',
        'CORRECTED_METHODOLOGY_SUMMARY.md',
        'DATASET_VALIDATION_REPORT.md',
        'DATA_VERIFICATION_REPORT.md',
        'ENHANCED_METHODOLOGY_SUMMARY.md',
        'LANGUAGE_CORRECTION_COMPLETION_SUMMARY.md',
        'LANGUAGE_CORRECTION_FINAL_STATUS.md',
        'LANGUAGE_CORRECTION_SUMMARY.md',
        'METHODOLOGY_CLEANING_SUMMARY.md',
        'METHODOLOGY_VALIDATION_REPORT.md',
        'METODOLOGI_SUMMARY.md',
        'HASIL_SUMMARY.md'
    ]
    
    # Files that might contain useful content
    content_files = [
        'METODOLOGI.md',
        'METHODOLOGY_ALGORITHMS.md',
        'DISSERTATION_FIGURES.md',
        'FIGURES_INDEX.md',
        'hasil.md'
    ]
    
    print(f"📊 ANALYSIS RESULTS:")
    print(f"Total markdown files: {len(root_files)}")
    print(f"Essential files: {len(essential_files)}")
    print(f"Potentially outdated: {len(potentially_outdated)}")
    print(f"Content files: {len(content_files)}")
    
    return {
        'all_files': root_files,
        'essential': essential_files,
        'outdated': potentially_outdated,
        'content': content_files
    }

def check_file_content_for_outdated_data(filename):
    """Check if file contains outdated data or methodology"""
    
    if not os.path.exists(filename):
        return False, "File not found"
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for outdated patterns
        outdated_patterns = [
            'baseline threshold',
            'baseline_threshold', 
            '0.025',
            '2025-06-16',
            'enhanced 5-fold',
            'Enhanced 5-Fold CV',
            '0.041823',  # Old Vangerven result
            '0.054573',  # Old MindBigData result
            '0.028666',  # Old Crell result
            '0.015272'   # Old Miyawaki result
        ]
        
        found_patterns = []
        for pattern in outdated_patterns:
            if pattern.lower() in content.lower():
                found_patterns.append(pattern)
        
        has_outdated = len(found_patterns) > 0
        return has_outdated, found_patterns
        
    except Exception as e:
        return False, f"Error reading file: {e}"

def create_cleanup_recommendations():
    """Create recommendations for file cleanup"""
    
    print("\n📋 CREATING CLEANUP RECOMMENDATIONS")
    print("=" * 50)
    
    analysis = analyze_root_markdown_files()
    
    recommendations = {
        'keep': [],
        'archive': [],
        'delete': [],
        'update_needed': []
    }
    
    # Check each file
    for filename in analysis['all_files']:
        if filename in analysis['essential']:
            recommendations['keep'].append(filename)
        elif filename in analysis['content']:
            # Check content files for outdated data
            has_outdated, patterns = check_file_content_for_outdated_data(filename)
            if has_outdated:
                recommendations['update_needed'].append({
                    'file': filename,
                    'issues': patterns
                })
            else:
                recommendations['keep'].append(filename)
        elif filename in analysis['outdated']:
            recommendations['archive'].append(filename)
        else:
            # Unknown files - check content
            has_outdated, patterns = check_file_content_for_outdated_data(filename)
            if has_outdated:
                recommendations['archive'].append(filename)
            else:
                recommendations['keep'].append(filename)
    
    return recommendations

def execute_cleanup(recommendations, dry_run=True):
    """Execute the cleanup based on recommendations"""
    
    print(f"\n🧹 {'DRY RUN - ' if dry_run else ''}EXECUTING CLEANUP")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = f"archive_root_markdown_{timestamp}"
    
    if not dry_run:
        os.makedirs(archive_dir, exist_ok=True)
    
    # Archive files
    archived_count = 0
    for filename in recommendations['archive']:
        if os.path.exists(filename):
            if dry_run:
                print(f"📦 Would archive: {filename}")
            else:
                shutil.move(filename, os.path.join(archive_dir, filename))
                print(f"📦 Archived: {filename}")
            archived_count += 1
    
    # Delete files (if any)
    deleted_count = 0
    for filename in recommendations['delete']:
        if os.path.exists(filename):
            if dry_run:
                print(f"🗑️ Would delete: {filename}")
            else:
                os.remove(filename)
                print(f"🗑️ Deleted: {filename}")
            deleted_count += 1
    
    # Report files that need updates
    update_count = len(recommendations['update_needed'])
    if update_count > 0:
        print(f"\n⚠️ FILES NEEDING UPDATES:")
        for item in recommendations['update_needed']:
            print(f"   📝 {item['file']}: {item['issues']}")
    
    # Report kept files
    keep_count = len(recommendations['keep'])
    print(f"\n✅ FILES TO KEEP ({keep_count}):")
    for filename in recommendations['keep']:
        print(f"   📄 {filename}")
    
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"   📦 Archived: {archived_count}")
    print(f"   🗑️ Deleted: {deleted_count}")
    print(f"   ⚠️ Need updates: {update_count}")
    print(f"   ✅ Kept: {keep_count}")
    
    if not dry_run and archived_count > 0:
        print(f"\n📁 Archive created: {archive_dir}")
    
    return {
        'archived': archived_count,
        'deleted': deleted_count,
        'updated_needed': update_count,
        'kept': keep_count,
        'archive_dir': archive_dir if not dry_run else None
    }

def create_cleanup_summary(recommendations, results):
    """Create summary of cleanup actions"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"ROOT_MARKDOWN_CLEANUP_SUMMARY_{timestamp}.md"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Root Markdown Files Cleanup Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write("**Purpose:** Cleanup and organization of root markdown files  \n")
        f.write("**Status:** Analysis and recommendations completed  \n\n")
        
        f.write("---\n\n")
        f.write("## 📊 Cleanup Results\n\n")
        
        f.write("### Files Processed:\n")
        f.write(f"- **Total files analyzed:** {len(recommendations['keep']) + len(recommendations['archive']) + len(recommendations['delete']) + len(recommendations['update_needed'])}\n")
        f.write(f"- **Files kept:** {results['kept']}\n")
        f.write(f"- **Files archived:** {results['archived']}\n")
        f.write(f"- **Files deleted:** {results['deleted']}\n")
        f.write(f"- **Files needing updates:** {results['updated_needed']}\n\n")
        
        f.write("### Essential Files (Kept):\n")
        for filename in recommendations['keep']:
            f.write(f"- ✅ `{filename}`\n")
        f.write("\n")
        
        if recommendations['archive']:
            f.write("### Archived Files:\n")
            for filename in recommendations['archive']:
                f.write(f"- 📦 `{filename}` - Outdated or redundant\n")
            f.write("\n")
        
        if recommendations['update_needed']:
            f.write("### Files Needing Updates:\n")
            for item in recommendations['update_needed']:
                f.write(f"- ⚠️ `{item['file']}` - Contains: {', '.join(item['issues'])}\n")
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## 🎯 Recommendations\n\n")
        f.write("### Immediate Actions:\n")
        f.write("1. **Review archived files** before permanent deletion\n")
        f.write("2. **Update files with outdated content** to use latest methodology\n")
        f.write("3. **Focus on essential files** for documentation\n")
        f.write("4. **Maintain clean root directory** going forward\n\n")
        
        f.write("### Essential Documentation Structure:\n")
        f.write("- `README.md` - Main project documentation\n")
        f.write("- `SOTA.md` - State-of-the-art comparison\n")
        f.write("- `research_documentation_index.md` - Complete research index\n")
        f.write("- Content files with current methodology only\n\n")
        
        f.write("---\n\n")
        f.write("**Cleanup Status:** ✅ **ANALYSIS COMPLETE**  \n")
        f.write("**Next Step:** Review recommendations and execute cleanup  \n")
    
    print(f"📝 Cleanup summary saved: {summary_file}")
    return summary_file

def main():
    """Main cleanup function"""
    
    print("🧹 ROOT MARKDOWN FILES CLEANUP ANALYSIS")
    print("=" * 60)
    print("Analyzing root folder markdown files for cleanup")
    print()
    
    try:
        # Analyze files
        recommendations = create_cleanup_recommendations()
        
        # Dry run first
        print("\n" + "="*60)
        results = execute_cleanup(recommendations, dry_run=True)
        
        # Create summary
        summary_file = create_cleanup_summary(recommendations, results)
        
        print(f"\n🎉 ANALYSIS COMPLETED!")
        print(f"✅ Recommendations created")
        print(f"✅ Summary saved: {summary_file}")
        print(f"✅ Ready for cleanup execution")
        
        print(f"\n📋 TO EXECUTE CLEANUP:")
        print(f"1. Review the recommendations in {summary_file}")
        print(f"2. Run this script with execute=True to perform actual cleanup")
        print(f"3. Files will be safely archived, not deleted")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
