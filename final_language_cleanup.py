#!/usr/bin/env python3
"""
Final Language Cleanup for SOTA.md
==================================

Mengganti semua kata informal yang tersisa dengan bahasa formal untuk disertasi.
"""

import re
from pathlib import Path

def final_language_cleanup():
    """Final cleanup of informal language in SOTA.md"""
    
    print("PEMBERSIHAN BAHASA INFORMAL FINAL")
    print("=" * 50)
    
    sota_path = Path("SOTA.md")
    if not sota_path.exists():
        print("SOTA.md tidak ditemukan")
        return False
    
    with open(sota_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dictionary untuk mengganti semua kata informal
    replacements = {
        # English informal words
        'competitive': 'kompetitif',
        'Competitive': 'Kompetitif',
        'excellent': 'sangat baik',
        'Excellent': 'Sangat baik',
        'good': 'baik',
        'Good': 'Baik',
        'poor': 'buruk',
        'Poor': 'Buruk',
        'clean': 'bersih',
        'Clean': 'Bersih',
        'awesome': 'luar biasa',
        'amazing': 'menakjubkan',
        'incredible': 'luar biasa',
        'revolutionary': 'revolusioner',
        'Revolutionary': 'Revolusioner',
        
        # Performance terms
        'Consistently poor': 'Konsisten buruk',
        'consistently poor': 'konsisten buruk',
        'Consistently Poor': 'Konsisten Buruk',
        'surprisingly excellent': 'mengejutkan sangat baik',
        'Surprisingly excellent': 'Mengejutkan sangat baik',
        'good performance': 'kinerja baik',
        'Good performance': 'Kinerja baik',
        'poor quality': 'kualitas buruk',
        'Poor quality': 'Kualitas buruk',
        'excellent/good': 'sangat baik/baik',
        
        # Comparison terms
        'vs ': 'dibandingkan dengan ',
        'better than': 'lebih baik dari',
        'worse than': 'lebih buruk dari',
        'superior to': 'superior terhadap',
        'competitive with': 'kompetitif dengan',
        
        # Technical terms
        'outperforms': 'mengungguli',
        'Outperforms': 'Mengungguli',
        'excels': 'unggul',
        'Excels': 'Unggul',
        'suitable': 'sesuai',
        'Suitable': 'Sesuai',
        
        # Specific phrases that need formal language
        'Competitive performance': 'Kinerja kompetitif',
        'competitive performance': 'kinerja kompetitif',
        'domain excellence': 'keunggulan domain',
        'Domain excellence': 'Keunggulan domain',
        'consistent quality': 'kualitas konsisten',
        'Consistent quality': 'Kualitas konsisten',
        
        # Remove remaining informal expressions
        'surprisingly': 'secara mengejutkan',
        'Surprisingly': 'Secara mengejutkan',
        'especially': 'terutama',
        'Especially': 'Terutama',
    }
    
    # Apply all replacements
    original_content = content
    changes_made = 0
    
    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            changes_made += 1
            print(f"Mengganti: '{old}' → '{new}'")
    
    # Write back to file
    with open(sota_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nTotal perubahan: {changes_made}")
    return True

def verify_cleanup():
    """Verify that all informal language has been cleaned up"""
    
    print("\nVERIFIKASI PEMBERSIHAN:")
    print("-" * 30)
    
    sota_path = Path("SOTA.md")
    with open(sota_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for remaining informal terms
    informal_terms = [
        'competitive', 'excellent', 'good', 'poor', 'clean',
        'awesome', 'amazing', 'incredible', 'revolutionary',
        'vs ', 'better than', 'worse than', 'outperforms',
        'surprisingly', 'especially'
    ]
    
    remaining_informal = []
    for term in informal_terms:
        if term.lower() in content.lower():
            remaining_informal.append(term)
    
    if remaining_informal:
        print(f"Kata informal yang tersisa: {remaining_informal}")
        return False
    else:
        print("Semua kata informal telah dibersihkan")
        return True

def main():
    """Main execution"""
    
    print("PEMBERSIHAN BAHASA INFORMAL FINAL UNTUK DISERTASI")
    print("Mengganti semua kata informal dengan bahasa formal")
    print("=" * 60)
    
    # Perform cleanup
    cleanup_success = final_language_cleanup()
    
    if cleanup_success:
        # Verify cleanup
        verification_success = verify_cleanup()
        
        if verification_success:
            print("\nPEMBERSIHAN BERHASIL!")
            print("SOTA.md siap untuk laporan disertasi")
        else:
            print("\nPEMBERSIHAN PERLU PERBAIKAN")
            print("Masih ada kata informal yang tersisa")
    else:
        print("\nPEMBERSIHAN GAGAL")

if __name__ == "__main__":
    main()
