#!/usr/bin/env python3
"""
Final Complete Cleanup for SOTA.md
==================================

Mengganti SEMUA kata informal yang tersisa dengan bahasa formal Indonesia untuk disertasi.
"""

import re
from pathlib import Path

def final_complete_cleanup():
    """Complete cleanup of all remaining informal language"""
    
    print("PEMBERSIHAN LENGKAP KATA INFORMAL")
    print("=" * 50)
    
    sota_path = Path("SOTA.md")
    if not sota_path.exists():
        print("SOTA.md tidak ditemukan")
        return False
    
    with open(sota_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dictionary lengkap untuk mengganti SEMUA kata informal
    replacements = {
        # English words that should be Indonesian
        'benar': 'tepat',
        'BENAR': 'TEPAT',
        'terbaik': 'optimal',
        'TERBAIK': 'OPTIMAL',
        'terburuk': 'paling rendah',
        'TERBURUK': 'PALING RENDAH',
        'lebih baik': 'superior',
        'lebih buruk': 'inferior',
        'Best': 'Optimal',
        'best': 'optimal',
        'Worst': 'Paling rendah',
        'worst': 'paling rendah',
        'Better': 'Superior',
        'better': 'superior',
        'Worse': 'Inferior',
        'worse': 'inferior',
        
        # Performance terms
        'performance': 'kinerja',
        'Performance': 'Kinerja',
        'outperform': 'mengungguli',
        'Outperform': 'Mengungguli',
        
        # Technical terms
        'correct': 'tepat',
        'Correct': 'Tepat',
        'valid': 'sahih',
        'Valid': 'Sahih',
        'invalid': 'tidak sahih',
        'Invalid': 'Tidak sahih',
        
        # Comparison terms
        'vs ': 'dibandingkan dengan ',
        'vs.': 'dibandingkan dengan',
        'versus': 'dibandingkan dengan',
        
        # Quality terms
        'quality': 'kualitas',
        'Quality': 'Kualitas',
        'excellence': 'keunggulan',
        'Excellence': 'Keunggulan',
        
        # Specific phrases
        'Most Consistent': 'Paling Konsisten',
        'most consistent': 'paling konsisten',
        'Baseline - terbaik': 'Baseline - optimal',
        'baseline - terbaik': 'baseline - optimal',
        'profile terbaik': 'profil optimal',
        'Profile terbaik': 'Profil optimal',
        
        # Percentage comparisons
        '% lebih baik': '% superior',
        '% lebih buruk': '% inferior',
        
        # Method names with informal language
        'kinerja terbaik': 'kinerja optimal',
        'Kinerja terbaik': 'Kinerja optimal',
        'performa terbaik': 'kinerja optimal',
        'Performa terbaik': 'Kinerja optimal',
        
        # Dataset descriptions
        'yang benar': 'yang tepat',
        'Yang benar': 'Yang tepat',
        'data yang benar': 'data yang tepat',
        'Data yang benar': 'Data yang tepat',
        'pemetaan yang benar': 'pemetaan yang tepat',
        'Pemetaan yang benar': 'Pemetaan yang tepat',
        
        # Results descriptions
        'hasil yang benar': 'hasil yang tepat',
        'Hasil yang benar': 'Hasil yang tepat',
        'mapping yang benar': 'pemetaan yang tepat',
        'Mapping yang benar': 'Pemetaan yang tepat',
        
        # Scientific terms
        'honest': 'jujur',
        'Honest': 'Jujur',
        'transparency': 'transparansi',
        'Transparency': 'Transparansi',
        'integrity': 'integritas',
        'Integrity': 'Integritas',
    }
    
    # Apply all replacements
    original_content = content
    changes_made = 0
    
    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            changes_made += 1
            print(f"Mengganti: '{old}' → '{new}'")
    
    # Additional regex replacements for patterns
    patterns = [
        (r'\b(\d+\.?\d*)% lebih baik\b', r'\1% superior'),
        (r'\b(\d+\.?\d*)% lebih buruk\b', r'\1% inferior'),
        (r'\bterbaik\b', 'optimal'),
        (r'\bterburuk\b', 'paling rendah'),
        (r'\bbenar\b', 'tepat'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    # Write back to file
    with open(sota_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nTotal perubahan: {changes_made}")
    return True

def verify_complete_cleanup():
    """Verify that ALL informal language has been cleaned up"""
    
    print("\nVERIFIKASI PEMBERSIHAN LENGKAP:")
    print("-" * 40)
    
    sota_path = Path("SOTA.md")
    with open(sota_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for ALL possible informal terms
    informal_terms = [
        'terbaik', 'TERBAIK', 'terburuk', 'TERBURUK',
        'benar', 'BENAR', 'lebih baik', 'lebih buruk',
        'best', 'Best', 'worst', 'Worst',
        'better', 'Better', 'worse', 'Worse',
        'good', 'Good', 'poor', 'Poor',
        'excellent', 'Excellent', 'competitive', 'Competitive',
        'vs ', 'versus', 'outperform', 'Outperform'
    ]
    
    remaining_informal = []
    for term in informal_terms:
        if term in content:
            remaining_informal.append(term)
    
    if remaining_informal:
        print(f"Kata informal yang tersisa: {remaining_informal}")
        return False
    else:
        print("SEMUA kata informal telah dibersihkan")
        return True

def main():
    """Main execution"""
    
    print("PEMBERSIHAN LENGKAP KATA INFORMAL UNTUK DISERTASI")
    print("Mengganti SEMUA kata informal dengan bahasa formal Indonesia")
    print("=" * 70)
    
    # Perform complete cleanup
    cleanup_success = final_complete_cleanup()
    
    if cleanup_success:
        # Verify complete cleanup
        verification_success = verify_complete_cleanup()
        
        if verification_success:
            print("\nPEMBERSIHAN LENGKAP BERHASIL!")
            print("SOTA.md siap untuk laporan disertasi dengan bahasa formal")
        else:
            print("\nPEMBERSIHAN PERLU PERBAIKAN")
            print("Masih ada kata informal yang tersisa")
    else:
        print("\nPEMBERSIHAN GAGAL")

if __name__ == "__main__":
    main()
