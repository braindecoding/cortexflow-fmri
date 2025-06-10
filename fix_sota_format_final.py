#!/usr/bin/env python3
"""
Fix SOTA Format Final
====================

Perbaiki format SOTA.md yang menjadi satu baris dan hapus kata-kata yang tidak sesuai untuk disertasi.
"""

import re
from pathlib import Path

def fix_sota_format():
    """Fix SOTA.md format and remove inappropriate terms"""
    
    print("MEMPERBAIKI FORMAT SOTA.md UNTUK DISERTASI")
    print("=" * 60)
    
    sota_path = Path("SOTA.md")
    if not sota_path.exists():
        print("SOTA.md tidak ditemukan")
        return False
    
    with open(sota_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix line breaks - restore proper markdown structure
    content = content.replace('## ', '\n\n## ')
    content = content.replace('### ', '\n\n### ')
    content = content.replace('#### ', '\n\n#### ')
    content = content.replace('**Gambar', '\n\n**Gambar')
    content = content.replace('**Tabel', '\n\n**Tabel')
    content = content.replace('![', '\n\n![')
    content = content.replace('| Peringkat', '\n\n| Peringkat')
    content = content.replace('**Dataset', '\n\n**Dataset')
    content = content.replace('**PENGATURAN', '\n\n**PENGATURAN')
    content = content.replace('**Protokol', '\n\n**Protokol')
    
    # Remove inappropriate terms for dissertation
    inappropriate_replacements = {
        'competitive': 'kompetitif',
        'vs ': 'dibandingkan dengan ',
        'better than': 'lebih baik dari',
        'worse than': 'lebih buruk dari',
        'simulasi proses diffusion': 'proses diffusion',
        'simulasi': 'pemodelan',
        'HONEST ASSESSMENT': 'Penilaian yang Jujur',
        'HONEST EVALUATION': 'Evaluasi yang Jujur',
        'HONEST RESULTS': 'Hasil yang Jujur',
        'valid RESULTS': 'Hasil yang Valid',
        'valid COMPARISON': 'Perbandingan yang Valid',
        'dikonfirmasi': 'terkonfirmasi',
        'validATED': 'tervalidasi',
        'terjaga': 'terjaga',
        'dipatuhi': 'dipatuhi',
    }
    
    for old, new in inappropriate_replacements.items():
        content = content.replace(old, new)
    
    # Fix multiple newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Ensure proper spacing around headers
    content = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', content)
    
    # Fix table formatting
    content = re.sub(r'\n(\|[^|]+\|)', r'\n\n\1', content)
    
    # Write back to file
    with open(sota_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Format SOTA.md telah diperbaiki")
    return True

def verify_final_format():
    """Verify final format is correct"""
    
    print("\nVERIFIKASI FORMAT FINAL:")
    print("-" * 40)
    
    sota_path = Path("SOTA.md")
    with open(sota_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count structure elements
    headers = content.count('## ')
    subheaders = content.count('### ')
    figures = content.count('**Gambar')
    tables = content.count('**Tabel')
    
    print(f"Headers (##): {headers}")
    print(f"Subheaders (###): {subheaders}")
    print(f"Figures: {figures}")
    print(f"Tables: {tables}")
    
    # Check for inappropriate terms
    inappropriate_terms = ['competitive', 'vs ', 'simulasi proses', 'HONEST']
    found_inappropriate = []
    
    for term in inappropriate_terms:
        if term in content:
            found_inappropriate.append(term)
    
    if found_inappropriate:
        print(f"Istilah tidak sesuai ditemukan: {found_inappropriate}")
        return False
    else:
        print("Tidak ada istilah tidak sesuai")
        return True

def main():
    """Main execution"""
    
    print("PERBAIKAN FINAL FORMAT SOTA.md")
    print("Memastikan format sesuai standar disertasi")
    print("=" * 60)
    
    # Fix format
    format_fixed = fix_sota_format()
    
    if format_fixed:
        # Verify format
        verification_success = verify_final_format()
        
        if verification_success:
            print("\nPERBAIKAN FORMAT BERHASIL!")
            print("SOTA.md siap untuk laporan disertasi")
        else:
            print("\nFORMAT PERLU PERBAIKAN LEBIH LANJUT")
    else:
        print("\nGAGAL MEMPERBAIKI FORMAT")

if __name__ == "__main__":
    main()
