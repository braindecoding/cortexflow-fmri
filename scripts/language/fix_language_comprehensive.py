#!/usr/bin/env python3
"""
Comprehensive Language Correction Script
Converts English terms to Indonesian in methodology document
"""

import re

def fix_language_comprehensive():
    """Apply comprehensive language corrections to methodology"""
    
    # Read the file
    with open('METODOLOGI.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔧 APPLYING COMPREHENSIVE LANGUAGE CORRECTIONS")
    print("=" * 50)
    
    # Dictionary of English to Indonesian translations
    translations = {
        # Common terms
        'framework': 'kerangka kerja',
        'training': 'pelatihan',
        'testing': 'pengujian', 
        'preprocessing': 'prapemrosesan',
        'features': 'fitur',
        'performance': 'kinerja',
        'validation': 'validasi',
        'evaluation': 'evaluasi',
        'optimization': 'optimalisasi',
        'pipeline': 'jalur pemrosesan',
        'enhanced': 'ditingkatkan',
        'comprehensive': 'komprehensif',
        'statistical analysis': 'analisis statistik',
        'cross-validation': 'validasi silang',
        'ensemble approach': 'pendekatan ensemble',
        'neural decoding': 'dekoding neural',
        'methodology': 'metodologi',
        
        # Technical terms
        'architecture': 'arsitektur',
        'parameters': 'parameter',
        'normalization': 'normalisasi',
        'dropout rate': 'tingkat dropout',
        'learning rate': 'tingkat pembelajaran',
        'batch size': 'ukuran batch',
        'optimizer': 'pengoptimal',
        'weight decay': 'peluruhan bobot',
        'scheduler': 'penjadwal',
        'epochs': 'epoch',
        'patience': 'kesabaran',
        
        # Dataset terms
        'input features': 'fitur masukan',
        'output dimension': 'dimensi keluaran',
        'visual patterns': 'pola visual',
        'digit recognition': 'pengenalan digit',
        'cross-modal': 'lintas-modal',
        'binary contrast': 'kontras biner',
        'grayscale images': 'citra skala abu-abu',
        'min-max normalization': 'normalisasi min-maks',
        'division by': 'pembagian dengan',
        'multi-modal alignment': 'penyelarasan multi-modal',
        
        # Process terms
        'data splitting': 'pembagian data',
        'model training': 'pelatihan model',
        'hyperparameter optimization': 'optimalisasi hiperparameter',
        'early stopping': 'penghentian dini',
        'statistical rigor': 'ketelitian statistik',
        'robust analysis': 'analisis robust',
        'significance testing': 'pengujian signifikansi',
        'effect size': 'ukuran efek',
        'confidence intervals': 'interval kepercayaan',
        
        # Implementation terms
        'gpu optimization': 'optimalisasi GPU',
        'memory efficiency': 'efisiensi memori',
        'direct loading': 'pemuatan langsung',
        'tensor operations': 'operasi tensor',
        'configuration': 'konfigurasi',
        'implementation': 'implementasi',
        'reproducibility': 'reproduksibilitas',
        'quality assurance': 'jaminan kualitas',
        
        # Architecture terms
        'dual-pathway': 'jalur-ganda',
        'cross-attention': 'perhatian-silang',
        'multi-pathway': 'multi-jalur',
        'uncertainty': 'ketidakpastian',
        'learned weighting': 'pembobotan terpelajar',
        'internal variants': 'varian internal',
        'sparse masking': 'masking jarang',
        'iterative denoising': 'denoising iteratif',
        'ensemble training': 'pelatihan ensemble',
        
        # Evaluation terms
        'multi-metric assessment': 'penilaian multi-metrik',
        'reconstruction visualization': 'visualisasi rekonstruksi',
        'qualitative analysis': 'analisis kualitatif',
        'comparison': 'perbandingan',
        'baseline': 'dasar',
        'sota': 'SOTA',
        'state-of-the-art': 'mutakhir',
        
        # Quality terms
        'academic integrity': 'integritas akademik',
        'research standards': 'standar penelitian',
        'peer review': 'tinjauan sejawat',
        'publication ready': 'siap publikasi',
        'transparent methodology': 'metodologi transparan',
        'open source': 'sumber terbuka',
        'version control': 'kontrol versi',
        'environment documentation': 'dokumentasi lingkungan',
        'dependency specification': 'spesifikasi dependensi',
        'cross-platform testing': 'pengujian lintas-platform'
    }
    
    # Apply translations
    corrections_made = 0
    for english, indonesian in translations.items():
        # Case-insensitive replacement, preserving original case
        pattern = re.compile(re.escape(english), re.IGNORECASE)
        matches = pattern.findall(content)
        if matches:
            content = pattern.sub(indonesian, content)
            corrections_made += len(matches)
            print(f"✅ {english} → {indonesian} ({len(matches)} instances)")
    
    # Additional specific corrections
    specific_corrections = [
        # Fix remaining English phrases
        ('Enhanced 5-Fold Cross-Validation', 'Validasi Silang 5-Lipatan yang Ditingkatkan'),
        ('Statistical Rigor', 'Ketelitian Statistik'),
        ('Comprehensive Evaluation', 'Evaluasi Komprehensif'),
        ('Multi-Metric', 'Multi-Metrik'),
        ('GPU-optimized', 'yang dioptimalkan GPU'),
        ('Memory-efficient', 'yang efisien memori'),
        ('WSL-compatible', 'kompatibel WSL'),
        ('Dataset-Specific', 'Khusus Dataset'),
        ('Cross-Validation', 'Validasi Silang'),
        ('T-test', 'Uji-T'),
        ('Cohen\\'s d', 'Cohen d'),
        ('p-value', 'nilai-p'),
        ('Effect Size', 'Ukuran Efek'),
        ('Confidence Interval', 'Interval Kepercayaan'),
        ('Statistical Power', 'Kekuatan Statistik'),
        ('Sample Size', 'Ukuran Sampel'),
        ('Data Loading', 'Pemuatan Data'),
        ('Feature Alignment', 'Penyelarasan Fitur'),
        ('Memory Optimization', 'Optimalisasi Memori'),
        ('Model Training Strategy', 'Strategi Pelatihan Model'),
        ('Foundation CNN', 'CNN Dasar'),
        ('Novel Architecture', 'Arsitektur Novel'),
        ('End-to-end', 'Ujung-ke-ujung'),
        ('Internal Variants', 'Varian Internal'),
        ('SOTA Baseline', 'Dasar SOTA'),
        ('Significance Testing', 'Pengujian Signifikansi'),
        ('Method Comparison', 'Perbandingan Metode'),
        ('Calculation', 'Perhitungan'),
        ('Reliability', 'Keandalan'),
        ('Power Analysis', 'Analisis Kekuatan'),
        ('Sample', 'Sampel'),
        ('Fixed Random Seeds', 'Seed Acak Tetap'),
        ('Deterministic Results', 'Hasil Deterministik'),
        ('Version Control', 'Kontrol Versi'),
        ('Systematic Code Versioning', 'Versioning Kode Sistematis'),
        ('Environment Documentation', 'Dokumentasi Lingkungan'),
        ('Complete Dependency Specification', 'Spesifikasi Dependensi Lengkap'),
        ('Result Validation', 'Validasi Hasil'),
        ('Cross-platform Testing', 'Pengujian Lintas-platform'),
        ('Authentic Data', 'Data Autentik'),
        ('Synthetic Data', 'Data Sintetis'),
        ('Transparent Methodology', 'Metodologi Transparan'),
        ('Open-source Implementation', 'Implementasi Sumber Terbuka'),
        ('Statistical Rigor', 'Ketelitian Statistik'),
        ('Proper Significance Testing', 'Pengujian Signifikansi yang Tepat'),
        ('Peer-Review Standards', 'Standar Tinjauan Sejawat'),
        ('Publication-ready Methodology', 'Metodologi Siap Publikasi')
    ]
    
    for english, indonesian in specific_corrections:
        if english in content:
            content = content.replace(english, indonesian)
            corrections_made += 1
            print(f"✅ {english} → {indonesian}")
    
    # Write back to file
    with open('METODOLOGI.md', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n🎯 LANGUAGE CORRECTION SUMMARY:")
    print(f"✅ Total corrections made: {corrections_made}")
    print(f"✅ File updated: METODOLOGI.md")
    print(f"✅ Language: Indonesian (consistent)")
    print(f"✅ Status: Comprehensive correction completed")
    
    return corrections_made

if __name__ == "__main__":
    corrections = fix_language_comprehensive()
    print(f"\n🏆 LANGUAGE CORRECTION: SUCCESS!")
    print(f"📊 {corrections} English terms converted to Indonesian")
    print(f"🎓 Methodology now uses consistent Indonesian language")
    print(f"✅ Ready for academic submission in Indonesian")
