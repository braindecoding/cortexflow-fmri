# Comprehensive Root Folder Cleanup Plan

**Generated:** 2025-06-17 21:40:00  
**Purpose:** Complete cleanup and reorganization of root folder  
**Status:** 🚨 **URGENT CLEANUP NEEDED** - Root folder is very messy!  

---

## 🚨 CURRENT PROBLEM

### Root Folder is EXTREMELY MESSY:
- **50+ files** in root directory (should be ~10 max)
- **20+ Python scripts** that should be in `scripts/` or `tools/`
- **25+ Markdown files** that should be organized
- **Multiple redundant files** from different development phases
- **Poor project structure** - unprofessional appearance

### Impact:
- ❌ **Hard to navigate** - Can't find essential files
- ❌ **Unprofessional** - Looks like development mess
- ❌ **Confusing for reviewers** - Academic reviewers will be confused
- ❌ **Maintenance nightmare** - Too many files to keep updated
- ❌ **Version control issues** - Git history cluttered

---

## 📊 FILE ANALYSIS

### 🔧 PYTHON SCRIPTS (MOVE TO scripts/ - 20+ files):
```
cleanup_and_update_methodology.py          → scripts/cleanup/
cleanup_root_markdown_files.py             → scripts/cleanup/
comprehensive_hypothesis_research_documentation.py → scripts/analysis/
convert_methodology.py                     → scripts/conversion/
convert_to_html.py                         → scripts/conversion/
convert_to_pdf.py                          → scripts/conversion/
corrected_statistical_analysis.py          → scripts/analysis/
create_architecture_figures.py             → scripts/figures/
create_dissertation_figures.py             → scripts/figures/
create_methodology_visuals.py              → scripts/figures/
create_overview_figure.py                  → scripts/figures/
dataset_complexity_hypothesis_testing.py   → scripts/analysis/
fix_language_comprehensive.py              → scripts/language/
generate_comprehensive_tables.py           → scripts/analysis/
individual_model_hypothesis_testing.py     → scripts/analysis/
test_8variant_ensemble.py                  → scripts/testing/
test_enhanced_ensemble.py                  → scripts/testing/
test_enhanced_training.py                  → scripts/testing/
verify_data_sources.py                     → scripts/verification/
```

### 📝 MARKDOWN FILES (ORGANIZE - 25+ files):

#### ✅ KEEP IN ROOT (5 files):
```
README.md                               → ROOT (main documentation)
SOTA.md                                → ROOT (SOTA comparison)
LICENSE                                → ROOT (license file)
requirements.txt                       → ROOT (dependencies)
research_documentation_index.md        → ROOT (research index)
```

#### 📁 MOVE TO docs/ (10 files):
```
METODOLOGI.md                          → docs/methodology/
METHODOLOGY_ALGORITHMS.md              → docs/methodology/
hasil.md                               → docs/results/
DISSERTATION_FIGURES.md                → docs/figures/
FIGURES_INDEX.md                       → docs/figures/
METODOLOGI_CortexFlow.docx             → docs/methodology/
METODOLOGI_CortexFlow.html             → docs/methodology/
SOTA_UPDATE_SUMMARY.md                 → docs/updates/
TRAIN_PY_UPDATE_SUMMARY.md             → docs/updates/
ROOT_MARKDOWN_CLEANUP_RECOMMENDATIONS.md → docs/cleanup/
```

#### 📦 ARCHIVE (15+ files):
```
BRANCH_V4_SUMMARY.md                   → archive/summaries/
CLEAN_STATE_SUMMARY.md                 → archive/summaries/
CONVERSION_SUMMARY.md                  → archive/summaries/
CORRECTED_METHODOLOGY_SUMMARY.md       → archive/summaries/
ENHANCED_METHODOLOGY_SUMMARY.md        → archive/summaries/
METHODOLOGY_CLEANING_SUMMARY.md        → archive/summaries/
METHODOLOGY_VALIDATION_REPORT.md       → archive/reports/
METODOLOGI_SUMMARY.md                  → archive/summaries/
HASIL_SUMMARY.md                       → archive/summaries/
LANGUAGE_CORRECTION_*.md (3 files)     → archive/language/
DATASET_VALIDATION_REPORT.md           → archive/reports/
DATA_VERIFICATION_REPORT.md            → archive/reports/
ROOT_MARKDOWN_CLEANUP_SUMMARY_*.md     → archive/cleanup/
```

### 🧪 TEST FILES (MOVE TO tests/ - 4 files):
```
test.py                                → tests/
test_8variant_ensemble.py              → tests/ensemble/
test_enhanced_ensemble.py              → tests/ensemble/
test_enhanced_training.py              → tests/training/
verify.py                              → tests/verification/
```

---

## 🎯 TARGET STRUCTURE

### AFTER CLEANUP - Professional Structure:
```
cortexflow-fmri/
├── README.md                          # Main documentation
├── SOTA.md                           # SOTA comparison
├── LICENSE                           # License
├── requirements.txt                  # Dependencies
├── research_documentation_index.md   # Research index
├── train.py                         # Main training script
├── src/                             # Source code
├── data/                            # Data files
├── results/                         # Training results
├── figures/                         # Architecture figures
├── configs/                         # Configuration files
├── docs/                            # Documentation
│   ├── methodology/                 # Methodology docs
│   ├── results/                     # Results docs
│   ├── figures/                     # Figure docs
│   └── updates/                     # Update summaries
├── scripts/                         # Utility scripts
│   ├── analysis/                    # Analysis scripts
│   ├── figures/                     # Figure generation
│   ├── cleanup/                     # Cleanup scripts
│   ├── conversion/                  # Conversion scripts
│   └── testing/                     # Test scripts
├── tests/                           # Test files
└── archive/                         # Archived files
    ├── summaries/                   # Old summaries
    ├── reports/                     # Old reports
    └── language/                    # Language correction files
```

---

## 🚀 EXECUTION PLAN

### Phase 1: Create Directory Structure
```bash
mkdir -p docs/methodology docs/results docs/figures docs/updates docs/cleanup
mkdir -p scripts/analysis scripts/figures scripts/cleanup scripts/conversion scripts/testing scripts/language scripts/verification
mkdir -p tests/ensemble tests/training tests/verification
mkdir -p archive/summaries archive/reports archive/language archive/cleanup
```

### Phase 2: Move Python Scripts
```bash
# Analysis scripts
mv comprehensive_hypothesis_research_documentation.py scripts/analysis/
mv corrected_statistical_analysis.py scripts/analysis/
mv dataset_complexity_hypothesis_testing.py scripts/analysis/
mv generate_comprehensive_tables.py scripts/analysis/
mv individual_model_hypothesis_testing.py scripts/analysis/

# Figure generation scripts
mv create_architecture_figures.py scripts/figures/
mv create_dissertation_figures.py scripts/figures/
mv create_methodology_visuals.py scripts/figures/
mv create_overview_figure.py scripts/figures/

# Cleanup scripts
mv cleanup_and_update_methodology.py scripts/cleanup/
mv cleanup_root_markdown_files.py scripts/cleanup/

# Conversion scripts
mv convert_methodology.py scripts/conversion/
mv convert_to_html.py scripts/conversion/
mv convert_to_pdf.py scripts/conversion/

# Language scripts
mv fix_language_comprehensive.py scripts/language/

# Verification scripts
mv verify_data_sources.py scripts/verification/
```

### Phase 3: Move Documentation
```bash
# Methodology docs
mv METODOLOGI.md docs/methodology/
mv METHODOLOGY_ALGORITHMS.md docs/methodology/
mv METODOLOGI_CortexFlow.docx docs/methodology/
mv METODOLOGI_CortexFlow.html docs/methodology/

# Results docs
mv hasil.md docs/results/

# Figure docs
mv DISSERTATION_FIGURES.md docs/figures/
mv FIGURES_INDEX.md docs/figures/

# Update summaries
mv SOTA_UPDATE_SUMMARY.md docs/updates/
mv TRAIN_PY_UPDATE_SUMMARY.md docs/updates/
mv ROOT_MARKDOWN_CLEANUP_RECOMMENDATIONS.md docs/cleanup/
```

### Phase 4: Move Test Files
```bash
mv test.py tests/
mv test_8variant_ensemble.py tests/ensemble/
mv test_enhanced_ensemble.py tests/ensemble/
mv test_enhanced_training.py tests/training/
mv verify.py tests/verification/
```

### Phase 5: Archive Old Files
```bash
# Summary files
mv BRANCH_V4_SUMMARY.md archive/summaries/
mv CLEAN_STATE_SUMMARY.md archive/summaries/
mv CONVERSION_SUMMARY.md archive/summaries/
mv CORRECTED_METHODOLOGY_SUMMARY.md archive/summaries/
mv ENHANCED_METHODOLOGY_SUMMARY.md archive/summaries/
mv METHODOLOGY_CLEANING_SUMMARY.md archive/summaries/
mv METODOLOGI_SUMMARY.md archive/summaries/
mv HASIL_SUMMARY.md archive/summaries/

# Report files
mv METHODOLOGY_VALIDATION_REPORT.md archive/reports/
mv DATASET_VALIDATION_REPORT.md archive/reports/
mv DATA_VERIFICATION_REPORT.md archive/reports/

# Language correction files
mv LANGUAGE_CORRECTION_*.md archive/language/

# Cleanup files
mv ROOT_MARKDOWN_CLEANUP_SUMMARY_*.md archive/cleanup/
```

---

## ✅ BENEFITS

### Professional Appearance:
- **Clean root directory** - Only 6 essential files
- **Organized structure** - Everything in logical folders
- **Easy navigation** - Clear hierarchy
- **Academic ready** - Professional project presentation

### Maintenance Benefits:
- **Easier updates** - Know where everything is
- **Better version control** - Cleaner Git history
- **Reduced confusion** - No duplicate files
- **Scalable structure** - Easy to add new content

### Academic Benefits:
- **Reviewer friendly** - Easy to navigate
- **Professional impression** - Well-organized project
- **Clear documentation** - Logical structure
- **Publication ready** - Academic standards met

---

**Cleanup Priority:** 🚨 **URGENT - Execute immediately for professional project structure**
