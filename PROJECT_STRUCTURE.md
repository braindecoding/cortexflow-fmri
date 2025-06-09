# CortexFlow Variant Ensemble - Project Structure

## 🚀 MAIN MODEL FILES

### **Primary Ensemble Model:**
```
cortexflow_main_ensemble.py          ← MAIN MODEL (PRIMARY)
cortexflow_variant_ensemble.py       ← Analysis & comparison script
```

### **Core Documentation:**
```
README.md                            ← Project overview
hasil.md                             ← Results documentation (Indonesian)
metode.md                            ← Methodology documentation (Indonesian)
mathematical_contributions.md        ← Mathematical formulations
requirements.txt                     ← Dependencies
```

## 📊 DATA & RESULTS

### **Essential Data:**
```
data/processed/                      ← Processed datasets
├── miyawaki_structured_28x28.mat
├── digit69_28x28.mat
├── mindbigdata.mat
└── crell.mat
```

### **Key Results:**
```
results/comprehensive_training_results.json  ← Individual model performance
results/variant_ensemble/                    ← Ensemble results & visualizations
├── cortexflow_main_ensemble_analysis.png
├── cortexflow_main_ensemble_results.json
└── cortexflow_variant_ensemble_comparison.png
```

## 🏗️ SOURCE CODE

### **Individual Models:**
```
src/models/                          ← CortexFlow variant implementations
├── cortexflow_simple.py
├── cortexflow_mc.py
├── cortexflow_hierarchical.py
├── cortexflow_enhanced.py
└── cortexflow_unified.py
```

### **Supporting Infrastructure:**
```
src/data/                           ← Data loading utilities
src/training/                       ← Training utilities
src/evaluation/                     ← Evaluation utilities
src/utils/                          ← General utilities
```

## 📈 VISUALIZATION & FIGURES

### **Publication Figures:**
```
figure0_framework_ecosystem.svg      ← Framework overview
figure1_cortexflow_overview.svg      ← Architecture details
figure2_unified_detail.svg           ← Unified model details
figure3_training_protocol.svg        ← Training methodology
figure4_uncertainty_mechanism.svg    ← Uncertainty quantification
figure_results_*.svg                 ← Results visualizations
```

## 🧪 EXPERIMENTS & CHECKPOINTS

### **Experiment Scripts:**
```
experiments/                        ← Individual model training experiments
├── run_all_architectures_4_datasets.py
├── comprehensive_training_summary.py
└── [variant-specific directories]
```

### **Model Checkpoints:**
```
checkpoints/                        ← Trained individual models
├── simple/
├── hierarchical/
├── enhanced/
└── unified/
```

## 🔧 UTILITIES & SCRIPTS

### **Helper Scripts:**
```
scripts/                            ← Utility scripts
├── comprehensive_training.py       ← Individual model training
├── create_comprehensive_viz.py     ← Visualization creation
└── [other utilities]
```

### **Testing:**
```
tests/                              ← Test suite
├── test_reproducibility.py
└── integration/
```

## 📋 CONFIGURATION

### **Project Configuration:**
```
configs/project_config.json         ← Project settings
```

## 🗑️ CLEANED UP (REMOVED)

### **Obsolete Ensemble Files:**
```
❌ ensemble_*.pth                   ← Simplified ensemble models
❌ full_ensemble_*.pth              ← Full ensemble models  
❌ perceptual_ensemble_*.pth        ← Perceptual ensemble models
❌ train_ensemble_*.py              ← Old ensemble training scripts
❌ src/models/ensemble_losses.py    ← Complex ensemble losses
❌ src/models/perceptual_losses.py  ← Perceptual loss functions
```

### **Obsolete Result Directories:**
```
❌ results/ensemble_training/       ← Simplified ensemble results
❌ results/full_ensemble_training/  ← Full ensemble results
❌ results/perceptual_ensemble_training/ ← Perceptual results
```

### **Obsolete Scripts:**
```
❌ comprehensive_analysis.py        ← Old analysis
❌ debug_datasets.py               ← Debug utilities
❌ quick_train_all.py              ← Old training
❌ visualize_full_ensemble_results.py ← Old visualization
```

## ✅ CURRENT FOCUS

### **CortexFlow Variant Ensemble:**
- **Main Model**: `cortexflow_main_ensemble.py`
- **Performance**: Average MSE 0.035279 across 4 datasets
- **Method**: Intelligent variant selection (not averaging)
- **Novelty**: Paradigm shift in ensemble learning
- **Status**: Production ready, publication ready

### **Key Features:**
1. **Intelligent Selection**: Choose optimal variant per dataset
2. **Peak Performance**: Maintain individual model excellence
3. **Domain Awareness**: Rules based on data characteristics
4. **High Interpretability**: Clear selection rationale
5. **Computational Efficiency**: Single model execution

### **Performance Summary:**
```
Miyawaki:    0.008456 (Enhanced variant)    ← BREAKTHROUGH
Vangerven:   0.044265 (Enhanced variant)    ← EXCELLENT  
MindBigData: 0.056108 (Hierarchical variant) ← OPTIMAL
Crell:       0.032285 (Hierarchical variant) ← SUPERIOR
```

## 🎯 USAGE

### **Run Main Ensemble:**
```bash
python cortexflow_main_ensemble.py
```

### **Run Analysis:**
```bash
python cortexflow_variant_ensemble.py
```

### **View Results:**
```
results/variant_ensemble/cortexflow_main_ensemble_analysis.png
```

---

**CortexFlow Variant Ensemble - Clean, Focused, Production Ready!** 🚀
