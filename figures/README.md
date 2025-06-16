# CortexFlow Architecture Figures

This directory contains publication-ready architecture diagrams for all CortexFlow models and baseline methods used in the neural decoding framework.

## 📊 **Generated Figures**

### **🏗️ CortexFlow Architectures**

#### **1. CortexFlow-Lite Architecture**
- **Files**: `cortexflow_lite_architecture.png`, `cortexflow_lite_architecture.svg`
- **Description**: Optimized neural decoding architecture with professional branding
- **Key Features**: 
  - BatchNorm1d normalization for stability
  - Dropout (0.2, 0.15) for optimal regularization
  - Architecture: input → 512 → 256 → 512 → 784 (output)
- **Performance**: Winner on Vangerven dataset (0.041823 MSE)

#### **2. CortexFlow-MC (Monte Carlo) Architecture**
- **Files**: `cortexflow_mc_architecture.png`, `cortexflow_mc_architecture.svg`
- **Description**: Uncertainty-aware predictions with Monte Carlo Dropout
- **Key Features**:
  - MCDropout (always active, even in eval mode)
  - Systematic uncertainty quantification
  - LayerNorm normalization
- **Mathematical**: `F.dropout(x, p=0.15, training=True)`

#### **3. CortexFlow-Hierarchical Architecture**
- **Files**: `cortexflow_hierarchical_architecture.png`, `cortexflow_hierarchical_architecture.svg`
- **Description**: Temporal pattern recognition with multi-scale processing
- **Key Features**:
  - HierarchicalBlock dengan temporal attention
  - Multi-scale processing (3 levels)
  - Adaptive dropout per level
- **Mathematical**: `x = LayerNorm(Linear(x)) * Sigmoid(MLP_temporal(x))`

#### **4. CortexFlow-Enhanced Architecture**
- **Files**: `cortexflow_enhanced_architecture.png`, `cortexflow_enhanced_architecture.svg`
- **Description**: Advanced feature processing with multiple mechanisms
- **Key Features**:
  - Integration: MC + Hierarchical + Feature Alignment
  - EnhancedBlock dengan multiple mechanisms
  - Residual connections for gradient flow
- **Mathematical**: `x_attended + feature_alignment(x_attended)`

#### **5. CortexFlow Multi-Pathway Architecture**
- **Files**: `cortexflow_multipathway_architecture.png`, `cortexflow_multipathway_architecture.svg`
- **Description**: Advanced multi-pathway processing with cross-attention
- **Key Features**:
  - Cross-pathway attention mechanism
  - Adaptive pathway weighting
  - Dynamic gated fusion
  - Uncertainty-aware decoder
- **Performance**: Winner on MindBigData (0.054573 MSE)
- **Mathematical**: Multi-pathway with cross-attention fusion

#### **6. CortexFlow-Ensemble Architecture**
- **Files**: `cortexflow_ensemble_architecture.png`, `cortexflow_ensemble_architecture.svg`
- **Description**: 8-variant ensemble with intelligent weighting
- **Key Features**:
  - Learned weighting of all 8 CortexFlow variants
  - Neural network-based weight learning
  - Adaptive combination based on input
- **Performance**: Winner on Crell dataset (0.028666 MSE)
- **Mathematical**: `y_ensemble = Σᵢ₌₁⁸ wᵢ · fᵢ(x)`

### **🔬 SOTA Baseline Architectures**

#### **7. Brain-Diffuser Architecture**
- **Files**: `brain_diffuser_architecture.png`, `brain_diffuser_architecture.svg`
- **Description**: Pure diffusion approach for neural decoding
- **Key Features**:
  - SiLU activation and LayerNorm
  - 10 timesteps with beta linear schedule
  - Iterative denoising inference
- **Performance**: Winner on Miyawaki dataset (0.015272 MSE)
- **Mathematical**: Noise prediction with diffusion process

#### **8. MinD-Vis Architecture**
- **Files**: `mindvis_architecture.png`, `mindvis_architecture.svg`
- **Description**: State-of-the-art conditional diffusion (CVPR 2023)
- **Key Features**:
  - Sparse masked modeling with 15% masking
  - Conditional diffusion decoder
  - Noise injection for diffusion simulation
- **Mathematical**: Conditional diffusion with noise schedule

### **📋 Complete Overview**

#### **9. CortexFlow Complete Overview**
- **Files**: `cortexflow_complete_overview.png`, `cortexflow_complete_overview.svg`
- **Description**: Comprehensive overview of all CortexFlow architectures
- **Content**:
  - All 8 CortexFlow variants
  - Ensemble architecture
  - SOTA baselines
  - Performance results summary

## 🎯 **Usage Guidelines**

### **Academic Publications**
- Use **SVG format** for vector graphics in LaTeX documents
- Use **PNG format** for presentations and web display
- All figures are publication-ready with 300 DPI resolution

### **Figure Captions (Suggested)**
```latex
\caption{CortexFlow-Lite Architecture: Optimized neural decoding framework with BatchNorm1d normalization and adaptive dropout for enhanced performance on structured digit recognition tasks.}
```

### **Citation Information**
When using these figures, please cite:
```
CortexFlow: Enhanced Neural Decoding Framework with 5-Fold Cross-Validation
Enhanced Statistical Rigor for fMRI-to-Visual Reconstruction
```

## 📊 **Performance Summary**

Based on enhanced 5-fold cross-validation results:

| **Architecture** | **Best Dataset** | **MSE Score** | **Rank** |
|------------------|------------------|---------------|----------|
| **CortexFlow-Lite** | Vangerven | 0.041823 | 🥇 1st |
| **CortexFlow Multi-Pathway** | MindBigData | 0.054573 | 🥇 1st |
| **CortexFlow-Ensemble** | Crell | 0.028666 | 🥇 1st |
| **Brain-Diffuser** | Miyawaki | 0.015272 | 🥇 1st |

**Overall**: CortexFlow wins 3 out of 4 datasets with enhanced statistical validation.

## 🔧 **Technical Specifications**

- **Resolution**: 300 DPI for publication quality
- **Formats**: PNG (raster) and SVG (vector)
- **Style**: Publication-ready with serif fonts
- **Color Scheme**: Consistent across all architectures
- **Mathematical Notation**: LaTeX-style formatting

## 📝 **Generation Scripts**

- **Main Script**: `../create_architecture_figures.py`
- **Overview Script**: `../create_overview_figure.py`
- **Dependencies**: matplotlib, numpy

To regenerate figures:
```bash
python create_architecture_figures.py
python create_overview_figure.py
```
