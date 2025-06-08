# CortexFlow: fMRI-to-Image Reconstruction

Advanced neural architectures for reconstructing visual stimuli from fMRI brain activity.

## 🏗️ Project Structure

```
cortexflow-fmri/
├── 📁 experiments/          # Experimental scripts
│   ├── simple/             # Basic CortexFlow
│   ├── hierarchical/       # Hierarchical architecture
│   ├── enhanced/           # Enhanced with MC + Alignment
│   ├── configs/            # Configuration files
│   └── logs/               # Training logs
├── 📁 src/                 # Source code
│   ├── models/             # Neural architectures
│   ├── data/               # Data utilities
│   ├── utils/              # Helper functions
│   ├── training/           # Training utilities
│   └── evaluation/         # Evaluation tools
├── 📁 data/                # Datasets
│   ├── raw/                # Original data
│   ├── processed/          # Preprocessed data
│   └── external/           # External datasets
├── 📁 results/             # Experimental results
│   ├── simple/             # Basic results
│   ├── hierarchical/       # Hierarchical results
│   ├── enhanced/           # Enhanced results
│   └── comparisons/        # Comparative analysis
├── 📁 checkpoints/         # Model checkpoints
│   ├── simple/             # Simple models
│   ├── hierarchical/       # Hierarchical models
│   └── enhanced/           # Enhanced models
├── 📁 tests/               # Test scripts
├── 📁 docs/                # Documentation
└── 📁 scripts/             # Utility scripts
```

## 🚀 Quick Start

### 1. Simple CortexFlow
```bash
cd experiments/simple
python cortexflow_training.py
```

### 2. Hierarchical CortexFlow
```bash
cd experiments/hierarchical
python hierarchical_training.py
```

### 3. Enhanced CortexFlow (Monte Carlo + Feature Alignment)
```bash
cd experiments/enhanced
python enhanced_hierarchical_training.py
```

## 🏆 Architecture Comparison

| Architecture | Miyawaki Loss | Vangerven Loss | Features |
|--------------|---------------|----------------|----------|
| Simple | 0.018006 | 0.037846 | Basic encoder-decoder |
| Hierarchical | 0.050046 | 0.050851 | Multi-scale + Progressive |
| Enhanced | 0.051336 | 0.078782 | MC Dropout + Alignment |

## 📊 Features

### Simple CortexFlow
- Basic encoder-decoder architecture
- Standard reconstruction loss
- Fast training and inference

### Hierarchical CortexFlow
- Multi-scale temporal processing
- Feature pyramid networks
- Progressive decoding
- Advanced loss functions

### Enhanced CortexFlow
- Monte Carlo Dropout for uncertainty estimation
- Feature alignment across scales
- Uncertainty visualization
- Research-grade performance

## 🧪 Testing

```bash
# Test hierarchical architecture
cd tests
python test_hierarchical.py

# Test reproducibility
python test_reproducibility.py
```

## 📈 Results

All experimental results are organized in the `results/` directory:
- Reconstruction comparisons
- Progressive visualization
- Uncertainty maps
- Performance metrics

## 🔧 Configuration

Configuration files are available in `experiments/configs/` for easy customization of:
- Model hyperparameters
- Training settings
- Architecture parameters

## 📚 Documentation

Detailed documentation is available in the `docs/` directory covering:
- Architecture details
- Training procedures
- Evaluation metrics
- API reference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Miyawaki et al. for the visual cortex dataset
- Vangerven et al. for the digit recognition dataset
- PyTorch team for the deep learning framework
