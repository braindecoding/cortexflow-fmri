# SPESIFIKASI FIGURE DAN TABEL UNTUK PUBLIKASI

## FIGURE YANG WAJIB DIBUAT

### Figure 1: Comprehensive CortexFlow Architecture Overview
**Tipe**: Multi-panel architectural diagram
**Ukuran**: Full page width (2-column spanning)
**Konten**:
- Panel A: Simple CortexFlow (encoder-decoder sederhana)
- Panel B: Monte Carlo Simple (dengan MC dropout sampling)
- Panel C: Hierarchical CortexFlow (multi-scale temporal processing)
- Panel D: Enhanced CortexFlow (hierarchical + uncertainty + alignment)
- Panel E: Unified CortexFlow (adaptive complexity mechanism)

**Detail Teknis**:
- Gunakan consistent color coding untuk komponen serupa
- Tunjukkan dimensi input/output pada setiap stage
- Highlight novel components dengan border/shading khusus
- Include legend untuk symbols dan notations

### Figure 2: Unified CortexFlow Detailed Architecture
**Tipe**: Detailed technical diagram
**Ukuran**: Full page width
**Konten**:
- Adaptive Complexity Module dengan complexity predictor
- Dual pathways (simple vs complex)
- Feature fusion mechanism dengan attention
- Dynamic routing berdasarkan complexity score
- Uncertainty estimation components

**Detail Teknis**:
- Flow arrows menunjukkan decision process
- Mathematical notations untuk key operations
- Color coding untuk different complexity levels
- Detailed component specifications

### Figure 3: Training Protocol Flowchart
**Tipe**: Process flowchart
**Ukuran**: Single column
**Konten**:
- Data preprocessing steps
- Training loop dengan validation
- Early stopping mechanism
- Model checkpointing
- Evaluation pipeline

**Detail Teknis**:
- Decision diamonds untuk conditional steps
- Clear start/end points
- Parallel processes untuk different architectures
- Time estimates untuk each major step

### Figure 4: Uncertainty Quantification Mechanism
**Tipe**: Conceptual diagram dengan mathematical illustration
**Ukuran**: Single column
**Konten**:
- Monte Carlo dropout sampling process
- Epistemic vs aleatoric uncertainty decomposition
- Uncertainty aggregation methods
- Calibration process

**Detail Teknis**:
- Mathematical formulas untuk uncertainty calculation
- Visual representation of sampling distributions
- Comparison plots showing different uncertainty types
- Calibration curves

## TABEL YANG SUDAH DIINTEGRASIKAN

### Table 1: Dataset Characteristics ✅
- Sudah diintegrasikan dalam metode.md
- Menunjukkan diversity dataset: 2 fMRI asli + 2 EEG-to-fMRI via NT-ViT
- Highlight cross-modal generalization capability dengan novel translation method

### Table 2: Architecture Comparison ✅
- Sudah diintegrasikan dalam metode.md
- Highlight progressive complexity dan novel features

### Table 3: Hyperparameter Configuration ✅
- Sudah diintegrasikan dalam metode.md
- Memastikan reproducibility

## TABEL TAMBAHAN YANG DIREKOMENDASIKAN

### Table 4: Performance Results Summary
**Lokasi**: Bagian Results (bukan Metode)
**Konten**:
- Test loss untuk semua architecture-dataset combinations
- Training time dan convergence epochs
- Uncertainty metrics (untuk applicable architectures)
- Statistical significance indicators

### Table 5: Computational Complexity Analysis
**Lokasi**: Bisa di Metode atau Results
**Konten**:
- Parameter counts
- FLOPs estimation
- Memory requirements
- Training/inference time
- Scalability analysis

## GUIDELINES UNTUK PEMBUATAN FIGURE

### Style Guidelines:
- **Font**: Arial atau Helvetica, minimum 8pt
- **Colors**: Colorblind-friendly palette
- **Resolution**: Minimum 300 DPI untuk print
- **Format**: Vector format (SVG/EPS) preferred

### Technical Requirements:
- **Consistency**: Same style across all figures
- **Clarity**: All text must be readable at publication size
- **Completeness**: Include all necessary labels dan legends
- **Accuracy**: Technical details must match implementation

### Content Guidelines:
- **Novelty Emphasis**: Highlight novel components clearly
- **Comparison**: Show clear differences between architectures
- **Flow**: Logical information flow in diagrams
- **Completeness**: Cover all major methodological aspects

## PRIORITAS PEMBUATAN

### Priority 1 (WAJIB):
1. Figure 1: Architecture Overview
2. Figure 2: Unified CortexFlow Detail
3. Algorithm 1: Adaptive Complexity Selection ✅
4. Algorithm 2: Monte Carlo Uncertainty Estimation ✅
5. Table 4: Performance Results (untuk Results section)

### Priority 2 (SANGAT DIREKOMENDASIKAN):
6. Figure 3: Training Protocol
7. Figure 4: Uncertainty Mechanism
8. Algorithm 3: NT-ViT Robust Training ✅
9. Table 5: Computational Analysis

### Priority 3 (OPSIONAL):
7. Figure 5: NT-ViT Cross-Modal Translation Pipeline
8. Supplementary figures untuk ablation studies
9. Additional tables untuk detailed hyperparameter sensitivity

### Figure 5: NT-ViT Cross-Modal Translation Pipeline (OPSIONAL)
**Tipe**: Technical pipeline diagram
**Ukuran**: Single column
**Konten**:
- Robust EEG preprocessing (outlier detection, MAD normalization)
- Spectrogram conversion dengan STFT
- NT-ViT encoder architecture dengan attention mechanism
- Domain matcher dengan conservative adversarial training
- Output fMRI representation validation

**Detail Teknis**:
- Dimension annotations pada setiap stage (N,C,T) → (N,3092)
- Mathematical formulations untuk MAD normalization dan clipping
- Robust training components (gradient clipping, loss monitoring)
- Validation metrics dan correlation analysis
- Comparison dengan ground truth fMRI

## TOOLS YANG DIREKOMENDASIKAN

### Untuk Architectural Diagrams:
- **Draw.io** (gratis, web-based)
- **Lucidchart** (professional)
- **Adobe Illustrator** (advanced)
- **TikZ/LaTeX** (untuk mathematical precision)

### Untuk Flowcharts:
- **Microsoft Visio**
- **Draw.io**
- **Graphviz** (untuk automated layouts)

### Untuk Mathematical Diagrams:
- **TikZ/LaTeX** (recommended)
- **Matplotlib** (untuk plots dengan annotations)
- **Inkscape** (vector graphics)

## CHECKLIST SEBELUM SUBMISSION

### Figure Quality:
- [ ] Resolusi minimum 300 DPI
- [ ] Text readable pada ukuran publikasi
- [ ] Colors accessible untuk colorblind readers
- [ ] Consistent styling across all figures

### Content Completeness:
- [ ] Semua novel components highlighted
- [ ] Mathematical notations consistent
- [ ] Legends dan captions complete
- [ ] Technical accuracy verified

### Integration:
- [ ] Figures referenced properly dalam text
- [ ] Captions explain key points
- [ ] Numbering consistent
- [ ] Placement optimal untuk readability
