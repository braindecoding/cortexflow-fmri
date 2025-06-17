# Individual Model Consistency Hypothesis Testing

**Generated:** 2025-06-17 19:30:08  
**Analysis Type:** Individual Model Consistency Testing  
**Methodology:** Multi-Criteria Consistency Assessment  

---

## 📋 Research Hypotheses

**H1:** Model MinD_Vis memiliki performa yang konsisten untuk semua dataset  
**H2:** Model CortexFlow_Lite memiliki performa yang konsisten untuk semua dataset  
**H3:** Model CortexFlow_Multi-Pathway memiliki performa yang konsisten untuk semua dataset  
**H4:** Model CortexFlow_Ensemble memiliki performa yang konsisten untuk semua dataset  
**H5:** Model Brain_Diffuser memiliki performa yang konsisten untuk semua dataset  

---

## 🔬 Statistical Testing Methodology

### Consistency Criteria:
1. **CV Coefficient Test**: CV < 0.3 (consistent variation)
2. **T-Test**: p > 0.05 (not significantly different from overall mean)
3. **Ranking Consistency**: Std < 1.5 (stable rankings)

**Decision Rule**: Hypothesis supported if ≥2/3 criteria met

---

## 📊 Individual Model Results

### Summary Table

| Hypothesis | Model | CV Coeff | Ranking Std | Consistency Score | Result |
|------------|-------|----------|-------------|-------------------|--------|
| H1 | MinD_Vis | 0.5715 | 0.50 | 2/3 | ✅ SUPPORTED |
| H2 | CortexFlow_Lite | 0.5306 | 0.50 | 2/3 | ✅ SUPPORTED |
| H3 | CortexFlow_Multi-Pathway | 0.2608 | 2.31 | 1/3 | ❌ REJECTED |
| H4 | CortexFlow_Ensemble | 0.4993 | 1.29 | 2/3 | ✅ SUPPORTED |
| H5 | Brain_Diffuser | 0.5851 | 1.91 | 1/3 | ❌ REJECTED |

### Detailed Results

#### H1: MinD_Vis

**Performance Across Datasets:**
- Datasets: MIYAWAKI, VANGERVEN, MINDBIGDATA, CRELL
- MSE Scores: ['0.009484', '0.048767', '0.058001', '0.032564']
- Rankings: [2, 2, 2, 3]
- Mean MSE: 0.037204 ± 0.021261
- Mean Ranking: 2.25 ± 0.50

**Consistency Tests:**
- CV Coefficient: 0.5715 (❌ Fail)
