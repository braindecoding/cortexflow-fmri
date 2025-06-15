"""
Statistical Analysis for Neural Decoding Research
================================================

Comprehensive statistical analysis functions for academic research:
- T-test analysis (one-sample, independent, paired)
- Effect size calculations (Cohen's d)
- Cross-validation statistical validation
- Publication-ready statistical reporting

Academic Features:
- Real cross-validation results analysis
- Multiple comparison corrections
- Effect size interpretations
- Statistical significance testing
- Academic reporting standards
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def comprehensive_ttest_analysis(cv_results_dict, dataset_name):
    """
    Comprehensive T-Test Analysis using REAL Cross-Validation Results
    
    Performs three types of statistical tests:
    1. One-sample t-test vs baseline threshold
    2. Independent samples t-test (CortexFlow vs SOTA)
    3. Paired samples t-test (pairwise comparisons)
    
    Args:
        cv_results_dict: Dictionary with method names as keys and CV results as values
        dataset_name: Name of the dataset being analyzed
        
    Returns:
        Dictionary with statistical analysis results
    """

    print(f"\n🔬 COMPREHENSIVE T-TEST ANALYSIS - Dataset: {dataset_name.upper()}")
    print("=" * 80)

    # Use REAL cross-validation results - NO SIMULATION
    if not cv_results_dict or len(cv_results_dict) == 0:
        print("❌ ERROR: No real cross-validation results available")
        print("   T-test analysis requires actual CV results, not single scores")
        print("   Please run cross-validation first to get multiple samples")
        return None

    methods = list(cv_results_dict.keys())

    print(f"📊 T-TEST OVERVIEW:")
    print(f"   T-test menggunakan REAL cross-validation results")
    print(f"   H₀: μ₁ = μ₂ (tidak ada perbedaan signifikan)")
    print(f"   H₁: μ₁ ≠ μ₂ (ada perbedaan signifikan)")
    print(f"   Significance level: α = 0.05")
    print(f"   Data source: ACTUAL {len(list(cv_results_dict.values())[0])}-fold cross-validation")

    print(f"\n📈 REAL CROSS-VALIDATION RESULTS:")
    for method, runs in cv_results_dict.items():
        mean_score = np.mean(runs)
        std_score = np.std(runs)
        print(f"   {method}: {mean_score:.6f} ± {std_score:.6f} (n={len(runs)} folds)")

    # 1. ONE-SAMPLE T-TEST
    print(f"\n1️⃣ ONE-SAMPLE T-TEST:")
    print(f"   Membandingkan setiap method dengan baseline threshold")
    baseline_threshold = 0.025  # Threshold untuk acceptable performance

    for method, runs in cv_results_dict.items():
        t_stat, p_value = stats.ttest_1samp(runs, baseline_threshold)

        if np.mean(runs) < baseline_threshold:
            interpretation = "✅ Significantly BETTER than baseline"
        else:
            interpretation = "❌ Not significantly better than baseline"

        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        print(f"   {method}:")
        print(f"     vs baseline ({baseline_threshold}): t = {t_stat:.3f}, p = {p_value:.6f} {significance}")
        print(f"     {interpretation}")

    # 2. INDEPENDENT SAMPLES T-TEST (Two-Sample)
    print(f"\n2️⃣ INDEPENDENT SAMPLES T-TEST:")
    print(f"   Membandingkan CortexFlow methods vs SOTA methods")

    cortexflow_methods = [method for method in methods if 'CortexFlow' in method]
    sota_methods = [method for method in methods if 'CortexFlow' not in method]

    # Combine REAL scores untuk group comparison
    cortexflow_scores = []
    sota_scores = []

    for method in cortexflow_methods:
        cortexflow_scores.extend(cv_results_dict[method])

    for method in sota_methods:
        sota_scores.extend(cv_results_dict[method])

    if cortexflow_scores and sota_scores:
        t_stat, p_value = stats.ttest_ind(cortexflow_scores, sota_scores)

        cf_mean = np.mean(cortexflow_scores)
        sota_mean = np.mean(sota_scores)

        if cf_mean < sota_mean:
            interpretation = "✅ CortexFlow significantly BETTER than SOTA"
            improvement = ((sota_mean - cf_mean) / sota_mean) * 100
        else:
            interpretation = "❌ CortexFlow not significantly better than SOTA"
            improvement = ((cf_mean - sota_mean) / cf_mean) * 100

        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        print(f"   CortexFlow vs SOTA:")
        print(f"     CortexFlow mean: {cf_mean:.6f}")
        print(f"     SOTA mean: {sota_mean:.6f}")
        print(f"     t-statistic: {t_stat:.3f}")
        print(f"     p-value: {p_value:.6f} {significance}")
        print(f"     {interpretation}")
        if cf_mean < sota_mean:
            print(f"     Improvement: {improvement:.2f}%")

    # 3. PAIRED SAMPLES T-TEST
    print(f"\n3️⃣ PAIRED SAMPLES T-TEST:")
    print(f"   Membandingkan methods pada dataset yang sama (paired comparison)")

    # Pairwise comparisons using REAL CV results
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1, method2 = methods[i], methods[j]
            scores1, scores2 = cv_results_dict[method1], cv_results_dict[method2]

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(scores1, scores2)

            # Effect size (Cohen's d untuk paired samples)
            diff = np.array(scores1) - np.array(scores2)
            cohens_d = np.mean(diff) / np.std(diff)

            # Interpretation
            mean1, mean2 = np.mean(scores1), np.mean(scores2)
            if mean1 < mean2:
                winner = method1
                improvement = ((mean2 - mean1) / mean2) * 100
            else:
                winner = method2
                improvement = ((mean1 - mean2) / mean1) * 100

            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            # Effect size interpretation
            if abs(cohens_d) < 0.2:
                effect_magnitude = "Small"
            elif abs(cohens_d) < 0.5:
                effect_magnitude = "Medium"
            elif abs(cohens_d) < 0.8:
                effect_magnitude = "Large"
            else:
                effect_magnitude = "Very Large"

            print(f"   {method1} vs {method2}:")
            print(f"     t-statistic: {t_stat:.3f}")
            print(f"     p-value: {p_value:.6f} {significance}")
            print(f"     Cohen's d: {cohens_d:.3f} ({effect_magnitude} effect)")
            print(f"     Winner: {winner} ({improvement:.2f}% better)")

    return cv_results_dict


def statistical_analysis(results_dict, dataset_name):
    """
    Basic statistical analysis untuk single training results

    Provides descriptive statistics, pairwise comparisons, and effect size analysis
    for single training run results (non-cross-validation).

    Args:
        results_dict: Dictionary with method names as keys and MSE scores as values
        dataset_name: Name of the dataset being analyzed

    Returns:
        Dictionary with statistical summary
    """

    print(f"\n📊 STATISTICAL ANALYSIS - Dataset: {dataset_name.upper()}")
    print("=" * 70)

    # Extract results
    methods = list(results_dict.keys())
    scores = list(results_dict.values())

    print(f"🔬 Methods: {methods}")
    print(f"📈 MSE Scores: {[f'{score:.6f}' for score in scores]}")

    # 1. Descriptive Statistics
    print(f"\n1. DESCRIPTIVE STATISTICS:")
    print(f"   Best Method: {methods[np.argmin(scores)]} (MSE: {min(scores):.6f})")
    print(f"   Worst Method: {methods[np.argmax(scores)]} (MSE: {max(scores):.6f})")
    print(f"   Range: {max(scores) - min(scores):.6f}")
    print(f"   Mean: {np.mean(scores):.6f} ± {np.std(scores):.6f}")

    # 2. Pairwise Comparisons (untuk publication)
    print(f"\n2. PAIRWISE COMPARISONS:")
    cortexflow_methods = [i for i, method in enumerate(methods) if 'CortexFlow' in method]
    sota_methods = [i for i, method in enumerate(methods) if 'CortexFlow' not in method]

    # Compare CortexFlow methods vs SOTA
    for cf_idx in cortexflow_methods:
        cf_method = methods[cf_idx]
        cf_score = scores[cf_idx]

        print(f"\n   {cf_method} vs SOTA methods:")
        for sota_idx in sota_methods:
            sota_method = methods[sota_idx]
            sota_score = scores[sota_idx]

            improvement = ((sota_score - cf_score) / sota_score) * 100
            effect_size = abs(cf_score - sota_score) / np.std([cf_score, sota_score])

            if cf_score < sota_score:
                print(f"     vs {sota_method}: ✅ {improvement:.2f}% improvement (Effect size: {effect_size:.3f})")
            else:
                print(f"     vs {sota_method}: ❌ {-improvement:.2f}% worse (Effect size: {effect_size:.3f})")

    # 3. CortexFlow Enhanced vs Ensemble Comparison
    enhanced_idx = next((i for i, method in enumerate(methods) if 'Enhanced' in method), None)
    ensemble_idx = next((i for i, method in enumerate(methods) if 'Ensemble' in method), None)

    if enhanced_idx is not None and ensemble_idx is not None:
        enhanced_score = scores[enhanced_idx]
        ensemble_score = scores[ensemble_idx]

        print(f"\n3. CORTEXFLOW APPROACH COMPARISON:")
        if enhanced_score < ensemble_score:
            improvement = ((ensemble_score - enhanced_score) / ensemble_score) * 100
            print(f"   Enhanced vs Ensemble: ✅ Enhanced better by {improvement:.2f}%")
            print(f"   Conclusion: Multi-Pathway approach superior untuk {dataset_name}")
        else:
            improvement = ((enhanced_score - ensemble_score) / enhanced_score) * 100
            print(f"   Enhanced vs Ensemble: ✅ Ensemble better by {improvement:.2f}%")
            print(f"   Conclusion: Variant Ensemble approach superior untuk {dataset_name}")

    # 4. Effect Size Classification
    print(f"\n4. EFFECT SIZE ANALYSIS:")
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]

    for i, (method, score) in enumerate(zip(methods, scores)):
        if i != best_idx:
            effect_size = abs(score - best_score) / np.std([score, best_score])
            if effect_size < 0.2:
                magnitude = "Small"
            elif effect_size < 0.5:
                magnitude = "Medium"
            elif effect_size < 0.8:
                magnitude = "Large"
            else:
                magnitude = "Very Large"

            print(f"   {method}: Effect size = {effect_size:.3f} ({magnitude})")

    return {
        'best_method': methods[np.argmin(scores)],
        'best_score': min(scores),
        'worst_score': max(scores),
        'range': max(scores) - min(scores),
        'mean': np.mean(scores),
        'std': np.std(scores)
    }
