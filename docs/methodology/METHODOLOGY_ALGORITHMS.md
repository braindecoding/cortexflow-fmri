# Algoritma Metodologi CortexFlow: Hypothesis-Driven Framework

## Gambaran Umum

Dokumen ini menyajikan 7 algoritma kunci yang digunakan dalam metodologi penelitian CortexFlow berbasis hypothesis-driven approach. Setiap algoritma dijelaskan dengan pseudocode yang detail dan implementasi yang dapat direproduksi untuk pengujian 7 hipotesis penelitian spesifik.

---

## Algoritma 1: Multi-Criteria Consistency Assessment (H1-H5)

### Deskripsi
Algoritma multi-criteria consistency assessment untuk menguji hipotesis konsistensi individual model (H1-H5) menggunakan 3 kriteria statistik dengan decision rule majority voting.

### Pseudocode
```
Algorithm 1: Multi-Criteria Consistency Assessment (H1-H5)
Input: CV_results, model_name, hypothesis_number
Output: Hypothesis_result = {supported: bool, consistency_score: int, details: dict}

1. INITIALIZE:
   - datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
   - consistency_threshold = {cv_coeff: 0.3, ranking_std: 1.5, t_test_alpha: 0.05}
   - criteria_results = []

2. EXTRACT MODEL PERFORMANCE:
   a. model_scores = []
   b. model_rankings = []
   c. FOR each dataset in datasets:
      i. score = CV_results[model_name][dataset]['mse']
      ii. ranking = get_ranking(CV_results, dataset, model_name)
      iii. model_scores.append(score)
      iv. model_rankings.append(ranking)
   d. END FOR

3. CRITERION 1: CV COEFFICIENT TEST
   a. mean_score = mean(model_scores)
   b. std_score = std(model_scores)
   c. cv_coefficient = std_score / mean_score
   d. cv_consistent = cv_coefficient < consistency_threshold['cv_coeff']
   e. criteria_results.append(cv_consistent)

4. CRITERION 2: ONE-SAMPLE T-TEST
   a. overall_mean = calculate_overall_mean(CV_results, datasets)
   b. t_statistic, p_value = ttest_1samp(model_scores, overall_mean)
   c. t_consistent = p_value > consistency_threshold['t_test_alpha']
   d. criteria_results.append(t_consistent)

5. CRITERION 3: RANKING CONSISTENCY TEST
   a. ranking_std = std(model_rankings)
   b. ranking_consistent = ranking_std < consistency_threshold['ranking_std']
   c. criteria_results.append(ranking_consistent)

6. DECISION RULE (MAJORITY VOTING):
   a. consistency_score = sum(criteria_results)  # Count of passed criteria
   b. hypothesis_supported = consistency_score >= 2  # ≥2/3 criteria

7. COMPILE RESULTS:
   a. result = {
      'hypothesis': f'H{hypothesis_number}',
      'model': model_name,
      'supported': hypothesis_supported,
      'consistency_score': f'{consistency_score}/3',
      'cv_coefficient': cv_coefficient,
      't_test_pvalue': p_value,
      'ranking_std': ranking_std,
      'criteria_details': {
         'cv_test': cv_consistent,
         't_test': t_consistent,
         'ranking_test': ranking_consistent
      }
   }

8. RETURN result
```

### Kompleksitas
- **Time Complexity**: O(n × d) dimana n=models, d=datasets
- **Space Complexity**: O(d) untuk storing scores per dataset
- **Statistical Power**: Multi-criteria approach mengurangi false positives/negatives

---

## Algoritma 2: Architecture-Dataset Complexity Correlation Analysis (H6)

### Deskripsi
Algoritma untuk menguji hipotesis H6 tentang korelasi antara kompleksitas arsitektur dan kompleksitas dataset menggunakan analisis korelasi Pearson.

### Pseudocode
```
Algorithm 2: Architecture-Dataset Complexity Correlation Analysis (H6)
Input: CV_results, architecture_specs, dataset_specs
Output: Correlation_result = {correlation_coeff: float, p_value: float, supported: bool}

1. INITIALIZE:
   - architectures = ['CortexFlow_Lite', 'MinD_Vis', 'Brain_Diffuser',
                     'CortexFlow_Multi-Pathway', 'CortexFlow_Ensemble']
   - datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']

2. DEFINE COMPLEXITY METRICS:
   a. architecture_complexity = {
      'CortexFlow_Lite': 1.0,           # Baseline complexity
      'MinD_Vis': 1.2,                  # Slightly more complex
      'Brain_Diffuser': 1.3,            # Diffusion complexity
      'CortexFlow_Multi-Pathway': 1.8,  # Multi-pathway complexity
      'CortexFlow_Ensemble': 2.5        # Highest complexity
   }

   b. dataset_complexity = {
      'miyawaki': 1.0,      # Single-modal, binary patterns
      'vangerven': 1.2,     # Single-modal, grayscale
      'mindbigdata': 1.8,   # Cross-modal, complex
      'crell': 1.6          # Cross-modal, moderate
   }

3. EXTRACT PERFORMANCE DATA:
   a. performance_matrix = []
   b. architecture_values = []
   c. dataset_values = []

   d. FOR each architecture in architectures:
      i. FOR each dataset in datasets:
         ii. performance = CV_results[architecture][dataset]['mse']
         iii. arch_complexity = architecture_complexity[architecture]
         iv. data_complexity = dataset_complexity[dataset]
         v. performance_matrix.append((arch_complexity, data_complexity, performance))
         vi. architecture_values.append(arch_complexity)
         vii. dataset_values.append(data_complexity)
      viii. END FOR
   e. END FOR

4. CORRELATION ANALYSIS:
   a. # Test H6: Positive correlation between architecture and dataset complexity
   b. correlation_coeff, p_value = pearsonr(architecture_values, dataset_values)

5. HYPOTHESIS TESTING:
   a. alpha = 0.05
   b. hypothesis_supported = (correlation_coeff > 0) AND (p_value < alpha)

6. ADDITIONAL ANALYSIS:
   a. # Best performer analysis by complexity matching
   b. best_performers = {}
   c. FOR each dataset in datasets:
      i. best_arch = find_best_performer(CV_results, dataset)
      ii. best_performers[dataset] = {
         'architecture': best_arch,
         'arch_complexity': architecture_complexity[best_arch],
         'data_complexity': dataset_complexity[dataset],
         'performance': CV_results[best_arch][dataset]['mse']
      }
   d. END FOR

7. COMPILE RESULTS:
   a. result = {
      'hypothesis': 'H6',
      'correlation_coefficient': correlation_coeff,
      'p_value': p_value,
      'supported': hypothesis_supported,
      'interpretation': get_correlation_interpretation(correlation_coeff, p_value),
      'best_performers': best_performers,
      'complexity_matching_evidence': analyze_complexity_matching(best_performers)
   }

8. RETURN result
```

### Kompleksitas
- **Time Complexity**: O(n × d) dimana n=architectures, d=datasets
- **Space Complexity**: O(n × d) untuk performance matrix
- **Statistical Power**: Pearson correlation dengan significance testing

---

## Algoritma 3: Modality Specialization Testing (H7)

### Deskripsi
Algoritma untuk menguji hipotesis H7 tentang spesialisasi model berdasarkan modalitas data (single-modal vs cross-modal) menggunakan independent samples t-test.

### Pseudocode
```
Algorithm 3: Modality Specialization Testing (H7)
Input: CV_results, modality_classification
Output: Specialization_result = {t_statistic: float, p_value: float, supported: bool, patterns: dict}

1. INITIALIZE:
   - architectures = ['CortexFlow_Lite', 'MinD_Vis', 'Brain_Diffuser',
                     'CortexFlow_Multi-Pathway', 'CortexFlow_Ensemble']
   - single_modal_datasets = ['miyawaki', 'vangerven']  # fMRI → Visual direct
   - cross_modal_datasets = ['mindbigdata', 'crell']   # EEG → fMRI → Visual

2. CLASSIFY DATASET MODALITIES:
   a. modality_types = {
      'miyawaki': 'single_modal',     # fMRI visual cortex only
      'vangerven': 'single_modal',    # fMRI digit recognition
      'mindbigdata': 'cross_modal',   # EEG → fMRI → Visual
      'crell': 'cross_modal'          # EEG → fMRI → Visual
   }

3. EXTRACT PERFORMANCE BY MODALITY:
   a. specialization_patterns = {}
   b. overall_single_modal = []
   c. overall_cross_modal = []

   d. FOR each architecture in architectures:
      i. single_modal_scores = []
      ii. cross_modal_scores = []

      iii. FOR each dataset in single_modal_datasets:
         iv. score = CV_results[architecture][dataset]['mse']
         v. single_modal_scores.append(score)
         vi. overall_single_modal.append(score)
      vii. END FOR

      viii. FOR each dataset in cross_modal_datasets:
         ix. score = CV_results[architecture][dataset]['mse']
         x. cross_modal_scores.append(score)
         xi. overall_cross_modal.append(score)
      xii. END FOR

      # Calculate specialization metrics
      xiii. single_modal_mean = mean(single_modal_scores)
      xiv. cross_modal_mean = mean(cross_modal_scores)
      xv. specialization_ratio = single_modal_mean / cross_modal_mean

      xvi. specialization_patterns[architecture] = {
         'single_modal_performance': single_modal_mean,
         'cross_modal_performance': cross_modal_mean,
         'specialization_ratio': specialization_ratio,
         'specialization_type': determine_specialization_type(specialization_ratio)
      }
   e. END FOR

4. STATISTICAL TESTING:
   a. # Independent samples t-test for overall modality difference
   b. t_statistic, p_value = ttest_ind(overall_single_modal, overall_cross_modal)
   c. alpha = 0.05
   d. hypothesis_supported = p_value < alpha

5. RANKING ANALYSIS:
   a. ranking_patterns = {}
   b. FOR each architecture in architectures:
      i. single_modal_ranks = []
      ii. cross_modal_ranks = []

      iii. FOR each dataset in single_modal_datasets:
         iv. rank = get_ranking(CV_results, dataset, architecture)
         v. single_modal_ranks.append(rank)
      vi. END FOR

      vii. FOR each dataset in cross_modal_datasets:
         viii. rank = get_ranking(CV_results, dataset, architecture)
         ix. cross_modal_ranks.append(rank)
      x. END FOR

      xi. ranking_patterns[architecture] = {
         'single_modal_avg_rank': mean(single_modal_ranks),
         'cross_modal_avg_rank': mean(cross_modal_ranks),
         'rank_difference': mean(single_modal_ranks) - mean(cross_modal_ranks)
      }
   c. END FOR

6. SPECIALIZATION CLASSIFICATION:
   a. FOR each architecture in architectures:
      i. single_rank = ranking_patterns[architecture]['single_modal_avg_rank']
      ii. cross_rank = ranking_patterns[architecture]['cross_modal_avg_rank']

      iii. IF single_rank < cross_rank - 1.0:
         specialization = 'Single-Modal Specialist'
      iv. ELIF cross_rank < single_rank - 1.0:
         specialization = 'Cross-Modal Specialist'
      v. ELSE:
         specialization = 'Balanced'
      vi. END IF

      vii. ranking_patterns[architecture]['specialization'] = specialization
   b. END FOR

7. COMPILE RESULTS:
   a. result = {
      'hypothesis': 'H7',
      't_statistic': t_statistic,
      'p_value': p_value,
      'supported': hypothesis_supported,
      'overall_single_modal_mean': mean(overall_single_modal),
      'overall_cross_modal_mean': mean(overall_cross_modal),
      'specialization_patterns': specialization_patterns,
      'ranking_patterns': ranking_patterns,
      'interpretation': generate_specialization_interpretation(ranking_patterns)
   }

8. RETURN result
```

### Kompleksitas
- **Time Complexity**: O(n × d) dimana n=architectures, d=datasets
- **Space Complexity**: O(n × d) untuk performance matrices
- **Statistical Power**: Independent samples t-test dengan ranking analysis

---

## Algoritma 4: Comprehensive Hypothesis Testing Framework

### Deskripsi
Algoritma master untuk menjalankan comprehensive hypothesis testing framework yang mengintegrasikan semua 7 hipotesis penelitian dengan koordinasi dan validasi hasil.

### Pseudocode
```
Algorithm 4: Comprehensive Hypothesis Testing Framework
Input: CV_results, configuration_parameters
Output: Complete_hypothesis_results = {H1-H7: results, summary: analysis}

1. INITIALIZE:
   - hypotheses = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']
   - models = ['MinD_Vis', 'CortexFlow_Lite', 'CortexFlow_Multi-Pathway',
              'CortexFlow_Ensemble', 'Brain_Diffuser']
   - results = {}

2. INDIVIDUAL MODEL CONSISTENCY TESTING (H1-H5):
   a. FOR i = 1 to 5:
      i. model_name = models[i-1]
      ii. hypothesis_num = i
      iii. result = Algorithm_1(CV_results, model_name, hypothesis_num)
      iv. results[f'H{i}'] = result
      v. log_hypothesis_result(f'H{i}', result)
   b. END FOR

3. ARCHITECTURE-DATASET COMPLEXITY ANALYSIS (H6):
   a. h6_result = Algorithm_2(CV_results, architecture_specs, dataset_specs)
   b. results['H6'] = h6_result
   c. log_hypothesis_result('H6', h6_result)

4. MODALITY SPECIALIZATION TESTING (H7):
   a. h7_result = Algorithm_3(CV_results, modality_classification)
   b. results['H7'] = h7_result
   c. log_hypothesis_result('H7', h7_result)

5. CROSS-HYPOTHESIS VALIDATION:
   a. # Validate consistency between related hypotheses
   b. consistency_check = validate_hypothesis_consistency(results)
   c. IF consistency_check.has_conflicts:
      i. flag_conflicts = consistency_check.conflicts
      ii. recommend_further_analysis(flag_conflicts)
   d. END IF

6. STATISTICAL POWER ANALYSIS:
   a. FOR each hypothesis in results:
      i. power_analysis = calculate_statistical_power(results[hypothesis])
      ii. results[hypothesis]['statistical_power'] = power_analysis
   b. END FOR

7. EFFECT SIZE AGGREGATION:
   a. effect_sizes = {}
   b. FOR each hypothesis in results:
      i. IF results[hypothesis].has_effect_size:
         ii. effect_sizes[hypothesis] = results[hypothesis]['effect_size']
      iii. END IF
   c. END FOR

8. COMPREHENSIVE SUMMARY GENERATION:
   a. summary = {
      'total_hypotheses': len(hypotheses),
      'supported_hypotheses': count_supported(results),
      'rejected_hypotheses': count_rejected(results),
      'support_rate': calculate_support_rate(results),
      'key_findings': extract_key_findings(results),
      'research_implications': generate_implications(results),
      'statistical_rigor': assess_statistical_rigor(results)
   }

9. RESEARCH DOCUMENTATION:
   a. generate_hypothesis_report(results, summary)
   b. create_statistical_tables(results)
   c. generate_visualization_data(results)
   d. compile_academic_documentation(results, summary)

10. VALIDATION AND QUALITY ASSURANCE:
    a. validate_statistical_assumptions(results)
    b. check_reproducibility_requirements(results)
    c. verify_academic_standards(results)

11. RETURN {
    'hypothesis_results': results,
    'comprehensive_summary': summary,
    'documentation_generated': True,
    'quality_validated': True
}
```

### Kompleksitas
- **Time Complexity**: O(H × M × D) dimana H=hypotheses, M=models, D=datasets
- **Space Complexity**: O(H × R) dimana R=results per hypothesis
- **Coordination**: Master algorithm untuk comprehensive testing framework

---

## Algoritma 5: Statistical Decision Rule Implementation

### Deskripsi
Algoritma untuk implementasi decision rule yang konsisten dan robust dalam pengambilan keputusan hipotesis berdasarkan multiple criteria dan statistical thresholds.

### Pseudocode
```
Algorithm 5: Statistical Decision Rule Implementation
Input: Statistical_test_results, Criteria_thresholds, Decision_parameters
Output: Decision_result = {decision: bool, confidence: float, reasoning: str}

1. INITIALIZE:
   - decision_criteria = {
      'cv_coefficient_threshold': 0.3,
      'ranking_std_threshold': 1.5,
      't_test_alpha': 0.05,
      'correlation_alpha': 0.05,
      'majority_rule_threshold': 0.67  # ≥2/3 criteria
   }

2. MULTI-CRITERIA DECISION RULE (H1-H5):
   a. FUNCTION apply_multi_criteria_rule(test_results):
      i. criteria_passed = []

      # Criterion 1: CV Coefficient
      ii. cv_pass = test_results['cv_coefficient'] < decision_criteria['cv_coefficient_threshold']
      iii. criteria_passed.append(cv_pass)

      # Criterion 2: T-Test
      iv. t_test_pass = test_results['t_test_pvalue'] > decision_criteria['t_test_alpha']
      v. criteria_passed.append(t_test_pass)

      # Criterion 3: Ranking Consistency
      vi. ranking_pass = test_results['ranking_std'] < decision_criteria['ranking_std_threshold']
      vii. criteria_passed.append(ranking_pass)

      # Apply majority rule
      viii. criteria_score = sum(criteria_passed) / len(criteria_passed)
      ix. decision = criteria_score >= decision_criteria['majority_rule_threshold']

      x. RETURN {
         'decision': decision,
         'criteria_score': criteria_score,
         'individual_criteria': criteria_passed,
         'confidence': calculate_confidence(criteria_score)
      }

3. CORRELATION DECISION RULE (H6):
   a. FUNCTION apply_correlation_rule(correlation_results):
      i. correlation_coeff = correlation_results['correlation_coefficient']
      ii. p_value = correlation_results['p_value']

      # H6: Positive correlation expected
      iii. positive_correlation = correlation_coeff > 0
      iv. statistically_significant = p_value < decision_criteria['correlation_alpha']

      v. decision = positive_correlation AND statistically_significant
      vi. confidence = calculate_correlation_confidence(correlation_coeff, p_value)

      vii. RETURN {
         'decision': decision,
         'correlation_coefficient': correlation_coeff,
         'p_value': p_value,
         'confidence': confidence,
         'reasoning': generate_correlation_reasoning(correlation_coeff, p_value)
      }

4. SPECIALIZATION DECISION RULE (H7):
   a. FUNCTION apply_specialization_rule(specialization_results):
      i. t_statistic = specialization_results['t_statistic']
      ii. p_value = specialization_results['p_value']

      # H7: Significant difference between modalities expected
      iii. statistically_significant = p_value < decision_criteria['correlation_alpha']

      # Additional evidence from ranking patterns
      iv. ranking_evidence = analyze_ranking_patterns(specialization_results['ranking_patterns'])
      v. strong_specialization_evidence = ranking_evidence['strong_patterns'] > 0

      vi. decision = statistically_significant OR strong_specialization_evidence
      vii. confidence = calculate_specialization_confidence(p_value, ranking_evidence)

      viii. RETURN {
         'decision': decision,
         't_statistic': t_statistic,
         'p_value': p_value,
         'ranking_evidence': ranking_evidence,
         'confidence': confidence,
         'reasoning': generate_specialization_reasoning(p_value, ranking_evidence)
      }

5. CONFIDENCE CALCULATION:
   a. FUNCTION calculate_confidence(criteria_score):
      IF criteria_score >= 1.0: RETURN 'Very High'
      ELIF criteria_score >= 0.67: RETURN 'High'
      ELIF criteria_score >= 0.33: RETURN 'Moderate'
      ELSE: RETURN 'Low'

6. DECISION AGGREGATION:
   a. FUNCTION aggregate_decisions(all_hypothesis_results):
      i. total_hypotheses = len(all_hypothesis_results)
      ii. supported_count = count_supported_hypotheses(all_hypothesis_results)
      iii. support_rate = supported_count / total_hypotheses

      iv. overall_confidence = calculate_overall_confidence(all_hypothesis_results)
      v. research_strength = assess_research_strength(support_rate, overall_confidence)

      vi. RETURN {
         'total_hypotheses': total_hypotheses,
         'supported_hypotheses': supported_count,
         'support_rate': support_rate,
         'overall_confidence': overall_confidence,
         'research_strength': research_strength
      }

7. QUALITY ASSURANCE:
   a. validate_decision_consistency(all_decisions)
   b. check_statistical_assumptions(test_results)
   c. verify_threshold_appropriateness(decision_criteria)

8. RETURN Decision_result with comprehensive reasoning and confidence metrics
```

### Kompleksitas
- **Time Complexity**: O(H × C) dimana H=hypotheses, C=criteria per hypothesis
- **Space Complexity**: O(H) untuk storing decision results
- **Robustness**: Multi-criteria approach dengan confidence assessment

---

## Algoritma 6: Cross-Validation dengan Hypothesis Validation

### Deskripsi
Algoritma enhanced cross-validation yang terintegrasi dengan hypothesis validation untuk memastikan robust statistical testing dengan proper data splitting.

### Pseudocode
```
Algorithm 6: Cross-Validation dengan Hypothesis Validation
Input: Dataset, Models, Hypotheses_framework, k=5
Output: CV_results_with_hypothesis_validation

1. INITIALIZE:
   - Set random_seed = 42 for reproducibility
   - Create KFold(n_splits=5, shuffle=True, random_state=42)
   - Initialize hypothesis_validation_results = {}

2. STRATIFIED CV WITH HYPOTHESIS AWARENESS:
   a. FOR each fold i = 1 to k:
      i. Split data: (X_train, y_train), (X_val, y_val) = split(X, y, fold_i)

      # Ensure balanced representation for hypothesis testing
      ii. validate_fold_balance(X_train, X_val, y_train, y_val)

      iii. FOR each model in models:
         iv. Train model on (X_train, y_train)
         v. Evaluate on (X_val, y_val)
         vi. Store fold_results[model][fold_i] = evaluation_metrics
      vii. END FOR

      # Hypothesis-specific validation per fold
      viii. fold_hypothesis_results = validate_hypotheses_per_fold(fold_results, fold_i)
      ix. hypothesis_validation_results[fold_i] = fold_hypothesis_results
   b. END FOR

3. HYPOTHESIS CONSISTENCY ACROSS FOLDS:
   a. FOR each hypothesis in [H1, H2, H3, H4, H5, H6, H7]:
      i. fold_consistency = []
      ii. FOR each fold in range(k):
         iii. fold_result = hypothesis_validation_results[fold][hypothesis]
         iv. fold_consistency.append(fold_result['supported'])
      v. END FOR

      # Check hypothesis stability across folds
      vi. consistency_rate = sum(fold_consistency) / k
      vii. hypothesis_stability = assess_stability(consistency_rate)
   b. END FOR

4. RETURN CV_results with integrated hypothesis validation
```

### Kompleksitas
- **Time Complexity**: O(k × n × T × H) dimana H=hypotheses
- **Space Complexity**: O(k × n × H) untuk hypothesis results per fold
- **Validation**: Integrated hypothesis testing dengan CV robustness

---

## Algoritma 7: Effect Size dan Statistical Power Calculation

### Deskripsi
Algoritma untuk menghitung effect size (Cohen's d) dan statistical power untuk memastikan meaningful dan reliable hypothesis testing results.

### Pseudocode
```
Algorithm 7: Effect Size dan Statistical Power Calculation
Input: Statistical_test_results, Sample_sizes, Alpha_level=0.05
Output: Effect_size_analysis = {cohens_d: float, power: float, interpretation: str}

1. INITIALIZE:
   - effect_size_thresholds = {
      'small': 0.2, 'medium': 0.5, 'large': 0.8, 'very_large': 1.2
   }
   - power_thresholds = {
      'low': 0.5, 'adequate': 0.8, 'high': 0.9
   }

2. COHEN'S D CALCULATION:
   a. FUNCTION calculate_cohens_d(group1_scores, group2_scores):
      i. mean1 = mean(group1_scores)
      ii. mean2 = mean(group2_scores)
      iii. std1 = std(group1_scores, ddof=1)
      iv. std2 = std(group2_scores, ddof=1)

      # Pooled standard deviation
      v. n1, n2 = len(group1_scores), len(group2_scores)
      vi. pooled_std = sqrt(((n1-1)*std1² + (n2-1)*std2²) / (n1+n2-2))

      # Cohen's d
      vii. cohens_d = (mean1 - mean2) / pooled_std
      viii. RETURN cohens_d

3. EFFECT SIZE INTERPRETATION:
   a. FUNCTION interpret_effect_size(cohens_d):
      i. abs_d = abs(cohens_d)
      ii. IF abs_d < effect_size_thresholds['small']:
         RETURN 'Negligible'
      iii. ELIF abs_d < effect_size_thresholds['medium']:
         RETURN 'Small'
      iv. ELIF abs_d < effect_size_thresholds['large']:
         RETURN 'Medium'
      v. ELIF abs_d < effect_size_thresholds['very_large']:
         RETURN 'Large'
      vi. ELSE:
         RETURN 'Very Large'

4. STATISTICAL POWER CALCULATION:
   a. FUNCTION calculate_statistical_power(effect_size, sample_size, alpha=0.05):
      # Using power analysis for t-test
      i. degrees_freedom = sample_size - 1
      ii. noncentrality_param = effect_size * sqrt(sample_size)
      iii. critical_t = t_distribution_critical(degrees_freedom, alpha)
      iv. power = 1 - t_distribution_cdf(critical_t, degrees_freedom, noncentrality_param)
      v. RETURN power

5. POWER INTERPRETATION:
   a. FUNCTION interpret_power(power_value):
      i. IF power_value < power_thresholds['low']:
         RETURN 'Insufficient'
      ii. ELIF power_value < power_thresholds['adequate']:
         RETURN 'Low'
      iii. ELIF power_value < power_thresholds['high']:
         RETURN 'Adequate'
      iv. ELSE:
         RETURN 'High'

6. COMPREHENSIVE ANALYSIS:
   a. FOR each hypothesis_test in statistical_results:
      i. effect_size = calculate_cohens_d(test['group1'], test['group2'])
      ii. effect_interpretation = interpret_effect_size(effect_size)
      iii. statistical_power = calculate_statistical_power(effect_size, test['sample_size'])
      iv. power_interpretation = interpret_power(statistical_power)

      v. analysis_result = {
         'cohens_d': effect_size,
         'effect_size_interpretation': effect_interpretation,
         'statistical_power': statistical_power,
         'power_interpretation': power_interpretation,
         'practical_significance': assess_practical_significance(effect_size),
         'recommendation': generate_recommendation(effect_size, statistical_power)
      }
   b. END FOR

7. SAMPLE SIZE RECOMMENDATIONS:
   a. FOR each underpowered_test:
      i. required_n = calculate_required_sample_size(desired_power=0.8, effect_size, alpha)
      ii. current_n = underpowered_test['sample_size']
      iii. additional_n_needed = max(0, required_n - current_n)
      iv. recommendations.append({
         'test': underpowered_test['name'],
         'current_power': underpowered_test['power'],
         'required_n': required_n,
         'additional_n_needed': additional_n_needed
      })
   b. END FOR

8. RETURN Comprehensive effect size and power analysis with recommendations
```

### Kompleksitas
- **Time Complexity**: O(T × N) dimana T=tests, N=sample_size
- **Space Complexity**: O(T) untuk analysis results
- **Statistical Rigor**: Complete effect size dan power analysis

---

## Kesimpulan Algoritma Hypothesis-Driven Framework

Ketujuh algoritma ini membentuk foundation metodologis hypothesis-driven yang robust untuk penelitian CortexFlow:

### **🔬 Core Hypothesis Testing Algorithms:**
1. **Multi-Criteria Consistency Assessment (H1-H5)**: 3-kriteria statistical validation dengan majority rule
2. **Architecture-Dataset Complexity Analysis (H6)**: Pearson correlation dengan complexity metrics
3. **Modality Specialization Testing (H7)**: Independent samples t-test dengan ranking analysis

### **🎯 Framework Integration Algorithms:**
4. **Comprehensive Hypothesis Testing Framework**: Master coordinator untuk 7 hipotesis
5. **Statistical Decision Rule Implementation**: Robust decision making dengan confidence assessment
6. **Cross-Validation dengan Hypothesis Validation**: Integrated CV dengan hypothesis stability
7. **Effect Size dan Statistical Power Calculation**: Complete statistical rigor assessment

### **✅ Keunggulan Framework:**
- **Definitive**: 7 hipotesis spesifik dengan clear research questions
- **Rigorous**: Multi-criteria statistical validation dengan proper effect size analysis
- **Integrated**: Coordinated testing framework dengan cross-validation robustness
- **Academic**: Publication-ready dengan comprehensive statistical power analysis
- **Reproducible**: Complete algorithmic specification untuk independent replication

### **📊 Statistical Standards:**
- **Multi-Criteria Assessment**: ≥2/3 criteria untuk hypothesis validation
- **Effect Size Analysis**: Cohen's d dengan practical significance assessment
- **Statistical Power**: Proper power analysis dengan sample size recommendations
- **Cross-Validation**: Hypothesis stability across 5-fold CV
- **Decision Rules**: Consistent dan transparent decision making framework

Setiap algoritma telah diimplementasikan dengan consideration untuk **hypothesis-driven research**, **statistical rigor**, **reproducibility**, dan **academic standards** yang diperlukan untuk penelitian neural decoding berkualitas tinggi dengan **definitive scientific conclusions**.
