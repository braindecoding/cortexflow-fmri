# Algoritma Metodologi CortexFlow

## Gambaran Umum

Dokumen ini menyajikan algoritma-algoritma kunci yang digunakan dalam metodologi penelitian CortexFlow. Setiap algoritma dijelaskan dengan pseudocode yang detail dan implementasi yang dapat direproduksi.

---

## Algoritma 1: Enhanced 5-Fold Cross-Validation

### Deskripsi
Algoritma enhanced 5-fold cross-validation dengan statistical rigor untuk robust model evaluation.

### Pseudocode
```
Algorithm 1: Enhanced 5-Fold Cross-Validation
Input: Dataset X, y; Models M = {m₁, m₂, ..., mₙ}; k = 5
Output: CV_scores, Statistical_analysis

1. INITIALIZE:
   - Set random_seed = 42 for reproducibility
   - Create KFold(n_splits=5, shuffle=True, random_state=42)
   - Initialize score_matrix[n_models][k_folds]

2. FOR each fold i = 1 to k:
   a. Split data: (X_train, y_train), (X_val, y_val) = split(X, y, fold_i)
   b. FOR each model mⱼ in M:
      i. Initialize model_copy = deep_copy(mⱼ)
      ii. Train model_copy on (X_train, y_train)
      iii. Evaluate: score = evaluate(model_copy, X_val, y_val)
      iv. Store: score_matrix[j][i] = score
   c. END FOR

3. STATISTICAL ANALYSIS:
   a. FOR each model mⱼ:
      i. Calculate: mean_score = mean(score_matrix[j])
      ii. Calculate: std_score = std(score_matrix[j])
      iii. Calculate: ci_lower, ci_upper = confidence_interval(score_matrix[j])
   b. END FOR
   
   c. FOR each pair (mᵢ, mⱼ):
      i. Perform: t_stat, p_value = paired_ttest(score_matrix[i], score_matrix[j])
      ii. Calculate: effect_size = cohens_d(score_matrix[i], score_matrix[j])
      iii. Store significance results
   d. END FOR

4. RETURN CV_scores, Statistical_analysis
```

### Kompleksitas
- **Time Complexity**: O(k × n × T) dimana T adalah training time per model
- **Space Complexity**: O(k × n) untuk menyimpan scores
- **Statistical Power**: Enhanced dengan n=5 samples untuk robust T-test

---

## Algoritma 2: Intelligent Ensemble Weighting

### Deskripsi
Algoritma learned weighting untuk intelligent ensemble combination dari 8 CortexFlow variants.

### Pseudocode
```
Algorithm 2: Intelligent Ensemble Weighting
Input: Base_models B = {b₁, b₂, ..., b₈}; Training_data (X, y)
Output: Ensemble_model E dengan learned weights

1. INITIALIZE:
   - Create WeightingNetwork W: input_dim → 512 → 256 → 128 → 8
   - Initialize weights w = [w₁, w₂, ..., w₈]
   - Set optimizer = Adam(lr=0.001)

2. TRAIN BASE MODELS:
   a. FOR each base_model bᵢ in B:
      i. Train bᵢ on (X, y) with individual configuration
      ii. Store trained model: B_trained[i] = bᵢ
   b. END FOR

3. ENSEMBLE TRAINING:
   a. FOR epoch = 1 to max_epochs:
      i. FOR each batch (x_batch, y_batch):
         
         # Forward pass through base models
         ii. predictions = []
         iii. FOR each bᵢ in B_trained:
             - pred_i = bᵢ(x_batch)  # No gradient update
             - predictions.append(pred_i)
         iv. END FOR
         
         # Learn optimal weights
         v. raw_weights = W(x_batch)  # Context-dependent weights
         vi. weights = softmax(raw_weights)  # Normalize to probabilities
         
         # Ensemble prediction
         vii. ensemble_pred = Σᵢ₌₁⁸ weights[i] × predictions[i]
         
         # Loss and backpropagation
         viii. loss = MSE(ensemble_pred, y_batch)
         ix. loss.backward()
         x. optimizer.step()
         xi. optimizer.zero_grad()
         
      b. END FOR batch
   c. END FOR epoch

4. ADAPTIVE WEIGHTING MECHANISM:
   a. FOR each input x:
      i. Compute context_weights = softmax(W(x))
      ii. Ensemble_output = Σᵢ₌₁⁸ context_weights[i] × bᵢ(x)
   b. END FOR

5. RETURN Ensemble_model E with learned WeightingNetwork W
```

### Kompleksitas
- **Time Complexity**: O(E × B × N) dimana E=epochs, B=batch_size, N=ensemble_size
- **Space Complexity**: O(8 × M) dimana M adalah model parameters
- **Adaptivity**: Context-dependent weighting berdasarkan input characteristics

---

## Algoritma 3: Multi-Metric Comprehensive Evaluation

### Deskripsi
Algoritma evaluasi komprehensif menggunakan 4 metrik: MSE, PSNR, SSIM, LPIPS.

### Pseudocode
```
Algorithm 3: Multi-Metric Comprehensive Evaluation
Input: Predictions P, Ground_truth G, Device
Output: Comprehensive_metrics = {MSE, PSNR, SSIM, LPIPS}

1. INITIALIZE:
   - Load LPIPS_model = LPIPS(net='alex').to(device)
   - Initialize metrics_dict = {}

2. PREPROCESSING:
   a. Ensure P, G are on same device
   b. Normalize P, G to [0, 1] range if needed
   c. Reshape to consistent format: [batch, channels, height, width]

3. COMPUTE MSE:
   a. mse_value = mean((P - G)²)
   b. metrics_dict['MSE'] = mse_value

4. COMPUTE PSNR:
   a. max_pixel_value = 1.0  # Assuming normalized data
   b. psnr_value = 20 × log₁₀(max_pixel_value / √mse_value)
   c. metrics_dict['PSNR'] = psnr_value

5. COMPUTE SSIM:
   a. # Structural Similarity Index
   b. μₚ = mean(P), μ_G = mean(G)
   c. σₚ = var(P), σ_G = var(G)
   d. σₚ_G = covariance(P, G)
   e. c₁ = (0.01 × max_pixel_value)²
   f. c₂ = (0.03 × max_pixel_value)²
   g. ssim_value = ((2×μₚ×μ_G + c₁) × (2×σₚ_G + c₂)) / 
                   ((μₚ² + μ_G² + c₁) × (σₚ² + σ_G² + c₂))
   h. metrics_dict['SSIM'] = ssim_value

6. COMPUTE LPIPS:
   a. # Convert to 3-channel for LPIPS
   b. P_3ch = repeat(P, channels=3)
   c. G_3ch = repeat(G, channels=3)
   d. lpips_value = LPIPS_model(P_3ch, G_3ch).mean()
   e. metrics_dict['LPIPS'] = lpips_value

7. STATISTICAL AGGREGATION:
   a. FOR each metric in metrics_dict:
      i. Compute batch statistics: mean, std, min, max
      ii. Store comprehensive results
   b. END FOR

8. RETURN Comprehensive_metrics
```

### Kompleksitas
- **Time Complexity**: O(N × H × W × C) untuk setiap metrik
- **Space Complexity**: O(N × H × W × C) untuk tensor operations
- **Accuracy**: Multi-perspective evaluation untuk robust assessment

---

## Algoritma 4: Statistical Significance Testing

### Deskripsi
Algoritma comprehensive statistical analysis dengan T-test dan effect size calculation.

### Pseudocode
```
Algorithm 4: Statistical Significance Testing
Input: CV_results = {method₁: [scores], method₂: [scores], ...}
Output: Significance_matrix, Effect_sizes, Confidence_intervals

1. INITIALIZE:
   - methods = list(CV_results.keys())
   - n_methods = len(methods)
   - p_matrix = zeros(n_methods, n_methods)
   - effect_matrix = zeros(n_methods, n_methods)

2. PAIRWISE STATISTICAL TESTING:
   a. FOR i = 1 to n_methods:
      b. FOR j = 1 to n_methods:
         IF i ≠ j:
            # Get scores for comparison
            i. scores_i = CV_results[methods[i]]
            ii. scores_j = CV_results[methods[j]]
            
            # Paired t-test (since same CV folds)
            iii. t_statistic, p_value = paired_ttest(scores_i, scores_j)
            iv. p_matrix[i][j] = p_value
            
            # Effect size calculation (Cohen's d)
            v. mean_diff = mean(scores_i) - mean(scores_j)
            vi. pooled_std = √((var(scores_i) + var(scores_j)) / 2)
            vii. cohens_d = mean_diff / pooled_std
            viii. effect_matrix[i][j] = cohens_d
         END IF
      c. END FOR j
   d. END FOR i

3. CONFIDENCE INTERVALS:
   a. FOR each method in methods:
      i. scores = CV_results[method]
      ii. n = len(scores)
      iii. mean_score = mean(scores)
      iv. std_score = std(scores)
      v. se = std_score / √n  # Standard error
      vi. t_critical = t_distribution(df=n-1, α=0.05)
      vii. margin_error = t_critical × se
      viii. ci_lower = mean_score - margin_error
      ix. ci_upper = mean_score + margin_error
      x. Store: CI[method] = (ci_lower, ci_upper)
   b. END FOR

4. MULTIPLE COMPARISON CORRECTION:
   a. # Bonferroni correction for multiple testing
   b. n_comparisons = n_methods × (n_methods - 1) / 2
   c. corrected_alpha = 0.05 / n_comparisons
   d. Apply correction to p_matrix

5. EFFECT SIZE INTERPRETATION:
   a. FOR each effect_size in effect_matrix:
      IF |effect_size| < 0.2: interpretation = "Small"
      ELIF |effect_size| < 0.5: interpretation = "Medium"  
      ELIF |effect_size| < 0.8: interpretation = "Large"
      ELSE: interpretation = "Very Large"
   b. END FOR

6. RETURN Significance_matrix, Effect_sizes, Confidence_intervals
```

### Kompleksitas
- **Time Complexity**: O(n²) untuk pairwise comparisons
- **Space Complexity**: O(n²) untuk matrices
- **Statistical Power**: Enhanced dengan n=5 CV folds

---

## Algoritma 5: GPU-Optimized Training Pipeline

### Deskripsi
Algoritma GPU-optimized training dengan memory management dan WSL optimization.

### Pseudocode
```
Algorithm 5: GPU-Optimized Training Pipeline
Input: Model, Dataset, Configuration, Device
Output: Trained_model, Training_history

1. GPU INITIALIZATION:
   a. Set CUDA_VISIBLE_DEVICES = 0
   b. Configure memory allocation: max_split_size_mb = 512
   c. Enable deterministic operations for reproducibility
   d. Set random seeds: torch.manual_seed(42)

2. MEMORY OPTIMIZATION:
   a. # Direct GPU loading
   b. X_train = torch.tensor(data['fmriTrn'], device=device)
   c. y_train = torch.tensor(data['stimTrn'], device=device)
   d. # Optimize batch size based on GPU memory
   e. optimal_batch_size = calculate_optimal_batch_size(model, device)

3. MODEL PREPARATION:
   a. model = model.to(device)
   b. optimizer = Adam(model.parameters(), lr=config['lr'])
   c. scheduler = ReduceLROnPlateau(optimizer, patience=config['patience'])
   d. criterion = MSELoss()

4. TRAINING LOOP:
   a. FOR epoch = 1 to max_epochs:
      i. model.train()
      ii. epoch_loss = 0
      
      iii. FOR batch in DataLoader(X_train, y_train, batch_size):
         # GPU-optimized forward pass
         iv. predictions = model(batch_x)
         v. loss = criterion(predictions, batch_y)
         
         # Memory-efficient backpropagation
         vi. optimizer.zero_grad()
         vii. loss.backward()
         viii. torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
         ix. optimizer.step()
         
         # Memory cleanup
         x. del predictions, loss
         xi. torch.cuda.empty_cache() if device == 'cuda'
         
         xii. epoch_loss += loss.item()
      iv. END FOR batch
      
      # Validation and scheduling
      v. val_loss = validate_model(model, X_val, y_val, device)
      vi. scheduler.step(val_loss)
      vii. Store training history
      
      # Early stopping check
      viii. IF early_stopping_criteria_met:
           BREAK
      ix. END IF
      
   b. END FOR epoch

5. WSL OPTIMIZATION:
   a. # Memory management for WSL environment
   b. Set OMP_NUM_THREADS = 4
   c. Configure virtual memory: vm.overcommit_memory = 1
   d. Optimize CUDA context for WSL

6. MEMORY CLEANUP:
   a. torch.cuda.empty_cache()
   b. del intermediate_variables
   c. gc.collect()

7. RETURN Trained_model, Training_history
```

### Kompleksitas
- **Time Complexity**: O(E × B × F) dimana E=epochs, B=batches, F=forward_pass
- **Space Complexity**: O(M + D) dimana M=model_params, D=data_size
- **Memory Efficiency**: GPU-optimized dengan automatic cleanup

---

## Kesimpulan Algoritma

Kelima algoritma ini membentuk foundation metodologis yang robust untuk penelitian CortexFlow:

1. **Enhanced 5-Fold CV**: Memberikan statistical rigor dengan n=5 samples
2. **Intelligent Ensemble**: Mengoptimalkan combination dari 8 variants
3. **Multi-Metric Evaluation**: Comprehensive assessment dari multiple perspectives
4. **Statistical Testing**: Rigorous significance analysis dengan effect sizes
5. **GPU-Optimized Training**: Efficient implementation untuk large-scale experiments

Setiap algoritma telah diimplementasikan dengan consideration untuk reproducibility, efficiency, dan academic rigor yang diperlukan untuk penelitian neural decoding berkualitas tinggi.
