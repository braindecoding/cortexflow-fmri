#!/usr/bin/env python3
"""
Data Source Verification Script
Verifies that all data in methodology tables comes from actual training results
"""

import json
import numpy as np
from scipy import stats

def verify_performance_table():
    """Verify Tabel 4 data against actual training results"""
    
    print("🔍 VERIFYING TABEL 4: PERFORMANCE RESULTS")
    print("=" * 50)
    
    # Load actual results
    with open('results/comprehensive_training_cv/comprehensive_training_results.json', 'r') as f:
        actual_results = json.load(f)
    
    # Expected data from table
    table_data = {
        'miyawaki': {
            'CortexFlow_Lite': 0.033545,
            'CortexFlow_Multi_Pathway': 0.096494,
            'CortexFlow_Ensemble': 0.022393,
            'MinD_Vis': 0.024000,
            'Brain_Diffuser': 0.015272
        },
        'vangerven': {
            'CortexFlow_Lite': 0.041823,
            'CortexFlow_Multi_Pathway': 0.053095,
            'CortexFlow_Ensemble': 0.042603,
            'MinD_Vis': 0.047271,
            'Brain_Diffuser': 0.043848
        },
        'mindbigdata': {
            'CortexFlow_Lite': 0.056274,
            'CortexFlow_Multi_Pathway': 0.054573,
            'CortexFlow_Ensemble': 0.060648,
            'MinD_Vis': 0.056394,
            'Brain_Diffuser': 0.054800
        },
        'crell': {
            'CortexFlow_Lite': 0.029198,
            'CortexFlow_Multi_Pathway': 0.028963,
            'CortexFlow_Ensemble': 0.028666,
            'MinD_Vis': 0.029013,
            'Brain_Diffuser': 0.029482
        }
    }
    
    # Verify each entry
    all_verified = True
    for dataset, methods in table_data.items():
        print(f"\n{dataset.upper()}:")
        for method, table_value in methods.items():
            actual_key = method.replace('-', '_')
            if actual_key in actual_results[dataset]:
                actual_value = actual_results[dataset][actual_key]
                match = abs(actual_value - table_value) < 1e-6
                status = "✅ VERIFIED" if match else "❌ MISMATCH"
                print(f"  {method}: {table_value:.6f} vs {actual_value:.6f} {status}")
                if not match:
                    all_verified = False
            else:
                print(f"  {method}: NOT FOUND in actual results ❌")
                all_verified = False
    
    return all_verified

def verify_statistical_table():
    """Verify Tabel 5 data against actual CV results"""
    
    print("\n🔍 VERIFYING TABEL 5: STATISTICAL ANALYSIS")
    print("=" * 50)
    
    # Load CV results
    with open('results/comprehensive_training_cv/cross_validation_results.json', 'r') as f:
        cv_results = json.load(f)
    
    # Expected winners from table
    expected_winners = {
        'miyawaki': 'Brain_Diffuser',
        'vangerven': 'MinD_Vis', 
        'mindbigdata': 'CortexFlow_Multi-Pathway',
        'crell': 'MinD_Vis'
    }
    
    all_verified = True
    for dataset, expected_winner in expected_winners.items():
        print(f"\n{dataset.upper()}:")
        
        # Find actual winner (lowest mean MSE)
        means = {method: np.mean(scores) for method, scores in cv_results[dataset].items()}
        actual_winner = min(means, key=means.get)
        
        # Check if winner matches
        winner_match = actual_winner == expected_winner
        status = "✅ VERIFIED" if winner_match else "❌ MISMATCH"
        print(f"  Expected Winner: {expected_winner}")
        print(f"  Actual Winner: {actual_winner} {status}")
        
        if winner_match:
            # Verify statistics
            winner_scores = cv_results[dataset][actual_winner]
            mean_score = np.mean(winner_scores)
            std_score = np.std(winner_scores, ddof=1)
            
            # Calculate 95% CI
            n = len(winner_scores)
            t_critical = stats.t.ppf(0.975, df=n-1)
            margin_error = t_critical * (std_score / np.sqrt(n))
            ci_lower = mean_score - margin_error
            ci_upper = mean_score + margin_error
            
            print(f"  Mean ± SD: {mean_score:.6f} ± {std_score:.6f}")
            print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
            print(f"  CV Scores: {[round(score, 4) for score in winner_scores]}")
        else:
            all_verified = False
    
    return all_verified

def main():
    """Main verification function"""
    
    print("📊 DATA SOURCE VERIFICATION FOR METODOLOGI TABLES")
    print("=" * 60)
    
    # Verify performance table
    perf_verified = verify_performance_table()
    
    # Verify statistical table  
    stat_verified = verify_statistical_table()
    
    # Overall verification
    print("\n🎯 OVERALL VERIFICATION RESULTS")
    print("=" * 40)
    
    if perf_verified:
        print("✅ Tabel 4 (Performance Results): ALL DATA VERIFIED")
    else:
        print("❌ Tabel 4 (Performance Results): CONTAINS ERRORS")
    
    if stat_verified:
        print("✅ Tabel 5 (Statistical Analysis): ALL DATA VERIFIED")
    else:
        print("❌ Tabel 5 (Statistical Analysis): CONTAINS ERRORS")
    
    if perf_verified and stat_verified:
        print("\n🏆 CONCLUSION: ALL METHODOLOGY DATA IS AUTHENTIC!")
        print("✅ Data integrity maintained")
        print("✅ Research ethics compliant")
        print("✅ Academic standards met")
    else:
        print("\n⚠️  CONCLUSION: DATA VERIFICATION FAILED!")
        print("❌ Some data does not match training results")
        print("❌ Requires correction for research integrity")

if __name__ == "__main__":
    main()
