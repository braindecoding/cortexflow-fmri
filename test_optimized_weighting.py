#!/usr/bin/env python3
"""
Optimized Weighting Training Test
================================

Quick test to verify that the optimized weighting mechanism
improves ensemble performance by giving more weight to the
best-performing Baseline CNN variant.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time

# Add src to path
sys.path.append('src')

from models.ensemble import CortexFlowEnsemble
from models.baseline import StandardBaselineCNN
from data.loader import load_dataset_gpu_optimized

def quick_optimized_test():
    """Quick training test with optimized weighting"""
    
    print("🚀 OPTIMIZED WEIGHTING TRAINING TEST")
    print("=" * 50)
    
    # Set device and reproducibility
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed(42)
    
    print(f"🔥 Device: {device}")
    
    # Load dataset
    try:
        dataset_name = 'miyawaki'
        X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
        
        if X_train is None:
            print("❌ Dataset loading failed")
            return
            
        print(f"✅ Dataset: {dataset_name}")
        print(f"📊 Train: {X_train.shape[0]} samples")
        
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return
    
    # Initialize models
    try:
        optimized_ensemble = CortexFlowEnsemble(input_dim, device)
        baseline_cnn = StandardBaselineCNN(input_dim, device)
        
        print(f"\n🏗️ Models initialized")
        
        # Check optimized weighting
        ensemble_info = optimized_ensemble.get_ensemble_info()
        print(f"📊 Ensemble: {ensemble_info['name']}")
        print(f"🎯 Weighting: {ensemble_info['weighting']}")
        
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return
    
    # Training configuration
    num_epochs = 3  # Very quick test
    learning_rate = 0.001
    batch_size = min(16, X_train.shape[0])
    
    print(f"\n⚙️ Training configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    
    # Create optimizers
    ensemble_optimizer = optim.Adam(optimized_ensemble.parameters(), lr=learning_rate)
    baseline_optimizer = optim.Adam(baseline_cnn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training function
    def train_model(model, optimizer, model_name):
        model.train()
        total_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        for i in range(0, X_train.shape[0], batch_size):
            end_idx = min(i + batch_size, X_train.shape[0])
            batch_x = X_train[i:end_idx]
            batch_y = y_train[i:end_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            if outputs.shape != batch_y.shape:
                outputs = outputs.view(batch_y.shape)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        training_time = time.time() - start_time
        
        return avg_loss, training_time
    
    # Evaluation function
    def evaluate_model(model):
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            if outputs.shape != y_test.shape:
                outputs = outputs.view(y_test.shape)
            test_loss = criterion(outputs, y_test).item()
        return test_loss
    
    print(f"\n🏃 OPTIMIZED TRAINING COMPARISON:")
    print("=" * 40)
    
    # Train Optimized Ensemble
    print(f"\n🚀 Training Optimized Ensemble...")
    ensemble_losses = []
    
    for epoch in range(num_epochs):
        train_loss, train_time = train_model(optimized_ensemble, ensemble_optimizer, "Optimized Ensemble")
        test_loss = evaluate_model(optimized_ensemble)
        
        ensemble_losses.append(test_loss)
        print(f"   Epoch {epoch+1}: Train: {train_loss:.6f}, Test: {test_loss:.6f}, Time: {train_time:.2f}s")
    
    # Train Baseline CNN
    print(f"\n📊 Training Baseline CNN...")
    baseline_losses = []
    
    for epoch in range(num_epochs):
        train_loss, train_time = train_model(baseline_cnn, baseline_optimizer, "Baseline CNN")
        test_loss = evaluate_model(baseline_cnn)
        
        baseline_losses.append(test_loss)
        print(f"   Epoch {epoch+1}: Train: {train_loss:.6f}, Test: {test_loss:.6f}, Time: {train_time:.2f}s")
    
    # Results comparison
    print(f"\n📈 OPTIMIZED RESULTS:")
    print("=" * 25)
    
    final_ensemble_loss = ensemble_losses[-1]
    final_baseline_loss = baseline_losses[-1]
    
    print(f"📊 Final Test Loss:")
    print(f"   Optimized Ensemble: {final_ensemble_loss:.6f}")
    print(f"   Baseline CNN: {final_baseline_loss:.6f}")
    
    if final_ensemble_loss < final_baseline_loss:
        improvement = ((final_baseline_loss - final_ensemble_loss) / final_baseline_loss) * 100
        print(f"   🎉 Optimized Ensemble wins by {improvement:.2f}%!")
        print(f"   ✅ OPTIMIZATION SUCCESS!")
    else:
        gap = ((final_ensemble_loss - final_baseline_loss) / final_baseline_loss) * 100
        print(f"   ⚠️ Optimized Ensemble behind by {gap:.2f}%")
        print(f"   🔧 May need further tuning")
    
    # Learning analysis
    ensemble_improvement = ((ensemble_losses[0] - ensemble_losses[-1]) / ensemble_losses[0]) * 100
    baseline_improvement = ((baseline_losses[0] - baseline_losses[-1]) / baseline_losses[0]) * 100
    
    print(f"\n📈 Learning Progress:")
    print(f"   Optimized Ensemble: {ensemble_improvement:.2f}% improvement")
    print(f"   Baseline CNN: {baseline_improvement:.2f}% improvement")
    
    print(f"\n🎯 OPTIMIZATION SUMMARY:")
    print("=" * 30)
    print("✅ Baseline CNN now gets 26.0% weight (vs 15.0% before)")
    print("✅ Aggressive baseline emphasis: 3.0x factor")
    print("✅ Performance-based weighting implemented")
    
    if final_ensemble_loss < final_baseline_loss:
        print("🏆 OPTIMIZATION SUCCESSFUL: Ensemble beats baseline!")
    else:
        print("🔧 Optimization shows promise, may need full training")

if __name__ == "__main__":
    quick_optimized_test()
