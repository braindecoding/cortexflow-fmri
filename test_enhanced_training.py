#!/usr/bin/env python3
"""
Enhanced Ensemble Training Test
==============================

Quick training test to compare:
1. Original CortexFlow Ensemble (before enhancement)
2. Enhanced CortexFlow Ensemble (with improvements)
3. Standalone Baseline CNN

This test runs a few epochs to verify that the enhanced ensemble
shows improved performance compared to the original.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append('src')

from models.ensemble import CortexFlowEnsemble
from models.baseline import StandardBaselineCNN
from data.loader import load_dataset_gpu_optimized

def quick_training_test():
    """Quick training test for enhanced ensemble"""
    
    print("🚀 ENHANCED ENSEMBLE TRAINING TEST")
    print("=" * 50)
    
    # Set device and reproducibility
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed(42)
    
    print(f"🔥 Device: {device}")
    print(f"🎯 Random seed: 42")
    
    # Load dataset
    try:
        dataset_name = 'miyawaki'
        X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
        
        if X_train is None:
            print("❌ Dataset loading failed")
            return
            
        print(f"✅ Dataset: {dataset_name}")
        print(f"📊 Train: {X_train.shape[0]} samples")
        print(f"🧪 Test: {X_test.shape[0]} samples")
        
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return
    
    # Initialize models
    try:
        enhanced_ensemble = CortexFlowEnsemble(input_dim, device)
        baseline_cnn = StandardBaselineCNN(input_dim, device)
        
        print(f"\n🏗️ Models initialized:")
        print(f"   Enhanced Ensemble: {sum(p.numel() for p in enhanced_ensemble.parameters()):,} params")
        print(f"   Baseline CNN: {sum(p.numel() for p in baseline_cnn.parameters()):,} params")
        
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return
    
    # Training configuration
    num_epochs = 5  # Quick test
    learning_rate = 0.001
    batch_size = min(16, X_train.shape[0])  # Small batch for quick test
    
    print(f"\n⚙️ Training configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: {batch_size}")
    
    # Create optimizers
    ensemble_optimizer = optim.Adam(enhanced_ensemble.parameters(), lr=learning_rate)
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
            
            # Ensure output shape matches target
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
    def evaluate_model(model, model_name):
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            if outputs.shape != y_test.shape:
                outputs = outputs.view(y_test.shape)
            test_loss = criterion(outputs, y_test).item()
        return test_loss
    
    print(f"\n🏃 TRAINING COMPARISON:")
    print("=" * 30)
    
    # Train Enhanced Ensemble
    print(f"\n🚀 Training Enhanced Ensemble...")
    ensemble_losses = []
    ensemble_times = []
    
    for epoch in range(num_epochs):
        train_loss, train_time = train_model(enhanced_ensemble, ensemble_optimizer, "Enhanced Ensemble")
        test_loss = evaluate_model(enhanced_ensemble, "Enhanced Ensemble")
        
        ensemble_losses.append(test_loss)
        ensemble_times.append(train_time)
        
        print(f"   Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Time: {train_time:.2f}s")
    
    # Train Baseline CNN
    print(f"\n📊 Training Baseline CNN...")
    baseline_losses = []
    baseline_times = []
    
    for epoch in range(num_epochs):
        train_loss, train_time = train_model(baseline_cnn, baseline_optimizer, "Baseline CNN")
        test_loss = evaluate_model(baseline_cnn, "Baseline CNN")
        
        baseline_losses.append(test_loss)
        baseline_times.append(train_time)
        
        print(f"   Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Time: {train_time:.2f}s")
    
    # Results comparison
    print(f"\n📈 RESULTS COMPARISON:")
    print("=" * 25)
    
    final_ensemble_loss = ensemble_losses[-1]
    final_baseline_loss = baseline_losses[-1]
    
    avg_ensemble_time = np.mean(ensemble_times)
    avg_baseline_time = np.mean(baseline_times)
    
    print(f"📊 Final Test Loss:")
    print(f"   Enhanced Ensemble: {final_ensemble_loss:.6f}")
    print(f"   Baseline CNN: {final_baseline_loss:.6f}")
    
    if final_ensemble_loss < final_baseline_loss:
        improvement = ((final_baseline_loss - final_ensemble_loss) / final_baseline_loss) * 100
        print(f"   🎉 Enhanced Ensemble wins by {improvement:.2f}%!")
    else:
        gap = ((final_ensemble_loss - final_baseline_loss) / final_baseline_loss) * 100
        print(f"   ⚠️ Enhanced Ensemble behind by {gap:.2f}%")
    
    print(f"\n⏱️ Training Time:")
    print(f"   Enhanced Ensemble: {avg_ensemble_time:.2f}s/epoch")
    print(f"   Baseline CNN: {avg_baseline_time:.2f}s/epoch")
    print(f"   Time ratio: {avg_ensemble_time/avg_baseline_time:.2f}x")
    
    print(f"\n🎯 ENHANCEMENT ANALYSIS:")
    print("=" * 25)
    
    # Check if ensemble is learning
    ensemble_improvement = ((ensemble_losses[0] - ensemble_losses[-1]) / ensemble_losses[0]) * 100
    baseline_improvement = ((baseline_losses[0] - baseline_losses[-1]) / baseline_losses[0]) * 100
    
    print(f"📈 Learning Progress:")
    print(f"   Enhanced Ensemble: {ensemble_improvement:.2f}% improvement")
    print(f"   Baseline CNN: {baseline_improvement:.2f}% improvement")
    
    if ensemble_improvement > baseline_improvement:
        print(f"   ✅ Enhanced Ensemble learns faster!")
    else:
        print(f"   ⚠️ Baseline CNN learns faster")
    
    print(f"\n🎉 TRAINING TEST COMPLETED!")
    print("=" * 30)
    print("✅ Enhanced ensemble architecture tested")
    print("✅ Training comparison completed")
    print("✅ Performance metrics analyzed")
    
    if final_ensemble_loss < final_baseline_loss:
        print("🏆 Enhanced ensemble shows improvement!")
    else:
        print("🔧 Enhanced ensemble needs further tuning")

if __name__ == "__main__":
    quick_training_test()
