"""
Crell Dataset Optimization Test
==============================

Quick test to see if CCCV3 strategy would improve Crell visual quality
compared to current CCCV2 selection.

Current Crell Results:
- CCCV2 (Selected): SSIM 0.5046, MSE 0.035859
- Goal: Test CCCV3 to see if we can achieve SSIM > 0.7

Strategy: Force CCCV3 selection for Crell and compare visual quality
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)
sys.path.append(parent_dir)

# Import models
try:
    from cccv3.src.models.cccv3_ultimate import create_cccv3_ultimate, create_cccv3_ultimate_trainer
    from src.data import load_dataset_gpu_optimized
except ImportError:
    print("❌ Could not import required models")
    sys.exit(1)

def setup_device():
    """Setup CUDA device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
        return device
    else:
        return torch.device('cpu')

def compute_visual_metrics(pred, target):
    """Compute visual quality metrics"""
    # MSE
    mse = np.mean((pred - target) ** 2)
    
    # PSNR
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Simplified SSIM
    mu1 = np.mean(pred)
    mu2 = np.mean(target)
    sigma1 = np.var(pred)
    sigma2 = np.var(target)
    sigma12 = np.mean((pred - mu1) * (target - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    
    return {'mse': mse, 'psnr': psnr, 'ssim': ssim}

def test_crell_with_cccv3():
    """Test Crell with CCCV3 strategy"""
    
    print("🔧 Testing Crell with CCCV3 Strategy")
    print("=" * 40)
    
    device = setup_device()
    
    # Load Crell dataset
    X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized('crell', device)
    dataset_size = len(X_train)
    
    print(f"✅ Crell loaded: Train={dataset_size}, Test={len(X_test)}, Input_dim={input_dim}")
    
    # Force CCCV3 strategy for Crell
    model = create_cccv3_ultimate(input_dim, 'crell', dataset_size, device)
    trainer = create_cccv3_ultimate_trainer(model, device)
    
    print(f"🧠 Using CCCV3 strategy: {model.strategy}")
    
    # Quick training
    from torch.utils.data import DataLoader, TensorDataset
    
    batch_size = 16
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    config = {'use_optimal_hyperparams': True, 'transfer_learning': True}
    
    print("🔧 Training CCCV3 for Crell...")
    training_results = trainer.train_ultimate_model(train_loader, val_loader, config)
    
    # Test on sample data
    model.eval()
    test_indices = np.random.choice(len(X_test), 6, replace=False)
    sample_X = X_test[test_indices].to(device)
    sample_y = y_test[test_indices].to(device)
    
    with torch.no_grad():
        reconstructions, strategy_info = model(sample_X)
    
    # Compute metrics
    targets = sample_y.cpu().numpy()
    preds = reconstructions.cpu().numpy()
    
    metrics_list = []
    for i in range(len(test_indices)):
        target_img = targets[i, 0]
        pred_img = preds[i, 0]
        metrics = compute_visual_metrics(pred_img, target_img)
        metrics_list.append(metrics)
    
    # Average metrics
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in metrics_list]),
        'psnr': np.mean([m['psnr'] for m in metrics_list if m['psnr'] != float('inf')]),
        'ssim': np.mean([m['ssim'] for m in metrics_list])
    }
    
    print(f"\n📊 CCCV3 Crell Results:")
    print(f"   Average MSE: {avg_metrics['mse']:.6f}")
    print(f"   Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"   Average SSIM: {avg_metrics['ssim']:.4f}")
    
    # Compare with CCCV2 results
    cccv2_results = {
        'mse': 0.035859,
        'psnr': 14.59,
        'ssim': 0.5046
    }
    
    print(f"\n📈 Comparison with CCCV2:")
    print(f"   MSE: {avg_metrics['mse']:.6f} vs {cccv2_results['mse']:.6f}")
    
    if avg_metrics['mse'] < cccv2_results['mse']:
        improvement = ((cccv2_results['mse'] - avg_metrics['mse']) / cccv2_results['mse']) * 100
        print(f"   🏆 CCCV3 MSE better by {improvement:.2f}%")
    else:
        gap = ((avg_metrics['mse'] - cccv2_results['mse']) / cccv2_results['mse']) * 100
        print(f"   📈 CCCV2 MSE better by {gap:.2f}%")
    
    print(f"   SSIM: {avg_metrics['ssim']:.4f} vs {cccv2_results['ssim']:.4f}")
    
    if avg_metrics['ssim'] > cccv2_results['ssim']:
        improvement = ((avg_metrics['ssim'] - cccv2_results['ssim']) / cccv2_results['ssim']) * 100
        print(f"   🏆 CCCV3 SSIM better by {improvement:.2f}%")
    else:
        gap = ((cccv2_results['ssim'] - avg_metrics['ssim']) / cccv2_results['ssim']) * 100
        print(f"   📈 CCCV2 SSIM better by {gap:.2f}%")
    
    # Quality assessment
    cccv3_quality = 'Excellent' if avg_metrics['ssim'] > 0.9 else 'Good' if avg_metrics['ssim'] > 0.7 else 'Fair'
    cccv2_quality = 'Fair'
    
    print(f"\n🎯 Quality Assessment:")
    print(f"   CCCV3: {cccv3_quality}")
    print(f"   CCCV2: {cccv2_quality}")
    
    if cccv3_quality != cccv2_quality:
        print(f"   🎉 CCCV3 achieves {cccv3_quality} quality vs CCCV2 {cccv2_quality}!")
    
    # Recommendation
    if avg_metrics['ssim'] > cccv2_results['ssim'] and avg_metrics['mse'] < cccv2_results['mse']:
        print(f"\n💡 RECOMMENDATION: Update Crell to use CCCV3 strategy!")
        print(f"   CCCV3 shows clear improvement over CCCV2 for Crell dataset")
    elif avg_metrics['ssim'] > 0.7:
        print(f"\n💡 RECOMMENDATION: Consider CCCV3 as alternative for Crell")
        print(f"   CCCV3 achieves good visual quality (SSIM > 0.7)")
    else:
        print(f"\n💡 RECOMMENDATION: Keep CCCV2 for Crell")
        print(f"   CCCV2 remains optimal choice for this dataset")
    
    return avg_metrics, cccv2_results

if __name__ == "__main__":
    test_crell_with_cccv3()
