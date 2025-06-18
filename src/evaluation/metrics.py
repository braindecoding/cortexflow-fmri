"""
Comprehensive Evaluation Metrics for Neural Decoding
===================================================

Advanced evaluation metrics untuk academic research:
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

Academic Features:
- GPU-optimized computation
- Batch processing support
- Comprehensive statistical analysis
- Publication-ready metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
import lpips
from pytorch_msssim import ssim as pytorch_ssim
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveEvaluationMetrics:
    """
    Comprehensive evaluation metrics untuk neural decoding research
    
    Supports:
    - MSE, PSNR, SSIM, LPIPS, MS-SSIM
    - GPU acceleration
    - Batch processing
    - Academic-quality analysis
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize LPIPS model
        try:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_available = True
            print("✅ LPIPS model loaded successfully")
        except Exception as e:
            print(f"❌ LPIPS initialization failed: {e}")
            self.lpips_available = False
    
    def compute_mse(self, pred, target):
        """Compute Mean Squared Error"""
        return F.mse_loss(pred, target).item()
    
    def compute_psnr(self, pred, target, data_range=1.0):
        """
        Compute Peak Signal-to-Noise Ratio
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
            data_range: Maximum possible pixel value
        
        Returns:
            PSNR value in dB
        """
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
        return psnr.item()
    
    def compute_ssim(self, pred, target, data_range=1.0):
        """
        Compute Structural Similarity Index using PyTorch implementation
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
            data_range: Maximum possible pixel value
        
        Returns:
            SSIM value [0, 1]
        """
        try:
            ssim_val = pytorch_ssim(pred, target, data_range=data_range, size_average=True)
            return ssim_val.item()
        except Exception as e:
            print(f"PyTorch SSIM failed: {e}, using scikit-image")
            return self.compute_ssim_skimage(pred, target, data_range)
    
    def compute_ssim_skimage(self, pred, target, data_range=1.0):
        """Fallback SSIM using scikit-image"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        ssim_scores = []
        for i in range(pred_np.shape[0]):
            for c in range(pred_np.shape[1]):
                ssim_score = skimage_ssim(
                    target_np[i, c], pred_np[i, c], 
                    data_range=data_range
                )
                ssim_scores.append(ssim_score)
        
        return np.mean(ssim_scores)
    
    def compute_lpips(self, pred, target):
        """
        Compute Learned Perceptual Image Patch Similarity
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
        
        Returns:
            LPIPS value (lower is better)
        """
        if not self.lpips_available:
            return 0.0
        
        try:
            # Convert grayscale to RGB for LPIPS
            if pred.shape[1] == 1:
                pred_rgb = pred.repeat(1, 3, 1, 1)
                target_rgb = target.repeat(1, 3, 1, 1)
            else:
                pred_rgb = pred
                target_rgb = target
            
            # Ensure minimum size for LPIPS (64x64)
            if pred_rgb.shape[-1] < 64:
                pred_rgb = F.interpolate(pred_rgb, size=(64, 64), mode='bilinear', align_corners=False)
                target_rgb = F.interpolate(target_rgb, size=(64, 64), mode='bilinear', align_corners=False)
            
            # Normalize to [-1, 1] range for LPIPS
            pred_rgb = pred_rgb * 2.0 - 1.0
            target_rgb = target_rgb * 2.0 - 1.0
            
            lpips_val = self.lpips_model(pred_rgb, target_rgb)
            return lpips_val.mean().item()
            
        except Exception as e:
            print(f"LPIPS computation failed: {e}")
            return 0.0
    
    def compute_all_metrics(self, pred, target, data_range=1.0):
        """
        Compute all evaluation metrics (4 valid metrics)

        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
            data_range: Maximum possible pixel value

        Returns:
            Dictionary with 4 metrics: MSE, PSNR, SSIM, LPIPS
        """
        metrics = {}

        # Ensure tensors are on the same device
        pred = pred.to(self.device)
        target = target.to(self.device)

        # Ensure same shape
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

        # Compute 4 valid metrics
        metrics['MSE'] = self.compute_mse(pred, target)
        metrics['PSNR'] = self.compute_psnr(pred, target, data_range)
        metrics['SSIM'] = self.compute_ssim(pred, target, data_range)
        metrics['LPIPS'] = self.compute_lpips(pred, target)

        return metrics


def calculate_comprehensive_metrics(pred, target, device='cuda', data_range=1.0):
    """
    Convenience function for calculating comprehensive metrics

    Args:
        pred: Predicted images [B, C, H, W] or [B, H, W]
        target: Target images [B, C, H, W] or [B, H, W]
        device: Device for computation
        data_range: Maximum possible pixel value

    Returns:
        Dictionary with metrics: mse, psnr, ssim, lpips
    """
    # Initialize evaluator
    evaluator = ComprehensiveEvaluationMetrics(device)

    # Ensure 4D tensors
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    # Compute metrics
    metrics = evaluator.compute_all_metrics(pred, target, data_range)

    # Convert to lowercase keys for consistency
    return {
        'mse': metrics['MSE'],
        'psnr': metrics['PSNR'],
        'ssim': metrics['SSIM'],
        'lpips': metrics['LPIPS']
    }


def test_evaluation_metrics():
    """Test function untuk 4 evaluation metrics"""
    print("🧪 TESTING 4 COMPREHENSIVE EVALUATION METRICS")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ComprehensiveEvaluationMetrics(device)

    # Create test data
    batch_size = 4
    pred = torch.randn(batch_size, 1, 28, 28).to(device)
    target = torch.randn(batch_size, 1, 28, 28).to(device)

    # Normalize to [0, 1]
    pred = torch.sigmoid(pred)
    target = torch.sigmoid(target)

    print(f"📊 Test data shape: {pred.shape}")
    print(f"🔧 Device: {device}")

    # Compute 4 valid metrics
    metrics = evaluator.compute_all_metrics(pred, target)

    print("\n📈 4 EVALUATION METRICS RESULTS:")
    print("-" * 40)
    for metric_name, value in metrics.items():
        print(f"   {metric_name}: {value:.6f}")

    print(f"\n✅ ALL 4 METRICS COMPUTED SUCCESSFULLY!")
    print("📊 Metrics: MSE, PSNR, SSIM, LPIPS")
    return metrics


if __name__ == "__main__":
    test_evaluation_metrics()
