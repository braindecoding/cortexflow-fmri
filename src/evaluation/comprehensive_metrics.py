#!/usr/bin/env python3
"""
Comprehensive Evaluation Metrics for CortexFlow
Implements multiple evaluation metrics: MSE, PSNR, SSIM, FID, LPIPS, CLIP Score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.linalg import sqrtm
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import lpips
    LPIPS_AVAILABLE = True
    print("✅ LPIPS imported successfully")
except ImportError:
    LPIPS_AVAILABLE = False
    print("❌ LPIPS not available. Install with: pip install lpips")

try:
    import clip
    CLIP_AVAILABLE = True
    print("✅ CLIP imported successfully")
except ImportError:
    CLIP_AVAILABLE = False
    print("❌ CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

class ComprehensiveMetrics:
    """
    Comprehensive evaluation metrics for neural decoding reconstruction
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Initialize LPIPS if available
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            print("✅ LPIPS model loaded")
        else:
            self.lpips_model = None
            
        # Initialize CLIP if available
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            print("✅ CLIP model loaded")
        else:
            self.clip_model = None
            
        # Initialize Inception model for FID
        self.inception_model = self._load_inception_model()
        print("✅ Inception model loaded for FID")
        
        print("🚀 Comprehensive metrics initialized")
    
    def _load_inception_model(self):
        """Load Inception model for FID calculation"""
        inception = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        inception.fc = nn.Identity()  # Remove final classification layer
        inception.eval()
        return inception.to(self.device)
    
    def compute_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute all available metrics
        
        Args:
            predictions: Predicted images [B, 784] or [B, 1, 28, 28]
            targets: Target images [B, 784] or [B, 1, 28, 28]
        
        Returns:
            Dictionary with all computed metrics
        """
        
        # Ensure correct shape [B, 1, 28, 28]
        if predictions.dim() == 2:
            predictions = predictions.view(-1, 1, 28, 28)
        if targets.dim() == 2:
            targets = targets.view(-1, 1, 28, 28)
        
        # Ensure values in [0, 1]
        predictions = torch.clamp(predictions, 0, 1)
        targets = torch.clamp(targets, 0, 1)
        
        metrics = {}
        
        # 1. MSE (Mean Squared Error)
        metrics['mse'] = self.compute_mse(predictions, targets)
        
        # 2. PSNR (Peak Signal-to-Noise Ratio)
        metrics['psnr'] = self.compute_psnr(predictions, targets)
        
        # 3. SSIM (Structural Similarity Index)
        metrics['ssim'] = self.compute_ssim(predictions, targets)
        
        # 4. FID (Fréchet Inception Distance)
        metrics['fid'] = self.compute_fid(predictions, targets)
        
        # 5. LPIPS (Learned Perceptual Image Patch Similarity)
        if self.lpips_model is not None:
            metrics['lpips'] = self.compute_lpips(predictions, targets)
        else:
            metrics['lpips'] = None
            
        # 6. CLIP Score
        if self.clip_model is not None:
            metrics['clip_score'] = self.compute_clip_score(predictions, targets)
        else:
            metrics['clip_score'] = None
        
        return metrics
    
    def compute_mse(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Mean Squared Error"""
        mse = F.mse_loss(predictions, targets)
        return mse.item()
    
    def compute_psnr(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio"""
        # Convert to numpy for skimage
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        psnr_values = []
        for i in range(pred_np.shape[0]):
            # PSNR for each image in batch
            psnr_val = psnr(target_np[i, 0], pred_np[i, 0], data_range=1.0)
            psnr_values.append(psnr_val)
        
        return np.mean(psnr_values)
    
    def compute_ssim(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Structural Similarity Index"""
        # Convert to numpy for skimage
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        ssim_values = []
        for i in range(pred_np.shape[0]):
            # SSIM for each image in batch
            ssim_val = ssim(target_np[i, 0], pred_np[i, 0], data_range=1.0)
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    def compute_fid(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Fréchet Inception Distance"""
        
        # Resize to 299x299 for Inception
        pred_resized = F.interpolate(predictions.repeat(1, 3, 1, 1), size=(299, 299), mode='bilinear')
        target_resized = F.interpolate(targets.repeat(1, 3, 1, 1), size=(299, 299), mode='bilinear')
        
        # Get Inception features
        with torch.no_grad():
            pred_features = self.inception_model(pred_resized)
            target_features = self.inception_model(target_resized)
        
        # Convert to numpy
        pred_features = pred_features.cpu().numpy()
        target_features = target_features.cpu().numpy()
        
        # Calculate FID
        fid_score = self._calculate_fid(pred_features, target_features)
        return fid_score
    
    def _calculate_fid(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate FID score between two sets of features"""
        
        # Calculate mean and covariance
        mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        
        # Handle numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)
    
    def compute_lpips(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute LPIPS (Learned Perceptual Image Patch Similarity)"""
        if self.lpips_model is None:
            return None
        
        # Convert to RGB for LPIPS
        pred_rgb = predictions.repeat(1, 3, 1, 1)
        target_rgb = targets.repeat(1, 3, 1, 1)
        
        # Resize to at least 64x64 for LPIPS
        pred_rgb = F.interpolate(pred_rgb, size=(64, 64), mode='bilinear')
        target_rgb = F.interpolate(target_rgb, size=(64, 64), mode='bilinear')
        
        # Normalize to [-1, 1] for LPIPS
        pred_rgb = pred_rgb * 2 - 1
        target_rgb = target_rgb * 2 - 1
        
        with torch.no_grad():
            lpips_scores = self.lpips_model(pred_rgb, target_rgb)
        
        return lpips_scores.mean().item()
    
    def compute_clip_score(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute CLIP similarity score"""
        if self.clip_model is None:
            return None
        
        # Convert to RGB and resize for CLIP
        pred_rgb = predictions.repeat(1, 3, 1, 1)
        target_rgb = targets.repeat(1, 3, 1, 1)
        
        # Resize to 224x224 for CLIP
        pred_rgb = F.interpolate(pred_rgb, size=(224, 224), mode='bilinear')
        target_rgb = F.interpolate(target_rgb, size=(224, 224), mode='bilinear')
        
        # Normalize for CLIP
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
        pred_rgb = normalize(pred_rgb)
        target_rgb = normalize(target_rgb)
        
        with torch.no_grad():
            # Get image features
            pred_features = self.clip_model.encode_image(pred_rgb)
            target_features = self.clip_model.encode_image(target_rgb)
            
            # Normalize features
            pred_features = pred_features / pred_features.norm(dim=-1, keepdim=True)
            target_features = target_features / target_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (pred_features * target_features).sum(dim=-1)
        
        return similarity.mean().item()
    
    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display"""
        
        formatted = []
        
        # MSE (lower is better)
        if 'mse' in metrics:
            formatted.append(f"MSE: {metrics['mse']:.6f}")
        
        # PSNR (higher is better)
        if 'psnr' in metrics:
            formatted.append(f"PSNR: {metrics['psnr']:.2f} dB")
        
        # SSIM (higher is better, 0-1)
        if 'ssim' in metrics:
            formatted.append(f"SSIM: {metrics['ssim']:.4f}")
        
        # FID (lower is better)
        if 'fid' in metrics:
            formatted.append(f"FID: {metrics['fid']:.2f}")
        
        # LPIPS (lower is better, 0-1)
        if 'lpips' in metrics and metrics['lpips'] is not None:
            formatted.append(f"LPIPS: {metrics['lpips']:.4f}")
        
        # CLIP Score (higher is better, -1 to 1)
        if 'clip_score' in metrics and metrics['clip_score'] is not None:
            formatted.append(f"CLIP: {metrics['clip_score']:.4f}")
        
        return " | ".join(formatted)

def install_missing_dependencies():
    """Install missing dependencies"""
    import subprocess
    import sys
    
    missing = []
    
    if not LPIPS_AVAILABLE:
        missing.append("lpips")
    
    if not CLIP_AVAILABLE:
        missing.append("git+https://github.com/openai/CLIP.git")
    
    if missing:
        print("📦 Installing missing dependencies...")
        for package in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
    else:
        print("✅ All dependencies available")

# Example usage
if __name__ == "__main__":
    # Install missing dependencies
    install_missing_dependencies()
    
    # Test metrics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = ComprehensiveMetrics(device=device)
    
    # Create dummy data
    batch_size = 4
    predictions = torch.rand(batch_size, 1, 28, 28).to(device)
    targets = torch.rand(batch_size, 1, 28, 28).to(device)
    
    # Compute all metrics
    results = metrics.compute_all_metrics(predictions, targets)
    
    print("\n📊 Comprehensive Metrics Test:")
    print("=" * 50)
    print(metrics.format_metrics(results))
    print("\n✅ All metrics computed successfully!")
