#!/usr/bin/env python3
"""
🧠⚡ FID Calculator for CortexFlow

Fréchet Inception Distance calculation for brain-to-image reconstruction evaluation.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from scipy import linalg
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


class InceptionFeatureExtractor(nn.Module):
    """Feature extractor using pre-trained Inception v3."""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Load pre-trained Inception v3
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # Remove final classification layer
        self.inception.eval()
        self.inception.to(device)
        
        # Preprocessing for Inception
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, x):
        """Extract features from images."""
        # x shape: [batch, channels, height, width]
        if x.shape[1] == 1:  # Grayscale to RGB
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")
        
        # Preprocess
        x = self.preprocess(x)
        
        # Extract features
        with torch.no_grad():
            features = self.inception(x)
        
        return features


class FIDCalculator:
    """Calculate Fréchet Inception Distance between real and generated images."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.feature_extractor = InceptionFeatureExtractor(device)
        
    def preprocess_images(self, images):
        """Preprocess images for FID calculation."""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        
        # Ensure correct shape and range
        if images.dim() == 3:  # [batch, height, width]
            images = images.unsqueeze(1)  # Add channel dimension
        
        if images.dim() == 2:  # [height, width]
            images = images.unsqueeze(0).unsqueeze(0)  # Add batch and channel
        
        # Ensure range [0, 1]
        if images.max() > 1.0:
            images = images / 255.0
        
        # Resize to at least 28x28 for better feature extraction
        if images.shape[-1] < 28:
            images = torch.nn.functional.interpolate(
                images, size=(28, 28), mode='bilinear', align_corners=False
            )
        
        return images.to(self.device)
    
    def extract_features(self, images):
        """Extract Inception features from images."""
        images = self.preprocess_images(images)

        # Process in batches to avoid memory issues
        batch_size = min(32, len(images))
        features_list = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_features = self.feature_extractor(batch)
            features_list.append(batch_features.cpu())

        features = torch.cat(features_list, dim=0)
        return features.numpy()
    
    def calculate_statistics(self, features):
        """Calculate mean and covariance of features with robust handling."""
        mu = np.mean(features, axis=0)

        # Handle small sample size
        n_samples, n_features = features.shape

        if n_samples < n_features:
            print(f"⚠️ Warning: Small sample size ({n_samples}) vs features ({n_features})")
            print("   Using regularized covariance calculation...")

            # Use regularized covariance for small samples
            sigma = np.cov(features, rowvar=False)

            # Add regularization to diagonal
            reg_factor = 1e-6
            sigma += reg_factor * np.eye(sigma.shape[0])

        else:
            # Standard covariance calculation
            sigma = np.cov(features, rowvar=False)

        return mu, sigma
    
    def calculate_fid(self, real_images, generated_images):
        """Calculate FID between real and generated images."""
        print("🔍 Extracting features from real images...")
        real_features = self.extract_features(real_images)
        
        print("🔍 Extracting features from generated images...")
        generated_features = self.extract_features(generated_images)
        
        print("📊 Calculating statistics...")
        mu1, sigma1 = self.calculate_statistics(real_features)
        mu2, sigma2 = self.calculate_statistics(generated_features)
        
        print("🧮 Computing FID...")
        fid = self.compute_fid(mu1, sigma1, mu2, sigma2)
        
        return fid
    
    def compute_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Compute FID given statistics of two distributions with robust handling."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, f"Mean shapes don't match: {mu1.shape} vs {mu2.shape}"
        assert sigma1.shape == sigma2.shape, f"Covariance shapes don't match: {sigma1.shape} vs {sigma2.shape}"

        diff = mu1 - mu2

        # Add regularization for numerical stability
        reg_eps = max(eps, 1e-6)
        sigma1_reg = sigma1 + reg_eps * np.eye(sigma1.shape[0])
        sigma2_reg = sigma2 + reg_eps * np.eye(sigma2.shape[0])

        try:
            # Product might be almost singular
            covmean, _ = linalg.sqrtm(sigma1_reg.dot(sigma2_reg), disp=False)

            if not np.isfinite(covmean).all():
                print(f"⚠️ Warning: Non-finite covmean, using larger regularization")
                larger_eps = reg_eps * 100
                sigma1_reg = sigma1 + larger_eps * np.eye(sigma1.shape[0])
                sigma2_reg = sigma2 + larger_eps * np.eye(sigma2.shape[0])
                covmean, _ = linalg.sqrtm(sigma1_reg.dot(sigma2_reg), disp=False)

            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    print(f"⚠️ Warning: Large imaginary component, taking real part")
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            fid = (diff.dot(diff) + np.trace(sigma1_reg) +
                   np.trace(sigma2_reg) - 2 * tr_covmean)

            return max(0, fid)  # Ensure non-negative

        except Exception as e:
            print(f"❌ FID computation failed: {e}")
            print("   Falling back to simplified distance metric...")

            # Fallback: Use Wasserstein-1 distance approximation
            return np.linalg.norm(mu1 - mu2) + np.trace(sigma1 + sigma2 - 2 * np.sqrt(np.diag(sigma1) * np.diag(sigma2)))


def calculate_fid_for_reconstruction(targets, predictions, device='cuda'):
    """
    Calculate FID for brain-to-image reconstruction.
    
    Args:
        targets: Ground truth images [batch, height, width] or [batch, channels, height, width]
        predictions: Reconstructed images [batch, height, width] or [batch, channels, height, width]
        device: Device to use for computation
    
    Returns:
        fid_score: FID score (lower is better)
    """
    calculator = FIDCalculator(device=device)
    
    try:
        fid_score = calculator.calculate_fid(targets, predictions)
        print(f"✅ FID Score: {fid_score:.4f}")
        return fid_score
    except Exception as e:
        print(f"❌ FID calculation failed: {e}")
        return float('inf')


def calculate_comprehensive_metrics_with_fid(predictions, targets, device='cuda'):
    """
    Calculate comprehensive metrics including FID.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        device: Device for computation
    
    Returns:
        dict: Comprehensive metrics including FID
    """
    import scipy.stats
    
    # Convert to numpy
    if torch.is_tensor(predictions):
        pred_np = predictions.cpu().numpy()
    else:
        pred_np = predictions
    
    if torch.is_tensor(targets):
        target_np = targets.cpu().numpy()
    else:
        target_np = targets
    
    # Flatten for pixel-wise metrics
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Basic metrics
    mse = np.mean((pred_flat - target_flat) ** 2)
    correlation = scipy.stats.pearsonr(pred_flat, target_flat)[0]
    
    # SSIM (simplified)
    def ssim_simple(img1, img2):
        mu1, mu2 = img1.mean(), img2.mean()
        sigma1, sigma2 = img1.std(), img2.std()
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        return ssim
    
    # Calculate SSIM for each sample
    ssim_scores = []
    for i in range(pred_np.shape[0]):
        if pred_np.shape[1] == 1:  # [batch, 1, 28, 28]
            pred_img = pred_np[i, 0]
            target_img = target_np[i].reshape(28, 28) if target_np.ndim == 2 else target_np[i, 0]
        else:  # [batch, 784] or [batch, 28, 28]
            pred_img = pred_np[i].reshape(28, 28) if pred_np[i].ndim == 1 else pred_np[i]
            target_img = target_np[i].reshape(28, 28) if target_np[i].ndim == 1 else target_np[i]
        
        ssim_score = ssim_simple(pred_img, target_img)
        ssim_scores.append(ssim_score)
    
    ssim = np.mean(ssim_scores)
    
    # PSNR
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # FID
    print("🔍 Calculating FID...")
    fid = calculate_fid_for_reconstruction(targets, predictions, device)
    
    return {
        'mse': mse,
        'correlation': correlation,
        'ssim': ssim,
        'psnr': psnr,
        'fid': fid
    }


if __name__ == "__main__":
    # Test FID calculation
    print("🧠⚡ Testing FID Calculator")
    
    # Create dummy data
    batch_size = 10
    real_images = torch.randn(batch_size, 1, 28, 28)
    fake_images = torch.randn(batch_size, 1, 28, 28)
    
    # Calculate FID
    fid_score = calculate_fid_for_reconstruction(real_images, fake_images)
    print(f"Test FID Score: {fid_score:.4f}")
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics_with_fid(fake_images, real_images)
    print("Test Metrics:", metrics)
