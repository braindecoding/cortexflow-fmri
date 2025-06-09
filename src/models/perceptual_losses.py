#!/usr/bin/env python3
"""
Perceptual Loss Functions for High-Quality Neural Decoding
Addresses blocky reconstruction issues with advanced loss formulations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, Tuple
import numpy as np

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using pre-trained VGG features
    Addresses blocky reconstruction by comparing high-level features
    """
    
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        
        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.vgg = vgg.eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Layer mapping
        self.layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
            'relu4_1': 18, 'relu4_2': 20, 'relu4_3': 22,
            'relu5_1': 25, 'relu5_2': 27, 'relu5_3': 29
        }
        
        self.target_layers = [self.layer_map[layer] for layer in layers]
        self.weights = [1.0, 1.0, 1.0, 1.0]  # Equal weights for all layers
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between prediction and target
        
        Args:
            pred: Predicted images [B, 784] -> reshape to [B, 1, 28, 28]
            target: Target images [B, 784] -> reshape to [B, 1, 28, 28]
        """
        
        # Reshape to image format
        pred_img = pred.view(-1, 1, 28, 28)
        target_img = target.view(-1, 1, 28, 28)
        
        # Convert grayscale to RGB for VGG
        pred_rgb = pred_img.repeat(1, 3, 1, 1)
        target_rgb = target_img.repeat(1, 3, 1, 1)
        
        # Resize to minimum VGG input size (224x224)
        pred_rgb = F.interpolate(pred_rgb, size=(224, 224), mode='bilinear', align_corners=False)
        target_rgb = F.interpolate(target_rgb, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize for VGG (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred_rgb - mean) / std
        target_norm = (target_rgb - mean) / std
        
        # Extract features
        pred_features = self._extract_features(pred_norm)
        target_features = self._extract_features(target_norm)
        
        # Compute perceptual loss
        perceptual_loss = 0.0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            perceptual_loss += self.weights[i] * F.mse_loss(pred_feat, target_feat)
        
        return perceptual_loss
    
    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract features from target layers"""
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.target_layers:
                features.append(x)
        return features

class StructuralSimilarityLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss
    Preserves structural information in reconstructions
    """
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 1
        
        # Create Gaussian window
        self.window = self._create_window(window_size, sigma)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM for minimization)
        """
        # Reshape to image format
        pred_img = pred.view(-1, 1, 28, 28)
        target_img = target.view(-1, 1, 28, 28)
        
        # Move window to correct device
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
        
        # Compute SSIM
        ssim_value = self._ssim(pred_img, target_img)
        
        # Return 1 - SSIM for loss minimization
        return 1 - ssim_value
    
    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian window for SSIM computation"""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        window = g.outer(g).unsqueeze(0).unsqueeze(0)
        return window
    
    def _ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between two images"""
        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()

class GradientLoss(nn.Module):
    """
    Gradient Loss for preserving edge information
    Helps maintain sharp edges in reconstructions
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss between prediction and target
        """
        # Reshape to image format
        pred_img = pred.view(-1, 1, 28, 28)
        target_img = target.view(-1, 1, 28, 28)
        
        # Compute gradients
        pred_grad_x = torch.abs(pred_img[:, :, :, :-1] - pred_img[:, :, :, 1:])
        pred_grad_y = torch.abs(pred_img[:, :, :-1, :] - pred_img[:, :, 1:, :])
        
        target_grad_x = torch.abs(target_img[:, :, :, :-1] - target_img[:, :, :, 1:])
        target_grad_y = torch.abs(target_img[:, :, :-1, :] - target_img[:, :, 1:, :])
        
        # Compute gradient loss
        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y

class FrequencyLoss(nn.Module):
    """
    Frequency Domain Loss using FFT
    Preserves frequency characteristics of images
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency domain loss
        """
        # Reshape to image format
        pred_img = pred.view(-1, 1, 28, 28)
        target_img = target.view(-1, 1, 28, 28)
        
        # Compute FFT
        pred_fft = torch.fft.fft2(pred_img)
        target_fft = torch.fft.fft2(target_img)
        
        # Compute magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Frequency loss
        freq_loss = F.mse_loss(pred_mag, target_mag)
        
        return freq_loss

class ComprehensiveReconstructionLoss(nn.Module):
    """
    Comprehensive loss combining multiple objectives
    Addresses blocky reconstruction with perceptual quality
    """
    
    def __init__(self, 
                 lambda_mse: float = 1.0,
                 lambda_perceptual: float = 0.1,
                 lambda_ssim: float = 0.1,
                 lambda_gradient: float = 0.05,
                 lambda_frequency: float = 0.02):
        super().__init__()
        
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_gradient = lambda_gradient
        self.lambda_frequency = lambda_frequency
        
        # Initialize loss components
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.ssim_loss = StructuralSimilarityLoss()
        self.gradient_loss = GradientLoss()
        self.frequency_loss = FrequencyLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive reconstruction loss
        
        Returns:
            Dictionary with individual loss components and total loss
        """
        
        # Individual loss components
        mse = self.mse_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        gradient = self.gradient_loss(pred, target)
        frequency = self.frequency_loss(pred, target)
        
        # Total comprehensive loss
        total_loss = (self.lambda_mse * mse +
                     self.lambda_perceptual * perceptual +
                     self.lambda_ssim * ssim +
                     self.lambda_gradient * gradient +
                     self.lambda_frequency * frequency)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse,
            'perceptual_loss': perceptual,
            'ssim_loss': ssim,
            'gradient_loss': gradient,
            'frequency_loss': frequency
        }

class AdaptiveReconstructionLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress
    """
    
    def __init__(self):
        super().__init__()
        self.comprehensive_loss = ComprehensiveReconstructionLoss()
        self.epoch = 0
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive loss with epoch-dependent weighting
        """
        
        # Adjust weights based on training progress
        if self.epoch < 10:
            # Early training: Focus on MSE
            self.comprehensive_loss.lambda_mse = 1.0
            self.comprehensive_loss.lambda_perceptual = 0.05
            self.comprehensive_loss.lambda_ssim = 0.05
        elif self.epoch < 20:
            # Mid training: Increase perceptual weight
            self.comprehensive_loss.lambda_mse = 0.8
            self.comprehensive_loss.lambda_perceptual = 0.15
            self.comprehensive_loss.lambda_ssim = 0.1
        else:
            # Late training: Full perceptual focus
            self.comprehensive_loss.lambda_mse = 0.6
            self.comprehensive_loss.lambda_perceptual = 0.2
            self.comprehensive_loss.lambda_ssim = 0.15
        
        loss_dict = self.comprehensive_loss(pred, target)
        return loss_dict['total_loss']
    
    def update_epoch(self, epoch: int):
        """Update current epoch for adaptive weighting"""
        self.epoch = epoch

# Utility function for easy integration
def create_advanced_loss(loss_type: str = 'comprehensive'):
    """
    Factory function to create advanced loss functions
    
    Args:
        loss_type: 'comprehensive', 'perceptual', 'ssim', 'adaptive'
    """
    
    if loss_type == 'comprehensive':
        return ComprehensiveReconstructionLoss()
    elif loss_type == 'perceptual':
        return PerceptualLoss()
    elif loss_type == 'ssim':
        return StructuralSimilarityLoss()
    elif loss_type == 'adaptive':
        return AdaptiveReconstructionLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
