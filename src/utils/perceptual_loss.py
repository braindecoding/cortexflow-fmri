#!/usr/bin/env python3
"""
🎨 CortexFlow Perceptual Loss Implementation

Advanced perceptual loss using pre-trained VGG features for better visual quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG19 features.
    
    This loss compares high-level features extracted from a pre-trained VGG19 network
    rather than raw pixel values, leading to more perceptually meaningful comparisons.
    """
    
    def __init__(self, feature_layers=[2, 7, 12, 21, 30], use_normalization=True):
        """
        Initialize VGG Perceptual Loss.

        Args:
            feature_layers: List of VGG layer indices to extract features from
            use_normalization: Whether to normalize inputs to VGG range
        """
        super(VGGPerceptualLoss, self).__init__()

        # Load pre-trained VGG19 (fix deprecation warning)
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Freeze VGG parameters and fix inplace operations
        for param in vgg.parameters():
            param.requires_grad = False

        # Fix inplace ReLU operations to avoid gradient issues
        for module in vgg.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        # Store the full VGG model for feature extraction
        self.vgg_features = vgg
        self.feature_layers = feature_layers
        
        # Normalization for VGG input (ImageNet stats)
        self.use_normalization = use_normalization
        if use_normalization:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    
    def forward(self, pred, target):
        """
        Compute perceptual loss between predicted and target images.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
        
        Returns:
            Perceptual loss value
        """
        # Ensure inputs are in correct format
        if pred.size(1) == 1:  # Grayscale to RGB
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:  # Grayscale to RGB
            target = target.repeat(1, 3, 1, 1)
        
        # Resize to minimum VGG input size if needed
        if pred.size(-1) < 224:
            pred = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize inputs if required (only for RGB images)
        if self.use_normalization and pred.size(1) == 3:
            pred = self.normalize(pred)
            target = self.normalize(target)
        
        # Extract features and compute loss
        total_loss = 0.0
        feature_count = 0

        # Forward through VGG and extract features at specified layers
        # Process pred and target separately to avoid gradient conflicts

        # Forward pass for predicted images
        x_pred = pred
        pred_features = []
        for i, layer in enumerate(self.vgg_features):
            x_pred = layer(x_pred)
            if i in self.feature_layers:
                pred_features.append(x_pred)

        # Forward pass for target images (no gradients needed)
        with torch.no_grad():
            x_target = target
            target_features = []
            for i, layer in enumerate(self.vgg_features):
                x_target = layer(x_target)
                if i in self.feature_layers:
                    target_features.append(x_target)

        # Compute losses between corresponding features
        for pred_feat, target_feat in zip(pred_features, target_features):
            feature_loss = F.mse_loss(pred_feat, target_feat)
            total_loss += feature_loss
            feature_count += 1

        return total_loss / max(feature_count, 1)


class CombinedPerceptualLoss(nn.Module):
    """
    Combined loss function with multiple perceptual components.
    """
    
    def __init__(self, 
                 vgg_weight=1.0,
                 l1_weight=0.1,
                 gradient_weight=0.1,
                 ssim_weight=0.1):
        """
        Initialize combined perceptual loss.
        
        Args:
            vgg_weight: Weight for VGG perceptual loss
            l1_weight: Weight for L1 reconstruction loss
            gradient_weight: Weight for gradient loss
            ssim_weight: Weight for SSIM loss
        """
        super(CombinedPerceptualLoss, self).__init__()
        
        self.vgg_loss = VGGPerceptualLoss()
        self.vgg_weight = vgg_weight
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.ssim_weight = ssim_weight
    
    def gradient_loss(self, pred, target):
        """Compute gradient-based edge preservation loss."""
        # Compute gradients
        grad_x_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        grad_y_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_x_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        grad_y_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        return F.l1_loss(grad_x_pred, grad_x_target) + F.l1_loss(grad_y_pred, grad_y_target)
    
    def ssim_loss(self, pred, target, window_size=11):
        """Compute SSIM-based structural loss."""
        # Simplified SSIM implementation
        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def forward(self, pred, target):
        """
        Compute combined perceptual loss.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
        
        Returns:
            Dictionary with individual loss components and total loss
        """
        losses = {}
        
        # VGG Perceptual Loss (primary)
        if self.vgg_weight > 0:
            losses['vgg_loss'] = self.vgg_loss(pred, target)
        else:
            losses['vgg_loss'] = torch.tensor(0.0, device=pred.device)
        
        # L1 Reconstruction Loss (secondary)
        if self.l1_weight > 0:
            losses['l1_loss'] = F.l1_loss(pred, target)
        else:
            losses['l1_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Gradient Loss (edge preservation)
        if self.gradient_weight > 0:
            losses['gradient_loss'] = self.gradient_loss(pred, target)
        else:
            losses['gradient_loss'] = torch.tensor(0.0, device=pred.device)
        
        # SSIM Loss (structural similarity)
        if self.ssim_weight > 0:
            losses['ssim_loss'] = self.ssim_loss(pred, target)
        else:
            losses['ssim_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Compute total loss
        total_loss = (
            self.vgg_weight * losses['vgg_loss'] +
            self.l1_weight * losses['l1_loss'] +
            self.gradient_weight * losses['gradient_loss'] +
            self.ssim_weight * losses['ssim_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses


class LightweightPerceptualLoss(nn.Module):
    """
    Lightweight perceptual loss for faster training.
    Uses fewer VGG layers and simpler computations.
    """
    
    def __init__(self):
        super(LightweightPerceptualLoss, self).__init__()
        
        # Use only early VGG layers for speed
        vgg = models.vgg19(pretrained=True).features[:16]  # Up to conv3_4
        
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.feature_extractor = vgg
    
    def forward(self, pred, target):
        """Compute lightweight perceptual loss."""
        # Convert grayscale to RGB if needed
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)
        
        # Resize if too small
        if pred.size(-1) < 64:
            pred = F.interpolate(pred, size=(64, 64), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # Compute feature loss
        return F.mse_loss(pred_features, target_features)


# Factory function for easy usage
def create_perceptual_loss(loss_type='combined', **kwargs):
    """
    Factory function to create perceptual loss.
    
    Args:
        loss_type: 'vgg', 'combined', or 'lightweight'
        **kwargs: Additional arguments for loss function
    
    Returns:
        Perceptual loss module
    """
    if loss_type == 'vgg':
        return VGGPerceptualLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedPerceptualLoss(**kwargs)
    elif loss_type == 'lightweight':
        return LightweightPerceptualLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test perceptual loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    pred = torch.randn(2, 1, 28, 28).to(device)
    target = torch.randn(2, 1, 28, 28).to(device)
    
    # Test different loss types
    print("🧪 Testing Perceptual Loss Implementations")
    
    # VGG Perceptual Loss
    vgg_loss = VGGPerceptualLoss().to(device)
    loss_value = vgg_loss(pred, target)
    print(f"✅ VGG Perceptual Loss: {loss_value.item():.6f}")
    
    # Combined Perceptual Loss
    combined_loss = CombinedPerceptualLoss().to(device)
    losses = combined_loss(pred, target)
    print(f"✅ Combined Loss: {losses['total_loss'].item():.6f}")
    print(f"   - VGG: {losses['vgg_loss'].item():.6f}")
    print(f"   - L1: {losses['l1_loss'].item():.6f}")
    print(f"   - Gradient: {losses['gradient_loss'].item():.6f}")
    print(f"   - SSIM: {losses['ssim_loss'].item():.6f}")
    
    # Lightweight Perceptual Loss
    lightweight_loss = LightweightPerceptualLoss().to(device)
    loss_value = lightweight_loss(pred, target)
    print(f"✅ Lightweight Loss: {loss_value.item():.6f}")
    
    print("🎉 All perceptual loss tests passed!")
