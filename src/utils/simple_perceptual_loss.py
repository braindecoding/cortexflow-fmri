#!/usr/bin/env python3
"""
🎨 CortexFlow Simple Perceptual Loss Implementation

Lightweight perceptual loss without requiring VGG download for immediate testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePerceptualLoss(nn.Module):
    """
    Simple perceptual loss using gradient and structural features.
    
    This is a lightweight alternative that doesn't require pre-trained models
    but still provides better perceptual quality than MSE.
    """
    
    def __init__(self, 
                 gradient_weight=1.0,
                 l1_weight=0.5,
                 ssim_weight=0.3,
                 edge_weight=0.2):
        """
        Initialize simple perceptual loss.
        
        Args:
            gradient_weight: Weight for gradient loss (edge preservation)
            l1_weight: Weight for L1 reconstruction loss
            ssim_weight: Weight for SSIM structural loss
            edge_weight: Weight for edge enhancement loss
        """
        super(SimplePerceptualLoss, self).__init__()
        
        self.gradient_weight = gradient_weight
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        
        # Sobel edge detection kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def gradient_loss(self, pred, target):
        """Compute gradient-based edge preservation loss."""
        # Compute gradients using simple differences
        grad_x_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        grad_y_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_x_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        grad_y_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        return F.l1_loss(grad_x_pred, grad_x_target) + F.l1_loss(grad_y_pred, grad_y_target)
    
    def edge_loss(self, pred, target):
        """Compute edge-based loss using Sobel operators."""
        # Apply Sobel filters
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_edge_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1)
        
        # Compute edge magnitude
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-8)
        
        return F.l1_loss(pred_edge, target_edge)
    
    def ssim_loss(self, pred, target, window_size=11, sigma=1.5):
        """Compute SSIM-based structural loss."""
        # Create Gaussian window
        coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
        coords -= window_size // 2
        
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        
        window = g.outer(g).unsqueeze(0).unsqueeze(0)
        window = window.expand(pred.size(1), 1, -1, -1)
        
        # Compute SSIM components
        mu1 = F.conv2d(pred, window, padding=window_size//2, groups=pred.size(1))
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=target.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=pred.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=target.size(1)) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=pred.size(1)) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def forward(self, pred, target):
        """
        Compute simple perceptual loss.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
        
        Returns:
            Dictionary with individual loss components and total loss
        """
        losses = {}
        
        # L1 Reconstruction Loss
        if self.l1_weight > 0:
            losses['l1_loss'] = F.l1_loss(pred, target)
        else:
            losses['l1_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Gradient Loss (edge preservation)
        if self.gradient_weight > 0:
            losses['gradient_loss'] = self.gradient_loss(pred, target)
        else:
            losses['gradient_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Edge Loss (Sobel-based)
        if self.edge_weight > 0:
            losses['edge_loss'] = self.edge_loss(pred, target)
        else:
            losses['edge_loss'] = torch.tensor(0.0, device=pred.device)
        
        # SSIM Loss (structural similarity)
        if self.ssim_weight > 0:
            losses['ssim_loss'] = self.ssim_loss(pred, target)
        else:
            losses['ssim_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Compute total loss
        total_loss = (
            self.l1_weight * losses['l1_loss'] +
            self.gradient_weight * losses['gradient_loss'] +
            self.edge_weight * losses['edge_loss'] +
            self.ssim_weight * losses['ssim_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses


class AdvancedSimplePerceptualLoss(nn.Module):
    """
    Advanced simple perceptual loss with frequency domain components.
    """
    
    def __init__(self, 
                 gradient_weight=1.0,
                 l1_weight=0.3,
                 ssim_weight=0.2,
                 edge_weight=0.2,
                 frequency_weight=0.3):
        """Initialize advanced simple perceptual loss."""
        super(AdvancedSimplePerceptualLoss, self).__init__()
        
        self.simple_loss = SimplePerceptualLoss(gradient_weight, l1_weight, ssim_weight, edge_weight)
        self.frequency_weight = frequency_weight
    
    def frequency_loss(self, pred, target):
        """Compute frequency domain loss using FFT."""
        # Apply 2D FFT
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Compute magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Focus on low-frequency components (structure)
        h, w = pred_mag.shape[-2:]
        center_h, center_w = h // 2, w // 2
        
        # Create low-frequency mask
        mask = torch.zeros_like(pred_mag)
        mask[:, :, center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 1.0
        
        # Apply mask and compute loss
        pred_low_freq = pred_mag * mask
        target_low_freq = target_mag * mask
        
        return F.l1_loss(pred_low_freq, target_low_freq)
    
    def forward(self, pred, target):
        """Compute advanced simple perceptual loss."""
        # Get simple perceptual loss components
        losses = self.simple_loss(pred, target)
        
        # Add frequency domain loss
        if self.frequency_weight > 0:
            losses['frequency_loss'] = self.frequency_loss(pred, target)
        else:
            losses['frequency_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Recompute total loss
        losses['total_loss'] = (
            losses['total_loss'] + 
            self.frequency_weight * losses['frequency_loss']
        )
        
        return losses


# Factory function for easy usage
def create_simple_perceptual_loss(loss_type='simple', **kwargs):
    """
    Factory function to create simple perceptual loss.
    
    Args:
        loss_type: 'simple' or 'advanced'
        **kwargs: Additional arguments for loss function
    
    Returns:
        Simple perceptual loss module
    """
    if loss_type == 'simple':
        return SimplePerceptualLoss(**kwargs)
    elif loss_type == 'advanced':
        return AdvancedSimplePerceptualLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test simple perceptual loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    pred = torch.randn(2, 1, 28, 28).to(device)
    target = torch.randn(2, 1, 28, 28).to(device)
    
    print("🧪 Testing Simple Perceptual Loss Implementations")
    
    # Simple Perceptual Loss
    simple_loss = SimplePerceptualLoss().to(device)
    losses = simple_loss(pred, target)
    print(f"✅ Simple Perceptual Loss: {losses['total_loss'].item():.6f}")
    print(f"   - L1: {losses['l1_loss'].item():.6f}")
    print(f"   - Gradient: {losses['gradient_loss'].item():.6f}")
    print(f"   - Edge: {losses['edge_loss'].item():.6f}")
    print(f"   - SSIM: {losses['ssim_loss'].item():.6f}")
    
    # Advanced Simple Perceptual Loss
    advanced_loss = AdvancedSimplePerceptualLoss().to(device)
    losses = advanced_loss(pred, target)
    print(f"✅ Advanced Simple Loss: {losses['total_loss'].item():.6f}")
    print(f"   - Frequency: {losses['frequency_loss'].item():.6f}")
    
    print("🎉 All simple perceptual loss tests passed!")
