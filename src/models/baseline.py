"""
Standard Baseline CNN for Neural Decoding
========================================

Standard CNN baseline implementation for fair comparison in neural decoding tasks.
This represents a typical approach found in neural decoding literature.

Architecture:
    - MLP projection: fMRI signals -> 784-dimensional vector
    - CNN processing: Refine and denoise projected features
    - Output: (1, 28, 28) reconstructed image

Features:
    - Standard MLP-to-CNN pipeline
    - BatchNormalization for regularization
    - ReLU activations
    - Dropout for overfitting prevention
    - Progressive channel reduction
    - Sigmoid output for [0,1] pixel values
"""

import torch
import torch.nn as nn


class StandardBaselineCNN(nn.Module):
    """CortexFlow Lite: Lightweight CNN for Neural Decoding (Efficient Implementation)"""

    def __init__(self, input_dim, device='cuda'):
        super(StandardBaselineCNN, self).__init__()
        self.name = "CortexFlow-Lite"
        self.device = device

        # Standard MLP projection (common baseline approach)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 784),
            nn.ReLU(inplace=True)
        ).to(device)

        # Standard CNN processing (common in neural decoding literature)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        """
        Forward pass through the baseline CNN.
        
        Args:
            x: Input fMRI signals [batch_size, input_dim]
            
        Returns:
            Reconstructed images [batch_size, 1, 28, 28]
        """
        projected = self.projection(x)
        reshaped = projected.view(-1, 1, 28, 28)
        output = self.cnn(reshaped)
        return output
