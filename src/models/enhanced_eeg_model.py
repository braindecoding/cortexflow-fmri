#!/usr/bin/env python3
"""
🧠⚡ Enhanced EEG Model for Better Reconstruction

Advanced architecture with attention, multi-scale processing, and alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleConv1D(nn.Module):
    """Multi-scale 1D convolution for temporal feature extraction."""
    
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.out_per_kernel = out_channels // len(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, self.out_per_kernel,
                     kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        # Calculate actual output channels
        actual_out_channels = self.out_per_kernel * len(kernel_sizes)
        self.bn = nn.BatchNorm1d(actual_out_channels)
        
    def forward(self, x):
        # x: [batch, channels, time]
        conv_outputs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = self.bn(x)
        return F.relu(x)


class ChannelAttention(nn.Module):
    """Channel attention mechanism for EEG spatial processing."""
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        
    def forward(self, x):
        # x: [batch, channels, time]
        b, c, t = x.size()
        
        # Global pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Attention weights
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1)
        
        return x * attention


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for EEG time series."""
    
    def __init__(self, channels, seq_length):
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        
        self.scale = math.sqrt(channels)
        
    def forward(self, x):
        # x: [batch, channels, time]
        b, c, t = x.size()
        
        # Reshape for attention: [batch, time, channels]
        x = x.transpose(1, 2)  # [batch, time, channels]
        
        # Compute Q, K, V
        Q = self.query(x)  # [batch, time, channels]
        K = self.key(x)    # [batch, time, channels]
        V = self.value(x)  # [batch, time, channels]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Residual connection
        output = attended + x
        
        # Reshape back: [batch, channels, time]
        return output.transpose(1, 2)


class EnhancedEEGEncoder(nn.Module):
    """Enhanced EEG encoder with multi-scale processing and attention."""
    
    def __init__(self, channels, seq_length, d_model=256):
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Multi-scale temporal processing
        self.temporal_conv1 = MultiScaleConv1D(channels, 64)
        self.temporal_conv2 = MultiScaleConv1D(64, 128)
        self.temporal_conv3 = MultiScaleConv1D(128, 256)
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(256)
        self.temporal_attention = TemporalAttention(256, seq_length)
        
        # Frequency domain processing
        self.freq_conv = nn.Conv1d(256, 256, kernel_size=1)
        
        # Final projection
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        # x: [batch, channels, time]
        
        # Multi-scale temporal processing
        x = self.temporal_conv1(x)
        x = self.temporal_conv2(x)
        x = self.temporal_conv3(x)
        
        # Channel attention
        x = self.channel_attention(x)
        
        # Temporal attention
        x = self.temporal_attention(x)
        
        # Frequency domain enhancement
        # Apply FFT, process, and convert back
        x_freq = torch.fft.fft(x, dim=-1)
        x_freq_real = x_freq.real
        x_freq_enhanced = self.freq_conv(x_freq_real)
        x = x + x_freq_enhanced  # Residual connection
        
        # Final projection
        features = self.projection(x)
        
        return features


class ProgressiveImageDecoder(nn.Module):
    """Progressive image decoder with skip connections."""
    
    def __init__(self, d_model=256, image_size=28):
        super().__init__()
        self.d_model = d_model
        self.image_size = image_size
        
        # Progressive upsampling
        self.fc1 = nn.Linear(d_model, 7 * 7 * 128)
        self.bn1 = nn.BatchNorm2d(128)
        
        # Decoder blocks with skip connections
        self.decoder_blocks = nn.ModuleList([
            # 7x7 -> 14x14
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            # 14x14 -> 28x28
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        ])
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Skip connection projections
        self.skip_projections = nn.ModuleList([
            nn.Conv2d(128, 64, kernel_size=1),  # For first skip
            nn.Conv2d(64, 32, kernel_size=1)    # For second skip
        ])
        
    def forward(self, x):
        # x: [batch, d_model]
        
        # Initial projection
        x = self.fc1(x)
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.bn1(x))
        
        skip_connections = [x]
        
        # Progressive decoding with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x)
            
            # Add skip connection from previous layer
            if i > 0 and i < len(skip_connections):
                skip = skip_connections[i]
                skip_proj = self.skip_projections[i-1](skip)
                # Resize skip connection to match current size
                skip_proj = F.interpolate(skip_proj, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip_proj
            
            skip_connections.append(x)
        
        # Final output
        output = self.final_conv(x)
        
        return output


class EnhancedEEGCortexFlow(nn.Module):
    """Enhanced EEG CortexFlow model with advanced preprocessing and architecture."""
    
    def __init__(self, channels, seq_length, image_size=28, d_model=256):
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        self.image_size = image_size
        self.d_model = d_model
        
        # Enhanced encoder
        self.encoder = EnhancedEEGEncoder(channels, seq_length, d_model)
        
        # Progressive decoder
        self.decoder = ProgressiveImageDecoder(d_model, image_size)
        
        # Feature alignment layer
        self.alignment_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        # x: [batch, channels, time]
        
        # Encode EEG features
        features = self.encoder(x)
        
        # Feature alignment
        aligned_features = self.alignment_layer(features)
        
        # Decode to image
        image = self.decoder(aligned_features)
        
        return image


class EnhancedEEGLoss(nn.Module):
    """Enhanced loss function for EEG reconstruction."""
    
    def __init__(self, mse_weight=1.0, perceptual_weight=0.2, consistency_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.consistency_weight = consistency_weight
        
        # Simple perceptual features (edge detection)
        self.edge_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                       dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, predictions, targets):
        # Ensure predictions and targets have same shape
        if predictions.shape != targets.shape:
            if len(targets.shape) == 2:  # [batch, 784]
                targets = targets.view(-1, 1, 28, 28)
            elif len(targets.shape) == 3:  # [batch, 28, 28]
                targets = targets.unsqueeze(1)
        
        # MSE Loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Perceptual Loss (edge consistency)
        if predictions.device != self.edge_kernel.device:
            self.edge_kernel = self.edge_kernel.to(predictions.device)
        
        pred_edges = F.conv2d(predictions, self.edge_kernel, padding=1)
        target_edges = F.conv2d(targets, self.edge_kernel, padding=1)
        perceptual_loss = F.mse_loss(pred_edges, target_edges)
        
        # Consistency Loss (spatial smoothness)
        pred_diff_h = torch.abs(predictions[:, :, 1:, :] - predictions[:, :, :-1, :])
        pred_diff_w = torch.abs(predictions[:, :, :, 1:] - predictions[:, :, :, :-1])
        consistency_loss = torch.mean(pred_diff_h) + torch.mean(pred_diff_w)
        
        # Total loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.perceptual_weight * perceptual_loss + 
                     self.consistency_weight * consistency_loss)
        
        return total_loss, {
            'mse': mse_loss,
            'perceptual': perceptual_loss,
            'consistency': consistency_loss,
            'total': total_loss
        }
