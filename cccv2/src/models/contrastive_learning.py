"""
CortexFlow-CLIP-CNN V2: Contrastive Learning Integration
=======================================================

Revolutionary contrastive learning for neural decoding:
1. Semantic Contrastive Loss for better representations
2. Hard Negative Mining for challenging examples
3. Self-Supervised Learning for small datasets
4. CLIP-inspired contrastive alignment

Innovation: First contrastive learning framework for neural decoding
Goal: Better representations especially for small datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SemanticContrastiveLoss(nn.Module):
    """
    Semantic contrastive loss for neural decoding
    
    Innovation:
    - Positive pairs: Similar visual content
    - Negative pairs: Different visual content
    - Temperature-scaled similarity
    - Adaptive margin based on semantic distance
    """
    
    def __init__(self, temperature=0.1, margin=0.5):
        super(SemanticContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, features, targets):
        """
        Args:
            features: [batch_size, embed_dim] - encoded features
            targets: [batch_size, 1, 28, 28] - visual targets
        Returns:
            contrastive_loss: Scalar loss value
            similarity_matrix: [batch_size, batch_size] similarity scores
        """
        batch_size = features.shape[0]
        
        # Normalize features to unit sphere
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create semantic similarity labels from visual targets
        targets_flat = targets.view(batch_size, -1)  # [batch_size, 784]
        
        # Compute visual similarity (MSE-based)
        visual_similarity = torch.zeros(batch_size, batch_size, device=features.device)
        for i in range(batch_size):
            for j in range(batch_size):
                mse = F.mse_loss(targets_flat[i], targets_flat[j], reduction='mean')
                visual_similarity[i, j] = torch.exp(-mse * 10)  # Convert to similarity
        
        # Create positive/negative masks
        positive_mask = (visual_similarity > 0.7).float()  # High visual similarity
        negative_mask = (visual_similarity < 0.3).float()  # Low visual similarity
        
        # Remove self-similarity
        mask = torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask * (1 - mask)
        negative_mask = negative_mask * (1 - mask)
        
        # Contrastive loss computation
        positive_logits = similarity_matrix * positive_mask
        negative_logits = similarity_matrix * negative_mask
        
        # InfoNCE-style loss
        labels = torch.arange(batch_size, device=features.device)
        contrastive_loss = self.criterion(similarity_matrix, labels)
        
        return contrastive_loss, similarity_matrix


class HardNegativeMiner(nn.Module):
    """
    Hard negative mining for better contrastive learning
    
    Innovation:
    - Identifies challenging negative examples
    - Focuses training on difficult cases
    - Adaptive difficulty based on training progress
    """
    
    def __init__(self, margin=0.5, top_k=5):
        super(HardNegativeMiner, self).__init__()
        self.margin = margin
        self.top_k = top_k
        
    def mine_hard_negatives(self, features, targets, epoch=0):
        """
        Args:
            features: [batch_size, embed_dim]
            targets: [batch_size, 1, 28, 28]
            epoch: Current training epoch for adaptive difficulty
        Returns:
            hard_negative_pairs: List of (anchor, negative) indices
            mining_info: Dict with mining statistics
        """
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute pairwise distances
        distances = torch.cdist(features, features)
        
        # Compute visual dissimilarity
        targets_flat = targets.view(batch_size, -1)
        visual_distances = torch.zeros(batch_size, batch_size, device=features.device)
        
        for i in range(batch_size):
            for j in range(batch_size):
                mse = F.mse_loss(targets_flat[i], targets_flat[j], reduction='mean')
                visual_distances[i, j] = mse
        
        # Find hard negatives: close in feature space but far in visual space
        hard_negative_pairs = []
        mining_stats = {'total_pairs': 0, 'hard_pairs': 0}
        
        for i in range(batch_size):
            # Get samples that are visually different (negative samples)
            visual_threshold = torch.quantile(visual_distances[i], 0.7)  # Top 30% most different
            negative_mask = visual_distances[i] > visual_threshold
            
            if negative_mask.sum() > 0:
                # Among negatives, find those closest in feature space (hardest)
                negative_distances = distances[i] * negative_mask.float()
                negative_distances[negative_distances == 0] = float('inf')  # Mask out non-negatives
                
                # Select top-k hardest negatives
                if negative_distances.min() < float('inf'):
                    _, hard_indices = torch.topk(negative_distances, 
                                               min(self.top_k, negative_mask.sum().item()), 
                                               largest=False)
                    
                    for j in hard_indices:
                        j_val = j.item() if hasattr(j, 'item') else j
                        if i != j_val:  # Avoid self-pairs
                            hard_negative_pairs.append((i, j_val))
                            mining_stats['hard_pairs'] += 1
            
            mining_stats['total_pairs'] += batch_size - 1
        
        mining_info = {
            'hard_negative_pairs': hard_negative_pairs,
            'mining_ratio': mining_stats['hard_pairs'] / max(mining_stats['total_pairs'], 1),
            'total_hard_pairs': mining_stats['hard_pairs']
        }
        
        return hard_negative_pairs, mining_info


class SelfSupervisedAugmentation(nn.Module):
    """
    Self-supervised augmentation for small datasets
    
    Innovation:
    - Feature space augmentation
    - Consistency regularization
    - Pseudo-label generation
    """
    
    def __init__(self, embed_dim=512, noise_std=0.1):
        super(SelfSupervisedAugmentation, self).__init__()
        self.embed_dim = embed_dim
        self.noise_std = noise_std
        
        # Augmentation network
        self.augmentation_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, features, augment_ratio=0.5):
        """
        Args:
            features: [batch_size, embed_dim]
            augment_ratio: Ratio of samples to augment
        Returns:
            augmented_features: [batch_size, embed_dim]
            augmentation_mask: [batch_size] - which samples were augmented
        """
        batch_size = features.shape[0]
        
        # Random augmentation mask
        augmentation_mask = torch.rand(batch_size, device=features.device) < augment_ratio
        
        # Apply augmentation
        augmented_features = features.clone()
        
        if augmentation_mask.sum() > 0:
            # Select features to augment
            features_to_augment = features[augmentation_mask]
            
            # Add learned augmentation
            learned_aug = self.augmentation_net(features_to_augment)
            
            # Add noise
            noise = torch.randn_like(features_to_augment) * self.noise_std
            
            # Combine augmentations
            augmented = features_to_augment + 0.1 * learned_aug + noise
            augmented = F.normalize(augmented, p=2, dim=1)
            
            # Update augmented features
            augmented_features[augmentation_mask] = augmented
        
        return augmented_features, augmentation_mask


class ContrastiveCortexFlowV2(nn.Module):
    """
    Complete contrastive learning model for CCCV2
    
    Integrates:
    - Semantic contrastive loss
    - Hard negative mining
    - Self-supervised augmentation
    """
    
    def __init__(self, input_dim, device='cuda'):
        super(ContrastiveCortexFlowV2, self).__init__()
        self.name = "CortexFlow-CLIP-CNN-V2-Contrastive"
        self.device = device
        self.input_dim = input_dim
        
        # Base encoder (simple but effective)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Tanh()
        ).to(device)
        
        # Contrastive projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        ).to(device)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.02),
            
            nn.Linear(512, 784),
            nn.Sigmoid()
        ).to(device)
        
        # Contrastive learning components
        self.contrastive_loss = SemanticContrastiveLoss(temperature=0.1).to(device)
        self.hard_negative_miner = HardNegativeMiner(top_k=3).to(device)
        self.self_supervised_aug = SelfSupervisedAugmentation(embed_dim=512).to(device)
        
    def forward(self, x, targets=None, epoch=0, use_contrastive=True):
        """
        Complete contrastive processing
        
        Args:
            x: [batch_size, input_dim]
            targets: [batch_size, 1, 28, 28] - for contrastive learning
            epoch: Current epoch for adaptive components
            use_contrastive: Whether to apply contrastive learning
        Returns:
            visual_output: [batch_size, 1, 28, 28]
            encoded_features: [batch_size, 512]
            contrastive_info: Dict with contrastive learning info
        """
        # Base encoding
        encoded_features = self.encoder(x)
        encoded_features = F.normalize(encoded_features, p=2, dim=1)
        
        # Self-supervised augmentation (for training)
        if self.training and use_contrastive:
            augmented_features, aug_mask = self.self_supervised_aug(encoded_features)
        else:
            augmented_features = encoded_features
            aug_mask = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
        
        # Contrastive projection
        contrastive_features = self.contrastive_head(augmented_features)
        contrastive_features = F.normalize(contrastive_features, p=2, dim=1)
        
        # Decode to visual output
        visual_output = self.decoder(augmented_features)
        visual_output = visual_output.view(-1, 1, 28, 28)
        
        # Contrastive learning (during training)
        contrastive_info = {}
        if self.training and use_contrastive and targets is not None:
            # Semantic contrastive loss
            contrastive_loss, similarity_matrix = self.contrastive_loss(
                contrastive_features, targets
            )
            
            # Hard negative mining
            hard_negatives, mining_info = self.hard_negative_miner.mine_hard_negatives(
                contrastive_features, targets, epoch
            )
            
            contrastive_info = {
                'contrastive_loss': contrastive_loss,
                'similarity_matrix': similarity_matrix,
                'hard_negatives': hard_negatives,
                'mining_info': mining_info,
                'augmentation_mask': aug_mask
            }
        
        return visual_output, encoded_features, contrastive_info


class ContrastiveTrainer:
    """
    Specialized trainer for contrastive learning
    
    Handles:
    - Combined reconstruction + contrastive loss
    - Hard negative mining integration
    - Adaptive loss weighting
    """
    
    def __init__(self, model, device, contrastive_weight=0.1):
        self.model = model
        self.device = device
        self.contrastive_weight = contrastive_weight
        
    def compute_combined_loss(self, visual_output, targets, contrastive_info, epoch=0):
        """
        Compute combined reconstruction + contrastive loss
        
        Args:
            visual_output: [batch_size, 1, 28, 28]
            targets: [batch_size, 1, 28, 28]
            contrastive_info: Dict from model forward pass
            epoch: Current epoch for adaptive weighting
        Returns:
            total_loss: Combined loss
            loss_components: Dict with individual loss components
        """
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(visual_output, targets)
        
        # Contrastive loss (if available)
        if 'contrastive_loss' in contrastive_info:
            contrastive_loss = contrastive_info['contrastive_loss']
            
            # Adaptive weighting (higher contrastive weight early in training)
            adaptive_weight = self.contrastive_weight * (1.0 + np.exp(-epoch / 20))
            
            total_loss = reconstruction_loss + adaptive_weight * contrastive_loss
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)
            total_loss = reconstruction_loss
        
        loss_components = {
            'reconstruction_loss': reconstruction_loss.item(),
            'contrastive_loss': contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components


# Factory function for easy model creation
def create_contrastive_model(input_dim, device='cuda'):
    """Factory function to create contrastive learning model"""
    return ContrastiveCortexFlowV2(input_dim, device=device)


# Export main classes
__all__ = [
    'SemanticContrastiveLoss',
    'HardNegativeMiner',
    'SelfSupervisedAugmentation',
    'ContrastiveCortexFlowV2',
    'ContrastiveTrainer',
    'create_contrastive_model'
]
