"""
CCCV2 Final Integration
======================

Final integration of best CCCV2 innovations:
1. ✅ Attention Mechanisms (75% success rate)
2. ⚠️ Hierarchical Fusion (50% success rate - large datasets only)
3. ❌ Contrastive Learning (0% success rate - needs refinement)

Strategy: Combine proven attention with selective hierarchical fusion
Goal: Achieve best possible performance across all datasets
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)

# Import utilities
try:
    from src.data import load_dataset_gpu_optimized
except ImportError:
    print("⚠️ Parent directory imports not available")

def setup_device():
    """Setup CUDA device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        return device
    else:
        return torch.device('cpu')

class AdaptiveAttentionModule(nn.Module):
    """Proven attention mechanism from CCCV2 testing"""
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(AdaptiveAttentionModule, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.layer_norm(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.layer_norm(x + ffn_output)
        
        return x, attn_weights

class SelectiveHierarchicalModule(nn.Module):
    """Selective hierarchical processing for large datasets"""
    
    def __init__(self, input_dim, embed_dim=512, use_hierarchical=True):
        super(SelectiveHierarchicalModule, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_hierarchical = use_hierarchical
        
        if use_hierarchical:
            # Multi-scale encoders (simplified)
            self.scale_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.SiLU(),
                    nn.Dropout(0.1)
                ),
                nn.Sequential(
                    nn.Linear(input_dim, embed_dim * 2),
                    nn.LayerNorm(embed_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.SiLU()
                )
            ])
            
            # Fusion
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.SiLU()
            )
        else:
            # Simple encoder for small datasets
            self.simple_encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.SiLU(),
                nn.Dropout(0.05)
            )
    
    def forward(self, x):
        if self.use_hierarchical:
            # Multi-scale processing
            scale_features = []
            for encoder in self.scale_encoders:
                features = encoder(x)
                scale_features.append(features)
            
            # Fuse scales
            concatenated = torch.cat(scale_features, dim=-1)
            fused_features = self.fusion(concatenated)
            return fused_features
        else:
            # Simple processing
            return self.simple_encoder(x)

class FinalCortexFlowV2(nn.Module):
    """
    Final CCCV2 model combining best innovations
    
    Strategy:
    - Always use attention (proven 75% success)
    - Use hierarchical for large datasets (>500 samples)
    - Skip contrastive learning (0% success)
    """
    
    def __init__(self, input_dim, dataset_size, device):
        super(FinalCortexFlowV2, self).__init__()
        self.name = "CortexFlow-CLIP-CNN-V2-Final"
        self.device = device
        self.dataset_size = dataset_size
        
        # Adaptive architecture based on dataset size
        use_hierarchical = dataset_size > 500  # Large dataset threshold
        
        print(f"   📊 Dataset size: {dataset_size}")
        print(f"   🏗️ Using hierarchical: {use_hierarchical}")
        
        # Hierarchical/Simple encoder
        self.hierarchical_encoder = SelectiveHierarchicalModule(
            input_dim, embed_dim=512, use_hierarchical=use_hierarchical
        ).to(device)
        
        # Attention mechanism (always used)
        self.attention = AdaptiveAttentionModule(512, num_heads=4, dropout=0.05).to(device)
        
        # CLIP-like encoder
        self.clip_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.03),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Tanh()
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
        
    def forward(self, x):
        # Hierarchical/Simple encoding
        encoded_features = self.hierarchical_encoder(x)  # [batch_size, 512]
        
        # Add sequence dimension for attention
        encoded_features = encoded_features.unsqueeze(1)  # [batch_size, 1, 512]
        
        # Apply attention
        attended_features, attn_weights = self.attention(encoded_features)  # [batch_size, 1, 512]
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(1)  # [batch_size, 512]
        
        # CLIP encoding
        clip_features = self.clip_encoder(attended_features)
        clip_features = torch.nn.functional.normalize(clip_features, p=2, dim=1)
        
        # Decode to visual output
        visual_output = self.decoder(clip_features)
        visual_output = visual_output.view(-1, 1, 28, 28)
        
        return visual_output, clip_features, attn_weights

class CCCV1Baseline(nn.Module):
    """CCCV1 baseline for comparison"""
    
    def __init__(self, input_dim, device):
        super(CCCV1Baseline, self).__init__()
        self.name = "CortexFlow-CLIP-CNN-V1-Baseline"
        self.device = device
        
        # Standard CCCV1 architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.06),
            
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.04),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.03),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Tanh()
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
        
    def forward(self, x):
        # Standard encoding
        encoded = self.encoder(x)
        encoded = torch.nn.functional.normalize(encoded, p=2, dim=1)
        
        # Decode to visual output
        visual_output = self.decoder(encoded)
        visual_output = visual_output.view(-1, 1, 28, 28)
        
        return visual_output, encoded

def train_model(model, train_loader, val_loader, config, device):
    """Train model with optimized configuration"""
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 1e-6),
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('epochs', 100), eta_min=1e-8
    )
    
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 20)
    
    for epoch in range(config.get('epochs', 100)):
        # Training
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Handle different model outputs
            outputs = model(data)
            if len(outputs) == 3:  # Final CCCV2
                visual_output, encoded_features, attn_weights = outputs
            else:  # CCCV1 baseline
                visual_output, encoded_features = outputs
            
            loss = criterion(visual_output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                outputs = model(data)
                if len(outputs) == 3:
                    visual_output, _, _ = outputs
                else:
                    visual_output, _ = outputs
                
                val_loss += criterion(visual_output, target).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"   Epoch {epoch+1:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return best_val_loss

def evaluate_model(model, test_loader, device):
    """Evaluate model"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            if len(outputs) == 3:
                visual_output, _, _ = outputs
            else:
                visual_output, _ = outputs
            
            all_predictions.append(visual_output.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = nn.MSELoss()(predictions, targets).item()
    return mse

def compare_final_models(dataset_name, device):
    """Compare Final CCCV2 vs CCCV1 baseline"""
    
    print(f"\n📁 Final CCCV2 Testing on {dataset_name.upper()}")
    print("=" * 50)
    
    # Load dataset
    try:
        if 'load_dataset_gpu_optimized' in globals():
            X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
        else:
            print("❌ Dataset loading function not available")
            return None
        
        if X_train is None:
            print(f"❌ Failed to load {dataset_name}")
            return None
        
        dataset_size = len(X_train)
        print(f"✅ Dataset loaded: Train={dataset_size}, Test={len(X_test)}, Input_dim={input_dim}")
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None
    
    # Prepare data loaders
    train_size = int(0.8 * len(X_train))
    X_train_split = X_train[:train_size]
    y_train_split = y_train[:train_size]
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    
    batch_size = min(16, len(X_train_split) // 4)
    
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Training configuration
    config = {
        'lr': 0.0005,
        'weight_decay': 1e-6,
        'epochs': 100,
        'patience': 20
    }
    
    results = {}
    
    # Test CCCV1 Baseline
    print(f"\n🔧 Training CCCV1 Baseline...")
    cccv1_model = CCCV1Baseline(input_dim, device)
    print(f"   Parameters: {sum(p.numel() for p in cccv1_model.parameters()):,}")
    
    cccv1_val_loss = train_model(cccv1_model, train_loader, val_loader, config, device)
    cccv1_test_mse = evaluate_model(cccv1_model, test_loader, device)
    
    results['cccv1'] = {
        'val_loss': cccv1_val_loss,
        'test_mse': cccv1_test_mse,
        'model_name': cccv1_model.name
    }
    
    print(f"   CCCV1 Results: Val={cccv1_val_loss:.6f}, Test={cccv1_test_mse:.6f}")
    
    # Test Final CCCV2
    print(f"\n🔧 Training Final CCCV2...")
    cccv2_model = FinalCortexFlowV2(input_dim, dataset_size, device)
    print(f"   Parameters: {sum(p.numel() for p in cccv2_model.parameters()):,}")
    
    cccv2_val_loss = train_model(cccv2_model, train_loader, val_loader, config, device)
    cccv2_test_mse = evaluate_model(cccv2_model, test_loader, device)
    
    results['cccv2'] = {
        'val_loss': cccv2_val_loss,
        'test_mse': cccv2_test_mse,
        'model_name': cccv2_model.name
    }
    
    print(f"   CCCV2 Results: Val={cccv2_val_loss:.6f}, Test={cccv2_test_mse:.6f}")
    
    # Compare results
    print(f"\n🎯 Final Comparison for {dataset_name.upper()}:")
    print(f"   CCCV1 Baseline: {cccv1_test_mse:.6f}")
    print(f"   Final CCCV2: {cccv2_test_mse:.6f}")
    
    if cccv2_test_mse < cccv1_test_mse:
        improvement = ((cccv1_test_mse - cccv2_test_mse) / cccv1_test_mse) * 100
        print(f"   🏆 CCCV2 WINS by {improvement:.2f}%!")
        results['winner'] = 'cccv2'
        results['improvement'] = improvement
    else:
        gap = ((cccv2_test_mse - cccv1_test_mse) / cccv1_test_mse) * 100
        print(f"   📈 CCCV1 wins by {gap:.2f}%")
        results['winner'] = 'cccv1'
        results['improvement'] = -gap
    
    return results

def main():
    """Main final integration testing function"""
    print("🎯 CCCV2 Final Integration Testing")
    print("=" * 40)
    print("🔬 Testing best CCCV2 innovations combined")
    print("📊 Goal: Achieve maximum performance across all datasets")
    print()
    
    device = setup_device()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Test datasets
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    # Results storage
    all_results = {}
    
    for dataset_name in datasets:
        result = compare_final_models(dataset_name, device)
        if result:
            all_results[dataset_name] = result
    
    # Final analysis
    print("\n🎉 CCCV2 Final Integration Testing Complete!")
    print("=" * 50)
    
    cccv2_wins = 0
    total_datasets = len(all_results)
    
    for dataset_name, result in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"   Winner: {result['winner']}")
        print(f"   Improvement: {result['improvement']:.2f}%")
        
        if result['winner'] == 'cccv2':
            cccv2_wins += 1
    
    print(f"\n🏆 FINAL CCCV2 INTEGRATION RESULTS:")
    print(f"CCCV2 wins: {cccv2_wins}/{total_datasets}")
    print(f"Success rate: {(cccv2_wins/total_datasets)*100:.1f}%")
    
    if cccv2_wins >= total_datasets * 0.75:
        print(f"\n🎉 CCCV2 INTEGRATION SUCCESS!")
        print(f"🚀 Final CCCV2 achieves excellent performance!")
    elif cccv2_wins >= total_datasets * 0.5:
        print(f"\n✅ CCCV2 INTEGRATION PROMISING!")
        print(f"🔧 Good results with room for improvement!")
    else:
        print(f"\n🔧 CCCV2 needs further refinement")
        print(f"📈 Consider architecture adjustments")
    
    print("\n🚀 CCCV2 Development Complete!")

if __name__ == "__main__":
    main()
