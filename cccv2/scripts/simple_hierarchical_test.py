"""
CCCV2 Simple Hierarchical Test
==============================

Simplified hierarchical feature fusion test:
1. Multi-Scale Feature Extraction
2. Simple Feature Fusion
3. Compare with Attention Baseline

Goal: Validate hierarchical concept with simpler implementation
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

class SimpleMultiScaleEncoder(nn.Module):
    """Simple multi-scale feature extraction"""
    
    def __init__(self, input_dim, embed_dim=512, num_scales=3):
        super(SimpleMultiScaleEncoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        
        # Multi-scale encoders
        self.scale_encoders = nn.ModuleList()
        
        for i in range(num_scales):
            # Different hidden dimensions for different scales
            hidden_dim = embed_dim * (2 ** i)  # 512, 1024, 2048
            
            encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.SiLU(),
                nn.Dropout(0.05)
            )
            
            self.scale_encoders.append(encoder)
        
        # Fusion mechanism
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Extract features at multiple scales
        scale_features = []
        for encoder in self.scale_encoders:
            features = encoder(x)
            scale_features.append(features)
        
        # Concatenate and fuse
        concatenated = torch.cat(scale_features, dim=-1)
        fused_features = self.fusion(concatenated)
        
        return fused_features, scale_features

class SimpleHierarchicalV2(nn.Module):
    """Simple hierarchical CortexFlow V2"""
    
    def __init__(self, input_dim, device):
        super(SimpleHierarchicalV2, self).__init__()
        self.name = "CortexFlow-CLIP-CNN-V2-Hierarchical-Simple"
        self.device = device
        
        # Multi-scale encoder
        self.multi_scale_encoder = SimpleMultiScaleEncoder(input_dim, embed_dim=512, num_scales=3).to(device)
        
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
        # Multi-scale feature extraction
        fused_features, scale_features = self.multi_scale_encoder(x)
        
        # Encode to CLIP-like space
        encoded = self.clip_encoder(fused_features)
        encoded = torch.nn.functional.normalize(encoded, p=2, dim=1)
        
        # Decode to visual output
        visual_output = self.decoder(encoded)
        visual_output = visual_output.view(-1, 1, 28, 28)
        
        return visual_output, encoded, scale_features

class AttentionBaseline(nn.Module):
    """Attention baseline from previous test"""
    
    def __init__(self, input_dim, device):
        super(AttentionBaseline, self).__init__()
        self.name = "CortexFlow-CLIP-CNN-V2-Attention-Baseline"
        self.device = device
        
        # Simple attention mechanism
        self.input_projection = nn.Linear(input_dim, 512).to(device)
        
        self.attention = nn.MultiheadAttention(
            512, num_heads=4, dropout=0.1, batch_first=True
        ).to(device)
        
        self.layer_norm = nn.LayerNorm(512).to(device)
        
        # Encoder
        self.encoder = nn.Sequential(
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
        # Project and add sequence dimension
        x = self.input_projection(x).unsqueeze(1)
        
        # Apply attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_output).squeeze(1)
        
        # Encode and decode
        encoded = self.encoder(x)
        encoded = torch.nn.functional.normalize(encoded, p=2, dim=1)
        
        visual_output = self.decoder(encoded)
        visual_output = visual_output.view(-1, 1, 28, 28)
        
        return visual_output, encoded

def train_model(model, train_loader, val_loader, config, device):
    """Train model with standard configuration"""
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 1e-6)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-8
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
            if len(outputs) == 3:  # Hierarchical model
                visual_output, encoded_features, scale_features = outputs
            else:  # Attention model
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
        
        scheduler.step(val_loss)
        
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

def compare_models(dataset_name, device):
    """Compare simple hierarchical vs attention baseline"""
    
    print(f"\n📁 Testing Simple Hierarchical on {dataset_name.upper()}")
    print("=" * 55)
    
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
        
        print(f"✅ Dataset loaded: Train={len(X_train)}, Test={len(X_test)}, Input_dim={input_dim}")
        
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
    
    # Test Attention Baseline
    print(f"\n🔧 Training Attention Baseline...")
    attention_model = AttentionBaseline(input_dim, device)
    print(f"   Parameters: {sum(p.numel() for p in attention_model.parameters()):,}")
    
    attention_val_loss = train_model(attention_model, train_loader, val_loader, config, device)
    attention_test_mse = evaluate_model(attention_model, test_loader, device)
    
    results['attention'] = {
        'val_loss': attention_val_loss,
        'test_mse': attention_test_mse,
        'model_name': attention_model.name
    }
    
    print(f"   Attention Results: Val={attention_val_loss:.6f}, Test={attention_test_mse:.6f}")
    
    # Test Simple Hierarchical
    print(f"\n🔧 Training Simple Hierarchical...")
    hierarchical_model = SimpleHierarchicalV2(input_dim, device)
    print(f"   Parameters: {sum(p.numel() for p in hierarchical_model.parameters()):,}")
    
    hierarchical_val_loss = train_model(hierarchical_model, train_loader, val_loader, config, device)
    hierarchical_test_mse = evaluate_model(hierarchical_model, test_loader, device)
    
    results['hierarchical'] = {
        'val_loss': hierarchical_val_loss,
        'test_mse': hierarchical_test_mse,
        'model_name': hierarchical_model.name
    }
    
    print(f"   Hierarchical Results: Val={hierarchical_val_loss:.6f}, Test={hierarchical_test_mse:.6f}")
    
    # Compare results
    print(f"\n🎯 Comparison Results for {dataset_name.upper()}:")
    print(f"   Attention Baseline: {attention_test_mse:.6f}")
    print(f"   Simple Hierarchical: {hierarchical_test_mse:.6f}")
    
    if hierarchical_test_mse < attention_test_mse:
        improvement = ((attention_test_mse - hierarchical_test_mse) / attention_test_mse) * 100
        print(f"   🏆 HIERARCHICAL WINS by {improvement:.2f}%!")
        results['winner'] = 'hierarchical'
        results['improvement'] = improvement
    else:
        gap = ((hierarchical_test_mse - attention_test_mse) / attention_test_mse) * 100
        print(f"   📈 Attention wins by {gap:.2f}%")
        results['winner'] = 'attention'
        results['improvement'] = -gap
    
    return results

def main():
    """Main simple hierarchical testing function"""
    print("🎯 CCCV2 Simple Hierarchical Testing")
    print("=" * 45)
    print("🔬 Testing simplified hierarchical processing")
    print("📊 Goal: Validate hierarchical concept vs attention")
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
        result = compare_models(dataset_name, device)
        if result:
            all_results[dataset_name] = result
    
    # Final analysis
    print("\n🎉 CCCV2 Simple Hierarchical Testing Complete!")
    print("=" * 55)
    
    hierarchical_wins = 0
    total_datasets = len(all_results)
    
    for dataset_name, result in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"   Winner: {result['winner']}")
        print(f"   Improvement: {result['improvement']:.2f}%")
        
        if result['winner'] == 'hierarchical':
            hierarchical_wins += 1
    
    print(f"\n🏆 FINAL SIMPLE HIERARCHICAL RESULTS:")
    print(f"Hierarchical wins: {hierarchical_wins}/{total_datasets}")
    print(f"Success rate: {(hierarchical_wins/total_datasets)*100:.1f}%")
    
    if hierarchical_wins >= total_datasets * 0.5:
        print(f"\n🎉 HIERARCHICAL CONCEPT VALIDATED!")
        print(f"🚀 Multi-scale processing shows promise for CCCV2!")
    else:
        print(f"\n🔧 Hierarchical processing needs refinement")
        print(f"📈 Consider different fusion strategies")
    
    print("\n🚀 Ready for next CCCV2 innovation: Contrastive Learning!")

if __name__ == "__main__":
    main()
