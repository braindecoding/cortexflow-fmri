"""
CCCV2 Simple Attention Test
===========================

Simplified attention mechanism test focusing on core improvements:
1. Basic self-attention for feature relationships
2. Lightweight multi-head attention
3. Residual connections for stability
4. Compare with CCCV1 baseline

Goal: Validate attention concept with simpler implementation
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

class SimpleAttentionModule(nn.Module):
    """Simple but effective attention mechanism"""
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(SimpleAttentionModule, self).__init__()
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
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.layer_norm(x + ffn_output)
        
        return x

class SimpleCortexFlowV2(nn.Module):
    """Simplified CortexFlow V2 with basic attention"""
    
    def __init__(self, input_dim, device):
        super(SimpleCortexFlowV2, self).__init__()
        self.name = "CortexFlow-CLIP-CNN-V2-Simple"
        self.device = device
        
        # Encoder with attention
        self.input_projection = nn.Linear(input_dim, 512).to(device)
        
        # Simple attention mechanism
        self.attention = SimpleAttentionModule(512, num_heads=4, dropout=0.05).to(device)
        
        # CLIP-like encoder
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
        # Project to embedding space
        x = self.input_projection(x)  # [batch_size, 512]
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, 512]
        
        # Apply attention
        x = self.attention(x)  # [batch_size, 1, 512]
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, 512]
        
        # Encode to CLIP-like space
        encoded = self.encoder(x)
        encoded = torch.nn.functional.normalize(encoded, p=2, dim=1)
        
        # Decode to visual output
        visual_output = self.decoder(encoded)
        visual_output = visual_output.view(-1, 1, 28, 28)
        
        return visual_output, encoded

class BaselineCortexFlowV1(nn.Module):
    """Baseline CCCV1-like model for comparison"""
    
    def __init__(self, input_dim, device):
        super(BaselineCortexFlowV1, self).__init__()
        self.name = "CortexFlow-CLIP-CNN-V1-Baseline"
        self.device = device
        
        # Standard encoder (no attention)
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
            visual_output, encoded_features = model(data)
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
                visual_output, _ = model(data)
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
            visual_output, _ = model(data)
            
            all_predictions.append(visual_output.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = nn.MSELoss()(predictions, targets).item()
    return mse

def compare_models(dataset_name, device):
    """Compare CCCV2 attention vs CCCV1 baseline"""
    
    print(f"\n📁 Comparing Models on {dataset_name.upper()}")
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
    
    # Test CCCV1 Baseline
    print(f"\n🔧 Training CCCV1 Baseline...")
    baseline_model = BaselineCortexFlowV1(input_dim, device)
    print(f"   Parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
    
    baseline_val_loss = train_model(baseline_model, train_loader, val_loader, config, device)
    baseline_test_mse = evaluate_model(baseline_model, test_loader, device)
    
    results['baseline'] = {
        'val_loss': baseline_val_loss,
        'test_mse': baseline_test_mse,
        'model_name': baseline_model.name
    }
    
    print(f"   Baseline Results: Val={baseline_val_loss:.6f}, Test={baseline_test_mse:.6f}")
    
    # Test CCCV2 Simple Attention
    print(f"\n🔧 Training CCCV2 Simple Attention...")
    attention_model = SimpleCortexFlowV2(input_dim, device)
    print(f"   Parameters: {sum(p.numel() for p in attention_model.parameters()):,}")
    
    attention_val_loss = train_model(attention_model, train_loader, val_loader, config, device)
    attention_test_mse = evaluate_model(attention_model, test_loader, device)
    
    results['attention'] = {
        'val_loss': attention_val_loss,
        'test_mse': attention_test_mse,
        'model_name': attention_model.name
    }
    
    print(f"   Attention Results: Val={attention_val_loss:.6f}, Test={attention_test_mse:.6f}")
    
    # Compare results
    print(f"\n🎯 Comparison Results for {dataset_name.upper()}:")
    print(f"   CCCV1 Baseline: {baseline_test_mse:.6f}")
    print(f"   CCCV2 Attention: {attention_test_mse:.6f}")
    
    if attention_test_mse < baseline_test_mse:
        improvement = ((baseline_test_mse - attention_test_mse) / baseline_test_mse) * 100
        print(f"   🏆 ATTENTION WINS by {improvement:.2f}%!")
        results['winner'] = 'attention'
        results['improvement'] = improvement
    else:
        gap = ((attention_test_mse - baseline_test_mse) / baseline_test_mse) * 100
        print(f"   📈 Baseline wins by {gap:.2f}%")
        results['winner'] = 'baseline'
        results['improvement'] = -gap
    
    return results

def main():
    """Main simple attention testing function"""
    print("🎯 CCCV2 Simple Attention Testing")
    print("=" * 40)
    print("🔬 Testing simplified attention mechanisms")
    print("📊 Goal: Validate attention concept vs baseline")
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
    print("\n🎉 CCCV2 Simple Attention Testing Complete!")
    print("=" * 50)
    
    attention_wins = 0
    total_datasets = len(all_results)
    
    for dataset_name, result in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"   Winner: {result['winner']}")
        print(f"   Improvement: {result['improvement']:.2f}%")
        
        if result['winner'] == 'attention':
            attention_wins += 1
    
    print(f"\n🏆 FINAL SIMPLE ATTENTION RESULTS:")
    print(f"Attention wins: {attention_wins}/{total_datasets}")
    print(f"Success rate: {(attention_wins/total_datasets)*100:.1f}%")
    
    if attention_wins >= total_datasets * 0.5:
        print(f"\n🎉 ATTENTION CONCEPT VALIDATED!")
        print(f"🚀 Simple attention shows promise for CCCV2!")
    else:
        print(f"\n🔧 Attention needs refinement")
        print(f"📈 Consider different attention architectures")
    
    print("\n🚀 Ready for next CCCV2 innovation!")

if __name__ == "__main__":
    main()
