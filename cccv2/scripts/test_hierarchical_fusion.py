"""
CCCV2 Hierarchical Feature Fusion Testing
=========================================

Test revolutionary hierarchical processing innovations:
1. Multi-Resolution Feature Extraction
2. Feature Pyramid Networks (FPN)
3. Adaptive Fusion Mechanisms
4. Compare with Attention Baseline

Goal: Validate hierarchical fusion improvements over attention mechanisms
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
sys.path.append(parent_dir)

# Import CCCV2 models
try:
    from cccv2.src.models.hierarchical_fusion import (
        HierarchicalCortexFlowV2,
        create_hierarchical_model
    )
except ImportError:
    try:
        sys.path.append(os.path.join(parent_dir, 'src', 'models'))
        from hierarchical_fusion import (
            HierarchicalCortexFlowV2,
            create_hierarchical_model
        )
    except ImportError:
        print("❌ Could not import CCCV2 hierarchical models")
        sys.exit(1)

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

class AttentionBaseline(nn.Module):
    """Attention baseline from previous test for comparison"""
    
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
    """Train model with advanced techniques"""
    
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
            
            # Handle different model types
            if hasattr(model, 'forward') and 'dataset_id' in model.forward.__code__.co_varnames:
                # Hierarchical model with dataset info
                batch_size = data.shape[0]
                dataset_stats = torch.tensor([[100.0, 0.5, 0.8]]).repeat(batch_size, 1).to(device)
                dataset_id = torch.tensor([0]).to(device)
                
                visual_output, encoded_features, hierarchy_info = model(
                    data, dataset_id=dataset_id, dataset_stats=dataset_stats
                )
            else:
                # Attention baseline
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
                
                if hasattr(model, 'forward') and 'dataset_id' in model.forward.__code__.co_varnames:
                    batch_size = data.shape[0]
                    dataset_stats = torch.tensor([[100.0, 0.5, 0.8]]).repeat(batch_size, 1).to(device)
                    dataset_id = torch.tensor([0]).to(device)
                    
                    visual_output, _, _ = model(data, dataset_id=dataset_id, dataset_stats=dataset_stats)
                else:
                    visual_output, _ = model(data)
                
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
    """Evaluate model and extract hierarchy info if available"""
    model.eval()
    all_predictions = []
    all_targets = []
    hierarchy_info = None
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if hasattr(model, 'forward') and 'dataset_id' in model.forward.__code__.co_varnames:
                batch_size = data.shape[0]
                dataset_stats = torch.tensor([[100.0, 0.5, 0.8]]).repeat(batch_size, 1).to(device)
                dataset_id = torch.tensor([0]).to(device)
                
                visual_output, encoded_features, hierarchy_info = model(
                    data, dataset_id=dataset_id, dataset_stats=dataset_stats
                )
            else:
                visual_output, encoded_features = model(data)
            
            all_predictions.append(visual_output.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = nn.MSELoss()(predictions, targets).item()
    
    return {
        'mse': mse,
        'predictions': predictions,
        'targets': targets,
        'hierarchy_info': hierarchy_info
    }

def analyze_hierarchy_info(hierarchy_info, save_path):
    """Analyze and visualize hierarchical processing"""
    
    if hierarchy_info is None:
        print("⚠️ No hierarchy information available")
        return
    
    print(f"\n🔍 Hierarchical Processing Analysis:")
    
    # Analyze fusion weights
    if 'fusion_weights' in hierarchy_info:
        fusion_weights = hierarchy_info['fusion_weights'][0].cpu().numpy()  # First sample
        print(f"   Fusion weights shape: {fusion_weights.shape}")
        print(f"   Fusion weights: {fusion_weights}")
        
        # Find dominant scales
        dominant_scales = np.argsort(fusion_weights)[-3:]  # Top 3 scales
        print(f"   Dominant scales: {dominant_scales}")
    
    # Analyze multi-resolution features
    if 'multi_res_features' in hierarchy_info:
        multi_res = hierarchy_info['multi_res_features']
        print(f"   Multi-resolution features: {len(multi_res)} scales")
        for i, features in enumerate(multi_res):
            print(f"     Scale {i}: {features.shape}")
    
    # Analyze pyramid features
    if 'pyramid_features' in hierarchy_info:
        pyramid = hierarchy_info['pyramid_features']
        print(f"   Pyramid features: {len(pyramid)} levels")
        for i, features in enumerate(pyramid):
            print(f"     Level {i}: {features.shape}")

def compare_models(dataset_name, device):
    """Compare hierarchical fusion vs attention baseline"""
    
    print(f"\n📁 Testing Hierarchical Fusion on {dataset_name.upper()}")
    print("=" * 60)
    
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
    attention_results = evaluate_model(attention_model, test_loader, device)
    
    results['attention'] = {
        'val_loss': attention_val_loss,
        'test_mse': attention_results['mse'],
        'model_name': attention_model.name
    }
    
    print(f"   Attention Results: Val={attention_val_loss:.6f}, Test={attention_results['mse']:.6f}")
    
    # Test Hierarchical Fusion
    print(f"\n🔧 Training Hierarchical Fusion...")
    hierarchical_model = HierarchicalCortexFlowV2(input_dim, device)
    print(f"   Parameters: {sum(p.numel() for p in hierarchical_model.parameters()):,}")
    
    hierarchical_val_loss = train_model(hierarchical_model, train_loader, val_loader, config, device)
    hierarchical_results = evaluate_model(hierarchical_model, test_loader, device)
    
    results['hierarchical'] = {
        'val_loss': hierarchical_val_loss,
        'test_mse': hierarchical_results['mse'],
        'model_name': hierarchical_model.name,
        'hierarchy_info': hierarchical_results['hierarchy_info']
    }
    
    print(f"   Hierarchical Results: Val={hierarchical_val_loss:.6f}, Test={hierarchical_results['mse']:.6f}")
    
    # Analyze hierarchical processing
    analyze_hierarchy_info(hierarchical_results['hierarchy_info'], f"hierarchy_{dataset_name}")
    
    # Compare results
    print(f"\n🎯 Comparison Results for {dataset_name.upper()}:")
    print(f"   Attention Baseline: {attention_results['mse']:.6f}")
    print(f"   Hierarchical Fusion: {hierarchical_results['mse']:.6f}")
    
    if hierarchical_results['mse'] < attention_results['mse']:
        improvement = ((attention_results['mse'] - hierarchical_results['mse']) / attention_results['mse']) * 100
        print(f"   🏆 HIERARCHICAL WINS by {improvement:.2f}%!")
        results['winner'] = 'hierarchical'
        results['improvement'] = improvement
    else:
        gap = ((hierarchical_results['mse'] - attention_results['mse']) / attention_results['mse']) * 100
        print(f"   📈 Attention wins by {gap:.2f}%")
        results['winner'] = 'attention'
        results['improvement'] = -gap
    
    return results

def main():
    """Main hierarchical fusion testing function"""
    print("🎯 CCCV2 Hierarchical Feature Fusion Testing")
    print("=" * 50)
    print("🔬 Testing revolutionary hierarchical processing")
    print("📊 Goal: Validate hierarchical improvements over attention")
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
    print("\n🎉 CCCV2 Hierarchical Fusion Testing Complete!")
    print("=" * 60)
    
    hierarchical_wins = 0
    total_datasets = len(all_results)
    
    for dataset_name, result in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"   Winner: {result['winner']}")
        print(f"   Improvement: {result['improvement']:.2f}%")
        
        if result['winner'] == 'hierarchical':
            hierarchical_wins += 1
    
    print(f"\n🏆 FINAL HIERARCHICAL FUSION RESULTS:")
    print(f"Hierarchical wins: {hierarchical_wins}/{total_datasets}")
    print(f"Success rate: {(hierarchical_wins/total_datasets)*100:.1f}%")
    
    if hierarchical_wins >= total_datasets * 0.5:
        print(f"\n🎉 HIERARCHICAL FUSION SUCCESS!")
        print(f"🚀 Multi-scale processing shows significant improvements!")
    else:
        print(f"\n🔧 Hierarchical fusion needs refinement")
        print(f"📈 Consider architecture adjustments")
    
    print("\n🚀 Ready for next CCCV2 innovation: Contrastive Learning!")

if __name__ == "__main__":
    main()
