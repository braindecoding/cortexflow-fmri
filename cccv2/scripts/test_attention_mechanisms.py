"""
CCCV2 Attention Mechanisms Testing
=================================

Test and validate the revolutionary attention mechanisms:
1. Multi-Scale Attention Pyramid
2. Cross-Modal Attention
3. Adaptive Attention Weighting
4. Complete Attention Encoder

Goal: Validate attention innovations and measure performance improvements
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)
sys.path.append(parent_dir)

# Import CCCV2 attention models
try:
    from cccv2.src.models.attention_models import (
        CortexFlowAttentionEncoder,
        MultiScaleAttentionPyramid,
        CrossModalAttention,
        AdaptiveAttentionWeighting,
        create_attention_encoder
    )
except ImportError:
    try:
        sys.path.append(os.path.join(parent_dir, 'src', 'models'))
        from attention_models import (
            CortexFlowAttentionEncoder,
            MultiScaleAttentionPyramid,
            CrossModalAttention,
            AdaptiveAttentionWeighting,
            create_attention_encoder
        )
    except ImportError:
        print("❌ Could not import CCCV2 attention models")
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

def create_attention_decoder(device):
    """Create simple decoder for testing"""
    return nn.Sequential(
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.SiLU(),
        nn.Dropout(0.02),
        nn.Linear(512, 784),
        nn.Sigmoid()
    ).to(device)

class AttentionCortexFlowV2(nn.Module):
    """Complete CCCV2 model with attention mechanisms"""
    
    def __init__(self, input_dim, device):
        super(AttentionCortexFlowV2, self).__init__()
        self.name = "CortexFlow-CLIP-CNN-V2-Attention"
        self.device = device
        
        # Attention-based encoder
        self.encoder = CortexFlowAttentionEncoder(input_dim, device=device)
        
        # Simple decoder for testing
        self.decoder = create_attention_decoder(device)
        
    def forward(self, x, dataset_id=None, dataset_stats=None):
        # Attention encoding
        encoded_features, attention_info = self.encoder(
            x, dataset_id=dataset_id, dataset_stats=dataset_stats
        )
        
        # Decode to visual output
        visual_output = self.decoder(encoded_features)
        visual_output = visual_output.view(-1, 1, 28, 28)
        
        return visual_output, encoded_features, attention_info

def train_attention_model(model, train_loader, val_loader, config, device):
    """Train attention model with advanced techniques"""
    
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
    patience = config.get('patience', 15)
    
    training_history = []
    
    for epoch in range(config.get('epochs', 100)):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Create dataset statistics (dummy for testing)
            batch_size = data.shape[0]
            dataset_stats = torch.tensor([[100.0, 0.5, 0.8]]).repeat(batch_size, 1).to(device)
            dataset_id = torch.tensor([0]).to(device)  # Dummy dataset ID
            
            optimizer.zero_grad()
            
            # Forward pass with attention
            visual_output, encoded_features, attention_info = model(
                data, dataset_id=dataset_id, dataset_stats=dataset_stats
            )
            
            loss = criterion(visual_output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                batch_size = data.shape[0]
                dataset_stats = torch.tensor([[100.0, 0.5, 0.8]]).repeat(batch_size, 1).to(device)
                dataset_id = torch.tensor([0]).to(device)
                
                visual_output, _, _ = model(data, dataset_id=dataset_id, dataset_stats=dataset_stats)
                val_loss += criterion(visual_output, target).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        # Track history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch+1:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return {
        'best_val_loss': best_val_loss,
        'training_history': training_history,
        'final_epoch': epoch + 1
    }

def evaluate_attention_model(model, test_loader, device):
    """Evaluate attention model and extract attention maps"""
    model.eval()
    all_predictions = []
    all_targets = []
    attention_maps = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            batch_size = data.shape[0]
            dataset_stats = torch.tensor([[100.0, 0.5, 0.8]]).repeat(batch_size, 1).to(device)
            dataset_id = torch.tensor([0]).to(device)
            
            visual_output, encoded_features, attention_info = model(
                data, dataset_id=dataset_id, dataset_stats=dataset_stats
            )
            
            all_predictions.append(visual_output.cpu())
            all_targets.append(target.cpu())
            attention_maps.append(attention_info)
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = nn.MSELoss()(predictions, targets).item()
    
    return {
        'mse': mse,
        'predictions': predictions,
        'targets': targets,
        'attention_maps': attention_maps
    }

def visualize_attention_maps(attention_maps, save_path):
    """Visualize attention patterns"""
    
    # Extract scale weights from first batch
    first_batch = attention_maps[0]
    scale_weights = first_batch['pyramid_maps']['scale_weights'].cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scale weights
    axes[0, 0].bar(['Local', 'Regional', 'Global'], scale_weights)
    axes[0, 0].set_title('Multi-Scale Attention Weights')
    axes[0, 0].set_ylabel('Weight')
    
    # Attention heatmap (if available)
    if 'layer_attentions' in first_batch and len(first_batch['layer_attentions']) > 0:
        layer_attn = first_batch['layer_attentions'][0][0, 0].cpu().numpy()  # First head, first sample
        im = axes[0, 1].imshow(layer_attn, cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Self-Attention Pattern')
        plt.colorbar(im, ax=axes[0, 1])
    
    # Adaptive weights (if available)
    if 'adaptive_weights' in first_batch and first_batch['adaptive_weights'] is not None:
        adaptive_weights = first_batch['adaptive_weights'][0].cpu().numpy()  # First sample
        axes[1, 0].plot(adaptive_weights[:50])  # Plot first 50 dimensions
        axes[1, 0].set_title('Adaptive Attention Weights')
        axes[1, 0].set_xlabel('Feature Dimension')
        axes[1, 0].set_ylabel('Weight')
    
    # Summary statistics
    axes[1, 1].text(0.1, 0.8, f'Local Weight: {scale_weights[0]:.3f}', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'Regional Weight: {scale_weights[1]:.3f}', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f'Global Weight: {scale_weights[2]:.3f}', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Attention Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Attention visualization saved to: {save_path}")

def test_single_dataset(dataset_name, device):
    """Test attention mechanisms on a single dataset"""
    
    print(f"\n📁 Testing Attention Mechanisms on {dataset_name.upper()}")
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
    
    # Create attention model
    model = AttentionCortexFlowV2(input_dim, device)
    
    print(f"🏗️ Model: {model.name}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare data loaders
    train_size = int(0.8 * len(X_train))
    X_train_split = X_train[:train_size]
    y_train_split = y_train[:train_size]
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    
    batch_size = min(16, len(X_train_split) // 4)  # Adaptive batch size
    
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
        'epochs': 80,
        'patience': 15
    }
    
    print(f"🔧 Training with attention mechanisms...")
    
    # Train model
    training_results = train_attention_model(model, train_loader, val_loader, config, device)
    
    # Evaluate model
    test_results = evaluate_attention_model(model, test_loader, device)
    
    print(f"\n🎯 Attention Results for {dataset_name.upper()}:")
    print(f"   Best Val Loss: {training_results['best_val_loss']:.6f}")
    print(f"   Test MSE: {test_results['mse']:.6f}")
    print(f"   Training Epochs: {training_results['final_epoch']}")
    
    # Compare with CCCV1 targets
    cccv1_targets = {
        'miyawaki': 0.005845,
        'vangerven': 0.046034,
        'mindbigdata': 0.057028,
        'crell': 0.032504
    }
    
    if dataset_name in cccv1_targets:
        cccv1_target = cccv1_targets[dataset_name]
        if test_results['mse'] < cccv1_target:
            improvement = ((cccv1_target - test_results['mse']) / cccv1_target) * 100
            print(f"   🏆 BEATS CCCV1 by {improvement:.2f}%!")
        else:
            gap = ((test_results['mse'] - cccv1_target) / cccv1_target) * 100
            print(f"   📈 Gap to CCCV1: +{gap:.2f}%")
    
    # Visualize attention patterns
    results_dir = f"cccv2/results/attention_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    attention_viz_path = f"{results_dir}/{dataset_name}_attention_patterns.png"
    visualize_attention_maps(test_results['attention_maps'], attention_viz_path)
    
    return {
        'dataset_name': dataset_name,
        'training_results': training_results,
        'test_results': test_results,
        'model': model,
        'results_dir': results_dir
    }

def main():
    """Main attention testing function"""
    print("🎯 CCCV2 Attention Mechanisms Testing")
    print("=" * 50)
    print("🔬 Testing revolutionary attention innovations")
    print("📊 Goal: Validate attention improvements over CCCV1")
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
        result = test_single_dataset(dataset_name, device)
        if result:
            all_results[dataset_name] = result
    
    # Final analysis
    print("\n🎉 CCCV2 Attention Testing Complete!")
    print("=" * 50)
    
    improvements = 0
    total_datasets = len(all_results)
    
    # CCCV1 baseline results
    cccv1_baselines = {
        'miyawaki': 0.005845,
        'vangerven': 0.046034,
        'mindbigdata': 0.057028,
        'crell': 0.032504
    }
    
    for dataset_name, result in all_results.items():
        test_mse = result['test_results']['mse']
        cccv1_baseline = cccv1_baselines.get(dataset_name, 0.05)
        
        print(f"\n{dataset_name.upper()}:")
        print(f"   CCCV1 Baseline: {cccv1_baseline:.6f}")
        print(f"   CCCV2 Attention: {test_mse:.6f}")
        
        if test_mse < cccv1_baseline:
            improvement = ((cccv1_baseline - test_mse) / cccv1_baseline) * 100
            print(f"   🏆 IMPROVEMENT: +{improvement:.2f}%")
            improvements += 1
        else:
            gap = ((test_mse - cccv1_baseline) / cccv1_baseline) * 100
            print(f"   📈 Gap: +{gap:.2f}%")
    
    print(f"\n🏆 FINAL ATTENTION TESTING RESULTS:")
    print(f"Improvements over CCCV1: {improvements}/{total_datasets}")
    print(f"Success rate: {(improvements/total_datasets)*100:.1f}%")
    
    if improvements >= total_datasets * 0.5:
        print(f"\n🎉 ATTENTION MECHANISMS SUCCESS!")
        print(f"🚀 Multi-scale attention shows promising improvements!")
    else:
        print(f"\n🔧 Attention mechanisms need refinement")
        print(f"📈 Consider architecture adjustments or training optimization")
    
    print("\n🚀 Ready for next CCCV2 innovation: Hierarchical Feature Fusion!")

if __name__ == "__main__":
    main()
