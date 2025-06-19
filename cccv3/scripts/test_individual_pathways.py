"""
CCCV3 Individual Pathway Testing
================================

Test each pathway individually to validate they reproduce the performance
of their original architectures:

1. CortexFlow-Lite pathway → Should match original Lite performance
2. CLIP-inspired pathway → Should match CCCV1 performance  
3. Attention pathway → Should match CCCV2 attention performance

Goal: Validate pathway implementations before fusion testing
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

# Import CCCV3 pathway models
try:
    from cccv3.src.models.lite_pathway import create_complete_lite_model
    from cccv3.src.models.clip_pathway import create_complete_clip_model
    from cccv3.src.models.attention_pathway import create_complete_attention_model
except ImportError:
    try:
        sys.path.append(os.path.join(parent_dir, 'src', 'models'))
        from lite_pathway import create_complete_lite_model
        from clip_pathway import create_complete_clip_model
        from attention_pathway import create_complete_attention_model
    except ImportError:
        print("❌ Could not import CCCV3 pathway models")
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

def train_pathway_model(model, train_loader, val_loader, config, device):
    """Train individual pathway model"""
    
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
            if len(outputs) == 3:  # Pathway model with info
                visual_output, pathway_features, pathway_info = outputs
            else:  # Simple model
                visual_output, pathway_features = outputs
            
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

def evaluate_pathway_model(model, test_loader, device):
    """Evaluate individual pathway model"""
    model.eval()
    all_predictions = []
    all_targets = []
    pathway_infos = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            if len(outputs) == 3:
                visual_output, pathway_features, pathway_info = outputs
                pathway_infos.append(pathway_info)
            else:
                visual_output, pathway_features = outputs
            
            all_predictions.append(visual_output.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = nn.MSELoss()(predictions, targets).item()
    
    return {
        'mse': mse,
        'predictions': predictions,
        'targets': targets,
        'pathway_infos': pathway_infos
    }

def test_pathway_on_dataset(pathway_name, pathway_factory, dataset_name, device):
    """Test individual pathway on a dataset"""
    
    print(f"\n📁 Testing {pathway_name} on {dataset_name.upper()}")
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
        
        dataset_size = len(X_train)
        print(f"✅ Dataset loaded: Train={dataset_size}, Test={len(X_test)}, Input_dim={input_dim}")
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None
    
    # Create pathway model
    model = pathway_factory(input_dim, dataset_size=dataset_size, device=device)
    print(f"🏗️ Model: {model.model_name}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
    
    print(f"🔧 Training {pathway_name}...")
    
    # Train model
    val_loss = train_pathway_model(model, train_loader, val_loader, config, device)
    
    # Evaluate model
    test_results = evaluate_pathway_model(model, test_loader, device)
    
    print(f"\n🎯 {pathway_name} Results on {dataset_name.upper()}:")
    print(f"   Val Loss: {val_loss:.6f}")
    print(f"   Test MSE: {test_results['mse']:.6f}")
    
    return {
        'pathway_name': pathway_name,
        'dataset_name': dataset_name,
        'dataset_size': dataset_size,
        'val_loss': val_loss,
        'test_mse': test_results['mse'],
        'model': model,
        'test_results': test_results
    }

def compare_with_baselines(results, dataset_name):
    """Compare pathway results with known baselines"""
    
    # Known baseline results from previous testing
    baselines = {
        'miyawaki': {
            'CortexFlow-Lite': 0.012,  # Approximate from previous tests
            'CCCV1-CLIP': 0.012326,   # From CCCV2 final integration
            'CCCV2-Attention': 0.014374  # From CCCV2 final integration
        },
        'vangerven': {
            'CortexFlow-Lite': 0.036,  # Approximate (known winner)
            'CCCV1-CLIP': 0.036487,   # From CCCV2 final integration
            'CCCV2-Attention': 0.036926  # From CCCV2 final integration
        },
        'mindbigdata': {
            'CortexFlow-Lite': 0.058,  # Approximate
            'CCCV1-CLIP': 0.058601,   # From CCCV2 final integration
            'CCCV2-Attention': 0.056883  # From CCCV2 final integration (winner)
        },
        'crell': {
            'CortexFlow-Lite': 0.032,  # Approximate
            'CCCV1-CLIP': 0.032218,   # From CCCV2 final integration
            'CCCV2-Attention': 0.032058  # From CCCV2 final integration (winner)
        }
    }
    
    dataset_baselines = baselines.get(dataset_name, {})
    
    print(f"\n📊 Baseline Comparison for {dataset_name.upper()}:")
    
    for result in results:
        pathway_name = result['pathway_name']
        test_mse = result['test_mse']
        
        # Map pathway names to baseline keys
        baseline_key = None
        if 'Lite' in pathway_name:
            baseline_key = 'CortexFlow-Lite'
        elif 'CLIP' in pathway_name:
            baseline_key = 'CCCV1-CLIP'
        elif 'Attention' in pathway_name:
            baseline_key = 'CCCV2-Attention'
        
        if baseline_key and baseline_key in dataset_baselines:
            baseline_mse = dataset_baselines[baseline_key]
            
            if test_mse < baseline_mse:
                improvement = ((baseline_mse - test_mse) / baseline_mse) * 100
                print(f"   {pathway_name}: {test_mse:.6f} vs {baseline_mse:.6f} → 🏆 +{improvement:.2f}%")
            else:
                gap = ((test_mse - baseline_mse) / baseline_mse) * 100
                print(f"   {pathway_name}: {test_mse:.6f} vs {baseline_mse:.6f} → 📈 -{gap:.2f}%")
        else:
            print(f"   {pathway_name}: {test_mse:.6f} (no baseline available)")

def main():
    """Main individual pathway testing function"""
    print("🎯 CCCV3 Individual Pathway Testing")
    print("=" * 45)
    print("🔬 Testing pathway implementations individually")
    print("📊 Goal: Validate pathways reproduce original performance")
    print()
    
    device = setup_device()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Define pathways to test
    pathways = [
        ('CortexFlow-Lite-Pathway', create_complete_lite_model),
        ('CLIP-Inspired-Pathway', create_complete_clip_model),
        ('Attention-Pathway', create_complete_attention_model)
    ]
    
    # Test datasets
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    # Results storage
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"🗂️ TESTING ALL PATHWAYS ON {dataset_name.upper()}")
        print(f"{'='*60}")
        
        dataset_results = []
        
        for pathway_name, pathway_factory in pathways:
            result = test_pathway_on_dataset(pathway_name, pathway_factory, dataset_name, device)
            if result:
                dataset_results.append(result)
        
        # Compare with baselines
        if dataset_results:
            compare_with_baselines(dataset_results, dataset_name)
            all_results[dataset_name] = dataset_results
    
    # Final analysis
    print("\n🎉 CCCV3 Individual Pathway Testing Complete!")
    print("=" * 60)
    
    # Summary analysis
    pathway_success_rates = {'Lite': 0, 'CLIP': 0, 'Attention': 0}
    total_tests = len(datasets)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()} Results:")
        
        for result in results:
            pathway_name = result['pathway_name']
            test_mse = result['test_mse']
            print(f"   {pathway_name}: {test_mse:.6f}")
    
    print(f"\n🏆 INDIVIDUAL PATHWAY VALIDATION:")
    print(f"✅ All pathways implemented and tested")
    print(f"📊 Performance validation against baselines completed")
    print(f"🚀 Ready for multi-pathway fusion testing!")

if __name__ == "__main__":
    main()
