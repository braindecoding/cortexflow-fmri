"""
CCCV3 Complete Multi-Pathway Testing
====================================

Test the complete CCCV3 multi-pathway model that combines:
1. CortexFlow-Lite pathway (simple & robust)
2. CLIP-inspired pathway (semantic understanding) 
3. Attention pathway (feature relationships)

With adaptive fusion for optimal performance.

Goal: Achieve 100% success rate with 5-15% improvements
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

# Import CCCV3 complete model
try:
    from cccv3.src.models.cccv3_model import create_cccv3_model, create_cccv3_trainer, get_cccv3_config
except ImportError:
    try:
        sys.path.append(os.path.join(parent_dir, 'src', 'models'))
        from cccv3_model import create_cccv3_model, create_cccv3_trainer, get_cccv3_config
    except ImportError:
        print("❌ Could not import CCCV3 complete model")
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

def train_cccv3_model(model, train_loader, val_loader, config, device):
    """Train CCCV3 multi-pathway model"""
    
    # Create specialized trainer
    trainer = create_cccv3_trainer(model, device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-6),
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('training_epochs', 100), eta_min=1e-8
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    
    training_history = []
    
    for epoch in range(config.get('training_epochs', 100)):
        # Training
        model.train()
        train_losses = {'total': 0.0, 'reconstruction': 0.0, 'regularization': 0.0}
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with pathway info
            visual_output, fused_features, pathway_info = model(data, return_pathway_info=True)
            
            # Compute multi-pathway loss
            total_loss, loss_components = trainer.compute_multi_pathway_loss(
                visual_output, target, pathway_info
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track losses
            train_losses['total'] += loss_components['total_loss']
            train_losses['reconstruction'] += loss_components['reconstruction_loss']
            train_losses['regularization'] += (loss_components['weight_regularization'] + 
                                             loss_components['entropy_regularization'])
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                visual_output, _ = model(data, return_pathway_info=False)
                val_loss += nn.MSELoss()(visual_output, target).item()
        
        # Average losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        # Track history
        training_history.append({
            'epoch': epoch + 1,
            'train_total': train_losses['total'],
            'train_reconstruction': train_losses['reconstruction'],
            'train_regularization': train_losses['regularization'],
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
        
        if epoch % 20 == 0:
            print(f"   Epoch {epoch+1:3d}: Total={train_losses['total']:.6f}, "
                  f"Recon={train_losses['reconstruction']:.6f}, "
                  f"Reg={train_losses['regularization']:.6f}, Val={val_loss:.6f}")
        
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

def evaluate_cccv3_model(model, test_loader, device):
    """Evaluate CCCV3 model with pathway analysis"""
    model.eval()
    all_predictions = []
    all_targets = []
    pathway_analyses = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Get predictions
            visual_output, _ = model(data, return_pathway_info=False)
            all_predictions.append(visual_output.cpu())
            all_targets.append(target.cpu())
            
            # Get pathway analysis for first batch only (for efficiency)
            if len(pathway_analyses) == 0:
                pathway_analysis = model.get_pathway_analysis(data)
                pathway_analyses.append(pathway_analysis)
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = nn.MSELoss()(predictions, targets).item()
    
    return {
        'mse': mse,
        'predictions': predictions,
        'targets': targets,
        'pathway_analysis': pathway_analyses[0] if pathway_analyses else None
    }

def test_cccv3_on_dataset(dataset_name, device):
    """Test complete CCCV3 on a dataset"""
    
    print(f"\n📁 Testing Complete CCCV3 on {dataset_name.upper()}")
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
    
    # Get CCCV3 configuration
    config = get_cccv3_config(dataset_name)
    print(f"📋 CCCV3 Config: {config['primary_strategy']}")
    print(f"🎯 Expected improvement: {config['expected_improvement']}")
    
    # Create CCCV3 model
    model = create_cccv3_model(input_dim, dataset_name, dataset_size, device)
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
    
    print(f"🔧 Training CCCV3 Multi-Pathway Model...")
    
    # Train model
    training_results = train_cccv3_model(model, train_loader, val_loader, config, device)
    
    # Evaluate model
    test_results = evaluate_cccv3_model(model, test_loader, device)
    
    print(f"\n🎯 CCCV3 Results on {dataset_name.upper()}:")
    print(f"   Val Loss: {training_results['best_val_loss']:.6f}")
    print(f"   Test MSE: {test_results['mse']:.6f}")
    print(f"   Training Epochs: {training_results['final_epoch']}")
    
    # Pathway analysis
    if test_results['pathway_analysis']:
        analysis = test_results['pathway_analysis']
        print(f"\n🔍 Pathway Analysis:")
        print(f"   Expected weights: {analysis['pathway_weights']['expected']}")
        print(f"   Actual weights: {analysis['pathway_weights']['actual']}")
        print(f"   Pathway contributions:")
        print(f"     Lite: {analysis['pathway_contributions']['lite']:.4f}")
        print(f"     CLIP: {analysis['pathway_contributions']['clip']:.4f}")
        print(f"     Attention: {analysis['pathway_contributions']['attention']:.4f}")
    
    return {
        'dataset_name': dataset_name,
        'dataset_size': dataset_size,
        'config': config,
        'training_results': training_results,
        'test_results': test_results,
        'model': model
    }

def compare_with_best_individual(cccv3_result, individual_results):
    """Compare CCCV3 with best individual pathway"""
    
    dataset_name = cccv3_result['dataset_name']
    cccv3_mse = cccv3_result['test_results']['mse']
    
    # Best individual results from previous testing
    individual_best = {
        'miyawaki': 0.008796,    # Attention pathway
        'vangerven': 0.036195,   # CLIP pathway
        'mindbigdata': 0.056781, # Attention pathway
        'crell': 0.032119        # Attention pathway
    }
    
    best_individual_mse = individual_best.get(dataset_name, 0.05)
    
    print(f"\n📊 CCCV3 vs Best Individual Pathway:")
    print(f"   Best Individual: {best_individual_mse:.6f}")
    print(f"   CCCV3 Multi-Pathway: {cccv3_mse:.6f}")
    
    if cccv3_mse < best_individual_mse:
        improvement = ((best_individual_mse - cccv3_mse) / best_individual_mse) * 100
        print(f"   🏆 CCCV3 WINS by {improvement:.2f}%!")
        return 'cccv3', improvement
    else:
        gap = ((cccv3_mse - best_individual_mse) / best_individual_mse) * 100
        print(f"   📈 Individual wins by {gap:.2f}%")
        return 'individual', -gap

def main():
    """Main CCCV3 complete testing function"""
    print("🎯 CCCV3 Complete Multi-Pathway Testing")
    print("=" * 50)
    print("🔬 Testing complete CCCV3 multi-pathway model")
    print("📊 Goal: 100% success rate with 5-15% improvements")
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
        result = test_cccv3_on_dataset(dataset_name, device)
        if result:
            all_results[dataset_name] = result
            
            # Compare with best individual
            winner, improvement = compare_with_best_individual(result, {})
            result['vs_individual'] = {'winner': winner, 'improvement': improvement}
    
    # Final analysis
    print("\n🎉 CCCV3 Complete Multi-Pathway Testing Complete!")
    print("=" * 60)
    
    cccv3_wins = 0
    total_datasets = len(all_results)
    
    for dataset_name, result in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"   CCCV3 MSE: {result['test_results']['mse']:.6f}")
        print(f"   vs Individual: {result['vs_individual']['winner']} (+{result['vs_individual']['improvement']:.2f}%)")
        
        if result['vs_individual']['winner'] == 'cccv3':
            cccv3_wins += 1
    
    print(f"\n🏆 FINAL CCCV3 MULTI-PATHWAY RESULTS:")
    print(f"CCCV3 wins: {cccv3_wins}/{total_datasets}")
    print(f"Success rate: {(cccv3_wins/total_datasets)*100:.1f}%")
    
    if cccv3_wins == total_datasets:
        print(f"\n🎉 PERFECT SUCCESS! 100% WIN RATE!")
        print(f"🚀 CCCV3 Multi-Pathway achieves complete dominance!")
    elif cccv3_wins >= total_datasets * 0.75:
        print(f"\n🎉 EXCELLENT SUCCESS! 75%+ WIN RATE!")
        print(f"🚀 CCCV3 Multi-Pathway shows strong improvements!")
    elif cccv3_wins >= total_datasets * 0.5:
        print(f"\n✅ GOOD SUCCESS! 50%+ WIN RATE!")
        print(f"🔧 CCCV3 shows promise with room for optimization!")
    else:
        print(f"\n🔧 NEEDS IMPROVEMENT")
        print(f"📈 Consider fusion strategy adjustments")
    
    print("\n🚀 CCCV3 Multi-Pathway Development Complete!")

if __name__ == "__main__":
    main()
