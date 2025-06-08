#!/usr/bin/env python3
"""
CortexFlow Training and Evaluation Script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy import stats
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from cortexflow_models import (
    CortexFlowSimple, CortexFlowMC, CortexFlowHierarchical, 
    CortexFlowEnhanced, CortexFlowUnified
)

class CortexFlowTrainer:
    def __init__(self, results_dir="results/actual_experiments"):
        self.results_dir = Path(results_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Trainer initialized on {self.device}")
        
        # Training history storage
        self.training_history = {}
        
    def create_model(self, variant, input_dim):
        """Create CortexFlow model variant"""
        models = {
            'simple': CortexFlowSimple(input_dim),
            'mc': CortexFlowMC(input_dim),
            'hierarchical': CortexFlowHierarchical(input_dim),
            'enhanced': CortexFlowEnhanced(input_dim),
            'unified': CortexFlowUnified(input_dim)
        }
        return models[variant].to(self.device)
    
    def train_model(self, model, train_loader, val_loader, variant, dataset_name, epochs=100):
        """Train a CortexFlow model variant"""
        print(f"🚀 Training {variant} on {dataset_name}...")
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass based on variant
                if variant == 'simple':
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                elif variant == 'mc':
                    output, uncertainty = model(batch_x)
                    recon_loss = criterion(output, batch_y)
                    uncertainty_loss = torch.mean(torch.log(uncertainty) + (batch_y - output)**2 / uncertainty)
                    loss = recon_loss + 0.1 * uncertainty_loss
                elif variant == 'hierarchical':
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                elif variant == 'enhanced':
                    output, uncertainty, alignment = model(batch_x)
                    recon_loss = criterion(output, batch_y)
                    uncertainty_loss = torch.mean(torch.log(uncertainty) + (batch_y - output)**2 / uncertainty)
                    alignment_loss = torch.mean(alignment**2)  # L2 regularization
                    loss = 0.6 * recon_loss + 0.3 * uncertainty_loss + 0.1 * alignment_loss
                elif variant == 'unified':
                    if epoch < 50:  # Simple config for first half
                        output = model(batch_x, config='simple')
                        loss = criterion(output, batch_y)
                    else:  # Balanced config for second half
                        output, uncertainty, complexity = model(batch_x, config='balanced')
                        recon_loss = criterion(output, batch_y)
                        uncertainty_loss = torch.mean(torch.log(uncertainty) + (batch_y - output)**2 / uncertainty)
                        complexity_loss = torch.mean((complexity - 0.5)**2)  # Encourage balanced complexity
                        loss = recon_loss + 0.1 * uncertainty_loss + 0.05 * complexity_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    if variant == 'simple':
                        output = model(batch_x)
                        val_loss = criterion(output, batch_y)
                    elif variant == 'mc':
                        output, _ = model(batch_x)
                        val_loss = criterion(output, batch_y)
                    elif variant == 'hierarchical':
                        output = model(batch_x)
                        val_loss = criterion(output, batch_y)
                    elif variant == 'enhanced':
                        output, _, _ = model(batch_x)
                        val_loss = criterion(output, batch_y)
                    elif variant == 'unified':
                        if epoch < 50:
                            output = model(batch_x, config='simple')
                        else:
                            output, _, _ = model(batch_x, config='balanced')
                        val_loss = criterion(output, batch_y)
                    
                    val_losses.append(val_loss.item())
            
            # Record history
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['epochs'].append(epoch)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 
                          self.results_dir / "models" / f"{variant}_{dataset_name}_best.pt")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        training_time = time.time() - start_time
        
        # Store training history
        history['training_time'] = training_time
        history['best_val_loss'] = best_val_loss
        history['final_epoch'] = epoch
        
        self.training_history[f"{variant}_{dataset_name}"] = history
        
        print(f"✅ Training completed in {training_time:.2f}s, Best Val Loss: {best_val_loss:.6f}")
        
        return model, history
    
    def evaluate_model(self, model, test_loader, variant, dataset_name):
        """Evaluate model and compute metrics"""
        print(f"📊 Evaluating {variant} on {dataset_name}...")
        
        model.eval()
        predictions = []
        targets = []
        uncertainties = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                if variant in ['mc', 'enhanced']:
                    # Monte Carlo sampling for uncertainty
                    if variant == 'mc':
                        pred, epistemic, aleatoric = model(batch_x, mc_samples=10)
                        uncertainty = epistemic + aleatoric
                    else:  # enhanced
                        pred, epistemic, aleatoric, _ = model(batch_x, mc_samples=10)
                        uncertainty = epistemic + aleatoric
                    
                    uncertainties.append(uncertainty.cpu().numpy())
                elif variant == 'unified':
                    pred, uncertainty, complexity = model(batch_x, mc_samples=10, config='balanced')
                    uncertainties.append(uncertainty.cpu().numpy())
                else:
                    if variant == 'simple':
                        pred = model(batch_x)
                    elif variant == 'hierarchical':
                        pred = model(batch_x)
                
                predictions.append(pred.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Compute metrics
        mse = mean_squared_error(targets.flatten(), predictions.flatten())
        
        # SSIM approximation (simplified)
        ssim_scores = []
        for i in range(len(predictions)):
            pred_img = predictions[i].reshape(28, 28)
            target_img = targets[i].reshape(28, 28)
            
            # Simplified SSIM calculation
            mu1, mu2 = pred_img.mean(), target_img.mean()
            sigma1, sigma2 = pred_img.std(), target_img.std()
            sigma12 = np.mean((pred_img - mu1) * (target_img - mu2))
            
            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
            ssim_scores.append(max(0, min(1, ssim)))  # Clamp to [0,1]
        
        ssim_mean = np.mean(ssim_scores)
        ssim_std = np.std(ssim_scores)
        
        results = {
            'mse': mse,
            'mse_std': np.std([mean_squared_error(targets[i], predictions[i]) for i in range(len(predictions))]),
            'ssim': ssim_mean,
            'ssim_std': ssim_std,
            'predictions': predictions,
            'targets': targets
        }
        
        if uncertainties:
            uncertainties = np.concatenate(uncertainties, axis=0)
            results['uncertainties'] = uncertainties
            results['uncertainty_mean'] = uncertainties.mean()
            results['uncertainty_std'] = uncertainties.std()
        
        print(f"✅ Evaluation complete: MSE = {mse:.6f}, SSIM = {ssim_mean:.3f}")
        
        return results

def main():
    print("🚀 Starting CortexFlow Training and Evaluation...")
    print("=" * 60)
    
    trainer = CortexFlowTrainer()
    
    # Load synthetic data
    data_path = trainer.results_dir / "data" / "synthetic_datasets.pt"
    if not data_path.exists():
        print("❌ Synthetic data not found. Run data generation first.")
        return
    
    data = torch.load(data_path)
    print("✅ Loaded synthetic datasets")
    
    # Training configuration
    variants = ['simple', 'mc', 'hierarchical', 'enhanced', 'unified']
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n📊 Processing dataset: {dataset_name}")
        
        X, y = data[dataset_name]['X'], data[dataset_name]['y']
        
        # Split data
        n_train = int(0.7 * len(X))
        n_val = int(0.15 * len(X))
        
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        dataset_results = {}
        
        for variant in variants:
            print(f"\n🔧 Training {variant} variant...")
            
            # Create and train model
            model = trainer.create_model(variant, X.shape[1])
            trained_model, history = trainer.train_model(
                model, train_loader, val_loader, variant, dataset_name, epochs=100
            )
            
            # Evaluate model
            results = trainer.evaluate_model(trained_model, test_loader, variant, dataset_name)
            
            dataset_results[variant] = {
                'training_history': history,
                'evaluation_results': results
            }
        
        all_results[dataset_name] = dataset_results
    
    # Save all results
    results_file = trainer.results_dir / "data" / "training_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for dataset, dataset_data in all_results.items():
        json_results[dataset] = {}
        for variant, variant_data in dataset_data.items():
            json_results[dataset][variant] = {
                'training_history': variant_data['training_history'],
                'evaluation_results': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in variant_data['evaluation_results'].items()
                    if k not in ['predictions', 'targets', 'uncertainties']  # Skip large arrays
                }
            }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    print("🎯 Training and evaluation complete!")

if __name__ == "__main__":
    main()
