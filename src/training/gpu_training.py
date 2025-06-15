"""
GPU-Optimized Training Functions
===============================

GPU-optimized training functions with mixed precision, advanced scheduling,
and WSL compatibility for neural decoding models.

Key Features:
    - Mixed precision training for memory efficiency
    - Advanced learning rate scheduling
    - Early stopping with patience
    - Gradient clipping for stability
    - WSL-compatible data loading
    - Comprehensive logging

Functions:
    gpu_optimized_training: Main GPU training function with mixed precision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time


def gpu_optimized_training(model, X_train, y_train, X_val, y_val, epochs=150, lr=0.001, batch_size=64, patience=35):
    """
    GPU-optimized training dengan mixed precision.
    
    This function provides state-of-the-art GPU training with:
    - Mixed precision for memory efficiency and speed
    - Advanced learning rate scheduling
    - Early stopping for optimal convergence
    - Gradient clipping for training stability
    - WSL-compatible configuration
    
    Args:
        model: Neural network model to train
        X_train: Training input data [batch_size, input_dim]
        y_train: Training target data [batch_size, 1, 28, 28]
        X_val: Validation input data [batch_size, input_dim]
        y_val: Validation target data [batch_size, 1, 28, 28]
        epochs: Maximum number of training epochs (default: 150)
        lr: Learning rate (default: 0.001)
        batch_size: Batch size for training (default: 64)
        patience: Early stopping patience (default: 35)
        
    Returns:
        float: Best validation loss achieved during training
    """
    
    print(f"🔥 GPU Training {model.name} dengan mixed precision...")
    
    # Setup untuk mixed precision
    scaler = torch.amp.GradScaler('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()
    
    # DataLoader untuk batch processing (no workers untuk WSL compatibility)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=False)
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        model.train()
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"   Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Time: {elapsed:.1f}s")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"✅ {model.name} training completed in {total_time:.1f}s, Best Loss: {best_loss:.6f}")
    return best_loss


def gpu_optimized_training_with_advanced_scheduling(model, X_train, y_train, X_val, y_val, 
                                                   epochs=100, lr=0.001, batch_size=64, patience=20):
    """
    GPU-optimized training with advanced OneCycleLR scheduling.
    
    This version includes:
    - OneCycleLR scheduler with warmup
    - Advanced optimization techniques
    - KNN memory updates (if supported by model)
    - Enhanced mixed precision training
    
    Args:
        model: Neural network model to train
        X_train: Training input data
        y_train: Training target data
        X_val: Validation input data
        y_val: Validation target data
        epochs: Maximum number of training epochs
        lr: Base learning rate
        batch_size: Batch size for training
        patience: Early stopping patience
        
    Returns:
        float: Best validation loss achieved during training
    """
    
    # Setup optimizer dan loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if model.device == 'cuda' else None

    # Check if model has KNN capabilities
    has_knn = hasattr(model, 'update_knn_memory')

    # ADVANCED: Learning rate scheduler with warmup
    warmup_epochs = min(10, epochs // 10)  # 10% of total epochs for warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 2,  # Peak LR is 2x base LR
        epochs=epochs,
        steps_per_epoch=len(X_train) // batch_size + 1,
        pct_start=warmup_epochs / epochs,  # Warmup percentage
        anneal_strategy='cos'  # Cosine annealing
    )

    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)

    best_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            if model.device == 'cuda' and scaler:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    output = model(batch_X)
                    # Handle tuple output from some models
                    if isinstance(output, tuple):
                        output = output[0]  # Use mean prediction
                    loss = criterion(output, batch_y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate
            else:
                # Standard training
                output = model(batch_X)
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            if model.device == 'cuda' and scaler:
                with torch.cuda.amp.autocast():
                    val_output = model(X_val)
                    if isinstance(val_output, tuple):
                        val_output = val_output[0]
                    val_loss = criterion(val_output, y_val).item()
            else:
                val_output = model(X_val)
                if isinstance(val_output, tuple):
                    val_output = val_output[0]
                val_loss = criterion(val_output, y_val).item()
        model.train()

        # KNN memory update (if supported)
        if has_knn and epoch % 10 == 0:
            try:
                model.update_knn_memory(X_train[:100], y_train[:100])  # Update with subset
            except:
                pass  # Ignore if update fails

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"   Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/len(train_loader):.6f}, "
                  f"Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    print(f"✅ {model.name} advanced training completed, Best Loss: {best_loss:.6f}")
    return best_loss
