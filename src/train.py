"""
FUNPOLYMER - Training Pipeline for Diffusion Coefficient Neural Network
========================================================================

This module implements the complete training pipeline including:
- Training with validation
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Training visualization
- L2 regularization via weight decay

Author: FUNPOLYMER Project - Universidad de Almería
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime
import os

from data_generator import create_dataloaders, SyntheticNMRDataGenerator, NMRDiffusionDataset
from models import DiffusionNet, DiffusionLoss, create_model, count_parameters


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class TrainingHistory:
    """Track training metrics."""
    
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_mse = []
        self.val_mse = []
        self.learning_rates = []
        self.epoch_times = []
        
    def update(self, train_loss: float, val_loss: float, 
               train_mse: float, val_mse: float, 
               lr: float, epoch_time: float):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_mse.append(train_mse)
        self.val_mse.append(val_mse)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        
    def save(self, path: str):
        data = {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.train_loss = data['train_loss']
        self.val_loss = data['val_loss']
        self.train_mse = data['train_mse']
        self.val_mse = data['val_mse']
        self.learning_rates = data['learning_rates']
        self.epoch_times = data['epoch_times']


class Trainer:
    """
    Neural network trainer for diffusion coefficient estimation.
    
    Implements training loop with:
    - L2 regularization (weight decay in optimizer)
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Gradient clipping
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'auto',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,    # L2 regularization
                 gradient_clip: float = 1.0,
                 checkpoint_dir: str = 'checkpoints'):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            device: 'cuda', 'cpu', or 'auto'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            gradient_clip: Maximum gradient norm
            checkpoint_dir: Directory for saving checkpoints
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Loss function
        self.criterion = DiffusionLoss(mse_weight=1.0, kl_weight=0.1, smoothness_weight=0.01)
        
        # Optimizer with L2 regularization (weight decay)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = TrainingHistory()
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average total loss
            avg_mse: Average MSE loss
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Calculate loss
            loss, loss_dict = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss_dict['total']
            total_mse += loss_dict['mse']
            n_batches += 1
            
        return total_loss / n_batches, total_mse / n_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Returns:
            avg_loss: Average total loss
            avg_mse: Average MSE loss
        """
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                loss, loss_dict = self.criterion(output, target)
                
                total_loss += loss_dict['total']
                total_mse += loss_dict['mse']
                n_batches += 1
                
        return total_loss / n_batches, total_mse / n_batches
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             n_epochs: int = 100,
             early_stopping_patience: int = 15,
             save_best: bool = True,
             verbose: bool = True) -> TrainingHistory:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Maximum number of epochs
            early_stopping_patience: Epochs to wait before stopping
            save_best: Save best model checkpoint
            verbose: Print progress
            
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print("-" * 70)
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_mse = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_mse = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history.update(train_loss, val_loss, train_mse, val_mse, current_lr, epoch_time)
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch, val_loss)
                
            # Print progress
            if verbose:
                print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Val MSE: {val_mse:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.1f}s")
            
            # Early stopping check
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        print("-" * 70)
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        # Save final model and history
        self.save_checkpoint('final_model.pth', epoch, val_loss)
        self.history.save(str(self.checkpoint_dir / 'training_history.json'))
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'model_config': {
                'input_size': self.model.input_size,
                'output_size': self.model.output_size,
                'hidden_sizes': self.model.hidden_sizes
            }
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']


def train_model(n_gradient_steps: int = 23,
               n_train: int = 50000,
               n_val: int = 5000,
               batch_size: int = 64,
               n_epochs: int = 100,
               learning_rate: float = 1e-3,
               weight_decay: float = 1e-4,
               checkpoint_dir: str = 'checkpoints',
               seed: int = 42) -> Tuple[nn.Module, TrainingHistory]:
    """
    Complete training pipeline.
    
    Args:
        n_gradient_steps: Number of gradient variations (23 or 46)
        n_train: Number of training samples
        n_val: Number of validation samples
        batch_size: Batch size
        n_epochs: Maximum epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        checkpoint_dir: Directory for checkpoints
        seed: Random seed
        
    Returns:
        Trained model and training history
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create data loaders
    print("=" * 70)
    print("FUNPOLYMER - NMR Diffusion Deep Learning Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Gradient steps: {n_gradient_steps}")
    print(f"  Training samples: {n_train:,}")
    print(f"  Validation samples: {n_val:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {n_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay (L2): {weight_decay}")
    
    train_loader, val_loader, _ = create_dataloaders(
        n_train=n_train,
        n_val=n_val,
        n_test=1000,
        batch_size=batch_size,
        n_gradient_steps=n_gradient_steps,
        seed=seed
    )
    
    # Create model
    model = create_model(
        model_type='standard',
        input_size=n_gradient_steps,
        output_size=1024,
        hidden_sizes=[512, 1024, 1024, 512],
        dropout_rate=0.2
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        early_stopping_patience=15
    )
    
    return model, history


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train FUNPOLYMER Diffusion Network')
    parser.add_argument('--gradient-steps', type=int, default=23, choices=[23, 46],
                       help='Number of gradient variations')
    parser.add_argument('--n-train', type=int, default=50000,
                       help='Number of training samples')
    parser.add_argument('--n-val', type=int, default=5000,
                       help='Number of validation samples')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='L2 regularization')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    model, history = train_model(
        n_gradient_steps=args.gradient_steps,
        n_train=args.n_train,
        n_val=args.n_val,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed
    )
    
    print("\nTraining completed successfully!")
    print(f"Best model saved to: {args.checkpoint_dir}/best_model.pth")


if __name__ == "__main__":
    main()
