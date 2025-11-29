"""
FUNPOLYMER - Deep Neural Network for Diffusion Coefficient Estimation
======================================================================

This module implements a deep neural network with 4 hidden layers for 
solving the inverse Laplace transform (ILT) problem in NMR diffusion
experiments.

Architecture:
- Input: Attenuation signal (n_gradient_steps points)
- 4 Hidden layers with configurable neurons
- Output: Diffusion coefficient distribution (1024 bins)

The network uses L2 regularization in the cost function as specified
in the FUNPOLYMER project documentation.

Author: FUNPOLYMER Project - Universidad de Almería
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class DiffusionNet(nn.Module):
    """
    Deep Neural Network for NMR Diffusion Coefficient Distribution Estimation.
    
    Architecture following FUNPOLYMER Activity 7 specifications:
    - 4 hidden layers
    - 1024 output positions for diffusion coefficient distribution
    - L2 regularization (weight decay)
    - Backpropagation training
    """
    
    def __init__(self,
                 input_size: int = 23,
                 output_size: int = 1024,
                 hidden_sizes: List[int] = [512, 1024, 1024, 512],
                 dropout_rate: float = 0.2,
                 activation: str = 'relu'):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of gradient steps in input (23 or 46 as per project)
            output_size: Resolution of output distribution (1024)
            hidden_sizes: List of hidden layer sizes (4 layers)
            dropout_rate: Dropout probability for regularization
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'selu')
        """
        super(DiffusionNet, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization for stable training
            layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1, inplace=True))
            elif activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            elif activation == 'selu':
                layers.append(nn.SELU(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
            
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # Build sequential model
        self.network = nn.Sequential(*layers)
        
        # Softmax for normalized distribution output
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input attenuation signal (batch_size, input_size)
            
        Returns:
            Predicted diffusion distribution (batch_size, output_size)
        """
        # Pass through network
        out = self.network(x)
        
        # Apply softmax to get normalized distribution
        out = self.softmax(out)
        
        return out
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            x: Input attenuation signal
            n_samples: Number of forward passes
            
        Returns:
            mean_prediction: Average prediction
            std_prediction: Standard deviation (uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        self.eval()  # Return to evaluation mode
        
        return mean_pred, std_pred


class DiffusionNetLarge(nn.Module):
    """
    Larger version of DiffusionNet for 46 gradient variations.
    
    Extended architecture for processing datasets with more gradient
    strength variations as specified in FUNPOLYMER Activity 7.
    """
    
    def __init__(self,
                 input_size: int = 46,
                 output_size: int = 1024,
                 hidden_sizes: List[int] = [512, 1024, 2048, 1024, 512],
                 dropout_rate: float = 0.3):
        """
        Initialize the larger network variant.
        """
        super(DiffusionNetLarge, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        self.middle = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.BatchNorm1d(hidden_sizes[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_sizes[3], hidden_sizes[4]),
            nn.BatchNorm1d(hidden_sizes[4]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_sizes[4], output_size),
        )
        
        self.softmax = nn.Softmax(dim=1)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return self.softmax(x)


class DiffusionResNet(nn.Module):
    """
    Residual Network variant for improved gradient flow.
    
    Uses skip connections for better training of deep networks.
    """
    
    def __init__(self,
                 input_size: int = 23,
                 output_size: int = 1024,
                 hidden_size: int = 512,
                 n_blocks: int = 4,
                 dropout_rate: float = 0.2):
        """
        Initialize ResNet-style architecture.
        """
        super(DiffusionResNet, self).__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate) for _ in range(n_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.output_proj(x)
        return self.softmax(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, hidden_size: int, dropout_rate: float = 0.2):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = out + residual
        return self.relu(out)


class DiffusionLoss(nn.Module):
    """
    Custom loss function for diffusion distribution prediction.
    
    Combines multiple loss components:
    - MSE loss for distribution accuracy
    - KL divergence for distribution matching
    - L2 regularization (handled by optimizer weight_decay)
    - Smoothness regularization for continuous distributions
    """
    
    def __init__(self,
                 mse_weight: float = 1.0,
                 kl_weight: float = 0.1,
                 smoothness_weight: float = 0.01):
        """
        Initialize loss function.
        
        Args:
            mse_weight: Weight for MSE loss
            kl_weight: Weight for KL divergence
            smoothness_weight: Weight for smoothness regularization
        """
        super(DiffusionLoss, self).__init__()
        
        self.mse_weight = mse_weight
        self.kl_weight = kl_weight
        self.smoothness_weight = smoothness_weight
        
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Calculate combined loss.
        
        Args:
            pred: Predicted distribution
            target: Target distribution
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # MSE loss
        mse_loss = self.mse(pred, target)
        
        # KL divergence (add small epsilon for numerical stability)
        eps = 1e-10
        kl_loss = self.kl(torch.log(pred + eps), target + eps)
        
        # Smoothness regularization (penalize high-frequency variations)
        smoothness_loss = torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))
        
        # Total loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.kl_weight * kl_loss + 
                     self.smoothness_weight * smoothness_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'kl': kl_loss.item(),
            'smoothness': smoothness_loss.item()
        }
        
        return total_loss, loss_dict


def create_model(model_type: str = 'standard',
                 input_size: int = 23,
                 output_size: int = 1024,
                 **kwargs) -> nn.Module:
    """
    Factory function to create diffusion models.
    
    Args:
        model_type: 'standard', 'large', or 'resnet'
        input_size: Number of input features
        output_size: Number of output bins
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model
    """
    if model_type == 'standard':
        return DiffusionNet(input_size=input_size, output_size=output_size, **kwargs)
    elif model_type == 'large':
        return DiffusionNetLarge(input_size=input_size, output_size=output_size, **kwargs)
    elif model_type == 'resnet':
        return DiffusionResNet(input_size=input_size, output_size=output_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    print("Testing Neural Network Models...")
    
    # Test standard model
    model = DiffusionNet(input_size=23, output_size=1024)
    print(f"\nStandard DiffusionNet:")
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(32, 23)
    y = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Output sum: {y.sum(dim=1).mean():.4f} (should be ~1.0)")
    
    # Test large model
    model_large = DiffusionNetLarge(input_size=46, output_size=1024)
    print(f"\nLarge DiffusionNet:")
    print(f"  Parameters: {count_parameters(model_large):,}")
    
    x_large = torch.randn(32, 46)
    y_large = model_large(x_large)
    print(f"  Input shape: {x_large.shape}")
    print(f"  Output shape: {y_large.shape}")
    
    # Test ResNet model
    model_resnet = DiffusionResNet(input_size=23, output_size=1024)
    print(f"\nDiffusionResNet:")
    print(f"  Parameters: {count_parameters(model_resnet):,}")
    
    # Test loss function
    loss_fn = DiffusionLoss()
    pred = torch.softmax(torch.randn(32, 1024), dim=1)
    target = torch.softmax(torch.randn(32, 1024), dim=1)
    loss, loss_dict = loss_fn(pred, target)
    print(f"\nLoss function test:")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  MSE: {loss_dict['mse']:.4f}")
    print(f"  KL: {loss_dict['kl']:.4f}")
    print(f"  Smoothness: {loss_dict['smoothness']:.4f}")
    
    print("\nModel tests completed successfully!")
