"""
FUNPOLYMER - Visualization Module
==================================

Tools for visualizing NMR diffusion data and neural network predictions.

Author: FUNPOLYMER Project - Universidad de Almería
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_attenuation(attenuation: np.ndarray,
                    g_values: Optional[np.ndarray] = None,
                    title: str = "NMR Diffusion Attenuation",
                    ax: Optional[plt.Axes] = None,
                    **kwargs) -> plt.Axes:
    """
    Plot attenuation signal vs gradient strength.
    
    Args:
        attenuation: Attenuation signal
        g_values: Gradient values (optional)
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        **kwargs: Additional plot arguments
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    n_points = len(attenuation)
    if g_values is None:
        g_values = np.arange(n_points)
        xlabel = 'Gradient Step'
    else:
        xlabel = 'Gradient Strength (T/m)'
    
    ax.semilogy(g_values, attenuation, 'o-', **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Normalized Signal Intensity')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_distribution(distribution: np.ndarray,
                     D_grid: np.ndarray,
                     title: str = "Diffusion Coefficient Distribution",
                     ax: Optional[plt.Axes] = None,
                     label: Optional[str] = None,
                     **kwargs) -> plt.Axes:
    """
    Plot diffusion coefficient distribution.
    
    Args:
        distribution: Distribution values
        D_grid: Diffusion coefficient grid
        title: Plot title
        ax: Matplotlib axes
        label: Legend label
        **kwargs: Additional plot arguments
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.semilogx(D_grid, distribution, label=label, **kwargs)
    ax.set_xlabel('Diffusion Coefficient (m²/s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if label:
        ax.legend()
    
    return ax


def plot_comparison(pred_dist: np.ndarray,
                   true_dist: np.ndarray,
                   D_grid: np.ndarray,
                   title: str = "Prediction vs Ground Truth",
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison between predicted and true distributions.
    
    Args:
        pred_dist: Predicted distribution
        true_dist: Ground truth distribution
        D_grid: Diffusion coefficient grid
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Distribution comparison
    axes[0].semilogx(D_grid, true_dist, 'b-', linewidth=2, label='Ground Truth')
    axes[0].semilogx(D_grid, pred_dist, 'r--', linewidth=2, label='Prediction')
    axes[0].set_xlabel('Diffusion Coefficient (m²/s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error plot
    error = pred_dist - true_dist
    axes[1].semilogx(D_grid, error, 'g-', linewidth=1.5)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Diffusion Coefficient (m²/s)')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Prediction Error')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_training_history(history_path: str,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training history from JSON file.
    
    Args:
        history_path: Path to training_history.json
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # MSE plot
    axes[0, 1].plot(epochs, history['train_mse'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_mse'], 'r-', label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('Mean Squared Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Learning rate
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Epoch time
    axes[1, 1].plot(epochs, history['epoch_times'], 'purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].set_title('Epoch Duration')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_noise_robustness(results: Dict,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot noise robustness evaluation results.
    
    Args:
        results: Dictionary from evaluate_noise_robustness()
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    snr = results['snr']
    
    # MSE vs SNR
    axes[0, 0].plot(snr, results['mse'], 'bo-', markersize=8)
    axes[0, 0].set_xlabel('SNR')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('MSE vs Signal-to-Noise Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    
    # MAE vs SNR
    axes[0, 1].plot(snr, results['mae'], 'rs-', markersize=8)
    axes[0, 1].set_xlabel('SNR')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('MAE vs Signal-to-Noise Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    
    # Correlation vs SNR
    axes[1, 0].plot(snr, results['pearson_r'], 'g^-', markersize=8)
    axes[1, 0].set_xlabel('SNR')
    axes[1, 0].set_ylabel('Pearson r')
    axes[1, 0].set_title('Correlation vs Signal-to-Noise Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_ylim([0.9, 1.0])
    
    # D error vs SNR
    axes[1, 1].plot(snr, results['D_mean_rel_error_%'], 'md-', markersize=8)
    axes[1, 1].set_xlabel('SNR')
    axes[1, 1].set_ylabel('Relative Error (%)')
    axes[1, 1].set_title('D Estimation Error vs SNR')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_sample_predictions(model,
                           data_generator,
                           n_samples: int = 6,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot sample predictions for different distribution types.
    
    Args:
        model: Trained model
        data_generator: SyntheticNMRDataGenerator instance
        n_samples: Number of samples to show
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    import torch
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    distribution_configs = [
        {'distribution_type': 'single', 'D_center': 5e-11, 'width': 0.08},
        {'distribution_type': 'single', 'D_center': 1e-10, 'width': 0.15},
        {'distribution_type': 'bimodal', 'D1': 2e-11, 'D2': 2e-10},
        {'distribution_type': 'bimodal', 'D1': 5e-11, 'D2': 1.5e-10},
        {'distribution_type': 'polydisperse', 'D_center': 1e-10, 'PDI': 1.5},
        {'distribution_type': 'polydisperse', 'D_center': 5e-11, 'PDI': 2.0},
    ]
    
    titles = [
        'Narrow Single Peak',
        'Broad Single Peak', 
        'Bimodal (Far)',
        'Bimodal (Close)',
        'Polydisperse (Low PDI)',
        'Polydisperse (High PDI)'
    ]
    
    D_grid = data_generator.dist_generator.D_grid
    
    for i, (config, title) in enumerate(zip(distribution_configs, titles)):
        attenuation, true_dist, _ = data_generator.generate_sample(
            noise_snr=100,
            **config
        )
        
        # Predict
        with torch.no_grad():
            x = torch.FloatTensor(attenuation).unsqueeze(0)
            pred_dist = model(x).cpu().numpy()[0]
        
        # Plot
        axes[i].semilogx(D_grid, true_dist, 'b-', linewidth=2, label='Ground Truth')
        axes[i].semilogx(D_grid, pred_dist, 'r--', linewidth=2, label='Prediction')
        axes[i].set_xlabel('D (m²/s)')
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(title)
        axes[i].legend(loc='upper right', fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def create_summary_figure(model_path: str,
                         history_path: str,
                         save_path: str = 'summary_figure.png'):
    """
    Create comprehensive summary figure for the model.
    
    Args:
        model_path: Path to model checkpoint
        history_path: Path to training history
        save_path: Path to save figure
    """
    import torch
    from data_generator import SyntheticNMRDataGenerator
    from models import create_model
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_model(
        model_type='standard',
        input_size=checkpoint['model_config']['input_size'],
        output_size=checkpoint['model_config']['output_size'],
        hidden_sizes=checkpoint['model_config']['hidden_sizes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create data generator
    generator = SyntheticNMRDataGenerator(n_gradient_steps=checkpoint['model_config']['input_size'])
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Training loss
    ax1 = fig.add_subplot(gs[0, 0:2])
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.semilogy(epochs, history['train_loss'], 'b-', label='Train')
    ax1.semilogy(epochs, history['val_loss'], 'r-', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.semilogy(epochs, history['learning_rates'], 'g-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    
    # Sample predictions
    D_grid = generator.dist_generator.D_grid
    
    configs = [
        {'distribution_type': 'single', 'D_center': 1e-10, 'width': 0.1},
        {'distribution_type': 'bimodal', 'D1': 3e-11, 'D2': 3e-10},
        {'distribution_type': 'polydisperse', 'D_center': 8e-11, 'PDI': 1.8},
        {'distribution_type': 'single', 'D_center': 5e-11, 'width': 0.2},
    ]
    
    for i, config in enumerate(configs):
        ax = fig.add_subplot(gs[1 + i // 2, (i % 2) * 2:(i % 2) * 2 + 2])
        
        attenuation, true_dist, _ = generator.generate_sample(noise_snr=100, **config)
        
        with torch.no_grad():
            x = torch.FloatTensor(attenuation).unsqueeze(0)
            pred_dist = model(x).cpu().numpy()[0]
        
        ax.semilogx(D_grid, true_dist, 'b-', linewidth=2, label='Target')
        ax.semilogx(D_grid, pred_dist, 'r--', linewidth=2, label='Predicted')
        ax.set_xlabel('D (m²/s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f"Sample {i+1}: {config['distribution_type'].capitalize()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Summary figure saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization module...")
    
    setup_publication_style()
    
    # Generate sample data
    from data_generator import SyntheticNMRDataGenerator
    
    generator = SyntheticNMRDataGenerator(n_gradient_steps=23)
    attenuation, distribution, _ = generator.generate_sample(
        noise_snr=100,
        distribution_type='bimodal',
        D1=5e-11,
        D2=5e-10
    )
    
    # Test plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    plot_attenuation(attenuation, ax=axes[0], title="Test Attenuation")
    plot_distribution(distribution, generator.dist_generator.D_grid, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('/tmp/test_visualization.png')
    print("Visualization test completed!")
