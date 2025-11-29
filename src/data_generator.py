"""
FUNPOLYMER - Synthetic NMR Diffusion Data Generator
====================================================

This module generates synthetic NMR diffusion decay data based on the 
Stejskal-Tanner equation for training deep learning models.

The attenuation signal follows:
    E(g) = ∫ A(D) * exp(-D * γ² * δ² * g² * Δ') dD

For discrete components:
    E(g) = Σ Aᵢ * exp(-Dᵢ * b)
    
where b = γ² * δ² * g² * Δ' (b-value)

Author: FUNPOLYMER Project - Universidad de Almería
License: MIT
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class NMRDiffusionParameters:
    """Physical parameters for NMR diffusion experiments."""
    gamma: float = 267.52218744e6  # Gyromagnetic ratio for 1H (rad/s/T)
    delta: float = 2e-3            # Gradient pulse duration (s)
    Delta: float = 50e-3           # Diffusion time (s)
    
    def calculate_b_value(self, g: float) -> float:
        """Calculate b-value for given gradient strength."""
        return (self.gamma ** 2) * (self.delta ** 2) * (g ** 2) * (self.Delta - self.delta / 3)


class DiffusionDistribution:
    """
    Class to represent diffusion coefficient distributions.
    
    Supports:
    - Single exponential (monodisperse)
    - Bimodal distributions
    - Continuous (polydisperse) distributions
    - Log-normal distributions
    """
    
    def __init__(self, 
                 D_min: float = 1e-12,      # m²/s
                 D_max: float = 1e-8,       # m²/s
                 n_bins: int = 1024):
        """
        Initialize diffusion distribution grid.
        
        Args:
            D_min: Minimum diffusion coefficient
            D_max: Maximum diffusion coefficient
            n_bins: Number of bins in the distribution (output resolution)
        """
        self.D_min = D_min
        self.D_max = D_max
        self.n_bins = n_bins
        
        # Create logarithmic grid for diffusion coefficients
        self.D_grid = np.logspace(np.log10(D_min), np.log10(D_max), n_bins)
        self.log_D_grid = np.log10(self.D_grid)
        
    def generate_single_peak(self, 
                            D_center: float, 
                            width: float = 0.1,
                            amplitude: float = 1.0) -> np.ndarray:
        """
        Generate a single Gaussian peak in log-D space.
        
        Args:
            D_center: Center diffusion coefficient
            width: Width in log10 units
            amplitude: Peak amplitude
        """
        log_D_center = np.log10(D_center)
        distribution = amplitude * np.exp(-0.5 * ((self.log_D_grid - log_D_center) / width) ** 2)
        return distribution / (np.sum(distribution) + 1e-10)
    
    def generate_bimodal(self,
                        D1: float, D2: float,
                        width1: float = 0.1, width2: float = 0.1,
                        ratio: float = 0.5) -> np.ndarray:
        """
        Generate bimodal distribution.
        
        Args:
            D1, D2: Center diffusion coefficients for each peak
            width1, width2: Widths of each peak
            ratio: Relative amplitude of first peak (0 to 1)
        """
        peak1 = self.generate_single_peak(D1, width1, ratio)
        peak2 = self.generate_single_peak(D2, width2, 1 - ratio)
        distribution = peak1 + peak2
        return distribution / (np.sum(distribution) + 1e-10)
    
    def generate_polydisperse(self,
                             D_center: float,
                             PDI: float = 1.5) -> np.ndarray:
        """
        Generate polydisperse distribution (broad, like polymers).
        
        Args:
            D_center: Center diffusion coefficient
            PDI: Polydispersity index (1 = monodisperse, >1 = polydisperse)
        """
        # Width related to PDI
        width = 0.1 * np.sqrt(PDI - 1 + 0.01)
        return self.generate_single_peak(D_center, width)
    
    def generate_random_distribution(self, 
                                    n_peaks: Optional[int] = None,
                                    seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Generate a random diffusion distribution for training.
        
        Returns:
            distribution: Normalized distribution array
            params: Dictionary with generation parameters
        """
        if seed is not None:
            np.random.seed(seed)
            
        if n_peaks is None:
            n_peaks = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        
        distribution = np.zeros(self.n_bins)
        params = {'n_peaks': n_peaks, 'peaks': []}
        
        for i in range(n_peaks):
            # Random D in range
            log_D = np.random.uniform(np.log10(self.D_min * 10), np.log10(self.D_max / 10))
            D_center = 10 ** log_D
            
            # Random width (narrower for small molecules, broader for polymers)
            width = np.random.uniform(0.05, 0.4)
            
            # Random amplitude
            amplitude = np.random.uniform(0.3, 1.0)
            
            peak = self.generate_single_peak(D_center, width, amplitude)
            distribution += peak
            
            params['peaks'].append({
                'D_center': D_center,
                'width': width,
                'amplitude': amplitude
            })
        
        # Normalize
        distribution = distribution / (np.sum(distribution) + 1e-10)
        return distribution, params


class SyntheticNMRDataGenerator:
    """
    Generate synthetic NMR diffusion decay data for neural network training.
    """
    
    def __init__(self,
                 n_gradient_steps: int = 23,
                 g_min: float = 0.01,           # T/m
                 g_max: float = 0.5,            # T/m
                 n_bins: int = 1024,
                 D_min: float = 1e-12,
                 D_max: float = 1e-8,
                 nmr_params: Optional[NMRDiffusionParameters] = None):
        """
        Initialize the data generator.
        
        Args:
            n_gradient_steps: Number of gradient strength variations
            g_min, g_max: Range of gradient strengths
            n_bins: Resolution of diffusion coefficient distribution
            D_min, D_max: Range of diffusion coefficients
            nmr_params: NMR experimental parameters
        """
        self.n_gradient_steps = n_gradient_steps
        self.g_values = np.linspace(g_min, g_max, n_gradient_steps)
        self.nmr_params = nmr_params or NMRDiffusionParameters()
        
        # Calculate b-values
        self.b_values = np.array([
            self.nmr_params.calculate_b_value(g) for g in self.g_values
        ])
        
        # Initialize distribution generator
        self.dist_generator = DiffusionDistribution(D_min, D_max, n_bins)
        
    def calculate_attenuation(self, distribution: np.ndarray) -> np.ndarray:
        """
        Calculate signal attenuation from diffusion distribution.
        
        E(b) = ∫ A(D) * exp(-b*D) dD
        
        Discretized as:
        E(b) = Σ A(Dᵢ) * exp(-b*Dᵢ) * ΔDᵢ
        """
        D_grid = self.dist_generator.D_grid
        
        # Create kernel matrix: K[i,j] = exp(-b[i] * D[j])
        kernel = np.exp(-np.outer(self.b_values, D_grid))
        
        # Calculate attenuation
        attenuation = kernel @ distribution
        
        # Normalize to initial intensity of 1
        if attenuation[0] > 0:
            attenuation = attenuation / attenuation[0]
            
        return attenuation
    
    def add_noise(self, 
                  attenuation: np.ndarray, 
                  snr: float = 100.0,
                  noise_type: str = 'gaussian') -> np.ndarray:
        """
        Add realistic noise to attenuation data.
        
        Args:
            attenuation: Clean attenuation signal
            snr: Signal-to-noise ratio
            noise_type: 'gaussian' or 'rician' (for magnitude NMR data)
        """
        noise_level = 1.0 / snr
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, attenuation.shape)
            noisy = attenuation + noise
        elif noise_type == 'rician':
            # Rician noise for magnitude data
            real_noise = np.random.normal(attenuation, noise_level)
            imag_noise = np.random.normal(0, noise_level, attenuation.shape)
            noisy = np.sqrt(real_noise**2 + imag_noise**2)
        else:
            noisy = attenuation
            
        return np.clip(noisy, 0, None)
    
    def generate_sample(self,
                       noise_snr: Optional[float] = None,
                       distribution_type: str = 'random',
                       **kwargs) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate a single training sample.
        
        Args:
            noise_snr: SNR for added noise (None = no noise)
            distribution_type: 'random', 'single', 'bimodal', 'polydisperse'
            **kwargs: Additional parameters for specific distribution types
            
        Returns:
            attenuation: Decay signal (input to neural network)
            distribution: Diffusion distribution (target output)
            params: Generation parameters
        """
        if distribution_type == 'random':
            distribution, params = self.dist_generator.generate_random_distribution(**kwargs)
        elif distribution_type == 'single':
            D_center = kwargs.get('D_center', 1e-10)
            width = kwargs.get('width', 0.1)
            distribution = self.dist_generator.generate_single_peak(D_center, width)
            params = {'type': 'single', 'D_center': D_center, 'width': width}
        elif distribution_type == 'bimodal':
            D1 = kwargs.get('D1', 5e-11)
            D2 = kwargs.get('D2', 5e-10)
            distribution = self.dist_generator.generate_bimodal(D1, D2, **kwargs)
            params = {'type': 'bimodal', 'D1': D1, 'D2': D2}
        elif distribution_type == 'polydisperse':
            D_center = kwargs.get('D_center', 1e-10)
            PDI = kwargs.get('PDI', 1.5)
            distribution = self.dist_generator.generate_polydisperse(D_center, PDI)
            params = {'type': 'polydisperse', 'D_center': D_center, 'PDI': PDI}
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
        
        # Calculate attenuation
        attenuation = self.calculate_attenuation(distribution)
        
        # Add noise if specified
        if noise_snr is not None:
            attenuation = self.add_noise(attenuation, noise_snr)
            params['noise_snr'] = noise_snr
            
        return attenuation, distribution, params
    
    def generate_dataset(self,
                        n_samples: int,
                        noise_snr_range: Tuple[float, float] = (50, 500),
                        include_clean: bool = True,
                        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete dataset for training.
        
        Args:
            n_samples: Number of samples to generate
            noise_snr_range: Range of SNR values (uniformly sampled)
            include_clean: Include 10% clean samples without noise
            seed: Random seed for reproducibility
            
        Returns:
            X: Input attenuation data (n_samples, n_gradient_steps)
            Y: Target distributions (n_samples, n_bins)
        """
        if seed is not None:
            np.random.seed(seed)
            
        X = np.zeros((n_samples, self.n_gradient_steps))
        Y = np.zeros((n_samples, self.dist_generator.n_bins))
        
        for i in range(n_samples):
            # Determine noise level
            if include_clean and np.random.random() < 0.1:
                noise_snr = None
            else:
                noise_snr = np.random.uniform(*noise_snr_range)
            
            attenuation, distribution, _ = self.generate_sample(
                noise_snr=noise_snr,
                distribution_type='random',
                seed=seed + i if seed else None
            )
            
            X[i] = attenuation
            Y[i] = distribution
            
        return X, Y


class NMRDiffusionDataset(Dataset):
    """PyTorch Dataset for NMR diffusion data."""
    
    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray,
                 transform=None):
        """
        Args:
            X: Input attenuation data
            Y: Target diffusion distributions
            transform: Optional transform to apply
        """
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


def create_dataloaders(n_train: int = 50000,
                      n_val: int = 5000,
                      n_test: int = 5000,
                      batch_size: int = 64,
                      n_gradient_steps: int = 23,
                      seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        n_train, n_val, n_test: Number of samples for each set
        batch_size: Batch size for training
        n_gradient_steps: Number of gradient variations
        seed: Random seed
        
    Returns:
        train_loader, val_loader, test_loader
    """
    generator = SyntheticNMRDataGenerator(n_gradient_steps=n_gradient_steps)
    
    # Generate datasets
    print("Generating training data...")
    X_train, Y_train = generator.generate_dataset(n_train, seed=seed)
    
    print("Generating validation data...")
    X_val, Y_val = generator.generate_dataset(n_val, seed=seed + 1000000)
    
    print("Generating test data...")
    X_test, Y_test = generator.generate_dataset(n_test, seed=seed + 2000000)
    
    # Create datasets
    train_dataset = NMRDiffusionDataset(X_train, Y_train)
    val_dataset = NMRDiffusionDataset(X_val, Y_val)
    test_dataset = NMRDiffusionDataset(X_test, Y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data generator
    print("Testing NMR Diffusion Data Generator...")
    
    generator = SyntheticNMRDataGenerator(n_gradient_steps=23)
    
    # Generate a sample
    attenuation, distribution, params = generator.generate_sample(
        noise_snr=100,
        distribution_type='bimodal',
        D1=1e-11,
        D2=1e-10
    )
    
    print(f"Attenuation shape: {attenuation.shape}")
    print(f"Distribution shape: {distribution.shape}")
    print(f"Parameters: {params}")
    
    # Generate dataset
    X, Y = generator.generate_dataset(n_samples=1000, seed=42)
    print(f"\nDataset shapes: X={X.shape}, Y={Y.shape}")
    
    print("\nData generation test completed successfully!")
