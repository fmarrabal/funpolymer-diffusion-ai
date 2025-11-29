"""
FUNPOLYMER - Model Evaluation and Testing
==========================================

This module provides comprehensive evaluation tools for the trained
diffusion coefficient neural network, including:
- Performance metrics (MSE, MAE, correlation)
- Distribution comparison
- Peak detection and accuracy
- Visualization of results
- Comparison with classical ILT methods

Author: FUNPOLYMER Project - Universidad de Almería
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import json
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr

from data_generator import SyntheticNMRDataGenerator, DiffusionDistribution
from models import DiffusionNet, create_model


class DiffusionEvaluator:
    """
    Comprehensive evaluator for diffusion coefficient predictions.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'auto',
                 D_min: float = 1e-12,
                 D_max: float = 1e-8,
                 n_bins: int = 1024):
        """
        Initialize evaluator.
        
        Args:
            model: Trained neural network model
            device: Computation device
            D_min, D_max: Diffusion coefficient range
            n_bins: Number of output bins
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.model.eval()
        
        # Create diffusion grid
        self.D_grid = np.logspace(np.log10(D_min), np.log10(D_max), n_bins)
        self.log_D_grid = np.log10(self.D_grid)
        
    def predict(self, attenuation: np.ndarray) -> np.ndarray:
        """
        Predict diffusion distribution from attenuation data.
        
        Args:
            attenuation: Input attenuation signal (n_samples, n_gradient_steps)
            
        Returns:
            Predicted distributions (n_samples, n_bins)
        """
        if attenuation.ndim == 1:
            attenuation = attenuation.reshape(1, -1)
            
        with torch.no_grad():
            x = torch.FloatTensor(attenuation).to(self.device)
            pred = self.model(x)
            return pred.cpu().numpy()
    
    def calculate_metrics(self, 
                         pred: np.ndarray, 
                         target: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            pred: Predicted distributions
            target: Ground truth distributions
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = np.mean((pred - target) ** 2)
        metrics['mae'] = np.mean(np.abs(pred - target))
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Correlation metrics
        flat_pred = pred.flatten()
        flat_target = target.flatten()
        metrics['pearson_r'], metrics['pearson_p'] = pearsonr(flat_pred, flat_target)
        metrics['spearman_r'], metrics['spearman_p'] = spearmanr(flat_pred, flat_target)
        
        # Distribution-specific metrics
        metrics['kl_divergence'] = self._kl_divergence(pred, target)
        metrics['js_divergence'] = self._js_divergence(pred, target)
        
        # Peak detection metrics
        peak_metrics = self._evaluate_peaks(pred, target)
        metrics.update(peak_metrics)
        
        return metrics
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """Calculate KL divergence."""
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.mean(np.sum(p * np.log(p / q), axis=-1))
    
    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence."""
        m = 0.5 * (p + q)
        return 0.5 * (self._kl_divergence(p, m) + self._kl_divergence(q, m))
    
    def _evaluate_peaks(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        Evaluate peak detection accuracy.
        
        Returns:
            Dictionary with peak-related metrics
        """
        peak_errors = []
        detected_peaks_match = []
        
        for i in range(len(pred)):
            # Find peaks in target
            target_peaks, _ = find_peaks(target[i], height=0.01, distance=20)
            
            # Find peaks in prediction
            pred_peaks, _ = find_peaks(pred[i], height=0.01, distance=20)
            
            # Check if number of peaks matches
            detected_peaks_match.append(len(pred_peaks) == len(target_peaks))
            
            # Calculate peak position error if peaks found
            if len(target_peaks) > 0 and len(pred_peaks) > 0:
                # For each target peak, find nearest predicted peak
                for tp in target_peaks:
                    if len(pred_peaks) > 0:
                        nearest = pred_peaks[np.argmin(np.abs(pred_peaks - tp))]
                        # Convert to log-D space
                        error = np.abs(self.log_D_grid[nearest] - self.log_D_grid[tp])
                        peak_errors.append(error)
        
        metrics = {
            'peak_count_accuracy': np.mean(detected_peaks_match),
            'mean_peak_position_error': np.mean(peak_errors) if peak_errors else 0.0,
            'std_peak_position_error': np.std(peak_errors) if peak_errors else 0.0
        }
        
        return metrics
    
    def get_diffusion_coefficient(self, distribution: np.ndarray) -> Dict[str, float]:
        """
        Extract diffusion coefficient statistics from distribution.
        
        Args:
            distribution: Predicted or target distribution
            
        Returns:
            Dictionary with D statistics
        """
        if distribution.ndim == 1:
            distribution = distribution.reshape(1, -1)
            
        results = []
        
        for dist in distribution:
            # Normalize
            dist = dist / (np.sum(dist) + 1e-10)
            
            # Calculate weighted mean (peak D)
            D_mean = np.sum(self.D_grid * dist)
            
            # Find maximum (mode)
            D_mode = self.D_grid[np.argmax(dist)]
            
            # Calculate weighted standard deviation
            D_var = np.sum(dist * (self.D_grid - D_mean) ** 2)
            D_std = np.sqrt(D_var)
            
            # Calculate percentiles
            cumsum = np.cumsum(dist)
            D_25 = self.D_grid[np.searchsorted(cumsum, 0.25)]
            D_50 = self.D_grid[np.searchsorted(cumsum, 0.50)]
            D_75 = self.D_grid[np.searchsorted(cumsum, 0.75)]
            
            results.append({
                'D_mean': D_mean,
                'D_mode': D_mode,
                'D_std': D_std,
                'D_25': D_25,
                'D_50': D_50,
                'D_75': D_75
            })
            
        return results if len(results) > 1 else results[0]
    
    def compare_D_estimation(self, 
                            pred: np.ndarray, 
                            target: np.ndarray) -> Dict[str, float]:
        """
        Compare estimated diffusion coefficients.
        
        Returns:
            Metrics comparing estimated D values
        """
        pred_stats = self.get_diffusion_coefficient(pred)
        target_stats = self.get_diffusion_coefficient(target)
        
        if not isinstance(pred_stats, list):
            pred_stats = [pred_stats]
            target_stats = [target_stats]
        
        errors = {
            'D_mean_error': [],
            'D_mode_error': [],
            'D_mean_rel_error': [],
            'D_mode_rel_error': []
        }
        
        for ps, ts in zip(pred_stats, target_stats):
            errors['D_mean_error'].append(np.abs(ps['D_mean'] - ts['D_mean']))
            errors['D_mode_error'].append(np.abs(ps['D_mode'] - ts['D_mode']))
            
            if ts['D_mean'] > 0:
                errors['D_mean_rel_error'].append(
                    np.abs(ps['D_mean'] - ts['D_mean']) / ts['D_mean'] * 100
                )
            if ts['D_mode'] > 0:
                errors['D_mode_rel_error'].append(
                    np.abs(ps['D_mode'] - ts['D_mode']) / ts['D_mode'] * 100
                )
        
        return {
            'mean_D_mean_error': np.mean(errors['D_mean_error']),
            'mean_D_mode_error': np.mean(errors['D_mode_error']),
            'mean_D_mean_rel_error_%': np.mean(errors['D_mean_rel_error']),
            'mean_D_mode_rel_error_%': np.mean(errors['D_mode_rel_error'])
        }


def evaluate_on_test_set(model_path: str,
                        n_test: int = 5000,
                        n_gradient_steps: int = 23,
                        seed: int = 12345) -> Dict[str, float]:
    """
    Evaluate trained model on test set.
    
    Args:
        model_path: Path to model checkpoint
        n_test: Number of test samples
        n_gradient_steps: Number of gradient steps
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = create_model(
        model_type='standard',
        input_size=checkpoint['model_config']['input_size'],
        output_size=checkpoint['model_config']['output_size'],
        hidden_sizes=checkpoint['model_config']['hidden_sizes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = DiffusionEvaluator(model)
    
    # Generate test data
    print(f"Generating {n_test} test samples...")
    generator = SyntheticNMRDataGenerator(n_gradient_steps=n_gradient_steps)
    X_test, Y_test = generator.generate_dataset(n_test, seed=seed)
    
    # Make predictions
    print("Making predictions...")
    predictions = evaluator.predict(X_test)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = evaluator.calculate_metrics(predictions, Y_test)
    
    # Add D estimation comparison
    D_metrics = evaluator.compare_D_estimation(predictions, Y_test)
    metrics.update(D_metrics)
    
    return metrics


def evaluate_noise_robustness(model_path: str,
                             snr_levels: List[float] = [20, 50, 100, 200, 500],
                             n_samples: int = 1000,
                             n_gradient_steps: int = 23) -> Dict[str, List[float]]:
    """
    Evaluate model robustness to different noise levels.
    
    Args:
        model_path: Path to model checkpoint
        snr_levels: List of SNR values to test
        n_samples: Number of samples per SNR level
        n_gradient_steps: Number of gradient steps
        
    Returns:
        Dictionary mapping SNR to metrics
    """
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = create_model(
        model_type='standard',
        input_size=checkpoint['model_config']['input_size'],
        output_size=checkpoint['model_config']['output_size'],
        hidden_sizes=checkpoint['model_config']['hidden_sizes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator = DiffusionEvaluator(model)
    generator = SyntheticNMRDataGenerator(n_gradient_steps=n_gradient_steps)
    
    results = {
        'snr': snr_levels,
        'mse': [],
        'mae': [],
        'pearson_r': [],
        'D_mean_rel_error_%': []
    }
    
    for snr in snr_levels:
        print(f"Evaluating at SNR = {snr}...")
        
        # Generate data with specific noise level
        X = np.zeros((n_samples, n_gradient_steps))
        Y = np.zeros((n_samples, 1024))
        
        for i in range(n_samples):
            attenuation, distribution, _ = generator.generate_sample(
                noise_snr=snr,
                distribution_type='random'
            )
            X[i] = attenuation
            Y[i] = distribution
        
        # Predict and evaluate
        predictions = evaluator.predict(X)
        metrics = evaluator.calculate_metrics(predictions, Y)
        D_metrics = evaluator.compare_D_estimation(predictions, Y)
        
        results['mse'].append(metrics['mse'])
        results['mae'].append(metrics['mae'])
        results['pearson_r'].append(metrics['pearson_r'])
        results['D_mean_rel_error_%'].append(D_metrics['mean_D_mean_rel_error_%'])
    
    return results


def evaluate_distribution_types(model_path: str,
                               n_samples: int = 500,
                               n_gradient_steps: int = 23) -> Dict[str, Dict]:
    """
    Evaluate model on different distribution types.
    
    Args:
        model_path: Path to model checkpoint
        n_samples: Number of samples per type
        n_gradient_steps: Number of gradient steps
        
    Returns:
        Metrics for each distribution type
    """
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = create_model(
        model_type='standard',
        input_size=checkpoint['model_config']['input_size'],
        output_size=checkpoint['model_config']['output_size'],
        hidden_sizes=checkpoint['model_config']['hidden_sizes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator = DiffusionEvaluator(model)
    generator = SyntheticNMRDataGenerator(n_gradient_steps=n_gradient_steps)
    
    distribution_types = {
        'single_narrow': {'distribution_type': 'single', 'width': 0.05},
        'single_broad': {'distribution_type': 'single', 'width': 0.2},
        'bimodal_close': {'distribution_type': 'bimodal', 'D1': 5e-11, 'D2': 1e-10},
        'bimodal_far': {'distribution_type': 'bimodal', 'D1': 1e-11, 'D2': 1e-9},
        'polydisperse_low': {'distribution_type': 'polydisperse', 'PDI': 1.2},
        'polydisperse_high': {'distribution_type': 'polydisperse', 'PDI': 2.0}
    }
    
    results = {}
    
    for name, params in distribution_types.items():
        print(f"Evaluating {name}...")
        
        X = np.zeros((n_samples, n_gradient_steps))
        Y = np.zeros((n_samples, 1024))
        
        for i in range(n_samples):
            # Vary D_center randomly
            D_center = 10 ** np.random.uniform(-11, -9)
            params_i = params.copy()
            if 'D_center' not in params_i:
                params_i['D_center'] = D_center
            
            attenuation, distribution, _ = generator.generate_sample(
                noise_snr=100,
                **params_i
            )
            X[i] = attenuation
            Y[i] = distribution
        
        predictions = evaluator.predict(X)
        metrics = evaluator.calculate_metrics(predictions, Y)
        D_metrics = evaluator.compare_D_estimation(predictions, Y)
        
        results[name] = {**metrics, **D_metrics}
    
    return results


def run_complete_evaluation(model_path: str, output_dir: str = 'evaluation_results'):
    """
    Run complete evaluation pipeline.
    
    Args:
        model_path: Path to trained model
        output_dir: Directory for saving results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FUNPOLYMER - Complete Model Evaluation")
    print("=" * 70)
    
    # 1. Test set evaluation
    print("\n1. Test Set Evaluation")
    print("-" * 40)
    test_metrics = evaluate_on_test_set(model_path)
    
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # 2. Noise robustness
    print("\n2. Noise Robustness Evaluation")
    print("-" * 40)
    noise_results = evaluate_noise_robustness(model_path)
    
    for snr, mse, mae in zip(noise_results['snr'], 
                             noise_results['mse'], 
                             noise_results['mae']):
        print(f"  SNR={snr:3d}: MSE={mse:.6f}, MAE={mae:.6f}")
    
    with open(output_dir / 'noise_robustness.json', 'w') as f:
        json.dump(noise_results, f, indent=2)
    
    # 3. Distribution type evaluation
    print("\n3. Distribution Type Evaluation")
    print("-" * 40)
    dist_results = evaluate_distribution_types(model_path)
    
    for dist_type, metrics in dist_results.items():
        print(f"  {dist_type}:")
        print(f"    MSE: {metrics['mse']:.6f}, "
              f"D_error: {metrics['mean_D_mean_rel_error_%']:.2f}%")
    
    with open(output_dir / 'distribution_types.json', 'w') as f:
        json.dump(dist_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Evaluation complete. Results saved to {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate FUNPOLYMER Diffusion Network')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    run_complete_evaluation(args.model, args.output_dir)
