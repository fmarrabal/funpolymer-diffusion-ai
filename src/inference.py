"""
FUNPOLYMER - Inference Module
==============================

This module provides easy-to-use inference functionality for predicting
diffusion coefficient distributions from experimental NMR data.

Includes:
- Model loading
- Real-time prediction
- ONNX export for deployment
- Integration helpers for MATLAB/C#

Author: FUNPOLYMER Project - Universidad de Almería
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import json

from models import DiffusionNet, create_model


class DiffusionPredictor:
    """
    High-level interface for NMR diffusion coefficient prediction.
    
    This class provides an easy-to-use API for loading trained models
    and making predictions on experimental or simulated data.
    """
    
    def __init__(self,
                 model_path: str,
                 device: str = 'auto',
                 D_min: float = 1e-12,
                 D_max: float = 1e-8):
        """
        Initialize predictor with a trained model.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
            device: 'cuda', 'cpu', or 'auto'
            D_min, D_max: Diffusion coefficient range
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on {self.device}...")
        
        # Load checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        config = self.checkpoint['model_config']
        self.model = create_model(
            model_type='standard',
            input_size=config['input_size'],
            output_size=config['output_size'],
            hidden_sizes=config['hidden_sizes']
        )
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Store configuration
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        
        # Create diffusion coefficient grid
        self.D_grid = np.logspace(np.log10(D_min), np.log10(D_max), config['output_size'])
        self.log_D_grid = np.log10(self.D_grid)
        
        print(f"Model loaded successfully.")
        print(f"  Input size: {self.input_size} gradient steps")
        print(f"  Output size: {self.output_size} bins")
        print(f"  D range: {D_min:.2e} to {D_max:.2e} m²/s")
    
    def predict(self, 
                attenuation: Union[np.ndarray, List[float], torch.Tensor]
                ) -> Dict[str, np.ndarray]:
        """
        Predict diffusion distribution from attenuation data.
        
        Args:
            attenuation: Input data with shape (n_gradient_steps,) or 
                        (n_samples, n_gradient_steps)
        
        Returns:
            Dictionary containing:
            - 'distribution': Predicted distribution(s)
            - 'D_grid': Diffusion coefficient grid
            - 'D_mean': Mean diffusion coefficient(s)
            - 'D_mode': Mode (peak) diffusion coefficient(s)
        """
        # Convert input
        if isinstance(attenuation, list):
            attenuation = np.array(attenuation)
        if isinstance(attenuation, np.ndarray):
            attenuation = torch.FloatTensor(attenuation)
        
        # Ensure batch dimension
        if attenuation.dim() == 1:
            attenuation = attenuation.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Validate input size
        if attenuation.shape[-1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} gradient steps, "
                           f"got {attenuation.shape[-1]}")
        
        # Move to device and predict
        attenuation = attenuation.to(self.device)
        
        with torch.no_grad():
            distribution = self.model(attenuation).cpu().numpy()
        
        # Calculate statistics
        D_mean = self._calculate_mean_D(distribution)
        D_mode = self._calculate_mode_D(distribution)
        
        # Remove batch dimension if single sample
        if single_sample:
            distribution = distribution[0]
            D_mean = D_mean[0]
            D_mode = D_mode[0]
        
        return {
            'distribution': distribution,
            'D_grid': self.D_grid,
            'D_mean': D_mean,
            'D_mode': D_mode
        }
    
    def _calculate_mean_D(self, distribution: np.ndarray) -> np.ndarray:
        """Calculate weighted mean diffusion coefficient."""
        return np.sum(distribution * self.D_grid, axis=-1)
    
    def _calculate_mode_D(self, distribution: np.ndarray) -> np.ndarray:
        """Calculate mode (peak) diffusion coefficient."""
        peak_indices = np.argmax(distribution, axis=-1)
        return self.D_grid[peak_indices]
    
    def predict_with_uncertainty(self,
                                 attenuation: Union[np.ndarray, List[float]],
                                 n_samples: int = 20) -> Dict[str, np.ndarray]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            attenuation: Input attenuation data
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with mean, std, and confidence intervals
        """
        # Convert input
        if isinstance(attenuation, list):
            attenuation = np.array(attenuation)
        attenuation = torch.FloatTensor(attenuation)
        
        if attenuation.dim() == 1:
            attenuation = attenuation.unsqueeze(0)
        
        attenuation = attenuation.to(self.device)
        
        # Enable dropout for MC sampling
        self.model.train()
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(attenuation).cpu().numpy()
                predictions.append(pred)
        
        self.model.eval()
        
        predictions = np.array(predictions)
        
        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'percentile_5': np.percentile(predictions, 5, axis=0),
            'percentile_95': np.percentile(predictions, 95, axis=0),
            'D_grid': self.D_grid
        }
    
    def reconstruct_attenuation(self, 
                                distribution: np.ndarray,
                                b_values: np.ndarray) -> np.ndarray:
        """
        Reconstruct attenuation signal from distribution (forward model).
        
        Args:
            distribution: Predicted diffusion distribution
            b_values: b-values for reconstruction
            
        Returns:
            Reconstructed attenuation signal
        """
        kernel = np.exp(-np.outer(b_values, self.D_grid))
        return kernel @ distribution
    
    def export_onnx(self, output_path: str):
        """
        Export model to ONNX format for deployment.
        
        Args:
            output_path: Path for ONNX file
        """
        dummy_input = torch.randn(1, self.input_size).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=['attenuation'],
            output_names=['distribution'],
            dynamic_axes={
                'attenuation': {0: 'batch_size'},
                'distribution': {0: 'batch_size'}
            },
            opset_version=11
        )
        print(f"Model exported to {output_path}")
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_sizes': self.checkpoint['model_config']['hidden_sizes'],
            'D_range': (self.D_grid[0], self.D_grid[-1]),
            'device': str(self.device),
            'checkpoint_epoch': self.checkpoint.get('epoch', 'unknown'),
            'checkpoint_val_loss': self.checkpoint.get('val_loss', 'unknown')
        }


def predict_from_file(model_path: str,
                      data_path: str,
                      output_path: Optional[str] = None) -> Dict:
    """
    Convenience function to predict from data file.
    
    Args:
        model_path: Path to model checkpoint
        data_path: Path to input data (numpy .npy or text file)
        output_path: Optional path to save results
        
    Returns:
        Prediction results
    """
    # Load model
    predictor = DiffusionPredictor(model_path)
    
    # Load data
    if data_path.endswith('.npy'):
        attenuation = np.load(data_path)
    else:
        attenuation = np.loadtxt(data_path)
    
    # Predict
    results = predictor.predict(attenuation)
    
    # Save if requested
    if output_path:
        np.savez(output_path,
                 distribution=results['distribution'],
                 D_grid=results['D_grid'],
                 D_mean=results['D_mean'],
                 D_mode=results['D_mode'])
        print(f"Results saved to {output_path}")
    
    return results


class MATLABInterface:
    """
    Interface for MATLAB integration.
    
    Provides methods for easy data exchange with MATLAB.
    """
    
    @staticmethod
    def save_for_matlab(results: Dict, output_path: str):
        """Save results in MATLAB-compatible format."""
        try:
            from scipy.io import savemat
            savemat(output_path, results)
            print(f"Results saved for MATLAB: {output_path}")
        except ImportError:
            print("scipy required for MATLAB export. Install with: pip install scipy")
    
    @staticmethod
    def load_from_matlab(input_path: str) -> np.ndarray:
        """Load attenuation data from MATLAB file."""
        try:
            from scipy.io import loadmat
            data = loadmat(input_path)
            # Assume attenuation is stored in 'attenuation' variable
            return data.get('attenuation', data.get('data', None))
        except ImportError:
            print("scipy required for MATLAB import. Install with: pip install scipy")
            return None


def generate_c_header():
    """
    Generate C header file for integration with Visual C#.
    
    Returns:
        String containing C header content
    """
    header = '''
/*
 * FUNPOLYMER - NMR Diffusion Coefficient Prediction
 * C/C++ Interface Header
 * 
 * For integration with Visual C# via P/Invoke or C++/CLI
 */

#ifndef FUNPOLYMER_DIFFUSION_H
#define FUNPOLYMER_DIFFUSION_H

#ifdef __cplusplus
extern "C" {
#endif

// Configuration structure
typedef struct {
    int input_size;      // Number of gradient steps (23 or 46)
    int output_size;     // Number of output bins (1024)
    double D_min;        // Minimum diffusion coefficient
    double D_max;        // Maximum diffusion coefficient
} DiffusionConfig;

// Result structure
typedef struct {
    double* distribution;  // Output distribution array
    double* D_grid;        // Diffusion coefficient grid
    double D_mean;         // Mean diffusion coefficient
    double D_mode;         // Mode diffusion coefficient
    int status;            // 0 = success, other = error code
} DiffusionResult;

// Function declarations
int initialize_model(const char* model_path, DiffusionConfig* config);
int predict_distribution(const double* attenuation, int n_samples, DiffusionResult* result);
void cleanup_model();
void free_result(DiffusionResult* result);

#ifdef __cplusplus
}
#endif

#endif // FUNPOLYMER_DIFFUSION_H
'''
    return header


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FUNPOLYMER Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str,
                       help='Path to input data file')
    parser.add_argument('--output', type=str,
                       help='Path to save results')
    parser.add_argument('--export-onnx', type=str,
                       help='Export model to ONNX format')
    parser.add_argument('--info', action='store_true',
                       help='Print model information')
    
    args = parser.parse_args()
    
    predictor = DiffusionPredictor(args.model)
    
    if args.info:
        info = predictor.get_model_info()
        print("\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    if args.export_onnx:
        predictor.export_onnx(args.export_onnx)
    
    if args.data:
        results = predict_from_file(args.model, args.data, args.output)
        print(f"\nPrediction Results:")
        print(f"  D_mean: {results['D_mean']:.4e} m²/s")
        print(f"  D_mode: {results['D_mode']:.4e} m²/s")
