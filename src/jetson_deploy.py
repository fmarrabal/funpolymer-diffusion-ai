"""
FUNPOLYMER - NVIDIA Jetson Deployment Module
=============================================

This module provides tools for deploying the trained neural network
on NVIDIA Jetson devices (Nano, Xavier, etc.) for high-performance
computing applications.

Implements:
- TensorRT optimization
- ONNX conversion
- Jetson-specific optimizations
- REST API server for embedded deployment

Author: FUNPOLYMER Project - Universidad de Almería
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union
import json
import time


def export_to_onnx(model: nn.Module,
                   input_size: int,
                   output_path: str,
                   opset_version: int = 11) -> str:
    """
    Export PyTorch model to ONNX format for TensorRT conversion.
    
    Args:
        model: Trained PyTorch model
        input_size: Number of input features
        output_path: Path for ONNX file
        opset_version: ONNX opset version
        
    Returns:
        Path to exported ONNX file
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, input_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['attenuation'],
        output_names=['distribution'],
        dynamic_axes={
            'attenuation': {0: 'batch_size'},
            'distribution': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {output_path}")
    return output_path


def optimize_for_jetson(onnx_path: str,
                        output_path: str,
                        precision: str = 'fp16',
                        workspace_size: int = 1 << 30) -> Optional[str]:
    """
    Optimize ONNX model using TensorRT for Jetson deployment.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path for TensorRT engine
        precision: 'fp32', 'fp16', or 'int8'
        workspace_size: TensorRT workspace size in bytes
        
    Returns:
        Path to TensorRT engine or None if TensorRT not available
    """
    try:
        import tensorrt as trt
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Create builder
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"ONNX Parse Error: {parser.get_error(error)}")
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        if engine is None:
            print("Failed to build TensorRT engine")
            return None
        
        # Serialize
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to: {output_path}")
        return output_path
        
    except ImportError:
        print("TensorRT not available. Install with: pip install tensorrt")
        print("For Jetson, use JetPack SDK which includes TensorRT.")
        return None


class JetsonInference:
    """
    Inference class optimized for NVIDIA Jetson devices.
    
    Uses TensorRT for accelerated inference when available,
    falls back to PyTorch otherwise.
    """
    
    def __init__(self,
                 model_path: str,
                 use_tensorrt: bool = True,
                 precision: str = 'fp16'):
        """
        Initialize Jetson inference engine.
        
        Args:
            model_path: Path to model (ONNX, TRT engine, or PyTorch checkpoint)
            use_tensorrt: Try to use TensorRT acceleration
            precision: Precision for TensorRT ('fp32', 'fp16', 'int8')
        """
        self.model_path = Path(model_path)
        self.use_tensorrt = use_tensorrt
        self.precision = precision
        
        self.engine = None
        self.context = None
        self.pytorch_model = None
        
        self._load_model()
        
    def _load_model(self):
        """Load model based on file type and available backends."""
        suffix = self.model_path.suffix.lower()
        
        if suffix == '.engine' and self.use_tensorrt:
            self._load_tensorrt_engine()
        elif suffix == '.onnx' and self.use_tensorrt:
            # Convert ONNX to TensorRT
            trt_path = self.model_path.with_suffix('.engine')
            if optimize_for_jetson(str(self.model_path), str(trt_path), self.precision):
                self.model_path = trt_path
                self._load_tensorrt_engine()
            else:
                self._load_pytorch_model()
        elif suffix == '.pth':
            self._load_pytorch_model()
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
    
    def _load_tensorrt_engine(self):
        """Load TensorRT engine."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            with open(self.model_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()
            
            print("TensorRT engine loaded successfully")
            
        except ImportError:
            print("TensorRT/PyCUDA not available, falling back to PyTorch")
            self._load_pytorch_model()
    
    def _load_pytorch_model(self):
        """Load PyTorch model as fallback."""
        from models import create_model
        
        if self.model_path.suffix == '.pth':
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            self.pytorch_model = create_model(
                model_type='standard',
                input_size=checkpoint['model_config']['input_size'],
                output_size=checkpoint['model_config']['output_size'],
                hidden_sizes=checkpoint['model_config']['hidden_sizes']
            )
            self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            self.pytorch_model.eval()
            
            # Use CUDA if available
            if torch.cuda.is_available():
                self.pytorch_model = self.pytorch_model.cuda()
            
            print("PyTorch model loaded")
        else:
            raise ValueError("Cannot load non-PyTorch model without TensorRT")
    
    def predict(self, attenuation: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            attenuation: Input attenuation data (n_samples, n_features)
            
        Returns:
            Predicted distribution (n_samples, n_bins)
        """
        if self.engine is not None:
            return self._predict_tensorrt(attenuation)
        else:
            return self._predict_pytorch(attenuation)
    
    def _predict_tensorrt(self, attenuation: np.ndarray) -> np.ndarray:
        """TensorRT inference."""
        import pycuda.driver as cuda
        
        if attenuation.ndim == 1:
            attenuation = attenuation.reshape(1, -1)
        
        batch_size = attenuation.shape[0]
        input_size = attenuation.shape[1]
        output_size = 1024  # Fixed output size
        
        # Allocate device memory
        d_input = cuda.mem_alloc(attenuation.nbytes)
        d_output = cuda.mem_alloc(batch_size * output_size * 4)  # float32
        
        # Copy input to device
        cuda.memcpy_htod(d_input, attenuation.astype(np.float32))
        
        # Run inference
        self.context.execute_v2([int(d_input), int(d_output)])
        
        # Copy output to host
        output = np.empty((batch_size, output_size), dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        
        return output
    
    def _predict_pytorch(self, attenuation: np.ndarray) -> np.ndarray:
        """PyTorch inference."""
        if attenuation.ndim == 1:
            attenuation = attenuation.reshape(1, -1)
        
        with torch.no_grad():
            x = torch.FloatTensor(attenuation)
            if torch.cuda.is_available():
                x = x.cuda()
            output = self.pytorch_model(x)
            return output.cpu().numpy()
    
    def benchmark(self, n_iterations: int = 100, batch_size: int = 1) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            n_iterations: Number of iterations
            batch_size: Batch size for benchmarking
            
        Returns:
            Dictionary with performance metrics
        """
        # Generate random input
        input_size = 23  # Default
        if self.pytorch_model is not None:
            input_size = self.pytorch_model.input_size
        
        test_input = np.random.randn(batch_size, input_size).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            _ = self.predict(test_input)
        
        # Benchmark
        times = []
        for _ in range(n_iterations):
            start = time.time()
            _ = self.predict(test_input)
            times.append(time.time() - start)
        
        times = np.array(times)
        
        return {
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'min_latency_ms': np.min(times) * 1000,
            'max_latency_ms': np.max(times) * 1000,
            'throughput_samples_per_sec': batch_size / np.mean(times),
            'batch_size': batch_size,
            'n_iterations': n_iterations
        }


def generate_jetson_deployment_package(model_path: str,
                                       output_dir: str,
                                       include_server: bool = True):
    """
    Generate complete deployment package for Jetson.
    
    Args:
        model_path: Path to trained model checkpoint
        output_dir: Output directory for deployment files
        include_server: Include REST API server code
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and export model
    checkpoint = torch.load(model_path, map_location='cpu')
    from models import create_model
    
    model = create_model(
        model_type='standard',
        input_size=checkpoint['model_config']['input_size'],
        output_size=checkpoint['model_config']['output_size'],
        hidden_sizes=checkpoint['model_config']['hidden_sizes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export to ONNX
    onnx_path = output_dir / 'model.onnx'
    export_to_onnx(model, checkpoint['model_config']['input_size'], str(onnx_path))
    
    # Save model config
    config = {
        'input_size': checkpoint['model_config']['input_size'],
        'output_size': checkpoint['model_config']['output_size'],
        'hidden_sizes': checkpoint['model_config']['hidden_sizes'],
        'D_min': 1e-12,
        'D_max': 1e-8
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Generate server code if requested
    if include_server:
        server_code = generate_flask_server()
        with open(output_dir / 'server.py', 'w') as f:
            f.write(server_code)
    
    # Generate requirements
    requirements = """
torch>=1.9.0
numpy>=1.19.0
flask>=2.0.0
onnx>=1.10.0
# tensorrt  # Install via JetPack SDK
# pycuda    # Install via JetPack SDK
""".strip()
    
    with open(output_dir / 'requirements.txt', 'w') as f:
        f.write(requirements)
    
    # Generate setup script for Jetson
    setup_script = """#!/bin/bash
# FUNPOLYMER Jetson Setup Script

echo "Setting up FUNPOLYMER on NVIDIA Jetson..."

# Install Python dependencies
pip3 install -r requirements.txt

# Convert ONNX to TensorRT (run on Jetson)
# python3 -c "from jetson_deploy import optimize_for_jetson; optimize_for_jetson('model.onnx', 'model.engine', 'fp16')"

echo "Setup complete!"
echo "Run the server with: python3 server.py"
"""
    
    with open(output_dir / 'setup.sh', 'w') as f:
        f.write(setup_script)
    
    print(f"Deployment package created in {output_dir}")
    print("Files created:")
    for f in output_dir.iterdir():
        print(f"  - {f.name}")


def generate_flask_server() -> str:
    """Generate Flask REST API server code for Jetson deployment."""
    return '''
"""
FUNPOLYMER - REST API Server for NMR Diffusion Prediction
=========================================================

Run on NVIDIA Jetson for real-time inference.
"""

from flask import Flask, request, jsonify
import numpy as np
import json

# Import from the same directory
from jetson_deploy import JetsonInference

app = Flask(__name__)

# Load model at startup
MODEL_PATH = 'model.onnx'  # or 'model.engine' if TensorRT optimized
inference_engine = None

with open('config.json', 'r') as f:
    CONFIG = json.load(f)


@app.before_first_request
def load_model():
    global inference_engine
    inference_engine = JetsonInference(MODEL_PATH, use_tensorrt=True)
    print(f"Model loaded: {MODEL_PATH}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': inference_engine is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict diffusion distribution from attenuation data.
    
    Request JSON:
        {
            "attenuation": [0.9, 0.8, 0.7, ...]  // n_gradient_steps values
        }
    
    Response JSON:
        {
            "distribution": [...],  // 1024 values
            "D_mean": 1.23e-10,
            "D_mode": 1.15e-10,
            "success": true
        }
    """
    try:
        data = request.get_json()
        attenuation = np.array(data['attenuation'], dtype=np.float32)
        
        # Validate input
        if len(attenuation) != CONFIG['input_size']:
            return jsonify({
                'success': False,
                'error': f"Expected {CONFIG['input_size']} values, got {len(attenuation)}"
            }), 400
        
        # Run inference
        distribution = inference_engine.predict(attenuation)[0]
        
        # Calculate statistics
        D_grid = np.logspace(np.log10(CONFIG['D_min']), np.log10(CONFIG['D_max']), CONFIG['output_size'])
        D_mean = float(np.sum(D_grid * distribution))
        D_mode = float(D_grid[np.argmax(distribution)])
        
        return jsonify({
            'success': True,
            'distribution': distribution.tolist(),
            'D_mean': D_mean,
            'D_mode': D_mode
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint."""
    try:
        data = request.get_json()
        attenuations = np.array(data['attenuations'], dtype=np.float32)
        
        distributions = inference_engine.predict(attenuations)
        
        return jsonify({
            'success': True,
            'distributions': distributions.tolist()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/benchmark', methods=['GET'])
def benchmark():
    """Run benchmark and return performance metrics."""
    n_iterations = request.args.get('iterations', 100, type=int)
    batch_size = request.args.get('batch_size', 1, type=int)
    
    metrics = inference_engine.benchmark(n_iterations, batch_size)
    
    return jsonify({
        'success': True,
        'metrics': metrics
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FUNPOLYMER Jetson Deployment')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='jetson_deploy',
                       help='Output directory for deployment package')
    parser.add_argument('--export-onnx', action='store_true',
                       help='Export to ONNX only')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark')
    
    args = parser.parse_args()
    
    if args.export_onnx:
        from models import create_model
        checkpoint = torch.load(args.model, map_location='cpu')
        model = create_model(
            model_type='standard',
            input_size=checkpoint['model_config']['input_size'],
            output_size=checkpoint['model_config']['output_size'],
            hidden_sizes=checkpoint['model_config']['hidden_sizes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        export_to_onnx(model, checkpoint['model_config']['input_size'], 'model.onnx')
    elif args.benchmark:
        inference = JetsonInference(args.model)
        metrics = inference.benchmark()
        print("Benchmark Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        generate_jetson_deployment_package(args.model, args.output_dir)
