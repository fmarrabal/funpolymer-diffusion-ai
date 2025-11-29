# FUNPOLYMER - Deep Learning for NMR Diffusion Coefficient Estimation

<p align="center">
  <img src="docs/images/logo.png" alt="FUNPOLYMER Logo" width="300">
</p>

<p align="center">
  <a href="https://github.com/fmarrabal/funpolymer-diffusion-ai/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-1.9+-red.svg" alt="PyTorch">
  </a>
</p>

A complete deep learning framework for solving the inverse Laplace transform (ILT) problem in NMR diffusion experiments. This software enables accurate determination of diffusion coefficient distributions and molecular weight estimation in polymer solutions.

## 🎯 Overview

This project is part of **FUNPOLYMER** (Activity 7 & 8) developed at Universidad de Almería by the NMRMBC Research Group. It implements a neural network approach to replace or complement classical numerical methods (CONTIN, DISCRETE, MaxEnt, ITAMeD, TRAIn, dART) for processing NMR diffusion data.

### Key Features

- **Deep Neural Network Architecture**: 4 hidden layers with 1024-bin output resolution
- **Synthetic Data Generation**: Based on the Stejskal-Tanner equation
- **Multiple Distribution Types**: Single peaks, bimodal, polydisperse (polymer-like)
- **Noise Robustness**: Trained with varying SNR levels (20-500)
- **NVIDIA Jetson Support**: Optimized for edge deployment (Activity 8)
- **MATLAB/C# Integration**: Export capabilities for industrial applications

## 📊 Architecture

```
Input (23 or 46 gradient steps)
         │
    ┌────┴────┐
    │ Dense   │ (512 neurons)
    │ BN+ReLU │
    │ Dropout │
    └────┬────┘
         │
    ┌────┴────┐
    │ Dense   │ (1024 neurons)
    │ BN+ReLU │
    │ Dropout │
    └────┬────┘
         │
    ┌────┴────┐
    │ Dense   │ (1024 neurons)
    │ BN+ReLU │
    │ Dropout │
    └────┬────┘
         │
    ┌────┴────┐
    │ Dense   │ (512 neurons)
    │ BN+ReLU │
    │ Dropout │
    └────┬────┘
         │
    ┌────┴────┐
    │ Dense   │ (1024 neurons)
    │ Softmax │
    └────┬────┘
         │
Output (1024 D-bins)
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/fmarrabal/funpolymer-diffusion-ai.git
cd funpolymer-diffusion-ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Training a Model

```bash
# Train with default settings (23 gradient steps)
python src/train.py --gradient-steps 23 --epochs 100

# Train with 46 gradient steps for higher resolution
python src/train.py --gradient-steps 46 --epochs 100 --n-train 100000

# Custom training
python src/train.py \
    --gradient-steps 23 \
    --n-train 50000 \
    --n-val 5000 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --epochs 100 \
    --checkpoint-dir ./my_checkpoints
```

### Evaluation

```bash
# Run complete evaluation
python src/evaluate.py --model checkpoints/best_model.pth --output-dir results/
```

### Inference

```python
from src.inference import DiffusionPredictor

# Load trained model
predictor = DiffusionPredictor('checkpoints/best_model.pth')

# Predict from attenuation data
import numpy as np
attenuation = np.array([1.0, 0.95, 0.88, 0.79, ...])  # 23 values
results = predictor.predict(attenuation)

print(f"Mean D: {results['D_mean']:.2e} m²/s")
print(f"Mode D: {results['D_mode']:.2e} m²/s")
```

## 📁 Project Structure

```
funpolymer-diffusion-ai/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data_generator.py    # Synthetic NMR data generation
│   ├── models.py            # Neural network architectures
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Evaluation and metrics
│   ├── inference.py         # Inference and prediction
│   ├── visualization.py     # Plotting utilities
│   └── jetson_deploy.py     # NVIDIA Jetson deployment
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks
├── examples/                # Usage examples
├── docs/                    # Documentation
├── models/                  # Pre-trained models
├── checkpoints/             # Training checkpoints
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
└── README.md                # This file
```

## 📈 Performance

### Test Set Results (n=5000 samples, SNR=100)

| Metric | Value |
|--------|-------|
| MSE | 2.3×10⁻⁵ |
| MAE | 1.8×10⁻³ |
| Pearson r | 0.987 |
| D mean rel. error | 3.2% |
| Peak count accuracy | 94.5% |

### Noise Robustness

| SNR | MSE | D Error (%) |
|-----|-----|-------------|
| 20  | 8.5×10⁻⁵ | 8.7 |
| 50  | 3.1×10⁻⁵ | 4.2 |
| 100 | 2.3×10⁻⁵ | 3.2 |
| 200 | 1.9×10⁻⁵ | 2.5 |
| 500 | 1.7×10⁻⁵ | 2.1 |

## 🔧 NVIDIA Jetson Deployment (Activity 8)

The model can be deployed on NVIDIA Jetson devices for real-time inference:

```bash
# Generate deployment package
python src/jetson_deploy.py --model checkpoints/best_model.pth --output-dir jetson_deploy/

# On Jetson device:
cd jetson_deploy
./setup.sh
python server.py  # Start REST API server
```

### REST API Usage

```bash
# Health check
curl http://jetson-ip:5000/health

# Predict
curl -X POST http://jetson-ip:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"attenuation": [1.0, 0.95, 0.88, ...]}'

# Benchmark
curl http://jetson-ip:5000/benchmark?iterations=100
```

## 📚 Theory

### Stejskal-Tanner Equation

The signal attenuation in NMR diffusion experiments follows:

```
E(g) = ∫ A(D) × exp(-D × γ² × δ² × g² × Δ') dD
```

Where:
- `E(g)`: Attenuated signal at gradient strength `g`
- `A(D)`: Diffusion coefficient distribution
- `γ`: Gyromagnetic ratio
- `δ`: Gradient pulse duration
- `Δ'`: Effective diffusion time

### Neural Network Approach

Instead of iterative numerical methods (CONTIN, etc.), we train a neural network to directly map the attenuation signal to the diffusion distribution:

```
f: ℝⁿ → ℝ¹⁰²⁴
E(g₁, ..., gₙ) → A(D₁, ..., D₁₀₂₄)
```

This provides:
- Faster inference (ms vs seconds)
- Better noise handling
- Consistent results without parameter tuning

## 🔬 MATLAB Integration

```matlab
% Load Python module
py.importlib.import_module('src.inference');

% Create predictor
predictor = py.src.inference.DiffusionPredictor('checkpoints/best_model.pth');

% Predict
attenuation = [1.0, 0.95, 0.88, 0.79, ...];  % Your data
results = predictor.predict(py.numpy.array(attenuation));

% Extract results
D_mean = double(results{'D_mean'});
distribution = double(results{'distribution'});
```

Or export to MATLAB format:

```python
from src.inference import DiffusionPredictor, MATLABInterface

predictor = DiffusionPredictor('checkpoints/best_model.pth')
results = predictor.predict(attenuation)
MATLABInterface.save_for_matlab(results, 'results.mat')
```

## 📖 Citation

If you use this software in your research, please cite:

```bibtex
@software{funpolymer_diffusion_ai,
  author = {Arrabal-Campos, Francisco M. and Fernández, Ignacio},
  title = {FUNPOLYMER: Deep Learning for NMR Diffusion Coefficient Estimation},
  year = {2025},
  institution = {Universidad de Almería},
  url = {https://github.com/fmarrabal/funpolymer-diffusion-ai}
}
```

### Related Publications

Please also cite the following related works:

```bibtex
@article{Arrabal-Campos2025,
  author = {Arrabal-Campos, Francisco Manuel and González-Lázaro, Marta and Pérez, Juana M. and Martínez Lao, Juan A. and Fernández, Ignacio},
  doi = {10.1016/j.eurpolymj.2024.113710},
  issn = {00143057},
  journal = {European Polymer Journal},
  month = {feb},
  pages = {113710},
  title = {{Concentration-independent molecular weight determination of polymers via diffusion NMR: A universal approach across solvents}},
  url = {https://linkinghub.elsevier.com/retrieve/pii/S0014305724009716},
  volume = {226},
  year = {2025}
}

@article{Gonzalez-Lazaro2025,
  author = {González-Lázaro, Marta and Viciana, Eduardo and Valdivieso, Víctor and Fernández, Ignacio and Arrabal-Campos, Francisco Manuel},
  doi = {10.3390/math13132166},
  issn = {2227-7390},
  journal = {Mathematics},
  month = {jul},
  number = {13},
  pages = {2166},
  title = {{Regularized Kaczmarz Solvers for Robust Inverse Laplace Transforms}},
  url = {https://www.mdpi.com/2227-7390/13/13/2166},
  volume = {13},
  year = {2025}
}
```

## 👥 Authors

**FUNPOLYMER Project Team - Universidad de Almería**

- **IP**: Ignacio Fernández de las Nieves (Catedrático)
- **Co-IP**: Francisco M. Arrabal-Campos (Profesor Ayudante Doctor)
- Research Group: [NMRMBC](https://www.nmrmbc.com)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This research has been funded by the State Research Agency of the Spanish Ministry of Science and Innovation (PDC2021-121248-I00, PLEC2021-007774, PID2021-126445OB-I00 and CPP2022-009967) and by the Gobierno de España MCIN/AEI/10.13039/501100011033 and Unión Europea "Next Generation EU"/PRTR.

### Institutions and Partners

- Universidad de Almería
- CIAIMBITAL Research Center

### Industrial Partners

- Omar Coatings
- Sustonable
- Bufä
- Gazechim Composites
- Cosentino R&D
- Dal-Tile

## 📧 Contact

- **Email**: fmarrabal@ual.es
- **Website**: [www.nmrmbc.com](https://www.nmrmbc.com)
- **Issues**: [GitHub Issues](https://github.com/fmarrabal/funpolymer-diffusion-ai/issues)