"""
FUNPOLYMER - Deep Learning for NMR Diffusion Coefficient Estimation
====================================================================

A complete software package for determining molecular weight distribution
in polymers using NMR diffusion experiments and deep neural networks.

Developed as part of the FUNPOLYMER project (Activity 7 & 8)
Universidad de Almería - NMRMBC Research Group

Author: FUNPOLYMER Project Team
License: MIT
"""

__version__ = '1.0.0'
__author__ = 'FUNPOLYMER Project - Universidad de Almería'

from .data_generator import (
    SyntheticNMRDataGenerator,
    DiffusionDistribution,
    NMRDiffusionParameters,
    NMRDiffusionDataset,
    create_dataloaders
)

from .models import (
    DiffusionNet,
    DiffusionNetLarge,
    DiffusionResNet,
    DiffusionLoss,
    create_model,
    count_parameters
)

from .train import (
    Trainer,
    TrainingHistory,
    EarlyStopping,
    train_model
)

from .evaluate import (
    DiffusionEvaluator,
    evaluate_on_test_set,
    evaluate_noise_robustness,
    evaluate_distribution_types,
    run_complete_evaluation
)

from .inference import (
    DiffusionPredictor,
    predict_from_file,
    MATLABInterface
)

__all__ = [
    # Data generation
    'SyntheticNMRDataGenerator',
    'DiffusionDistribution',
    'NMRDiffusionParameters',
    'NMRDiffusionDataset',
    'create_dataloaders',
    
    # Models
    'DiffusionNet',
    'DiffusionNetLarge',
    'DiffusionResNet',
    'DiffusionLoss',
    'create_model',
    'count_parameters',
    
    # Training
    'Trainer',
    'TrainingHistory',
    'EarlyStopping',
    'train_model',
    
    # Evaluation
    'DiffusionEvaluator',
    'evaluate_on_test_set',
    'evaluate_noise_robustness',
    'evaluate_distribution_types',
    'run_complete_evaluation',
    
    # Inference
    'DiffusionPredictor',
    'predict_from_file',
    'MATLABInterface'
]
