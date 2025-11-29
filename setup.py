#!/usr/bin/env python
"""
FUNPOLYMER - Deep Learning for NMR Diffusion
Setup configuration for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name="funpolymer-diffusion",
    version="1.0.0",
    author="FUNPOLYMER Project - Universidad de Almería",
    author_email="nmrmbc@ual.es",
    description="Deep Learning for NMR Diffusion Coefficient Estimation in Polymers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nmrmbc/funpolymer-diffusion-ai",
    project_urls={
        "Bug Tracker": "https://github.com/nmrmbc/funpolymer-diffusion-ai/issues",
        "Documentation": "https://github.com/nmrmbc/funpolymer-diffusion-ai#readme",
        "Research Group": "https://www.nmrmbc.com",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "onnx": [
            "onnx>=1.10.0",
        ],
        "server": [
            "flask>=2.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "funpolymer-train=src.train:main",
            "funpolymer-evaluate=src.evaluate:run_complete_evaluation",
            "funpolymer-predict=src.inference:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "NMR",
        "diffusion",
        "polymer",
        "deep learning",
        "neural network",
        "inverse Laplace transform",
        "molecular weight",
        "DOSY",
    ],
)
