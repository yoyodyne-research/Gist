"""Latent layer training modules for Kitsune.

This package provides tools for training the PCA + GMM latent layer:
- pca: IncrementalPCA training with Varimax rotation
- gmm: Gaussian Mixture Model training
- export: Export trained models to JSON for C++ runtime
- collect_features: Utilities for feature data collection from Bela
"""

from .pca import train_pca, varimax_rotation
from .gmm import train_gmm
from .export import export_model

__all__ = ['train_pca', 'varimax_rotation', 'train_gmm', 'export_model']
