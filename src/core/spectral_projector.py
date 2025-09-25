#!/usr/bin/env python3
"""
Spectral Projector - ΨQRH Core Component

Implements spectral projection operations for transformer reformulation,
providing efficient dimensionality reduction and feature extraction
capabilities for large language models.

Classification: ΨQRH-Core-v1.0
"""

import numpy as np
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
from scipy.linalg import svd


class SpectralProjector:
    """
    Spectral projector for transformer attention reformulation

    This component implements spectral decomposition and projection
    techniques to optimize attention mechanisms in large language models.
    """

    def __init__(self,
                 projection_dim: int = 512,
                 spectral_threshold: float = 0.1,
                 use_torch: bool = True):
        """
        Initialize spectral projector

        Args:
            projection_dim: Target dimensionality for spectral projection
            spectral_threshold: Threshold for singular value retention
            use_torch: Whether to use PyTorch or NumPy/SciPy operations
        """
        self.projection_dim = projection_dim
        self.spectral_threshold = spectral_threshold
        self.use_torch = use_torch

        # Projection matrices
        self.projection_matrix = None
        self.inverse_projection = None

        # Spectral statistics
        self.singular_values = None
        self.explained_variance = None

    def fit(self, data_matrix: Union[np.ndarray, torch.Tensor]) -> 'SpectralProjector':
        """
        Fit spectral projector to data matrix

        Args:
            data_matrix: Input data matrix (n_samples x n_features)

        Returns:
            self: Fitted spectral projector
        """
        if self.use_torch and isinstance(data_matrix, torch.Tensor):
            return self._fit_torch(data_matrix)
        else:
            return self._fit_numpy(data_matrix)

    def _fit_torch(self, data_matrix: torch.Tensor) -> 'SpectralProjector':
        """Fit using PyTorch SVD"""
        # Center the data
        data_centered = data_matrix - data_matrix.mean(dim=0)

        # Compute SVD
        U, S, Vt = torch.linalg.svd(data_centered, full_matrices=False)

        # Filter singular values
        significant_sv = S > self.spectral_threshold
        S_filtered = S[significant_sv]
        Vt_filtered = Vt[significant_sv, :]

        # Store spectral information
        self.singular_values = S_filtered.cpu().numpy()
        self.explained_variance = (S_filtered ** 2 / (data_matrix.shape[0] - 1)).cpu().numpy()

        # Create projection matrix
        if len(S_filtered) > self.projection_dim:
            self.projection_matrix = Vt_filtered[:self.projection_dim, :].T
        else:
            self.projection_matrix = Vt_filtered.T

        # Create pseudo-inverse for reconstruction
        if len(S_filtered) > 0:
            S_inv = 1.0 / S_filtered
            self.inverse_projection = Vt_filtered.T @ torch.diag(S_inv) @ U[:, :len(S_filtered)].T

        return self

    def _fit_numpy(self, data_matrix: np.ndarray) -> 'SpectralProjector':
        """Fit using NumPy/SciPy SVD"""
        # Center the data
        data_centered = data_matrix - data_matrix.mean(axis=0)

        # Compute SVD
        U, S, Vt = svd(data_centered, full_matrices=False)

        # Filter singular values
        significant_sv = S > self.spectral_threshold
        S_filtered = S[significant_sv]
        Vt_filtered = Vt[significant_sv, :]

        # Store spectral information
        self.singular_values = S_filtered
        self.explained_variance = (S_filtered ** 2 / (data_matrix.shape[0] - 1))

        # Create projection matrix
        if len(S_filtered) > self.projection_dim:
            self.projection_matrix = Vt_filtered[:self.projection_dim, :].T
        else:
            self.projection_matrix = Vt_filtered.T

        # Create pseudo-inverse for reconstruction
        if len(S_filtered) > 0:
            S_inv = 1.0 / S_filtered
            self.inverse_projection = Vt_filtered.T @ np.diag(S_inv) @ U[:, :len(S_filtered)].T

        return self

    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Project data to spectral subspace

        Args:
            data: Input data to project

        Returns:
            Projected data in spectral subspace
        """
        if self.projection_matrix is None:
            raise ValueError("Spectral projector must be fitted before transformation")

        if self.use_torch and isinstance(data, torch.Tensor):
            return data @ self.projection_matrix
        else:
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            return data @ self.projection_matrix.cpu().numpy()

    def inverse_transform(self, projected_data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Reconstruct data from spectral subspace

        Args:
            projected_data: Data in spectral subspace

        Returns:
            Reconstructed data in original space
        """
        if self.inverse_projection is None:
            raise ValueError("Spectral projector must be fitted before inverse transformation")

        if self.use_torch and isinstance(projected_data, torch.Tensor):
            return projected_data @ self.inverse_projection.T
        else:
            if isinstance(projected_data, torch.Tensor):
                projected_data = projected_data.cpu().numpy()
            return projected_data @ self.inverse_projection.cpu().numpy().T

    def get_spectral_statistics(self) -> dict:
        """
        Get spectral statistics from fitted projector

        Returns:
            Dictionary containing spectral statistics
        """
        if self.singular_values is None:
            raise ValueError("Spectral projector must be fitted before getting statistics")

        return {
            'singular_values': self.singular_values,
            'explained_variance': self.explained_variance,
            'total_variance': self.explained_variance.sum(),
            'effective_dimension': len(self.singular_values),
            'projection_dimension': self.projection_matrix.shape[1] if self.projection_matrix is not None else 0
        }

    def save(self, filepath: str):
        """Save fitted projector to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'projection_matrix': self.projection_matrix,
                'inverse_projection': self.inverse_projection,
                'singular_values': self.singular_values,
                'explained_variance': self.explained_variance,
                'config': {
                    'projection_dim': self.projection_dim,
                    'spectral_threshold': self.spectral_threshold,
                    'use_torch': self.use_torch
                }
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'SpectralProjector':
        """Load fitted projector from file"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        projector = cls(**data['config'])
        projector.projection_matrix = data['projection_matrix']
        projector.inverse_projection = data['inverse_projection']
        projector.singular_values = data['singular_values']
        projector.explained_variance = data['explained_variance']

        return projector


def create_random_spectral_data(n_samples: int = 1000, n_features: int = 1024) -> torch.Tensor:
    """
    Create random spectral data for testing

    Args:
        n_samples: Number of samples
        n_features: Number of features

    Returns:
        Random spectral data tensor
    """
    # Create data with spectral structure
    torch.manual_seed(42)

    # Create basis vectors with decaying eigenvalues
    eigenvalues = torch.exp(-torch.arange(n_features) / 50)
    basis = torch.randn(n_features, n_features)

    # Orthogonalize basis
    Q, _ = torch.linalg.qr(basis)

    # Create correlated data
    data = torch.randn(n_samples, n_features) @ (Q * eigenvalues).T

    return data


def test_spectral_projector():
    """Test function for spectral projector"""
    # Create test data
    data = create_random_spectral_data(100, 256)

    # Initialize and fit projector
    projector = SpectralProjector(projection_dim=50)
    projector.fit(data)

    # Test transformation
    projected = projector.transform(data)
    reconstructed = projector.inverse_transform(projected)

    # Calculate reconstruction error
    error = torch.norm(data - reconstructed) / torch.norm(data)

    print(f"Original data shape: {data.shape}")
    print(f"Projected data shape: {projected.shape}")
    print(f"Reconstruction error: {error:.6f}")

    # Print spectral statistics
    stats = projector.get_spectral_statistics()
    print(f"Effective dimension: {stats['effective_dimension']}")
    print(f"Total variance explained: {stats['total_variance']:.4f}")

    return error < 0.1  # Success if reconstruction error < 10%


if __name__ == "__main__":
    # Run test if executed directly
    success = test_spectral_projector()
    print(f"Test {'PASSED' if success else 'FAILED'}")