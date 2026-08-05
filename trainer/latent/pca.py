"""PCA training with Varimax rotation for the latent layer.

Uses IncrementalPCA for memory-efficient training on potentially large datasets.
Includes Varimax rotation for improved interpretability of latent axes.
"""

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Iterator


def varimax_rotation(
    Phi: np.ndarray,
    gamma: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Varimax rotation of factor loading matrix.

    Varimax maximizes the variance of squared loadings within each component,
    producing axes that load heavily on a few features rather than moderately
    on many. This improves interpretability.

    Args:
        Phi: [p, k] loading matrix (features x components)
        gamma: 1.0 for varimax, 0.0 for quartimax
        max_iter: Maximum iterations for convergence
        tol: Convergence tolerance

    Returns:
        Tuple of (rotated_loadings [p, k], rotation_matrix [k, k])
    """
    p, k = Phi.shape
    R = np.eye(k, dtype=Phi.dtype)

    for iteration in range(max_iter):
        Lambda = Phi @ R
        # Varimax criterion gradient
        Lambda_sq = Lambda ** 2
        col_sums = np.sum(Lambda_sq, axis=0, keepdims=True)
        gradient = Phi.T @ (Lambda ** 3 - (gamma / p) * Lambda * col_sums)

        # SVD to find optimal rotation update
        u, s, vh = np.linalg.svd(gradient)
        R_new = u @ vh

        # Check convergence
        if np.max(np.abs(R_new - R)) < tol:
            break
        R = R_new

    return Phi @ R, R


class LatentPCA:
    """
    Incremental PCA with standardization and optional Varimax rotation.

    Designed for streaming/batch training without loading all data into memory.
    """

    def __init__(
        self,
        n_components: int = 8,
        batch_size: int = 1024,
        apply_varimax: bool = True
    ):
        """
        Args:
            n_components: Number of PCA dimensions to retain
            batch_size: Batch size for incremental fitting
            apply_varimax: Whether to apply Varimax rotation after fitting
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.apply_varimax = apply_varimax

        self.scaler = StandardScaler()
        self.ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

        # Fitted parameters
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None  # [n_components, n_features]
        self.rotation_matrix_: Optional[np.ndarray] = None  # [n_components, n_components]

    def partial_fit(self, X: np.ndarray) -> 'LatentPCA':
        """
        Incrementally fit on a batch of data.

        Call this multiple times with batches, then call finalize().

        Args:
            X: [batch_size, n_features] feature matrix
        """
        # Update scaler statistics
        self.scaler.partial_fit(X)

        # Transform batch and fit PCA incrementally
        X_scaled = self.scaler.transform(X)
        self.ipca.partial_fit(X_scaled)

        return self

    def fit(self, X: np.ndarray) -> 'LatentPCA':
        """
        Fit on all data at once (if it fits in memory).

        Args:
            X: [n_samples, n_features] feature matrix
        """
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)

        # Fit PCA
        self.ipca.fit(X_scaled)

        # Extract and optionally rotate
        self._finalize()

        return self

    def fit_batches(self, batches: Iterator[np.ndarray]) -> 'LatentPCA':
        """
        Fit incrementally from a batch iterator.

        Args:
            batches: Iterator yielding [batch_size, n_features] arrays
        """
        for batch in batches:
            self.partial_fit(batch)

        self._finalize()
        return self

    def _finalize(self) -> None:
        """Extract fitted parameters and apply Varimax if enabled."""
        self.mean_ = self.scaler.mean_.astype(np.float32)
        self.std_ = np.sqrt(self.scaler.var_).astype(np.float32)

        # Get PCA components [n_components, n_features]
        components = self.ipca.components_.astype(np.float32)

        if self.apply_varimax:
            # Varimax expects [n_features, n_components], transpose in and out
            rotated, R = varimax_rotation(components.T)
            self.components_ = rotated.T  # Back to [n_components, n_features]
            self.rotation_matrix_ = R.astype(np.float32)
        else:
            self.components_ = components
            self.rotation_matrix_ = np.eye(self.n_components, dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to latent space.

        Args:
            X: [n_samples, n_features] feature matrix

        Returns:
            Z: [n_samples, n_components] latent vectors
        """
        if self.mean_ is None:
            raise RuntimeError("PCA not fitted. Call fit() or partial_fit() first.")

        # Standardize
        X_scaled = (X - self.mean_) / (self.std_ + 1e-8)

        # Project
        Z = X_scaled @ self.components_.T

        return Z.astype(np.float32)

    def get_params(self) -> dict:
        """Get parameters for export."""
        if self.mean_ is None:
            raise RuntimeError("PCA not fitted.")

        return {
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist(),
            'components': self.components_.tolist(),
            'rotation_matrix': self.rotation_matrix_.tolist(),
            'n_components': self.n_components,
            'explained_variance_ratio': self.ipca.explained_variance_ratio_.tolist()
        }


def train_pca(
    X: np.ndarray,
    n_components: int = 8,
    apply_varimax: bool = True
) -> LatentPCA:
    """
    Train PCA on feature data.

    Args:
        X: [n_samples, n_features] feature matrix
        n_components: Number of PCA dimensions
        apply_varimax: Apply Varimax rotation for interpretability

    Returns:
        Fitted LatentPCA instance
    """
    pca = LatentPCA(n_components=n_components, apply_varimax=apply_varimax)
    pca.fit(X)
    return pca


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    X = np.random.randn(1000, 150).astype(np.float32)

    pca = train_pca(X, n_components=8, apply_varimax=True)
    Z = pca.transform(X)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Z.shape}")
    print(f"Components shape: {pca.components_.shape}")
    print(f"Explained variance ratios: {pca.ipca.explained_variance_ratio_}")
