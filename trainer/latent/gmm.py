"""GMM training for soft technique mode detection.

Trains a Gaussian Mixture Model in PCA space to identify gesture clusters
that emerge naturally from playing data.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Optional


class LatentGMM:
    """
    Gaussian Mixture Model for soft mode detection in latent space.

    Uses diagonal covariances for efficiency and stability.
    """

    def __init__(
        self,
        n_components: int = 14,
        random_state: int = 42
    ):
        """
        Args:
            n_components: Number of mixture components (gesture clusters)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state

        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='diag',
            random_state=random_state,
            n_init=3,  # Multiple initializations for stability
            max_iter=200
        )

        # Fitted parameters
        self.means_: Optional[np.ndarray] = None  # [n_components, n_dims]
        self.covariances_: Optional[np.ndarray] = None  # [n_components, n_dims]
        self.weights_: Optional[np.ndarray] = None  # [n_components]

    def fit(self, Z: np.ndarray) -> 'LatentGMM':
        """
        Fit GMM on PCA-transformed data.

        Args:
            Z: [n_samples, n_dims] latent vectors from PCA

        Returns:
            self
        """
        self.gmm.fit(Z)

        # Extract parameters
        self.means_ = self.gmm.means_.astype(np.float32)
        self.covariances_ = self.gmm.covariances_.astype(np.float32)
        self.weights_ = self.gmm.weights_.astype(np.float32)

        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute responsibilities (soft cluster assignments).

        Args:
            Z: [n_samples, n_dims] latent vectors

        Returns:
            R: [n_samples, n_components] responsibilities summing to 1
        """
        return self.gmm.predict_proba(Z).astype(np.float32)

    def score_samples(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood for each sample.

        Args:
            Z: [n_samples, n_dims] latent vectors

        Returns:
            log_prob: [n_samples] log probability under the model
        """
        return self.gmm.score_samples(Z).astype(np.float32)

    def get_params(self) -> dict:
        """Get parameters for export."""
        if self.means_ is None:
            raise RuntimeError("GMM not fitted.")

        return {
            'n_components': self.n_components,
            'means': self.means_.tolist(),
            'covariances': self.covariances_.tolist(),
            'weights': self.weights_.tolist()
        }

    def analyze_clusters(self, Z: np.ndarray) -> dict:
        """
        Analyze cluster statistics for interpretability.

        Args:
            Z: [n_samples, n_dims] latent vectors used for training

        Returns:
            Dictionary with cluster analysis
        """
        if self.means_ is None:
            raise RuntimeError("GMM not fitted.")

        R = self.predict_proba(Z)
        hard_labels = np.argmax(R, axis=1)

        analysis = {
            'cluster_sizes': [],
            'cluster_mean_responsibility': [],
            'bic': self.gmm.bic(Z),
            'aic': self.gmm.aic(Z)
        }

        for k in range(self.n_components):
            mask = hard_labels == k
            analysis['cluster_sizes'].append(int(np.sum(mask)))
            analysis['cluster_mean_responsibility'].append(
                float(np.mean(R[mask, k])) if np.sum(mask) > 0 else 0.0
            )

        return analysis


def train_gmm(
    Z: np.ndarray,
    n_components: int = 14,
    random_state: int = 42
) -> LatentGMM:
    """
    Train GMM on PCA-transformed latent vectors.

    Args:
        Z: [n_samples, n_dims] latent vectors from PCA
        n_components: Number of mixture components
        random_state: Random seed

    Returns:
        Fitted LatentGMM instance
    """
    gmm = LatentGMM(n_components=n_components, random_state=random_state)
    gmm.fit(Z)
    return gmm


def select_n_components(
    Z: np.ndarray,
    min_k: int = 8,
    max_k: int = 20,
    criterion: str = 'bic'
) -> int:
    """
    Select optimal number of GMM components using BIC or AIC.

    Args:
        Z: [n_samples, n_dims] latent vectors
        min_k: Minimum number of components to try
        max_k: Maximum number of components to try
        criterion: 'bic' or 'aic'

    Returns:
        Optimal number of components
    """
    scores = []
    for k in range(min_k, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='diag',
            random_state=42,
            n_init=3
        )
        gmm.fit(Z)
        score = gmm.bic(Z) if criterion == 'bic' else gmm.aic(Z)
        scores.append((k, score))
        print(f"k={k}: {criterion.upper()}={score:.2f}")

    # Return k with lowest score
    best_k = min(scores, key=lambda x: x[1])[0]
    return best_k


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    Z = np.random.randn(1000, 8).astype(np.float32)

    gmm = train_gmm(Z, n_components=14)
    R = gmm.predict_proba(Z)

    print(f"Input shape: {Z.shape}")
    print(f"Responsibilities shape: {R.shape}")
    print(f"Weights: {gmm.weights_}")

    analysis = gmm.analyze_clusters(Z)
    print(f"Cluster sizes: {analysis['cluster_sizes']}")
    print(f"BIC: {analysis['bic']:.2f}")
